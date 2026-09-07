# Connect Four Arena

This file provides guidance to Coding Agents when working with code in this repository.

## Commands

- Run the Gradio app locally: `python app.py` (entry point loads `.env`, builds the UI from `arena/c4.py`).
- Install deps: `pip install -r requirements.txt` (HuggingFace Spaces target; no `uv` / `pyproject.toml` here).
- No test suite or lint config is wired up. `black` is listed in requirements for manual formatting.
- `prototype.ipynb` is an exploratory notebook kept alongside the app; it is not part of the runtime path.

### Gradio version (pinned — do not "upgrade")

`requirements.txt` pins `gradio==5.15.0` deliberately, matching `sdk_version` in `README.md` (HF Spaces uses that value at deploy time). Gradio **6.x regresses the Leaderboard**: switching to the tab freezes the browser's main thread indefinitely after the SSE payload arrives — the `js=` parameter on `Blocks` also stops auto-calling the named function there. Pinning is the fix; do not bump it without a matching HF `sdk_version` bump *and* a live re-test of the Leaderboard tab.

## Required env (.env at project root)

- Provider keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`. Only the keys corresponding to enabled models need to be present.
- `MONGO_URI` — optional. Without it, `arena/record.py` silently returns no history and does not persist games (the Leaderboard tab just shows empty tables). The DB name is `outsmart`, collection `connect`.
- `MODELS` — optional comma-separated allowlist. `LLM.all_model_names()` intersects this with the registered models to populate the dropdowns; useful for Spaces deployments that should expose a subset.

## Architecture

Single-process Gradio app where two LLMs play Connect 4 against each other. The layering is deliberate:

- `app.py` → `arena/c4.py::make_display` builds the Gradio `Blocks` and wires callbacks. All UI state is held in a single `gr.State` containing a `Game`.
- `arena/game.py::Game` owns a `Board` and two `Player`s keyed by color constant (`RED=1`, `YELLOW=-1`, see `arena/board.py`). `Game.record()` writes results via `arena/record.py`.
- `arena/board.py::Board` is pure game logic plus multiple render formats (`__repr__`, `html`, `svg` via `board_view.py`, `display`, `next_drops`). `display()` is a labelled ASCII grid with row numbers and an explicit "bottom (pieces stack upward)" marker; `next_drops()` is a single line listing which row each legal column's next piece lands on. These two are what's sent to the LLM — the older `json()`/`alternative()` dual representation was removed once per-token cost and diagonal blindness on small models drove a redesign.
- `arena/player.py::Player` owns the prompt templates. Builds a dynamic Pydantic `MoveSchema` per turn (`move_column: Literal[*legal_moves]`) so the structured output layer makes illegal columns unrepresentable. `process_move` receives a parsed schema instance; the only forfeit path is an exhausted-retry API failure.
- `arena/llm.py::LLM` wraps **LiteLLM** (`litellm.completion`). One code path for OpenAI, Anthropic, Gemini, xAI (Grok), and OpenRouter, selected by the model-name prefix each subclass attaches (`anthropic/claude-...`, `gemini/gemini-...`, `xai/grok-...`, `openrouter/moonshotai/...`, or plain `gpt-...`). Returns parsed Pydantic instances.

### LLM layer (LiteLLM)

- `send(system, user, schema_cls) -> BaseModel | None` is the single entry point. Internally calls `litellm.completion(..., response_format=schema_cls)` and retries up to 3 times; returns `None` only when retries are exhausted (the forfeit signal). The old JSON-substring extraction hack is gone — structured outputs either return valid JSON or raise.
- **OpenAI path** (`LLM._send`, inherited by `GPT`): non-streaming, no `max_tokens`, just `response_format=schema_cls`.
- **OpenRouter path** (`OpenRouter._send`): same as OpenAI shape plus `reasoning={"effort": "high"}` so reasoning models like `moonshotai/kimi-k2.6` run at max effort, not OpenRouter's default. Verified on the wire. Valid OpenRouter effort levels are `"low"`, `"medium"`, `"high"` — LiteLLM's param validator rejects `"xhigh"`/`"max"`, so do not reintroduce those. OpenRouter also rejects LiteLLM's unified `reasoning_effort=` param, which is why we use the `reasoning={...}` dict form instead. OpenRouter has no Anthropic-style timeout, so streaming is unnecessary even though some calls can take a couple of minutes.
- **Gemini path** (`Gemini._send`): uses LiteLLM's unified `reasoning_effort="high"` (which maps to Gemini 3+'s `thinking_level`), but *only* for non-Lite variants. Flash-Lite variants are non-reasoning by design and LiteLLM raises `UnsupportedParamsError` if `reasoning_effort` is passed — the class skips the kwarg when `"lite"` is in the model name. Temperature is left at Gemini's default (1.0); values < 1.0 can cause infinite loops per LiteLLM provider docs.
- **xAI Grok path** (`Grok._send`): inherits the default OpenAI-style `_send` with *no* reasoning param. Grok 4.20 rejects `reasoningEffort` on every variant including the `-reasoning` one (xAI's error message: "Model ... does not support parameter reasoningEffort."). The `-reasoning` variant reasons at its built-in level; the `-non-reasoning` variant doesn't.
- **Anthropic path** (`Claude._send`): always streams with `max_tokens=64_000` to avoid Anthropic's 10-min non-streaming timeout at high token ceilings. Chunks are rebuilt via `litellm.stream_chunk_builder`, then parsed by `_parse_message`, which handles both delivery shapes LiteLLM may use (`message.content` JSON or `message.tool_calls[0].function.arguments`).
- **Opus-4.7-only kwargs** (`Claude._is_opus_4_7()` gate):
  - `thinking={"type": "adaptive"}` — adaptive extended reasoning.
  - `output_config={"effort": "max"}` — **must-have** for Opus. Without this it runs at a weaker reasoning level and is unfairly disadvantaged on the leaderboard.
  - These are Opus-4.7-only because on older Claude models (Opus 4.1, Sonnet 4.5, Haiku 4.5) LiteLLM falls back to forced tool-use to implement `response_format`, and Anthropic rejects any `thinking` combined with forced tool-use ("Thinking may not be enabled when tool_choice forces tool use"). Opus 4.7 uses a different internal path in LiteLLM (no `tool_choice` on the wire — confirmed via packet capture) and accepts the combination. Don't try to re-enable thinking on the older models without changing the structured-output path.
- **LiteLLM `output_config` monkey-patch** (top of `arena/llm.py`): LiteLLM's `AnthropicConfig._apply_output_config` hardcodes `effort='max'` as "Opus 4.6 only" (see `litellm/llms/anthropic/chat/transformation.py:1539`) and raises before the request leaves the process. We replace that method with a minimal version that forwards `output_config` as-is and lets the Anthropic API be the source of truth. This patch is load-bearing — if LiteLLM is upgraded, re-check that the method name, signature, and validation logic haven't changed, and that our patch still attaches correctly. Verify by capturing the outbound HTTPS body and confirming `output_config` is at the top level.

### Adding a model

1. Add the display name to the appropriate `model_names` list on `GPT`, `Claude`, `Gemini`, `Grok`, or `OpenRouter` in `arena/llm.py`. Display names may include a trailing space suffix for UI friendliness; `api_model_name()` strips it before the call.
2. Each subclass's `api_model_name()` auto-prefixes for LiteLLM: `anthropic/`, `gemini/`, `xai/`, `openrouter/`. `GPT` needs no prefix. For a brand-new provider, add a subclass with its own prefix override.
3. Use the provider's own model-listing endpoint to verify the exact model string rather than guessing — `https://generativelanguage.googleapis.com/v1beta/models` for Gemini, `https://api.x.ai/v1/models` for xAI. LiteLLM's provider docs lag behind; the provider's live model list is authoritative.
4. If adding another Opus-4.7-class model that also supports `thinking`+`output_config`, widen the `_is_opus_4_7()` check — but first re-verify with packet capture that LiteLLM doesn't inject `tool_choice` for that model, otherwise the thinking-vs-forced-tool-use conflict reappears.
5. No other registration step — `ALL_MODEL_NAMES` in `c4.py` is computed at import time from `LLM.all_model_names()`.

### Leaderboard / ELO

`arena/record.py` exposes three query helpers, all going through `_fetch_games(query, sort_direction, limit)`:

- `get_games(limit=20)` — used by the Recent games table. Sort-desc + limit + reverse so Python gets the newest-first window in chronological order. Capped because the Gradio Dataframe payload was large enough to cause visible client-side lag.
- `count_games()` — `collection.count_documents({})`, used to show the "Most recent 20 of N games played" label on the Recent games table. Server-side count, no row transfer.
- `get_supported_games(supported_models)` — `$in` filter pushed into the Mongo query so ELO only pulls games where *both* players are currently registered in `LLM.all_supported_model_names()`. Retired-model games stay in the DB for history but don't influence ratings.

ELO computation (K=32, default 1000, self-play excluded via `calculate_elo_ratings`'s `exclude_self_play=True`) runs on the filtered set only, which keeps the tab responsive even as total game history grows large (currently 9k+ rows, of which only ~100 involve current models).

If the query ever feels slow at the DB layer, a compound index on `(red_player, yellow_player)` would make the `$in` filter near-instant — that's a one-time `createIndex` against the `outsmart.connect` collection.
