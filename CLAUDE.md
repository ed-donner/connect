# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run the Gradio app locally: `python app.py` (entry point loads `.env`, builds the UI from `arena/c4.py`).
- Install deps: `pip install -r requirements.txt` (HuggingFace Spaces target; no `uv` / `pyproject.toml` here).
- No test suite or lint config is wired up. `black` is listed in requirements for manual formatting.
- `prototype.ipynb` is an exploratory notebook kept alongside the app; it is not part of the runtime path.

## Required env (.env at project root)

- Provider keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` (for anything routed via OpenRouter, e.g. `moonshotai/kimi-k2.6`).
- `MONGO_URI` — optional. Without it, `arena/record.py` silently returns no history and does not persist games (the Leaderboard tab just shows empty tables). The DB name is `outsmart`, collection `connect`.
- `MODELS` — optional comma-separated allowlist. `LLM.all_model_names()` intersects this with the registered models to populate the dropdowns; useful for Spaces deployments that should expose a subset.

## Architecture

Single-process Gradio app where two LLMs play Connect 4 against each other. The layering is deliberate:

- `app.py` → `arena/c4.py::make_display` builds the Gradio `Blocks` and wires callbacks. All UI state is held in a single `gr.State` containing a `Game`.
- `arena/game.py::Game` owns a `Board` and two `Player`s keyed by color constant (`RED=1`, `YELLOW=-1`, see `arena/board.py`). `Game.record()` writes results via `arena/record.py`.
- `arena/board.py::Board` is pure game logic plus multiple render formats (`__repr__`, `html`, `svg` via `board_view.py`, `json`, `alternative`). The JSON and `alternative` ASCII forms are both sent to the LLM in the same prompt so the model sees the position two ways.
- `arena/player.py::Player` owns the prompt templates. Builds a dynamic Pydantic `MoveSchema` per turn (`move_column: Literal[*legal_moves]`) so the structured output layer makes illegal columns unrepresentable. `process_move` receives a parsed schema instance; the only forfeit path is an exhausted-retry API failure.
- `arena/llm.py::LLM` wraps **LiteLLM** (`litellm.completion`). One code path for OpenAI, Anthropic, and OpenRouter, selected by the model-name prefix each subclass attaches (`anthropic/claude-...`, `openrouter/moonshotai/...`, or plain `gpt-...`). Returns parsed Pydantic instances.

### LLM layer (LiteLLM)

- `send(system, user, schema_cls) -> BaseModel | None` is the single entry point. Internally calls `litellm.completion(..., response_format=schema_cls)` and retries up to 3 times; returns `None` only when retries are exhausted (the forfeit signal). The old JSON-substring extraction hack is gone — structured outputs either return valid JSON or raise.
- **OpenAI path** (`LLM._send`, inherited by `GPT`): non-streaming, no `max_tokens`, just `response_format=schema_cls`.
- **OpenRouter path** (`OpenRouter._send`): same as OpenAI shape plus `reasoning={"effort": "high"}` so reasoning models like `moonshotai/kimi-k2.6` run at max effort, not OpenRouter's default. Verified on the wire. Valid OpenRouter effort levels are `"low"`, `"medium"`, `"high"` — LiteLLM's param validator rejects `"xhigh"`/`"max"`, so do not reintroduce those. OpenRouter has no Anthropic-style timeout, so streaming is unnecessary even though kimi calls can take a couple of minutes.
- **Anthropic path** (`Claude._send`): always streams with `max_tokens=64_000` to avoid Anthropic's 10-min non-streaming timeout at high token ceilings. Chunks are rebuilt via `litellm.stream_chunk_builder`, then parsed by `_parse_message`, which handles both delivery shapes LiteLLM may use (`message.content` JSON or `message.tool_calls[0].function.arguments`).
- **Opus-4.7-only kwargs** (`Claude._is_opus_4_7()` gate):
  - `thinking={"type": "adaptive"}` — adaptive extended reasoning.
  - `output_config={"effort": "max"}` — **must-have** for Opus. Without this it runs at a weaker reasoning level and is unfairly disadvantaged on the leaderboard.
  - These are Opus-4.7-only because on older Claude models (Opus 4.1, Sonnet 4.5, Haiku 4.5) LiteLLM falls back to forced tool-use to implement `response_format`, and Anthropic rejects any `thinking` combined with forced tool-use ("Thinking may not be enabled when tool_choice forces tool use"). Opus 4.7 uses a different internal path in LiteLLM (no `tool_choice` on the wire — confirmed via packet capture) and accepts the combination. Don't try to re-enable thinking on the older models without changing the structured-output path.
- **LiteLLM `output_config` monkey-patch** (top of `arena/llm.py`): LiteLLM's `AnthropicConfig._apply_output_config` hardcodes `effort='max'` as "Opus 4.6 only" (see `litellm/llms/anthropic/chat/transformation.py:1539`) and raises before the request leaves the process. We replace that method with a minimal version that forwards `output_config` as-is and lets the Anthropic API be the source of truth. This patch is load-bearing — if LiteLLM is upgraded, re-check that the method name, signature, and validation logic haven't changed, and that our patch still attaches correctly. Verify by capturing the outbound HTTPS body and confirming `output_config` is at the top level.

### Adding a model

1. Add the display name to the appropriate `model_names` list on `GPT`, `Claude`, or `OpenRouter` in `arena/llm.py`. Display names may include a trailing space suffix for UI friendliness; `api_model_name()` strips it before the call.
2. `Claude.api_model_name()` auto-prefixes with `anthropic/`; `OpenRouter.api_model_name()` auto-prefixes with `openrouter/`; `GPT` needs no prefix. For a brand-new provider, add a subclass with its own prefix override.
3. If adding another Opus-4.7-class model that also supports `thinking`+`output_config`, widen the `_is_opus_4_7()` check — but first re-verify with packet capture that LiteLLM doesn't inject `tool_choice` for that model, otherwise the thinking-vs-forced-tool-use conflict reappears.
4. No other registration step — `ALL_MODEL_NAMES` in `c4.py` is computed at import time from `LLM.all_model_names()`.

### Leaderboard / ELO

`arena/record.py` computes ELO on the fly from the full game history every time the Leaderboard tab is selected (K=32, default 1000, self-play excluded). There is no cached ratings table. `Game.get_ratings()` then filters to models currently registered in `LLM.all_supported_model_names()`, so retired model names disappear from the leaderboard even though their games remain in the DB.
