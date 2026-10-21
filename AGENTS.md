# Connect Four Arena

This file provides guidance to Coding Agents when working with code in this repository.

## Commands

- Install deps: `uv sync` (creates `.venv` from `uv.lock`, including the `dev` group: `black`, `jupyterlab`, `ipywidgets`).
- Run the Gradio app locally: `uv run app.py` (entry point loads `.env`, builds the UI from `arena/c4.py`).
- No test suite or lint config is wired up. Format manually with `uv run black`.
- `prototype.ipynb` is an exploratory notebook kept alongside the app; it is not part of the runtime path.
- Two git remotes: `origin` (GitHub, `ed-donner/connect`) and `hf` (the HuggingFace Space, `ed-donner/connect`). Pushing to `hf` deploys the Space.

### Local env mirrors the Space

- `pyproject.toml` is the source of truth for dependencies. `requires-python = "==3.12.*"` (plus `.python-version`) matches `python_version` in the README front matter.
- HF Spaces only reads `requirements.txt`, so it is **generated** from `uv.lock` with exact pins, runtime deps only. Never hand-edit it; after any dependency change run:
  `uv export --no-dev --no-hashes --no-emit-project --no-annotate --output-file requirements.txt`
- The Space build also installs `gradio[oauth]==<sdk_version>`, `spaces` and `uvicorn` itself. Checked with `uv pip compile` that these resolve against the pins without changing any pinned version.

### Gradio version (pinned — do not "upgrade")

`pyproject.toml` (and so `requirements.txt`) pins `gradio==5.15.0` deliberately, matching `sdk_version` in `README.md` (HF Spaces uses that value at deploy time). Gradio **6.x regresses the Leaderboard**: switching to the tab freezes the browser's main thread indefinitely after the SSE payload arrives — the `js=` parameter on `Blocks` also stops auto-calling the named function there. Pinning is the fix; do not bump it without a matching HF `sdk_version` bump *and* a live re-test of the Leaderboard tab.

## Required env (.env at project root)

- `OPENROUTER_API_KEY` — the only provider key; every model goes through OpenRouter.
- `MONGO_URI` — optional. Without it, `arena/record.py` silently returns no history and does not persist games (the Leaderboard tab just shows empty tables). The DB name is `outsmart`, collection `connect`. Local runs and the Space share this DB, so a locally completed "Run game" lands on the live leaderboard ("Next move" does not record).
- `MODELS` — optional comma-separated allowlist. `LLM.all_model_names()` intersects this with the registered models to populate the dropdowns. The HF Space has set it in the past (in Space settings, not the repo) to expose only cheap models. Gotchas:
  - Entries are not whitespace-trimmed, so `a, b` silently drops `b`. Names must match `model_names` exactly, e.g. `openai/gpt-6-luna` (no `openrouter/` prefix).
  - Order sets the dropdown order; the first two entries are the default Red and Yellow players. Without `MODELS`, the defaults are the first two entries of `OpenRouter.model_names`.
  - At least two entries must survive the filter, or `c4.py` hits an IndexError on `ALL_MODEL_NAMES[1]` at startup. Renaming or replacing models can trigger this on the Space if its `MODELS` still lists old names.
  - Read once at import time, so changing it needs a restart. It does not filter the Leaderboard, which uses `all_supported_model_names()`.

## Architecture

Single-process Gradio app where two LLMs play Connect 4 against each other. The layering is deliberate:

- `app.py` → `arena/c4.py::make_display` builds the Gradio `Blocks` and wires callbacks. All UI state is held in a single `gr.State` containing a `Game`. The `js=` function on `Blocks` only forces the dark theme.
- `arena/game.py::Game` owns a `Board` and two `Player`s keyed by color constant (`RED=1`, `YELLOW=-1`, see `arena/board.py`). `Game.record()` writes results via `arena/record.py`.
- `arena/board.py::Board` is pure game logic plus multiple render formats (`__repr__`, `html`, `svg` via `board_view.py`, `display`, `next_drops`). `display()` is a labelled ASCII grid with row numbers and an explicit "bottom (pieces stack upward)" marker; `next_drops()` is a single line listing which row each legal column's next piece lands on. These two are what's sent to the LLM — the older `json()`/`alternative()` dual representation was removed once per-token cost and diagonal blindness on small models drove a redesign. The drop animation and clack sound are inlined into an `<img onload>` attribute (`_DROP_TRIGGER_JS` in `board_view.py`), so they don't depend on Gradio's `js=` parameter.
- `arena/player.py::Player` owns the prompt templates. Builds a dynamic Pydantic `MoveSchema` per turn (`move_column: Literal[*legal_moves]`) so the structured output layer makes illegal columns unrepresentable. `process_move` receives a parsed schema instance; the only forfeit path is an exhausted-retry API failure.
- `arena/llm.py::LLM` wraps **LiteLLM** (`litellm.completion`). All models are on the `OpenRouter` subclass, which adds the `openrouter/` prefix. Returns parsed Pydantic instances.

### LLM layer (LiteLLM via OpenRouter)

- `send(system, user, schema_cls) -> BaseModel | None` is the single entry point. It retries `_send` up to 3 times; returns `None` only when retries are exhausted (the forfeit signal). Structured outputs either return valid JSON or raise.
- `OpenRouter._send` is non-streaming and sends three things beyond the messages. All verified on the wire (outbound body captured by wrapping `httpx.Client.send`) with LiteLLM 1.106.2:
  - `response_format=schema_cls` → `json_schema` with `strict: true` and the legal columns as an `enum`.
  - `reasoning={"effort": ...}` — OpenRouter's normalized reasoning control, which it maps onto each provider's native knob. Used instead of LiteLLM's `reasoning_effort=` because not every model lists that param (MiMo doesn't). Default is `REASONING_EFFORT = "low"`. Higher levels up to `xhigh` also work: LiteLLM 1.106 accepts `xhigh` for OpenRouter (older LiteLLM rejected it).
  - `provider={"require_parameters": True}` — **must-have**. Without it OpenRouter may route to hosts that don't support structured outputs (for Haiku: Bedrock and Claude Platform on AWS; for DeepSeek: DeepSeek's own API, Novita, Alibaba, BaseTen; for MiMo: Relace, Novita), which return output that fails schema validation and leads to forfeits.
- **Per-model effort overrides** (`OpenRouter.reasoning_efforts`): DeepSeek V4.1 Flash and MiMo V2.6 Flash run at `"none"`. Their hosts (InferenceNet, Makora) ignore the effort level: at `xhigh` one move took 93-129s (9.5-15k reasoning tokens), `minimal` was no better (120-134s, 18-20k), and `reasoning={"max_tokens": 1024}` still produced 9-48k. Only `"none"` takes effect (0 reasoning tokens, ~1-6s). GPT-6 Luna, Haiku 5.5 and Gemini 3.5 Flash-Lite run at the default `low`. Measured on one test position: Luna ~300-430 reasoning tokens and 7.5-9s per move (no lower than at `medium`), Haiku ~210 tokens and ~4s, Gemini 0 reasoning tokens and ~1s (`low` effectively turns its reasoning off). At `xhigh` they used ~1-2k tokens and up to 17s.
- `_parse_message` handles both delivery shapes LiteLLM may use (`message.content` JSON or `message.tool_calls[0].function.arguments`).
- Direct-provider subclasses (OpenAI, Anthropic, Gemini, xAI) were removed when the app moved to OpenRouter. If direct providers come back, commit `4ddcf00` has the provider-specific workarounds: Anthropic streaming at high `max_tokens`, the `output_config` monkey-patch for LiteLLM's effort whitelist, the Fable `output_format` path, and the thinking-vs-forced-tool-use conflict on older Claude models.

### Adding a model

1. Add the OpenRouter model ID to `OpenRouter.model_names` in `arena/llm.py`. The first two entries are the default players. Display names may include a trailing space suffix for UI friendliness; `api_model_name()` strips it before the call.
2. Check the exact ID and capabilities against OpenRouter's live list rather than guessing: `https://openrouter.ai/api/v1/models` (`supported_parameters` must include `structured_outputs` and `reasoning`), and `https://openrouter.ai/api/v1/models/<id>/endpoints` for per-host support.
3. Time a real move. If it reasons for minutes, check whether lower efforts actually take effect on its hosts (compare `usage.completion_tokens_details.reasoning_tokens`) before adding an override to `reasoning_efforts`.
4. No other registration step — `ALL_MODEL_NAMES` in `c4.py` is computed at import time from `LLM.all_model_names()`.

### Leaderboard / ELO

`arena/record.py` exposes three query helpers, all going through `_fetch_games(query, sort_direction, limit)`:

- `get_games(limit=20)` — used by the Recent games table. Sort-desc + limit + reverse so Python gets the newest-first window in chronological order. Capped because the Gradio Dataframe payload was large enough to cause visible client-side lag.
- `count_games()` — `collection.count_documents({})`, used to show the "Most recent 20 of N games played" label on the Recent games table. Server-side count, no row transfer.
- `get_supported_games(supported_models)` — `$in` filter pushed into the Mongo query so ELO only pulls games where *both* players are currently registered in `LLM.all_supported_model_names()`. Retired-model games stay in the DB for history but don't influence ratings. Games are stored under the display name (e.g. `openai/gpt-6-luna`), so renaming a model starts its rating from scratch.

ELO computation (K=32, default 1000, self-play excluded via `calculate_elo_ratings`'s `exclude_self_play=True`) runs on the filtered set only, which keeps the tab responsive even as total game history grows large (12.5k+ rows as of Oct 2026).

If the query ever feels slow at the DB layer, a compound index on `(red_player, yellow_player)` would make the `$in` filter near-instant — that's a one-time `createIndex` against the `outsmart.connect` collection.
