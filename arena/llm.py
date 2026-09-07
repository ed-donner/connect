"""
LLM layer for the Connect 4 arena, built on LiteLLM.

Single code path for OpenAI, Anthropic, Google (Gemini), xAI (Grok), and
OpenRouter. Structured outputs via a Pydantic schema passed through
`response_format`; LiteLLM implements this as tool-use under the hood for
providers that lack native JSON-schema support. Anthropic calls stream (to
dodge the 10-min non-streaming timeout at high max_tokens); Opus 4.7 carries
`thinking` + `output_config` and Fable 5 carries `output_format` +
`output_config` so both run at full reasoning strength (see Claude docstring).
Gemini and Grok use LiteLLM's unified `reasoning_effort="high"` knob so they
also play at maximum reasoning.
"""

from abc import ABC
import logging
import os
import time
from typing import Dict, List, Self, Type

import litellm
from dotenv import load_dotenv
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.utils import type_to_response_format_param
from pydantic import BaseModel

load_dotenv(override=True)


def _apply_output_config(self, data, model, optional_params):
    """
    Forward `output_config` to the Anthropic request body without LiteLLM's
    stale model-name whitelist. LiteLLM hardcodes `effort='max'` as Opus-4.6-only,
    but Opus 4.7 accepts it too. Let the Anthropic API be the source of truth.
    """
    output_config = optional_params.get("output_config")
    if isinstance(output_config, dict) and output_config:
        data["output_config"] = output_config


AnthropicConfig._apply_output_config = _apply_output_config

logger = logging.getLogger(__name__)

ANTHROPIC_MAX_TOKENS = 128_000
HAIKU_MAX_TOKENS = 64_000  # Haiku 4.5's output ceiling; the API rejects anything higher
RETRIES = 3
RETRY_DELAY_SECONDS = 2


class LLMException(Exception):
    pass


class LLM(ABC):
    """Abstract superclass. Default `_send` is the OpenAI path via LiteLLM."""

    model_names: List[str] = []

    def __init__(self, model_name: str):
        self.model_name = model_name

    def api_model_name(self) -> str:
        """Strip the UI-only suffix after the first space."""
        return self.model_name.split(" ", 1)[0]

    def send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel | None:
        """
        Retry up to RETRIES times. Return a parsed Pydantic instance, or None
        if every attempt fails. `None` is the forfeit signal for the caller.
        """
        retries = RETRIES
        while retries:
            retries -= 1
            try:
                return self._send(system, user, schema_cls)
            except Exception as e:
                logger.error(f"LLM call failed for {self.model_name}: {e}")
                if retries:
                    time.sleep(RETRY_DELAY_SECONDS)
        return None

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        """OpenAI path: no streaming, no max_tokens."""
        response = litellm.completion(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema_cls,
        )
        return _parse_message(response.choices[0].message, schema_cls)

    @classmethod
    def model_map(cls) -> Dict[str, Type[Self]]:
        mapping: Dict[str, Type[Self]] = {}
        for subclass in cls.__subclasses__():
            for model_name in subclass.model_names:
                mapping[model_name] = subclass
        return mapping

    @classmethod
    def all_supported_model_names(cls) -> List[str]:
        return list(cls.model_map().keys())

    @classmethod
    def all_model_names(cls) -> List[str]:
        models = cls.all_supported_model_names()
        allowed = os.getenv("MODELS")
        if allowed:
            return [m for m in allowed.split(",") if m in models]
        return models

    @classmethod
    def create(cls, model_name: str) -> Self:
        subclass = cls.model_map().get(model_name)
        if not subclass:
            raise LLMException(f"Unrecognized LLM model name: {model_name}")
        return subclass(model_name)


class GPT(LLM):
    """
    OpenAI gpt-5.x models. Sends `reasoning_effort="xhigh"` so models reason
    at maximum strength rather than OpenAI's default level. The base `_send`
    doesn't pass reasoning controls, so we override here.
    """

    model_names = ["gpt-5.6-sol", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        response = litellm.completion(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema_cls,
            reasoning_effort="xhigh",
        )
        return _parse_message(response.choices[0].message, schema_cls)


class Claude(LLM):
    """
    Anthropic models. Streams the response so we can set max_tokens high without
    hitting the 10-min non-streaming timeout. Reasoning is controlled per model:

    - Fable 5: thinking is always on and cannot be configured; effort is set via
      `output_config={"effort": "xhigh"}`. LiteLLM's `reasoning_effort` param is
      unusable here (its Anthropic mapping only knows models up to Opus 4.7), and
      its `response_format` mapping falls back to forced tool use for models not
      in its hardcoded allowlist — which Anthropic rejects when thinking is on.
      So we bypass both: we build the native `output_format` json_schema payload
      with LiteLLM's own mapper and pass it through directly (verified on the
      wire: no tools/tool_choice, structured-outputs beta header auto-added).
    - Opus 4.7: adaptive thinking + `output_config={"effort": "xhigh"}`. Its
      `response_format` takes LiteLLM's native output_format path already.
    - Sonnet 4.6, Haiku 4.5: no reasoning kwargs. LiteLLM implements their
      `response_format` via forced tool use, and Anthropic rejects thinking
      combined with forced tool_choice.
    """

    model_names = [
        "claude-fable-5",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ]

    def api_model_name(self) -> str:
        return f"anthropic/{super().api_model_name()}"

    def _is_fable(self) -> bool:
        return "fable" in self.api_model_name().lower()

    def _is_opus_4_7(self) -> bool:
        return "opus-4-7" in self.api_model_name().lower()

    def _max_tokens(self) -> int:
        return HAIKU_MAX_TOKENS if "haiku" in self.api_model_name().lower() else ANTHROPIC_MAX_TOKENS

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        kwargs = dict(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens(),
            stream=True,
        )
        if self._is_fable():
            response_format = type_to_response_format_param(response_format=schema_cls)
            kwargs["output_format"] = AnthropicConfig().map_response_format_to_anthropic_output_format(
                response_format
            )
            kwargs["output_config"] = {"effort": "xhigh"}
        else:
            kwargs["response_format"] = schema_cls
            if self._is_opus_4_7():
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": "xhigh"}
        stream = litellm.completion(**kwargs)
        rebuilt = litellm.stream_chunk_builder(list(stream))
        return _parse_message(rebuilt.choices[0].message, schema_cls)


class OpenRouter(LLM):
    """
    Models accessed via OpenRouter. LiteLLM routes via the `openrouter/` prefix
    and reads `OPENROUTER_API_KEY` from the environment. Non-streaming — OpenRouter
    has no Anthropic-style high-max_tokens timeout.

    `reasoning={"effort": "high"}` is passed so reasoning models like kimi-k2.6
    run at max effort, not OpenRouter's default. Valid OpenRouter effort levels
    are "low", "medium", "high" — do not use "xhigh"/"max" (LiteLLM rejects).
    Note: OpenRouter takes `reasoning={...}`, not LiteLLM's unified
    `reasoning_effort=` param (which OpenRouter explicitly rejects).
    """

    model_names = [
        "moonshotai/kimi-k2.6",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "z-ai/glm-5.1",
        "nvidia/nemotron-3-super-120b-a12b",
    ]

    def api_model_name(self) -> str:
        return f"openrouter/{super().api_model_name()}"

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        response = litellm.completion(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema_cls,
            reasoning={"effort": "high"},
        )
        return _parse_message(response.choices[0].message, schema_cls)


class Gemini(LLM):
    """
    Google Gemini models via LiteLLM. Reads `GEMINI_API_KEY` from environment.
    Pro/Flash variants use LiteLLM's `reasoning_effort` (mapped to Gemini 3+'s
    thinking_level) so they reason at full strength. Flash-Lite variants are
    non-reasoning by design — LiteLLM rejects `reasoning_effort` for them — so
    we skip the param for any model whose name contains "lite".
    Temperature is left at the Gemini default (1.0); values < 1.0 can cause
    infinite loops on Gemini 3 per the LiteLLM provider docs.
    """

    model_names = [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
    ]

    def api_model_name(self) -> str:
        return f"gemini/{super().api_model_name()}"

    def _supports_reasoning(self) -> bool:
        return "lite" not in self.model_name.lower()

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        kwargs = dict(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema_cls,
        )
        if self._supports_reasoning():
            kwargs["reasoning_effort"] = "high"
        response = litellm.completion(**kwargs)
        return _parse_message(response.choices[0].message, schema_cls)


class Grok(LLM):
    """
    xAI Grok models via LiteLLM. Reads `XAI_API_KEY` from environment.
    Inherits the default OpenAI-style `_send` — no `reasoning_effort` because
    Grok 4.20 rejects it on every variant (including the *-reasoning one):
    xAI returns "Model ... does not support parameter reasoningEffort." The
    -reasoning variant reasons at its built-in level; -non-reasoning doesn't.
    """

    model_names = [
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
    ]

    def api_model_name(self) -> str:
        return f"xai/{super().api_model_name()}"


def _parse_message(message, schema_cls: Type[BaseModel]) -> BaseModel:
    """
    LiteLLM delivers structured outputs either as `message.content` (JSON text)
    or as `message.tool_calls[0].function.arguments` when it uses tool-use under
    the hood. Try content first, fall back to the tool call.
    """
    content = getattr(message, "content", None)
    if content:
        return schema_cls.model_validate_json(content)
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        return schema_cls.model_validate_json(tool_calls[0].function.arguments)
    raise LLMException("Response had neither content nor tool_calls")
