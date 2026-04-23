"""
LLM layer for the Connect 4 arena, built on LiteLLM.

Single code path for OpenAI, Anthropic, and OpenRouter. Structured outputs via
a Pydantic schema passed through `response_format`; LiteLLM implements this as
tool-use under the hood for providers that lack native JSON-schema support.
Anthropic calls stream (to dodge the 10-min non-streaming timeout at high
max_tokens) and carry `thinking` + `output_config` so Opus 4.7 runs at full
reasoning strength.
"""

from abc import ABC
import logging
import os
import time
from typing import Dict, List, Self, Type

import litellm
from dotenv import load_dotenv
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
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

ANTHROPIC_MAX_TOKENS = 64_000
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
    """OpenAI models. Inherits the default `_send`."""

    model_names = ["gpt-5.4", "gpt-5", "gpt-5.4-mini", "gpt-5.4-nano"]


class Claude(LLM):
    """
    Anthropic models. Streams the response so we can set max_tokens high without
    hitting the 10-min non-streaming timeout. For Opus 4.7 we additionally enable
    adaptive thinking and `output_config={"effort": "max"}` so Opus runs at full
    reasoning strength. Older Claude models (Sonnet 4.5, Haiku 4.5, Opus 4.1)
    reject thinking when combined with tool_choice-forced structured outputs —
    LiteLLM implements `response_format` via forced tool use — so we skip those
    kwargs for them.
    """

    model_names = [
        "claude-opus-4-7",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ]

    def api_model_name(self) -> str:
        return f"anthropic/{super().api_model_name()}"

    def _is_opus_4_7(self) -> bool:
        return "opus-4-7" in self.api_model_name().lower()

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        kwargs = dict(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema_cls,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            stream=True,
        )
        if self._is_opus_4_7():
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": "max"}
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
    """

    model_names = ["moonshotai/kimi-k2.6", "openai/gpt-oss-120b"]

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
