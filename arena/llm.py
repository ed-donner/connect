"""
LLM layer for the Connect 4 arena, built on LiteLLM and routed through OpenRouter.

Every model goes through OpenRouter (the `openrouter/` prefix in LiteLLM), so
one key, `OPENROUTER_API_KEY`, covers them all. Structured outputs via a
Pydantic schema passed through `response_format`. Reasoning effort defaults to
low, with per-model overrides for models that are too slow at it.
"""

from abc import ABC, abstractmethod
import logging
import os
import time
from typing import Dict, List, Self, Type

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

logger = logging.getLogger(__name__)

RETRIES = 3
RETRY_DELAY_SECONDS = 2
REASONING_EFFORT = "low"


class LLMException(Exception):
    pass


class LLM(ABC):
    """Abstract superclass. Subclasses list their `model_names` and implement `_send`."""

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

    @abstractmethod
    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        pass

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


class OpenRouter(LLM):
    """
    Models accessed via OpenRouter. LiteLLM routes via the `openrouter/` prefix
    and reads `OPENROUTER_API_KEY` from the environment. Non-streaming.

    - `reasoning={"effort": ...}` is OpenRouter's normalized reasoning control,
      which it maps onto each provider's native knob. Used instead of
      `reasoning_effort=` because not every model lists that param (MiMo doesn't).
    - `provider={"require_parameters": True}` stops OpenRouter from routing to
      hosts that don't support every param we send. Several hosts lack
      structured outputs (e.g. Bedrock for Claude, DeepSeek's own API) and
      would return output that fails schema validation.

    The first two names are the default Red and Yellow players in the UI.
    """

    model_names = [
        "openai/gpt-6-luna",
        "anthropic/claude-haiku-5.5",
        "deepseek/deepseek-v4.1-flash-1017",
        "xiaomi/mimo-v2.6-flash",
        "google/gemini-3.5-flash-lite",
    ]

    # These reason for minutes per move, and their OpenRouter hosts ignore the
    # effort level: "minimal" and even `reasoning.max_tokens=1024` still produced
    # 9-48k reasoning tokens. Only "none" takes effect (0 reasoning tokens, ~2s).
    reasoning_efforts = {
        "deepseek/deepseek-v4.1-flash-1017": "none",
        "xiaomi/mimo-v2.6-flash": "none",
    }

    def api_model_name(self) -> str:
        return f"openrouter/{super().api_model_name()}"

    def reasoning_effort(self) -> str:
        return self.reasoning_efforts.get(self.model_name, REASONING_EFFORT)

    def _send(self, system: str, user: str, schema_cls: Type[BaseModel]) -> BaseModel:
        response = litellm.completion(
            model=self.api_model_name(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema_cls,
            reasoning={"effort": self.reasoning_effort()},
            provider={"require_parameters": True},
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
