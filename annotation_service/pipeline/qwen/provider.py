from __future__ import annotations

import base64
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .contract import (
    QWEN_ENRICHMENT_PROMPT_VERSION,
    QWEN_JOINT_ENRICHMENT_PROMPT_VERSION,
    QwenImageInput,
    QwenJointVisualFacts,
    QwenJointVisualContext,
    QwenPromptSet,
    QwenVisualContext,
    QwenVisualFacts,
    build_direct_joint_prompt_messages,
    build_direct_prompt_messages,
    parse_prompt_set,
)


class QwenProviderError(RuntimeError):
    """Raised when the OpenAI-compatible Qwen endpoint cannot be used."""


Transport = Callable[
    [str, dict[str, str], bytes, float],
    dict[str, Any],
]


@dataclass(frozen=True)
class QwenProviderConfig:
    base_url: str
    model: str = "qwen25vl"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_tokens: int = 1200
    facts_temperature: float = 0.1
    prompts_temperature: float = 0.3

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "Qwen base_url must be an absolute HTTP(S) URL"
            )
        if not self.model.strip():
            raise ValueError("Qwen model must not be blank")
        if not 1 <= self.max_tokens <= 16_384:
            raise ValueError("Qwen max_tokens must be between 1 and 16384")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ValueError(
                "Qwen timeout_seconds must be between 1 and 3600"
            )
        for name, value in (
            ("facts_temperature", self.facts_temperature),
            ("prompts_temperature", self.prompts_temperature),
        ):
            if not 0 <= value <= 2:
                raise ValueError(f"{name} must be between 0 and 2")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


@dataclass(frozen=True)
class QwenGenerationResult:
    facts: QwenVisualFacts | QwenJointVisualFacts | None
    prompt_set: QwenPromptSet
    provider: str
    model: str
    facts_prompt_version: str
    enrichment_prompt_version: str
    timings_ms: dict[str, float | int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        if self.facts is None:
            facts = None
        else:
            model_dump = getattr(self.facts, "model_dump", None)
            facts = (
                model_dump(mode="json")
                if callable(model_dump)
                else json.loads(self.facts.json())
            )
        prompt_dump = getattr(self.prompt_set, "model_dump", None)
        prompt_set = (
            prompt_dump(mode="json")
            if callable(prompt_dump)
            else json.loads(self.prompt_set.json())
        )
        return {
            "facts": facts,
            **prompt_set,
            "provenance": {
                "qwen_provider": self.provider,
                "qwen_model": self.model,
                "qwen_facts_prompt_version": self.facts_prompt_version,
                "qwen_enrichment_prompt_version": (
                    self.enrichment_prompt_version
                ),
            },
            "timings_ms": self.timings_ms,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class _QwenCompletion:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)


def image_file_to_input(
    path: Path,
    *,
    media_type: str,
    label: str,
) -> QwenImageInput:
    if media_type not in {"image/jpeg", "image/png"}:
        raise ValueError("Qwen image media type must be JPEG or PNG")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Qwen image is empty: {path.name}")
    encoded = base64.b64encode(raw).decode("ascii")
    return QwenImageInput(
        label=label,
        media_type=media_type,
        data_url=f"data:{media_type};base64,{encoded}",
    )


def _default_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise QwenProviderError(
            f"Qwen endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        raise QwenProviderError("Qwen endpoint is unavailable") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QwenProviderError(
            "Qwen endpoint returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise QwenProviderError(
            "Qwen endpoint response must be a JSON object"
        )
    return payload


class Qwen25VLProvider:
    """Generate final prompts directly with an OpenAI-compatible Qwen-VL."""

    def __init__(
        self,
        config: QwenProviderConfig,
        *,
        transport: Transport | None = None,
    ):
        self.config = config
        self._transport = transport or _default_transport

    def _complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        force_json_response: bool,
    ) -> _QwenCompletion:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
        }
        if force_json_response:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = self._transport(
            self.config.chat_completions_url,
            headers,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            self.config.timeout_seconds,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenProviderError(
                "Qwen response does not contain assistant content"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise QwenProviderError("Qwen assistant content is empty")
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return _QwenCompletion(content=content, usage=usage)

    def generate(
        self,
        *,
        context: QwenVisualContext,
        images: list[QwenImageInput],
        custom_instruction: str | None = None,
    ) -> QwenGenerationResult:
        if not images:
            raise ValueError(
                "Qwen-VL generation requires at least one image"
            )
        total_started = time.perf_counter()
        generation_started = time.perf_counter()
        completion = self._complete(
            messages=build_direct_prompt_messages(
                context,
                custom_instruction=custom_instruction,
                images=images,
            ),
            temperature=self.config.prompts_temperature,
            force_json_response=custom_instruction is None,
        )
        generation_ms = (
            time.perf_counter() - generation_started
        ) * 1000
        prompt_set = parse_prompt_set(completion.content)
        return QwenGenerationResult(
            facts=None,
            prompt_set=prompt_set,
            provider="vllm-openai-compatible",
            model=self.config.model,
            facts_prompt_version="not-used-direct-generation",
            enrichment_prompt_version=QWEN_ENRICHMENT_PROMPT_VERSION,
            timings_ms={
                "model_calls": 1,
                "generation_call_ms": round(generation_ms, 3),
                "prompt_call_ms": round(generation_ms, 3),
                "total_ms": round(
                    (time.perf_counter() - total_started) * 1000,
                    3,
                ),
            },
            usage=completion.usage,
        )

    def generate_joint(
        self,
        *,
        context: QwenJointVisualContext,
        images: list[QwenImageInput],
        custom_instruction: str | None = None,
    ) -> QwenGenerationResult:
        if not images:
            raise ValueError(
                "joint Qwen-VL generation requires images"
            )
        total_started = time.perf_counter()
        generation_started = time.perf_counter()
        completion = self._complete(
            messages=build_direct_joint_prompt_messages(
                context,
                custom_instruction=custom_instruction,
                images=images,
            ),
            temperature=self.config.prompts_temperature,
            force_json_response=custom_instruction is None,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        prompt_set = parse_prompt_set(completion.content)
        return QwenGenerationResult(
            facts=None,
            prompt_set=prompt_set,
            provider="vllm-openai-compatible",
            model=self.config.model,
            facts_prompt_version="not-used-direct-generation",
            enrichment_prompt_version=(
                QWEN_JOINT_ENRICHMENT_PROMPT_VERSION
            ),
            timings_ms={
                "model_calls": 1,
                "generation_call_ms": round(generation_ms, 3),
                "prompt_call_ms": round(generation_ms, 3),
                "total_ms": round(
                    (time.perf_counter() - total_started) * 1000,
                    3,
                ),
            },
            usage=completion.usage,
        )
