from __future__ import annotations

import base64
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .qwen_contract import (
    QWEN_ENRICHMENT_PROMPT_VERSION,
    QWEN_FACTS_PROMPT_VERSION,
    QwenImageInput,
    QwenPromptSet,
    QwenVisualContext,
    QwenVisualFacts,
    build_prompt_enrichment_messages,
    build_visual_facts_messages,
    parse_prompt_set,
    parse_visual_facts,
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
    facts: QwenVisualFacts
    prompt_set: QwenPromptSet
    provider: str
    model: str
    facts_prompt_version: str
    enrichment_prompt_version: str

    def as_dict(self) -> dict[str, Any]:
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
        }


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
    """Two-stage Qwen2.5-VL generation over a vLLM-compatible endpoint."""

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
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
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
        return content

    def generate(
        self,
        *,
        context: QwenVisualContext,
        images: list[QwenImageInput],
    ) -> QwenGenerationResult:
        if not images:
            raise ValueError(
                "Qwen2.5-VL generation requires at least one image"
            )
        raw_facts = self._complete(
            messages=build_visual_facts_messages(
                context,
                images=images,
            ),
            temperature=self.config.facts_temperature,
        )
        facts = parse_visual_facts(raw_facts)
        raw_prompts = self._complete(
            messages=build_prompt_enrichment_messages(
                category=context.category,
                facts=facts,
            ),
            temperature=self.config.prompts_temperature,
        )
        prompt_set = parse_prompt_set(raw_prompts)
        return QwenGenerationResult(
            facts=facts,
            prompt_set=prompt_set,
            provider="vllm-openai-compatible",
            model=self.config.model,
            facts_prompt_version=QWEN_FACTS_PROMPT_VERSION,
            enrichment_prompt_version=QWEN_ENRICHMENT_PROMPT_VERSION,
        )
