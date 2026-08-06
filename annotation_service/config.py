from __future__ import annotations

import os
from dataclasses import dataclass

from .prompt_normalization import (
    PROMPT_NORMALIZATION_MODES,
    PROMPT_NORMALIZATION_PROFILE_NAMES,
    PROMPT_TRANSLATION_FAILURE_POLICIES,
)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _get_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_origins(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    origins: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin or origin in seen:
            continue
        seen.add(origin)
        origins.append(origin)
    return tuple(origins)


def _get_optional_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip() or None


@dataclass(frozen=True)
class Settings:
    service_version: str = "1.5.1"
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    cors_allow_credentials: bool = False
    max_request_bytes: int = 30 * 1024 * 1024
    max_image_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    max_metadata_chars: int = 16_384
    max_queued_jobs: int = 100
    docs_enabled: bool = True
    storage_enabled: bool = False
    storage_root: str = "./annotation-data"
    prompt_normalization_mode: str = "terminal_period"
    prompt_normalization_profile: str = "construction_safety_v1"
    prompt_translation_failure_policy: str = (
        "fallback_canonical_terms"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            service_version=os.getenv(
                "ANNOTATION_SERVICE_VERSION", "1.5.1"
            ).strip(),
            api_key=_get_optional_secret("ANNOTATION_API_KEY"),
            cors_origins=_get_origins("ANNOTATION_CORS_ORIGINS"),
            cors_allow_credentials=_get_bool(
                "ANNOTATION_CORS_ALLOW_CREDENTIALS", False
            ),
            max_request_bytes=_get_int(
                "ANNOTATION_MAX_REQUEST_BYTES",
                30 * 1024 * 1024,
            ),
            max_image_bytes=_get_int(
                "ANNOTATION_MAX_IMAGE_BYTES",
                20 * 1024 * 1024,
            ),
            max_image_pixels=_get_int(
                "ANNOTATION_MAX_IMAGE_PIXELS",
                25_000_000,
            ),
            max_metadata_chars=_get_int(
                "ANNOTATION_MAX_METADATA_CHARS",
                16_384,
            ),
            max_queued_jobs=_get_int(
                "ANNOTATION_MAX_QUEUED_JOBS",
                100,
            ),
            docs_enabled=_get_bool("ANNOTATION_DOCS_ENABLED", True),
            storage_enabled=_get_bool("ANNOTATION_STORAGE_ENABLED", False),
            storage_root=os.getenv(
                "ANNOTATION_STORAGE_ROOT", "./annotation-data"
            ).strip(),
            prompt_normalization_mode=os.getenv(
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE",
                "terminal_period",
            ).strip(),
            prompt_normalization_profile=os.getenv(
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE",
                "construction_safety_v1",
            ).strip(),
            prompt_translation_failure_policy=os.getenv(
                "ANNOTATION_GROUNDING_DINO_PROMPT_TRANSLATION_FAILURE_POLICY",
                "fallback_canonical_terms",
            ).strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.service_version:
            raise ValueError("ANNOTATION_SERVICE_VERSION must not be empty")
        if self.max_request_bytes <= self.max_image_bytes:
            raise ValueError(
                "ANNOTATION_MAX_REQUEST_BYTES must be greater than "
                "ANNOTATION_MAX_IMAGE_BYTES to allow multipart metadata"
            )
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "credentialed CORS must use explicit origins, not '*'"
            )
        if self.storage_enabled and not self.storage_root:
            raise ValueError(
                "ANNOTATION_STORAGE_ROOT must not be empty when storage is enabled"
            )
        if self.prompt_normalization_mode not in PROMPT_NORMALIZATION_MODES:
            raise ValueError(
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE must be "
                f"one of: {', '.join(PROMPT_NORMALIZATION_MODES)}"
            )
        if not self.prompt_normalization_profile:
            raise ValueError(
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE must "
                "not be empty"
            )
        if (
            self.prompt_normalization_profile
            not in PROMPT_NORMALIZATION_PROFILE_NAMES
        ):
            raise ValueError(
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE must "
                "be one of: "
                f"{', '.join(PROMPT_NORMALIZATION_PROFILE_NAMES)}"
            )
        expected_profile = {
            "canonical_terms": "construction_safety_v1",
            "llm_grounding_caption": "open_semantic_zh_en_v1",
        }.get(self.prompt_normalization_mode)
        if (
            expected_profile is not None
            and self.prompt_normalization_profile != expected_profile
        ):
            raise ValueError(
                f"{self.prompt_normalization_mode} requires profile "
                f"{expected_profile}"
            )
        if (
            self.prompt_translation_failure_policy
            not in PROMPT_TRANSLATION_FAILURE_POLICIES
        ):
            raise ValueError(
                "ANNOTATION_GROUNDING_DINO_PROMPT_TRANSLATION_FAILURE_POLICY "
                "must be one of: "
                f"{', '.join(PROMPT_TRANSLATION_FAILURE_POLICIES)}"
            )
