from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol


PromptNormalizationMode = Literal[
    "off",
    "terminal_period",
    "canonical_terms",
    "llm_grounding_caption",
]

PromptNormalizationProfile = Literal[
    "construction_safety_v1",
    "open_semantic_zh_en_v1",
]

PromptTranslationFailurePolicy = Literal[
    "fail_job",
    "fallback_canonical_terms",
    "fallback_terminal_period",
]

PROMPT_NORMALIZATION_MODES = (
    "off",
    "terminal_period",
    "canonical_terms",
    "llm_grounding_caption",
)
PROMPT_NORMALIZATION_PROFILE_NAMES = (
    "construction_safety_v1",
    "open_semantic_zh_en_v1",
)
PROMPT_TRANSLATION_FAILURE_POLICIES = (
    "fail_job",
    "fallback_canonical_terms",
    "fallback_terminal_period",
)

_TRAILING_PERIOD = " ."

_CONSTRUCTION_SAFETY_V1: tuple[tuple[str, str], ...] = (
    ("安全帽", "helmet"),
    ("头盔", "helmet"),
    ("hard hat", "helmet"),
    ("safety helmet", "helmet"),
    ("护栏", "guardrail"),
    ("栏杆", "guardrail"),
    ("guard rail", "guardrail"),
    ("临边防护", "guardrail"),
    ("安全带", "safety harness"),
    ("安全绳", "safety harness"),
    ("反光背心", "safety vest"),
    ("荧光背心", "safety vest"),
    ("安全背心", "safety vest"),
    ("挖掘机", "excavator"),
    ("吊车", "crane"),
    ("叉车", "forklift"),
    ("洞口", "opening"),
    ("楼板洞口", "opening"),
    ("开口", "opening"),
    ("临边", "platform edge"),
    ("材料堆", "construction material"),
    ("杂物", "debris"),
    ("通道", "walkway"),
    ("施工人员", "person"),
    ("作业人员", "person"),
    ("工人", "person"),
    ("worker", "person"),
)

PROMPT_NORMALIZATION_PROFILES: dict[
    PromptNormalizationProfile,
    tuple[tuple[str, str], ...],
] = {
    "construction_safety_v1": _CONSTRUCTION_SAFETY_V1,
}


@dataclass(frozen=True)
class PromptTranslation:
    translated_prompt: str
    target_entities: tuple[str, ...]
    preserved_constraints: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provider: str = "openai-compatible"
    model: str = ""
    prompt_version: str = "open-semantic-zh-en-v1"
    latency_ms: float | None = None
    cache_hit: bool = False


class GroundingPromptTranslator(Protocol):
    def translate(
        self,
        prompt: str,
        *,
        profile: PromptNormalizationProfile,
    ) -> PromptTranslation:
        ...


class PromptRouteFailure(RuntimeError):
    """Translation failure that still exposes the completed routing trace."""

    def __init__(self, message: str):
        super().__init__(message)
        self.route = {
            "rule_attempted": True,
            "rule_matched": False,
            "llm_attempted": True,
            "llm_succeeded": False,
            "fallback_used": False,
        }


@dataclass(frozen=True)
class PromptNormalizationResult:
    original_prompt: str
    normalized_prompt: str
    mode: PromptNormalizationMode
    profile: PromptNormalizationProfile | None
    applied_aliases: tuple[tuple[str, str], ...] = ()
    translation_provider: str | None = None
    translation_model: str | None = None
    translation_prompt_version: str | None = None
    translation_latency_ms: float | None = None
    translation_cache_hit: bool = False
    translation_fallback_used: bool = False
    translation_fallback_mode: str | None = None
    translation_target_entities: tuple[str, ...] = ()
    translation_preserved_constraints: tuple[str, ...] = ()
    translation_warnings: tuple[str, ...] = ()

    def as_route(self) -> dict[str, bool]:
        rule_attempted = self.mode in {
            "canonical_terms",
            "llm_grounding_caption",
        }
        if self.mode == "canonical_terms":
            rule_matched = bool(self.applied_aliases)
        else:
            rule_matched = (
                self.translation_provider
                == "deterministic-direct-targets"
            )
        llm_attempted = (
            self.mode == "llm_grounding_caption"
            and not rule_matched
        )
        llm_succeeded = (
            llm_attempted
            and not self.translation_fallback_used
            and self.translation_provider is not None
        )
        return {
            "rule_attempted": rule_attempted,
            "rule_matched": rule_matched,
            "llm_attempted": llm_attempted,
            "llm_succeeded": llm_succeeded,
            "fallback_used": self.translation_fallback_used,
        }

    def as_metadata(self) -> dict[str, object]:
        return {
            "grounding_prompt_raw": self.original_prompt,
            "grounding_prompt_normalized": self.normalized_prompt,
            "grounding_prompt_normalization_mode": self.mode,
            "grounding_prompt_normalization_profile": self.profile,
            "grounding_prompt_applied_aliases": [
                {"source": source, "target": target}
                for source, target in self.applied_aliases
            ],
            "grounding_prompt_translation_provider": (
                self.translation_provider
            ),
            "grounding_prompt_translation_model": self.translation_model,
            "grounding_prompt_translation_prompt_version": (
                self.translation_prompt_version
            ),
            "grounding_prompt_translation_latency_ms": (
                self.translation_latency_ms
            ),
            "grounding_prompt_translation_cache_hit": (
                self.translation_cache_hit
            ),
            "grounding_prompt_translation_fallback_used": (
                self.translation_fallback_used
            ),
            "grounding_prompt_translation_fallback_mode": (
                self.translation_fallback_mode
            ),
            "grounding_prompt_translation_target_entities": list(
                self.translation_target_entities
            ),
            "grounding_prompt_translation_preserved_constraints": list(
                self.translation_preserved_constraints
            ),
            "grounding_prompt_translation_warnings": list(
                self.translation_warnings
            ),
        }


def _ensure_terminal_period(prompt: str) -> str:
    return prompt if prompt.endswith(".") else prompt + _TRAILING_PERIOD


def _replace_aliases(
    prompt: str,
    *,
    profile: PromptNormalizationProfile,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    replacements = PROMPT_NORMALIZATION_PROFILES.get(profile)
    if replacements is None:
        raise ValueError(
            f"unsupported prompt normalization profile: {profile}"
        )
    normalized = prompt
    applied: list[tuple[str, str]] = []
    for source, target in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source.isascii():
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            updated, count = pattern.subn(target, normalized)
        else:
            updated = normalized.replace(source, target)
            count = 1 if updated != normalized else 0
        if count > 0:
            applied.append((source, target))
            normalized = updated
    return normalized, tuple(applied)


def _direct_target_caption(
    prompt: str,
) -> tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]] | None:
    parts = [
        part.strip().strip("。.!！?？")
        for part in re.split(r"[,，、;；]+", prompt)
    ]
    if not parts or any(not part for part in parts):
        return None
    alias_map: dict[str, tuple[str, str]] = {}
    for source, target in _CONSTRUCTION_SAFETY_V1:
        alias_map[re.sub(r"\s+", " ", source).casefold()] = (
            source,
            target,
        )
        alias_map[target.casefold()] = (target, target)
    targets: list[str] = []
    applied: list[tuple[str, str]] = []
    for part in parts:
        lookup = re.sub(r"\s+", " ", part).casefold()
        match = alias_map.get(lookup)
        if match is None:
            return None
        _, target = match
        applied.append((part, target))
        if target not in targets:
            targets.append(target)
    return " . ".join(targets), tuple(applied), tuple(targets)


def _translation_fallback(
    prompt: str,
    *,
    policy: PromptTranslationFailurePolicy,
    profile: PromptNormalizationProfile,
    error: Exception,
) -> PromptNormalizationResult:
    warning = f"prompt translation failed: {type(error).__name__}"
    if policy == "fail_job":
        raise PromptRouteFailure(warning) from error
    if policy == "fallback_canonical_terms":
        normalized, applied = _replace_aliases(
            prompt,
            profile="construction_safety_v1",
        )
        fallback_mode = "canonical_terms"
    elif policy == "fallback_terminal_period":
        normalized = prompt
        applied = ()
        fallback_mode = "terminal_period"
    else:
        raise ValueError(
            f"unsupported prompt translation failure policy: {policy}"
        )
    return PromptNormalizationResult(
        original_prompt=prompt,
        normalized_prompt=_ensure_terminal_period(normalized),
        mode="llm_grounding_caption",
        profile=profile,
        applied_aliases=applied,
        translation_fallback_used=True,
        translation_fallback_mode=fallback_mode,
        translation_warnings=(warning,),
    )


def normalize_grounding_prompt(
    prompt: str,
    *,
    mode: PromptNormalizationMode = "terminal_period",
    profile: PromptNormalizationProfile = "construction_safety_v1",
    translator: GroundingPromptTranslator | None = None,
    translation_failure_policy: PromptTranslationFailurePolicy = (
        "fallback_canonical_terms"
    ),
) -> PromptNormalizationResult:
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("GroundingDINO prompt must not be blank")
    if mode == "off":
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=normalized,
            mode=mode,
            profile=None,
        )
    if mode == "terminal_period":
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=_ensure_terminal_period(normalized),
            mode=mode,
            profile=None,
        )
    if mode == "canonical_terms":
        canonical, applied = _replace_aliases(
            normalized,
            profile=profile,
        )
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=_ensure_terminal_period(canonical),
            mode=mode,
            profile=profile,
            applied_aliases=applied,
        )
    if mode == "llm_grounding_caption":
        if profile != "open_semantic_zh_en_v1":
            raise ValueError(
                "llm_grounding_caption requires profile "
                "open_semantic_zh_en_v1"
            )
        direct = _direct_target_caption(normalized)
        if direct is not None:
            caption, applied, targets = direct
            return PromptNormalizationResult(
                original_prompt=normalized,
                normalized_prompt=_ensure_terminal_period(caption),
                mode=mode,
                profile=profile,
                applied_aliases=applied,
                translation_provider="deterministic-direct-targets",
                translation_prompt_version="direct-targets-v1",
                translation_target_entities=targets,
            )
        if translator is None:
            return _translation_fallback(
                normalized,
                policy=translation_failure_policy,
                profile=profile,
                error=RuntimeError("prompt translator is not configured"),
            )
        try:
            translated = translator.translate(
                normalized,
                profile=profile,
            )
            translated_prompt = re.sub(
                r"\s+",
                " ",
                translated.translated_prompt,
            ).strip().rstrip(".")
            if not translated_prompt:
                raise ValueError("translated prompt must not be blank")
            if len(translated_prompt) > 1000:
                raise ValueError(
                    "translated prompt must contain at most 1000 characters"
                )
        except Exception as exc:
            return _translation_fallback(
                normalized,
                policy=translation_failure_policy,
                profile=profile,
                error=exc,
            )
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=_ensure_terminal_period(
                translated_prompt
            ),
            mode=mode,
            profile=profile,
            translation_provider=translated.provider,
            translation_model=translated.model or None,
            translation_prompt_version=translated.prompt_version,
            translation_latency_ms=translated.latency_ms,
            translation_cache_hit=translated.cache_hit,
            translation_target_entities=translated.target_entities,
            translation_preserved_constraints=(
                translated.preserved_constraints
            ),
            translation_warnings=translated.warnings,
        )
    raise ValueError(
        f"unsupported prompt normalization mode: {mode}"
    )
