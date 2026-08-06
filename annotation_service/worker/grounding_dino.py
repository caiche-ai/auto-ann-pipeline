from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image

from ..prompt_normalization import (
    GroundingPromptTranslator,
    PromptNormalizationMode,
    PromptNormalizationProfile,
    PromptTranslationFailurePolicy,
    normalize_grounding_prompt as normalize_prompt_result,
)
from ..schemas import AnnotationCategory


CATEGORY_ENTITIES: dict[AnnotationCategory, tuple[str, ...]] = {
    AnnotationCategory.HELMET_MISSING: ("person", "helmet"),
    AnnotationCategory.NO_HELMET: ("person", "helmet"),
    AnnotationCategory.NO_JACKET: (
        "person",
        "safety vest",
        "reflective vest",
    ),
    AnnotationCategory.HARNESS_MISSING: (
        "person",
        "safety harness",
    ),
    AnnotationCategory.EQUIPMENT_PROXIMITY: (
        "person",
        "excavator",
        "construction vehicle",
        "crane",
        "forklift",
    ),
    AnnotationCategory.OPENING_UNPROTECTED: (
        "opening",
        "floor hole",
        "cover",
        "guardrail",
        "barricade",
    ),
    AnnotationCategory.GUARDRAIL_MISSING: (
        "guardrail",
        "platform edge",
        "floor opening",
    ),
    AnnotationCategory.POOR_HOUSEKEEPING: (
        "construction material",
        "debris",
        "walkway",
    ),
    AnnotationCategory.SAFE: (
        "person",
        "helmet",
        "safety vest",
        "guardrail",
    ),
    AnnotationCategory.UNSAFE: (
        "person",
        "helmet",
        "safety vest",
        "safety harness",
        "construction vehicle",
        "excavator",
        "crane",
        "forklift",
        "opening",
        "floor hole",
        "platform edge",
        "guardrail",
        "cover",
        "barricade",
        "construction material",
        "debris",
        "walkway",
    ),
}


@dataclass(frozen=True)
class GroundingDINOModelConfig:
    root: Path
    config_path: Path
    checkpoint_path: Path
    bert_path: Path
    device: str = "cuda"
    model_version: str = "groundingdino-swint-ogc"
    prompt_version: str = "free-form-v1"
    prompt_normalization_mode: PromptNormalizationMode = "terminal_period"
    prompt_normalization_profile: PromptNormalizationProfile = (
        "construction_safety_v1"
    )
    prompt_translation_failure_policy: PromptTranslationFailurePolicy = (
        "fallback_canonical_terms"
    )
    box_threshold: float = 0.35
    text_threshold: float = 0.25


@dataclass(frozen=True)
class GroundingDINODetection:
    entity: str
    box_xyxy: tuple[float, float, float, float]
    box_score: float
    phrase_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_storage_payload(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "box_xyxy": list(self.box_xyxy),
            "box_score": self.box_score,
            "phrase_score": self.phrase_score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GroundingPromptPreparation:
    caption: str
    requested_entities: tuple[str, ...]
    requested_prompt: str
    metadata: dict[str, object]
    route: dict[str, bool] | None


class DetectionPredictor(Protocol):
    model_version: str
    prompt_version: str

    def predict(
        self,
        *,
        image_path: Path,
        width: int,
        height: int,
        prompt: str | None = None,
        categories: Sequence[str | AnnotationCategory] | None = None,
        prompt_normalization_mode: PromptNormalizationMode | None = None,
        prompt_normalization_profile: PromptNormalizationProfile | None = None,
        prompt_translation_failure_policy: (
            PromptTranslationFailurePolicy | None
        ) = None,
        prepared_prompt: GroundingPromptPreparation | None = None,
    ) -> list[GroundingDINODetection]:
        ...


def entities_for_categories(
    categories: Sequence[str | AnnotationCategory],
) -> tuple[str, ...]:
    entities: list[str] = []
    seen: set[str] = set()
    for value in categories:
        category = (
            value
            if isinstance(value, AnnotationCategory)
            else AnnotationCategory(str(value))
        )
        for entity in CATEGORY_ENTITIES[category]:
            if entity not in seen:
                seen.add(entity)
                entities.append(entity)
    return tuple(entities)


def build_caption(entities: Sequence[str]) -> str:
    normalized = [
        value.strip().strip(".")
        for value in entities
        if value.strip().strip(".")
    ]
    if not normalized:
        raise ValueError("GroundingDINO caption must not be empty")
    return " . ".join(normalized) + " ."


def normalize_grounding_prompt(prompt: str) -> str:
    """Apply only GroundingDINO's terminal-period convention."""

    return normalize_prompt_result(
        prompt,
        mode="terminal_period",
    ).normalized_prompt


def normalized_cxcywh_to_xyxy(
    box: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if len(box) != 4:
        raise ValueError("normalized box must contain four coordinates")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    center_x, center_y, box_width, box_height = [
        float(value) for value in box
    ]
    x1 = max(0.0, min(float(width), (center_x - box_width / 2) * width))
    y1 = max(
        0.0,
        min(float(height), (center_y - box_height / 2) * height),
    )
    x2 = max(0.0, min(float(width), (center_x + box_width / 2) * width))
    y2 = max(
        0.0,
        min(float(height), (center_y + box_height / 2) * height),
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _normalize_phrase(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().strip(".")
    return normalized or "unknown"


class GroundingDINOAdapter:
    """Lazy GroundingDINO adapter used only by the GPU worker process."""

    def __init__(
        self,
        config: GroundingDINOModelConfig,
        *,
        prompt_translator: GroundingPromptTranslator | None = None,
    ):
        self.config = config
        self.prompt_translator = prompt_translator
        self.model_version = config.model_version
        self.prompt_version = config.prompt_version
        self._model: Any | None = None
        self._torch: Any | None = None
        self._image_transform: Any | None = None
        self._get_phrases_from_posmap: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        for label, path in (
            ("GroundingDINO root", self.config.root),
            ("GroundingDINO config", self.config.config_path),
            ("GroundingDINO checkpoint", self.config.checkpoint_path),
            ("BERT directory", self.config.bert_path),
        ):
            expects_directory = "root" in label or "directory" in label
            expected = (
                path.is_dir() if expects_directory else path.is_file()
            )
            if not expected:
                raise FileNotFoundError(f"{label} not found: {path}")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        root_value = str(self.config.root)
        if root_value not in sys.path:
            sys.path.insert(0, root_value)

        import torch
        import groundingdino.datasets.transforms as transforms
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import (
            clean_state_dict,
            get_phrases_from_posmap,
        )

        args = SLConfig.fromfile(str(self.config.config_path))
        args.device = self.config.device
        args.text_encoder_type = str(self.config.bert_path)
        model = build_model(args)
        checkpoint = torch.load(
            str(self.config.checkpoint_path),
            map_location="cpu",
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError(
                "GroundingDINO checkpoint does not contain a model state"
            )
        model.load_state_dict(
            clean_state_dict(checkpoint["model"]),
            strict=False,
        )
        self._model = model.eval().to(self.config.device)
        self._torch = torch
        self._image_transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self._get_phrases_from_posmap = get_phrases_from_posmap

    def prepare_prompt(
        self,
        *,
        prompt: str | None = None,
        categories: Sequence[str | AnnotationCategory] | None = None,
        prompt_normalization_mode: PromptNormalizationMode | None = None,
        prompt_normalization_profile: PromptNormalizationProfile | None = None,
        prompt_translation_failure_policy: (
            PromptTranslationFailurePolicy | None
        ) = None,
    ) -> GroundingPromptPreparation:
        effective_mode = (
            prompt_normalization_mode
            if prompt_normalization_mode is not None
            else self.config.prompt_normalization_mode
        )
        effective_profile = (
            prompt_normalization_profile
            if prompt_normalization_profile is not None
            else self.config.prompt_normalization_profile
        )
        effective_failure_policy = (
            prompt_translation_failure_policy
            if prompt_translation_failure_policy is not None
            else self.config.prompt_translation_failure_policy
        )
        if prompt is not None:
            prompt_result = normalize_prompt_result(
                prompt,
                mode=effective_mode,
                profile=effective_profile,
                translator=self.prompt_translator,
                translation_failure_policy=effective_failure_policy,
            )
            return GroundingPromptPreparation(
                caption=prompt_result.normalized_prompt,
                requested_entities=(),
                requested_prompt=prompt_result.original_prompt,
                metadata=prompt_result.as_metadata(),
                route=prompt_result.as_route(),
            )
        requested_entities = entities_for_categories(categories or ())
        caption = build_caption(requested_entities)
        return GroundingPromptPreparation(
            caption=caption,
            requested_entities=requested_entities,
            requested_prompt=caption,
            metadata={
                "grounding_prompt_raw": caption,
                "grounding_prompt_normalized": caption,
                "grounding_prompt_normalization_mode": "categories",
                "grounding_prompt_normalization_profile": (
                    effective_profile
                    if effective_mode == "canonical_terms"
                    else None
                ),
                "grounding_prompt_applied_aliases": [],
            },
            route=None,
        )

    def predict(
        self,
        *,
        image_path: Path,
        width: int,
        height: int,
        prompt: str | None = None,
        categories: Sequence[str | AnnotationCategory] | None = None,
        prompt_normalization_mode: PromptNormalizationMode | None = None,
        prompt_normalization_profile: PromptNormalizationProfile | None = None,
        prompt_translation_failure_policy: (
            PromptTranslationFailurePolicy | None
        ) = None,
        prepared_prompt: GroundingPromptPreparation | None = None,
    ) -> list[GroundingDINODetection]:
        preparation = prepared_prompt or self.prepare_prompt(
            prompt=prompt,
            categories=categories,
            prompt_normalization_mode=prompt_normalization_mode,
            prompt_normalization_profile=prompt_normalization_profile,
            prompt_translation_failure_policy=(
                prompt_translation_failure_policy
            ),
        )
        caption = preparation.caption
        requested_entities = list(preparation.requested_entities)
        requested_prompt = preparation.requested_prompt
        prompt_metadata = preparation.metadata
        self.load()
        with Image.open(image_path) as source:
            image_source = source.convert("RGB")
        image, _ = self._image_transform(image_source, None)
        with self._torch.no_grad():
            outputs = self._model(
                image[None].to(self.config.device),
                captions=[caption],
            )
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
        prediction_boxes = outputs["pred_boxes"].cpu()[0]
        mask = (
            prediction_logits.max(dim=1)[0]
            > self.config.box_threshold
        )
        logits = prediction_logits[mask]
        boxes = prediction_boxes[mask]
        tokenizer = self._model.tokenizer
        tokenized = tokenizer(caption)
        phrases = [
            self._get_phrases_from_posmap(
                logit > self.config.text_threshold,
                tokenized,
                tokenizer,
            ).replace(".", "")
            for logit in logits
        ]
        scores = logits.max(dim=1)[0]
        detections: list[GroundingDINODetection] = []
        for box, score, phrase in zip(boxes, scores, phrases):
            box_values = (
                box.detach().cpu().tolist()
                if hasattr(box, "detach")
                else list(box)
            )
            xyxy = normalized_cxcywh_to_xyxy(
                box_values,
                width=width,
                height=height,
            )
            if xyxy is None:
                continue
            confidence = float(
                score.detach().cpu().item()
                if hasattr(score, "detach")
                else score
            )
            detections.append(
                GroundingDINODetection(
                    entity=_normalize_phrase(str(phrase)),
                    box_xyxy=xyxy,
                    box_score=confidence,
                    phrase_score=confidence,
                    metadata={
                        "raw_phrase": str(phrase),
                        "caption": caption,
                        "grounding_prompt": requested_prompt,
                        **prompt_metadata,
                        "requested_entities": requested_entities,
                        "model_version": self.model_version,
                        "prompt_version": self.prompt_version,
                        "box_threshold": self.config.box_threshold,
                        "text_threshold": self.config.text_threshold,
                        "score_semantics": "grounding_token_max",
                    },
                )
            )
        return detections
