from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from ...api.schemas import AnnotationCategory
from ..remote import (
    RemoteProviderConfig,
    RemoteProviderError,
    RemoteTransport,
    default_remote_transport,
    image_payload,
    post_json,
)
from .adapter import (
    GroundingDINODetection,
    GroundingPromptPreparation,
    prepare_grounding_prompt,
)
from .prompt import (
    GroundingPromptTranslator,
    PromptNormalizationMode,
    PromptNormalizationProfile,
    PromptTranslationFailurePolicy,
)


class RemoteGroundingDINOPredictor:
    """GroundingDINO predictor backed by the standalone inference API."""

    def __init__(
        self,
        config: RemoteProviderConfig,
        *,
        model_version: str = "groundingdino-swint-ogc",
        prompt_version: str = "free-form-v1",
        prompt_normalization_mode: PromptNormalizationMode = "terminal_period",
        prompt_normalization_profile: PromptNormalizationProfile = (
            "construction_safety_v1"
        ),
        prompt_translation_failure_policy: PromptTranslationFailurePolicy = (
            "fallback_canonical_terms"
        ),
        prompt_translator: GroundingPromptTranslator | None = None,
        transport: RemoteTransport | None = None,
    ):
        self.config = config
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.prompt_normalization_mode = prompt_normalization_mode
        self.prompt_normalization_profile = prompt_normalization_profile
        self.prompt_translation_failure_policy = prompt_translation_failure_policy
        self.prompt_translator = prompt_translator
        self._transport = transport or default_remote_transport

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
        return prepare_grounding_prompt(
            prompt=prompt,
            categories=categories,
            prompt_normalization_mode=(
                prompt_normalization_mode or self.prompt_normalization_mode
            ),
            prompt_normalization_profile=(
                prompt_normalization_profile or self.prompt_normalization_profile
            ),
            prompt_translation_failure_policy=(
                prompt_translation_failure_policy
                or self.prompt_translation_failure_policy
            ),
            prompt_translator=self.prompt_translator,
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
            prompt_translation_failure_policy=(prompt_translation_failure_policy),
        )
        media_type, encoded = image_payload(
            image_path,
            max_bytes=self.config.max_image_bytes,
        )
        response = post_json(
            config=self.config,
            path="/v1/inference/grounding-dino",
            payload={
                "image_base64": encoded,
                "media_type": media_type,
                "width": width,
                "height": height,
                "prepared_prompt": {
                    "caption": preparation.caption,
                    "requested_entities": list(preparation.requested_entities),
                    "requested_prompt": preparation.requested_prompt,
                    "metadata": preparation.metadata,
                    "route": preparation.route,
                },
            },
            transport=self._transport,
        )
        detections = response.get("detections")
        if not isinstance(detections, list):
            raise RemoteProviderError(
                "remote GroundingDINO response must contain detections"
            )
        remote_model = response.get("model_version")
        remote_prompt = response.get("prompt_version")
        if isinstance(remote_model, str) and remote_model.strip():
            self.model_version = remote_model.strip()
        if isinstance(remote_prompt, str) and remote_prompt.strip():
            self.prompt_version = remote_prompt.strip()
        return [
            self._parse_detection(item, width=width, height=height)
            for item in detections
        ]

    @staticmethod
    def _parse_detection(
        payload: Any,
        *,
        width: int,
        height: int,
    ) -> GroundingDINODetection:
        if not isinstance(payload, dict):
            raise RemoteProviderError("remote detection must be an object")
        try:
            entity = str(payload["entity"]).strip()
            box = tuple(float(value) for value in payload["box_xyxy"])
            box_score = float(payload["box_score"])
            phrase_score = float(payload["phrase_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteProviderError("remote detection is malformed") from exc
        if not entity or len(entity) > 300 or len(box) != 4:
            raise RemoteProviderError("remote detection is malformed")
        if not all(math.isfinite(value) for value in box):
            raise RemoteProviderError("remote detection box is not finite")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise RemoteProviderError("remote detection box is out of bounds")
        if (
            not math.isfinite(box_score)
            or not math.isfinite(phrase_score)
            or not 0 <= box_score <= 1
            or not 0 <= phrase_score <= 1
        ):
            raise RemoteProviderError("remote detection score is out of range")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RemoteProviderError("remote detection metadata must be an object")
        return GroundingDINODetection(
            entity=entity,
            box_xyxy=box,  # type: ignore[arg-type]
            box_score=box_score,
            phrase_score=phrase_score,
            metadata={**metadata, "inference_provider": "remote"},
        )
