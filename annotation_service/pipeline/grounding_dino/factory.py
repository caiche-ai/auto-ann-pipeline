from __future__ import annotations

from .adapter import GroundingDINOAdapter, GroundingDINOModelConfig
from .remote import RemoteGroundingDINOPredictor
from .settings import GroundingDINOWorkerSettings


def build_detection_predictor(settings: GroundingDINOWorkerSettings):
    translator = settings.prompt_translator()
    if settings.provider == "remote":
        return RemoteGroundingDINOPredictor(
            settings.remote_config(),
            model_version=settings.model_version,
            prompt_version=settings.prompt_version,
            prompt_normalization_mode=settings.prompt_normalization_mode,
            prompt_normalization_profile=(settings.prompt_normalization_profile),
            prompt_translation_failure_policy=(
                settings.prompt_translation_failure_policy
            ),
            prompt_translator=translator,
        )
    if (
        settings.grounding_dino_root is None
        or settings.config_path is None
        or settings.checkpoint_path is None
        or settings.bert_path is None
    ):
        raise ValueError("local GroundingDINO model paths are incomplete")
    return GroundingDINOAdapter(
        GroundingDINOModelConfig(
            root=settings.grounding_dino_root,
            config_path=settings.config_path,
            checkpoint_path=settings.checkpoint_path,
            bert_path=settings.bert_path,
            device=settings.device,
            model_version=settings.model_version,
            prompt_version=settings.prompt_version,
            prompt_normalization_mode=settings.prompt_normalization_mode,
            prompt_normalization_profile=(settings.prompt_normalization_profile),
            prompt_translation_failure_policy=(
                settings.prompt_translation_failure_policy
            ),
            box_threshold=settings.box_threshold,
            text_threshold=settings.text_threshold,
        ),
        prompt_translator=translator,
    )
