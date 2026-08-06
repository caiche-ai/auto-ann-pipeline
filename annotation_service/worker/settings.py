from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _integer(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _floating(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


@dataclass(frozen=True)
class GroundingDINOWorkerSettings:
    storage_root: Path
    grounding_dino_root: Path
    config_path: Path
    checkpoint_path: Path
    bert_path: Path
    device: str
    model_version: str
    prompt_version: str
    box_threshold: float
    text_threshold: float
    worker_id: str
    lease_seconds: int
    heartbeat_seconds: int
    poll_seconds: float

    @classmethod
    def from_env(cls) -> "GroundingDINOWorkerSettings":
        grounding_root = Path(
            _required("ANNOTATION_GROUNDING_DINO_ROOT")
        ).expanduser().resolve()

        def model_path(name: str, default: str) -> Path:
            configured = Path(os.getenv(name, default).strip()).expanduser()
            if configured.is_absolute():
                return configured.resolve()
            return (grounding_root / configured).resolve()

        lease_seconds = _integer(
            "ANNOTATION_WORKER_LEASE_SECONDS",
            300,
            minimum=30,
            maximum=86_400,
        )
        heartbeat_seconds = _integer(
            "ANNOTATION_WORKER_HEARTBEAT_SECONDS",
            60,
            minimum=5,
            maximum=3_600,
        )
        if heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "ANNOTATION_WORKER_HEARTBEAT_SECONDS must be less than "
                "ANNOTATION_WORKER_LEASE_SECONDS"
            )
        worker_id = os.getenv(
            "ANNOTATION_WORKER_ID",
            f"{socket.gethostname()}-{os.getpid()}",
        ).strip()
        if not worker_id or len(worker_id) > 128:
            raise ValueError(
                "ANNOTATION_WORKER_ID must contain between 1 and 128 "
                "characters"
            )
        device = os.getenv(
            "ANNOTATION_GROUNDING_DINO_DEVICE",
            "cuda",
        ).strip()
        if not device:
            raise ValueError(
                "ANNOTATION_GROUNDING_DINO_DEVICE must not be empty"
            )
        model_version = os.getenv(
            "ANNOTATION_GROUNDING_DINO_MODEL_VERSION",
            "groundingdino-swint-ogc",
        ).strip()
        prompt_version = os.getenv(
            "ANNOTATION_GROUNDING_DINO_PROMPT_VERSION",
            "free-form-v1",
        ).strip()
        if not model_version or not prompt_version:
            raise ValueError(
                "GroundingDINO model and prompt versions must not be empty"
            )
        return cls(
            storage_root=Path(
                _required("ANNOTATION_STORAGE_ROOT")
            ).expanduser().resolve(),
            grounding_dino_root=grounding_root,
            config_path=model_path(
                "ANNOTATION_GROUNDING_DINO_CONFIG",
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ),
            checkpoint_path=model_path(
                "ANNOTATION_GROUNDING_DINO_CHECKPOINT",
                "weights/groundingdino_swint_ogc.pth",
            ),
            bert_path=model_path(
                "ANNOTATION_GROUNDING_DINO_BERT",
                "weights/bert-base-uncased",
            ),
            device=device,
            model_version=model_version,
            prompt_version=prompt_version,
            box_threshold=_floating(
                "ANNOTATION_GROUNDING_DINO_BOX_THRESHOLD",
                0.35,
                minimum=0.0,
                maximum=1.0,
            ),
            text_threshold=_floating(
                "ANNOTATION_GROUNDING_DINO_TEXT_THRESHOLD",
                0.25,
                minimum=0.0,
                maximum=1.0,
            ),
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            poll_seconds=_floating(
                "ANNOTATION_WORKER_POLL_SECONDS",
                2.0,
                minimum=0.1,
                maximum=60.0,
            ),
        )

    def validate_model_files(self) -> None:
        if not self.grounding_dino_root.is_dir():
            raise FileNotFoundError(
                f"GroundingDINO root not found: {self.grounding_dino_root}"
            )
        for label, path in (
            ("config", self.config_path),
            ("checkpoint", self.checkpoint_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"GroundingDINO {label} not found: {path}"
                )
        if not self.bert_path.is_dir():
            raise FileNotFoundError(
                f"GroundingDINO BERT directory not found: {self.bert_path}"
            )
        for filename in (
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        ):
            path = self.bert_path / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"GroundingDINO BERT file missing or empty: {path}"
                )
