from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .qwen_worker import OperationHeartbeat
from .sam_adapter import SAMAdapter, SAMMaskCandidate, SAMModelConfig
from .storage import AnnotationStore


LOGGER = logging.getLogger(__name__)


class MaskPredictor(Protocol):
    def predict(
        self,
        *,
        image_path: Path,
        box_xyxy: list[float],
    ) -> SAMMaskCandidate:
        ...


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


@dataclass(frozen=True)
class SAMWorkerSettings:
    storage_root: Path
    checkpoint_path: Path
    model_type: str
    device: str
    python_package: str
    model_version: str
    polygon_epsilon: float
    worker_id: str
    lease_seconds: int
    heartbeat_seconds: int
    poll_seconds: float

    @classmethod
    def from_env(cls) -> "SAMWorkerSettings":
        storage_root = os.getenv("ANNOTATION_STORAGE_ROOT", "").strip()
        checkpoint = os.getenv(
            "ANNOTATION_SAM_CHECKPOINT",
            "",
        ).strip()
        if not storage_root:
            raise ValueError("ANNOTATION_STORAGE_ROOT must not be empty")
        if not checkpoint:
            raise ValueError("ANNOTATION_SAM_CHECKPOINT must not be empty")
        lease = _integer(
            "ANNOTATION_SAM_LEASE_SECONDS",
            300,
            30,
            3600,
        )
        heartbeat = _integer(
            "ANNOTATION_SAM_HEARTBEAT_SECONDS",
            60,
            5,
            600,
        )
        if heartbeat >= lease:
            raise ValueError(
                "ANNOTATION_SAM_HEARTBEAT_SECONDS must be less than lease"
            )
        return cls(
            storage_root=Path(storage_root).expanduser().resolve(),
            checkpoint_path=Path(checkpoint).expanduser().resolve(),
            model_type=os.getenv(
                "ANNOTATION_SAM_MODEL_TYPE",
                "vit_h",
            ).strip(),
            device=os.getenv(
                "ANNOTATION_SAM_DEVICE",
                "cuda",
            ).strip(),
            python_package=os.getenv(
                "ANNOTATION_SAM_PYTHON_PACKAGE",
                "model.segment_anything",
            ).strip(),
            model_version=os.getenv(
                "ANNOTATION_SAM_MODEL_VERSION",
                "sam-vit-h-4b8939",
            ).strip(),
            polygon_epsilon=float(
                os.getenv("ANNOTATION_SAM_POLYGON_EPSILON", "1.0")
            ),
            worker_id=os.getenv(
                "ANNOTATION_SAM_WORKER_ID",
                f"{socket.gethostname()}-{os.getpid()}",
            ).strip(),
            lease_seconds=lease,
            heartbeat_seconds=heartbeat,
            poll_seconds=float(
                os.getenv("ANNOTATION_SAM_POLL_SECONDS", "2")
            ),
        )

    def model_config(self) -> SAMModelConfig:
        return SAMModelConfig(
            checkpoint_path=self.checkpoint_path,
            model_type=self.model_type,
            device=self.device,
            python_package=self.python_package,
            model_version=self.model_version,
            polygon_epsilon=self.polygon_epsilon,
        )


def persist_sam_candidate(
    store: AnnotationStore,
    *,
    task_id: str,
    operation_id: str | None,
    candidate: SAMMaskCandidate,
) -> dict:
    metadata = {
        "box_xyxy": candidate.box_xyxy,
        "predicted_iou": candidate.predicted_iou,
        "mask_area_pixels": candidate.mask_area_pixels,
        "sam_version": candidate.model_version,
    }
    mask = store.store_artifact(
        task_id=task_id,
        artifact_type="mask",
        data=candidate.mask_png,
        media_type="image/png",
        operation_id=operation_id,
        metadata=metadata,
    )
    overlay = store.store_artifact(
        task_id=task_id,
        artifact_type="mask-overlay",
        data=candidate.overlay_png,
        media_type="image/png",
        operation_id=operation_id,
        metadata=metadata,
    )
    crop = store.store_artifact(
        task_id=task_id,
        artifact_type="crop",
        data=candidate.crop_png,
        media_type="image/png",
        operation_id=operation_id,
        metadata=metadata,
    )
    return {
        "box_xyxy": candidate.box_xyxy,
        "predicted_iou": candidate.predicted_iou,
        "mask_area_pixels": candidate.mask_area_pixels,
        "shapes": candidate.shapes,
        "artifacts": {
            "mask": mask["url"],
            "mask_overlay": overlay["url"],
            "crop": crop["url"],
        },
        "provenance": {
            "sam_version": candidate.model_version,
        },
    }


class SAMMaskWorker:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        predictor: MaskPredictor,
        worker_id: str,
        lease_seconds: int = 300,
        heartbeat_seconds: int = 60,
        poll_seconds: float = 2.0,
    ):
        if heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be less than lease_seconds"
            )
        self.store = store
        self.predictor = predictor
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds

    def run_once(self) -> bool:
        operation = self.store.claim_next_operation(
            operation_type="mask_candidate",
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if operation is None:
            return False
        self._process(operation)
        return True

    def run_forever(
        self,
        stop_event: threading.Event | None = None,
    ) -> None:
        stop = stop_event or threading.Event()
        while not stop.is_set():
            try:
                processed = self.run_once()
            except Exception:
                LOGGER.exception("unexpected SAM worker loop failure")
                processed = False
            if not processed:
                stop.wait(self.poll_seconds)

    def _process(self, operation: dict) -> None:
        operation_id = operation["operation_id"]
        heartbeat = OperationHeartbeat(
            store=self.store,
            operation_id=operation_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_seconds,
        )
        heartbeat.start()
        try:
            task = self.store.get_task(operation["task_id"])
            if task["version"] != operation["task_version"]:
                raise ValueError(
                    "task version changed after mask operation was queued"
                )
            image_path, _ = self.store.asset_file(
                task["asset"]["asset_id"]
            )
            candidate = self.predictor.predict(
                image_path=image_path,
                box_xyxy=operation["request"]["box_xyxy"],
            )
            heartbeat.ensure_healthy()
            if self.store.get_operation(operation_id)["status"] != "running":
                raise ValueError(
                    "mask operation was cancelled before artifacts were saved"
                )
            result = persist_sam_candidate(
                self.store,
                task_id=task["task_id"],
                operation_id=operation_id,
                candidate=candidate,
            )
            self.store.complete_operation(
                operation_id,
                worker_id=self.worker_id,
                result=result,
            )
        except Exception as exc:
            LOGGER.exception(
                "SAM mask generation failed",
                extra={"operation_id": operation_id},
            )
            try:
                self.store.fail_operation(
                    operation_id,
                    worker_id=self.worker_id,
                    code="model_unavailable",
                    message=str(exc),
                )
            except Exception:
                LOGGER.exception(
                    "failed to persist SAM operation error",
                    extra={"operation_id": operation_id},
                )
        finally:
            heartbeat.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAM mask candidate worker",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = SAMWorkerSettings.from_env()
    config = settings.model_config()
    config.validate()
    store = AnnotationStore(settings.storage_root)
    store.initialize()
    worker = SAMMaskWorker(
        store=store,
        predictor=SAMAdapter(config),
        worker_id=settings.worker_id,
        lease_seconds=settings.lease_seconds,
        heartbeat_seconds=settings.heartbeat_seconds,
        poll_seconds=settings.poll_seconds,
    )
    try:
        if args.once:
            return 0 if worker.run_once() else 3
        worker.run_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
