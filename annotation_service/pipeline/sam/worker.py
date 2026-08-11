from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, replace
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ..runtime import OperationHeartbeat
from .adapter import (
    SAMAdapter,
    SAMMaskCandidate,
    SAMModelConfig,
    render_sam_candidate,
)
from ...storage.repository import AnnotationStore


LOGGER = logging.getLogger(__name__)


def combine_sam_candidates(
    *,
    image_path: Path,
    candidates: list[SAMMaskCandidate],
    detection_ids: list[str] | None = None,
    polygon_epsilon: float = 1.0,
) -> SAMMaskCandidate:
    """Combine per-box SAM results into one sample-level candidate."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    ids = detection_ids or []
    if ids and len(ids) != len(candidates):
        raise ValueError("detection_ids must match SAM candidates")
    if len(candidates) == 1:
        candidate = candidates[0]
        detection_id = ids[0] if ids else None
        shapes = []
        for shape in candidate.shapes:
            tagged = dict(shape)
            if detection_id is not None:
                tagged["source_detection_id"] = detection_id
            shapes.append(tagged)
        return replace(
            candidate,
            shapes=shapes,
            timings_ms={
                **candidate.timings_ms,
                "instance_count": 1,
                "combined_mask": False,
            },
        )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    union_mask: np.ndarray | None = None
    shapes = []
    for instance_index, candidate in enumerate(candidates, start=1):
        with Image.open(BytesIO(candidate.mask_png)) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        union_mask = mask if union_mask is None else union_mask | mask
        detection_id = ids[instance_index - 1] if ids else None
        for shape_index, shape in enumerate(candidate.shapes, start=1):
            combined_shape = dict(shape)
            combined_shape["shape_id"] = (
                f"sam-instance-{instance_index}-{shape_index}"
            )
            if detection_id is not None:
                combined_shape["source_detection_id"] = detection_id
            shapes.append(combined_shape)
    boxes = [candidate.box_xyxy for candidate in candidates]
    union_box = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    combined = render_sam_candidate(
        image=image,
        mask=union_mask,
        box_xyxy=union_box,
        predicted_iou=sum(
            candidate.predicted_iou for candidate in candidates
        )
        / len(candidates),
        model_version=candidates[0].model_version,
        polygon_epsilon=polygon_epsilon,
    )
    return replace(
        combined,
        shapes=shapes,
        timings_ms={
            **candidates[0].timings_ms,
            "instance_count": len(candidates),
            "combined_mask": True,
        },
    )


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
    max_batch_size: int
    image_embedding_cache_size: int
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
            86_400,
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
                "third_party.segment_anything",
            ).strip(),
            model_version=os.getenv(
                "ANNOTATION_SAM_MODEL_VERSION",
                "sam-vit-h-4b8939",
            ).strip(),
            polygon_epsilon=float(
                os.getenv("ANNOTATION_SAM_POLYGON_EPSILON", "1.0")
            ),
            max_batch_size=_integer(
                "ANNOTATION_SAM_MAX_BATCH_SIZE",
                16,
                1,
                128,
            ),
            image_embedding_cache_size=_integer(
                "ANNOTATION_SAM_IMAGE_CACHE_SIZE",
                2,
                0,
                16,
            ),
            worker_id=os.getenv(
                "ANNOTATION_SAM_WORKER_ID",
                f"{socket.gethostname()}-{os.getpid()}",
            ).strip(),
            lease_seconds=lease,
            heartbeat_seconds=heartbeat,
            poll_seconds=float(
                os.getenv("ANNOTATION_SAM_POLL_SECONDS", "0.2")
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
            image_embedding_cache_size=(
                self.image_embedding_cache_size
            ),
        )


def persist_sam_candidate(
    store: AnnotationStore,
    *,
    task_id: str,
    operation_id: str | None,
    candidate: SAMMaskCandidate,
) -> dict:
    artifact_started = time.perf_counter()
    metadata = {
        "box_xyxy": candidate.box_xyxy,
        "predicted_iou": candidate.predicted_iou,
        "mask_area_pixels": candidate.mask_area_pixels,
        "sam_version": candidate.model_version,
        "timings_ms": candidate.timings_ms,
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
    timings_ms = {
        **candidate.timings_ms,
        "artifact_write_ms": round(
            (time.perf_counter() - artifact_started) * 1000,
            3,
        ),
    }
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
        "timings_ms": timings_ms,
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
        poll_seconds: float = 0.2,
        max_batch_size: int = 16,
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
        if max_batch_size < 1 or max_batch_size > 128:
            raise ValueError(
                "max_batch_size must be between 1 and 128"
            )
        self.max_batch_size = max_batch_size
        self._prefetch_created_after = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
        self._prefetched_asset_ids: set[str] = set()

    def run_once(self) -> bool:
        operations = self.store.claim_mask_operation_batch(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            max_batch_size=self.max_batch_size,
        )
        if not operations:
            return self._prefetch_once()
        self._process_batch(operations)
        return True

    def _prefetch_once(self) -> bool:
        precompute = getattr(self.predictor, "precompute", None)
        if not callable(precompute):
            return False
        asset_ids = self.store.list_detection_asset_ids_created_after(
            created_after=self._prefetch_created_after,
            limit=100,
        )
        asset_id = next(
            (
                candidate
                for candidate in asset_ids
                if candidate not in self._prefetched_asset_ids
            ),
            None,
        )
        if asset_id is None:
            return False
        self._prefetched_asset_ids.add(asset_id)
        try:
            image_path, _ = self.store.asset_file(asset_id)
            timings = precompute(image_path=image_path)
            LOGGER.info(
                "SAM background prefetch completed: asset_id=%s "
                "timings=%s",
                asset_id,
                timings,
            )
        except Exception:
            LOGGER.exception(
                "SAM background prefetch failed",
                extra={"asset_id": asset_id},
            )
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

    def _fail_running_operation(
        self,
        operation: dict,
        error: Exception,
    ) -> None:
        operation_id = operation["operation_id"]
        try:
            current = self.store.get_operation(operation_id)
            if current["status"] != "running":
                return
            self.store.fail_operation(
                operation_id,
                worker_id=self.worker_id,
                code="model_unavailable",
                message=str(error),
            )
        except Exception:
            LOGGER.exception(
                "failed to persist SAM operation error",
                extra={"operation_id": operation_id},
            )

    def _predict_many(
        self,
        *,
        image_path: Path,
        boxes_xyxy: list[list[float]],
    ) -> list[SAMMaskCandidate]:
        batch_predict = getattr(self.predictor, "predict_many", None)
        if callable(batch_predict):
            return batch_predict(
                image_path=image_path,
                boxes_xyxy=boxes_xyxy,
            )
        return [
            self.predictor.predict(
                image_path=image_path,
                box_xyxy=box_xyxy,
            )
            for box_xyxy in boxes_xyxy
        ]

    def _process_batch(self, operations: list[dict]) -> None:
        heartbeats = [
            OperationHeartbeat(
                store=self.store,
                operation_id=operation["operation_id"],
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                interval_seconds=self.heartbeat_seconds,
            )
            for operation in operations
        ]
        for heartbeat in heartbeats:
            heartbeat.start()
        try:
            tasks = [
                self.store.get_task(operation["task_id"])
                for operation in operations
            ]
            for task, operation in zip(tasks, operations):
                if task["version"] != operation["task_version"]:
                    raise ValueError(
                        "task version changed after mask operation was queued"
                    )
            asset_ids = {
                task["asset"]["asset_id"] for task in tasks
            }
            if len(asset_ids) != 1:
                raise ValueError(
                    "SAM batch operations must reference one image"
                )
            image_path, _ = self.store.asset_file(asset_ids.pop())
            box_groups = []
            for operation in operations:
                request = operation["request"]
                boxes = request.get("boxes_xyxy")
                if boxes is None:
                    boxes = [request["box_xyxy"]]
                box_groups.append(boxes)
            flat_boxes = [
                box
                for boxes in box_groups
                for box in boxes
            ]
            predicted = self._predict_many(
                image_path=image_path,
                boxes_xyxy=flat_boxes,
            )
            if len(predicted) != len(flat_boxes):
                raise ValueError(
                    "SAM batch result count does not match box count"
                )
            candidates = []
            offset = 0
            polygon_epsilon = getattr(
                getattr(self.predictor, "config", None),
                "polygon_epsilon",
                1.0,
            )
            for operation, boxes in zip(operations, box_groups):
                operation_candidates = predicted[
                    offset:offset + len(boxes)
                ]
                offset += len(boxes)
                detection_ids = operation["request"].get(
                    "detection_ids",
                    [],
                )
                candidates.append(
                    combine_sam_candidates(
                        image_path=image_path,
                        candidates=operation_candidates,
                        detection_ids=detection_ids,
                        polygon_epsilon=polygon_epsilon,
                    )
                )
            for heartbeat in heartbeats:
                heartbeat.ensure_healthy()
        except Exception as exc:
            LOGGER.exception(
                "SAM batch mask generation failed",
                extra={
                    "operation_ids": [
                        operation["operation_id"]
                        for operation in operations
                    ]
                },
            )
            for operation in operations:
                self._fail_running_operation(operation, exc)
            for heartbeat in heartbeats:
                heartbeat.stop()
            return

        for operation, task, candidate in zip(
            operations,
            tasks,
            candidates,
        ):
            operation_id = operation["operation_id"]
            try:
                if (
                    self.store.get_operation(operation_id)["status"]
                    != "running"
                ):
                    continue
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
                    "SAM mask candidate persistence failed",
                    extra={"operation_id": operation_id},
                )
                self._fail_running_operation(operation, exc)
        for heartbeat in heartbeats:
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
    predictor = SAMAdapter(config)
    if not args.once:
        predictor.warmup()
        LOGGER.info("SAM model preloaded and ready")
    worker = SAMMaskWorker(
        store=store,
        predictor=predictor,
        worker_id=settings.worker_id,
        lease_seconds=settings.lease_seconds,
        heartbeat_seconds=settings.heartbeat_seconds,
        poll_seconds=settings.poll_seconds,
        max_batch_size=settings.max_batch_size,
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
