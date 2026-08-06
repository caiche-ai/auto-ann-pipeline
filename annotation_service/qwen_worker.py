from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .qwen_contract import QwenContractError, QwenVisualContext
from .qwen_provider import (
    Qwen25VLProvider,
    QwenGenerationResult,
    QwenProviderConfig,
    QwenProviderError,
    image_file_to_input,
)
from .schemas import Detection
from .storage import AnnotationStore


LOGGER = logging.getLogger(__name__)


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
class QwenWorkerSettings:
    storage_root: Path
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    max_tokens: int
    facts_temperature: float
    prompts_temperature: float
    worker_id: str
    lease_seconds: int
    heartbeat_seconds: int
    poll_seconds: float

    @classmethod
    def from_env(cls) -> "QwenWorkerSettings":
        storage_root = os.getenv("ANNOTATION_STORAGE_ROOT", "").strip()
        base_url = os.getenv("ANNOTATION_QWEN_BASE_URL", "").strip()
        model = os.getenv(
            "ANNOTATION_QWEN_MODEL",
            "qwen25vl",
        ).strip()
        if not storage_root:
            raise ValueError("ANNOTATION_STORAGE_ROOT must not be empty")
        if not base_url:
            raise ValueError("ANNOTATION_QWEN_BASE_URL must not be empty")
        if not model:
            raise ValueError("ANNOTATION_QWEN_MODEL must not be empty")
        lease_seconds = _integer(
            "ANNOTATION_QWEN_LEASE_SECONDS",
            300,
            minimum=30,
            maximum=3600,
        )
        heartbeat_seconds = _integer(
            "ANNOTATION_QWEN_HEARTBEAT_SECONDS",
            60,
            minimum=5,
            maximum=600,
        )
        if heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "ANNOTATION_QWEN_HEARTBEAT_SECONDS must be less than "
                "ANNOTATION_QWEN_LEASE_SECONDS"
            )
        worker_id = os.getenv(
            "ANNOTATION_QWEN_WORKER_ID",
            f"{socket.gethostname()}-{os.getpid()}",
        ).strip()
        if not worker_id or len(worker_id) > 128:
            raise ValueError(
                "ANNOTATION_QWEN_WORKER_ID must contain 1 to 128 characters"
            )
        api_key = os.getenv("ANNOTATION_QWEN_API_KEY", "").strip() or None
        return cls(
            storage_root=Path(storage_root).expanduser().resolve(),
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=_floating(
                "ANNOTATION_QWEN_TIMEOUT_SECONDS",
                120.0,
                minimum=1,
                maximum=3600,
            ),
            max_tokens=_integer(
                "ANNOTATION_QWEN_MAX_TOKENS",
                1200,
                minimum=1,
                maximum=16_384,
            ),
            facts_temperature=_floating(
                "ANNOTATION_QWEN_FACTS_TEMPERATURE",
                0.1,
                minimum=0,
                maximum=2,
            ),
            prompts_temperature=_floating(
                "ANNOTATION_QWEN_PROMPTS_TEMPERATURE",
                0.3,
                minimum=0,
                maximum=2,
            ),
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            poll_seconds=_floating(
                "ANNOTATION_QWEN_POLL_SECONDS",
                2.0,
                minimum=0.1,
                maximum=60,
            ),
        )

    def provider_config(self) -> QwenProviderConfig:
        return QwenProviderConfig(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            facts_temperature=self.facts_temperature,
            prompts_temperature=self.prompts_temperature,
        )


class PromptProvider(Protocol):
    def generate(
        self,
        *,
        context: QwenVisualContext,
        images: list[Any],
    ) -> QwenGenerationResult:
        ...


class OperationHeartbeat:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        operation_id: str,
        worker_id: str,
        lease_seconds: int,
        interval_seconds: int,
    ):
        self.store = store
        self.operation_id = operation_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"qwen-heartbeat-{self.operation_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.store.heartbeat_operation(
                    self.operation_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:
                self._error = exc
                self._stop.set()
                return

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1, self.interval_seconds + 1))


def _target_box(task: dict[str, Any]) -> list[float]:
    source_hazard = task.get("source_hazard")
    if source_hazard and source_hazard.get("box_xyxy"):
        return [float(value) for value in source_hazard["box_xyxy"]]
    points = [
        point
        for shape in task["annotation"].get("shapes", [])
        if shape.get("label") == "target"
        for point in shape.get("points", [])
    ]
    if points:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        box = [min(xs), min(ys), max(xs), max(ys)]
        if box[2] > box[0] and box[3] > box[1]:
            return box
    if task.get("detections"):
        return [
            float(value)
            for value in task["detections"][0]["box_xyxy"]
        ]
    raise ValueError(
        "task has no target box, target polygon, or detection"
    )


def _visual_context(task: dict[str, Any]) -> QwenVisualContext:
    source_hazard = task.get("source_hazard") or {}
    return QwenVisualContext(
        asset_id=task["asset"]["asset_id"],
        category=task["category"],
        target_box_xyxy=_target_box(task),
        target_detection_ids=source_hazard.get(
            "target_detection_ids",
            [],
        ),
        detections=[
            Detection(**detection)
            for detection in task.get("detections", [])
        ],
        hazard_evidence=source_hazard.get("evidence", []),
        requires_visual_verification=True,
        mask_available=bool(
            task.get("artifacts", {}).get("mask_overlay_url")
            or task.get("artifacts", {}).get("mask_png_url")
        ),
    )


class QwenPromptWorker:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        provider: PromptProvider,
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
        self.provider = provider
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds

    def run_once(self) -> bool:
        operation = self.store.claim_next_operation(
            operation_type="prompt_enrichment",
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
                LOGGER.exception("unexpected Qwen worker loop failure")
                processed = False
            if not processed:
                stop.wait(self.poll_seconds)

    def _process(self, operation: dict[str, Any]) -> None:
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
                    "task version changed after prompt operation was queued"
                )
            asset_path, asset_media_type = self.store.asset_file(
                task["asset"]["asset_id"]
            )
            try:
                mask_path, mask_media_type = self.store.artifact_file(
                    task["task_id"],
                    "mask-overlay",
                )
                mask_label = "SAM mask叠加图"
            except Exception:
                mask_path, mask_media_type = self.store.artifact_file(
                    task["task_id"],
                    "mask",
                )
                mask_label = "SAM二值mask"
            images = [
                image_file_to_input(
                    asset_path,
                    media_type=asset_media_type,
                    label="原图",
                ),
                image_file_to_input(
                    mask_path,
                    media_type=mask_media_type,
                    label=mask_label,
                ),
            ]
            if task.get("artifacts", {}).get("crop_url"):
                crop_path, crop_media_type = self.store.artifact_file(
                    task["task_id"],
                    "crop",
                )
                images.append(
                    image_file_to_input(
                        crop_path,
                        media_type=crop_media_type,
                        label="目标局部裁剪图",
                    )
                )
            result = self.provider.generate(
                context=_visual_context(task),
                images=images,
            )
            heartbeat.ensure_healthy()
            if self.store.get_operation(operation_id)["status"] != "running":
                raise ValueError(
                    "prompt operation was cancelled before results were saved"
                )
            self.store.complete_operation(
                operation_id,
                worker_id=self.worker_id,
                result=result.as_dict(),
            )
        except Exception as exc:
            LOGGER.exception(
                "Qwen prompt generation failed",
                extra={"operation_id": operation_id},
            )
            try:
                if isinstance(exc, QwenProviderError):
                    error_code = "model_unavailable"
                elif isinstance(exc, QwenContractError):
                    error_code = "validation_error"
                elif "task version changed" in str(exc):
                    error_code = "version_conflict"
                else:
                    error_code = "internal_error"
                self.store.fail_operation(
                    operation_id,
                    worker_id=self.worker_id,
                    code=error_code,
                    message=str(exc),
                )
            except Exception:
                LOGGER.exception(
                    "failed to persist Qwen operation error",
                    extra={"operation_id": operation_id},
                )
        finally:
            heartbeat.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Qwen2.5-VL prompt generation worker",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one prompt operation and exit",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = QwenWorkerSettings.from_env()
    store = AnnotationStore(settings.storage_root)
    store.initialize()
    worker = QwenPromptWorker(
        store=store,
        provider=Qwen25VLProvider(settings.provider_config()),
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
