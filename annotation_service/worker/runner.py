from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..detection_overlay import render_detection_overlay
from ..errors import VersionConflictError
from ..schemas import JobStatus, PipelineStage
from ..storage import AnnotationStore
from .grounding_dino import DetectionPredictor


LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LeaseHeartbeat:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        job_id: str,
        worker_id: str,
        lease_seconds: int,
        interval_seconds: int,
    ):
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-heartbeat-{self.job_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.store.heartbeat_job(
                    self.job_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:
                self._error = exc
                self._stop.set()
                return

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise VersionConflictError(
                "annotation job lease heartbeat failed"
            ) from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1, self.interval_seconds + 1))


class GroundingDINOJobWorker:
    """Run free-form GroundingDINO detection jobs from the queue."""

    def __init__(
        self,
        *,
        store: AnnotationStore,
        predictor: DetectionPredictor,
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
        job = self.store.claim_next_job(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            required_stop_after=PipelineStage.GROUNDING_DINO,
            grounding_prompt_required=True,
        )
        if job is None:
            return False
        self._process_claimed_job(job)
        return True

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or threading.Event()
        while not stop.is_set():
            try:
                processed = self.run_once()
            except VersionConflictError:
                LOGGER.warning(
                    "GroundingDINO job lease was lost; returning to queue",
                    exc_info=True,
                )
                processed = True
            except Exception:
                LOGGER.exception(
                    "unexpected GroundingDINO worker loop failure"
                )
                processed = False
            if not processed:
                stop.wait(self.poll_seconds)

    def _process_claimed_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        heartbeat = LeaseHeartbeat(
            store=self.store,
            job_id=job_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_seconds,
        )
        heartbeat.start()
        errors: list[dict[str, Any]] = []
        succeeded_assets = 0
        terminal_stage = self._terminal_stage(job)
        try:
            for asset_id in job["asset_ids"]:
                heartbeat.ensure_healthy()
                error = self._process_asset(
                    job=job,
                    asset_id=asset_id,
                )
                if error is None:
                    succeeded_assets += 1
                else:
                    errors.append(error)
                self._update_running_progress(
                    job=job,
                    completed_assets=succeeded_assets + len(errors),
                    errors=errors,
                    terminal_stage=terminal_stage,
                )
            heartbeat.ensure_healthy()
        except VersionConflictError:
            raise
        except Exception:
            LOGGER.exception(
                "GroundingDINO job failed unexpectedly",
                extra={"job_id": job_id},
            )
            errors.append(
                {
                    "asset_id": None,
                    "stage": terminal_stage.value,
                    "code": "annotation_worker_failed",
                    "message": (
                        "annotation worker failed before all assets were "
                        "processed"
                    ),
                }
            )
        finally:
            heartbeat.stop()

        self._complete_job(
            job=job,
            succeeded_assets=succeeded_assets,
            errors=errors,
        )

    def _process_asset(
        self,
        *,
        job: dict[str, Any],
        asset_id: str,
    ) -> dict[str, Any] | None:
        job_id = job["job_id"]
        self.store.update_job_asset(
            job_id=job_id,
            asset_id=asset_id,
            status="running",
            worker_id=self.worker_id,
        )
        try:
            asset = self.store.get_asset(asset_id)
            image_path, _ = self.store.asset_file(asset_id)
            detections = self.predictor.predict(
                image_path=Path(image_path),
                width=int(asset["width"]),
                height=int(asset["height"]),
                prompt=job["grounding_prompt"],
            )
            saved_detections = self.store.replace_detections(
                job_id=job_id,
                asset_id=asset_id,
                detections=[
                    detection.as_storage_payload()
                    for detection in detections
                ],
                worker_id=self.worker_id,
            )
            overlay = render_detection_overlay(
                image_path=Path(image_path),
                detections=saved_detections,
            )
            self.store.store_job_artifact(
                job_id=job_id,
                asset_id=asset_id,
                artifact_type="bbox-image",
                data=overlay,
                media_type="image/png",
                worker_id=self.worker_id,
                metadata={
                    "grounding_prompt": job["grounding_prompt"],
                    "detection_count": len(saved_detections),
                    "model_version": self.predictor.model_version,
                    "prompt_version": self.predictor.prompt_version,
                },
            )
            self.store.update_job_asset(
                job_id=job_id,
                asset_id=asset_id,
                status="succeeded",
                worker_id=self.worker_id,
            )
            return None
        except VersionConflictError:
            raise
        except Exception:
            LOGGER.exception(
                "annotation pipeline failed for asset",
                extra={
                    "job_id": job_id,
                    "asset_id": asset_id,
                },
            )
            error = {
                "asset_id": asset_id,
                "stage": PipelineStage.GROUNDING_DINO.value,
                "code": "grounding_dino_asset_failed",
                "message": "GroundingDINO inference failed for this asset",
            }
            self.store.update_job_asset(
                job_id=job_id,
                asset_id=asset_id,
                status="failed",
                worker_id=self.worker_id,
                error=error,
            )
            return error

    def _update_running_progress(
        self,
        *,
        job: dict[str, Any],
        completed_assets: int,
        errors: list[dict[str, Any]],
        terminal_stage: PipelineStage,
    ) -> None:
        stages = dict(job["stages"])
        stages[PipelineStage.GROUNDING_DINO.value] = {
            **stages[PipelineStage.GROUNDING_DINO.value],
            "status": "running",
            "message": (
                f"processed {completed_assets}/{len(job['asset_ids'])} assets"
            ),
        }
        self.store.update_job(
            job["job_id"],
            expected_status=JobStatus.RUNNING,
            stage=terminal_stage,
            progress={
                "total_assets": len(job["asset_ids"]),
                "completed_assets": completed_assets,
                "generated_tasks": 0,
            },
            stages=stages,
            errors=errors,
            worker_id=self.worker_id,
        )
        job["stages"] = stages

    def _complete_job(
        self,
        *,
        job: dict[str, Any],
        succeeded_assets: int,
        errors: list[dict[str, Any]],
    ) -> None:
        total_assets = len(job["asset_ids"])
        processed_assets = succeeded_assets + sum(
            1 for error in errors if error.get("asset_id") is not None
        )
        now = _now()
        stages = dict(job["stages"])
        dino_errors = [
            error
            for error in errors
            if error.get("stage") == PipelineStage.GROUNDING_DINO.value
        ]
        terminal_stage = self._terminal_stage(job)
        stages[PipelineStage.GROUNDING_DINO.value] = {
            **stages[PipelineStage.GROUNDING_DINO.value],
            "status": "succeeded" if not dino_errors else "failed",
            "completed_at": now,
            "message": (
                f"{total_assets - len(dino_errors)}/{total_assets} assets "
                f"detected; {len(dino_errors)} failed"
            ),
        }
        if not errors:
            status = JobStatus.SUCCEEDED
        elif succeeded_assets:
            status = JobStatus.PARTIAL_FAILED
        else:
            status = JobStatus.FAILED
        self.store.update_job(
            job["job_id"],
            expected_status=JobStatus.RUNNING,
            status=status,
            stage=terminal_stage,
            progress={
                "total_assets": total_assets,
                "completed_assets": processed_assets,
                "generated_tasks": 0,
            },
            stages=stages,
            errors=errors,
            worker_id=self.worker_id,
        )
        LOGGER.info(
            "GroundingDINO job completed",
            extra={
                "job_id": job["job_id"],
                "status": status.value,
                "succeeded_assets": succeeded_assets,
                "failed_assets": len(errors),
            },
        )

    @staticmethod
    def _terminal_stage(job: dict[str, Any]) -> PipelineStage:
        return PipelineStage.GROUNDING_DINO
