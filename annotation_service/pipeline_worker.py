from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hazard_rules import HazardRuleEngine
from .qwen_provider import Qwen25VLProvider, image_file_to_input
from .qwen_worker import PromptProvider, _visual_context
from .review_task_builder import materialize_candidate_tasks
from .sam_adapter import SAMAdapter
from .sam_worker import (
    MaskPredictor,
    SAMWorkerSettings,
    persist_sam_candidate,
)
from .schemas import JobStatus, PipelineStage
from .storage import AnnotationStore
from .worker.grounding_dino import (
    DetectionPredictor,
    GroundingDINOAdapter,
    GroundingDINOModelConfig,
)
from .worker.runner import LeaseHeartbeat
from .worker.settings import GroundingDINOWorkerSettings
from .qwen_worker import QwenWorkerSettings


LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineExecutionError(RuntimeError):
    def __init__(self, stage: PipelineStage, message: str):
        super().__init__(message)
        self.stage = stage


class FullAnnotationPipelineWorker:
    """Run one complete GroundingDINO -> SAM -> Qwen annotation job."""

    def __init__(
        self,
        *,
        store: AnnotationStore,
        detection_predictor: DetectionPredictor,
        mask_predictor: MaskPredictor,
        prompt_provider: PromptProvider,
        worker_id: str,
        hazard_engine: HazardRuleEngine | None = None,
        lease_seconds: int = 900,
        heartbeat_seconds: int = 60,
        poll_seconds: float = 2.0,
    ):
        if heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be less than lease_seconds"
            )
        self.store = store
        self.detection_predictor = detection_predictor
        self.mask_predictor = mask_predictor
        self.prompt_provider = prompt_provider
        self.hazard_engine = hazard_engine or HazardRuleEngine()
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds

    def run_once(self) -> bool:
        job = self.store.claim_next_job(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            full_pipeline_only=True,
        )
        if job is None:
            return False
        self._process(job)
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
                LOGGER.exception("unexpected full pipeline worker failure")
                processed = False
            if not processed:
                stop.wait(self.poll_seconds)

    def _process(self, job: dict[str, Any]) -> None:
        heartbeat = LeaseHeartbeat(
            store=self.store,
            job_id=job["job_id"],
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_seconds,
        )
        heartbeat.start()
        errors: list[dict[str, Any]] = []
        successful_assets: set[str] = set()
        stage_failures: dict[str, int] = {
            stage.value: 0 for stage in PipelineStage
        }
        try:
            for asset_id in job["asset_ids"]:
                heartbeat.ensure_healthy()
                try:
                    self._detect_and_derive(job, asset_id)
                    successful_assets.add(asset_id)
                except Exception as exc:
                    LOGGER.exception(
                        "full pipeline detection/rule stage failed",
                        extra={
                            "job_id": job["job_id"],
                            "asset_id": asset_id,
                        },
                    )
                    stage = (
                        exc.stage
                        if isinstance(exc, PipelineExecutionError)
                        else PipelineStage.GROUNDING_DINO
                    )
                    stage_failures[stage.value] += 1
                    errors.append(
                        {
                            "asset_id": asset_id,
                            "stage": stage.value,
                            "code": "pipeline_asset_failed",
                            "message": str(exc)[:1000],
                        }
                    )
                    self.store.update_job_asset(
                        job_id=job["job_id"],
                        asset_id=asset_id,
                        status="failed",
                        worker_id=self.worker_id,
                        error=errors[-1],
                    )

            current_job = self.store.get_job(job["job_id"])
            task_ids = materialize_candidate_tasks(
                self.store,
                job=current_job,
            )
            for task_id in task_ids:
                heartbeat.ensure_healthy()
                task = self.store.get_task(task_id)
                if task["asset"]["asset_id"] not in successful_assets:
                    continue
                try:
                    self._segment_and_prompt(task)
                except Exception as exc:
                    LOGGER.exception(
                        "full pipeline SAM/Qwen stage failed",
                        extra={
                            "job_id": job["job_id"],
                            "task_id": task_id,
                        },
                    )
                    stage = (
                        exc.stage
                        if isinstance(exc, PipelineExecutionError)
                        else PipelineStage.SAM
                    )
                    if stage in {
                        PipelineStage.QWEN_FACTS,
                        PipelineStage.QWEN_PROMPTS,
                    }:
                        stage_failures[
                            PipelineStage.QWEN_PROMPTS.value
                        ] += 1
                    stage_failures[stage.value] += 1
                    errors.append(
                        {
                            "asset_id": task["asset"]["asset_id"],
                            "stage": stage.value,
                            "code": "pipeline_task_failed",
                            "message": str(exc)[:1000],
                        }
                    )
                    successful_assets.discard(
                        task["asset"]["asset_id"]
                    )
                    self.store.update_job_asset(
                        job_id=job["job_id"],
                        asset_id=task["asset"]["asset_id"],
                        status="failed",
                        worker_id=self.worker_id,
                        error=errors[-1],
                    )
            for asset_id in successful_assets:
                self.store.update_job_asset(
                    job_id=job["job_id"],
                    asset_id=asset_id,
                    status="succeeded",
                    worker_id=self.worker_id,
                )
            heartbeat.ensure_healthy()
            self._complete(
                job=job,
                errors=errors,
                successful_assets=successful_assets,
                stage_failures=stage_failures,
            )
        except Exception as exc:
            LOGGER.exception(
                "full annotation pipeline aborted",
                extra={"job_id": job["job_id"]},
            )
            try:
                self.store.update_job(
                    job["job_id"],
                    expected_status=JobStatus.RUNNING,
                    status=JobStatus.FAILED,
                    stage=PipelineStage.BUILD_REVIEW_TASKS,
                    errors=[
                        *errors,
                        {
                            "asset_id": None,
                            "stage": None,
                            "code": "pipeline_worker_failed",
                            "message": str(exc)[:1000],
                        },
                    ],
                    worker_id=self.worker_id,
                )
            except Exception:
                LOGGER.exception("failed to persist pipeline failure")
        finally:
            heartbeat.stop()

    def _detect_and_derive(
        self,
        job: dict[str, Any],
        asset_id: str,
    ) -> None:
        self.store.update_job_asset(
            job_id=job["job_id"],
            asset_id=asset_id,
            status="running",
            worker_id=self.worker_id,
        )
        asset = self.store.get_asset(asset_id)
        image_path, _ = self.store.asset_file(asset_id)
        try:
            detections = self.detection_predictor.predict(
                image_path=image_path,
                width=asset["width"],
                height=asset["height"],
                categories=job["requested_categories"],
            )
            saved = self.store.replace_detections(
                job_id=job["job_id"],
                asset_id=asset_id,
                detections=[
                    detection.as_storage_payload()
                    for detection in detections
                ],
                worker_id=self.worker_id,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                PipelineStage.GROUNDING_DINO,
                f"GroundingDINO failed: {exc}",
            ) from exc
        try:
            candidates = self.hazard_engine.infer(
                detections=saved,
                requested_categories=job["requested_categories"],
                width=asset["width"],
                height=asset["height"],
            )
            self.store.replace_hazard_candidates(
                job_id=job["job_id"],
                asset_id=asset_id,
                candidates=[
                    candidate.as_storage_payload()
                    for candidate in candidates
                ],
                worker_id=self.worker_id,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                PipelineStage.HAZARD_RULES,
                f"hazard rules failed: {exc}",
            ) from exc

    def _segment_and_prompt(self, task: dict[str, Any]) -> None:
        source_hazard = task.get("source_hazard")
        if not source_hazard:
            raise ValueError("pipeline task has no source hazard")
        image_path, image_media_type = self.store.asset_file(
            task["asset"]["asset_id"]
        )
        try:
            sam = self.mask_predictor.predict(
                image_path=image_path,
                box_xyxy=source_hazard["box_xyxy"],
            )
            persist_sam_candidate(
                self.store,
                task_id=task["task_id"],
                operation_id=None,
                candidate=sam,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                PipelineStage.SAM,
                f"SAM failed: {exc}",
            ) from exc
        refreshed = self.store.get_task(task["task_id"])
        overlay_path, overlay_media_type = self.store.artifact_file(
            task["task_id"],
            "mask-overlay",
        )
        crop_path, crop_media_type = self.store.artifact_file(
            task["task_id"],
            "crop",
        )
        try:
            generated = self.prompt_provider.generate(
                context=_visual_context(refreshed),
                images=[
                    image_file_to_input(
                        image_path,
                        media_type=image_media_type,
                        label="原图",
                    ),
                    image_file_to_input(
                        overlay_path,
                        media_type=overlay_media_type,
                        label="SAM mask叠加图",
                    ),
                    image_file_to_input(
                        crop_path,
                        media_type=crop_media_type,
                        label="目标局部裁剪图",
                    ),
                ],
            )
        except Exception as exc:
            raise PipelineExecutionError(
                PipelineStage.QWEN_FACTS,
                f"Qwen2.5-VL failed: {exc}",
            ) from exc
        result = generated.as_dict()
        facts = result["facts"]
        annotation = {
            "target_object": facts["target_object"],
            "instance_count": facts["instance_count"],
            "visual_anchor": facts["visual_anchor"],
            "mask_granularity": facts["mask_granularity"],
            "risk_semantics": facts.get("risk_semantics"),
            "shapes": sam.shapes,
            "prompts": result["prompts"],
        }
        provenance = {
            "sam_version": sam.model_version,
            **result["provenance"],
        }
        warnings = [
            "SAM mask 和 Qwen Prompt 均为模型候选，必须经过人工审核。"
        ]
        if source_hazard["metadata"].get(
            "requires_visual_verification"
        ):
            warnings.append("隐患规则使用弱负证据，必须人工确认。")
        self.store.replace_generated_task_content(
            task["task_id"],
            expected_version=task["version"],
            annotation=annotation,
            provenance_updates=provenance,
            warnings=warnings,
        )

    def _complete(
        self,
        *,
        job: dict[str, Any],
        errors: list[dict[str, Any]],
        successful_assets: set[str],
        stage_failures: dict[str, int],
    ) -> None:
        current = self.store.get_job(job["job_id"])
        now = _now()
        stages = dict(current["stages"])
        task_count = len(current["task_ids"])
        messages = {
            PipelineStage.GROUNDING_DINO: "entity detection completed",
            PipelineStage.HAZARD_RULES: "hazard candidates derived",
            PipelineStage.SAM: f"SAM processed {task_count} task(s)",
            PipelineStage.QWEN_FACTS: (
                f"Qwen visual facts processed {task_count} task(s)"
            ),
            PipelineStage.QWEN_PROMPTS: (
                f"Qwen prompts processed {task_count} task(s)"
            ),
            PipelineStage.BUILD_REVIEW_TASKS: (
                f"materialized {task_count} review task(s)"
            ),
        }
        for stage in PipelineStage:
            failed = stage_failures.get(stage.value, 0)
            stages[stage.value] = {
                "status": "failed" if failed else "succeeded",
                "started_at": (
                    stages.get(stage.value, {}).get("started_at") or now
                ),
                "completed_at": now,
                "message": (
                    f"{messages[stage]}; {failed} failure(s)"
                ),
            }
        if not errors:
            status = JobStatus.SUCCEEDED
        elif successful_assets:
            status = JobStatus.PARTIAL_FAILED
        else:
            status = JobStatus.FAILED
        self.store.update_job(
            job["job_id"],
            expected_status=JobStatus.RUNNING,
            status=status,
            stage=PipelineStage.BUILD_REVIEW_TASKS,
            progress={
                "total_assets": len(job["asset_ids"]),
                "completed_assets": len(job["asset_ids"]),
                "generated_tasks": task_count,
            },
            stages=stages,
            errors=errors,
            worker_id=self.worker_id,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete annotation GPU pipeline worker",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dino = GroundingDINOWorkerSettings.from_env()
    sam = SAMWorkerSettings.from_env()
    qwen = QwenWorkerSettings.from_env()
    dino.validate_model_files()
    sam_config = sam.model_config()
    sam_config.validate()
    worker_id = os.getenv(
        "ANNOTATION_PIPELINE_WORKER_ID",
        f"{socket.gethostname()}-{os.getpid()}",
    ).strip()
    lease_seconds = int(
        os.getenv("ANNOTATION_PIPELINE_LEASE_SECONDS", "900")
    )
    heartbeat_seconds = int(
        os.getenv("ANNOTATION_PIPELINE_HEARTBEAT_SECONDS", "60")
    )
    store = AnnotationStore(dino.storage_root)
    store.initialize()
    worker = FullAnnotationPipelineWorker(
        store=store,
        detection_predictor=GroundingDINOAdapter(
            GroundingDINOModelConfig(
                root=dino.grounding_dino_root,
                config_path=dino.config_path,
                checkpoint_path=dino.checkpoint_path,
                bert_path=dino.bert_path,
                device=dino.device,
                model_version=dino.model_version,
                prompt_version=dino.prompt_version,
                box_threshold=dino.box_threshold,
                text_threshold=dino.text_threshold,
            )
        ),
        mask_predictor=SAMAdapter(sam_config),
        prompt_provider=Qwen25VLProvider(qwen.provider_config()),
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        heartbeat_seconds=heartbeat_seconds,
        poll_seconds=float(
            os.getenv("ANNOTATION_PIPELINE_POLL_SECONDS", "2")
        ),
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
