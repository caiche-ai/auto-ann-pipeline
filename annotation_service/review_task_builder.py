from __future__ import annotations

from typing import Any

from .errors import InvalidStateTransitionError, ValidationServiceError
from .schemas import JobStatus
from .storage import AnnotationStore, utc_now


BUILDER_VERSION = "hazard-candidate-manual-task-v1"


ENTITY_NAMES = {
    "person": "作业人员",
    "worker": "作业人员",
    "helmet": "安全帽",
    "safety vest": "反光背心",
    "excavator": "施工设备",
    "construction equipment": "施工设备",
    "opening": "洞口",
    "platform edge": "临边",
    "debris": "建筑垃圾",
    "material": "现场材料",
}

GRANULARITY_BY_CATEGORY = {
    "helmet_missing": "人员整体",
    "no_helmet": "人员整体",
    "no_jacket": "人员整体",
    "harness_missing": "人员整体",
    "equipment_proximity": "人员整体",
    "opening_unprotected": "洞口或开口区域",
    "guardrail_missing": "临边或洞口区域",
    "poor_housekeeping": "垃圾或占道材料整体",
    "safe": "待人工确认",
    "unsafe": "待人工确认",
}

RISK_BY_CATEGORY = {
    "helmet_missing": "候选风险：头部防护可能缺失",
    "no_helmet": "候选风险：头部防护可能缺失",
    "no_jacket": "候选风险：反光防护服可能缺失",
    "harness_missing": "候选风险：高处防坠保护可能缺失",
    "equipment_proximity": "候选风险：人员与施工设备距离可能过近",
    "opening_unprotected": "候选风险：洞口防护可能缺失",
    "guardrail_missing": "候选风险：临边防护可能缺失",
    "poor_housekeeping": "候选风险：现场整理或通道占用问题",
    "safe": None,
    "unsafe": "候选风险：存在待具体化的不安全因素",
}


def _position_anchor(
    box_xyxy: list[float],
    *,
    width: int,
    height: int,
) -> str:
    center_x = (box_xyxy[0] + box_xyxy[2]) / 2 / width
    center_y = (box_xyxy[1] + box_xyxy[3]) / 2 / height
    horizontal = "左侧" if center_x < 1 / 3 else "右侧" if center_x > 2 / 3 else "中央"
    vertical = "上部" if center_y < 1 / 3 else "下部" if center_y > 2 / 3 else "中部"
    return f"目标框位于画面{horizontal}{vertical}"


def _provenance(
    *,
    job: dict[str, Any],
    candidate: dict[str, Any],
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = detections[0].get("metadata", {}) if detections else {}
    thresholds = {
        key: float(value)
        for key, value in metadata.get("thresholds", {}).items()
        if isinstance(value, (int, float))
    }
    return {
        "pipeline_version": job["pipeline_version"],
        "hazard_candidate_id": candidate["hazard_id"],
        "review_task_builder_version": BUILDER_VERSION,
        "grounding_dino_version": metadata.get("model_version"),
        "grounding_dino_prompt_version": metadata.get("prompt_version"),
        "grounding_dino_thresholds": thresholds,
        "hazard_rule_version": candidate["rule_version"],
    }


def materialize_candidate_tasks(
    store: AnnotationStore,
    *,
    job: dict[str, Any],
) -> list[str]:
    candidates = store.list_job_hazard_candidates(job_id=job["job_id"])
    detections = store.list_job_detections(job_id=job["job_id"])
    detections_by_asset: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        detections_by_asset.setdefault(detection["asset_id"], []).append(
            detection
        )

    task_ids: list[str] = []
    for candidate in candidates:
        asset = store.get_asset(candidate["asset_id"])
        entity_name = ENTITY_NAMES.get(
            candidate["target_entity"].lower(),
            candidate["target_entity"],
        )
        annotation = {
            "target_object": f"待人工复核的{entity_name}",
            "instance_count": 1,
            "visual_anchor": [
                _position_anchor(
                    candidate["box_xyxy"],
                    width=asset["width"],
                    height=asset["height"],
                )
            ],
            "mask_granularity": GRANULARITY_BY_CATEGORY[
                candidate["category"]
            ],
            "risk_semantics": RISK_BY_CATEGORY[candidate["category"]],
            "shapes": [],
            "prompts": [],
        }
        warnings = [
            "该 Task 来自隐患规则候选，尚未经过视觉事实确认。",
            "当前没有 SAM mask；提交前必须由人工补齐 target polygon。",
            "当前没有千问富化结果；提交前必须由人工补齐 3+2+1 Prompt。",
        ]
        if candidate["metadata"].get("requires_visual_verification"):
            warnings.append("规则使用弱负证据，必须人工确认候选是否成立。")
        task = store.create_task(
            job_id=job["job_id"],
            asset_id=candidate["asset_id"],
            category=candidate["category"],
            annotation=annotation,
            provenance=_provenance(
                job=job,
                candidate=candidate,
                detections=detections_by_asset.get(
                    candidate["asset_id"],
                    [],
                ),
            ),
            warnings=warnings,
            source_hazard_id=candidate["hazard_id"],
        )
        task_ids.append(task["task_id"])
    return task_ids


def build_review_tasks(
    store: AnnotationStore,
    *,
    job_id: str,
) -> dict[str, Any]:
    """Materialize manual drafts from persisted hazard candidates.

    This is intentionally model-free. It never turns the candidate box into a
    polygon and never treats weak negative evidence as a confirmed fact.
    """

    job = store.get_job(job_id)
    if job["status"] not in {
        JobStatus.SUCCEEDED.value,
        JobStatus.PARTIAL_FAILED.value,
    }:
        raise InvalidStateTransitionError(
            "review tasks can only be built from a completed hazard job"
        )
    hazard_stage = job["stages"].get("hazard_rules", {})
    if hazard_stage.get("status") != "succeeded":
        raise InvalidStateTransitionError(
            "hazard_rules must succeed before review tasks are built"
        )

    before = set(job["task_ids"])
    task_ids = materialize_candidate_tasks(store, job=job)

    updated_job = store.get_job(job_id)
    progress = dict(updated_job["progress"])
    progress["generated_tasks"] = len(updated_job["task_ids"])
    stages = dict(updated_job["stages"])
    build_stage = stages.get("build_review_tasks", {})
    if build_stage.get("status") != "succeeded":
        completed_at = utc_now()
        stages["build_review_tasks"] = {
            "status": "succeeded",
            "started_at": completed_at,
            "completed_at": completed_at,
            "message": (
                f"materialized {len(task_ids)} manual review task(s)"
            ),
        }
    store.update_job(
        job_id,
        expected_status=updated_job["status"],
        progress=progress,
        stages=stages,
    )
    created = len(set(task_ids) - before)
    return {
        "job_id": job_id,
        "task_ids": task_ids,
        "created_count": created,
        "existing_count": len(task_ids) - created,
    }


def build_detection_review_tasks(
    store: AnnotationStore,
    *,
    job_id: str,
    detection_ids: list[str] | None,
    category: str,
) -> dict[str, Any]:
    """Create one idempotent annotation task for each selected detection."""

    job = store.get_job(job_id)
    if job["status"] not in {
        JobStatus.SUCCEEDED.value,
        JobStatus.PARTIAL_FAILED.value,
    }:
        raise InvalidStateTransitionError(
            "review tasks can only be built from a completed detection job"
        )

    detections = store.list_job_detections(job_id=job_id)
    by_id = {
        detection["detection_id"]: detection
        for detection in detections
    }
    selected_ids = (
        detection_ids
        if detection_ids is not None
        else list(by_id)
    )
    missing = [
        detection_id
        for detection_id in selected_ids
        if detection_id not in by_id
    ]
    if missing:
        raise ValidationServiceError(
            "one or more detections do not belong to this job",
            details=[
                {
                    "field": "detection_ids",
                    "reason": (
                        "unknown detection IDs: "
                        + ", ".join(missing[:10])
                    ),
                }
            ],
        )

    before = set(job["task_ids"])
    task_ids: list[str] = []
    for detection_id in selected_ids:
        detection = by_id[detection_id]
        asset = store.get_asset(detection["asset_id"])
        metadata = detection.get("metadata", {})
        thresholds = {}
        for key in ("box_threshold", "text_threshold"):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                thresholds[key] = float(value)
        entity = detection["entity"].strip() or "目标对象"
        task = store.create_task(
            job_id=job_id,
            asset_id=detection["asset_id"],
            category=category,
            annotation={
                "target_object": entity,
                "instance_count": 1,
                "visual_anchor": [
                    _position_anchor(
                        detection["box_xyxy"],
                        width=asset["width"],
                        height=asset["height"],
                    )
                ],
                "mask_granularity": "目标对象整体",
                "risk_semantics": None,
                "shapes": [],
                "prompts": [],
            },
            provenance={
                "pipeline_version": job["pipeline_version"],
                "source_detection_id": detection_id,
                "grounding_prompt": job["grounding_prompt"],
                "grounding_dino_version": metadata.get("model_version"),
                "grounding_dino_prompt_version": metadata.get(
                    "prompt_version"
                ),
                "grounding_dino_thresholds": thresholds,
                "review_task_builder_version": (
                    "free-form-detection-task-v1"
                ),
            },
            warnings=[
                "该 Task 来自自由文本检测框，类别和目标语义需要人工确认。",
                "当前没有 SAM mask；提交前需生成或手工补齐 target polygon。",
                "当前没有 Prompt 候选；提交前需选择或手工补齐 Prompt。",
            ],
            source_detection_id=detection_id,
        )
        task_ids.append(task["task_id"])

    created = len(set(task_ids) - before)
    return {
        "job_id": job_id,
        "task_ids": task_ids,
        "created_count": created,
        "existing_count": len(task_ids) - created,
    }
