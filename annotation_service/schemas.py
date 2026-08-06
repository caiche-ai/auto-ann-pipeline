from __future__ import annotations

from datetime import datetime
from enum import Enum
from math import isclose
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator

from .prompt_normalization import (
    PromptNormalizationMode,
    PromptNormalizationProfile,
    PromptTranslationFailurePolicy,
)


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True


class AnnotationCategory(str, Enum):
    HELMET_MISSING = "helmet_missing"
    NO_HELMET = "no_helmet"
    NO_JACKET = "no_jacket"
    HARNESS_MISSING = "harness_missing"
    EQUIPMENT_PROXIMITY = "equipment_proximity"
    OPENING_UNPROTECTED = "opening_unprotected"
    GUARDRAIL_MISSING = "guardrail_missing"
    POOR_HOUSEKEEPING = "poor_housekeeping"
    SAFE = "safe"
    UNSAFE = "unsafe"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    GROUNDING_DINO = "grounding_dino"
    HAZARD_RULES = "hazard_rules"
    SAM = "sam"
    QWEN_FACTS = "qwen_facts"
    QWEN_PROMPTS = "qwen_prompts"
    BUILD_REVIEW_TASKS = "build_review_tasks"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskStatus(str, Enum):
    GENERATED = "generated"
    ANNOTATING = "annotating"
    REVIEW_PENDING = "review_pending"
    CHANGES_REQUESTED = "changes_requested"
    NEEDS_EXPERT = "needs_expert"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FROZEN = "frozen"


class BadCaseType(str, Enum):
    PROMPT_OK = "prompt_ok"
    PROMPT_REWRITTEN = "prompt_rewritten"
    PROMPT_MASK_MISMATCH = "prompt_mask_mismatch"
    MASK_OVERFLOW = "mask_overflow"
    MASK_MISSING = "mask_missing"
    INSTANCE_AMBIGUOUS = "instance_ambiguous"
    TARGET_UNRECOGNIZABLE = "target_unrecognizable"
    DINO_FALSE_POSITIVE = "dino_false_positive"
    DINO_FALSE_NEGATIVE = "dino_false_negative"
    HAZARD_RULE_ERROR = "hazard_rule_error"
    QWEN_VISUAL_HALLUCINATION = "qwen_visual_hallucination"
    QWEN_PROMPT_SEMANTIC_DRIFT = "qwen_prompt_semantic_drift"
    OTHER = "other"


class PromptType(str, Enum):
    VISUAL = "visual"
    RISK = "risk"
    AGENT = "agent"


class ShapeLabel(str, Enum):
    TARGET = "target"
    IGNORE = "ignore"


class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    REQUEST_CHANGES = "request_changes"
    NEEDS_EXPERT = "needs_expert"
    REJECT = "reject"


class OperationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationType(str, Enum):
    MASK_CANDIDATE = "mask_candidate"
    PROMPT_ENRICHMENT = "prompt_enrichment"
    JOINT_PROMPT_ENRICHMENT = "joint_prompt_enrichment"


class ReleaseStatus(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    ANNOTATION_VALIDATION_FAILED = "annotation_validation_failed"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    VERSION_CONFLICT = "version_conflict"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INVALID_STATE_TRANSITION = "invalid_state_transition"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    REQUEST_TOO_LARGE = "request_too_large"
    QUEUE_FULL = "queue_full"
    MODEL_UNAVAILABLE = "model_unavailable"
    DOWNSTREAM_UNAVAILABLE = "downstream_unavailable"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(StrictModel):
    field: Optional[str] = None
    reason: str


class ErrorPayload(StrictModel):
    request_id: Optional[str] = None
    code: ErrorCode
    message: str
    details: List[ErrorDetail] = Field(default_factory=list)


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadinessResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    dependencies: Dict[str, str] = Field(default_factory=dict)


class Asset(StrictModel):
    asset_id: str
    source_id: Optional[str] = None
    group_id: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    sha256: str
    media_type: Literal["image/jpeg", "image/png"]
    content_url: str
    duplicate_of: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @validator("sha256")
    def sha256_must_be_lowercase_hex(cls, value: str) -> str:
        if len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError("sha256 must be 64 lowercase hexadecimal chars")
        return value


class JobOptions(StrictModel):
    generate_masks: bool = True
    enrich_prompts: bool = True
    prompt_count: Literal[6] = 6
    stop_after: Optional[PipelineStage] = None
    grounding_prompt_normalization_mode: PromptNormalizationMode = (
        "terminal_period"
    )
    grounding_prompt_normalization_profile: PromptNormalizationProfile = (
        "construction_safety_v1"
    )
    grounding_prompt_translation_failure_policy: (
        PromptTranslationFailurePolicy
    ) = "fallback_canonical_terms"

    @root_validator(skip_on_failure=True)
    def staged_execution_must_match_requested_outputs(cls, values):
        if values.get("stop_after") in {
            PipelineStage.GROUNDING_DINO,
            PipelineStage.HAZARD_RULES,
        } and (
            values.get("generate_masks") or values.get("enrich_prompts")
        ):
            raise ValueError(
                "stop_after=grounding_dino or hazard_rules requires "
                "generate_masks=false and enrich_prompts=false"
            )
        if values.get("stop_after") is None and (
            not values.get("generate_masks")
            or not values.get("enrich_prompts")
        ):
            raise ValueError(
                "a full pipeline job requires generate_masks=true and "
                "enrich_prompts=true"
            )
        mode = values.get("grounding_prompt_normalization_mode")
        profile = values.get("grounding_prompt_normalization_profile")
        if (
            mode == "canonical_terms"
            and profile != "construction_safety_v1"
        ):
            raise ValueError(
                "canonical_terms requires profile "
                "construction_safety_v1"
            )
        if (
            mode == "llm_grounding_caption"
            and profile != "open_semantic_zh_en_v1"
        ):
            raise ValueError(
                "llm_grounding_caption requires profile "
                "open_semantic_zh_en_v1"
            )
        return values


class CreateJobRequest(StrictModel):
    asset_ids: List[str] = Field(..., min_items=1, max_items=500)
    grounding_prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )
    grounding_prompt_normalization_mode: PromptNormalizationMode = (
        "terminal_period"
    )
    grounding_prompt_normalization_profile: PromptNormalizationProfile = (
        "construction_safety_v1"
    )
    grounding_prompt_translation_failure_policy: (
        PromptTranslationFailurePolicy
    ) = "fallback_canonical_terms"
    pipeline_version: str = Field(
        default="groundingdino-free-form-v1",
        min_length=1,
        max_length=128,
    )

    @validator("asset_ids")
    def values_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("items must be unique")
        return value

    @validator(
        "grounding_prompt",
        "pipeline_version",
        "grounding_prompt_normalization_profile",
    )
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @validator("grounding_prompt_normalization_mode")
    def mode_must_be_supported(cls, value: str) -> str:
        if value not in {
            "off",
            "terminal_period",
            "canonical_terms",
            "llm_grounding_caption",
        }:
            raise ValueError(
                "grounding_prompt_normalization_mode must be one of: "
                "off, terminal_period, canonical_terms, "
                "llm_grounding_caption"
            )
        return value

    @root_validator(skip_on_failure=True)
    def normalization_mode_must_match_profile(cls, values):
        mode = values.get("grounding_prompt_normalization_mode")
        profile = values.get("grounding_prompt_normalization_profile")
        if (
            mode == "canonical_terms"
            and profile != "construction_safety_v1"
        ):
            raise ValueError(
                "canonical_terms requires profile "
                "construction_safety_v1"
            )
        if (
            mode == "llm_grounding_caption"
            and profile != "open_semantic_zh_en_v1"
        ):
            raise ValueError(
                "llm_grounding_caption requires profile "
                "open_semantic_zh_en_v1"
            )
        return values


class JobProgress(StrictModel):
    total_assets: int = Field(..., ge=0)
    completed_assets: int = Field(..., ge=0)
    generated_tasks: int = Field(..., ge=0)

    @root_validator(skip_on_failure=True)
    def completed_must_not_exceed_total(cls, values):
        completed = values.get("completed_assets")
        total = values.get("total_assets")
        if completed is not None and total is not None and completed > total:
            raise ValueError("completed_assets must not exceed total_assets")
        return values


class StageResult(StrictModel):
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    message: Optional[str] = None


class JobError(StrictModel):
    asset_id: Optional[str] = None
    stage: Optional[Literal["grounding_dino"]] = None
    code: str
    message: str


class DetectionJobProgress(StrictModel):
    total_assets: int = Field(..., ge=0)
    completed_assets: int = Field(..., ge=0)

    @root_validator(skip_on_failure=True)
    def completed_must_not_exceed_total(cls, values):
        completed = values.get("completed_assets")
        total = values.get("total_assets")
        if completed is not None and total is not None and completed > total:
            raise ValueError("completed_assets must not exceed total_assets")
        return values


class GroundingPromptRoute(StrictModel):
    rule_attempted: bool
    rule_matched: bool
    llm_attempted: bool
    llm_succeeded: bool
    fallback_used: bool


class Job(StrictModel):
    job_id: str
    status: JobStatus
    stage: Optional[Literal["grounding_dino"]] = None
    pipeline_version: str
    grounding_prompt: str
    grounding_prompt_normalization_mode: PromptNormalizationMode = (
        "terminal_period"
    )
    grounding_prompt_normalization_profile: PromptNormalizationProfile = (
        "construction_safety_v1"
    )
    grounding_prompt_translation_failure_policy: (
        PromptTranslationFailurePolicy
    ) = "fallback_canonical_terms"
    grounding_prompt_route: Optional[GroundingPromptRoute] = None
    progress: DetectionJobProgress
    stages: Dict[Literal["grounding_dino"], StageResult]
    errors: List[JobError] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def _validate_box(value: List[float]) -> List[float]:
    if len(value) != 4:
        raise ValueError("box_xyxy must contain exactly four coordinates")
    if any(coordinate < 0 for coordinate in value):
        raise ValueError("box coordinates must be non-negative")
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box_xyxy must have positive width and height")
    return value


def _validate_points(value: List[List[float]]) -> List[List[float]]:
    if len(value) < 3:
        raise ValueError("polygon must contain at least three points")
    for point in value:
        if len(point) != 2:
            raise ValueError("each polygon point must contain x and y")
        if any(coordinate < 0 for coordinate in point):
            raise ValueError("polygon coordinates must be non-negative")
    return value


class PolygonShape(StrictModel):
    shape_id: str
    label: ShapeLabel
    shape_type: Literal["polygon"] = "polygon"
    points: List[List[float]]

    _points_are_valid = validator("points", allow_reuse=True)(_validate_points)


class AnnotationPrompt(StrictModel):
    prompt_id: str
    type: PromptType
    text: str = Field(..., min_length=1, max_length=200)

    @validator("text")
    def prompt_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt text must not be blank")
        return normalized


class AnnotationContent(StrictModel):
    target_object: str = Field(..., max_length=300)
    instance_count: int = Field(..., ge=1)
    visual_anchor: List[str] = Field(default_factory=list, max_items=10)
    mask_granularity: str = Field(..., max_length=200)
    risk_semantics: Optional[str] = Field(default=None, max_length=500)
    shapes: List[PolygonShape] = Field(default_factory=list)
    prompts: List[AnnotationPrompt] = Field(
        default_factory=list,
        max_items=6,
    )


class Detection(StrictModel):
    detection_id: str
    entity: str
    box_xyxy: List[float]
    box_score: float = Field(..., ge=0, le=1)
    phrase_score: float = Field(..., ge=0, le=1)

    _box_is_valid = validator("box_xyxy", allow_reuse=True)(_validate_box)


class JobDetection(Detection):
    asset_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class JobDetectionsResponse(StrictModel):
    job_id: str
    items: List[JobDetection] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class HazardCandidate(StrictModel):
    hazard_id: str
    asset_id: str
    category: AnnotationCategory
    target_entity: str
    target_detection_ids: List[str] = Field(..., min_items=1)
    box_xyxy: List[float]
    confidence: float = Field(..., ge=0, le=1)
    rule_id: str
    rule_version: str
    evidence: List[str] = Field(..., min_items=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    _box_is_valid = validator("box_xyxy", allow_reuse=True)(_validate_box)


class JobHazardCandidatesResponse(StrictModel):
    job_id: str
    items: List[HazardCandidate] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class ReviewTaskBuildItem(StrictModel):
    detection_id: str
    task_id: str
    task_version: int = Field(..., ge=1)
    asset_id: str
    box_xyxy: List[float]
    created: bool

    _box_is_valid = validator("box_xyxy", allow_reuse=True)(_validate_box)


class DetectionOverlapWarning(StrictModel):
    detection_ids: List[str] = Field(..., min_items=2, max_items=2)
    box_iou: float = Field(..., ge=0, le=1)
    message: str


class BuildReviewTasksResponse(StrictModel):
    job_id: str
    task_ids: List[str] = Field(default_factory=list)
    created_count: int = Field(..., ge=0)
    existing_count: int = Field(..., ge=0)
    items: List[ReviewTaskBuildItem] = Field(default_factory=list)
    overlap_warnings: List[DetectionOverlapWarning] = Field(
        default_factory=list
    )


class BuildDetectionTasksRequest(StrictModel):
    detection_ids: Optional[List[str]] = Field(
        default=None,
        min_items=1,
        max_items=500,
    )
    category: AnnotationCategory = AnnotationCategory.UNSAFE

    @validator("detection_ids")
    def detection_ids_must_be_unique(cls, value):
        if value is not None and len(value) != len(set(value)):
            raise ValueError("detection_ids must be unique")
        return value


class CancelJobRequest(StrictModel):
    actor_id: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(..., min_length=1, max_length=2000)

    @validator("actor_id", "reason")
    def cancellation_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class TaskAsset(StrictModel):
    asset_id: str
    group_id: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    image_url: str


class ArtifactLinks(StrictModel):
    detection_overlay_url: Optional[str] = None
    mask_overlay_url: Optional[str] = None
    mask_png_url: Optional[str] = None
    crop_url: Optional[str] = None


class Provenance(StrictModel):
    pipeline_version: str
    source_detection_id: Optional[str] = None
    grounding_prompt: Optional[str] = None
    hazard_candidate_id: Optional[str] = None
    review_task_builder_version: Optional[str] = None
    grounding_dino_version: Optional[str] = None
    grounding_dino_prompt_version: Optional[str] = None
    grounding_dino_thresholds: Dict[str, float] = Field(
        default_factory=dict
    )
    hazard_rule_version: Optional[str] = None
    sam_version: Optional[str] = None
    sam_qc_version: Optional[str] = None
    qwen_provider: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_facts_prompt_version: Optional[str] = None
    qwen_enrichment_prompt_version: Optional[str] = None


class AnnotationTask(StrictModel):
    task_id: str
    job_id: str
    asset: TaskAsset
    category: AnnotationCategory
    status: TaskStatus
    version: int = Field(..., ge=1)
    detections: List[Detection] = Field(default_factory=list)
    annotation: AnnotationContent
    artifacts: ArtifactLinks = Field(default_factory=ArtifactLinks)
    provenance: Provenance
    source_detection_id: Optional[str] = None
    source_hazard: Optional[HazardCandidate] = None
    warnings: List[str] = Field(default_factory=list)
    primary_result: Optional[BadCaseType] = None
    annotator_id: Optional[str] = None
    reviewer_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TaskSummary(StrictModel):
    task_id: str
    asset_id: str
    group_id: str
    category: AnnotationCategory
    status: TaskStatus
    version: int = Field(..., ge=1)
    source_detection_id: Optional[str] = None
    source_hazard_id: Optional[str] = None
    primary_result: Optional[BadCaseType] = None
    annotator_id: Optional[str] = None
    reviewer_id: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TaskList(StrictModel):
    items: List[TaskSummary] = Field(default_factory=list)
    next_cursor: Optional[str] = None


class TaskVersionRecord(StrictModel):
    version: int = Field(..., ge=1)
    annotation: AnnotationContent
    status: TaskStatus
    editor_id: Optional[str] = None
    change_kind: Literal[
        "generated",
        "draft",
        "submit",
        "review",
        "freeze",
        "invalidate",
    ]
    comment: Optional[str] = None
    created_at: datetime


class TaskVersionList(StrictModel):
    task_id: str
    items: List[TaskVersionRecord] = Field(default_factory=list)


class ReviewRecord(StrictModel):
    review_id: str
    task_id: str
    task_version: int = Field(..., ge=1)
    reviewer_id: str
    decision: ReviewDecision
    primary_result: BadCaseType
    comment: Optional[str] = None
    created_at: datetime


class ReviewList(StrictModel):
    task_id: str
    items: List[ReviewRecord] = Field(default_factory=list)


class SaveDraftRequest(StrictModel):
    expected_version: int = Field(..., ge=1)
    annotation: AnnotationContent
    editor_id: str = Field(..., min_length=1, max_length=128)


class SubmitTaskRequest(StrictModel):
    expected_version: int = Field(..., ge=1)
    annotator_id: str = Field(..., min_length=1, max_length=128)
    primary_result: BadCaseType
    comment: Optional[str] = Field(default=None, max_length=2000)


class ReviewTaskRequest(StrictModel):
    expected_version: int = Field(..., ge=1)
    reviewer_id: str = Field(..., min_length=1, max_length=128)
    decision: ReviewDecision
    primary_result: BadCaseType
    comment: Optional[str] = Field(default=None, max_length=2000)


class InvalidateTaskRequest(StrictModel):
    expected_version: int = Field(..., ge=1)
    actor_id: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(..., min_length=1, max_length=2000)

    @validator("actor_id", "reason")
    def invalidation_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class CreateMaskCandidateRequest(StrictModel):
    expected_version: int = Field(..., ge=1)
    box_xyxy: List[float]

    _box_is_valid = validator("box_xyxy", allow_reuse=True)(_validate_box)


class CreatePromptEnrichmentRequest(StrictModel):
    expected_version: int = Field(..., ge=1)


class BatchMaskCandidateItem(StrictModel):
    task_id: str = Field(..., min_length=1, max_length=128)
    expected_version: int = Field(..., ge=1)
    box_xyxy: List[float]

    _box_is_valid = validator("box_xyxy", allow_reuse=True)(_validate_box)


class BatchMaskCandidatesRequest(StrictModel):
    items: List[BatchMaskCandidateItem] = Field(
        ...,
        min_items=1,
        max_items=500,
    )

    @validator("items")
    def task_ids_must_be_unique(cls, value):
        task_ids = [item.task_id for item in value]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("batch task_id values must be unique")
        return value


class BatchPromptEnrichmentItem(StrictModel):
    task_id: str = Field(..., min_length=1, max_length=128)
    expected_version: int = Field(..., ge=1)


class BatchPromptEnrichmentsRequest(StrictModel):
    items: List[BatchPromptEnrichmentItem] = Field(
        ...,
        min_items=1,
        max_items=500,
    )

    @validator("items")
    def task_ids_must_be_unique(cls, value):
        task_ids = [item.task_id for item in value]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("batch task_id values must be unique")
        return value


class JointPromptEnrichmentRequest(StrictModel):
    items: List[BatchPromptEnrichmentItem] = Field(
        ...,
        min_items=2,
        max_items=16,
    )
    mode: Literal["joint"] = "joint"

    @validator("items")
    def task_ids_must_be_unique(cls, value):
        task_ids = [item.task_id for item in value]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("joint task_id values must be unique")
        return value


class CancelOperationRequest(StrictModel):
    actor_id: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(..., min_length=1, max_length=2000)

    @validator("actor_id", "reason")
    def operation_cancellation_text_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class OperationAccepted(StrictModel):
    operation_id: str
    status: Literal["queued", "running"] = "queued"
    created_at: datetime


class BatchOperationItemResult(StrictModel):
    task_id: str
    operation_id: Optional[str] = None
    status: Literal["queued", "running", "rejected"]
    created_at: Optional[datetime] = None
    error: Optional[ErrorPayload] = None


class BatchOperationsAccepted(StrictModel):
    items: List[BatchOperationItemResult] = Field(default_factory=list)
    accepted_count: int = Field(..., ge=0)
    rejected_count: int = Field(..., ge=0)


class TaskGroupOperationAccepted(StrictModel):
    task_group_id: str
    operation_id: str
    status: Literal["queued", "running"] = "queued"
    created_at: datetime


class TaskGroupMember(StrictModel):
    task_id: str
    task_version: int = Field(..., ge=1)


class TaskGroup(StrictModel):
    task_group_id: str
    asset_id: str
    mode: Literal["joint"] = "joint"
    source_task_ids: List[str] = Field(..., min_items=2, max_items=16)
    items: List[TaskGroupMember] = Field(..., min_items=2, max_items=16)
    operation_id: str
    status: OperationStatus
    facts: Optional[Dict[str, Any]] = None
    prompts: List[AnnotationPrompt] = Field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = None
    error: Optional[ErrorPayload] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AnnotationOperation(StrictModel):
    operation_id: str
    operation_type: OperationType
    task_id: Optional[str] = None
    task_version: Optional[int] = Field(default=None, ge=1)
    task_group_id: Optional[str] = None
    status: OperationStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorPayload] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @root_validator(skip_on_failure=True)
    def operation_has_one_owner(cls, values):
        operation_type = values.get("operation_type")
        task_id = values.get("task_id")
        task_version = values.get("task_version")
        task_group_id = values.get("task_group_id")
        if operation_type == OperationType.JOINT_PROMPT_ENRICHMENT:
            if not task_group_id or task_id is not None or task_version is not None:
                raise ValueError(
                    "joint prompt operation must belong only to a Task Group"
                )
        elif (
            not task_id
            or task_version is None
            or task_group_id is not None
        ):
            raise ValueError(
                "single-task operation must belong only to one Task"
            )
        return values


class SplitPolicy(StrictModel):
    type: Literal["grouped"] = "grouped"
    group_field: Literal["group_id"] = "group_id"
    train_ratio: float = Field(..., ge=0, le=1)
    val_ratio: float = Field(..., ge=0, le=1)
    golden_ratio: float = Field(..., ge=0, le=1)
    seed: int

    @root_validator(skip_on_failure=True)
    def ratios_must_sum_to_one(cls, values):
        ratios = [
            values.get("train_ratio"),
            values.get("val_ratio"),
            values.get("golden_ratio"),
        ]
        if all(value is not None for value in ratios) and not isclose(
            sum(ratios),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("split ratios must sum to 1.0")
        return values


class ReleaseTaskFilter(StrictModel):
    status: Literal["accepted"] = "accepted"
    categories: Optional[List[AnnotationCategory]] = Field(
        default=None,
        min_items=1,
    )

    @validator("categories")
    def categories_must_be_unique(cls, value):
        if value is not None and len(value) != len(set(value)):
            raise ValueError("categories must be unique")
        return value


class CreateReleaseRequest(StrictModel):
    name: str
    task_filter: ReleaseTaskFilter
    split_policy: SplitPolicy

    @validator("name")
    def name_must_be_safe(cls, value: str) -> str:
        if not value or len(value) > 128:
            raise ValueError("release name must contain 1 to 128 characters")
        if not value[0].isalnum() or any(
            not (char.isalnum() or char in "._-") for char in value
        ):
            raise ValueError(
                "release name may contain only letters, digits, '.', '_', "
                "or '-' and must start with an alphanumeric character"
            )
        return value


class ReleaseCounts(StrictModel):
    train: int = Field(..., ge=0)
    val: int = Field(..., ge=0)
    golden: int = Field(..., ge=0)


class Release(StrictModel):
    release_id: str
    name: str
    status: ReleaseStatus
    counts: Optional[ReleaseCounts] = None
    manifest_url: Optional[str] = None
    archive_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
