from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import FileResponse

from ..auth import AuthDependency
from ..config import Settings
from ..errors import StorageUnavailableError
from ..schemas import (
    BuildDetectionTasksRequest,
    BuildReviewTasksResponse,
    CancelJobRequest,
    CreateJobRequest,
    ErrorPayload,
    Job,
    JobDetectionsResponse,
)
from ..review_task_builder import build_detection_review_tasks
from ..storage import AnnotationStore


COMMON_RESPONSES = {
    401: {"model": ErrorPayload, "description": "Authentication failed."},
    503: {"model": ErrorPayload, "description": "Storage is unavailable."},
}
CREATE_JOB_RESPONSES = {
    **COMMON_RESPONSES,
    404: {"model": ErrorPayload, "description": "Asset was not found."},
    409: {"model": ErrorPayload, "description": "Idempotency conflict."},
    422: {"model": ErrorPayload, "description": "Job request is invalid."},
    429: {"model": ErrorPayload, "description": "Job queue is full."},
}
GET_JOB_RESPONSES = {
    **COMMON_RESPONSES,
    404: {"model": ErrorPayload, "description": "Job was not found."},
}
PUBLIC_JOB_FIELDS = {
    "job_id",
    "status",
    "stage",
    "pipeline_version",
    "grounding_prompt",
    "grounding_prompt_normalization_mode",
    "grounding_prompt_normalization_profile",
    "grounding_prompt_translation_failure_policy",
    "grounding_prompt_route",
    "progress",
    "stages",
    "errors",
    "created_at",
    "started_at",
    "completed_at",
}


def _model_json(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(model.json())


def _public_job(payload: dict[str, Any]) -> dict[str, Any]:
    public = {
        field: payload.get(field)
        for field in PUBLIC_JOB_FIELDS
    }
    public["progress"] = {
        "total_assets": payload["progress"]["total_assets"],
        "completed_assets": payload["progress"]["completed_assets"],
    }
    public["stages"] = {
        "grounding_dino": payload["stages"]["grounding_dino"]
    }
    return public


def build_jobs_router(
    *,
    settings: Settings,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/jobs",
        tags=["Jobs"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    @router.post(
        "",
        operation_id="createAnnotationJob",
        response_model=Job,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses=CREATE_JOB_RESPONSES,
    )
    async def create_annotation_job(
        request: CreateJobRequest,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
        ),
    ) -> dict[str, Any]:
        store = require_storage()
        request_payload = _model_json(request)
        fields_set = getattr(request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(request, "__fields_set__", set())
        if "grounding_prompt_normalization_mode" not in fields_set:
            request_payload["grounding_prompt_normalization_mode"] = (
                settings.prompt_normalization_mode
            )
        if "grounding_prompt_normalization_profile" not in fields_set:
            request_payload["grounding_prompt_normalization_profile"] = (
                settings.prompt_normalization_profile
            )
        if "grounding_prompt_translation_failure_policy" not in fields_set:
            request_payload[
                "grounding_prompt_translation_failure_policy"
            ] = settings.prompt_translation_failure_policy
        job = await asyncio.to_thread(
            store.create_job,
            asset_ids=request_payload["asset_ids"],
            grounding_prompt=request_payload["grounding_prompt"],
            pipeline_version=request_payload["pipeline_version"],
            options={
                "generate_masks": False,
                "enrich_prompts": False,
                "prompt_count": 6,
                "stop_after": "grounding_dino",
                "grounding_prompt_normalization_mode": (
                    request_payload[
                        "grounding_prompt_normalization_mode"
                    ]
                ),
                "grounding_prompt_normalization_profile": (
                    request_payload[
                        "grounding_prompt_normalization_profile"
                    ]
                ),
                "grounding_prompt_translation_failure_policy": (
                    request_payload[
                        "grounding_prompt_translation_failure_policy"
                    ]
                ),
            },
            max_queued_jobs=settings.max_queued_jobs,
            idempotency_key=idempotency_key,
            idempotency_request=(
                request_payload if idempotency_key is not None else None
            ),
        )
        return _public_job(job)

    @router.get(
        "/{job_id}",
        operation_id="getAnnotationJob",
        response_model=Job,
        dependencies=[Depends(authenticate)],
        responses=GET_JOB_RESPONSES,
    )
    async def get_annotation_job(job_id: str) -> dict[str, Any]:
        store = require_storage()
        job = await asyncio.to_thread(store.get_job, job_id)
        return _public_job(job)

    @router.post(
        "/{job_id}/cancel",
        operation_id="cancelAnnotationJob",
        response_model=Job,
        dependencies=[Depends(authenticate)],
        responses={
            **GET_JOB_RESPONSES,
            409: {
                "model": ErrorPayload,
                "description": "Job is already in a terminal state.",
            },
        },
    )
    async def cancel_annotation_job(
        job_id: str,
        request: CancelJobRequest,
    ) -> dict[str, Any]:
        job = await asyncio.to_thread(
            require_storage().cancel_job,
            job_id,
            actor_id=request.actor_id,
            reason=request.reason,
        )
        return _public_job(job)

    @router.get(
        "/{job_id}/detections",
        operation_id="listAnnotationJobDetections",
        response_model=JobDetectionsResponse,
        dependencies=[Depends(authenticate)],
        responses=GET_JOB_RESPONSES,
    )
    async def list_annotation_job_detections(
        job_id: str,
        asset_id: str | None = Query(
            default=None,
            min_length=1,
            max_length=128,
        ),
    ) -> dict[str, Any]:
        store = require_storage()
        items = await asyncio.to_thread(
            store.list_job_detections,
            job_id=job_id,
            asset_id=asset_id,
        )
        return {
            "job_id": job_id,
            "items": items,
            "total": len(items),
        }

    @router.post(
        "/{job_id}/review-tasks",
        operation_id="buildDetectionReviewTasks",
        response_model=BuildReviewTasksResponse,
        dependencies=[Depends(authenticate)],
        responses={
            **GET_JOB_RESPONSES,
            409: {
                "model": ErrorPayload,
                "description": "Detection job has not completed.",
            },
            422: {
                "model": ErrorPayload,
                "description": "Detection selection is invalid.",
            },
        },
    )
    async def build_detection_tasks(
        job_id: str,
        request: BuildDetectionTasksRequest,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            build_detection_review_tasks,
            require_storage(),
            job_id=job_id,
            detection_ids=request.detection_ids,
            category=request.category.value,
        )

    @router.get(
        "/{job_id}/assets/{asset_id}/bbox-image",
        operation_id="getAnnotationJobBoundingBoxImage",
        response_class=FileResponse,
        dependencies=[Depends(authenticate)],
        responses={
            **GET_JOB_RESPONSES,
            200: {
                "description": "PNG image with GroundingDINO bounding boxes.",
                "content": {
                    "image/png": {
                        "schema": {
                            "type": "string",
                            "format": "binary",
                        }
                    }
                },
            },
        },
    )
    async def get_annotation_job_bbox_image(
        job_id: str,
        asset_id: str,
    ) -> FileResponse:
        path, media_type, sha256 = await asyncio.to_thread(
            require_storage().job_artifact_file,
            job_id=job_id,
            asset_id=asset_id,
            artifact_type="bbox-image",
        )
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "ETag": f'"{sha256}"',
                "Cache-Control": "private, no-cache",
            },
        )

    return router
