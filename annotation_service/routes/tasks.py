from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from ..auth import AuthDependency
from ..errors import StorageUnavailableError
from ..schemas import (
    AnnotationCategory,
    AnnotationTask,
    BadCaseType,
    CreateMaskCandidateRequest,
    CreatePromptEnrichmentRequest,
    ErrorPayload,
    InvalidateTaskRequest,
    OperationAccepted,
    ReviewList,
    ReviewTaskRequest,
    SaveDraftRequest,
    SubmitTaskRequest,
    TaskList,
    TaskStatus,
    TaskVersionList,
)
from ..storage import AnnotationStore


COMMON_RESPONSES = {
    401: {"model": ErrorPayload, "description": "Authentication failed."},
    404: {"model": ErrorPayload, "description": "Task was not found."},
    503: {"model": ErrorPayload, "description": "Storage is unavailable."},
}
WRITE_RESPONSES = {
    **COMMON_RESPONSES,
    409: {
        "model": ErrorPayload,
        "description": "Version conflict or invalid state transition.",
    },
    422: {
        "model": ErrorPayload,
        "description": "Annotation payload is invalid.",
    },
}


def _model_json(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(model.json())


def build_tasks_router(
    *,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/tasks",
        tags=["Tasks"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    @router.get(
        "",
        operation_id="listAnnotationTasks",
        response_model=TaskList,
        dependencies=[Depends(authenticate)],
        responses={
            401: COMMON_RESPONSES[401],
            422: {
                "model": ErrorPayload,
                "description": "Task filters or cursor are invalid.",
            },
            503: COMMON_RESPONSES[503],
        },
    )
    async def list_annotation_tasks(
        status: TaskStatus | None = Query(default=None),
        category: AnnotationCategory | None = Query(default=None),
        group_id: str | None = Query(default=None, max_length=256),
        job_id: str | None = Query(default=None, max_length=128),
        annotator_id: str | None = Query(default=None, max_length=128),
        reviewer_id: str | None = Query(default=None, max_length=128),
        bad_case_type: BadCaseType | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = Query(default=None, min_length=1),
    ) -> dict[str, Any]:
        store = require_storage()
        return await asyncio.to_thread(
            store.list_tasks,
            status=status,
            category=category,
            group_id=group_id,
            job_id=job_id,
            annotator_id=annotator_id,
            reviewer_id=reviewer_id,
            bad_case_type=bad_case_type,
            limit=limit,
            cursor=cursor,
        )

    @router.get(
        "/{task_id}",
        operation_id="getAnnotationTask",
        response_model=AnnotationTask,
        dependencies=[Depends(authenticate)],
        responses=COMMON_RESPONSES,
    )
    async def get_annotation_task(task_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().get_task,
            task_id,
        )

    @router.get(
        "/{task_id}/versions",
        operation_id="listAnnotationTaskVersions",
        response_model=TaskVersionList,
        dependencies=[Depends(authenticate)],
        responses=COMMON_RESPONSES,
    )
    async def list_annotation_task_versions(
        task_id: str,
    ) -> dict[str, Any]:
        items = await asyncio.to_thread(
            require_storage().list_task_versions,
            task_id,
        )
        return {"task_id": task_id, "items": items}

    @router.get(
        "/{task_id}/reviews",
        operation_id="listAnnotationTaskReviews",
        response_model=ReviewList,
        dependencies=[Depends(authenticate)],
        responses=COMMON_RESPONSES,
    )
    async def list_annotation_task_reviews(
        task_id: str,
    ) -> dict[str, Any]:
        items = await asyncio.to_thread(
            require_storage().list_reviews,
            task_id,
        )
        return {"task_id": task_id, "items": items}

    @router.put(
        "/{task_id}/draft",
        operation_id="saveAnnotationDraft",
        response_model=AnnotationTask,
        dependencies=[Depends(authenticate)],
        responses=WRITE_RESPONSES,
    )
    async def save_annotation_draft(
        task_id: str,
        request: SaveDraftRequest,
    ) -> dict[str, Any]:
        payload = _model_json(request)
        return await asyncio.to_thread(
            require_storage().save_task_draft,
            task_id,
            expected_version=payload["expected_version"],
            annotation=payload["annotation"],
            editor_id=payload["editor_id"],
        )

    @router.post(
        "/{task_id}/submit",
        operation_id="submitAnnotationTask",
        response_model=AnnotationTask,
        dependencies=[Depends(authenticate)],
        responses=WRITE_RESPONSES,
    )
    async def submit_annotation_task(
        task_id: str,
        request: SubmitTaskRequest,
    ) -> dict[str, Any]:
        store = require_storage()
        payload = _model_json(request)
        return await asyncio.to_thread(
            store.submit_task,
            task_id,
            expected_version=payload["expected_version"],
            annotator_id=payload["annotator_id"],
            primary_result=payload["primary_result"],
            comment=payload["comment"],
        )

    @router.post(
        "/{task_id}/invalidate",
        operation_id="invalidateAnnotationTask",
        response_model=AnnotationTask,
        dependencies=[Depends(authenticate)],
        responses=WRITE_RESPONSES,
    )
    async def invalidate_annotation_task(
        task_id: str,
        request: InvalidateTaskRequest,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().invalidate_task,
            task_id,
            expected_version=request.expected_version,
            actor_id=request.actor_id,
            reason=request.reason,
        )

    @router.post(
        "/{task_id}/review",
        operation_id="reviewAnnotationTask",
        response_model=AnnotationTask,
        dependencies=[Depends(authenticate)],
        responses=WRITE_RESPONSES,
    )
    async def review_annotation_task(
        task_id: str,
        request: ReviewTaskRequest,
    ) -> dict[str, Any]:
        store = require_storage()
        payload = _model_json(request)
        return await asyncio.to_thread(
            store.review_task,
            task_id,
            expected_version=payload["expected_version"],
            reviewer_id=payload["reviewer_id"],
            decision=payload["decision"],
            primary_result=payload["primary_result"],
            comment=payload["comment"],
        )

    @router.post(
        "/{task_id}/mask-candidates",
        operation_id="createMaskCandidate",
        response_model=OperationAccepted,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses={
            **WRITE_RESPONSES,
            202: {
                "model": OperationAccepted,
                "description": "SAM mask generation operation accepted.",
            },
        },
    )
    async def create_mask_candidate(
        task_id: str,
        request: CreateMaskCandidateRequest,
    ) -> dict[str, Any]:
        operation = await asyncio.to_thread(
            require_storage().create_mask_candidate_operation,
            task_id=task_id,
            expected_version=request.expected_version,
            box_xyxy=request.box_xyxy,
        )
        return {
            "operation_id": operation["operation_id"],
            "status": operation["status"],
            "created_at": operation["created_at"],
        }

    @router.post(
        "/{task_id}/prompt-enrichments",
        operation_id="createPromptEnrichment",
        response_model=OperationAccepted,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses={
            **WRITE_RESPONSES,
            202: {
                "model": OperationAccepted,
                "description": (
                    "Qwen2.5-VL prompt generation operation accepted."
                ),
            },
        },
    )
    async def create_prompt_enrichment(
        task_id: str,
        request: CreatePromptEnrichmentRequest,
    ) -> dict[str, Any]:
        operation = await asyncio.to_thread(
            require_storage().create_prompt_enrichment_operation,
            task_id=task_id,
            expected_version=request.expected_version,
        )
        return {
            "operation_id": operation["operation_id"],
            "status": operation["status"],
            "created_at": operation["created_at"],
        }

    @router.get(
        "/{task_id}/artifacts/{artifact_type}",
        operation_id="getTaskArtifact",
        response_class=FileResponse,
        dependencies=[Depends(authenticate)],
        responses={
            **COMMON_RESPONSES,
            200: {
                "description": "Task image artifact.",
                "content": {
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                    "image/png": {"schema": {"type": "string", "format": "binary"}},
                },
            },
        },
    )
    async def get_task_artifact(
        task_id: str,
        artifact_type: Literal[
            "detections",
            "mask",
            "mask-overlay",
            "crop",
        ],
    ) -> FileResponse:
        path, media_type = await asyncio.to_thread(
            require_storage().artifact_file,
            task_id,
            artifact_type,
        )
        return FileResponse(path, media_type=media_type)

    return router
