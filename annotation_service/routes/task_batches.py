from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends

from ..auth import AuthDependency
from ..errors import ServiceError, StorageUnavailableError
from ..schemas import (
    BatchMaskCandidatesRequest,
    BatchOperationsAccepted,
    BatchPromptEnrichmentsRequest,
    ErrorPayload,
)
from ..storage import AnnotationStore


def _model_json(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(model.json())


def _rejected_item(
    task_id: str,
    error: ServiceError | ValueError,
) -> dict[str, Any]:
    if isinstance(error, ServiceError):
        code = error.code
        message = error.message
        details = error.details
    else:
        code = "validation_error"
        message = str(error)
        details = []
    return {
        "task_id": task_id,
        "operation_id": None,
        "status": "rejected",
        "created_at": None,
        "error": {
            "request_id": None,
            "code": code,
            "message": message,
            "details": details,
        },
    }


def _accepted_item(
    task_id: str,
    operation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "operation_id": operation["operation_id"],
        "status": operation["status"],
        "created_at": operation["created_at"],
        "error": None,
    }


def _batch_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = sum(item["status"] == "rejected" for item in items)
    return {
        "items": items,
        "accepted_count": len(items) - rejected,
        "rejected_count": rejected,
    }


def build_task_batches_router(
    *,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/task-batches",
        tags=["Task Batches"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    @router.post(
        "/mask-candidates",
        operation_id="createBatchMaskCandidates",
        response_model=BatchOperationsAccepted,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses={
            202: {
                "model": BatchOperationsAccepted,
                "description": (
                    "One SAM operation is accepted per valid Task. Rejected "
                    "items do not prevent other Tasks from being queued."
                ),
            },
            401: {
                "model": ErrorPayload,
                "description": "Authentication failed.",
            },
            422: {
                "model": ErrorPayload,
                "description": "Batch request structure is invalid.",
            },
            503: {
                "model": ErrorPayload,
                "description": "Storage is unavailable.",
            },
        },
    )
    async def create_batch_mask_candidates(
        request: BatchMaskCandidatesRequest,
    ) -> dict[str, Any]:
        store = require_storage()
        items: list[dict[str, Any]] = []
        for item in _model_json(request)["items"]:
            try:
                operation = await asyncio.to_thread(
                    store.create_mask_candidate_operation,
                    task_id=item["task_id"],
                    expected_version=item["expected_version"],
                    box_xyxy=item["box_xyxy"],
                )
                items.append(
                    _accepted_item(item["task_id"], operation)
                )
            except (ServiceError, ValueError) as exc:
                items.append(_rejected_item(item["task_id"], exc))
        return _batch_response(items)

    @router.post(
        "/prompt-enrichments",
        operation_id="createBatchPromptEnrichments",
        response_model=BatchOperationsAccepted,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses={
            202: {
                "model": BatchOperationsAccepted,
                "description": (
                    "One Qwen operation is accepted per valid Task. Rejected "
                    "items do not prevent other Tasks from being queued."
                ),
            },
            401: {
                "model": ErrorPayload,
                "description": "Authentication failed.",
            },
            422: {
                "model": ErrorPayload,
                "description": "Batch request structure is invalid.",
            },
            503: {
                "model": ErrorPayload,
                "description": "Storage is unavailable.",
            },
        },
    )
    async def create_batch_prompt_enrichments(
        request: BatchPromptEnrichmentsRequest,
    ) -> dict[str, Any]:
        store = require_storage()
        items: list[dict[str, Any]] = []
        for item in _model_json(request)["items"]:
            try:
                operation = await asyncio.to_thread(
                    store.create_prompt_enrichment_operation,
                    task_id=item["task_id"],
                    expected_version=item["expected_version"],
                )
                items.append(
                    _accepted_item(item["task_id"], operation)
                )
            except (ServiceError, ValueError) as exc:
                items.append(_rejected_item(item["task_id"], exc))
        return _batch_response(items)

    return router
