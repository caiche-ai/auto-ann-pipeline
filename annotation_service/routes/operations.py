from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends

from ..auth import AuthDependency
from ..errors import StorageUnavailableError
from ..schemas import (
    AnnotationOperation,
    CancelOperationRequest,
    ErrorPayload,
)
from ..storage import AnnotationStore


def build_operations_router(
    *,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/operations",
        tags=["Operations"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    @router.get(
        "/{operation_id}",
        operation_id="getAnnotationOperation",
        response_model=AnnotationOperation,
        dependencies=[Depends(authenticate)],
        responses={
            401: {
                "model": ErrorPayload,
                "description": "Authentication failed.",
            },
            404: {
                "model": ErrorPayload,
                "description": "Operation was not found.",
            },
            503: {
                "model": ErrorPayload,
                "description": "Storage is unavailable.",
            },
        },
    )
    async def get_annotation_operation(
        operation_id: str,
    ) -> dict[str, Any]:
        operation = await asyncio.to_thread(
            require_storage().get_operation,
            operation_id,
        )
        return {
            "operation_id": operation["operation_id"],
            "operation_type": operation["operation_type"],
            "task_id": operation["task_id"],
            "task_version": operation["task_version"],
            "task_group_id": operation["task_group_id"],
            "status": operation["status"],
            "result": operation["result"],
            "error": operation["error"],
            "created_at": operation["created_at"],
            "started_at": operation["started_at"],
            "completed_at": operation["completed_at"],
        }

    @router.post(
        "/{operation_id}/cancel",
        operation_id="cancelAnnotationOperation",
        response_model=AnnotationOperation,
        dependencies=[Depends(authenticate)],
        responses={
            401: {
                "model": ErrorPayload,
                "description": "Authentication failed.",
            },
            404: {
                "model": ErrorPayload,
                "description": "Operation was not found.",
            },
            409: {
                "model": ErrorPayload,
                "description": "Operation is already in a terminal state.",
            },
            503: {
                "model": ErrorPayload,
                "description": "Storage is unavailable.",
            },
        },
    )
    async def cancel_annotation_operation(
        operation_id: str,
        request: CancelOperationRequest,
    ) -> dict[str, Any]:
        operation = await asyncio.to_thread(
            require_storage().cancel_operation,
            operation_id,
            actor_id=request.actor_id,
            reason=request.reason,
        )
        return {
            "operation_id": operation["operation_id"],
            "operation_type": operation["operation_type"],
            "task_id": operation["task_id"],
            "task_version": operation["task_version"],
            "task_group_id": operation["task_group_id"],
            "status": operation["status"],
            "result": operation["result"],
            "error": operation["error"],
            "created_at": operation["created_at"],
            "started_at": operation["started_at"],
            "completed_at": operation["completed_at"],
        }

    return router
