from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends

from ..auth import AuthDependency
from ..errors import StorageUnavailableError
from ..schemas import (
    ErrorPayload,
    JointPromptEnrichmentRequest,
    TaskGroup,
    TaskGroupOperationAccepted,
)
from ..storage import AnnotationStore


def _model_json(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return model.dict()


def build_task_groups_router(
    *,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/task-groups",
        tags=["Task Groups"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    common_responses = {
        401: {
            "model": ErrorPayload,
            "description": "Authentication failed.",
        },
        404: {
            "model": ErrorPayload,
            "description": "Task or Task Group was not found.",
        },
        409: {
            "model": ErrorPayload,
            "description": "At least one Task version changed.",
        },
        422: {
            "model": ErrorPayload,
            "description": (
                "Tasks are invalid, use different images, or lack mask/crop "
                "artifacts."
            ),
        },
        503: {
            "model": ErrorPayload,
            "description": "Storage is unavailable.",
        },
    }

    @router.post(
        "/prompt-enrichments",
        operation_id="createJointPromptEnrichment",
        response_model=TaskGroupOperationAccepted,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses={
            **common_responses,
            202: {
                "model": TaskGroupOperationAccepted,
                "description": (
                    "One Qwen operation is accepted for the complete Task "
                    "Group."
                ),
            },
        },
    )
    async def create_joint_prompt_enrichment(
        request: JointPromptEnrichmentRequest,
    ) -> dict[str, Any]:
        payload = _model_json(request)
        operation = await asyncio.to_thread(
            require_storage().create_joint_prompt_enrichment_operation,
            items=payload["items"],
            mode=payload["mode"],
        )
        return {
            "task_group_id": operation["task_group_id"],
            "operation_id": operation["operation_id"],
            "status": operation["status"],
            "created_at": operation["created_at"],
        }

    @router.get(
        "/{task_group_id}",
        operation_id="getAnnotationTaskGroup",
        response_model=TaskGroup,
        dependencies=[Depends(authenticate)],
        responses=common_responses,
    )
    async def get_annotation_task_group(
        task_group_id: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().get_task_group,
            task_group_id,
        )

    return router
