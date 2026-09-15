from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, Query, Response, UploadFile

from ...storage.image_io import validate_image_bytes
from ...storage.repository import AnnotationStore, sha256_bytes
from ...storage.validation import parse_metadata_json
from ..auth import AuthDependency
from ..config import Settings
from ..errors import StorageUnavailableError, ValidationServiceError
from ..schemas import (
    CreateWorkspaceTaskRequest,
    ErrorPayload,
    UpdateWorkspaceTaskItemRequest,
    WorkspaceTask,
    WorkspaceTaskItem,
    WorkspaceTaskList,
)


ERROR_RESPONSES = {
    401: {"model": ErrorPayload, "description": "Authentication failed."},
    404: {
        "model": ErrorPayload,
        "description": "Workspace task or item was not found.",
    },
    409: {"model": ErrorPayload, "description": "Idempotency conflict."},
    413: {"model": ErrorPayload, "description": "Image is too large."},
    415: {"model": ErrorPayload, "description": "Unsupported image type."},
    422: {"model": ErrorPayload, "description": "Request is invalid."},
    503: {"model": ErrorPayload, "description": "Storage is unavailable."},
}


def build_workspace_tasks_router(
    *,
    settings: Settings,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/workspace-tasks",
        tags=["Workspace Tasks"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError("annotation storage is disabled")
        return storage

    @router.post(
        "",
        operation_id="createWorkspaceTask",
        response_model=WorkspaceTask,
        status_code=201,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def create_workspace_task(
        request: CreateWorkspaceTaskRequest,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().create_workspace_task,
            name=request.name,
            description=request.description,
        )

    @router.get(
        "",
        operation_id="listWorkspaceTasks",
        response_model=WorkspaceTaskList,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def list_workspace_tasks(
        limit: int = Query(default=100, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().list_workspace_tasks,
            limit=limit,
            offset=offset,
        )

    @router.get(
        "/{workspace_task_id}",
        operation_id="getWorkspaceTask",
        response_model=WorkspaceTask,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def get_workspace_task(workspace_task_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().get_workspace_task,
            workspace_task_id,
        )

    @router.get(
        "/{workspace_task_id}/assets",
        operation_id="listWorkspaceTaskAssets",
        response_model=list[WorkspaceTaskItem],
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def list_workspace_task_assets(
        workspace_task_id: str,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            require_storage().list_workspace_items,
            workspace_task_id,
        )

    @router.post(
        "/{workspace_task_id}/assets",
        operation_id="uploadWorkspaceTaskAsset",
        response_model=WorkspaceTaskItem,
        status_code=201,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def upload_workspace_task_asset(
        workspace_task_id: str,
        file: UploadFile = File(...),
        source_id: str | None = Form(default=None, max_length=256),
        metadata_json: str | None = Form(
            default=None,
            max_length=settings.max_metadata_chars,
        ),
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
        ),
    ) -> dict[str, Any]:
        store = require_storage()
        await asyncio.to_thread(store.get_workspace_task, workspace_task_id)
        original_filename = (file.filename or "").replace("\\", "/")
        original_filename = original_filename.rsplit("/", 1)[-1].strip()
        try:
            raw = await file.read(settings.max_image_bytes + 1)
        finally:
            await file.close()
        image = await asyncio.to_thread(
            validate_image_bytes,
            raw,
            max_image_bytes=settings.max_image_bytes,
            max_image_pixels=settings.max_image_pixels,
        )
        try:
            metadata = parse_metadata_json(
                metadata_json,
                max_chars=settings.max_metadata_chars,
            )
            if original_filename:
                metadata = {**metadata, "original_filename": original_filename}
        except ValueError as exc:
            raise ValidationServiceError(
                "asset metadata is invalid",
                details=[{"field": "metadata_json", "reason": str(exc)}],
            ) from exc
        normalized_source_id = source_id.strip() if source_id is not None else None
        normalized_source_id = normalized_source_id or None
        request_payload = {
            "image_sha256": sha256_bytes(image.raw),
            "source_id": normalized_source_id,
            "group_id": workspace_task_id,
            "metadata": metadata,
        }
        asset = await asyncio.to_thread(
            store.create_asset,
            image_bytes=image.raw,
            media_type=image.media_type,
            width=image.width,
            height=image.height,
            group_id=workspace_task_id,
            source_id=normalized_source_id,
            metadata=metadata,
            idempotency_key=idempotency_key,
            idempotency_request=(request_payload if idempotency_key else None),
        )
        return await asyncio.to_thread(
            store.attach_workspace_asset,
            workspace_task_id,
            asset["asset_id"],
        )

    @router.patch(
        "/{workspace_task_id}/assets/{asset_id}",
        operation_id="updateWorkspaceTaskAsset",
        response_model=WorkspaceTaskItem,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def update_workspace_task_asset(
        workspace_task_id: str,
        asset_id: str,
        request: UpdateWorkspaceTaskItemRequest,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            require_storage().update_workspace_item,
            workspace_task_id,
            asset_id,
            request.dict(exclude_unset=True),
        )

    @router.delete(
        "/{workspace_task_id}/assets/{asset_id}",
        operation_id="removeWorkspaceTaskAsset",
        status_code=204,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def remove_workspace_task_asset(
        workspace_task_id: str,
        asset_id: str,
    ) -> Response:
        await asyncio.to_thread(
            require_storage().remove_workspace_item,
            workspace_task_id,
            asset_id,
        )
        return Response(status_code=204)

    return router
