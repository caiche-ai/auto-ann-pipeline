from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, Header
from fastapi.responses import FileResponse

from ..auth import AuthDependency
from ..errors import StorageUnavailableError
from ..schemas import CreateReleaseRequest, ErrorPayload, Release
from ..storage import AnnotationStore


PUBLIC_RELEASE_FIELDS = {
    "release_id",
    "name",
    "status",
    "counts",
    "manifest_url",
    "archive_url",
    "error",
    "created_at",
    "completed_at",
}
COMMON_RESPONSES = {
    401: {"model": ErrorPayload, "description": "Authentication failed."},
    404: {"model": ErrorPayload, "description": "Release was not found."},
    503: {"model": ErrorPayload, "description": "Storage is unavailable."},
}


def _model_json(model: Any) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(model.json())


def _public_release(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload[field] for field in PUBLIC_RELEASE_FIELDS}


def build_releases_router(
    *,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/releases",
        tags=["Releases"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    @router.post(
        "",
        operation_id="createAnnotationRelease",
        response_model=Release,
        status_code=202,
        dependencies=[Depends(authenticate)],
        responses={
            **COMMON_RESPONSES,
            409: {
                "model": ErrorPayload,
                "description": "Release name or idempotency conflict.",
            },
            422: {
                "model": ErrorPayload,
                "description": "Release request is invalid.",
            },
        },
    )
    async def create_annotation_release(
        request: CreateReleaseRequest,
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
        ),
    ) -> dict[str, Any]:
        payload = _model_json(request)
        release = await asyncio.to_thread(
            require_storage().create_release,
            name=payload["name"],
            task_filter=payload["task_filter"],
            split_policy=payload["split_policy"],
            idempotency_key=idempotency_key,
            idempotency_request=(
                payload if idempotency_key is not None else None
            ),
        )
        return _public_release(release)

    @router.get(
        "/{release_id}",
        operation_id="getAnnotationRelease",
        response_model=Release,
        dependencies=[Depends(authenticate)],
        responses=COMMON_RESPONSES,
    )
    async def get_annotation_release(
        release_id: str,
    ) -> dict[str, Any]:
        release = await asyncio.to_thread(
            require_storage().get_release,
            release_id,
        )
        return _public_release(release)

    @router.get(
        "/{release_id}/manifest",
        operation_id="getAnnotationReleaseManifest",
        response_class=FileResponse,
        dependencies=[Depends(authenticate)],
        responses={
            **COMMON_RESPONSES,
            409: {
                "model": ErrorPayload,
                "description": "Release is not ready.",
            },
        },
    )
    async def get_annotation_release_manifest(
        release_id: str,
    ) -> FileResponse:
        path, media_type = await asyncio.to_thread(
            require_storage().release_file,
            release_id,
            "manifest",
        )
        return FileResponse(
            path,
            media_type=media_type,
            filename="manifest.json",
        )

    @router.get(
        "/{release_id}/archive",
        operation_id="getAnnotationReleaseArchive",
        response_class=FileResponse,
        dependencies=[Depends(authenticate)],
        responses={
            **COMMON_RESPONSES,
            409: {
                "model": ErrorPayload,
                "description": "Release is not ready.",
            },
            200: {
                "description": "Immutable ReasonSeg release archive.",
                "content": {
                    "application/zip": {
                        "schema": {
                            "type": "string",
                            "format": "binary",
                        }
                    }
                },
            },
        },
    )
    async def get_annotation_release_archive(
        release_id: str,
    ) -> FileResponse:
        store = require_storage()
        release = await asyncio.to_thread(
            store.get_release,
            release_id,
        )
        path, media_type = await asyncio.to_thread(
            store.release_file,
            release_id,
            "archive",
        )
        return FileResponse(
            path,
            media_type=media_type,
            filename=f"{release['name']}.zip",
        )

    return router
