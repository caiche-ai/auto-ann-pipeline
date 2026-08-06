from __future__ import annotations

import asyncio
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    UploadFile,
)
from fastapi.responses import FileResponse

from ..auth import AuthDependency
from ..config import Settings
from ..errors import (
    StorageUnavailableError,
    ValidationServiceError,
)
from ..image_io import validate_image_bytes
from ..schemas import Asset, ErrorPayload
from ..storage import AnnotationStore, sha256_bytes
from ..validation import parse_metadata_json, validate_group_id


ERROR_RESPONSES = {
    401: {"model": ErrorPayload, "description": "Authentication failed."},
    404: {"model": ErrorPayload, "description": "Asset was not found."},
    409: {"model": ErrorPayload, "description": "Idempotency conflict."},
    413: {"model": ErrorPayload, "description": "Image is too large."},
    415: {"model": ErrorPayload, "description": "Unsupported image type."},
    422: {"model": ErrorPayload, "description": "Upload is invalid."},
    503: {"model": ErrorPayload, "description": "Storage is unavailable."},
}
CONTENT_RESPONSE = {
    200: {
        "description": "Original JPEG or PNG bytes.",
        "content": {
            "image/jpeg": {
                "schema": {"type": "string", "format": "binary"}
            },
            "image/png": {
                "schema": {"type": "string", "format": "binary"}
            },
        },
    }
}


def build_assets_router(
    *,
    settings: Settings,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/assets",
        tags=["Assets"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError(
                "annotation storage is disabled"
            )
        return storage

    @router.post(
        "",
        operation_id="createAsset",
        response_model=Asset,
        status_code=201,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def create_asset(
        file: UploadFile = File(...),
        group_id: str = Form(..., min_length=1, max_length=256),
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
            normalized_group_id = validate_group_id(group_id)
            metadata = parse_metadata_json(
                metadata_json,
                max_chars=settings.max_metadata_chars,
            )
        except ValueError as exc:
            field = (
                "metadata_json"
                if "metadata_json" in str(exc)
                else "group_id"
            )
            raise ValidationServiceError(
                "asset metadata is invalid",
                details=[{"field": field, "reason": str(exc)}],
            ) from exc

        normalized_source_id = (
            source_id.strip() if source_id is not None else None
        )
        normalized_source_id = normalized_source_id or None
        request_payload = {
            "image_sha256": sha256_bytes(image.raw),
            "source_id": normalized_source_id,
            "group_id": normalized_group_id,
            "metadata": metadata,
        }
        return await asyncio.to_thread(
            store.create_asset,
            image_bytes=image.raw,
            media_type=image.media_type,
            width=image.width,
            height=image.height,
            group_id=normalized_group_id,
            source_id=normalized_source_id,
            metadata=metadata,
            idempotency_key=idempotency_key,
            idempotency_request=(
                request_payload if idempotency_key is not None else None
            ),
        )

    @router.get(
        "/{asset_id}",
        operation_id="getAsset",
        response_model=Asset,
        dependencies=[Depends(authenticate)],
        responses=ERROR_RESPONSES,
    )
    async def get_asset(asset_id: str) -> dict[str, Any]:
        store = require_storage()
        return await asyncio.to_thread(store.get_asset, asset_id)

    @router.get(
        "/{asset_id}/content",
        operation_id="getAssetContent",
        response_class=FileResponse,
        dependencies=[Depends(authenticate)],
        responses={**ERROR_RESPONSES, **CONTENT_RESPONSE},
    )
    async def get_asset_content(asset_id: str) -> FileResponse:
        store = require_storage()
        asset, file_result = await asyncio.gather(
            asyncio.to_thread(store.get_asset, asset_id),
            asyncio.to_thread(store.asset_file, asset_id),
        )
        path, media_type = file_result
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "ETag": f'"{asset["sha256"]}"',
                "Cache-Control": "private, no-cache",
            },
        )

    return router
