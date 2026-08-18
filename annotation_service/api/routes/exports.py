from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from ...review.submission_export import export_submitted_samples
from ...storage.repository import AnnotationStore
from ..auth import AuthDependency
from ..errors import StorageUnavailableError, ValidationServiceError
from ..schemas import ErrorPayload


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def build_exports_router(
    *,
    storage: AnnotationStore | None,
    authenticate: AuthDependency,
) -> APIRouter:
    router = APIRouter(
        prefix="/v1/annotation/export",
        tags=["Exports"],
    )

    def require_storage() -> AnnotationStore:
        if storage is None:
            raise StorageUnavailableError("annotation storage is disabled")
        return storage

    @router.get(
        "",
        operation_id="exportSubmittedAnnotations",
        response_class=FileResponse,
        dependencies=[Depends(authenticate)],
        responses={
            200: {
                "description": "ZIP containing submitted annotation samples.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            401: {"model": ErrorPayload, "description": "Authentication failed."},
            422: {
                "model": ErrorPayload,
                "description": "The time range is invalid or no samples matched.",
            },
            503: {"model": ErrorPayload, "description": "Storage is unavailable."},
        },
    )
    async def export_submitted_annotations(
        start_time: datetime | None = Query(
            default=None,
            description="Inclusive submission time lower bound (ISO 8601).",
        ),
        end_time: datetime | None = Query(
            default=None,
            description="Inclusive submission time upper bound (ISO 8601).",
        ),
        task_id: list[str] | None = Query(
            default=None,
            description="Optional task IDs; repeat the parameter to select several.",
        ),
    ) -> FileResponse:
        start = _utc(start_time)
        end = _utc(end_time)
        if start is not None and end is not None and start > end:
            raise ValidationServiceError(
                "start_time must be earlier than or equal to end_time"
            )
        try:
            path, count = await asyncio.to_thread(
                export_submitted_samples,
                require_storage(),
                start_time=start,
                end_time=end,
                task_ids=task_id,
            )
        except ValueError as exc:
            raise ValidationServiceError(str(exc)) from exc
        response = FileResponse(
            path,
            media_type="application/zip",
            filename=path.name,
        )
        response.headers["X-Annotation-Sample-Count"] = str(count)
        return response

    return router
