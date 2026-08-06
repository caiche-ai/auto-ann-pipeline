from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Callable, Mapping, cast

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import build_authenticate
from .config import Settings
from .errors import ServiceError
from .middleware import RequestBodyLimitMiddleware, RequestContextMiddleware
from .schemas import (
    ErrorCode,
    ErrorDetail,
    ErrorPayload,
    HealthResponse,
    ReadinessResponse,
)
from .routes.assets import build_assets_router
from .routes.jobs import build_jobs_router
from .routes.operations import build_operations_router
from .routes.releases import build_releases_router
from .routes.task_batches import build_task_batches_router
from .routes.task_groups import build_task_groups_router
from .routes.tasks import build_tasks_router
from .storage import AnnotationStore, StorageBackend


LOGGER = logging.getLogger(__name__)
ReadinessProvider = Callable[[], Mapping[str, str]]


def _model_dict(model) -> dict:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return model.dict()


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def _error_payload(
    request: Request,
    *,
    code: ErrorCode,
    message: str,
    details: list[ErrorDetail] | None = None,
) -> dict:
    return _model_dict(
        ErrorPayload(
            request_id=_request_id(request),
            code=code,
            message=message,
            details=details or [],
        )
    )


def create_app(
    settings: Settings | None = None,
    readiness_provider: ReadinessProvider | None = None,
    storage: StorageBackend | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    if storage is None and settings.storage_enabled:
        storage = AnnotationStore(settings.storage_root)
    authenticate = build_authenticate(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if storage is not None:
            await asyncio.to_thread(storage.initialize)
        try:
            yield
        finally:
            if storage is not None:
                await asyncio.to_thread(storage.close)

    app = FastAPI(
        title="Construction Safety Annotation API",
        version=settings.service_version,
        description=(
            "Free-form GroundingDINO detection, SAM box-prompt segmentation, "
            "Qwen prompt enrichment, manual annotation review, and dataset "
            "release API. The API process does not load model weights."
        ),
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )
    app.openapi_version = "3.0.3"
    app.state.settings = settings
    app.state.readiness_provider = readiness_provider
    app.state.storage = storage

    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=settings.max_request_bytes,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "PUT", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Idempotency-Key",
                "X-API-Key",
                "X-Request-ID",
            ],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(ServiceError)
    async def handle_service_error(
        request: Request,
        exc: ServiceError,
    ) -> JSONResponse:
        try:
            code = ErrorCode(exc.code)
        except ValueError:
            code = ErrorCode.INTERNAL_ERROR
        details = [
            ErrorDetail(
                field=item.get("field"),
                reason=str(item.get("reason", "request failed")),
            )
            for item in exc.details
        ]
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request,
                code=code,
                message=exc.message,
                details=details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = []
        for item in exc.errors():
            location = ".".join(str(part) for part in item.get("loc", []))
            details.append(
                ErrorDetail(
                    field=location or None,
                    reason=str(item.get("msg", "invalid value")),
                )
            )
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                request,
                code=ErrorCode.VALIDATION_ERROR,
                message="request payload is invalid",
                details=details,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        code_by_status = {
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            413: ErrorCode.REQUEST_TOO_LARGE,
            415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            422: ErrorCode.VALIDATION_ERROR,
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request,
                code=code_by_status.get(
                    exc.status_code,
                    ErrorCode.INTERNAL_ERROR,
                ),
                message=str(exc.detail),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        LOGGER.exception(
            "unexpected annotation API error",
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                request,
                code=ErrorCode.INTERNAL_ERROR,
                message="internal server error",
            ),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Service"],
        operation_id="getHealth",
    )
    async def health() -> HealthResponse:
        return HealthResponse(version=settings.service_version)

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        tags=["Service"],
        operation_id="getReadiness",
        dependencies=[Depends(authenticate)],
    )
    async def ready() -> JSONResponse:
        dependencies = {"api": "ready"}
        if readiness_provider is not None:
            dependencies.update(readiness_provider())
        if storage is not None:
            dependencies.update(storage.readiness())
        is_ready = bool(dependencies) and all(
            status == "ready" for status in dependencies.values()
        )
        payload = ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            dependencies=dependencies,
        )
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content=_model_dict(payload),
        )

    app.include_router(
        build_assets_router(
            settings=settings,
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_jobs_router(
            settings=settings,
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_tasks_router(
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_task_batches_router(
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_task_groups_router(
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_operations_router(
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    app.include_router(
        build_releases_router(
            storage=cast(AnnotationStore | None, storage),
            authenticate=authenticate,
        )
    )
    return app


app = create_app()
