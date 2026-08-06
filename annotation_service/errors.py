from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    code = "internal_error"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        details: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or []


class UnauthorizedError(ServiceError):
    code = "unauthorized"
    status_code = 401


class ForbiddenError(ServiceError):
    code = "forbidden"
    status_code = 403


class ResourceNotFoundError(ServiceError):
    code = "not_found"
    status_code = 404


class VersionConflictError(ServiceError):
    code = "version_conflict"
    status_code = 409


class IdempotencyConflictError(ServiceError):
    code = "idempotency_conflict"
    status_code = 409


class InvalidStateTransitionError(ServiceError):
    code = "invalid_state_transition"
    status_code = 409


class UnsupportedMediaTypeError(ServiceError):
    code = "unsupported_media_type"
    status_code = 415


class RequestTooLargeError(ServiceError):
    code = "request_too_large"
    status_code = 413


class ValidationServiceError(ServiceError):
    code = "validation_error"
    status_code = 422


class AnnotationValidationError(ServiceError):
    code = "annotation_validation_failed"
    status_code = 422


class QueueFullError(ServiceError):
    code = "queue_full"
    status_code = 429


class ModelUnavailableError(ServiceError):
    code = "model_unavailable"
    status_code = 503


class DownstreamUnavailableError(ServiceError):
    code = "downstream_unavailable"
    status_code = 503


class StorageError(ServiceError):
    code = "internal_error"
    status_code = 500


class StorageUnavailableError(ServiceError):
    code = "downstream_unavailable"
    status_code = 503
