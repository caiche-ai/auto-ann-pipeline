from __future__ import annotations

import secrets
from typing import Awaitable, Callable

from fastapi import Security
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from .config import Settings
from .errors import UnauthorizedError


AuthDependency = Callable[..., Awaitable[None]]
API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    scheme_name="apiKeyAuth",
    auto_error=False,
)
BEARER_AUTH = HTTPBearer(
    scheme_name="bearerAuth",
    auto_error=False,
)


def build_authenticate(settings: Settings) -> AuthDependency:
    async def authenticate(
        x_api_key: str | None = Security(API_KEY_HEADER),
        bearer: HTTPAuthorizationCredentials | None = Security(BEARER_AUTH),
    ) -> None:
        if settings.api_key is None:
            return

        candidate = x_api_key
        if candidate is None and bearer is not None:
            candidate = bearer.credentials

        if candidate is None or not secrets.compare_digest(
            candidate,
            settings.api_key,
        ):
            raise UnauthorizedError("invalid API credential")

    return authenticate
