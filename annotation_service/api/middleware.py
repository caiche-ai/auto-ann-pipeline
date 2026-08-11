from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable


AsgiMessage = dict[str, Any]
AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
AsgiSend = Callable[[AsgiMessage], Awaitable[None]]
AsgiApp = Callable[[dict[str, Any], AsgiReceive, AsgiSend], Awaitable[None]]


class _RequestBodyTooLarge(Exception):
    pass


def _header_value(scope: dict[str, Any], target: bytes) -> str | None:
    for name, value in scope.get("headers", []):
        if name.lower() != target:
            continue
        decoded = value.decode("latin-1").strip()
        return decoded or None
    return None


def _request_id(scope: dict[str, Any]) -> str | None:
    state = scope.get("state")
    if isinstance(state, dict):
        value = state.get("request_id")
        if isinstance(value, str) and value:
            return value[:128]
    value = _header_value(scope, b"x-request-id")
    return value[:128] if value else None


def _content_lengths(scope: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for name, raw in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            value = int(raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            continue
        if value >= 0:
            values.append(value)
    return values


class RequestContextMiddleware:
    def __init__(self, app: AsgiApp):
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = _header_value(scope, b"x-request-id")
        request_id = (request_id or str(uuid.uuid4()))[:128]
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        async def send_with_request_id(message: AsgiMessage) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (b"x-request-id", request_id.encode("latin-1", "replace"))
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class RequestBodyLimitMiddleware:
    def __init__(self, app: AsgiApp, *, max_bytes: int):
        if max_bytes < 1:
            raise ValueError("request body limit must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(
        self,
        scope: dict[str, Any],
        send: AsgiSend,
    ) -> None:
        body = json.dumps(
            {
                "request_id": _request_id(scope),
                "code": "request_too_large",
                "message": "request body exceeds configured limit",
                "details": [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        if any(
            length > self.max_bytes
            for length in _content_lengths(scope)
        ):
            await self._reject(scope, send)
            return

        received_bytes = 0
        body_too_large = False
        response_started = False

        async def limited_receive() -> AsgiMessage:
            nonlocal body_too_large, received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    body_too_large = True
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: AsgiMessage) -> None:
            nonlocal response_started
            if body_too_large:
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, send)
            return

        if body_too_large:
            if response_started:
                raise RuntimeError(
                    "request body exceeded the limit after response start"
                )
            await self._reject(scope, send)
