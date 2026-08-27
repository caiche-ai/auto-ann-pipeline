from __future__ import annotations

import base64
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class RemoteProviderError(RuntimeError):
    """Raised when a remote inference provider cannot be used safely."""


RemoteTransport = Callable[
    [str, dict[str, str], bytes, float],
    dict[str, Any],
]


@dataclass(frozen=True)
class RemoteProviderConfig:
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_image_bytes: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("remote provider base_url must be an absolute HTTP(S) URL")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ValueError(
                "remote provider timeout_seconds must be between 1 and 3600"
            )
        if not 1 <= self.max_image_bytes <= 100 * 1024 * 1024:
            raise ValueError(
                "remote provider max_image_bytes must be between 1 and 100 MiB"
            )

    def endpoint(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


def default_remote_transport(
    url: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(256 * 1024 * 1024 + 1)
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RemoteProviderError(
            f"remote inference endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        raise RemoteProviderError("remote inference endpoint is unavailable") from exc
    if len(raw) > 256 * 1024 * 1024:
        raise RemoteProviderError("remote inference response is too large")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RemoteProviderError(
            "remote inference endpoint returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise RemoteProviderError("remote inference response must be a JSON object")
    return payload


def image_payload(path: Path, *, max_bytes: int) -> tuple[str, str]:
    raw = path.read_bytes()
    if not raw:
        raise RemoteProviderError(f"image is empty: {path.name}")
    if len(raw) > max_bytes:
        raise RemoteProviderError(
            f"image exceeds remote provider limit of {max_bytes} bytes"
        )
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    else:
        raise RemoteProviderError("remote provider supports only JPEG and PNG")
    return media_type, base64.b64encode(raw).decode("ascii")


def post_json(
    *,
    config: RemoteProviderConfig,
    path: str,
    payload: dict[str, Any],
    transport: RemoteTransport,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["X-Inference-API-Key"] = config.api_key
    return transport(
        config.endpoint(path),
        headers,
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        config.timeout_seconds,
    )


def decode_base64_field(payload: dict[str, Any], field: str) -> bytes:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RemoteProviderError(f"remote response field {field} is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise RemoteProviderError(
            f"remote response field {field} is not valid base64"
        ) from exc
    if not decoded:
        raise RemoteProviderError(f"remote response field {field} is empty")
    return decoded
