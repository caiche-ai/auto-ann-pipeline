from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

from .errors import (
    IdempotencyConflictError,
    InvalidStateTransitionError,
    QueueFullError,
    ResourceNotFoundError,
    StorageError,
    StorageUnavailableError,
    ValidationServiceError,
    VersionConflictError,
)
from .image_io import validate_image_bytes
from .schemas import (
    AnnotationCategory,
    AnnotationContent,
    BadCaseType,
    CreateReleaseRequest,
    JobOptions,
    JobProgress,
    JobStatus,
    OperationStatus,
    PipelineStage,
    Provenance,
    ReleaseStatus,
    ReviewDecision,
    StageResult,
    TaskStatus,
)
from .state_machine import (
    ensure_job_transition,
    ensure_release_transition,
    ensure_task_transition,
)
from .storage_schema import (
    SCHEMA_V1,
    SCHEMA_V2,
    SCHEMA_V3,
    SCHEMA_V4,
    SCHEMA_V5,
    SCHEMA_V6,
    SCHEMA_V7,
    SCHEMA_V8,
    SCHEMA_VERSION,
)
from .validation import validate_annotation_for_submission


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _enum_value(value: str | Enum) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _model_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    dictionary = getattr(value, "dict", None)
    if callable(dictionary):
        raw = dictionary()
        return _to_json_value(raw)
    raise TypeError(f"expected dict or Pydantic model, got {type(value)!r}")


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(_enum_value(key)): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class StorageBackend(Protocol):
    def initialize(self) -> None:
        ...

    def close(self) -> None:
        ...

    def readiness(self) -> dict[str, str]:
        ...


class AnnotationStore:
    """SQLite metadata and filesystem-backed annotation artifacts.

    Database rows store only relative file paths. Public payloads expose API
    URLs, never host filesystem paths. Each method opens its own SQLite
    connection so an API process and a later worker process can safely share
    the same WAL database.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if self.root == Path(self.root.anchor):
            raise ValueError("annotation storage root must not be a filesystem root")
        self.db_path = self.root / "annotation.db"
        self.images_root = self.root / "images"
        self.masks_root = self.root / "masks"
        self.overlays_root = self.root / "overlays"
        self.crops_root = self.root / "crops"
        self.exports_root = self.root / "exports"
        self.tmp_root = self.root / "tmp"
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                for path in (
                    self.root,
                    self.images_root,
                    self.masks_root,
                    self.overlays_root,
                    self.crops_root,
                    self.exports_root,
                    self.tmp_root,
                ):
                    path.mkdir(parents=True, exist_ok=True)

                with self._connect() as connection:
                    current = (
                        connection.execute(
                            "SELECT COALESCE(MAX(version), 0) "
                            "FROM schema_migrations"
                        ).fetchone()[0]
                        if self._table_exists(
                            connection,
                            "schema_migrations",
                        )
                        else 0
                    )
                    if current > SCHEMA_VERSION:
                        raise RuntimeError(
                            f"database schema {current} is newer than supported "
                            f"version {SCHEMA_VERSION}"
                        )
                    if current < 1:
                        connection.executescript(SCHEMA_V1)
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO schema_migrations(
                                version, applied_at
                            ) VALUES (?, ?)
                            """,
                            (1, utc_now()),
                        )
                        current = 1
                    if current < 2:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V2:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (2, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                        current = 2
                    if current < 3:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V3:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (3, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                        current = 3
                    if current < 4:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V4:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (4, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                        current = 4
                    if current < 5:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V5:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (5, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                        current = 5
                    if current < 6:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V6:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (6, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                        current = 6
                    if current < 7:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V7:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (7, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                        current = 7
                    if current < 8:
                        connection.execute("BEGIN IMMEDIATE")
                        try:
                            for statement in SCHEMA_V8:
                                connection.execute(statement)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations(
                                    version, applied_at
                                ) VALUES (?, ?)
                                """,
                                (8, utc_now()),
                            )
                            connection.execute("COMMIT")
                        except Exception:
                            connection.execute("ROLLBACK")
                            raise
                self._initialized = True
            except Exception as exc:
                raise StorageUnavailableError(
                    "annotation storage is unavailable"
                ) from exc

    def close(self) -> None:
        with self._lock:
            self._initialized = False

    def readiness(self) -> dict[str, str]:
        if not self._initialized:
            return {"storage": "not_ready"}
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return {"storage": "ready"}
        except Exception:
            return {"storage": "not_ready"}

    def schema_version(self) -> int:
        self._ensure_initialized()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
        return int(row[0])

    def _table_exists(self, connection: sqlite3.Connection, name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _resolve_relative(self, relative: str | Path) -> Path:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise StorageError("stored path must be relative")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise StorageError("stored path escapes annotation storage root") from exc
        return candidate

    def _atomic_write(self, relative: Path, data: bytes) -> Path:
        if not data:
            raise StorageError("refusing to persist an empty file")
        destination = self._resolve_relative(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.tmp_root / f"{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception as exc:
            raise StorageError("failed to persist annotation file") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _atomic_copy(self, source: Path, relative: Path) -> Path:
        if not source.is_file() or source.stat().st_size == 0:
            raise StorageError("release staging file is missing or empty")
        destination = self._resolve_relative(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.tmp_root / f"{uuid.uuid4().hex}.tmp"
        try:
            with source.open("rb") as input_handle, temporary.open(
                "wb"
            ) as output_handle:
                shutil.copyfileobj(
                    input_handle,
                    output_handle,
                    length=1024 * 1024,
                )
                output_handle.flush()
                os.fsync(output_handle.fileno())
            os.replace(temporary, destination)
        except Exception as exc:
            raise StorageError("failed to publish release file") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _row_or_not_found(
        self,
        connection: sqlite3.Connection,
        query: str,
        parameters: tuple[Any, ...],
        resource: str,
    ) -> sqlite3.Row:
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise ResourceNotFoundError(f"{resource} was not found")
        return row

    # ------------------------------------------------------------------ assets
    def create_asset(
        self,
        *,
        image_bytes: bytes,
        media_type: str,
        width: int,
        height: int,
        group_id: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotency_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if media_type not in {"image/jpeg", "image/png"}:
            raise ValueError("media_type must be image/jpeg or image/png")
        if width < 1 or height < 1:
            raise ValueError("image dimensions must be positive")
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")
        if not group_id.strip():
            raise ValueError("group_id must not be blank")

        digest = sha256_bytes(image_bytes)
        idempotency_digest: str | None = None
        if idempotency_key is not None:
            if idempotency_request is None:
                raise ValueError(
                    "idempotency_request is required with idempotency_key"
                )
            existing = self.find_idempotency(
                scope="create-asset",
                key=idempotency_key,
                request_payload=idempotency_request,
            )
            if existing is not None:
                return existing["response"]
            idempotency_digest = sha256_bytes(
                canonical_json(idempotency_request).encode("utf-8")
            )
        extension = "jpg" if media_type == "image/jpeg" else "png"
        relative = Path("images") / digest[:2] / f"{digest}.{extension}"
        self._atomic_write(relative, image_bytes)
        asset_id = _new_id("ast")
        created_at = utc_now()
        duplicate_of: str | None = None

        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if idempotency_key is not None:
                    connection.execute(
                        """
                        DELETE FROM idempotency_keys
                        WHERE scope = 'create-asset' AND idempotency_key = ?
                          AND expires_at IS NOT NULL AND expires_at <= ?
                        """,
                        (idempotency_key, created_at),
                    )
                    idempotent = connection.execute(
                        """
                        SELECT * FROM idempotency_keys
                        WHERE scope = 'create-asset' AND idempotency_key = ?
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if idempotent is not None:
                        if idempotent["request_sha256"] != idempotency_digest:
                            raise IdempotencyConflictError(
                                "idempotency key was reused with a different request"
                            )
                        response = _json_loads(
                            idempotent["response_json"], {}
                        )
                        connection.execute("COMMIT")
                        return response
                canonical = connection.execute(
                    """
                    SELECT asset_id, image_path FROM assets
                    WHERE sha256 = ? AND duplicate_of IS NULL
                    """,
                    (digest,),
                ).fetchone()
                if canonical is not None:
                    duplicate_of = canonical["asset_id"]
                    relative = Path(canonical["image_path"])
                connection.execute(
                    """
                    INSERT INTO assets (
                        asset_id, source_id, group_id, width, height, sha256,
                        media_type, image_path, size_bytes, metadata_json,
                        duplicate_of, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        source_id,
                        group_id,
                        width,
                        height,
                        digest,
                        media_type,
                        relative.as_posix(),
                        len(image_bytes),
                        canonical_json(metadata or {}),
                        duplicate_of,
                        created_at,
                    ),
                )
                if idempotency_key is not None:
                    response = {
                        "asset_id": asset_id,
                        "source_id": source_id,
                        "group_id": group_id,
                        "width": width,
                        "height": height,
                        "sha256": digest,
                        "media_type": media_type,
                        "content_url": (
                            f"/v1/annotation/assets/{asset_id}/content"
                        ),
                        "duplicate_of": duplicate_of,
                        "metadata": metadata or {},
                        "created_at": created_at,
                    }
                    connection.execute(
                        """
                        INSERT INTO idempotency_keys (
                            scope, idempotency_key, request_sha256,
                            resource_type, resource_id, response_json,
                            created_at
                        ) VALUES ('create-asset', ?, ?, 'asset', ?, ?, ?)
                        """,
                        (
                            idempotency_key,
                            idempotency_digest,
                            asset_id,
                            canonical_json(response),
                            created_at,
                        ),
                    )
                connection.execute("COMMIT")
        except (IdempotencyConflictError, StorageError):
            raise
        except Exception as exc:
            raise StorageError("failed to create annotation asset") from exc
        return self.get_asset(asset_id)

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM assets WHERE asset_id = ?",
                (asset_id,),
                "asset",
            )
        return self._asset_payload(row)

    def _asset_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        asset_id = row["asset_id"]
        return {
            "asset_id": asset_id,
            "source_id": row["source_id"],
            "group_id": row["group_id"],
            "width": row["width"],
            "height": row["height"],
            "sha256": row["sha256"],
            "media_type": row["media_type"],
            "content_url": f"/v1/annotation/assets/{asset_id}/content",
            "duplicate_of": row["duplicate_of"],
            "metadata": _json_loads(row["metadata_json"], {}),
            "created_at": row["created_at"],
        }

    def asset_file(self, asset_id: str) -> tuple[Path, str]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                "SELECT image_path, media_type FROM assets WHERE asset_id = ?",
                (asset_id,),
                "asset",
            )
        path = self._resolve_relative(row["image_path"])
        if not path.is_file():
            raise StorageError("asset image file is missing")
        return path, row["media_type"]

    # -------------------------------------------------------------------- jobs
    def create_job(
        self,
        *,
        asset_ids: list[str],
        grounding_prompt: str | None = None,
        requested_categories: (
            list[str | AnnotationCategory] | None
        ) = None,
        pipeline_version: str,
        options: dict[str, Any] | JobOptions | None = None,
        max_queued_jobs: int = 100,
        idempotency_key: str | None = None,
        idempotency_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if not asset_ids or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset_ids must be non-empty and unique")
        if max_queued_jobs < 1:
            raise ValueError("max_queued_jobs must be positive")
        categories = [
            _enum_value(value)
            for value in (requested_categories or [])
        ]
        if len(categories) != len(set(categories)):
            raise ValueError("requested_categories must be unique")
        for category in categories:
            AnnotationCategory(category)
        normalized_prompt = (
            grounding_prompt.strip()
            if grounding_prompt is not None
            else ""
        )
        if grounding_prompt is not None and not normalized_prompt:
            raise ValueError("grounding_prompt must not be blank")
        if len(normalized_prompt) > 2000:
            raise ValueError(
                "grounding_prompt must contain at most 2000 characters"
            )
        if not normalized_prompt and not categories:
            raise ValueError(
                "grounding_prompt is required for a free detection job"
            )
        normalized_pipeline_version = pipeline_version.strip()
        if not normalized_pipeline_version:
            raise ValueError("pipeline_version must not be blank")
        option_model = (
            options
            if isinstance(options, JobOptions)
            else JobOptions(**(options or {}))
        )
        option_payload = _model_dict(option_model)
        idempotency_digest: str | None = None
        if idempotency_key is not None:
            if idempotency_request is None:
                raise ValueError(
                    "idempotency_request is required with idempotency_key"
                )
            existing = self.find_idempotency(
                scope="create-job",
                key=idempotency_key,
                request_payload=idempotency_request,
            )
            if existing is not None:
                return existing["response"]
            idempotency_digest = sha256_bytes(
                canonical_json(idempotency_request).encode("utf-8")
            )
        progress = {
            "total_assets": len(asset_ids),
            "completed_assets": 0,
            "generated_tasks": 0,
        }
        stages = (
            {
                PipelineStage.GROUNDING_DINO.value: {
                    "status": "pending"
                }
            }
            if normalized_prompt
            else {
                stage.value: {"status": "pending"}
                for stage in PipelineStage
            }
        )
        job_id = _new_id("job")
        created_at = utc_now()

        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if idempotency_key is not None:
                    connection.execute(
                        """
                        DELETE FROM idempotency_keys
                        WHERE scope = 'create-job' AND idempotency_key = ?
                          AND expires_at IS NOT NULL AND expires_at <= ?
                        """,
                        (idempotency_key, created_at),
                    )
                    idempotent = connection.execute(
                        """
                        SELECT * FROM idempotency_keys
                        WHERE scope = 'create-job' AND idempotency_key = ?
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if idempotent is not None:
                        if idempotent["request_sha256"] != idempotency_digest:
                            raise IdempotencyConflictError(
                                "idempotency key was reused with a "
                                "different request"
                            )
                        response = _json_loads(
                            idempotent["response_json"],
                            {},
                        )
                        connection.execute("COMMIT")
                        return response
                queued_jobs = connection.execute(
                    """
                    SELECT COUNT(*) FROM annotation_jobs
                    WHERE status = 'queued'
                    """
                ).fetchone()[0]
                if queued_jobs >= max_queued_jobs:
                    raise QueueFullError(
                        "annotation job queue is full"
                    )
                placeholders = ",".join("?" for _ in asset_ids)
                found = {
                    row[0]
                    for row in connection.execute(
                        "SELECT asset_id FROM assets "
                        f"WHERE asset_id IN ({placeholders})",
                        asset_ids,
                    ).fetchall()
                }
                missing = [asset_id for asset_id in asset_ids if asset_id not in found]
                if missing:
                    raise ResourceNotFoundError(
                        f"assets were not found: {', '.join(missing)}"
                    )
                connection.execute(
                    """
                    INSERT INTO annotation_jobs (
                        job_id, status, pipeline_version,
                        grounding_prompt, requested_categories_json,
                        options_json,
                        progress_json, stages_json, errors_json, created_at
                    ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, '[]', ?)
                    """,
                    (
                        job_id,
                        normalized_pipeline_version,
                        normalized_prompt,
                        canonical_json(categories),
                        canonical_json(option_payload),
                        canonical_json(progress),
                        canonical_json(stages),
                        created_at,
                    ),
                )
                for ordinal, asset_id in enumerate(asset_ids):
                    connection.execute(
                        """
                        INSERT INTO job_assets(
                            job_id, asset_id, ordinal
                        ) VALUES (?, ?, ?)
                        """,
                        (job_id, asset_id, ordinal),
                    )
                if idempotency_key is not None:
                    response = {
                        "job_id": job_id,
                        "status": JobStatus.QUEUED.value,
                        "stage": None,
                        "pipeline_version": normalized_pipeline_version,
                        "grounding_prompt": normalized_prompt,
                        "requested_categories": categories,
                        "options": option_payload,
                        "progress": progress,
                        "stages": stages,
                        "task_ids": [],
                        "asset_ids": asset_ids,
                        "errors": [],
                        "created_at": created_at,
                        "started_at": None,
                        "completed_at": None,
                        "claimed_by": None,
                        "lease_expires_at": None,
                        "heartbeat_at": None,
                        "attempt_count": 0,
                    }
                    connection.execute(
                        """
                        INSERT INTO idempotency_keys (
                            scope, idempotency_key, request_sha256,
                            resource_type, resource_id, response_json,
                            created_at
                        ) VALUES (
                            'create-job', ?, ?, 'job', ?, ?, ?
                        )
                        """,
                        (
                            idempotency_key,
                            idempotency_digest,
                            job_id,
                            canonical_json(response),
                            created_at,
                        ),
                    )
                connection.execute("COMMIT")
        except (
            IdempotencyConflictError,
            QueueFullError,
            ResourceNotFoundError,
            ValueError,
        ):
            raise
        except Exception as exc:
            raise StorageError("failed to create annotation job") from exc
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM annotation_jobs WHERE job_id = ?",
                (job_id,),
                "annotation job",
            )
            asset_ids = [
                item["asset_id"]
                for item in connection.execute(
                    "SELECT asset_id FROM job_assets WHERE job_id = ? ORDER BY ordinal",
                    (job_id,),
                ).fetchall()
            ]
            task_ids = [
                item["task_id"]
                for item in connection.execute(
                    "SELECT task_id FROM annotation_tasks WHERE job_id = ? ORDER BY created_at",
                    (job_id,),
                ).fetchall()
            ]
        return self._job_payload(row, asset_ids, task_ids)

    def _job_payload(
        self,
        row: sqlite3.Row,
        asset_ids: list[str],
        task_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "stage": row["stage"],
            "pipeline_version": row["pipeline_version"],
            "grounding_prompt": row["grounding_prompt"],
            "requested_categories": _json_loads(
                row["requested_categories_json"], []
            ),
            "options": _json_loads(row["options_json"], {}),
            "progress": _json_loads(row["progress_json"], {}),
            "stages": _json_loads(row["stages_json"], {}),
            "task_ids": task_ids,
            "asset_ids": asset_ids,
            "errors": _json_loads(row["errors_json"], []),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "claimed_by": row["claimed_by"],
            "lease_expires_at": row["lease_expires_at"],
            "heartbeat_at": row["heartbeat_at"],
            "attempt_count": row["attempt_count"],
        }

    def update_job(
        self,
        job_id: str,
        *,
        expected_status: str | JobStatus,
        status: str | JobStatus | None = None,
        stage: str | PipelineStage | None = None,
        progress: dict[str, Any] | JobProgress | None = None,
        stages: dict[str, Any] | None = None,
        errors: list[dict[str, Any]] | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        expected = JobStatus(_enum_value(expected_status))
        target = expected if status is None else JobStatus(_enum_value(status))
        if target != expected:
            ensure_job_transition(expected, target)
        stage_value = _enum_value(stage) if stage is not None else None
        if stage_value is not None:
            PipelineStage(stage_value)
        progress_payload = _model_dict(progress) if progress is not None else None
        if progress_payload is not None:
            JobProgress(**progress_payload)
        stages_payload: dict[str, Any] | None = None
        if stages is not None:
            stages_payload = {}
            for stage_name, stage_result in stages.items():
                normalized_name = PipelineStage(_enum_value(stage_name)).value
                model = (
                    stage_result
                    if isinstance(stage_result, StageResult)
                    else StageResult(**stage_result)
                )
                stages_payload[normalized_name] = _model_dict(model)
        errors_payload = _to_json_value(errors) if errors is not None else None
        normalized_worker_id = (
            self._validate_worker_id(worker_id)
            if worker_id is not None
            else None
        )

        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM annotation_jobs WHERE job_id = ?",
                (job_id,),
                "annotation job",
            )
            if row["status"] != expected.value:
                connection.execute("ROLLBACK")
                raise VersionConflictError(
                    "annotation job status has changed",
                    details=[
                        {
                            "field": "expected_status",
                            "reason": (
                                f"expected {expected.value} but current status "
                                f"is {row['status']}"
                            ),
                        }
                    ],
                )
            if normalized_worker_id is not None and (
                row["claimed_by"] != normalized_worker_id
                or row["lease_expires_at"] is None
                or row["lease_expires_at"] <= now
            ):
                connection.execute("ROLLBACK")
                raise VersionConflictError(
                    "annotation job lease is not owned by this worker"
                )
            started_at = row["started_at"]
            completed_at = row["completed_at"]
            claimed_by = row["claimed_by"]
            lease_expires_at = row["lease_expires_at"]
            heartbeat_at = row["heartbeat_at"]
            if target == JobStatus.RUNNING and started_at is None:
                started_at = now
            if target in {
                JobStatus.SUCCEEDED,
                JobStatus.PARTIAL_FAILED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                completed_at = completed_at or now
                claimed_by = None
                lease_expires_at = None
                heartbeat_at = None
            connection.execute(
                """
                UPDATE annotation_jobs
                SET status = ?, stage = ?, progress_json = ?, stages_json = ?,
                    errors_json = ?, started_at = ?, completed_at = ?,
                    claimed_by = ?, lease_expires_at = ?, heartbeat_at = ?
                WHERE job_id = ?
                """,
                (
                    target.value,
                    stage_value if stage is not None else row["stage"],
                    canonical_json(progress_payload)
                    if progress_payload is not None
                    else row["progress_json"],
                    canonical_json(stages_payload)
                    if stages_payload is not None
                    else row["stages_json"],
                    canonical_json(errors_payload)
                    if errors_payload is not None
                    else row["errors_json"],
                    started_at,
                    completed_at,
                    claimed_by,
                    lease_expires_at,
                    heartbeat_at,
                    job_id,
                ),
            )
            connection.execute("COMMIT")
        return self.get_job(job_id)

    def cancel_job(
        self,
        job_id: str,
        *,
        actor_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Cancel a queued or running job without deleting its audit trail."""

        normalized_actor = self._validate_worker_id(actor_id)
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("cancellation reason must not be blank")
        job = self.get_job(job_id)
        current = JobStatus(job["status"])
        if current not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise InvalidStateTransitionError(
                f"cannot cancel a job in {current.value} status"
            )
        return self.update_job(
            job_id,
            expected_status=current,
            status=JobStatus.CANCELLED,
            errors=[
                *job["errors"],
                {
                    "asset_id": None,
                    "stage": (
                        job["stage"]
                        if job["stage"] == PipelineStage.GROUNDING_DINO.value
                        else None
                    ),
                    "code": "cancelled",
                    "message": (
                        f"cancelled by {normalized_actor}: "
                        f"{normalized_reason[:1800]}"
                    ),
                },
            ],
        )

    def list_recoverable_jobs(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id FROM annotation_jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at, job_id
                """
            ).fetchall()
        return [self.get_job(row["job_id"]) for row in rows]

    @staticmethod
    def _validate_worker_id(worker_id: str) -> str:
        normalized = worker_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError(
                "worker_id must contain between 1 and 128 characters"
            )
        return normalized

    def claim_next_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        required_stop_after: (
            str
            | PipelineStage
            | Sequence[str | PipelineStage]
            | None
        ) = None,
        full_pipeline_only: bool = False,
        grounding_prompt_required: bool = False,
    ) -> dict[str, Any] | None:
        """Atomically claim one queued or expired job for a GPU worker."""

        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        if lease_seconds < 1 or lease_seconds > 86_400:
            raise ValueError(
                "lease_seconds must be between 1 and 86400"
            )
        claimed_at = datetime.now(timezone.utc)
        claimed_at_value = claimed_at.isoformat()
        lease_expires_at = (
            claimed_at + timedelta(seconds=lease_seconds)
        ).isoformat()
        if full_pipeline_only and required_stop_after is not None:
            raise ValueError(
                "full_pipeline_only cannot be combined with "
                "required_stop_after"
            )
        if required_stop_after is None:
            stop_after_values: tuple[str, ...] = ()
        elif isinstance(required_stop_after, (str, PipelineStage)):
            stop_after_values = (
                PipelineStage(_enum_value(required_stop_after)).value,
            )
        else:
            stop_after_values = tuple(
                PipelineStage(_enum_value(value)).value
                for value in required_stop_after
            )
            if not stop_after_values:
                raise ValueError(
                    "required_stop_after sequence must not be empty"
                )

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stop_after_filter = ""
            parameters: tuple[Any, ...] = (claimed_at_value,)
            if full_pipeline_only:
                stop_after_filter = (
                    "AND json_extract(options_json, '$.stop_after') IS NULL"
                )
            elif stop_after_values:
                placeholders = ",".join("?" for _ in stop_after_values)
                stop_after_filter = (
                    "AND json_extract(options_json, '$.stop_after') "
                    f"IN ({placeholders})"
                )
                parameters += stop_after_values
            if grounding_prompt_required:
                stop_after_filter += " AND grounding_prompt <> ''"
            row = connection.execute(
                f"""
                SELECT * FROM annotation_jobs
                WHERE (
                        status = 'queued'
                        OR (
                            status = 'running'
                            AND (
                                claimed_by IS NULL
                                OR lease_expires_at IS NULL
                                OR lease_expires_at <= ?
                            )
                        )
                    )
                  {stop_after_filter}
                ORDER BY
                    CASE status WHEN 'running' THEN 0 ELSE 1 END,
                    created_at,
                    job_id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None

            stages = _json_loads(row["stages_json"], {})
            stage = row["stage"]
            started_at = row["started_at"]
            if row["status"] == JobStatus.QUEUED.value:
                stage = PipelineStage.GROUNDING_DINO.value
                started_at = started_at or claimed_at_value
                stages[stage] = {
                    "status": "running",
                    "started_at": claimed_at_value,
                    "completed_at": None,
                    "message": None,
                }
            connection.execute(
                """
                UPDATE annotation_jobs
                SET status = 'running', stage = ?, stages_json = ?,
                    started_at = ?, claimed_by = ?,
                    lease_expires_at = ?, heartbeat_at = ?,
                    attempt_count = attempt_count + 1
                WHERE job_id = ?
                """,
                (
                    stage,
                    canonical_json(stages),
                    started_at,
                    normalized_worker_id,
                    lease_expires_at,
                    claimed_at_value,
                    row["job_id"],
                ),
            )
            connection.execute("COMMIT")
        return self.get_job(row["job_id"])

    def heartbeat_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        """Extend a live job lease without changing pipeline progress."""

        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        if lease_seconds < 1 or lease_seconds > 86_400:
            raise ValueError(
                "lease_seconds must be between 1 and 86400"
            )
        heartbeat_at = datetime.now(timezone.utc)
        heartbeat_value = heartbeat_at.isoformat()
        lease_expires_at = (
            heartbeat_at + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE annotation_jobs
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE job_id = ?
                  AND status = 'running'
                  AND claimed_by = ?
                  AND lease_expires_at > ?
                """,
                (
                    heartbeat_value,
                    lease_expires_at,
                    job_id,
                    normalized_worker_id,
                    heartbeat_value,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    """
                    SELECT 1 FROM annotation_jobs
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if exists is None:
                    raise ResourceNotFoundError(
                        "annotation job was not found"
                    )
                raise VersionConflictError(
                    "annotation job lease is no longer active"
                )
        return self.get_job(job_id)

    # -------------------------------------------------------------- detections
    @staticmethod
    def _normalize_detection(
        detection: dict[str, Any],
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        entity = str(detection.get("entity", "")).strip()
        if not entity or len(entity) > 200:
            raise ValueError(
                "detection entity must contain between 1 and 200 characters"
            )
        box_xyxy = detection.get("box_xyxy")
        if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
            raise ValueError("box_xyxy must contain four coordinates")
        x1, y1, x2, y2 = [float(value) for value in box_xyxy]
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(
                "box_xyxy must have non-negative positive area"
            )
        if width is not None and x2 > width:
            raise ValueError("detection x2 exceeds asset width")
        if height is not None and y2 > height:
            raise ValueError("detection y2 exceeds asset height")
        box_score = float(detection.get("box_score"))
        phrase_score = float(detection.get("phrase_score"))
        if not 0 <= box_score <= 1 or not 0 <= phrase_score <= 1:
            raise ValueError(
                "detection scores must be between 0 and 1"
            )
        metadata = detection.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("detection metadata must be an object")
        canonical_json(metadata)
        return {
            "entity": entity,
            "box_xyxy": [x1, y1, x2, y2],
            "box_score": box_score,
            "phrase_score": phrase_score,
            "metadata": metadata,
        }

    @staticmethod
    def _assert_active_job_lease(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        worker_id: str,
        now: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT status, claimed_by, lease_expires_at
            FROM annotation_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ResourceNotFoundError(
                "annotation job was not found"
            )
        if (
            row["status"] != JobStatus.RUNNING.value
            or row["claimed_by"] != worker_id
            or row["lease_expires_at"] is None
            or row["lease_expires_at"] <= now
        ):
            raise VersionConflictError(
                "annotation job lease is no longer active"
            )

    def update_job_asset(
        self,
        *,
        job_id: str,
        asset_id: str,
        status: str,
        worker_id: str,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        if status not in {"pending", "running", "succeeded", "failed"}:
            raise ValueError("invalid job asset status")
        if error is not None and not isinstance(error, dict):
            raise ValueError("job asset error must be an object")
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_job_lease(
                connection,
                job_id=job_id,
                worker_id=normalized_worker_id,
                now=now,
            )
            cursor = connection.execute(
                """
                UPDATE job_assets
                SET status = ?, error_json = ?
                WHERE job_id = ? AND asset_id = ?
                """,
                (
                    status,
                    canonical_json(error) if error is not None else None,
                    job_id,
                    asset_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.execute("ROLLBACK")
                raise ResourceNotFoundError(
                    "asset is not part of the annotation job"
                )
            connection.execute("COMMIT")
        return {
            "job_id": job_id,
            "asset_id": asset_id,
            "status": status,
            "error": error,
        }

    def replace_detections(
        self,
        *,
        job_id: str,
        asset_id: str,
        detections: list[dict[str, Any]],
        worker_id: str,
    ) -> list[dict[str, Any]]:
        """Atomically replace one asset's detections under an active lease."""

        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        now = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._assert_active_job_lease(
                    connection,
                    job_id=job_id,
                    worker_id=normalized_worker_id,
                    now=now,
                )
                asset = connection.execute(
                    """
                    SELECT a.width, a.height
                    FROM job_assets ja
                    JOIN assets a ON a.asset_id = ja.asset_id
                    WHERE ja.job_id = ? AND ja.asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if asset is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )
                normalized = [
                    self._normalize_detection(
                        item,
                        width=int(asset["width"]),
                        height=int(asset["height"]),
                    )
                    for item in detections
                ]
                connection.execute(
                    """
                    DELETE FROM hazard_candidates
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                )
                connection.execute(
                    """
                    DELETE FROM detections
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                )
                saved: list[dict[str, Any]] = []
                for ordinal, item in enumerate(normalized):
                    identity = canonical_json(
                        {
                            "job_id": job_id,
                            "asset_id": asset_id,
                            "ordinal": ordinal,
                            **item,
                        }
                    )
                    detection_id = (
                        "det_"
                        + sha256_bytes(identity.encode("utf-8"))[:32]
                    )
                    metadata = {
                        **item["metadata"],
                        "ordinal": ordinal,
                    }
                    connection.execute(
                        """
                        INSERT INTO detections (
                            detection_id, job_id, asset_id, entity,
                            x1, y1, x2, y2, box_score, phrase_score,
                            metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            detection_id,
                            job_id,
                            asset_id,
                            item["entity"],
                            *item["box_xyxy"],
                            item["box_score"],
                            item["phrase_score"],
                            canonical_json(metadata),
                            now,
                        ),
                    )
                    saved.append(
                        {
                            "detection_id": detection_id,
                            **item,
                            "metadata": metadata,
                            "created_at": now,
                        }
                    )
                connection.execute("COMMIT")
                return saved
        except (
            ResourceNotFoundError,
            VersionConflictError,
            ValueError,
        ):
            raise
        except sqlite3.IntegrityError as exc:
            raise StorageError(
                "failed to replace annotation detections"
            ) from exc

    def add_detection(
        self,
        *,
        job_id: str,
        asset_id: str,
        entity: str,
        box_xyxy: list[float],
        box_score: float,
        phrase_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        normalized = self._normalize_detection(
            {
                "entity": entity,
                "box_xyxy": box_xyxy,
                "box_score": box_score,
                "phrase_score": phrase_score,
                "metadata": metadata or {},
            }
        )
        x1, y1, x2, y2 = normalized["box_xyxy"]
        detection_id = _new_id("det")
        created_at = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                membership = connection.execute(
                    """
                    SELECT 1 FROM job_assets
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if membership is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )
                connection.execute(
                    """
                    INSERT INTO detections (
                        detection_id, job_id, asset_id, entity,
                        x1, y1, x2, y2, box_score, phrase_score,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        detection_id,
                        job_id,
                        asset_id,
                        normalized["entity"],
                        x1,
                        y1,
                        x2,
                        y2,
                        normalized["box_score"],
                        normalized["phrase_score"],
                        canonical_json(normalized["metadata"]),
                        created_at,
                    ),
                )
                connection.execute("COMMIT")
        except ResourceNotFoundError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ResourceNotFoundError(
                "job or asset for detection was not found"
            ) from exc
        return {
            "detection_id": detection_id,
            "entity": normalized["entity"],
            "box_xyxy": [x1, y1, x2, y2],
            "box_score": normalized["box_score"],
            "phrase_score": normalized["phrase_score"],
            "metadata": normalized["metadata"],
            "created_at": created_at,
        }

    def list_detections(
        self,
        *,
        job_id: str,
        asset_id: str,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM detections
                WHERE job_id = ? AND asset_id = ?
                ORDER BY created_at, detection_id
                """,
                (job_id, asset_id),
            ).fetchall()
        return [
            {
                "detection_id": row["detection_id"],
                "entity": row["entity"],
                "box_xyxy": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "box_score": row["box_score"],
                "phrase_score": row["phrase_score"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_job_detections(
        self,
        *,
        job_id: str,
        asset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List public detection results for a job in stable asset order."""

        self._ensure_initialized()
        with self._connect() as connection:
            job = connection.execute(
                "SELECT 1 FROM annotation_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise ResourceNotFoundError(
                    "annotation job was not found"
                )
            if asset_id is not None:
                membership = connection.execute(
                    """
                    SELECT 1 FROM job_assets
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if membership is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )

            query = """
                SELECT d.*, ja.ordinal AS asset_ordinal
                FROM detections d
                JOIN job_assets ja
                  ON ja.job_id = d.job_id
                 AND ja.asset_id = d.asset_id
                WHERE d.job_id = ?
            """
            parameters: list[Any] = [job_id]
            if asset_id is not None:
                query += " AND d.asset_id = ?"
                parameters.append(asset_id)
            query += """
                ORDER BY ja.ordinal, d.created_at, d.detection_id
            """
            rows = connection.execute(query, parameters).fetchall()

        return [
            {
                "detection_id": row["detection_id"],
                "asset_id": row["asset_id"],
                "entity": row["entity"],
                "box_xyxy": [
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"],
                ],
                "box_score": row["box_score"],
                "phrase_score": row["phrase_score"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------- hazard candidates
    @staticmethod
    def _normalize_hazard_candidate(
        candidate: dict[str, Any],
        *,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        category = AnnotationCategory(
            _enum_value(candidate.get("category", ""))
        ).value
        target_entity = str(
            candidate.get("target_entity", "")
        ).strip()
        if not target_entity or len(target_entity) > 200:
            raise ValueError(
                "hazard target_entity must contain between 1 and 200 "
                "characters"
            )
        detection_ids = candidate.get("target_detection_ids")
        if (
            not isinstance(detection_ids, (list, tuple))
            or not detection_ids
        ):
            raise ValueError(
                "hazard target_detection_ids must be a non-empty array"
            )
        normalized_detection_ids = [
            str(value).strip() for value in detection_ids
        ]
        if (
            any(not value for value in normalized_detection_ids)
            or len(normalized_detection_ids)
            != len(set(normalized_detection_ids))
        ):
            raise ValueError(
                "hazard target_detection_ids must be unique non-empty "
                "strings"
            )
        box_value = candidate.get("box_xyxy")
        if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
            raise ValueError(
                "hazard box_xyxy must contain four coordinates"
            )
        x1, y1, x2, y2 = [float(value) for value in box_value]
        if (
            min(x1, y1) < 0
            or x2 <= x1
            or y2 <= y1
            or x2 > width
            or y2 > height
        ):
            raise ValueError(
                "hazard box_xyxy must be within the asset dimensions"
            )
        confidence = float(candidate.get("confidence"))
        if not 0 <= confidence <= 1:
            raise ValueError(
                "hazard confidence must be between 0 and 1"
            )
        rule_id = str(candidate.get("rule_id", "")).strip()
        rule_version = str(candidate.get("rule_version", "")).strip()
        if not rule_id or len(rule_id) > 200:
            raise ValueError(
                "hazard rule_id must contain between 1 and 200 characters"
            )
        if not rule_version or len(rule_version) > 128:
            raise ValueError(
                "hazard rule_version must contain between 1 and 128 "
                "characters"
            )
        evidence = candidate.get("evidence")
        if not isinstance(evidence, (list, tuple)) or not evidence:
            raise ValueError(
                "hazard evidence must be a non-empty array"
            )
        normalized_evidence = [str(value).strip() for value in evidence]
        if any(not value for value in normalized_evidence):
            raise ValueError(
                "hazard evidence entries must be non-empty strings"
            )
        metadata = candidate.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("hazard metadata must be an object")
        canonical_json(metadata)
        return {
            "category": category,
            "target_entity": target_entity,
            "target_detection_ids": normalized_detection_ids,
            "box_xyxy": [x1, y1, x2, y2],
            "confidence": confidence,
            "rule_id": rule_id,
            "rule_version": rule_version,
            "evidence": normalized_evidence,
            "metadata": metadata,
        }

    def replace_hazard_candidates(
        self,
        *,
        job_id: str,
        asset_id: str,
        candidates: list[dict[str, Any]],
        worker_id: str,
    ) -> list[dict[str, Any]]:
        """Replace derived hazard candidates under an active job lease."""

        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        now = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._assert_active_job_lease(
                    connection,
                    job_id=job_id,
                    worker_id=normalized_worker_id,
                    now=now,
                )
                asset = connection.execute(
                    """
                    SELECT a.width, a.height
                    FROM job_assets ja
                    JOIN assets a ON a.asset_id = ja.asset_id
                    WHERE ja.job_id = ? AND ja.asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if asset is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )
                normalized = [
                    self._normalize_hazard_candidate(
                        candidate,
                        width=int(asset["width"]),
                        height=int(asset["height"]),
                    )
                    for candidate in candidates
                ]
                referenced_ids = {
                    detection_id
                    for candidate in normalized
                    for detection_id in candidate[
                        "target_detection_ids"
                    ]
                }
                if referenced_ids:
                    placeholders = ",".join(
                        "?" for _ in referenced_ids
                    )
                    found_ids = {
                        row["detection_id"]
                        for row in connection.execute(
                            """
                            SELECT detection_id FROM detections
                            WHERE job_id = ? AND asset_id = ?
                              AND detection_id IN (
                            """
                            + placeholders
                            + ")",
                            (
                                job_id,
                                asset_id,
                                *sorted(referenced_ids),
                            ),
                        ).fetchall()
                    }
                    missing_ids = sorted(referenced_ids - found_ids)
                    if missing_ids:
                        raise ValueError(
                            "hazard candidates reference detections outside "
                            "the job asset: "
                            + ", ".join(missing_ids)
                        )
                connection.execute(
                    """
                    DELETE FROM hazard_candidates
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                )
                saved: list[dict[str, Any]] = []
                for ordinal, item in enumerate(normalized):
                    identity = canonical_json(
                        {
                            "job_id": job_id,
                            "asset_id": asset_id,
                            "ordinal": ordinal,
                            **item,
                        }
                    )
                    hazard_id = (
                        "haz_"
                        + sha256_bytes(identity.encode("utf-8"))[:32]
                    )
                    metadata = {
                        **item["metadata"],
                        "ordinal": ordinal,
                    }
                    connection.execute(
                        """
                        INSERT INTO hazard_candidates (
                            hazard_id, job_id, asset_id, category,
                            target_entity, target_detection_ids_json,
                            x1, y1, x2, y2, confidence, rule_id,
                            rule_version, evidence_json, metadata_json,
                            created_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            hazard_id,
                            job_id,
                            asset_id,
                            item["category"],
                            item["target_entity"],
                            canonical_json(
                                item["target_detection_ids"]
                            ),
                            *item["box_xyxy"],
                            item["confidence"],
                            item["rule_id"],
                            item["rule_version"],
                            canonical_json(item["evidence"]),
                            canonical_json(metadata),
                            now,
                        ),
                    )
                    saved.append(
                        {
                            "hazard_id": hazard_id,
                            "asset_id": asset_id,
                            **item,
                            "metadata": metadata,
                            "created_at": now,
                        }
                    )
                connection.execute("COMMIT")
                return saved
        except (
            ResourceNotFoundError,
            VersionConflictError,
            ValueError,
        ):
            raise
        except sqlite3.IntegrityError as exc:
            raise StorageError(
                "failed to replace hazard candidates"
            ) from exc

    def list_job_hazard_candidates(
        self,
        *,
        job_id: str,
        asset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List derived hazard candidates in stable asset/rule order."""

        self._ensure_initialized()
        with self._connect() as connection:
            job = connection.execute(
                "SELECT 1 FROM annotation_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise ResourceNotFoundError(
                    "annotation job was not found"
                )
            if asset_id is not None:
                membership = connection.execute(
                    """
                    SELECT 1 FROM job_assets
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if membership is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )
            query = """
                SELECT h.*, ja.ordinal AS asset_ordinal
                FROM hazard_candidates h
                JOIN job_assets ja
                  ON ja.job_id = h.job_id
                 AND ja.asset_id = h.asset_id
                WHERE h.job_id = ?
            """
            parameters: list[Any] = [job_id]
            if asset_id is not None:
                query += " AND h.asset_id = ?"
                parameters.append(asset_id)
            query += """
                ORDER BY ja.ordinal, h.category, h.rule_id, h.hazard_id
            """
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "hazard_id": row["hazard_id"],
                "asset_id": row["asset_id"],
                "category": row["category"],
                "target_entity": row["target_entity"],
                "target_detection_ids": _json_loads(
                    row["target_detection_ids_json"],
                    [],
                ),
                "box_xyxy": [
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"],
                ],
                "confidence": row["confidence"],
                "rule_id": row["rule_id"],
                "rule_version": row["rule_version"],
                "evidence": _json_loads(row["evidence_json"], []),
                "metadata": _json_loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------- tasks
    def create_task(
        self,
        *,
        job_id: str,
        asset_id: str,
        category: str | AnnotationCategory,
        annotation: dict[str, Any] | AnnotationContent,
        provenance: dict[str, Any] | Provenance,
        warnings: list[str] | None = None,
        source_hazard_id: str | None = None,
        source_detection_id: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        category_value = AnnotationCategory(_enum_value(category)).value
        annotation_payload = _model_dict(
            annotation
            if isinstance(annotation, AnnotationContent)
            else AnnotationContent(**annotation)
        )
        provenance_payload = _model_dict(
            provenance
            if isinstance(provenance, Provenance)
            else Provenance(**provenance)
        )
        task_id = _new_id("tsk")
        now = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if source_hazard_id is not None:
                    existing = connection.execute(
                        """
                        SELECT task_id FROM annotation_tasks
                        WHERE source_hazard_id = ?
                        """,
                        (source_hazard_id,),
                    ).fetchone()
                    if existing is not None:
                        connection.execute("COMMIT")
                        return self.get_task(existing["task_id"])
                if source_detection_id is not None:
                    existing = connection.execute(
                        """
                        SELECT task_id FROM annotation_tasks
                        WHERE source_detection_id = ?
                        """,
                        (source_detection_id,),
                    ).fetchone()
                    if existing is not None:
                        connection.execute("COMMIT")
                        return self.get_task(existing["task_id"])
                membership = connection.execute(
                    """
                    SELECT 1 FROM job_assets
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if membership is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )
                if source_hazard_id is not None:
                    hazard = connection.execute(
                        """
                        SELECT category FROM hazard_candidates
                        WHERE hazard_id = ? AND job_id = ? AND asset_id = ?
                        """,
                        (source_hazard_id, job_id, asset_id),
                    ).fetchone()
                    if hazard is None:
                        raise ResourceNotFoundError(
                            "hazard candidate was not found for this job asset"
                        )
                    if hazard["category"] != category_value:
                        raise ValueError(
                            "task category must match the hazard candidate"
                        )
                if source_detection_id is not None:
                    detection = connection.execute(
                        """
                        SELECT 1 FROM detections
                        WHERE detection_id = ? AND job_id = ? AND asset_id = ?
                        """,
                        (source_detection_id, job_id, asset_id),
                    ).fetchone()
                    if detection is None:
                        raise ResourceNotFoundError(
                            "source detection was not found for this job asset"
                        )
                connection.execute(
                    """
                    INSERT INTO annotation_tasks (
                        task_id, job_id, asset_id, category, status, version,
                        annotation_json, provenance_json, warnings_json,
                        source_hazard_id, source_detection_id,
                        created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, 'generated', 1, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        task_id,
                        job_id,
                        asset_id,
                        category_value,
                        canonical_json(annotation_payload),
                        canonical_json(provenance_payload),
                        canonical_json(warnings or []),
                        source_hazard_id,
                        source_detection_id,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO task_versions (
                        task_id, version, annotation_json, status,
                        change_kind, created_at
                    ) VALUES (?, 1, ?, 'generated', 'generated', ?)
                    """,
                    (task_id, canonical_json(annotation_payload), now),
                )
                connection.execute("COMMIT")
        except ResourceNotFoundError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ResourceNotFoundError(
                "job or asset for annotation task was not found"
            ) from exc
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                """
                SELECT t.*, a.group_id, a.width, a.height
                FROM annotation_tasks t
                JOIN assets a ON a.asset_id = t.asset_id
                WHERE t.task_id = ?
                """,
                (task_id,),
                "annotation task",
            )
            detections = connection.execute(
                """
                SELECT * FROM detections
                WHERE job_id = ? AND asset_id = ?
                ORDER BY created_at, detection_id
                """,
                (row["job_id"], row["asset_id"]),
            ).fetchall()
            artifacts = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE task_id = ? ORDER BY created_at, artifact_id
                """,
                (task_id,),
            ).fetchall()
            source_hazard = (
                connection.execute(
                    """
                    SELECT * FROM hazard_candidates WHERE hazard_id = ?
                    """,
                    (row["source_hazard_id"],),
                ).fetchone()
                if row["source_hazard_id"] is not None
                else None
            )
        return self._task_payload(
            row,
            detections,
            artifacts,
            source_hazard,
        )

    def _task_payload(
        self,
        row: sqlite3.Row,
        detections: list[sqlite3.Row],
        artifacts: list[sqlite3.Row],
        source_hazard: sqlite3.Row | None,
    ) -> dict[str, Any]:
        links: dict[str, str | None] = {
            "detection_overlay_url": None,
            "mask_overlay_url": None,
            "mask_png_url": None,
            "crop_url": None,
        }
        link_keys = {
            "detections": "detection_overlay_url",
            "mask-overlay": "mask_overlay_url",
            "mask": "mask_png_url",
            "crop": "crop_url",
        }
        for artifact in artifacts:
            links[link_keys[artifact["artifact_type"]]] = (
                f"/v1/annotation/tasks/{row['task_id']}/artifacts/"
                f"{artifact['artifact_type']}"
            )
        return {
            "task_id": row["task_id"],
            "job_id": row["job_id"],
            "asset": {
                "asset_id": row["asset_id"],
                "group_id": row["group_id"],
                "width": row["width"],
                "height": row["height"],
                "image_url": (
                    f"/v1/annotation/assets/{row['asset_id']}/content"
                ),
            },
            "category": row["category"],
            "status": row["status"],
            "version": row["version"],
            "detections": [
                {
                    "detection_id": item["detection_id"],
                    "entity": item["entity"],
                    "box_xyxy": [item["x1"], item["y1"], item["x2"], item["y2"]],
                    "box_score": item["box_score"],
                    "phrase_score": item["phrase_score"],
                }
                for item in detections
            ],
            "annotation": _json_loads(row["annotation_json"], {}),
            "artifacts": links,
            "provenance": _json_loads(row["provenance_json"], {}),
            "source_detection_id": row["source_detection_id"],
            "source_hazard": (
                {
                    "hazard_id": source_hazard["hazard_id"],
                    "asset_id": source_hazard["asset_id"],
                    "category": source_hazard["category"],
                    "target_entity": source_hazard["target_entity"],
                    "target_detection_ids": _json_loads(
                        source_hazard["target_detection_ids_json"],
                        [],
                    ),
                    "box_xyxy": [
                        source_hazard["x1"],
                        source_hazard["y1"],
                        source_hazard["x2"],
                        source_hazard["y2"],
                    ],
                    "confidence": source_hazard["confidence"],
                    "rule_id": source_hazard["rule_id"],
                    "rule_version": source_hazard["rule_version"],
                    "evidence": _json_loads(
                        source_hazard["evidence_json"],
                        [],
                    ),
                    "metadata": _json_loads(
                        source_hazard["metadata_json"],
                        {},
                    ),
                    "created_at": source_hazard["created_at"],
                }
                if source_hazard is not None
                else None
            ),
            "warnings": _json_loads(row["warnings_json"], []),
            "primary_result": row["primary_result"],
            "annotator_id": row["annotator_id"],
            "reviewer_id": row["reviewer_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _encode_task_cursor(updated_at: str, task_id: str) -> str:
        payload = canonical_json([updated_at, task_id]).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_task_cursor(cursor: str) -> tuple[str, str]:
        try:
            padding = "=" * (-len(cursor) % 4)
            decoded = base64.urlsafe_b64decode(
                (cursor + padding).encode("ascii")
            )
            value = json.loads(decoded.decode("utf-8"))
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(isinstance(item, str) and item for item in value)
            ):
                raise ValueError
            return value[0], value[1]
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValidationServiceError(
                "task cursor is invalid",
                details=[
                    {
                        "field": "cursor",
                        "reason": "cursor must be returned by this API",
                    }
                ],
            ) from exc

    def list_tasks(
        self,
        *,
        status: str | TaskStatus | None = None,
        category: str | AnnotationCategory | None = None,
        group_id: str | None = None,
        job_id: str | None = None,
        annotator_id: str | None = None,
        reviewer_id: str | None = None,
        bad_case_type: str | BadCaseType | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")

        conditions: list[str] = []
        parameters: list[Any] = []
        enum_filters = (
            ("t.status", status, TaskStatus),
            ("t.category", category, AnnotationCategory),
            ("t.primary_result", bad_case_type, BadCaseType),
        )
        for column, value, enum_type in enum_filters:
            if value is not None:
                normalized = enum_type(_enum_value(value)).value
                conditions.append(f"{column} = ?")
                parameters.append(normalized)
        for column, value in (
            ("a.group_id", group_id),
            ("t.job_id", job_id),
            ("t.annotator_id", annotator_id),
            ("t.reviewer_id", reviewer_id),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                parameters.append(value)
        if cursor is not None:
            cursor_updated_at, cursor_task_id = self._decode_task_cursor(
                cursor
            )
            conditions.append(
                "(t.updated_at < ? OR "
                "(t.updated_at = ? AND t.task_id < ?))"
            )
            parameters.extend(
                [cursor_updated_at, cursor_updated_at, cursor_task_id]
            )

        where = (
            "WHERE " + " AND ".join(conditions) if conditions else ""
        )
        parameters.append(limit + 1)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT t.*, a.group_id,
                    (
                        SELECT ar.artifact_type
                        FROM artifacts ar
                        WHERE ar.task_id = t.task_id
                          AND ar.artifact_type IN ('crop', 'mask-overlay')
                        ORDER BY
                            CASE ar.artifact_type
                                WHEN 'crop' THEN 0 ELSE 1
                            END,
                            ar.created_at DESC,
                            ar.artifact_id DESC
                        LIMIT 1
                    ) AS thumbnail_artifact_type
                FROM annotation_tasks t
                JOIN assets a ON a.asset_id = t.asset_id
                {where}
                ORDER BY t.updated_at DESC, t.task_id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        has_more = len(rows) > limit
        page = rows[:limit]
        items = []
        for row in page:
            artifact_type = row["thumbnail_artifact_type"]
            items.append(
                {
                    "task_id": row["task_id"],
                    "asset_id": row["asset_id"],
                    "group_id": row["group_id"],
                    "category": row["category"],
                    "status": row["status"],
                    "version": row["version"],
                    "source_detection_id": row["source_detection_id"],
                    "source_hazard_id": row["source_hazard_id"],
                    "primary_result": row["primary_result"],
                    "annotator_id": row["annotator_id"],
                    "reviewer_id": row["reviewer_id"],
                    "thumbnail_url": (
                        f"/v1/annotation/tasks/{row['task_id']}/artifacts/"
                        f"{artifact_type}"
                        if artifact_type is not None
                        else None
                    ),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        next_cursor = (
            self._encode_task_cursor(
                page[-1]["updated_at"],
                page[-1]["task_id"],
            )
            if has_more and page
            else None
        )
        return {"items": items, "next_cursor": next_cursor}

    def _update_task_snapshot(
        self,
        connection: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        expected_version: int,
        annotation_payload: dict[str, Any],
        target_status: TaskStatus,
        editor_id: str,
        change_kind: str,
        primary_result: str | None = None,
        annotator_id: str | None = None,
        reviewer_id: str | None = None,
        comment: str | None = None,
    ) -> int:
        if row["version"] != expected_version:
            raise VersionConflictError(
                "annotation task has been modified",
                details=[
                    {
                        "field": "expected_version",
                        "reason": (
                            f"expected {expected_version} but current version "
                            f"is {row['version']}"
                        ),
                    }
                ],
            )
        current_status = TaskStatus(row["status"])
        if target_status != current_status:
            ensure_task_transition(current_status, target_status)
        version = expected_version + 1
        now = utc_now()
        cursor = connection.execute(
            """
            UPDATE annotation_tasks
            SET status = ?, version = ?, annotation_json = ?,
                primary_result = COALESCE(?, primary_result),
                annotator_id = COALESCE(?, annotator_id),
                reviewer_id = COALESCE(?, reviewer_id),
                updated_at = ?
            WHERE task_id = ? AND version = ?
            """,
            (
                target_status.value,
                version,
                canonical_json(annotation_payload),
                primary_result,
                annotator_id,
                reviewer_id,
                now,
                row["task_id"],
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise VersionConflictError("annotation task has been modified")
        connection.execute(
            """
            INSERT INTO task_versions (
                task_id, version, annotation_json, status,
                editor_id, change_kind, comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["task_id"],
                version,
                canonical_json(annotation_payload),
                target_status.value,
                editor_id,
                change_kind,
                comment,
                now,
            ),
        )
        return version

    def save_task_draft(
        self,
        task_id: str,
        *,
        expected_version: int,
        annotation: dict[str, Any] | AnnotationContent,
        editor_id: str,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        annotation_payload = _model_dict(
            annotation
            if isinstance(annotation, AnnotationContent)
            else AnnotationContent(**annotation)
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM annotation_tasks WHERE task_id = ?",
                (task_id,),
                "annotation task",
            )
            current = TaskStatus(row["status"])
            if current in {TaskStatus.GENERATED, TaskStatus.CHANGES_REQUESTED}:
                target = TaskStatus.ANNOTATING
            elif current == TaskStatus.ANNOTATING:
                target = current
            else:
                raise InvalidStateTransitionError(
                    f"cannot edit a task in {current.value} status"
                )
            self._update_task_snapshot(
                connection,
                row=row,
                expected_version=expected_version,
                annotation_payload=annotation_payload,
                target_status=target,
                editor_id=editor_id,
                change_kind="draft",
            )
            connection.execute("COMMIT")
        return self.get_task(task_id)

    def submit_task(
        self,
        task_id: str,
        *,
        expected_version: int,
        annotator_id: str,
        primary_result: str | BadCaseType,
        comment: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        result = BadCaseType(_enum_value(primary_result)).value
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                """
                SELECT t.*, a.width, a.height
                FROM annotation_tasks t
                JOIN assets a ON a.asset_id = t.asset_id
                WHERE t.task_id = ?
                """,
                (task_id,),
                "annotation task",
            )
            if row["version"] != expected_version:
                raise VersionConflictError(
                    "annotation task has been modified",
                    details=[
                        {
                            "field": "expected_version",
                            "reason": (
                                f"expected {expected_version} but current "
                                f"version is {row['version']}"
                            ),
                        }
                    ],
                )
            validate_annotation_for_submission(
                _json_loads(row["annotation_json"], {}),
                width=row["width"],
                height=row["height"],
                category=row["category"],
            )
            self._update_task_snapshot(
                connection,
                row=row,
                expected_version=expected_version,
                annotation_payload=_json_loads(row["annotation_json"], {}),
                target_status=TaskStatus.REVIEW_PENDING,
                editor_id=annotator_id,
                change_kind="submit",
                primary_result=result,
                annotator_id=annotator_id,
                comment=comment,
            )
            connection.execute("COMMIT")
        return self.get_task(task_id)

    def review_task(
        self,
        task_id: str,
        *,
        expected_version: int,
        reviewer_id: str,
        decision: str | ReviewDecision,
        primary_result: str | BadCaseType,
        comment: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        decision_value = ReviewDecision(_enum_value(decision))
        result = BadCaseType(_enum_value(primary_result)).value
        target_by_decision = {
            ReviewDecision.ACCEPT: TaskStatus.ACCEPTED,
            ReviewDecision.REQUEST_CHANGES: TaskStatus.CHANGES_REQUESTED,
            ReviewDecision.NEEDS_EXPERT: TaskStatus.NEEDS_EXPERT,
            ReviewDecision.REJECT: TaskStatus.REJECTED,
        }
        review_id = _new_id("rev")
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                """
                SELECT t.*, a.width, a.height
                FROM annotation_tasks t
                JOIN assets a ON a.asset_id = t.asset_id
                WHERE t.task_id = ?
                """,
                (task_id,),
                "annotation task",
            )
            if row["version"] != expected_version:
                raise VersionConflictError(
                    "annotation task has been modified",
                    details=[
                        {
                            "field": "expected_version",
                            "reason": (
                                f"expected {expected_version} but current "
                                f"version is {row['version']}"
                            ),
                        }
                    ],
                )
            if decision_value == ReviewDecision.ACCEPT:
                validate_annotation_for_submission(
                    _json_loads(row["annotation_json"], {}),
                    width=row["width"],
                    height=row["height"],
                    category=row["category"],
                )
            version = self._update_task_snapshot(
                connection,
                row=row,
                expected_version=expected_version,
                annotation_payload=_json_loads(row["annotation_json"], {}),
                target_status=target_by_decision[decision_value],
                editor_id=reviewer_id,
                change_kind="review",
                primary_result=result,
                reviewer_id=reviewer_id,
                comment=comment,
            )
            connection.execute(
                """
                INSERT INTO reviews (
                    review_id, task_id, task_version, reviewer_id,
                    decision, primary_result, comment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    task_id,
                    version,
                    reviewer_id,
                    decision_value.value,
                    result,
                    comment,
                    now,
                ),
            )
            connection.execute("COMMIT")
        return self.get_task(task_id)

    def invalidate_task(
        self,
        task_id: str,
        *,
        expected_version: int,
        actor_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Mark an editable sample as rejected while preserving all versions."""

        normalized_actor = self._validate_worker_id(actor_id)
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("invalidation reason must not be blank")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM annotation_tasks WHERE task_id = ?",
                (task_id,),
                "annotation task",
            )
            current = TaskStatus(row["status"])
            if current not in {
                TaskStatus.GENERATED,
                TaskStatus.ANNOTATING,
                TaskStatus.REVIEW_PENDING,
                TaskStatus.CHANGES_REQUESTED,
                TaskStatus.NEEDS_EXPERT,
            }:
                connection.execute("ROLLBACK")
                raise InvalidStateTransitionError(
                    f"cannot invalidate a task in {current.value} status"
                )
            self._update_task_snapshot(
                connection,
                row=row,
                expected_version=expected_version,
                annotation_payload=_json_loads(row["annotation_json"], {}),
                target_status=TaskStatus.REJECTED,
                editor_id=normalized_actor,
                change_kind="invalidate",
                primary_result=BadCaseType.OTHER.value,
                comment=f"作废：{normalized_reason}",
            )
            connection.execute("COMMIT")
        return self.get_task(task_id)

    def freeze_task(
        self,
        task_id: str,
        *,
        expected_version: int,
        editor_id: str,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM annotation_tasks WHERE task_id = ?",
                (task_id,),
                "annotation task",
            )
            self._update_task_snapshot(
                connection,
                row=row,
                expected_version=expected_version,
                annotation_payload=_json_loads(row["annotation_json"], {}),
                target_status=TaskStatus.FROZEN,
                editor_id=editor_id,
                change_kind="freeze",
            )
            connection.execute("COMMIT")
        return self.get_task(task_id)

    def list_task_versions(self, task_id: str) -> list[dict[str, Any]]:
        self._ensure_initialized()
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM annotation_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if exists is None:
                raise ResourceNotFoundError("annotation task was not found")
            rows = connection.execute(
                """
                SELECT * FROM task_versions
                WHERE task_id = ? ORDER BY version
                """,
                (task_id,),
            ).fetchall()
        return [
            {
                "version": row["version"],
                "annotation": _json_loads(row["annotation_json"], {}),
                "status": row["status"],
                "editor_id": row["editor_id"],
                "change_kind": row["change_kind"],
                "comment": row["comment"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_reviews(self, task_id: str) -> list[dict[str, Any]]:
        self._ensure_initialized()
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM annotation_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if exists is None:
                raise ResourceNotFoundError(
                    "annotation task was not found"
                )
            rows = connection.execute(
                "SELECT * FROM reviews WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -------------------------------------------------------------- operations
    def create_prompt_enrichment_operation(
        self,
        *,
        task_id: str,
        expected_version: int,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if expected_version < 1:
            raise ValueError("expected_version must be positive")
        operation_id = _new_id("op")
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self._row_or_not_found(
                connection,
                """
                SELECT version FROM annotation_tasks WHERE task_id = ?
                """,
                (task_id,),
                "annotation task",
            )
            if task["version"] != expected_version:
                connection.execute("ROLLBACK")
                raise VersionConflictError(
                    "annotation task version has changed",
                    details=[
                        {
                            "field": "expected_version",
                            "reason": (
                                f"expected {expected_version} but current "
                                f"version is {task['version']}"
                            ),
                        }
                    ],
                )
            available_artifact = connection.execute(
                """
                SELECT 1 FROM artifacts
                WHERE task_id = ?
                  AND artifact_type IN ('mask-overlay', 'mask')
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if available_artifact is None:
                connection.execute("ROLLBACK")
                raise ValidationServiceError(
                    "prompt generation requires a SAM mask artifact",
                    details=[
                        {
                            "field": "task_id",
                            "reason": (
                                "store mask-overlay or mask before requesting "
                                "Qwen prompt generation"
                            ),
                        }
                    ],
                )
            existing = connection.execute(
                """
                SELECT operation_id FROM annotation_operations
                WHERE operation_type = 'prompt_enrichment'
                  AND task_id = ? AND task_version = ?
                  AND status IN ('queued', 'running')
                ORDER BY created_at, operation_id LIMIT 1
                """,
                (task_id, expected_version),
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return self.get_operation(existing["operation_id"])
            connection.execute(
                """
                INSERT INTO annotation_operations(
                    operation_id, operation_type, task_id, task_version,
                    status, request_json, created_at
                ) VALUES (
                    ?, 'prompt_enrichment', ?, ?, 'queued', '{}', ?
                )
                """,
                (
                    operation_id,
                    task_id,
                    expected_version,
                    created_at,
                ),
            )
            connection.execute("COMMIT")
        return self.get_operation(operation_id)

    def create_mask_candidate_operation(
        self,
        *,
        task_id: str,
        expected_version: int,
        box_xyxy: list[float],
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if expected_version < 1:
            raise ValueError("expected_version must be positive")
        if len(box_xyxy) != 4:
            raise ValueError("box_xyxy must contain four coordinates")
        box = [float(value) for value in box_xyxy]
        x1, y1, x2, y2 = box
        if min(box) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(
                "box_xyxy must have non-negative positive area"
            )
        request_payload = {"box_xyxy": box}
        request_json = canonical_json(request_payload)
        operation_id = _new_id("op")
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self._row_or_not_found(
                connection,
                """
                SELECT t.version, a.width, a.height
                FROM annotation_tasks t
                JOIN assets a ON a.asset_id = t.asset_id
                WHERE t.task_id = ?
                """,
                (task_id,),
                "annotation task",
            )
            if task["version"] != expected_version:
                connection.execute("ROLLBACK")
                raise VersionConflictError(
                    "annotation task version has changed",
                    details=[
                        {
                            "field": "expected_version",
                            "reason": (
                                f"expected {expected_version} but current "
                                f"version is {task['version']}"
                            ),
                        }
                    ],
                )
            if x2 > task["width"] or y2 > task["height"]:
                connection.execute("ROLLBACK")
                raise ValidationServiceError(
                    "mask candidate box exceeds image bounds",
                    details=[
                        {
                            "field": "box_xyxy",
                            "reason": (
                                f"box must fit inside {task['width']}x"
                                f"{task['height']} image"
                            ),
                        }
                    ],
                )
            existing = connection.execute(
                """
                SELECT operation_id FROM annotation_operations
                WHERE operation_type = 'mask_candidate'
                  AND task_id = ? AND task_version = ?
                  AND request_json = ?
                  AND status IN ('queued', 'running')
                ORDER BY created_at, operation_id LIMIT 1
                """,
                (task_id, expected_version, request_json),
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return self.get_operation(existing["operation_id"])
            connection.execute(
                """
                INSERT INTO annotation_operations(
                    operation_id, operation_type, task_id, task_version,
                    status, request_json, created_at
                ) VALUES (
                    ?, 'mask_candidate', ?, ?, 'queued', ?, ?
                )
                """,
                (
                    operation_id,
                    task_id,
                    expected_version,
                    request_json,
                    created_at,
                ),
            )
            connection.execute("COMMIT")
        return self.get_operation(operation_id)

    def replace_generated_task_content(
        self,
        task_id: str,
        *,
        expected_version: int,
        annotation: dict[str, Any] | AnnotationContent,
        provenance_updates: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        """Replace model-generated version 1 without creating a user edit."""

        self._ensure_initialized()
        annotation_payload = _model_dict(
            annotation
            if isinstance(annotation, AnnotationContent)
            else AnnotationContent(**annotation)
        )
        canonical_json(provenance_updates)
        canonical_json(warnings)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                """
                SELECT * FROM annotation_tasks WHERE task_id = ?
                """,
                (task_id,),
                "annotation task",
            )
            if (
                row["status"] != TaskStatus.GENERATED.value
                or row["version"] != expected_version
            ):
                connection.execute("ROLLBACK")
                raise VersionConflictError(
                    "generated task changed before model output was applied"
                )
            provenance = _json_loads(row["provenance_json"], {})
            provenance.update(provenance_updates)
            Provenance(**provenance)
            now = utc_now()
            connection.execute(
                """
                UPDATE annotation_tasks
                SET annotation_json = ?, provenance_json = ?,
                    warnings_json = ?, updated_at = ?
                WHERE task_id = ? AND version = ?
                  AND status = 'generated'
                """,
                (
                    canonical_json(annotation_payload),
                    canonical_json(provenance),
                    canonical_json(warnings),
                    now,
                    task_id,
                    expected_version,
                ),
            )
            connection.execute(
                """
                UPDATE task_versions
                SET annotation_json = ?
                WHERE task_id = ? AND version = ?
                """,
                (
                    canonical_json(annotation_payload),
                    task_id,
                    expected_version,
                ),
            )
            connection.execute("COMMIT")
        return self.get_task(task_id)

    @staticmethod
    def _operation_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "operation_id": row["operation_id"],
            "operation_type": row["operation_type"],
            "task_id": row["task_id"],
            "task_version": row["task_version"],
            "status": row["status"],
            "request": _json_loads(row["request_json"], {}),
            "result": _json_loads(row["result_json"], None),
            "error": _json_loads(row["error_json"], None),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "claimed_by": row["claimed_by"],
            "lease_expires_at": row["lease_expires_at"],
            "heartbeat_at": row["heartbeat_at"],
            "attempt_count": row["attempt_count"],
        }

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                """
                SELECT * FROM annotation_operations
                WHERE operation_id = ?
                """,
                (operation_id,),
                "annotation operation",
            )
        return self._operation_payload(row)

    def cancel_operation(
        self,
        operation_id: str,
        *,
        actor_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Cancel a queued or running SAM/Qwen operation."""

        normalized_actor = self._validate_worker_id(actor_id)
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("cancellation reason must not be blank")
        now = utc_now()
        result = {
            "cancelled_by": normalized_actor,
            "reason": normalized_reason,
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row_or_not_found(
                connection,
                """
                SELECT status FROM annotation_operations
                WHERE operation_id = ?
                """,
                (operation_id,),
                "annotation operation",
            )
            if row["status"] not in {
                OperationStatus.QUEUED.value,
                OperationStatus.RUNNING.value,
            }:
                connection.execute("ROLLBACK")
                raise InvalidStateTransitionError(
                    "only queued or running operations can be cancelled"
                )
            connection.execute(
                """
                UPDATE annotation_operations
                SET status = 'cancelled', result_json = ?,
                    error_json = NULL, completed_at = ?,
                    claimed_by = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL
                WHERE operation_id = ?
                """,
                (canonical_json(result), now, operation_id),
            )
            connection.execute("COMMIT")
        return self.get_operation(operation_id)

    def claim_next_operation(
        self,
        *,
        operation_type: str,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        if operation_type not in {"mask_candidate", "prompt_enrichment"}:
            raise ValueError("unsupported operation_type")
        if lease_seconds < 1 or lease_seconds > 86_400:
            raise ValueError(
                "lease_seconds must be between 1 and 86400"
            )
        claimed_at = datetime.now(timezone.utc)
        now = claimed_at.isoformat()
        expires_at = (
            claimed_at + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM annotation_operations
                WHERE operation_type = ?
                  AND (
                    status = 'queued'
                    OR (
                      status = 'running'
                      AND (
                        claimed_by IS NULL
                        OR lease_expires_at IS NULL
                        OR lease_expires_at <= ?
                      )
                    )
                  )
                ORDER BY
                  CASE status WHEN 'running' THEN 0 ELSE 1 END,
                  created_at, operation_id
                LIMIT 1
                """,
                (operation_type, now),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                """
                UPDATE annotation_operations
                SET status = 'running',
                    claimed_by = ?,
                    lease_expires_at = ?,
                    heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    attempt_count = attempt_count + 1,
                    result_json = NULL,
                    error_json = NULL,
                    completed_at = NULL
                WHERE operation_id = ?
                """,
                (
                    normalized_worker_id,
                    expires_at,
                    now,
                    now,
                    row["operation_id"],
                ),
            )
            connection.execute("COMMIT")
        return self.get_operation(row["operation_id"])

    def heartbeat_operation(
        self,
        operation_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        heartbeat_at = datetime.now(timezone.utc)
        now = heartbeat_at.isoformat()
        expires_at = (
            heartbeat_at + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE annotation_operations
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE operation_id = ?
                  AND status = 'running'
                  AND claimed_by = ?
                  AND lease_expires_at > ?
                """,
                (
                    now,
                    expires_at,
                    operation_id,
                    normalized_worker_id,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError(
                    "annotation operation lease is no longer active"
                )
        return self.get_operation(operation_id)

    def complete_operation(
        self,
        operation_id: str,
        *,
        worker_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        result_json = canonical_json(result)
        now = utc_now()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE annotation_operations
                SET status = 'succeeded', result_json = ?,
                    error_json = NULL, completed_at = ?,
                    claimed_by = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL
                WHERE operation_id = ?
                  AND status = 'running'
                  AND claimed_by = ?
                  AND lease_expires_at > ?
                """,
                (
                    result_json,
                    now,
                    operation_id,
                    normalized_worker_id,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError(
                    "annotation operation lease is no longer active"
                )
        return self.get_operation(operation_id)

    def fail_operation(
        self,
        operation_id: str,
        *,
        worker_id: str,
        code: str,
        message: str,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        safe_code = code.strip()[:128] or "internal_error"
        safe_message = message.strip()[:2000] or "operation failed"
        now = utc_now()
        error = {
            "request_id": None,
            "code": safe_code,
            "message": safe_message,
            "details": [],
        }
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE annotation_operations
                SET status = 'failed', result_json = NULL,
                    error_json = ?, completed_at = ?,
                    claimed_by = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL
                WHERE operation_id = ?
                  AND status = 'running'
                  AND claimed_by = ?
                """,
                (
                    canonical_json(error),
                    now,
                    operation_id,
                    normalized_worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError(
                    "annotation operation lease is no longer owned by worker"
                )
        return self.get_operation(operation_id)

    # ---------------------------------------------------------- job artifacts
    def store_job_artifact(
        self,
        *,
        job_id: str,
        asset_id: str,
        artifact_type: str,
        data: bytes,
        media_type: str,
        worker_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a detection result image under an active job lease."""

        self._ensure_initialized()
        if artifact_type != "bbox-image":
            raise ValueError(
                f"unsupported job artifact_type: {artifact_type}"
            )
        if media_type != "image/png":
            raise ValueError("bbox-image artifacts must use image/png")
        normalized_worker_id = self._validate_worker_id(worker_id)
        validated = validate_image_bytes(
            data,
            max_image_bytes=max(1, len(data)),
            max_image_pixels=(2**63) - 1,
        )
        if validated.image_format != "png":
            raise ValueError("bbox-image data must be encoded as PNG")
        metadata_payload = metadata or {}
        canonical_json(metadata_payload)
        artifact_id = _new_id("jart")
        relative = (
            Path("overlays")
            / "jobs"
            / job_id
            / asset_id
            / f"{artifact_id}.png"
        )
        self._atomic_write(relative, data)
        previous_path: str | None = None
        now = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._assert_active_job_lease(
                    connection,
                    job_id=job_id,
                    worker_id=normalized_worker_id,
                    now=now,
                )
                membership = connection.execute(
                    """
                    SELECT 1 FROM job_assets
                    WHERE job_id = ? AND asset_id = ?
                    """,
                    (job_id, asset_id),
                ).fetchone()
                if membership is None:
                    raise ResourceNotFoundError(
                        "asset is not part of the annotation job"
                    )
                previous = connection.execute(
                    """
                    SELECT file_path FROM job_artifacts
                    WHERE job_id = ? AND asset_id = ?
                      AND artifact_type = ?
                    """,
                    (job_id, asset_id, artifact_type),
                ).fetchone()
                if previous is not None:
                    previous_path = previous["file_path"]
                    connection.execute(
                        """
                        DELETE FROM job_artifacts
                        WHERE job_id = ? AND asset_id = ?
                          AND artifact_type = ?
                        """,
                        (job_id, asset_id, artifact_type),
                    )
                connection.execute(
                    """
                    INSERT INTO job_artifacts (
                        artifact_id, job_id, asset_id, artifact_type,
                        file_path, media_type, sha256, size_bytes,
                        width, height, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        job_id,
                        asset_id,
                        artifact_type,
                        relative.as_posix(),
                        media_type,
                        sha256_bytes(data),
                        len(data),
                        validated.width,
                        validated.height,
                        canonical_json(metadata_payload),
                        now,
                    ),
                )
                connection.execute("COMMIT")
        except (
            ResourceNotFoundError,
            VersionConflictError,
            ValueError,
        ):
            self._resolve_relative(relative).unlink(missing_ok=True)
            raise
        except Exception as exc:
            self._resolve_relative(relative).unlink(missing_ok=True)
            raise StorageError("failed to store job artifact") from exc
        if previous_path and previous_path != relative.as_posix():
            self._resolve_relative(previous_path).unlink(missing_ok=True)
        return {
            "artifact_id": artifact_id,
            "job_id": job_id,
            "asset_id": asset_id,
            "artifact_type": artifact_type,
            "media_type": media_type,
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
            "width": validated.width,
            "height": validated.height,
            "metadata": metadata_payload,
            "url": (
                f"/v1/annotation/jobs/{job_id}/assets/"
                f"{asset_id}/bbox-image"
            ),
        }

    def job_artifact_file(
        self,
        *,
        job_id: str,
        asset_id: str,
        artifact_type: str,
    ) -> tuple[Path, str, str]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_path, media_type, sha256
                FROM job_artifacts
                WHERE job_id = ? AND asset_id = ?
                  AND artifact_type = ?
                """,
                (job_id, asset_id, artifact_type),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("job artifact was not found")
        path = self._resolve_relative(row["file_path"])
        if not path.is_file():
            raise StorageError("job artifact file is missing")
        return path, row["media_type"], row["sha256"]

    # --------------------------------------------------------------- artifacts
    def store_artifact(
        self,
        *,
        task_id: str,
        artifact_type: str,
        data: bytes,
        media_type: str,
        width: int | None = None,
        height: int | None = None,
        operation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        roots = {
            "detections": "overlays",
            "mask": "masks",
            "mask-overlay": "overlays",
            "crop": "crops",
        }
        if artifact_type not in roots:
            raise ValueError(f"unsupported artifact_type: {artifact_type}")
        if media_type not in {"image/jpeg", "image/png"}:
            raise ValueError("artifact media_type must be image/jpeg or image/png")
        if artifact_type == "mask" and media_type != "image/png":
            raise ValueError("mask artifacts must use image/png")
        if (width is None) != (height is None):
            raise ValueError("artifact width and height must be provided together")
        if width is not None and (width < 1 or height is None or height < 1):
            raise ValueError("artifact dimensions must be positive")
        validated = validate_image_bytes(
            data,
            max_image_bytes=max(1, len(data)),
            max_image_pixels=(2**63) - 1,
        )
        expected_media_type = (
            "image/png"
            if validated.image_format == "png"
            else "image/jpeg"
        )
        if media_type != expected_media_type:
            raise ValueError(
                "artifact media_type does not match the encoded image"
            )
        if width is not None and (
            width != validated.width or height != validated.height
        ):
            raise ValueError(
                "artifact dimensions do not match the encoded image"
            )
        width = validated.width
        height = validated.height
        extension = "jpg" if media_type == "image/jpeg" else "png"
        artifact_id = _new_id("art")
        relative = (
            Path(roots[artifact_type])
            / task_id
            / f"{artifact_id}.{extension}"
        )
        self._atomic_write(relative, data)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id, task_id, artifact_type, operation_id,
                        file_path, media_type, sha256, size_bytes,
                        width, height, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        task_id,
                        artifact_type,
                        operation_id,
                        relative.as_posix(),
                        media_type,
                        sha256_bytes(data),
                        len(data),
                        width,
                        height,
                        canonical_json(metadata or {}),
                        utc_now(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            self._resolve_relative(relative).unlink(missing_ok=True)
            raise ResourceNotFoundError("artifact task was not found") from exc
        except Exception as exc:
            self._resolve_relative(relative).unlink(missing_ok=True)
            raise StorageError("failed to register task artifact") from exc
        return {
            "artifact_id": artifact_id,
            "task_id": task_id,
            "artifact_type": artifact_type,
            "operation_id": operation_id,
            "media_type": media_type,
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
            "width": width,
            "height": height,
            "metadata": metadata or {},
            "url": (
                f"/v1/annotation/tasks/{task_id}/artifacts/{artifact_type}"
            ),
        }

    def artifact_file(
        self,
        task_id: str,
        artifact_type: str,
    ) -> tuple[Path, str]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_path, media_type FROM artifacts
                WHERE task_id = ? AND artifact_type = ?
                ORDER BY created_at DESC, artifact_id DESC LIMIT 1
                """,
                (task_id, artifact_type),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError("task artifact was not found")
        path = self._resolve_relative(row["file_path"])
        if not path.is_file():
            raise StorageError("task artifact file is missing")
        return path, row["media_type"]

    # ---------------------------------------------------------------- releases
    def create_release(
        self,
        *,
        name: str,
        task_filter: dict[str, Any],
        split_policy: dict[str, Any],
        idempotency_key: str | None = None,
        idempotency_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        request = CreateReleaseRequest(
            name=name,
            task_filter=task_filter,
            split_policy=split_policy,
        )
        payload = _model_dict(request)
        release_id = _new_id("rel")
        created_at = utc_now()
        request_payload = idempotency_request or payload
        request_digest = sha256_bytes(
            canonical_json(request_payload).encode("utf-8")
        )
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if idempotency_key is not None:
                    idempotent = connection.execute(
                        """
                        SELECT * FROM idempotency_keys
                        WHERE scope = 'create-release'
                          AND idempotency_key = ?
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if idempotent is not None:
                        if idempotent["request_sha256"] != request_digest:
                            raise IdempotencyConflictError(
                                "idempotency key was reused with a "
                                "different request"
                            )
                        response = _json_loads(
                            idempotent["response_json"],
                            {},
                        )
                        connection.execute("COMMIT")
                        return response
                connection.execute(
                    """
                    INSERT INTO releases (
                        release_id, name, status, task_filter_json,
                        split_policy_json, created_at
                    ) VALUES (?, ?, 'queued', ?, ?, ?)
                    """,
                    (
                        release_id,
                        name,
                        canonical_json(payload["task_filter"]),
                        canonical_json(payload["split_policy"]),
                        created_at,
                    ),
                )
                response = {
                    "release_id": release_id,
                    "name": name,
                    "status": ReleaseStatus.QUEUED.value,
                    "task_filter": payload["task_filter"],
                    "split_policy": payload["split_policy"],
                    "counts": None,
                    "manifest_url": None,
                    "archive_url": None,
                    "error": None,
                    "created_at": created_at,
                    "completed_at": None,
                    "claimed_by": None,
                    "claim_token": None,
                    "lease_expires_at": None,
                    "heartbeat_at": None,
                    "started_at": None,
                    "attempt_count": 0,
                }
                if idempotency_key is not None:
                    connection.execute(
                        """
                        INSERT INTO idempotency_keys (
                            scope, idempotency_key, request_sha256,
                            resource_type, resource_id, response_json,
                            created_at
                        ) VALUES (
                            'create-release', ?, ?, 'release', ?, ?, ?
                        )
                        """,
                        (
                            idempotency_key,
                            request_digest,
                            release_id,
                            canonical_json(response),
                            created_at,
                        ),
                    )
                connection.execute("COMMIT")
                return response
        except IdempotencyConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise IdempotencyConflictError(
                "release name already exists"
            ) from exc

    def get_release(self, release_id: str) -> dict[str, Any]:
        self._ensure_initialized()
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                "SELECT * FROM releases WHERE release_id = ?",
                (release_id,),
                "annotation release",
            )
        return self._release_payload(row)

    def _release_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        release_id = row["release_id"]
        return {
            "release_id": release_id,
            "name": row["name"],
            "status": row["status"],
            "task_filter": _json_loads(row["task_filter_json"], {}),
            "split_policy": _json_loads(row["split_policy_json"], {}),
            "counts": _json_loads(row["counts_json"], None),
            "manifest_url": (
                f"/v1/annotation/releases/{release_id}/manifest"
                if row["manifest_path"]
                else None
            ),
            "archive_url": (
                f"/v1/annotation/releases/{release_id}/archive"
                if row["archive_path"]
                else None
            ),
            "error": row["error"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "claimed_by": row["claimed_by"],
            "claim_token": row["claim_token"],
            "lease_expires_at": row["lease_expires_at"],
            "heartbeat_at": row["heartbeat_at"],
            "started_at": row["started_at"],
            "attempt_count": row["attempt_count"],
        }

    def claim_next_release(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        self._ensure_initialized()
        normalized_worker_id = self._validate_worker_id(worker_id)
        if lease_seconds < 1 or lease_seconds > 86_400:
            raise ValueError(
                "lease_seconds must be between 1 and 86400"
            )
        now = datetime.now(timezone.utc)
        now_value = now.isoformat()
        lease_expires_at = (
            now + timedelta(seconds=lease_seconds)
        ).isoformat()
        claim_token = uuid.uuid4().hex
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM releases
                WHERE status = 'queued'
                   OR (
                       status = 'building'
                       AND (
                           claim_token IS NULL
                           OR lease_expires_at IS NULL
                           OR lease_expires_at <= ?
                       )
                   )
                ORDER BY
                    CASE status WHEN 'building' THEN 0 ELSE 1 END,
                    created_at,
                    release_id
                LIMIT 1
                """,
                (now_value,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                """
                UPDATE releases
                SET status = 'building', claimed_by = ?, claim_token = ?,
                    lease_expires_at = ?, heartbeat_at = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = NULL, error = NULL,
                    attempt_count = attempt_count + 1
                WHERE release_id = ?
                """,
                (
                    normalized_worker_id,
                    claim_token,
                    lease_expires_at,
                    now_value,
                    now_value,
                    row["release_id"],
                ),
            )
            connection.execute("COMMIT")
        return self.get_release(row["release_id"])

    def heartbeat_release(
        self,
        release_id: str,
        *,
        claim_token: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if not claim_token:
            raise ValueError("claim_token must not be blank")
        if lease_seconds < 1 or lease_seconds > 86_400:
            raise ValueError(
                "lease_seconds must be between 1 and 86400"
            )
        now = datetime.now(timezone.utc)
        now_value = now.isoformat()
        lease_expires_at = (
            now + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE releases
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE release_id = ? AND status = 'building'
                  AND claim_token = ? AND lease_expires_at > ?
                """,
                (
                    now_value,
                    lease_expires_at,
                    release_id,
                    claim_token,
                    now_value,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM releases WHERE release_id = ?",
                    (release_id,),
                ).fetchone()
                if exists is None:
                    raise ResourceNotFoundError(
                        "annotation release was not found"
                    )
                raise VersionConflictError(
                    "annotation release claim is no longer active"
                )
        return self.get_release(release_id)

    def get_release_export_snapshot(
        self,
        release_id: str,
        *,
        claim_token: str,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        now = utc_now()
        with self._connect() as connection:
            release = self._row_or_not_found(
                connection,
                "SELECT * FROM releases WHERE release_id = ?",
                (release_id,),
                "annotation release",
            )
            if (
                release["status"] != ReleaseStatus.BUILDING.value
                or release["claim_token"] != claim_token
                or release["lease_expires_at"] is None
                or release["lease_expires_at"] <= now
            ):
                raise VersionConflictError(
                    "annotation release claim is no longer active"
                )
            task_filter = _json_loads(
                release["task_filter_json"],
                {},
            )
            categories = task_filter.get("categories")
            query = """
                SELECT
                    t.task_id, t.job_id, t.asset_id, t.category,
                    t.status, t.version, t.annotation_json,
                    t.provenance_json, t.primary_result,
                    t.annotator_id, t.reviewer_id,
                    t.created_at, t.updated_at,
                    a.group_id, a.image_path, a.media_type,
                    a.width, a.height, a.sha256 AS image_sha256
                FROM annotation_tasks t
                JOIN assets a ON a.asset_id = t.asset_id
                WHERE t.status = 'accepted'
            """
            parameters: list[Any] = []
            if categories:
                placeholders = ",".join("?" for _ in categories)
                query += f" AND t.category IN ({placeholders})"
                parameters.extend(categories)
            query += " ORDER BY t.task_id"
            rows = connection.execute(query, parameters).fetchall()
            snapshots = []
            for row in rows:
                reviews = [
                    dict(item)
                    for item in connection.execute(
                        """
                        SELECT * FROM reviews
                        WHERE task_id = ? ORDER BY created_at, review_id
                        """,
                        (row["task_id"],),
                    ).fetchall()
                ]
                snapshots.append(
                    {
                        **dict(row),
                        "annotation": _json_loads(
                            row["annotation_json"],
                            {},
                        ),
                        "provenance": _json_loads(
                            row["provenance_json"],
                            {},
                        ),
                        "image_path": self._resolve_relative(
                            row["image_path"]
                        ),
                        "reviews": reviews,
                    }
                )
        return snapshots

    def fail_release(
        self,
        release_id: str,
        *,
        claim_token: str,
        error: str,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        safe_error = error.strip()[:2000] or "release build failed"
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE releases
                SET status = 'failed', error = ?, completed_at = ?,
                    claimed_by = NULL, claim_token = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL
                WHERE release_id = ? AND status = 'building'
                  AND claim_token = ?
                """,
                (
                    safe_error,
                    utc_now(),
                    release_id,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError(
                    "annotation release claim is no longer active"
                )
        return self.get_release(release_id)

    def complete_release_from_files(
        self,
        release_id: str,
        *,
        claim_token: str,
        manifest_path: Path,
        archive_path: Path,
        counts: dict[str, int],
    ) -> dict[str, Any]:
        self._ensure_initialized()
        required_counts = {"train", "val", "golden"}
        if set(counts) != required_counts or any(
            not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError(
                "release counts must contain non-negative train, val, golden"
            )
        try:
            manifest_value = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "release manifest must be a UTF-8 JSON object"
            ) from exc
        if not isinstance(manifest_value, dict):
            raise ValueError("release manifest must contain a JSON object")

        unique_root = Path("exports") / release_id / claim_token
        manifest_relative = unique_root / "manifest.json"
        archive_relative = unique_root / "reasonseg.zip"
        published_manifest = self._atomic_copy(
            manifest_path,
            manifest_relative,
        )
        try:
            published_archive = self._atomic_copy(
                archive_path,
                archive_relative,
            )
        except Exception:
            published_manifest.unlink(missing_ok=True)
            raise
        try:
            with self._lock, self._connect() as connection:
                now = utc_now()
                cursor = connection.execute(
                    """
                    UPDATE releases
                    SET status = 'succeeded', counts_json = ?,
                        manifest_path = ?, manifest_sha256 = ?,
                        archive_path = ?, archive_sha256 = ?,
                        error = NULL, completed_at = ?,
                        claimed_by = NULL, claim_token = NULL,
                        lease_expires_at = NULL, heartbeat_at = NULL
                    WHERE release_id = ? AND status = 'building'
                      AND claim_token = ? AND lease_expires_at > ?
                    """,
                    (
                        canonical_json(counts),
                        manifest_relative.as_posix(),
                        sha256_file(published_manifest),
                        archive_relative.as_posix(),
                        sha256_file(published_archive),
                        now,
                        release_id,
                        claim_token,
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    raise VersionConflictError(
                        "annotation release claim is no longer active"
                    )
        except Exception:
            published_manifest.unlink(missing_ok=True)
            published_archive.unlink(missing_ok=True)
            raise
        return self.get_release(release_id)

    def transition_release(
        self,
        release_id: str,
        *,
        expected_status: str | ReleaseStatus,
        status: str | ReleaseStatus,
        error: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        expected = ReleaseStatus(_enum_value(expected_status))
        target = ReleaseStatus(_enum_value(status))
        ensure_release_transition(expected, target)
        completed_at = utc_now() if target == ReleaseStatus.FAILED else None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE releases SET status = ?, error = ?, completed_at = ?
                WHERE release_id = ? AND status = ?
                """,
                (
                    target.value,
                    error,
                    completed_at,
                    release_id,
                    expected.value,
                ),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT status FROM releases WHERE release_id = ?",
                    (release_id,),
                ).fetchone()
                if row is None:
                    raise ResourceNotFoundError("annotation release was not found")
                raise VersionConflictError(
                    f"expected release status {expected.value} but current "
                    f"status is {row['status']}"
                )
        return self.get_release(release_id)

    def complete_release(
        self,
        release_id: str,
        *,
        manifest: bytes,
        archive: bytes,
        counts: dict[str, int],
    ) -> dict[str, Any]:
        self._ensure_initialized()
        try:
            manifest_value = json.loads(manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("release manifest must be a UTF-8 JSON document") from exc
        if not isinstance(manifest_value, dict):
            raise ValueError("release manifest must contain a JSON object")
        required_counts = {"train", "val", "golden"}
        if set(counts) != required_counts or any(
            not isinstance(value, int) or value < 0 for value in counts.values()
        ):
            raise ValueError("release counts must contain non-negative train, val, golden")

        release = self.get_release(release_id)
        if release["status"] != ReleaseStatus.BUILDING.value:
            raise InvalidStateTransitionError(
                "release outputs can only be stored while building"
            )
        manifest_relative = Path("exports") / release_id / "manifest.json"
        archive_relative = Path("exports") / release_id / "reasonseg.zip"
        self._atomic_write(manifest_relative, manifest)
        self._atomic_write(archive_relative, archive)
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE releases
                    SET status = 'succeeded', counts_json = ?,
                        manifest_path = ?, manifest_sha256 = ?,
                        archive_path = ?, archive_sha256 = ?,
                        error = NULL, completed_at = ?
                    WHERE release_id = ? AND status = 'building'
                    """,
                    (
                        canonical_json(counts),
                        manifest_relative.as_posix(),
                        sha256_bytes(manifest),
                        archive_relative.as_posix(),
                        sha256_bytes(archive),
                        utc_now(),
                        release_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise VersionConflictError("annotation release status has changed")
        except Exception:
            self._resolve_relative(manifest_relative).unlink(missing_ok=True)
            self._resolve_relative(archive_relative).unlink(missing_ok=True)
            raise
        return self.get_release(release_id)

    def release_file(self, release_id: str, kind: str) -> tuple[Path, str]:
        if kind not in {"manifest", "archive"}:
            raise ValueError("release file kind must be manifest or archive")
        self._ensure_initialized()
        column = "manifest_path" if kind == "manifest" else "archive_path"
        with self._connect() as connection:
            row = self._row_or_not_found(
                connection,
                f"SELECT status, {column} FROM releases WHERE release_id = ?",
                (release_id,),
                "annotation release",
            )
        if row["status"] != ReleaseStatus.SUCCEEDED.value or not row[column]:
            raise InvalidStateTransitionError("annotation release is not ready")
        path = self._resolve_relative(row[column])
        if not path.is_file():
            raise StorageError("annotation release file is missing")
        media_type = "application/json" if kind == "manifest" else "application/zip"
        return path, media_type

    # ------------------------------------------------------------- idempotency
    def find_idempotency(
        self,
        *,
        scope: str,
        key: str,
        request_payload: Any,
    ) -> dict[str, Any] | None:
        self._ensure_initialized()
        digest = sha256_bytes(canonical_json(request_payload).encode("utf-8"))
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM idempotency_keys
                WHERE scope = ? AND idempotency_key = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (scope, key, utc_now()),
            ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != digest:
            raise IdempotencyConflictError(
                "idempotency key was reused with a different request"
            )
        return {
            "resource_type": row["resource_type"],
            "resource_id": row["resource_id"],
            "response": _json_loads(row["response_json"], {}),
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
        }

    def save_idempotency(
        self,
        *,
        scope: str,
        key: str,
        request_payload: Any,
        resource_type: str,
        resource_id: str,
        response: dict[str, Any],
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        digest = sha256_bytes(canonical_json(request_payload).encode("utf-8"))
        created_at = utc_now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    DELETE FROM idempotency_keys
                    WHERE scope = ? AND idempotency_key = ?
                      AND expires_at IS NOT NULL AND expires_at <= ?
                    """,
                    (scope, key, created_at),
                )
                connection.execute(
                    """
                    INSERT INTO idempotency_keys (
                        scope, idempotency_key, request_sha256,
                        resource_type, resource_id, response_json,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope,
                        key,
                        digest,
                        resource_type,
                        resource_id,
                        canonical_json(response),
                        created_at,
                        expires_at,
                    ),
                )
                connection.execute("COMMIT")
        except sqlite3.IntegrityError:
            existing = self.find_idempotency(
                scope=scope,
                key=key,
                request_payload=request_payload,
            )
            if existing is None:
                raise IdempotencyConflictError("idempotency key is unavailable")
            return existing
        return {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "response": response,
            "created_at": created_at,
            "expires_at": expires_at,
        }
