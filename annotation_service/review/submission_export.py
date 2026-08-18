from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..release.builder import (
    _filename_prefix,
    _source_filename,
    write_sample_directory,
)

if TYPE_CHECKING:
    from ..storage.repository import AnnotationStore


REQUIRED_PROCESS_ARTIFACTS = (
    "sam_mask",
    "sam_overlay",
)


def validate_task_process_artifacts(
    store: AnnotationStore,
    task_id: str,
) -> None:
    task = store.get_task(task_id)
    process = store.get_task_process_export_snapshot(task_id)
    artifacts = process.get("process_artifacts", {})
    missing = [name for name in REQUIRED_PROCESS_ARTIFACTS if name not in artifacts]
    if missing:
        raise ValueError(
            "submitted sample is missing required process artifacts: "
            + ", ".join(missing)
        )


def _snapshots_with_prefixes(
    store: AnnotationStore,
) -> list[dict[str, Any]]:
    snapshots = store.list_submission_export_snapshots()
    counts: dict[str, int] = {}
    for snapshot in snapshots:
        source_filename = _source_filename(snapshot)
        base = _filename_prefix(source_filename)
        key = base.casefold()
        counts[key] = counts.get(key, 0) + 1
        occurrence = counts[key]
        snapshot["source_filename"] = source_filename
        snapshot["file_prefix"] = (
            base if occurrence == 1 else f"{base}_{occurrence}"
        )
    return snapshots


def _replace_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.bak"
    replaced = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            replaced = True
        os.replace(source, destination)
    except Exception:
        if replaced and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()


def materialize_submission_snapshot(
    store: AnnotationStore,
    snapshot: dict[str, Any],
) -> tuple[Path, list[Path]]:
    prefix = snapshot["file_prefix"]
    staging_root = store.tmp_root / f"submission-{uuid.uuid4().hex}"
    staging_sample = staging_root / prefix
    destination = store.submissions_root / prefix
    try:
        written = write_sample_directory(
            snapshot=snapshot,
            sample_root=staging_sample,
            prefix=prefix,
            require_process_artifacts=False,
        )
        relative_files = [path.relative_to(staging_sample) for path in written]
        _replace_directory(staging_sample, destination)
        return destination, [destination / path for path in relative_files]
    finally:
        if staging_root.is_dir():
            shutil.rmtree(staging_root)


def materialize_submitted_task(
    store: AnnotationStore,
    task_id: str,
) -> Path:
    for snapshot in _snapshots_with_prefixes(store):
        if snapshot["task_id"] == task_id:
            return materialize_submission_snapshot(store, snapshot)[0]
    raise ValueError("submitted annotation task was not found")


def _in_time_range(
    snapshot: dict[str, Any],
    *,
    start_time: datetime | None,
    end_time: datetime | None,
) -> bool:
    submitted_at = datetime.fromisoformat(snapshot["submitted_at"])
    if submitted_at.tzinfo is None:
        submitted_at = submitted_at.replace(tzinfo=timezone.utc)
    submitted_at = submitted_at.astimezone(timezone.utc)
    if start_time is not None and submitted_at < start_time:
        return False
    if end_time is not None and submitted_at > end_time:
        return False
    return True


def export_submitted_samples(
    store: AnnotationStore,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    task_ids: Iterable[str] | None = None,
) -> tuple[Path, int]:
    selected_task_ids = set(task_ids or [])
    snapshots = [
        snapshot
        for snapshot in _snapshots_with_prefixes(store)
        if (not selected_task_ids or snapshot["task_id"] in selected_task_ids)
        and _in_time_range(
            snapshot,
            start_time=start_time,
            end_time=end_time,
        )
    ]
    if not snapshots:
        raise ValueError("no submitted annotation samples matched the filters")

    store.exports_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = store.exports_root / (
        f"annotation-export-{timestamp}-{uuid.uuid4().hex[:8]}.zip"
    )
    temporary = store.tmp_root / f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for snapshot in snapshots:
                _, files = materialize_submission_snapshot(
                    store,
                    snapshot,
                )
                for path in sorted(files):
                    relative = path.relative_to(store.submissions_root).as_posix()
                    info = zipfile.ZipInfo(relative)
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, path.read_bytes())
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return archive_path, len(snapshots)
