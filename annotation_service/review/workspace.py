from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from ..api.schemas import TaskStatus

if TYPE_CHECKING:
    from ..storage.repository import AnnotationStore


REVIEW_SNAPSHOT_SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_component(value: str, *, field: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{field} contains unsafe path characters")
    return value


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_review_submission(
    submissions_root: Path,
    payload: dict[str, Any],
    *,
    strict_existing: bool = True,
) -> Path:
    """Persist an immutable annotator submission and update its latest pointer."""

    task_id = _safe_component(str(payload["task_id"]), field="task_id")
    version = int(payload["task_version"])
    if version < 1:
        raise ValueError("task_version must be positive")
    normalized = {
        "schema_version": REVIEW_SNAPSHOT_SCHEMA_VERSION,
        **payload,
        "task_version": version,
    }
    task_root = submissions_root / task_id
    version_path = task_root / f"v{version:08d}.json"
    if version_path.exists():
        existing = json.loads(version_path.read_text(encoding="utf-8"))
        if strict_existing and existing != normalized:
            raise ValueError("review submission version already has other content")
        normalized = existing
    else:
        _atomic_json_write(version_path, normalized)
    _atomic_json_write(task_root / "current.json", normalized)
    return version_path


def load_review_submissions(
    submissions_root: Path,
    *,
    task_ids: Iterable[str] | None = None,
    latest_only: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    if not submissions_root.is_dir():
        return []
    selected = set(task_ids or [])
    items: list[tuple[Path, dict[str, Any]]] = []
    pattern = "*/current.json" if latest_only else "*/v[0-9]*.json"
    for path in sorted(submissions_root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if selected and payload.get("task_id") not in selected:
            continue
        items.append((path, payload))
    return items


def collect_task_export_snapshots(
    store: AnnotationStore,
    *,
    status: str | TaskStatus = TaskStatus.ACCEPTED,
    categories: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect current task data in the shape expected by the LISA builder."""

    status_value = TaskStatus(status).value
    category_filter = set(categories or [])
    selected_ids = list(dict.fromkeys(task_ids or []))
    tasks: list[dict[str, Any]] = []
    if selected_ids:
        for task_id in selected_ids:
            task = store.get_task(task_id)
            if task["status"] != status_value:
                continue
            if category_filter and task["category"] not in category_filter:
                continue
            tasks.append(task)
    else:
        cursor: str | None = None
        while True:
            page = store.list_tasks(
                status=status_value,
                limit=200,
                cursor=cursor,
            )
            for item in page["items"]:
                if category_filter and item["category"] not in category_filter:
                    continue
                tasks.append(store.get_task(item["task_id"]))
            cursor = page.get("next_cursor")
            if cursor is None:
                break

    snapshots: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: item["task_id"]):
        asset_id = task["asset"]["asset_id"]
        asset = store.get_asset(asset_id)
        image_path, _ = store.asset_file(asset_id)
        snapshots.append(
            {
                "task_id": task["task_id"],
                "job_id": task["job_id"],
                "asset_id": asset_id,
                "category": task["category"],
                "status": task["status"],
                "version": task["version"],
                "annotation": task["annotation"],
                "provenance": task["provenance"],
                "primary_result": task["primary_result"],
                "annotator_id": task["annotator_id"],
                "reviewer_id": task["reviewer_id"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"],
                "group_id": task["asset"]["group_id"],
                "image_path": image_path,
                "media_type": asset["media_type"],
                "width": task["asset"]["width"],
                "height": task["asset"]["height"],
                "image_sha256": asset["sha256"],
                "reviews": store.list_reviews(task["task_id"]),
            }
        )
    return snapshots


def backfill_review_submissions(store: AnnotationStore) -> list[Path]:
    """Idempotently materialize historical submit versions in the review area."""

    return [
        persist_review_submission(
            store.submissions_root,
            payload,
            strict_existing=False,
        )
        for payload in store.list_historical_review_submissions()
    ]
