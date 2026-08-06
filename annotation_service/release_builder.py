from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import ServiceError, VersionConflictError
from .storage import AnnotationStore, sha256_file
from .validation import validate_annotation_for_submission


SPLITS = ("train", "val", "golden")
BUILDER_VERSION = "reasonseg-release-v1"
LOGGER = logging.getLogger(__name__)


def split_for_group(group_id: str, policy: dict[str, Any]) -> str:
    seed = int(policy["seed"])
    digest = hashlib.sha256(
        f"{seed}\0{group_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest, "big") / (1 << 256)
    train_cutoff = float(policy["train_ratio"])
    val_cutoff = train_cutoff + float(policy["val_ratio"])
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "golden"


def _polygon_area(points: list[list[int]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def _export_shapes(
    shapes: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    exported = []
    has_target = False
    for shape in shapes:
        label = shape["label"]
        if label not in {"target", "ignore"}:
            raise ValueError("release shape label must be target or ignore")
        points = [
            [
                min(max(int(round(float(point[0]))), 0), width - 1),
                min(max(int(round(float(point[1]))), 0), height - 1),
            ]
            for point in shape["points"]
        ]
        if len(points) < 3 or _polygon_area(points) <= 0:
            raise ValueError(
                "release polygon collapses after integer conversion"
            )
        has_target = has_target or label == "target"
        exported.append({"label": label, "points": points})
    if not has_target:
        raise ValueError("release sample has no target polygon")
    return exported


def _write_jpeg(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.load()
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        image.save(
            destination,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )


def _write_deterministic_zip(source_root: Path, archive_path: Path) -> None:
    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
    )
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in files:
            relative = path.relative_to(source_root.parent).as_posix()
            info = zipfile.ZipInfo(relative)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


@dataclass(frozen=True)
class ReleaseBuildFiles:
    manifest_path: Path
    archive_path: Path
    counts: dict[str, int]


def build_release_files(
    *,
    release: dict[str, Any],
    snapshots: list[dict[str, Any]],
    output_root: Path,
) -> ReleaseBuildFiles:
    if not snapshots:
        raise ValueError("release contains no accepted tasks")
    dataset_root = output_root / release["name"]
    for split in SPLITS:
        (dataset_root / split).mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in SPLITS}
    manifest_items = []
    jsonl_items = []
    member_hashes: dict[str, str] = {}
    for snapshot in snapshots:
        annotation = snapshot["annotation"]
        validate_annotation_for_submission(
            annotation,
            width=int(snapshot["width"]),
            height=int(snapshot["height"]),
            category=snapshot["category"],
        )
        split = split_for_group(
            snapshot["group_id"],
            release["split_policy"],
        )
        counts[split] += 1
        stem = f"{snapshot['task_id']}__{snapshot['category']}"
        image_path = dataset_root / split / f"{stem}.jpg"
        json_path = dataset_root / split / f"{stem}.json"
        _write_jpeg(snapshot["image_path"], image_path)
        prompts = [item["text"] for item in annotation["prompts"]]
        reasonseg = {
            "shapes": _export_shapes(
                annotation["shapes"],
                width=int(snapshot["width"]),
                height=int(snapshot["height"]),
            ),
            "text": prompts,
            "is_sentence": True,
            "source": {
                "sample_id": snapshot["task_id"],
                "sample_key": snapshot["category"],
                "group_id": snapshot["group_id"],
            },
        }
        json_path.write_text(
            json.dumps(
                reasonseg,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        image_relative = image_path.relative_to(dataset_root).as_posix()
        json_relative = json_path.relative_to(dataset_root).as_posix()
        image_digest = sha256_file(image_path)
        json_digest = sha256_file(json_path)
        member_hashes[image_relative] = image_digest
        member_hashes[json_relative] = json_digest
        manifest_items.append(
            {
                "task_id": snapshot["task_id"],
                "task_version": snapshot["version"],
                "asset_id": snapshot["asset_id"],
                "category": snapshot["category"],
                "group_id": snapshot["group_id"],
                "split": split,
                "image": image_relative,
                "annotation": json_relative,
                "image_sha256": image_digest,
                "annotation_sha256": json_digest,
            }
        )
        jsonl_items.append(
            {
                **manifest_items[-1],
                "prompt_records": annotation["prompts"],
                "provenance": snapshot["provenance"],
                "primary_result": snapshot["primary_result"],
                "annotator_id": snapshot["annotator_id"],
                "reviewer_id": snapshot["reviewer_id"],
                "reviews": snapshot["reviews"],
                "source_image_sha256": snapshot["image_sha256"],
            }
        )

    jsonl_path = dataset_root / "annotation_manifest.jsonl"
    jsonl_path.write_text(
        "".join(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in jsonl_items
        ),
        encoding="utf-8",
    )
    summary = {
        "builder_version": BUILDER_VERSION,
        "release_id": release["release_id"],
        "release_name": release["name"],
        "counts": counts,
        "split_policy": release["split_policy"],
        "task_count": len(snapshots),
    }
    summary_path = dataset_root / "build_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    card_path = dataset_root / "dataset_card.md"
    card_path.write_text(
        f"# {release['name']}\n\n"
        "ReasonSeg construction-safety annotation release.\n\n"
        f"- Builder: `{BUILDER_VERSION}`\n"
        f"- Tasks: {len(snapshots)}\n"
        f"- Train: {counts['train']}\n"
        f"- Val: {counts['val']}\n"
        f"- Golden: {counts['golden']}\n"
        "- Split isolation key: `group_id`\n",
        encoding="utf-8",
    )
    for path in (jsonl_path, summary_path, card_path):
        member_hashes[
            path.relative_to(dataset_root).as_posix()
        ] = sha256_file(path)

    manifest = {
        **summary,
        "samples": manifest_items,
        "member_sha256": dict(sorted(member_hashes.items())),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    archive_path = output_root / "reasonseg.zip"
    _write_deterministic_zip(dataset_root, archive_path)
    return ReleaseBuildFiles(
        manifest_path=manifest_path,
        archive_path=archive_path,
        counts=counts,
    )


class ReleaseLeaseHeartbeat:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        release_id: str,
        claim_token: str,
        lease_seconds: int,
        interval_seconds: int,
    ):
        self.store = store
        self.release_id = release_id
        self.claim_token = claim_token
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"release-heartbeat-{self.release_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.store.heartbeat_release(
                    self.release_id,
                    claim_token=self.claim_token,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:
                self._error = exc
                self._stop.set()

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise VersionConflictError(
                "annotation release heartbeat failed"
            ) from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(
                timeout=max(1, self.interval_seconds + 1)
            )


class ReleaseWorker:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        worker_id: str,
        lease_seconds: int = 300,
        heartbeat_seconds: int = 60,
        poll_seconds: float = 2.0,
    ):
        if heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be less than lease_seconds"
            )
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.store = store
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds

    def run_once(self) -> bool:
        release = self.store.claim_next_release(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if release is None:
            return False
        token = release["claim_token"]
        heartbeat = ReleaseLeaseHeartbeat(
            store=self.store,
            release_id=release["release_id"],
            claim_token=token,
            lease_seconds=self.lease_seconds,
            interval_seconds=self.heartbeat_seconds,
        )
        heartbeat.start()
        try:
            snapshots = self.store.get_release_export_snapshot(
                release["release_id"],
                claim_token=token,
            )
            with tempfile.TemporaryDirectory(
                dir=self.store.tmp_root,
                prefix=f"{release['release_id']}-",
            ) as temporary:
                files = build_release_files(
                    release=release,
                    snapshots=snapshots,
                    output_root=Path(temporary),
                )
                heartbeat.ensure_healthy()
                self.store.complete_release_from_files(
                    release["release_id"],
                    claim_token=token,
                    manifest_path=files.manifest_path,
                    archive_path=files.archive_path,
                    counts=files.counts,
                )
        except VersionConflictError:
            raise
        except Exception as exc:
            if isinstance(exc, ServiceError):
                public_message = exc.message
            elif isinstance(exc, ValueError):
                public_message = str(exc)
            else:
                public_message = "release build failed"
            if isinstance(exc, (ServiceError, ValueError)):
                LOGGER.warning(
                    "annotation release build rejected: %s",
                    public_message,
                    extra={"release_id": release["release_id"]},
                )
            else:
                LOGGER.exception(
                    "annotation release build failed",
                    extra={"release_id": release["release_id"]},
                )
            try:
                self.store.fail_release(
                    release["release_id"],
                    claim_token=token,
                    error=(
                        f"{type(exc).__name__}: {public_message}"
                    ),
                )
            except VersionConflictError:
                LOGGER.warning(
                    "release failure was not persisted because the claim "
                    "is no longer active",
                    extra={"release_id": release["release_id"]},
                )
        finally:
            heartbeat.stop()
        return True

    def run_forever(self) -> None:
        while True:
            processed = self.run_once()
            if not processed:
                threading.Event().wait(self.poll_seconds)
