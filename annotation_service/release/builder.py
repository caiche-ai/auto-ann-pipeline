from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import threading
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..api.errors import ServiceError, VersionConflictError
from ..storage.repository import AnnotationStore, sha256_file


SPLITS = ("train", "val", "golden")
BUILDER_VERSION = "reasonseg-release-v2"
LOGGER = logging.getLogger(__name__)
UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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
        exported.append({"label": label, "points": points})
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


def _write_grounding_dino_boxes(
    source: Path,
    destination: Path,
    detections: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) / 300))
    for detection in detections:
        box = detection.get("box_xyxy") or []
        if len(box) != 4:
            continue
        coordinates = tuple(float(value) for value in box)
        draw.rectangle(coordinates, outline=(255, 40, 40), width=line_width)
        label = str(detection.get("entity") or "target")
        draw.text(
            (coordinates[0] + line_width, coordinates[1] + line_width),
            label,
            fill=(255, 40, 40),
        )
    rendered.save(destination, format="PNG")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_process_artifact(
    artifact: dict[str, Any],
    destination: Path,
) -> None:
    source = Path(artifact["path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_sha256 = str(artifact["sha256"])
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"release process artifact checksum changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _artifact_manifest_record(
    artifact: dict[str, Any],
    *,
    file_path: str,
) -> dict[str, Any]:
    return {
        "file": file_path,
        "artifact_id": artifact["artifact_id"],
        "operation_id": artifact.get("operation_id"),
        "media_type": artifact["media_type"],
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
        "width": artifact["width"],
        "height": artifact["height"],
        "metadata": artifact["metadata"],
        "created_at": artifact["created_at"],
    }


def _source_filename(snapshot: dict[str, Any]) -> str:
    metadata = snapshot.get("asset_metadata") or {}
    return str(
        metadata.get("original_filename")
        or metadata.get("filename")
        or snapshot.get("asset_source_id")
        or f"image-{snapshot['image_sha256'][:12]}"
    )


def _filename_prefix(source_filename: str) -> str:
    basename = source_filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(basename).stem
    normalized = unicodedata.normalize("NFC", stem)
    normalized = UNSAFE_FILENAME.sub("_", normalized).strip(". ")
    if not normalized:
        normalized = "image"
    return normalized[:120].rstrip(". ") or "image"


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


def write_sample_directory(
    *,
    snapshot: dict[str, Any],
    sample_root: Path,
    prefix: str,
    require_process_artifacts: bool = False,
) -> list[Path]:
    """Write one submitted sample in the user-facing six-file layout."""

    annotation = snapshot["annotation"]
    process_artifacts = snapshot.get("process_artifacts", {})
    required = ("sam_mask", "sam_overlay")
    missing = [name for name in required if name not in process_artifacts]
    if require_process_artifacts and missing:
        raise ValueError(
            "submitted sample is missing required process artifacts: "
            + ", ".join(missing)
        )

    image_path = sample_root / f"{prefix}.jpg"
    lisa_path = sample_root / f"{prefix}_lisa.json"
    grounding_path = sample_root / f"{prefix}_grounding-dino.json"
    bbox_path = sample_root / f"{prefix}_grounding-dino-boxes.png"
    mask_path = sample_root / f"{prefix}_sam-mask.png"
    overlay_path = sample_root / f"{prefix}_sam-overlay.png"
    _write_jpeg(Path(snapshot["image_path"]), image_path)
    _write_json(
        lisa_path,
        {
            "shapes": _export_shapes(
                annotation["shapes"],
                width=int(snapshot["width"]),
                height=int(snapshot["height"]),
            ),
            "text": [item["text"] for item in annotation["prompts"]],
            "is_sentence": True,
            "source": {
                "sample_id": snapshot["task_id"],
                "sample_key": snapshot["category"],
                "group_id": snapshot["group_id"],
            },
        },
    )
    provenance = snapshot["provenance"]
    selected_detection_ids = provenance.get("source_detection_ids") or []
    if not selected_detection_ids and provenance.get("source_detection_id"):
        selected_detection_ids = [provenance["source_detection_id"]]
    _write_json(
        grounding_path,
        {
            "schema_version": 1,
            "task_id": snapshot["task_id"],
            "job_id": snapshot["job_id"],
            "asset_id": snapshot["asset_id"],
            "model": provenance.get("grounding_dino_version"),
            "prompt_version": provenance.get(
                "grounding_dino_prompt_version"
            ),
            "prompt": provenance.get("grounding_prompt"),
            "thresholds": provenance.get("grounding_dino_thresholds"),
            "selected_detection_ids": selected_detection_ids,
            "detections": snapshot.get("grounding_dino_detections", []),
        },
    )
    artifact_destinations = {
        "grounding_dino_bbox_image": bbox_path,
        "sam_mask": mask_path,
        "sam_overlay": overlay_path,
    }
    written = [image_path, lisa_path, grounding_path]
    if "grounding_dino_bbox_image" not in process_artifacts:
        _write_grounding_dino_boxes(
            Path(snapshot["image_path"]),
            bbox_path,
            snapshot.get("grounding_dino_detections", []),
        )
        written.append(bbox_path)
    for name, destination in artifact_destinations.items():
        artifact = process_artifacts.get(name)
        if artifact is None:
            continue
        _copy_process_artifact(artifact, destination)
        written.append(destination)
    return written


def build_release_files(
    *,
    release: dict[str, Any],
    snapshots: list[dict[str, Any]],
    output_root: Path,
) -> ReleaseBuildFiles:
    if not snapshots:
        raise ValueError("release contains no accepted tasks")
    dataset_root = output_root / release["name"]
    dataset_root.mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in SPLITS}
    process_counts = {
        "grounding_dino_detections": 0,
        "grounding_dino_bbox_images": 0,
        "sam_masks": 0,
        "sam_overlays": 0,
    }
    manifest_items = []
    jsonl_items = []
    member_hashes: dict[str, str] = {}
    prefix_counts: dict[str, int] = {}
    for snapshot in snapshots:
        annotation = snapshot["annotation"]
        split = split_for_group(
            snapshot["group_id"],
            release["split_policy"],
        )
        counts[split] += 1
        source_filename = _source_filename(snapshot)
        base_prefix = _filename_prefix(source_filename)
        prefix_key = base_prefix.casefold()
        prefix_counts[prefix_key] = prefix_counts.get(prefix_key, 0) + 1
        occurrence = prefix_counts[prefix_key]
        prefix = base_prefix if occurrence == 1 else f"{base_prefix}_{occurrence}"
        sample_root = dataset_root / prefix
        image_path = sample_root / f"{prefix}.jpg"
        json_path = sample_root / f"{prefix}_lisa.json"
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
        _write_json(json_path, reasonseg)

        provenance = snapshot["provenance"]
        selected_detection_ids = provenance.get("source_detection_ids") or []
        if not selected_detection_ids and provenance.get("source_detection_id"):
            selected_detection_ids = [provenance["source_detection_id"]]
        detections = snapshot.get("grounding_dino_detections", [])
        grounding_dino_path = sample_root / f"{prefix}_grounding-dino.json"
        _write_json(
            grounding_dino_path,
            {
                "schema_version": 1,
                "task_id": snapshot["task_id"],
                "job_id": snapshot["job_id"],
                "asset_id": snapshot["asset_id"],
                "model": provenance.get("grounding_dino_version"),
                "prompt_version": provenance.get(
                    "grounding_dino_prompt_version"
                ),
                "prompt": provenance.get("grounding_prompt"),
                "thresholds": provenance.get("grounding_dino_thresholds"),
                "selected_detection_ids": selected_detection_ids,
                "detections": detections,
            },
        )
        process_counts["grounding_dino_detections"] += len(detections)

        process_files: dict[str, Any] = {
            "grounding_dino": {
                "detections": grounding_dino_path.relative_to(
                    dataset_root
                ).as_posix(),
                "detection_count": len(detections),
                "selected_detection_ids": selected_detection_ids,
                "bbox_image": None,
            },
            "sam": {"mask": None, "overlay": None},
        }
        artifact_destinations = {
            "grounding_dino_bbox_image": (
                "grounding_dino",
                "bbox_image",
                sample_root / f"{prefix}_grounding-dino-boxes.png",
            ),
            "sam_mask": (
                "sam",
                "mask",
                sample_root / f"{prefix}_sam-mask.png",
            ),
            "sam_overlay": (
                "sam",
                "overlay",
                sample_root / f"{prefix}_sam-overlay.png",
            ),
        }
        process_artifacts = snapshot.get("process_artifacts", {})
        for artifact_name, (
            process_name,
            record_name,
            destination,
        ) in artifact_destinations.items():
            artifact = process_artifacts.get(artifact_name)
            if artifact is None:
                continue
            _copy_process_artifact(artifact, destination)
            relative = destination.relative_to(dataset_root).as_posix()
            process_files[process_name][record_name] = (
                _artifact_manifest_record(artifact, file_path=relative)
            )
            if artifact_name == "grounding_dino_bbox_image":
                process_counts["grounding_dino_bbox_images"] += 1
            elif artifact_name == "sam_mask":
                process_counts["sam_masks"] += 1
            elif artifact_name == "sam_overlay":
                process_counts["sam_overlays"] += 1

        image_relative = image_path.relative_to(dataset_root).as_posix()
        json_relative = json_path.relative_to(dataset_root).as_posix()
        image_digest = sha256_file(image_path)
        json_digest = sha256_file(json_path)
        member_hashes[image_relative] = image_digest
        member_hashes[json_relative] = json_digest
        member_hashes[
            grounding_dino_path.relative_to(dataset_root).as_posix()
        ] = sha256_file(grounding_dino_path)
        for artifact_group in process_files.values():
            for record in artifact_group.values():
                if not isinstance(record, dict) or "file" not in record:
                    continue
                member_hashes[record["file"]] = record["sha256"]
        manifest_items.append(
            {
                "task_id": snapshot["task_id"],
                "task_version": snapshot["version"],
                "asset_id": snapshot["asset_id"],
                "source_filename": source_filename,
                "file_prefix": prefix,
                "category": snapshot["category"],
                "group_id": snapshot["group_id"],
                "image": image_relative,
                "annotation": json_relative,
                "image_sha256": image_digest,
                "annotation_sha256": json_digest,
                "process": process_files,
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
        "process_counts": process_counts,
    }
    card_path = dataset_root / "dataset_card.md"
    card_path.write_text(
        f"# {release['name']}\n\n"
        "ReasonSeg construction-safety annotation release.\n\n"
        f"- Builder: `{BUILDER_VERSION}`\n"
        f"- Tasks: {len(snapshots)}\n"
        f"- GroundingDINO detections: "
        f"{process_counts['grounding_dino_detections']}\n"
        f"- SAM masks: {process_counts['sam_masks']}\n",
        encoding="utf-8",
    )
    for path in (jsonl_path, card_path):
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
