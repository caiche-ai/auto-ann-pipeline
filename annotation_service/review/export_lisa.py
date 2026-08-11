from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

from ..release.builder import ReleaseBuildFiles, build_release_files
from ..storage.repository import AnnotationStore
from .workspace import collect_task_export_snapshots


_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def export_lisa_dataset(
    *,
    store: AnnotationStore,
    output_root: Path,
    dataset_name: str,
    categories: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    golden_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> ReleaseBuildFiles:
    if not _DATASET_NAME.fullmatch(dataset_name):
        raise ValueError("dataset_name contains unsupported characters")
    ratios = (train_ratio, val_ratio, golden_ratio)
    if any(value < 0 or value > 1 for value in ratios):
        raise ValueError("split ratios must be between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    dataset_root = output_root / dataset_name
    generated_paths = (
        dataset_root,
        output_root / "manifest.json",
        output_root / "reasonseg.zip",
    )
    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            f"LISA export already exists: {existing_paths[0]}; use --overwrite"
        )
    if overwrite:
        if dataset_root.is_dir():
            shutil.rmtree(dataset_root)
        elif dataset_root.exists():
            dataset_root.unlink()
        for path in generated_paths[1:]:
            path.unlink(missing_ok=True)
    snapshots = collect_task_export_snapshots(
        store,
        status="accepted",
        categories=categories,
        task_ids=task_ids,
    )
    if not snapshots:
        raise ValueError("no accepted annotation tasks matched the export filters")
    release = {
        "release_id": f"manual-{dataset_name}",
        "name": dataset_name,
        "split_policy": {
            "type": "grouped",
            "group_field": "group_id",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "golden_ratio": golden_ratio,
            "seed": seed,
        },
    }
    return build_release_files(
        release=release,
        snapshots=snapshots,
        output_root=output_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export accepted annotation tasks as a LISA ReasonSeg dataset",
    )
    parser.add_argument(
        "--storage-root",
        default=os.getenv("ANNOTATION_STORAGE_ROOT", "./annotation-data"),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--dataset-name", default="ReasonSegReviewed")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--golden-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    storage_root = Path(args.storage_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (
            Path(os.getenv("ANNOTATION_REVIEW_WORKSPACE", "./review_workspace"))
            .expanduser()
            .resolve()
            / "exports"
            / f"{args.dataset_name}-release"
        )
    )
    store = AnnotationStore(storage_root)
    store.initialize()
    try:
        files = export_lisa_dataset(
            store=store,
            output_root=output_root,
            dataset_name=args.dataset_name,
            categories=args.categories,
            task_ids=args.task_ids,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            golden_ratio=args.golden_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    finally:
        store.close()
    print(
        json.dumps(
            {
                "dataset": str(output_root / args.dataset_name),
                "archive": str(files.archive_path),
                "manifest": str(files.manifest_path),
                "counts": files.counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
