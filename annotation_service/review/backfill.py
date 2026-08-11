from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..storage.repository import AnnotationStore
from .workspace import backfill_review_submissions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill immutable review files from historical submissions",
    )
    parser.add_argument(
        "--storage-root",
        default=os.getenv("ANNOTATION_STORAGE_ROOT", "./annotation-data"),
    )
    args = parser.parse_args()
    storage_root = Path(args.storage_root).expanduser().resolve()
    store = AnnotationStore(storage_root)
    store.initialize()
    try:
        paths = backfill_review_submissions(store)
    finally:
        store.close()
    print(
        json.dumps(
            {
                "submissions": len(paths),
                "submissions_root": str(storage_root / "submissions"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
