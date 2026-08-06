from __future__ import annotations

import argparse
import logging
import os
import socket

from .release_builder import ReleaseWorker
from .storage import AnnotationStore


def _integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the model-free ReasonSeg release worker",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    storage_root = os.environ["ANNOTATION_STORAGE_ROOT"]
    worker_id = os.getenv(
        "ANNOTATION_RELEASE_WORKER_ID",
        f"{socket.gethostname()}-{os.getpid()}",
    )
    lease_seconds = _integer(
        "ANNOTATION_RELEASE_LEASE_SECONDS",
        300,
    )
    heartbeat_seconds = _integer(
        "ANNOTATION_RELEASE_HEARTBEAT_SECONDS",
        60,
    )
    poll_seconds = float(
        os.getenv("ANNOTATION_RELEASE_POLL_SECONDS", "2")
    )
    store = AnnotationStore(storage_root)
    store.initialize()
    worker = ReleaseWorker(
        store=store,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        heartbeat_seconds=heartbeat_seconds,
        poll_seconds=poll_seconds,
    )
    try:
        if args.once:
            return 0 if worker.run_once() else 3
        worker.run_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
