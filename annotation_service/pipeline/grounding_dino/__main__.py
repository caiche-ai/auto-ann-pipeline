from __future__ import annotations

import argparse
import logging

from ...storage.repository import AnnotationStore
from .factory import build_detection_predictor
from .worker import GroundingDINOJobWorker
from .settings import GroundingDINOWorkerSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the GroundingDINO annotation GPU worker",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one compatible job and exit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = GroundingDINOWorkerSettings.from_env()
    settings.validate_model_files()
    store = AnnotationStore(settings.storage_root)
    store.initialize()
    predictor = build_detection_predictor(settings)
    if not args.once and settings.provider == "local":
        predictor.load()
        logging.getLogger(__name__).info("GroundingDINO model preloaded and ready")
    worker = GroundingDINOJobWorker(
        store=store,
        predictor=predictor,
        worker_id=settings.worker_id,
        lease_seconds=settings.lease_seconds,
        heartbeat_seconds=settings.heartbeat_seconds,
        poll_seconds=settings.poll_seconds,
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
