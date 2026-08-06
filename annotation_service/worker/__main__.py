from __future__ import annotations

import argparse
import logging

from ..storage import AnnotationStore
from .grounding_dino import GroundingDINOAdapter, GroundingDINOModelConfig
from .runner import GroundingDINOJobWorker
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
    predictor = GroundingDINOAdapter(
        GroundingDINOModelConfig(
            root=settings.grounding_dino_root,
            config_path=settings.config_path,
            checkpoint_path=settings.checkpoint_path,
            bert_path=settings.bert_path,
            device=settings.device,
            model_version=settings.model_version,
            prompt_version=settings.prompt_version,
            prompt_normalization_mode=settings.prompt_normalization_mode,
            prompt_normalization_profile=(
                settings.prompt_normalization_profile
            ),
            prompt_translation_failure_policy=(
                settings.prompt_translation_failure_policy
            ),
            box_threshold=settings.box_threshold,
            text_threshold=settings.text_threshold,
        ),
        prompt_translator=settings.prompt_translator(),
    )
    if not args.once:
        predictor.load()
        logging.getLogger(__name__).info(
            "GroundingDINO model preloaded and ready"
        )
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
