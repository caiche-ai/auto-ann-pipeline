import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from annotation_service.storage import AnnotationStore
from annotation_service.prompt_normalization import (
    normalize_grounding_prompt as normalize_grounding_prompt_result,
)
from annotation_service.worker.grounding_dino import (
    GroundingDINODetection,
    GroundingPromptPreparation,
    normalize_grounding_prompt,
    normalized_cxcywh_to_xyxy,
)
from annotation_service.worker.runner import GroundingDINOJobWorker
from annotation_service.worker.settings import GroundingDINOWorkerSettings


def png_bytes(
    color: tuple[int, int, int] = (10, 20, 30),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", (40, 20), color).save(output, format="PNG")
    return output.getvalue()


class FakePredictor:
    model_version = "fake-grounding-dino-v1"
    prompt_version = "free-form-v1"

    def __init__(self, *, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, str | None]] = []

    def predict(
        self,
        *,
        image_path: Path,
        width: int,
        height: int,
        prompt: str | None = None,
        categories=None,
        prompt_normalization_mode=None,
        prompt_normalization_profile=None,
        prompt_translation_failure_policy=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "prompt_normalization_mode": prompt_normalization_mode,
                "prompt_normalization_profile": prompt_normalization_profile,
                "prompt_translation_failure_policy": (
                    prompt_translation_failure_policy
                ),
            }
        )
        if (
            self.fail_on_call is not None
            and self.fail_on_call == len(self.calls)
        ):
            raise RuntimeError("synthetic detector failure")
        return [
            GroundingDINODetection(
                entity="person",
                box_xyxy=(1.0, 1.0, width - 1.0, height - 1.0),
                box_score=0.91,
                phrase_score=0.87,
                metadata={
                    "model_version": self.model_version,
                    "prompt_version": self.prompt_version,
                    "grounding_prompt": prompt,
                    "image_name": image_path.name,
                },
            )
        ]


class RouteAwareEmptyPredictor:
    model_version = "fake-grounding-dino-v1"
    prompt_version = "free-form-v1"

    def __init__(self, route: dict[str, bool]):
        self.route = route
        self.prepared_prompt_seen = False

    def prepare_prompt(self, *, prompt: str, **kwargs):
        return GroundingPromptPreparation(
            caption="person .",
            requested_entities=(),
            requested_prompt=prompt,
            metadata={},
            route=self.route,
        )

    def predict(self, *, prepared_prompt=None, **kwargs):
        self.prepared_prompt_seen = prepared_prompt is not None
        return []


class GroundingDINOHelpersTest(unittest.TestCase):
    def test_free_prompt_only_adds_terminal_period(self):
        self.assertEqual(
            normalize_grounding_prompt("person near excavator"),
            "person near excavator .",
        )
        self.assertEqual(
            normalize_grounding_prompt("任意提示词。"),
            "任意提示词。 .",
        )
        self.assertEqual(
            normalize_grounding_prompt("person."),
            "person.",
        )
        with self.assertRaises(ValueError):
            normalize_grounding_prompt("   ")

    def test_canonical_terms_mode_maps_safety_aliases(self):
        result = normalize_grounding_prompt_result(
            "安全帽 near the excavator",
            mode="canonical_terms",
        )
        self.assertEqual(
            result.normalized_prompt,
            "helmet near the excavator .",
        )
        self.assertEqual(
            result.applied_aliases,
            (("安全帽", "helmet"),),
        )
        self.assertEqual(
            result.original_prompt,
            "安全帽 near the excavator",
        )

    def test_normalized_box_conversion_clamps_to_image(self):
        self.assertEqual(
            normalized_cxcywh_to_xyxy(
                (0.5, 0.5, 0.4, 0.6),
                width=100,
                height=50,
            ),
            (30.0, 10.0, 70.0, 40.0),
        )
        self.assertIsNone(
            normalized_cxcywh_to_xyxy(
                (2.0, 2.0, 0.1, 0.1),
                width=100,
                height=50,
            )
        )


class GroundingDINOWorkerSettingsTest(unittest.TestCase):
    def test_absolute_model_store_paths_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "GroundingDINO"
            model_store = Path(temporary) / "MODEL_STORE"
            bert = model_store / "text_encoder" / "bert-base-uncased"
            config = model_store / "GroundingDINO_SwinT_OGC.py"
            checkpoint = model_store / "groundingdino_swint_ogc.pth"
            root.mkdir()
            bert.mkdir(parents=True)
            config.write_text("text_encoder_type = 'bert-base-uncased'")
            checkpoint.write_bytes(b"checkpoint")
            for filename in (
                "config.json",
                "model.safetensors",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt",
            ):
                (bert / filename).write_bytes(b"x")
            environment = {
                "ANNOTATION_STORAGE_ROOT": str(
                    Path(temporary) / "annotation-data"
                ),
                "ANNOTATION_GROUNDING_DINO_ROOT": str(root),
                "ANNOTATION_GROUNDING_DINO_CONFIG": str(config),
                "ANNOTATION_GROUNDING_DINO_CHECKPOINT": str(checkpoint),
                "ANNOTATION_GROUNDING_DINO_BERT": str(bert),
                "ANNOTATION_WORKER_ID": "worker-test",
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_MODE": (
                    "llm_grounding_caption"
                ),
                "ANNOTATION_GROUNDING_DINO_PROMPT_NORMALIZATION_PROFILE": (
                    "open_semantic_zh_en_v1"
                ),
                "ANNOTATION_PROMPT_TRANSLATOR_BASE_URL": (
                    "http://127.0.0.1:18000/qwen25/v1"
                ),
                "ANNOTATION_PROMPT_TRANSLATOR_MODEL": "qwen25vl",
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = GroundingDINOWorkerSettings.from_env()
            settings.validate_model_files()

            self.assertEqual(settings.config_path, config.resolve())
            self.assertEqual(settings.checkpoint_path, checkpoint.resolve())
            self.assertEqual(settings.bert_path, bert.resolve())
            self.assertEqual(settings.prompt_version, "free-form-v1")
            self.assertEqual(
                settings.prompt_normalization_mode,
                "llm_grounding_caption",
            )
            self.assertIsNotNone(settings.prompt_translator())


class GroundingDINOJobWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.store.initialize()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def create_asset(
        self,
        color: tuple[int, int, int] = (10, 20, 30),
    ) -> dict:
        return self.store.create_asset(
            image_bytes=png_bytes(color),
            media_type="image/png",
            width=40,
            height=20,
            group_id="site01:camera01",
        )

    def create_job(
        self,
        asset_ids: list[str],
        prompt: str = "worker beside excavator",
    ) -> dict:
        return self.store.create_job(
            asset_ids=asset_ids,
            grounding_prompt=prompt,
            pipeline_version="groundingdino-free-form-v1",
            options={
                "generate_masks": False,
                "enrich_prompts": False,
                "stop_after": "grounding_dino",
                "grounding_prompt_normalization_mode": "off",
                "grounding_prompt_normalization_profile": (
                    "construction_safety_v1"
                ),
                "grounding_prompt_translation_failure_policy": (
                    "fallback_canonical_terms"
                ),
            },
        )

    def make_worker(self, predictor: FakePredictor):
        return GroundingDINOJobWorker(
            store=self.store,
            predictor=predictor,
            worker_id="test-gpu-worker",
            lease_seconds=30,
            heartbeat_seconds=5,
            poll_seconds=0.1,
        )

    def test_worker_uses_free_prompt_and_generates_bbox_image(self):
        asset = self.create_asset()
        prompt = "the blue machine and nearby worker"
        job = self.create_job([asset["asset_id"]], prompt)
        predictor = FakePredictor()

        self.assertTrue(self.make_worker(predictor).run_once())

        completed = self.store.get_job(job["job_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["grounding_prompt"], prompt)
        self.assertEqual(
            predictor.calls,
            [
                {
                    "prompt": prompt,
                    "prompt_normalization_mode": "off",
                    "prompt_normalization_profile": (
                        "construction_safety_v1"
                    ),
                    "prompt_translation_failure_policy": (
                        "fallback_canonical_terms"
                    ),
                }
            ],
        )
        self.assertEqual(
            completed["stages"]["grounding_dino"]["status"],
            "succeeded",
        )
        self.assertEqual(set(completed["stages"]), {"grounding_dino"})
        detections = self.store.list_detections(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
        )
        self.assertEqual(len(detections), 1)
        path, media_type, sha256 = self.store.job_artifact_file(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            artifact_type="bbox-image",
        )
        self.assertEqual(media_type, "image/png")
        self.assertEqual(len(sha256), 64)
        with Image.open(path) as overlay:
            self.assertEqual(overlay.size, (40, 20))
            self.assertNotEqual(
                overlay.convert("RGB").getpixel((1, 1)),
                (10, 20, 30),
            )

    def test_worker_records_partial_asset_failure(self):
        first = self.create_asset()
        second = self.create_asset((30, 40, 50))
        job = self.create_job(
            [first["asset_id"], second["asset_id"]]
        )

        self.assertTrue(
            self.make_worker(FakePredictor(fail_on_call=2)).run_once()
        )

        completed = self.store.get_job(job["job_id"])
        self.assertEqual(completed["status"], "partial_failed")
        self.assertEqual(len(completed["errors"]), 1)
        self.assertEqual(
            completed["errors"][0]["asset_id"],
            second["asset_id"],
        )

    def test_job_route_is_persisted_when_detection_result_is_empty(self):
        asset = self.create_asset()
        job = self.create_job(
            [asset["asset_id"]],
            "找出右侧没有佩戴安全帽的人员",
        )
        route = {
            "rule_attempted": True,
            "rule_matched": False,
            "llm_attempted": True,
            "llm_succeeded": True,
            "fallback_used": False,
        }
        predictor = RouteAwareEmptyPredictor(route)

        self.assertTrue(self.make_worker(predictor).run_once())

        completed = self.store.get_job(job["job_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["grounding_prompt_route"], route)
        self.assertTrue(predictor.prepared_prompt_seen)
        self.assertEqual(
            self.store.list_detections(
                job_id=job["job_id"],
                asset_id=asset["asset_id"],
            ),
            [],
        )

    def test_legacy_category_job_is_not_claimed(self):
        asset = self.create_asset()
        self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="legacy-v1",
            options={
                "generate_masks": False,
                "enrich_prompts": False,
                "stop_after": "grounding_dino",
            },
        )
        self.assertFalse(self.make_worker(FakePredictor()).run_once())


if __name__ == "__main__":
    unittest.main()
