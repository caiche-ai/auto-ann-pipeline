import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from annotation_service.pipeline_worker import (
    FullAnnotationPipelineWorker,
)
from annotation_service.qwen_contract import (
    QwenPromptSet,
    QwenVisualFacts,
)
from annotation_service.qwen_provider import QwenGenerationResult
from annotation_service.sam_adapter import SAMMaskCandidate
from annotation_service.storage import AnnotationStore
from annotation_service.worker.grounding_dino import (
    GroundingDINODetection,
)


def png_bytes(color=(10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 20), color).save(output, format="PNG")
    return output.getvalue()


class FakeDINO:
    def predict(
        self,
        *,
        image_path,
        width,
        height,
        categories,
    ):
        return [
            GroundingDINODetection(
                entity="person",
                box_xyxy=(2, 2, 18, 18),
                box_score=0.95,
                phrase_score=0.91,
                metadata={
                    "model_version": "dino-test",
                    "prompt_version": "prompt-test",
                },
            )
        ]


class FakeSAM:
    def predict(self, *, image_path, box_xyxy):
        return SAMMaskCandidate(
            mask_png=png_bytes((255, 255, 255)),
            overlay_png=png_bytes((255, 64, 64)),
            crop_png=png_bytes(),
            shapes=[
                {
                    "shape_id": "sam-target-1",
                    "label": "target",
                    "shape_type": "polygon",
                    "points": [[2, 2], [18, 2], [18, 18], [2, 18]],
                }
            ],
            box_xyxy=box_xyxy,
            predicted_iou=0.94,
            mask_area_pixels=256,
            model_version="sam-test",
        )


class FakeQwen:
    def generate(self, *, context, images):
        return QwenGenerationResult(
            facts=QwenVisualFacts(
                target_object="画面中央的一名作业人员",
                instance_count=1,
                visual_anchor=["位于画面中央"],
                mask_granularity="人员整体",
                visible_facts=["人员头部未见安全帽"],
                risk_semantics="头部防护缺失",
            ),
            prompt_set=QwenPromptSet(
                prompts=[
                    {"prompt_id": "v1", "type": "visual", "text": "分割中央人员。"},
                    {"prompt_id": "v2", "type": "visual", "text": "标出中部人员。"},
                    {"prompt_id": "v3", "type": "visual", "text": "提取中央作业人员。"},
                    {"prompt_id": "r1", "type": "risk", "text": "分割中央未戴安全帽人员。"},
                    {"prompt_id": "r2", "type": "risk", "text": "标出中央头部防护缺失人员。"},
                    {"prompt_id": "a1", "type": "agent", "text": "请定位并分割中央未戴安全帽人员。"},
                ]
            ),
            provider="test",
            model="qwen25vl",
            facts_prompt_version="facts-v1",
            enrichment_prompt_version="prompts-v1",
        )


class FullPipelineWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.store.initialize()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_default_job_runs_through_sam_qwen_and_task_creation(self):
        asset = self.store.create_asset(
            image_bytes=png_bytes(),
            media_type="image/png",
            width=20,
            height=20,
            group_id="site-1",
        )
        staged = self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="staged-v1",
            options={
                "generate_masks": False,
                "enrich_prompts": False,
                "prompt_count": 6,
                "stop_after": "hazard_rules",
            },
        )
        job = self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="full-v1",
        )
        worker = FullAnnotationPipelineWorker(
            store=self.store,
            detection_predictor=FakeDINO(),
            mask_predictor=FakeSAM(),
            prompt_provider=FakeQwen(),
            worker_id="pipeline-test",
            lease_seconds=60,
            heartbeat_seconds=10,
        )

        self.assertTrue(worker.run_once())

        completed = self.store.get_job(job["job_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(
            self.store.get_job(staged["job_id"])["status"],
            "queued",
        )
        self.assertEqual(completed["progress"]["generated_tasks"], 1)
        self.assertEqual(
            completed["stages"]["qwen_prompts"]["status"],
            "succeeded",
        )
        task = self.store.get_task(completed["task_ids"][0])
        self.assertEqual(len(task["annotation"]["prompts"]), 6)
        self.assertEqual(len(task["annotation"]["shapes"]), 1)
        self.assertEqual(task["provenance"]["sam_version"], "sam-test")
        self.assertEqual(task["provenance"]["qwen_model"], "qwen25vl")
        self.assertTrue(task["artifacts"]["mask_png_url"])
        self.assertEqual(task["version"], 1)


if __name__ == "__main__":
    unittest.main()
