import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from annotation_service.qwen_contract import (
    QwenPromptSet,
    QwenVisualFacts,
)
from annotation_service.qwen_provider import QwenGenerationResult
from annotation_service.qwen_worker import QwenPromptWorker
from annotation_service.storage import AnnotationStore


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), (10, 20, 30)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate(self, *, context, images):
        self.calls.append((context, images))
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
            provider="test-provider",
            model="qwen2.5-vl-7b-instruct",
            facts_prompt_version="facts-v1",
            enrichment_prompt_version="prompts-v1",
        )


class QwenPromptWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.store.initialize()
        asset = self.store.create_asset(
            image_bytes=png_bytes(),
            media_type="image/png",
            width=10,
            height=10,
            source_id="sample-1",
            group_id="site-1",
            metadata={},
        )
        job = self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="grounded-qwen-v1",
        )
        self.task = self.store.create_task(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            category="helmet_missing",
            annotation={
                "target_object": "候选人员",
                "instance_count": 1,
                "visual_anchor": [],
                "mask_granularity": "人员整体",
                "risk_semantics": None,
                "shapes": [
                    {
                        "shape_id": "shape-1",
                        "label": "target",
                        "shape_type": "polygon",
                        "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                    }
                ],
                "prompts": [],
            },
            provenance={"pipeline_version": "grounded-qwen-v1"},
        )
        self.store.store_artifact(
            task_id=self.task["task_id"],
            artifact_type="mask-overlay",
            data=png_bytes(),
            media_type="image/png",
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_worker_generates_candidate_without_overwriting_task(self):
        operation = self.store.create_prompt_enrichment_operation(
            task_id=self.task["task_id"],
            expected_version=1,
        )
        provider = FakeProvider()
        worker = QwenPromptWorker(
            store=self.store,
            provider=provider,
            worker_id="qwen-test-worker",
            lease_seconds=60,
            heartbeat_seconds=10,
        )

        self.assertTrue(worker.run_once())

        completed = self.store.get_operation(
            operation["operation_id"]
        )
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(len(completed["result"]["prompts"]), 6)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.calls[0][1]), 2)
        unchanged = self.store.get_task(self.task["task_id"])
        self.assertEqual(unchanged["version"], 1)
        self.assertEqual(unchanged["annotation"]["prompts"], [])


if __name__ == "__main__":
    unittest.main()
