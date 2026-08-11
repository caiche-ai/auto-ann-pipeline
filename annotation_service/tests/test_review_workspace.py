import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from annotation_service.review.export_lisa import export_lisa_dataset
from annotation_service.review.workspace import backfill_review_submissions
from annotation_service.storage.repository import AnnotationStore
from annotation_service.tools.render_annotation_results import (
    find_review_font,
    render_annotation_results,
)


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (80, 100, 120)).save(output, format="PNG")
    return output.getvalue()


def _annotation() -> dict:
    return {
        "target_object": "未戴安全帽的作业人员",
        "instance_count": 1,
        "visual_anchor": ["画面中央"],
        "mask_granularity": "人员整体",
        "risk_semantics": "头部防护缺失",
        "shapes": [
            {
                "shape_id": "shape-1",
                "label": "target",
                "shape_type": "polygon",
                "points": [[4, 3], [27, 3], [27, 20], [4, 20]],
            }
        ],
        "prompts": [
            {"prompt_id": "v1", "type": "visual", "text": "分割目标人员。"},
            {"prompt_id": "v2", "type": "visual", "text": "标出目标人员。"},
            {"prompt_id": "v3", "type": "visual", "text": "提取目标人员。"},
            {"prompt_id": "r1", "type": "risk", "text": "分割未戴安全帽的人员。"},
            {"prompt_id": "r2", "type": "risk", "text": "标出头部防护缺失人员。"},
            {"prompt_id": "a1", "type": "agent", "text": "找出并分割目标。"},
        ],
    }


class ReviewWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "annotation-data"
        self.store = AnnotationStore(self.root)
        self.store.initialize()
        asset = self.store.create_asset(
            image_bytes=_image_bytes(),
            media_type="image/png",
            width=32,
            height=24,
            group_id="site01:camera01",
        )
        job = self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="manual-v1",
        )
        self.task = self.store.create_task(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            category="helmet_missing",
            annotation=_annotation(),
            provenance={"pipeline_version": "manual-v1"},
        )
        self.submitted = self.store.submit_task(
            self.task["task_id"],
            expected_version=self.task["version"],
            annotator_id="annotator-7",
            primary_result="prompt_ok",
            comment="请审核",
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_renderer_embeds_annotation_and_prompts_in_one_image(self):
        output_root = Path(self.temporary.name) / "review-workspace" / "outputs"
        rendered = render_annotation_results(
            storage_root=self.root,
            output_root=output_root,
        )

        self.assertEqual(len(rendered), 1)
        self.assertTrue(rendered[0].is_file())
        with Image.open(rendered[0]) as image:
            self.assertGreaterEqual(image.width, 900)
            self.assertGreater(image.height, 24)
        self.assertEqual(
            rendered[0].parent,
            output_root,
        )
        self.assertTrue(find_review_font().is_file())

    def test_historical_submission_backfill_is_idempotent(self):
        first = backfill_review_submissions(self.store)
        second = backfill_review_submissions(self.store)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertTrue(first[0].is_file())

    def test_exporter_writes_lisa_reasonseg_pairs_for_accepted_tasks(self):
        self.store.review_task(
            self.task["task_id"],
            expected_version=self.submitted["version"],
            reviewer_id="reviewer-9",
            decision="accept",
            primary_result="prompt_ok",
            comment="通过",
        )
        output_root = self.root / "review" / "lisa_exports"
        files = export_lisa_dataset(
            store=self.store,
            output_root=output_root,
            dataset_name="ReasonSegReviewedTest",
            train_ratio=1.0,
            val_ratio=0.0,
            golden_ratio=0.0,
        )

        dataset_root = output_root / "ReasonSegReviewedTest" / "train"
        images = list(dataset_root.glob("*.jpg"))
        annotations = list(dataset_root.glob("*.json"))
        self.assertEqual(len(images), 1)
        self.assertEqual(len(annotations), 1)
        payload = json.loads(annotations[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["shapes"][0]["label"], "target")
        self.assertEqual(len(payload["text"]), 6)
        self.assertTrue(files.archive_path.is_file())
        self.assertTrue(files.manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
