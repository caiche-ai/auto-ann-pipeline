import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from annotation_service.sam_adapter import SAMMaskCandidate
from annotation_service.sam_worker import SAMMaskWorker
from annotation_service.storage import AnnotationStore


def png_bytes(color=(10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), color).save(output, format="PNG")
    return output.getvalue()


class FakeSAMPredictor:
    def predict(self, *, image_path, box_xyxy):
        return SAMMaskCandidate(
            mask_png=png_bytes((255, 255, 255)),
            overlay_png=png_bytes((255, 64, 64)),
            crop_png=png_bytes((10, 20, 30)),
            shapes=[
                {
                    "shape_id": "sam-target-1",
                    "label": "target",
                    "shape_type": "polygon",
                    "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                }
            ],
            box_xyxy=box_xyxy,
            predicted_iou=0.93,
            mask_area_pixels=64,
            model_version="sam-test",
        )


class SAMMaskWorkerTest(unittest.TestCase):
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
            group_id="site-1",
        )
        job = self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="full-v1",
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
                "shapes": [],
                "prompts": [],
            },
            provenance={"pipeline_version": "full-v1"},
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def test_worker_persists_all_sam_artifacts_and_shapes(self):
        operation = self.store.create_mask_candidate_operation(
            task_id=self.task["task_id"],
            expected_version=1,
            box_xyxy=[1, 1, 9, 9],
        )
        worker = SAMMaskWorker(
            store=self.store,
            predictor=FakeSAMPredictor(),
            worker_id="sam-test-worker",
            lease_seconds=60,
            heartbeat_seconds=10,
        )

        self.assertTrue(worker.run_once())

        completed = self.store.get_operation(
            operation["operation_id"]
        )
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(
            completed["result"]["shapes"][0]["label"],
            "target",
        )
        task = self.store.get_task(self.task["task_id"])
        self.assertTrue(task["artifacts"]["mask_png_url"])
        self.assertTrue(task["artifacts"]["mask_overlay_url"])
        self.assertTrue(task["artifacts"]["crop_url"])
        self.assertEqual(task["annotation"]["shapes"], [])


if __name__ == "__main__":
    unittest.main()
