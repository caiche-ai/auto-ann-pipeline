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


class BatchFakeSAMPredictor(FakeSAMPredictor):
    def __init__(self):
        self.batch_calls = 0
        self.last_boxes = []

    def predict_many(self, *, image_path, boxes_xyxy):
        self.batch_calls += 1
        self.last_boxes = boxes_xyxy
        return [
            self.predict(
                image_path=image_path,
                box_xyxy=box_xyxy,
            )
            for box_xyxy in boxes_xyxy
        ]


class PrefetchFakeSAMPredictor(FakeSAMPredictor):
    def __init__(self):
        self.precompute_calls = []

    def precompute(self, *, image_path):
        self.precompute_calls.append(image_path)
        return {
            "embedding_cache_hit": False,
            "image_encode_ms": 10.0,
        }


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
        self.assertIn(
            "artifact_write_ms",
            completed["result"]["timings_ms"],
        )
        task = self.store.get_task(self.task["task_id"])
        self.assertTrue(task["artifacts"]["mask_png_url"])
        self.assertTrue(task["artifacts"]["mask_overlay_url"])
        self.assertTrue(task["artifacts"]["crop_url"])
        self.assertEqual(task["annotation"]["shapes"], [])

    def test_worker_batches_boxes_for_one_image(self):
        second = self.store.create_task(
            job_id=self.task["job_id"],
            asset_id=self.task["asset"]["asset_id"],
            category="helmet_missing",
            annotation={
                "target_object": "另一名候选人员",
                "instance_count": 1,
                "visual_anchor": [],
                "mask_granularity": "人员整体",
                "risk_semantics": None,
                "shapes": [],
                "prompts": [],
            },
            provenance={"pipeline_version": "full-v1"},
        )
        first_operation = self.store.create_mask_candidate_operation(
            task_id=self.task["task_id"],
            expected_version=1,
            box_xyxy=[1, 1, 5, 9],
        )
        second_operation = self.store.create_mask_candidate_operation(
            task_id=second["task_id"],
            expected_version=1,
            box_xyxy=[5, 1, 9, 9],
        )
        predictor = BatchFakeSAMPredictor()
        worker = SAMMaskWorker(
            store=self.store,
            predictor=predictor,
            worker_id="sam-batch-worker",
            lease_seconds=60,
            heartbeat_seconds=10,
            max_batch_size=16,
        )

        self.assertTrue(worker.run_once())

        self.assertEqual(predictor.batch_calls, 1)
        self.assertEqual(
            predictor.last_boxes,
            [[1.0, 1.0, 5.0, 9.0], [5.0, 1.0, 9.0, 9.0]],
        )
        self.assertEqual(
            self.store.get_operation(
                first_operation["operation_id"]
            )["status"],
            "succeeded",
        )
        self.assertEqual(
            self.store.get_operation(
                second_operation["operation_id"]
            )["status"],
            "succeeded",
        )

    def test_worker_prefetches_recently_detected_asset_when_idle(self):
        self.store.add_detection(
            job_id=self.task["job_id"],
            asset_id=self.task["asset"]["asset_id"],
            entity="person",
            box_xyxy=[1, 1, 9, 9],
            box_score=0.9,
            phrase_score=0.8,
        )
        predictor = PrefetchFakeSAMPredictor()
        worker = SAMMaskWorker(
            store=self.store,
            predictor=predictor,
            worker_id="sam-prefetch-worker",
            lease_seconds=60,
            heartbeat_seconds=10,
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(len(predictor.precompute_calls), 1)
        self.assertFalse(worker.run_once())


if __name__ == "__main__":
    unittest.main()
