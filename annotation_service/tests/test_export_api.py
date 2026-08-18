import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.api.app import create_app
from annotation_service.api.config import Settings
from annotation_service.storage.repository import AnnotationStore
from annotation_service.tests.test_task_api import complete_annotation


def png_bytes(color=(10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), color).save(output, format="PNG")
    return output.getvalue()


class ExportApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "annotation-data"
        self.store = AnnotationStore(self.root)
        settings = Settings(
            service_version="test-v1",
            api_key=None,
            docs_enabled=True,
            storage_enabled=False,
            max_image_bytes=4096,
            max_image_pixels=1000,
        )
        self.client_context = TestClient(
            create_app(settings, storage=self.store)
        )
        self.client = self.client_context.__enter__()
        asset = self.client.post(
            "/v1/annotation/assets",
            files={"file": ("现场照片01.png", png_bytes(), "image/png")},
            data={"group_id": "frontend-ai-annotation"},
        ).json()
        self.asset_id = asset["asset_id"]
        job = self.store.create_job(
            asset_ids=[self.asset_id],
            requested_categories=["unsafe"],
            pipeline_version="groundingdino-free-form-v1",
        )
        self.job_id = job["job_id"]
        detection = self.store.add_detection(
            job_id=self.job_id,
            asset_id=self.asset_id,
            entity="person",
            box_xyxy=[1, 1, 8, 8],
            box_score=0.91,
            phrase_score=0.88,
        )
        self.task = self.store.create_task(
            job_id=self.job_id,
            asset_id=self.asset_id,
            category="unsafe",
            annotation=complete_annotation("目标人员"),
            provenance={
                "pipeline_version": "groundingdino-free-form-v1",
                "grounding_dino_version": "groundingdino-swint-ogc",
                "grounding_prompt": "person",
                "source_detection_id": detection["detection_id"],
            },
        )
        for artifact_type in ("mask", "mask-overlay"):
            self.store.store_artifact(
                task_id=self.task["task_id"],
                artifact_type=artifact_type,
                data=png_bytes((255, 255, 255)),
                media_type="image/png",
            )

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_submit_materializes_six_files_and_export_filters_by_time(self):
        submitted = self.client.post(
            f"/v1/annotation/tasks/{self.task['task_id']}/submit",
            json={
                "expected_version": self.task["version"],
                "annotator_id": "annotator-1",
                "primary_result": "prompt_ok",
                "comment": "final",
            },
        )
        self.assertEqual(submitted.status_code, 200, submitted.text)
        self.assertEqual(submitted.json()["status"], "accepted")

        sample_root = self.root / "submissions" / "现场照片01"
        expected_names = {
            "现场照片01.jpg",
            "现场照片01_lisa.json",
            "现场照片01_grounding-dino.json",
            "现场照片01_grounding-dino-boxes.png",
            "现场照片01_sam-mask.png",
            "现场照片01_sam-overlay.png",
        }
        self.assertEqual(
            {path.name for path in sample_root.iterdir()},
            expected_names,
        )
        grounding = json.loads(
            (sample_root / "现场照片01_grounding-dino.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(grounding["detections"][0]["box_xyxy"], [1.0, 1.0, 8.0, 8.0])

        exported = self.client.get(
            "/v1/annotation/export",
            params={
                "start_time": "2000-01-01T00:00:00Z",
                "end_time": "2100-01-01T00:00:00Z",
                "task_id": self.task["task_id"],
            },
        )
        self.assertEqual(exported.status_code, 200, exported.text)
        self.assertEqual(exported.headers["x-annotation-sample-count"], "1")
        with zipfile.ZipFile(BytesIO(exported.content)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {f"现场照片01/{name}" for name in expected_names},
            )

        empty = self.client.get(
            "/v1/annotation/export",
            params={"end_time": "2000-01-01T00:00:00Z"},
        )
        self.assertEqual(empty.status_code, 422, empty.text)


if __name__ == "__main__":
    unittest.main()
