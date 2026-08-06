import tempfile
import unittest
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.app import create_app
from annotation_service.config import Settings
from annotation_service.storage import AnnotationStore


def png_bytes(
    color: tuple[int, int, int] = (10, 20, 30),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 10), color).save(output, format="PNG")
    return output.getvalue()


def make_settings(**overrides) -> Settings:
    values = {
        "service_version": "test-v1",
        "api_key": None,
        "cors_origins": (),
        "cors_allow_credentials": False,
        "max_request_bytes": 4096,
        "max_image_bytes": 2048,
        "max_image_pixels": 1000,
        "max_metadata_chars": 100,
        "max_queued_jobs": 100,
        "docs_enabled": True,
        "storage_enabled": False,
        "storage_root": "./annotation-data",
    }
    values.update(overrides)
    return Settings(**values)


class JobApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.client_context = TestClient(
            create_app(make_settings(), storage=self.store)
        )
        self.client = self.client_context.__enter__()
        self.asset_id = self.upload_asset()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def upload_asset(self) -> str:
        response = self.client.post(
            "/v1/annotation/assets",
            files={
                "file": (
                    "sample.png",
                    png_bytes(),
                    "image/png",
                )
            },
            data={"group_id": "site01:video03"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["asset_id"]

    def request_payload(self, **overrides) -> dict:
        payload = {
            "asset_ids": [self.asset_id],
            "grounding_prompt": (
                "worker without a helmet near the excavator"
            ),
            "pipeline_version": "groundingdino-free-form-v1",
        }
        payload.update(overrides)
        return payload

    def create_job(
        self,
        *,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ):
        return self.client.post(
            "/v1/annotation/jobs",
            json=payload or self.request_payload(),
            headers=headers or {},
        )

    def test_create_and_query_free_detection_job(self):
        response = self.create_job()
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(
            job["grounding_prompt"],
            "worker without a helmet near the excavator",
        )
        self.assertEqual(
            set(job["stages"]),
            {"grounding_dino"},
        )
        self.assertEqual(
            job["stages"]["grounding_dino"]["status"],
            "pending",
        )
        self.assertNotIn("requested_categories", job)
        self.assertNotIn("options", job)
        self.assertNotIn("claimed_by", job)
        self.assertNotIn("asset_ids", job)

        detail = self.client.get(
            f"/v1/annotation/jobs/{job['job_id']}"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json(), job)

    def test_prompt_content_is_not_restricted_by_category(self):
        prompt = "任意中文描述：找出蓝色设备旁边的人！@#$%^&*()"
        response = self.create_job(
            payload=self.request_payload(grounding_prompt=prompt)
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.json()["grounding_prompt"],
            prompt,
        )

        old_contract = self.request_payload()
        old_contract.pop("grounding_prompt")
        old_contract["requested_categories"] = ["helmet_missing"]
        rejected = self.create_job(payload=old_contract)
        self.assertEqual(rejected.status_code, 422)

    def test_blank_or_oversized_prompt_is_rejected(self):
        blank = self.create_job(
            payload=self.request_payload(grounding_prompt="   ")
        )
        self.assertEqual(blank.status_code, 422)

        oversized = self.create_job(
            payload=self.request_payload(grounding_prompt="x" * 2001)
        )
        self.assertEqual(oversized.status_code, 422)

    def test_query_detections_and_bbox_image(self):
        response = self.create_job()
        job_id = response.json()["job_id"]

        empty = self.client.get(
            f"/v1/annotation/jobs/{job_id}/detections"
        )
        self.assertEqual(
            empty.json(),
            {"job_id": job_id, "items": [], "total": 0},
        )
        missing_image = self.client.get(
            f"/v1/annotation/jobs/{job_id}/assets/"
            f"{self.asset_id}/bbox-image"
        )
        self.assertEqual(missing_image.status_code, 404)

        claimed = self.store.claim_next_job(
            worker_id="test-worker",
            lease_seconds=30,
            required_stop_after="grounding_dino",
            grounding_prompt_required=True,
        )
        self.assertEqual(claimed["job_id"], job_id)
        saved = self.store.replace_detections(
            job_id=job_id,
            asset_id=self.asset_id,
            detections=[
                {
                    "entity": "person",
                    "box_xyxy": [1, 1, 19, 9],
                    "box_score": 0.91,
                    "phrase_score": 0.83,
                    "metadata": {
                        "grounding_prompt": claimed[
                            "grounding_prompt"
                        ]
                    },
                }
            ],
            worker_id="test-worker",
        )
        self.store.store_job_artifact(
            job_id=job_id,
            asset_id=self.asset_id,
            artifact_type="bbox-image",
            data=png_bytes((30, 40, 50)),
            media_type="image/png",
            worker_id="test-worker",
            metadata={"detection_count": 1},
        )

        detections = self.client.get(
            f"/v1/annotation/jobs/{job_id}/detections"
        )
        self.assertEqual(detections.status_code, 200)
        self.assertEqual(detections.json()["total"], 1)
        self.assertEqual(
            detections.json()["items"][0]["detection_id"],
            saved[0]["detection_id"],
        )

        image = self.client.get(
            f"/v1/annotation/jobs/{job_id}/assets/"
            f"{self.asset_id}/bbox-image"
        )
        self.assertEqual(image.status_code, 200, image.text)
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertTrue(image.headers["etag"])
        self.assertEqual(image.content, png_bytes((30, 40, 50)))

    def test_completed_detection_can_build_idempotent_review_task(self):
        job_id = self.create_job().json()["job_id"]
        worker_id = "task-builder-test-worker"
        claimed = self.store.claim_next_job(
            worker_id=worker_id,
            lease_seconds=30,
            required_stop_after="grounding_dino",
            grounding_prompt_required=True,
        )
        saved = self.store.replace_detections(
            job_id=job_id,
            asset_id=self.asset_id,
            detections=[
                {
                    "entity": "worker",
                    "box_xyxy": [1, 1, 19, 9],
                    "box_score": 0.91,
                    "phrase_score": 0.83,
                    "metadata": {
                        "grounding_prompt": claimed["grounding_prompt"],
                        "model_version": "dino-test",
                        "prompt_version": "free-form-v1",
                        "box_threshold": 0.35,
                        "text_threshold": 0.25,
                    },
                }
            ],
            worker_id=worker_id,
        )
        self.store.update_job_asset(
            job_id=job_id,
            asset_id=self.asset_id,
            status="succeeded",
            worker_id=worker_id,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        self.store.update_job(
            job_id,
            expected_status="running",
            status="succeeded",
            stage="grounding_dino",
            progress={
                "total_assets": 1,
                "completed_assets": 1,
                "generated_tasks": 0,
            },
            stages={
                "grounding_dino": {
                    "status": "succeeded",
                    "started_at": claimed["started_at"],
                    "completed_at": completed_at,
                    "message": "1/1 assets detected",
                }
            },
            worker_id=worker_id,
        )

        payload = {
            "detection_ids": [saved[0]["detection_id"]],
            "category": "unsafe",
        }
        created = self.client.post(
            f"/v1/annotation/jobs/{job_id}/review-tasks",
            json=payload,
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["created_count"], 1)
        task_id = created.json()["task_ids"][0]
        task = self.client.get(
            f"/v1/annotation/tasks/{task_id}"
        ).json()
        self.assertEqual(
            task["source_detection_id"],
            saved[0]["detection_id"],
        )
        self.assertEqual(task["provenance"]["grounding_prompt"], claimed[
            "grounding_prompt"
        ])

        repeated = self.client.post(
            f"/v1/annotation/jobs/{job_id}/review-tasks",
            json=payload,
        )
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()["created_count"], 0)
        self.assertEqual(repeated.json()["existing_count"], 1)
        self.assertEqual(repeated.json()["task_ids"], [task_id])

    def test_queued_job_can_be_cancelled(self):
        job_id = self.create_job().json()["job_id"]
        response = self.client.post(
            f"/v1/annotation/jobs/{job_id}/cancel",
            json={
                "actor_id": "spring-test",
                "reason": "用户取消检测步骤",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertEqual(response.json()["errors"][-1]["code"], "cancelled")

    def test_idempotent_retry_and_conflict(self):
        headers = {"Idempotency-Key": "frontend-job-001"}
        first = self.create_job(headers=headers)
        retry = self.create_job(headers=headers)
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertEqual(
            retry.json()["job_id"],
            first.json()["job_id"],
        )

        conflict = self.create_job(
            payload=self.request_payload(
                grounding_prompt="a different prompt"
            ),
            headers=headers,
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)

    def test_missing_asset_and_queue_limit(self):
        missing = self.create_job(
            payload=self.request_payload(asset_ids=["ast_missing"])
        )
        self.assertEqual(missing.status_code, 404, missing.text)

        limited_store = AnnotationStore(
            Path(self.temporary.name) / "limited-data"
        )
        app = create_app(
            make_settings(max_queued_jobs=1),
            storage=limited_store,
        )
        with TestClient(app) as client:
            asset = limited_store.create_asset(
                image_bytes=png_bytes(),
                media_type="image/png",
                width=20,
                height=10,
                group_id="site01:video04",
            )
            payload = self.request_payload(
                asset_ids=[asset["asset_id"]]
            )
            first = client.post("/v1/annotation/jobs", json=payload)
            second = client.post("/v1/annotation/jobs", json=payload)
            self.assertEqual(first.status_code, 202)
            self.assertEqual(second.status_code, 429)


if __name__ == "__main__":
    unittest.main()
