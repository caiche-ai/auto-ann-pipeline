import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from annotation_service.app import create_app
from annotation_service.config import Settings
from annotation_service.release_builder import ReleaseWorker
from annotation_service.storage import AnnotationStore


def make_settings(**overrides) -> Settings:
    values = {
        "service_version": "test-v1",
        "api_key": None,
        "cors_origins": (),
        "cors_allow_credentials": False,
        "max_request_bytes": 8192,
        "max_image_bytes": 4096,
        "max_image_pixels": 1000,
        "max_metadata_chars": 100,
        "max_queued_jobs": 100,
        "docs_enabled": True,
        "storage_enabled": False,
        "storage_root": "./annotation-data",
    }
    values.update(overrides)
    return Settings(**values)


def request_payload(name="release-v1") -> dict:
    return {
        "name": name,
        "task_filter": {"status": "accepted"},
        "split_policy": {
            "type": "grouped",
            "group_field": "group_id",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "golden_ratio": 0.1,
            "seed": 42,
        },
    }


class ReleaseApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.client_context = TestClient(
            create_app(make_settings(), storage=self.store)
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_create_query_idempotency_and_not_ready_download(self):
        headers = {"Idempotency-Key": "release-request-001"}
        first = self.client.post(
            "/v1/annotation/releases",
            json=request_payload(),
            headers=headers,
        )
        retry = self.client.post(
            "/v1/annotation/releases",
            json=request_payload(),
            headers=headers,
        )
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertEqual(
            first.json()["release_id"],
            retry.json()["release_id"],
        )
        release_id = first.json()["release_id"]
        detail = self.client.get(
            f"/v1/annotation/releases/{release_id}"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["status"], "queued")
        self.assertNotIn("claim_token", detail.json())

        conflict = self.client.post(
            "/v1/annotation/releases",
            json=request_payload("another-release"),
            headers=headers,
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        not_ready = self.client.get(
            f"/v1/annotation/releases/{release_id}/archive"
        )
        self.assertEqual(not_ready.status_code, 409, not_ready.text)

    def test_failed_release_is_visible(self):
        created = self.client.post(
            "/v1/annotation/releases",
            json=request_payload("empty-release"),
        )
        release_id = created.json()["release_id"]
        ReleaseWorker(
            store=self.store,
            worker_id="release-test",
            lease_seconds=30,
            heartbeat_seconds=5,
        ).run_once()
        detail = self.client.get(
            f"/v1/annotation/releases/{release_id}"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["status"], "failed")

    def test_release_routes_require_authentication(self):
        with TestClient(
            create_app(
                make_settings(api_key="secret"),
                storage=self.store,
            )
        ) as secured:
            response = secured.post(
                "/v1/annotation/releases",
                json=request_payload(),
            )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
