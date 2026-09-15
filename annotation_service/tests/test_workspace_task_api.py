import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.api.app import create_app
from annotation_service.api.config import Settings
from annotation_service.storage.repository import AnnotationStore


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 3), (20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def settings() -> Settings:
    return Settings(
        service_version="test-v1",
        api_key=None,
        cors_origins=(),
        cors_allow_credentials=False,
        max_request_bytes=4096,
        max_image_bytes=2048,
        max_image_pixels=1000,
        max_metadata_chars=500,
        docs_enabled=True,
        storage_enabled=False,
        storage_root="./annotation-data",
    )


class WorkspaceTaskApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name) / "annotation-data"
        self.app = create_app(settings(), storage=AnnotationStore(self.root))
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)
        self.temporary.cleanup()

    def create_task(self) -> dict:
        response = self.client.post(
            "/v1/annotation/workspace-tasks",
            json={"name": "安全帽标注", "description": "测试任务"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def upload(self, workspace_task_id: str) -> dict:
        response = self.client.post(
            f"/v1/annotation/workspace-tasks/{workspace_task_id}/assets",
            files={"file": ("site.png", png_bytes(), "image/png")},
            data={"source_id": "site.png"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_task_asset_and_prompt_survive_restart(self):
        task = self.create_task()
        task_id = task["workspace_task_id"]
        item = self.upload(task_id)
        asset_id = item["asset"]["asset_id"]

        updated = self.client.patch(
            f"/v1/annotation/workspace-tasks/{task_id}/assets/{asset_id}",
            json={"prompt": "person. helmet."},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["prompt"], "person. helmet.")

        self.context.__exit__(None, None, None)
        self.app = create_app(settings(), storage=AnnotationStore(self.root))
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()

        tasks = self.client.get("/v1/annotation/workspace-tasks")
        self.assertEqual(tasks.status_code, 200, tasks.text)
        self.assertEqual(tasks.json()["total"], 1)
        self.assertEqual(tasks.json()["items"][0]["item_count"], 1)

        items = self.client.get(f"/v1/annotation/workspace-tasks/{task_id}/assets")
        self.assertEqual(items.status_code, 200, items.text)
        self.assertEqual(items.json()[0]["prompt"], "person. helmet.")
        self.assertEqual(
            items.json()[0]["asset"]["metadata"]["original_filename"], "site.png"
        )
        content = self.client.get(items.json()[0]["asset"]["content_url"])
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, png_bytes())

    def test_remove_persists_queue_membership(self):
        task = self.create_task()
        task_id = task["workspace_task_id"]
        item = self.upload(task_id)
        asset_id = item["asset"]["asset_id"]

        removed = self.client.delete(
            f"/v1/annotation/workspace-tasks/{task_id}/assets/{asset_id}"
        )
        self.assertEqual(removed.status_code, 204, removed.text)
        self.assertEqual(
            self.client.get(f"/v1/annotation/workspace-tasks/{task_id}/assets").json(),
            [],
        )
        self.assertEqual(
            self.client.get(f"/v1/annotation/assets/{asset_id}").status_code,
            200,
        )

    def test_upload_requires_existing_task(self):
        response = self.client.post(
            "/v1/annotation/workspace-tasks/wst_missing/assets",
            files={"file": ("site.png", png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 404, response.text)


if __name__ == "__main__":
    unittest.main()
