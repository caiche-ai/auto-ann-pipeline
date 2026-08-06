import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.app import create_app
from annotation_service.config import Settings
from annotation_service.storage import AnnotationStore


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), (10, 20, 30)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


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


def complete_annotation(target: str = "画面中央的一名作业人员") -> dict:
    return {
        "target_object": target,
        "instance_count": 1,
        "visual_anchor": ["画面中央"],
        "mask_granularity": "人员整体",
        "risk_semantics": "头部防护缺失",
        "shapes": [
            {
                "shape_id": "shape-1",
                "label": "target",
                "shape_type": "polygon",
                "points": [[1, 1], [8, 1], [8, 8], [1, 8]],
            }
        ],
        "prompts": [
            {"prompt_id": "v1", "type": "visual", "text": "分割中央人员。"},
            {"prompt_id": "v2", "type": "visual", "text": "标出深色上衣人员。"},
            {"prompt_id": "v3", "type": "visual", "text": "提取画面中的目标人员。"},
            {"prompt_id": "r1", "type": "risk", "text": "分割未戴安全帽的人员。"},
            {"prompt_id": "r2", "type": "risk", "text": "标出头部防护缺失人员。"},
            {"prompt_id": "a1", "type": "agent", "text": "找出并分割违规人员。"},
        ],
    }


class TaskApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.client_context = TestClient(
            create_app(make_settings(), storage=self.store)
        )
        self.client = self.client_context.__enter__()
        asset_response = self.client.post(
            "/v1/annotation/assets",
            files={"file": ("sample.png", png_bytes(), "image/png")},
            data={"group_id": "site01:video03"},
        )
        self.asset_id = asset_response.json()["asset_id"]
        job = self.store.create_job(
            asset_ids=[self.asset_id],
            requested_categories=["helmet_missing"],
            pipeline_version="grounded-qwen-v1",
        )
        self.job_id = job["job_id"]
        self.task = self.store.create_task(
            job_id=self.job_id,
            asset_id=self.asset_id,
            category="helmet_missing",
            annotation=complete_annotation(),
            provenance={"pipeline_version": "grounded-qwen-v1"},
        )

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_list_detail_draft_submit_and_review(self):
        listed = self.client.get(
            "/v1/annotation/tasks",
            params={
                "job_id": self.job_id,
                "category": "helmet_missing",
                "status": "generated",
            },
        )
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(len(listed.json()["items"]), 1)
        self.assertEqual(
            listed.json()["items"][0]["task_id"],
            self.task["task_id"],
        )

        detail = self.client.get(
            f"/v1/annotation/tasks/{self.task['task_id']}"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        annotation = detail.json()["annotation"]
        annotation["visual_anchor"].append("穿深色上衣")
        draft = self.client.put(
            f"/v1/annotation/tasks/{self.task['task_id']}/draft",
            json={
                "expected_version": 1,
                "annotation": annotation,
                "editor_id": "annotator-1",
            },
        )
        self.assertEqual(draft.status_code, 200, draft.text)
        self.assertEqual(draft.json()["status"], "annotating")
        self.assertEqual(draft.json()["version"], 2)

        submitted = self.client.post(
            f"/v1/annotation/tasks/{self.task['task_id']}/submit",
            json={
                "expected_version": 2,
                "annotator_id": "annotator-1",
                "primary_result": "prompt_ok",
                "comment": "ready",
            },
        )
        self.assertEqual(submitted.status_code, 200, submitted.text)
        self.assertEqual(submitted.json()["status"], "review_pending")
        self.assertEqual(submitted.json()["version"], 3)

        accepted = self.client.post(
            f"/v1/annotation/tasks/{self.task['task_id']}/review",
            json={
                "expected_version": 3,
                "reviewer_id": "reviewer-1",
                "decision": "accept",
                "primary_result": "prompt_ok",
                "comment": "accepted",
            },
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()["status"], "accepted")
        self.assertEqual(accepted.json()["version"], 4)

        versions = self.client.get(
            f"/v1/annotation/tasks/{self.task['task_id']}/versions"
        )
        self.assertEqual(versions.status_code, 200, versions.text)
        self.assertEqual(
            [item["change_kind"] for item in versions.json()["items"]],
            ["generated", "draft", "submit", "review"],
        )
        self.assertEqual(
            versions.json()["items"][2]["comment"],
            "ready",
        )

        reviews = self.client.get(
            f"/v1/annotation/tasks/{self.task['task_id']}/reviews"
        )
        self.assertEqual(reviews.status_code, 200, reviews.text)
        self.assertEqual(len(reviews.json()["items"]), 1)
        self.assertEqual(
            reviews.json()["items"][0]["decision"],
            "accept",
        )

    def test_submit_rejects_incomplete_annotation(self):
        incomplete = complete_annotation()
        incomplete["shapes"] = []
        saved = self.client.put(
            f"/v1/annotation/tasks/{self.task['task_id']}/draft",
            json={
                "expected_version": 1,
                "annotation": incomplete,
                "editor_id": "annotator-1",
            },
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        submitted = self.client.post(
            f"/v1/annotation/tasks/{self.task['task_id']}/submit",
            json={
                "expected_version": 2,
                "annotator_id": "annotator-1",
                "primary_result": "mask_missing",
            },
        )
        self.assertEqual(submitted.status_code, 422, submitted.text)
        self.assertEqual(
            submitted.json()["code"],
            "annotation_validation_failed",
        )

    def test_generated_task_can_be_invalidated_with_audit_version(self):
        invalidated = self.client.post(
            f"/v1/annotation/tasks/{self.task['task_id']}/invalidate",
            json={
                "expected_version": 1,
                "actor_id": "spring-user-1",
                "reason": "检测框不是需要标注的目标",
            },
        )
        self.assertEqual(invalidated.status_code, 200, invalidated.text)
        self.assertEqual(invalidated.json()["status"], "rejected")
        self.assertEqual(invalidated.json()["version"], 2)
        self.assertEqual(
            invalidated.json()["primary_result"],
            "other",
        )
        versions = self.client.get(
            f"/v1/annotation/tasks/{self.task['task_id']}/versions"
        )
        self.assertEqual(
            versions.json()["items"][-1]["change_kind"],
            "invalidate",
        )
        self.assertEqual(
            versions.json()["items"][-1]["comment"],
            "作废：检测框不是需要标注的目标",
        )

    def test_prompt_enrichment_operation_is_queued_and_queryable(self):
        self.store.store_artifact(
            task_id=self.task["task_id"],
            artifact_type="mask-overlay",
            data=png_bytes(),
            media_type="image/png",
        )
        accepted = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/prompt-enrichments"
            ),
            json={"expected_version": 1},
        )
        self.assertEqual(accepted.status_code, 202, accepted.text)
        operation_id = accepted.json()["operation_id"]

        operation = self.client.get(
            f"/v1/annotation/operations/{operation_id}"
        )
        self.assertEqual(operation.status_code, 200, operation.text)
        self.assertEqual(
            operation.json()["operation_type"],
            "prompt_enrichment",
        )
        self.assertEqual(operation.json()["task_version"], 1)
        self.assertEqual(operation.json()["status"], "queued")

        duplicate = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/prompt-enrichments"
            ),
            json={"expected_version": 1},
        )
        self.assertEqual(duplicate.status_code, 202, duplicate.text)
        self.assertEqual(
            duplicate.json()["operation_id"],
            operation_id,
        )

    def test_mask_candidate_operation_is_queued(self):
        accepted = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/mask-candidates"
            ),
            json={
                "expected_version": 1,
                "box_xyxy": [1, 1, 9, 9],
            },
        )
        self.assertEqual(accepted.status_code, 202, accepted.text)
        operation = self.client.get(
            (
                "/v1/annotation/operations/"
                f"{accepted.json()['operation_id']}"
            )
        )
        self.assertEqual(operation.status_code, 200)
        self.assertEqual(
            operation.json()["operation_type"],
            "mask_candidate",
        )

        outside = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/mask-candidates"
            ),
            json={
                "expected_version": 1,
                "box_xyxy": [1, 1, 11, 9],
            },
        )
        self.assertEqual(outside.status_code, 422)

    def test_queued_operation_can_be_cancelled(self):
        accepted = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/mask-candidates"
            ),
            json={
                "expected_version": 1,
                "box_xyxy": [1, 1, 9, 9],
            },
        )
        operation_id = accepted.json()["operation_id"]
        cancelled = self.client.post(
            f"/v1/annotation/operations/{operation_id}/cancel",
            json={
                "actor_id": "spring-user-1",
                "reason": "用户返回上一步",
            },
        )
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertEqual(
            cancelled.json()["result"]["reason"],
            "用户返回上一步",
        )
        repeated = self.client.post(
            f"/v1/annotation/operations/{operation_id}/cancel",
            json={
                "actor_id": "spring-user-1",
                "reason": "重复取消",
            },
        )
        self.assertEqual(repeated.status_code, 409, repeated.text)

    def test_prompt_enrichment_requires_mask_and_current_version(self):
        missing_mask = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/prompt-enrichments"
            ),
            json={"expected_version": 1},
        )
        self.assertEqual(missing_mask.status_code, 422)

        self.store.store_artifact(
            task_id=self.task["task_id"],
            artifact_type="mask",
            data=png_bytes(),
            media_type="image/png",
        )
        stale = self.client.post(
            (
                f"/v1/annotation/tasks/{self.task['task_id']}"
                "/prompt-enrichments"
            ),
            json={"expected_version": 2},
        )
        self.assertEqual(stale.status_code, 409)

    def test_submit_rejects_duplicate_ids_and_boundary_coordinates(self):
        invalid = complete_annotation()
        invalid["shapes"].append(
            {
                "shape_id": "shape-1",
                "label": "ignore",
                "shape_type": "polygon",
                "points": [[0, 0], [10, 0], [10, 2], [0, 2]],
            }
        )
        invalid["prompts"][1]["prompt_id"] = "v1"
        saved = self.client.put(
            f"/v1/annotation/tasks/{self.task['task_id']}/draft",
            json={
                "expected_version": 1,
                "annotation": invalid,
                "editor_id": "annotator-1",
            },
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        submitted = self.client.post(
            f"/v1/annotation/tasks/{self.task['task_id']}/submit",
            json={
                "expected_version": 2,
                "annotator_id": "annotator-1",
                "primary_result": "prompt_ok",
            },
        )
        self.assertEqual(submitted.status_code, 422, submitted.text)
        reasons = {
            item["reason"] for item in submitted.json()["details"]
        }
        self.assertIn("shape_id values must be unique", reasons)
        self.assertIn("prompt_id values must be unique", reasons)
        self.assertIn(
            "point must be inside the 10x10 image",
            reasons,
        )

    def test_task_cursor_and_artifact_download(self):
        second = self.store.create_task(
            job_id=self.job_id,
            asset_id=self.asset_id,
            category="helmet_missing",
            annotation=complete_annotation("另一名作业人员"),
            provenance={"pipeline_version": "grounded-qwen-v1"},
        )
        first_page = self.client.get(
            "/v1/annotation/tasks",
            params={"limit": 1},
        )
        self.assertEqual(first_page.status_code, 200, first_page.text)
        cursor = first_page.json()["next_cursor"]
        self.assertIsNotNone(cursor)
        second_page = self.client.get(
            "/v1/annotation/tasks",
            params={"limit": 1, "cursor": cursor},
        )
        self.assertEqual(second_page.status_code, 200, second_page.text)
        task_ids = {
            first_page.json()["items"][0]["task_id"],
            second_page.json()["items"][0]["task_id"],
        }
        self.assertEqual(task_ids, {self.task["task_id"], second["task_id"]})

        self.store.store_artifact(
            task_id=self.task["task_id"],
            artifact_type="mask",
            data=png_bytes(),
            media_type="image/png",
            width=10,
            height=10,
        )
        artifact = self.client.get(
            f"/v1/annotation/tasks/{self.task['task_id']}/artifacts/mask"
        )
        self.assertEqual(artifact.status_code, 200, artifact.text)
        self.assertEqual(artifact.headers["content-type"], "image/png")
        self.assertEqual(artifact.content, png_bytes())

        invalid_cursor = self.client.get(
            "/v1/annotation/tasks",
            params={"cursor": "not-a-valid-cursor"},
        )
        self.assertEqual(invalid_cursor.status_code, 422)

    def test_task_routes_require_authentication(self):
        secured = TestClient(
            create_app(
                make_settings(api_key="secret"),
                storage=self.store,
            )
        )
        response = secured.get(
            f"/v1/annotation/tasks/{self.task['task_id']}"
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
