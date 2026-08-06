import unittest
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from annotation_service.app import create_app
from annotation_service.config import Settings
from annotation_service.errors import ResourceNotFoundError


def settings(**overrides) -> Settings:
    values = {
        "service_version": "test-v1",
        "api_key": None,
        "cors_origins": (),
        "cors_allow_credentials": False,
        "max_request_bytes": 1024,
        "max_image_bytes": 512,
        "max_image_pixels": 1000,
        "max_metadata_chars": 100,
        "docs_enabled": True,
    }
    values.update(overrides)
    return Settings(**values)


class AnnotationAppTest(unittest.TestCase):
    def test_health_and_request_id(self):
        client = TestClient(create_app(settings()))
        response = client.get(
            "/health",
            headers={"X-Request-ID": "frontend-request"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(
            response.headers["x-request-id"],
            "frontend-request",
        )

    def test_readiness_reports_dependency_failure(self):
        client = TestClient(
            create_app(
                settings(),
                readiness_provider=lambda: {
                    "api": "ready",
                    "storage": "not_ready",
                },
            )
        )
        response = client.get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")

    def test_api_key_accepts_bearer_or_explicit_header(self):
        client = TestClient(create_app(settings(api_key="test-secret")))

        self.assertEqual(client.get("/ready").status_code, 401)
        self.assertEqual(
            client.get(
                "/ready",
                headers={"Authorization": "Bearer test-secret"},
            ).status_code,
            200,
        )
        self.assertEqual(
            client.get(
                "/ready",
                headers={"X-API-Key": "test-secret"},
            ).status_code,
            200,
        )

    def test_service_errors_use_stable_payload(self):
        app = create_app(settings())

        @app.get("/test/not-found")
        async def fail():
            raise ResourceNotFoundError("missing test resource")

        client = TestClient(app)
        response = client.get("/test/not-found")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "not_found")
        self.assertEqual(response.json()["details"], [])
        self.assertTrue(response.headers["x-request-id"])

    def test_router_not_found_uses_stable_payload(self):
        client = TestClient(create_app(settings()))
        response = client.get("/missing-route")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "not_found")
        self.assertEqual(response.json()["details"], [])

    def test_request_body_limit_rejects_before_routing(self):
        client = TestClient(
            create_app(
                settings(max_request_bytes=16, max_image_bytes=8)
            )
        )
        response = client.post(
            "/not-implemented",
            content=b"x" * 17,
            headers={"Content-Type": "application/octet-stream"},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["code"], "request_too_large")
        self.assertEqual(response.json()["details"], [])

    def test_docs_can_be_disabled(self):
        app = create_app(settings(docs_enabled=False))
        paths = {route.path for route in app.routes}

        self.assertNotIn("/docs", paths)
        self.assertNotIn("/redoc", paths)
        self.assertNotIn("/openapi.json", paths)

    def test_runtime_openapi_exposes_asset_contract(self):
        document = create_app(settings()).openapi()
        self.assertEqual(document["openapi"], "3.0.3")

        expected_operations = {
            ("/v1/annotation/assets", "post"): "createAsset",
            ("/v1/annotation/assets/{asset_id}", "get"): "getAsset",
            (
                "/v1/annotation/assets/{asset_id}/content",
                "get",
            ): "getAssetContent",
            (
                "/v1/annotation/jobs",
                "post",
            ): "createAnnotationJob",
            (
                "/v1/annotation/jobs/{job_id}",
                "get",
            ): "getAnnotationJob",
            (
                "/v1/annotation/jobs/{job_id}/detections",
                "get",
            ): "listAnnotationJobDetections",
            (
                "/v1/annotation/jobs/{job_id}/assets/"
                "{asset_id}/bbox-image",
                "get",
            ): "getAnnotationJobBoundingBoxImage",
            (
                "/v1/annotation/jobs/{job_id}/review-tasks",
                "post",
            ): "buildDetectionReviewTasks",
            (
                "/v1/annotation/tasks/{task_id}/mask-candidates",
                "post",
            ): "createMaskCandidate",
            (
                "/v1/annotation/tasks/{task_id}/prompt-enrichments",
                "post",
            ): "createPromptEnrichment",
            (
                "/v1/annotation/tasks/{task_id}/submit",
                "post",
            ): "submitAnnotationTask",
            (
                "/v1/annotation/tasks/{task_id}/invalidate",
                "post",
            ): "invalidateAnnotationTask",
        }
        for (path, method), operation_id in expected_operations.items():
            self.assertEqual(
                document["paths"][path][method]["operationId"],
                operation_id,
            )

        upload = document["paths"]["/v1/annotation/assets"]["post"]
        self.assertIn(
            "multipart/form-data",
            upload["requestBody"]["content"],
        )
        content = document["paths"][
            "/v1/annotation/assets/{asset_id}/content"
        ]["get"]["responses"]["200"]["content"]
        self.assertEqual(set(content), {"image/jpeg", "image/png"})
        self.assertEqual(
            set(document["components"]["securitySchemes"]),
            {"apiKeyAuth", "bearerAuth"},
        )
        self.assertEqual(
            upload["security"],
            [{"apiKeyAuth": []}, {"bearerAuth": []}],
        )
        self.assertIn("/v1/annotation/tasks", document["paths"])
        self.assertIn("/v1/annotation/releases", document["paths"])

    def test_storage_lifecycle_is_reflected_in_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = create_app(
                settings(
                    storage_enabled=True,
                    storage_root=str(Path(temporary) / "annotation-data"),
                )
            )
            store = app.state.storage
            self.assertEqual(store.readiness(), {"storage": "not_ready"})
            with TestClient(app) as client:
                response = client.get("/ready")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json()["dependencies"]["storage"],
                    "ready",
                )
            self.assertEqual(store.readiness(), {"storage": "not_ready"})


if __name__ == "__main__":
    unittest.main()
