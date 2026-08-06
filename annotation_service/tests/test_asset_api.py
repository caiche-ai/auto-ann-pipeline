import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.app import create_app
from annotation_service.config import Settings
from annotation_service.storage import AnnotationStore


def png_bytes(
    *,
    size: tuple[int, int] = (3, 2),
    color: tuple[int, int, int] = (10, 20, 30),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
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
        "docs_enabled": True,
        "storage_enabled": False,
        "storage_root": "./annotation-data",
    }
    values.update(overrides)
    return Settings(**values)


class AssetApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "annotation-data"
        self.store = AnnotationStore(self.root)
        self.app = create_app(
            make_settings(),
            storage=self.store,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def upload(
        self,
        raw: bytes | None = None,
        *,
        group_id: str = "site01:video03",
        metadata_json: str = '{"camera":"north"}',
        headers: dict[str, str] | None = None,
    ):
        return self.client.post(
            "/v1/annotation/assets",
            files={
                "file": (
                    "sample.png",
                    raw if raw is not None else png_bytes(),
                    "image/png",
                )
            },
            data={
                "source_id": "frontend-image-1",
                "group_id": group_id,
                "metadata_json": metadata_json,
            },
            headers=headers or {},
        )

    def test_upload_query_and_content(self):
        raw = png_bytes()
        created = self.upload(raw)
        self.assertEqual(created.status_code, 201, created.text)
        payload = created.json()
        self.assertEqual(payload["width"], 3)
        self.assertEqual(payload["height"], 2)
        self.assertEqual(payload["group_id"], "site01:video03")
        self.assertEqual(payload["metadata"], {"camera": "north"})
        self.assertIsNone(payload["duplicate_of"])

        detail = self.client.get(
            f"/v1/annotation/assets/{payload['asset_id']}"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json(), payload)

        content = self.client.get(payload["content_url"])
        self.assertEqual(content.status_code, 200, content.text)
        self.assertEqual(content.content, raw)
        self.assertEqual(content.headers["content-type"], "image/png")
        self.assertEqual(
            content.headers["etag"],
            f'"{payload["sha256"]}"',
        )

    def test_duplicate_without_idempotency_creates_linked_asset(self):
        first = self.upload().json()
        second_response = self.upload(group_id="site01:video04")
        self.assertEqual(second_response.status_code, 201)
        second = second_response.json()

        self.assertNotEqual(second["asset_id"], first["asset_id"])
        self.assertEqual(second["duplicate_of"], first["asset_id"])

    def test_idempotent_upload_returns_original_asset(self):
        headers = {"Idempotency-Key": "frontend-upload-001"}
        first = self.upload(headers=headers)
        second = self.upload(headers=headers)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(
            second.json()["asset_id"],
            first.json()["asset_id"],
        )

    def test_idempotency_conflict_does_not_store_second_image(self):
        headers = {"Idempotency-Key": "frontend-upload-002"}
        first = self.upload(headers=headers)
        conflict = self.upload(
            png_bytes(color=(200, 10, 10)),
            headers=headers,
        )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(
            conflict.json()["code"],
            "idempotency_conflict",
        )
        self.assertEqual(len(list(self.root.rglob("*.png"))), 1)

    def test_invalid_metadata_and_group_are_rejected(self):
        invalid_metadata = self.upload(metadata_json="[1,2,3]")
        self.assertEqual(invalid_metadata.status_code, 422)
        self.assertEqual(
            invalid_metadata.json()["details"][0]["field"],
            "metadata_json",
        )

        invalid_group = self.upload(group_id="../site")
        self.assertEqual(invalid_group.status_code, 422)
        self.assertEqual(
            invalid_group.json()["details"][0]["field"],
            "group_id",
        )

    def test_unsupported_corrupt_and_oversized_images(self):
        unsupported = self.upload(b"GIF89a")
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(
            unsupported.json()["code"],
            "unsupported_media_type",
        )

        corrupt = self.upload(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        )
        self.assertEqual(corrupt.status_code, 422)

        pixel_limit = self.upload(png_bytes(size=(50, 50)))
        self.assertEqual(pixel_limit.status_code, 422)

    def test_image_byte_limit_is_enforced(self):
        app = create_app(
            make_settings(
                max_request_bytes=4096,
                max_image_bytes=50,
            ),
            storage=AnnotationStore(
                Path(self.temporary.name) / "small-limit-data"
            ),
        )
        with TestClient(app) as client:
            response = client.post(
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
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(response.json()["code"], "request_too_large")

    def test_authentication_is_applied_to_all_asset_routes(self):
        app = create_app(
            make_settings(api_key="asset-secret"),
            storage=AnnotationStore(
                Path(self.temporary.name) / "auth-data"
            ),
        )
        with TestClient(app) as client:
            upload = client.post(
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
        self.assertEqual(upload.status_code, 401)

    def test_storage_disabled_returns_503(self):
        with TestClient(create_app(make_settings())) as client:
            response = client.post(
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
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "downstream_unavailable")


if __name__ == "__main__":
    unittest.main()
