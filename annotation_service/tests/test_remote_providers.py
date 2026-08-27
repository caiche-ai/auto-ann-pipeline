import base64
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from annotation_service.pipeline.grounding_dino.remote import (
    RemoteGroundingDINOPredictor,
)
from annotation_service.pipeline.grounding_dino.settings import (
    GroundingDINOWorkerSettings,
)
from annotation_service.pipeline.remote import RemoteProviderConfig
from annotation_service.pipeline.sam.remote import RemoteSAMProvider
from annotation_service.pipeline.sam.worker import SAMWorkerSettings


def png_bytes(color=(10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), color).save(output, format="PNG")
    return output.getvalue()


class RemoteGroundingDINOProviderTest(unittest.TestCase):
    def test_posts_prepared_prompt_and_parses_detections(self):
        calls = []

        def transport(url, headers, body, timeout):
            calls.append((url, headers, json.loads(body), timeout))
            return {
                "model_version": "remote-dino-v2",
                "prompt_version": "remote-prompt-v1",
                "detections": [
                    {
                        "entity": "person",
                        "box_xyxy": [1, 2, 8, 9],
                        "box_score": 0.91,
                        "phrase_score": 0.88,
                        "metadata": {"server": "gpu-1"},
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            image_path.write_bytes(png_bytes())
            provider = RemoteGroundingDINOPredictor(
                RemoteProviderConfig(
                    base_url="https://inference.example.test",
                    api_key="secret",
                    timeout_seconds=45,
                ),
                transport=transport,
            )
            detections = provider.predict(
                image_path=image_path,
                width=10,
                height=10,
                prompt="person",
            )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].entity, "person")
        self.assertEqual(detections[0].metadata["inference_provider"], "remote")
        self.assertEqual(provider.model_version, "remote-dino-v2")
        url, headers, payload, timeout = calls[0]
        self.assertEqual(
            url,
            "https://inference.example.test/v1/inference/grounding-dino",
        )
        self.assertEqual(headers["X-Inference-API-Key"], "secret")
        self.assertEqual(payload["prepared_prompt"]["caption"], "person .")
        self.assertEqual(timeout, 45)

    def test_remote_settings_do_not_require_local_model_paths(self):
        environment = {
            "ANNOTATION_STORAGE_ROOT": "./annotation-data",
            "ANNOTATION_GROUNDING_DINO_PROVIDER": "remote",
            "ANNOTATION_GROUNDING_DINO_REMOTE_BASE_URL": (
                "https://inference.example.test"
            ),
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = GroundingDINOWorkerSettings.from_env()
        settings.validate_model_files()
        self.assertEqual(settings.provider, "remote")
        self.assertIsNone(settings.checkpoint_path)


class RemoteSAMProviderTest(unittest.TestCase):
    def test_posts_boxes_and_parses_binary_artifacts(self):
        artifact = png_bytes((255, 255, 255))

        def transport(url, headers, body, timeout):
            del headers, timeout
            payload = json.loads(body)
            self.assertTrue(url.endswith("/v1/inference/sam"))
            self.assertEqual(payload["boxes_xyxy"], [[1, 1, 9, 9]])
            encoded = base64.b64encode(artifact).decode("ascii")
            return {
                "model_version": "sam-remote-v1",
                "candidates": [
                    {
                        "mask_png_base64": encoded,
                        "overlay_png_base64": encoded,
                        "crop_png_base64": encoded,
                        "shapes": [],
                        "box_xyxy": [1, 1, 9, 9],
                        "predicted_iou": 0.94,
                        "mask_area_pixels": 64,
                        "model_version": "sam-remote-v1",
                        "timings_ms": {"mask_decode_ms": 12.3},
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            image_path.write_bytes(png_bytes())
            provider = RemoteSAMProvider(
                RemoteProviderConfig(base_url="https://inference.example.test"),
                transport=transport,
            )
            candidates = provider.predict_many(
                image_path=image_path,
                boxes_xyxy=[[1, 1, 9, 9]],
            )

        self.assertEqual(candidates[0].mask_png, artifact)
        self.assertEqual(candidates[0].model_version, "sam-remote-v1")
        self.assertEqual(candidates[0].timings_ms["inference_provider"], "remote")

    def test_remote_settings_do_not_require_checkpoint(self):
        environment = {
            "ANNOTATION_STORAGE_ROOT": "./annotation-data",
            "ANNOTATION_SAM_PROVIDER": "remote",
            "ANNOTATION_SAM_REMOTE_BASE_URL": ("https://inference.example.test"),
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = SAMWorkerSettings.from_env()
        self.assertEqual(settings.provider, "remote")
        self.assertIsNone(settings.checkpoint_path)
        self.assertEqual(
            settings.remote_config().base_url,
            "https://inference.example.test",
        )


if __name__ == "__main__":
    unittest.main()
