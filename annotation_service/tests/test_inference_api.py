import base64
import os
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.inference.app import create_app
from annotation_service.pipeline.grounding_dino.adapter import (
    GroundingDINODetection,
)
from annotation_service.pipeline.sam.adapter import SAMMaskCandidate


def png_bytes(color=(10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (10, 10), color).save(output, format="PNG")
    return output.getvalue()


class FakeDINO:
    model_version = "dino-test"
    prompt_version = "prompt-test"

    def load(self):
        return None

    def predict(self, *, prepared_prompt, **kwargs):
        del kwargs
        return [
            GroundingDINODetection(
                entity=prepared_prompt.caption.rstrip(" ."),
                box_xyxy=(1, 1, 9, 9),
                box_score=0.9,
                phrase_score=0.8,
            )
        ]


class FakeSAMConfig:
    model_version = "sam-test"


class FakeSAM:
    config = FakeSAMConfig()

    def warmup(self):
        return None

    def predict_many(self, *, boxes_xyxy, **kwargs):
        del kwargs
        artifact = png_bytes((255, 255, 255))
        return [
            SAMMaskCandidate(
                mask_png=artifact,
                overlay_png=artifact,
                crop_png=artifact,
                shapes=[],
                box_xyxy=box,
                predicted_iou=0.9,
                mask_area_pixels=64,
                model_version="sam-test",
            )
            for box in boxes_xyxy
        ]


class InferenceAPITest(unittest.TestCase):
    def setUp(self):
        environment = {"ANNOTATION_INFERENCE_PRELOAD": "false"}
        patcher = patch.dict(os.environ, environment)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(
            create_app(
                dino_predictor=FakeDINO(),
                sam_predictor=FakeSAM(),
                api_key="inference-secret",
            )
        )
        self.headers = {"X-Inference-API-Key": "inference-secret"}
        self.image = base64.b64encode(png_bytes()).decode("ascii")

    def test_requires_inference_api_key(self):
        response = self.client.post(
            "/v1/inference/sam",
            json={
                "image_base64": self.image,
                "media_type": "image/png",
                "boxes_xyxy": [[1, 1, 9, 9]],
            },
        )
        self.assertEqual(response.status_code, 401)

    def test_grounding_dino_contract(self):
        response = self.client.post(
            "/v1/inference/grounding-dino",
            headers=self.headers,
            json={
                "image_base64": self.image,
                "media_type": "image/png",
                "width": 10,
                "height": 10,
                "prepared_prompt": {
                    "caption": "person .",
                    "requested_entities": [],
                    "requested_prompt": "person",
                    "metadata": {},
                    "route": None,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"][0]["entity"], "person")

    def test_sam_contract(self):
        response = self.client.post(
            "/v1/inference/sam",
            headers=self.headers,
            json={
                "image_base64": self.image,
                "media_type": "image/png",
                "boxes_xyxy": [[1, 1, 9, 9]],
            },
        )
        self.assertEqual(response.status_code, 200)
        candidate = response.json()["candidates"][0]
        self.assertEqual(candidate["model_version"], "sam-test")
        self.assertTrue(base64.b64decode(candidate["mask_png_base64"]))


if __name__ == "__main__":
    unittest.main()
