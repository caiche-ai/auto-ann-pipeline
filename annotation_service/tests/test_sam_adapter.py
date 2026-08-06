import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from annotation_service.sam_adapter import SAMAdapter, SAMModelConfig


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def argmax(self, dim):
        return FakeTensor(np.argmax(self.value, axis=dim))

    def __getitem__(self, item):
        return FakeTensor(self.value[item])

    def item(self):
        return self.value.item()

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    float32 = np.float32

    @staticmethod
    def as_tensor(value, *, dtype, device):
        del device
        return FakeTensor(np.asarray(value, dtype=dtype))


class FakeTransform:
    def __init__(self):
        self.calls = 0

    def apply_boxes_torch(self, boxes, image_shape):
        self.calls += 1
        self.image_shape = image_shape
        return boxes


class FakePredictor:
    device = "cpu"

    def __init__(self):
        self.set_image_calls = 0
        self.predict_torch_calls = 0
        self.transform = FakeTransform()

    def set_image(self, image, image_format):
        self.set_image_calls += 1
        self.features = object()
        self.original_size = image.shape[:2]
        self.input_size = image.shape[:2]
        self.is_image_set = True
        self.image_format = image_format

    def predict_torch(
        self,
        *,
        point_coords,
        point_labels,
        boxes,
        multimask_output,
        return_logits,
    ):
        del point_coords, point_labels, multimask_output, return_logits
        self.predict_torch_calls += 1
        batch_size = boxes.shape[0]
        masks = np.zeros((batch_size, 3, 10, 10), dtype=bool)
        masks[:, 1, 2:8, 2:8] = True
        scores = np.tile(
            np.asarray([[0.1, 0.9, 0.2]], dtype=np.float32),
            (batch_size, 1),
        )
        return FakeTensor(masks), FakeTensor(scores), None


class SAMAdapterTest(unittest.TestCase):
    def test_batches_box_decoder_and_reuses_image_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "asset.png"
            Image.new("RGB", (10, 10), (20, 30, 40)).save(image_path)
            adapter = SAMAdapter(
                SAMModelConfig(
                    checkpoint_path=Path(directory) / "unused.pth",
                    image_embedding_cache_size=2,
                )
            )
            predictor = FakePredictor()
            adapter._predictor = predictor
            real_import = importlib.import_module

            def import_module(name):
                if name == "torch":
                    return FakeTorch
                return real_import(name)

            with (
                patch(
                    "annotation_service.sam_adapter.importlib.import_module",
                    side_effect=import_module,
                ),
                patch(
                    "annotation_service.sam_adapter._mask_to_shapes",
                    return_value=[
                        {
                            "shape_id": "sam-target-1",
                            "label": "target",
                            "shape_type": "polygon",
                            "points": [
                                [2, 2],
                                [7, 2],
                                [7, 7],
                                [2, 7],
                            ],
                        }
                    ],
                ),
            ):
                first = adapter.predict_many(
                    image_path=image_path,
                    boxes_xyxy=[
                        [1, 1, 5, 9],
                        [5, 1, 9, 9],
                    ],
                )
                second = adapter.predict_many(
                    image_path=image_path,
                    boxes_xyxy=[[2, 2, 8, 8]],
                )

        self.assertEqual(predictor.set_image_calls, 1)
        self.assertEqual(predictor.predict_torch_calls, 2)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            first[0].timings_ms["decoder_mode"],
            "batched_predict_torch",
        )
        self.assertFalse(first[0].timings_ms["embedding_cache_hit"])
        self.assertTrue(second[0].timings_ms["embedding_cache_hit"])


if __name__ == "__main__":
    unittest.main()
