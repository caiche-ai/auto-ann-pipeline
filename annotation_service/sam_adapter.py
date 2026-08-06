from __future__ import annotations

import importlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class SAMInferenceError(RuntimeError):
    """Raised when SAM cannot produce a usable mask."""


@dataclass(frozen=True)
class SAMModelConfig:
    checkpoint_path: Path
    model_type: str = "vit_h"
    device: str = "cuda"
    python_package: str = "model.segment_anything"
    model_version: str = "sam-vit-h-4b8939"
    polygon_epsilon: float = 1.0

    def validate(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.checkpoint_path}"
            )
        if self.checkpoint_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"SAM checkpoint is empty: {self.checkpoint_path}"
            )
        if self.model_type not in {"vit_h", "vit_l", "vit_b", "default"}:
            raise ValueError("unsupported SAM model_type")
        if not self.device.strip():
            raise ValueError("SAM device must not be blank")
        if self.polygon_epsilon < 0:
            raise ValueError("SAM polygon_epsilon must be non-negative")


@dataclass(frozen=True)
class SAMMaskCandidate:
    mask_png: bytes
    overlay_png: bytes
    crop_png: bytes
    shapes: list[dict[str, Any]]
    box_xyxy: list[float]
    predicted_iou: float
    mask_area_pixels: int
    model_version: str


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _mask_to_shapes(
    mask: np.ndarray,
    *,
    epsilon: float,
) -> list[dict[str, Any]]:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise SAMInferenceError(
            "OpenCV is required to convert SAM masks to polygons"
        ) from exc
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    shapes: list[dict[str, Any]] = []
    ordered = sorted(
        contours,
        key=lambda contour: float(cv2.contourArea(contour)),
        reverse=True,
    )
    for index, contour in enumerate(ordered, start=1):
        if float(cv2.contourArea(contour)) <= 0:
            continue
        approximated = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )
        points = [
            [float(point[0][0]), float(point[0][1])]
            for point in approximated
        ]
        if len(points) < 3:
            continue
        shapes.append(
            {
                "shape_id": f"sam-target-{index}",
                "label": "target",
                "shape_type": "polygon",
                "points": points,
            }
        )
    if not shapes:
        raise SAMInferenceError(
            "SAM mask has no polygon with positive area"
        )
    return shapes


def render_sam_candidate(
    *,
    image: Image.Image,
    mask: np.ndarray,
    box_xyxy: list[float],
    predicted_iou: float,
    model_version: str,
    polygon_epsilon: float,
) -> SAMMaskCandidate:
    rgb = image.convert("RGB")
    width, height = rgb.size
    boolean_mask = np.asarray(mask, dtype=bool)
    if boolean_mask.shape != (height, width):
        raise SAMInferenceError(
            "SAM mask dimensions do not match the source image"
        )
    area = int(boolean_mask.sum())
    if area <= 0:
        raise SAMInferenceError("SAM returned an empty mask")
    mask_image = Image.fromarray(
        boolean_mask.astype(np.uint8) * 255,
        mode="L",
    )
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    color = np.array([255, 64, 64], dtype=np.float32)
    overlay[boolean_mask] = (
        overlay[boolean_mask].astype(np.float32) * 0.45
        + color * 0.55
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGB")
    x1, y1, x2, y2 = box_xyxy
    crop = rgb.crop(
        (
            max(0, int(np.floor(x1))),
            max(0, int(np.floor(y1))),
            min(width, int(np.ceil(x2))),
            min(height, int(np.ceil(y2))),
        )
    )
    if crop.width < 1 or crop.height < 1:
        raise SAMInferenceError("SAM target crop is empty")
    return SAMMaskCandidate(
        mask_png=_png_bytes(mask_image),
        overlay_png=_png_bytes(overlay_image),
        crop_png=_png_bytes(crop),
        shapes=_mask_to_shapes(
            boolean_mask,
            epsilon=polygon_epsilon,
        ),
        box_xyxy=[float(value) for value in box_xyxy],
        predicted_iou=float(predicted_iou),
        mask_area_pixels=area,
        model_version=model_version,
    )


class SAMAdapter:
    """Lazily load Segment Anything and predict from one bounding box."""

    def __init__(self, config: SAMModelConfig):
        self.config = config
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        self.config.validate()
        package = importlib.import_module(self.config.python_package)
        registry = getattr(package, "sam_model_registry")
        predictor_class = getattr(package, "SamPredictor")
        model = registry[self.config.model_type](
            checkpoint=str(self.config.checkpoint_path)
        )
        model.to(device=self.config.device)
        self._predictor = predictor_class(model)
        return self._predictor

    def predict(
        self,
        *,
        image_path: Path,
        box_xyxy: list[float],
    ) -> SAMMaskCandidate:
        predictor = self._load()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image_array = np.asarray(image)
        predictor.set_image(image_array, image_format="RGB")
        masks, scores, _ = predictor.predict(
            box=np.asarray(box_xyxy, dtype=np.float32),
            multimask_output=True,
            return_logits=False,
        )
        if len(scores) == 0:
            raise SAMInferenceError("SAM returned no mask candidates")
        best = int(np.argmax(scores))
        return render_sam_candidate(
            image=image,
            mask=masks[best],
            box_xyxy=box_xyxy,
            predicted_iou=float(scores[best]),
            model_version=self.config.model_version,
            polygon_epsilon=self.config.polygon_epsilon,
        )
