from __future__ import annotations

import importlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


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
    image_embedding_cache_size: int = 2

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
        if not 0 <= self.image_embedding_cache_size <= 16:
            raise ValueError(
                "SAM image_embedding_cache_size must be between 0 and 16"
            )


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
    timings_ms: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SAMImageEmbedding:
    features: Any
    original_size: tuple[int, int]
    input_size: tuple[int, int]


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", compress_level=3)
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
    """Lazily load SAM and reuse one image embedding for one or more boxes."""

    def __init__(self, config: SAMModelConfig):
        self.config = config
        self._predictor = None
        self._image_embeddings: OrderedDict[
            tuple[str, int, int],
            _SAMImageEmbedding,
        ] = OrderedDict()

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

    def warmup(self) -> None:
        """Load model weights before the first user operation."""

        self._load()

    @staticmethod
    def _image_cache_key(image_path: Path) -> tuple[str, int, int]:
        resolved = image_path.resolve()
        stat = resolved.stat()
        return (str(resolved), stat.st_size, stat.st_mtime_ns)

    def _set_image(
        self,
        *,
        predictor,
        image_path: Path,
        image_array: np.ndarray,
    ) -> bool:
        key = self._image_cache_key(image_path)
        cached = self._image_embeddings.pop(key, None)
        if cached is not None:
            self._image_embeddings[key] = cached
            predictor.features = cached.features
            predictor.original_size = cached.original_size
            predictor.input_size = cached.input_size
            predictor.is_image_set = True
            return True

        predictor.set_image(image_array, image_format="RGB")
        if self.config.image_embedding_cache_size > 0:
            self._image_embeddings[key] = _SAMImageEmbedding(
                features=predictor.features,
                original_size=tuple(predictor.original_size),
                input_size=tuple(predictor.input_size),
            )
            while (
                len(self._image_embeddings)
                > self.config.image_embedding_cache_size
            ):
                self._image_embeddings.popitem(last=False)
        return False

    @staticmethod
    def _predict_boxes_together(
        *,
        predictor,
        boxes_xyxy: list[list[float]],
        image_shape: tuple[int, int],
    ) -> tuple[list[np.ndarray], list[float]]:
        torch = importlib.import_module("torch")
        box_tensor = torch.as_tensor(
            np.asarray(boxes_xyxy, dtype=np.float32),
            dtype=torch.float32,
            device=predictor.device,
        )
        transformed_boxes = predictor.transform.apply_boxes_torch(
            box_tensor,
            image_shape,
        )
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
            return_logits=False,
        )
        if scores.shape[-1] == 0:
            raise SAMInferenceError("SAM returned no mask candidates")
        best_indices = scores.argmax(dim=1)
        selected_masks: list[np.ndarray] = []
        selected_scores: list[float] = []
        for index in range(len(boxes_xyxy)):
            best = int(best_indices[index].item())
            selected_masks.append(
                masks[index, best].detach().cpu().numpy()
            )
            selected_scores.append(float(scores[index, best].item()))
        return selected_masks, selected_scores

    @staticmethod
    def _predict_boxes_sequentially(
        *,
        predictor,
        boxes_xyxy: list[list[float]],
    ) -> tuple[list[np.ndarray], list[float]]:
        selected_masks: list[np.ndarray] = []
        selected_scores: list[float] = []
        for box_xyxy in boxes_xyxy:
            masks, scores, _ = predictor.predict(
                box=np.asarray(box_xyxy, dtype=np.float32),
                multimask_output=True,
                return_logits=False,
            )
            if len(scores) == 0:
                raise SAMInferenceError(
                    "SAM returned no mask candidates"
                )
            best = int(np.argmax(scores))
            selected_masks.append(masks[best])
            selected_scores.append(float(scores[best]))
        return selected_masks, selected_scores

    def predict(
        self,
        *,
        image_path: Path,
        box_xyxy: list[float],
    ) -> SAMMaskCandidate:
        return self.predict_many(
            image_path=image_path,
            boxes_xyxy=[box_xyxy],
        )[0]

    def precompute(self, *, image_path: Path) -> dict[str, Any]:
        """Populate the image embedding cache before boxes are selected."""

        total_started = time.perf_counter()
        predictor = self._load()
        image_started = time.perf_counter()
        with Image.open(image_path) as source:
            image_array = np.asarray(source.convert("RGB"))
        image_read_ms = (time.perf_counter() - image_started) * 1000
        encode_started = time.perf_counter()
        cache_hit = self._set_image(
            predictor=predictor,
            image_path=image_path,
            image_array=image_array,
        )
        image_encode_ms = (time.perf_counter() - encode_started) * 1000
        timings = {
            "embedding_cache_hit": cache_hit,
            "image_read_ms": round(image_read_ms, 3),
            "image_encode_ms": round(image_encode_ms, 3),
            "total_ms": round(
                (time.perf_counter() - total_started) * 1000,
                3,
            ),
        }
        LOGGER.info(
            "SAM embedding precomputed: cache_hit=%s "
            "image_read_ms=%.1f image_encode_ms=%.1f total_ms=%.1f",
            cache_hit,
            image_read_ms,
            image_encode_ms,
            timings["total_ms"],
        )
        return timings

    def predict_many(
        self,
        *,
        image_path: Path,
        boxes_xyxy: list[list[float]],
    ) -> list[SAMMaskCandidate]:
        if not boxes_xyxy:
            raise ValueError("boxes_xyxy must not be empty")
        total_started = time.perf_counter()
        predictor = self._load()
        image_started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image_array = np.asarray(image)
        image_read_ms = (time.perf_counter() - image_started) * 1000
        encode_started = time.perf_counter()
        cache_hit = self._set_image(
            predictor=predictor,
            image_path=image_path,
            image_array=image_array,
        )
        image_encode_ms = (time.perf_counter() - encode_started) * 1000
        decode_started = time.perf_counter()
        if callable(getattr(predictor, "predict_torch", None)) and hasattr(
            predictor,
            "transform",
        ):
            selected_masks, selected_scores = (
                self._predict_boxes_together(
                    predictor=predictor,
                    boxes_xyxy=boxes_xyxy,
                    image_shape=image_array.shape[:2],
                )
            )
            decoder_mode = "batched_predict_torch"
        else:
            selected_masks, selected_scores = (
                self._predict_boxes_sequentially(
                    predictor=predictor,
                    boxes_xyxy=boxes_xyxy,
                )
            )
            decoder_mode = "sequential_fallback"
        mask_decode_ms = (time.perf_counter() - decode_started) * 1000
        candidates: list[SAMMaskCandidate] = []
        render_timings: list[float] = []
        for box_xyxy, mask, score in zip(
            boxes_xyxy,
            selected_masks,
            selected_scores,
        ):
            render_started = time.perf_counter()
            candidate = render_sam_candidate(
                image=image,
                mask=mask,
                box_xyxy=box_xyxy,
                predicted_iou=score,
                model_version=self.config.model_version,
                polygon_epsilon=self.config.polygon_epsilon,
            )
            render_timings.append(
                (time.perf_counter() - render_started) * 1000
            )
            candidates.append(candidate)
        total_ms = (time.perf_counter() - total_started) * 1000
        shared_timings = {
            "batch_size": len(boxes_xyxy),
            "embedding_cache_hit": cache_hit,
            "image_read_ms": round(image_read_ms, 3),
            "image_encode_ms": round(image_encode_ms, 3),
            "mask_decode_ms": round(mask_decode_ms, 3),
            "batch_total_ms": round(total_ms, 3),
            "decoder_mode": decoder_mode,
        }
        candidates = [
            replace(
                candidate,
                timings_ms={
                    **shared_timings,
                    "artifact_render_ms": round(render_ms, 3),
                },
            )
            for candidate, render_ms in zip(
                candidates,
                render_timings,
            )
        ]
        LOGGER.info(
            "SAM batch completed: boxes=%d cache_hit=%s mode=%s "
            "image_read_ms=%.1f image_encode_ms=%.1f "
            "mask_decode_ms=%.1f total_ms=%.1f",
            len(boxes_xyxy),
            cache_hit,
            decoder_mode,
            image_read_ms,
            image_encode_ms,
            mask_decode_ms,
            total_ms,
        )
        return candidates
