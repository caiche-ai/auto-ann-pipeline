from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..remote import (
    RemoteProviderConfig,
    RemoteProviderError,
    RemoteTransport,
    decode_base64_field,
    default_remote_transport,
    image_payload,
    post_json,
)
from .adapter import SAMMaskCandidate


class RemoteSAMProvider:
    """SAM mask predictor backed by the standalone inference API."""

    def __init__(
        self,
        config: RemoteProviderConfig,
        *,
        model_version: str = "sam-vit-h-4b8939",
        polygon_epsilon: float = 1.0,
        transport: RemoteTransport | None = None,
    ):
        if polygon_epsilon < 0:
            raise ValueError("SAM polygon_epsilon must be non-negative")
        self.remote_config = config
        # SAMMaskWorker reads predictor.config.polygon_epsilon when combining
        # multiple masks, so expose a small compatible configuration object.
        self.config = _RemoteSAMRuntimeConfig(
            model_version=model_version,
            polygon_epsilon=polygon_epsilon,
        )
        self._transport = transport or default_remote_transport

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

    def predict_many(
        self,
        *,
        image_path: Path,
        boxes_xyxy: list[list[float]],
    ) -> list[SAMMaskCandidate]:
        if not boxes_xyxy:
            raise ValueError("boxes_xyxy must not be empty")
        media_type, encoded = image_payload(
            image_path,
            max_bytes=self.remote_config.max_image_bytes,
        )
        response = post_json(
            config=self.remote_config,
            path="/v1/inference/sam",
            payload={
                "image_base64": encoded,
                "media_type": media_type,
                "boxes_xyxy": boxes_xyxy,
            },
            transport=self._transport,
        )
        candidates = response.get("candidates")
        if not isinstance(candidates, list):
            raise RemoteProviderError("remote SAM response must contain candidates")
        if len(candidates) != len(boxes_xyxy):
            raise RemoteProviderError(
                "remote SAM candidate count does not match box count"
            )
        return [
            self._parse_candidate(item, expected_box=expected_box)
            for item, expected_box in zip(candidates, boxes_xyxy)
        ]

    @staticmethod
    def _parse_candidate(
        payload: Any,
        *,
        expected_box: list[float],
    ) -> SAMMaskCandidate:
        if not isinstance(payload, dict):
            raise RemoteProviderError("remote SAM candidate must be an object")
        try:
            box = [float(value) for value in payload["box_xyxy"]]
            predicted_iou = float(payload["predicted_iou"])
            mask_area_pixels = int(payload["mask_area_pixels"])
            model_version = str(payload["model_version"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteProviderError("remote SAM candidate is malformed") from exc
        shapes = payload.get("shapes")
        timings = payload.get("timings_ms", {})
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in box)
            or box != [float(value) for value in expected_box]
            or not math.isfinite(predicted_iou)
            or not 0 <= predicted_iou <= 1
            or mask_area_pixels < 1
            or not model_version
            or not isinstance(shapes, list)
            or not isinstance(timings, dict)
        ):
            raise RemoteProviderError("remote SAM candidate is malformed")
        mask_png = decode_base64_field(payload, "mask_png_base64")
        overlay_png = decode_base64_field(payload, "overlay_png_base64")
        crop_png = decode_base64_field(payload, "crop_png_base64")
        if not all(
            value.startswith(b"\x89PNG\r\n\x1a\n")
            for value in (mask_png, overlay_png, crop_png)
        ):
            raise RemoteProviderError("remote SAM artifacts must be PNG images")
        return SAMMaskCandidate(
            mask_png=mask_png,
            overlay_png=overlay_png,
            crop_png=crop_png,
            shapes=shapes,
            box_xyxy=box,
            predicted_iou=predicted_iou,
            mask_area_pixels=mask_area_pixels,
            model_version=model_version,
            timings_ms={**timings, "inference_provider": "remote"},
        )


class _RemoteSAMRuntimeConfig:
    def __init__(self, *, model_version: str, polygon_epsilon: float):
        self.model_version = model_version
        self.polygon_epsilon = polygon_epsilon
