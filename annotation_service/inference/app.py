from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, validator

from ..api.middleware import RequestBodyLimitMiddleware
from ..pipeline.grounding_dino.adapter import (
    GroundingDINOAdapter,
    GroundingDINOModelConfig,
    GroundingPromptPreparation,
)
from ..pipeline.sam.adapter import SAMAdapter, SAMModelConfig


LOGGER = logging.getLogger(__name__)


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"


class PreparedPromptRequest(StrictModel):
    caption: str = Field(..., min_length=1, max_length=2000)
    requested_entities: list[str] = Field(default_factory=list, max_items=100)
    requested_prompt: str = Field(..., min_length=1, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    route: dict[str, bool] | None = None


class GroundingDINOInferenceRequest(StrictModel):
    image_base64: str = Field(..., min_length=1)
    media_type: Literal["image/jpeg", "image/png"]
    width: int = Field(..., ge=1, le=100_000)
    height: int = Field(..., ge=1, le=100_000)
    prepared_prompt: PreparedPromptRequest


class SAMInferenceRequest(StrictModel):
    image_base64: str = Field(..., min_length=1)
    media_type: Literal["image/jpeg", "image/png"]
    boxes_xyxy: list[list[float]] = Field(..., min_items=1, max_items=128)

    @validator("boxes_xyxy")
    def boxes_are_valid(cls, value: list[list[float]]) -> list[list[float]]:
        for box in value:
            if len(box) != 4:
                raise ValueError("each box must contain four coordinates")
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box coordinates must have positive area")
        return value


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _model_path(root: Path, name: str, default: str) -> Path:
    configured = Path(os.getenv(name, default).strip()).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (root / configured).resolve()
    )


def _dino_model_config() -> GroundingDINOModelConfig:
    root = Path(_required("ANNOTATION_GROUNDING_DINO_ROOT")).expanduser()
    root = root.resolve()
    return GroundingDINOModelConfig(
        root=root,
        config_path=_model_path(
            root,
            "ANNOTATION_GROUNDING_DINO_CONFIG",
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        checkpoint_path=_model_path(
            root,
            "ANNOTATION_GROUNDING_DINO_CHECKPOINT",
            "weights/groundingdino_swint_ogc.pth",
        ),
        bert_path=_model_path(
            root,
            "ANNOTATION_GROUNDING_DINO_BERT",
            "weights/bert-base-uncased",
        ),
        device=os.getenv("ANNOTATION_GROUNDING_DINO_DEVICE", "cuda").strip(),
        model_version=os.getenv(
            "ANNOTATION_GROUNDING_DINO_MODEL_VERSION",
            "groundingdino-swint-ogc",
        ).strip(),
        prompt_version=os.getenv(
            "ANNOTATION_GROUNDING_DINO_PROMPT_VERSION", "free-form-v1"
        ).strip(),
        box_threshold=float(
            os.getenv("ANNOTATION_GROUNDING_DINO_BOX_THRESHOLD", "0.35")
        ),
        text_threshold=float(
            os.getenv("ANNOTATION_GROUNDING_DINO_TEXT_THRESHOLD", "0.25")
        ),
    )


def _sam_model_config() -> SAMModelConfig:
    return SAMModelConfig(
        checkpoint_path=Path(_required("ANNOTATION_SAM_CHECKPOINT"))
        .expanduser()
        .resolve(),
        model_type=os.getenv("ANNOTATION_SAM_MODEL_TYPE", "vit_h").strip(),
        device=os.getenv("ANNOTATION_SAM_DEVICE", "cuda").strip(),
        python_package=os.getenv(
            "ANNOTATION_SAM_PYTHON_PACKAGE",
            "third_party.segment_anything",
        ).strip(),
        model_version=os.getenv(
            "ANNOTATION_SAM_MODEL_VERSION", "sam-vit-h-4b8939"
        ).strip(),
        polygon_epsilon=float(os.getenv("ANNOTATION_SAM_POLYGON_EPSILON", "1.0")),
        image_embedding_cache_size=int(
            os.getenv("ANNOTATION_SAM_IMAGE_CACHE_SIZE", "2")
        ),
    )


def _validate_dino_config(config: GroundingDINOModelConfig) -> None:
    for label, path, directory in (
        ("root", config.root, True),
        ("config", config.config_path, False),
        ("checkpoint", config.checkpoint_path, False),
        ("BERT", config.bert_path, True),
    ):
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            raise FileNotFoundError(f"GroundingDINO {label} not found: {path}")
        if not directory and path.stat().st_size == 0:
            raise FileNotFoundError(f"GroundingDINO {label} is empty: {path}")
    for filename in (
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    ):
        path = config.bert_path / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"GroundingDINO BERT file missing: {path}")


@contextmanager
def _temporary_image(
    *,
    encoded: str,
    media_type: str,
    max_image_bytes: int,
) -> Iterator[Path]:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "image_base64 is invalid") from exc
    if not raw:
        raise HTTPException(422, "image is empty")
    if len(raw) > max_image_bytes:
        raise HTTPException(413, "image exceeds inference service limit")
    suffix = ".png" if media_type == "image/png" else ".jpg"
    with tempfile.TemporaryDirectory(prefix="annotation-inference-") as root:
        path = Path(root) / f"input{suffix}"
        path.write_bytes(raw)
        try:
            with Image.open(path) as image:
                expected = "PNG" if media_type == "image/png" else "JPEG"
                if image.format != expected:
                    raise HTTPException(422, "image content does not match media_type")
                image.verify()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, "image is invalid") from exc
        yield path


def create_app(
    *,
    dino_predictor=None,
    sam_predictor=None,
    api_key: str | None = None,
    max_image_bytes: int | None = None,
) -> FastAPI:
    configured_key = (
        api_key
        if api_key is not None
        else os.getenv("ANNOTATION_INFERENCE_API_KEY", "").strip() or None
    )
    image_limit = max_image_bytes or int(
        os.getenv("ANNOTATION_INFERENCE_MAX_IMAGE_BYTES", str(20 * 1024 * 1024))
    )
    dino_holder = {"value": dino_predictor}
    sam_holder = {"value": sam_predictor}
    dino_lock = threading.RLock()
    sam_lock = threading.RLock()

    def authenticate(
        x_inference_api_key: str | None = Header(
            default=None,
            alias="X-Inference-API-Key",
        ),
    ) -> None:
        if configured_key is not None and (
            x_inference_api_key is None
            or not hmac.compare_digest(configured_key, x_inference_api_key)
        ):
            raise HTTPException(401, "invalid inference API key")

    def get_dino():
        with dino_lock:
            if dino_holder["value"] is None:
                dino_holder["value"] = GroundingDINOAdapter(_dino_model_config())
            return dino_holder["value"]

    def get_sam():
        with sam_lock:
            if sam_holder["value"] is None:
                sam_holder["value"] = SAMAdapter(_sam_model_config())
            return sam_holder["value"]

    app = FastAPI(
        title="Annotation Model Inference API",
        version="1.0.0",
    )
    # Base64 expands the image by roughly 4/3; leave room for JSON, prompts,
    # boxes and headers while still rejecting unbounded request bodies before
    # FastAPI materializes them in memory.
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=(image_limit * 4 // 3) + 1024 * 1024,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authenticate)])
    def ready() -> dict[str, str]:
        # Configuration is checked without loading multi-gigabyte weights.
        _validate_dino_config(_dino_model_config())
        _sam_model_config().validate()
        return {"status": "ready"}

    @app.on_event("startup")
    def preload_models() -> None:
        enabled = os.getenv("ANNOTATION_INFERENCE_PRELOAD", "true").strip().lower()
        if enabled not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError("ANNOTATION_INFERENCE_PRELOAD must be a boolean")
        if enabled in {"0", "false", "no", "off"}:
            return
        with dino_lock:
            get_dino().load()
        with sam_lock:
            get_sam().warmup()
        LOGGER.info("GroundingDINO and SAM inference models preloaded")

    @app.post(
        "/v1/inference/grounding-dino",
        dependencies=[Depends(authenticate)],
    )
    def grounding_dino(
        request: GroundingDINOInferenceRequest,
    ) -> dict[str, Any]:
        with _temporary_image(
            encoded=request.image_base64,
            media_type=request.media_type,
            max_image_bytes=image_limit,
        ) as image_path:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
            if (request.width, request.height) != (
                actual_width,
                actual_height,
            ):
                raise HTTPException(422, "image dimensions do not match")
            prepared = GroundingPromptPreparation(
                caption=request.prepared_prompt.caption,
                requested_entities=tuple(request.prepared_prompt.requested_entities),
                requested_prompt=request.prepared_prompt.requested_prompt,
                metadata=request.prepared_prompt.metadata,
                route=request.prepared_prompt.route,
            )
            try:
                with dino_lock:
                    predictor = get_dino()
                    detections = predictor.predict(
                        image_path=image_path,
                        width=actual_width,
                        height=actual_height,
                        prepared_prompt=prepared,
                    )
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("remote GroundingDINO inference failed")
                raise HTTPException(503, "GroundingDINO inference failed") from exc
        return {
            "model_version": predictor.model_version,
            "prompt_version": predictor.prompt_version,
            "detections": [item.as_storage_payload() for item in detections],
        }

    @app.post(
        "/v1/inference/sam",
        dependencies=[Depends(authenticate)],
    )
    def sam(request: SAMInferenceRequest) -> dict[str, Any]:
        with _temporary_image(
            encoded=request.image_base64,
            media_type=request.media_type,
            max_image_bytes=image_limit,
        ) as image_path:
            try:
                with sam_lock:
                    predictor = get_sam()
                    candidates = predictor.predict_many(
                        image_path=image_path,
                        boxes_xyxy=request.boxes_xyxy,
                    )
            except HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("remote SAM inference failed")
                raise HTTPException(503, "SAM inference failed") from exc
        return {
            "model_version": predictor.config.model_version,
            "candidates": [
                {
                    "mask_png_base64": base64.b64encode(item.mask_png).decode("ascii"),
                    "overlay_png_base64": base64.b64encode(item.overlay_png).decode(
                        "ascii"
                    ),
                    "crop_png_base64": base64.b64encode(item.crop_png).decode("ascii"),
                    "shapes": item.shapes,
                    "box_xyxy": item.box_xyxy,
                    "predicted_iou": item.predicted_iou,
                    "mask_area_pixels": item.mask_area_pixels,
                    "model_version": item.model_version,
                    "timings_ms": item.timings_ms,
                }
                for item in candidates
            ],
        }

    return app


app = create_app()
