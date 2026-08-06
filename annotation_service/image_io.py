from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .errors import (
    RequestTooLargeError,
    UnsupportedMediaTypeError,
    ValidationServiceError,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}
JPEG_STANDALONE_MARKERS = {
    0x01,
    0xD8,
    0xD9,
    *range(0xD0, 0xD8),
}


@dataclass(frozen=True)
class ValidatedImage:
    raw: bytes
    image_format: str
    media_type: str
    width: int
    height: int


def detect_image_format(raw: bytes) -> str:
    if raw.startswith(PNG_SIGNATURE):
        return "png"
    if raw.startswith(JPEG_SIGNATURE):
        return "jpeg"
    raise UnsupportedMediaTypeError(
        "only JPEG and PNG images are supported"
    )


def _png_dimensions(raw: bytes) -> tuple[int, int]:
    if len(raw) < 24 or raw[12:16] != b"IHDR":
        raise ValidationServiceError("PNG header is invalid")
    return (
        int.from_bytes(raw[16:20], "big"),
        int.from_bytes(raw[20:24], "big"),
    )


def _jpeg_dimensions(raw: bytes) -> tuple[int, int]:
    position = 2
    while position < len(raw):
        if raw[position] != 0xFF:
            position += 1
            continue
        while position < len(raw) and raw[position] == 0xFF:
            position += 1
        if position >= len(raw):
            break
        marker = raw[position]
        position += 1
        if marker in JPEG_STANDALONE_MARKERS:
            continue
        if position + 2 > len(raw):
            break
        segment_length = int.from_bytes(
            raw[position : position + 2],
            "big",
        )
        if segment_length < 2 or position + segment_length > len(raw):
            break
        if marker in JPEG_SOF_MARKERS:
            if segment_length < 7:
                break
            height = int.from_bytes(
                raw[position + 3 : position + 5],
                "big",
            )
            width = int.from_bytes(
                raw[position + 5 : position + 7],
                "big",
            )
            return width, height
        if marker == 0xDA:
            break
        position += segment_length
    raise ValidationServiceError(
        "JPEG header does not contain valid dimensions"
    )


def encoded_image_dimensions(
    raw: bytes,
    image_format: str,
) -> tuple[int, int]:
    if image_format == "png":
        return _png_dimensions(raw)
    if image_format == "jpeg":
        return _jpeg_dimensions(raw)
    raise UnsupportedMediaTypeError(
        "only JPEG and PNG images are supported"
    )


def _validate_dimensions(
    width: int,
    height: int,
    *,
    max_image_pixels: int,
) -> None:
    if width <= 0 or height <= 0:
        raise ValidationServiceError("image dimensions are invalid")
    if width * height > max_image_pixels:
        raise ValidationServiceError(
            f"image exceeds {max_image_pixels} decoded pixels",
            details=[
                {
                    "field": "file",
                    "reason": (
                        f"decoded dimensions {width}x{height} exceed "
                        f"{max_image_pixels} pixels"
                    ),
                }
            ],
        )


def validate_image_bytes(
    raw: bytes,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
) -> ValidatedImage:
    if not raw:
        raise ValidationServiceError(
            "uploaded image is empty",
            details=[{"field": "file", "reason": "image is empty"}],
        )
    if len(raw) > max_image_bytes:
        raise RequestTooLargeError(
            f"image exceeds {max_image_bytes} bytes",
            details=[
                {
                    "field": "file",
                    "reason": (
                        f"decoded image has {len(raw)} bytes; maximum is "
                        f"{max_image_bytes}"
                    ),
                }
            ],
        )

    image_format = detect_image_format(raw)
    width, height = encoded_image_dimensions(raw, image_format)
    _validate_dimensions(
        width,
        height,
        max_image_pixels=max_image_pixels,
    )

    expected_pillow_format = "PNG" if image_format == "png" else "JPEG"
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != expected_pillow_format:
                raise ValidationServiceError(
                    "image content does not match its file signature"
                )
            if image.size != (width, height):
                raise ValidationServiceError(
                    "image dimensions do not match its encoded header"
                )
            if getattr(image, "n_frames", 1) != 1:
                raise ValidationServiceError(
                    "animated images are not supported"
                )
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            image.load()
    except ValidationServiceError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValidationServiceError(
            "uploaded JPEG or PNG image is invalid",
            details=[
                {
                    "field": "file",
                    "reason": "image decoding or integrity check failed",
                }
            ],
        ) from exc

    return ValidatedImage(
        raw=raw,
        image_format=image_format,
        media_type=(
            "image/png" if image_format == "png" else "image/jpeg"
        ),
        width=width,
        height=height,
    )
