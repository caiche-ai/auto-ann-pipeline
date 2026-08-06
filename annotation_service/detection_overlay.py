from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


def _color_for(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return tuple(64 + value % 160 for value in digest[:3])


def _drawable_text(value: str) -> str:
    """Keep rendering deterministic when Pillow's default font lacks glyphs."""

    return value.encode("ascii", errors="replace").decode("ascii")


def render_detection_overlay(
    *,
    image_path: Path,
    detections: Sequence[dict[str, Any]],
) -> bytes:
    """Render absolute xyxy GroundingDINO detections as a PNG image."""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 300))

    for detection in detections:
        box = detection.get("box_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("detection box_xyxy must contain four values")
        x1, y1, x2, y2 = [float(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detection box must have positive area")
        entity = str(detection.get("entity", "object")).strip() or "object"
        score = float(detection.get("box_score", 0.0))
        label = _drawable_text(f"{entity} {score:.3f}")
        color = _color_for(entity)
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width,
        )
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(0.0, min(x1, image.width - text_width - 4))
        label_y = max(0.0, y1 - text_height - 5)
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 4,
                label_y + text_height + 4,
            ),
            fill=color,
        )
        draw.text(
            (label_x + 2, label_y + 2),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
