from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .errors import AnnotationValidationError


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def normalize_request_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized[:128] or None


def validate_group_id(value: str) -> str:
    normalized = value.strip()
    if not SAFE_IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            "group_id must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_', ':', or '-'"
        )
    return normalized


def parse_metadata_json(
    value: str | None,
    *,
    max_chars: int,
) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}
    if len(value) > max_chars:
        raise ValueError(
            f"metadata_json exceeds the {max_chars}-character limit"
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must encode a JSON object")
    return parsed


def _polygon_area(points: list[list[float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def validate_annotation_for_submission(
    annotation: dict[str, Any],
    *,
    width: int,
    height: int,
    category: str,
) -> None:
    """Validate invariants required before review or acceptance.

    Drafts deliberately remain permissive. This validator is called only at
    submit time and again before an accepting review so generated candidates
    can never bypass the human-review contract.
    """

    details: list[dict[str, str]] = []
    target_object = str(annotation.get("target_object", "")).strip()
    mask_granularity = str(
        annotation.get("mask_granularity", "")
    ).strip()
    if not target_object:
        details.append(
            {
                "field": "annotation.target_object",
                "reason": "target_object must identify a concrete object",
            }
        )
    if not mask_granularity:
        details.append(
            {
                "field": "annotation.mask_granularity",
                "reason": "mask_granularity must not be blank",
            }
        )
    if category in {"safe", "unsafe"} and target_object in {
        "安全区域",
        "危险区域",
        "不安全目标",
        "安全目标",
    }:
        details.append(
            {
                "field": "annotation.target_object",
                "reason": (
                    "safe/unsafe must be concretized to a visible, "
                    "segmentable object"
                ),
            }
        )

    shapes = annotation.get("shapes", [])
    shape_ids = [str(shape.get("shape_id", "")).strip() for shape in shapes]
    if any(not shape_id for shape_id in shape_ids):
        details.append(
            {
                "field": "annotation.shapes",
                "reason": "shape_id must not be blank",
            }
        )
    if len(set(shape_ids)) != len(shape_ids):
        details.append(
            {
                "field": "annotation.shapes",
                "reason": "shape_id values must be unique",
            }
        )

    target_shapes = []
    for shape_index, shape in enumerate(shapes):
        if shape.get("label") == "target":
            target_shapes.append(shape)
        points = shape.get("points", [])
        for point_index, point in enumerate(points):
            if (
                len(point) != 2
                or point[0] < 0
                or point[1] < 0
                or point[0] >= width
                or point[1] >= height
            ):
                details.append(
                    {
                        "field": (
                            f"annotation.shapes.{shape_index}.points."
                            f"{point_index}"
                        ),
                        "reason": (
                            f"point must be inside the {width}x{height} image"
                        ),
                    }
                )
        if len(points) >= 3 and _polygon_area(points) <= 0:
            details.append(
                {
                    "field": f"annotation.shapes.{shape_index}.points",
                    "reason": "polygon must have non-zero area",
                }
            )
    if not target_shapes:
        details.append(
            {
                "field": "annotation.shapes",
                "reason": "at least one target polygon is required",
            }
        )

    prompts = annotation.get("prompts", [])
    prompt_ids = [
        str(item.get("prompt_id", "")).strip() for item in prompts
    ]
    if any(not prompt_id for prompt_id in prompt_ids):
        details.append(
            {
                "field": "annotation.prompts",
                "reason": "prompt_id must not be blank",
            }
        )
    if len(set(prompt_ids)) != len(prompt_ids):
        details.append(
            {
                "field": "annotation.prompts",
                "reason": "prompt_id values must be unique",
            }
        )
    counts = Counter(str(item.get("type", "")) for item in prompts)
    expected = {"visual": 3, "risk": 2, "agent": 1}
    for prompt_type, expected_count in expected.items():
        actual = counts.get(prompt_type, 0)
        if actual != expected_count:
            details.append(
                {
                    "field": "annotation.prompts",
                    "reason": (
                        f"expected {expected_count} {prompt_type} prompts, "
                        f"got {actual}"
                    ),
                }
            )
    normalized_prompts = [
        str(item.get("text", "")).strip() for item in prompts
    ]
    if any(not text for text in normalized_prompts):
        details.append(
            {
                "field": "annotation.prompts",
                "reason": "prompt text must not be blank",
            }
        )
    if len(set(normalized_prompts)) != len(normalized_prompts):
        details.append(
            {
                "field": "annotation.prompts",
                "reason": "prompt texts must be unique",
            }
        )

    if details:
        raise AnnotationValidationError(
            "annotation payload is invalid",
            details=details,
        )
