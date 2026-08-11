from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from ..review.workspace import load_review_submissions


COLORS = {
    "target": ((25, 185, 90, 90), (0, 125, 55, 255)),
    "ignore": ((235, 75, 65, 75), (185, 35, 30, 255)),
}
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


def default_review_workspace() -> Path:
    configured = os.getenv("ANNOTATION_REVIEW_WORKSPACE")
    return Path(configured or "./review_workspace").expanduser().resolve()


def find_review_font(explicit: str | Path | None = None) -> Path:
    configured = explicit or os.getenv("ANNOTATION_REVIEW_FONT")
    candidates = ([str(configured)] if configured else []) + list(FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "no Chinese font found; set --font or ANNOTATION_REVIEW_FONT"
    )


def _source_path(storage_root: Path, relative: str) -> Path:
    candidate = (storage_root / relative).resolve()
    try:
        candidate.relative_to(storage_root.resolve())
    except ValueError as exc:
        raise ValueError("review snapshot image escapes storage root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in (text.splitlines() or [""]):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and font.getlength(candidate) > width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def _render_polygons(image: Image.Image, shapes: list[dict[str, Any]]) -> Image.Image:
    image = image.convert("RGBA")
    fill_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer, "RGBA")
    valid_shapes: list[tuple[list[tuple[float, float]], tuple[int, ...]]] = []
    for shape in shapes:
        points = [tuple(map(float, point)) for point in shape["points"]]
        if len(points) < 3:
            continue
        fill, outline = COLORS.get(
            shape.get("label"),
            ((70, 130, 230, 70), (40, 90, 190, 255)),
        )
        fill_draw.polygon(points, fill=fill)
        valid_shapes.append((points, outline))
    image = Image.alpha_composite(image, fill_layer)
    outline_draw = ImageDraw.Draw(image, "RGBA")
    for points, outline in valid_shapes:
        outline_draw.line(points + [points[0]], fill=outline, width=4)
        for x, y in points:
            radius = 4
            outline_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=outline,
            )
    return image


def _detail_lines(submission: dict[str, Any]) -> list[tuple[str, str]]:
    annotation = submission["annotation"]
    details = [
        ("任务", f"{submission['task_id']}  v{submission['task_version']}"),
        ("标注人员", str(submission["annotator_id"])),
        ("类别", str(submission["category"])),
        ("目标", str(annotation.get("target_object") or "-")),
        ("实例数", str(annotation.get("instance_count") or "-")),
        ("标注粒度", str(annotation.get("mask_granularity") or "-")),
    ]
    if annotation.get("risk_semantics"):
        details.append(("风险语义", str(annotation["risk_semantics"])))
    return details


def render_annotation_result(
    *,
    storage_root: Path,
    submission: dict[str, Any],
    destination: Path,
    font_path: Path,
    max_image_width: int = 1400,
) -> Path:
    source = _source_path(storage_root, submission["asset"]["image_path"])
    with Image.open(source) as opened:
        annotated = _render_polygons(
            opened,
            submission["annotation"].get("shapes", []),
        )
    if annotated.width > max_image_width:
        ratio = max_image_width / annotated.width
        annotated = annotated.resize(
            (max_image_width, max(1, round(annotated.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    canvas_width = max(900, annotated.width)
    padding = 24
    title_font = ImageFont.truetype(str(font_path), 30)
    body_font = ImageFont.truetype(str(font_path), 22)
    small_font = ImageFont.truetype(str(font_path), 19)
    line_gap = 9
    content_width = canvas_width - padding * 2
    blocks: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]] = []
    blocks.append(("标注结果审核图", title_font, (20, 30, 45)))
    for label, value in _detail_lines(submission):
        blocks.append((f"{label}：{value}", body_font, (40, 50, 65)))
    blocks.append(("Prompt", title_font, (20, 30, 45)))
    prompts = submission["annotation"].get("prompts", [])
    for index, prompt in enumerate(prompts, start=1):
        prompt_type = str(prompt.get("type", "prompt")).upper()
        blocks.append(
            (
                f"{index}. [{prompt_type}] {prompt['text']}",
                small_font,
                (35, 65, 105),
            )
        )
    if submission.get("comment"):
        blocks.append(
            (f"提交备注：{submission['comment']}", small_font, (105, 70, 30))
        )

    rendered_blocks: list[tuple[list[str], ImageFont.FreeTypeFont, tuple[int, ...]]] = []
    panel_height = padding
    for text, font, color in blocks:
        lines = _wrap_text(text, font, content_width)
        rendered_blocks.append((lines, font, color))
        line_height = font.getbbox("国Ag")[3] - font.getbbox("国Ag")[1]
        panel_height += len(lines) * (line_height + line_gap) + 7
    panel_height += padding

    canvas = Image.new(
        "RGB",
        (canvas_width, annotated.height + panel_height),
        (245, 247, 250),
    )
    canvas.paste(
        annotated.convert("RGB"),
        ((canvas_width - annotated.width) // 2, 0),
    )
    draw = ImageDraw.Draw(canvas)
    draw.line(
        (0, annotated.height, canvas_width, annotated.height),
        fill=(190, 198, 210),
        width=2,
    )
    y = annotated.height + padding
    for lines, font, color in rendered_blocks:
        line_height = font.getbbox("国Ag")[3] - font.getbbox("国Ag")[1]
        for line in lines:
            draw.text((padding, y), line, font=font, fill=color)
            y += line_height + line_gap
        y += 7
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return destination


def render_annotation_results(
    *,
    storage_root: Path,
    output_root: Path | None = None,
    task_ids: Iterable[str] | None = None,
    latest_only: bool = True,
    font_path: str | Path | None = None,
    max_image_width: int = 1400,
) -> list[Path]:
    output = output_root or default_review_workspace() / "outputs"
    font = find_review_font(font_path)
    submissions = load_review_submissions(
        storage_root / "submissions",
        task_ids=task_ids,
        latest_only=latest_only,
    )
    rendered: list[Path] = []
    for _, submission in submissions:
        task_id = submission["task_id"]
        version = int(submission["task_version"])
        destination = output / f"{task_id}__v{version:08d}__annotated.png"
        rendered.append(
            render_annotation_result(
                storage_root=storage_root,
                submission=submission,
                destination=destination,
                font_path=font,
                max_image_width=max_image_width,
            )
        )
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate self-contained annotation images with polygons and prompts",
    )
    parser.add_argument(
        "--storage-root",
        default=os.getenv("ANNOTATION_STORAGE_ROOT", "./annotation-data"),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--all-versions", action="store_true")
    parser.add_argument("--font")
    parser.add_argument("--max-image-width", type=int, default=1400)
    args = parser.parse_args()
    if args.max_image_width < 320:
        parser.error("--max-image-width must be at least 320")
    storage_root = Path(args.storage_root).expanduser().resolve()
    rendered = render_annotation_results(
        storage_root=storage_root,
        output_root=(
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else None
        ),
        task_ids=args.task_ids,
        latest_only=not args.all_versions,
        font_path=args.font,
        max_image_width=args.max_image_width,
    )
    print(
        json.dumps(
            {"rendered": len(rendered), "images": [str(path) for path in rendered]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
