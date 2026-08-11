"""Sequential debugger for DINO -> SAM -> Qwen prompt generation."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TERMINAL_JOB_STATUSES = {
    "succeeded", "partial_failed", "failed", "cancelled"
}
TERMINAL_OPERATION_STATUSES = {
    "succeeded", "failed", "cancelled"
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> dict:
    request = Request(url, method=method, headers=headers or {}, data=body)
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"API unavailable: {exc.reason}") from exc


def _post_json(url: str, api_key: str, payload: dict) -> dict:
    return _json_request(
        url,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        },
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def _get_json(url: str, api_key: str) -> dict:
    return _json_request(url, headers={"X-API-Key": api_key})


def _multipart_image(
    path: Path,
    *,
    field: str = "file",
    fields: dict[str, str] | None = None,
) -> tuple[str, bytes]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if media_type not in {"image/jpeg", "image/png"}:
        raise ValueError("image must be JPEG or PNG")
    boundary = f"----annotation-debug-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in (fields or {}).items():
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(),
            b"\r\n",
        ])
    parts.extend([
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="{field}"; '
            f'filename="{path.name}"\r\n'
        ).encode(),
        f"Content-Type: {media_type}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _upload_asset(base_url: str, api_key: str, path: Path) -> dict:
    content_type, body = _multipart_image(
        path,
        fields={"group_id": "debug", "source_id": "debug-client"},
    )
    return _json_request(
        f"{base_url}/v1/annotation/assets",
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": content_type,
            "Idempotency-Key": f"debug-asset-{uuid.uuid4().hex}",
        },
        body=body,
    )


def _wait_for_api(base_url: str) -> None:
    for _ in range(120):
        try:
            _json_request(f"{base_url}/health")
            return
        except RuntimeError:
            time.sleep(0.5)
    raise RuntimeError("API did not become healthy within 60 seconds")


def _poll(
    fetch,
    *,
    terminal_statuses: set[str],
    poll_seconds: float,
    timeout_seconds: float,
    label: str,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while time.monotonic() < deadline:
        current = fetch()
        status = current.get("status")
        stage = current.get("stage")
        marker = (status, stage)
        if marker != previous:
            suffix = f" stage={stage}" if stage else ""
            print(f"{label}: status={status}{suffix}", flush=True)
            previous = marker
        if status in terminal_statuses:
            return current
        time.sleep(poll_seconds)
    raise RuntimeError(f"timed out waiting for {label}")


def _parse_selection(value: str, count: int) -> list[int]:
    normalized = value.strip().lower()
    if normalized in {"all", "a", "全部", "全选"}:
        return list(range(count))
    indexes: list[int] = []
    for token in normalized.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(token) - 1
        except ValueError as exc:
            raise ValueError("请输入逗号分隔的编号，例如 1,3，或 all") from exc
        if index < 0 or index >= count:
            raise ValueError(f"检测框编号超出范围: {index + 1}")
        if index not in indexes:
            indexes.append(index)
    if not indexes:
        raise ValueError("至少选择一个检测框")
    return indexes


def _choose_detections(detections: list[dict]) -> list[dict]:
    print("\nGroundingDINO 检测结果：")
    for index, detection in enumerate(detections, start=1):
        score = float(detection.get("box_score", 0.0))
        phrase_score = float(detection.get("phrase_score", 0.0))
        print(
            f"  [{index}] {detection.get('entity')} "
            f"box_score={score:.3f} phrase_score={phrase_score:.3f} "
            f"box={detection.get('box_xyxy')}"
        )
    while True:
        try:
            value = input("选择交给 SAM 的检测框（如 1,3 或 all）: ")
            return [detections[index] for index in _parse_selection(value, len(detections))]
        except ValueError as exc:
            print(f"输入无效：{exc}", file=sys.stderr)


def _prompt_image_path(initial: Path | None) -> Path:
    candidate = initial
    while True:
        if candidate is None:
            raw = input("图片路径（服务器路径）: ").strip().strip('"')
            if not raw:
                print("图片路径不能为空，请重新输入。", file=sys.stderr)
                continue
            candidate = Path(raw)
        path = candidate.expanduser().resolve()
        if path.is_file():
            return path
        print(f"图片不存在: {path}，请重新输入。", file=sys.stderr)
        candidate = None


def _prompt_grounding_prompt(initial: str | None) -> str:
    candidate = initial
    while True:
        value = candidate.strip() if candidate is not None else input(
            "GroundingDINO Prompt: "
        ).strip()
        if value:
            return value
        print("Prompt 不能为空，请重新输入。", file=sys.stderr)
        candidate = None


def _run_stage(name: str, module: str) -> bool:
    choice = input(
        f"\n按 Enter 开始【{name}】；输入 q 在此结束调试: "
    ).strip().lower()
    if choice in {"q", "quit", "exit", "stop"}:
        print(f"已在【{name}】之前停止。")
        return False
    storage_root = Path(
        os.environ["ANNOTATION_STORAGE_ROOT"]
    ).expanduser().resolve()
    log_directory = storage_root / "stage_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_name = module.rsplit(".", 1)[-1]
    log_path = log_directory / f"{log_name}-{time.time_ns()}.log"
    print(f"\n========== {name} 开始 ==========", flush=True)
    print(f"阶段日志: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--once"],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        try:
            tail = "\n".join(
                log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()[-30:]
            )
        except OSError:
            tail = ""
        raise RuntimeError(
            f"{name} worker exited with code {completed.returncode}; "
            f"see {log_path}"
            + (f"\n--- log tail ---\n{tail}" if tail else "")
        )
    print(f"========== {name} 完成 ==========\n", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential DINO, SAM and Qwen prompt debugger"
    )
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--category", default="unsafe")
    parser.add_argument("--api-url", default="http://127.0.0.1:8008")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()

    path = _prompt_image_path(args.image)
    prompt_value = _prompt_grounding_prompt(args.prompt)
    api_key = os.getenv("ANNOTATION_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANNOTATION_API_KEY 未配置")

    base_url = args.api_url.rstrip("/")
    _wait_for_api(base_url)
    asset = _upload_asset(base_url, api_key, path)
    job = _post_json(
        f"{base_url}/v1/annotation/jobs",
        api_key,
        {
            "asset_ids": [asset["asset_id"]],
            "grounding_prompt": prompt_value,
            "grounding_prompt_normalization_mode": "canonical_terms",
            "grounding_prompt_normalization_profile": "construction_safety_v1",
            "pipeline_version": "debug-staged-dino-sam-qwen-v1",
        },
    )
    job_id = job["job_id"]
    print(json.dumps({"asset_id": asset["asset_id"], "job_id": job_id}))
    if not _run_stage(
        "阶段 1/3 GroundingDINO",
        "annotation_service.pipeline.grounding_dino",
    ):
        return 0
    job = _get_json(f"{base_url}/v1/annotation/jobs/{job_id}", api_key)
    if job["status"] not in {"succeeded", "partial_failed"}:
        raise RuntimeError(
            "GroundingDINO job failed: "
            + json.dumps(job, ensure_ascii=False)
        )

    response = _get_json(
        f"{base_url}/v1/annotation/jobs/{job_id}/detections", api_key
    )
    detections = response["items"]
    if not detections:
        print("GroundingDINO 没有产生检测框，本次不会创建 Task。")
        return 0
    selected = _choose_detections(detections)
    selection = _post_json(
        f"{base_url}/v1/annotation/jobs/{job_id}/review-tasks",
        api_key,
        {
            "detection_ids": [item["detection_id"] for item in selected],
            "category": args.category,
        },
    )
    if len(selection["items"]) != 1:
        raise RuntimeError(
            "single-image detection selection must create exactly one Task"
        )
    item = selection["items"][0]
    task_id = item["task_id"]
    operation = _post_json(
        f"{base_url}/v1/annotation/tasks/{task_id}/mask-candidates",
        api_key,
        {
            "expected_version": item["task_version"],
            "boxes_xyxy": item["boxes_xyxy"],
            "detection_ids": item["detection_ids"],
        },
    )
    operation_id = operation["operation_id"]
    print(
        json.dumps(
            {
                "task_id": task_id,
                "selected_detection_ids": item["detection_ids"],
                "sam_operation_id": operation_id,
            },
            ensure_ascii=False,
        )
    )
    if not _run_stage("阶段 2/3 SAM", "annotation_service.pipeline.sam"):
        return 0
    operation = _get_json(
        f"{base_url}/v1/annotation/operations/{operation_id}", api_key
    )
    print(json.dumps(operation, ensure_ascii=False, indent=2))
    if operation["status"] != "succeeded":
        raise RuntimeError(
            "SAM operation failed: "
            + json.dumps(operation.get("error"), ensure_ascii=False)
        )
    print(
        "\nSAM 完成：人工选中的多个检测框属于同一个 Task，"
        "已生成实例 polygons 和联合 mask。"
    )

    prompt_operation = _post_json(
        f"{base_url}/v1/annotation/tasks/{task_id}/prompt-enrichments",
        api_key,
        {"expected_version": item["task_version"]},
    )
    prompt_operation_id = prompt_operation["operation_id"]
    print(
        json.dumps(
            {
                "task_id": task_id,
                "prompt_operation_id": prompt_operation_id,
            },
            ensure_ascii=False,
        )
    )
    if not _run_stage(
        "阶段 3/3 Qwen Prompt 生成",
        "annotation_service.pipeline.qwen",
    ):
        return 0
    prompt_operation = _get_json(
        f"{base_url}/v1/annotation/operations/{prompt_operation_id}",
        api_key,
    )
    print(json.dumps(prompt_operation, ensure_ascii=False, indent=2))
    if prompt_operation["status"] != "succeeded":
        raise RuntimeError(
            "Qwen prompt operation failed: "
            + json.dumps(
                prompt_operation.get("error"),
                ensure_ascii=False,
            )
        )
    print("\n完成：GroundingDINO、SAM 和 3+2+1 Prompt 已分步执行。")
    return 0


if __name__ == "__main__":
    main()
