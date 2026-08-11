"""End-to-end concurrency benchmark for DINO -> SAM -> Qwen.

The benchmark automatically selects every GroundingDINO detection.  It is
intended for an isolated benchmark deployment, not production annotation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .debug_client import _get_json, _post_json, _upload_asset


TERMINAL = {"succeeded", "partial_failed", "failed", "cancelled"}


def _wait(fetch, *, poll_seconds: float, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = fetch()
        if value.get("status") in TERMINAL:
            return value
        time.sleep(poll_seconds)
    raise TimeoutError("operation did not finish before the timeout")


def _milliseconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds() * 1000


def _operation_metrics(operation: dict, prefix: str) -> dict[str, float | None]:
    return {
        f"{prefix}_queue_ms": _milliseconds(
            operation.get("created_at"), operation.get("started_at")
        ),
        f"{prefix}_worker_ms": _milliseconds(
            operation.get("started_at"), operation.get("completed_at")
        ),
    }


def _one_request(
    index: int,
    *,
    image: Path,
    prompt: str,
    base_url: str,
    api_key: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    asset = _upload_asset(base_url, api_key, image)
    uploaded = time.perf_counter()
    job = _post_json(
        f"{base_url}/v1/annotation/jobs",
        api_key,
        {
            "asset_ids": [asset["asset_id"]],
            "grounding_prompt": prompt,
            "grounding_prompt_normalization_mode": "canonical_terms",
            "grounding_prompt_normalization_profile": "construction_safety_v1",
            "pipeline_version": "concurrency-benchmark-v1",
        },
    )
    job = _wait(
        lambda: _get_json(
            f"{base_url}/v1/annotation/jobs/{job['job_id']}", api_key
        ),
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    if job["status"] not in {"succeeded", "partial_failed"}:
        raise RuntimeError(f"DINO job ended as {job['status']}")
    detections = _get_json(
        f"{base_url}/v1/annotation/jobs/{job['job_id']}/detections", api_key
    )["items"]
    if not detections:
        raise RuntimeError("DINO returned no detections")
    selection = _post_json(
        f"{base_url}/v1/annotation/jobs/{job['job_id']}/review-tasks",
        api_key,
        {
            "detection_ids": [item["detection_id"] for item in detections],
            "category": "unsafe",
        },
    )
    if len(selection["items"]) != 1:
        raise RuntimeError("detection selection did not create one Task")
    task = selection["items"][0]
    sam = _post_json(
        f"{base_url}/v1/annotation/tasks/{task['task_id']}/mask-candidates",
        api_key,
        {
            "expected_version": task["task_version"],
            "boxes_xyxy": task["boxes_xyxy"],
            "detection_ids": task["detection_ids"],
        },
    )
    sam = _wait(
        lambda: _get_json(
            f"{base_url}/v1/annotation/operations/{sam['operation_id']}",
            api_key,
        ),
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    if sam["status"] != "succeeded":
        raise RuntimeError(f"SAM operation ended as {sam['status']}: {sam.get('error')}")
    qwen = _post_json(
        f"{base_url}/v1/annotation/tasks/{task['task_id']}/prompt-enrichments",
        api_key,
        {"expected_version": task["task_version"]},
    )
    qwen = _wait(
        lambda: _get_json(
            f"{base_url}/v1/annotation/operations/{qwen['operation_id']}",
            api_key,
        ),
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    if qwen["status"] != "succeeded":
        raise RuntimeError(
            f"Qwen operation ended as {qwen['status']}: {qwen.get('error')}"
        )
    usage = (qwen.get("result") or {}).get("usage") or {}
    completed = time.perf_counter()
    result: dict[str, Any] = {
        "index": index,
        "status": "succeeded",
        "total_ms": (completed - started) * 1000,
        "upload_ms": (uploaded - started) * 1000,
        "detection_count": len(detections),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    result.update(
        {
            "dino_queue_ms": _milliseconds(
                job.get("created_at"), job.get("started_at")
            ),
            "dino_worker_ms": _milliseconds(
                job.get("started_at"), job.get("completed_at")
            ),
        }
    )
    result.update(_operation_metrics(sam, "sam"))
    result.update(_operation_metrics(qwen, "qwen"))
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[position]


def _summarize(results: list[dict], elapsed_seconds: float) -> dict[str, Any]:
    succeeded = [item for item in results if item["status"] == "succeeded"]
    summary: dict[str, Any] = {
        "requests": len(results),
        "succeeded": len(succeeded),
        "failed": len(results) - len(succeeded),
        "elapsed_seconds": elapsed_seconds,
        "throughput_rps": len(succeeded) / elapsed_seconds,
    }
    for key in (
        "total_ms",
        "dino_queue_ms",
        "dino_worker_ms",
        "sam_queue_ms",
        "sam_worker_ms",
        "qwen_queue_ms",
        "qwen_worker_ms",
    ):
        values = [
            float(item[key]) for item in succeeded if item.get(key) is not None
        ]
        summary[key] = {
            "mean": statistics.fmean(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
        }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [int(item[key]) for item in succeeded if item.get(key) is not None]
        summary[key] = {
            "sum": sum(values),
            "mean": statistics.fmean(values) if values else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--api-url", default="http://127.0.0.1:8010")
    parser.add_argument(
        "--api-key",
        default=os.getenv("ANNOTATION_API_KEY", "").strip(),
        help="defaults to ANNOTATION_API_KEY",
    )
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")
    if not args.api_key:
        parser.error("--api-key or ANNOTATION_API_KEY is required")
    image = args.image.expanduser().resolve()
    if not image.is_file():
        parser.error(f"image does not exist: {image}")

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _one_request,
                index,
                image=image,
                prompt=args.prompt,
                base_url=args.api_url.rstrip("/"),
                api_key=args.api_key,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
            for index in range(1, args.requests + 1)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                print(
                    f"[{len(results) + 1}/{args.requests}] ok "
                    f"request={result['index']} total={result['total_ms']:.1f}ms",
                    flush=True,
                )
            except Exception as exc:
                result = {"status": "failed", "error": str(exc)}
                print(
                    f"[{len(results) + 1}/{args.requests}] failed: {exc}",
                    flush=True,
                )
            results.append(result)
    elapsed = time.perf_counter() - started
    report = {
        "concurrency": args.concurrency,
        "summary": _summarize(results, elapsed),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
