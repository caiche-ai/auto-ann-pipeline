"""Run one isolated, sequential DINO -> SAM -> Qwen debug session."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_URL = "http://127.0.0.1:8008"


def _tail_log(path: Path, line_count: int = 30) -> str:
    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def _wait_for_api(process: subprocess.Popen, log_path: Path) -> None:
    for _ in range(120):
        if process.poll() is not None:
            tail = _tail_log(log_path)
            raise RuntimeError(
                "Annotation API failed to start"
                + (f":\n{tail}" if tail else f"; see {log_path}")
            )
        try:
            with urlopen(f"{API_URL}/health", timeout=1):
                return
        except (URLError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError(
        f"Annotation API did not become healthy; see {log_path}"
    )


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    debug_root = Path(
        os.getenv(
            "ANNOTATION_DEBUG_ROOT",
            os.getenv("ANNOTATION_DEBUG_WORKSPACE", str(PROJECT_ROOT / "debug")),
        )
    ).expanduser().resolve()
    session_root = debug_root / "sessions" / run_id
    session_root.mkdir(parents=True, exist_ok=True)
    api_log_path = session_root / "api.log"
    session_env = os.environ.copy()
    session_env["ANNOTATION_STORAGE_ROOT"] = str(session_root)
    session_env["PYTHONUNBUFFERED"] = "1"
    # Debugpy pauses heartbeat threads together with the model thread. Use a
    # long lease in this isolated session so inspecting tensors at a breakpoint
    # does not invalidate the claimed Job or Operation.
    session_env["ANNOTATION_WORKER_LEASE_SECONDS"] = "86400"
    session_env["ANNOTATION_WORKER_HEARTBEAT_SECONDS"] = "3600"
    session_env["ANNOTATION_SAM_LEASE_SECONDS"] = "86400"
    session_env["ANNOTATION_SAM_HEARTBEAT_SECONDS"] = "600"
    session_env["ANNOTATION_QWEN_LEASE_SECONDS"] = "86400"
    session_env["ANNOTATION_QWEN_HEARTBEAT_SECONDS"] = "600"

    processes: list[subprocess.Popen] = []
    api_log = api_log_path.open("w", encoding="utf-8", buffering=1)
    try:
        print(f"调试数据目录: {session_root}", flush=True)
        print(f"启动 Annotation API ... 日志: {api_log_path}", flush=True)
        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "annotation_service.api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8008",
            ],
            cwd=PROJECT_ROOT,
            env=session_env,
            stdout=api_log,
            stderr=subprocess.STDOUT,
        )
        processes.append(api)
        _wait_for_api(api, api_log_path)
        print(
            "\n分步调试已就绪：GroundingDINO -> SAM -> Prompt 生成。",
            flush=True,
        )

        while True:
            client = subprocess.Popen(
                [sys.executable, "-m", "annotation_service.tools.debug_client"],
                cwd=PROJECT_ROOT,
                env=session_env,
            )
            processes.append(client)
            while client.poll() is None:
                if api.poll() is not None:
                    client.terminate()
                    client.wait(timeout=5)
                    raise RuntimeError(
                        "Annotation API exited while processing:\n"
                        + _tail_log(api_log_path)
                    )
                time.sleep(0.5)
            processes.pop()
            if client.returncode != 0:
                print(
                    f"本次分步调试异常结束：{client.returncode}",
                    flush=True,
                )
            try:
                choice = input("\n继续提交新图片？[Y/n]: ").strip().lower()
            except EOFError:
                return 0
            if choice in {"n", "no", "q", "quit", "exit"}:
                return 0
    except KeyboardInterrupt:
        print("\n正在关闭分步调试 ...", flush=True)
        return 0
    finally:
        _stop(processes)
        api_log.close()


if __name__ == "__main__":
    main()
