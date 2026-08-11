from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from annotation_service.tools import debug_client, debug_runner


def _png(path: Path) -> None:
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, format="PNG")


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class DebugClientTest(unittest.TestCase):
    def test_multipart_contains_required_asset_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            _png(image_path)
            content_type, body = debug_client._multipart_image(
                image_path,
                fields={"group_id": "debug", "source_id": "debug-client"},
            )
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'name="group_id"', body)
        self.assertIn(b'name="source_id"', body)
        self.assertIn(b'name="file"; filename="sample.png"', body)
        self.assertIn(b"Content-Type: image/png", body)

    def test_selection_accepts_all_and_unique_indexes(self):
        self.assertEqual(debug_client._parse_selection("all", 3), [0, 1, 2])
        self.assertEqual(debug_client._parse_selection("3,1,3", 3), [2, 0])
        with self.assertRaises(ValueError):
            debug_client._parse_selection("4", 3)

    def test_blank_image_path_and_prompt_are_reprompted(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            _png(image_path)
            with (
                patch(
                    "builtins.input",
                    side_effect=["", str(image_path)],
                ),
                patch("sys.stderr", io.StringIO()),
            ):
                selected_path = debug_client._prompt_image_path(None)
            with (
                patch("builtins.input", side_effect=["", "helmet"]),
                patch("sys.stderr", io.StringIO()),
            ):
                prompt = debug_client._prompt_grounding_prompt(None)

        self.assertEqual(selected_path, image_path.resolve())
        self.assertEqual(prompt, "helmet")

    def test_stage_worker_output_is_redirected_to_its_own_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                **os.environ,
                "ANNOTATION_STORAGE_ROOT": temporary,
            }
            completed = MagicMock(returncode=0)
            with (
                patch.dict(os.environ, environment, clear=True),
                patch("builtins.input", return_value=""),
                patch.object(
                    debug_client.subprocess,
                    "run",
                    return_value=completed,
                ) as run,
                patch("sys.stdout", io.StringIO()),
            ):
                result = debug_client._run_stage(
                    "SAM",
                    "annotation_service.pipeline.sam",
                )

            logs = list(
                (Path(temporary) / "stage_logs").glob("sam-*.log")
            )

        self.assertTrue(result)
        self.assertEqual(len(logs), 1)
        self.assertEqual(
            run.call_args.args[0],
            [
                debug_client.sys.executable,
                "-m",
                "annotation_service.pipeline.sam",
                "--once",
            ],
        )
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertIsNotNone(run.call_args.kwargs["stdout"])

    def test_main_runs_dino_sam_and_prompt_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            _png(image_path)
            argv = ["debug_client", str(image_path), "helmet"]
            environment = {**os.environ, "ANNOTATION_API_KEY": "test-key"}
            detections = [
                {
                    "detection_id": "det-1",
                    "entity": "helmet",
                    "box_xyxy": [1, 1, 4, 4],
                    "box_score": 0.9,
                    "phrase_score": 0.8,
                },
                {
                    "detection_id": "det-2",
                    "entity": "helmet",
                    "box_xyxy": [4, 1, 7, 4],
                    "box_score": 0.85,
                    "phrase_score": 0.75,
                },
            ]

            def post_json(url, api_key, payload):
                if url.endswith("/jobs"):
                    return {"job_id": "job-debug", "status": "queued"}
                if url.endswith("/review-tasks"):
                    self.assertEqual(
                        payload["detection_ids"], ["det-1", "det-2"]
                    )
                    return {
                        "items": [
                            {
                                "task_id": "tsk-debug",
                                "task_version": 1,
                                "detection_ids": ["det-1", "det-2"],
                                "boxes_xyxy": [[1, 1, 4, 4], [4, 1, 7, 4]],
                            }
                        ]
                    }
                if url.endswith("/prompt-enrichments"):
                    self.assertEqual(payload["expected_version"], 1)
                    return {
                        "operation_id": "op-prompt",
                        "status": "queued",
                    }
                self.assertEqual(payload["detection_ids"], ["det-1", "det-2"])
                self.assertEqual(len(payload["boxes_xyxy"]), 2)
                return {"operation_id": "op-debug", "status": "queued"}

            def get_json(url, api_key):
                if url.endswith("/detections"):
                    return {"items": detections, "total": 2}
                if "/operations/" in url:
                    return {
                        "operation_id": "op-debug",
                        "status": "succeeded",
                        "result": {"shapes": [{}, {}]},
                    }
                return {"job_id": "job-debug", "status": "succeeded"}

            output = io.StringIO()
            with (
                patch.object(debug_client.sys, "argv", argv),
                patch.dict(os.environ, environment, clear=True),
                patch.object(debug_client, "_wait_for_api"),
                patch.object(
                    debug_client,
                    "_upload_asset",
                    return_value={"asset_id": "ast-debug"},
                ),
                patch.object(debug_client, "_post_json", side_effect=post_json),
                patch.object(debug_client, "_get_json", side_effect=get_json),
                patch.object(debug_client, "_run_stage", return_value=True),
                patch("builtins.input", return_value="all"),
                patch("sys.stdout", output),
            ):
                result = debug_client.main()

        self.assertEqual(result, 0)
        self.assertIn("同一个 Task", output.getvalue())
        self.assertIn("3+2+1 Prompt", output.getvalue())


class DebugRunnerTest(unittest.TestCase):
    def test_runner_starts_isolated_api_and_staged_client(self):
        api = FakeProcess()
        client = FakeProcess(returncode=0)
        popen = MagicMock(side_effect=[api, client])
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(debug_runner.subprocess, "Popen", popen),
                patch.object(debug_runner, "PROJECT_ROOT", Path(temporary)),
                patch.object(debug_runner, "_wait_for_api"),
                patch("builtins.input", return_value="n"),
            ):
                result = debug_runner.main()

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 2)
        self.assertTrue(api.terminated)
        self.assertFalse(client.terminated)
        self.assertIsNotNone(popen.call_args_list[0].kwargs["stdout"])
        self.assertEqual(
            popen.call_args_list[0].kwargs["stderr"],
            subprocess.STDOUT,
        )
        session_root = popen.call_args_list[0].kwargs["env"][
            "ANNOTATION_STORAGE_ROOT"
        ]
        self.assertIn("debug", session_root)
        self.assertIn("sessions", session_root)
        self.assertEqual(
            popen.call_args_list[1].kwargs["env"][
                "ANNOTATION_STORAGE_ROOT"
            ],
            session_root,
        )
        debug_environment = popen.call_args_list[1].kwargs["env"]
        self.assertEqual(
            debug_environment["ANNOTATION_WORKER_LEASE_SECONDS"],
            "86400",
        )
        self.assertEqual(
            debug_environment["ANNOTATION_SAM_LEASE_SECONDS"],
            "86400",
        )
        self.assertEqual(
            debug_environment["ANNOTATION_QWEN_LEASE_SECONDS"],
            "86400",
        )


if __name__ == "__main__":
    unittest.main()
