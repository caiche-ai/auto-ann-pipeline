import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from annotation_service.errors import VersionConflictError
from annotation_service.release_builder import (
    ReleaseWorker,
    build_release_files,
    split_for_group,
)
from annotation_service.storage import AnnotationStore, sha256_file


def image_bytes(
    color=(20, 40, 60),
    *,
    image_format="PNG",
) -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 12), color).save(
        output,
        format=image_format,
    )
    return output.getvalue()


def complete_annotation(target: str) -> dict:
    return {
        "target_object": target,
        "instance_count": 1,
        "visual_anchor": ["画面中央"],
        "mask_granularity": "人员整体",
        "risk_semantics": "头部防护缺失",
        "shapes": [
            {
                "shape_id": "shape-1",
                "label": "target",
                "shape_type": "polygon",
                "points": [[2, 2], [15, 2], [15, 10], [2, 10]],
            }
        ],
        "prompts": [
            {"prompt_id": "v1", "type": "visual", "text": f"分割{target}。"},
            {"prompt_id": "v2", "type": "visual", "text": f"标出{target}。"},
            {"prompt_id": "v3", "type": "visual", "text": f"提取{target}。"},
            {"prompt_id": "r1", "type": "risk", "text": f"分割未戴安全帽的{target}。"},
            {"prompt_id": "r2", "type": "risk", "text": f"标出头部防护缺失的{target}。"},
            {"prompt_id": "a1", "type": "agent", "text": f"找出并分割{target}。"},
        ],
    }


class ReleaseBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "annotation-data"
        self.store = AnnotationStore(self.root)
        self.store.initialize()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def accepted_task(
        self,
        *,
        group_id: str,
        target: str,
        color=(20, 40, 60),
    ) -> dict:
        asset = self.store.create_asset(
            image_bytes=image_bytes(color),
            media_type="image/png",
            width=20,
            height=12,
            group_id=group_id,
        )
        job = self.store.create_job(
            asset_ids=[asset["asset_id"]],
            requested_categories=["helmet_missing"],
            pipeline_version="manual-v1",
        )
        task = self.store.create_task(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            category="helmet_missing",
            annotation=complete_annotation(target),
            provenance={"pipeline_version": "manual-v1"},
        )
        submitted = self.store.submit_task(
            task["task_id"],
            expected_version=1,
            annotator_id="annotator-1",
            primary_result="prompt_ok",
            comment="ready",
        )
        return self.store.review_task(
            task["task_id"],
            expected_version=submitted["version"],
            reviewer_id="reviewer-1",
            decision="accept",
            primary_result="prompt_ok",
            comment="accepted",
        )

    def create_release(self, name="ReasonSegGroundedV1") -> dict:
        return self.store.create_release(
            name=name,
            task_filter={"status": "accepted"},
            split_policy={
                "type": "grouped",
                "group_field": "group_id",
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "golden_ratio": 0.1,
                "seed": 42,
            },
        )

    def test_release_worker_exports_reasonseg_and_manifest(self):
        first = self.accepted_task(
            group_id="site01:video01",
            target="第一名作业人员",
        )
        second = self.accepted_task(
            group_id="site01:video01",
            target="第二名作业人员",
            color=(80, 30, 10),
        )
        third = self.accepted_task(
            group_id="site02:video01",
            target="第三名作业人员",
            color=(10, 80, 30),
        )
        release = self.create_release()
        worker = ReleaseWorker(
            store=self.store,
            worker_id="release-test",
            lease_seconds=30,
            heartbeat_seconds=5,
            poll_seconds=0.1,
        )
        self.assertTrue(worker.run_once())

        completed = self.store.get_release(release["release_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(sum(completed["counts"].values()), 3)
        manifest_path, _ = self.store.release_file(
            release["release_id"],
            "manifest",
        )
        archive_path, _ = self.store.release_file(
            release["release_id"],
            "archive",
        )
        manifest = json.loads(manifest_path.read_text())
        by_task = {
            item["task_id"]: item for item in manifest["samples"]
        }
        self.assertEqual(
            by_task[first["task_id"]]["split"],
            by_task[second["task_id"]]["split"],
        )
        self.assertIn(third["task_id"], by_task)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            self.assertEqual(names, sorted(names))
            self.assertIn(
                "ReasonSegGroundedV1/annotation_manifest.jsonl",
                names,
            )
            json_names = [
                name
                for name in names
                if name.endswith(".json")
                and "/build_summary.json" not in name
            ]
            sample_json = json.loads(archive.read(json_names[0]))
            self.assertEqual(len(sample_json["text"]), 6)
            self.assertTrue(sample_json["is_sentence"])
            self.assertTrue(
                any(
                    shape["label"] == "target"
                    for shape in sample_json["shapes"]
                )
            )
            for name in names:
                self.assertFalse(name.startswith("/"))
                self.assertNotIn("..", Path(name).parts)

    def test_build_is_deterministic_and_group_split_is_stable(self):
        self.accepted_task(
            group_id="site01:video01",
            target="目标人员",
        )
        release = self.create_release("deterministic-v1")
        claimed = self.store.claim_next_release(
            worker_id="release-test",
            lease_seconds=30,
        )
        snapshots = self.store.get_release_export_snapshot(
            release["release_id"],
            claim_token=claimed["claim_token"],
        )
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_release_files(
                release=claimed,
                snapshots=snapshots,
                output_root=Path(first_dir),
            )
            second = build_release_files(
                release=claimed,
                snapshots=snapshots,
                output_root=Path(second_dir),
            )
            self.assertEqual(
                sha256_file(first.archive_path),
                sha256_file(second.archive_path),
            )
        split = split_for_group(
            "site01:video01",
            claimed["split_policy"],
        )
        self.assertIn(split, {"train", "val", "golden"})

    def test_export_is_readable_by_existing_reasonseg_mask_loader(self):
        try:
            import cv2
            from utils.data_processing import get_mask_from_json
        except ImportError:
            self.skipTest("OpenCV is not installed in this test environment")

        self.accepted_task(
            group_id="site01:video01",
            target="兼容性测试人员",
        )
        release = self.create_release("loader-compatible-v1")
        ReleaseWorker(
            store=self.store,
            worker_id="release-test",
            lease_seconds=30,
            heartbeat_seconds=5,
        ).run_once()
        archive_path, _ = self.store.release_file(
            release["release_id"],
            "archive",
        )
        with tempfile.TemporaryDirectory() as extracted:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extracted)
            dataset_root = Path(extracted) / "loader-compatible-v1"
            image_path = next(
                path
                for split in ("train", "val", "golden")
                for path in (dataset_root / split).glob("*.jpg")
            )
            json_path = image_path.with_suffix(".json")
            image = cv2.imread(str(image_path))
            mask, prompts, is_sentence = get_mask_from_json(
                str(json_path),
                image,
            )
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertGreater(int((mask == 1).sum()), 0)
        self.assertEqual(len(prompts), 6)
        self.assertTrue(is_sentence)

    def test_claim_is_exclusive_and_expired_claim_is_recovered(self):
        release = self.create_release("lease-v1")
        first = self.store.claim_next_release(
            worker_id="release-1",
            lease_seconds=30,
        )
        self.assertEqual(first["release_id"], release["release_id"])
        self.assertIsNone(
            self.store.claim_next_release(
                worker_id="release-2",
                lease_seconds=30,
            )
        )
        with self.store._connect() as connection:
            connection.execute(
                """
                UPDATE releases
                SET lease_expires_at = '2000-01-01T00:00:00+00:00'
                WHERE release_id = ?
                """,
                (release["release_id"],),
            )
        recovered = self.store.claim_next_release(
            worker_id="release-2",
            lease_seconds=30,
        )
        self.assertNotEqual(
            first["claim_token"],
            recovered["claim_token"],
        )
        self.assertEqual(recovered["attempt_count"], 2)
        with self.assertRaises(VersionConflictError):
            self.store.heartbeat_release(
                release["release_id"],
                claim_token=first["claim_token"],
                lease_seconds=30,
            )

    def test_empty_release_fails_without_artifacts(self):
        release = self.create_release("empty-v1")
        worker = ReleaseWorker(
            store=self.store,
            worker_id="release-test",
            lease_seconds=30,
            heartbeat_seconds=5,
        )
        self.assertTrue(worker.run_once())
        failed = self.store.get_release(release["release_id"])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("no accepted tasks", failed["error"])

    def test_release_worker_rejects_invalid_timing(self):
        with self.assertRaises(ValueError):
            ReleaseWorker(
                store=self.store,
                worker_id="release-test",
                lease_seconds=30,
                heartbeat_seconds=30,
            )
        with self.assertRaises(ValueError):
            ReleaseWorker(
                store=self.store,
                worker_id="release-test",
                lease_seconds=30,
                heartbeat_seconds=5,
                poll_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
