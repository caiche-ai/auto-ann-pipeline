import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

from annotation_service.errors import (
    IdempotencyConflictError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
    VersionConflictError,
)
from annotation_service.storage import AnnotationStore
from annotation_service.storage_schema import SCHEMA_V1, SCHEMA_V2


def png_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 10), color).save(output, format="PNG")
    return output.getvalue()


def annotation(target: str = "一名作业人员") -> dict:
    return {
        "target_object": target,
        "instance_count": 1,
        "visual_anchor": ["画面中央"],
        "mask_granularity": "人员整体",
        "risk_semantics": None,
        "shapes": [
            {
                "shape_id": "shape-1",
                "label": "target",
                "shape_type": "polygon",
                "points": [[1, 1], [10, 1], [10, 8], [1, 8]],
            }
        ],
        "prompts": [
            {"prompt_id": "v1", "type": "visual", "text": "分割中央人员。"},
            {"prompt_id": "v2", "type": "visual", "text": "标出目标人员。"},
            {"prompt_id": "v3", "type": "visual", "text": "提取画面中的人员。"},
            {"prompt_id": "r1", "type": "risk", "text": "分割存在风险的人员。"},
            {"prompt_id": "r2", "type": "risk", "text": "标出需复核的人员。"},
            {"prompt_id": "a1", "type": "agent", "text": "找出并分割目标人员。"},
        ],
    }


class AnnotationStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "annotation-data"
        self.store = AnnotationStore(self.root)
        self.store.initialize()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def create_asset(
        self,
        *,
        content: bytes = b"test-image",
        group_id: str = "site01:video01",
    ) -> dict:
        return self.store.create_asset(
            image_bytes=content,
            media_type="image/png",
            width=20,
            height=10,
            group_id=group_id,
            source_id="frontend-1",
            metadata={"camera": "north"},
        )

    def create_job(self, asset_id: str) -> dict:
        return self.store.create_job(
            asset_ids=[asset_id],
            requested_categories=["helmet_missing"],
            pipeline_version="grounded-qwen-v1",
        )

    def create_task(self, job_id: str, asset_id: str) -> dict:
        return self.store.create_task(
            job_id=job_id,
            asset_id=asset_id,
            category="helmet_missing",
            annotation=annotation(),
            provenance={"pipeline_version": "grounded-qwen-v1"},
        )

    def test_initialize_creates_versioned_schema_and_directories(self):
        self.assertEqual(self.store.schema_version(), 10)
        self.assertEqual(self.store.readiness(), {"storage": "ready"})
        for name in (
            "images",
            "masks",
            "overlays",
            "crops",
            "exports",
            "tmp",
        ):
            self.assertTrue((self.root / name).is_dir())

        with sqlite3.connect(self.root / "annotation.db") as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertTrue(
            {
                "assets",
                "annotation_jobs",
                "job_assets",
                "detections",
                "job_artifacts",
                "hazard_candidates",
                "annotation_tasks",
                "annotation_task_groups",
                "annotation_task_group_members",
                "task_versions",
                "reviews",
                "artifacts",
                "releases",
                "idempotency_keys",
            }.issubset(tables)
        )
        with self.store._connect() as connection:
            job_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(annotation_jobs)"
                ).fetchall()
            }
        self.assertTrue(
            {
                "claimed_by",
                "lease_expires_at",
                "heartbeat_at",
                "attempt_count",
                "grounding_prompt",
                "grounding_prompt_route_json",
            }.issubset(job_columns)
        )
        with self.store._connect() as connection:
            task_version_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(task_versions)"
                ).fetchall()
            }
        self.assertIn("comment", task_version_columns)
        with self.store._connect() as connection:
            task_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(annotation_tasks)"
                ).fetchall()
            }
            release_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(releases)"
                ).fetchall()
            }
        self.assertIn("source_hazard_id", task_columns)
        self.assertIn("source_detection_id", task_columns)
        self.assertTrue(
            {
                "claimed_by",
                "claim_token",
                "lease_expires_at",
                "heartbeat_at",
                "started_at",
                "attempt_count",
            }.issubset(release_columns)
        )

    def test_existing_v1_database_is_migrated(self):
        root = Path(self.temporary.name) / "v1-data"
        root.mkdir()
        with sqlite3.connect(root / "annotation.db") as connection:
            connection.executescript(SCHEMA_V1)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, applied_at)
                VALUES (1, '2026-01-01T00:00:00+00:00')
                """
            )

        migrated = AnnotationStore(root)
        migrated.initialize()
        self.assertEqual(migrated.schema_version(), 10)
        with migrated._connect() as connection:
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(annotation_jobs)"
                ).fetchall()
            }
        migrated.close()
        self.assertIn("lease_expires_at", columns)

    def test_existing_v2_database_is_migrated_with_hazard_table(self):
        root = Path(self.temporary.name) / "v2-data"
        root.mkdir()
        with sqlite3.connect(root / "annotation.db") as connection:
            connection.executescript(SCHEMA_V1)
            for statement in SCHEMA_V2:
                connection.execute(statement)
            connection.executemany(
                """
                INSERT INTO schema_migrations(version, applied_at)
                VALUES (?, '2026-01-01T00:00:00+00:00')
                """,
                [(1,), (2,)],
            )

        migrated = AnnotationStore(root)
        migrated.initialize()
        self.assertEqual(migrated.schema_version(), 10)
        with migrated._connect() as connection:
            table = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name = 'hazard_candidates'
                """
            ).fetchone()
        migrated.close()
        self.assertIsNotNone(table)

    def test_asset_uses_relative_content_addressed_path_and_duplicate_link(self):
        first = self.create_asset()
        duplicate = self.create_asset(group_id="site01:video02")

        self.assertIsNone(first["duplicate_of"])
        self.assertEqual(duplicate["duplicate_of"], first["asset_id"])
        first_path, media_type = self.store.asset_file(first["asset_id"])
        duplicate_path, _ = self.store.asset_file(duplicate["asset_id"])
        self.assertEqual(first_path, duplicate_path)
        self.assertEqual(first_path.read_bytes(), b"test-image")
        self.assertEqual(media_type, "image/png")
        self.assertNotIn(str(self.root), first["content_url"])

        with self.store._connect() as connection:
            stored = connection.execute(
                "SELECT image_path FROM assets WHERE asset_id = ?",
                (first["asset_id"],),
            ).fetchone()[0]
        self.assertFalse(Path(stored).is_absolute())

    def test_concurrent_duplicate_assets_create_one_canonical_file(self):
        def create(index: int) -> dict:
            return self.create_asset(group_id=f"site{index}:video01")

        with ThreadPoolExecutor(max_workers=4) as executor:
            assets = list(executor.map(create, range(8)))

        self.assertEqual(
            sum(item["duplicate_of"] is None for item in assets),
            1,
        )
        image_files = list((self.root / "images").rglob("*.png"))
        self.assertEqual(len(image_files), 1)

    def test_jobs_survive_restart_and_remain_recoverable(self):
        asset = self.create_asset()
        job = self.create_job(asset["asset_id"])
        running = self.store.update_job(
            job["job_id"],
            expected_status="queued",
            status="running",
            stage="grounding_dino",
        )
        self.assertEqual(running["status"], "running")

        restarted = AnnotationStore(self.root)
        restarted.initialize()
        recovered = restarted.list_recoverable_jobs()
        restarted.close()

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["job_id"], job["job_id"])
        self.assertEqual(recovered[0]["stage"], "grounding_dino")

    def test_only_one_worker_claims_a_job(self):
        asset = self.create_asset()
        job = self.create_job(asset["asset_id"])
        stores = [AnnotationStore(self.root) for _ in range(4)]
        for store in stores:
            store.initialize()

        def claim(item):
            index, store = item
            return store.claim_next_job(
                worker_id=f"gpu-worker-{index}",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            claims = list(executor.map(claim, enumerate(stores)))
        for store in stores:
            store.close()

        claimed = [item for item in claims if item is not None]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0]["job_id"], job["job_id"])
        self.assertEqual(claimed[0]["status"], "running")
        self.assertEqual(claimed[0]["stage"], "grounding_dino")
        self.assertEqual(claimed[0]["attempt_count"], 1)
        self.assertEqual(
            claimed[0]["stages"]["grounding_dino"]["status"],
            "running",
        )

    def test_concurrent_idempotent_job_creation_across_stores(self):
        asset = self.create_asset()
        request = {
            "asset_ids": [asset["asset_id"]],
            "requested_categories": ["helmet_missing"],
            "pipeline_version": "grounded-qwen-v1",
            "options": {
                "generate_masks": True,
                "enrich_prompts": True,
                "prompt_count": 6,
            },
        }
        stores = [AnnotationStore(self.root) for _ in range(4)]
        for store in stores:
            store.initialize()

        def create(store: AnnotationStore) -> dict:
            return store.create_job(
                asset_ids=request["asset_ids"],
                requested_categories=request[
                    "requested_categories"
                ],
                pipeline_version=request["pipeline_version"],
                options=request["options"],
                idempotency_key="concurrent-job-001",
                idempotency_request=request,
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = list(executor.map(create, stores))
        for store in stores:
            store.close()

        self.assertEqual(
            len({job["job_id"] for job in jobs}),
            1,
        )
        with self.store._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM annotation_jobs"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_worker_lease_heartbeat_completion_and_reclaim(self):
        asset = self.create_asset()
        job = self.create_job(asset["asset_id"])
        first = self.store.claim_next_job(
            worker_id="gpu-worker-1",
            lease_seconds=60,
        )
        self.assertIsNotNone(first)
        heartbeat = self.store.heartbeat_job(
            job["job_id"],
            worker_id="gpu-worker-1",
            lease_seconds=120,
        )
        self.assertEqual(heartbeat["claimed_by"], "gpu-worker-1")

        with self.assertRaises(VersionConflictError):
            self.store.heartbeat_job(
                job["job_id"],
                worker_id="gpu-worker-2",
            )
        with self.store._connect() as connection:
            connection.execute(
                """
                UPDATE annotation_jobs
                SET lease_expires_at = '2000-01-01T00:00:00+00:00'
                WHERE job_id = ?
                """,
                (job["job_id"],),
            )

        reclaimed = self.store.claim_next_job(
            worker_id="gpu-worker-2",
        )
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed["job_id"], job["job_id"])
        self.assertEqual(reclaimed["claimed_by"], "gpu-worker-2")
        self.assertEqual(reclaimed["attempt_count"], 2)

        with self.assertRaises(VersionConflictError):
            self.store.update_job(
                job["job_id"],
                expected_status="running",
                status="succeeded",
                worker_id="gpu-worker-1",
            )
        completed = self.store.update_job(
            job["job_id"],
            expected_status="running",
            status="succeeded",
            worker_id="gpu-worker-2",
        )
        self.assertEqual(completed["status"], "succeeded")
        self.assertIsNone(completed["claimed_by"])
        self.assertIsNone(completed["lease_expires_at"])

    def test_detection_requires_asset_membership_in_job(self):
        first = self.create_asset()
        second = self.create_asset(content=b"second-image")
        job = self.create_job(first["asset_id"])

        with self.assertRaises(ResourceNotFoundError):
            self.store.add_detection(
                job_id=job["job_id"],
                asset_id=second["asset_id"],
                entity="person",
                box_xyxy=[1, 1, 10, 9],
                box_score=0.9,
                phrase_score=0.8,
            )

    def test_task_versions_review_and_optimistic_lock(self):
        asset = self.create_asset()
        job = self.create_job(asset["asset_id"])
        self.store.add_detection(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            entity="person",
            box_xyxy=[1, 1, 10, 9],
            box_score=0.9,
            phrase_score=0.8,
        )
        task = self.create_task(job["job_id"], asset["asset_id"])
        self.assertEqual(task["version"], 1)
        self.assertEqual(task["status"], "generated")
        self.assertEqual(len(task["detections"]), 1)

        draft = self.store.save_task_draft(
            task["task_id"],
            expected_version=1,
            annotation=annotation("画面中央未戴安全帽的一名人员"),
            editor_id="annotator-1",
        )
        self.assertEqual(draft["version"], 2)
        self.assertEqual(draft["status"], "annotating")

        with self.assertRaises(VersionConflictError):
            self.store.save_task_draft(
                task["task_id"],
                expected_version=1,
                annotation=annotation(),
                editor_id="stale-editor",
            )

        submitted = self.store.submit_task(
            task["task_id"],
            expected_version=2,
            annotator_id="annotator-1",
            primary_result="prompt_rewritten",
            comment="一级标注已确认",
        )
        accepted = self.store.review_task(
            task["task_id"],
            expected_version=submitted["version"],
            reviewer_id="reviewer-1",
            decision="accept",
            primary_result="prompt_rewritten",
        )
        frozen = self.store.freeze_task(
            task["task_id"],
            expected_version=accepted["version"],
            editor_id="release-builder",
        )

        self.assertEqual(frozen["status"], "frozen")
        versions = self.store.list_task_versions(task["task_id"])
        self.assertEqual([item["version"] for item in versions], [1, 2, 3, 4, 5])
        self.assertEqual(versions[2]["comment"], "一级标注已确认")
        self.assertEqual(len(self.store.list_reviews(task["task_id"])), 1)

        with self.assertRaises(InvalidStateTransitionError):
            self.store.save_task_draft(
                task["task_id"],
                expected_version=5,
                annotation=annotation(),
                editor_id="late-editor",
            )

    def test_artifacts_are_atomic_and_latest_candidate_is_returned(self):
        asset = self.create_asset()
        job = self.create_job(asset["asset_id"])
        task = self.create_task(job["job_id"], asset["asset_id"])

        self.store.store_artifact(
            task_id=task["task_id"],
            artifact_type="mask",
            data=png_bytes((1, 2, 3)),
            media_type="image/png",
            width=20,
            height=10,
        )
        expected = png_bytes((4, 5, 6))
        latest = self.store.store_artifact(
            task_id=task["task_id"],
            artifact_type="mask",
            data=expected,
            media_type="image/png",
            width=20,
            height=10,
            operation_id="op-2",
        )
        path, media_type = self.store.artifact_file(task["task_id"], "mask")

        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(media_type, "image/png")
        self.assertEqual(latest["operation_id"], "op-2")
        self.assertNotIn(str(self.root), latest["url"])

        with self.assertRaises(ValueError):
            self.store.store_artifact(
                task_id=task["task_id"],
                artifact_type="mask",
                data=expected,
                media_type="image/jpeg",
            )
        with self.assertRaises(ValueError):
            self.store.store_artifact(
                task_id=task["task_id"],
                artifact_type="mask",
                data=expected,
                media_type="image/png",
                width=21,
                height=10,
            )

    def test_release_files_are_persisted_and_not_overwritten(self):
        release = self.store.create_release(
            name="ReasonSegGroundedV1",
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
        building = self.store.transition_release(
            release["release_id"],
            expected_status="queued",
            status="building",
        )
        self.assertEqual(building["status"], "building")
        completed = self.store.complete_release(
            release["release_id"],
            manifest=b'{"name":"ReasonSegGroundedV1"}',
            archive=b"PK-test-archive",
            counts={"train": 8, "val": 1, "golden": 1},
        )
        self.assertEqual(completed["status"], "succeeded")
        manifest_path, manifest_type = self.store.release_file(
            release["release_id"], "manifest"
        )
        archive_path, archive_type = self.store.release_file(
            release["release_id"], "archive"
        )
        self.assertEqual(manifest_type, "application/json")
        self.assertEqual(archive_type, "application/zip")
        self.assertEqual(manifest_path.read_bytes(), b'{"name":"ReasonSegGroundedV1"}')
        self.assertEqual(archive_path.read_bytes(), b"PK-test-archive")

        with self.assertRaises(IdempotencyConflictError):
            self.store.create_release(
                name="ReasonSegGroundedV1",
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

    def test_idempotency_reuses_same_request_and_rejects_different_request(self):
        request = {"asset_ids": ["a"], "pipeline_version": "v1"}
        saved = self.store.save_idempotency(
            scope="create-job",
            key="frontend-key-001",
            request_payload=request,
            resource_type="job",
            resource_id="job-1",
            response={"job_id": "job-1"},
        )
        reused = self.store.save_idempotency(
            scope="create-job",
            key="frontend-key-001",
            request_payload=request,
            resource_type="job",
            resource_id="job-1",
            response={"job_id": "job-1"},
        )
        self.assertEqual(reused["resource_id"], saved["resource_id"])

        with self.assertRaises(IdempotencyConflictError):
            self.store.find_idempotency(
                scope="create-job",
                key="frontend-key-001",
                request_payload={"asset_ids": ["different"]},
            )

    def test_root_and_escaping_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            AnnotationStore(Path(self.root.anchor))
        with self.assertRaises(Exception):
            self.store._resolve_relative("../outside")


if __name__ == "__main__":
    unittest.main()
