from __future__ import annotations


SCHEMA_VERSION = 10

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    source_id TEXT,
    group_id TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    sha256 TEXT NOT NULL,
    media_type TEXT NOT NULL
        CHECK (media_type IN ('image/jpeg', 'image/png')),
    image_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    duplicate_of TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (duplicate_of) REFERENCES assets(asset_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_canonical_sha256
ON assets(sha256) WHERE duplicate_of IS NULL;
CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets(sha256);
CREATE INDEX IF NOT EXISTS idx_assets_group_id ON assets(group_id);
CREATE INDEX IF NOT EXISTS idx_assets_source_id ON assets(source_id);

CREATE TABLE IF NOT EXISTS annotation_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN (
            'queued', 'running', 'succeeded', 'partial_failed',
            'failed', 'cancelled'
        )
    ),
    stage TEXT CHECK (
        stage IS NULL OR stage IN (
            'grounding_dino', 'hazard_rules', 'sam', 'qwen_facts',
            'qwen_prompts', 'build_review_tasks'
        )
    ),
    pipeline_version TEXT NOT NULL,
    requested_categories_json TEXT NOT NULL,
    options_json TEXT NOT NULL,
    progress_json TEXT NOT NULL,
    stages_json TEXT NOT NULL,
    errors_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status
ON annotation_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS job_assets (
    job_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    error_json TEXT,
    PRIMARY KEY (job_id, asset_id),
    UNIQUE (job_id, ordinal),
    FOREIGN KEY (job_id)
        REFERENCES annotation_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id)
);

CREATE INDEX IF NOT EXISTS idx_job_assets_status
ON job_assets(job_id, status);

CREATE TABLE IF NOT EXISTS detections (
    detection_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    x1 REAL NOT NULL CHECK (x1 >= 0),
    y1 REAL NOT NULL CHECK (y1 >= 0),
    x2 REAL NOT NULL CHECK (x2 > x1),
    y2 REAL NOT NULL CHECK (y2 > y1),
    box_score REAL NOT NULL CHECK (box_score >= 0 AND box_score <= 1),
    phrase_score REAL NOT NULL
        CHECK (phrase_score >= 0 AND phrase_score <= 1),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id)
        REFERENCES annotation_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_job_asset
ON detections(job_id, asset_id);

CREATE TABLE IF NOT EXISTS annotation_tasks (
    task_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    category TEXT NOT NULL CHECK (
        category IN (
            'helmet_missing', 'no_helmet', 'no_jacket',
            'harness_missing', 'equipment_proximity',
            'opening_unprotected', 'guardrail_missing',
            'poor_housekeeping', 'safe', 'unsafe'
        )
    ),
    status TEXT NOT NULL CHECK (
        status IN (
            'generated', 'annotating', 'review_pending',
            'changes_requested', 'needs_expert', 'accepted',
            'rejected', 'frozen'
        )
    ),
    version INTEGER NOT NULL CHECK (version >= 1),
    annotation_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    primary_result TEXT,
    annotator_id TEXT,
    reviewer_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id)
        REFERENCES annotation_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (asset_id)
        REFERENCES assets(asset_id),
    UNIQUE (task_id, version)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status
ON annotation_tasks(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_tasks_asset
ON annotation_tasks(asset_id);
CREATE INDEX IF NOT EXISTS idx_tasks_job
ON annotation_tasks(job_id);
CREATE INDEX IF NOT EXISTS idx_tasks_category
ON annotation_tasks(category, status);

CREATE TABLE IF NOT EXISTS task_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    annotation_json TEXT NOT NULL,
    status TEXT NOT NULL,
    editor_id TEXT,
    change_kind TEXT NOT NULL CHECK (
        change_kind IN ('generated', 'draft', 'submit', 'review', 'freeze')
    ),
    created_at TEXT NOT NULL,
    UNIQUE (task_id, version),
    FOREIGN KEY (task_id)
        REFERENCES annotation_tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_version INTEGER NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (
        decision IN ('accept', 'request_changes', 'needs_expert', 'reject')
    ),
    primary_result TEXT NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id)
        REFERENCES annotation_tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_task
ON reviews(task_id, created_at);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL CHECK (
        artifact_type IN ('detections', 'mask', 'mask-overlay', 'crop')
    ),
    operation_id TEXT,
    file_path TEXT NOT NULL,
    media_type TEXT NOT NULL
        CHECK (media_type IN ('image/jpeg', 'image/png')),
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    width INTEGER CHECK (width IS NULL OR width > 0),
    height INTEGER CHECK (height IS NULL OR height > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id)
        REFERENCES annotation_tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_task_type
ON artifacts(task_id, artifact_type, created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_operation
ON artifacts(operation_id);

CREATE TABLE IF NOT EXISTS releases (
    release_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'building', 'succeeded', 'failed')
    ),
    task_filter_json TEXT NOT NULL,
    split_policy_json TEXT NOT NULL,
    counts_json TEXT,
    manifest_path TEXT,
    manifest_sha256 TEXT,
    archive_path TEXT,
    archive_sha256 TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_releases_status
ON releases(status, created_at);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    scope TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_sha256 TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    PRIMARY KEY (scope, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_idempotency_expires_at
ON idempotency_keys(expires_at);
"""


SCHEMA_V2 = (
    """
    ALTER TABLE annotation_jobs
    ADD COLUMN claimed_by TEXT
    """,
    """
    ALTER TABLE annotation_jobs
    ADD COLUMN lease_expires_at TEXT
    """,
    """
    ALTER TABLE annotation_jobs
    ADD COLUMN heartbeat_at TEXT
    """,
    """
    ALTER TABLE annotation_jobs
    ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_count >= 0)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_jobs_claimable
    ON annotation_jobs(status, lease_expires_at, created_at)
    """,
)


SCHEMA_V3 = (
    """
    CREATE TABLE IF NOT EXISTS hazard_candidates (
        hazard_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        category TEXT NOT NULL CHECK (
            category IN (
                'helmet_missing', 'no_helmet', 'no_jacket',
                'harness_missing', 'equipment_proximity',
                'opening_unprotected', 'guardrail_missing',
                'poor_housekeeping', 'safe', 'unsafe'
            )
        ),
        target_entity TEXT NOT NULL,
        target_detection_ids_json TEXT NOT NULL,
        x1 REAL NOT NULL CHECK (x1 >= 0),
        y1 REAL NOT NULL CHECK (y1 >= 0),
        x2 REAL NOT NULL CHECK (x2 > x1),
        y2 REAL NOT NULL CHECK (y2 > y1),
        confidence REAL NOT NULL
            CHECK (confidence >= 0 AND confidence <= 1),
        rule_id TEXT NOT NULL,
        rule_version TEXT NOT NULL,
        evidence_json TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (job_id)
            REFERENCES annotation_jobs(job_id) ON DELETE CASCADE,
        FOREIGN KEY (asset_id)
            REFERENCES assets(asset_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_hazard_candidates_job_asset
    ON hazard_candidates(job_id, asset_id, category)
    """,
)


SCHEMA_V4 = (
    """
    ALTER TABLE task_versions
    ADD COLUMN comment TEXT
    """,
)


SCHEMA_V5 = (
    """
    ALTER TABLE annotation_tasks
    ADD COLUMN source_hazard_id TEXT
        REFERENCES hazard_candidates(hazard_id)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_source_hazard
    ON annotation_tasks(source_hazard_id)
    WHERE source_hazard_id IS NOT NULL
    """,
    """
    ALTER TABLE releases
    ADD COLUMN claimed_by TEXT
    """,
    """
    ALTER TABLE releases
    ADD COLUMN claim_token TEXT
    """,
    """
    ALTER TABLE releases
    ADD COLUMN lease_expires_at TEXT
    """,
    """
    ALTER TABLE releases
    ADD COLUMN heartbeat_at TEXT
    """,
    """
    ALTER TABLE releases
    ADD COLUMN started_at TEXT
    """,
    """
    ALTER TABLE releases
    ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_count >= 0)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_releases_claimable
    ON releases(status, lease_expires_at, created_at)
    """,
)


SCHEMA_V6 = (
    """
    CREATE TABLE IF NOT EXISTS annotation_operations (
        operation_id TEXT PRIMARY KEY,
        operation_type TEXT NOT NULL CHECK (
            operation_type IN ('mask_candidate', 'prompt_enrichment')
        ),
        task_id TEXT NOT NULL,
        task_version INTEGER NOT NULL CHECK (task_version >= 1),
        status TEXT NOT NULL CHECK (
            status IN ('queued', 'running', 'succeeded', 'failed')
        ),
        request_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        error_json TEXT,
        claimed_by TEXT,
        lease_expires_at TEXT,
        heartbeat_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0
            CHECK (attempt_count >= 0),
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        FOREIGN KEY (task_id)
            REFERENCES annotation_tasks(task_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_operations_claimable
    ON annotation_operations(
        operation_type, status, lease_expires_at, created_at
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_operations_task
    ON annotation_operations(task_id, created_at)
    """,
)


SCHEMA_V7 = (
    """
    ALTER TABLE annotation_jobs
    ADD COLUMN grounding_prompt TEXT NOT NULL DEFAULT ''
    """,
    """
    CREATE TABLE IF NOT EXISTS job_artifacts (
        artifact_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        artifact_type TEXT NOT NULL CHECK (
            artifact_type IN ('bbox-image')
        ),
        file_path TEXT NOT NULL,
        media_type TEXT NOT NULL CHECK (media_type = 'image/png'),
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
        width INTEGER NOT NULL CHECK (width > 0),
        height INTEGER NOT NULL CHECK (height > 0),
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE (job_id, asset_id, artifact_type),
        FOREIGN KEY (job_id, asset_id)
            REFERENCES job_assets(job_id, asset_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_job_artifacts_job_asset
    ON job_artifacts(job_id, asset_id, artifact_type)
    """,
)


SCHEMA_V8 = (
    """
    ALTER TABLE annotation_tasks
    ADD COLUMN source_detection_id TEXT
        REFERENCES detections(detection_id)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_source_detection
    ON annotation_tasks(source_detection_id)
    WHERE source_detection_id IS NOT NULL
    """,
    """
    ALTER TABLE task_versions RENAME TO task_versions_v7
    """,
    """
    CREATE TABLE task_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 1),
        annotation_json TEXT NOT NULL,
        status TEXT NOT NULL,
        editor_id TEXT,
        change_kind TEXT NOT NULL CHECK (
            change_kind IN (
                'generated', 'draft', 'submit', 'review', 'freeze',
                'invalidate'
            )
        ),
        comment TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (task_id, version),
        FOREIGN KEY (task_id)
            REFERENCES annotation_tasks(task_id) ON DELETE CASCADE
    )
    """,
    """
    INSERT INTO task_versions (
        id, task_id, version, annotation_json, status, editor_id,
        change_kind, comment, created_at
    )
    SELECT
        id, task_id, version, annotation_json, status, editor_id,
        change_kind, comment, created_at
    FROM task_versions_v7
    """,
    """
    DROP TABLE task_versions_v7
    """,
    """
    ALTER TABLE annotation_operations RENAME TO annotation_operations_v7
    """,
    """
    CREATE TABLE annotation_operations (
        operation_id TEXT PRIMARY KEY,
        operation_type TEXT NOT NULL CHECK (
            operation_type IN ('mask_candidate', 'prompt_enrichment')
        ),
        task_id TEXT NOT NULL,
        task_version INTEGER NOT NULL CHECK (task_version >= 1),
        status TEXT NOT NULL CHECK (
            status IN (
                'queued', 'running', 'succeeded', 'failed', 'cancelled'
            )
        ),
        request_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        error_json TEXT,
        claimed_by TEXT,
        lease_expires_at TEXT,
        heartbeat_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0
            CHECK (attempt_count >= 0),
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        FOREIGN KEY (task_id)
            REFERENCES annotation_tasks(task_id) ON DELETE CASCADE
    )
    """,
    """
    INSERT INTO annotation_operations (
        operation_id, operation_type, task_id, task_version, status,
        request_json, result_json, error_json, claimed_by,
        lease_expires_at, heartbeat_at, attempt_count, created_at,
        started_at, completed_at
    )
    SELECT
        operation_id, operation_type, task_id, task_version, status,
        request_json, result_json, error_json, claimed_by,
        lease_expires_at, heartbeat_at, attempt_count, created_at,
        started_at, completed_at
    FROM annotation_operations_v7
    """,
    """
    DROP TABLE annotation_operations_v7
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_operations_claimable
    ON annotation_operations(
        operation_type, status, lease_expires_at, created_at
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_operations_task
    ON annotation_operations(task_id, created_at)
    """,
)


SCHEMA_V9 = (
    """
    ALTER TABLE annotation_jobs
    ADD COLUMN grounding_prompt_route_json TEXT
    """,
)


SCHEMA_V10 = (
    """
    CREATE TABLE annotation_task_groups (
        task_group_id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode = 'joint'),
        created_at TEXT NOT NULL,
        FOREIGN KEY (asset_id)
            REFERENCES assets(asset_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE annotation_task_group_members (
        task_group_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        task_version INTEGER NOT NULL CHECK (task_version >= 1),
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        PRIMARY KEY (task_group_id, task_id),
        UNIQUE (task_group_id, ordinal),
        FOREIGN KEY (task_group_id)
            REFERENCES annotation_task_groups(task_group_id) ON DELETE CASCADE,
        FOREIGN KEY (task_id)
            REFERENCES annotation_tasks(task_id) ON DELETE RESTRICT
    )
    """,
    """
    CREATE INDEX idx_task_group_members_task
    ON annotation_task_group_members(task_id, task_group_id)
    """,
    """
    ALTER TABLE annotation_operations
    RENAME TO annotation_operations_v9
    """,
    """
    CREATE TABLE annotation_operations (
        operation_id TEXT PRIMARY KEY,
        operation_type TEXT NOT NULL CHECK (
            operation_type IN (
                'mask_candidate',
                'prompt_enrichment',
                'joint_prompt_enrichment'
            )
        ),
        task_id TEXT,
        task_version INTEGER CHECK (
            task_version IS NULL OR task_version >= 1
        ),
        task_group_id TEXT UNIQUE,
        status TEXT NOT NULL CHECK (
            status IN (
                'queued', 'running', 'succeeded', 'failed', 'cancelled'
            )
        ),
        request_json TEXT NOT NULL DEFAULT '{}',
        result_json TEXT,
        error_json TEXT,
        claimed_by TEXT,
        lease_expires_at TEXT,
        heartbeat_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0
            CHECK (attempt_count >= 0),
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        CHECK (
            (
                operation_type = 'joint_prompt_enrichment'
                AND task_group_id IS NOT NULL
                AND task_id IS NULL
                AND task_version IS NULL
            )
            OR
            (
                operation_type IN (
                    'mask_candidate', 'prompt_enrichment'
                )
                AND task_group_id IS NULL
                AND task_id IS NOT NULL
                AND task_version IS NOT NULL
            )
        ),
        FOREIGN KEY (task_id)
            REFERENCES annotation_tasks(task_id) ON DELETE CASCADE,
        FOREIGN KEY (task_group_id)
            REFERENCES annotation_task_groups(task_group_id) ON DELETE CASCADE
    )
    """,
    """
    INSERT INTO annotation_operations (
        operation_id, operation_type, task_id, task_version,
        task_group_id, status, request_json, result_json, error_json,
        claimed_by, lease_expires_at, heartbeat_at, attempt_count,
        created_at, started_at, completed_at
    )
    SELECT
        operation_id, operation_type, task_id, task_version,
        NULL, status, request_json, result_json, error_json,
        claimed_by, lease_expires_at, heartbeat_at, attempt_count,
        created_at, started_at, completed_at
    FROM annotation_operations_v9
    """,
    """
    DROP TABLE annotation_operations_v9
    """,
    """
    CREATE INDEX idx_operations_claimable
    ON annotation_operations(
        operation_type, status, lease_expires_at, created_at
    )
    """,
    """
    CREATE INDEX idx_operations_task
    ON annotation_operations(task_id, created_at)
    """,
)
