from __future__ import annotations

from .errors import InvalidStateTransitionError
from .schemas import JobStatus, ReleaseStatus, TaskStatus


TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.GENERATED: frozenset(
        {
            TaskStatus.ANNOTATING,
            TaskStatus.REVIEW_PENDING,
            TaskStatus.REJECTED,
        }
    ),
    TaskStatus.ANNOTATING: frozenset(
        {TaskStatus.REVIEW_PENDING, TaskStatus.REJECTED}
    ),
    TaskStatus.REVIEW_PENDING: frozenset(
        {
            TaskStatus.ACCEPTED,
            TaskStatus.CHANGES_REQUESTED,
            TaskStatus.NEEDS_EXPERT,
            TaskStatus.REJECTED,
        }
    ),
    TaskStatus.CHANGES_REQUESTED: frozenset(
        {TaskStatus.ANNOTATING, TaskStatus.REJECTED}
    ),
    TaskStatus.NEEDS_EXPERT: frozenset(
        {
            TaskStatus.ACCEPTED,
            TaskStatus.CHANGES_REQUESTED,
            TaskStatus.REJECTED,
        }
    ),
    TaskStatus.ACCEPTED: frozenset({TaskStatus.FROZEN}),
    TaskStatus.REJECTED: frozenset(),
    TaskStatus.FROZEN: frozenset(),
}

JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.SUCCEEDED,
            JobStatus.PARTIAL_FAILED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.PARTIAL_FAILED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}

RELEASE_TRANSITIONS: dict[ReleaseStatus, frozenset[ReleaseStatus]] = {
    ReleaseStatus.QUEUED: frozenset(
        {ReleaseStatus.BUILDING, ReleaseStatus.FAILED}
    ),
    ReleaseStatus.BUILDING: frozenset(
        {ReleaseStatus.SUCCEEDED, ReleaseStatus.FAILED}
    ),
    ReleaseStatus.SUCCEEDED: frozenset(),
    ReleaseStatus.FAILED: frozenset(),
}


def _ensure_transition(
    resource: str,
    current,
    target,
    transitions,
) -> None:
    if target not in transitions[current]:
        raise InvalidStateTransitionError(
            f"cannot transition {resource} from {current.value} "
            f"to {target.value}",
            details=[
                {
                    "field": "status",
                    "reason": (
                        f"{current.value} -> {target.value} is not allowed"
                    ),
                }
            ],
        )


def ensure_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    _ensure_transition("task", current, target, TASK_TRANSITIONS)


def ensure_job_transition(current: JobStatus, target: JobStatus) -> None:
    _ensure_transition("job", current, target, JOB_TRANSITIONS)


def ensure_release_transition(
    current: ReleaseStatus,
    target: ReleaseStatus,
) -> None:
    _ensure_transition("release", current, target, RELEASE_TRANSITIONS)
