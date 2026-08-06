import unittest

from annotation_service.errors import InvalidStateTransitionError
from annotation_service.schemas import JobStatus, ReleaseStatus, TaskStatus
from annotation_service.state_machine import (
    ensure_job_transition,
    ensure_release_transition,
    ensure_task_transition,
)


class StateMachineTest(unittest.TestCase):
    def test_task_happy_path(self):
        ensure_task_transition(
            TaskStatus.GENERATED,
            TaskStatus.ANNOTATING,
        )
        ensure_task_transition(
            TaskStatus.ANNOTATING,
            TaskStatus.REVIEW_PENDING,
        )
        ensure_task_transition(
            TaskStatus.REVIEW_PENDING,
            TaskStatus.ACCEPTED,
        )
        ensure_task_transition(
            TaskStatus.ACCEPTED,
            TaskStatus.FROZEN,
        )

    def test_frozen_task_is_terminal(self):
        with self.assertRaises(InvalidStateTransitionError):
            ensure_task_transition(
                TaskStatus.FROZEN,
                TaskStatus.ANNOTATING,
            )

    def test_job_terminal_state_cannot_restart(self):
        with self.assertRaises(InvalidStateTransitionError):
            ensure_job_transition(
                JobStatus.SUCCEEDED,
                JobStatus.RUNNING,
            )

    def test_release_happy_path(self):
        ensure_release_transition(
            ReleaseStatus.QUEUED,
            ReleaseStatus.BUILDING,
        )
        ensure_release_transition(
            ReleaseStatus.BUILDING,
            ReleaseStatus.SUCCEEDED,
        )


if __name__ == "__main__":
    unittest.main()
