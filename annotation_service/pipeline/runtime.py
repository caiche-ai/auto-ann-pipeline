from __future__ import annotations

import threading

from ..storage.repository import AnnotationStore


class OperationHeartbeat:
    """Keep an asynchronous operation lease alive while a worker runs."""

    def __init__(
        self,
        *,
        store: AnnotationStore,
        operation_id: str,
        worker_id: str,
        lease_seconds: int,
        interval_seconds: int,
    ):
        self.store = store
        self.operation_id = operation_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"operation-heartbeat-{self.operation_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.store.heartbeat_operation(
                    self.operation_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as exc:
                self._error = exc
                self._stop.set()
                return

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1, self.interval_seconds + 1))
