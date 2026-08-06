"""GPU worker components.

This package is intentionally not imported by the FastAPI application. Model
dependencies are loaded lazily only after a compatible queued job is claimed.
"""

from .grounding_dino import (
    GroundingDINOAdapter,
    GroundingDINODetection,
    GroundingDINOModelConfig,
)
from .runner import GroundingDINOJobWorker
from .settings import GroundingDINOWorkerSettings

__all__ = [
    "GroundingDINOAdapter",
    "GroundingDINODetection",
    "GroundingDINOJobWorker",
    "GroundingDINOModelConfig",
    "GroundingDINOWorkerSettings",
]
