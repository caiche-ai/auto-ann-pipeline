from __future__ import annotations

from .adapter import SAMAdapter
from .remote import RemoteSAMProvider


def build_mask_predictor(settings):
    if settings.provider == "remote":
        return RemoteSAMProvider(
            settings.remote_config(),
            model_version=settings.model_version,
            polygon_epsilon=settings.polygon_epsilon,
        )
    config = settings.model_config()
    config.validate()
    return SAMAdapter(config)
