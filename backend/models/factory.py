from __future__ import annotations

from ..config import Settings
from .base import ModelAdapter
from .http_model import HttpChatModel
from .mock import MockModel


def build_adapter(settings: Settings) -> ModelAdapter:
    if settings.model_backend == "http":
        return HttpChatModel(settings.model_url, settings.model_name, settings.model_api_key,
                             settings.model_json_mode,
                             completion_tokens={"plan": settings.model_plan_tokens,
                                                "review": settings.model_review_tokens,
                                                "verbalize": settings.model_reply_tokens,
                                                "root": settings.model_reply_tokens,
                                                "repair": settings.model_reply_tokens})
    return MockModel(latency_s=settings.mock_latency_s)
