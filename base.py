"""Model adapter interface. Components never talk to a provider directly; they use the gateway."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

TASKS = ("plan", "review", "verbalize")


class ModelAdapter(ABC):
    name: str = "unnamed"
    is_mock: bool = False

    @abstractmethod
    async def run(self, task: str, data: Dict[str, Any], *, timeout: float) -> Any:
        """Execute one structured task and return the parsed JSON-compatible result.

        Must raise ModelUnavailable / ModelTimeout / ModelOutputError instead of returning garbage.
        `data` is untrusted-input-bearing, JSON-serialisable context assembled by the attention component.
        """
