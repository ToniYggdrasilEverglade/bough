"""Exception types and small helpers shared across the backend."""
from __future__ import annotations

from typing import Iterable


class BrainError(Exception):
    """Base class for expected, user-reportable errors."""


class InputRejected(BrainError):
    """The user's input (or an identifier) failed validation."""


class NotFound(BrainError):
    """A referenced session, cycle or memory does not exist."""


class MemoryRejected(BrainError):
    """A long-term memory write was refused (empty, too long, limit reached...)."""


class ModelError(BrainError):
    """Base class for model-adapter problems."""


class ModelUnavailable(ModelError):
    """Endpoint unreachable, HTTP error, or refused connection."""


class ModelTimeout(ModelError):
    """The model did not answer within the configured timeout."""


class ModelOutputError(ModelError):
    """The model answered, but the output was not valid for the requested schema."""


class ModelFailure(BrainError):
    """Raised by the gateway after retries/budget are exhausted. Agents catch it and fall back."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class LoopGuardError(BrainError):
    """Too many events of the same kind in one cycle (runaway loop protection)."""


class StaleCycleError(BrainError):
    """A write was attempted for a cycle that is no longer the session's current cycle."""


class CycleSuperseded(BrainError):
    """The cycle was cancelled or replaced by a newer request."""


def redact(text: str, secrets: Iterable[str]) -> str:
    """Remove known secret strings from text before it is logged or sent to the UI."""
    out = text
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return out
