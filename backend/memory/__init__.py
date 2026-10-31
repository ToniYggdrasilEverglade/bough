"""Memory backends. `build_store(settings)` picks the configured one."""
from __future__ import annotations

from ..config import Settings
from .base import LOCAL, MemoryBackend, Owner


def build_store(settings: Settings) -> MemoryBackend:
    if settings.memory_backend == "supabase":
        from .supabase_store import SupabaseStore
        return SupabaseStore(settings.supabase_url, settings.supabase_publishable_key, settings.max_memory_records)
    from .store import MemoryStore
    return MemoryStore(settings.db_path, settings.max_memory_records)


__all__ = ["LOCAL", "MemoryBackend", "Owner", "build_store"]
