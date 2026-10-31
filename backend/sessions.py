"""Session registry. Each session owns its workspace, memory toggle, pending task and feedback state."""
from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import Settings
from .context import CycleContext
from .errors import InputRejected, NotFound
from .memory.base import LOCAL, Owner
from .schemas import utcnow_iso
from .workspace import Workspace

HISTORY_MAX_EXCHANGES = 12
HISTORY_MAX_CHARS = 24000

_SID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Internal ids are "<owner user id>:<client sid>" when auth is on, so two users' "default" sessions
# (and their event streams) can never collide. Owner ids are Supabase UUIDs.
_SCOPED = re.compile(r"^(?:[0-9a-fA-F-]{1,36}:)?[A-Za-z0-9_-]{1,64}$")


@dataclass
class Session:
    session_id: str
    workspace: Workspace
    owner: Owner = LOCAL                             # refreshed on every authenticated request (token rotates)
    memory_enabled: bool = False                      # persistent memory is opt-in
    current_cycle_id: Optional[str] = None
    task: Optional["asyncio.Task"] = None
    ctx: Optional[CycleContext] = None
    bias: Dict[str, float] = field(default_factory=dict)     # session-only adaptation (memory off)
    records: "OrderedDict[str, Dict[str, Any]]" = field(default_factory=OrderedDict)
    created_at: str = field(default_factory=utcnow_iso)

    conversation: list[Dict[str, str]] = field(default_factory=list)
    history_dropped: int = 0

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        self.conversation.append({"user": user_text, "assistant": assistant_text})
        while (len(self.conversation) > HISTORY_MAX_EXCHANGES or
               sum(len(x["user"]) + len(x["assistant"]) for x in self.conversation) > HISTORY_MAX_CHARS):
            self.conversation.pop(0)
            self.history_dropped += 1

    def conversation_snapshot(self) -> Dict[str, Any]:
        messages = []
        for exchange in self.conversation:
            messages.extend([
                {"role": "user", "content": exchange["user"]},
                {"role": "assistant", "content": exchange["assistant"]},
            ])
        return {"messages": messages, "dropped_exchanges": self.history_dropped,
                "max_exchanges": HISTORY_MAX_EXCHANGES, "max_chars": HISTORY_MAX_CHARS}

    def clear_conversation(self) -> None:
        self.conversation.clear()
        self.history_dropped = 0

    def add_record(self, cycle_id: str, rec: Dict[str, Any], limit: int = 50) -> None:
        self.records[cycle_id] = rec
        while len(self.records) > limit:
            self.records.popitem(last=False)


class SessionManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()

    @staticmethod
    def validate_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SID.match(session_id):
            raise InputRejected("session id must be 1-64 characters: letters, digits, '-' or '_'")
        return session_id

    @classmethod
    def scoped_id(cls, owner: Owner, session_id: str) -> str:
        """Namespace a client-chosen session id by its owner. LOCAL (auth off) keeps the bare id."""
        cls.validate_id(session_id)
        return session_id if owner is LOCAL or owner.user_id == LOCAL.user_id else f"{owner.user_id}:{session_id}"

    def get(self, session_id: str, create: bool = True) -> Session:
        if not isinstance(session_id, str) or not _SCOPED.match(session_id):
            raise InputRejected("session id must be 1-64 characters: letters, digits, '-' or '_'")
        s = self._sessions.get(session_id)
        if s is None:
            if not create:
                raise NotFound("unknown session")
            s = Session(session_id, Workspace(session_id, self.settings.workspace_max_items))
            self._sessions[session_id] = s
            while len(self._sessions) > self.settings.max_sessions:
                for sid, old in self._sessions.items():
                    if old.task is None or old.task.done():
                        del self._sessions[sid]
                        break
                else:
                    break
        else:
            self._sessions.move_to_end(session_id)
        return s

    def __len__(self) -> int:
        return len(self._sessions)
