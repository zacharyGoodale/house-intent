"""SessionStore (spec §6.9): in-memory, keyed by session_id, with TTLs for pending clarifications and recent targets."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ha_jev.models import PlanItem


@dataclass
class Pending:
    """A pending clarification (§6.9). `expires` is on the store's clock (monotonic); policy never reads a clock."""

    kind: str
    question: str
    options: Mapping[str, str]
    plan: Mapping[str, PlanItem]
    expires: float


@dataclass
class SessionSnapshot:
    recent_targets: list[str] = field(default_factory=list)
    pending: Pending | None = None
    luna_response_id: str | None = None


@dataclass
class _Entry:
    pending: Pending | None = None
    recent_targets: list[str] = field(default_factory=list)
    recent_expires: float = 0.0
    luna_response_id: str | None = None
    hermes_expires: float = 0.0
    hermes_person: str | None = None


class SessionStore:
    def __init__(self, pending_ttl_s: float = 60, recent_ttl_s: float = 300,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.pending_ttl_s = pending_ttl_s
        self.recent_ttl_s = recent_ttl_s
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    def snapshot(self, session_id: str | None) -> SessionSnapshot:
        if not session_id or session_id not in self._entries:
            return SessionSnapshot()
        e = self._entries[session_id]
        now = self._clock()
        pending = e.pending if e.pending is not None and e.pending.expires > now else None
        recent = list(e.recent_targets) if e.recent_expires > now else []
        return SessionSnapshot(recent_targets=recent, pending=pending, luna_response_id=e.luna_response_id)

    def clear_pending(self, session_id: str | None) -> None:
        if session_id and session_id in self._entries:
            self._entries[session_id].pending = None

    def set_pending(self, session_id: str | None, kind: str, question: str, options, plan) -> None:
        if not session_id:
            return
        e = self._entries.setdefault(session_id, _Entry())
        e.pending = Pending(kind=kind, question=question, options=dict(options), plan=dict(plan),
                            expires=self._clock() + self.pending_ttl_s)

    def set_recent(self, session_id: str | None, targets: list[str]) -> None:
        if not session_id or not targets:
            return
        e = self._entries.setdefault(session_id, _Entry())
        e.recent_targets = list(dict.fromkeys(targets))
        e.recent_expires = self._clock() + self.recent_ttl_s

    def set_luna_response(self, session_id: str | None, response_id: str | None) -> None:
        if not session_id or not response_id:
            return
        self._entries.setdefault(session_id, _Entry()).luna_response_id = response_id

    def hermes_response(self, session_id: str | None, person: str | None) -> str | None:
        """Expire the sticky conversation and its history together, using a monotonic idle clock."""
        e = self._entries.get(session_id or "")
        if e is None or not e.hermes_expires:
            return None
        if e.hermes_expires <= self._clock() or e.hermes_person != person:
            self.clear_hermes(session_id)
            return None
        return e.luna_response_id

    def keep_hermes(self, session_id: str | None, person: str | None, idle_s: float) -> None:
        if not session_id:
            return
        e = self._entries.setdefault(session_id, _Entry())
        e.hermes_expires = self._clock() + idle_s
        e.hermes_person = person
        # Agent-side changes cannot update Jev's device targets/clarifications reliably.
        e.pending = None
        e.recent_targets = []

    def clear_hermes(self, session_id: str | None) -> None:
        e = self._entries.get(session_id or "")
        if e is not None:
            e.hermes_expires = 0.0
            e.hermes_person = None
            e.luna_response_id = None
