"""
Session Manager: The core state that the existing ecosystem lacks.

LangChain middleware, NeMo, Guardrails AI, LLM Guard — all stateless.
They evaluate each request independently. The Session Manager maintains
cumulative state across an entire agent session: cost, violation counts,
action count, duration.

This is the gap AgentTrace fills.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class SessionState(str, Enum):
    """Session lifecycle states."""
    ACTIVE = "active"
    ALERT = "alert"          # Budget alert threshold hit
    KILLED = "killed"        # Kill-switch triggered
    COMPLETED = "completed"  # Graceful completion
    EXPIRED = "expired"      # Max duration exceeded


@dataclass
class ViolationRecord:
    """A single violation event."""
    violation_type: str
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)
    action_index: int = 0


@dataclass
class ActionRecord:
    """A single action taken by the agent."""
    action_name: str
    timestamp: float
    cost: float = 0.0
    blocked: bool = False
    block_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Session:
    """
    Represents a single agent session with cumulative state.

    This is what doesn't exist in the current ecosystem:
    - LangChain PIIMiddleware blocks one PII instance but can't say
      "after 3 PII blocks, terminate the session"
    - NeMo tracks dialog state but not cost or violation counts
    - Langfuse observes but doesn't enforce

    Session tracks everything needed for policy enforcement:
    - Cumulative cost (sum of all actions)
    - Violation counts by type (pii_blocked: 2, scope_violation: 0, ...)
    - Total action count
    - Session duration
    - Full action history for audit
    """

    def __init__(
        self,
        session_id: str | None = None,
        agent_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.state = SessionState.ACTIVE
        self.created_at = time.time()
        self.metadata = metadata or {}

        # Cumulative state — the whole point of AgentTrace
        self._total_cost: float = 0.0
        self._action_count: int = 0
        self._violation_counts: dict[str, int] = {}
        self._actions: list[ActionRecord] = []
        self._violations: list[ViolationRecord] = []
        self._kill_reason: str | None = None

        self._lock = Lock()

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def violation_counts(self) -> dict[str, int]:
        return dict(self._violation_counts)

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return time.time() - self.created_at

    @property
    def is_active(self) -> bool:
        return self.state == SessionState.ACTIVE or self.state == SessionState.ALERT

    @property
    def kill_reason(self) -> str | None:
        return self._kill_reason

    def record_action(self, action: ActionRecord) -> None:
        """Record an action and update cumulative state."""
        with self._lock:
            if not self.is_active:
                raise SessionKilledError(
                    f"Session {self.session_id} is {self.state.value}: {self._kill_reason}"
                )
            action.action_index = self._action_count  # type: ignore[attr-defined]
            self._actions.append(action)
            self._action_count += 1
            self._total_cost += action.cost

    def record_violation(self, violation_type: str, details: dict[str, Any] | None = None) -> int:
        """
        Record a violation and return the new cumulative count for this type.

        This is the key operation that stateless tools can't do.
        LangChain PIIMiddleware blocks PII — but it doesn't know this is the 3rd time.
        We do.
        """
        with self._lock:
            count = self._violation_counts.get(violation_type, 0) + 1
            self._violation_counts[violation_type] = count
            self._violations.append(ViolationRecord(
                violation_type=violation_type,
                timestamp=time.time(),
                details=details or {},
                action_index=self._action_count,
            ))
            return count

    def kill(self, reason: str) -> None:
        """Hard stop this session."""
        with self._lock:
            self.state = SessionState.KILLED
            self._kill_reason = reason

    def set_alert(self) -> None:
        """Mark session as in alert state (approaching limits)."""
        with self._lock:
            if self.state == SessionState.ACTIVE:
                self.state = SessionState.ALERT

    def complete(self) -> None:
        """Mark session as gracefully completed."""
        with self._lock:
            if self.is_active:
                self.state = SessionState.COMPLETED

    def to_audit_dict(self) -> dict[str, Any]:
        """Export session state for audit logging."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "duration_seconds": self.duration,
            "total_cost_usd": round(self._total_cost, 6),
            "action_count": self._action_count,
            "violation_counts": dict(self._violation_counts),
            "kill_reason": self._kill_reason,
            "actions": [
                {
                    "name": a.action_name,
                    "cost": a.cost,
                    "blocked": a.blocked,
                    "block_reason": a.block_reason,
                    "timestamp": a.timestamp,
                }
                for a in self._actions
            ],
            "violations": [
                {
                    "type": v.violation_type,
                    "timestamp": v.timestamp,
                    "details": v.details,
                }
                for v in self._violations
            ],
        }


class SessionManager:
    """
    Manages multiple concurrent sessions.
    Thread-safe session registry with lookup and cleanup.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def create_session(
        self,
        agent_id: str = "default",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = Session(session_id=session_id, agent_id=agent_id, metadata=metadata)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def kill_session(self, session_id: str, reason: str) -> bool:
        session = self._sessions.get(session_id)
        if session and session.is_active:
            session.kill(reason)
            return True
        return False

    def active_sessions(self) -> list[Session]:
        return [s for s in self._sessions.values() if s.is_active]

    def cleanup_completed(self) -> int:
        """Remove completed/killed sessions. Returns count removed."""
        with self._lock:
            to_remove = [
                sid for sid, s in self._sessions.items() if not s.is_active
            ]
            for sid in to_remove:
                del self._sessions[sid]
            return len(to_remove)


class SessionKilledError(Exception):
    """Raised when an action is attempted on a killed/completed session."""
    pass
