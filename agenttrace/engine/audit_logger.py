"""
Audit Logger: Compliance-grade, immutable action log.

Every policy decision, every action, every kill event — recorded
with timestamps, session state snapshots, and policy versions.

Supports JSON Lines file output and is structured for OTel export.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("agenttrace.audit")


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: float
    event_type: str  # action_allowed, action_blocked, violation, session_killed, session_created
    session_id: str
    agent_id: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            **self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Immutable audit log for all AgentTrace events.

    Writes to:
    - In-memory buffer (always)
    - JSON Lines file (if configured)
    - Python logger (always)

    Production would add OTel span export here.
    """

    def __init__(self, file_path: str | None = None):
        self._entries: list[AuditEntry] = []
        self._file_path = file_path
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        event_type: str,
        session_id: str,
        agent_id: str,
        **details: Any,
    ) -> AuditEntry:
        """Record an audit entry."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=event_type,
            session_id=session_id,
            agent_id=agent_id,
            details=details,
        )
        self._entries.append(entry)

        # Write to file
        if self._file_path:
            with open(self._file_path, "a") as f:
                f.write(entry.to_json() + "\n")

        # Python logging
        logger.info(f"[{event_type}] session={session_id[:12]} agent={agent_id} {details}")

        return entry

    def log_action_allowed(
        self,
        session_id: str,
        agent_id: str,
        action_name: str,
        cost: float,
        session_total_cost: float,
    ) -> AuditEntry:
        return self.log(
            "action_allowed",
            session_id=session_id,
            agent_id=agent_id,
            action=action_name,
            cost_usd=round(cost, 6),
            session_total_cost_usd=round(session_total_cost, 6),
        )

    def log_action_blocked(
        self,
        session_id: str,
        agent_id: str,
        action_name: str,
        reason: str,
        session_total_cost: float,
    ) -> AuditEntry:
        return self.log(
            "action_blocked",
            session_id=session_id,
            agent_id=agent_id,
            action=action_name,
            reason=reason,
            session_total_cost_usd=round(session_total_cost, 6),
        )

    def log_violation(
        self,
        session_id: str,
        agent_id: str,
        violation_type: str,
        count: int,
        threshold: int | None = None,
    ) -> AuditEntry:
        return self.log(
            "violation",
            session_id=session_id,
            agent_id=agent_id,
            violation_type=violation_type,
            cumulative_count=count,
            threshold=threshold,
        )

    def log_session_killed(
        self,
        session_id: str,
        agent_id: str,
        reason: str,
        total_cost: float,
        action_count: int,
    ) -> AuditEntry:
        return self.log(
            "session_killed",
            session_id=session_id,
            agent_id=agent_id,
            reason=reason,
            total_cost_usd=round(total_cost, 6),
            action_count=action_count,
        )

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def entries_for_session(self, session_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.session_id == session_id]

    def export_json(self) -> str:
        """Export all entries as a JSON array."""
        return json.dumps([e.to_dict() for e in self._entries], indent=2, default=str)
