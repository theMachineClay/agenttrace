"""
AgentTrace: The session-aware policy engine for AI agents.

This is the top-level orchestrator. It wires together:
- SessionManager (cumulative state)
- PolicyEngine (YAML → decisions)
- CostTracker (tiktoken → budget enforcement)
- KillSwitch (hard stop + notifications)
- AuditLogger (compliance-grade logging)

Usage:
    trace = AgentTrace.from_yaml("policy.yaml")
    session = trace.create_session(agent_id="customer-support-bot")

    # Before each agent action:
    decision = trace.pre_action(session.session_id, "llm_call", estimated_cost=0.03)
    if not decision.action_allowed:
        # Session was killed or action was blocked
        ...

    # After the action:
    trace.post_action(session.session_id, "llm_call", actual_cost=0.028)

    # When a violation is detected (by LangChain PIIMiddleware, etc.):
    trace.record_violation(session.session_id, "pii_blocked")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agenttrace.engine.audit_logger import AuditLogger
from agenttrace.engine.cost_tracker import CostTracker
from agenttrace.engine.kill_switch import KillSwitch, KillEvent
from agenttrace.engine.policy_engine import PolicyAction, PolicyDecision, PolicyEngine
from agenttrace.engine.session import (
    ActionRecord,
    Session,
    SessionKilledError,
    SessionManager,
)

import time

logger = logging.getLogger("agenttrace")


class AgentTrace:
    """
    Top-level API for session-aware policy enforcement.

    This is what doesn't exist in the ecosystem:
    - LangChain middleware is stateless → AgentTrace adds session state
    - NeMo Guardrails handles dialog → AgentTrace handles budgets + violations
    - Langfuse observes → AgentTrace enforces
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        audit_log_path: str | None = None,
    ):
        self.policy_engine = policy_engine
        self.session_manager = SessionManager()
        self.cost_tracker = CostTracker()
        self.audit = AuditLogger(file_path=audit_log_path)

        # Build kill switch from policy
        ks_policy = policy_engine.policy.kill_switch
        self.kill_switch = KillSwitch(
            webhooks=ks_policy.webhooks,
            pagerduty_services=ks_policy.pagerduty_services,
            grace_period_seconds=ks_policy.grace_period_seconds,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentTrace:
        """Create AgentTrace from a YAML policy file."""
        engine = PolicyEngine.from_yaml(path)
        audit_path = engine.policy.audit.file_path if engine.policy.audit.enabled else None
        return cls(policy_engine=engine, audit_log_path=audit_path)

    @classmethod
    def from_dict(cls, policy_dict: dict[str, Any]) -> AgentTrace:
        """Create AgentTrace from a policy dictionary."""
        engine = PolicyEngine.from_dict(policy_dict)
        audit_path = engine.policy.audit.file_path if engine.policy.audit.enabled else None
        return cls(policy_engine=engine, audit_log_path=audit_path)

    @property
    def policy(self):
        return self.policy_engine.policy

    # ── Session lifecycle ──────────────────────────────────────────

    def create_session(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new tracked session."""
        aid = agent_id or self.policy.agent_id
        session = self.session_manager.create_session(
            agent_id=aid, session_id=session_id, metadata=metadata
        )
        self.audit.log(
            "session_created",
            session_id=session.session_id,
            agent_id=aid,
            policy_version=self.policy.version,
            budget_limit=self.policy.budget.max_cost_per_session,
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.session_manager.get_session(session_id)

    # ── Pre-action check (BEFORE the agent acts) ──────────────────

    def pre_action(
        self,
        session_id: str,
        action_name: str,
        estimated_cost: float = 0.0,
        model: str | None = None,
        input_text: str | None = None,
    ) -> PolicyDecision:
        """
        Check policy BEFORE an action executes.

        If model + input_text are provided, estimates cost via tiktoken.
        Otherwise uses the provided estimated_cost.

        Returns a PolicyDecision. If action_allowed is False, the caller
        should not proceed with the action.
        """
        session = self._get_active_session(session_id)

        # Estimate cost if we have text
        if model and input_text and estimated_cost == 0.0:
            estimate = self.cost_tracker.estimate_cost(
                model=model, input_text=input_text
            )
            estimated_cost = estimate.total_cost

        # Evaluate against policy
        decision = self.policy_engine.evaluate_pre_action(
            session_total_cost=session.total_cost,
            session_action_count=session.action_count,
            session_duration=session.duration,
            estimated_cost=estimated_cost,
            action_name=action_name,
        )

        # Act on the decision
        if not decision.action_allowed:
            self.audit.log_action_blocked(
                session_id=session.session_id,
                agent_id=session.agent_id,
                action_name=action_name,
                reason=decision.reason,
                session_total_cost=session.total_cost,
            )
            if decision.action_taken == PolicyAction.KILL:
                self._execute_kill(session, decision.reason)
        elif decision.action_taken == PolicyAction.ALERT:
            session.set_alert()
            self.audit.log(
                "budget_alert",
                session_id=session.session_id,
                agent_id=session.agent_id,
                reason=decision.reason,
            )

        return decision

    # ── Post-action record (AFTER the agent acts) ─────────────────

    def post_action(
        self,
        session_id: str,
        action_name: str,
        actual_cost: float = 0.0,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an action after it completes.
        Updates cumulative cost and action count.
        """
        session = self._get_active_session(session_id)

        # Calculate actual cost if tokens provided
        if model and (input_tokens or output_tokens):
            estimate = self.cost_tracker.estimate_cost(
                model=model,
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
            )
            actual_cost = estimate.total_cost

        record = ActionRecord(
            action_name=action_name,
            timestamp=time.time(),
            cost=actual_cost,
            metadata=metadata or {},
        )
        session.record_action(record)

        self.audit.log_action_allowed(
            session_id=session.session_id,
            agent_id=session.agent_id,
            action_name=action_name,
            cost=actual_cost,
            session_total_cost=session.total_cost,
        )

    # ── Violation recording ───────────────────────────────────────

    def record_violation(
        self,
        session_id: str,
        violation_type: str,
        details: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """
        Record a violation detected by an external scanner.

        This is the integration point with LangChain PIIMiddleware,
        LLM Guard, Presidio, etc. They detect; AgentTrace counts and
        enforces thresholds.

        Example:
            # LangChain PIIMiddleware blocks a PII instance
            # Your integration layer calls:
            decision = trace.record_violation(session_id, "pii_blocked")
            # AgentTrace checks: is this the 3rd PII block? → kill
        """
        session = self._get_active_session(session_id)

        # Record and get cumulative count
        count = session.record_violation(violation_type, details)

        # Get threshold for this violation type
        threshold_config = self.policy.get_violation_threshold(violation_type)
        threshold_val = threshold_config.max_count if threshold_config else None

        self.audit.log_violation(
            session_id=session.session_id,
            agent_id=session.agent_id,
            violation_type=violation_type,
            count=count,
            threshold=threshold_val,
        )

        # Evaluate against policy
        decision = self.policy_engine.evaluate_violation(
            violation_type=violation_type,
            cumulative_count=count,
        )

        if not decision.action_allowed and decision.action_taken == PolicyAction.KILL:
            self._execute_kill(session, decision.reason)

        return decision

    # ── Kill execution ────────────────────────────────────────────

    def _execute_kill(self, session: Session, reason: str) -> KillEvent:
        """Execute the kill switch on a session."""
        event = self.kill_switch.execute_sync(session, reason)
        self.audit.log_session_killed(
            session_id=session.session_id,
            agent_id=session.agent_id,
            reason=reason,
            total_cost=session.total_cost,
            action_count=session.action_count,
        )
        return event

    # ── Session completion ────────────────────────────────────────

    def complete_session(self, session_id: str) -> dict[str, Any]:
        """Gracefully complete a session and return audit summary."""
        session = self._get_active_session(session_id)
        session.complete()
        self.audit.log(
            "session_completed",
            session_id=session.session_id,
            agent_id=session.agent_id,
            total_cost=round(session.total_cost, 6),
            action_count=session.action_count,
            violation_counts=session.violation_counts,
            duration_seconds=round(session.duration, 2),
        )
        return session.to_audit_dict()

    # ── Helpers ───────────────────────────────────────────────────

    def _get_active_session(self, session_id: str) -> Session:
        session = self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if not session.is_active:
            raise SessionKilledError(
                f"Session {session_id} is {session.state.value}: {session.kill_reason}"
            )
        return session
