"""
Policy Engine: Declarative YAML → runtime enforcement.

Loads policy-as-code from YAML files and evaluates session state against
policy thresholds. This is the decision brain of AgentTrace.

The key insight: policies are about cumulative state, not individual events.
"Block PII" is a scanner's job. "After 3 PII blocks, kill the session" is
a policy engine's job. That's what we do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class PolicyAction(str, Enum):
    """What to do when a threshold is breached."""
    KILL = "kill"       # Terminate the session
    ALERT = "alert"     # Mark session as alert, send notification
    LOG = "log"         # Log only, don't intervene
    BLOCK = "block"     # Block this action but continue session


@dataclass
class BudgetPolicy:
    max_cost_per_session: float = 5.00
    max_cost_per_action: float = 0.50
    alert_at: float = 0.80  # Fraction of budget
    on_exceed: PolicyAction = PolicyAction.KILL


@dataclass
class SessionPolicy:
    max_duration_seconds: int = 1800  # 30 minutes
    max_actions: int = 100


@dataclass
class ViolationThreshold:
    violation_type: str
    max_count: int
    on_threshold: PolicyAction = PolicyAction.KILL


@dataclass
class KillSwitchPolicy:
    enabled: bool = True
    webhooks: list[str] = field(default_factory=list)
    pagerduty_services: list[str] = field(default_factory=list)
    grace_period_seconds: float = 5.0


@dataclass
class AuditPolicy:
    enabled: bool = True
    include_fields: list[str] = field(default_factory=lambda: [
        "action_name", "policy_decision", "cost_incurred", "session_state"
    ])
    otel_endpoint: str | None = None
    file_path: str | None = None


@dataclass
class Policy:
    """Complete policy configuration for an agent."""
    version: str = "1.0"
    agent_id: str = "default"
    budget: BudgetPolicy = field(default_factory=BudgetPolicy)
    session: SessionPolicy = field(default_factory=SessionPolicy)
    violation_thresholds: list[ViolationThreshold] = field(default_factory=list)
    kill_switch: KillSwitchPolicy = field(default_factory=KillSwitchPolicy)
    audit: AuditPolicy = field(default_factory=AuditPolicy)

    def get_violation_threshold(self, violation_type: str) -> ViolationThreshold | None:
        for vt in self.violation_thresholds:
            if vt.violation_type == violation_type:
                return vt
        return None


@dataclass
class PolicyDecision:
    """Result of evaluating session state against policy."""
    action_allowed: bool
    action_taken: PolicyAction
    reason: str
    violation_type: str | None = None
    session_state_snapshot: dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """
    Loads YAML policies and evaluates session state against them.

    Usage:
        engine = PolicyEngine.from_yaml("policy.yaml")
        decision = engine.evaluate_action(session, action_name, estimated_cost)
        decision = engine.evaluate_violation(session, "pii_blocked")
    """

    def __init__(self, policy: Policy):
        self.policy = policy

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyEngine:
        """Load policy from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        policy = cls._parse_policy(raw)
        return cls(policy)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyEngine:
        """Load policy from a dictionary."""
        policy = cls._parse_policy(raw)
        return cls(policy)

    @classmethod
    def _parse_policy(cls, raw: dict[str, Any]) -> Policy:
        """Parse raw YAML dict into a Policy object."""
        policy = Policy(
            version=raw.get("version", "1.0"),
            agent_id=raw.get("agent_id", "default"),
        )

        # Budget
        if "budget" in raw:
            b = raw["budget"]
            policy.budget = BudgetPolicy(
                max_cost_per_session=b.get("max_cost_per_session", 5.00),
                max_cost_per_action=b.get("max_cost_per_action", 0.50),
                alert_at=b.get("alert_at", 0.80),
                on_exceed=PolicyAction(b.get("on_exceed", "kill")),
            )

        # Session
        if "session" in raw:
            s = raw["session"]
            duration = s.get("max_duration", "30m")
            policy.session = SessionPolicy(
                max_duration_seconds=cls._parse_duration(duration),
                max_actions=s.get("max_actions", 100),
            )

        # Violation thresholds
        if "violations" in raw and "thresholds" in raw["violations"]:
            default_action = PolicyAction(
                raw["violations"].get("on_threshold", "kill")
            )
            for vtype, max_count in raw["violations"]["thresholds"].items():
                policy.violation_thresholds.append(ViolationThreshold(
                    violation_type=vtype,
                    max_count=max_count,
                    on_threshold=default_action,
                ))

        # Kill-switch
        if "kill_switch" in raw:
            ks = raw["kill_switch"]
            notify = ks.get("notify", [])
            webhooks = []
            pagerduty = []
            for n in notify:
                if isinstance(n, dict):
                    if "webhook" in n:
                        webhooks.append(n["webhook"])
                    if "pagerduty" in n:
                        pagerduty.append(n["pagerduty"])
            policy.kill_switch = KillSwitchPolicy(
                enabled=ks.get("enabled", True),
                webhooks=webhooks,
                pagerduty_services=pagerduty,
                grace_period_seconds=cls._parse_duration(
                    ks.get("grace_period", "5s")
                ),
            )

        # Audit
        if "audit" in raw:
            a = raw["audit"]
            export = a.get("export", {})
            otel_endpoint = None
            file_path = None
            if isinstance(export, list):
                for e in export:
                    if isinstance(e, dict):
                        otel_endpoint = otel_endpoint or e.get("otel")
                        file_path = file_path or e.get("file")
            policy.audit = AuditPolicy(
                enabled=a.get("enabled", True),
                include_fields=a.get("include", []),
                otel_endpoint=otel_endpoint,
                file_path=file_path,
            )

        return policy

    @staticmethod
    def _parse_duration(value: str | int | float) -> float:
        """Parse duration string (e.g. '30m', '5s', '1h') to seconds."""
        if isinstance(value, (int, float)):
            return float(value)
        value = str(value).strip().lower()
        if value.endswith("s"):
            return float(value[:-1])
        elif value.endswith("m"):
            return float(value[:-1]) * 60
        elif value.endswith("h"):
            return float(value[:-1]) * 3600
        return float(value)

    def evaluate_pre_action(
        self,
        session_total_cost: float,
        session_action_count: int,
        session_duration: float,
        estimated_cost: float,
        action_name: str,
    ) -> PolicyDecision:
        """
        Evaluate whether an action should proceed, BEFORE execution.
        Checks: budget, action count, session duration.
        """
        snapshot = {
            "total_cost": session_total_cost,
            "action_count": session_action_count,
            "duration_seconds": session_duration,
            "estimated_cost": estimated_cost,
            "action": action_name,
        }

        # Check session duration
        if session_duration > self.policy.session.max_duration_seconds:
            return PolicyDecision(
                action_allowed=False,
                action_taken=PolicyAction.KILL,
                reason=(
                    f"Session duration {session_duration:.0f}s exceeds "
                    f"limit {self.policy.session.max_duration_seconds}s"
                ),
                session_state_snapshot=snapshot,
            )

        # Check action count
        if session_action_count >= self.policy.session.max_actions:
            return PolicyDecision(
                action_allowed=False,
                action_taken=PolicyAction.KILL,
                reason=(
                    f"Action count {session_action_count} reached "
                    f"limit {self.policy.session.max_actions}"
                ),
                session_state_snapshot=snapshot,
            )

        # Check per-action cost
        if estimated_cost > self.policy.budget.max_cost_per_action:
            return PolicyDecision(
                action_allowed=False,
                action_taken=PolicyAction.BLOCK,
                reason=(
                    f"Action cost ${estimated_cost:.4f} exceeds "
                    f"per-action limit ${self.policy.budget.max_cost_per_action:.4f}"
                ),
                session_state_snapshot=snapshot,
            )

        # Check session budget
        cost_after = session_total_cost + estimated_cost
        if cost_after > self.policy.budget.max_cost_per_session:
            return PolicyDecision(
                action_allowed=False,
                action_taken=self.policy.budget.on_exceed,
                reason=(
                    f"Session cost would reach ${cost_after:.4f}, "
                    f"exceeding budget ${self.policy.budget.max_cost_per_session:.2f}"
                ),
                session_state_snapshot=snapshot,
            )

        # Check alert threshold
        utilization = cost_after / self.policy.budget.max_cost_per_session
        if utilization >= self.policy.budget.alert_at:
            return PolicyDecision(
                action_allowed=True,
                action_taken=PolicyAction.ALERT,
                reason=(
                    f"Budget utilization at {utilization:.0%} "
                    f"(${cost_after:.4f} / ${self.policy.budget.max_cost_per_session:.2f})"
                ),
                session_state_snapshot=snapshot,
            )

        return PolicyDecision(
            action_allowed=True,
            action_taken=PolicyAction.LOG,
            reason="Action within policy limits",
            session_state_snapshot=snapshot,
        )

    def evaluate_violation(
        self,
        violation_type: str,
        cumulative_count: int,
    ) -> PolicyDecision:
        """
        Evaluate a violation against thresholds.

        This is the core of session-aware enforcement:
        "This is the 3rd PII block. Threshold is 3. Kill the session."
        """
        threshold = self.policy.get_violation_threshold(violation_type)

        if threshold is None:
            return PolicyDecision(
                action_allowed=True,
                action_taken=PolicyAction.LOG,
                reason=f"No threshold configured for '{violation_type}'",
                violation_type=violation_type,
            )

        if cumulative_count >= threshold.max_count:
            return PolicyDecision(
                action_allowed=False,
                action_taken=threshold.on_threshold,
                reason=(
                    f"Violation '{violation_type}' count {cumulative_count} "
                    f"reached threshold {threshold.max_count}"
                ),
                violation_type=violation_type,
                session_state_snapshot={
                    "violation_type": violation_type,
                    "count": cumulative_count,
                    "threshold": threshold.max_count,
                },
            )

        return PolicyDecision(
            action_allowed=True,
            action_taken=PolicyAction.LOG,
            reason=(
                f"Violation '{violation_type}' count {cumulative_count} "
                f"below threshold {threshold.max_count}"
            ),
            violation_type=violation_type,
        )
