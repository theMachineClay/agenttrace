"""Tests for AgentTrace core components."""

import pytest
import time


from agenttrace.engine.session import Session, SessionState, SessionKilledError
from agenttrace.engine.cost_tracker import CostTracker
from agenttrace.engine.policy_engine import PolicyEngine, PolicyAction
from agenttrace.engine.agent_trace import AgentTrace


# ── Session Tests ─────────────────────────────────────────────────

class TestSession:
    def test_create_session(self):
        session = Session(agent_id="test-agent")
        assert session.is_active
        assert session.total_cost == 0.0
        assert session.action_count == 0
        assert session.violation_counts == {}

    def test_record_violation_cumulative(self):
        """The core test: violations accumulate across a session."""
        session = Session(agent_id="test-agent")
        
        count1 = session.record_violation("pii_blocked")
        assert count1 == 1
        
        count2 = session.record_violation("pii_blocked")
        assert count2 == 2
        
        count3 = session.record_violation("pii_blocked")
        assert count3 == 3
        
        assert session.violation_counts == {"pii_blocked": 3}

    def test_kill_prevents_actions(self):
        session = Session(agent_id="test-agent")
        session.kill("budget exceeded")
        
        assert not session.is_active
        assert session.state == SessionState.KILLED
        assert session.kill_reason == "budget exceeded"
        
        from agenttrace.engine.session import ActionRecord
        with pytest.raises(SessionKilledError):
            session.record_action(ActionRecord(
                action_name="test", timestamp=time.time(), cost=0.01
            ))

    def test_audit_dict(self):
        session = Session(agent_id="test-agent")
        session.record_violation("pii_blocked")
        audit = session.to_audit_dict()
        
        assert audit["agent_id"] == "test-agent"
        assert audit["violation_counts"] == {"pii_blocked": 1}
        assert audit["state"] == "active"


# ── Cost Tracker Tests ────────────────────────────────────────────

class TestCostTracker:
    def test_token_counting(self):
        tracker = CostTracker()
        try:
            count = tracker.count_tokens("Hello, world!", "gpt-4o")
            assert count > 0
        except Exception:
            pytest.skip("tiktoken encoding download requires network access")

    def test_cost_estimation(self):
        tracker = CostTracker()
        estimate = tracker.estimate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 500
        assert estimate.total_cost > 0
        # GPT-4o: $2.50/1M input + $10.00/1M output
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(estimate.total_cost - expected) < 0.0001

    def test_budget_check_allowed(self):
        tracker = CostTracker()
        decision = tracker.check_budget(
            session_total_cost=1.00,
            estimated_action_cost=0.10,
            max_cost_per_session=5.00,
        )
        assert decision.allowed
        assert not decision.alert

    def test_budget_check_blocked(self):
        tracker = CostTracker()
        decision = tracker.check_budget(
            session_total_cost=4.80,
            estimated_action_cost=0.30,
            max_cost_per_session=5.00,
        )
        assert not decision.allowed
        assert decision.reason is not None

    def test_budget_check_alert(self):
        tracker = CostTracker()
        decision = tracker.check_budget(
            session_total_cost=4.00,
            estimated_action_cost=0.10,
            max_cost_per_session=5.00,
            alert_threshold=0.80,
        )
        assert decision.allowed
        assert decision.alert  # 4.10 / 5.00 = 82% > 80%


# ── Policy Engine Tests ───────────────────────────────────────────

class TestPolicyEngine:
    def _make_engine(self) -> PolicyEngine:
        return PolicyEngine.from_dict({
            "version": "1.0",
            "agent_id": "test",
            "budget": {
                "max_cost_per_session": 5.00,
                "max_cost_per_action": 0.50,
                "alert_at": 0.80,
                "on_exceed": "kill",
            },
            "session": {"max_duration": "30m", "max_actions": 10},
            "violations": {
                "thresholds": {"pii_blocked": 3, "scope_violation": 1},
                "on_threshold": "kill",
            },
        })

    def test_action_within_budget(self):
        engine = self._make_engine()
        decision = engine.evaluate_pre_action(
            session_total_cost=1.00,
            session_action_count=2,
            session_duration=60.0,
            estimated_cost=0.10,
            action_name="test",
        )
        assert decision.action_allowed

    def test_action_exceeds_budget(self):
        engine = self._make_engine()
        decision = engine.evaluate_pre_action(
            session_total_cost=4.80,
            session_action_count=5,
            session_duration=60.0,
            estimated_cost=0.30,
            action_name="test",
        )
        assert not decision.action_allowed
        assert decision.action_taken == PolicyAction.KILL

    def test_action_exceeds_per_action_limit(self):
        engine = self._make_engine()
        decision = engine.evaluate_pre_action(
            session_total_cost=1.00,
            session_action_count=2,
            session_duration=60.0,
            estimated_cost=0.60,  # > 0.50 limit
            action_name="test",
        )
        assert not decision.action_allowed
        assert decision.action_taken == PolicyAction.BLOCK

    def test_action_count_exceeded(self):
        engine = self._make_engine()
        decision = engine.evaluate_pre_action(
            session_total_cost=1.00,
            session_action_count=10,  # = max
            session_duration=60.0,
            estimated_cost=0.10,
            action_name="test",
        )
        assert not decision.action_allowed

    def test_violation_below_threshold(self):
        engine = self._make_engine()
        decision = engine.evaluate_violation("pii_blocked", 2)
        assert decision.action_allowed

    def test_violation_at_threshold(self):
        """The core test: 3rd PII block → kill."""
        engine = self._make_engine()
        decision = engine.evaluate_violation("pii_blocked", 3)
        assert not decision.action_allowed
        assert decision.action_taken == PolicyAction.KILL

    def test_violation_unknown_type(self):
        engine = self._make_engine()
        decision = engine.evaluate_violation("unknown_violation", 100)
        assert decision.action_allowed  # No threshold → log only

    def test_parse_duration(self):
        assert PolicyEngine._parse_duration("30m") == 1800.0
        assert PolicyEngine._parse_duration("5s") == 5.0
        assert PolicyEngine._parse_duration("1h") == 3600.0
        assert PolicyEngine._parse_duration(120) == 120.0


# ── AgentTrace Integration Tests ──────────────────────────────────

class TestAgentTrace:
    def _make_trace(self) -> AgentTrace:
        return AgentTrace.from_dict({
            "version": "1.0",
            "agent_id": "test-agent",
            "budget": {
                "max_cost_per_session": 2.00,
                "max_cost_per_action": 0.50,
                "alert_at": 0.80,
                "on_exceed": "kill",
            },
            "session": {"max_duration": "30m", "max_actions": 50},
            "violations": {
                "thresholds": {"pii_blocked": 3},
                "on_threshold": "kill",
            },
            "kill_switch": {"enabled": True, "notify": [], "grace_period": "1s"},
            "audit": {"enabled": True},
        })

    def test_full_lifecycle(self):
        """Test: create session → actions → complete."""
        trace = self._make_trace()
        session = trace.create_session()

        # Action within budget
        decision = trace.pre_action(session.session_id, "test_action", estimated_cost=0.10)
        assert decision.action_allowed

        trace.post_action(session.session_id, "test_action", actual_cost=0.10)
        assert session.total_cost == 0.10

        # Complete session
        summary = trace.complete_session(session.session_id)
        assert summary["state"] == "completed"
        assert summary["total_cost_usd"] == 0.10

    def test_budget_kill(self):
        """Test: session killed when budget exceeded."""
        trace = self._make_trace()
        session = trace.create_session()

        # Spend most of the budget
        for i in range(9):
            trace.pre_action(session.session_id, f"action_{i}", estimated_cost=0.20)
            trace.post_action(session.session_id, f"action_{i}", actual_cost=0.20)

        assert session.total_cost == pytest.approx(1.80)

        # This should trigger kill (1.80 + 0.30 > 2.00)
        decision = trace.pre_action(session.session_id, "final_action", estimated_cost=0.30)
        assert not decision.action_allowed
        assert decision.action_taken == PolicyAction.KILL
        assert not session.is_active

    def test_violation_kill(self):
        """Test: session killed after 3 PII violations."""
        trace = self._make_trace()
        session = trace.create_session()

        # First 2 violations — session stays active
        trace.record_violation(session.session_id, "pii_blocked")
        trace.record_violation(session.session_id, "pii_blocked")
        assert session.is_active
        assert session.violation_counts["pii_blocked"] == 2

        # 3rd violation — session killed
        decision = trace.record_violation(session.session_id, "pii_blocked")
        assert not decision.action_allowed
        assert not session.is_active
        assert session.state == SessionState.KILLED

    def test_action_after_kill_raises(self):
        """Test: can't take actions on a killed session."""
        trace = self._make_trace()
        session = trace.create_session()

        # Kill via violations
        for _ in range(3):
            trace.record_violation(session.session_id, "pii_blocked")

        with pytest.raises(SessionKilledError):
            trace.pre_action(session.session_id, "blocked_action", estimated_cost=0.01)

    def test_audit_log_populated(self):
        """Test: audit log captures all events."""
        trace = self._make_trace()
        session = trace.create_session()

        trace.pre_action(session.session_id, "action_1", estimated_cost=0.10)
        trace.post_action(session.session_id, "action_1", actual_cost=0.10)
        trace.record_violation(session.session_id, "pii_blocked")

        entries = trace.audit.entries_for_session(session.session_id)
        event_types = [e.event_type for e in entries]
        
        assert "session_created" in event_types
        assert "action_allowed" in event_types
        assert "violation" in event_types
