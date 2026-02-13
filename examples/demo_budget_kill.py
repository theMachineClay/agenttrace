#!/usr/bin/env python3
"""
AgentTrace Demo: Session-Aware Policy Enforcement

This demo simulates an agent session that:
1. Makes several LLM calls (cost accumulates)
2. Triggers PII violations (violation count accumulates)
3. Eventually hits the budget limit → session killed → webhook notification

This is the gap in the ecosystem:
- LangChain PIIMiddleware blocks PII but doesn't count violations across a session
- No existing tool says "you've spent $1.80 of $2.00, this $0.30 call would exceed budget"
- No existing tool kills a session after 3 cumulative PII blocks

AgentTrace does all of this.

Run: python examples/demo_budget_kill.py
"""

import json
import logging
import sys
import time

# Add parent to path for development
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from agenttrace import AgentTrace, SessionKilledError

# ── Setup logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def main():
    print("=" * 70)
    print("  AgentTrace Demo: Session-Aware Policy Enforcement")
    print("  The circuit breaker for AI agent sessions")
    print("=" * 70)
    print()

    # ── 1. Create AgentTrace with inline policy ───────────────────
    # (In production, load from YAML file)
    
    policy = {
        "version": "1.0",
        "agent_id": "customer-support-bot",
        "session": {
            "max_duration": "30m",
            "max_actions": 50,
        },
        "budget": {
            "max_cost_per_session": 2.00,  # $2 budget for demo
            "max_cost_per_action": 0.50,
            "alert_at": 0.80,
            "on_exceed": "kill",
        },
        "violations": {
            "thresholds": {
                "pii_blocked": 3,       # Kill after 3 PII blocks
                "scope_violation": 1,   # Kill after 1 scope violation
            },
            "on_threshold": "kill",
        },
        "kill_switch": {
            "enabled": True,
            "notify": [
                # In production, this would be a real Slack webhook URL.
                # The demo will attempt the POST and log the failure gracefully.
                {"webhook": "https://hooks.slack.com/services/DEMO/WEBHOOK/URL"},
            ],
            "grace_period": "5s",
        },
        "audit": {
            "enabled": True,
            "export": [
                {"file": "/tmp/agenttrace/demo_audit.jsonl"},
            ],
        },
    }

    trace = AgentTrace.from_dict(policy)
    session = trace.create_session(agent_id="customer-support-bot")

    print(f"📋 Session created: {session.session_id[:12]}...")
    print(f"💰 Budget: ${trace.policy.budget.max_cost_per_session:.2f}")
    print(f"🚨 PII violation threshold: {trace.policy.violation_thresholds[0].max_count}")
    print()

    # ── 2. Simulate agent actions ─────────────────────────────────

    actions = [
        # (action_name, model, input_tokens, output_tokens, triggers_pii)
        ("greeting",        "gpt-4o", 500,   200,  False),
        ("lookup_account",  "gpt-4o", 1000,  500,  False),
        ("draft_response",  "gpt-4o", 2000,  1000, True),   # PII leak #1
        ("refine_response", "gpt-4o", 1500,  800,  False),
        ("send_email",      "gpt-4o", 800,   300,  True),   # PII leak #2
        ("log_interaction", "gpt-4o", 600,   200,  False),
        ("followup_draft",  "gpt-4o", 3000,  1500, True),   # PII leak #3 → KILL
        ("unreachable",     "gpt-4o", 100,   50,   False),  # Should never execute
    ]

    for i, (action_name, model, in_tokens, out_tokens, triggers_pii) in enumerate(actions):
        print(f"─── Action {i+1}: {action_name} ───")

        # Estimate cost
        estimate = trace.cost_tracker.estimate_cost(
            model=model, input_tokens=in_tokens, output_tokens=out_tokens
        )
        print(f"  📊 Estimated cost: ${estimate.total_cost:.4f} "
              f"({in_tokens} in / {out_tokens} out tokens)")

        # Pre-action check (BEFORE execution)
        try:
            decision = trace.pre_action(
                session_id=session.session_id,
                action_name=action_name,
                estimated_cost=estimate.total_cost,
            )
        except SessionKilledError as e:
            print(f"  🛑 SESSION ALREADY DEAD: {e}")
            break

        if not decision.action_allowed:
            print(f"  ❌ BLOCKED: {decision.reason}")
            print(f"  🛑 Action: {decision.action_taken.value}")
            break

        if decision.action_taken.value == "alert":
            print(f"  ⚠️  ALERT: {decision.reason}")

        # Simulate execution
        print(f"  ✅ Executing {action_name}...")
        time.sleep(0.1)  # Simulate work

        # Post-action (record actual cost)
        try:
            trace.post_action(
                session_id=session.session_id,
                action_name=action_name,
                model=model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        except SessionKilledError as e:
            print(f"  🛑 SESSION KILLED during post-action: {e}")
            break

        print(f"  💰 Session total: ${session.total_cost:.4f} / "
              f"${trace.policy.budget.max_cost_per_session:.2f}")

        # Simulate PII violation (as if LangChain PIIMiddleware caught it)
        if triggers_pii:
            print(f"  🔍 PII detected by scanner! Recording violation...")
            try:
                v_decision = trace.record_violation(
                    session_id=session.session_id,
                    violation_type="pii_blocked",
                    details={"field": "customer_email", "action": "redacted"},
                )
                pii_count = session.violation_counts.get("pii_blocked", 0)
                threshold = trace.policy.violation_thresholds[0].max_count
                print(f"  🚨 PII violations: {pii_count} / {threshold}")

                if not v_decision.action_allowed:
                    print(f"  🛑 THRESHOLD BREACHED: {v_decision.reason}")
                    break
            except SessionKilledError as e:
                print(f"  🛑 SESSION KILLED: {e}")
                break

        print()

    # ── 3. Print session summary ──────────────────────────────────

    print()
    print("=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)

    audit = session.to_audit_dict()
    print(f"  Session ID:    {audit['session_id'][:12]}...")
    print(f"  Agent:         {audit['agent_id']}")
    print(f"  State:         {audit['state']}")
    print(f"  Duration:      {audit['duration_seconds']:.1f}s")
    print(f"  Total Cost:    ${audit['total_cost_usd']:.4f}")
    print(f"  Actions:       {audit['action_count']}")
    print(f"  Violations:    {audit['violation_counts']}")
    if audit['kill_reason']:
        print(f"  Kill Reason:   {audit['kill_reason']}")
    print()

    # ── 4. Show audit log ─────────────────────────────────────────

    print("─── Audit Log ───")
    for entry in trace.audit.entries:
        d = entry.to_dict()
        event = d.pop("event_type")
        sid = d.pop("session_id")[:8]
        agent = d.pop("agent_id")
        ts = d.pop("timestamp")
        # Filter to interesting fields
        extras = {k: v for k, v in d.items() if v is not None}
        print(f"  [{event:20s}] {json.dumps(extras, default=str)}")

    print()

    # ── 5. Show kill switch history ───────────────────────────────

    if trace.kill_switch.kill_history:
        print("─── Kill Switch Events ───")
        for event in trace.kill_switch.kill_history:
            print(f"  🛑 Killed session {event.session_id[:12]}...")
            print(f"     Reason: {event.reason}")
            print(f"     Cost: ${event.session_cost:.4f}")
            print(f"     Notifications: {len(event.notifications_sent)} sent")
            for n in event.notifications_sent:
                status = n.get("status", "unknown")
                print(f"       → {n.get('type', 'webhook')}: {status}")
        print()

    print("─── What just happened ───")
    print("  The agent session was tracked with cumulative state.")
    print("  Each LLM call's cost was estimated via tiktoken and")
    print("  checked against the session budget BEFORE execution.")
    print("  PII violations were counted across the session.")
    print("  When the 3rd PII violation hit, the session was killed")
    print("  and a Slack webhook notification was fired.")
    print()
    print("  No existing tool does this:")
    print("  • LangChain PIIMiddleware → stateless (blocks one at a time)")
    print("  • NeMo Guardrails → no cost tracking or violation counting")
    print("  • Langfuse → observes but doesn't enforce")
    print("  • AgentTrace → session-aware enforcement with kill-switch")
    print()


if __name__ == "__main__":
    main()
