#!/usr/bin/env python3
"""
AgentTrace Demo: Budget Ceiling Kill

This demo simulates an agent session that hits the COST LIMIT:
1. Agent makes increasingly expensive LLM calls
2. Cost accumulates toward the $0.50 budget
3. Alert fires at 80% utilization
4. Session killed when the next action would exceed budget

This is the question enterprise customers ask:
"How do I enforce a $5 budget per agent session?"

LangChain can intercept. Langfuse can observe.
Neither tracks cumulative cost across a session.
AgentTrace does.

Run: python examples/demo_cost_kill.py
"""

import json
import logging
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from agenttrace import AgentTrace, SessionKilledError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def main():
    print("=" * 70)
    print("  AgentTrace Demo: Budget Ceiling Kill")
    print("  Enterprise question: 'How do I enforce a budget per session?'")
    print("=" * 70)
    print()

    # ── 1. Create AgentTrace with a tight budget ──────────────────
    trace = AgentTrace.from_dict({
        "version": "1.0",
        "agent_id": "research-agent",
        "session": {
            "max_duration": "10m",
            "max_actions": 50,
        },
        "budget": {
            "max_cost_per_session": 0.15,  # Tight $0.15 budget
            "max_cost_per_action": 0.10,   # No single action > $0.10
            "alert_at": 0.70,              # Alert at 70% ($0.105)
            "on_exceed": "kill",
        },
        "violations": {
            "thresholds": {},
            "on_threshold": "kill",
        },
        "kill_switch": {
            "enabled": True,
            "notify": [
                {"webhook": "https://hooks.slack.com/services/DEMO/WEBHOOK/URL"},
            ],
            "grace_period": "5s",
        },
        "audit": {"enabled": True},
    })

    session = trace.create_session(agent_id="research-agent")

    print(f"📋 Session created: {session.session_id[:12]}...")
    print(f"💰 Budget: ${trace.policy.budget.max_cost_per_session:.2f}")
    print(f"⚠️  Alert at: ${trace.policy.budget.max_cost_per_session * trace.policy.budget.alert_at:.2f} "
          f"({trace.policy.budget.alert_at:.0%})")
    print(f"🚫 Per-action limit: ${trace.policy.budget.max_cost_per_action:.2f}")
    print()

    # ── 2. Simulate a research agent making LLM calls ─────────────
    # Each call gets bigger as the agent processes more context

    actions = [
        # (action_name, model, input_tokens, output_tokens, description)
        ("init_query",          "gpt-4o", 500,   200,   "Initial user query"),
        ("search_results",      "gpt-4o", 2000,  800,   "Process search results"),
        ("analyze_paper_1",     "gpt-4o", 8000,  2000,  "Analyze first paper"),
        ("analyze_paper_2",     "gpt-4o", 10000, 3000,  "Analyze second paper"),
        ("cross_reference",     "gpt-4o", 15000, 4000,  "Cross-reference findings"),
        ("synthesize_report",   "gpt-4o", 20000, 5000,  "Synthesize final report"),  # Budget hit here
        ("format_output",       "gpt-4o", 5000,  2000,  "Format for user"),          # Never reached
    ]

    for i, (action_name, model, in_tokens, out_tokens, description) in enumerate(actions):
        print(f"─── Action {i+1}: {action_name} ───")
        print(f"  📝 {description}")

        # Estimate cost before execution
        estimate = trace.cost_tracker.estimate_cost(
            model=model, input_tokens=in_tokens, output_tokens=out_tokens
        )
        print(f"  📊 Estimated: ${estimate.total_cost:.4f} "
              f"({in_tokens:,} in / {out_tokens:,} out)")

        # Pre-action budget check
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
            utilization = session.total_cost / trace.policy.budget.max_cost_per_session
            print(f"  ⚠️  BUDGET ALERT: {decision.reason}")

        # Execute
        print(f"  ✅ Executing...")
        time.sleep(0.1)

        # Record actual cost
        try:
            trace.post_action(
                session_id=session.session_id,
                action_name=action_name,
                model=model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        except SessionKilledError as e:
            print(f"  🛑 SESSION KILLED: {e}")
            break

        remaining = trace.policy.budget.max_cost_per_session - session.total_cost
        utilization = session.total_cost / trace.policy.budget.max_cost_per_session
        bar_len = 30
        filled = int(bar_len * utilization)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  💰 [{bar}] ${session.total_cost:.4f} / "
              f"${trace.policy.budget.max_cost_per_session:.2f} "
              f"({utilization:.0%}) — ${remaining:.4f} remaining")
        print()

    # ── 3. Session summary ────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)

    audit = session.to_audit_dict()
    print(f"  Session ID:    {audit['session_id'][:12]}...")
    print(f"  Agent:         {audit['agent_id']}")
    print(f"  State:         {audit['state']}")
    print(f"  Duration:      {audit['duration_seconds']:.1f}s")
    print(f"  Total Cost:    ${audit['total_cost_usd']:.4f} / "
          f"${trace.policy.budget.max_cost_per_session:.2f}")
    print(f"  Actions:       {audit['action_count']}")
    if audit['kill_reason']:
        print(f"  Kill Reason:   {audit['kill_reason']}")
    print()

    # ── 4. Audit trail ────────────────────────────────────────────
    print("─── Audit Log ───")
    for entry in trace.audit.entries:
        d = entry.to_dict()
        event = d.pop("event_type")
        d.pop("session_id", None)
        d.pop("agent_id", None)
        d.pop("timestamp", None)
        extras = {k: v for k, v in d.items() if v is not None}
        print(f"  [{event:20s}] {json.dumps(extras, default=str)}")
    print()

    # ── 5. The point ──────────────────────────────────────────────
    print("─── What just happened ───")
    print("  The research agent made increasingly expensive LLM calls.")
    print("  AgentTrace tracked cumulative cost across the session.")
    print("  At 80% budget utilization, it fired an alert.")
    print(f"  When the next call (${estimate.total_cost:.4f}) would have")
    print(f"  pushed the session past ${trace.policy.budget.max_cost_per_session:.2f}, "
          "it killed the session.")
    print()
    print("  The enterprise question: 'How do I enforce a budget?'")
    print("  • LangChain → can intercept, can't track cumulative cost")
    print("  • Langfuse  → can observe cost, can't enforce limits")
    print("  • AgentTrace → tracks, enforces, and kills. Pre-execution.")
    print()


if __name__ == "__main__":
    main()
