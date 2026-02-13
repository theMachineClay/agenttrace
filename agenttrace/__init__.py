"""
AgentTrace: Session-aware policy engine for AI agents.

AgentTrace doesn't replace Presidio, LLM Guard, or NeMo Guardrails.
It orchestrates them with session state — tracking cumulative cost,
counting violations, and killing sessions when thresholds are breached.

The existing guardrails ecosystem is stateless. AgentTrace adds state.
"""

from agenttrace.engine.session import SessionManager, Session, SessionState, SessionKilledError
from agenttrace.engine.cost_tracker import CostTracker
from agenttrace.engine.policy_engine import PolicyEngine
from agenttrace.engine.kill_switch import KillSwitch
from agenttrace.engine.audit_logger import AuditLogger
from agenttrace.engine.agent_trace import AgentTrace

__version__ = "0.1.0"

__all__ = [
    "AgentTrace",
    "SessionManager",
    "Session",
    "SessionState",
    "SessionKilledError",
    "CostTracker",
    "PolicyEngine",
    "KillSwitch",
    "AuditLogger",
]
