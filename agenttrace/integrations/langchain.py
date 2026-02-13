"""
LangChain Integration: Session-aware wrapper for LangChain middleware.

LangChain 1.0 has PIIMiddleware, HumanInTheLoopMiddleware, wrap_tool_call, etc.
These are STATELESS — they evaluate each request independently.

This module wraps them with AgentTrace session state so that:
- PIIMiddleware blocking a PII instance → AgentTrace counts it
- After N blocks → AgentTrace kills the session
- Every LLM call → AgentTrace tracks cost against budget

We don't replace LangChain middleware. We orchestrate it with state.

Usage:
    from agenttrace.integrations.langchain import AgentTraceMiddleware

    trace = AgentTrace.from_yaml("policy.yaml")
    session = trace.create_session()

    # Wrap a LangChain agent with session-aware enforcement
    middleware = AgentTraceMiddleware(trace, session.session_id)

    # Use as a LangChain callback handler
    agent.invoke({"input": "..."}, config={"callbacks": [middleware]})
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from agenttrace.engine.agent_trace import AgentTrace
from agenttrace.engine.policy_engine import PolicyAction
from agenttrace.engine.session import SessionKilledError

logger = logging.getLogger("agenttrace.langchain")

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Stub for when LangChain isn't installed
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    class LLMResult:  # type: ignore[no-redef]
        pass


class AgentTraceCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that integrates with AgentTrace.

    Hooks into LangChain's callback system to:
    - Check budget before LLM calls
    - Track cost after LLM calls
    - Count violations when PII or other issues are detected
    - Kill sessions when thresholds are breached

    This is the bridge between LangChain's stateless middleware
    and AgentTrace's session-aware policy engine.
    """

    def __init__(
        self,
        agent_trace: AgentTrace,
        session_id: str,
        model: str = "gpt-4o",
    ):
        if not HAS_LANGCHAIN:
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install agenttrace[langchain]"
            )
        super().__init__()
        self.trace = agent_trace
        self.session_id = session_id
        self.model = model
        self._current_action_start: float | None = None

    # ── LLM callbacks ─────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Check budget before LLM call."""
        self._current_action_start = time.time()
        input_text = "\n".join(prompts)

        try:
            decision = self.trace.pre_action(
                session_id=self.session_id,
                action_name="llm_call",
                model=self.model,
                input_text=input_text,
            )
            if not decision.action_allowed:
                raise SessionKilledError(
                    f"Action blocked by AgentTrace: {decision.reason}"
                )
        except SessionKilledError:
            raise
        except Exception as e:
            logger.error(f"AgentTrace pre-action check failed: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Track cost after LLM call completes."""
        try:
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})

            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

            self.trace.post_action(
                session_id=self.session_id,
                action_name="llm_call",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except SessionKilledError:
            raise
        except Exception as e:
            logger.error(f"AgentTrace post-action tracking failed: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record LLM errors as potential violations."""
        logger.warning(f"LLM error in session {self.session_id}: {error}")

    # ── Tool callbacks ────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Check policy before tool execution."""
        tool_name = serialized.get("name", "unknown_tool")
        try:
            decision = self.trace.pre_action(
                session_id=self.session_id,
                action_name=f"tool:{tool_name}",
                estimated_cost=0.001,  # Tool calls have minimal direct cost
            )
            if not decision.action_allowed:
                raise SessionKilledError(
                    f"Tool call blocked by AgentTrace: {decision.reason}"
                )
        except SessionKilledError:
            raise
        except Exception as e:
            logger.error(f"AgentTrace tool pre-check failed: {e}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record tool completion."""
        pass  # Cost already tracked in pre_action estimate

    # ── Violation integration ─────────────────────────────────────

    def report_pii_violation(self, details: dict[str, Any] | None = None) -> None:
        """
        Call this when LangChain PIIMiddleware (or any scanner) blocks PII.

        This is the key integration point:
        - PIIMiddleware blocks one instance (stateless)
        - AgentTrace counts it (stateful)
        - After N blocks, AgentTrace kills the session

        Usage in a custom middleware wrapper:
            if pii_detected:
                handler.report_pii_violation({"field": "email", "action": "block"})
        """
        self.trace.record_violation(
            session_id=self.session_id,
            violation_type="pii_blocked",
            details=details,
        )

    def report_violation(
        self, violation_type: str, details: dict[str, Any] | None = None
    ) -> None:
        """Report any violation type to AgentTrace."""
        self.trace.record_violation(
            session_id=self.session_id,
            violation_type=violation_type,
            details=details,
        )
