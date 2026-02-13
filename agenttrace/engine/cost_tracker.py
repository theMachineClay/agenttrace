"""
Cost Tracker: Real-time budget enforcement for agent sessions.

Enterprise question: "How do I enforce a $5 budget per agent session?"
LangChain can intercept. Langfuse can observe. Neither tracks cumulative cost.
CostTracker does.

Uses tiktoken for token counting, maps to provider pricing, and integrates
with the Session to enforce budget limits before actions execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tiktoken


# Pricing per 1M tokens (USD) — update as providers change pricing
# These are approximate; production would pull from a config or API
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # Defaults
    "default": {"input": 2.00, "output": 8.00},
}

# tiktoken encoding mapping
MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
    # Anthropic models — tiktoken doesn't have their tokenizer,
    # so we approximate with cl100k_base. Production would use
    # Anthropic's own token counting API.
    "claude-sonnet-4-5-20250514": "cl100k_base",
    "claude-haiku-3-5-20241022": "cl100k_base",
    "claude-opus-4-20250514": "cl100k_base",
    "default": "cl100k_base",
}


@dataclass
class CostEstimate:
    """Result of a cost estimation."""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost_usd": round(self.input_cost, 6),
            "output_cost_usd": round(self.output_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
            "model": self.model,
        }


@dataclass
class BudgetDecision:
    """Result of a budget check."""
    allowed: bool
    session_cost_before: float
    estimated_action_cost: float
    session_cost_after: float
    budget_limit: float
    budget_remaining: float
    alert: bool  # True if past alert threshold
    reason: str | None = None  # Set if blocked


class CostTracker:
    """
    Tracks and enforces cost budgets for agent sessions.

    Core operations:
    1. estimate_cost(model, input_text, output_text) → CostEstimate
    2. check_budget(session, estimated_cost, policy) → BudgetDecision
    3. count_tokens(text, model) → int
    """

    def __init__(self):
        self._encoders: dict[str, tiktoken.Encoding] = {}

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        encoding_name = MODEL_ENCODINGS.get(model, MODEL_ENCODINGS["default"])
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        return self._encoders[encoding_name]

    def _get_pricing(self, model: str) -> dict[str, float]:
        return MODEL_PRICING.get(model, MODEL_PRICING["default"])

    def count_tokens(self, text: str, model: str = "default") -> int:
        """Count tokens in text using tiktoken."""
        encoder = self._get_encoder(model)
        return len(encoder.encode(text))

    def estimate_cost(
        self,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
    ) -> CostEstimate:
        """
        Estimate the cost of an LLM call.
        Accepts either token counts directly or text to count.
        """
        if input_tokens is None:
            input_tokens = self.count_tokens(input_text or "", model)
        if output_tokens is None:
            output_tokens = self.count_tokens(output_text or "", model)

        pricing = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            model=model,
        )

    def check_budget(
        self,
        session_total_cost: float,
        estimated_action_cost: float,
        max_cost_per_session: float,
        max_cost_per_action: float | None = None,
        alert_threshold: float = 0.8,
    ) -> BudgetDecision:
        """
        Check if an action is within budget.

        This is the core enforcement that no existing tool provides:
        "The session has spent $4.50. This action costs ~$0.60. Budget is $5.00. BLOCKED."
        """
        cost_after = session_total_cost + estimated_action_cost
        remaining = max_cost_per_session - session_total_cost
        alert = session_total_cost >= (max_cost_per_session * alert_threshold)

        # Check per-action limit
        if max_cost_per_action and estimated_action_cost > max_cost_per_action:
            return BudgetDecision(
                allowed=False,
                session_cost_before=session_total_cost,
                estimated_action_cost=estimated_action_cost,
                session_cost_after=session_total_cost,  # not applied
                budget_limit=max_cost_per_session,
                budget_remaining=remaining,
                alert=alert,
                reason=(
                    f"Action cost ${estimated_action_cost:.4f} exceeds "
                    f"per-action limit ${max_cost_per_action:.4f}"
                ),
            )

        # Check session budget
        if cost_after > max_cost_per_session:
            return BudgetDecision(
                allowed=False,
                session_cost_before=session_total_cost,
                estimated_action_cost=estimated_action_cost,
                session_cost_after=session_total_cost,  # not applied
                budget_limit=max_cost_per_session,
                budget_remaining=remaining,
                alert=True,
                reason=(
                    f"Session cost would reach ${cost_after:.4f}, "
                    f"exceeding budget ${max_cost_per_session:.2f} "
                    f"(remaining: ${remaining:.4f})"
                ),
            )

        return BudgetDecision(
            allowed=True,
            session_cost_before=session_total_cost,
            estimated_action_cost=estimated_action_cost,
            session_cost_after=cost_after,
            budget_limit=max_cost_per_session,
            budget_remaining=remaining - estimated_action_cost,
            alert=alert,
        )
