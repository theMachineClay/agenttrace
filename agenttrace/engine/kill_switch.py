"""
Kill Switch: Hard stop for runaway agents + webhook notifications.

When thresholds are breached, the kill switch:
1. Terminates the session (no more actions allowed)
2. Fires webhook notifications (Slack, PagerDuty, custom)
3. Logs the termination event for audit

This is the circuit breaker pattern applied to AI agents.
"""

from __future__ import annotations

import asyncio

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from agenttrace.engine.session import Session

logger = logging.getLogger("agenttrace.kill_switch")


@dataclass
class KillEvent:
    """Record of a session termination."""
    session_id: str
    agent_id: str
    reason: str
    timestamp: float
    session_cost: float
    action_count: int
    violation_counts: dict[str, int]
    notifications_sent: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "session_killed",
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "session_cost_usd": round(self.session_cost, 6),
            "action_count": self.action_count,
            "violation_counts": self.violation_counts,
            "notifications_sent": self.notifications_sent,
        }


class KillSwitch:
    """
    Terminates agent sessions and sends notifications.

    Usage:
        kill_switch = KillSwitch(
            webhooks=["https://hooks.slack.com/..."],
            on_kill=[my_callback],
        )
        event = await kill_switch.execute(session, "Budget exceeded: $5.02 > $5.00")
    """

    def __init__(
        self,
        webhooks: list[str] | None = None,
        pagerduty_services: list[str] | None = None,
        grace_period_seconds: float = 5.0,
        on_kill: list[Callable[[KillEvent], None]] | None = None,
    ):
        self.webhooks = webhooks or []
        self.pagerduty_services = pagerduty_services or []
        self.grace_period_seconds = grace_period_seconds
        self._on_kill_callbacks = on_kill or []
        self._kill_history: list[KillEvent] = []

    async def execute(self, session: Session, reason: str) -> KillEvent:
        """
        Execute the kill switch on a session.

        1. Mark session as killed
        2. Fire all webhook notifications
        3. Record the kill event
        4. Call registered callbacks
        """
        # 1. Kill the session
        session.kill(reason)

        # 2. Build the kill event
        event = KillEvent(
            session_id=session.session_id,
            agent_id=session.agent_id,
            reason=reason,
            timestamp=time.time(),
            session_cost=session.total_cost,
            action_count=session.action_count,
            violation_counts=session.violation_counts,
            notifications_sent=[],
        )

        # 3. Fire notifications (async, best-effort)
        notification_tasks = []
        for webhook_url in self.webhooks:
            notification_tasks.append(
                self._send_webhook(webhook_url, event)
            )

        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Webhook notification failed: {self.webhooks[i]} — {result}"
                    )
                    event.notifications_sent.append({
                        "type": "webhook",
                        "url": self.webhooks[i],
                        "status": "failed",
                        "error": str(result),
                    })
                else:
                    event.notifications_sent.append(result)

        # 4. Record and callback
        self._kill_history.append(event)
        for callback in self._on_kill_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Kill callback failed: {e}")

        logger.warning(
            f"🛑 SESSION KILLED | {session.session_id} | "
            f"Agent: {session.agent_id} | Reason: {reason} | "
            f"Cost: ${session.total_cost:.4f} | Actions: {session.action_count}"
        )

        return event

    def execute_sync(self, session: Session, reason: str) -> KillEvent:
        """Synchronous version — creates event loop if needed."""
        try:
            asyncio.get_running_loop()
            # We're already in an async context, schedule it
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run, self.execute(session, reason)
                ).result()
        except RuntimeError:
            return asyncio.run(self.execute(session, reason))

    async def _send_webhook(self, url: str, event: KillEvent) -> dict[str, Any]:
        """Send a webhook notification (Slack-compatible format)."""
        payload = self._format_slack_payload(event)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            return {
                "type": "webhook",
                "url": url,
                "status": "sent" if response.is_success else "failed",
                "status_code": response.status_code,
            }

    @staticmethod
    def _format_slack_payload(event: KillEvent) -> dict[str, Any]:
        """Format kill event as a Slack message."""
        violations_str = ", ".join(
            f"{k}: {v}" for k, v in event.violation_counts.items()
        ) or "none"

        return {
            "text": f"🛑 Agent Session Killed: {event.agent_id}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🛑 AgentTrace: Session Killed",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Agent:*\n{event.agent_id}"},
                        {"type": "mrkdwn", "text": f"*Session:*\n`{event.session_id[:12]}...`"},
                        {"type": "mrkdwn", "text": f"*Cost:*\n${event.session_cost:.4f}"},
                        {"type": "mrkdwn", "text": f"*Actions:*\n{event.action_count}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Reason:*\n{event.reason}",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Violations:*\n{violations_str}",
                    },
                },
            ],
        }

    @property
    def kill_history(self) -> list[KillEvent]:
        return list(self._kill_history)
