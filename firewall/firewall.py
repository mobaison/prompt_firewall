"""
firewall/firewall.py — Main Firewall Coordinator
=================================================
Runs all 5 layers in sequence. Stops at first failure.
Returns a structured FirewallResult to app.py.

Layer execution order:
  1. InputSanitizer       — input quality
  2. RateLimiter          — abuse prevention   (before topic, saves compute)
  3. InjectionDetector    — prompt attacks
  4. TopicGuard           — relevance check
  5. ContentFilter        — sensitive content

Also logs every blocked attempt with reason and layer for audit.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from .sanitizer          import InputSanitizer
from .rate_limiter       import RateLimiter
from .injection_detector import InjectionDetector
from .topic_guard        import TopicGuard
from .content_filter     import ContentFilter

# ── Audit logger ──────────────────────────────────────────────────
logging.basicConfig(
    filename="firewall_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
audit = logging.getLogger("firewall.audit")


@dataclass
class FirewallResult:
    allowed     : bool
    message     : str           = ""   # response to send user if blocked
    layer       : str           = ""   # which layer blocked it
    action      : str           = "allow"  # allow | block | redirect | warn
    warning     : str           = ""   # soft warning (rate limit approaching)
    latency_ms  : float         = 0.0  # how long firewall took


class PromptFirewall:

    def __init__(self):
        self.sanitizer  = InputSanitizer()
        self.rate       = RateLimiter()
        self.injection  = InjectionDetector()
        self.topic      = TopicGuard()
        self.content    = ContentFilter()
        print("✅ Prompt Firewall initialized (5 layers active)")

    def check(self, text: str, session_id: str = "default") -> FirewallResult:
        """
        Run all firewall layers against the input.
        Returns FirewallResult — check .allowed before proceeding.
        """
        start = time.time()

        # ════════════════════════════════════════════════
        # LAYER 1 — Input Sanitization
        # ════════════════════════════════════════════════
        ok, reason = self.sanitizer.check(text)
        if not ok:
            return self._blocked("sanitizer", reason, text, session_id, start)

        # ════════════════════════════════════════════════
        # LAYER 2 — Rate Limiting
        # ════════════════════════════════════════════════
        ok, reason = self.rate.check(session_id)
        if not ok:
            return self._blocked("rate_limiter", reason, text, session_id, start)

        # Soft warning from rate limiter (allowed but approaching limit)
        rate_warning = reason if ok and reason else ""

        # ════════════════════════════════════════════════
        # LAYER 3 — Injection Detection
        # ════════════════════════════════════════════════
        ok, reason = self.injection.check(text)
        if not ok:
            return self._blocked("injection_detector", reason, text, session_id, start)

        # ════════════════════════════════════════════════
        # LAYER 4 — Topic Guard
        # ════════════════════════════════════════════════
        ok, reason = self.topic.check(text)
        if not ok:
            return self._blocked("topic_guard", reason, text, session_id, start)

        # ════════════════════════════════════════════════
        # LAYER 5 — Content Filter
        # ════════════════════════════════════════════════
        ok, reason, action = self.content.check(text)
        if not ok:
            self._log_block("content_filter", action, text, session_id)
            latency = (time.time() - start) * 1000
            return FirewallResult(
                allowed    = False,
                message    = reason,
                layer      = "content_filter",
                action     = action,       # could be "redirect" for crisis
                latency_ms = latency,
            )

        # ════════════════════════════════════════════════
        # ALL LAYERS PASSED
        # ════════════════════════════════════════════════
        latency = (time.time() - start) * 1000
        return FirewallResult(
            allowed    = True,
            action     = "allow",
            warning    = rate_warning,
            latency_ms = latency,
        )

    # ── Helpers ───────────────────────────────────────────────────
    def _blocked(self, layer: str, reason: str,
                 text: str, session_id: str, start: float) -> FirewallResult:
        self._log_block(layer, "block", text, session_id)
        latency = (time.time() - start) * 1000
        return FirewallResult(
            allowed    = False,
            message    = reason,
            layer      = layer,
            action     = "block",
            latency_ms = latency,
        )

    def _log_block(self, layer: str, action: str, text: str, session_id: str):
        """Write blocked attempt to audit log."""
        safe_text = text[:120].replace("\n", " ")
        audit.info(
            f"BLOCKED | layer={layer} | action={action} | "
            f"session={session_id[:16]} | input=\"{safe_text}\""
        )

    def stats(self, session_id: str) -> dict:
        """Return rate limit stats for a session."""
        return self.rate.get_stats(session_id)