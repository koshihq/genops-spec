"""Policy enforcement for AI governance."""

from __future__ import annotations

import functools
import logging
from enum import Enum
from typing import Any, Callable, TypeVar

from opentelemetry import trace

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class PolicyResult(Enum):
    """Policy enforcement results."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    WARNING = "warning"
    RATE_LIMITED = "rate_limited"


class PolicyViolationError(Exception):
    """Raised when a policy violation blocks an operation."""

    def __init__(
        self, policy_name: str, reason: str, metadata: dict[str, Any] | None = None
    ):
        self.policy_name = policy_name
        self.reason = reason
        self.metadata = metadata or {}
        super().__init__(f"Policy '{policy_name}' violation: {reason}")


class PolicyEvaluationResult:
    """Result of policy evaluation with details."""

    def __init__(
        self,
        policy_name: str,
        result: PolicyResult,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.policy_name = policy_name
        self.result = result
        self.reason = reason
        self.metadata = metadata or {}


class PolicyConfig:
    """Configuration for a governance policy."""

    def __init__(
        self,
        name: str,
        description: str = "",
        enabled: bool = True,
        enforcement_level: PolicyResult = PolicyResult.BLOCKED,
        conditions: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.enabled = enabled
        self.enforcement_level = enforcement_level
        self.conditions = conditions or {}


class PolicyEngine:
    """Core policy enforcement engine."""

    def __init__(self):
        self.policies: dict[str, PolicyConfig] = {}
        self.telemetry = GenOpsTelemetry()

    def register_policy(self, policy: PolicyConfig) -> None:
        """Register a new policy."""
        self.policies[policy.name] = policy
        logger.info(f"Registered policy: {policy.name}")

    def evaluate_policy(
        self, policy_name: str, operation_context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """
        Evaluate a policy against an operation context.

        Returns:
            PolicyEvaluationResult: Policy evaluation result with details
        """
        if policy_name not in self.policies:
            logger.warning(f"Unknown policy: {policy_name}")
            return PolicyEvaluationResult(policy_name, PolicyResult.ALLOWED, "Policy not found")

        policy = self.policies[policy_name]

        if not policy.enabled:
            return PolicyEvaluationResult(policy_name, PolicyResult.ALLOWED, "Policy disabled")

        # Example policy evaluations - extend as needed
        if policy.name == "cost_limit":
            return self._evaluate_cost_limit(policy, operation_context)
        elif policy.name == "rate_limit":
            return self._evaluate_rate_limit(policy, operation_context)
        elif policy.name == "content_filter":
            return self._evaluate_content_filter(policy, operation_context)
        elif policy.name == "team_access":
            return self._evaluate_team_access(policy, operation_context)

        return PolicyEvaluationResult(policy_name, PolicyResult.ALLOWED, None)

    def _evaluate_cost_limit(
        self, policy: PolicyConfig, context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """Evaluate cost limit policy."""
        max_cost = policy.conditions.get("max_cost", float("inf"))
        # Check both 'cost' and 'estimated_cost' for backwards compatibility
        estimated_cost = context.get("cost", context.get("estimated_cost", 0))

        if estimated_cost > max_cost:
            return PolicyEvaluationResult(
                policy.name,
                policy.enforcement_level,
                f"Cost limit exceeded: ${estimated_cost:.4f} exceeds limit ${max_cost:.4f}",
                metadata={"limit": max_cost, "actual": estimated_cost}
            )

        return PolicyEvaluationResult(policy.name, PolicyResult.ALLOWED, None)

    def _evaluate_rate_limit(
        self, policy: PolicyConfig, context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """Evaluate rate limit policy."""
        # Simplified rate limiting - in production, use Redis or similar
        max_requests = policy.conditions.get("max_requests_per_minute", policy.conditions.get("max_requests", 100))
        # Check multiple keys for backwards compatibility
        current_requests = context.get("request_count", context.get("requests_count", context.get("current_requests", 0)))

        if current_requests >= max_requests:
            return PolicyEvaluationResult(
                policy.name,
                policy.enforcement_level,  # Use configured enforcement level
                f"Rate limit exceeded: {current_requests}/{max_requests} requests per minute",
            )

        return PolicyEvaluationResult(policy.name, PolicyResult.ALLOWED, None)

    def _evaluate_content_filter(
        self, policy: PolicyConfig, context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """Evaluate content filtering policy."""
        blocked_patterns = policy.conditions.get("blocked_patterns", [])
        content = context.get("content", "")

        for pattern in blocked_patterns:
            if pattern.lower() in content.lower():
                return PolicyEvaluationResult(
                    policy.name,
                    policy.enforcement_level,
                    f"Content contains blocked pattern: {pattern}",
                )

        return PolicyEvaluationResult(policy.name, PolicyResult.ALLOWED, None)

    def _evaluate_team_access(
        self, policy: PolicyConfig, context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """Evaluate team access policy."""
        allowed_teams = policy.conditions.get("allowed_teams", [])
        team = context.get("team")

        if allowed_teams and team not in allowed_teams:
            return PolicyEvaluationResult(
                policy.name,
                policy.enforcement_level,
                f"Team '{team}' not in allowed teams: {allowed_teams}",
            )

        return PolicyEvaluationResult(policy.name, PolicyResult.ALLOWED, None)


# Global policy engine instance
_policy_engine = PolicyEngine()
_global_policy_engine = _policy_engine  # Alias for testing compatibility


def register_policy(
    name: str,
    description: str = "",
    enabled: bool = True,
    enforcement_level: PolicyResult = PolicyResult.BLOCKED,
    **conditions: Any,
) -> None:
    """Register a new governance policy."""
    policy = PolicyConfig(
        name=name,
        description=description,
        enabled=enabled,
        enforcement_level=enforcement_level,
        conditions=conditions,
    )
    _policy_engine.register_policy(policy)


def enforce_policy(
    policies: str | list[str], operation_context: dict[str, Any] | None = None
) -> Callable[[F], F]:
    """
    Decorator to enforce governance policies on AI operations.

    Args:
        policies: Policy name(s) to enforce
        operation_context: Additional context for policy evaluation

    Example:
        @enforce_policy(["cost_limit", "content_filter"])
        def generate_content(prompt: str) -> str:
            return llm.complete(prompt)
    """
    if isinstance(policies, str):
        policies = [policies]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build operation context
            context = operation_context or {}
            context.update(
                {
                    "function_name": func.__name__,
                    "module": func.__module__,
                    "args": args,
                    "kwargs": kwargs,
                }
            )

            # Get current span for telemetry
            current_span = trace.get_current_span()

            # Evaluate each policy
            for policy_name in policies:
                policy_result = _policy_engine.evaluate_policy(policy_name, context)

                # Record policy telemetry
                if current_span and current_span.is_recording():
                    _policy_engine.telemetry.record_policy(
                        span=current_span,
                        policy_name=policy_name,
                        policy_result=policy_result.result.value,
                        policy_reason=policy_result.reason,
                    )

                # Handle policy violations
                if policy_result.result == PolicyResult.BLOCKED:
                    raise PolicyViolationError(
                        policy_name, policy_result.reason or "Policy violation"
                    )
                elif policy_result.result == PolicyResult.WARNING:
                    logger.warning(f"Policy warning for '{policy_name}': {policy_result.reason}")
                elif policy_result.result == PolicyResult.RATE_LIMITED:
                    raise PolicyViolationError(
                        policy_name, policy_result.reason or "Rate limit exceeded"
                    )

            # All policies passed, execute the function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_policy(
    policy_name: str, operation_context: dict[str, Any]
) -> PolicyEvaluationResult:
    """
    Manually check a policy without enforcement.

    Returns:
        PolicyEvaluationResult: Policy evaluation result with details
    """
    return _policy_engine.evaluate_policy(policy_name, operation_context)
