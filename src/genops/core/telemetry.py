"""Core telemetry engine for GenOps AI governance."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class GenOpsTelemetry:
    """Central telemetry engine for GenOps governance signals."""

    def __init__(self, tracer_name: str = "genops-ai"):
        self.tracer = trace.get_tracer(tracer_name)

    def create_span(
        self, name: str, attributes: Optional[Dict[str, Any]] = None, **kwargs
    ) -> trace.Span:
        """Create a new span with GenOps governance attributes."""
        span = self.tracer.start_span(name, **kwargs)

        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)

        return span

    @contextmanager
    def trace_operation(
        self, operation_name: str, operation_type: str = "ai.inference", **attributes
    ):
        """Context manager for tracing AI operations with governance metadata."""
        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Set core operation attributes
                span.set_attribute("genops.operation.type", operation_type)
                span.set_attribute("genops.operation.name", operation_name)
                span.set_attribute("genops.timestamp", int(time.time()))

                # Set additional attributes
                for key, value in attributes.items():
                    if value is not None:
                        span.set_attribute(f"genops.{key}", value)

                yield span

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def record_cost(
        self,
        span: trace.Span,
        cost: float,
        currency: str = "USD",
        provider: str = "",
        model: str = "",
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        **metadata,
    ) -> None:
        """Record cost telemetry on a span."""
        span.set_attribute("genops.cost.amount", cost)
        span.set_attribute("genops.cost.currency", currency)

        if provider:
            span.set_attribute("genops.cost.provider", provider)
        if model:
            span.set_attribute("genops.cost.model", model)
        if tokens_input is not None:
            span.set_attribute("genops.cost.tokens.input", tokens_input)
        if tokens_output is not None:
            span.set_attribute("genops.cost.tokens.output", tokens_output)

        # Record additional cost metadata
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(f"genops.cost.{key}", value)

    def record_policy(
        self,
        span: trace.Span,
        policy_name: str,
        policy_result: str = "allowed",
        policy_reason: Optional[str] = None,
        **metadata,
    ) -> None:
        """Record policy enforcement telemetry."""
        span.set_attribute("genops.policy.name", policy_name)
        span.set_attribute("genops.policy.result", policy_result)

        if policy_reason:
            span.set_attribute("genops.policy.reason", policy_reason)

        # Record additional policy metadata
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(f"genops.policy.{key}", value)

    def record_evaluation(
        self,
        span: trace.Span,
        evaluation_name: str,
        score: float,
        threshold: Optional[float] = None,
        passed: Optional[bool] = None,
        **metadata,
    ) -> None:
        """Record evaluation telemetry."""
        span.set_attribute("genops.eval.name", evaluation_name)
        span.set_attribute("genops.eval.score", score)

        if threshold is not None:
            span.set_attribute("genops.eval.threshold", threshold)
        if passed is not None:
            span.set_attribute("genops.eval.passed", passed)

        # Record additional evaluation metadata
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(f"genops.eval.{key}", value)

    def record_budget(
        self,
        span: trace.Span,
        budget_name: str,
        budget_limit: float,
        budget_used: float,
        budget_remaining: Optional[float] = None,
        **metadata,
    ) -> None:
        """Record budget telemetry."""
        span.set_attribute("genops.budget.name", budget_name)
        span.set_attribute("genops.budget.limit", budget_limit)
        span.set_attribute("genops.budget.used", budget_used)

        if budget_remaining is None:
            budget_remaining = budget_limit - budget_used
        span.set_attribute("genops.budget.remaining", budget_remaining)

        # Calculate and record budget utilization percentage
        utilization = (budget_used / budget_limit) * 100 if budget_limit > 0 else 0
        span.set_attribute("genops.budget.utilization_percent", utilization)

        # Record additional budget metadata
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(f"genops.budget.{key}", value)
