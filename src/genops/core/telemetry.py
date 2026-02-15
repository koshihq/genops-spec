"""Core telemetry engine for GenOps AI governance."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class GenOpsTelemetry:
    """Central telemetry engine for GenOps governance signals."""

    def __init__(self, tracer_name: str = "genops-ai"):
        self.tracer = trace.get_tracer(tracer_name)

    def create_span(
        self, name: str, attributes: dict[str, Any] | None = None, **kwargs
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
        # Get effective attributes (defaults + context + overrides)
        try:
            from genops.core.context import get_effective_attributes

            effective_attributes = get_effective_attributes(**attributes)
        except ImportError:
            # Fallback if context module not available
            effective_attributes = attributes

        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Set core operation attributes
                span.set_attribute("genops.operation.type", operation_type)
                span.set_attribute("genops.operation.name", operation_name)
                span.set_attribute("genops.timestamp", int(time.time()))

                # Set effective attributes (includes defaults, context, and overrides)
                for key, value in effective_attributes.items():
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
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        tokens_total: int | None = None,
        input_tokens: int | None = None,  # alias for tokens_input
        output_tokens: int | None = None,  # alias for tokens_output
        **metadata,
    ) -> None:
        """Record cost telemetry on a span."""
        span.set_attribute("genops.cost.total", cost)  # Use 'total' instead of 'amount'
        span.set_attribute("genops.cost.currency", currency)

        if provider:
            span.set_attribute("genops.cost.provider", provider)
        if model:
            span.set_attribute("genops.cost.model", model)

        # Handle token parameters with backward compatibility
        input_tokens_value = tokens_input if tokens_input is not None else input_tokens
        output_tokens_value = (
            tokens_output if tokens_output is not None else output_tokens
        )

        if input_tokens_value is not None:
            span.set_attribute("genops.tokens.input", input_tokens_value)
        if output_tokens_value is not None:
            span.set_attribute("genops.tokens.output", output_tokens_value)

        # Calculate total tokens if not provided
        if (
            tokens_total is None
            and input_tokens_value is not None
            and output_tokens_value is not None
        ):
            tokens_total = input_tokens_value + output_tokens_value
        if tokens_total is not None:
            span.set_attribute("genops.tokens.total", tokens_total)

        # Record additional cost metadata - handle special cases
        for key, value in metadata.items():
            if value is not None:
                if key == "cost_type":
                    span.set_attribute(
                        "genops.cost.type", value
                    )  # Map cost_type to type
                else:
                    span.set_attribute(f"genops.cost.{key}", value)

    def record_policy(
        self,
        span: trace.Span,
        policy_name: str,
        policy_result: str | None = None,
        policy_reason: str | None = None,
        result: str | None = None,  # alias for policy_result
        reason: str | None = None,  # alias for policy_reason
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Record policy enforcement telemetry."""
        span.set_attribute("genops.policy.name", policy_name)

        # Handle result parameter with backward compatibility
        result_value = policy_result if policy_result is not None else result
        if result_value is not None:
            span.set_attribute("genops.policy.result", result_value)

        # Handle reason parameter with backward compatibility
        reason_value = policy_reason if policy_reason is not None else reason
        if reason_value is not None:
            span.set_attribute("genops.policy.reason", reason_value)

        # Handle metadata parameter separately and flatten it
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    span.set_attribute(f"genops.policy.metadata.{key}", value)

        # Record additional policy metadata from kwargs
        for key, value in kwargs.items():
            if value is not None:
                span.set_attribute(f"genops.policy.{key}", value)

    def record_evaluation(
        self,
        span: trace.Span,
        evaluation_name: str | None = None,
        score: float = 0.0,
        threshold: float | None = None,
        passed: bool | None = None,
        metric_name: str | None = None,  # alias for evaluation_name
        evaluator: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Record evaluation telemetry."""
        # Handle name parameter with backward compatibility
        name_value = evaluation_name if evaluation_name is not None else metric_name
        if name_value is not None:
            span.set_attribute(
                "genops.eval.metric", name_value
            )  # Use 'metric' instead of 'name'

        span.set_attribute("genops.eval.score", score)

        if threshold is not None:
            span.set_attribute("genops.eval.threshold", threshold)
        if passed is not None:
            span.set_attribute("genops.eval.passed", passed)
        if evaluator is not None:
            span.set_attribute("genops.eval.evaluator", evaluator)

        # Handle metadata parameter separately and flatten it
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    span.set_attribute(f"genops.eval.metadata.{key}", value)

        # Record additional evaluation metadata from kwargs
        for key, value in kwargs.items():
            if value is not None:
                span.set_attribute(f"genops.eval.{key}", value)

    def record_budget(
        self,
        span: trace.Span,
        budget_name: str,
        budget_limit: float | None = None,
        budget_used: float | None = None,
        budget_remaining: float | None = None,
        allocated: float | None = None,  # alias for budget_limit
        consumed: float | None = None,  # alias for budget_used
        remaining: float | None = None,  # alias for budget_remaining
        **metadata,
    ) -> None:
        """Record budget telemetry."""
        span.set_attribute("genops.budget.name", budget_name)

        # Handle parameter aliases
        limit_value = budget_limit if budget_limit is not None else allocated
        used_value = budget_used if budget_used is not None else consumed
        remaining_value = (
            budget_remaining if budget_remaining is not None else remaining
        )

        if limit_value is not None:
            span.set_attribute(
                "genops.budget.allocated", limit_value
            )  # Use 'allocated' instead of 'limit'
        if used_value is not None:
            span.set_attribute(
                "genops.budget.consumed", used_value
            )  # Use 'consumed' instead of 'used'

        # Calculate remaining if not provided but limit and used are available
        if (
            remaining_value is None
            and limit_value is not None
            and used_value is not None
        ):
            remaining_value = limit_value - used_value
        if remaining_value is not None:
            span.set_attribute("genops.budget.remaining", remaining_value)

        # Calculate and record budget utilization percentage
        if limit_value is not None and used_value is not None and limit_value > 0:
            utilization = (used_value / limit_value) * 100
            span.set_attribute("genops.budget.utilization_percent", utilization)

        # Record additional budget metadata
        for key, value in metadata.items():
            if value is not None:
                span.set_attribute(f"genops.budget.{key}", value)


# NOTE: TelemetryExporter is an alias for GenOpsTelemetry, used by test imports
TelemetryExporter = GenOpsTelemetry
