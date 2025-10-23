"""Context manager for block-level AI governance tracking."""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry import trace

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)


@contextmanager
def track(
    operation_name: str,
    operation_type: str = "ai.inference",
    team: Optional[str] = None,
    project: Optional[str] = None,
    feature: Optional[str] = None,
    customer: Optional[str] = None,
    environment: Optional[str] = None,
    **attributes: Any,
) -> Generator[trace.Span, None, None]:
    """
    Context manager for tracking AI operations with governance telemetry.

    Args:
        operation_name: Name of the operation
        operation_type: Type of AI operation (ai.inference, ai.training, etc.)
        team: Team responsible for this operation
        project: Project this operation belongs to
        feature: Feature this operation supports
        customer: Customer this operation serves
        environment: Environment (dev, staging, prod)
        **attributes: Additional governance attributes

    Yields:
        span: The OpenTelemetry span for this operation

    Example:
        with genops.track(
            operation_name="batch_inference",
            team="ml-platform",
            project="recommendation-engine",
            customer="enterprise-customer-123"
        ) as span:
            results = model.predict_batch(inputs)

            # Record cost and evaluation manually
            genops.track_cost(
                cost=0.15,
                provider="openai",
                model="gpt-4",
                tokens_input=1500,
                tokens_output=500
            )

            genops.track_evaluation(
                evaluation_name="relevance_score",
                score=0.87,
                threshold=0.8,
                passed=True
            )
    """
    telemetry = GenOpsTelemetry()

    # Build governance attributes
    governance_attrs = {}
    if team:
        governance_attrs["team"] = team
    if project:
        governance_attrs["project"] = project
    if feature:
        governance_attrs["feature"] = feature
    if customer:
        governance_attrs["customer"] = customer
    if environment:
        governance_attrs["environment"] = environment

    # Add custom attributes
    governance_attrs.update(attributes)

    with telemetry.trace_operation(
        operation_name=operation_name, operation_type=operation_type, **governance_attrs
    ) as span:
        yield span


class GenOpsSpan:
    """
    Convenience wrapper around OpenTelemetry span with GenOps-specific methods.
    """

    def __init__(self, span: trace.Span):
        self.span = span
        self.telemetry = GenOpsTelemetry()

    def record_cost(
        self,
        cost: float,
        currency: str = "USD",
        provider: str = "",
        model: str = "",
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        **metadata: Any,
    ) -> None:
        """Record cost telemetry on this span."""
        self.telemetry.record_cost(
            span=self.span,
            cost=cost,
            currency=currency,
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            **metadata,
        )

    def record_policy(
        self,
        policy_name: str,
        policy_result: str = "allowed",
        policy_reason: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Record policy enforcement telemetry."""
        self.telemetry.record_policy(
            span=self.span,
            policy_name=policy_name,
            policy_result=policy_result,
            policy_reason=policy_reason,
            **metadata,
        )

    def record_evaluation(
        self,
        evaluation_name: str,
        score: float,
        threshold: Optional[float] = None,
        passed: Optional[bool] = None,
        **metadata: Any,
    ) -> None:
        """Record evaluation telemetry."""
        self.telemetry.record_evaluation(
            span=self.span,
            evaluation_name=evaluation_name,
            score=score,
            threshold=threshold,
            passed=passed,
            **metadata,
        )

    def record_budget(
        self,
        budget_name: str,
        budget_limit: float,
        budget_used: float,
        budget_remaining: Optional[float] = None,
        **metadata: Any,
    ) -> None:
        """Record budget telemetry."""
        self.telemetry.record_budget(
            span=self.span,
            budget_name=budget_name,
            budget_limit=budget_limit,
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            **metadata,
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the underlying span."""
        self.span.set_attribute(key, value)


@contextmanager
def track_enhanced(
    operation_name: str,
    operation_type: str = "ai.inference",
    team: Optional[str] = None,
    project: Optional[str] = None,
    feature: Optional[str] = None,
    customer: Optional[str] = None,
    environment: Optional[str] = None,
    **attributes: Any,
) -> Generator[GenOpsSpan, None, None]:
    """
    Enhanced context manager that returns a GenOpsSpan with convenience methods.

    Example:
        with genops.track_enhanced(
            operation_name="content_generation",
            team="content-ai",
            project="blog-writer"
        ) as span:
            content = llm.generate(prompt)

            span.record_cost(
                cost=0.05,
                provider="anthropic",
                model="claude-3-sonnet"
            )

            span.record_evaluation(
                evaluation_name="content_quality",
                score=0.92
            )
    """
    with track(
        operation_name=operation_name,
        operation_type=operation_type,
        team=team,
        project=project,
        feature=feature,
        customer=customer,
        environment=environment,
        **attributes,
    ) as span:
        yield GenOpsSpan(span)
