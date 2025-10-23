"""Function-level instrumentation decorator for GenOps AI governance."""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def track_usage(
    operation_name: Optional[str] = None,
    operation_type: str = "ai.inference",
    team: Optional[str] = None,
    project: Optional[str] = None,
    feature: Optional[str] = None,
    customer: Optional[str] = None,
    environment: Optional[str] = None,
    **attributes: Any,
) -> Callable[[F], F]:
    """
    Decorator to track AI operations with governance telemetry.

    Args:
        operation_name: Name of the operation. Defaults to function name.
        operation_type: Type of AI operation (ai.inference, ai.training, etc.)
        team: Team responsible for this operation
        project: Project this operation belongs to
        feature: Feature this operation supports
        customer: Customer this operation serves
        environment: Environment (dev, staging, prod)
        **attributes: Additional governance attributes

    Example:
        @track_usage(
            operation_name="user_query_processing",
            team="ai-platform",
            project="customer-support",
            feature="chat-assistant"
        )
        def process_user_query(query: str) -> str:
            return llm.complete(query)
    """

    def decorator(func: F) -> F:
        telemetry = GenOpsTelemetry()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"

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
                operation_name=span_name,
                operation_type=operation_type,
                **governance_attrs,
            ) as span:
                # Execute the wrapped function
                result = func(*args, **kwargs)

                # If the result contains cost information, record it
                if hasattr(result, "__dict__") and "cost" in result.__dict__:
                    telemetry.record_cost(
                        span=span,
                        cost=result.cost,
                        provider=getattr(result, "provider", ""),
                        model=getattr(result, "model", ""),
                        tokens_input=getattr(result, "tokens_input", None),
                        tokens_output=getattr(result, "tokens_output", None),
                    )

                return result

        return wrapper

    return decorator


def track_cost(
    cost: float,
    currency: str = "USD",
    provider: str = "",
    model: str = "",
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    **metadata: Any,
) -> None:
    """
    Manually record cost telemetry for the current span.

    Args:
        cost: Cost amount
        currency: Currency code (default: USD)
        provider: AI provider name
        model: Model name
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        **metadata: Additional cost metadata
    """
    from opentelemetry import trace

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        telemetry = GenOpsTelemetry()
        telemetry.record_cost(
            span=current_span,
            cost=cost,
            currency=currency,
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            **metadata,
        )
    else:
        logger.warning("No active span found to record cost telemetry")


def track_evaluation(
    evaluation_name: str,
    score: float,
    threshold: Optional[float] = None,
    passed: Optional[bool] = None,
    **metadata: Any,
) -> None:
    """
    Manually record evaluation telemetry for the current span.

    Args:
        evaluation_name: Name of the evaluation
        score: Evaluation score
        threshold: Score threshold for passing
        passed: Whether the evaluation passed
        **metadata: Additional evaluation metadata
    """
    from opentelemetry import trace

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        telemetry = GenOpsTelemetry()
        telemetry.record_evaluation(
            span=current_span,
            evaluation_name=evaluation_name,
            score=score,
            threshold=threshold,
            passed=passed,
            **metadata,
        )
    else:
        logger.warning("No active span found to record evaluation telemetry")
