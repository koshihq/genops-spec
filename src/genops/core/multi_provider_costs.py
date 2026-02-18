"""
Multi-Provider Cost Aggregation Utilities

This module provides utilities for aggregating and comparing costs across multiple AI providers
(OpenAI, Anthropic, etc.) with unified tracking and governance telemetry.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProviderCostEntry:
    """Single cost entry from a specific provider."""

    provider: str
    model: str
    operation_type: str
    cost: float
    currency: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    timestamp: datetime
    operation_id: str | None = None
    governance_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiProviderCostSummary:
    """Aggregated cost summary across multiple providers."""

    total_cost: float
    currency: str = "USD"
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_operation: dict[str, float] = field(default_factory=dict)
    unique_providers: set[str] = field(default_factory=set)
    unique_models: set[str] = field(default_factory=set)
    total_tokens: int = 0
    total_operations: int = 0
    time_range: tuple | None = None
    governance_attributes: dict[str, Any] = field(default_factory=dict)


class MultiProviderCostAggregator:
    """Aggregates costs across multiple AI providers with governance tracking."""

    def __init__(self, session_id: str | None = None):
        """Initialize cost aggregator.

        Args:
            session_id: Optional session identifier for cost tracking
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.cost_entries: list[ProviderCostEntry] = []
        self.start_time = datetime.now()
        self._governance_context: dict[str, Any] = {}

    def set_governance_context(self, **attributes):
        """Set governance context for all cost tracking.

        Args:
            **attributes: Governance attributes (team, project, customer_id, etc.)
        """
        self._governance_context.update(attributes)
        logger.debug(f"Set governance context: {attributes}")

    def add_cost_entry(
        self,
        provider: str,
        model: str,
        operation_type: str,
        cost: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        currency: str = "USD",
        operation_id: str | None = None,
        **governance_attrs,
    ) -> None:
        """Add a cost entry from a provider operation.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name used
            operation_type: Type of operation (e.g., "completion", "embedding")
            cost: Cost in specified currency
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
            currency: Currency (default: USD)
            operation_id: Optional operation identifier
            **governance_attrs: Additional governance attributes
        """
        # Merge with session-level governance context
        merged_governance = {**self._governance_context, **governance_attrs}

        entry = ProviderCostEntry(
            provider=provider,
            model=model,
            operation_type=operation_type,
            cost=cost,
            currency=currency,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_input + tokens_output,
            timestamp=datetime.now(),
            operation_id=operation_id,
            governance_attributes=merged_governance,
        )

        self.cost_entries.append(entry)
        logger.info(f"Added cost entry: {provider}/{model} - ${cost:.6f}")

    def add_openai_cost(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
        operation_type: str = "completion",
        **governance_attrs,
    ) -> float:
        """Add OpenAI cost entry with automatic cost calculation.

        Args:
            model: OpenAI model name
            tokens_input: Input tokens
            tokens_output: Output tokens
            operation_type: Operation type
            **governance_attrs: Governance attributes

        Returns:
            Calculated cost
        """
        cost = self._calculate_openai_cost(model, tokens_input, tokens_output)

        self.add_cost_entry(
            provider="openai",
            model=model,
            operation_type=operation_type,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            **governance_attrs,
        )

        return cost

    def add_anthropic_cost(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
        operation_type: str = "message",
        **governance_attrs,
    ) -> float:
        """Add Anthropic cost entry with automatic cost calculation.

        Args:
            model: Anthropic model name
            tokens_input: Input tokens
            tokens_output: Output tokens
            operation_type: Operation type
            **governance_attrs: Governance attributes

        Returns:
            Calculated cost
        """
        cost = self._calculate_anthropic_cost(model, tokens_input, tokens_output)

        self.add_cost_entry(
            provider="anthropic",
            model=model,
            operation_type=operation_type,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            **governance_attrs,
        )

        return cost

    def get_summary(self) -> MultiProviderCostSummary:
        """Get aggregated cost summary across all providers.

        Returns:
            MultiProviderCostSummary with aggregated data
        """
        if not self.cost_entries:
            return MultiProviderCostSummary(total_cost=0.0)

        total_cost = sum(entry.cost for entry in self.cost_entries)

        # Aggregate by provider
        cost_by_provider = {}
        for entry in self.cost_entries:
            cost_by_provider[entry.provider] = (
                cost_by_provider.get(entry.provider, 0.0) + entry.cost
            )

        # Aggregate by model
        cost_by_model = {}
        for entry in self.cost_entries:
            model_key = f"{entry.provider}/{entry.model}"
            cost_by_model[model_key] = cost_by_model.get(model_key, 0.0) + entry.cost

        # Aggregate by operation type
        cost_by_operation = {}
        for entry in self.cost_entries:
            cost_by_operation[entry.operation_type] = (
                cost_by_operation.get(entry.operation_type, 0.0) + entry.cost
            )

        # Collect unique providers and models
        unique_providers = {entry.provider for entry in self.cost_entries}
        unique_models = {
            f"{entry.provider}/{entry.model}" for entry in self.cost_entries
        }

        # Calculate totals
        total_tokens = sum(entry.tokens_total for entry in self.cost_entries)

        # Time range
        timestamps = [entry.timestamp for entry in self.cost_entries]
        time_range = (min(timestamps), max(timestamps)) if timestamps else None

        return MultiProviderCostSummary(
            total_cost=total_cost,
            cost_by_provider=cost_by_provider,
            cost_by_model=cost_by_model,
            cost_by_operation=cost_by_operation,
            unique_providers=unique_providers,
            unique_models=unique_models,
            total_tokens=total_tokens,
            total_operations=len(self.cost_entries),
            time_range=time_range,
            governance_attributes=self._governance_context,
        )

    def get_cost_breakdown(self) -> dict[str, Any]:
        """Get detailed cost breakdown with analysis.

        Returns:
            Dictionary with cost analysis and recommendations
        """
        summary = self.get_summary()

        # Calculate efficiency metrics
        avg_cost_per_token = summary.total_cost / max(summary.total_tokens, 1)
        avg_cost_per_operation = summary.total_cost / max(summary.total_operations, 1)

        # Find most/least expensive providers
        if summary.cost_by_provider:
            most_expensive_provider = max(
                summary.cost_by_provider.items(), key=lambda x: x[1]
            )
            least_expensive_provider = min(
                summary.cost_by_provider.items(), key=lambda x: x[1]
            )
        else:
            most_expensive_provider = ("none", 0.0)
            least_expensive_provider = ("none", 0.0)

        # Calculate provider cost ratios
        provider_ratios = {}
        if summary.total_cost > 0:
            for provider, cost in summary.cost_by_provider.items():
                provider_ratios[provider] = cost / summary.total_cost

        return {
            "summary": summary,
            "efficiency_metrics": {
                "avg_cost_per_token": avg_cost_per_token,
                "avg_cost_per_operation": avg_cost_per_operation,
                "total_cost": summary.total_cost,
                "total_tokens": summary.total_tokens,
                "total_operations": summary.total_operations,
            },
            "cost_leaders": {
                "most_expensive_provider": most_expensive_provider,
                "least_expensive_provider": least_expensive_provider,
            },
            "provider_distribution": provider_ratios,
            "recommendations": self._generate_recommendations(summary),
        }

    def export_telemetry(self) -> None:
        """Export cost telemetry to observability platform."""
        try:
            from genops.core.telemetry import GenOpsTelemetry

            telemetry = GenOpsTelemetry()
            summary = self.get_summary()

            # Create telemetry span for multi-provider costs
            with telemetry.trace_operation(
                operation_name="multi_provider_cost_summary",
                operation_type="cost.aggregation",
                session_id=self.session_id,
                **summary.governance_attributes,
            ) as span:
                # Set cost attributes
                span.set_attribute("multi_provider.total_cost", summary.total_cost)
                span.set_attribute("multi_provider.total_tokens", summary.total_tokens)
                span.set_attribute(
                    "multi_provider.total_operations", summary.total_operations
                )
                span.set_attribute(
                    "multi_provider.unique_providers", len(summary.unique_providers)
                )
                span.set_attribute(
                    "multi_provider.unique_models", len(summary.unique_models)
                )

                # Set provider-specific costs
                for provider, cost in summary.cost_by_provider.items():
                    span.set_attribute(f"multi_provider.cost.{provider}", cost)

                # Set operation-specific costs
                for operation, cost in summary.cost_by_operation.items():
                    span.set_attribute(
                        f"multi_provider.cost.operation.{operation}", cost
                    )

                logger.info(
                    f"Exported multi-provider cost telemetry: ${summary.total_cost:.6f}"
                )

        except Exception as e:
            logger.warning(f"Failed to export cost telemetry: {e}")

    def _calculate_openai_cost(
        self, model: str, tokens_input: int, tokens_output: int
    ) -> float:
        """Calculate OpenAI cost based on model pricing."""
        # Current OpenAI pricing (as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
            "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
            "gpt-3.5-turbo-instruct": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
            "text-embedding-3-small": {"input": 0.00002 / 1000, "output": 0.0},
            "text-embedding-3-large": {"input": 0.00013 / 1000, "output": 0.0},
        }

        # Default to GPT-3.5-Turbo pricing for unknown models
        default_pricing = {"input": 0.0015 / 1000, "output": 0.002 / 1000}
        model_pricing = pricing.get(model, default_pricing)

        input_cost = tokens_input * model_pricing["input"]
        output_cost = tokens_output * model_pricing["output"]

        return input_cost + output_cost

    def _calculate_anthropic_cost(
        self, model: str, tokens_input: int, tokens_output: int
    ) -> float:
        """Calculate Anthropic cost based on model pricing."""
        # Current Anthropic pricing (as of 2024)
        pricing = {
            "claude-3-5-sonnet-20241022": {
                "input": 3.00 / 1000000,
                "output": 15.00 / 1000000,
            },
            "claude-3-5-sonnet-20240620": {
                "input": 3.00 / 1000000,
                "output": 15.00 / 1000000,
            },
            "claude-3-5-haiku-20241022": {
                "input": 1.00 / 1000000,
                "output": 5.00 / 1000000,
            },
            "claude-3-opus-20240229": {
                "input": 15.00 / 1000000,
                "output": 75.00 / 1000000,
            },
            "claude-3-sonnet-20240229": {
                "input": 3.00 / 1000000,
                "output": 15.00 / 1000000,
            },
            "claude-3-haiku-20240307": {
                "input": 0.25 / 1000000,
                "output": 1.25 / 1000000,
            },
        }

        # Default to Claude 3.5 Sonnet pricing for unknown models
        default_pricing = {"input": 3.00 / 1000000, "output": 15.00 / 1000000}
        model_pricing = pricing.get(model, default_pricing)

        input_cost = tokens_input * model_pricing["input"]
        output_cost = tokens_output * model_pricing["output"]

        return input_cost + output_cost

    def _generate_recommendations(self, summary: MultiProviderCostSummary) -> list[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        if not summary.cost_by_provider:
            return ["No cost data available for recommendations"]

        # Provider cost analysis
        if len(summary.unique_providers) > 1:
            providers_by_cost = sorted(
                summary.cost_by_provider.items(), key=lambda x: x[1]
            )
            cheapest_provider = providers_by_cost[0][0]
            providers_by_cost[-1][0]

            cost_diff = providers_by_cost[-1][1] - providers_by_cost[0][1]
            if cost_diff > summary.total_cost * 0.2:  # >20% difference
                recommendations.append(
                    f"Consider using {cheapest_provider} more frequently - could save ${cost_diff:.4f}"
                )

        # Token efficiency analysis
        if summary.total_tokens > 0:
            cost_per_token = summary.total_cost / summary.total_tokens
            if cost_per_token > 0.0001:  # High cost per token threshold
                recommendations.append(
                    "High cost per token detected - consider using more efficient models"
                )

        # Operation type analysis
        if summary.cost_by_operation:
            operations_by_cost = sorted(
                summary.cost_by_operation.items(), key=lambda x: x[1], reverse=True
            )
            most_expensive_op = operations_by_cost[0]
            if most_expensive_op[1] > summary.total_cost * 0.5:  # >50% of total cost
                recommendations.append(
                    f"Operation '{most_expensive_op[0]}' accounts for {most_expensive_op[1] / summary.total_cost * 100:.1f}% of costs - review for optimization"
                )

        return recommendations or [
            "No specific optimization recommendations at this time"
        ]


@contextmanager
def multi_provider_cost_tracking(session_id: str | None = None, **governance_attrs):
    """Context manager for multi-provider cost tracking.

    Args:
        session_id: Optional session identifier
        **governance_attrs: Governance attributes (team, project, customer_id, etc.)

    Yields:
        MultiProviderCostAggregator instance
    """
    aggregator = MultiProviderCostAggregator(session_id)
    aggregator.set_governance_context(**governance_attrs)

    try:
        yield aggregator
    finally:
        # Export telemetry on context exit
        aggregator.export_telemetry()


def compare_provider_costs(cost_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare costs across providers for similar operations.

    Args:
        cost_entries: List of cost entry dictionaries with provider, model, cost, etc.

    Returns:
        Comparison analysis dictionary
    """
    aggregator = MultiProviderCostAggregator()

    # Add cost entries to aggregator
    for entry in cost_entries:
        aggregator.add_cost_entry(
            provider=entry.get("provider", "unknown"),
            model=entry.get("model", "unknown"),
            operation_type=entry.get("operation_type", "unknown"),
            cost=entry.get("cost", 0.0),
            tokens_input=entry.get("tokens_input", 0),
            tokens_output=entry.get("tokens_output", 0),
        )

    return aggregator.get_cost_breakdown()


def estimate_migration_costs(
    current_usage: dict[str, Any], target_provider: str, target_model: str
) -> dict[str, Any]:
    """Estimate costs for migrating to a different provider/model.

    Args:
        current_usage: Dictionary with current usage patterns
        target_provider: Target provider name
        target_model: Target model name

    Returns:
        Migration cost analysis
    """
    # This is a simplified estimation - in production you'd want more sophisticated modeling
    current_cost = current_usage.get("total_cost", 0.0)
    current_tokens = current_usage.get("total_tokens", 0)

    # Estimate target costs (simplified)
    aggregator = MultiProviderCostAggregator()

    if target_provider == "openai":
        estimated_cost = aggregator._calculate_openai_cost(
            target_model,
            current_tokens // 2,  # Rough input/output split
            current_tokens // 2,
        )
    elif target_provider == "anthropic":
        estimated_cost = aggregator._calculate_anthropic_cost(
            target_model, current_tokens // 2, current_tokens // 2
        )
    else:
        estimated_cost = current_cost  # No change if unknown provider

    cost_difference = estimated_cost - current_cost
    percentage_change = (
        (cost_difference / current_cost * 100) if current_cost > 0 else 0
    )

    return {
        "current_cost": current_cost,
        "estimated_new_cost": estimated_cost,
        "cost_difference": cost_difference,
        "percentage_change": percentage_change,
        "recommendation": "migrate" if cost_difference < 0 else "evaluate",
        "savings_potential": abs(cost_difference) if cost_difference < 0 else 0,
    }
