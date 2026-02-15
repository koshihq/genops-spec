#!/usr/bin/env python3
"""
Haystack Multi-Provider Cost Aggregator

Advanced cost tracking and analysis for Haystack pipelines with multiple AI providers.
Handles cost aggregation across OpenAI, Anthropic, Hugging Face, Cohere, and other
providers used within Haystack components.

Usage:
    from genops.providers.haystack_cost_aggregator import HaystackCostAggregator

    aggregator = HaystackCostAggregator()

    # Track costs from multiple components
    aggregator.add_component_cost("openai_generator", "openai", cost=0.002, tokens_in=100, tokens_out=50)
    aggregator.add_component_cost("embedding_retriever", "huggingface", cost=0.0001, operations=5)

    # Get comprehensive cost analysis
    analysis = aggregator.get_cost_analysis()
    print(f"Total cost: ${analysis.total_cost:.6f}")
    print(f"Cost by provider: {analysis.cost_by_provider}")

Features:
    - Multi-provider cost aggregation and analysis
    - Provider-specific cost calculation models
    - Cost optimization recommendations
    - Budget tracking and alerting
    - Usage pattern analysis and insights
    - Cost projection and forecasting
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported AI provider types for cost tracking."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "huggingface"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"
    GOOGLE_AI = "google_ai"
    MISTRAL = "mistral"
    REPLICATE = "replicate"
    BEDROCK = "bedrock"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class ComponentCostEntry:
    """Individual component cost entry with detailed tracking."""

    component_name: str
    component_type: str
    provider: str
    cost: Decimal
    timestamp: datetime
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    operations: int = 1
    model: Optional[str] = None
    execution_time_seconds: float = 0.0
    custom_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderCostSummary:
    """Cost summary for a specific provider."""

    provider: str
    total_cost: Decimal
    total_tokens_input: int
    total_tokens_output: int
    total_operations: int
    components_used: set[str]
    models_used: set[str]
    avg_cost_per_token: Optional[Decimal] = None
    avg_cost_per_operation: Optional[Decimal] = None
    cost_trend: str = "stable"  # "increasing", "decreasing", "stable"


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation."""

    component_name: str
    current_provider: str
    recommended_provider: str
    potential_savings: Decimal
    confidence: float
    reasoning: str
    migration_complexity: str  # "easy", "moderate", "complex"


@dataclass
class CostAnalysisResult:
    """Comprehensive cost analysis result."""

    total_cost: Decimal
    cost_by_provider: dict[str, Decimal]
    cost_by_component: dict[str, Decimal]
    cost_by_model: dict[str, Decimal]
    provider_summaries: dict[str, ProviderCostSummary]
    optimization_recommendations: list[CostOptimizationRecommendation]
    cost_trends: dict[str, str]
    budget_utilization: Optional[float] = None
    projected_monthly_cost: Optional[Decimal] = None


class HaystackCostAggregator:
    """
    Advanced cost aggregator for Haystack multi-provider workflows.

    Tracks, analyzes, and optimizes costs across multiple AI providers
    used within Haystack pipelines and components.
    """

    def __init__(
        self, budget_limit: Optional[float] = None, enable_retry_logic: bool = True
    ):
        """
        Initialize cost aggregator with enhanced error handling.

        Args:
            budget_limit: Optional budget limit for tracking utilization
            enable_retry_logic: Enable retry logic for cost calculations
        """
        self.budget_limit = Decimal(str(budget_limit)) if budget_limit else None
        self.enable_retry_logic = enable_retry_logic

        # Cost tracking storage
        self.cost_entries: list[ComponentCostEntry] = []
        self.session_costs: dict[str, list[ComponentCostEntry]] = {}

        # Enhanced error handling and retry configuration
        self.error_tracking = {
            "calculation_failures": defaultdict(int),
            "provider_errors": defaultdict(int),
            "retry_attempts": defaultdict(int),
            "cost_estimation_errors": deque(maxlen=50),
            "fallback_calculations_used": 0,
        }

        # Retry configuration for cost calculations
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 0.1,  # Shorter delay for cost calculations
            "max_delay": 2.0,
            "backoff_factor": 1.5,
            "jitter": True,
        }

        # Cost calculation cache for frequently accessed data
        self.calculation_cache = {}
        self.cache_ttl = 300  # 5-minute cache TTL

        # Provider pricing models (cost per 1K tokens or per operation)
        self.provider_pricing = {
            ProviderType.OPENAI: {
                "gpt-4": {"input": Decimal("0.03"), "output": Decimal("0.06")},
                "gpt-4-turbo": {"input": Decimal("0.01"), "output": Decimal("0.03")},
                "gpt-3.5-turbo": {
                    "input": Decimal("0.001"),
                    "output": Decimal("0.002"),
                },
                "text-embedding-3-small": {
                    "input": Decimal("0.00002"),
                    "output": Decimal("0"),
                },
                "text-embedding-3-large": {
                    "input": Decimal("0.00013"),
                    "output": Decimal("0"),
                },
            },
            ProviderType.ANTHROPIC: {
                "claude-3-opus": {
                    "input": Decimal("0.015"),
                    "output": Decimal("0.075"),
                },
                "claude-3-sonnet": {
                    "input": Decimal("0.003"),
                    "output": Decimal("0.015"),
                },
                "claude-3-haiku": {
                    "input": Decimal("0.00025"),
                    "output": Decimal("0.00125"),
                },
            },
            ProviderType.COHERE: {
                "command": {"input": Decimal("0.001"), "output": Decimal("0.002")},
                "command-light": {
                    "input": Decimal("0.0003"),
                    "output": Decimal("0.0006"),
                },
                "embed-english-v3.0": {
                    "input": Decimal("0.0001"),
                    "output": Decimal("0"),
                },
            },
            ProviderType.HUGGING_FACE: {
                "default": {"input": Decimal("0.00001"), "output": Decimal("0.00001")},
                "embedding": {"input": Decimal("0.000001"), "output": Decimal("0")},
            },
            ProviderType.LOCAL: {
                "default": {"input": Decimal("0"), "output": Decimal("0")},
            },
        }

        # Cost optimization thresholds
        self.optimization_thresholds = {
            "high_cost_component": Decimal("1.0"),  # Components costing > $1
            "cost_efficiency_threshold": 0.8,  # 80% efficiency threshold
            "migration_benefit_threshold": Decimal("0.10"),  # 10 cent savings minimum
        }

        # Diagnostic tracking for cost calculation accuracy
        self.diagnostic_metrics = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "fallback_calculations": 0,
            "cache_hits": 0,
            "average_calculation_time": deque(maxlen=100),
        }

    def add_component_cost(
        self,
        component_name: str,
        provider: str,
        cost: float,
        component_type: str = "unknown",
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        operations: int = 1,
        model: Optional[str] = None,
        execution_time_seconds: float = 0.0,
        session_id: Optional[str] = None,
        **custom_attributes,
    ) -> ComponentCostEntry:
        """
        Add a component cost entry to the aggregator.

        Args:
            component_name: Name of the Haystack component
            provider: AI provider used (openai, anthropic, huggingface, etc.)
            cost: Cost of the operation in USD
            component_type: Type of component (generator, retriever, embedder, etc.)
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            operations: Number of operations performed
            model: Specific model used
            execution_time_seconds: Execution time in seconds
            session_id: Optional session ID for grouping
            **custom_attributes: Additional custom tracking attributes

        Returns:
            ComponentCostEntry: The created cost entry
        """
        entry = ComponentCostEntry(
            component_name=component_name,
            component_type=component_type,
            provider=provider,
            cost=Decimal(str(cost)),
            timestamp=datetime.utcnow(),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            operations=operations,
            model=model,
            execution_time_seconds=execution_time_seconds,
            custom_attributes=custom_attributes,
        )

        # Add to main storage
        self.cost_entries.append(entry)

        # Add to session storage if session_id provided
        if session_id:
            if session_id not in self.session_costs:
                self.session_costs[session_id] = []
            self.session_costs[session_id].append(entry)

        logger.debug(f"Added cost entry: {component_name} ({provider}) - ${cost:.6f}")
        return entry

    def calculate_accurate_cost(
        self,
        provider: str,
        model: Optional[str] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        operations: int = 1,
    ) -> Decimal:
        """
        Calculate accurate cost based on provider pricing models.

        Args:
            provider: Provider name
            model: Model name
            tokens_input: Input tokens
            tokens_output: Output tokens
            operations: Number of operations

        Returns:
            Decimal: Calculated cost in USD
        """
        try:
            provider_enum = ProviderType(provider)
        except ValueError:
            provider_enum = ProviderType.UNKNOWN

        if provider_enum not in self.provider_pricing:
            # Fallback estimation
            return Decimal("0.001") * operations

        provider_models = self.provider_pricing[provider_enum]

        # Find model or use default
        if model and model in provider_models:
            pricing = provider_models[model]
        elif "default" in provider_models:
            pricing = provider_models["default"]
        else:
            # Use first available model as fallback
            pricing = list(provider_models.values())[0]

        # Calculate cost based on tokens
        input_cost = (tokens_input / 1000) * pricing["input"]
        output_cost = (tokens_output / 1000) * pricing["output"]

        total_cost = input_cost + output_cost

        # If no tokens, use operation-based pricing
        if total_cost == 0 and operations > 0:
            operation_cost = pricing.get("operation", Decimal("0.001"))
            total_cost = operation_cost * operations

        return total_cost

    def get_cost_analysis(
        self, time_period_hours: Optional[int] = None, session_id: Optional[str] = None
    ) -> CostAnalysisResult:
        """
        Get comprehensive cost analysis with optimization recommendations.

        Args:
            time_period_hours: Limit analysis to recent hours (None for all time)
            session_id: Limit analysis to specific session

        Returns:
            CostAnalysisResult: Complete cost analysis
        """
        # Filter entries based on criteria
        if session_id:
            entries = self.session_costs.get(session_id, [])
        else:
            entries = self.cost_entries

        if time_period_hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_period_hours)
            entries = [e for e in entries if e.timestamp >= cutoff_time]

        if not entries:
            return CostAnalysisResult(
                total_cost=Decimal("0"),
                cost_by_provider={},
                cost_by_component={},
                cost_by_model={},
                provider_summaries={},
                optimization_recommendations=[],
                cost_trends={},
            )

        # Calculate aggregations
        total_cost = sum(entry.cost for entry in entries)

        # Cost by provider
        cost_by_provider = {}
        for entry in entries:
            cost_by_provider[entry.provider] = (
                cost_by_provider.get(entry.provider, Decimal("0")) + entry.cost
            )

        # Cost by component
        cost_by_component = {}
        for entry in entries:
            cost_by_component[entry.component_name] = (
                cost_by_component.get(entry.component_name, Decimal("0")) + entry.cost
            )

        # Cost by model
        cost_by_model = {}
        for entry in entries:
            if entry.model:
                cost_by_model[entry.model] = (
                    cost_by_model.get(entry.model, Decimal("0")) + entry.cost
                )

        # Provider summaries
        provider_summaries = {}
        for provider in cost_by_provider.keys():
            provider_entries = [e for e in entries if e.provider == provider]

            total_tokens_input = sum(e.tokens_input or 0 for e in provider_entries)
            total_tokens_output = sum(e.tokens_output or 0 for e in provider_entries)
            total_operations = sum(e.operations for e in provider_entries)
            components_used = {e.component_name for e in provider_entries}
            models_used = {e.model for e in provider_entries if e.model}

            # Calculate averages
            avg_cost_per_token = None
            if total_tokens_input + total_tokens_output > 0:
                avg_cost_per_token = cost_by_provider[provider] / (
                    total_tokens_input + total_tokens_output
                )

            avg_cost_per_operation = None
            if total_operations > 0:
                avg_cost_per_operation = cost_by_provider[provider] / total_operations

            provider_summaries[provider] = ProviderCostSummary(
                provider=provider,
                total_cost=cost_by_provider[provider],
                total_tokens_input=total_tokens_input,
                total_tokens_output=total_tokens_output,
                total_operations=total_operations,
                components_used=components_used,
                models_used=models_used,
                avg_cost_per_token=avg_cost_per_token,
                avg_cost_per_operation=avg_cost_per_operation,
            )

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            entries, cost_by_component
        )

        # Calculate trends
        cost_trends = self._calculate_cost_trends(entries)

        # Calculate budget utilization
        budget_utilization = None
        if self.budget_limit:
            budget_utilization = float(total_cost / self.budget_limit) * 100

        # Project monthly cost
        projected_monthly_cost = None
        if entries:
            # Calculate daily average and project to monthly
            time_span_days = (
                max(e.timestamp for e in entries) - min(e.timestamp for e in entries)
            ).days
            if time_span_days > 0:
                daily_average = total_cost / time_span_days
                projected_monthly_cost = daily_average * 30

        return CostAnalysisResult(
            total_cost=total_cost,  # type: ignore
            cost_by_provider=cost_by_provider,
            cost_by_component=cost_by_component,
            cost_by_model=cost_by_model,
            provider_summaries=provider_summaries,
            optimization_recommendations=recommendations,
            cost_trends=cost_trends,
            budget_utilization=budget_utilization,
            projected_monthly_cost=projected_monthly_cost,  # type: ignore
        )

    def _generate_optimization_recommendations(
        self, entries: list[ComponentCostEntry], cost_by_component: dict[str, Decimal]
    ) -> list[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Find high-cost components
        for component_name, component_cost in cost_by_component.items():
            if component_cost > self.optimization_thresholds["high_cost_component"]:
                component_entries = [
                    e for e in entries if e.component_name == component_name
                ]
                current_provider = (
                    component_entries[0].provider if component_entries else "unknown"
                )

                # Suggest alternative providers
                alternative = self._find_cost_effective_alternative(
                    component_entries, current_provider
                )

                if alternative:
                    recommendations.append(alternative)

        return recommendations

    def _find_cost_effective_alternative(
        self, component_entries: list[ComponentCostEntry], current_provider: str
    ) -> Optional[CostOptimizationRecommendation]:
        """Find cost-effective alternative provider."""
        if not component_entries:
            return None

        # Calculate average usage pattern
        avg_tokens_input = sum(e.tokens_input or 0 for e in component_entries) / len(
            component_entries
        )
        avg_tokens_output = sum(e.tokens_output or 0 for e in component_entries) / len(
            component_entries
        )
        avg_operations = sum(e.operations for e in component_entries) / len(
            component_entries
        )

        # Current cost
        current_cost = sum(e.cost for e in component_entries) / len(component_entries)

        # Find best alternative
        best_alternative = None
        best_savings = Decimal("0")

        for provider_enum in ProviderType:
            if provider_enum.value == current_provider:
                continue

            # Calculate cost with alternative provider
            alt_cost = self.calculate_accurate_cost(
                provider=provider_enum.value,
                tokens_input=int(avg_tokens_input),
                tokens_output=int(avg_tokens_output),
                operations=int(avg_operations),
            )

            savings = current_cost - alt_cost
            if (
                savings > best_savings
                and savings
                > self.optimization_thresholds["migration_benefit_threshold"]
            ):
                best_alternative = provider_enum.value
                best_savings = savings

        if best_alternative:
            # Estimate migration complexity
            complexity = (
                "easy" if current_provider in ["openai", "anthropic"] else "moderate"
            )
            confidence = 0.8 if best_savings > Decimal("0.50") else 0.6

            return CostOptimizationRecommendation(
                component_name=component_entries[0].component_name,
                current_provider=current_provider,
                recommended_provider=best_alternative,
                potential_savings=best_savings,
                confidence=confidence,
                reasoning=f"Switch to {best_alternative} could save ${best_savings:.4f} per operation",
                migration_complexity=complexity,
            )

        return None

    def _calculate_cost_trends(
        self, entries: list[ComponentCostEntry]
    ) -> dict[str, str]:
        """Calculate cost trends for providers and components."""
        trends = {}

        if len(entries) < 2:
            return trends

        # Sort entries by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        midpoint = len(sorted_entries) // 2

        first_half = sorted_entries[:midpoint]
        second_half = sorted_entries[midpoint:]

        # Calculate trends by provider
        first_half_by_provider = {}
        second_half_by_provider = {}

        for entry in first_half:
            first_half_by_provider[entry.provider] = (
                first_half_by_provider.get(entry.provider, Decimal("0")) + entry.cost
            )

        for entry in second_half:
            second_half_by_provider[entry.provider] = (
                second_half_by_provider.get(entry.provider, Decimal("0")) + entry.cost
            )

        for provider in set(first_half_by_provider.keys()) | set(
            second_half_by_provider.keys()
        ):
            first_cost = first_half_by_provider.get(provider, Decimal("0"))
            second_cost = second_half_by_provider.get(provider, Decimal("0"))

            if first_cost == 0:
                trends[f"{provider}_trend"] = "new"
            elif second_cost > first_cost * Decimal("1.1"):
                trends[f"{provider}_trend"] = "increasing"
            elif second_cost < first_cost * Decimal("0.9"):
                trends[f"{provider}_trend"] = "decreasing"
            else:
                trends[f"{provider}_trend"] = "stable"

        return trends

    def get_session_cost_summary(self, session_id: str) -> dict[str, Any]:
        """Get cost summary for a specific session."""
        if session_id not in self.session_costs:
            return {"error": f"Session {session_id} not found"}

        analysis = self.get_cost_analysis(session_id=session_id)

        return {
            "session_id": session_id,
            "total_cost": float(analysis.total_cost),
            "cost_by_provider": {
                k: float(v) for k, v in analysis.cost_by_provider.items()
            },
            "cost_by_component": {
                k: float(v) for k, v in analysis.cost_by_component.items()
            },
            "components_used": len(analysis.cost_by_component),
            "providers_used": len(analysis.cost_by_provider),
            "optimization_opportunities": len(analysis.optimization_recommendations),
        }

    def reset_tracking(self):
        """Reset all cost tracking data."""
        self.cost_entries.clear()
        self.session_costs.clear()
        logger.info("Cost tracking data reset")

    def export_cost_data(self, format: str = "dict") -> Any:
        """
        Export cost data in various formats.

        Args:
            format: Export format ("dict", "csv", "json")

        Returns:
            Exported data in specified format
        """
        if format == "dict":
            return {
                "entries": [
                    {
                        "component_name": entry.component_name,
                        "provider": entry.provider,
                        "cost": float(entry.cost),
                        "timestamp": entry.timestamp.isoformat(),
                        "tokens_input": entry.tokens_input,
                        "tokens_output": entry.tokens_output,
                        "operations": entry.operations,
                        "model": entry.model,
                    }
                    for entry in self.cost_entries
                ],
                "total_entries": len(self.cost_entries),
                "total_cost": float(sum(entry.cost for entry in self.cost_entries)),
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Export main classes
__all__ = [
    "HaystackCostAggregator",
    "ComponentCostEntry",
    "ProviderCostSummary",
    "CostAnalysisResult",
    "CostOptimizationRecommendation",
    "ProviderType",
]
