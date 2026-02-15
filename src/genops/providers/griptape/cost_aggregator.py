#!/usr/bin/env python3
"""
Griptape Cost Aggregation for GenOps Governance

Provides multi-provider cost tracking and aggregation for Griptape AI framework operations,
including structure-level cost breakdown, provider-specific attribution, and budget management.

Usage:
    from genops.providers.griptape.cost_aggregator import GriptapeCostAggregator

    aggregator = GriptapeCostAggregator()

    # Track costs for Griptape structures
    aggregator.add_structure_cost("agent-123", "openai", "gpt-4", 150, 300)
    aggregator.add_structure_cost("pipeline-456", "anthropic", "claude-3", 200, 400)

    # Get cost summary
    summary = aggregator.get_cost_summary()
    print(f"Total cost: ${summary.total_cost:.6f}")
    print(f"Providers: {list(summary.cost_by_provider.keys())}")

Features:
    - Multi-provider cost tracking (OpenAI, Anthropic, Google, Cohere, etc.)
    - Structure-level cost attribution (Agent, Pipeline, Workflow)
    - Real-time cost aggregation with budget monitoring
    - Provider-specific pricing with fallback strategies
    - Daily, weekly, monthly cost breakdown analytics
    - Export capabilities for financial reporting
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional, Union

# Import existing cost calculators from GenOps providers (optional)
try:
    from genops.providers.openai_cost_calculator import OpenAICostCalculator
except ImportError:
    OpenAICostCalculator = None
try:
    from genops.providers.anthropic_cost_calculator import AnthropicCostCalculator
except ImportError:
    AnthropicCostCalculator = None
try:
    from genops.providers.google_cost_calculator import GoogleCostCalculator
except ImportError:
    GoogleCostCalculator = None
try:
    from genops.providers.bedrock_cost_calculator import BedrockCostCalculator
except ImportError:
    BedrockCostCalculator = None

logger = logging.getLogger(__name__)


@dataclass
class GriptapeCostBreakdown:
    """Cost breakdown for a specific Griptape operation."""

    # Core identification
    structure_id: str
    structure_type: str  # agent, pipeline, workflow, engine, memory
    provider: str
    model: str

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost information
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal

    # Timing and metadata
    timestamp: datetime
    operation_type: str = "run"
    duration: Optional[float] = None

    # Governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    environment: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export/serialization."""
        return {
            "structure_id": self.structure_id,
            "structure_type": self.structure_type,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "total_cost": float(self.total_cost),
            "timestamp": self.timestamp.isoformat(),
            "operation_type": self.operation_type,
            "duration": self.duration,
            "team": self.team,
            "project": self.project,
            "customer_id": self.customer_id,
            "environment": self.environment,
        }


@dataclass
class GriptapeCostSummary:
    """Aggregated cost summary for Griptape operations."""

    # Total costs
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))

    # Provider breakdown
    cost_by_provider: dict[str, Decimal] = field(default_factory=dict)
    cost_by_model: dict[str, Decimal] = field(default_factory=dict)

    # Structure breakdown
    cost_by_structure_type: dict[str, Decimal] = field(default_factory=dict)
    cost_by_structure_id: dict[str, Decimal] = field(default_factory=dict)

    # Usage statistics
    total_requests: int = 0
    total_tokens: int = 0
    unique_providers: set[str] = field(default_factory=set)
    unique_models: set[str] = field(default_factory=set)

    # Time period
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Governance breakdown
    cost_by_team: dict[str, Decimal] = field(default_factory=dict)
    cost_by_project: dict[str, Decimal] = field(default_factory=dict)
    cost_by_customer: dict[str, Decimal] = field(default_factory=dict)
    cost_by_environment: dict[str, Decimal] = field(default_factory=dict)

    def get_top_providers(self, limit: int = 5) -> list[tuple]:
        """Get top providers by cost."""
        return sorted(self.cost_by_provider.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

    def get_top_models(self, limit: int = 5) -> list[tuple]:
        """Get top models by cost."""
        return sorted(self.cost_by_model.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

    def get_cost_efficiency(self) -> dict[str, float]:
        """Calculate cost efficiency metrics."""
        if self.total_tokens == 0:
            return {"cost_per_token": 0.0, "cost_per_request": 0.0}

        cost_per_token = float(self.total_cost) / self.total_tokens
        cost_per_request = float(self.total_cost) / max(self.total_requests, 1)

        return {
            "cost_per_token": cost_per_token,
            "cost_per_request": cost_per_request,
            "tokens_per_request": self.total_tokens / max(self.total_requests, 1),
        }


class GriptapeCostAggregator:
    """
    Multi-provider cost aggregation for Griptape framework operations.

    Tracks costs across all supported LLM providers and provides unified
    reporting and analytics for governance and financial management.
    """

    def __init__(self):
        """Initialize cost aggregator with provider calculators."""

        # Cost breakdown storage
        self.cost_breakdowns: list[GriptapeCostBreakdown] = []
        self._lock = threading.Lock()

        # Provider-specific cost calculators
        self.calculators = {}
        if OpenAICostCalculator is not None:
            self.calculators["openai"] = OpenAICostCalculator()
        if AnthropicCostCalculator is not None:
            self.calculators["anthropic"] = AnthropicCostCalculator()
        if GoogleCostCalculator is not None:
            self.calculators["google"] = GoogleCostCalculator()
        if BedrockCostCalculator is not None:
            self.calculators["bedrock"] = BedrockCostCalculator()

        # Fallback pricing (per 1K tokens) for unsupported providers
        self.fallback_pricing = {
            "cohere": {"input": Decimal("0.0015"), "output": Decimal("0.002")},
            "mistral": {"input": Decimal("0.0007"), "output": Decimal("0.002")},
            "ollama": {"input": Decimal("0"), "output": Decimal("0")},  # Local models
            "huggingface": {"input": Decimal("0.0005"), "output": Decimal("0.0005")},
        }

        logger.info("Griptape cost aggregator initialized with provider support")

    def calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> dict[str, Decimal]:
        """Calculate cost breakdown for a provider/model combination."""

        provider_lower = provider.lower()

        try:
            # Try provider-specific calculator
            if provider_lower in self.calculators:
                calculator = self.calculators[provider_lower]

                if hasattr(calculator, "calculate_cost"):
                    result = calculator.calculate_cost(
                        model, input_tokens, output_tokens
                    )
                    if isinstance(result, dict) and "total_cost" in result:
                        return {
                            "input_cost": result.get("input_cost", Decimal("0")),
                            "output_cost": result.get("output_cost", Decimal("0")),
                            "total_cost": result["total_cost"],
                        }

                # Alternative method names
                for method_name in ["get_cost", "calculate_pricing", "get_pricing"]:
                    if hasattr(calculator, method_name):
                        result = getattr(calculator, method_name)(
                            model, input_tokens, output_tokens
                        )
                        if result:
                            total_cost = (
                                result
                                if isinstance(result, (Decimal, float))
                                else result.get("total_cost", 0)
                            )
                            return {
                                "input_cost": Decimal(str(total_cost))
                                * Decimal("0.6"),  # Estimate
                                "output_cost": Decimal(str(total_cost))
                                * Decimal("0.4"),  # Estimate
                                "total_cost": Decimal(str(total_cost)),
                            }

            # Use fallback pricing
            if provider_lower in self.fallback_pricing:
                pricing = self.fallback_pricing[provider_lower]
                input_cost = (Decimal(str(input_tokens)) / 1000) * pricing["input"]
                output_cost = (Decimal(str(output_tokens)) / 1000) * pricing["output"]

                logger.debug(f"Using fallback pricing for {provider}/{model}")
                return {
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": input_cost + output_cost,
                }

            # Generic fallback - conservative estimate
            logger.warning(
                f"No pricing data for {provider}/{model}, using generic fallback"
            )
            cost_per_1k_tokens = Decimal("0.002")  # Conservative estimate
            total_tokens = input_tokens + output_tokens
            total_cost = (Decimal(str(total_tokens)) / 1000) * cost_per_1k_tokens

            return {
                "input_cost": total_cost * Decimal("0.6"),
                "output_cost": total_cost * Decimal("0.4"),
                "total_cost": total_cost,
            }

        except Exception as e:
            logger.error(f"Error calculating cost for {provider}/{model}: {e}")

            # Emergency fallback
            total_cost = Decimal("0.01")  # Minimal fallback cost
            return {
                "input_cost": total_cost * Decimal("0.6"),
                "output_cost": total_cost * Decimal("0.4"),
                "total_cost": total_cost,
            }

    def add_structure_cost(
        self,
        structure_id: str,
        structure_type: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation_type: str = "run",
        duration: Optional[float] = None,
        governance_attrs: Optional[dict[str, Any]] = None,
    ) -> GriptapeCostBreakdown:
        """Add cost tracking for a Griptape structure operation."""

        # Calculate costs
        cost_breakdown = self.calculate_cost(
            provider, model, input_tokens, output_tokens
        )

        # Create cost breakdown record
        breakdown = GriptapeCostBreakdown(
            structure_id=structure_id,
            structure_type=structure_type,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=cost_breakdown["input_cost"],
            output_cost=cost_breakdown["output_cost"],
            total_cost=cost_breakdown["total_cost"],
            timestamp=datetime.now(),
            operation_type=operation_type,
            duration=duration,
        )

        # Add governance attributes
        if governance_attrs:
            breakdown.team = governance_attrs.get("team")
            breakdown.project = governance_attrs.get("project")
            breakdown.customer_id = governance_attrs.get("customer_id")
            breakdown.environment = governance_attrs.get("environment")

        # Thread-safe storage
        with self._lock:
            self.cost_breakdowns.append(breakdown)

        logger.debug(
            f"Added cost breakdown: {structure_type}={structure_id}, "
            f"{provider}/{model}, ${breakdown.total_cost:.6f}"
        )

        return breakdown

    def get_cost_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        structure_type: Optional[str] = None,
        provider: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
    ) -> GriptapeCostSummary:
        """Get aggregated cost summary with optional filtering."""

        # Filter breakdowns based on criteria
        filtered_breakdowns = []

        with self._lock:
            for breakdown in self.cost_breakdowns:
                # Time filtering
                if start_time and breakdown.timestamp < start_time:
                    continue
                if end_time and breakdown.timestamp > end_time:
                    continue

                # Structure filtering
                if structure_type and breakdown.structure_type != structure_type:
                    continue

                # Provider filtering
                if provider and breakdown.provider.lower() != provider.lower():
                    continue

                # Governance filtering
                if team and breakdown.team != team:
                    continue
                if project and breakdown.project != project:
                    continue

                filtered_breakdowns.append(breakdown)

        # Build summary
        summary = GriptapeCostSummary()

        if not filtered_breakdowns:
            return summary

        # Set time bounds
        summary.start_time = min(b.timestamp for b in filtered_breakdowns)
        summary.end_time = max(b.timestamp for b in filtered_breakdowns)

        # Aggregate costs and metrics
        for breakdown in filtered_breakdowns:
            # Total costs
            summary.total_cost += breakdown.total_cost
            summary.total_requests += 1
            summary.total_tokens += breakdown.total_tokens

            # Provider breakdown
            if breakdown.provider not in summary.cost_by_provider:
                summary.cost_by_provider[breakdown.provider] = Decimal("0")
            summary.cost_by_provider[breakdown.provider] += breakdown.total_cost

            # Model breakdown
            model_key = f"{breakdown.provider}/{breakdown.model}"
            if model_key not in summary.cost_by_model:
                summary.cost_by_model[model_key] = Decimal("0")
            summary.cost_by_model[model_key] += breakdown.total_cost

            # Structure breakdown
            if breakdown.structure_type not in summary.cost_by_structure_type:
                summary.cost_by_structure_type[breakdown.structure_type] = Decimal("0")
            summary.cost_by_structure_type[breakdown.structure_type] += (
                breakdown.total_cost
            )

            if breakdown.structure_id not in summary.cost_by_structure_id:
                summary.cost_by_structure_id[breakdown.structure_id] = Decimal("0")
            summary.cost_by_structure_id[breakdown.structure_id] += breakdown.total_cost

            # Governance breakdown
            if breakdown.team:
                if breakdown.team not in summary.cost_by_team:
                    summary.cost_by_team[breakdown.team] = Decimal("0")
                summary.cost_by_team[breakdown.team] += breakdown.total_cost

            if breakdown.project:
                if breakdown.project not in summary.cost_by_project:
                    summary.cost_by_project[breakdown.project] = Decimal("0")
                summary.cost_by_project[breakdown.project] += breakdown.total_cost

            if breakdown.customer_id:
                if breakdown.customer_id not in summary.cost_by_customer:
                    summary.cost_by_customer[breakdown.customer_id] = Decimal("0")
                summary.cost_by_customer[breakdown.customer_id] += breakdown.total_cost

            if breakdown.environment:
                if breakdown.environment not in summary.cost_by_environment:
                    summary.cost_by_environment[breakdown.environment] = Decimal("0")
                summary.cost_by_environment[breakdown.environment] += (
                    breakdown.total_cost
                )

            # Track unique values
            summary.unique_providers.add(breakdown.provider)
            summary.unique_models.add(f"{breakdown.provider}/{breakdown.model}")

        return summary

    def get_daily_costs(self, date: Optional[datetime] = None) -> Decimal:
        """Get total costs for a specific day."""
        target_date = date or datetime.now()
        start_of_day = datetime.combine(target_date.date(), datetime.min.time())
        end_of_day = start_of_day + timedelta(days=1)

        summary = self.get_cost_summary(start_time=start_of_day, end_time=end_of_day)
        return summary.total_cost

    def get_weekly_costs(self, date: Optional[datetime] = None) -> Decimal:
        """Get total costs for a specific week."""
        target_date = date or datetime.now()
        days_since_monday = target_date.weekday()
        start_of_week = target_date - timedelta(days=days_since_monday)
        start_of_week = datetime.combine(start_of_week.date(), datetime.min.time())
        end_of_week = start_of_week + timedelta(days=7)

        summary = self.get_cost_summary(start_time=start_of_week, end_time=end_of_week)
        return summary.total_cost

    def get_monthly_costs(self, date: Optional[datetime] = None) -> Decimal:
        """Get total costs for a specific month."""
        target_date = date or datetime.now()
        start_of_month = datetime.combine(
            target_date.replace(day=1).date(), datetime.min.time()
        )

        # Calculate end of month
        if target_date.month == 12:
            end_of_month = start_of_month.replace(year=target_date.year + 1, month=1)
        else:
            end_of_month = start_of_month.replace(month=target_date.month + 1)

        summary = self.get_cost_summary(
            start_time=start_of_month, end_time=end_of_month
        )
        return summary.total_cost

    def export_cost_data(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Union[list[dict], str]:
        """Export cost data in specified format."""

        # Get filtered breakdowns
        filtered_breakdowns = []

        with self._lock:
            for breakdown in self.cost_breakdowns:
                if start_time and breakdown.timestamp < start_time:
                    continue
                if end_time and breakdown.timestamp > end_time:
                    continue

                filtered_breakdowns.append(breakdown)

        # Convert to dictionaries
        data = [breakdown.to_dict() for breakdown in filtered_breakdowns]

        if format.lower() == "json":
            import json

            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            import io

            if not data:
                return ""

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()
        else:
            return data

    def clear_old_data(self, days_to_keep: int = 30) -> int:
        """Clear cost data older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self._lock:
            original_count = len(self.cost_breakdowns)
            self.cost_breakdowns = [
                breakdown
                for breakdown in self.cost_breakdowns
                if breakdown.timestamp >= cutoff_date
            ]
            removed_count = original_count - len(self.cost_breakdowns)

        if removed_count > 0:
            logger.info(
                f"Cleared {removed_count} old cost records (>{days_to_keep} days)"
            )

        return removed_count
