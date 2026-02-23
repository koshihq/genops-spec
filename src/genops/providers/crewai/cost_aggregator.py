#!/usr/bin/env python3
"""
CrewAI Multi-Provider Cost Aggregator

Advanced cost tracking and analysis for CrewAI multi-agent workflows, supporting
provider-agnostic cost aggregation, budget enforcement, and optimization recommendations.

Usage:
    from genops.providers.crewai import CrewAICostAggregator, create_crewai_cost_context

    # Create cost context for tracking
    with create_crewai_cost_context("research-crew") as context:
        context.add_agent_cost("researcher", "openai", "gpt-4", 150, 300)
        context.add_agent_cost("analyst", "anthropic", "claude-3", 200, 400)

    # Get comprehensive analysis
    analysis = context.get_cost_analysis()

Features:
    - Multi-provider cost aggregation (OpenAI, Anthropic, Google, etc.)
    - Agent-level cost attribution and tracking
    - Budget monitoring and enforcement
    - Cost optimization recommendations
    - Real-time cost analysis and reporting
"""

import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported AI providers for cost tracking."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    REPLICATE = "replicate"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"
    UNKNOWN = "unknown"


@dataclass
class AgentCostEntry:
    """Cost entry for a single agent execution."""

    agent_name: str
    agent_role: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: Decimal
    execution_time_seconds: float
    timestamp: datetime
    task_context: Optional[str] = None
    custom_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrewCostSummary:
    """Cost summary for a crew execution."""

    crew_name: str
    crew_id: str
    total_cost: Decimal
    cost_by_provider: dict[str, Decimal]
    cost_by_agent: dict[str, Decimal]
    cost_by_model: dict[str, Decimal]
    total_tokens_input: int
    total_tokens_output: int
    total_execution_time_seconds: float
    agent_count: int
    task_count: int
    timestamp: datetime
    unique_providers: set[str] = field(default_factory=set)


@dataclass
class ProviderCostSummary:
    """Cost summary for a specific provider."""

    provider: str
    total_cost: Decimal
    total_operations: int
    agents_used: set[str]
    models_used: set[str]
    total_tokens_input: int
    total_tokens_output: int
    average_cost_per_operation: Decimal
    peak_usage_hour: Optional[datetime] = None


@dataclass
class CostOptimizationRecommendation:
    """Recommendation for cost optimization."""

    agent_name: str
    current_provider: str
    recommended_provider: str
    current_model: str
    recommended_model: str
    potential_savings: Decimal
    confidence_score: float
    reasoning: str
    estimated_performance_impact: str


@dataclass
class CostAnalysisResult:
    """Comprehensive cost analysis result."""

    total_cost: Decimal
    cost_by_provider: dict[str, Decimal]
    cost_by_agent: dict[str, Decimal]
    cost_by_model: dict[str, Decimal]
    crew_summaries: list[CrewCostSummary]
    provider_summaries: dict[str, ProviderCostSummary]
    optimization_recommendations: list[CostOptimizationRecommendation]
    budget_status: dict[str, Any]
    time_period_hours: int
    analysis_timestamp: datetime


class CrewAICostAggregator:
    """Advanced cost aggregator for CrewAI multi-agent workflows."""

    def __init__(
        self,
        budget_limit: float = 100.0,
        time_window_hours: int = 24,
        enable_optimization_recommendations: bool = True,
    ):
        """
        Initialize the cost aggregator.

        Args:
            budget_limit: Daily budget limit in USD
            time_window_hours: Time window for cost analysis
            enable_optimization_recommendations: Enable cost optimization analysis
        """
        self.budget_limit = Decimal(str(budget_limit))
        self.time_window_hours = time_window_hours
        self.enable_optimization_recommendations = enable_optimization_recommendations

        # Thread-safe storage
        self._lock = threading.RLock()
        self._agent_costs: list[AgentCostEntry] = []
        self._crew_summaries: list[CrewCostSummary] = []

        # Provider cost estimation (USD per 1K tokens)
        self._provider_costs = {
            ProviderType.OPENAI.value: {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            },
            ProviderType.ANTHROPIC.value: {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            },
            ProviderType.GOOGLE.value: {
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
                "gemini-pro-vision": {"input": 0.0005, "output": 0.0015},
                "gemini-1.5-pro": {"input": 0.001, "output": 0.003},
                "gemini-1.5-flash": {"input": 0.00015, "output": 0.0006},
            },
            ProviderType.COHERE.value: {
                "command": {"input": 0.001, "output": 0.002},
                "command-light": {"input": 0.0003, "output": 0.0006},
                "command-r": {"input": 0.0005, "output": 0.0015},
                "command-r-plus": {"input": 0.003, "output": 0.015},
            },
        }

        # Default fallback costs for unknown models
        self._default_costs = {
            ProviderType.HUGGINGFACE.value: {"input": 0.0002, "output": 0.0002},
            ProviderType.MISTRAL.value: {"input": 0.0007, "output": 0.0007},
            ProviderType.REPLICATE.value: {"input": 0.001, "output": 0.001},
            ProviderType.TOGETHER.value: {"input": 0.0008, "output": 0.0008},
            ProviderType.FIREWORKS.value: {"input": 0.0002, "output": 0.0002},
            ProviderType.UNKNOWN.value: {"input": 0.001, "output": 0.001},
        }

    def add_agent_execution(
        self,
        agent_name: str,
        agent_role: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        execution_time_seconds: float,
        task_context: Optional[str] = None,
        **custom_attributes,
    ) -> Decimal:
        """
        Add an agent execution cost entry.

        Args:
            agent_name: Name of the agent
            agent_role: Role of the agent
            provider: AI provider used
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            execution_time_seconds: Execution time
            task_context: Optional task context
            **custom_attributes: Additional attributes

        Returns:
            Decimal: Calculated cost for this execution
        """
        with self._lock:
            cost = self._calculate_cost(provider, model, input_tokens, output_tokens)

            entry = AgentCostEntry(
                agent_name=agent_name,
                agent_role=agent_role,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                execution_time_seconds=execution_time_seconds,
                timestamp=datetime.now(),
                task_context=task_context,
                custom_attributes=custom_attributes,
            )

            self._agent_costs.append(entry)
            return cost

    def add_crew_execution(self, crew_result):
        """Add a complete crew execution result."""
        with self._lock:
            # Convert from adapter's CrewAICrewResult
            crew_summary = CrewCostSummary(
                crew_name=crew_result.crew_name,
                crew_id=crew_result.crew_id,
                total_cost=crew_result.total_cost,
                cost_by_provider=crew_result.cost_by_provider,
                cost_by_agent=crew_result.cost_by_agent,
                cost_by_model={},  # TODO: Extract from agent results
                total_tokens_input=sum(
                    r.tokens_input or 0 for r in crew_result.agent_results
                ),
                total_tokens_output=sum(
                    r.tokens_output or 0 for r in crew_result.agent_results
                ),
                total_execution_time_seconds=crew_result.total_execution_time_seconds,
                agent_count=crew_result.total_agents,
                task_count=crew_result.total_tasks,
                timestamp=datetime.now(),
                unique_providers=set(crew_result.cost_by_provider.keys()),
            )

            self._crew_summaries.append(crew_summary)

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Calculate cost for a provider/model combination."""
        # Normalize provider name
        provider_key = provider.lower()

        # Get provider costs
        if provider_key in self._provider_costs:
            model_costs = self._provider_costs[provider_key]
            if model in model_costs:
                rates = model_costs[model]
            else:
                # Use first available model costs as fallback
                rates = next(iter(model_costs.values()))
        else:
            # Use default costs
            rates = self._default_costs.get(
                provider_key, self._default_costs[ProviderType.UNKNOWN.value]
            )

        # Calculate cost (rates are per 1K tokens)
        input_cost = (
            Decimal(str(rates["input"])) * Decimal(str(input_tokens)) / Decimal("1000")
        )
        output_cost = (
            Decimal(str(rates["output"]))
            * Decimal(str(output_tokens))
            / Decimal("1000")
        )

        return input_cost + output_cost

    def get_cost_summary(self, time_period_hours: int = None) -> dict[str, Any]:  # type: ignore[assignment]
        """Get cost summary for the specified time period."""
        if time_period_hours is None:
            time_period_hours = self.time_window_hours

        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)

        with self._lock:
            # Filter entries by time
            recent_entries = [
                entry for entry in self._agent_costs if entry.timestamp >= cutoff_time
            ]

            if not recent_entries:
                return {
                    "total_cost": 0.0,
                    "cost_by_provider": {},
                    "cost_by_agent": {},
                    "agent_executions": 0,
                    "budget_remaining": float(self.budget_limit),
                    "budget_utilization": 0.0,
                }

            # Calculate totals
            total_cost = sum(entry.cost for entry in recent_entries)

            # Group by provider
            cost_by_provider = defaultdict(Decimal)
            for entry in recent_entries:
                cost_by_provider[entry.provider] += entry.cost

            # Group by agent
            cost_by_agent = defaultdict(Decimal)
            for entry in recent_entries:
                cost_by_agent[entry.agent_name] += entry.cost

            budget_utilization = (
                (total_cost / self.budget_limit * 100) if self.budget_limit > 0 else 0
            )

            return {
                "total_cost": float(total_cost),
                "cost_by_provider": {k: float(v) for k, v in cost_by_provider.items()},
                "cost_by_agent": {k: float(v) for k, v in cost_by_agent.items()},
                "agent_executions": len(recent_entries),
                "budget_remaining": float(max(0, self.budget_limit - total_cost)),  # type: ignore
                "budget_utilization": float(budget_utilization),
                "time_period_hours": time_period_hours,
            }

    def get_cost_analysis(self, time_period_hours: int = None) -> CostAnalysisResult:  # type: ignore[assignment]
        """Get comprehensive cost analysis."""
        if time_period_hours is None:
            time_period_hours = self.time_window_hours

        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)

        with self._lock:
            # Filter recent data
            recent_entries = [
                e for e in self._agent_costs if e.timestamp >= cutoff_time
            ]
            recent_crews = [
                c for c in self._crew_summaries if c.timestamp >= cutoff_time
            ]

            # Calculate aggregates
            total_cost = sum(entry.cost for entry in recent_entries)

            cost_by_provider = defaultdict(Decimal)
            cost_by_agent = defaultdict(Decimal)
            cost_by_model = defaultdict(Decimal)

            for entry in recent_entries:
                cost_by_provider[entry.provider] += entry.cost
                cost_by_agent[entry.agent_name] += entry.cost
                cost_by_model[f"{entry.provider}:{entry.model}"] += entry.cost

            # Generate provider summaries
            provider_summaries = {}
            for provider in cost_by_provider:
                provider_entries = [e for e in recent_entries if e.provider == provider]

                provider_summaries[provider] = ProviderCostSummary(
                    provider=provider,
                    total_cost=cost_by_provider[provider],
                    total_operations=len(provider_entries),
                    agents_used={e.agent_name for e in provider_entries},
                    models_used={e.model for e in provider_entries},
                    total_tokens_input=sum(e.input_tokens for e in provider_entries),
                    total_tokens_output=sum(e.output_tokens for e in provider_entries),
                    average_cost_per_operation=cost_by_provider[provider]
                    / max(1, len(provider_entries)),
                )

            # Generate optimization recommendations
            recommendations = []
            if self.enable_optimization_recommendations:
                recommendations = self._generate_optimization_recommendations(
                    recent_entries
                )

            # Budget status
            budget_status = {
                "limit": float(self.budget_limit),
                "used": float(total_cost),
                "remaining": float(max(0, self.budget_limit - total_cost)),  # type: ignore
                "utilization_percentage": float(
                    (total_cost / self.budget_limit * 100)
                    if self.budget_limit > 0
                    else 0
                ),
                "is_over_budget": total_cost > self.budget_limit,
            }

            return CostAnalysisResult(
                total_cost=total_cost,  # type: ignore
                cost_by_provider=dict(cost_by_provider),
                cost_by_agent=dict(cost_by_agent),
                cost_by_model=dict(cost_by_model),
                crew_summaries=recent_crews,
                provider_summaries=provider_summaries,
                optimization_recommendations=recommendations,
                budget_status=budget_status,
                time_period_hours=time_period_hours,
                analysis_timestamp=datetime.now(),
            )

    def _generate_optimization_recommendations(
        self, entries: list[AgentCostEntry]
    ) -> list[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Group by agent
        agent_costs = defaultdict(list)
        for entry in entries:
            agent_costs[entry.agent_name].append(entry)

        for agent_name, agent_entries in agent_costs.items():
            if len(agent_entries) < 3:  # Need sufficient data
                continue

            # Find most used provider/model
            current_provider = max(
                {e.provider for e in agent_entries},
                key=lambda p: sum(1 for e in agent_entries if e.provider == p),
            )
            current_model = max(
                {e.model for e in agent_entries},
                key=lambda m: sum(1 for e in agent_entries if e.model == m),
            )

            # Calculate current average cost
            current_avg_cost = sum(e.cost for e in agent_entries) / len(agent_entries)

            # Simple optimization heuristic - suggest cheaper alternatives
            cheaper_alternatives = self._find_cheaper_alternatives(
                current_provider, current_model
            )

            for alt_provider, alt_model, potential_savings_pct in cheaper_alternatives:
                if potential_savings_pct > 0.2:  # At least 20% savings
                    estimated_savings = current_avg_cost * Decimal(
                        str(potential_savings_pct)
                    )

                    recommendations.append(
                        CostOptimizationRecommendation(
                            agent_name=agent_name,
                            current_provider=current_provider,
                            recommended_provider=alt_provider,
                            current_model=current_model,
                            recommended_model=alt_model,
                            potential_savings=estimated_savings,
                            confidence_score=0.7,  # Conservative confidence
                            reasoning=f"Switch from {current_provider}:{current_model} to {alt_provider}:{alt_model} "
                            f"for ~{potential_savings_pct * 100:.0f}% cost reduction",
                            estimated_performance_impact="minimal",
                        )
                    )

        return recommendations[:5]  # Return top 5 recommendations

    def _find_cheaper_alternatives(
        self, current_provider: str, current_model: str
    ) -> list[tuple]:
        """Find cheaper provider/model alternatives."""
        alternatives = []

        # Get current cost rates
        current_costs = self._provider_costs.get(current_provider, {}).get(
            current_model
        )
        if not current_costs:
            return alternatives

        current_avg_cost = (current_costs["input"] + current_costs["output"]) / 2

        # Check alternatives
        for provider, models in self._provider_costs.items():
            if provider == current_provider:
                continue

            for model, costs in models.items():
                alt_avg_cost = (costs["input"] + costs["output"]) / 2
                if alt_avg_cost < current_avg_cost:
                    savings_pct = (current_avg_cost - alt_avg_cost) / current_avg_cost
                    alternatives.append((provider, model, savings_pct))

        # Sort by potential savings
        alternatives.sort(key=lambda x: x[2], reverse=True)
        return alternatives[:3]  # Top 3 alternatives


# Context manager for cost tracking
@contextmanager
def create_crewai_cost_context(crew_name: str):
    """
    Create a cost tracking context for a CrewAI execution.

    Args:
        crew_name: Name of the crew being tracked

    Example:
        with create_crewai_cost_context("research-crew") as context:
            context.add_agent_cost("researcher", "openai", "gpt-4", 150, 300)
            analysis = context.get_cost_analysis()
    """
    aggregator = CrewAICostAggregator()

    class CostContext:
        def __init__(self, agg):
            self.aggregator = agg
            self.crew_name = crew_name

        def add_agent_cost(
            self,
            agent_name: str,
            provider: str,
            model: str,
            input_tokens: int,
            output_tokens: int,
            execution_time: float = 1.0,
        ):
            return self.aggregator.add_agent_execution(
                agent_name,
                agent_name,
                provider,
                model,
                input_tokens,
                output_tokens,
                execution_time,
            )

        def get_cost_analysis(self):
            return self.aggregator.get_cost_analysis()

    context = CostContext(aggregator)
    yield context


# CLAUDE.md standard functions
def multi_provider_cost_tracking():
    """CLAUDE.md standard function for multi-provider cost tracking."""
    return CrewAICostAggregator(enable_optimization_recommendations=True)


def create_chain_cost_context(chain_id: str):
    """CLAUDE.md standard alias for create_crewai_cost_context."""
    return create_crewai_cost_context(chain_id)


# Export main classes and functions
__all__ = [
    "CrewAICostAggregator",
    "AgentCostEntry",
    "CrewCostSummary",
    "ProviderCostSummary",
    "CostOptimizationRecommendation",
    "CostAnalysisResult",
    "ProviderType",
    "create_crewai_cost_context",
    "multi_provider_cost_tracking",
    "create_chain_cost_context",
]
