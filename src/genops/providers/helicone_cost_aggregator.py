#!/usr/bin/env python3
"""
GenOps Helicone Cost Aggregator

This module provides comprehensive cost aggregation and intelligence for Helicone
AI gateway operations across multiple providers. It tracks costs at the gateway
level while providing detailed breakdown by provider, model, and operation type.

Features:
- Cross-provider cost aggregation through Helicone gateway
- Real-time cost tracking with provider-specific breakdowns
- Gateway overhead analysis and optimization insights
- Multi-provider routing cost comparison
- Enterprise-grade cost attribution and reporting
- Cost optimization recommendations across providers
- Historical cost trend analysis and forecasting

Key Concepts:
- Gateway Cost: Total cost including provider + Helicone fees
- Provider Cost: Direct cost charged by AI provider (OpenAI, Anthropic, etc.)
- Helicone Cost: Gateway service fees and overhead
- Routing Intelligence: Cost-based provider selection
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class GatewayCostBreakdown:
    """Detailed cost breakdown for a gateway operation."""
    operation_id: str
    timestamp: datetime
    provider: str
    model: str
    operation_type: str  # chat, embed, etc.

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost breakdown
    provider_cost: float
    helicone_cost: float
    total_cost: float

    # Performance metrics
    request_time: float
    gateway_overhead: float
    tokens_per_second: float
    cost_per_token: float

    # Governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    environment: str = "production"
    cost_center: Optional[str] = None

    # Routing context
    routing_strategy: Optional[str] = None
    alternative_providers: List[str] = field(default_factory=list)
    cost_savings: float = 0.0  # Savings compared to alternatives

@dataclass
class ProviderCostSummary:
    """Cost summary for a specific provider through gateway."""
    provider: str
    operations: int = 0
    total_cost: float = 0.0
    provider_cost: float = 0.0
    helicone_cost: float = 0.0
    total_tokens: int = 0
    avg_cost_per_operation: float = 0.0
    avg_cost_per_token: float = 0.0
    avg_request_time: float = 0.0
    avg_gateway_overhead: float = 0.0
    models_used: Dict[str, int] = field(default_factory=dict)
    operation_types: Dict[str, int] = field(default_factory=dict)

@dataclass
class GatewayCostSummary:
    """Comprehensive gateway cost summary across all providers."""
    session_id: str
    start_time: datetime
    end_time: datetime

    # Overall totals
    total_operations: int = 0
    total_cost: float = 0.0
    total_provider_cost: float = 0.0
    total_helicone_cost: float = 0.0
    total_tokens: int = 0

    # Gateway intelligence
    unique_providers: int = 0
    routing_decisions: int = 0
    cost_optimizations: int = 0
    total_savings: float = 0.0

    # Performance aggregates
    avg_request_time: float = 0.0
    avg_gateway_overhead: float = 0.0
    avg_cost_per_operation: float = 0.0
    avg_cost_per_token: float = 0.0

    # Provider breakdown
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    operations_by_provider: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)

    # Governance breakdown
    cost_by_team: Dict[str, float] = field(default_factory=dict)
    cost_by_project: Dict[str, float] = field(default_factory=dict)
    cost_by_customer: Dict[str, float] = field(default_factory=dict)

@dataclass
class CostOptimizationInsight:
    """Cost optimization recommendation."""
    insight_type: str
    description: str
    potential_savings: float
    confidence: float  # 0.0 to 1.0
    actionable_steps: List[str]
    affected_operations: int
    provider_recommendations: Dict[str, str] = field(default_factory=dict)

class HeliconeCostAggregator:
    """
    Advanced cost aggregation system for Helicone AI gateway operations.
    
    Provides comprehensive cost tracking, analysis, and optimization insights
    across multiple AI providers routed through the Helicone gateway.
    """

    def __init__(
        self,
        session_id: str,
        enable_optimization_insights: bool = True,
        cost_tracking_granularity: str = "operation"  # operation, minute, hour
    ):
        """
        Initialize cost aggregator for gateway operations.
        
        Args:
            session_id: Unique session identifier
            enable_optimization_insights: Generate cost optimization recommendations
            cost_tracking_granularity: Level of cost tracking detail
        """
        self.session_id = session_id
        self.enable_optimization_insights = enable_optimization_insights
        self.cost_tracking_granularity = cost_tracking_granularity

        # Cost tracking storage
        self.operations: List[GatewayCostBreakdown] = []
        self.provider_summaries: Dict[str, ProviderCostSummary] = {}

        # Session metadata
        self.start_time = datetime.utcnow()
        self.last_operation_time = self.start_time

        # Optimization tracking
        self.routing_decisions: List[Dict[str, Any]] = []
        self.cost_optimizations: List[Dict[str, Any]] = []
        self.insights_cache: List[CostOptimizationInsight] = []

        logger.debug(f"Initialized Helicone cost aggregator for session {session_id}")

    def add_gateway_operation(
        self,
        operation_id: str,
        provider: str,
        model: str,
        operation_type: str,
        input_tokens: int,
        output_tokens: int,
        provider_cost: float,
        helicone_cost: float,
        request_time: float,
        gateway_overhead: float,
        **governance_kwargs
    ) -> GatewayCostBreakdown:
        """
        Add a gateway operation to cost tracking.
        
        Args:
            operation_id: Unique operation identifier
            provider: AI provider (openai, anthropic, etc.)
            model: Model name used
            operation_type: Type of operation (chat, embed, etc.)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens  
            provider_cost: Cost charged by the provider
            helicone_cost: Helicone gateway service cost
            request_time: Total request time in seconds
            gateway_overhead: Gateway processing overhead in seconds
            **governance_kwargs: Team, project, customer_id, etc.
        
        Returns:
            GatewayCostBreakdown with complete cost analysis
        """
        timestamp = datetime.utcnow()
        total_tokens = input_tokens + output_tokens
        total_cost = provider_cost + helicone_cost

        # Create cost breakdown
        cost_breakdown = GatewayCostBreakdown(
            operation_id=operation_id,
            timestamp=timestamp,
            provider=provider,
            model=model,
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            provider_cost=provider_cost,
            helicone_cost=helicone_cost,
            total_cost=total_cost,
            request_time=request_time,
            gateway_overhead=gateway_overhead,
            tokens_per_second=total_tokens / max(request_time, 0.001),
            cost_per_token=total_cost / max(total_tokens, 1),
            team=governance_kwargs.get("team"),
            project=governance_kwargs.get("project"),
            customer_id=governance_kwargs.get("customer_id"),
            environment=governance_kwargs.get("environment", "production"),
            cost_center=governance_kwargs.get("cost_center"),
            routing_strategy=governance_kwargs.get("routing_strategy")
        )

        # Add to operations log
        self.operations.append(cost_breakdown)
        self.last_operation_time = timestamp

        # Update provider summaries
        self._update_provider_summary(cost_breakdown)

        # Track routing decisions
        if governance_kwargs.get("routing_strategy"):
            self._track_routing_decision(cost_breakdown, governance_kwargs)

        # Generate optimization insights if enabled
        if self.enable_optimization_insights and len(self.operations) % 10 == 0:
            self._generate_optimization_insights()

        logger.debug(f"Added gateway operation {operation_id}: {provider}/{model} - ${total_cost:.6f}")
        return cost_breakdown

    def _update_provider_summary(self, operation: GatewayCostBreakdown):
        """Update provider-specific cost summary."""
        provider = operation.provider

        if provider not in self.provider_summaries:
            self.provider_summaries[provider] = ProviderCostSummary(provider=provider)

        summary = self.provider_summaries[provider]

        # Update counters and totals
        summary.operations += 1
        summary.total_cost += operation.total_cost
        summary.provider_cost += operation.provider_cost
        summary.helicone_cost += operation.helicone_cost
        summary.total_tokens += operation.total_tokens

        # Update averages
        summary.avg_cost_per_operation = summary.total_cost / summary.operations
        summary.avg_cost_per_token = summary.total_cost / max(summary.total_tokens, 1)

        # Update timing averages
        total_request_time = getattr(summary, '_total_request_time', 0.0) + operation.request_time
        total_gateway_overhead = getattr(summary, '_total_gateway_overhead', 0.0) + operation.gateway_overhead

        summary._total_request_time = total_request_time
        summary._total_gateway_overhead = total_gateway_overhead
        summary.avg_request_time = total_request_time / summary.operations
        summary.avg_gateway_overhead = total_gateway_overhead / summary.operations

        # Update model usage
        summary.models_used[operation.model] = summary.models_used.get(operation.model, 0) + 1
        summary.operation_types[operation.operation_type] = summary.operation_types.get(operation.operation_type, 0) + 1

    def _track_routing_decision(self, operation: GatewayCostBreakdown, governance_kwargs: Dict[str, Any]):
        """Track routing decision for analysis."""
        routing_info = {
            "operation_id": operation.operation_id,
            "timestamp": operation.timestamp.isoformat(),
            "selected_provider": operation.provider,
            "routing_strategy": governance_kwargs.get("routing_strategy"),
            "alternative_providers": governance_kwargs.get("alternative_providers", []),
            "decision_factors": governance_kwargs.get("decision_factors", {}),
            "cost_impact": governance_kwargs.get("cost_savings", 0.0)
        }

        self.routing_decisions.append(routing_info)

        # Track cost savings from routing
        cost_savings = governance_kwargs.get("cost_savings", 0.0)
        if cost_savings > 0:
            self.cost_optimizations.append({
                "operation_id": operation.operation_id,
                "optimization_type": "routing",
                "savings": cost_savings,
                "strategy": governance_kwargs.get("routing_strategy"),
                "timestamp": operation.timestamp.isoformat()
            })

    def _generate_optimization_insights(self):
        """Generate cost optimization insights based on recent operations."""
        if not self.operations:
            return

        insights = []
        recent_ops = self.operations[-50:]  # Analyze last 50 operations

        # Provider cost comparison insight
        provider_costs = defaultdict(list)
        for op in recent_ops:
            provider_costs[op.provider].append(op.cost_per_token)

        if len(provider_costs) > 1:
            avg_costs = {p: sum(costs)/len(costs) for p, costs in provider_costs.items()}
            cheapest_provider = min(avg_costs.keys(), key=lambda x: avg_costs[x])
            most_expensive = max(avg_costs.keys(), key=lambda x: avg_costs[x])

            potential_savings = (avg_costs[most_expensive] - avg_costs[cheapest_provider]) * sum(op.total_tokens for op in recent_ops)

            if potential_savings > 0.01:  # Only suggest if savings > 1 cent
                insights.append(CostOptimizationInsight(
                    insight_type="provider_optimization",
                    description=f"Routing more requests to {cheapest_provider} could reduce costs",
                    potential_savings=potential_savings,
                    confidence=0.8,
                    actionable_steps=[
                        f"Configure routing to prefer {cheapest_provider} for similar tasks",
                        "Set up cost-based routing strategy",
                        "Monitor quality impact of provider switching"
                    ],
                    affected_operations=len([op for op in recent_ops if op.provider == most_expensive]),
                    provider_recommendations={
                        "preferred": cheapest_provider,
                        "avoid": most_expensive
                    }
                ))

        # Gateway overhead insight
        high_overhead_ops = [op for op in recent_ops if op.gateway_overhead > 0.5]  # > 500ms
        if len(high_overhead_ops) > len(recent_ops) * 0.2:  # More than 20% have high overhead
            avg_overhead = sum(op.gateway_overhead for op in high_overhead_ops) / len(high_overhead_ops)

            insights.append(CostOptimizationInsight(
                insight_type="gateway_performance",
                description=f"High gateway overhead detected (avg {avg_overhead:.2f}s)",
                potential_savings=0.0,  # Performance issue, not direct cost
                confidence=0.9,
                actionable_steps=[
                    "Consider caching frequently used prompts",
                    "Implement request batching where possible",
                    "Evaluate self-hosted gateway for high-volume usage"
                ],
                affected_operations=len(high_overhead_ops)
            ))

        # Model right-sizing insight
        model_usage = defaultdict(list)
        for op in recent_ops:
            model_usage[op.model].append((op.total_tokens, op.total_cost))

        for model, usage_data in model_usage.items():
            avg_tokens = sum(tokens for tokens, _ in usage_data) / len(usage_data)
            if model in ["gpt-4", "claude-3-opus", "mistral-large"] and avg_tokens < 100:
                # Using expensive model for simple tasks
                insights.append(CostOptimizationInsight(
                    insight_type="model_optimization",
                    description=f"Using {model} for low-token operations (avg {avg_tokens:.0f} tokens)",
                    potential_savings=sum(cost * 0.6 for _, cost in usage_data),  # Estimate 60% savings
                    confidence=0.7,
                    actionable_steps=[
                        "Consider using cheaper models for simple tasks",
                        "Implement task complexity-based model selection",
                        "Set up A/B testing to validate quality with cheaper models"
                    ],
                    affected_operations=len(usage_data),
                    provider_recommendations={
                        "alternatives": "gpt-3.5-turbo, claude-3-haiku, mistral-small"
                    }
                ))

        # Cache insights from recent operations
        self.insights_cache.extend(insights)

        # Keep only recent insights (last 100)
        self.insights_cache = self.insights_cache[-100:]

    def get_cost_summary(self) -> GatewayCostSummary:
        """Get comprehensive cost summary for the session."""
        if not self.operations:
            return GatewayCostSummary(
                session_id=self.session_id,
                start_time=self.start_time,
                end_time=self.last_operation_time
            )

        # Calculate totals
        total_operations = len(self.operations)
        total_cost = sum(op.total_cost for op in self.operations)
        total_provider_cost = sum(op.provider_cost for op in self.operations)
        total_helicone_cost = sum(op.helicone_cost for op in self.operations)
        total_tokens = sum(op.total_tokens for op in self.operations)

        # Calculate averages
        avg_request_time = sum(op.request_time for op in self.operations) / total_operations
        avg_gateway_overhead = sum(op.gateway_overhead for op in self.operations) / total_operations
        avg_cost_per_operation = total_cost / total_operations
        avg_cost_per_token = total_cost / max(total_tokens, 1)

        # Provider breakdowns
        cost_by_provider = {}
        operations_by_provider = {}
        for provider, summary in self.provider_summaries.items():
            cost_by_provider[provider] = summary.total_cost
            operations_by_provider[provider] = summary.operations

        # Model breakdown
        cost_by_model = defaultdict(float)
        for op in self.operations:
            cost_by_model[op.model] += op.total_cost

        # Governance breakdowns
        cost_by_team = defaultdict(float)
        cost_by_project = defaultdict(float)
        cost_by_customer = defaultdict(float)

        for op in self.operations:
            if op.team:
                cost_by_team[op.team] += op.total_cost
            if op.project:
                cost_by_project[op.project] += op.total_cost
            if op.customer_id:
                cost_by_customer[op.customer_id] += op.total_cost

        # Gateway intelligence metrics
        unique_providers = len(self.provider_summaries)
        routing_decisions = len(self.routing_decisions)
        cost_optimizations = len(self.cost_optimizations)
        total_savings = sum(opt["savings"] for opt in self.cost_optimizations)

        return GatewayCostSummary(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=self.last_operation_time,
            total_operations=total_operations,
            total_cost=total_cost,
            total_provider_cost=total_provider_cost,
            total_helicone_cost=total_helicone_cost,
            total_tokens=total_tokens,
            unique_providers=unique_providers,
            routing_decisions=routing_decisions,
            cost_optimizations=cost_optimizations,
            total_savings=total_savings,
            avg_request_time=avg_request_time,
            avg_gateway_overhead=avg_gateway_overhead,
            avg_cost_per_operation=avg_cost_per_operation,
            avg_cost_per_token=avg_cost_per_token,
            cost_by_provider=dict(cost_by_provider),
            operations_by_provider=dict(operations_by_provider),
            cost_by_model=dict(cost_by_model),
            cost_by_team=dict(cost_by_team),
            cost_by_project=dict(cost_by_project),
            cost_by_customer=dict(cost_by_customer)
        )

    def get_provider_summary(self, provider: str) -> Optional[ProviderCostSummary]:
        """Get cost summary for a specific provider."""
        return self.provider_summaries.get(provider)

    def get_optimization_insights(self) -> List[CostOptimizationInsight]:
        """Get current cost optimization insights."""
        if self.enable_optimization_insights:
            # Ensure insights are up to date
            self._generate_optimization_insights()

        return self.insights_cache.copy()

    def get_routing_analysis(self) -> Dict[str, Any]:
        """Get analysis of routing decisions and their cost impact."""
        if not self.routing_decisions:
            return {
                "total_routing_decisions": 0,
                "strategies_used": [],
                "cost_impact": 0.0,
                "provider_selection_frequency": {}
            }

        strategies_used = list(set(decision.get("routing_strategy") for decision in self.routing_decisions))
        total_cost_impact = sum(decision.get("cost_impact", 0.0) for decision in self.routing_decisions)

        provider_selections = defaultdict(int)
        for decision in self.routing_decisions:
            provider_selections[decision["selected_provider"]] += 1

        return {
            "total_routing_decisions": len(self.routing_decisions),
            "strategies_used": [s for s in strategies_used if s],
            "cost_impact": total_cost_impact,
            "provider_selection_frequency": dict(provider_selections),
            "avg_savings_per_decision": total_cost_impact / max(len(self.routing_decisions), 1)
        }

    def export_cost_data(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export comprehensive cost data for external analysis.
        
        Args:
            format: Export format ("json", "dict")
            
        Returns:
            Cost data in requested format
        """
        summary = self.get_cost_summary()
        insights = self.get_optimization_insights()
        routing_analysis = self.get_routing_analysis()

        export_data = {
            "session_summary": asdict(summary),
            "provider_summaries": {
                provider: asdict(summary) for provider, summary in self.provider_summaries.items()
            },
            "recent_operations": [
                asdict(op) for op in self.operations[-20:]  # Last 20 operations
            ],
            "optimization_insights": [
                asdict(insight) for insight in insights
            ],
            "routing_analysis": routing_analysis,
            "export_timestamp": datetime.utcnow().isoformat()
        }

        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data

    def reset_session(self, new_session_id: Optional[str] = None):
        """Reset cost tracking for new session."""
        self.session_id = new_session_id or self.session_id
        self.operations.clear()
        self.provider_summaries.clear()
        self.routing_decisions.clear()
        self.cost_optimizations.clear()
        self.insights_cache.clear()

        self.start_time = datetime.utcnow()
        self.last_operation_time = self.start_time

        logger.info(f"Reset cost aggregator for new session: {self.session_id}")

# Convenience functions for integration
def create_cost_aggregator(session_id: str, **kwargs) -> HeliconeCostAggregator:
    """Create a new cost aggregator instance."""
    return HeliconeCostAggregator(session_id=session_id, **kwargs)

def aggregate_multi_session_costs(aggregators: List[HeliconeCostAggregator]) -> Dict[str, Any]:
    """Aggregate costs across multiple sessions."""
    if not aggregators:
        return {}

    total_cost = sum(agg.get_cost_summary().total_cost for agg in aggregators)
    total_operations = sum(agg.get_cost_summary().total_operations for agg in aggregators)

    # Merge provider costs
    all_provider_costs = defaultdict(float)
    for agg in aggregators:
        for provider, cost in agg.get_cost_summary().cost_by_provider.items():
            all_provider_costs[provider] += cost

    return {
        "sessions": len(aggregators),
        "total_operations": total_operations,
        "total_cost": total_cost,
        "avg_cost_per_session": total_cost / len(aggregators),
        "cost_by_provider": dict(all_provider_costs),
        "session_ids": [agg.session_id for agg in aggregators]
    }

__all__ = [
    "HeliconeCostAggregator",
    "GatewayCostBreakdown",
    "ProviderCostSummary",
    "GatewayCostSummary",
    "CostOptimizationInsight",
    "create_cost_aggregator",
    "aggregate_multi_session_costs"
]
