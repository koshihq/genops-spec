"""LlamaIndex cost aggregator for GenOps AI governance."""

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Default provider pricing per 1K tokens
PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "openai": {"input": 0.03, "output": 0.06},
    "anthropic": {"input": 0.025, "output": 0.05},
    "google": {"input": 0.0005, "output": 0.0015},
    "cohere": {"input": 0.015, "output": 0.015},
    "huggingface": {"input": 0.001, "output": 0.001},
    "local": {"input": 0.0, "output": 0.0},
}


@dataclass
class RAGCostBreakdown:
    """Detailed cost breakdown for RAG pipeline operations."""

    total_cost: float
    embedding_cost: float = 0.0
    retrieval_cost: float = 0.0  # Vector store operations
    synthesis_cost: float = 0.0  # LLM generation
    agent_cost: float = 0.0  # Agent tool usage

    embedding_tokens: int = 0
    synthesis_tokens: int = 0
    retrieval_operations: int = 0
    agent_steps: int = 0

    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_operation: dict[str, float] = field(default_factory=dict)

    optimization_suggestions: list[str] = field(default_factory=list)


@dataclass
class LlamaIndexCostSummary:
    """Comprehensive cost summary for LlamaIndex operations."""

    total_cost: float
    operation_count: int
    rag_pipelines: int = 0
    agent_interactions: int = 0

    # Cost breakdown
    cost_breakdown: RAGCostBreakdown = field(
        default_factory=lambda: RAGCostBreakdown(0.0)
    )

    # Provider and model tracking
    unique_providers: set[str] = field(default_factory=set)
    unique_models: set[str] = field(default_factory=set)

    # Performance metrics
    avg_query_latency_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_synthesis_latency_ms: float = 0.0

    # Quality metrics
    retrieval_accuracy: Optional[float] = None
    synthesis_quality_score: Optional[float] = None

    # Budget tracking
    budget_status: Optional[dict[str, Any]] = None
    efficiency_metrics: Optional[dict[str, float]] = None


@dataclass
class BudgetAlert:
    """Budget monitoring alert for LlamaIndex operations."""

    alert_type: str  # 'warning', 'critical', 'exceeded'
    current_cost: float
    budget_limit: float
    percentage_used: float
    remaining_budget: float
    operation_type: Optional[str] = None  # 'query', 'embedding', 'agent'
    recommendation: Optional[str] = None


class LlamaIndexCostAggregator:
    """
    Advanced cost aggregator for LlamaIndex RAG and agent operations.

    Provides intelligent cost tracking across:
    - Query engines and RAG pipelines
    - Embedding operations and vector stores
    - Agent workflows and tool usage
    - Multi-provider LLM operations
    """

    def __init__(
        self,
        context_name: str,
        budget_limit: Optional[float] = None,
        enable_alerts: bool = True,
        embedding_cost_per_1k: float = 0.0001,  # Default embedding cost
        retrieval_cost_per_op: float = 0.00001,  # Default vector search cost
        **governance_defaults,
    ):
        """
        Initialize LlamaIndex cost aggregator.

        Args:
            context_name: Name for this cost tracking context
            budget_limit: Maximum allowed cost in USD
            enable_alerts: Enable budget alerts and warnings
            embedding_cost_per_1k: Cost per 1K embedding tokens
            retrieval_cost_per_op: Cost per retrieval operation
            **governance_defaults: Default governance attributes
        """
        self.context_name = context_name
        self.context_id = str(uuid.uuid4())
        self.budget_limit = budget_limit
        self.enable_alerts = enable_alerts
        self.governance_defaults = governance_defaults

        # Cost calculation defaults
        self.embedding_cost_per_1k = embedding_cost_per_1k
        self.retrieval_cost_per_op = retrieval_cost_per_op

        # Tracking state
        self.operations: list[dict[str, Any]] = []
        self.start_time = time.time()
        self.total_cost = 0.0
        self.alerts: list[BudgetAlert] = []

        # Cost breakdown tracking
        self._cost_by_provider: dict[str, float] = defaultdict(float)
        self._cost_by_model: dict[str, float] = defaultdict(float)
        self._cost_by_operation: dict[str, float] = defaultdict(float)

        # Performance tracking
        self._query_latencies: list[float] = []
        self._retrieval_latencies: list[float] = []
        self._synthesis_latencies: list[float] = []

        # Load provider pricing if available
        self._load_provider_pricing()

    def _load_provider_pricing(self):
        """Load provider-specific pricing information."""
        # Standard pricing for common providers (as of 2024)
        self.provider_pricing = {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "text-embedding-ada-002": {"embedding": 0.0001},
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
            "google": {
                "gemini-pro": {"input": 0.00025, "output": 0.0005},
                "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
            },
        }

    def add_llamaindex_operation(self, operation_data: dict[str, Any]) -> str:
        """
        Add a LlamaIndex operation to cost tracking.

        Args:
            operation_data: Operation details with cost information

        Returns:
            Operation ID for reference
        """
        operation_id = operation_data.get("operation_id", str(uuid.uuid4()))

        # Calculate cost if not provided
        if "cost_usd" not in operation_data or operation_data["cost_usd"] is None:
            operation_data["cost_usd"] = self._calculate_operation_cost(operation_data)

        # Add to tracking
        self.operations.append(operation_data)
        self.total_cost += operation_data.get("cost_usd", 0.0)

        # Update aggregated metrics
        provider = operation_data.get("provider", "unknown")
        model = operation_data.get("model", "unknown")
        operation_type = operation_data.get("operation_type", "unknown")
        cost = operation_data.get("cost_usd", 0.0)

        self._cost_by_provider[provider] += cost
        self._cost_by_model[model] += cost
        self._cost_by_operation[operation_type] += cost

        # Track latencies
        duration_ms = operation_data.get("duration_ms", 0.0)
        if operation_type == "query":
            self._query_latencies.append(duration_ms)
        elif operation_type == "retrieve":
            self._retrieval_latencies.append(duration_ms)
        elif operation_type in ["llm_call", "synthesize"]:
            self._synthesis_latencies.append(duration_ms)

        # Check budget constraints
        if self.enable_alerts and self.budget_limit:
            self._check_budget_alerts()

        # Record telemetry
        with tracer.start_as_current_span("llamaindex.cost_aggregation") as span:
            span.set_attributes(
                {
                    "genops.context_name": self.context_name,
                    "genops.context_id": self.context_id,
                    "genops.operation_id": operation_id,
                    "genops.operation_type": operation_type,
                    "genops.provider": provider,
                    "genops.model": model,
                    "genops.cost_usd": cost,
                    "genops.total_cost": self.total_cost,
                }
            )

        return operation_id

    def _calculate_operation_cost(self, operation: dict[str, Any]) -> float:
        """Calculate cost for an operation based on type and usage."""
        operation_type = operation.get("operation_type", "unknown")
        provider = operation.get("provider", "unknown")
        model = operation.get("model", "unknown")
        tokens = operation.get("tokens_consumed", 0)

        # Use explicit cost if available
        if "cost_usd" in operation and operation["cost_usd"] is not None:
            return operation["cost_usd"]

        cost = 0.0

        if operation_type == "embed":
            # Embedding cost calculation
            if (
                provider in self.provider_pricing
                and model in self.provider_pricing[provider]
            ):
                embedding_rate = self.provider_pricing[provider][model].get(
                    "embedding", self.embedding_cost_per_1k
                )
                cost = (tokens / 1000) * embedding_rate
            else:
                cost = (tokens / 1000) * self.embedding_cost_per_1k

        elif operation_type in ["llm_call", "synthesize"]:
            # LLM generation cost calculation
            if (
                provider in self.provider_pricing
                and model in self.provider_pricing[provider]
            ):
                pricing = self.provider_pricing[provider][model]
                # Assume half input, half output tokens (rough estimate)
                input_tokens = tokens // 2
                output_tokens = tokens - input_tokens
                cost = (input_tokens / 1000) * pricing.get("input", 0.001) + (
                    output_tokens / 1000
                ) * pricing.get("output", 0.002)
            else:
                # Fallback pricing
                cost = (tokens / 1000) * 0.002  # Default $0.002 per 1K tokens

        elif operation_type == "retrieve":
            # Retrieval operation cost
            cost = self.retrieval_cost_per_op

        elif operation_type == "agent_step":
            # Agent step cost (includes tool usage)
            cost = (tokens / 1000) * 0.003  # Slightly higher for agent operations

        return round(cost, 6)

    def get_current_summary(self) -> LlamaIndexCostSummary:
        """
        Get comprehensive cost summary for current operations.

        Returns:
            LlamaIndexCostSummary with detailed breakdown and metrics
        """
        if not self.operations:
            return LlamaIndexCostSummary(total_cost=0.0, operation_count=0)

        # Count operation types
        rag_pipelines = len(
            [op for op in self.operations if op.get("operation_type") == "query"]
        )
        agent_interactions = len(
            [op for op in self.operations if op.get("operation_type") == "agent_step"]
        )

        # Create cost breakdown
        breakdown = RAGCostBreakdown(
            total_cost=self.total_cost,
            embedding_cost=self._cost_by_operation.get("embed", 0.0),
            retrieval_cost=self._cost_by_operation.get("retrieve", 0.0),
            synthesis_cost=self._cost_by_operation.get("llm_call", 0.0)
            + self._cost_by_operation.get("synthesize", 0.0),
            agent_cost=self._cost_by_operation.get("agent_step", 0.0),
            cost_by_provider=dict(self._cost_by_provider),
            cost_by_model=dict(self._cost_by_model),
            cost_by_operation=dict(self._cost_by_operation),
        )

        # Calculate performance metrics
        avg_query_latency = (
            sum(self._query_latencies) / len(self._query_latencies)
            if self._query_latencies
            else 0.0
        )
        avg_retrieval_latency = (
            sum(self._retrieval_latencies) / len(self._retrieval_latencies)
            if self._retrieval_latencies
            else 0.0
        )
        avg_synthesis_latency = (
            sum(self._synthesis_latencies) / len(self._synthesis_latencies)
            if self._synthesis_latencies
            else 0.0
        )

        # Collect providers and models
        unique_providers = {op.get("provider", "unknown") for op in self.operations}
        unique_models = {op.get("model", "unknown") for op in self.operations}

        # Create summary
        summary = LlamaIndexCostSummary(
            total_cost=self.total_cost,
            operation_count=len(self.operations),
            rag_pipelines=rag_pipelines,
            agent_interactions=agent_interactions,
            cost_breakdown=breakdown,
            unique_providers=unique_providers,
            unique_models=unique_models,
            avg_query_latency_ms=avg_query_latency,
            avg_retrieval_latency_ms=avg_retrieval_latency,
            avg_synthesis_latency_ms=avg_synthesis_latency,
        )

        # Add budget information
        if self.budget_limit:
            percentage_used = (self.total_cost / self.budget_limit) * 100
            remaining = self.budget_limit - self.total_cost

            summary.budget_status = {
                "budget_limit": self.budget_limit,
                "percentage_used": percentage_used,
                "remaining_budget": remaining,
                "alerts": [asdict(alert) for alert in self.alerts],
            }

        # Generate optimization suggestions
        breakdown.optimization_suggestions = self._generate_optimization_suggestions()

        # Calculate efficiency metrics
        summary.efficiency_metrics = self._calculate_efficiency_metrics()

        return summary

    def _check_budget_alerts(self):
        """Check budget constraints and generate alerts."""
        if not self.budget_limit:
            return

        percentage_used = (self.total_cost / self.budget_limit) * 100
        remaining = self.budget_limit - self.total_cost

        # Clear previous alerts
        self.alerts = []

        if self.total_cost >= self.budget_limit:
            # Budget exceeded
            self.alerts.append(
                BudgetAlert(
                    alert_type="exceeded",
                    current_cost=self.total_cost,
                    budget_limit=self.budget_limit,
                    percentage_used=percentage_used,
                    remaining_budget=remaining,
                    recommendation="Stop operations immediately - budget exceeded",
                )
            )
        elif percentage_used >= 90:
            # Critical warning (90%+ used)
            self.alerts.append(
                BudgetAlert(
                    alert_type="critical",
                    current_cost=self.total_cost,
                    budget_limit=self.budget_limit,
                    percentage_used=percentage_used,
                    remaining_budget=remaining,
                    recommendation="Approaching budget limit - consider switching to cheaper models",
                )
            )
        elif percentage_used >= 75:
            # Warning (75%+ used)
            self.alerts.append(
                BudgetAlert(
                    alert_type="warning",
                    current_cost=self.total_cost,
                    budget_limit=self.budget_limit,
                    percentage_used=percentage_used,
                    remaining_budget=remaining,
                    recommendation="Budget 75% consumed - monitor remaining operations",
                )
            )

    def _generate_optimization_suggestions(self) -> list[str]:
        """Generate intelligent cost optimization suggestions."""
        suggestions = []

        if not self.operations:
            return suggestions

        # Analyze cost distribution
        total_embedding_cost = self._cost_by_operation.get("embed", 0.0)
        total_synthesis_cost = self._cost_by_operation.get(
            "llm_call", 0.0
        ) + self._cost_by_operation.get("synthesize", 0.0)

        # Embedding optimization
        if total_embedding_cost > self.total_cost * 0.3:  # >30% of costs
            suggestions.append(
                f"Embedding costs are ${total_embedding_cost:.4f} ({total_embedding_cost / self.total_cost * 100:.1f}% of total) - "
                f"consider caching embeddings or using smaller embedding models"
            )

        # Synthesis optimization
        if total_synthesis_cost > self.total_cost * 0.6:  # >60% of costs
            suggestions.append(
                f"LLM synthesis costs are ${total_synthesis_cost:.4f} ({total_synthesis_cost / self.total_cost * 100:.1f}% of total) - "
                f"consider using cheaper models for simpler queries"
            )

        # Provider optimization
        most_expensive_provider = (
            max(self._cost_by_provider.items(), key=lambda x: x[1])
            if self._cost_by_provider
            else None
        )
        if (
            most_expensive_provider
            and most_expensive_provider[1] > self.total_cost * 0.7
        ):
            suggestions.append(
                f"Provider '{most_expensive_provider[0]}' accounts for {most_expensive_provider[1] / self.total_cost * 100:.1f}% of costs - "
                f"consider mixing providers for better cost efficiency"
            )

        # Retrieval efficiency
        retrieval_ops = len(
            [op for op in self.operations if op.get("operation_type") == "retrieve"]
        )
        if retrieval_ops > len(self._query_latencies) * 3:  # Many retrievals per query
            suggestions.append(
                f"High retrieval-to-query ratio ({retrieval_ops}:{len(self._query_latencies)}) - "
                f"consider optimizing retrieval parameters or using hybrid search"
            )

        # Agent efficiency
        agent_cost = self._cost_by_operation.get("agent_step", 0.0)
        if agent_cost > self.total_cost * 0.5:
            suggestions.append(
                f"Agent operations cost ${agent_cost:.4f} ({agent_cost / self.total_cost * 100:.1f}% of total) - "
                f"consider optimizing agent prompts or reducing tool usage"
            )

        return suggestions[:5]  # Limit to top 5 suggestions

    def _calculate_efficiency_metrics(self) -> dict[str, float]:
        """Calculate efficiency metrics for performance optimization."""
        if not self.operations:
            return {}

        metrics = {}

        # Cost per operation type
        for op_type, cost in self._cost_by_operation.items():
            op_count = len(
                [op for op in self.operations if op.get("operation_type") == op_type]
            )
            if op_count > 0:
                metrics[f"avg_cost_per_{op_type}"] = cost / op_count

        # Token efficiency
        total_tokens = sum(op.get("tokens_consumed", 0) for op in self.operations)
        if total_tokens > 0:
            metrics["cost_per_1k_tokens"] = (self.total_cost / total_tokens) * 1000

        # Query efficiency
        if self._query_latencies:
            metrics["avg_cost_per_query"] = self.total_cost / len(self._query_latencies)
            metrics["queries_per_dollar"] = len(self._query_latencies) / max(
                self.total_cost, 0.001
            )

        # Retrieval efficiency
        retrieval_cost = self._cost_by_operation.get("retrieve", 0.0)
        retrieval_count = len(
            [op for op in self.operations if op.get("operation_type") == "retrieve"]
        )
        if retrieval_count > 0:
            metrics["cost_per_retrieval"] = retrieval_cost / retrieval_count

        return metrics

    def get_cost_optimization_recommendation(self) -> dict[str, Any]:
        """Get cost optimization recommendation based on usage patterns."""
        if not self._cost_by_provider:
            return {"recommendation": "No provider data available"}

        # Find most cost-effective provider
        provider_efficiency = {}
        for provider, cost in self._cost_by_provider.items():
            operation_count = sum(
                1 for op in self.operations if op.get("provider") == provider
            )
            if operation_count > 0:
                provider_efficiency[provider] = cost / operation_count

        if provider_efficiency:
            best_provider = min(provider_efficiency.items(), key=lambda x: x[1])
            worst_provider = max(provider_efficiency.items(), key=lambda x: x[1])

            potential_savings = (worst_provider[1] - best_provider[1]) * len(
                self.operations
            )

            return {
                "best_provider": best_provider[0],
                "best_cost_per_operation": best_provider[1],
                "worst_provider": worst_provider[0],
                "worst_cost_per_operation": worst_provider[1],
                "potential_savings": potential_savings,
                "recommendation": f"Switch to {best_provider[0]} for {potential_savings:.4f} USD savings",
            }

        return {"recommendation": "Insufficient data for optimization"}

    def enforce_budget_constraints(
        self, operation_cost: float, customer_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Real-time budget enforcement with automatic cost controls."""
        enforcement_result = {
            "allowed": True,
            "reason": "",
            "alternative_suggestion": None,
            "budget_status": {
                "daily_remaining": self.budget_limit - self.daily_cost
                if self.budget_limit
                else float("inf"),
                "projected_daily_spend": self.daily_cost + operation_cost,
            },
        }

        # Check global daily budget
        if self.budget_limit and (self.daily_cost + operation_cost) > self.budget_limit:
            enforcement_result["allowed"] = False
            enforcement_result["reason"] = (
                f"Operation would exceed daily budget: ${self.daily_cost + operation_cost:.4f} > ${self.budget_limit:.4f}"
            )

            # Suggest cheaper alternatives
            if operation_cost > 0.01:  # Only for significant costs
                cheaper_cost = operation_cost * 0.5  # 50% cost reduction
                if (self.daily_cost + cheaper_cost) <= self.budget_limit:
                    enforcement_result["alternative_suggestion"] = {
                        "action": "use_cheaper_model",
                        "estimated_cost": cheaper_cost,
                        "budget_remaining_after": self.budget_limit
                        - (self.daily_cost + cheaper_cost),
                    }

        # Check customer-specific budget (if provider-specific budgets configured)
        if hasattr(self, "_provider_budgets") and customer_id:
            customer_budget = self._provider_budgets.get(customer_id, float("inf"))
            customer_current = sum(
                op.get("cost_usd", 0)
                for op in self.operations
                if op.get("customer_id") == customer_id
            )

            if (customer_current + operation_cost) > customer_budget:
                enforcement_result["allowed"] = False
                enforcement_result["reason"] = (
                    f"Customer {customer_id} would exceed budget: ${customer_current + operation_cost:.4f} > ${customer_budget:.4f}"
                )

        # Check usage velocity (prevent runaway costs)
        recent_operations = [
            op for op in self.operations if time.time() - op.get("start_time", 0) < 3600
        ]  # Last hour
        hourly_cost = sum(op.get("cost_usd", 0) for op in recent_operations)

        if (
            hourly_cost > (self.budget_limit or 10.0) * 0.1
        ):  # More than 10% of daily budget in 1 hour
            enforcement_result["velocity_warning"] = True
            enforcement_result["hourly_burn_rate"] = hourly_cost

        return enforcement_result

    def optimize_provider_selection(
        self, complexity: str, max_cost: Optional[float] = None
    ) -> dict[str, Any]:
        """Intelligent provider selection based on cost, quality, and performance history."""
        if not hasattr(self, "_tracked_providers"):
            return {"recommendation": "No provider tracking data available"}

        provider_scores = {}

        for provider in self._tracked_providers:
            # Get historical performance for this provider
            provider_ops = [
                op for op in self.operations if op.get("provider") == provider
            ]

            if not provider_ops:
                continue

            avg_cost = sum(op.get("cost_usd", 0) for op in provider_ops) / len(
                provider_ops
            )
            avg_latency = sum(op.get("duration_ms", 0) for op in provider_ops) / len(
                provider_ops
            )
            success_rate = sum(
                1 for op in provider_ops if op.get("success", True)
            ) / len(provider_ops)

            # Quality score based on complexity handling
            complexity_bonus = {
                "high": 0.2 if provider in ["openai", "anthropic"] else 0.0,
                "medium": 0.1,
                "low": 0.0,
            }.get(complexity, 0.0)

            # Calculate composite score (higher is better)
            cost_score = max(0, 1 - (avg_cost / 0.1))  # Normalize to $0.1 baseline
            latency_score = max(0, 1 - (avg_latency / 5000))  # Normalize to 5s baseline
            quality_score = success_rate + complexity_bonus

            composite_score = (
                cost_score * 0.4 + latency_score * 0.3 + quality_score * 0.3
            )

            # Apply cost constraint if specified
            if max_cost and avg_cost > max_cost:
                composite_score *= 0.1  # Heavily penalize over-budget providers

            provider_scores[provider] = {
                "composite_score": composite_score,
                "avg_cost": avg_cost,
                "avg_latency_ms": avg_latency,
                "success_rate": success_rate,
                "total_operations": len(provider_ops),
            }

        if not provider_scores:
            return {"recommendation": "No provider performance data available"}

        best_provider = max(
            provider_scores.items(), key=lambda x: x[1]["composite_score"]
        )

        return {
            "recommended_provider": best_provider[0],
            "provider_scores": provider_scores,
            "reasoning": {
                "cost_efficiency": best_provider[1]["avg_cost"],
                "performance": f"{best_provider[1]['avg_latency_ms']:.0f}ms avg",
                "reliability": f"{best_provider[1]['success_rate']:.1%} success rate",
                "experience": f"{best_provider[1]['total_operations']} operations",
            },
        }

    def implement_cost_circuit_breaker(
        self, cost_threshold: float, time_window_seconds: int = 3600
    ) -> dict[str, Any]:
        """Implement circuit breaker pattern for cost control."""
        current_time = time.time()
        window_start = current_time - time_window_seconds

        # Get operations in time window
        recent_operations = [
            op for op in self.operations if op.get("start_time", 0) >= window_start
        ]

        window_cost = sum(op.get("cost_usd", 0) for op in recent_operations)

        circuit_status = {
            "is_open": window_cost >= cost_threshold,
            "current_cost": window_cost,
            "cost_threshold": cost_threshold,
            "time_window_seconds": time_window_seconds,
            "operations_count": len(recent_operations),
            "time_until_reset": max(
                0,
                time_window_seconds
                - (
                    current_time
                    - min(
                        op.get("start_time", current_time) for op in recent_operations
                    )
                    if recent_operations
                    else 0
                ),
            ),
        }

        if circuit_status["is_open"]:
            circuit_status["action"] = "BLOCK_NEW_OPERATIONS"
            circuit_status["message"] = (
                f"Cost circuit breaker open: ${window_cost:.4f} >= ${cost_threshold:.4f} in {time_window_seconds}s window"
            )
        else:
            remaining_budget = cost_threshold - window_cost
            circuit_status["action"] = "ALLOW_OPERATIONS"
            circuit_status["message"] = (
                f"Circuit breaker closed: ${remaining_budget:.4f} budget remaining"
            )

        return circuit_status

    def generate_cost_forecast(self, days_ahead: int = 7) -> dict[str, Any]:
        """Generate cost forecasting based on historical usage patterns."""
        if len(self.operations) < 10:  # Need minimum data for forecasting
            return {
                "forecast": "Insufficient data for forecasting (minimum 10 operations required)"
            }

        # Calculate daily averages
        daily_costs = defaultdict(float)
        daily_operations = defaultdict(int)

        for op in self.operations:
            operation_date = datetime.fromtimestamp(
                op.get("start_time", time.time())
            ).date()
            daily_costs[operation_date] += op.get("cost_usd", 0)
            daily_operations[operation_date] += 1

        if not daily_costs:
            return {"forecast": "No historical cost data available"}

        # Simple forecasting based on recent trends
        recent_days = sorted(daily_costs.keys())[-7:]  # Last 7 days
        avg_daily_cost = sum(daily_costs[day] for day in recent_days) / len(recent_days)
        avg_daily_operations = sum(daily_operations[day] for day in recent_days) / len(
            recent_days
        )

        # Calculate trend (simple linear)
        if len(recent_days) >= 3:
            early_avg = sum(daily_costs[day] for day in recent_days[:3]) / 3
            late_avg = sum(daily_costs[day] for day in recent_days[-3:]) / 3
            trend_factor = late_avg / early_avg if early_avg > 0 else 1.0
        else:
            trend_factor = 1.0

        # Generate forecast
        forecast_data = {
            "forecast_period_days": days_ahead,
            "avg_daily_cost": avg_daily_cost,
            "avg_daily_operations": avg_daily_operations,
            "trend_factor": trend_factor,
            "daily_forecasts": [],
            "total_forecast_cost": 0.0,
        }

        base_date = datetime.now().date()
        for i in range(1, days_ahead + 1):
            forecast_date = base_date + timedelta(days=i)

            # Apply trend with some smoothing
            trend_multiplier = 1.0 + (trend_factor - 1.0) * (i / days_ahead) * 0.5
            daily_forecast = avg_daily_cost * trend_multiplier

            forecast_data["daily_forecasts"].append(
                {
                    "date": forecast_date.isoformat(),
                    "forecast_cost": daily_forecast,
                    "forecast_operations": int(avg_daily_operations * trend_multiplier),
                }
            )

            forecast_data["total_forecast_cost"] += daily_forecast

        # Add budget impact analysis
        if self.budget_limit:
            days_until_budget_exceeded = None
            cumulative_cost = 0

            for i, day_forecast in enumerate(forecast_data["daily_forecasts"]):
                cumulative_cost += day_forecast["forecast_cost"]
                if (
                    cumulative_cost > self.budget_limit
                    and days_until_budget_exceeded is None
                ):
                    days_until_budget_exceeded = i + 1

            forecast_data["budget_analysis"] = {
                "current_budget": self.budget_limit,
                "days_until_budget_exceeded": days_until_budget_exceeded,
                "budget_utilization_at_end": (
                    forecast_data["total_forecast_cost"] / self.budget_limit
                )
                * 100
                if self.budget_limit
                else 0,
            }

        return forecast_data

    def export_detailed_report(self) -> dict[str, Any]:
        """Export detailed cost and performance report."""
        summary = self.get_current_summary()

        return {
            "context_info": {
                "name": self.context_name,
                "id": self.context_id,
                "start_time": self.start_time,
                "duration_seconds": time.time() - self.start_time,
                "budget_limit": self.budget_limit,
            },
            "cost_summary": asdict(summary),
            "operations": self.operations,
            "performance_analysis": {
                "query_latencies": self._query_latencies,
                "retrieval_latencies": self._retrieval_latencies,
                "synthesis_latencies": self._synthesis_latencies,
            },
            "governance_context": self.governance_defaults,
        }


def multi_provider_cost_tracking(
    providers: Optional[list[str]] = None,
    budget_per_provider: Optional[dict[str, float]] = None,
    enable_cost_optimization: bool = True,
    **kwargs,
) -> LlamaIndexCostAggregator:
    """
    Create unified cost tracking across multiple AI providers.

    Args:
        providers: List of provider names to track (e.g., ['openai', 'anthropic', 'google'])
        budget_per_provider: Budget limits per provider
        enable_cost_optimization: Enable automatic cost optimization recommendations
        **kwargs: Additional governance attributes

    Returns:
        LlamaIndexCostAggregator configured for multi-provider tracking

    Example:
        tracker = multi_provider_cost_tracking(
            providers=['openai', 'anthropic', 'google'],
            budget_per_provider={'openai': 10.0, 'anthropic': 15.0, 'google': 5.0},
            team="ai-research",
            project="multi-provider-rag"
        )

        # Use with different providers
        tracker.add_synthesis_cost("openai", "gpt-4", 1000, 500, 0.045)
        tracker.add_synthesis_cost("anthropic", "claude-3", 1000, 500, 0.015)

        # Get cross-provider analysis
        summary = tracker.get_current_summary()
        print(f"Best value provider: {tracker.get_cost_optimization_recommendation()}")
    """
    if providers is None:
        providers = ["openai", "anthropic", "google", "cohere"]

    # Calculate total budget
    total_budget = None
    if budget_per_provider:
        total_budget = sum(budget_per_provider.values())

    aggregator = LlamaIndexCostAggregator(
        context_name="multi_provider_tracking",
        budget_limit=total_budget,
        enable_alerts=True,
        **kwargs,
    )

    # Configure multi-provider settings
    aggregator._provider_budgets = budget_per_provider or {}
    aggregator._tracked_providers = set(providers)
    aggregator._enable_cost_optimization = enable_cost_optimization

    # Add real-time budget enforcement methods
    def add_operation_with_enforcement(operation_data: dict[str, Any]) -> str:
        """Add operation with real-time budget enforcement."""
        operation_cost = operation_data.get("cost_usd", 0.0)
        customer_id = operation_data.get("customer_id")

        # Check budget constraints
        enforcement = aggregator.enforce_budget_constraints(operation_cost, customer_id)

        if not enforcement["allowed"]:
            logger.warning(
                f"Operation blocked by budget enforcement: {enforcement['reason']}"
            )
            raise ValueError(f"Budget constraint violation: {enforcement['reason']}")

        if enforcement.get("velocity_warning"):
            logger.warning(
                f"High cost velocity detected: ${enforcement['hourly_burn_rate']:.4f}/hour"
            )

        return aggregator.add_llamaindex_operation(operation_data)

    # Replace the standard method with the enforcing version
    aggregator.add_llamaindex_operation_with_enforcement = (
        add_operation_with_enforcement
    )

    return aggregator


@contextmanager
def create_llamaindex_cost_context(
    context_name: str,
    budget_limit: Optional[float] = None,
    enable_alerts: bool = True,
    **kwargs,
) -> Iterator[LlamaIndexCostAggregator]:
    """
    Create a cost tracking context for LlamaIndex operations.

    Args:
        context_name: Descriptive name for the RAG/agent workflow
        budget_limit: Maximum allowed cost in USD
        enable_alerts: Enable budget monitoring alerts
        **kwargs: Additional configuration options

    Yields:
        LlamaIndexCostAggregator instance for tracking operations

    Example:
        with create_llamaindex_cost_context("rag_pipeline", budget_limit=5.0) as context:
            # Query operations
            context.add_llamaindex_operation({
                'operation_type': 'query',
                'provider': 'openai',
                'model': 'gpt-4',
                'tokens_consumed': 1500,
                'cost_usd': 0.045
            })

            # Get final summary
            summary = context.get_current_summary()
            print(f"Total RAG pipeline cost: ${summary.total_cost:.4f}")
    """

    # Create aggregator
    aggregator = LlamaIndexCostAggregator(
        context_name=context_name,
        budget_limit=budget_limit,
        enable_alerts=enable_alerts,
        **kwargs,
    )

    with tracer.start_as_current_span(
        "llamaindex.cost_context",
        attributes={
            "genops.context_name": context_name,
            "genops.context_id": aggregator.context_id,
            "genops.budget_limit": budget_limit or 0,
        },
    ) as span:
        try:
            yield aggregator

            # Record final metrics
            final_summary = aggregator.get_current_summary()
            span.set_attributes(
                {
                    "genops.total_cost": final_summary.total_cost,
                    "genops.operation_count": final_summary.operation_count,
                    "genops.rag_pipelines": final_summary.rag_pipelines,
                    "genops.agent_interactions": final_summary.agent_interactions,
                    "genops.success": True,
                }
            )

            # Log completion
            logger.info(
                f"LlamaIndex cost context '{context_name}' completed: "
                f"${final_summary.total_cost:.4f} across {final_summary.operation_count} operations"
            )

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(f"Error in LlamaIndex cost context '{context_name}': {e}")
            raise


# Global cost aggregator instance
_current_aggregator: Optional[LlamaIndexCostAggregator] = None


def get_cost_aggregator() -> Optional[LlamaIndexCostAggregator]:
    """Get the current cost aggregator instance."""
    return _current_aggregator


def set_cost_aggregator(aggregator: LlamaIndexCostAggregator) -> None:
    """Set the current cost aggregator instance."""
    global _current_aggregator
    _current_aggregator = aggregator


# CLAUDE.md compliant aliases for API consistency
create_chain_cost_context = create_llamaindex_cost_context  # Standard naming alias


# Export main classes and functions
__all__ = [
    "LlamaIndexCostAggregator",
    "create_llamaindex_cost_context",
    "create_chain_cost_context",  # CLAUDE.md standard alias
    "multi_provider_cost_tracking",  # CLAUDE.md standard function
    "RAGCostBreakdown",
    "LlamaIndexCostSummary",
    "BudgetAlert",
    "get_cost_aggregator",
    "set_cost_aggregator",
]
