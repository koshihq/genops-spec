#!/usr/bin/env python3
"""
GenOps Mistral AI Cost Aggregator

This module provides advanced cost analytics and aggregation for Mistral AI operations.
It enables enterprise-grade cost intelligence, optimization insights, and governance
for European AI workloads with comprehensive reporting capabilities.

Features:
- Real-time cost aggregation across all Mistral operations
- Time-based cost analysis with trend detection
- Team and project cost attribution with detailed breakdowns
- European AI provider cost optimization insights
- Budget tracking and alerting with governance controls
- Performance vs cost efficiency analysis
- Multi-dimensional cost analytics for enterprise reporting

Usage:
    from genops.providers.mistral_cost_aggregator import MistralCostAggregator
    
    aggregator = MistralCostAggregator()
    aggregator.record_operation("mistral-small-latest", "chat", cost_breakdown, team="ai-team")
    
    summary = aggregator.get_cost_summary()
    insights = aggregator.get_cost_optimization_insights()
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class TimeWindow(Enum):
    """Time window options for cost analysis."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

class CostCategory(Enum):
    """Cost categorization for analysis."""
    COMPUTE = "compute"
    TOKENS = "tokens"
    OPERATIONS = "operations"
    TOTAL = "total"

@dataclass
class OperationRecord:
    """Individual operation record for cost tracking."""
    timestamp: float
    model: str
    operation_type: str  # chat, embed, completion
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    request_time: float

    # Governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    environment: str = "development"
    session_id: Optional[str] = None
    operation_id: Optional[str] = None

    # Performance metrics
    tokens_per_second: float = 0.0
    cost_per_token: float = 0.0
    efficiency_score: float = 1.0  # Relative performance metric

@dataclass
class CostSummary:
    """Comprehensive cost summary with multiple dimensions."""
    total_cost: float
    total_operations: int
    total_tokens: int
    average_cost_per_operation: float
    average_cost_per_token: float
    time_window: str

    # Cost breakdowns
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_operation: Dict[str, float] = field(default_factory=dict)
    cost_by_team: Dict[str, float] = field(default_factory=dict)
    cost_by_project: Dict[str, float] = field(default_factory=dict)
    cost_by_customer: Dict[str, float] = field(default_factory=dict)
    cost_by_environment: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    total_request_time: float = 0.0
    average_tokens_per_second: float = 0.0
    efficiency_trends: Dict[str, float] = field(default_factory=dict)

    # Time-based analysis
    cost_trend: List[Tuple[str, float]] = field(default_factory=list)
    peak_usage_times: List[str] = field(default_factory=list)

    # European AI advantages
    gdpr_compliance_cost_savings: float = 0.0
    eu_data_residency_value: float = 0.0

@dataclass
class OptimizationInsight:
    """Cost optimization insight with actionable recommendations."""
    category: str  # model_selection, token_efficiency, usage_patterns, european_advantages
    priority: str  # high, medium, low
    insight: str
    potential_savings_usd: float
    potential_savings_percent: float
    recommended_actions: List[str]
    confidence_score: float  # 0.0 to 1.0
    implementation_effort: str  # low, medium, high

    # Supporting data
    current_state: Dict[str, Any] = field(default_factory=dict)
    target_state: Dict[str, Any] = field(default_factory=dict)
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)

class MistralCostAggregator:
    """Advanced cost aggregation and analytics for Mistral AI operations."""

    def __init__(
        self,
        retention_days: int = 90,
        enable_real_time_alerts: bool = True,
        cost_alert_threshold: float = 100.0,  # USD
        efficiency_alert_threshold: float = 0.7  # Below 70% efficiency
    ):
        """
        Initialize cost aggregator with configuration.
        
        Args:
            retention_days: How long to keep detailed records
            enable_real_time_alerts: Whether to enable cost alerts
            cost_alert_threshold: Cost threshold for alerts (USD)
            efficiency_alert_threshold: Efficiency threshold for alerts
        """
        self.retention_days = retention_days
        self.enable_real_time_alerts = enable_real_time_alerts
        self.cost_alert_threshold = cost_alert_threshold
        self.efficiency_alert_threshold = efficiency_alert_threshold

        # Storage for operation records
        self.operations: List[OperationRecord] = []

        # Real-time aggregations
        self.current_session_cost = 0.0
        self.current_session_operations = 0
        self.session_start_time = time.time()

        # Cost tracking by dimensions
        self.cost_by_model = defaultdict(float)
        self.cost_by_team = defaultdict(float)
        self.cost_by_project = defaultdict(float)
        self.cost_by_customer = defaultdict(float)
        self.cost_by_environment = defaultdict(float)
        self.cost_by_operation_type = defaultdict(float)

        # Performance tracking
        self.token_efficiency_history = []
        self.cost_efficiency_history = []

        # Budgets and limits
        self.team_budgets: Dict[str, float] = {}
        self.project_budgets: Dict[str, float] = {}
        self.customer_budgets: Dict[str, float] = {}

        logger.info("Mistral cost aggregator initialized")

    def record_operation(
        self,
        model: str,
        operation_type: str,
        cost_breakdown: Dict[str, Any],
        performance_metrics: Optional[Dict[str, Any]] = None,
        **governance_attrs
    ) -> str:
        """
        Record a Mistral operation for cost tracking and analysis.
        
        Args:
            model: Mistral model used
            operation_type: Type of operation (chat, embed, completion)
            cost_breakdown: Cost information
            performance_metrics: Performance data
            **governance_attrs: Governance attributes (team, project, customer_id, etc.)
            
        Returns:
            Operation ID for reference
        """
        current_time = time.time()

        # Create operation record
        record = OperationRecord(
            timestamp=current_time,
            model=model,
            operation_type=operation_type,
            input_tokens=cost_breakdown.get('input_tokens', 0),
            output_tokens=cost_breakdown.get('output_tokens', 0),
            total_tokens=cost_breakdown.get('total_tokens', 0),
            input_cost=cost_breakdown.get('input_cost', 0.0),
            output_cost=cost_breakdown.get('output_cost', 0.0),
            total_cost=cost_breakdown.get('total_cost', 0.0),
            request_time=performance_metrics.get('request_time', 0.0) if performance_metrics else 0.0,

            # Governance
            team=governance_attrs.get('team'),
            project=governance_attrs.get('project'),
            customer_id=governance_attrs.get('customer_id'),
            environment=governance_attrs.get('environment', 'development'),
            session_id=governance_attrs.get('session_id'),
            operation_id=governance_attrs.get('operation_id'),

            # Performance
            tokens_per_second=performance_metrics.get('tokens_per_second', 0.0) if performance_metrics else 0.0,
            cost_per_token=cost_breakdown.get('cost_per_token', 0.0),
            efficiency_score=performance_metrics.get('efficiency_score', 1.0) if performance_metrics else 1.0
        )

        # Store the record
        self.operations.append(record)

        # Update real-time aggregations
        self._update_real_time_aggregations(record)

        # Check for alerts
        if self.enable_real_time_alerts:
            self._check_alerts(record)

        # Cleanup old records
        self._cleanup_old_records()

        return record.operation_id or f"op_{int(current_time * 1000)}"

    def _update_real_time_aggregations(self, record: OperationRecord):
        """Update real-time cost aggregations."""
        self.current_session_cost += record.total_cost
        self.current_session_operations += 1

        # Update dimensional aggregations
        self.cost_by_model[record.model] += record.total_cost
        self.cost_by_operation_type[record.operation_type] += record.total_cost

        if record.team:
            self.cost_by_team[record.team] += record.total_cost
        if record.project:
            self.cost_by_project[record.project] += record.total_cost
        if record.customer_id:
            self.cost_by_customer[record.customer_id] += record.total_cost

        self.cost_by_environment[record.environment] += record.total_cost

        # Track efficiency
        if record.tokens_per_second > 0:
            self.token_efficiency_history.append((record.timestamp, record.tokens_per_second))

        if record.cost_per_token > 0:
            self.cost_efficiency_history.append((record.timestamp, record.cost_per_token))

    def _check_alerts(self, record: OperationRecord):
        """Check for cost and efficiency alerts."""
        # Cost threshold alerts
        if record.total_cost > self.cost_alert_threshold / 100:  # Per operation threshold
            logger.warning(f"High-cost operation detected: ${record.total_cost:.6f} for {record.model}")

        # Session cost alerts
        if self.current_session_cost > self.cost_alert_threshold:
            logger.warning(f"Session cost threshold exceeded: ${self.current_session_cost:.2f}")

        # Team budget alerts
        if record.team and record.team in self.team_budgets:
            team_cost = self.cost_by_team[record.team]
            team_budget = self.team_budgets[record.team]
            if team_cost > team_budget * 0.8:  # 80% of budget
                logger.warning(f"Team {record.team} approaching budget: ${team_cost:.2f}/${team_budget:.2f}")

        # Efficiency alerts
        if record.efficiency_score < self.efficiency_alert_threshold:
            logger.warning(f"Low efficiency operation: {record.efficiency_score:.2f} for {record.model}")

    def _cleanup_old_records(self):
        """Remove old records beyond retention period."""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        self.operations = [op for op in self.operations if op.timestamp > cutoff_time]

    def get_cost_summary(
        self,
        time_window: TimeWindow = TimeWindow.DAY,
        team: Optional[str] = None,
        project: Optional[str] = None,
        customer_id: Optional[str] = None
    ) -> CostSummary:
        """
        Get comprehensive cost summary for specified time window and filters.
        
        Args:
            time_window: Time window for analysis
            team: Filter by team
            project: Filter by project
            customer_id: Filter by customer
            
        Returns:
            Comprehensive cost summary
        """
        # Calculate time window
        current_time = time.time()
        window_seconds = self._get_time_window_seconds(time_window)
        start_time = current_time - window_seconds

        # Filter operations
        filtered_ops = [
            op for op in self.operations
            if op.timestamp >= start_time and
            (not team or op.team == team) and
            (not project or op.project == project) and
            (not customer_id or op.customer_id == customer_id)
        ]

        if not filtered_ops:
            return CostSummary(
                total_cost=0.0,
                total_operations=0,
                total_tokens=0,
                average_cost_per_operation=0.0,
                average_cost_per_token=0.0,
                time_window=time_window.value
            )

        # Calculate aggregated metrics
        total_cost = sum(op.total_cost for op in filtered_ops)
        total_operations = len(filtered_ops)
        total_tokens = sum(op.total_tokens for op in filtered_ops)
        total_request_time = sum(op.request_time for op in filtered_ops)

        avg_cost_per_op = total_cost / max(total_operations, 1)
        avg_cost_per_token = total_cost / max(total_tokens, 1)
        avg_tokens_per_sec = sum(op.tokens_per_second for op in filtered_ops) / max(total_operations, 1)

        # Build dimensional breakdowns
        cost_by_model = defaultdict(float)
        cost_by_operation = defaultdict(float)
        cost_by_team = defaultdict(float)
        cost_by_project = defaultdict(float)
        cost_by_customer = defaultdict(float)
        cost_by_environment = defaultdict(float)

        for op in filtered_ops:
            cost_by_model[op.model] += op.total_cost
            cost_by_operation[op.operation_type] += op.total_cost
            if op.team:
                cost_by_team[op.team] += op.total_cost
            if op.project:
                cost_by_project[op.project] += op.total_cost
            if op.customer_id:
                cost_by_customer[op.customer_id] += op.total_cost
            cost_by_environment[op.environment] += op.total_cost

        # Calculate cost trend
        cost_trend = self._calculate_cost_trend(filtered_ops, time_window)

        # Estimate European AI advantages
        gdpr_savings = total_cost * 0.1  # 10% estimated compliance cost savings
        eu_residency_value = total_cost * 0.05  # 5% estimated data residency value

        return CostSummary(
            total_cost=total_cost,
            total_operations=total_operations,
            total_tokens=total_tokens,
            average_cost_per_operation=avg_cost_per_op,
            average_cost_per_token=avg_cost_per_token,
            time_window=time_window.value,
            cost_by_model=dict(cost_by_model),
            cost_by_operation=dict(cost_by_operation),
            cost_by_team=dict(cost_by_team),
            cost_by_project=dict(cost_by_project),
            cost_by_customer=dict(cost_by_customer),
            cost_by_environment=dict(cost_by_environment),
            total_request_time=total_request_time,
            average_tokens_per_second=avg_tokens_per_sec,
            cost_trend=cost_trend,
            gdpr_compliance_cost_savings=gdpr_savings,
            eu_data_residency_value=eu_residency_value
        )

    def _get_time_window_seconds(self, window: TimeWindow) -> int:
        """Convert time window enum to seconds."""
        window_map = {
            TimeWindow.HOUR: 3600,
            TimeWindow.DAY: 86400,
            TimeWindow.WEEK: 604800,
            TimeWindow.MONTH: 2592000,  # 30 days
            TimeWindow.YEAR: 31536000   # 365 days
        }
        return window_map.get(window, 86400)  # Default to day

    def _calculate_cost_trend(
        self,
        operations: List[OperationRecord],
        time_window: TimeWindow
    ) -> List[Tuple[str, float]]:
        """Calculate cost trend over time."""
        if not operations:
            return []

        # Group operations by time buckets
        bucket_size = self._get_time_window_seconds(time_window) // 10  # 10 data points
        cost_by_bucket = defaultdict(float)

        min_time = min(op.timestamp for op in operations)

        for op in operations:
            bucket = int((op.timestamp - min_time) // bucket_size)
            cost_by_bucket[bucket] += op.total_cost

        # Convert to time series
        trend = []
        for bucket in sorted(cost_by_bucket.keys()):
            timestamp = min_time + (bucket * bucket_size)
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M' if time_window == TimeWindow.HOUR else '%m-%d')
            trend.append((time_str, cost_by_bucket[bucket]))

        return trend

    def get_cost_optimization_insights(
        self,
        min_savings_threshold: float = 5.0,  # Minimum $5 savings to recommend
        lookback_days: int = 7
    ) -> List[OptimizationInsight]:
        """
        Generate cost optimization insights based on usage patterns.
        
        Args:
            min_savings_threshold: Minimum savings to generate insights
            lookback_days: Days to analyze for patterns
            
        Returns:
            List of optimization insights with recommendations
        """
        insights = []

        # Get recent operations for analysis
        cutoff_time = time.time() - (lookback_days * 24 * 3600)
        recent_ops = [op for op in self.operations if op.timestamp >= cutoff_time]

        if not recent_ops:
            return insights

        # Analyze model usage patterns
        insights.extend(self._analyze_model_optimization(recent_ops, min_savings_threshold))

        # Analyze token efficiency
        insights.extend(self._analyze_token_efficiency(recent_ops, min_savings_threshold))

        # Analyze usage patterns
        insights.extend(self._analyze_usage_patterns(recent_ops, min_savings_threshold))

        # European AI advantages
        insights.extend(self._analyze_european_advantages(recent_ops, min_savings_threshold))

        # Sort by potential savings (highest first)
        insights.sort(key=lambda x: x.potential_savings_usd, reverse=True)

        return insights

    def _analyze_model_optimization(
        self,
        operations: List[OperationRecord],
        min_savings: float
    ) -> List[OptimizationInsight]:
        """Analyze model selection for cost optimization."""
        insights = []

        # Group by model and calculate stats
        model_stats = defaultdict(lambda: {"cost": 0.0, "operations": 0, "tokens": 0})

        for op in operations:
            stats = model_stats[op.model]
            stats["cost"] += op.total_cost
            stats["operations"] += 1
            stats["tokens"] += op.total_tokens

        # Find expensive models with alternatives
        total_cost = sum(stats["cost"] for stats in model_stats.values())

        for model, stats in model_stats.items():
            if stats["cost"] < total_cost * 0.1:  # Skip models with <10% of total cost
                continue

            # Check for more cost-effective alternatives
            avg_tokens_per_op = stats["tokens"] / max(stats["operations"], 1)

            # Suggest cheaper alternatives based on usage patterns
            if model == "mistral-large-latest" and avg_tokens_per_op < 2000:
                potential_savings = stats["cost"] * 0.6  # ~60% savings with medium
                if potential_savings >= min_savings:
                    insights.append(OptimizationInsight(
                        category="model_selection",
                        priority="high",
                        insight=f"Switch from {model} to mistral-medium-latest for simple tasks",
                        potential_savings_usd=potential_savings,
                        potential_savings_percent=60.0,
                        recommended_actions=[
                            "Test mistral-medium-latest for your use cases",
                            "Implement model selection logic based on task complexity",
                            "Monitor quality metrics during transition"
                        ],
                        confidence_score=0.8,
                        implementation_effort="medium",
                        current_state={"model": model, "cost": stats["cost"]},
                        target_state={"model": "mistral-medium-latest", "estimated_cost": stats["cost"] * 0.4}
                    ))

            elif model == "mistral-medium-latest" and avg_tokens_per_op < 1000:
                potential_savings = stats["cost"] * 0.4  # ~40% savings with small
                if potential_savings >= min_savings:
                    insights.append(OptimizationInsight(
                        category="model_selection",
                        priority="medium",
                        insight="Consider mistral-small-latest for simple queries",
                        potential_savings_usd=potential_savings,
                        potential_savings_percent=40.0,
                        recommended_actions=[
                            "Analyze query complexity distribution",
                            "A/B test with mistral-small-latest for simple tasks",
                            "Implement tiered model selection"
                        ],
                        confidence_score=0.7,
                        implementation_effort="low",
                        current_state={"model": model, "avg_tokens": avg_tokens_per_op},
                        target_state={"model": "mistral-small-latest", "complexity": "simple"}
                    ))

        return insights

    def _analyze_token_efficiency(
        self,
        operations: List[OperationRecord],
        min_savings: float
    ) -> List[OptimizationInsight]:
        """Analyze token usage efficiency."""
        insights = []

        # Calculate token efficiency metrics
        high_output_ops = [op for op in operations if op.output_tokens > op.input_tokens * 2]

        if len(high_output_ops) > len(operations) * 0.2:  # >20% of operations
            output_cost_waste = sum(op.output_cost * 0.3 for op in high_output_ops)  # 30% potential reduction

            if output_cost_waste >= min_savings:
                insights.append(OptimizationInsight(
                    category="token_efficiency",
                    priority="medium",
                    insight=f"{len(high_output_ops)} operations have high output/input ratio",
                    potential_savings_usd=output_cost_waste,
                    potential_savings_percent=30.0,
                    recommended_actions=[
                        "Implement max_tokens limits for simple queries",
                        "Use more specific prompts to reduce output length",
                        "Consider response length requirements by use case"
                    ],
                    confidence_score=0.9,
                    implementation_effort="low",
                    supporting_metrics={
                        "high_output_operations": len(high_output_ops),
                        "avg_output_input_ratio": sum(op.output_tokens / max(op.input_tokens, 1) for op in high_output_ops) / len(high_output_ops)
                    }
                ))

        return insights

    def _analyze_usage_patterns(
        self,
        operations: List[OperationRecord],
        min_savings: float
    ) -> List[OptimizationInsight]:
        """Analyze usage patterns for optimization opportunities."""
        insights = []

        # Analyze time-based usage patterns
        usage_by_hour = defaultdict(int)
        for op in operations:
            hour = datetime.fromtimestamp(op.timestamp).hour
            usage_by_hour[hour] += 1

        # Find peak usage times
        if usage_by_hour:
            peak_hours = [hour for hour, count in usage_by_hour.items()
                         if count > sum(usage_by_hour.values()) / len(usage_by_hour) * 1.5]

            if len(peak_hours) < 8:  # Concentrated usage
                # Suggest batch processing for cost optimization
                batch_savings = sum(op.total_cost for op in operations) * 0.15  # 15% batch discount

                if batch_savings >= min_savings:
                    insights.append(OptimizationInsight(
                        category="usage_patterns",
                        priority="low",
                        insight="Usage concentrated in specific hours - batch processing could reduce costs",
                        potential_savings_usd=batch_savings,
                        potential_savings_percent=15.0,
                        recommended_actions=[
                            "Consider batching non-urgent requests",
                            "Negotiate volume pricing with Mistral",
                            "Implement request queuing for off-peak processing"
                        ],
                        confidence_score=0.6,
                        implementation_effort="high",
                        supporting_metrics={"peak_hours": peak_hours, "concentration_ratio": len(peak_hours) / 24}
                    ))

        return insights

    def _analyze_european_advantages(
        self,
        operations: List[OperationRecord],
        min_savings: float
    ) -> List[OptimizationInsight]:
        """Analyze European AI provider advantages."""
        insights = []

        total_cost = sum(op.total_cost for op in operations)

        if total_cost > min_savings:
            # GDPR compliance savings
            gdpr_savings = total_cost * 0.1  # 10% estimated compliance cost savings

            insights.append(OptimizationInsight(
                category="european_advantages",
                priority="high",
                insight="Mistral provides GDPR-compliant AI with EU data residency",
                potential_savings_usd=gdpr_savings,
                potential_savings_percent=10.0,
                recommended_actions=[
                    "Leverage EU data residency for compliance requirements",
                    "Avoid cross-border data transfer costs and complexity",
                    "Highlight GDPR compliance in data governance reports"
                ],
                confidence_score=0.9,
                implementation_effort="low",
                supporting_metrics={
                    "gdpr_compliance_value": gdpr_savings,
                    "data_residency": "EU",
                    "regulatory_benefits": "GDPR compliant"
                }
            ))

            # Cost competitiveness vs US providers
            if total_cost > 50.0:  # For significant workloads
                competitive_savings = total_cost * 0.2  # 20% competitive advantage

                insights.append(OptimizationInsight(
                    category="european_advantages",
                    priority="medium",
                    insight="Mistral offers cost-competitive European AI alternative to US providers",
                    potential_savings_usd=competitive_savings,
                    potential_savings_percent=20.0,
                    recommended_actions=[
                        "Compare costs with OpenAI/Anthropic for similar workloads",
                        "Factor in data sovereignty and regulatory benefits",
                        "Consider Mistral for European customer-facing applications"
                    ],
                    confidence_score=0.7,
                    implementation_effort="medium",
                    supporting_metrics={"cost_competitiveness": "vs_us_providers", "market_position": "european_ai"}
                ))

        return insights

    def set_budget(self, budget_type: str, identifier: str, amount: float):
        """Set budget limits for teams, projects, or customers."""
        if budget_type == "team":
            self.team_budgets[identifier] = amount
        elif budget_type == "project":
            self.project_budgets[identifier] = amount
        elif budget_type == "customer":
            self.customer_budgets[identifier] = amount

        logger.info(f"Budget set for {budget_type} '{identifier}': ${amount:.2f}")

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget utilization status."""
        status = {
            "teams": {},
            "projects": {},
            "customers": {}
        }

        # Team budget status
        for team, budget in self.team_budgets.items():
            current_cost = self.cost_by_team.get(team, 0.0)
            status["teams"][team] = {
                "budget": budget,
                "spent": current_cost,
                "remaining": budget - current_cost,
                "utilization_percent": (current_cost / budget) * 100 if budget > 0 else 0
            }

        # Project budget status
        for project, budget in self.project_budgets.items():
            current_cost = self.cost_by_project.get(project, 0.0)
            status["projects"][project] = {
                "budget": budget,
                "spent": current_cost,
                "remaining": budget - current_cost,
                "utilization_percent": (current_cost / budget) * 100 if budget > 0 else 0
            }

        # Customer budget status
        for customer, budget in self.customer_budgets.items():
            current_cost = self.cost_by_customer.get(customer, 0.0)
            status["customers"][customer] = {
                "budget": budget,
                "spent": current_cost,
                "remaining": budget - current_cost,
                "utilization_percent": (current_cost / budget) * 100 if budget > 0 else 0
            }

        return status

    def export_analytics_data(self, format: str = "json") -> str:
        """Export analytics data for external reporting."""
        data = {
            "summary": {
                "total_operations": len(self.operations),
                "total_cost": sum(op.total_cost for op in self.operations),
                "session_cost": self.current_session_cost,
                "session_operations": self.current_session_operations
            },
            "cost_breakdowns": {
                "by_model": dict(self.cost_by_model),
                "by_team": dict(self.cost_by_team),
                "by_project": dict(self.cost_by_project),
                "by_customer": dict(self.cost_by_customer),
                "by_environment": dict(self.cost_by_environment),
                "by_operation_type": dict(self.cost_by_operation_type)
            },
            "budget_status": self.get_budget_status(),
            "metadata": {
                "retention_days": self.retention_days,
                "export_timestamp": datetime.now().isoformat(),
                "total_records": len(self.operations)
            }
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            return str(data)  # Basic string representation

    def reset_session(self):
        """Reset current session statistics."""
        self.current_session_cost = 0.0
        self.current_session_operations = 0
        self.session_start_time = time.time()
        logger.info("Cost aggregator session reset")

# Convenience functions
def create_mistral_cost_aggregator(**kwargs) -> MistralCostAggregator:
    """Create a new Mistral cost aggregator with configuration."""
    return MistralCostAggregator(**kwargs)

if __name__ == "__main__":
    # Demo and testing
    print("Mistral AI Cost Aggregator Demo")
    print("=" * 40)

    aggregator = MistralCostAggregator()

    # Simulate some operations
    import random
    models = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
    teams = ["ai-team", "research-team", "product-team"]

    for i in range(10):
        model = random.choice(models)
        team = random.choice(teams)

        cost_breakdown = {
            "input_tokens": random.randint(100, 1000),
            "output_tokens": random.randint(50, 500),
            "total_tokens": 0,  # Will be calculated
            "input_cost": random.uniform(0.001, 0.01),
            "output_cost": random.uniform(0.001, 0.02),
            "total_cost": 0.0,  # Will be calculated
            "cost_per_token": 0.0
        }

        cost_breakdown["total_tokens"] = cost_breakdown["input_tokens"] + cost_breakdown["output_tokens"]
        cost_breakdown["total_cost"] = cost_breakdown["input_cost"] + cost_breakdown["output_cost"]
        cost_breakdown["cost_per_token"] = cost_breakdown["total_cost"] / cost_breakdown["total_tokens"]

        aggregator.record_operation(
            model=model,
            operation_type="chat",
            cost_breakdown=cost_breakdown,
            team=team,
            project="demo-project"
        )

    # Get summary
    summary = aggregator.get_cost_summary(TimeWindow.DAY)
    print(f"Total cost: ${summary.total_cost:.6f}")
    print(f"Total operations: {summary.total_operations}")
    print(f"Cost by model: {summary.cost_by_model}")
    print(f"Cost by team: {summary.cost_by_team}")

    # Get insights
    insights = aggregator.get_cost_optimization_insights(min_savings_threshold=0.001)  # Low threshold for demo
    if insights:
        print("\nOptimization Insights:")
        for insight in insights[:2]:  # Top 2
            print(f"  • {insight.insight}")
            print(f"    Potential savings: ${insight.potential_savings_usd:.6f}")
            print(f"    Actions: {', '.join(insight.recommended_actions[:2])}")
