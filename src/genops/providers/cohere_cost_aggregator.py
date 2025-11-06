"""Cost aggregation and analytics system for Cohere operations."""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from .cohere_pricing import CohereCalculator, CostBreakdown
except ImportError:
    logger.warning("Could not import Cohere pricing calculator")


class TimeWindow(Enum):
    """Time window options for cost aggregation."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    ALL_TIME = "all_time"


@dataclass
class CostMetrics:
    """Comprehensive cost metrics for analysis."""
    
    # Basic metrics
    total_cost: float = 0.0
    total_operations: int = 0
    
    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # Operation-specific metrics
    total_embeddings: int = 0
    total_searches: int = 0
    total_image_tokens: int = 0
    
    # Cost breakdown
    input_token_costs: float = 0.0
    output_token_costs: float = 0.0
    embedding_costs: float = 0.0
    search_costs: float = 0.0
    image_costs: float = 0.0
    
    # Performance metrics
    avg_cost_per_operation: float = 0.0
    avg_tokens_per_operation: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Time metrics
    first_operation: Optional[float] = None
    last_operation: Optional[float] = None
    time_span_hours: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.total_tokens = self.total_input_tokens + self.total_output_tokens
        
        if self.total_operations > 0:
            self.avg_cost_per_operation = self.total_cost / self.total_operations
            self.avg_tokens_per_operation = self.total_tokens / self.total_operations
        
        if self.first_operation and self.last_operation:
            self.time_span_hours = (self.last_operation - self.first_operation) / 3600
    
    def update(self, cost_breakdown: CostBreakdown, latency_ms: float = 0.0, timestamp: float = None):
        """Update metrics with new cost data."""
        if timestamp is None:
            timestamp = time.time()
        
        # Update basic metrics
        self.total_cost += cost_breakdown.total_cost
        self.total_operations += 1
        
        # Update token metrics
        self.total_input_tokens += cost_breakdown.input_tokens
        self.total_output_tokens += cost_breakdown.output_tokens
        
        # Update operation-specific metrics
        self.total_embeddings += cost_breakdown.embedding_units
        self.total_searches += cost_breakdown.search_units
        self.total_image_tokens += cost_breakdown.image_tokens
        
        # Update cost breakdown
        self.input_token_costs += cost_breakdown.input_token_cost
        self.output_token_costs += cost_breakdown.output_token_cost
        self.embedding_costs += cost_breakdown.embedding_cost
        self.search_costs += cost_breakdown.search_cost
        self.image_costs += cost_breakdown.image_token_cost
        
        # Update performance metrics
        if latency_ms > 0:
            total_latency = self.avg_latency_ms * (self.total_operations - 1) + latency_ms
            self.avg_latency_ms = total_latency / self.total_operations
        
        # Update time metrics
        if self.first_operation is None:
            self.first_operation = timestamp
        self.last_operation = timestamp
        
        # Recalculate derived metrics
        self.__post_init__()


@dataclass
class OperationSummary:
    """Summary of operations by type, model, and governance attributes."""
    
    # Operation breakdown
    operations_by_type: Dict[str, int] = field(default_factory=dict)
    operations_by_model: Dict[str, int] = field(default_factory=dict)
    costs_by_type: Dict[str, float] = field(default_factory=dict)
    costs_by_model: Dict[str, float] = field(default_factory=dict)
    
    # Governance breakdown
    costs_by_team: Dict[str, float] = field(default_factory=dict)
    costs_by_project: Dict[str, float] = field(default_factory=dict)
    costs_by_customer: Dict[str, float] = field(default_factory=dict)
    costs_by_environment: Dict[str, float] = field(default_factory=dict)
    
    # Top usage patterns
    top_models_by_cost: List[Tuple[str, float]] = field(default_factory=list)
    top_teams_by_cost: List[Tuple[str, float]] = field(default_factory=list)
    top_operations_by_cost: List[Tuple[str, float]] = field(default_factory=list)


class CohereCostAggregator:
    """
    Comprehensive cost aggregation and analytics system for Cohere operations.
    
    Features:
    - Real-time cost tracking across all Cohere operations
    - Multi-dimensional cost attribution (team, project, customer, model)
    - Time-based cost analysis with configurable windows
    - Cost optimization insights and recommendations
    - Budget tracking and alerting
    - Detailed usage analytics and reporting
    """
    
    def __init__(
        self,
        enable_detailed_tracking: bool = True,
        cost_alert_threshold: Optional[float] = None,
        budget_period_hours: int = 24,
        max_history_days: int = 30
    ):
        """
        Initialize cost aggregator.
        
        Args:
            enable_detailed_tracking: Enable detailed per-operation tracking
            cost_alert_threshold: Optional cost threshold for alerts
            budget_period_hours: Budget period in hours for rate limiting
            max_history_days: Maximum days to retain detailed history
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.cost_alert_threshold = cost_alert_threshold
        self.budget_period_hours = budget_period_hours
        self.max_history_days = max_history_days
        
        # Initialize calculator
        try:
            self.calculator = CohereCalculator()
        except Exception as e:
            logger.warning(f"Could not initialize Cohere calculator: {e}")
            self.calculator = None
        
        # Cost tracking data structures
        self.total_metrics = CostMetrics()
        
        # Multi-dimensional cost tracking
        self.costs_by_model: Dict[str, CostMetrics] = defaultdict(CostMetrics)
        self.costs_by_operation: Dict[str, CostMetrics] = defaultdict(CostMetrics)
        self.costs_by_team: Dict[str, CostMetrics] = defaultdict(CostMetrics)
        self.costs_by_project: Dict[str, CostMetrics] = defaultdict(CostMetrics)
        self.costs_by_customer: Dict[str, CostMetrics] = defaultdict(CostMetrics)
        self.costs_by_environment: Dict[str, CostMetrics] = defaultdict(CostMetrics)
        
        # Detailed operation history (if enabled)
        self.operation_history: List[Dict[str, Any]] = []
        
        # Time-based tracking
        self.hourly_costs: Dict[str, float] = {}  # hourly_key -> cost
        self.daily_costs: Dict[str, float] = {}   # daily_key -> cost
        
        # Budget tracking
        self.current_period_cost = 0.0
        self.current_period_start = time.time()
        
        logger.info("Cohere cost aggregator initialized")
    
    def record_operation(
        self,
        model: str,
        operation_type: str,
        cost_breakdown: CostBreakdown,
        latency_ms: float = 0.0,
        timestamp: Optional[float] = None,
        **governance_attrs
    ):
        """
        Record a new operation for cost tracking.
        
        Args:
            model: Model name used
            operation_type: Type of operation (CHAT, EMBED, RERANK, etc.)
            cost_breakdown: Detailed cost breakdown
            latency_ms: Operation latency in milliseconds
            timestamp: Operation timestamp (defaults to current time)
            **governance_attrs: Governance attributes (team, project, customer_id, etc.)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update total metrics
        self.total_metrics.update(cost_breakdown, latency_ms, timestamp)
        
        # Update dimensional metrics
        self.costs_by_model[model].update(cost_breakdown, latency_ms, timestamp)
        self.costs_by_operation[operation_type].update(cost_breakdown, latency_ms, timestamp)
        
        # Update governance metrics
        team = governance_attrs.get("team", "unknown")
        project = governance_attrs.get("project", "unknown")
        customer_id = governance_attrs.get("customer_id", "unknown")
        environment = governance_attrs.get("environment", "unknown")
        
        if team != "unknown":
            self.costs_by_team[team].update(cost_breakdown, latency_ms, timestamp)
        if project != "unknown":
            self.costs_by_project[project].update(cost_breakdown, latency_ms, timestamp)
        if customer_id != "unknown":
            self.costs_by_customer[customer_id].update(cost_breakdown, latency_ms, timestamp)
        if environment != "unknown":
            self.costs_by_environment[environment].update(cost_breakdown, latency_ms, timestamp)
        
        # Update time-based tracking
        self._update_time_based_costs(cost_breakdown.total_cost, timestamp)
        
        # Update budget tracking
        self._update_budget_tracking(cost_breakdown.total_cost, timestamp)
        
        # Store detailed history if enabled
        if self.enable_detailed_tracking:
            operation_record = {
                "timestamp": timestamp,
                "model": model,
                "operation_type": operation_type,
                "cost_breakdown": cost_breakdown,
                "latency_ms": latency_ms,
                **governance_attrs
            }
            self.operation_history.append(operation_record)
            
            # Clean up old history
            self._cleanup_old_history()
        
        # Check for cost alerts
        if self.cost_alert_threshold and self.total_metrics.total_cost >= self.cost_alert_threshold:
            logger.warning(f"Cost alert: Total cost ${self.total_metrics.total_cost:.6f} exceeded threshold ${self.cost_alert_threshold:.6f}")
    
    def _update_time_based_costs(self, cost: float, timestamp: float):
        """Update hourly and daily cost tracking."""
        dt = datetime.fromtimestamp(timestamp)
        
        # Hourly tracking
        hour_key = dt.strftime("%Y-%m-%d-%H")
        self.hourly_costs[hour_key] = self.hourly_costs.get(hour_key, 0.0) + cost
        
        # Daily tracking
        day_key = dt.strftime("%Y-%m-%d")
        self.daily_costs[day_key] = self.daily_costs.get(day_key, 0.0) + cost
    
    def _update_budget_tracking(self, cost: float, timestamp: float):
        """Update budget period tracking."""
        # Check if we need to reset the budget period
        if timestamp - self.current_period_start > (self.budget_period_hours * 3600):
            self.current_period_cost = 0.0
            self.current_period_start = timestamp
        
        self.current_period_cost += cost
    
    def _cleanup_old_history(self):
        """Remove old operation history beyond retention period."""
        if not self.operation_history:
            return
        
        cutoff_time = time.time() - (self.max_history_days * 24 * 3600)
        self.operation_history = [
            op for op in self.operation_history 
            if op["timestamp"] > cutoff_time
        ]
    
    def get_cost_summary(self, time_window: TimeWindow = TimeWindow.ALL_TIME) -> Dict[str, Any]:
        """
        Get comprehensive cost summary.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            Dictionary with cost summary and analytics
        """
        # Get time-filtered metrics
        if time_window == TimeWindow.ALL_TIME:
            base_metrics = self.total_metrics
        else:
            base_metrics = self._get_time_filtered_metrics(time_window)
        
        summary = {
            "overview": {
                "total_cost": round(base_metrics.total_cost, 6),
                "total_operations": base_metrics.total_operations,
                "avg_cost_per_operation": round(base_metrics.avg_cost_per_operation, 6),
                "time_window": time_window.value,
                "time_span_hours": round(base_metrics.time_span_hours, 2)
            },
            
            "usage_metrics": {
                "total_tokens": base_metrics.total_tokens,
                "input_tokens": base_metrics.total_input_tokens,
                "output_tokens": base_metrics.total_output_tokens,
                "embeddings": base_metrics.total_embeddings,
                "searches": base_metrics.total_searches,
                "avg_tokens_per_operation": round(base_metrics.avg_tokens_per_operation, 1),
                "avg_latency_ms": round(base_metrics.avg_latency_ms, 1)
            },
            
            "cost_breakdown": {
                "input_token_costs": round(base_metrics.input_token_costs, 6),
                "output_token_costs": round(base_metrics.output_token_costs, 6),
                "embedding_costs": round(base_metrics.embedding_costs, 6),
                "search_costs": round(base_metrics.search_costs, 6),
                "image_costs": round(base_metrics.image_costs, 6)
            },
            
            "budget_tracking": {
                "current_period_cost": round(self.current_period_cost, 6),
                "budget_period_hours": self.budget_period_hours,
                "cost_alert_threshold": self.cost_alert_threshold,
                "period_utilization": round((self.current_period_cost / self.cost_alert_threshold * 100) if self.cost_alert_threshold else 0, 2)
            }
        }
        
        return summary
    
    def get_operation_summary(self, time_window: TimeWindow = TimeWindow.ALL_TIME) -> OperationSummary:
        """
        Get detailed operation summary with breakdowns.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            OperationSummary with detailed breakdowns
        """
        # Filter operations by time window if needed
        operations = self.operation_history
        if time_window != TimeWindow.ALL_TIME:
            cutoff_time = self._get_time_window_cutoff(time_window)
            operations = [op for op in operations if op["timestamp"] > cutoff_time]
        
        summary = OperationSummary()
        
        # Aggregate by different dimensions
        for op in operations:
            op_type = op["operation_type"]
            model = op["model"]
            cost = op["cost_breakdown"].total_cost
            
            # Operation type breakdown
            summary.operations_by_type[op_type] = summary.operations_by_type.get(op_type, 0) + 1
            summary.costs_by_type[op_type] = summary.costs_by_type.get(op_type, 0.0) + cost
            
            # Model breakdown
            summary.operations_by_model[model] = summary.operations_by_model.get(model, 0) + 1
            summary.costs_by_model[model] = summary.costs_by_model.get(model, 0.0) + cost
            
            # Governance breakdown
            team = op.get("team", "unknown")
            project = op.get("project", "unknown")
            customer_id = op.get("customer_id", "unknown")
            environment = op.get("environment", "unknown")
            
            if team != "unknown":
                summary.costs_by_team[team] = summary.costs_by_team.get(team, 0.0) + cost
            if project != "unknown":
                summary.costs_by_project[project] = summary.costs_by_project.get(project, 0.0) + cost
            if customer_id != "unknown":
                summary.costs_by_customer[customer_id] = summary.costs_by_customer.get(customer_id, 0.0) + cost
            if environment != "unknown":
                summary.costs_by_environment[environment] = summary.costs_by_environment.get(environment, 0.0) + cost
        
        # Generate top lists
        summary.top_models_by_cost = sorted(summary.costs_by_model.items(), key=lambda x: x[1], reverse=True)[:10]
        summary.top_teams_by_cost = sorted(summary.costs_by_team.items(), key=lambda x: x[1], reverse=True)[:10]
        summary.top_operations_by_cost = sorted(summary.costs_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return summary
    
    def get_cost_optimization_insights(self) -> Dict[str, Any]:
        """
        Generate cost optimization insights and recommendations.
        
        Returns:
            Dictionary with optimization recommendations
        """
        insights = {
            "recommendations": [],
            "cost_efficiency": {},
            "model_comparisons": {},
            "usage_patterns": {}
        }
        
        if not self.calculator:
            insights["recommendations"].append("⚠️ Cost calculator unavailable - install pricing module for optimization insights")
            return insights
        
        # Analyze model efficiency
        model_efficiency = {}
        for model, metrics in self.costs_by_model.items():
            if metrics.total_operations > 0:
                cost_per_token = metrics.total_cost / max(1, metrics.total_tokens)
                cost_per_operation = metrics.avg_cost_per_operation
                
                model_efficiency[model] = {
                    "cost_per_token": cost_per_token,
                    "cost_per_operation": cost_per_operation,
                    "total_operations": metrics.total_operations,
                    "avg_latency_ms": metrics.avg_latency_ms
                }
        
        insights["cost_efficiency"] = model_efficiency
        
        # Generate recommendations
        if len(model_efficiency) > 1:
            # Find most and least cost-efficient models
            sorted_by_efficiency = sorted(
                model_efficiency.items(), 
                key=lambda x: x[1]["cost_per_operation"]
            )
            
            most_efficient = sorted_by_efficiency[0]
            least_efficient = sorted_by_efficiency[-1]
            
            efficiency_diff = least_efficient[1]["cost_per_operation"] / most_efficient[1]["cost_per_operation"]
            
            if efficiency_diff > 2.0:  # More than 2x difference
                insights["recommendations"].append(
                    f"💰 Consider switching from {least_efficient[0]} to {most_efficient[0]} "
                    f"for {efficiency_diff:.1f}x cost reduction per operation"
                )
        
        # Analyze usage patterns
        total_cost = self.total_metrics.total_cost
        if total_cost > 0:
            # Check if embedding costs are high relative to generation
            embedding_ratio = self.total_metrics.embedding_costs / total_cost
            if embedding_ratio > 0.5:
                insights["recommendations"].append(
                    f"📊 Embedding costs are {embedding_ratio:.1%} of total - consider optimizing embedding frequency or batching"
                )
            
            # Check if search costs are significant
            search_ratio = self.total_metrics.search_costs / total_cost
            if search_ratio > 0.3:
                insights["recommendations"].append(
                    f"🔍 Search costs are {search_ratio:.1%} of total - consider caching search results or optimizing query frequency"
                )
        
        # Budget utilization insights
        if self.cost_alert_threshold:
            utilization = (self.current_period_cost / self.cost_alert_threshold) * 100
            if utilization > 80:
                insights["recommendations"].append(
                    f"⚠️ Budget utilization is {utilization:.1f}% - consider cost controls or budget increase"
                )
        
        return insights
    
    def _get_time_window_cutoff(self, time_window: TimeWindow) -> float:
        """Get timestamp cutoff for time window."""
        now = time.time()
        
        if time_window == TimeWindow.HOUR:
            return now - 3600
        elif time_window == TimeWindow.DAY:
            return now - (24 * 3600)
        elif time_window == TimeWindow.WEEK:
            return now - (7 * 24 * 3600)
        elif time_window == TimeWindow.MONTH:
            return now - (30 * 24 * 3600)
        else:
            return 0
    
    def _get_time_filtered_metrics(self, time_window: TimeWindow) -> CostMetrics:
        """Get metrics filtered by time window."""
        cutoff_time = self._get_time_window_cutoff(time_window)
        
        filtered_metrics = CostMetrics()
        
        for op in self.operation_history:
            if op["timestamp"] > cutoff_time:
                filtered_metrics.update(
                    op["cost_breakdown"], 
                    op["latency_ms"], 
                    op["timestamp"]
                )
        
        return filtered_metrics
    
    def export_cost_data(self, format: str = "dict") -> Any:
        """
        Export cost data in specified format.
        
        Args:
            format: Export format ("dict", "json", "csv")
            
        Returns:
            Cost data in requested format
        """
        data = {
            "total_metrics": {
                "total_cost": self.total_metrics.total_cost,
                "total_operations": self.total_metrics.total_operations,
                "total_tokens": self.total_metrics.total_tokens,
                "avg_cost_per_operation": self.total_metrics.avg_cost_per_operation,
                "time_span_hours": self.total_metrics.time_span_hours
            },
            "costs_by_model": {
                model: {
                    "total_cost": metrics.total_cost,
                    "operations": metrics.total_operations,
                    "avg_cost": metrics.avg_cost_per_operation
                }
                for model, metrics in self.costs_by_model.items()
            },
            "costs_by_team": {
                team: {
                    "total_cost": metrics.total_cost,
                    "operations": metrics.total_operations
                }
                for team, metrics in self.costs_by_team.items()
            },
            "hourly_costs": dict(self.hourly_costs),
            "daily_costs": dict(self.daily_costs)
        }
        
        if format == "json":
            import json
            return json.dumps(data, indent=2)
        elif format == "csv":
            # Return CSV-formatted string for operations
            lines = ["timestamp,model,operation_type,cost,team,project"]
            for op in self.operation_history:
                lines.append(f"{op['timestamp']},{op['model']},{op['operation_type']},{op['cost_breakdown'].total_cost},{op.get('team', '')},{op.get('project', '')}")
            return "\n".join(lines)
        
        return data
    
    def reset_metrics(self):
        """Reset all metrics and history."""
        self.total_metrics = CostMetrics()
        self.costs_by_model.clear()
        self.costs_by_operation.clear()
        self.costs_by_team.clear()
        self.costs_by_project.clear()
        self.costs_by_customer.clear()
        self.costs_by_environment.clear()
        self.operation_history.clear()
        self.hourly_costs.clear()
        self.daily_costs.clear()
        self.current_period_cost = 0.0
        self.current_period_start = time.time()
        
        logger.info("Cost aggregator metrics reset")


# Export main classes
__all__ = [
    "CohereCostAggregator",
    "CostMetrics",
    "OperationSummary",
    "TimeWindow"
]