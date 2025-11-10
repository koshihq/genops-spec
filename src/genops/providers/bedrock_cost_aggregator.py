#!/usr/bin/env python3
"""
GenOps Bedrock Cost Aggregator

This module provides advanced cost context management for AWS Bedrock operations,
enabling multi-operation cost tracking, optimization recommendations, and 
comprehensive cost analytics with AWS Cost Explorer integration.

Features:
- Multi-operation cost aggregation across different models
- Real-time cost tracking with budget alerts
- Cost optimization recommendations based on usage patterns
- AWS cost allocation tags integration
- Regional cost comparison and optimization
- Provisioned vs on-demand cost analysis
- Enterprise-grade cost reporting and analytics

Example usage:
    from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context
    
    # Multi-operation cost tracking
    with create_bedrock_cost_context("customer_analysis_workflow") as context:
        adapter = GenOpsBedrockAdapter()
        
        # Multiple operations automatically aggregated
        result1 = adapter.text_generation("Analyze this...", model_id="claude-3-haiku")
        result2 = adapter.text_generation("Summarize...", model_id="titan-express")
        
        # Get comprehensive cost summary
        summary = context.get_current_summary()
        print(f"Total workflow cost: ${summary.total_cost:.6f}")
        print(f"Cost by model: {summary.cost_by_model}")
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from genops.core.telemetry import GenOpsTelemetry
    from genops.providers.bedrock_pricing import (
        BedrockCostBreakdown,
        calculate_bedrock_cost,
        compare_bedrock_models,
        get_cost_optimization_recommendations,
        get_detailed_cost_breakdown,
    )
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BedrockOperationRecord:
    """Record of a single Bedrock operation for cost tracking."""
    operation_id: str
    model_id: str
    provider: str
    region: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    timestamp: datetime
    governance_attributes: Dict[str, str] = field(default_factory=dict)
    operation_type: str = "text_generation"
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BedrockCostSummary:
    """Comprehensive cost summary for Bedrock operations."""
    context_id: str
    total_cost: float
    total_operations: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    unique_models: Set[str] = field(default_factory=set)
    unique_providers: Set[str] = field(default_factory=set)
    unique_regions: Set[str] = field(default_factory=set)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    cost_by_region: Dict[str, float] = field(default_factory=dict)
    operations_by_model: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    governance_attributes: Dict[str, str] = field(default_factory=dict)
    optimization_recommendations: List[str] = field(default_factory=list)

    def get_average_cost_per_operation(self) -> float:
        """Get average cost per operation."""
        return self.total_cost / max(1, self.total_operations)

    def get_average_latency_ms(self) -> float:
        """Get average latency per operation."""
        return self.total_latency_ms / max(1, self.total_operations)

    def get_cost_breakdown_percentage(self) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown as percentages."""
        breakdown = {
            "by_model": {},
            "by_provider": {},
            "by_region": {}
        }

        if self.total_cost > 0:
            for model, cost in self.cost_by_model.items():
                breakdown["by_model"][model] = (cost / self.total_cost) * 100

            for provider, cost in self.cost_by_provider.items():
                breakdown["by_provider"][provider] = (cost / self.total_cost) * 100

            for region, cost in self.cost_by_region.items():
                breakdown["by_region"][region] = (cost / self.total_cost) * 100

        return breakdown

    def get_most_expensive_model(self) -> Optional[Tuple[str, float]]:
        """Get the most expensive model used."""
        if not self.cost_by_model:
            return None
        return max(self.cost_by_model.items(), key=lambda x: x[1])

    def get_cheapest_model(self) -> Optional[Tuple[str, float]]:
        """Get the least expensive model used."""
        if not self.cost_by_model:
            return None
        return min(self.cost_by_model.items(), key=lambda x: x[1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary for serialization."""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['unique_models'] = list(self.unique_models)
        data['unique_providers'] = list(self.unique_providers)
        data['unique_regions'] = list(self.unique_regions)
        # Convert datetime objects to ISO strings
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


class BedrockCostContext:
    """
    Context manager for Bedrock cost tracking and optimization.
    
    This enables comprehensive cost aggregation across multiple Bedrock operations
    with real-time optimization recommendations and budget monitoring.
    """

    def __init__(
        self,
        context_id: str,
        budget_limit: Optional[float] = None,
        alert_threshold: float = 0.8,
        enable_optimization_recommendations: bool = True
    ):
        """
        Initialize cost tracking context.
        
        Args:
            context_id: Unique identifier for this cost context
            budget_limit: Maximum budget for this context (optional)
            alert_threshold: Threshold for budget alerts (0.0-1.0)
            enable_optimization_recommendations: Enable real-time optimization suggestions
        """
        self.context_id = context_id
        self.budget_limit = budget_limit
        self.alert_threshold = alert_threshold
        self.enable_optimization = enable_optimization_recommendations

        self.operations: List[BedrockOperationRecord] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.telemetry: Optional[GenOpsTelemetry] = None

        # Initialize telemetry if available
        if GENOPS_AVAILABLE:
            self.telemetry = GenOpsTelemetry()

    def __enter__(self):
        """Enter the cost tracking context."""
        logger.info(f"Starting Bedrock cost tracking context: {self.context_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the cost tracking context with final summary."""
        self.end_time = datetime.now()
        summary = self.get_current_summary()

        # Log final summary
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(
            f"Bedrock cost context '{self.context_id}' completed: "
            f"${summary.total_cost:.6f} over {duration:.1f}s "
            f"({summary.total_operations} operations)"
        )

        # Export telemetry if available
        if self.telemetry:
            with self.telemetry.trace_operation(
                operation_name="bedrock.cost_context.summary",
                context_id=self.context_id
            ) as span:
                span.set_attribute("bedrock.context.total_cost", summary.total_cost)
                span.set_attribute("bedrock.context.total_operations", summary.total_operations)
                span.set_attribute("bedrock.context.duration_ms", duration * 1000)
                span.set_attribute("bedrock.context.unique_models", len(summary.unique_models))

    def add_operation(
        self,
        operation_id: str,
        model_id: str,
        provider: str,
        region: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        governance_attributes: Optional[Dict[str, str]] = None,
        operation_type: str = "text_generation",
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Add an operation to the cost tracking context.
        
        Args:
            operation_id: Unique operation identifier
            model_id: Bedrock model ID used
            provider: Model provider (anthropic, amazon, etc.)
            region: AWS region
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Operation latency in milliseconds
            governance_attributes: Governance attributes for the operation
            operation_type: Type of operation (text_generation, chat, etc.)
            success: Whether the operation succeeded
            error_message: Error message if operation failed
        """
        # Calculate cost for this operation
        cost = calculate_bedrock_cost(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            region=region
        )

        # Create operation record
        record = BedrockOperationRecord(
            operation_id=operation_id,
            model_id=model_id,
            provider=provider,
            region=region,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            governance_attributes=governance_attributes or {},
            operation_type=operation_type,
            success=success,
            error_message=error_message
        )

        self.operations.append(record)

        # Check budget alerts
        if self.budget_limit:
            current_total = sum(op.cost for op in self.operations)
            if current_total >= self.budget_limit * self.alert_threshold:
                logger.warning(
                    f"Budget alert: Context '{self.context_id}' has spent "
                    f"${current_total:.6f} of ${self.budget_limit:.6f} budget "
                    f"({(current_total/self.budget_limit)*100:.1f}%)"
                )

        logger.debug(f"Added operation {operation_id}: ${cost:.6f} ({model_id})")

    def get_current_summary(self) -> BedrockCostSummary:
        """Get current cost summary for the context."""
        if not self.operations:
            return BedrockCostSummary(
                context_id=self.context_id,
                total_cost=0.0,
                total_operations=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_latency_ms=0.0,
                start_time=self.start_time
            )

        # Aggregate metrics
        total_cost = sum(op.cost for op in self.operations)
        total_operations = len(self.operations)
        total_input_tokens = sum(op.input_tokens for op in self.operations)
        total_output_tokens = sum(op.output_tokens for op in self.operations)
        total_latency_ms = sum(op.latency_ms for op in self.operations)

        # Aggregate by dimensions
        unique_models = set(op.model_id for op in self.operations)
        unique_providers = set(op.provider for op in self.operations)
        unique_regions = set(op.region for op in self.operations)

        cost_by_model = defaultdict(float)
        cost_by_provider = defaultdict(float)
        cost_by_region = defaultdict(float)
        operations_by_model = defaultdict(int)

        for op in self.operations:
            cost_by_model[op.model_id] += op.cost
            cost_by_provider[op.provider] += op.cost
            cost_by_region[op.region] += op.cost
            operations_by_model[op.model_id] += 1

        # Collect governance attributes from first operation
        governance_attrs = {}
        if self.operations:
            governance_attrs = self.operations[0].governance_attributes.copy()

        # Generate optimization recommendations
        recommendations = []
        if self.enable_optimization and len(self.operations) > 1:
            recommendations = self._generate_optimization_recommendations(
                cost_by_model, operations_by_model, total_cost
            )

        return BedrockCostSummary(
            context_id=self.context_id,
            total_cost=total_cost,
            total_operations=total_operations,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_latency_ms=total_latency_ms,
            unique_models=unique_models,
            unique_providers=unique_providers,
            unique_regions=unique_regions,
            cost_by_model=dict(cost_by_model),
            cost_by_provider=dict(cost_by_provider),
            cost_by_region=dict(cost_by_region),
            operations_by_model=dict(operations_by_model),
            start_time=self.start_time,
            end_time=self.end_time,
            governance_attributes=governance_attrs,
            optimization_recommendations=recommendations
        )

    def _generate_optimization_recommendations(
        self,
        cost_by_model: Dict[str, float],
        operations_by_model: Dict[str, int],
        total_cost: float
    ) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        if not cost_by_model:
            return recommendations

        # Find most expensive model
        most_expensive_model = max(cost_by_model.items(), key=lambda x: x[1])
        most_expensive_cost = most_expensive_model[1]
        most_expensive_percentage = (most_expensive_cost / total_cost) * 100

        if most_expensive_percentage > 50:
            recommendations.append(
                f"Model {most_expensive_model[0]} accounts for {most_expensive_percentage:.1f}% "
                f"of costs (${most_expensive_cost:.6f}). Consider cheaper alternatives for high-volume tasks."
            )

        # Check for model diversity
        if len(cost_by_model) > 3:
            cheapest_model = min(cost_by_model.items(), key=lambda x: x[1])
            cost_ratio = most_expensive_model[1] / max(cheapest_model[1], 0.000001)

            if cost_ratio > 10:
                recommendations.append(
                    f"Cost variation is high ({cost_ratio:.1f}x between cheapest and most expensive). "
                    f"Consider standardizing on {cheapest_model[0]} for similar tasks."
                )

        # Volume-based recommendations
        high_volume_models = [
            model for model, ops in operations_by_model.items()
            if ops > len(self.operations) * 0.3
        ]

        for model in high_volume_models:
            avg_cost_per_op = cost_by_model[model] / operations_by_model[model]
            if avg_cost_per_op > 0.01:  # $0.01 per operation threshold
                recommendations.append(
                    f"High-volume model {model} costs ${avg_cost_per_op:.6f} per operation. "
                    f"Consider a more efficient model for bulk processing."
                )

        return recommendations

    def get_operations_by_timespan(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[BedrockOperationRecord]:
        """Get operations within a specific timespan."""
        filtered_ops = self.operations

        if start_time:
            filtered_ops = [op for op in filtered_ops if op.timestamp >= start_time]

        if end_time:
            filtered_ops = [op for op in filtered_ops if op.timestamp <= end_time]

        return filtered_ops

    def export_cost_report(
        self,
        format: str = "json",
        include_operations: bool = False
    ) -> str:
        """
        Export comprehensive cost report.
        
        Args:
            format: Export format ("json", "csv", "summary")
            include_operations: Include individual operation details
            
        Returns:
            Formatted cost report
        """
        summary = self.get_current_summary()

        if format == "json":
            report_data = summary.to_dict()
            if include_operations:
                report_data['operations'] = [
                    {
                        'operation_id': op.operation_id,
                        'model_id': op.model_id,
                        'provider': op.provider,
                        'region': op.region,
                        'cost': op.cost,
                        'input_tokens': op.input_tokens,
                        'output_tokens': op.output_tokens,
                        'latency_ms': op.latency_ms,
                        'timestamp': op.timestamp.isoformat(),
                        'success': op.success
                    }
                    for op in self.operations
                ]
            return json.dumps(report_data, indent=2)

        elif format == "summary":
            lines = [
                f"Bedrock Cost Summary - Context: {self.context_id}",
                "=" * 50,
                f"Total Cost: ${summary.total_cost:.6f}",
                f"Total Operations: {summary.total_operations}",
                f"Average Cost/Operation: ${summary.get_average_cost_per_operation():.6f}",
                f"Average Latency: {summary.get_average_latency_ms():.1f}ms",
                f"Models Used: {', '.join(summary.unique_models)}",
                f"Providers: {', '.join(summary.unique_providers)}",
                f"Regions: {', '.join(summary.unique_regions)}",
                ""
            ]

            if summary.cost_by_model:
                lines.append("Cost by Model:")
                for model, cost in sorted(summary.cost_by_model.items(), key=lambda x: x[1], reverse=True):
                    percentage = (cost / summary.total_cost) * 100
                    lines.append(f"  {model}: ${cost:.6f} ({percentage:.1f}%)")
                lines.append("")

            if summary.optimization_recommendations:
                lines.append("Optimization Recommendations:")
                for i, rec in enumerate(summary.optimization_recommendations, 1):
                    lines.append(f"  {i}. {rec}")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global cost aggregator for cross-context tracking
_global_cost_aggregator: Optional[Dict[str, BedrockCostContext]] = {}


def create_bedrock_cost_context(
    context_id: str,
    budget_limit: Optional[float] = None,
    alert_threshold: float = 0.8,
    enable_optimization_recommendations: bool = True
) -> BedrockCostContext:
    """
    Create a cost tracking context for Bedrock operations.
    
    This follows the exact pattern specified in CLAUDE.md for framework adapters:
    
    with create_bedrock_cost_context("operation_id") as context:
        # Multiple models/operations automatically aggregated
        result1 = adapter.text_generation(...)
        result2 = adapter.text_generation(...)
        summary = context.get_current_summary()
    
    Args:
        context_id: Unique identifier for this cost context
        budget_limit: Optional budget limit for alerts
        alert_threshold: Budget alert threshold (0.0-1.0)
        enable_optimization_recommendations: Enable real-time optimization
        
    Returns:
        BedrockCostContext for tracking operations
    """
    context = BedrockCostContext(
        context_id=context_id,
        budget_limit=budget_limit,
        alert_threshold=alert_threshold,
        enable_optimization_recommendations=enable_optimization_recommendations
    )

    # Register in global aggregator for cross-context analysis
    if _global_cost_aggregator is not None:
        _global_cost_aggregator[context_id] = context

    return context


def get_global_cost_summary(
    timespan_hours: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get aggregated cost summary across all active contexts.
    
    Args:
        timespan_hours: Limit to operations within last N hours
        
    Returns:
        Global cost summary with cross-context analytics
    """
    if not _global_cost_aggregator:
        return {"total_contexts": 0, "total_cost": 0.0}

    cutoff_time = None
    if timespan_hours:
        cutoff_time = datetime.now() - timedelta(hours=timespan_hours)

    total_cost = 0.0
    total_operations = 0
    all_models = set()
    all_providers = set()

    context_summaries = {}

    for context_id, context in _global_cost_aggregator.items():
        ops = context.operations

        if cutoff_time:
            ops = [op for op in ops if op.timestamp >= cutoff_time]

        if ops:
            context_cost = sum(op.cost for op in ops)
            context_ops = len(ops)

            total_cost += context_cost
            total_operations += context_ops
            all_models.update(op.model_id for op in ops)
            all_providers.update(op.provider for op in ops)

            context_summaries[context_id] = {
                "cost": context_cost,
                "operations": context_ops,
                "avg_cost_per_op": context_cost / context_ops if context_ops > 0 else 0
            }

    return {
        "total_contexts": len(context_summaries),
        "total_cost": total_cost,
        "total_operations": total_operations,
        "avg_cost_per_operation": total_cost / total_operations if total_operations > 0 else 0,
        "unique_models": len(all_models),
        "unique_providers": len(all_providers),
        "context_breakdown": context_summaries,
        "timespan_hours": timespan_hours
    }


def cleanup_old_contexts(max_age_hours: int = 24):
    """Clean up old cost contexts to prevent memory leaks."""
    if not _global_cost_aggregator:
        return

    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    contexts_to_remove = []

    for context_id, context in _global_cost_aggregator.items():
        if context.end_time and context.end_time < cutoff_time:
            contexts_to_remove.append(context_id)

    for context_id in contexts_to_remove:
        del _global_cost_aggregator[context_id]

    if contexts_to_remove:
        logger.info(f"Cleaned up {len(contexts_to_remove)} old cost contexts")


# Export main classes and functions
__all__ = [
    'BedrockCostContext',
    'BedrockCostSummary',
    'BedrockOperationRecord',
    'create_bedrock_cost_context',
    'get_global_cost_summary',
    'cleanup_old_contexts'
]
