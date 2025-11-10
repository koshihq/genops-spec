#!/usr/bin/env python3
"""
Gemini cost aggregation and multi-operation tracking for GenOps.

This module provides context manager patterns for tracking costs across
multiple Gemini operations, enabling unified cost attribution and optimization
recommendations across complex AI workflows.

Features:
- Multi-operation cost aggregation with automatic finalization
- Cross-model cost comparison within workflows
- Budget-constrained operation strategies
- Real-time cost monitoring and alerts
- Integration with GenOps governance framework

Usage:
    from genops.providers.gemini_cost_aggregator import create_gemini_cost_context
    
    # Track costs across multiple operations
    with create_gemini_cost_context("ai_workflow_analysis") as context:
        # Multiple operations automatically tracked
        result1 = adapter.text_generation(prompt1, model="gemini-2.5-pro")
        result2 = adapter.text_generation(prompt2, model="gemini-2.5-flash")
        
        # Get unified cost summary
        summary = context.get_current_summary()
        print(f"Total workflow cost: ${summary.total_cost:.6f}")
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from genops.providers.gemini_pricing import (
    calculate_gemini_cost,
)

try:
    from genops.core.telemetry import GenOpsTelemetry
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CostAlertLevel(Enum):
    """Cost alert levels for budget monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class GeminiOperation:
    """Individual Gemini operation with cost and metadata."""
    operation_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: float
    governance_attributes: Dict[str, str] = field(default_factory=dict)
    operation_type: str = "text_generation"
    context_cache_tokens: Optional[int] = None


@dataclass
class GeminiCostSummary:
    """Aggregated cost summary for multiple Gemini operations."""
    total_cost: float
    currency: str
    total_operations: int
    unique_models: Set[str]
    cost_by_model: Dict[str, float]
    cost_by_operation_type: Dict[str, float]
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    operations: List[GeminiOperation]
    governance_attributes: Dict[str, str]
    optimization_recommendations: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def get_average_cost_per_operation(self) -> float:
        """Calculate average cost per operation."""
        return self.total_cost / self.total_operations if self.total_operations > 0 else 0.0

    def get_average_latency_ms(self) -> float:
        """Calculate average latency per operation."""
        return self.total_latency_ms / self.total_operations if self.total_operations > 0 else 0.0

    def get_cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score (lower is better)."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return (self.total_cost / total_tokens) * 1000 if total_tokens > 0 else 0.0


class GeminiCostContext:
    """Context manager for tracking Gemini costs across multiple operations."""

    def __init__(
        self,
        context_id: str,
        budget_limit: Optional[float] = None,
        enable_optimization: bool = True,
        enable_alerts: bool = True,
        governance_attributes: Optional[Dict[str, str]] = None
    ):
        """
        Initialize Gemini cost context.
        
        Args:
            context_id: Unique identifier for this cost context
            budget_limit: Maximum cost limit in USD (optional)
            enable_optimization: Enable automatic optimization recommendations
            enable_alerts: Enable budget alert monitoring
            governance_attributes: Default governance attributes for all operations
        """
        self.context_id = context_id
        self.budget_limit = budget_limit
        self.enable_optimization = enable_optimization
        self.enable_alerts = enable_alerts
        self.governance_attributes = governance_attributes or {}

        # Track operations and costs
        self.operations: List[GeminiOperation] = []
        self.total_cost = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Cost monitoring
        self.cost_alerts: List[Dict[str, Any]] = []
        self.budget_warnings_sent: Set[CostAlertLevel] = set()

        # Initialize telemetry if available
        self.telemetry = GenOpsTelemetry() if GENOPS_AVAILABLE else None

        logger.info(f"Initialized Gemini cost context: {context_id}")

    def __enter__(self) -> 'GeminiCostContext':
        """Enter the cost tracking context."""
        self.start_time = time.time()

        if self.telemetry:
            # Start a span for the entire context
            self.span = self.telemetry.start_span(
                f"gemini_cost_context_{self.context_id}",
                attributes={
                    "genops.provider": "gemini",
                    "genops.context_id": self.context_id,
                    "genops.operation_type": "cost_aggregation",
                    **{f"genops.{k}": str(v) for k, v in self.governance_attributes.items()}
                }
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the cost tracking context and finalize costs."""
        self.end_time = time.time()

        # Generate final summary
        summary = self.get_current_summary()

        # Add optimization recommendations if enabled
        if self.enable_optimization:
            self._generate_optimization_recommendations(summary)

        # Finalize telemetry
        if self.telemetry and hasattr(self, 'span'):
            self.span.set_attributes({
                "genops.cost.total": summary.total_cost,
                "genops.cost.currency": "USD",
                "genops.operations.count": summary.total_operations,
                "genops.tokens.total_input": summary.total_input_tokens,
                "genops.tokens.total_output": summary.total_output_tokens,
                "genops.latency.total_ms": summary.total_latency_ms,
                "genops.models.unique_count": len(summary.unique_models),
                "genops.context.duration_ms": (self.end_time - self.start_time) * 1000
            })

            if exc_type:
                self.span.set_status(status="ERROR", description=str(exc_val))
            else:
                self.span.set_status(status="OK")

            self.span.end()

        logger.info(f"Finalized Gemini cost context {self.context_id}: ${summary.total_cost:.6f}")

    def add_operation(
        self,
        operation_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        operation_type: str = "text_generation",
        context_cache_tokens: Optional[int] = None,
        governance_attributes: Optional[Dict[str, str]] = None
    ) -> GeminiOperation:
        """
        Add an operation to the cost context.
        
        Args:
            operation_id: Unique operation identifier
            model_id: Gemini model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Operation latency in milliseconds
            operation_type: Type of operation
            context_cache_tokens: Context cache tokens used
            governance_attributes: Operation-specific governance attributes
        
        Returns:
            GeminiOperation object representing the added operation
        """
        # Calculate cost for this operation
        cost_usd = calculate_gemini_cost(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_cache_tokens=context_cache_tokens
        )

        # Merge governance attributes
        merged_attrs = {**self.governance_attributes, **(governance_attributes or {})}

        # Create operation record
        operation = GeminiOperation(
            operation_id=operation_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=time.time(),
            governance_attributes=merged_attrs,
            operation_type=operation_type,
            context_cache_tokens=context_cache_tokens
        )

        # Add to tracking
        self.operations.append(operation)
        self.total_cost += cost_usd

        # Check budget alerts
        if self.enable_alerts and self.budget_limit:
            self._check_budget_alerts()

        logger.debug(f"Added Gemini operation {operation_id}: {model_id}, ${cost_usd:.6f}")

        return operation

    def get_current_summary(self) -> GeminiCostSummary:
        """
        Get current cost summary for all operations in this context.
        
        Returns:
            GeminiCostSummary with aggregated cost information
        """
        if not self.operations:
            return GeminiCostSummary(
                total_cost=0.0,
                currency="USD",
                total_operations=0,
                unique_models=set(),
                cost_by_model={},
                cost_by_operation_type={},
                total_input_tokens=0,
                total_output_tokens=0,
                total_latency_ms=0.0,
                operations=[],
                governance_attributes=self.governance_attributes,
                start_time=self.start_time,
                end_time=self.end_time
            )

        # Aggregate by model
        cost_by_model = {}
        for op in self.operations:
            cost_by_model[op.model_id] = cost_by_model.get(op.model_id, 0.0) + op.cost_usd

        # Aggregate by operation type
        cost_by_operation_type = {}
        for op in self.operations:
            cost_by_operation_type[op.operation_type] = (
                cost_by_operation_type.get(op.operation_type, 0.0) + op.cost_usd
            )

        summary = GeminiCostSummary(
            total_cost=self.total_cost,
            currency="USD",
            total_operations=len(self.operations),
            unique_models={op.model_id for op in self.operations},
            cost_by_model=cost_by_model,
            cost_by_operation_type=cost_by_operation_type,
            total_input_tokens=sum(op.input_tokens for op in self.operations),
            total_output_tokens=sum(op.output_tokens for op in self.operations),
            total_latency_ms=sum(op.latency_ms for op in self.operations),
            operations=self.operations.copy(),
            governance_attributes=self.governance_attributes,
            start_time=self.start_time,
            end_time=self.end_time or time.time()
        )

        return summary

    def _check_budget_alerts(self) -> None:
        """Check budget thresholds and generate alerts."""
        if not self.budget_limit:
            return

        utilization = self.total_cost / self.budget_limit

        # Define alert thresholds
        thresholds = [
            (0.5, CostAlertLevel.INFO, "50% budget utilized"),
            (0.75, CostAlertLevel.WARNING, "75% budget utilized"),
            (0.9, CostAlertLevel.CRITICAL, "90% budget utilized"),
            (1.0, CostAlertLevel.BUDGET_EXCEEDED, "Budget exceeded")
        ]

        for threshold, alert_level, message in thresholds:
            if utilization >= threshold and alert_level not in self.budget_warnings_sent:
                self._create_budget_alert(alert_level, message, utilization)
                self.budget_warnings_sent.add(alert_level)

    def _create_budget_alert(self, level: CostAlertLevel, message: str, utilization: float) -> None:
        """Create a budget alert."""
        alert = {
            "level": level.value,
            "message": message,
            "current_cost": self.total_cost,
            "budget_limit": self.budget_limit,
            "utilization_percent": utilization * 100,
            "timestamp": time.time(),
            "context_id": self.context_id,
            "operations_count": len(self.operations)
        }

        self.cost_alerts.append(alert)

        # Log the alert
        log_level = logging.WARNING if level in [CostAlertLevel.CRITICAL, CostAlertLevel.BUDGET_EXCEEDED] else logging.INFO
        logger.log(log_level, f"Budget alert [{level.value}]: {message} (${self.total_cost:.4f}/${self.budget_limit:.4f})")

    def _generate_optimization_recommendations(self, summary: GeminiCostSummary) -> None:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Check for opportunities to use cheaper models
        if "gemini-2.5-pro" in summary.cost_by_model and summary.cost_by_model["gemini-2.5-pro"] > 0:
            pro_cost = summary.cost_by_model["gemini-2.5-pro"]
            flash_cost_estimate = pro_cost * 0.24  # Flash is ~24% of Pro cost
            savings = pro_cost - flash_cost_estimate
            if savings > 0.001:  # Meaningful savings
                recommendations.append(
                    f"Consider using Gemini 2.5 Flash instead of Pro for some operations (potential savings: ${savings:.4f})"
                )

        # Check for high token usage patterns
        avg_tokens_per_op = (summary.total_input_tokens + summary.total_output_tokens) / summary.total_operations
        if avg_tokens_per_op > 2000:
            recommendations.append(
                f"High token usage detected ({avg_tokens_per_op:.0f} avg tokens/op). Consider prompt optimization or context caching"
            )

        # Check for single-model usage (missed optimization opportunities)
        if len(summary.unique_models) == 1 and summary.total_operations > 5:
            recommendations.append(
                "Using single model for all operations. Consider task-specific model selection for cost optimization"
            )

        # Check latency vs cost trade-offs
        avg_latency = summary.get_average_latency_ms()
        if avg_latency > 3000 and "gemini-2.5-flash-lite" not in summary.unique_models:
            recommendations.append(
                f"High latency detected ({avg_latency:.0f}ms avg). Consider Gemini 2.5 Flash-Lite for faster, cheaper operations"
            )

        summary.optimization_recommendations = recommendations

    def get_model_performance_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Compare performance metrics across models used in this context.
        
        Returns:
            Dictionary with performance metrics by model
        """
        model_stats = {}

        for model in self.get_current_summary().unique_models:
            model_operations = [op for op in self.operations if op.model_id == model]

            if model_operations:
                total_cost = sum(op.cost_usd for op in model_operations)
                total_tokens = sum(op.input_tokens + op.output_tokens for op in model_operations)
                avg_latency = sum(op.latency_ms for op in model_operations) / len(model_operations)
                cost_per_1k_tokens = (total_cost / total_tokens) * 1000 if total_tokens > 0 else 0

                model_stats[model] = {
                    "operations_count": len(model_operations),
                    "total_cost": total_cost,
                    "average_cost_per_operation": total_cost / len(model_operations),
                    "total_tokens": total_tokens,
                    "cost_per_1k_tokens": cost_per_1k_tokens,
                    "average_latency_ms": avg_latency,
                    "cost_efficiency_score": cost_per_1k_tokens / avg_latency * 1000  # Lower is better
                }

        return model_stats


@contextmanager
def create_gemini_cost_context(
    context_id: str,
    budget_limit: Optional[float] = None,
    enable_optimization: bool = True,
    enable_alerts: bool = True,
    **governance_attributes
) -> Iterator[GeminiCostContext]:
    """
    Create a Gemini cost tracking context manager.
    
    Args:
        context_id: Unique identifier for this cost context
        budget_limit: Maximum cost limit in USD
        enable_optimization: Enable automatic optimization recommendations
        enable_alerts: Enable budget alert monitoring
        **governance_attributes: Default governance attributes for all operations
    
    Yields:
        GeminiCostContext instance for cost tracking
    
    Usage:
        with create_gemini_cost_context("ai_analysis", budget_limit=5.00) as context:
            # Operations are automatically tracked
            context.add_operation(
                operation_id="op1",
                model_id="gemini-2.5-flash",
                input_tokens=1000,
                output_tokens=500,
                latency_ms=800
            )
            
            summary = context.get_current_summary()
            print(f"Current cost: ${summary.total_cost:.6f}")
    """
    context = GeminiCostContext(
        context_id=context_id,
        budget_limit=budget_limit,
        enable_optimization=enable_optimization,
        enable_alerts=enable_alerts,
        governance_attributes=governance_attributes
    )

    try:
        with context:
            yield context
    except Exception as e:
        logger.error(f"Error in Gemini cost context {context_id}: {e}")
        raise


def aggregate_multiple_contexts(contexts: List[GeminiCostContext]) -> GeminiCostSummary:
    """
    Aggregate cost summaries from multiple cost contexts.
    
    Args:
        contexts: List of GeminiCostContext instances
    
    Returns:
        Aggregated GeminiCostSummary across all contexts
    """
    all_operations = []
    total_cost = 0.0
    all_governance_attrs = {}

    for context in contexts:
        all_operations.extend(context.operations)
        total_cost += context.total_cost
        all_governance_attrs.update(context.governance_attributes)

    if not all_operations:
        return GeminiCostSummary(
            total_cost=0.0,
            currency="USD",
            total_operations=0,
            unique_models=set(),
            cost_by_model={},
            cost_by_operation_type={},
            total_input_tokens=0,
            total_output_tokens=0,
            total_latency_ms=0.0,
            operations=[],
            governance_attributes=all_governance_attrs
        )

    # Aggregate by model and operation type
    cost_by_model = {}
    cost_by_operation_type = {}

    for op in all_operations:
        cost_by_model[op.model_id] = cost_by_model.get(op.model_id, 0.0) + op.cost_usd
        cost_by_operation_type[op.operation_type] = (
            cost_by_operation_type.get(op.operation_type, 0.0) + op.cost_usd
        )

    return GeminiCostSummary(
        total_cost=total_cost,
        currency="USD",
        total_operations=len(all_operations),
        unique_models={op.model_id for op in all_operations},
        cost_by_model=cost_by_model,
        cost_by_operation_type=cost_by_operation_type,
        total_input_tokens=sum(op.input_tokens for op in all_operations),
        total_output_tokens=sum(op.output_tokens for op in all_operations),
        total_latency_ms=sum(op.latency_ms for op in all_operations),
        operations=all_operations,
        governance_attributes=all_governance_attrs,
        start_time=min(ctx.start_time for ctx in contexts if ctx.start_time),
        end_time=max(ctx.end_time for ctx in contexts if ctx.end_time)
    )


# Export main classes and functions
__all__ = [
    'GeminiOperation',
    'GeminiCostSummary',
    'GeminiCostContext',
    'CostAlertLevel',
    'create_gemini_cost_context',
    'aggregate_multiple_contexts'
]
