#!/usr/bin/env python3
"""
GenOps Replicate Cost Aggregator

Advanced cost tracking and aggregation for complex Replicate workflows involving
multiple models, batch operations, and multi-modal processing. Provides intelligent
cost optimization recommendations and unified governance across model types.

Features:
- Multi-model cost aggregation with category breakdowns
- Context managers for workflow-level cost tracking
- Real-time budget monitoring and alerts
- Intelligent model selection based on cost/performance trade-offs
- Batch processing optimization
- Cross-modal cost comparisons and recommendations

Usage:
    from genops.providers.replicate_cost_aggregator import create_replicate_cost_context
    
    # Workflow-level cost tracking
    with create_replicate_cost_context("multi_modal_workflow", budget_limit=10.0) as context:
        # Text generation
        context.add_operation("text", "meta/llama-2-70b-chat", input_tokens=1000, output_tokens=500, cost=0.75)
        
        # Image generation  
        context.add_operation("image", "black-forest-labs/flux-pro", num_images=3, cost=0.12)
        
        # Get optimization recommendations
        summary = context.get_current_summary()
        print(f"Total cost: ${summary.total_cost:.4f}")
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Iterator, Set, Tuple
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

@dataclass
class ReplicateOperation:
    """Individual Replicate operation within a cost context."""
    
    operation_id: str
    model: str
    category: str  # 'text', 'image', 'video', 'audio', 'multimodal'
    cost_usd: float
    timestamp: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    output_units: Optional[int] = None  # images, videos, etc.
    latency_ms: Optional[float] = None
    hardware_type: Optional[str] = None
    governance_attributes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.governance_attributes is None:
            self.governance_attributes = {}

@dataclass
class ReplicateCostSummary:
    """Comprehensive cost summary for Replicate operations."""
    
    total_cost: float
    operation_count: int
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_category: Dict[str, float] = field(default_factory=dict)
    unique_models: Set[str] = field(default_factory=set)
    unique_categories: Set[str] = field(default_factory=set)
    total_tokens: int = 0
    total_output_units: int = 0
    total_time_ms: float = 0.0
    most_expensive_model: Optional[str] = None
    cheapest_model: Optional[str] = None
    optimization_recommendations: List[str] = field(default_factory=list)
    budget_status: Optional[Dict[str, Any]] = None
    efficiency_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.cost_by_model:
            self.most_expensive_model = max(self.cost_by_model.items(), key=lambda x: x[1])[0]
            self.cheapest_model = min(self.cost_by_model.items(), key=lambda x: x[1])[0]

@dataclass
class BudgetAlert:
    """Budget monitoring alert with specific details."""
    
    alert_type: str  # 'warning', 'critical', 'exceeded'
    current_cost: float
    budget_limit: float
    percentage_used: float
    remaining_budget: float
    projected_cost: Optional[float] = None
    recommendation: Optional[str] = None

class ReplicateCostAggregator:
    """
    Advanced cost aggregator for Replicate operations.
    
    Tracks costs across multiple models and categories, provides intelligent
    optimization recommendations, and manages budget constraints.
    """
    
    def __init__(
        self,
        context_name: str,
        budget_limit: Optional[float] = None,
        enable_alerts: bool = True,
        optimization_threshold: float = 0.10  # 10% potential savings
    ):
        """
        Initialize cost aggregator with budget controls.
        
        Args:
            context_name: Name for this cost tracking context
            budget_limit: Maximum allowed cost in USD
            enable_alerts: Enable budget alerts and warnings
            optimization_threshold: Minimum savings threshold for recommendations
        """
        self.context_name = context_name
        self.context_id = str(uuid.uuid4())
        self.budget_limit = budget_limit
        self.enable_alerts = enable_alerts
        self.optimization_threshold = optimization_threshold
        
        # Tracking state
        self.operations: List[ReplicateOperation] = []
        self.start_time = time.time()
        self.total_cost = 0.0
        self.alerts: List[BudgetAlert] = []
        
        # Performance metrics
        self._cost_by_model: Dict[str, float] = defaultdict(float)
        self._cost_by_category: Dict[str, float] = defaultdict(float)
        self._operation_count_by_model: Dict[str, int] = defaultdict(int)
        
        # Import pricing calculator for recommendations
        try:
            from .replicate_pricing import ReplicatePricingCalculator
            self._pricing_calculator = ReplicatePricingCalculator()
        except ImportError:
            logger.warning("Replicate pricing calculator not available")
            self._pricing_calculator = None
    
    def add_operation(
        self,
        model: str,
        category: str,
        cost_usd: float,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        output_units: Optional[int] = None,
        latency_ms: Optional[float] = None,
        hardware_type: Optional[str] = None,
        **governance_attrs
    ) -> str:
        """
        Add a Replicate operation to the cost context.
        
        Args:
            model: Replicate model name
            category: Model category (text, image, video, audio, multimodal)
            cost_usd: Operation cost in USD
            input_tokens: Number of input tokens (for text models)
            output_tokens: Number of output tokens (for text models)
            output_units: Number of output units (images, videos, etc.)
            latency_ms: Operation latency in milliseconds
            hardware_type: Hardware used for operation
            **governance_attrs: Additional governance attributes
            
        Returns:
            Operation ID for reference
        """
        operation_id = str(uuid.uuid4())
        
        # Create operation record
        operation = ReplicateOperation(
            operation_id=operation_id,
            model=model,
            category=category,
            cost_usd=cost_usd,
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_units=output_units,
            latency_ms=latency_ms,
            hardware_type=hardware_type,
            governance_attributes=governance_attrs
        )
        
        # Add to tracking
        self.operations.append(operation)
        self.total_cost += cost_usd
        
        # Update aggregated metrics
        self._cost_by_model[model] += cost_usd
        self._cost_by_category[category] += cost_usd
        self._operation_count_by_model[model] += 1
        
        # Check budget constraints
        if self.enable_alerts and self.budget_limit:
            self._check_budget_alerts()
        
        # Record telemetry
        with tracer.start_as_current_span("replicate.cost_aggregation") as span:
            span.set_attributes({
                "genops.context_name": self.context_name,
                "genops.context_id": self.context_id,
                "genops.operation_id": operation_id,
                "genops.model": model,
                "genops.category": category,
                "genops.cost_usd": cost_usd,
                "genops.total_cost": self.total_cost,
                "genops.operation_count": len(self.operations)
            })
        
        return operation_id
    
    def get_current_summary(self) -> ReplicateCostSummary:
        """
        Get comprehensive cost summary for current operations.
        
        Returns:
            ReplicateCostSummary with detailed breakdown and recommendations
        """
        if not self.operations:
            return ReplicateCostSummary(
                total_cost=0.0,
                operation_count=0
            )
        
        # Calculate aggregated metrics
        total_tokens = sum(
            (op.input_tokens or 0) + (op.output_tokens or 0)
            for op in self.operations
        )
        
        total_output_units = sum(
            op.output_units or 0 for op in self.operations
        )
        
        total_time_ms = sum(
            op.latency_ms or 0 for op in self.operations
        )
        
        unique_models = set(op.model for op in self.operations)
        unique_categories = set(op.category for op in self.operations)
        
        # Create summary
        summary = ReplicateCostSummary(
            total_cost=self.total_cost,
            operation_count=len(self.operations),
            cost_by_model=dict(self._cost_by_model),
            cost_by_category=dict(self._cost_by_category),
            unique_models=unique_models,
            unique_categories=unique_categories,
            total_tokens=total_tokens,
            total_output_units=total_output_units,
            total_time_ms=total_time_ms
        )
        
        # Add budget information
        if self.budget_limit:
            percentage_used = (self.total_cost / self.budget_limit) * 100
            remaining = self.budget_limit - self.total_cost
            
            summary.budget_status = {
                "budget_limit": self.budget_limit,
                "percentage_used": percentage_used,
                "remaining_budget": remaining,
                "alerts": [asdict(alert) for alert in self.alerts]
            }
        
        # Calculate efficiency metrics
        summary.efficiency_metrics = self._calculate_efficiency_metrics()
        
        # Generate optimization recommendations
        summary.optimization_recommendations = self._generate_optimization_recommendations()
        
        return summary
    
    def _check_budget_alerts(self):
        """Check budget constraints and generate alerts."""
        
        if not self.budget_limit:
            return
        
        percentage_used = (self.total_cost / self.budget_limit) * 100
        remaining = self.budget_limit - self.total_cost
        
        # Clear previous alerts for fresh assessment
        self.alerts = []
        
        if self.total_cost >= self.budget_limit:
            # Budget exceeded
            self.alerts.append(BudgetAlert(
                alert_type="exceeded",
                current_cost=self.total_cost,
                budget_limit=self.budget_limit,
                percentage_used=percentage_used,
                remaining_budget=remaining,
                recommendation="Stop operations immediately - budget exceeded"
            ))
        elif percentage_used >= 90:
            # Critical warning (90%+ used)
            self.alerts.append(BudgetAlert(
                alert_type="critical",
                current_cost=self.total_cost,
                budget_limit=self.budget_limit,
                percentage_used=percentage_used,
                remaining_budget=remaining,
                recommendation="Approaching budget limit - review remaining operations"
            ))
        elif percentage_used >= 75:
            # Warning (75%+ used)
            self.alerts.append(BudgetAlert(
                alert_type="warning",
                current_cost=self.total_cost,
                budget_limit=self.budget_limit,
                percentage_used=percentage_used,
                remaining_budget=remaining,
                recommendation="Budget 75% consumed - monitor remaining operations"
            ))
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics for optimization insights."""
        
        if not self.operations:
            return {}
        
        metrics = {}
        
        # Cost per operation by category
        for category in self._cost_by_category:
            category_ops = [op for op in self.operations if op.category == category]
            if category_ops:
                avg_cost = self._cost_by_category[category] / len(category_ops)
                metrics[f"avg_cost_per_{category}_operation"] = avg_cost
        
        # Token efficiency for text models
        text_ops = [op for op in self.operations if op.category == "text"]
        if text_ops:
            total_text_tokens = sum((op.input_tokens or 0) + (op.output_tokens or 0) for op in text_ops)
            total_text_cost = sum(op.cost_usd for op in text_ops)
            
            if total_text_tokens > 0:
                metrics["cost_per_1k_tokens"] = (total_text_cost / total_text_tokens) * 1000
        
        # Output efficiency for generative models
        image_ops = [op for op in self.operations if op.category == "image"]
        if image_ops:
            total_images = sum(op.output_units or 1 for op in image_ops)
            total_image_cost = sum(op.cost_usd for op in image_ops)
            metrics["cost_per_image"] = total_image_cost / total_images
        
        # Latency efficiency
        timed_ops = [op for op in self.operations if op.latency_ms]
        if timed_ops:
            total_latency = sum(op.latency_ms for op in timed_ops)
            total_cost_timed = sum(op.cost_usd for op in timed_ops)
            metrics["cost_per_second"] = total_cost_timed / (total_latency / 1000)
        
        return metrics
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate intelligent cost optimization recommendations."""
        
        recommendations = []
        
        if not self.operations:
            return recommendations
        
        # Model distribution analysis
        if len(self._cost_by_model) > 1:
            most_expensive = max(self._cost_by_model.items(), key=lambda x: x[1])
            if most_expensive[1] > self.total_cost * 0.5:
                recommendations.append(
                    f"Model {most_expensive[0]} accounts for {most_expensive[1]/self.total_cost*100:.1f}% "
                    f"of costs - consider alternatives"
                )
        
        # Category-specific recommendations
        for category, cost in self._cost_by_category.items():
            category_ops = [op for op in self.operations if op.category == category]
            
            if category == "text" and len(category_ops) > 1:
                # Check for token efficiency
                avg_tokens_per_op = sum(
                    (op.input_tokens or 0) + (op.output_tokens or 0) 
                    for op in category_ops
                ) / len(category_ops)
                
                if avg_tokens_per_op > 1000:
                    recommendations.append(
                        "High token usage detected - consider breaking large prompts into smaller chunks"
                    )
            
            elif category == "image" and len(category_ops) > 5:
                recommendations.append(
                    f"Multiple image generations ({len(category_ops)}) - "
                    f"consider batch processing for efficiency"
                )
        
        # Budget-based recommendations
        if self.budget_limit and self.total_cost > self.budget_limit * 0.8:
            recommendations.append(
                "Approaching budget limit - prioritize essential operations only"
            )
        
        # Model alternatives (if pricing calculator available)
        if self._pricing_calculator:
            for model in self._cost_by_model:
                alternatives = self._pricing_calculator.get_model_alternatives(model)
                if alternatives:
                    cheaper_model, cost_ratio, reason = alternatives[0]
                    if cost_ratio < 0.7:  # 30% savings
                        recommendations.append(
                            f"Consider {cheaper_model} instead of {model} - {reason}"
                        )
        
        # Efficiency recommendations
        efficiency = self._calculate_efficiency_metrics()
        if "cost_per_1k_tokens" in efficiency:
            cost_per_1k = efficiency["cost_per_1k_tokens"]
            if cost_per_1k > 2.0:  # High token cost
                recommendations.append(
                    f"High token cost (${cost_per_1k:.2f}/1K) - consider more efficient models"
                )
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_model_performance(self, model: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific model."""
        
        model_ops = [op for op in self.operations if op.model == model]
        if not model_ops:
            return None
        
        total_cost = sum(op.cost_usd for op in model_ops)
        avg_latency = sum(op.latency_ms or 0 for op in model_ops) / len(model_ops)
        
        return {
            "model": model,
            "operation_count": len(model_ops),
            "total_cost": total_cost,
            "average_cost": total_cost / len(model_ops),
            "average_latency_ms": avg_latency,
            "cost_percentage": (total_cost / self.total_cost) * 100 if self.total_cost > 0 else 0
        }
    
    def export_summary(self) -> Dict[str, Any]:
        """Export complete summary for external analysis."""
        
        summary = self.get_current_summary()
        
        export_data = {
            "context_info": {
                "name": self.context_name,
                "id": self.context_id,
                "start_time": self.start_time,
                "duration_seconds": time.time() - self.start_time,
                "budget_limit": self.budget_limit
            },
            "cost_summary": asdict(summary),
            "operations": [asdict(op) for op in self.operations],
            "model_performance": {
                model: self.get_model_performance(model)
                for model in self._cost_by_model
            }
        }
        
        return export_data

@contextmanager
def create_replicate_cost_context(
    context_name: str,
    budget_limit: Optional[float] = None,
    enable_alerts: bool = True,
    **kwargs
) -> Iterator[ReplicateCostAggregator]:
    """
    Create a cost tracking context for Replicate operations.
    
    This context manager provides automatic cost aggregation and budget
    monitoring for complex workflows involving multiple Replicate models.
    
    Args:
        context_name: Descriptive name for the workflow
        budget_limit: Maximum allowed cost in USD
        enable_alerts: Enable budget monitoring alerts
        **kwargs: Additional configuration options
        
    Yields:
        ReplicateCostAggregator instance for tracking operations
        
    Example:
        with create_replicate_cost_context("multi_modal_pipeline", budget_limit=5.0) as context:
            # Text processing
            context.add_operation("meta/llama-2-70b-chat", "text", cost=0.50)
            
            # Image generation
            context.add_operation("black-forest-labs/flux-pro", "image", cost=0.08)
            
            # Get final summary
            summary = context.get_current_summary()
            print(f"Total workflow cost: ${summary.total_cost:.4f}")
    """
    
    # Create aggregator
    aggregator = ReplicateCostAggregator(
        context_name=context_name,
        budget_limit=budget_limit,
        enable_alerts=enable_alerts,
        **kwargs
    )
    
    with tracer.start_as_current_span(
        "replicate.cost_context",
        attributes={
            "genops.context_name": context_name,
            "genops.context_id": aggregator.context_id,
            "genops.budget_limit": budget_limit or 0
        }
    ) as span:
        
        try:
            yield aggregator
            
            # Record final metrics
            final_summary = aggregator.get_current_summary()
            span.set_attributes({
                "genops.total_cost": final_summary.total_cost,
                "genops.operation_count": final_summary.operation_count,
                "genops.unique_models": len(final_summary.unique_models),
                "genops.success": True
            })
            
            # Log completion
            logger.info(
                f"Replicate cost context '{context_name}' completed: "
                f"${final_summary.total_cost:.4f} across {final_summary.operation_count} operations"
            )
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            logger.error(f"Error in Replicate cost context '{context_name}': {e}")
            raise

# Export main classes and functions
__all__ = [
    'ReplicateCostAggregator',
    'create_replicate_cost_context',
    'ReplicateOperation',
    'ReplicateCostSummary',
    'BudgetAlert'
]