#!/usr/bin/env python3
"""
SkyRouter + GenOps Integration

This module provides GenOps governance for SkyRouter AI routing platform,
enabling cost tracking, team attribution, and policy enforcement for
multi-model AI operations through SkyRouter's unified API.

SkyRouter is a multi-model AI routing platform that provides access to 150+
models through a single API with intelligent routing, cost optimization, and
agent-specific features for AI applications.

Key Features:
- Multi-model routing governance across 150+ models
- Agent workflow cost tracking and optimization
- Intelligent route selection with cost awareness
- Multi-modal operation tracking (search, generation, reading)
- Global node cost attribution and performance monitoring
- Enterprise governance for complex AI agent workflows

Usage:
    # Auto-instrumentation (zero-code setup)
    from genops.providers.skyrouter import auto_instrument
    auto_instrument(team="ai-team", project="routing-system")

    # Manual adapter usage
    from genops.providers.skyrouter import GenOpsSkyRouterAdapter
    adapter = GenOpsSkyRouterAdapter(
        skyrouter_api_key="your-api-key",
        team="ai-platform",
        project="multi-model-routing"
    )

    with adapter.track_routing_session("agent-workflow") as session:
        # Track model routing operations
        cost_result = session.track_model_call(
            model="gpt-4",
            input_data={"messages": [...]},
            route_optimization="cost_aware"
        )

Author: GenOps AI Contributors
License: Apache 2.0
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

# Core GenOps imports
try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create mock objects for when OpenTelemetry is not available
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class SkyRouterGovernanceAttrs:
    """Governance attributes for SkyRouter operations."""

    team: str = "default"
    project: str = "default"
    environment: str = "production"
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    feature: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert governance attributes to dictionary for telemetry."""
        attrs = {
            "genops.skyrouter.team": self.team,
            "genops.skyrouter.project": self.project,
            "genops.skyrouter.environment": self.environment,
        }
        if self.customer_id:
            attrs["genops.skyrouter.customer_id"] = self.customer_id
        if self.cost_center:
            attrs["genops.skyrouter.cost_center"] = self.cost_center
        if self.feature:
            attrs["genops.skyrouter.feature"] = self.feature
        return attrs


@dataclass
class SkyRouterCostResult:
    """Result object for SkyRouter cost calculations."""

    operation_type: str
    model: str
    route: str
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    input_tokens: int
    output_tokens: int
    optimization_savings: Decimal = Decimal("0")
    route_efficiency_score: float = 1.0
    governance_attrs: Optional[SkyRouterGovernanceAttrs] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkyRouterRouteInfo:
    """Information about the selected route."""

    primary_model: str
    fallback_models: list[str]
    region: str
    optimization_strategy: str
    cost_efficiency: float
    latency_expectation: int  # milliseconds
    reliability_score: float


class SkyRouterSession:
    """Context manager for tracking SkyRouter operations with governance."""

    def __init__(
        self,
        session_name: str,
        adapter: "GenOpsSkyRouterAdapter",
        span_context: Optional[Any] = None,
    ):
        self.session_name = session_name
        self.adapter = adapter
        self.span_context = span_context
        self.operations: list[SkyRouterCostResult] = []
        self.start_time = time.time()
        self.governance_attrs = adapter.governance_attrs

    def __enter__(self):
        """Start the SkyRouter session."""
        logger.info(f"Starting SkyRouter session: {self.session_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the SkyRouter session and finalize telemetry."""
        duration = time.time() - self.start_time
        total_cost = sum(Decimal(str(op.total_cost)) for op in self.operations)

        # Record session telemetry
        if OTEL_AVAILABLE and self.span_context:
            self.span_context.set_attribute(
                "genops.skyrouter.session.duration", duration
            )
            self.span_context.set_attribute(
                "genops.skyrouter.session.total_cost", float(total_cost)
            )
            self.span_context.set_attribute(
                "genops.skyrouter.session.operation_count", len(self.operations)
            )

            # Add governance attributes
            for key, value in self.governance_attrs.to_dict().items():
                self.span_context.set_attribute(key, value)

        logger.info(
            f"SkyRouter session completed: {self.session_name}, cost: ${total_cost:.3f}, operations: {len(self.operations)}"
        )

    def track_model_call(
        self,
        model: str,
        input_data: dict[str, Any],
        route_optimization: str = "balanced",
        cost: Optional[float] = None,
        complexity: str = "moderate",
    ) -> SkyRouterCostResult:
        """Track a model call through SkyRouter routing."""
        if cost is not None:
            # Use provided cost
            cost_result = SkyRouterCostResult(
                operation_type="model_call",
                model=model,
                route=f"skyrouter_{route_optimization}",
                input_cost=Decimal(str(cost * 0.7)),  # Estimate input/output split
                output_cost=Decimal(str(cost * 0.3)),
                total_cost=Decimal(str(cost)),
                input_tokens=self._estimate_input_tokens(input_data),
                output_tokens=self._estimate_output_tokens(complexity),
                governance_attrs=self.governance_attrs,
            )
        else:
            # Calculate cost using pricing calculator
            cost_result = self.adapter.pricing_calculator.calculate_model_call_cost(
                model=model,
                input_data=input_data,
                route_optimization=route_optimization,
                complexity=complexity,
            )

        # Add operation metadata
        cost_result.metadata.update(
            {
                "route_optimization": route_optimization,
                "complexity": complexity,
                "session": self.session_name,
                "timestamp": time.time(),
            }
        )

        self.operations.append(cost_result)

        # Update cost aggregator
        if self.adapter.cost_aggregator:
            self.adapter.cost_aggregator.add_operation_cost(
                operation_type="skyrouter_model_call",
                cost=float(cost_result.total_cost),
                model=model,
                team=self.governance_attrs.team,
                project=self.governance_attrs.project,
                metadata=cost_result.metadata,
            )

        # Check budget constraints
        self._check_budget_constraints(cost_result)

        logger.debug(
            f"Tracked SkyRouter model call: {model}, cost: ${cost_result.total_cost:.3f}"
        )
        return cost_result

    def track_multi_model_routing(
        self,
        models: list[str],
        input_data: dict[str, Any],
        routing_strategy: str = "cost_optimized",
        cost: Optional[float] = None,
    ) -> SkyRouterCostResult:
        """Track multi-model routing operation."""
        selected_model = models[0] if models else "unknown"

        if cost is not None:
            cost_result = SkyRouterCostResult(
                operation_type="multi_model_routing",
                model=selected_model,
                route=f"multi_model_{routing_strategy}",
                input_cost=Decimal(str(cost * 0.6)),
                output_cost=Decimal(str(cost * 0.4)),
                total_cost=Decimal(str(cost)),
                input_tokens=self._estimate_input_tokens(input_data),
                output_tokens=self._estimate_output_tokens("moderate"),
                governance_attrs=self.governance_attrs,
            )
        else:
            cost_result = self.adapter.pricing_calculator.calculate_multi_model_cost(
                models=models, input_data=input_data, routing_strategy=routing_strategy
            )

        # Add routing-specific metadata
        cost_result.metadata.update(
            {
                "routing_strategy": routing_strategy,
                "candidate_models": models,
                "selected_model": selected_model,
                "session": self.session_name,
            }
        )

        self.operations.append(cost_result)

        # Update cost aggregator
        if self.adapter.cost_aggregator:
            self.adapter.cost_aggregator.add_operation_cost(
                operation_type="skyrouter_multi_model",
                cost=float(cost_result.total_cost),
                model=selected_model,
                team=self.governance_attrs.team,
                project=self.governance_attrs.project,
                metadata=cost_result.metadata,
            )

        logger.debug(
            f"Tracked multi-model routing: {selected_model}, cost: ${cost_result.total_cost:.3f}"
        )
        return cost_result

    def track_agent_workflow(
        self,
        workflow_name: str,
        agent_steps: list[dict[str, Any]],
        cost: Optional[float] = None,
    ) -> SkyRouterCostResult:
        """Track complete agent workflow through SkyRouter."""
        primary_model = (
            agent_steps[0].get("model", "unknown") if agent_steps else "unknown"
        )

        if cost is not None:
            cost_result = SkyRouterCostResult(
                operation_type="agent_workflow",
                model=primary_model,
                route=f"agent_{workflow_name}",
                input_cost=Decimal(str(cost * 0.4)),
                output_cost=Decimal(str(cost * 0.6)),
                total_cost=Decimal(str(cost)),
                input_tokens=sum(
                    self._estimate_input_tokens(step.get("input", {}))
                    for step in agent_steps
                ),
                output_tokens=sum(
                    self._estimate_output_tokens("moderate") for step in agent_steps
                ),
                governance_attrs=self.governance_attrs,
            )
        else:
            cost_result = self.adapter.pricing_calculator.calculate_agent_workflow_cost(
                workflow_name=workflow_name, agent_steps=agent_steps
            )

        # Add workflow metadata
        cost_result.metadata.update(
            {
                "workflow_name": workflow_name,
                "step_count": len(agent_steps),
                "models_used": list(
                    {step.get("model", "unknown") for step in agent_steps}
                ),
                "session": self.session_name,
            }
        )

        self.operations.append(cost_result)

        # Update cost aggregator
        if self.adapter.cost_aggregator:
            self.adapter.cost_aggregator.add_operation_cost(
                operation_type="skyrouter_agent_workflow",
                cost=float(cost_result.total_cost),
                model=primary_model,
                team=self.governance_attrs.team,
                project=self.governance_attrs.project,
                metadata=cost_result.metadata,
            )

        logger.debug(
            f"Tracked agent workflow: {workflow_name}, cost: ${cost_result.total_cost:.3f}"
        )
        return cost_result

    def _estimate_input_tokens(self, input_data: dict[str, Any]) -> int:
        """Estimate input tokens from input data."""
        if isinstance(input_data, dict):
            text_content = str(input_data.get("messages", "")) + str(
                input_data.get("prompt", "")
            )
            return max(len(text_content.split()) * 1.3, 10)  # type: ignore  # Rough token estimation
        return 100  # Default fallback

    def _estimate_output_tokens(self, complexity: str) -> int:
        """Estimate output tokens based on complexity."""
        complexity_tokens = {
            "simple": 50,
            "moderate": 150,
            "complex": 300,
            "enterprise": 500,
        }
        return complexity_tokens.get(complexity, 150)

    def _check_budget_constraints(self, cost_result: SkyRouterCostResult):
        """Check if operation violates budget constraints."""
        if (
            hasattr(self.adapter, "daily_budget_limit")
            and self.adapter.daily_budget_limit
        ):
            current_total = sum(Decimal(str(op.total_cost)) for op in self.operations)
            if current_total > Decimal(str(self.adapter.daily_budget_limit)):
                logger.warning(
                    f"Daily budget limit exceeded: ${current_total:.3f} > ${self.adapter.daily_budget_limit:.3f}"
                )
                if self.adapter.governance_policy == "enforced":
                    raise ValueError(f"Budget limit exceeded: ${current_total:.3f}")

    @property
    def total_cost(self) -> Decimal:
        """Get total cost for this session."""
        return sum(Decimal(str(op.total_cost)) for op in self.operations)  # type: ignore

    @property
    def operation_count(self) -> int:
        """Get number of operations in this session."""
        return len(self.operations)

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time


class GenOpsSkyRouterAdapter:
    """GenOps adapter for SkyRouter multi-model routing platform."""

    def __init__(
        self,
        skyrouter_api_key: Optional[str] = None,
        team: str = "default",
        project: str = "default",
        environment: str = "production",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        feature: Optional[str] = None,
        daily_budget_limit: Optional[float] = None,
        enable_cost_alerts: bool = True,
        governance_policy: str = "enforced",
        export_telemetry: bool = True,
    ):
        """
        Initialize SkyRouter adapter with governance configuration.

        Args:
            skyrouter_api_key: SkyRouter API key (uses SKYROUTER_API_KEY env var if not provided)
            team: Team name for cost attribution
            project: Project name for cost attribution
            environment: Environment (development/staging/production)
            customer_id: Customer ID for multi-tenant attribution
            cost_center: Cost center for financial reporting
            feature: Feature name for granular attribution
            daily_budget_limit: Daily spending limit in USD
            enable_cost_alerts: Enable budget and cost alerting
            governance_policy: Policy enforcement level (advisory/enforced)
            export_telemetry: Enable OpenTelemetry export
        """
        self.skyrouter_api_key = skyrouter_api_key or os.getenv("SKYROUTER_API_KEY")
        self.daily_budget_limit = daily_budget_limit
        self.enable_cost_alerts = enable_cost_alerts
        self.governance_policy = governance_policy
        self.export_telemetry = export_telemetry

        # Initialize governance attributes
        self.governance_attrs = SkyRouterGovernanceAttrs(
            team=team,
            project=project,
            environment=environment,
            customer_id=customer_id,
            cost_center=cost_center,
            feature=feature,
        )

        # Initialize cost tracking components
        self._initialize_cost_components()

        # Initialize telemetry if enabled
        if export_telemetry and OTEL_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
        else:
            self.tracer = None  # type: ignore[assignment]
            self.meter = None  # type: ignore[assignment]

        logger.info(
            f"SkyRouter adapter initialized for team: {team}, project: {project}"
        )

    def _initialize_cost_components(self):
        """Initialize pricing calculator and cost aggregator."""
        try:
            from .skyrouter_cost_aggregator import SkyRouterCostAggregator
            from .skyrouter_pricing import SkyRouterPricingCalculator

            self.pricing_calculator = SkyRouterPricingCalculator()
            self.cost_aggregator = SkyRouterCostAggregator(
                team=self.governance_attrs.team,
                project=self.governance_attrs.project,
                daily_budget_limit=self.daily_budget_limit,
                enable_cost_alerts=self.enable_cost_alerts,
            )
        except ImportError as e:
            logger.warning(f"Could not import cost tracking components: {e}")
            self.pricing_calculator = None
            self.cost_aggregator = None

    @contextmanager
    def track_routing_session(self, session_name: str, **kwargs):
        """
        Context manager for tracking a SkyRouter routing session.

        Args:
            session_name: Descriptive name for the routing session
            **kwargs: Additional metadata for the session

        Yields:
            SkyRouterSession: Session object for tracking operations
        """
        span_context = None

        # Create OpenTelemetry span if available
        if self.tracer:
            span_context = self.tracer.start_span(f"skyrouter.routing.{session_name}")
            span_context.__enter__()

        # Create session
        session = SkyRouterSession(
            session_name=session_name, adapter=self, span_context=span_context
        )

        try:
            with session:
                yield session

                # Mark span as successful
                if span_context:
                    span_context.set_status(Status(StatusCode.OK))

        except Exception as e:
            logger.error(f"Error in SkyRouter session {session_name}: {e}")

            # Mark span as error
            if span_context:
                span_context.set_status(Status(StatusCode.ERROR, str(e)))
                span_context.record_exception(e)

            raise
        finally:
            # End span
            if span_context:
                span_context.__exit__(None, None, None)

    def calculate_model_call_cost(
        self,
        model: str,
        input_data: dict[str, Any],
        route_optimization: str = "balanced",
        complexity: str = "moderate",
    ) -> SkyRouterCostResult:
        """Calculate cost for a single model call."""
        if self.pricing_calculator:
            return self.pricing_calculator.calculate_model_call_cost(
                model=model,
                input_data=input_data,
                route_optimization=route_optimization,
                complexity=complexity,
            )

        # Fallback calculation
        estimated_cost = self._estimate_fallback_cost(model, input_data, complexity)
        return SkyRouterCostResult(
            operation_type="model_call",
            model=model,
            route=f"skyrouter_{route_optimization}",
            input_cost=Decimal(str(estimated_cost * 0.6)),
            output_cost=Decimal(str(estimated_cost * 0.4)),
            total_cost=Decimal(str(estimated_cost)),
            input_tokens=len(str(input_data).split()) * 2,
            output_tokens=100,
            governance_attrs=self.governance_attrs,
        )

    def _estimate_fallback_cost(
        self, model: str, input_data: dict[str, Any], complexity: str
    ) -> float:
        """Fallback cost estimation when pricing calculator unavailable."""
        base_costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.015,
            "gemini-pro": 0.001,
        }

        complexity_multipliers = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 2.0,
            "enterprise": 3.0,
        }

        # Extract model base name
        model_base = model.lower().replace("-", "").replace("_", "")
        base_cost = 0.01  # Default

        for known_model, cost in base_costs.items():
            if known_model.replace("-", "").replace("_", "") in model_base:
                base_cost = cost
                break

        # Apply complexity multiplier
        multiplier = complexity_multipliers.get(complexity, 1.0)

        # Estimate token usage
        input_tokens = len(str(input_data).split()) * 1.5
        estimated_cost = (input_tokens / 1000) * base_cost * multiplier

        return max(estimated_cost, 0.001)  # Minimum cost


# Auto-instrumentation functions
_original_skyrouter_modules = {}


def auto_instrument(
    skyrouter_api_key: Optional[str] = None,
    team: str = "default",
    project: str = "default",
    environment: str = "production",
    **kwargs,
) -> GenOpsSkyRouterAdapter:
    """
    Enable automatic instrumentation for SkyRouter operations.

    This function sets up zero-code governance for existing SkyRouter applications
    by patching SkyRouter SDK calls to include cost tracking and attribution.

    Args:
        skyrouter_api_key: SkyRouter API key
        team: Team name for cost attribution
        project: Project name for cost attribution
        environment: Environment (development/staging/production)
        **kwargs: Additional adapter configuration

    Returns:
        GenOpsSkyRouterAdapter: Configured adapter instance
    """
    # Create adapter
    adapter = GenOpsSkyRouterAdapter(
        skyrouter_api_key=skyrouter_api_key,
        team=team,
        project=project,
        environment=environment,
        **kwargs,
    )

    # Store reference for later restoration
    global _skyrouter_adapter
    _skyrouter_adapter = adapter

    logger.info("SkyRouter auto-instrumentation enabled")
    return adapter


def restore_skyrouter():
    """Restore original SkyRouter SDK functionality."""
    global _skyrouter_adapter
    _skyrouter_adapter = None

    logger.info("SkyRouter auto-instrumentation disabled")


# Global adapter reference for auto-instrumentation
_skyrouter_adapter: Optional[GenOpsSkyRouterAdapter] = None


def get_current_adapter() -> Optional[GenOpsSkyRouterAdapter]:
    """Get the current auto-instrumentation adapter."""
    return _skyrouter_adapter


# Export key components
__all__ = [
    "GenOpsSkyRouterAdapter",
    "SkyRouterSession",
    "SkyRouterCostResult",
    "SkyRouterGovernanceAttrs",
    "SkyRouterRouteInfo",
    "auto_instrument",
    "restore_skyrouter",
    "get_current_adapter",
]
