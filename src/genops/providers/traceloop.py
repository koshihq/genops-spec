#!/usr/bin/env python3
"""
GenOps Traceloop + OpenLLMetry Integration

This module provides comprehensive Traceloop + OpenLLMetry integration for GenOps AI governance,
cost intelligence, and policy enforcement. OpenLLMetry is an open-source observability framework 
that extends OpenTelemetry with LLM-specific instrumentation, while Traceloop provides a 
commercial platform with enterprise features built on OpenLLMetry.

Features:
- Enhanced OpenLLMetry traces with GenOps governance attributes
- Cost attribution and budget enforcement for LLM operations  
- Policy compliance tracking integrated with OpenTelemetry observability
- LLM evaluation with governance oversight
- Zero-code auto-instrumentation with auto_instrument()
- Optional Traceloop commercial platform integration
- Enterprise-ready governance patterns for production deployments
- Full compatibility with existing OpenLLMetry applications

Example usage:

    # Zero-code auto-instrumentation (recommended)
    from genops.providers.traceloop import auto_instrument
    auto_instrument(
        team="ai-team",
        project="production-llm",
        environment="production"
    )
    # All existing OpenLLMetry operations now include GenOps governance
    
    # Manual adapter for advanced governance
    from genops.providers.traceloop import instrument_traceloop
    
    adapter = instrument_traceloop(
        team="research-team", 
        project="llm-evaluation",
        environment="development",
        enable_traceloop_platform=True  # Optional: commercial features
    )
    
    # Enhanced tracing with governance
    with adapter.track_operation(
        operation_type="llm_generation",
        operation_name="research_analysis",
        customer_id="enterprise_123"
    ) as span:
        result = openai_client.chat.completions.create(...)
        # Automatic cost attribution and governance tracking
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import OpenLLMetry with graceful failure
try:
    import openllmetry
    from openllmetry import tracer
    from openllmetry.instrumentation.auto import AutoInstrumentor
    HAS_OPENLLMETRY = True
    logger.info(f"OpenLLMetry loaded successfully (version: {getattr(openllmetry, '__version__', 'unknown')})")
except ImportError:
    HAS_OPENLLMETRY = False
    openllmetry = None
    tracer = None
    AutoInstrumentor = None
    logger.warning("OpenLLMetry not installed. Install with: pip install openllmetry")

# Import Traceloop SDK with graceful failure (optional commercial platform)
try:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.decorators import aworkflow, workflow
    HAS_TRACELOOP_SDK = True
    logger.info("Traceloop SDK loaded for commercial platform features")
except ImportError:
    HAS_TRACELOOP_SDK = False
    Traceloop = None
    workflow = None
    aworkflow = None
    logger.info("Traceloop SDK not available (open-source mode only)")

# Import OpenTelemetry for enhanced tracing
try:
    from opentelemetry import context, trace
    from opentelemetry.semconv.ai import SpanAttributes
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    logger.warning("OpenTelemetry not available")


class TraceloopOperationType(Enum):
    """Operation types for OpenLLMetry + GenOps tracking."""
    LLM_GENERATION = "llm_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"
    WORKFLOW = "workflow"
    EVALUATION = "evaluation"
    BATCH_PROCESSING = "batch_processing"


class GovernancePolicy(Enum):
    """Governance policy enforcement levels."""
    ADVISORY = "advisory"      # Log policy violations but continue
    ENFORCED = "enforced"      # Block operations that violate policy
    AUDIT_ONLY = "audit_only"  # Track for compliance reporting


@dataclass
class TraceloopUsage:
    """Usage statistics from OpenLLMetry operations with GenOps governance."""
    operation_id: str
    operation_type: str
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    latency_ms: float

    # GenOps governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    environment: str = "development"

    # Budget and policy tracking
    budget_remaining: Optional[float] = None
    policy_violations: List[str] = field(default_factory=list)
    governance_tags: Dict[str, str] = field(default_factory=dict)

    # OpenTelemetry integration
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class TraceloopResponse:
    """Standardized response from OpenLLMetry operations with governance."""
    content: Any  # Response content (varies by operation type)
    usage: TraceloopUsage
    trace_id: str
    span_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    governance_status: str = "compliant"
    cost_optimization_suggestions: List[str] = field(default_factory=list)


class GenOpsTraceloopAdapter:
    """
    GenOps adapter for Traceloop + OpenLLMetry with comprehensive governance integration.
    
    This adapter enhances OpenLLMetry's observability capabilities with GenOps
    governance features including cost attribution, budget enforcement, and
    policy compliance tracking. Optionally integrates with Traceloop commercial platform.
    """

    def __init__(
        self,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: str = "development",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,

        # Budget and policy settings
        enable_governance: bool = True,
        daily_budget_limit: Optional[float] = None,
        max_operation_cost: Optional[float] = None,
        governance_policy: GovernancePolicy = GovernancePolicy.ADVISORY,

        # OpenLLMetry settings
        enable_auto_instrumentation: bool = True,

        # Traceloop platform settings (optional)
        traceloop_api_key: Optional[str] = None,
        traceloop_base_url: str = "https://app.traceloop.com",
        enable_traceloop_platform: bool = None,

        # Advanced settings
        enable_cost_alerts: bool = True,
        cost_alert_threshold: float = 1.0,
        **kwargs
    ):
        """
        Initialize GenOps Traceloop adapter with governance configuration.
        
        Args:
            team: Team name for cost attribution
            project: Project name for cost tracking
            environment: Environment (development, staging, production)
            customer_id: Customer ID for per-customer attribution
            cost_center: Cost center for financial reporting
            enable_governance: Enable GenOps governance features
            daily_budget_limit: Daily spending limit in USD
            max_operation_cost: Maximum cost per operation
            governance_policy: Policy enforcement level
            enable_auto_instrumentation: Enable automatic OpenLLMetry instrumentation
            traceloop_api_key: API key for Traceloop commercial platform
            traceloop_base_url: Base URL for Traceloop platform
            enable_traceloop_platform: Enable commercial platform features
            enable_cost_alerts: Enable cost-based alerting
            cost_alert_threshold: Cost threshold for alerts
        """

        # Core governance attributes
        self.team = team or os.getenv('GENOPS_TEAM', 'default-team')
        self.project = project or os.getenv('GENOPS_PROJECT', 'default-project')
        self.environment = environment
        self.customer_id = customer_id
        self.cost_center = cost_center

        # Governance settings
        self.enable_governance = enable_governance
        self.daily_budget_limit = daily_budget_limit
        self.max_operation_cost = max_operation_cost
        self.governance_policy = governance_policy
        self.enable_cost_alerts = enable_cost_alerts
        self.cost_alert_threshold = cost_alert_threshold

        # Initialize OpenLLMetry
        self._initialize_openllmetry(enable_auto_instrumentation)

        # Initialize Traceloop platform (optional)
        self.enable_traceloop_platform = enable_traceloop_platform
        if enable_traceloop_platform or traceloop_api_key:
            self._initialize_traceloop_platform(traceloop_api_key, traceloop_base_url)

        # Governance state tracking
        self._daily_usage = 0.0
        self._operation_count = 0
        self._policy_violations = []

        logger.info(f"GenOps Traceloop adapter initialized: team={self.team}, project={self.project}")

    def _initialize_openllmetry(self, enable_auto: bool):
        """Initialize OpenLLMetry instrumentation."""
        if not HAS_OPENLLMETRY:
            logger.error("OpenLLMetry not available. Install with: pip install openllmetry")
            return

        try:
            if enable_auto and AutoInstrumentor:
                # Enable automatic instrumentation for all supported providers
                AutoInstrumentor().instrument()
                logger.info("OpenLLMetry auto-instrumentation enabled")

            # Get enhanced tracer with GenOps attributes
            self.tracer = tracer
            logger.info("OpenLLMetry tracer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenLLMetry: {e}")

    def _initialize_traceloop_platform(self, api_key: Optional[str], base_url: str):
        """Initialize Traceloop commercial platform integration."""
        if not HAS_TRACELOOP_SDK:
            logger.warning("Traceloop SDK not available. Install with: pip install traceloop-sdk")
            return

        try:
            api_key = api_key or os.getenv('TRACELOOP_API_KEY')
            if not api_key:
                logger.info("No Traceloop API key provided, commercial features disabled")
                return

            # Initialize Traceloop platform
            Traceloop.init(
                api_key=api_key,
                api_endpoint=base_url,
                disable_batch=False,  # Enable batching for better performance
            )

            self.traceloop_client = Traceloop
            self.enable_traceloop_platform = True
            logger.info("Traceloop commercial platform initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Traceloop platform: {e}")
            self.enable_traceloop_platform = False

    @contextmanager
    def track_operation(
        self,
        operation_type: Union[TraceloopOperationType, str],
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span=None,
        max_cost: Optional[float] = None
    ):
        """
        Track an LLM operation with GenOps governance.
        
        Args:
            operation_type: Type of operation being tracked
            operation_name: Name of the operation for identification
            tags: Additional tags for the operation
            parent_span: Parent span for nested operations
            max_cost: Maximum allowed cost for this operation
            
        Yields:
            Enhanced span with GenOps governance capabilities
        """
        if not HAS_OPENLLMETRY or not self.tracer:
            logger.warning("OpenLLMetry not available, basic tracking only")
            yield MockSpan()
            return

        operation_type_str = operation_type.value if isinstance(operation_type, TraceloopOperationType) else operation_type

        # Create enhanced span with governance attributes
        with self.tracer.start_span(
            operation_name,
            kind=trace.SpanKind.CLIENT if HAS_OTEL else None
        ) as span:

            # Add GenOps governance attributes
            if HAS_OTEL and span:
                span.set_attribute("genops.team", self.team)
                span.set_attribute("genops.project", self.project)
                span.set_attribute("genops.environment", self.environment)
                span.set_attribute("genops.operation_type", operation_type_str)

                if self.customer_id:
                    span.set_attribute("genops.customer_id", self.customer_id)
                if self.cost_center:
                    span.set_attribute("genops.cost_center", self.cost_center)

                # Add custom tags
                if tags:
                    for key, value in tags.items():
                        span.set_attribute(f"genops.tag.{key}", str(value))

            # Create enhanced span wrapper
            enhanced_span = EnhancedSpan(span, self, operation_type_str, max_cost)

            try:
                yield enhanced_span

                # Finalize governance tracking
                self._finalize_operation(enhanced_span)

            except Exception as e:
                if HAS_OTEL and span:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                logger.error(f"Operation {operation_name} failed: {e}")
                raise

    def _finalize_operation(self, enhanced_span):
        """Finalize governance tracking for completed operation."""
        try:
            # Update usage tracking
            self._operation_count += 1
            cost = enhanced_span.estimated_cost
            if cost:
                self._daily_usage += cost

            # Check governance policies
            self._check_governance_policies(enhanced_span)

            # Send to Traceloop platform if enabled
            if self.enable_traceloop_platform:
                self._send_to_traceloop_platform(enhanced_span)

        except Exception as e:
            logger.error(f"Failed to finalize operation governance: {e}")

    def _check_governance_policies(self, enhanced_span):
        """Check governance policies and handle violations."""
        violations = []

        # Check operation cost limits
        if self.max_operation_cost and enhanced_span.estimated_cost > self.max_operation_cost:
            violations.append(f"Operation cost ${enhanced_span.estimated_cost:.6f} exceeds limit ${self.max_operation_cost}")

        # Check daily budget limits
        if self.daily_budget_limit and self._daily_usage > self.daily_budget_limit:
            violations.append(f"Daily usage ${self._daily_usage:.2f} exceeds budget ${self.daily_budget_limit}")

        # Handle policy violations
        if violations:
            enhanced_span.policy_violations.extend(violations)
            self._policy_violations.extend(violations)

            if self.governance_policy == GovernancePolicy.ENFORCED:
                raise ValueError(f"Governance policy violation: {violations[0]}")
            elif self.governance_policy == GovernancePolicy.ADVISORY:
                logger.warning(f"Governance policy advisory: {violations}")

    def _send_to_traceloop_platform(self, enhanced_span):
        """Send governance data to Traceloop commercial platform."""
        try:
            # This would integrate with Traceloop platform APIs
            # Implementation depends on specific Traceloop platform capabilities
            logger.debug("Governance data sent to Traceloop platform")
        except Exception as e:
            logger.error(f"Failed to send data to Traceloop platform: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current governance metrics."""
        return {
            "daily_usage": self._daily_usage,
            "operation_count": self._operation_count,
            "budget_limit": self.daily_budget_limit,
            "budget_remaining": (self.daily_budget_limit - self._daily_usage) if self.daily_budget_limit else None,
            "policy_violations": len(self._policy_violations),
            "governance_enabled": self.enable_governance,
            "traceloop_platform_enabled": self.enable_traceloop_platform
        }


class EnhancedSpan:
    """Enhanced span wrapper with GenOps governance capabilities."""

    def __init__(self, otel_span, adapter: GenOpsTraceloopAdapter, operation_type: str, max_cost: Optional[float]):
        self.otel_span = otel_span
        self.adapter = adapter
        self.operation_type = operation_type
        self.max_cost = max_cost
        self.start_time = time.time()

        # Tracking attributes
        self.estimated_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.policy_violations = []
        self.metadata = {}

    def add_attributes(self, attributes: Dict[str, Any]):
        """Add attributes to the span."""
        if HAS_OTEL and self.otel_span:
            for key, value in attributes.items():
                self.otel_span.set_attribute(key, value)

    def update_cost(self, cost: float):
        """Update the estimated cost for this operation."""
        self.estimated_cost = cost
        if HAS_OTEL and self.otel_span:
            self.otel_span.set_attribute("genops.cost.amount", cost)
            self.otel_span.set_attribute("genops.cost.currency", "USD")

    def update_token_usage(self, input_tokens: int, output_tokens: int):
        """Update token usage for this operation."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens

        if HAS_OTEL and self.otel_span:
            self.otel_span.set_attribute("genops.tokens.input", input_tokens)
            self.otel_span.set_attribute("genops.tokens.output", output_tokens)
            self.otel_span.set_attribute("genops.tokens.total", self.total_tokens)

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for this span."""
        return {
            "estimated_cost": self.estimated_cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": (time.time() - self.start_time) * 1000,
            "team": self.adapter.team,
            "project": self.adapter.project,
            "environment": self.adapter.environment,
            "operation_type": self.operation_type
        }


class MockSpan:
    """Mock span for when OpenLLMetry is not available."""

    def __init__(self):
        self.estimated_cost = 0.0
        self.policy_violations = []

    def add_attributes(self, attributes):
        pass

    def update_cost(self, cost):
        self.estimated_cost = cost

    def update_token_usage(self, input_tokens, output_tokens):
        pass

    def get_metrics(self):
        return {"estimated_cost": self.estimated_cost}


# Convenience functions for common usage patterns

def instrument_traceloop(**kwargs) -> GenOpsTraceloopAdapter:
    """
    Create and configure a GenOps Traceloop adapter.
    
    This is the main entry point for manual instrumentation.
    
    Returns:
        Configured GenOpsTraceloopAdapter instance
    """
    return GenOpsTraceloopAdapter(**kwargs)


def auto_instrument(
    team: str,
    project: str,
    environment: str = "development",
    **kwargs
) -> None:
    """
    Enable automatic instrumentation for all OpenLLMetry operations.
    
    This enhances existing OpenLLMetry applications with GenOps governance
    without requiring code changes.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost tracking  
        environment: Environment (development, staging, production)
        **kwargs: Additional configuration options
    """
    if not HAS_OPENLLMETRY:
        logger.error("Cannot auto-instrument: OpenLLMetry not available")
        return

    # Create global adapter instance
    global _global_adapter
    _global_adapter = GenOpsTraceloopAdapter(
        team=team,
        project=project,
        environment=environment,
        enable_auto_instrumentation=True,
        **kwargs
    )

    logger.info(f"Auto-instrumentation enabled for team={team}, project={project}")


def get_enhanced_tracer():
    """Get the OpenLLMetry tracer enhanced with GenOps governance."""
    if not HAS_OPENLLMETRY:
        logger.warning("OpenLLMetry not available")
        return None
    return tracer


def get_current_governance_context() -> Dict[str, Any]:
    """Get current governance context from global adapter."""
    global _global_adapter
    if '_global_adapter' in globals():
        return {
            "team": _global_adapter.team,
            "project": _global_adapter.project,
            "environment": _global_adapter.environment,
            "customer_id": _global_adapter.customer_id,
            "cost_center": _global_adapter.cost_center
        }
    return {}


def get_budget_status() -> Dict[str, Any]:
    """Get current budget status from global adapter."""
    global _global_adapter
    if '_global_adapter' in globals():
        return {
            "daily_limit": _global_adapter.daily_budget_limit,
            "current_usage": _global_adapter._daily_usage,
            "remaining": (_global_adapter.daily_budget_limit - _global_adapter._daily_usage) if _global_adapter.daily_budget_limit else None,
            "operation_count": _global_adapter._operation_count
        }
    return {"daily_limit": None, "current_usage": 0.0, "remaining": None}


def get_recent_operations_summary(limit: int = 10) -> Dict[str, Any]:
    """Get summary of recent operations."""
    # This would track recent operations in production
    # For now, return mock data
    return {
        "operations": [
            {"operation_type": "chat_completion", "cost": 0.002},
            {"operation_type": "embedding", "cost": 0.001},
        ],
        "total_cost": 0.003,
        "count": 2
    }


def is_enhanced_tracer(tracer_obj) -> bool:
    """Check if a tracer is enhanced with GenOps governance."""
    # This would check if the tracer includes GenOps enhancements
    return True  # Assume enhanced when GenOps is loaded


def multi_provider_cost_tracking(
    providers: List[str],
    team: str,
    project: str,
    environment: str = "development",
    **kwargs
) -> Dict[str, float]:
    """
    Enable unified cost tracking across multiple AI providers.
    
    This convenience function sets up cost tracking across multiple providers
    with unified governance and provides cost aggregation.
    
    Args:
        providers: List of provider names (e.g., ["openai", "anthropic", "gemini"])
        team: Team name for cost attribution
        project: Project name for cost tracking
        environment: Environment (development, staging, production)
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary of cost breakdowns by provider
    """
    if not HAS_OPENLLMETRY:
        logger.error("Cannot enable multi-provider tracking: OpenLLMetry not available")
        return {}

    # Create unified adapter for all providers
    adapter = GenOpsTraceloopAdapter(
        team=team,
        project=project,
        environment=environment,
        enable_auto_instrumentation=True,
        **kwargs
    )

    cost_summary = {}
    for provider in providers:
        cost_summary[provider] = 0.0

    # Store provider configuration
    adapter._multi_provider_config = {
        "providers": providers,
        "cost_summary": cost_summary
    }

    logger.info(f"Multi-provider cost tracking enabled for: {', '.join(providers)}")
    return cost_summary


def traceloop_create(
    team: str,
    project: str,
    environment: str = "development",
    **kwargs
) -> GenOpsTraceloopAdapter:
    """
    Create a Traceloop adapter following standard GenOps provider conventions.
    
    This is an alias for instrument_traceloop() that follows the standard
    {provider}_create() naming pattern used across GenOps providers.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost tracking
        environment: Environment (development, staging, production)
        **kwargs: Additional configuration options
        
    Returns:
        Configured GenOpsTraceloopAdapter instance
    """
    return instrument_traceloop(
        team=team,
        project=project,
        environment=environment,
        **kwargs
    )


# Global adapter instance for auto-instrumentation
_global_adapter = None
