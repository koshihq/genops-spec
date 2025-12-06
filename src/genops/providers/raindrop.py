#!/usr/bin/env python3
"""
GenOps Raindrop AI Integration

This module provides comprehensive Raindrop AI integration for GenOps governance,
cost intelligence, and policy enforcement. Raindrop AI is an AI monitoring platform
that discovers silent agent failures and provides performance insights for AI systems.

Features:
- Enhanced agent monitoring with GenOps governance attributes and cost tracking
- Cost attribution and budget enforcement for agent monitoring operations
- Policy compliance tracking integrated with agent performance monitoring
- Signal monitoring with governance oversight and cost optimization
- Alert management and dashboard analytics with unified cost intelligence
- Zero-code auto-instrumentation with auto_instrument()
- Enterprise-ready governance patterns for production AI agent observability

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.raindrop import auto_instrument
    auto_instrument(
        raindrop_api_key="your-raindrop-api-key",
        team="ai-ops-team",
        project="agent-monitoring"
    )
    
    # Your existing Raindrop code now includes GenOps governance
    import raindrop
    
    client = raindrop.Client(api_key="your-api-key")
    response = client.track_interaction(
        agent_id="agent-123",
        interaction_data={
            "input": "user_query",
            "output": "agent_response",
            "performance_signals": {"latency": 150, "accuracy": 0.95}
        }
    )
    # Automatically tracked with cost attribution and governance
    
    # Manual adapter usage for advanced governance
    from genops.providers.raindrop import GenOpsRaindropAdapter
    
    adapter = GenOpsRaindropAdapter(
        raindrop_api_key="your-raindrop-api-key",
        team="ai-platform-team",
        project="production-monitoring",
        enable_cost_alerts=True,
        daily_budget_limit=100.0
    )
    
    with adapter.track_agent_monitoring_session("fraud-detection-agents") as session:
        # Multi-agent monitoring with unified cost tracking
        session.track_agent_interaction("agent-1", interaction_data, cost=0.05)
        session.track_performance_signal("accuracy_drop", {"threshold": 0.1}, cost=0.02)
        session.create_alert("performance_degradation", alert_config, cost=0.10)
        
        # Automatic cost aggregation and governance telemetry export
        print(f"Session cost: ${session.total_cost:.3f}")

Author: GenOps AI Contributors
License: Apache 2.0
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Union
from decimal import Decimal
import uuid
import logging

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None
    OTLPSpanExporter = None

from .raindrop_pricing import RaindropPricingCalculator, RaindropCostResult
from .raindrop_cost_aggregator import RaindropCostAggregator, RaindropSessionSummary
from .raindrop_validation import validate_setup, print_validation_result

logger = logging.getLogger(__name__)

@dataclass
class RaindropGovernanceAttributes:
    """Standard governance attributes for Raindrop AI operations."""
    team: str
    project: str
    environment: str = "production"
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    feature: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for OpenTelemetry attributes."""
        attrs = {
            "genops.team": self.team,
            "genops.project": self.project,
            "genops.environment": self.environment,
            "genops.session_id": self.session_id,
            "genops.provider": "raindrop",
        }
        if self.customer_id:
            attrs["genops.customer_id"] = self.customer_id
        if self.cost_center:
            attrs["genops.cost_center"] = self.cost_center
        if self.feature:
            attrs["genops.feature"] = self.feature
        return attrs

class GenOpsRaindropAdapter:
    """
    GenOps adapter for Raindrop AI with comprehensive governance features.
    
    Provides cost tracking, budget enforcement, and governance telemetry
    for Raindrop AI agent monitoring operations.
    """
    
    def __init__(
        self,
        raindrop_api_key: Optional[str] = None,
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
        **kwargs
    ):
        """
        Initialize GenOps Raindrop AI adapter.
        
        Args:
            raindrop_api_key: Raindrop AI API key (or set RAINDROP_API_KEY env var)
            team: Team identifier for cost attribution
            project: Project identifier for cost attribution
            environment: Environment (development, staging, production)
            customer_id: Customer identifier for multi-tenant scenarios
            cost_center: Cost center for financial reporting
            feature: Feature identifier for granular attribution
            daily_budget_limit: Daily spending limit in USD
            enable_cost_alerts: Enable budget and cost alerting
            governance_policy: Policy enforcement level (advisory, enforced)
            export_telemetry: Enable OpenTelemetry export
        """
        self.raindrop_api_key = raindrop_api_key or os.getenv("RAINDROP_API_KEY")
        self.daily_budget_limit = daily_budget_limit
        self.enable_cost_alerts = enable_cost_alerts
        self.governance_policy = governance_policy
        self.export_telemetry = export_telemetry
        
        # Initialize governance attributes
        self.governance_attrs = RaindropGovernanceAttributes(
            team=team,
            project=project,
            environment=environment,
            customer_id=customer_id,
            cost_center=cost_center,
            feature=feature
        )
        
        # Initialize pricing and cost tracking
        self.pricing_calculator = RaindropPricingCalculator()
        self.cost_aggregator = RaindropCostAggregator()
        
        # Initialize OpenTelemetry
        self.tracer = None
        if export_telemetry and trace:
            self._setup_telemetry()
        
        # Validate setup with comprehensive error handling
        if self.governance_policy == "enforced":
            try:
                validation_result = validate_setup(self.raindrop_api_key)
                if not validation_result.is_valid:
                    error_messages = []
                    for error in validation_result.errors[:3]:  # Show first 3 errors
                        error_msg = f"• {error.message}"
                        if error.fix_suggestion:
                            error_msg += f"\n  💡 Fix: {error.fix_suggestion}"
                        error_messages.append(error_msg)
                    
                    raise ValueError(
                        f"Raindrop AI setup validation failed:\n" + 
                        "\n".join(error_messages) + 
                        f"\n\n🔧 Run the following for complete validation:\n" +
                        "from genops.providers.raindrop_validation import validate_setup_interactive\n" +
                        "validate_setup_interactive()"
                    )
            except Exception as validation_error:
                if self.governance_policy == "enforced":
                    logger.error(f"Critical validation failure: {validation_error}")
                    raise
                else:
                    logger.warning(f"Validation failed but continuing in advisory mode: {validation_error}")
        
        logger.info(f"GenOps Raindrop adapter initialized for team='{team}', project='{project}'")
    
    def _setup_telemetry(self):
        """Initialize OpenTelemetry tracing with comprehensive error handling."""
        try:
            if not trace.get_tracer_provider():
                trace.set_tracer_provider(TracerProvider())
                
            # Configure OTLP exporter with error handling
            try:
                otlp_exporter = OTLPSpanExporter()
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
                logger.debug("OpenTelemetry OTLP exporter configured successfully")
            except Exception as otlp_error:
                logger.warning(f"OTLP exporter configuration failed: {otlp_error}")
                logger.info("Telemetry will work locally but won't export to external collectors")
            
            self.tracer = trace.get_tracer("genops.raindrop")
            logger.debug("OpenTelemetry tracer initialized for GenOps Raindrop")
            
        except ImportError as import_error:
            logger.error(f"OpenTelemetry dependencies missing: {import_error}")
            logger.error("Install with: pip install genops[raindrop] to enable telemetry")
            if self.governance_policy == "enforced":
                raise ValueError(
                    f"OpenTelemetry is required in enforced governance mode.\n"
                    f"Install with: pip install genops[raindrop]\n"
                    f"Or switch to advisory mode: governance_policy='advisory'"
                ) from import_error
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry: {e}")
            logger.info("Continuing without telemetry export - local tracking will still work")
            if self.governance_policy == "enforced":
                logger.error("Telemetry setup failed in enforced mode")
                raise ValueError(
                    f"Telemetry setup failed in enforced governance mode: {e}\n"
                    f"Fix telemetry configuration or switch to advisory mode"
                ) from e
    
    @contextmanager
    def track_agent_monitoring_session(self, session_name: str, **kwargs):
        """
        Context manager for tracking an agent monitoring session.
        
        Args:
            session_name: Name identifier for the monitoring session
            **kwargs: Additional session parameters
            
        Yields:
            RaindropMonitoringSession: Session object for tracking operations
        """
        session = RaindropMonitoringSession(
            name=session_name,
            adapter=self,
            governance_attrs=self.governance_attrs,
            **kwargs
        )
        
        # Start telemetry span
        if self.tracer:
            with self.tracer.start_as_current_span(f"raindrop.monitoring.session.{session_name}") as span:
                span.set_attributes(self.governance_attrs.to_dict())
                span.set_attribute("genops.operation", "agent_monitoring_session")
                span.set_attribute("genops.session.name", session_name)
                
                try:
                    yield session
                    
                    # Finalize session
                    session._finalize()
                    
                    # Add cost and performance metrics to span
                    span.set_attribute("genops.cost.total", float(session.total_cost))
                    span.set_attribute("genops.cost.currency", "USD")
                    span.set_attribute("genops.session.operations", session.operation_count)
                    span.set_attribute("genops.session.duration_seconds", session.duration_seconds)
                    span.set_status(Status(StatusCode.OK))
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("genops.error", str(e))
                    raise
        else:
            try:
                yield session
                session._finalize()
            except Exception:
                raise

class RaindropMonitoringSession:
    """
    Context for tracking Raindrop AI agent monitoring operations with cost attribution.
    """
    
    def __init__(self, name: str, adapter: GenOpsRaindropAdapter, governance_attrs: RaindropGovernanceAttributes, **kwargs):
        self.name = name
        self.adapter = adapter
        self.governance_attrs = governance_attrs
        self.start_time = time.time()
        self.operations: List[Dict[str, Any]] = []
        self.total_cost = Decimal('0.00')
        self.finalized = False
    
    @property
    def operation_count(self) -> int:
        """Number of operations in this session."""
        return len(self.operations)
    
    @property
    def duration_seconds(self) -> float:
        """Duration of the session in seconds."""
        return time.time() - self.start_time
    
    def track_agent_interaction(
        self, 
        agent_id: str, 
        interaction_data: Dict[str, Any], 
        cost: Optional[float] = None
    ) -> RaindropCostResult:
        """
        Track an agent interaction with cost attribution.
        
        Args:
            agent_id: Identifier for the agent
            interaction_data: Interaction data and performance signals
            cost: Override cost calculation (optional)
            
        Returns:
            RaindropCostResult: Cost calculation result
        """
        # Calculate cost
        if cost is not None:
            cost_result = RaindropCostResult(
                operation_type="agent_interaction",
                base_cost=Decimal(str(cost)),
                total_cost=Decimal(str(cost)),
                currency="USD",
                agent_id=agent_id
            )
        else:
            cost_result = self.adapter.pricing_calculator.calculate_interaction_cost(
                agent_id=agent_id,
                interaction_data=interaction_data
            )
        
        # Track operation
        operation = {
            "type": "agent_interaction",
            "agent_id": agent_id,
            "interaction_data": interaction_data,
            "cost": float(cost_result.total_cost),
            "timestamp": time.time()
        }
        self.operations.append(operation)
        self.total_cost += cost_result.total_cost
        
        # Check budget if enabled
        if self.adapter.daily_budget_limit and self.adapter.enable_cost_alerts:
            if float(self.total_cost) > self.adapter.daily_budget_limit:
                if self.adapter.governance_policy == "enforced":
                    raise ValueError(f"Session would exceed daily budget limit: ${self.adapter.daily_budget_limit}")
                else:
                    logger.warning(f"Session cost ${float(self.total_cost)} exceeds budget ${self.adapter.daily_budget_limit}")
        
        return cost_result
    
    def track_performance_signal(
        self, 
        signal_name: str, 
        signal_data: Dict[str, Any], 
        cost: Optional[float] = None
    ) -> RaindropCostResult:
        """
        Track a performance signal with cost attribution.
        
        Args:
            signal_name: Name of the performance signal
            signal_data: Signal configuration and data
            cost: Override cost calculation (optional)
            
        Returns:
            RaindropCostResult: Cost calculation result
        """
        if cost is not None:
            cost_result = RaindropCostResult(
                operation_type="performance_signal",
                base_cost=Decimal(str(cost)),
                total_cost=Decimal(str(cost)),
                currency="USD",
                signal_name=signal_name
            )
        else:
            cost_result = self.adapter.pricing_calculator.calculate_signal_cost(
                signal_name=signal_name,
                signal_data=signal_data
            )
        
        operation = {
            "type": "performance_signal",
            "signal_name": signal_name,
            "signal_data": signal_data,
            "cost": float(cost_result.total_cost),
            "timestamp": time.time()
        }
        self.operations.append(operation)
        self.total_cost += cost_result.total_cost
        
        return cost_result
    
    def create_alert(
        self, 
        alert_name: str, 
        alert_config: Dict[str, Any], 
        cost: Optional[float] = None
    ) -> RaindropCostResult:
        """
        Create an alert with cost attribution.
        
        Args:
            alert_name: Name of the alert
            alert_config: Alert configuration
            cost: Override cost calculation (optional)
            
        Returns:
            RaindropCostResult: Cost calculation result
        """
        if cost is not None:
            cost_result = RaindropCostResult(
                operation_type="alert_creation",
                base_cost=Decimal(str(cost)),
                total_cost=Decimal(str(cost)),
                currency="USD",
                alert_name=alert_name
            )
        else:
            cost_result = self.adapter.pricing_calculator.calculate_alert_cost(
                alert_name=alert_name,
                alert_config=alert_config
            )
        
        operation = {
            "type": "alert_creation",
            "alert_name": alert_name,
            "alert_config": alert_config,
            "cost": float(cost_result.total_cost),
            "timestamp": time.time()
        }
        self.operations.append(operation)
        self.total_cost += cost_result.total_cost
        
        return cost_result
    
    def _finalize(self):
        """Finalize the session and export telemetry."""
        if self.finalized:
            return
        
        self.finalized = True
        
        # Create session summary
        session_summary = RaindropSessionSummary(
            session_id=self.governance_attrs.session_id,
            session_name=self.name,
            total_cost=float(self.total_cost),
            operation_count=self.operation_count,
            duration_seconds=self.duration_seconds,
            operations=self.operations,
            governance_attributes=self.governance_attrs.to_dict()
        )
        
        # Add to cost aggregator
        self.adapter.cost_aggregator.add_session(session_summary)

# Global auto-instrumentation state
_auto_instrumented = False
_original_raindrop_client = None

def auto_instrument(
    raindrop_api_key: Optional[str] = None,
    team: str = "default",
    project: str = "default",
    environment: str = "production",
    **kwargs
) -> GenOpsRaindropAdapter:
    """
    Enable zero-code auto-instrumentation for Raindrop AI.
    
    This function patches the Raindrop AI client to automatically include
    GenOps governance attributes and cost tracking without code changes.
    
    Args:
        raindrop_api_key: Raindrop AI API key (or set RAINDROP_API_KEY env var)
        team: Team identifier for cost attribution
        project: Project identifier for cost attribution
        environment: Environment (development, staging, production)
        **kwargs: Additional GenOpsRaindropAdapter parameters
        
    Returns:
        GenOpsRaindropAdapter: Configured adapter instance
    """
    global _auto_instrumented, _original_raindrop_client
    
    if _auto_instrumented:
        logger.warning("Raindrop AI auto-instrumentation already enabled")
        return
    
    # Create adapter
    adapter = GenOpsRaindropAdapter(
        raindrop_api_key=raindrop_api_key,
        team=team,
        project=project,
        environment=environment,
        **kwargs
    )
    
    try:
        # Attempt to patch Raindrop AI client (if available)
        import raindrop
        
        _original_raindrop_client = raindrop.Client
        
        class InstrumentedRaindropClient(_original_raindrop_client):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._genops_adapter = adapter
            
            def track_interaction(self, *args, **kwargs):
                # Add governance attributes to interaction tracking
                result = super().track_interaction(*args, **kwargs)
                
                # Track cost and governance
                with adapter.track_agent_monitoring_session("auto_instrumented") as session:
                    session.track_agent_interaction(
                        agent_id=kwargs.get('agent_id', 'unknown'),
                        interaction_data=kwargs,
                        cost=0.001  # Default interaction cost
                    )
                
                return result
        
        # Replace the client class
        raindrop.Client = InstrumentedRaindropClient
        _auto_instrumented = True
        
        logger.info("✅ Raindrop AI auto-instrumentation enabled")
        
    except ImportError:
        logger.info("📋 Raindrop AI SDK not found - governance adapter ready for manual use")
    except Exception as e:
        logger.warning(f"⚠️ Failed to enable auto-instrumentation: {e}")
    
    return adapter

def restore_raindrop():
    """Restore original Raindrop AI client (disable auto-instrumentation)."""
    global _auto_instrumented, _original_raindrop_client
    
    if not _auto_instrumented:
        return
    
    try:
        import raindrop
        if _original_raindrop_client:
            raindrop.Client = _original_raindrop_client
        _auto_instrumented = False
        logger.info("✅ Raindrop AI auto-instrumentation disabled")
    except Exception as e:
        logger.warning(f"Failed to restore Raindrop AI client: {e}")

# Export main classes and functions
__all__ = [
    'GenOpsRaindropAdapter',
    'RaindropMonitoringSession', 
    'RaindropGovernanceAttributes',
    'auto_instrument',
    'restore_raindrop',
    'validate_setup',
    'print_validation_result'
]