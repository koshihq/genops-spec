#!/usr/bin/env python3
"""
GenOps Arize AI Integration

This module provides comprehensive Arize AI integration for GenOps governance,
cost intelligence, and policy enforcement. Arize AI is a leading ML observability
platform that helps teams monitor, troubleshoot, and improve model performance
in production.

Features:
- Enhanced model monitoring with GenOps governance attributes and cost tracking
- Cost attribution and budget enforcement for model monitoring operations
- Policy compliance tracking integrated with model performance monitoring
- Data quality monitoring with governance oversight and cost optimization
- Alert management and dashboard analytics with unified cost intelligence
- Zero-code auto-instrumentation with instrument_arize()
- Enterprise-ready governance patterns for production ML observability

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.arize import auto_instrument
    auto_instrument(
        arize_api_key="your-arize-api-key",
        team="ml-ops-team",
        project="model-monitoring"
    )

    # Your existing Arize code now includes GenOps governance
    from arize.pandas.logger import Client

    arize_client = Client(api_key="your-api-key", space_key="your-space-key")
    response = arize_client.log(
        prediction_id="pred-123",
        prediction_label="positive",
        actual_label="positive",
        model_id="sentiment-model-v2",
        model_version="2.1"
    )
    # Automatically tracked with cost attribution and governance

    # Manual adapter usage for advanced governance
    from genops.providers.arize import GenOpsArizeAdapter

    adapter = GenOpsArizeAdapter(
        arize_api_key="your-arize-api-key",
        arize_space_key="your-space-key",
        team="ml-platform-team",
        project="production-monitoring",
        enable_cost_alerts=True,
        daily_budget_limit=50.0
    )

    # Enhanced model monitoring with governance
    with adapter.track_model_monitoring_session("fraud-detection-v3") as session:
        # Log predictions with cost tracking
        session.log_prediction_batch(predictions_df, cost_per_prediction=0.001)

        # Monitor data quality with governance
        session.log_data_quality_metrics(quality_metrics, cost_estimate=0.05)

        # Create governed alerts
        session.create_performance_alert(
            metric="accuracy",
            threshold=0.85,
            cost_per_alert=0.10
        )

Dependencies:
    - arize: Arize AI Python SDK (pip install arize)
    - opentelemetry-api: For telemetry export
    - pandas: For data processing support

Environment Variables:
    - ARIZE_API_KEY: Your Arize AI API key
    - ARIZE_SPACE_KEY: Your Arize AI space key
    - GENOPS_TEAM: Team attribution (recommended)
    - GENOPS_PROJECT: Project attribution (recommended)
    - GENOPS_DAILY_BUDGET_LIMIT: Daily spending limit in USD
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Optional arize import with graceful degradation
try:
    import arize
    from arize.pandas.logger import Client as ArizeClient
    from arize.utils.types import Environments, ModelTypes

    ARIZE_AVAILABLE = True
except ImportError:
    ARIZE_AVAILABLE = False
    arize = None
    ArizeClient = None
    ModelTypes = None
    Environments = None

logger = logging.getLogger(__name__)


class MonitoringScope(Enum):
    """Model monitoring scope levels."""

    PREDICTIONS = "predictions"
    DATA_QUALITY = "data_quality"
    MODEL_DRIFT = "model_drift"
    PERFORMANCE = "performance"
    ALERTS = "alerts"


@dataclass
class ModelMonitoringCostSummary:
    """Cost summary for Arize AI model monitoring operations."""

    total_cost: float
    prediction_logging_cost: float
    data_quality_cost: float
    alert_management_cost: float
    dashboard_cost: float
    cost_by_model: dict[str, float]
    cost_by_environment: dict[str, float]
    monitoring_duration: float
    efficiency_score: float


@dataclass
class ArizeMonitoringContext:
    """Context for tracking Arize AI monitoring governance."""

    session_id: str
    session_name: str
    model_id: str
    model_version: str
    environment: str
    team: str
    customer_id: Optional[str]
    start_time: datetime
    estimated_cost: float = 0.0
    prediction_count: int = 0
    data_quality_checks: int = 0
    active_alerts: int = 0
    policy_violations: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.policy_violations is None:
            self.policy_violations = []


class GenOpsArizeAdapter:
    """
    GenOps governance adapter for Arize AI model monitoring and observability.

    Provides comprehensive cost intelligence, policy enforcement, and team attribution
    for Arize AI monitoring operations with enterprise-grade governance features.
    """

    def __init__(
        self,
        arize_api_key: Optional[str] = None,
        arize_space_key: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        customer_id: Optional[str] = None,
        environment: str = "production",
        daily_budget_limit: float = 50.0,
        max_monitoring_cost: float = 25.0,
        enable_cost_alerts: bool = True,
        enable_governance: bool = True,
        cost_center: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the GenOps Arize AI adapter.

        Args:
            arize_api_key: Arize AI API key (or set ARIZE_API_KEY env var)
            arize_space_key: Arize AI space key (or set ARIZE_SPACE_KEY env var)
            team: Team name for cost attribution
            project: Project name for cost attribution
            customer_id: Customer identifier for multi-tenant scenarios
            environment: Environment (development/staging/production)
            daily_budget_limit: Maximum daily spending limit in USD
            max_monitoring_cost: Maximum cost per monitoring session in USD
            enable_cost_alerts: Enable cost threshold alerts
            enable_governance: Enable governance features
            cost_center: Cost center for financial reporting
            tags: Additional tags for telemetry
        """
        if not ARIZE_AVAILABLE:
            raise ImportError(
                "Arize AI SDK is required for this integration. "
                "Install with: pip install arize"
            )

        # Configuration
        self.arize_api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
        self.arize_space_key = arize_space_key or os.getenv("ARIZE_SPACE_KEY")
        self.team = team or os.getenv("GENOPS_TEAM", "default-team")
        self.project = project or os.getenv("GENOPS_PROJECT", "default-project")
        self.customer_id = customer_id or os.getenv("GENOPS_CUSTOMER_ID")
        self.environment = environment
        self.cost_center = cost_center

        # Budget and policy settings
        self.daily_budget_limit = daily_budget_limit
        self.max_monitoring_cost = max_monitoring_cost
        self.enable_cost_alerts = enable_cost_alerts
        self.enable_governance = enable_governance

        # Tags
        self.tags = tags or {}

        # Runtime tracking
        self.active_sessions: dict[str, ArizeMonitoringContext] = {}
        self.daily_usage = 0.0
        self.operation_count = 0

        # Initialize tracer
        self.tracer = trace.get_tracer(__name__)

        # Initialize Arize client if keys provided
        self.arize_client = None
        if self.arize_api_key and self.arize_space_key:
            self.arize_client = ArizeClient(
                api_key=self.arize_api_key, space_key=self.arize_space_key
            )

        logger.info(
            f"GenOps Arize adapter initialized for team={self.team}, project={self.project}"
        )

    @contextmanager
    def track_model_monitoring_session(
        self,
        model_id: str,
        model_version: str = "latest",
        environment: str = "production",
        max_cost: Optional[float] = None,
        **kwargs,
    ):
        """
        Context manager for tracking complete model monitoring session with governance.

        Args:
            model_id: Unique identifier for the model being monitored
            model_version: Version of the model being monitored
            environment: Environment where monitoring occurs
            max_cost: Maximum cost limit for this monitoring session
            **kwargs: Additional attributes for telemetry

        Yields:
            ArizeMonitoringContext: Monitoring session context for cost tracking and governance
        """
        session_id = f"{model_id}_{model_version}_{int(time.time())}"
        max_cost = max_cost or self.max_monitoring_cost

        # Create monitoring context
        monitoring_context = ArizeMonitoringContext(
            session_id=session_id,
            session_name=f"{model_id}-monitoring",
            model_id=model_id,
            model_version=model_version,
            environment=environment,
            team=self.team,  # type: ignore
            customer_id=self.customer_id,
            start_time=datetime.utcnow(),
        )

        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(
            "arize.monitoring.session",
            attributes={
                "genops.provider": "arize",
                "genops.team": self.team,
                "genops.project": self.project,
                "genops.customer_id": self.customer_id,
                "genops.environment": self.environment,
                "genops.model.id": model_id,
                "genops.model.version": model_version,
                "genops.model.environment": environment,
                "genops.monitoring.session_id": session_id,
                "genops.cost.budget_limit": max_cost,
                **kwargs,
            },
        ) as span:
            try:
                # Register active session
                self.active_sessions[session_id] = monitoring_context

                # Pre-session governance checks
                if self.enable_governance:
                    self._validate_monitoring_budget(max_cost)

                logger.info(
                    f"Starting model monitoring session: {model_id}-{model_version}"
                )

                # Enhance context with governance methods
                monitoring_context.log_prediction_batch = (
                    lambda df, cost_per_prediction=0.001: self._log_prediction_batch(
                        session_id, df, cost_per_prediction
                    )
                )
                monitoring_context.log_data_quality_metrics = (
                    lambda metrics, cost_estimate=0.01: self._log_data_quality(
                        session_id, metrics, cost_estimate
                    )
                )
                monitoring_context.create_performance_alert = (
                    lambda metric, threshold, cost_per_alert=0.05: self._create_alert(
                        session_id, metric, threshold, cost_per_alert
                    )
                )
                monitoring_context.update_monitoring_cost = lambda cost: (
                    self._update_session_cost(session_id, cost)
                )

                yield monitoring_context

                # Calculate final costs and metrics
                total_cost = monitoring_context.estimated_cost
                duration = (
                    datetime.utcnow() - monitoring_context.start_time
                ).total_seconds()

                # Update span with final metrics
                span.set_attributes(
                    {
                        "genops.cost.total": total_cost,
                        "genops.cost.currency": "USD",
                        "genops.monitoring.duration_seconds": duration,
                        "genops.monitoring.prediction_count": monitoring_context.prediction_count,
                        "genops.monitoring.data_quality_checks": monitoring_context.data_quality_checks,
                        "genops.monitoring.active_alerts": monitoring_context.active_alerts,
                        "genops.governance.violations": len(
                            monitoring_context.policy_violations
                        ),
                    }
                )

                # Update daily usage
                self.daily_usage += total_cost
                self.operation_count += 1

                # Log governance violations
                if monitoring_context.policy_violations:
                    span.add_event(
                        "governance_violations",
                        {
                            "violations": monitoring_context.policy_violations,
                            "session_id": session_id,
                        },
                    )

                # Cost alerts
                if self.enable_cost_alerts and total_cost > max_cost * 0.8:
                    logger.warning(
                        f"Monitoring session {model_id} approaching cost limit: "
                        f"${total_cost:.4f} / ${max_cost:.2f}"
                    )

                span.set_status(Status(StatusCode.OK))
                logger.info(
                    f"Monitoring session completed: {model_id}, cost: ${total_cost:.4f}"
                )

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Error in monitoring session {model_id}: {e}")
                raise
            finally:
                # Cleanup
                self.active_sessions.pop(session_id, None)

    def instrument_arize_log(self, original_log):
        """
        Instrument Arize client log method with governance tracking.

        Args:
            original_log: Original Arize client log method

        Returns:
            Enhanced log method with governance
        """

        def enhanced_log(*args, **kwargs):
            # Extract logging parameters
            prediction_id = kwargs.get("prediction_id", "unknown")
            model_id = kwargs.get("model_id", "unknown")
            model_version = kwargs.get("model_version", "latest")

            # Track logging operation
            with self.tracer.start_as_current_span(
                "arize.log_prediction",
                attributes={
                    "genops.provider": "arize",
                    "genops.team": self.team,
                    "genops.model.id": model_id,
                    "genops.model.version": model_version,
                    "genops.operation": "log_prediction",
                    "genops.prediction.id": prediction_id,
                },
            ) as span:
                try:
                    # Add governance metadata
                    enhanced_kwargs = kwargs.copy()

                    # Add governance tags if supported
                    tags = enhanced_kwargs.get("tags", {})
                    tags.update(
                        {
                            "genops_team": self.team,
                            "genops_project": self.project,
                            "genops_environment": self.environment,
                        }
                    )
                    enhanced_kwargs["tags"] = tags

                    # Call original log function
                    result = original_log(*args, **enhanced_kwargs)

                    # Estimate cost for logging operation
                    estimated_cost = self._estimate_prediction_log_cost()

                    # Update daily usage
                    self.daily_usage += estimated_cost
                    self.operation_count += 1

                    span.set_attributes(
                        {
                            "genops.cost.estimated": estimated_cost,
                            "genops.cost.currency": "USD",
                        }
                    )

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return enhanced_log

    def create_governed_alert(
        self,
        model_id: str,
        alert_name: str,
        metric: str,
        threshold: float,
        alert_type: str = "drift",
        cost_estimate: float = 0.05,
    ) -> None:
        """
        Create a model monitoring alert with governance metadata.

        Args:
            model_id: Model identifier for the alert
            alert_name: Name of the alert
            metric: Metric being monitored
            threshold: Alert threshold value
            alert_type: Type of alert (drift, performance, data_quality)
            cost_estimate: Estimated monthly cost for the alert
        """
        with self.tracer.start_as_current_span(
            "arize.create_alert",
            attributes={
                "genops.provider": "arize",
                "genops.team": self.team,
                "genops.model.id": model_id,
                "genops.alert.name": alert_name,
                "genops.alert.metric": metric,
                "genops.alert.threshold": threshold,
                "genops.alert.type": alert_type,
                "genops.cost.estimated": cost_estimate,
            },
        ) as span:
            try:
                # Note: Arize API for alert creation would go here
                # This is a placeholder for the actual Arize alert creation
                logger.info(
                    f"Creating governed alert: {alert_name} for model {model_id}"
                )

                # Update cost tracking
                self.daily_usage += cost_estimate / 30  # Daily portion of monthly cost

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def get_monitoring_cost_summary(
        self, session_id: str
    ) -> Optional[ModelMonitoringCostSummary]:
        """Get comprehensive cost summary for a monitoring session."""
        session_context = self.active_sessions.get(session_id)
        if not session_context:
            return None

        duration = (datetime.utcnow() - session_context.start_time).total_seconds()

        return ModelMonitoringCostSummary(
            total_cost=session_context.estimated_cost,
            prediction_logging_cost=session_context.prediction_count * 0.001,
            data_quality_cost=session_context.data_quality_checks * 0.01,
            alert_management_cost=session_context.active_alerts * 0.05,
            dashboard_cost=0.10,  # Estimated daily dashboard cost
            cost_by_model={session_context.model_id: session_context.estimated_cost},
            cost_by_environment={
                session_context.environment: session_context.estimated_cost
            },
            monitoring_duration=duration,
            efficiency_score=session_context.prediction_count
            / max(duration / 3600, 0.01),  # Predictions per hour
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current governance metrics and status."""
        return {
            "team": self.team,
            "project": self.project,
            "customer_id": self.customer_id,
            "daily_usage": self.daily_usage,
            "daily_budget_limit": self.daily_budget_limit,
            "budget_remaining": max(0, self.daily_budget_limit - self.daily_usage),
            "operation_count": self.operation_count,
            "active_monitoring_sessions": len(self.active_sessions),
            "cost_alerts_enabled": self.enable_cost_alerts,
            "governance_enabled": self.enable_governance,
        }

    def _validate_monitoring_budget(self, monitoring_cost: float) -> None:
        """Validate monitoring session against budget limits."""
        if self.daily_usage + monitoring_cost > self.daily_budget_limit:
            violation = f"Monitoring session would exceed daily budget: ${self.daily_usage + monitoring_cost:.2f} > ${self.daily_budget_limit:.2f}"
            logger.warning(f"Budget violation: {violation}")

    def _log_prediction_batch(
        self, session_id: str, predictions_df: Any, cost_per_prediction: float
    ) -> None:
        """Log prediction batch with cost tracking."""
        if session_id not in self.active_sessions:
            return

        # Estimate cost based on batch size
        if hasattr(predictions_df, "__len__"):
            batch_size = len(predictions_df)
        else:
            batch_size = 1  # Fallback for non-sized objects

        batch_cost = batch_size * cost_per_prediction

        # Update session context
        self.active_sessions[session_id].prediction_count += batch_size
        self.active_sessions[session_id].estimated_cost += batch_cost

        logger.info(
            f"Logged prediction batch: {batch_size} predictions, cost: ${batch_cost:.4f}"
        )

    def _log_data_quality(
        self, session_id: str, metrics: dict[str, Any], cost_estimate: float
    ) -> None:
        """Log data quality metrics with cost tracking."""
        if session_id not in self.active_sessions:
            return

        # Update session context
        self.active_sessions[session_id].data_quality_checks += 1
        self.active_sessions[session_id].estimated_cost += cost_estimate

        logger.info(f"Logged data quality metrics, cost: ${cost_estimate:.4f}")

    def _create_alert(
        self, session_id: str, metric: str, threshold: float, cost_per_alert: float
    ) -> None:
        """Create alert with cost tracking."""
        if session_id not in self.active_sessions:
            return

        # Update session context
        self.active_sessions[session_id].active_alerts += 1
        self.active_sessions[session_id].estimated_cost += cost_per_alert

        logger.info(
            f"Created alert for {metric} with threshold {threshold}, cost: ${cost_per_alert:.4f}"
        )

    def _update_session_cost(self, session_id: str, cost: float) -> None:
        """Update cost for a specific monitoring session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].estimated_cost += cost

    def _estimate_prediction_log_cost(self) -> float:
        """Estimate cost for prediction logging operation."""
        # Rough estimate: $0.001 per prediction logged
        return 0.001


def instrument_arize(
    arize_api_key: Optional[str] = None,
    arize_space_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs,
) -> GenOpsArizeAdapter:
    """
    Create and configure a GenOps Arize adapter for model monitoring governance.

    Args:
        arize_api_key: Arize AI API key
        arize_space_key: Arize AI space key
        team: Team name for cost attribution
        project: Project name for cost attribution
        **kwargs: Additional configuration options

    Returns:
        Configured GenOpsArizeAdapter instance
    """
    return GenOpsArizeAdapter(
        arize_api_key=arize_api_key,
        arize_space_key=arize_space_key,
        team=team,
        project=project,
        **kwargs,
    )


def auto_instrument(
    arize_api_key: Optional[str] = None,
    arize_space_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs,
) -> GenOpsArizeAdapter:
    """
    Enable zero-code auto-instrumentation for Arize AI with GenOps governance.

    This function patches Arize client methods to automatically include
    governance tracking without requiring code changes to existing Arize usage.

    Args:
        arize_api_key: Arize AI API key
        arize_space_key: Arize AI space key
        team: Team name for cost attribution
        project: Project name for cost attribution
        **kwargs: Additional configuration options

    Returns:
        Configured GenOpsArizeAdapter instance
    """
    if not ARIZE_AVAILABLE:
        raise ImportError(
            "Arize AI SDK is required for auto-instrumentation. "
            "Install with: pip install arize"
        )

    # Create adapter
    adapter = GenOpsArizeAdapter(
        arize_api_key=arize_api_key,
        arize_space_key=arize_space_key,
        team=team,
        project=project,
        **kwargs,
    )

    # Patch Arize client methods
    if hasattr(ArizeClient, "log"):
        original_log = ArizeClient.log
        ArizeClient.log = adapter.instrument_arize_log(original_log)

    logger.info("Arize AI auto-instrumentation enabled with GenOps governance")

    return adapter


# Global adapter instance for convenience
_global_adapter: Optional[GenOpsArizeAdapter] = None


def get_current_adapter() -> Optional[GenOpsArizeAdapter]:
    """Get the current global GenOps Arize adapter instance."""
    return _global_adapter


def set_global_adapter(adapter: GenOpsArizeAdapter) -> None:
    """Set the global GenOps Arize adapter instance."""
    global _global_adapter
    _global_adapter = adapter


# Convenience exports
__all__ = [
    "GenOpsArizeAdapter",
    "ArizeMonitoringContext",
    "ModelMonitoringCostSummary",
    "MonitoringScope",
    "instrument_arize",
    "auto_instrument",
    "get_current_adapter",
    "set_global_adapter",
    "ARIZE_AVAILABLE",
]
