"""
GenOps Elasticsearch Adapter - Main adapter class for governance telemetry export.

Provides high-level API for tracking AI operations, recording cost/policy telemetry,
and exporting to Elasticsearch with configurable modes (BATCH/REALTIME/HYBRID).
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .client import ElasticAPIClient
from .event_exporter import EventExporter, ExportMode

logger = logging.getLogger(__name__)


class GenOpsElasticAdapter:
    """
    Main adapter for GenOps Elasticsearch integration.

    Provides:
    - Context manager for tracking AI operations
    - Cost telemetry recording
    - Policy enforcement recording
    - Budget tracking
    - Configurable export modes (BATCH/REALTIME/HYBRID)
    """

    def __init__(
        self,
        elastic_url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        api_id: Optional[str] = None,
        index_prefix: str = "genops-ai",
        namespace: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: str = "development",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        feature: Optional[str] = None,
        export_mode: str = "batch",
        batch_size: int = 100,
        batch_interval_seconds: int = 60,
        ilm_enabled: bool = True,
        ilm_retention_days: int = 90,
        verify_certs: bool = True,
        ca_certs: Optional[str] = None,
        auto_validate: bool = True,
    ):
        """
        Initialize Elasticsearch adapter for GenOps governance telemetry.

        Args:
            elastic_url: Elasticsearch cluster URL (env: ELASTIC_URL)
            cloud_id: Elastic Cloud deployment ID (env: ELASTIC_CLOUD_ID)
            username: Basic auth username (env: ELASTIC_USERNAME)
            password: Basic auth password (env: ELASTIC_PASSWORD)
            api_key: API key for authentication (env: ELASTIC_API_KEY)
            api_id: API key ID (env: ELASTIC_API_ID)
            index_prefix: Prefix for index names (default: "genops-ai")
            namespace: Optional namespace for multi-tenant indexing
            team: Default team for governance attribution
            project: Default project for cost tracking
            environment: Environment (development/staging/production)
            customer_id: Customer ID for multi-tenant tracking
            cost_center: Cost center for financial reporting
            feature: Feature name for cost attribution
            export_mode: Export mode - "batch", "realtime", or "hybrid"
            batch_size: Maximum batch size before flush (batch mode)
            batch_interval_seconds: Flush interval in seconds (batch mode)
            ilm_enabled: Enable Index Lifecycle Management
            ilm_retention_days: Data retention period in days
            verify_certs: Verify SSL certificates
            ca_certs: Path to CA certificate bundle
            auto_validate: Automatically validate setup on initialization
        """
        # Environment variable fallbacks
        self.elastic_url = elastic_url or os.getenv("ELASTIC_URL")
        self.cloud_id = cloud_id or os.getenv("ELASTIC_CLOUD_ID")
        self.username = username or os.getenv("ELASTIC_USERNAME")
        self.password = password or os.getenv("ELASTIC_PASSWORD")
        self.api_key = api_key or os.getenv("ELASTIC_API_KEY")
        self.api_id = api_id or os.getenv("ELASTIC_API_ID")

        # Governance attributes
        self.team = team
        self.project = project
        self.environment = environment
        self.customer_id = customer_id
        self.cost_center = cost_center
        self.feature = feature

        # Configuration
        self.index_prefix = index_prefix
        self.namespace = namespace or team  # Use team as namespace if not specified
        self.ilm_enabled = ilm_enabled
        self.ilm_retention_days = ilm_retention_days

        # Parse export mode
        try:
            self.export_mode = ExportMode(export_mode.lower())
        except ValueError:
            logger.warning(f"Invalid export_mode '{export_mode}', defaulting to BATCH")
            self.export_mode = ExportMode.BATCH

        # Initialize Elasticsearch client
        self.client = ElasticAPIClient(
            elastic_url=self.elastic_url,
            cloud_id=self.cloud_id,
            username=self.username,
            password=self.password,
            api_key=self.api_key,
            api_id=self.api_id,
            verify_certs=verify_certs,
            ca_certs=ca_certs,
        )

        # Initialize event exporter
        self.exporter = EventExporter(
            client=self.client,
            index_prefix=self.index_prefix,
            namespace=self.namespace,
            export_mode=self.export_mode,
            batch_size=batch_size,
            batch_interval_seconds=batch_interval_seconds,
            enable_background_flush=(self.export_mode == ExportMode.BATCH),
        )

        # Get OpenTelemetry tracer
        self.tracer = trace.get_tracer(__name__)

        # Setup ILM if enabled
        if self.ilm_enabled:
            self._setup_ilm()

        # Validate setup if requested
        if auto_validate:
            self._validate_setup()

        logger.info(
            f"GenOpsElasticAdapter initialized "
            f"(mode: {self.export_mode.value}, namespace: {self.namespace})"
        )

    def _validate_setup(self):
        """Validate Elasticsearch connection and configuration."""
        try:
            if not self.client.health_check():
                logger.warning("Elasticsearch health check failed")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")

    def _setup_ilm(self):
        """Setup Index Lifecycle Management policy."""
        try:
            policy_name = f"{self.index_prefix}-ilm-policy"
            self.client.create_ilm_policy(
                policy_name=policy_name,
                retention_days=self.ilm_retention_days,
            )
            logger.info(
                f"ILM policy created: {policy_name} (retention: {self.ilm_retention_days} days)"
            )
        except Exception as e:
            logger.warning(f"ILM setup failed (may not be supported): {e}")

    @contextmanager
    def track_ai_operation(
        self,
        operation_name: str,
        operation_type: str = "ai_operation",
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: Optional[str] = None,
        customer_id: Optional[str] = None,
        feature: Optional[str] = None,
        **attributes,
    ) -> Iterator[Span]:
        """
        Track an AI operation with OpenTelemetry span and Elasticsearch export.

        Usage:
            with adapter.track_ai_operation("gpt4-completion") as span:
                # AI operation code
                adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

        Args:
            operation_name: Name of the operation
            operation_type: Type of operation (default: "ai_operation")
            team: Override default team
            project: Override default project
            environment: Override default environment
            customer_id: Override default customer_id
            feature: Override default feature
            **attributes: Additional custom attributes

        Yields:
            OpenTelemetry Span for the operation
        """
        # Use provided values or fall back to defaults
        final_team = team or self.team
        final_project = project or self.project
        final_environment = environment or self.environment
        final_customer_id = customer_id or self.customer_id
        final_feature = feature or self.feature

        # Create span
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add governance attributes
            if final_team:
                span.set_attribute("genops.team", final_team)
            if final_project:
                span.set_attribute("genops.project", final_project)
            if final_environment:
                span.set_attribute("genops.environment", final_environment)
            if final_customer_id:
                span.set_attribute("genops.customer_id", final_customer_id)
            if self.cost_center:
                span.set_attribute("genops.cost_center", self.cost_center)
            if final_feature:
                span.set_attribute("genops.feature", final_feature)

            # Add operation type
            span.set_attribute("genops.operation_type", operation_type)

            # Add custom attributes
            for key, value in attributes.items():
                span.set_attribute(f"genops.{key}", value)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                # Export span to Elasticsearch
                self._export_span(span, operation_type=operation_type)

    def _export_span(self, span: Span, operation_type: str = "ai_operation"):
        """Export span to Elasticsearch via event exporter."""
        try:
            # Get span context
            span_context = span.get_span_context()

            # Extract span data
            span_data = {
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
                "name": span.name,
                "operation_type": operation_type,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status": {
                    "status_code": span.status.status_code.name
                    if span.status
                    else "UNSET"
                },
                "attributes": span.attributes if hasattr(span, "attributes") else {},
            }

            # Determine if critical (for HYBRID mode)
            is_critical = (
                span_data["status"]["status_code"] == "ERROR"
                or span_data["attributes"].get("genops.policy.result") == "blocked"
            )

            # Export
            self.exporter.export_span(span_data, is_critical=is_critical)

        except Exception as e:
            logger.error(f"Failed to export span: {e}")

    def record_cost(
        self,
        span: Span,
        cost: float,
        provider: str,
        model: str,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost_input: Optional[float] = None,
        cost_output: Optional[float] = None,
    ):
        """
        Record cost telemetry for an AI operation.

        Args:
            span: OpenTelemetry span to attach cost data
            cost: Total cost in USD
            provider: AI provider (e.g., "openai", "anthropic", "bedrock")
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            tokens_input: Input tokens consumed
            tokens_output: Output tokens generated
            cost_input: Input token cost (if calculated separately)
            cost_output: Output token cost (if calculated separately)
        """
        span.set_attribute("genops.cost.total", cost)
        span.set_attribute("genops.cost.provider", provider)
        span.set_attribute("genops.cost.model", model)

        if cost_input is not None:
            span.set_attribute("genops.cost.input", cost_input)
        if cost_output is not None:
            span.set_attribute("genops.cost.output", cost_output)

        if tokens_input is not None:
            span.set_attribute("genops.tokens.input", tokens_input)
        if tokens_output is not None:
            span.set_attribute("genops.tokens.output", tokens_output)
        if tokens_input and tokens_output:
            span.set_attribute("genops.tokens.total", tokens_input + tokens_output)

        logger.debug(
            f"Recorded cost: ${cost:.4f} ({provider}/{model}, "
            f"tokens: {tokens_input or 0}+{tokens_output or 0})"
        )

    def record_policy(
        self,
        span: Span,
        policy_name: str,
        result: str,
        reason: Optional[str] = None,
    ):
        """
        Record policy enforcement telemetry.

        Args:
            span: OpenTelemetry span to attach policy data
            policy_name: Name of the policy evaluated
            result: Policy result ("allowed", "blocked", "warning")
            reason: Optional reason for the decision
        """
        span.set_attribute("genops.policy.name", policy_name)
        span.set_attribute("genops.policy.result", result)

        if reason:
            span.set_attribute("genops.policy.reason", reason)

        logger.debug(f"Recorded policy: {policy_name} -> {result}")

    def record_budget(
        self,
        span: Span,
        budget_id: str,
        limit: float,
        consumed: float,
        remaining: float,
    ):
        """
        Record budget tracking telemetry.

        Args:
            span: OpenTelemetry span to attach budget data
            budget_id: Budget identifier
            limit: Budget limit in USD
            consumed: Amount consumed so far
            remaining: Amount remaining
        """
        span.set_attribute("genops.budget.id", budget_id)
        span.set_attribute("genops.budget.limit", limit)
        span.set_attribute("genops.budget.consumed", consumed)
        span.set_attribute("genops.budget.remaining", remaining)

        logger.debug(
            f"Recorded budget: {budget_id} "
            f"(${consumed:.2f}/${limit:.2f}, ${remaining:.2f} remaining)"
        )

    def flush(self) -> int:
        """
        Force flush of batch buffer.

        Returns:
            Number of documents exported
        """
        return self.exporter.flush()

    def shutdown(self):
        """Gracefully shutdown adapter, flushing pending data."""
        logger.info("Shutting down GenOpsElasticAdapter")
        self.exporter.shutdown()
        self.client.close()

    def get_metrics(self) -> dict[str, Any]:
        """
        Get adapter metrics and statistics.

        Returns:
            Dictionary with export stats, cluster info, etc.
        """
        return {
            "adapter": {
                "export_mode": self.export_mode.value,
                "index_prefix": self.index_prefix,
                "namespace": self.namespace,
                "team": self.team,
                "project": self.project,
                "environment": self.environment,
            },
            "exporter": self.exporter.get_stats(),
            "cluster": {
                "version": self.client.get_version(),
            },
        }

    def get_export_summary(self) -> dict[str, Any]:
        """
        Get export performance summary.

        Returns:
            Dictionary with export statistics
        """
        return self.exporter.get_stats()
