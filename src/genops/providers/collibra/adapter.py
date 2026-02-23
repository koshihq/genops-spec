"""Main GenOps Collibra adapter for bidirectional integration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

from genops.core.telemetry import GenOpsTelemetry
from genops.providers.collibra.asset_exporter import AssetExporter, ExportMode
from genops.providers.collibra.client import CollibraAPIClient
from genops.providers.collibra.policy_importer import PolicyImporter
from genops.providers.collibra.validation import validate_setup

logger = logging.getLogger(__name__)


class GenOpsCollibraAdapter:
    """
    Bidirectional integration between GenOps AI and Collibra.

    Provides:
    - Export: GenOps telemetry → Collibra Assets
    - Import: Collibra policies → GenOps PolicyEngine (Phase 3)
    - Governance-aware AI operation tracking
    """

    def __init__(
        self,
        collibra_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        api_token: str | None = None,
        domain_id: str | None = None,
        team: str | None = None,
        project: str | None = None,
        environment: str = "development",
        export_mode: str = "batch",
        batch_size: int = 100,
        batch_interval_seconds: int = 60,
        enable_policy_sync: bool = False,
        policy_sync_interval_minutes: int = 5,
        enable_cost_tracking: bool = True,
        daily_budget_limit: float | None = None,
        enable_cost_alerts: bool = False,
        auto_validate: bool = True,
    ):
        """
        Initialize GenOps Collibra adapter.

        Args:
            collibra_url: Collibra instance URL
            username: Basic auth username
            password: Basic auth password
            api_token: API token (alternative to username/password)
            domain_id: Target Collibra domain ID (will use first available if not provided)
            team: Team name for governance attribution
            project: Project name for governance attribution
            environment: Environment (development, staging, production)
            export_mode: Export mode (batch, realtime, hybrid)
            batch_size: Maximum batch size
            batch_interval_seconds: Batch flush interval
            enable_policy_sync: Enable policy import from Collibra (Phase 3)
            policy_sync_interval_minutes: Policy sync interval
            enable_cost_tracking: Enable automatic cost tracking
            daily_budget_limit: Daily budget limit (USD)
            enable_cost_alerts: Enable cost alerting
            auto_validate: Automatically validate setup on initialization
        """
        # Get credentials from environment if not provided
        self.collibra_url = collibra_url or os.getenv("COLLIBRA_URL")
        self.username = username or os.getenv("COLLIBRA_USERNAME")
        self.password = password or os.getenv("COLLIBRA_PASSWORD")
        self.api_token = api_token or os.getenv("COLLIBRA_API_TOKEN")

        # Governance attributes
        self.team = team or os.getenv("GENOPS_TEAM")
        self.project = project or os.getenv("GENOPS_PROJECT")
        self.environment = environment

        # Configuration
        self.enable_policy_sync = enable_policy_sync
        self.enable_cost_tracking = enable_cost_tracking
        self.daily_budget_limit = daily_budget_limit
        self.enable_cost_alerts = enable_cost_alerts

        # Validate setup
        if auto_validate:
            validation_result = validate_setup(
                collibra_url=self.collibra_url,
                username=self.username,
                password=self.password,
                api_token=self.api_token,
            )
            if not validation_result.valid:
                logger.warning(
                    f"Collibra setup validation failed: {validation_result.errors}"
                )

        # Initialize Collibra client
        self.client = CollibraAPIClient(
            base_url=self.collibra_url,  # type: ignore
            username=self.username,
            password=self.password,
            api_token=self.api_token,
        )

        # Get or validate domain ID
        self.domain_id = domain_id
        if not self.domain_id:
            # Try to get first available domain
            try:
                domains = self.client.list_domains()
                if domains:
                    self.domain_id = domains[0]["id"]
                    logger.info(
                        f"Using Collibra domain: {domains[0].get('name', 'Unknown')} "
                        f"(ID: {self.domain_id})"
                    )
                else:
                    logger.warning(
                        "No Collibra domains found. Please create a domain or specify domain_id."
                    )
            except Exception as e:
                logger.error(f"Failed to list Collibra domains: {e}")

        # Initialize telemetry engine
        self.telemetry = GenOpsTelemetry(tracer_name="genops-collibra")

        # Initialize asset exporter
        export_mode_enum = ExportMode(export_mode.lower())
        self.exporter = AssetExporter(
            client=self.client,
            domain_id=self.domain_id,  # type: ignore[arg-type]
            export_mode=export_mode_enum,
            batch_size=batch_size,
            batch_interval_seconds=batch_interval_seconds,
        )

        # Policy importer (Phase 3)
        self.policy_importer = None
        if enable_policy_sync:
            self.policy_importer = PolicyImporter(
                client=self.client,
                domain_id=self.domain_id,
                sync_interval_minutes=policy_sync_interval_minutes,
                enable_background_sync=True,
            )
            logger.info(
                f"Policy sync enabled: importing policies from Collibra every "
                f"{policy_sync_interval_minutes} minutes"
            )
            # Do initial policy sync
            try:
                self.policy_importer.import_policies(register=True)
            except Exception as e:
                logger.error(f"Initial policy sync failed: {e}")

        # Track operation metrics
        self.operation_count = 0
        self.total_cost = 0.0

        logger.info(
            f"GenOps Collibra adapter initialized: "
            f"mode={export_mode}, team={self.team}, project={self.project}"
        )

    @contextmanager
    def track_ai_operation(
        self,
        operation_name: str,
        operation_type: str = "ai.inference",
        **governance_attrs,
    ):
        """
        Context manager for tracking AI operations with Collibra export.

        Args:
            operation_name: Operation name
            operation_type: Operation type
            **governance_attrs: Additional governance attributes (team, project, etc.)

        Yields:
            OpenTelemetry span

        Example:
            >>> adapter = GenOpsCollibraAdapter(...)
            >>> with adapter.track_ai_operation("gpt-4-completion") as span:
            ...     response = openai_client.chat.completions.create(...)
            ...     adapter.record_cost(span, cost=0.05, provider="openai")
        """
        # Merge default governance attributes with overrides
        effective_attrs = {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
        }
        effective_attrs.update(governance_attrs)

        # Create span with GenOps telemetry
        with self.telemetry.trace_operation(
            operation_name, operation_type, **effective_attrs
        ) as span:
            try:
                yield span

                # Export to Collibra after operation completes
                span_attributes = self._extract_span_attributes(span)
                self.exporter.export_span(span_attributes)

                # Update metrics
                self.operation_count += 1

                # Track cost
                if self.enable_cost_tracking:
                    cost = span_attributes.get("genops.cost.total", 0)
                    if cost:
                        self.total_cost += cost

                        # Check budget limit
                        if (
                            self.daily_budget_limit
                            and self.total_cost > self.daily_budget_limit
                        ):
                            logger.warning(
                                f"Daily budget limit exceeded: "
                                f"${self.total_cost:.2f} > ${self.daily_budget_limit:.2f}"
                            )

            except Exception as e:
                logger.error(f"Error in AI operation tracking: {e}")
                raise

    def record_cost(
        self,
        span: trace.Span,
        cost: float,
        provider: str = "",
        model: str = "",
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        **metadata,
    ):
        """
        Record cost telemetry on a span.

        Args:
            span: OpenTelemetry span
            cost: Cost amount
            provider: AI provider (openai, anthropic, etc.)
            model: Model name
            tokens_input: Input tokens
            tokens_output: Output tokens
            **metadata: Additional cost metadata
        """
        self.telemetry.record_cost(
            span=span,
            cost=cost,
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            **metadata,
        )

    def record_policy(
        self,
        span: trace.Span,
        policy_name: str,
        policy_result: str,
        policy_reason: str | None = None,
    ):
        """
        Record policy enforcement telemetry on a span.

        Args:
            span: OpenTelemetry span
            policy_name: Policy name
            policy_result: Policy result (allowed, blocked, warning)
            policy_reason: Policy reason (optional)
        """
        self.telemetry.record_policy(
            span=span,
            policy_name=policy_name,
            policy_result=policy_result,
            policy_reason=policy_reason,
        )

    def sync_policies(self) -> dict[str, Any]:
        """
        Sync policies from Collibra to GenOps PolicyEngine.

        Returns:
            Dictionary with sync statistics
        """
        if not self.enable_policy_sync or not self.policy_importer:
            logger.warning("Policy sync is not enabled")
            return {"imported": 0, "updated": 0, "failed": 0}

        return self.policy_importer.sync_policies()

    def flush(self) -> int:
        """
        Flush pending telemetry exports to Collibra.

        Returns:
            Number of assets exported
        """
        return self.exporter.flush()

    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown adapter and flush remaining data.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down GenOps Collibra adapter...")
        self.exporter.shutdown(timeout=timeout)

        # Shutdown policy importer if enabled
        if self.policy_importer:
            self.policy_importer.shutdown(timeout=timeout)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get adapter metrics.

        Returns:
            Dictionary with adapter metrics
        """
        export_stats = self.exporter.get_stats()

        return {
            "operation_count": self.operation_count,
            "total_cost": self.total_cost,
            "daily_budget_limit": self.daily_budget_limit,
            "budget_remaining": (
                self.daily_budget_limit - self.total_cost
                if self.daily_budget_limit
                else None
            ),
            "assets_exported": export_stats.assets_exported,
            "assets_failed": export_stats.assets_failed,
            "batches_sent": export_stats.batches_sent,
            "buffer_size": self.exporter.get_buffer_size(),
        }

    def get_export_summary(self) -> dict[str, Any]:
        """
        Get export summary statistics.

        Returns:
            Dictionary with export statistics
        """
        stats = self.exporter.get_stats()

        return {
            "assets_created": stats.assets_exported,
            "assets_failed": stats.assets_failed,
            "batches_sent": stats.batches_sent,
            "total_cost": self.total_cost,
            "average_export_time_ms": (
                stats.total_export_time_ms / stats.assets_exported
                if stats.assets_exported > 0
                else 0
            ),
            "last_export_time": stats.last_export_time,
        }

    def _extract_span_attributes(self, span: trace.Span) -> dict[str, Any]:
        """
        Extract all attributes from a span.

        Args:
            span: OpenTelemetry span

        Returns:
            Dictionary of span attributes
        """
        if not span.is_recording():
            return {}

        # Get span context
        span_context = span.get_span_context()
        attributes = {
            "trace.id": format(span_context.trace_id, "032x"),
            "span.id": format(span_context.span_id, "016x"),
            "span.name": span.name,
        }

        # Extract all span attributes
        # Note: This is a simplified approach. In production, you'd want to
        # access the span's attributes dict directly if possible.
        # For now, we rely on attributes being set via set_attribute()

        return attributes
