"""
Event exporter for GenOps Elasticsearch integration.

Handles telemetry export with multiple modes:
- BATCH: Buffer events and export in bulk for optimal performance
- REALTIME: Export each event immediately for low-latency monitoring
- HYBRID: Critical events in realtime, others batched
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from queue import Queue, Empty

from .client import ElasticAPIClient, ElasticDocument

logger = logging.getLogger(__name__)


class ExportMode(Enum):
    """Export mode for telemetry data."""
    BATCH = "batch"
    REALTIME = "realtime"
    HYBRID = "hybrid"


@dataclass
class ExportStats:
    """Statistics for export operations."""
    total_exported: int = 0
    total_failed: int = 0
    total_batches: int = 0
    total_realtime: int = 0
    last_export_timestamp: Optional[str] = None
    last_batch_size: int = 0
    last_export_duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def record_success(self, count: int, duration_ms: float, is_batch: bool = True):
        """Record a successful export."""
        self.total_exported += count
        if is_batch:
            self.total_batches += 1
            self.last_batch_size = count
        else:
            self.total_realtime += 1
        self.last_export_timestamp = datetime.utcnow().isoformat()
        self.last_export_duration_ms = duration_ms

    def record_failure(self, error: str):
        """Record a failed export."""
        self.total_failed += 1
        self.errors.append(f"{datetime.utcnow().isoformat()}: {error}")
        # Keep only last 10 errors
        if len(self.errors) > 10:
            self.errors = self.errors[-10:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_exported": self.total_exported,
            "total_failed": self.total_failed,
            "total_batches": self.total_batches,
            "total_realtime": self.total_realtime,
            "last_export_timestamp": self.last_export_timestamp,
            "last_batch_size": self.last_batch_size,
            "last_export_duration_ms": self.last_export_duration_ms,
            "recent_errors": self.errors,
        }


class EventExporter:
    """
    Manages export of GenOps telemetry to Elasticsearch with configurable modes.

    Supports:
    - BATCH: Buffer up to batch_size events, flush every batch_interval_seconds
    - REALTIME: Export each event immediately
    - HYBRID: Critical events (policy violations, errors) in realtime, others batched
    """

    def __init__(
        self,
        client: ElasticAPIClient,
        index_prefix: str = "genops-ai",
        namespace: Optional[str] = None,
        export_mode: ExportMode = ExportMode.BATCH,
        batch_size: int = 100,
        batch_interval_seconds: int = 60,
        enable_background_flush: bool = True,
    ):
        """
        Initialize event exporter.

        Args:
            client: ElasticAPIClient instance
            index_prefix: Prefix for index names (e.g., "genops-ai")
            namespace: Optional namespace for multi-tenant indexing
            export_mode: Export mode (BATCH, REALTIME, or HYBRID)
            batch_size: Maximum batch size before forcing flush
            batch_interval_seconds: Time interval for periodic batch flush
            enable_background_flush: Enable background thread for periodic flushing
        """
        self.client = client
        self.index_prefix = index_prefix
        self.namespace = namespace
        self.export_mode = export_mode
        self.batch_size = batch_size
        self.batch_interval_seconds = batch_interval_seconds

        # Batch buffer and thread safety
        self._buffer: List[ElasticDocument] = []
        self._buffer_lock = threading.Lock()

        # Statistics
        self.stats = ExportStats()

        # Background flush thread
        self._background_flush_enabled = enable_background_flush
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush_thread = threading.Event()

        if self._background_flush_enabled and export_mode == ExportMode.BATCH:
            self._start_background_flush()

    def export_span(
        self,
        span_data: Dict[str, Any],
        is_critical: bool = False,
    ) -> bool:
        """
        Export a span to Elasticsearch based on export mode.

        Args:
            span_data: Span data to export
            is_critical: Whether this is a critical event (affects HYBRID mode)

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            # Convert span data to ElasticDocument
            document = self._span_to_document(span_data)

            # Determine export strategy
            should_export_immediately = (
                self.export_mode == ExportMode.REALTIME
                or (self.export_mode == ExportMode.HYBRID and is_critical)
            )

            if should_export_immediately:
                return self._export_realtime(document)
            else:
                return self._add_to_batch(document)

        except Exception as e:
            logger.error(f"Failed to export span: {e}")
            self.stats.record_failure(str(e))
            return False

    def _span_to_document(self, span_data: Dict[str, Any]) -> ElasticDocument:
        """Convert span data to ElasticDocument."""
        # Extract core fields
        trace_id = span_data.get("trace_id", "unknown")
        span_id = span_data.get("span_id", "unknown")
        operation_name = span_data.get("name", "unknown")
        operation_type = span_data.get("operation_type", "ai_operation")

        # Extract timestamps
        start_time = span_data.get("start_time")
        end_time = span_data.get("end_time")

        # Calculate duration
        duration_ms = None
        if start_time and end_time:
            duration_ms = (end_time - start_time) / 1_000_000  # Convert nanoseconds to milliseconds

        # Use end_time as timestamp, fallback to start_time or current time
        timestamp = end_time or start_time or time.time_ns()
        timestamp_iso = datetime.utcfromtimestamp(timestamp / 1_000_000_000).isoformat()

        # Extract attributes
        attributes = span_data.get("attributes", {})

        # Build document with governance fields
        document = ElasticDocument(
            timestamp=timestamp_iso,
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            operation_type=operation_type,
            team=attributes.get("genops.team"),
            project=attributes.get("genops.project"),
            environment=attributes.get("genops.environment", "production"),
            customer_id=attributes.get("genops.customer_id"),
            cost_center=attributes.get("genops.cost_center"),
            feature=attributes.get("genops.feature"),
            cost_total=attributes.get("genops.cost.total"),
            cost_input=attributes.get("genops.cost.input"),
            cost_output=attributes.get("genops.cost.output"),
            cost_provider=attributes.get("genops.cost.provider"),
            cost_model=attributes.get("genops.cost.model"),
            tokens_input=attributes.get("genops.tokens.input"),
            tokens_output=attributes.get("genops.tokens.output"),
            tokens_total=attributes.get("genops.tokens.total"),
            policy_name=attributes.get("genops.policy.name"),
            policy_result=attributes.get("genops.policy.result"),
            policy_reason=attributes.get("genops.policy.reason"),
            budget_id=attributes.get("genops.budget.id"),
            budget_limit=attributes.get("genops.budget.limit"),
            budget_consumed=attributes.get("genops.budget.consumed"),
            budget_remaining=attributes.get("genops.budget.remaining"),
            duration_ms=duration_ms,
            status=span_data.get("status", {}).get("status_code", "success"),
            attributes={k: v for k, v in attributes.items() if not k.startswith("genops.")},
        )

        return document

    def _export_realtime(self, document: ElasticDocument) -> bool:
        """Export a single document immediately."""
        try:
            start_time = time.time()
            index_name = self._get_index_name()

            self.client.index_document(index=index_name, document=document)

            duration_ms = (time.time() - start_time) * 1000
            self.stats.record_success(count=1, duration_ms=duration_ms, is_batch=False)

            logger.debug(f"Exported document to {index_name} in {duration_ms:.2f}ms")
            return True

        except Exception as e:
            logger.error(f"Realtime export failed: {e}")
            self.stats.record_failure(str(e))
            return False

    def _add_to_batch(self, document: ElasticDocument) -> bool:
        """Add document to batch buffer."""
        with self._buffer_lock:
            self._buffer.append(document)

            # Check if batch is full
            if len(self._buffer) >= self.batch_size:
                logger.debug(f"Batch full ({len(self._buffer)} documents), triggering flush")
                self._flush_batch()

        return True

    def flush(self) -> int:
        """
        Force flush of batch buffer.

        Returns:
            Number of documents exported
        """
        with self._buffer_lock:
            return self._flush_batch()

    def _flush_batch(self) -> int:
        """
        Flush batch buffer to Elasticsearch (must be called with lock held).

        Returns:
            Number of documents exported
        """
        if not self._buffer:
            return 0

        try:
            start_time = time.time()
            index_name = self._get_index_name()

            # Export batch
            result = self.client.bulk_index(index=index_name, documents=self._buffer)

            duration_ms = (time.time() - start_time) * 1000
            success_count = result.get("success", 0)

            self.stats.record_success(count=success_count, duration_ms=duration_ms, is_batch=True)

            # Log errors if any
            if result.get("errors"):
                for error in result["errors"][:5]:  # Log first 5 errors
                    logger.warning(f"Bulk index error: {error}")
                    self.stats.record_failure(str(error))

            logger.info(
                f"Flushed {success_count}/{len(self._buffer)} documents to {index_name} "
                f"in {duration_ms:.2f}ms"
            )

            # Clear buffer
            exported_count = len(self._buffer)
            self._buffer.clear()

            return exported_count

        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            self.stats.record_failure(str(e))
            return 0

    def _get_index_name(self) -> str:
        """
        Generate index name with time-based suffix.

        Format: {prefix}-{namespace}-{YYYY.MM.DD}
        Example: genops-ai-ml-team-2025.01.18
        """
        date_suffix = datetime.utcnow().strftime("%Y.%m.%d")

        if self.namespace:
            return f"{self.index_prefix}-{self.namespace}-{date_suffix}"
        else:
            return f"{self.index_prefix}-{date_suffix}"

    def _start_background_flush(self):
        """Start background thread for periodic batch flushing."""
        if self._flush_thread and self._flush_thread.is_alive():
            logger.warning("Background flush thread already running")
            return

        self._stop_flush_thread.clear()
        self._flush_thread = threading.Thread(
            target=self._background_flush_loop,
            daemon=True,
            name="elastic-background-flush",
        )
        self._flush_thread.start()
        logger.info(
            f"Started background flush thread (interval: {self.batch_interval_seconds}s)"
        )

    def _background_flush_loop(self):
        """Background thread loop for periodic flushing."""
        while not self._stop_flush_thread.is_set():
            # Wait for interval or stop signal
            if self._stop_flush_thread.wait(timeout=self.batch_interval_seconds):
                break

            # Flush if buffer has data
            with self._buffer_lock:
                if self._buffer:
                    logger.debug(
                        f"Background flush triggered ({len(self._buffer)} documents buffered)"
                    )
                    self._flush_batch()

        logger.info("Background flush thread stopped")

    def shutdown(self):
        """
        Gracefully shutdown exporter, flushing any pending data.
        """
        logger.info("Shutting down event exporter")

        # Stop background thread
        if self._flush_thread and self._flush_thread.is_alive():
            self._stop_flush_thread.set()
            self._flush_thread.join(timeout=5)

        # Final flush
        with self._buffer_lock:
            if self._buffer:
                logger.info(f"Final flush: {len(self._buffer)} documents")
                self._flush_batch()

        logger.info("Event exporter shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        return self.stats.to_dict()
