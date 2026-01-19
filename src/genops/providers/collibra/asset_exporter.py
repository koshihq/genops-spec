"""Asset exporter for sending GenOps telemetry to Collibra."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from genops.providers.collibra.client import CollibraAPIClient, CollibraAPIError
from genops.providers.collibra.mapping import create_collibra_asset_from_span

logger = logging.getLogger(__name__)


class ExportMode(Enum):
    """Export modes for telemetry."""

    BATCH = "batch"  # Accumulate and send in batches
    REALTIME = "realtime"  # Send immediately
    HYBRID = "hybrid"  # Critical events real-time, others batched


@dataclass
class ExportStats:
    """Statistics for export operations."""

    assets_exported: int = 0
    assets_failed: int = 0
    batches_sent: int = 0
    total_export_time_ms: float = 0
    last_export_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    def record_success(self, count: int = 1, duration_ms: float = 0):
        """Record successful export."""
        self.assets_exported += count
        self.total_export_time_ms += duration_ms
        self.last_export_time = time.time()

    def record_failure(self, count: int = 1, error: Optional[str] = None):
        """Record failed export."""
        self.assets_failed += count
        if error:
            self.errors.append(error)

    def record_batch(self):
        """Record batch sent."""
        self.batches_sent += 1


class AssetExporter:
    """
    Export GenOps telemetry to Collibra as assets.

    Supports multiple export modes:
    - Batch: Accumulate spans and export in batches (reduces API calls)
    - Real-time: Export immediately after span completion
    - Hybrid: Critical events real-time, regular operations batched
    """

    def __init__(
        self,
        client: CollibraAPIClient,
        domain_id: str,
        export_mode: ExportMode = ExportMode.BATCH,
        batch_size: int = 100,
        batch_interval_seconds: int = 60,
        enable_background_flush: bool = True,
    ):
        """
        Initialize asset exporter.

        Args:
            client: Collibra API client
            domain_id: Target Collibra domain ID for assets
            export_mode: Export mode (batch, realtime, hybrid)
            batch_size: Maximum batch size before auto-flush
            batch_interval_seconds: Time interval for batch flush
            enable_background_flush: Enable background thread for periodic flush
        """
        self.client = client
        self.domain_id = domain_id
        self.export_mode = export_mode
        self.batch_size = batch_size
        self.batch_interval_seconds = batch_interval_seconds

        # Batch buffer
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_lock = threading.Lock()

        # Statistics
        self.stats = ExportStats()

        # Background flush thread
        self.background_flush_enabled = enable_background_flush
        self.background_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        if self.background_flush_enabled and export_mode == ExportMode.BATCH:
            self._start_background_flush()

    def export_span(
        self, span_attributes: Dict[str, Any], asset_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Export GenOps span as Collibra asset.

        Args:
            span_attributes: GenOps span attributes
            asset_type: Override asset type (optional)

        Returns:
            Created asset data (for real-time mode) or None (for batch mode)
        """
        if self.export_mode == ExportMode.REALTIME:
            return self._export_realtime(span_attributes, asset_type)
        elif self.export_mode == ExportMode.BATCH:
            self._export_batch(span_attributes, asset_type)
            return None
        elif self.export_mode == ExportMode.HYBRID:
            # Check if this is a critical event
            if self._is_critical_event(span_attributes):
                return self._export_realtime(span_attributes, asset_type)
            else:
                self._export_batch(span_attributes, asset_type)
                return None

    def _export_realtime(
        self, span_attributes: Dict[str, Any], asset_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Export span immediately to Collibra.

        Args:
            span_attributes: GenOps span attributes
            asset_type: Override asset type

        Returns:
            Created asset data or None on failure
        """
        try:
            start_time = time.time()

            # Create asset payload
            asset_payload = create_collibra_asset_from_span(
                span_attributes, self.domain_id, asset_type
            )

            # Send to Collibra
            result = self.client.create_asset(**asset_payload)

            # Record success
            duration_ms = (time.time() - start_time) * 1000
            self.stats.record_success(count=1, duration_ms=duration_ms)

            logger.debug(
                f"Exported asset to Collibra: {result.get('id')} "
                f"({asset_payload['typeId']}) in {duration_ms:.1f}ms"
            )

            return result

        except CollibraAPIError as e:
            self.stats.record_failure(count=1, error=str(e))
            logger.error(f"Failed to export asset to Collibra: {e}")
            return None
        except Exception as e:
            # Catch all other exceptions
            self.stats.record_failure(count=1, error=str(e))
            logger.error(f"Failed to export asset to Collibra: {e}")
            return None

    def _export_batch(
        self, span_attributes: Dict[str, Any], asset_type: Optional[str] = None
    ):
        """
        Add span to batch buffer for later export.

        Args:
            span_attributes: GenOps span attributes
            asset_type: Override asset type
        """
        # Create asset payload
        asset_payload = create_collibra_asset_from_span(
            span_attributes, self.domain_id, asset_type
        )

        # Add to buffer
        with self.buffer_lock:
            self.buffer.append(asset_payload)

            # Auto-flush if batch size reached
            if len(self.buffer) >= self.batch_size:
                logger.debug(
                    f"Batch size limit reached ({self.batch_size}), flushing buffer"
                )
                self._flush_buffer_locked()

    def _is_critical_event(self, span_attributes: Dict[str, Any]) -> bool:
        """
        Check if span represents a critical event that should be exported immediately.

        Args:
            span_attributes: GenOps span attributes

        Returns:
            True if event is critical
        """
        # Policy violations are critical
        policy_result = span_attributes.get("genops.policy.result")
        if policy_result in ["blocked", "rate_limited"]:
            return True

        # High-cost operations are critical
        cost = span_attributes.get("genops.cost.total", 0)
        if cost > 10.0:  # Threshold: $10
            return True

        # Budget exceeded is critical
        budget_remaining = span_attributes.get("genops.budget.remaining", float("inf"))
        if budget_remaining <= 0:
            return True

        return False

    def flush(self) -> int:
        """
        Flush batch buffer immediately.

        Returns:
            Number of assets exported
        """
        with self.buffer_lock:
            return self._flush_buffer_locked()

    def _flush_buffer_locked(self) -> int:
        """
        Flush batch buffer (assumes lock is held).

        Returns:
            Number of assets exported
        """
        if not self.buffer:
            return 0

        buffer_copy = self.buffer.copy()
        self.buffer.clear()

        logger.info(f"Flushing {len(buffer_copy)} assets to Collibra")

        # Release lock before making API calls
        # (API calls can be slow, don't want to block new spans)

        return self._send_batch(buffer_copy)

    def _send_batch(self, assets: List[Dict[str, Any]]) -> int:
        """
        Send batch of assets to Collibra.

        Args:
            assets: List of asset payloads

        Returns:
            Number of successfully exported assets
        """
        start_time = time.time()
        success_count = 0
        failure_count = 0

        for asset_payload in assets:
            try:
                self.client.create_asset(**asset_payload)
                success_count += 1
            except CollibraAPIError as e:
                failure_count += 1
                logger.error(
                    f"Failed to export asset '{asset_payload.get('name')}': {e}"
                )

        # Record statistics
        duration_ms = (time.time() - start_time) * 1000
        self.stats.record_success(count=success_count, duration_ms=duration_ms)
        self.stats.record_failure(count=failure_count)
        self.stats.record_batch()

        logger.info(
            f"Batch export complete: {success_count} succeeded, "
            f"{failure_count} failed in {duration_ms:.1f}ms"
        )

        return success_count

    def _start_background_flush(self):
        """Start background thread for periodic batch flushing."""
        if self.background_thread is not None:
            logger.warning("Background flush thread already running")
            return

        self.shutdown_event.clear()
        self.background_thread = threading.Thread(
            target=self._background_flush_loop, daemon=True, name="ColliburaExportFlush"
        )
        self.background_thread.start()
        logger.info(
            f"Started background flush thread "
            f"(interval: {self.batch_interval_seconds}s)"
        )

    def _background_flush_loop(self):
        """Background thread loop for periodic flushing."""
        while not self.shutdown_event.is_set():
            # Wait for interval or shutdown signal
            if self.shutdown_event.wait(timeout=self.batch_interval_seconds):
                break  # Shutdown requested

            # Flush buffer
            try:
                with self.buffer_lock:
                    if self.buffer:
                        logger.debug("Background flush triggered")
                        self._flush_buffer_locked()
            except Exception as e:
                logger.error(f"Error in background flush: {e}")

        logger.info("Background flush thread stopped")

    def shutdown(self, timeout: float = 5.0) -> bool:
        """
        Shutdown exporter and flush remaining data.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            True if shutdown completed successfully
        """
        logger.info("Shutting down asset exporter...")

        # Signal background thread to stop
        self.shutdown_event.set()

        # Wait for background thread
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=timeout)

        # Final flush
        remaining = self.flush()
        if remaining > 0:
            logger.info(f"Flushed {remaining} remaining assets during shutdown")

        return True

    def get_stats(self) -> ExportStats:
        """
        Get export statistics.

        Returns:
            Export statistics
        """
        return self.stats

    def get_buffer_size(self) -> int:
        """
        Get current buffer size.

        Returns:
            Number of assets in buffer
        """
        with self.buffer_lock:
            return len(self.buffer)
