"""Core Prometheus exporter for GenOps governance telemetry.

This module exports GenOps governance telemetry as Prometheus metrics using
OpenTelemetry's PrometheusMetricReader. It maintains GenOps's OpenTelemetry-first
architecture while enabling Prometheus scraping.

Architecture:
    AI Application → GenOps Instrumentation → OpenTelemetry Metrics → PrometheusMetricReader → /metrics endpoint
"""

from __future__ import annotations

import logging
import socket
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

from .config import PrometheusConfig
from .metrics import (
    ALL_METRICS,
    MetricType,
    filter_labels,
)

logger = logging.getLogger(__name__)

# Try to import Prometheus dependencies
try:
    from prometheus_client import start_http_server, REGISTRY
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning(
        "Prometheus dependencies not installed. "
        "Install with: pip install genops-ai[prometheus]"
    )


class PrometheusExporter:
    """GenOps Prometheus metrics exporter.

    Exports governance telemetry as Prometheus metrics:
    - Cost metrics (counters, gauges)
    - Token metrics (counters)
    - Policy metrics (counters, gauges)
    - Evaluation metrics (histograms)
    - Budget metrics (gauges, counters)

    Example:
        from genops.exporters.prometheus import PrometheusExporter, PrometheusConfig

        config = PrometheusConfig(port=8000, namespace="genops")
        exporter = PrometheusExporter(config)
        exporter.start()

        # Metrics now available at http://localhost:8000/metrics
    """

    def __init__(
        self,
        config: Optional[PrometheusConfig] = None,
        validate: bool = True
    ):
        """Initialize Prometheus exporter.

        Args:
            config: Prometheus configuration (uses defaults if not provided)
            validate: Validate configuration before starting (default: True)

        Raises:
            ImportError: If Prometheus dependencies not installed
            ValueError: If configuration is invalid
        """
        if not HAS_PROMETHEUS:
            raise ImportError(
                "Prometheus dependencies not installed. "
                "Install with: pip install genops-ai[prometheus]"
            )

        self.config = config or PrometheusConfig()

        # Validate configuration
        if validate:
            is_valid, errors = self.config.validate()
            if not is_valid:
                raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        self._meter_provider: Optional[MeterProvider] = None
        self._meter: Optional[metrics.Meter] = None
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._metrics_cache: Dict[str, Any] = {}

        # Initialize metrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Set up OpenTelemetry metrics with Prometheus reader."""
        try:
            # Create Prometheus metric reader
            reader = PrometheusMetricReader(prefix=self.config.namespace)

            # Create resource with service info
            resource = Resource.create({
                "service.name": "genops-ai",
                "service.namespace": self.config.namespace,
            })

            # Create meter provider
            self._meter_provider = MeterProvider(
                metric_readers=[reader],
                resource=resource
            )

            # Set as global meter provider
            metrics.set_meter_provider(self._meter_provider)

            # Get meter for creating instruments
            self._meter = self._meter_provider.get_meter(
                "genops.exporters.prometheus",
                version="0.1.0"
            )

            # Pre-create metric instruments
            self._create_metric_instruments()

            logger.info(f"Prometheus metrics initialized with namespace: {self.config.namespace}")

        except Exception as e:
            logger.error(f"Failed to set up Prometheus metrics: {e}")
            raise

    def _create_metric_instruments(self) -> None:
        """Pre-create OpenTelemetry metric instruments."""
        for metric_name, metric_def in ALL_METRICS.items():
            try:
                full_name = f"{metric_def.name}"

                if metric_def.metric_type == MetricType.COUNTER:
                    instrument = self._meter.create_counter(
                        name=full_name,
                        description=metric_def.description,
                        unit=metric_def.unit
                    )
                elif metric_def.metric_type == MetricType.GAUGE:
                    instrument = self._meter.create_observable_gauge(
                        name=full_name,
                        description=metric_def.description,
                        unit=metric_def.unit
                    )
                elif metric_def.metric_type == MetricType.HISTOGRAM:
                    instrument = self._meter.create_histogram(
                        name=full_name,
                        description=metric_def.description,
                        unit=metric_def.unit
                    )
                else:
                    logger.warning(f"Unsupported metric type for {metric_name}: {metric_def.metric_type}")
                    continue

                self._metrics_cache[metric_name] = instrument
                logger.debug(f"Created metric instrument: {full_name}")

            except Exception as e:
                logger.warning(f"Failed to create metric {metric_name}: {e}")

    def start(self) -> None:
        """Start the Prometheus metrics HTTP server.

        Raises:
            RuntimeError: If server is already running or port is in use
        """
        if self._is_running:
            logger.warning("Prometheus exporter already running")
            return

        # Check if port is available
        if not self._is_port_available(self.config.port):
            raise RuntimeError(
                f"Port {self.config.port} is already in use. "
                f"Configure a different port or stop the conflicting service."
            )

        try:
            # Start HTTP server for metrics endpoint
            start_http_server(
                port=self.config.port,
                addr='0.0.0.0'
            )

            self._is_running = True
            logger.info(
                f"Prometheus metrics server started at http://localhost:{self.config.port}{self.config.metrics_path}"
            )

        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
            raise

    def stop(self) -> None:
        """Stop the Prometheus metrics server."""
        if not self._is_running:
            logger.warning("Prometheus exporter not running")
            return

        self._is_running = False
        logger.info("Prometheus metrics server stopped")

    def record_cost(
        self,
        cost: float,
        provider: str,
        model: str,
        **labels
    ) -> None:
        """Record cost metric.

        Args:
            cost: Cost in USD
            provider: AI provider (openai, anthropic, etc.)
            model: Model name
            **labels: Additional governance labels (team, customer_id, etc.)
        """
        if "cost_total" not in self._metrics_cache:
            logger.warning("Cost metric not initialized")
            return

        try:
            # Filter labels based on configuration
            filtered_labels = self._filter_labels({
                "provider": provider,
                "model": model,
                **labels
            })

            counter = self._metrics_cache["cost_total"]
            counter.add(cost, attributes=filtered_labels)

        except Exception as e:
            logger.error(f"Failed to record cost metric: {e}")

    def record_tokens(
        self,
        tokens_input: int,
        tokens_output: int,
        provider: str,
        model: str,
        **labels
    ) -> None:
        """Record token metrics.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            provider: AI provider
            model: Model name
            **labels: Additional governance labels
        """
        try:
            filtered_labels = self._filter_labels({
                "provider": provider,
                "model": model,
                **labels
            })

            if "tokens_input_total" in self._metrics_cache:
                self._metrics_cache["tokens_input_total"].add(
                    tokens_input, attributes=filtered_labels
                )

            if "tokens_output_total" in self._metrics_cache:
                self._metrics_cache["tokens_output_total"].add(
                    tokens_output, attributes=filtered_labels
                )

            if "tokens_total" in self._metrics_cache:
                self._metrics_cache["tokens_total"].add(
                    tokens_input + tokens_output, attributes=filtered_labels
                )

        except Exception as e:
            logger.error(f"Failed to record token metrics: {e}")

    def record_operation_latency(
        self,
        latency: float,
        operation_type: str,
        provider: str,
        model: str,
        **labels
    ) -> None:
        """Record operation latency.

        Args:
            latency: Latency in seconds
            operation_type: Type of operation
            provider: AI provider
            model: Model name
            **labels: Additional governance labels
        """
        if "operation_latency_seconds" not in self._metrics_cache:
            logger.warning("Latency metric not initialized")
            return

        try:
            filtered_labels = self._filter_labels({
                "operation_type": operation_type,
                "provider": provider,
                "model": model,
                **labels
            })

            histogram = self._metrics_cache["operation_latency_seconds"]
            histogram.record(latency, attributes=filtered_labels)

        except Exception as e:
            logger.error(f"Failed to record latency metric: {e}")

    def _filter_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Filter labels based on configuration.

        Args:
            labels: Original labels

        Returns:
            Filtered labels
        """
        return filter_labels(
            labels,
            include=self.config.include_labels if self.config.include_labels else None,
            exclude=self.config.exclude_labels if self.config.exclude_labels else None
        )

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available.

        Args:
            port: Port number to check

        Returns:
            True if port is available, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0
        except Exception as e:
            logger.warning(f"Failed to check port availability: {e}")
            return False

    @contextmanager
    def context(self):
        """Context manager for automatic start/stop.

        Example:
            with exporter.context():
                # Metrics server is running
                pass
            # Metrics server is stopped
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop()
        return False
