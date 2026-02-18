"""
GenOps Prometheus Exporter - Governance Metrics for Prometheus

Exports GenOps governance telemetry as Prometheus metrics using OpenTelemetry:
- Cost metrics across all AI providers
- Token usage and efficiency metrics
- Policy compliance and violations
- Evaluation scores and quality metrics
- Budget tracking and utilization

Quick Start:
    from genops.exporters.prometheus import instrument_prometheus

    # Zero-code auto-instrumentation
    exporter = instrument_prometheus()

    # Now use any GenOps-instrumented AI provider
    from genops.providers.openai import instrument_openai
    client = instrument_openai()

    # Metrics available at http://localhost:8000/metrics

For detailed documentation, see: docs/prometheus-quickstart.md
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Core components
from .config import PrometheusConfig  # noqa: E402
from .exporter import PrometheusExporter  # noqa: E402
from .metrics import (  # noqa: E402
    ALL_METRICS,
    MetricDefinition,
    MetricType,
    get_full_metric_name,
    get_metric_definition,
)

# Validation utilities
from .validation import (  # noqa: E402
    PrometheusValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationLevel,
    ValidationResult,
    print_validation_result,
    validate_setup,
)

# Version info
__version__ = "0.1.0"

# Global exporter instance for auto-instrumentation
_global_exporter: PrometheusExporter | None = None


def instrument_prometheus(
    port: int = None,  # type: ignore[assignment]
    namespace: str = None,  # type: ignore[assignment]
    prometheus_url: str = None,  # type: ignore[assignment]
    validate: bool = True,
    auto_start: bool = True,
    **config_kwargs,
) -> PrometheusExporter:
    """Instrument Prometheus metrics exporter for GenOps governance telemetry.

    This is the main entry point for manual Prometheus integration. It creates
    and configures a PrometheusExporter instance with the specified settings.

    Args:
        port: Port for metrics endpoint (default: 8000)
        namespace: Metrics namespace prefix (default: genops)
        prometheus_url: Prometheus server URL for validation (default: http://localhost:9090)
        validate: Run setup validation before starting (default: True)
        auto_start: Automatically start the metrics server (default: True)
        **config_kwargs: Additional PrometheusConfig parameters

    Returns:
        PrometheusExporter instance

    Example:
        from genops.exporters.prometheus import instrument_prometheus

        # Basic usage
        exporter = instrument_prometheus()

        # Custom configuration
        exporter = instrument_prometheus(
            port=8001,
            namespace="myapp",
            max_label_cardinality=5000
        )

        # Manual start/stop control
        exporter = instrument_prometheus(auto_start=False)
        exporter.start()
        # ... use exporter
        exporter.stop()

    Raises:
        ImportError: If Prometheus dependencies not installed
        ValueError: If configuration is invalid
    """
    # Build configuration
    config_dict = {
        "port": port or int(os.getenv("PROMETHEUS_EXPORTER_PORT", "8000")),
        "namespace": namespace or os.getenv("PROMETHEUS_NAMESPACE", "genops"),
        "prometheus_url": prometheus_url
        or os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
    }
    config_dict.update(config_kwargs)

    config = PrometheusConfig(**config_dict)  # type: ignore

    # Run validation if requested
    if validate:
        from .validation import validate_setup as run_validation

        result = run_validation(
            port=config.port,
            prometheus_url=config.prometheus_url,
            namespace=config.namespace,
        )

        if result.has_critical_issues:
            logger.error("Critical validation issues detected")
            print_validation_result(result)
            raise ValueError(
                "Prometheus exporter validation failed with critical issues"
            )

        if result.has_errors:
            logger.warning("Validation errors detected")
            print_validation_result(result)

        if not result.success:
            logger.warning(
                f"Validation completed with warnings (score: {result.score:.1f}%)"
            )

    # Create exporter
    exporter = PrometheusExporter(config, validate=False)  # Already validated above

    # Auto-start if requested
    if auto_start:
        exporter.start()
        logger.info(
            f"Prometheus exporter started at http://localhost:{config.port}/metrics"
        )

    return exporter


def auto_instrument() -> PrometheusExporter:
    """Zero-code auto-instrumentation for Prometheus metrics export.

    Reads configuration from environment variables and starts the exporter
    automatically. This is the simplest way to enable Prometheus metrics.

    Environment Variables:
        PROMETHEUS_EXPORTER_PORT: Port for metrics endpoint (default: 8000)
        PROMETHEUS_NAMESPACE: Metrics namespace (default: genops)
        PROMETHEUS_URL: Prometheus server URL (default: http://localhost:9090)
        PROMETHEUS_MAX_CARDINALITY: Max label cardinality (default: 10000)
        PROMETHEUS_SAMPLING_RATE: Sampling rate 0-1 (default: 1.0)

    Returns:
        PrometheusExporter instance (already started)

    Example:
        from genops.exporters.prometheus import auto_instrument

        # Zero-code setup
        auto_instrument()

        # Now use any GenOps provider - metrics are automatically exported
        from genops.providers.openai import instrument_openai
        client = instrument_openai()

    Raises:
        ImportError: If Prometheus dependencies not installed
        ValueError: If configuration is invalid
    """
    global _global_exporter

    if _global_exporter is not None:
        logger.warning("Prometheus exporter already initialized via auto_instrument()")
        return _global_exporter

    # Load configuration from environment
    config = PrometheusConfig.from_env()

    # Create and start exporter
    _global_exporter = PrometheusExporter(config, validate=True)
    _global_exporter.start()

    logger.info(
        f"Prometheus auto-instrumentation enabled at http://localhost:{config.port}/metrics"
    )

    return _global_exporter


def get_exporter() -> PrometheusExporter | None:
    """Get the global auto-instrumented exporter instance.

    Returns:
        PrometheusExporter instance if auto_instrument() was called, None otherwise

    Example:
        from genops.exporters.prometheus import auto_instrument, get_exporter

        auto_instrument()

        # Later, get reference to exporter
        exporter = get_exporter()
        if exporter:
            print(f"Metrics at http://localhost:{exporter.config.port}/metrics")
    """
    return _global_exporter


def disable_auto_instrument() -> None:
    """Disable and stop the auto-instrumented exporter.

    Example:
        from genops.exporters.prometheus import auto_instrument, disable_auto_instrument

        auto_instrument()
        # ... use metrics

        # Clean up
        disable_auto_instrument()
    """
    global _global_exporter

    if _global_exporter is not None:
        _global_exporter.stop()
        _global_exporter = None
        logger.info("Prometheus auto-instrumentation disabled")
    else:
        logger.warning("No auto-instrumented exporter to disable")


# Export public API
__all__ = [
    # Main API
    "instrument_prometheus",
    "auto_instrument",
    "get_exporter",
    "disable_auto_instrument",
    # Configuration
    "PrometheusConfig",
    # Exporter
    "PrometheusExporter",
    # Metrics
    "MetricDefinition",
    "MetricType",
    "ALL_METRICS",
    "get_metric_definition",
    "get_full_metric_name",
    # Validation
    "validate_setup",
    "print_validation_result",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "ValidationCategory",
    "PrometheusValidator",
]
