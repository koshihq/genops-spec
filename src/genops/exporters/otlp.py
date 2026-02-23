"""
OpenTelemetry OTLP exporter configuration for GenOps AI.

Provides simplified API for configuring OTLP export to observability backends
like Honeycomb, Datadog, Grafana, etc.
"""

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_otlp_exporter(
    endpoint: str,
    headers: Optional[dict[str, str]] = None,
    service_name: Optional[str] = None,
    environment: Optional[str] = None,
    sampling_rate: float = 1.0,
) -> None:
    """
    Configure OpenTelemetry OTLP exporter for GenOps telemetry.

    This function provides a simplified API for setting up OTLP export to
    observability platforms. It configures the global tracer provider with
    the specified endpoint and authentication headers.

    Args:
        endpoint: OTLP endpoint URL (e.g., "https://api.honeycomb.io/v1/traces")
        headers: HTTP headers for authentication (e.g., {"X-Honeycomb-Team": "api_key"})
        service_name: Service name for telemetry (default: from OTEL_SERVICE_NAME env var)
        environment: Environment name (default: from ENVIRONMENT env var)
        sampling_rate: Sampling rate 0.0-1.0 (default: 1.0 = 100%)

    Example:
        Basic configuration for Honeycomb:

        >>> import os
        >>> from genops.exporters.otlp import configure_otlp_exporter
        >>>
        >>> configure_otlp_exporter(
        ...     endpoint="https://api.honeycomb.io/v1/traces",
        ...     headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}
        ... )

        With custom service name and sampling:

        >>> configure_otlp_exporter(
        ...     endpoint="https://api.honeycomb.io/v1/traces",
        ...     headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")},
        ...     service_name="my-ai-service",
        ...     environment="production",
        ...     sampling_rate=0.1  # 10% sampling
        ... )
    """
    # Create resource with service metadata
    resource_attrs = {
        "service.name": service_name or os.getenv("OTEL_SERVICE_NAME", "genops-ai"),
        "service.version": "1.0.0",
    }

    if environment:
        resource_attrs["deployment.environment"] = environment
    elif os.getenv("ENVIRONMENT"):
        resource_attrs["deployment.environment"] = os.getenv("ENVIRONMENT")  # type: ignore

    resource = Resource.create(resource_attrs)

    # Create OTLP exporter
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or {})

    # Set up tracing with optional sampling
    if sampling_rate < 1.0:
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        sampler = TraceIdRatioBased(sampling_rate)
        trace_provider = TracerProvider(resource=resource, sampler=sampler)
    else:
        trace_provider = TracerProvider(resource=resource)

    # Add batch span processor for efficient export
    trace_provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(trace_provider)
