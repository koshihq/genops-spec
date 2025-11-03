"""OpenTelemetry setup examples for GenOps AI."""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def setup_console_exporter():
    """Set up OpenTelemetry with console output for testing."""
    print("Setting up OpenTelemetry with console exporter...")

    # Create resource with service information
    resource = Resource.create({
        "service.name": "genops-ai-demo",
        "service.version": "0.1.0",
        "deployment.environment": "development"
    })

    # Set up tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Add console exporter for testing
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    tracer_provider.add_span_processor(span_processor)

    print("✓ Console exporter configured")
    return tracer_provider


def setup_otlp_exporter(endpoint: str = "http://localhost:4317"):
    """Set up OpenTelemetry with OTLP exporter for production."""
    print(f"Setting up OpenTelemetry with OTLP exporter to {endpoint}...")

    # Create resource with service information
    resource = Resource.create({
        "service.name": "genops-ai-app",
        "service.version": "0.1.0",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "production")
    })

    # Set up tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "api-key": os.getenv("OTEL_API_KEY", "")
        } if os.getenv("OTEL_API_KEY") else None
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    print("✓ OTLP exporter configured")
    return tracer_provider


def setup_jaeger_exporter():
    """Set up OpenTelemetry with Jaeger exporter."""
    print("Setting up OpenTelemetry with Jaeger exporter...")

    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        # Create resource
        resource = Resource.create({
            "service.name": "genops-ai-app",
            "service.version": "0.1.0"
        })

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
        )
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)

        print("✓ Jaeger exporter configured")
        return tracer_provider

    except ImportError:
        print("Jaeger exporter not available. Install with: pip install opentelemetry-exporter-jaeger")
        return None


def setup_datadog_exporter():
    """Set up OpenTelemetry with Datadog exporter."""
    print("Setting up OpenTelemetry with Datadog exporter...")

    try:
        from opentelemetry.exporter.datadog import DatadogExporter

        # Create resource
        resource = Resource.create({
            "service.name": "genops-ai-app",
            "service.version": "0.1.0"
        })

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Add Datadog exporter
        datadog_exporter = DatadogExporter(
            agent_url=os.getenv("DD_TRACE_AGENT_URL", "http://localhost:8126"),
            service_name="genops-ai-app"
        )
        span_processor = BatchSpanProcessor(datadog_exporter)
        tracer_provider.add_span_processor(span_processor)

        print("✓ Datadog exporter configured")
        return tracer_provider

    except ImportError:
        print("Datadog exporter not available. Install with: pip install opentelemetry-exporter-datadog")
        return None


def demo_with_setup():
    """Demo GenOps AI with OpenTelemetry setup."""
    print("GenOps AI + OpenTelemetry Demo")
    print("=" * 40)

    # Choose exporter based on environment
    exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "console")

    if exporter_type == "console":
        setup_console_exporter()
    elif exporter_type == "otlp":
        setup_otlp_exporter(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
    elif exporter_type == "jaeger":
        setup_jaeger_exporter()
    elif exporter_type == "datadog":
        setup_datadog_exporter()
    else:
        print(f"Unknown exporter type: {exporter_type}, using console")
        setup_console_exporter()

    # Now run GenOps AI examples
    print("\nRunning GenOps AI operations...")

    from genops import track, track_usage
    from genops.core.tracker import track_cost, track_evaluation

    @track_usage(
        operation_name="ai_inference",
        team="demo-team",
        project="otel-demo"
    )
    def demo_ai_operation():
        # Simulate AI operation
        track_cost(
            cost=0.05,
            provider="openai",
            model="gpt-3.5-turbo",
            tokens_input=100,
            tokens_output=50
        )

        track_evaluation(
            evaluation_name="quality_score",
            score=0.85,
            threshold=0.8,
            passed=True
        )

        return "Demo AI result"

    # Execute demo operations
    result1 = demo_ai_operation()
    print(f"Operation 1 result: {result1}")

    with track(
        operation_name="batch_processing",
        team="demo-team",
        project="otel-demo",
        customer="demo-customer"
    ) as span:
        span.set_attribute("batch_size", 5)

        for i in range(5):
            span.set_attribute(f"item_{i}_processed", True)

        track_cost(
            cost=0.25,
            provider="anthropic",
            model="claude-3-sonnet",
            batch_size=5
        )

    print("Batch processing completed")

    print("\n✓ Demo completed!")
    print("Check your configured exporter for telemetry data.")


if __name__ == "__main__":
    demo_with_setup()
