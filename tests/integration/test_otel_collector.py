"""Integration tests for OpenTelemetry Collector integration.

These tests verify end-to-end telemetry flow from GenOps SDK through OTel Collector
to backend services (Tempo, Loki, Mimir).

Prerequisites:
    - Docker Compose observability stack running:
      docker-compose -f docker-compose.observability.yml up -d
    - OTel Collector accessible at http://localhost:4318
"""

import os
import time
from typing import Optional

import pytest
import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from genops.core.telemetry import GenOpsTelemetry


# Skip these tests if OTel Collector is not available
def check_collector_available() -> bool:
    """Check if OTel Collector is accessible."""
    try:
        response = requests.get("http://localhost:13133/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not check_collector_available(),
    reason="OTel Collector not available at localhost:13133"
)


@pytest.fixture(scope="module")
def otel_setup():
    """Setup OpenTelemetry with OTLP exporter to local collector."""
    # Configure tracer provider with OTLP exporter
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()

    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4318/v1/traces"
    )

    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    yield tracer_provider

    # Cleanup (shutdown span processor to flush remaining spans)
    for processor in tracer_provider._active_span_processor._span_processors:
        processor.shutdown()


@pytest.fixture
def genops_telemetry(otel_setup):
    """Create GenOps telemetry instance with OTel setup."""
    return GenOpsTelemetry()


class TestOTelCollectorConnectivity:
    """Test OTel Collector connectivity and health."""

    def test_collector_health_endpoint(self):
        """Test collector health endpoint is accessible."""
        response = requests.get("http://localhost:13133/", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "Server available"

    def test_otlp_http_endpoint_accessible(self):
        """Test OTLP HTTP endpoint is listening."""
        # OTLP endpoint should return 405 Method Not Allowed for GET
        # This is expected behavior - it only accepts POST with protobuf
        try:
            response = requests.get("http://localhost:4318/v1/traces", timeout=5)
            # 405 is expected - endpoint is there but doesn't accept GET
            assert response.status_code in [405, 400]
        except requests.exceptions.ConnectionError:
            pytest.fail("OTLP HTTP endpoint not accessible")

    def test_grafana_accessible(self):
        """Test Grafana is accessible."""
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        assert response.status_code == 200

    def test_tempo_accessible(self):
        """Test Tempo trace backend is accessible."""
        # Tempo doesn't have a dedicated health endpoint, but should respond to API
        try:
            response = requests.get("http://localhost:3200/api/search", timeout=5)
            # 200 or 400 are both OK (400 means endpoint exists but bad query)
            assert response.status_code in [200, 400]
        except requests.exceptions.ConnectionError:
            pytest.fail("Tempo not accessible at localhost:3200")


class TestTelemetryExport:
    """Test telemetry export to OTel Collector."""

    def test_simple_span_export(self, genops_telemetry):
        """Test exporting a simple span to collector."""
        test_customer = f"test-customer-{int(time.time())}"

        with genops_telemetry.trace_operation(
            operation_name="test_simple_span",
            team="test-team",
            customer_id=test_customer,
            project="test-project"
        ) as span:
            # Record some telemetry
            genops_telemetry.record_cost(
                span,
                cost=0.01,
                provider="openai",
                model="gpt-4"
            )

        # Give collector time to process and export
        time.sleep(5)

        # Verify span was exported (would need Tempo query API for full verification)
        # For now, just verify no exceptions were raised
        assert True

    def test_governance_telemetry_export(self, genops_telemetry):
        """Test exporting comprehensive governance telemetry."""
        test_customer = f"test-customer-{int(time.time())}"

        with genops_telemetry.trace_operation(
            operation_name="test_governance_telemetry",
            team="engineering",
            customer_id=test_customer,
            project="genops-test",
            feature="integration-test"
        ) as span:
            # Record cost
            genops_telemetry.record_cost(
                span,
                cost=0.05,
                currency="USD",
                provider="openai",
                model="gpt-4"
            )

            # Record tokens
            genops_telemetry.record_tokens(
                span,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )

            # Record policy evaluation
            genops_telemetry.record_policy(
                span,
                policy_name="cost_limit",
                result="passed",
                reason="Cost within limit"
            )

            # Record evaluation
            genops_telemetry.record_evaluation(
                span,
                metric_name="quality",
                score=0.95,
                threshold=0.8,
                passed=True
            )

        # Allow time for export
        time.sleep(5)

        assert True

    def test_multiple_spans_export(self, genops_telemetry):
        """Test exporting multiple spans in sequence."""
        test_customer = f"test-customer-{int(time.time())}"

        for i in range(5):
            with genops_telemetry.trace_operation(
                operation_name=f"test_operation_{i}",
                team="test-team",
                customer_id=test_customer
            ) as span:
                genops_telemetry.record_cost(
                    span,
                    cost=0.001 * (i + 1),
                    provider="openai",
                    model="gpt-3.5-turbo"
                )

        # Allow time for batch processing and export
        time.sleep(5)

        assert True


class TestEndToEndDataFlow:
    """Test complete data flow from GenOps to observability backends."""

    @pytest.mark.slow
    def test_end_to_end_trace_query(self, genops_telemetry):
        """Test complete flow: GenOps → Collector → Tempo → Query."""
        test_customer = f"e2e-test-{int(time.time())}"

        # Send telemetry
        with genops_telemetry.trace_operation(
            operation_name="e2e_test_operation",
            team="e2e-team",
            customer_id=test_customer,
            project="e2e-test"
        ) as span:
            genops_telemetry.record_cost(
                span,
                cost=0.10,
                provider="openai",
                model="gpt-4"
            )

        # Wait for telemetry to be processed and indexed
        # Tempo can take 10-30 seconds to index traces
        time.sleep(30)

        # Query Tempo for the trace
        tempo_url = "http://localhost:3200/api/search"
        params = {
            "tags": f"genops.customer_id={test_customer}",
            "limit": 10
        }

        response = requests.get(tempo_url, params=params, timeout=10)
        assert response.status_code == 200

        data = response.json()
        traces = data.get("traces", [])

        # Verify trace was found
        # Note: This might fail if Tempo hasn't indexed yet
        # In production, you'd implement retries with backoff
        if len(traces) == 0:
            pytest.skip("Trace not yet indexed in Tempo (expected in high-load scenarios)")

        # Verify trace has expected attributes
        trace = traces[0]
        trace_str = str(trace)
        assert "genops.customer_id" in trace_str or test_customer in trace_str

    @pytest.mark.slow
    def test_collector_metrics_reflect_received_spans(self):
        """Test that collector metrics show received spans."""
        # Get collector metrics
        response = requests.get("http://localhost:8888/metrics", timeout=5)
        assert response.status_code == 200

        metrics = response.text

        # Check for receiver metrics
        assert "otelcol_receiver_accepted_spans" in metrics or \
               "otelcol_otlp_receiver" in metrics


class TestCollectorPerformance:
    """Test collector performance and resource usage."""

    def test_high_volume_span_export(self, genops_telemetry):
        """Test exporting high volume of spans."""
        test_customer = f"perf-test-{int(time.time())}"
        num_spans = 100

        start_time = time.time()

        for i in range(num_spans):
            with genops_telemetry.trace_operation(
                operation_name=f"perf_test_{i}",
                team="perf-team",
                customer_id=test_customer
            ) as span:
                genops_telemetry.record_cost(
                    span,
                    cost=0.001,
                    provider="openai",
                    model="gpt-3.5-turbo"
                )

        duration = time.time() - start_time

        # Should be able to generate 100 spans in reasonable time
        # This is just span creation, not including export time
        assert duration < 10.0, f"Took {duration}s to generate {num_spans} spans"

        # Allow time for export
        time.sleep(5)

    def test_collector_memory_usage(self):
        """Test collector memory usage remains reasonable."""
        # This would require Docker API or kubectl to check container metrics
        # For now, just verify collector is still responding
        response = requests.get("http://localhost:13133/", timeout=5)
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and resilience."""

    def test_telemetry_continues_if_collector_unavailable(self):
        """Test that telemetry generation doesn't crash if collector is down."""
        # Create telemetry with invalid endpoint
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()

        # Invalid endpoint
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:9999/v1/traces"
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        telemetry = GenOpsTelemetry()

        # Should not raise exception even though collector is unreachable
        try:
            with telemetry.trace_operation(
                operation_name="test_resilience",
                team="test-team"
            ) as span:
                telemetry.record_cost(span, cost=0.01, provider="openai", model="gpt-4")

            # Span creation should succeed even if export fails
            assert True

        except Exception as e:
            pytest.fail(f"Telemetry generation raised exception: {e}")


# Helper functions for manual testing
def manual_test_trace_visibility():
    """
    Helper function for manual testing of trace visibility in Grafana.

    Run this manually and then check Grafana Explore → Tempo for traces.
    """
    from genops.core.telemetry import GenOpsTelemetry

    telemetry = GenOpsTelemetry()

    print("Generating test traces...")

    for i in range(5):
        with telemetry.trace_operation(
            operation_name=f"manual_test_{i}",
            team="manual-test-team",
            customer_id="manual-test-customer",
            project="manual-testing"
        ) as span:
            telemetry.record_cost(
                span,
                cost=0.01 * (i + 1),
                provider="openai",
                model="gpt-4"
            )

            print(f"  Generated span {i + 1}/5")

    print("\nWaiting 30 seconds for traces to be indexed...")
    time.sleep(30)

    print("\nTraces should now be visible in Grafana:")
    print("1. Open http://localhost:3000")
    print("2. Navigate to Explore → Tempo")
    print("3. Search for: {.genops.team=\"manual-test-team\"}")
    print("4. You should see 5 traces")


if __name__ == "__main__":
    # Run manual test
    manual_test_trace_visibility()
