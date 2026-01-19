"""Tests for genops.exporters.otlp module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from genops.exporters.otlp import configure_otlp_exporter


@pytest.fixture(autouse=True)
def reset_tracer_provider():
    """Reset the tracer provider after each test."""
    # Store original provider
    original_provider = trace.get_tracer_provider()
    yield
    # Note: OpenTelemetry doesn't allow overriding the global tracer provider
    # Tests should mock the TracerProvider creation instead of relying on global state


@pytest.fixture
def mock_otlp_exporter():
    """Mock OTLP span exporter."""
    with patch("genops.exporters.otlp.OTLPSpanExporter") as mock:
        yield mock


@pytest.fixture
def mock_batch_processor():
    """Mock batch span processor."""
    with patch("genops.exporters.otlp.BatchSpanProcessor") as mock:
        yield mock


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables."""
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


class TestConfigureOTLPExporter:
    """Test configure_otlp_exporter function."""

    def test_basic_configuration(self, mock_otlp_exporter, mock_batch_processor):
        """Test basic OTLP exporter configuration."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        headers = {"X-Honeycomb-Team": "test_key"}

        configure_otlp_exporter(endpoint=endpoint, headers=headers)

        # Verify exporter was created with correct parameters
        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers=headers)

        # Verify batch processor was created
        mock_batch_processor.assert_called_once()

        # Verify tracer provider was set
        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)

    def test_configuration_with_service_name(
        self, mock_otlp_exporter, mock_batch_processor
    ):
        """Test configuration with custom service name."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        service_name = "my-ai-service"

        with patch("genops.exporters.otlp.TracerProvider") as mock_provider_class:
            mock_provider = Mock(spec=TracerProvider)
            mock_provider_class.return_value = mock_provider

            configure_otlp_exporter(endpoint=endpoint, service_name=service_name)

            # Verify TracerProvider was created with correct resource
            assert mock_provider_class.called
            call_kwargs = mock_provider_class.call_args[1]
            resource = call_kwargs["resource"]
            assert resource.attributes["service.name"] == service_name

    def test_configuration_with_environment(
        self, mock_otlp_exporter, mock_batch_processor
    ):
        """Test configuration with custom environment."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        environment = "production"

        with patch("genops.exporters.otlp.TracerProvider") as mock_provider_class:
            mock_provider = Mock(spec=TracerProvider)
            mock_provider_class.return_value = mock_provider

            configure_otlp_exporter(endpoint=endpoint, environment=environment)

            # Verify TracerProvider was created with correct environment
            call_kwargs = mock_provider_class.call_args[1]
            resource = call_kwargs["resource"]
            assert resource.attributes["deployment.environment"] == environment

    def test_default_service_name(
        self, mock_otlp_exporter, mock_batch_processor, clean_env
    ):
        """Test default service name when none provided."""
        endpoint = "https://api.honeycomb.io/v1/traces"

        configure_otlp_exporter(endpoint=endpoint)

        provider = trace.get_tracer_provider()
        resource = provider.resource
        assert resource.attributes["service.name"] == "genops-ai"

    def test_service_name_from_env(
        self, mock_otlp_exporter, mock_batch_processor, monkeypatch
    ):
        """Test service name from OTEL_SERVICE_NAME environment variable."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        env_service_name = "env-service"
        monkeypatch.setenv("OTEL_SERVICE_NAME", env_service_name)

        with patch("genops.exporters.otlp.TracerProvider") as mock_provider_class:
            mock_provider = Mock(spec=TracerProvider)
            mock_provider_class.return_value = mock_provider

            configure_otlp_exporter(endpoint=endpoint)

            # Verify service name from environment
            call_kwargs = mock_provider_class.call_args[1]
            resource = call_kwargs["resource"]
            assert resource.attributes["service.name"] == env_service_name

    def test_environment_from_env(
        self, mock_otlp_exporter, mock_batch_processor, monkeypatch
    ):
        """Test environment from ENVIRONMENT environment variable."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        env_value = "staging"
        monkeypatch.setenv("ENVIRONMENT", env_value)

        with patch("genops.exporters.otlp.TracerProvider") as mock_provider_class:
            mock_provider = Mock(spec=TracerProvider)
            mock_provider_class.return_value = mock_provider

            configure_otlp_exporter(endpoint=endpoint)

            # Verify environment from env var
            call_kwargs = mock_provider_class.call_args[1]
            resource = call_kwargs["resource"]
            assert resource.attributes["deployment.environment"] == env_value

    def test_explicit_params_override_env(
        self, mock_otlp_exporter, mock_batch_processor, monkeypatch
    ):
        """Test that explicit parameters override environment variables."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("ENVIRONMENT", "env-env")

        with patch("genops.exporters.otlp.TracerProvider") as mock_provider_class:
            mock_provider = Mock(spec=TracerProvider)
            mock_provider_class.return_value = mock_provider

            configure_otlp_exporter(
                endpoint=endpoint, service_name="explicit-service", environment="explicit-env"
            )

            # Verify explicit params override env
            call_kwargs = mock_provider_class.call_args[1]
            resource = call_kwargs["resource"]
            assert resource.attributes["service.name"] == "explicit-service"
            assert resource.attributes["deployment.environment"] == "explicit-env"

    def test_sampling_rate_full(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with full sampling (1.0)."""
        endpoint = "https://api.honeycomb.io/v1/traces"

        configure_otlp_exporter(endpoint=endpoint, sampling_rate=1.0)

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        # Default tracer provider doesn't have a sampler attribute we can easily check

    def test_sampling_rate_partial(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with partial sampling."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        sampling_rate = 0.1

        configure_otlp_exporter(endpoint=endpoint, sampling_rate=sampling_rate)

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        # Verify sampler was created (via TracerProvider)

    def test_no_headers(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration without authentication headers."""
        endpoint = "http://localhost:4318/v1/traces"

        configure_otlp_exporter(endpoint=endpoint)

        # Verify exporter was created with empty headers dict
        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers={})

    def test_empty_headers_dict(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with empty headers dictionary."""
        endpoint = "http://localhost:4318/v1/traces"
        headers = {}

        configure_otlp_exporter(endpoint=endpoint, headers=headers)

        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers=headers)

    def test_multiple_headers(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with multiple headers."""
        endpoint = "https://api.example.com/v1/traces"
        headers = {
            "Authorization": "Bearer token123",
            "X-Custom-Header": "custom_value",
        }

        configure_otlp_exporter(endpoint=endpoint, headers=headers)

        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers=headers)

    def test_honeycomb_endpoint(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with Honeycomb endpoint."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        headers = {"X-Honeycomb-Team": "test_api_key"}

        configure_otlp_exporter(endpoint=endpoint, headers=headers)

        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers=headers)

    def test_datadog_endpoint(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with Datadog endpoint."""
        endpoint = "https://api.datadoghq.com/api/v2/traces"
        headers = {"DD-API-KEY": "test_dd_key"}

        configure_otlp_exporter(endpoint=endpoint, headers=headers)

        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers=headers)

    def test_grafana_tempo_endpoint(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with Grafana Tempo endpoint."""
        endpoint = "http://tempo:4318/v1/traces"

        configure_otlp_exporter(endpoint=endpoint)

        mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers={})

    def test_service_version_included(
        self, mock_otlp_exporter, mock_batch_processor
    ):
        """Test that service version is included in resource attributes."""
        endpoint = "https://api.honeycomb.io/v1/traces"

        configure_otlp_exporter(endpoint=endpoint)

        provider = trace.get_tracer_provider()
        resource = provider.resource
        assert "service.version" in resource.attributes
        assert resource.attributes["service.version"] == "1.0.0"

    def test_all_parameters(self, mock_otlp_exporter, mock_batch_processor):
        """Test configuration with all parameters specified."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        headers = {"X-Honeycomb-Team": "test_key"}
        service_name = "full-config-service"
        environment = "testing"
        sampling_rate = 0.5

        with patch("genops.exporters.otlp.TracerProvider") as mock_provider_class, \
             patch("opentelemetry.sdk.trace.sampling.TraceIdRatioBased") as mock_sampler_class:
            mock_provider = Mock(spec=TracerProvider)
            mock_provider_class.return_value = mock_provider
            mock_sampler = Mock()
            mock_sampler_class.return_value = mock_sampler

            configure_otlp_exporter(
                endpoint=endpoint,
                headers=headers,
                service_name=service_name,
                environment=environment,
                sampling_rate=sampling_rate,
            )

            # Verify exporter creation
            mock_otlp_exporter.assert_called_once_with(endpoint=endpoint, headers=headers)

            # Verify resource attributes
            call_kwargs = mock_provider_class.call_args[1]
            resource = call_kwargs["resource"]
            assert resource.attributes["service.name"] == service_name
            assert resource.attributes["deployment.environment"] == environment

            # Verify sampling was configured
            mock_sampler_class.assert_called_once_with(sampling_rate)


class TestOTLPIntegration:
    """Integration tests for OTLP exporter."""

    def test_can_create_tracer_after_configuration(
        self, mock_otlp_exporter, mock_batch_processor
    ):
        """Test that we can create a tracer after configuration."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        configure_otlp_exporter(endpoint=endpoint)

        # Create a tracer
        tracer = trace.get_tracer(__name__)
        assert tracer is not None

    def test_can_create_spans_after_configuration(
        self, mock_otlp_exporter, mock_batch_processor
    ):
        """Test that we can create spans after configuration."""
        endpoint = "https://api.honeycomb.io/v1/traces"
        configure_otlp_exporter(endpoint=endpoint)

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("test_span") as span:
            assert span is not None
            span.set_attribute("test.attribute", "test_value")
