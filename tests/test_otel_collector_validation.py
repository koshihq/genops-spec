"""Unit tests for OpenTelemetry Collector validation utilities."""

import socket
from unittest.mock import Mock, patch

import pytest

# Import validation functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'observability'))

from otel_collector_validation import (
    OTelCollectorValidationResult,
    check_port_open,
    validate_url_format,
    validate_setup,
)


class TestPortChecking:
    """Test port availability checking."""

    def test_check_port_open_with_open_port(self):
        """Test checking an open port."""
        # Create a temporary socket to bind to a port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))  # Bind to any available port
        sock.listen(1)
        port = sock.getsockname()[1]

        try:
            result = check_port_open('localhost', port, timeout=1.0)
            assert result is True
        finally:
            sock.close()

    def test_check_port_open_with_closed_port(self):
        """Test checking a closed port."""
        # Use a port that's very unlikely to be in use
        result = check_port_open('localhost', 59999, timeout=0.5)
        assert result is False

    def test_check_port_open_with_invalid_host(self):
        """Test checking port on invalid host."""
        result = check_port_open('invalid.host.example.com', 80, timeout=0.5)
        assert result is False


class TestURLValidation:
    """Test URL format validation."""

    def test_validate_url_format_valid_http(self):
        """Test validation of valid HTTP URL."""
        valid, error = validate_url_format("http://localhost:4318")
        assert valid is True
        assert error is None

    def test_validate_url_format_valid_https(self):
        """Test validation of valid HTTPS URL."""
        valid, error = validate_url_format("https://collector.example.com:4318")
        assert valid is True
        assert error is None

    def test_validate_url_format_empty_url(self):
        """Test validation of empty URL."""
        valid, error = validate_url_format("")
        assert valid is False
        assert "empty" in error.lower()

    def test_validate_url_format_missing_scheme(self):
        """Test validation of URL without scheme."""
        valid, error = validate_url_format("localhost:4318")
        assert valid is False
        assert "scheme" in error.lower()

    def test_validate_url_format_invalid_scheme(self):
        """Test validation of URL with invalid scheme."""
        valid, error = validate_url_format("ftp://localhost:4318")
        assert valid is False
        assert "scheme" in error.lower()

    def test_validate_url_format_missing_domain(self):
        """Test validation of URL without domain."""
        valid, error = validate_url_format("http://")
        assert valid is False
        assert "domain" in error.lower()


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = OTelCollectorValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_validation_result_has_errors(self):
        """Test has_errors property."""
        result = OTelCollectorValidationResult(
            valid=False,
            errors=["Error 1", "Error 2"]
        )
        assert result.has_errors is True

        result_no_errors = OTelCollectorValidationResult(valid=True)
        assert result_no_errors.has_errors is False

    def test_validation_result_has_warnings(self):
        """Test has_warnings property."""
        result = OTelCollectorValidationResult(
            valid=True,
            warnings=["Warning 1"]
        )
        assert result.has_warnings is True

        result_no_warnings = OTelCollectorValidationResult(valid=True)
        assert result_no_warnings.has_warnings is False


class TestValidateSetup:
    """Test the main validate_setup function."""

    @patch('otel_collector_validation.HAS_REQUESTS', False)
    def test_validate_setup_without_requests_library(self):
        """Test validation when requests library is not available."""
        result = validate_setup(check_connectivity=True)

        assert result.valid is False
        assert any("requests library not installed" in err for err in result.errors)
        assert any("pip install requests" in rec for rec in result.recommendations)

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    def test_validate_setup_with_invalid_endpoint_url(self):
        """Test validation with invalid endpoint URL."""
        result = validate_setup(
            collector_endpoint="invalid-url",
            check_connectivity=False
        )

        assert result.valid is False
        assert any("Invalid collector endpoint URL" in err for err in result.errors)

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_collector_healthy(self, mock_get):
        """Test validation when collector is healthy."""
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Server available"}
        mock_get.return_value = mock_response

        # Mock port checks
        with patch('otel_collector_validation.check_port_open', return_value=True):
            result = validate_setup(
                collector_endpoint="http://localhost:4318",
                check_connectivity=True,
                check_backends=False
            )

        assert result.collector_healthy is True
        assert result.otlp_http_accessible is True
        assert result.valid is True

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_collector_connection_refused(self, mock_get):
        """Test validation when collector connection is refused."""
        # Mock connection refused
        mock_get.side_effect = Exception("Connection refused")

        with patch('otel_collector_validation.check_port_open', return_value=False):
            result = validate_setup(
                collector_endpoint="http://localhost:4318",
                check_connectivity=True,
                check_backends=False
            )

        assert result.valid is False
        assert result.collector_healthy is False
        assert any("not accessible" in err for err in result.errors)
        assert any("docker-compose" in rec.lower() for rec in result.recommendations)

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_collector_timeout(self, mock_get):
        """Test validation when collector health check times out."""
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout()

        result = validate_setup(
            collector_endpoint="http://localhost:4318",
            check_connectivity=True,
            check_backends=False
        )

        assert result.valid is False
        assert any("timeout" in err.lower() for err in result.errors)

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_with_backends(self, mock_get):
        """Test validation including backend services."""
        # Mock collector health check
        collector_response = Mock()
        collector_response.status_code = 200
        collector_response.json.return_value = {"status": "Server available"}

        # Mock Grafana health check
        grafana_response = Mock()
        grafana_response.status_code = 200

        mock_get.side_effect = [collector_response, grafana_response]

        with patch('otel_collector_validation.check_port_open', return_value=True):
            result = validate_setup(
                collector_endpoint="http://localhost:4318",
                grafana_endpoint="http://localhost:3000",
                check_connectivity=True,
                check_backends=True
            )

        assert result.collector_healthy is True
        assert result.grafana_accessible is True
        assert result.tempo_accessible is True
        assert result.loki_accessible is True
        assert result.mimir_accessible is True

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    def test_validate_setup_without_connectivity_check(self):
        """Test validation with connectivity check disabled."""
        result = validate_setup(
            collector_endpoint="http://localhost:4318",
            check_connectivity=False
        )

        # Should pass validation if URL format is valid
        assert result.valid is True
        assert result.collector_healthy is False  # Not checked

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_recommendations_on_success(self, mock_get):
        """Test that recommendations are provided on successful validation."""
        # Mock successful setup
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Server available"}
        mock_get.return_value = mock_response

        with patch('otel_collector_validation.check_port_open', return_value=True):
            result = validate_setup(
                collector_endpoint="http://localhost:4318",
                check_connectivity=True,
                check_backends=False
            )

        assert result.valid is True
        assert len(result.recommendations) > 0
        assert any("successfully" in rec.lower() for rec in result.recommendations)

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_backend_warnings(self, mock_get):
        """Test that backend service warnings don't fail validation."""
        # Mock collector healthy but backends unavailable
        collector_response = Mock()
        collector_response.status_code = 200
        collector_response.json.return_value = {"status": "Server available"}

        # Grafana not accessible
        grafana_response = Mock()
        grafana_response.side_effect = Exception("Connection refused")

        mock_get.side_effect = [collector_response, grafana_response]

        with patch('otel_collector_validation.check_port_open') as mock_port:
            # Collector ports open, backend ports closed
            def port_check_side_effect(host, port):
                if port in [4318, 4317]:
                    return True
                return False

            mock_port.side_effect = port_check_side_effect

            result = validate_setup(
                collector_endpoint="http://localhost:4318",
                check_connectivity=True,
                check_backends=True
            )

        # Should still be valid (backends are optional)
        assert result.valid is True
        assert result.collector_healthy is True
        assert result.grafana_accessible is False
        assert len(result.warnings) > 0


class TestEnvironmentVariables:
    """Test environment variable handling."""

    @patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://env-collector:4318'})
    @patch('otel_collector_validation.HAS_REQUESTS', True)
    def test_validate_setup_uses_env_var(self):
        """Test that validation uses OTEL_EXPORTER_OTLP_ENDPOINT env var."""
        result = validate_setup(
            check_connectivity=False
        )

        # Should use environment variable
        assert result.valid is True

    @patch.dict(os.environ, {}, clear=True)
    @patch('otel_collector_validation.HAS_REQUESTS', True)
    def test_validate_setup_defaults_to_localhost(self):
        """Test that validation defaults to localhost when no env var."""
        result = validate_setup(
            check_connectivity=False
        )

        # Should use default localhost:4318
        assert result.valid is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_with_none_endpoint(self, mock_get):
        """Test validation with None as endpoint."""
        # Should use default endpoint
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Server available"}
        mock_get.return_value = mock_response

        with patch('otel_collector_validation.check_port_open', return_value=True):
            result = validate_setup(
                collector_endpoint=None,  # Should use default
                check_connectivity=True,
                check_backends=False
            )

        assert result.valid is True

    @patch('otel_collector_validation.HAS_REQUESTS', True)
    @patch('otel_collector_validation.requests.get')
    def test_validate_setup_with_non_standard_port(self, mock_get):
        """Test validation with non-standard port."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Server available"}
        mock_get.return_value = mock_response

        with patch('otel_collector_validation.check_port_open', return_value=True):
            result = validate_setup(
                collector_endpoint="http://localhost:9999",
                check_connectivity=True,
                check_backends=False
            )

        # Should work with non-standard port
        assert result.collector_healthy is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
