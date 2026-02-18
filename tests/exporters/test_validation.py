"""Tests for genops.exporters.validation module."""

from unittest.mock import Mock, patch

import pytest

from genops.exporters.validation import (
    ValidationResult,
    print_validation_result,
    validate_export_setup,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables."""
    monkeypatch.delenv("HONEYCOMB_API_KEY", raising=False)
    monkeypatch.delenv("HONEYCOMB_DATASET", raising=False)
    monkeypatch.delenv("DD_API_KEY", raising=False)
    monkeypatch.delenv("DD_SITE", raising=False)
    monkeypatch.delenv("DD_SERVICE", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("TEMPO_ENDPOINT", raising=False)
    monkeypatch.delenv("TEMPO_AUTH_HEADER", raising=False)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            provider="test", passed=True, checks=[{"name": "test", "passed": True}]
        )

        assert result.provider == "test"
        assert result.passed is True
        assert len(result.checks) == 1
        assert result.error_message is None

    def test_validation_result_with_error(self):
        """Test ValidationResult with error message."""
        error_msg = "Configuration error"
        result = ValidationResult(
            provider="test", passed=False, checks=[], error_message=error_msg
        )

        assert result.passed is False
        assert result.error_message == error_msg


class TestValidateExportSetup:
    """Test validate_export_setup function."""

    def test_unsupported_provider(self):
        """Test validation with unsupported provider."""
        result = validate_export_setup(provider="unsupported")

        assert result.passed is False
        assert "not implemented" in result.error_message.lower()
        assert result.provider == "unsupported"

    def test_case_insensitive_provider_name(self, monkeypatch):
        """Test that provider names are case-insensitive."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "test_key")

        result1 = validate_export_setup(provider="honeycomb")
        result2 = validate_export_setup(provider="Honeycomb")
        result3 = validate_export_setup(provider="HONEYCOMB")

        assert result1.provider == "honeycomb"
        assert result2.provider == "honeycomb"
        assert result3.provider == "honeycomb"


class TestHoneycombValidation:
    """Test Honeycomb-specific validation."""

    def test_honeycomb_missing_api_key(self, clean_env):
        """Test Honeycomb validation when API key is missing."""
        result = validate_export_setup(provider="honeycomb")

        assert result.passed is False
        assert result.provider == "honeycomb"

        # Check that API key check failed
        api_key_check = next(
            (c for c in result.checks if c["name"] == "HONEYCOMB_API_KEY"), None
        )
        assert api_key_check is not None
        assert api_key_check["passed"] is False
        assert "fix" in api_key_check

    def test_honeycomb_with_api_key(self, monkeypatch):
        """Test Honeycomb validation with API key set."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "test_key")

        with patch("genops.exporters.validation.REQUESTS_AVAILABLE", False):
            result = validate_export_setup(provider="honeycomb")

        # Should pass basic checks even without connectivity test
        api_key_check = next(
            (c for c in result.checks if c["name"] == "HONEYCOMB_API_KEY"), None
        )
        assert api_key_check is not None
        assert api_key_check["passed"] is True

    def test_honeycomb_default_dataset(self, clean_env):
        """Test Honeycomb validation uses default dataset."""
        result = validate_export_setup(provider="honeycomb")

        dataset_check = next(
            (c for c in result.checks if c["name"] == "HONEYCOMB_DATASET"), None
        )
        assert dataset_check is not None
        assert dataset_check["passed"] is True
        assert dataset_check["message"] == "genops-ai"

    def test_honeycomb_custom_dataset(self, monkeypatch):
        """Test Honeycomb validation with custom dataset."""
        custom_dataset = "my-dataset"
        monkeypatch.setenv("HONEYCOMB_DATASET", custom_dataset)

        result = validate_export_setup(provider="honeycomb")

        dataset_check = next(
            (c for c in result.checks if c["name"] == "HONEYCOMB_DATASET"), None
        )
        assert dataset_check is not None
        assert dataset_check["message"] == custom_dataset

    @patch("genops.exporters.validation.REQUESTS_AVAILABLE", True)
    @patch("genops.exporters.validation.requests.get")
    def test_honeycomb_connectivity_success(self, mock_get, monkeypatch):
        """Test Honeycomb connectivity check success."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "valid_key")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = validate_export_setup(provider="honeycomb")

        # Connectivity check should pass
        conn_check = next(
            (c for c in result.checks if c["name"] == "Connectivity"), None
        )
        assert conn_check is not None
        assert conn_check["passed"] is True
        assert result.passed is True

    @patch("genops.exporters.validation.REQUESTS_AVAILABLE", True)
    @patch("genops.exporters.validation.requests.get")
    def test_honeycomb_connectivity_failure(self, mock_get, monkeypatch):
        """Test Honeycomb connectivity check failure."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "invalid_key")
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        result = validate_export_setup(provider="honeycomb")

        # Connectivity check should fail
        conn_check = next(
            (c for c in result.checks if c["name"] == "Connectivity"), None
        )
        assert conn_check is not None
        assert conn_check["passed"] is False
        assert "401" in conn_check["message"]
        assert result.passed is False

    @patch("genops.exporters.validation.REQUESTS_AVAILABLE", True)
    @patch("genops.exporters.validation.requests.get")
    def test_honeycomb_connectivity_network_error(self, mock_get, monkeypatch):
        """Test Honeycomb connectivity with network error."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "test_key")
        mock_get.side_effect = Exception("Network error")

        result = validate_export_setup(provider="honeycomb")

        # Connectivity check should fail with error message
        conn_check = next(
            (c for c in result.checks if c["name"] == "Connectivity"), None
        )
        assert conn_check is not None
        assert conn_check["passed"] is False
        assert "Network error" in conn_check["message"]

    @patch("genops.exporters.validation.REQUESTS_AVAILABLE", False)
    def test_honeycomb_connectivity_skipped_no_requests(self, monkeypatch):
        """Test Honeycomb connectivity check skipped when requests unavailable."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "test_key")

        result = validate_export_setup(provider="honeycomb")

        # Connectivity check should be skipped
        conn_check = next(
            (c for c in result.checks if c["name"] == "Connectivity"), None
        )
        assert conn_check is not None
        assert conn_check["passed"] is True
        assert "Skipped" in conn_check["message"]


class TestDatadogValidation:
    """Test Datadog-specific validation."""

    def test_datadog_missing_api_key(self, clean_env):
        """Test Datadog validation when API key is missing."""
        result = validate_export_setup(provider="datadog")

        assert result.passed is False
        assert result.provider == "datadog"

        api_key_check = next(
            (c for c in result.checks if c["name"] == "DD_API_KEY"), None
        )
        assert api_key_check is not None
        assert api_key_check["passed"] is False

    def test_datadog_with_api_key(self, monkeypatch):
        """Test Datadog validation with API key."""
        monkeypatch.setenv("DD_API_KEY", "test_dd_key")
        monkeypatch.setenv("DD_SERVICE", "test-service")

        result = validate_export_setup(provider="datadog")

        assert result.passed is True
        api_key_check = next(
            (c for c in result.checks if c["name"] == "DD_API_KEY"), None
        )
        assert api_key_check["passed"] is True

    def test_datadog_default_site(self, clean_env):
        """Test Datadog uses default site."""
        result = validate_export_setup(provider="datadog")

        site_check = next((c for c in result.checks if c["name"] == "DD_SITE"), None)
        assert site_check is not None
        assert site_check["message"] == "datadoghq.com"

    def test_datadog_custom_site(self, monkeypatch):
        """Test Datadog with custom site."""
        custom_site = "datadoghq.eu"
        monkeypatch.setenv("DD_SITE", custom_site)

        result = validate_export_setup(provider="datadog")

        site_check = next((c for c in result.checks if c["name"] == "DD_SITE"), None)
        assert site_check["message"] == custom_site

    def test_datadog_missing_service_name(self, clean_env):
        """Test Datadog validation without service name."""
        result = validate_export_setup(provider="datadog")

        service_check = next(
            (c for c in result.checks if c["name"] == "DD_SERVICE"), None
        )
        assert service_check is not None
        assert service_check["passed"] is False

    def test_datadog_service_from_dd_service(self, monkeypatch):
        """Test Datadog service name from DD_SERVICE."""
        monkeypatch.setenv("DD_SERVICE", "my-service")

        result = validate_export_setup(provider="datadog")

        service_check = next(
            (c for c in result.checks if c["name"] == "DD_SERVICE"), None
        )
        assert service_check["passed"] is True
        assert service_check["message"] == "my-service"

    def test_datadog_service_from_otel_service_name(self, monkeypatch):
        """Test Datadog service name from OTEL_SERVICE_NAME."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "otel-service")

        result = validate_export_setup(provider="datadog")

        service_check = next(
            (c for c in result.checks if c["name"] == "DD_SERVICE"), None
        )
        assert service_check["passed"] is True
        assert service_check["message"] == "otel-service"


class TestGrafanaValidation:
    """Test Grafana/Tempo-specific validation."""

    def test_grafana_missing_endpoint(self, clean_env):
        """Test Grafana validation when endpoint is missing."""
        result = validate_export_setup(provider="grafana")

        assert result.passed is False
        assert result.provider == "grafana"

        endpoint_check = next(
            (c for c in result.checks if c["name"] == "TEMPO_ENDPOINT"), None
        )
        assert endpoint_check is not None
        assert endpoint_check["passed"] is False

    def test_grafana_with_endpoint(self, monkeypatch):
        """Test Grafana validation with endpoint."""
        monkeypatch.setenv("TEMPO_ENDPOINT", "http://tempo:4318/v1/traces")

        result = validate_export_setup(provider="grafana")

        assert result.passed is True
        endpoint_check = next(
            (c for c in result.checks if c["name"] == "TEMPO_ENDPOINT"), None
        )
        assert endpoint_check["passed"] is True

    def test_grafana_with_auth_header(self, monkeypatch):
        """Test Grafana validation with auth header."""
        monkeypatch.setenv("TEMPO_ENDPOINT", "http://tempo:4318/v1/traces")
        monkeypatch.setenv("TEMPO_AUTH_HEADER", "Bearer token123")

        result = validate_export_setup(provider="grafana")

        assert result.passed is True
        auth_check = next(
            (c for c in result.checks if c["name"] == "TEMPO_AUTH_HEADER"), None
        )
        assert auth_check is not None
        assert auth_check["passed"] is True


class TestPrintValidationResult:
    """Test print_validation_result function."""

    def test_print_passing_result(self, capsys):
        """Test printing a passing validation result."""
        result = ValidationResult(
            provider="honeycomb",
            passed=True,
            checks=[
                {"name": "API_KEY", "passed": True, "message": "Set"},
                {"name": "Connectivity", "passed": True, "message": "Connected"},
            ],
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "✅" in captured.out
        assert "Honeycomb" in captured.out
        assert "API_KEY" in captured.out
        assert "All checks passed" in captured.out

    def test_print_failing_result(self, capsys):
        """Test printing a failing validation result."""
        result = ValidationResult(
            provider="honeycomb",
            passed=False,
            checks=[
                {
                    "name": "API_KEY",
                    "passed": False,
                    "message": "Not set",
                    "fix": "export HONEYCOMB_API_KEY='your_key'",
                },
            ],
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "❌" in captured.out
        assert "API_KEY" in captured.out
        assert "Not set" in captured.out
        assert "Fix:" in captured.out
        assert "export HONEYCOMB_API_KEY" in captured.out
        assert "Some checks failed" in captured.out

    def test_print_result_with_error_message(self, capsys):
        """Test printing result with error message."""
        error_msg = "Provider not supported"
        result = ValidationResult(
            provider="unknown", passed=False, checks=[], error_message=error_msg
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "❌" in captured.out
        assert error_msg in captured.out

    def test_print_result_formatting(self, capsys):
        """Test that result formatting is user-friendly."""
        result = ValidationResult(
            provider="honeycomb",
            passed=True,
            checks=[
                {"name": "Check1", "passed": True, "message": "OK"},
                {"name": "Check2", "passed": True, "message": "OK"},
            ],
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        # Check indentation and structure
        assert "Configuration:" in captured.out
        assert "  ✅" in captured.out  # Indented checks


class TestValidationIntegration:
    """Integration tests for validation functionality."""

    def test_full_honeycomb_validation_flow(self, monkeypatch):
        """Test complete validation flow for Honeycomb."""
        monkeypatch.setenv("HONEYCOMB_API_KEY", "test_key")
        monkeypatch.setenv("HONEYCOMB_DATASET", "test-dataset")

        with patch("genops.exporters.validation.REQUESTS_AVAILABLE", False):
            result = validate_export_setup(provider="honeycomb")

        assert result.provider == "honeycomb"
        assert len(result.checks) >= 2  # At least API key and dataset

    def test_multiple_provider_validation(self, monkeypatch):
        """Test validating multiple providers."""
        # Set up for multiple providers
        monkeypatch.setenv("HONEYCOMB_API_KEY", "hc_key")
        monkeypatch.setenv("DD_API_KEY", "dd_key")
        monkeypatch.setenv("DD_SERVICE", "test-service")

        with patch("genops.exporters.validation.REQUESTS_AVAILABLE", False):
            hc_result = validate_export_setup(provider="honeycomb")
            dd_result = validate_export_setup(provider="datadog")

        assert hc_result.provider == "honeycomb"
        assert dd_result.provider == "datadog"
