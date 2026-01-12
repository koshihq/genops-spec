"""Unit tests for Collibra validation utilities."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from genops.providers.collibra.validation import (
    CollibraValidationResult,
    print_validation_result,
    validate_setup,
    validate_url_format,
)


# ============================================================================
# URL Validation Tests (3 tests)
# ============================================================================


def test_validate_url_format_valid():
    """Test validation of valid URLs."""
    # Test HTTPS URL
    valid, error = validate_url_format("https://company.collibra.com")
    assert valid is True
    assert error is None

    # Test HTTP URL
    valid, error = validate_url_format("http://localhost:8080")
    assert valid is True
    assert error is None

    # Test URL with path
    valid, error = validate_url_format("https://company.collibra.com/api")
    assert valid is True
    assert error is None


def test_validate_url_format_missing_scheme():
    """Test validation fails for URLs missing scheme."""
    valid, error = validate_url_format("company.collibra.com")
    assert valid is False
    assert "missing scheme" in error.lower()


def test_validate_url_format_invalid_format():
    """Test validation fails for invalid URL formats."""
    # Empty URL
    valid, error = validate_url_format("")
    assert valid is False
    assert "empty" in error.lower()

    # Invalid scheme
    valid, error = validate_url_format("ftp://company.collibra.com")
    assert valid is False
    assert "invalid url scheme" in error.lower()

    # Missing domain
    valid, error = validate_url_format("https://")
    assert valid is False
    assert "missing domain" in error.lower()


# ============================================================================
# Environment Variable Tests (5 tests)
# ============================================================================


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_with_all_env_vars(mock_client_class, monkeypatch):
    """Test validation with all environment variables set."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Mock successful API calls
    mock_client = MagicMock()
    mock_client.health_check.return_value = True
    mock_client.get_application_info.return_value = {"version": "2.0"}
    mock_client.list_domains.return_value = [{"id": "domain-1", "name": "Test Domain"}]
    mock_client.list_policies.return_value = []
    mock_client_class.return_value = mock_client

    result = validate_setup()

    assert result.valid is True
    assert result.connectivity is True
    assert len(result.errors) == 0


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_missing_url(mock_client_class, monkeypatch):
    """Test validation fails when COLLIBRA_URL is missing."""
    # Clear URL environment variable
    monkeypatch.delenv("COLLIBRA_URL", raising=False)
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    result = validate_setup()

    assert result.valid is False
    assert any("COLLIBRA_URL" in error for error in result.errors)


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_missing_auth(mock_client_class, monkeypatch):
    """Test validation fails when authentication credentials are missing."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.delenv("COLLIBRA_USERNAME", raising=False)
    monkeypatch.delenv("COLLIBRA_PASSWORD", raising=False)
    monkeypatch.delenv("COLLIBRA_API_TOKEN", raising=False)

    result = validate_setup()

    assert result.valid is False
    assert any("authentication" in error.lower() for error in result.errors)


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_with_api_token(mock_client_class, monkeypatch):
    """Test validation with API token authentication."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_API_TOKEN", "test-api-token")
    monkeypatch.delenv("COLLIBRA_USERNAME", raising=False)
    monkeypatch.delenv("COLLIBRA_PASSWORD", raising=False)

    # Mock successful API calls
    mock_client = MagicMock()
    mock_client.health_check.return_value = True
    mock_client.get_application_info.return_value = {"version": "2.0"}
    mock_client.list_domains.return_value = [{"id": "domain-1", "name": "Test Domain"}]
    mock_client.list_policies.return_value = []
    mock_client_class.return_value = mock_client

    result = validate_setup()

    assert result.valid is True
    assert result.connectivity is True
    assert len(result.errors) == 0


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_with_basic_auth(mock_client_class, monkeypatch):
    """Test validation with basic authentication."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")
    monkeypatch.delenv("COLLIBRA_API_TOKEN", raising=False)

    # Mock successful API calls
    mock_client = MagicMock()
    mock_client.health_check.return_value = True
    mock_client.get_application_info.return_value = {"version": "2.0"}
    mock_client.list_domains.return_value = [{"id": "domain-1", "name": "Test Domain"}]
    mock_client.list_policies.return_value = []
    mock_client_class.return_value = mock_client

    result = validate_setup()

    assert result.valid is True
    assert result.connectivity is True
    assert len(result.errors) == 0


# ============================================================================
# Authentication Tests (4 tests)
# ============================================================================


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_authentication_success(mock_client_class, monkeypatch):
    """Test successful authentication."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Mock successful authentication
    mock_client = MagicMock()
    mock_client.health_check.return_value = True
    mock_client.get_application_info.return_value = {"version": "2.0"}
    mock_client.list_domains.return_value = [{"id": "domain-1", "name": "Test Domain"}]
    mock_client.list_policies.return_value = []
    mock_client_class.return_value = mock_client

    result = validate_setup()

    assert result.valid is True
    assert result.connectivity is True
    assert "domain-1" in str(result.available_domains[0])


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_authentication_401(mock_client_class, monkeypatch):
    """Test authentication failure with 401 Unauthorized."""
    from genops.providers.collibra.client import CollibraAPIError

    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "wrong_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "wrong_password")

    # Mock 401 authentication error
    mock_client = MagicMock()
    mock_client.health_check.side_effect = CollibraAPIError("Unauthorized", status_code=401)
    mock_client_class.return_value = mock_client

    result = validate_setup()

    assert result.valid is False
    assert result.connectivity is False


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_authentication_403(mock_client_class, monkeypatch):
    """Test authentication failure with 403 Forbidden."""
    from genops.providers.collibra.client import CollibraAPIError

    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Mock 403 permission error
    mock_client = MagicMock()
    mock_client.health_check.side_effect = CollibraAPIError("Forbidden", status_code=403)
    mock_client_class.return_value = mock_client

    result = validate_setup()

    assert result.valid is False
    assert result.connectivity is False


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_invalid_credentials(mock_client_class, monkeypatch):
    """Test with invalid credentials format."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "")

    result = validate_setup()

    assert result.valid is False
    assert any("authentication" in error.lower() for error in result.errors)


# ============================================================================
# Connectivity Tests (3 tests)
# ============================================================================


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_connectivity_success(mock_client_class, monkeypatch):
    """Test successful connectivity check."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Mock successful connectivity
    mock_client = MagicMock()
    mock_client.health_check.return_value = True
    mock_client.get_application_info.return_value = {"version": "2.0"}
    mock_client.list_domains.return_value = [{"id": "domain-1", "name": "Test Domain"}]
    mock_client.list_policies.return_value = [{"id": "policy-1"}]
    mock_client_class.return_value = mock_client

    result = validate_setup(check_connectivity=True)

    assert result.connectivity is True
    assert len(result.available_domains) > 0


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_connectivity_timeout(mock_client_class, monkeypatch):
    """Test connectivity failure with timeout."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Mock timeout error
    mock_client = MagicMock()
    mock_client.health_check.side_effect = TimeoutError("Connection timeout")
    mock_client_class.return_value = mock_client

    result = validate_setup(check_connectivity=True)

    assert result.connectivity is False
    assert result.valid is False


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_connectivity_network_error(mock_client_class, monkeypatch):
    """Test connectivity failure with network error."""
    monkeypatch.setenv("COLLIBRA_URL", "https://unreachable.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Mock network error
    mock_client = MagicMock()
    mock_client.health_check.side_effect = ConnectionError("Network unreachable")
    mock_client_class.return_value = mock_client

    result = validate_setup(check_connectivity=True)

    assert result.connectivity is False
    assert result.valid is False


# ============================================================================
# Error Message Tests (3 tests)
# ============================================================================


def test_validation_result_structure():
    """Test CollibraValidationResult structure and properties."""
    result = CollibraValidationResult(
        valid=False,
        errors=["Error 1", "Error 2"],
        warnings=["Warning 1"],
        recommendations=["Fix error 1", "Check configuration"],
        connectivity=False,
        api_version=None,
        available_domains=[],
        policy_count=0,
    )

    assert result.has_errors is True
    assert result.has_warnings is True
    assert len(result.errors) == 2
    assert len(result.warnings) == 1
    assert len(result.recommendations) == 2
    assert result.valid is False


def test_validation_error_recommendations():
    """Test that validation provides actionable recommendations."""
    result = CollibraValidationResult(
        valid=False,
        errors=["COLLIBRA_URL not set"],
        recommendations=[
            "Set environment variable: export COLLIBRA_URL=https://your-instance.collibra.com"
        ],
    )

    assert len(result.recommendations) > 0
    assert any("export COLLIBRA_URL" in rec for rec in result.recommendations)


def test_validation_warnings_vs_errors():
    """Test distinction between warnings and errors."""
    # Warnings should not prevent validation from passing
    result_with_warnings = CollibraValidationResult(
        valid=True, errors=[], warnings=["No policies found"], connectivity=True
    )

    assert result_with_warnings.valid is True
    assert result_with_warnings.has_warnings is True
    assert result_with_warnings.has_errors is False

    # Errors should prevent validation from passing
    result_with_errors = CollibraValidationResult(
        valid=False, errors=["Authentication failed"], warnings=[], connectivity=False
    )

    assert result_with_errors.valid is False
    assert result_with_errors.has_errors is True
    assert result_with_errors.has_warnings is False


# ============================================================================
# Output Formatting Tests (2 tests)
# ============================================================================


def test_print_validation_result_success(capsys):
    """Test printing successful validation result."""
    result = CollibraValidationResult(
        valid=True,
        connectivity=True,
        api_version="2.0",
        available_domains=["Test Domain 1", "Test Domain 2"],
        policy_count=5,
    )

    print_validation_result(result)
    captured = capsys.readouterr()

    assert "[SUCCESS]" in captured.out
    assert "PASSED" in captured.out
    assert "Connected" in captured.out
    assert "Test Domain 1" in captured.out
    assert "5 policies available" in captured.out


def test_print_validation_result_with_errors(capsys):
    """Test printing validation result with errors."""
    result = CollibraValidationResult(
        valid=False,
        connectivity=False,
        errors=["COLLIBRA_URL not set", "Authentication failed"],
        warnings=["No policies found"],
        recommendations=[
            "Set COLLIBRA_URL environment variable",
            "Check authentication credentials",
        ],
    )

    print_validation_result(result)
    captured = capsys.readouterr()

    assert "[ERROR]" in captured.out
    assert "FAILED" in captured.out
    assert "COLLIBRA_URL not set" in captured.out
    assert "Authentication failed" in captured.out
    assert "[WARNING]" in captured.out
    assert "No policies found" in captured.out
    assert "[INFO]" in captured.out
    assert "Set COLLIBRA_URL environment variable" in captured.out


# ============================================================================
# Additional Edge Cases (2 tests)
# ============================================================================


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_skip_connectivity_check(mock_client_class, monkeypatch):
    """Test validation with connectivity check disabled."""
    monkeypatch.setenv("COLLIBRA_URL", "https://test.collibra.com")
    monkeypatch.setenv("COLLIBRA_USERNAME", "test_user")
    monkeypatch.setenv("COLLIBRA_PASSWORD", "test_password")

    # Don't mock client calls since we're skipping connectivity
    result = validate_setup(check_connectivity=False)

    # Should pass basic validation even without connectivity
    assert result.valid is True or len(result.errors) == 0


@patch("genops.providers.collibra.validation.CollibraAPIClient")
def test_validate_setup_explicit_parameters(mock_client_class):
    """Test validation with explicit parameters instead of env vars."""
    # Mock successful API calls
    mock_client = MagicMock()
    mock_client.health_check.return_value = True
    mock_client.get_application_info.return_value = {"version": "2.0"}
    mock_client.list_domains.return_value = [{"id": "domain-1", "name": "Test Domain"}]
    mock_client.list_policies.return_value = []
    mock_client_class.return_value = mock_client

    result = validate_setup(
        collibra_url="https://explicit.collibra.com",
        username="explicit_user",
        password="explicit_pass",
    )

    assert result.valid is True
    assert result.connectivity is True
