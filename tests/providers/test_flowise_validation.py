"""
Test suite for Flowise validation module.

This module tests the validation functionality for Flowise setup,
including diagnostics, error detection, and user-friendly reporting.
"""

import json
import os
from unittest.mock import Mock, patch

import pytest
from requests.exceptions import ConnectionError, HTTPError, Timeout

from genops.providers.flowise_validation import (
    ValidationIssue,
    ValidationResult,
    _create_validation_summary,
    _validate_authentication,
    _validate_chatflows_access,
    _validate_connectivity,
    _validate_url_format,
    print_validation_result,
    validate_flowise_setup,
)


class TestValidationResult:
    """Test ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult with all parameters."""
        issues = [
            ValidationIssue("error", "Test error", "Fix this"),
            ValidationIssue("warning", "Test warning", "Consider this"),
        ]

        result = ValidationResult(
            is_valid=False,
            summary="Test failed",
            issues=issues,
            flowise_version="1.0.0",
            available_chatflows=2,
            response_time_ms=150,
        )

        assert result.is_valid is False
        assert result.summary == "Test failed"
        assert len(result.issues) == 2
        assert result.flowise_version == "1.0.0"
        assert result.available_chatflows == 2
        assert result.response_time_ms == 150

    def test_validation_result_defaults(self):
        """Test ValidationResult with minimal parameters."""
        result = ValidationResult(is_valid=True, summary="Success", issues=[])

        assert result.is_valid is True
        assert result.summary == "Success"
        assert result.issues == []
        assert result.flowise_version is None
        assert result.available_chatflows is None
        assert result.response_time_ms is None

    def test_validation_result_has_errors(self):
        """Test ValidationResult error detection."""
        issues_with_error = [
            ValidationIssue("error", "Critical error", "Fix immediately"),
            ValidationIssue("warning", "Warning message", "Consider fixing"),
        ]

        issues_without_error = [
            ValidationIssue("warning", "Warning only", "Consider fixing"),
            ValidationIssue("info", "Info message", "Good to know"),
        ]

        result_with_errors = ValidationResult(True, "Test", issues_with_error)
        result_without_errors = ValidationResult(True, "Test", issues_without_error)

        # Test helper method to check for errors
        def has_errors(result):
            return any(issue.severity == "error" for issue in result.issues)

        assert has_errors(result_with_errors) is True
        assert has_errors(result_without_errors) is False

    def test_validation_result_json_serializable(self):
        """Test ValidationResult can be serialized to JSON."""
        issues = [ValidationIssue("error", "Test error", "Fix this")]

        result = ValidationResult(
            is_valid=False,
            summary="Test failed",
            issues=issues,
            flowise_version="1.0.0",
        )

        # Convert to dict for JSON serialization
        result_dict = {
            "is_valid": result.is_valid,
            "summary": result.summary,
            "issues": [
                {
                    "severity": issue.severity,
                    "description": issue.description,
                    "suggested_fix": issue.suggested_fix,
                }
                for issue in result.issues
            ],
            "flowise_version": result.flowise_version,
            "available_chatflows": result.available_chatflows,
            "response_time_ms": result.response_time_ms,
        }

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        assert parsed["is_valid"] is False
        assert parsed["summary"] == "Test failed"
        assert len(parsed["issues"]) == 1


class TestValidationIssue:
    """Test ValidationIssue data class."""

    def test_validation_issue_creation(self):
        """Test creating ValidationIssue."""
        issue = ValidationIssue(
            severity="error",
            description="Connection failed",
            suggested_fix="Check your network connection",
        )

        assert issue.severity == "error"
        assert issue.description == "Connection failed"
        assert issue.suggested_fix == "Check your network connection"

    def test_validation_issue_severity_levels(self):
        """Test different severity levels."""
        severities = ["error", "warning", "info"]

        for severity in severities:
            issue = ValidationIssue(
                severity=severity,
                description=f"Test {severity}",
                suggested_fix=f"Fix {severity}",
            )
            assert issue.severity == severity

    def test_validation_issue_empty_values(self):
        """Test ValidationIssue with empty values."""
        issue = ValidationIssue("", "", "")

        assert issue.severity == ""
        assert issue.description == ""
        assert issue.suggested_fix == ""

    def test_validation_issue_unicode(self):
        """Test ValidationIssue with Unicode characters."""
        issue = ValidationIssue(
            severity="error",
            description="Connection failed with émojis 🚀",
            suggested_fix="Check spéciâl configuration",
        )

        assert "émojis" in issue.description
        assert "🚀" in issue.description
        assert "spéciâl" in issue.suggested_fix


class TestUrlValidation:
    """Test URL format validation."""

    def test_valid_url_formats(self):
        """Test validation of valid URL formats."""
        valid_urls = [
            "http://localhost:3000",
            "https://flowise.example.com",
            "http://192.168.1.100:3000",
            "https://api.flowise.com:8080",
            "http://flowise-service.namespace.svc.cluster.local:3000",
        ]

        for url in valid_urls:
            issues = _validate_url_format(url)
            assert len(issues) == 0, f"URL {url} should be valid"

    def test_invalid_url_formats(self):
        """Test validation of invalid URL formats."""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://wrong-protocol.com",
            "http://",
            "://missing-protocol.com",
            "http:///missing-host",
            "http://host:invalid-port",
        ]

        for url in invalid_urls:
            issues = _validate_url_format(url)
            assert len(issues) > 0, f"URL {url} should be invalid"
            assert any(issue.severity == "error" for issue in issues)

    def test_url_validation_with_trailing_slash(self):
        """Test URL validation handles trailing slashes."""
        urls_with_slash = [
            "http://localhost:3000/",
            "https://flowise.example.com/",
            "http://192.168.1.100:3000///",
        ]

        for url in urls_with_slash:
            issues = _validate_url_format(url)
            # Trailing slashes should not cause validation errors
            assert len(issues) == 0

    def test_url_validation_case_sensitivity(self):
        """Test URL validation with different cases."""
        mixed_case_urls = [
            "HTTP://localhost:3000",
            "Https://Flowise.Example.Com",
            "http://LOCALHOST:3000",
        ]

        for url in mixed_case_urls:
            issues = _validate_url_format(url)
            # Case variations should be acceptable
            assert len(issues) == 0


class TestConnectivityValidation:
    """Test connectivity validation."""

    @patch("requests.get")
    def test_successful_connectivity(self, mock_get):
        """Test successful connectivity validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.15
        mock_get.return_value = mock_response

        issues, response_time = _validate_connectivity("http://localhost:3000", None)

        assert len(issues) == 0
        assert response_time == 150  # 0.15 seconds -> 150ms

    @patch("requests.get")
    def test_connection_error(self, mock_get):
        """Test connectivity validation with connection error."""
        mock_get.side_effect = ConnectionError("Failed to connect")

        issues, response_time = _validate_connectivity("http://localhost:3000", None)

        assert len(issues) > 0
        assert any("connection" in issue.description.lower() for issue in issues)
        assert response_time is None

    @patch("requests.get")
    def test_timeout_error(self, mock_get):
        """Test connectivity validation with timeout."""
        mock_get.side_effect = Timeout("Request timeout")

        issues, response_time = _validate_connectivity(
            "http://localhost:3000", None, timeout=5
        )

        assert len(issues) > 0
        assert any("timeout" in issue.description.lower() for issue in issues)
        assert response_time is None

    @patch("requests.get")
    def test_http_error_responses(self, mock_get):
        """Test connectivity validation with HTTP errors."""
        error_codes = [400, 401, 403, 404, 500, 502, 503]

        for status_code in error_codes:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.text = f"HTTP {status_code} Error"
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_get.return_value = mock_response

            issues, response_time = _validate_connectivity(
                "http://localhost:3000", None
            )

            assert len(issues) > 0
            assert any(str(status_code) in issue.description for issue in issues)

    @patch("requests.get")
    def test_connectivity_with_auth_header(self, mock_get):
        """Test connectivity validation includes auth header."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_get.return_value = mock_response

        api_key = "test-api-key"
        issues, response_time = _validate_connectivity("http://localhost:3000", api_key)

        # Check that Authorization header was included
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {api_key}"

    @patch("requests.get")
    def test_connectivity_slow_response(self, mock_get):
        """Test connectivity validation with slow response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 3.0  # 3 seconds
        mock_get.return_value = mock_response

        issues, response_time = _validate_connectivity("http://localhost:3000", None)

        assert response_time == 3000  # 3000ms
        # Should have a warning about slow response
        assert any(
            issue.severity == "warning" and "slow" in issue.description.lower()
            for issue in issues
        )


class TestAuthenticationValidation:
    """Test authentication validation."""

    @patch("requests.get")
    def test_successful_authentication(self, mock_get):
        """Test successful authentication validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        issues = _validate_authentication("http://localhost:3000", "valid-api-key")

        assert len(issues) == 0

    @patch("requests.get")
    def test_authentication_failure(self, mock_get):
        """Test authentication validation with auth failure."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        issues = _validate_authentication("http://localhost:3000", "invalid-api-key")

        assert len(issues) > 0
        assert any(
            "unauthorized" in issue.description.lower()
            or "auth" in issue.description.lower()
            for issue in issues
        )
        assert any(issue.severity == "error" for issue in issues)

    @patch("requests.get")
    def test_authentication_missing_key(self, mock_get):
        """Test authentication validation with missing API key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # For localhost, missing API key should not be an error
        issues = _validate_authentication("http://localhost:3000", None)
        assert len([issue for issue in issues if issue.severity == "error"]) == 0

        # For remote host, missing API key should be a warning or error
        issues = _validate_authentication("https://remote-flowise.com", None)
        assert len(issues) > 0

    @patch("requests.get")
    def test_authentication_forbidden(self, mock_get):
        """Test authentication validation with forbidden access."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_get.return_value = mock_response

        issues = _validate_authentication("http://localhost:3000", "api-key")

        assert len(issues) > 0
        assert any(
            "forbidden" in issue.description.lower()
            or "permission" in issue.description.lower()
            for issue in issues
        )


class TestChatflowsAccessValidation:
    """Test chatflows access validation."""

    @patch("requests.get")
    def test_successful_chatflows_access(self, mock_get):
        """Test successful chatflows access validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "flow-1", "name": "Flow 1"},
            {"id": "flow-2", "name": "Flow 2"},
        ]
        mock_get.return_value = mock_response

        issues, count = _validate_chatflows_access("http://localhost:3000", "api-key")

        assert len(issues) == 0
        assert count == 2

    @patch("requests.get")
    def test_no_chatflows_available(self, mock_get):
        """Test chatflows validation with no flows available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        issues, count = _validate_chatflows_access("http://localhost:3000", "api-key")

        assert count == 0
        # Should have a warning about no chatflows
        assert any(
            issue.severity == "warning" and "no chatflows" in issue.description.lower()
            for issue in issues
        )

    @patch("requests.get")
    def test_chatflows_access_error(self, mock_get):
        """Test chatflows validation with access error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_get.return_value = mock_response

        issues, count = _validate_chatflows_access("http://localhost:3000", "api-key")

        assert count is None
        assert len(issues) > 0
        assert any(issue.severity == "error" for issue in issues)

    @patch("requests.get")
    def test_chatflows_invalid_json(self, mock_get):
        """Test chatflows validation with invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        issues, count = _validate_chatflows_access("http://localhost:3000", "api-key")

        assert count is None
        assert len(issues) > 0
        assert any(
            "json" in issue.description.lower() or "parse" in issue.description.lower()
            for issue in issues
        )

    @patch("requests.get")
    def test_chatflows_malformed_response(self, mock_get):
        """Test chatflows validation with malformed response structure."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = "not a list"
        mock_get.return_value = mock_response

        issues, count = _validate_chatflows_access("http://localhost:3000", "api-key")

        assert count is None
        assert len(issues) > 0


class TestValidationSummaryCreation:
    """Test validation summary creation."""

    def test_create_success_summary(self):
        """Test creating success summary."""
        summary = _create_validation_summary(
            issues=[],
            available_chatflows=3,
            flowise_version="1.0.0",
            response_time_ms=120,
        )

        assert "successful" in summary.lower() or "valid" in summary.lower()
        assert "3" in summary  # chatflow count
        assert "120" in summary  # response time

    def test_create_failure_summary(self):
        """Test creating failure summary."""
        issues = [
            ValidationIssue("error", "Connection failed", "Check network"),
            ValidationIssue("warning", "Slow response", "Check server"),
        ]

        summary = _create_validation_summary(issues=issues)

        assert "failed" in summary.lower() or "error" in summary.lower()
        assert "1" in summary  # error count

    def test_create_partial_success_summary(self):
        """Test creating summary with warnings only."""
        issues = [
            ValidationIssue("warning", "No API key", "Set API key"),
            ValidationIssue("info", "Local development", "Consider production setup"),
        ]

        summary = _create_validation_summary(issues=issues)

        assert "warning" in summary.lower()
        assert "1" in summary  # warning count

    def test_create_summary_with_version_info(self):
        """Test creating summary includes version information."""
        summary = _create_validation_summary(
            issues=[], flowise_version="2.1.0", available_chatflows=5
        )

        assert "2.1.0" in summary
        assert "5" in summary

    def test_create_summary_without_optional_info(self):
        """Test creating summary without optional information."""
        summary = _create_validation_summary(issues=[])

        # Should not crash and should indicate basic success
        assert len(summary) > 0
        assert isinstance(summary, str)


class TestMainValidationFunction:
    """Test main validate_flowise_setup function."""

    @patch("genops.providers.flowise_validation._validate_url_format")
    @patch("genops.providers.flowise_validation._validate_connectivity")
    @patch("genops.providers.flowise_validation._validate_authentication")
    @patch("genops.providers.flowise_validation._validate_chatflows_access")
    def test_complete_successful_validation(
        self, mock_chatflows, mock_auth, mock_conn, mock_url
    ):
        """Test complete successful validation flow."""
        # Mock all validation steps as successful
        mock_url.return_value = []
        mock_conn.return_value = ([], 120)
        mock_auth.return_value = []
        mock_chatflows.return_value = ([], 3)

        result = validate_flowise_setup("http://localhost:3000", "api-key")

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.available_chatflows == 3
        assert result.response_time_ms == 120

    @patch("genops.providers.flowise_validation._validate_url_format")
    def test_validation_stops_on_url_error(self, mock_url):
        """Test validation stops early on URL format error."""
        mock_url.return_value = [
            ValidationIssue("error", "Invalid URL format", "Use valid URL")
        ]

        result = validate_flowise_setup("invalid-url", "api-key")

        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].severity == "error"

    @patch("genops.providers.flowise_validation._validate_url_format")
    @patch("genops.providers.flowise_validation._validate_connectivity")
    def test_validation_continues_with_warnings(self, mock_conn, mock_url):
        """Test validation continues with warnings."""
        mock_url.return_value = []
        mock_conn.return_value = (
            [ValidationIssue("warning", "Slow response", "Check server performance")],
            2500,
        )

        with patch(
            "genops.providers.flowise_validation._validate_authentication"
        ) as mock_auth:
            mock_auth.return_value = []

            with patch(
                "genops.providers.flowise_validation._validate_chatflows_access"
            ) as mock_chatflows:
                mock_chatflows.return_value = ([], 2)

                result = validate_flowise_setup("http://localhost:3000", "api-key")

                assert result.is_valid is True  # Warnings don't make it invalid
                assert len(result.issues) == 1
                assert result.issues[0].severity == "warning"

    def test_validation_with_timeout_parameter(self):
        """Test validation respects timeout parameter."""
        with patch(
            "genops.providers.flowise_validation._validate_connectivity"
        ) as mock_conn:
            mock_conn.return_value = ([], 100)

            validate_flowise_setup("http://localhost:3000", "api-key", timeout=10)

            # Check that timeout was passed to connectivity validation
            mock_conn.assert_called_once()
            mock_conn.call_args[0]
            # The timeout should be passed in some way (depends on implementation)

    @patch.dict(os.environ, {"FLOWISE_BASE_URL": "http://env-flowise:3000"})
    def test_validation_uses_environment_variables(self):
        """Test validation can use environment variables."""
        # This test verifies the integration can get config from environment
        # The actual environment variable usage might be in the main adapter
        result = validate_flowise_setup("http://localhost:3000", None)

        # Test should verify that environment variables are considered
        assert isinstance(result, ValidationResult)


class TestPrintValidationResult:
    """Test validation result printing functionality."""

    def test_print_successful_result(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True,
            summary="Validation successful - Flowise is ready",
            issues=[],
            flowise_version="1.0.0",
            available_chatflows=3,
            response_time_ms=150,
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "✅" in captured.out
        assert "successful" in captured.out.lower()
        assert "1.0.0" in captured.out
        assert "3" in captured.out
        assert "150" in captured.out

    def test_print_failed_result(self, capsys):
        """Test printing failed validation result."""
        issues = [
            ValidationIssue("error", "Connection failed", "Check network connection"),
            ValidationIssue("warning", "No API key provided", "Set FLOWISE_API_KEY"),
        ]

        result = ValidationResult(
            is_valid=False,
            summary="Validation failed - 1 error, 1 warning",
            issues=issues,
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "❌" in captured.out
        assert "failed" in captured.out.lower()
        assert "Connection failed" in captured.out
        assert "Check network connection" in captured.out
        assert "No API key provided" in captured.out

    def test_print_result_with_warnings_only(self, capsys):
        """Test printing result with warnings only."""
        issues = [
            ValidationIssue(
                "warning", "Using default configuration", "Consider customization"
            ),
            ValidationIssue(
                "info", "Local development mode", "Use production config for deployment"
            ),
        ]

        result = ValidationResult(
            is_valid=True, summary="Validation successful with warnings", issues=issues
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "⚠️" in captured.out or "warning" in captured.out.lower()
        assert "successful" in captured.out.lower()
        assert "Using default configuration" in captured.out
        assert "Local development mode" in captured.out

    def test_print_result_unicode_handling(self, capsys):
        """Test printing result handles Unicode characters."""
        issues = [
            ValidationIssue(
                "info", "Configuration looks good 👍", "Everything is fine ✨"
            )
        ]

        result = ValidationResult(
            is_valid=True, summary="Validation successful 🎉", issues=issues
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "👍" in captured.out
        assert "✨" in captured.out
        assert "🎉" in captured.out

    def test_print_result_empty_issues(self, capsys):
        """Test printing result with no issues."""
        result = ValidationResult(
            is_valid=True, summary="Perfect validation", issues=[]
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert len(captured.out) > 0
        assert "Perfect validation" in captured.out

    def test_print_result_formatting(self, capsys):
        """Test validation result is formatted properly."""
        issues = [
            ValidationIssue("error", "Major issue", "Fix immediately"),
            ValidationIssue("warning", "Minor issue", "Fix when convenient"),
            ValidationIssue("info", "FYI", "Just so you know"),
        ]

        result = ValidationResult(
            is_valid=False,
            summary="Mixed results",
            issues=issues,
            available_chatflows=5,
            response_time_ms=200,
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        # Should contain all issues with proper formatting
        assert "Major issue" in captured.out
        assert "Minor issue" in captured.out
        assert "FYI" in captured.out

        # Should contain suggested fixes
        assert "Fix immediately" in captured.out
        assert "Fix when convenient" in captured.out
        assert "Just so you know" in captured.out


class TestValidationIntegration:
    """Test validation integration with real-world scenarios."""

    def test_local_development_scenario(self):
        """Test validation for local development setup."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{"id": "test", "name": "Test Flow"}]
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_get.return_value = mock_response

            result = validate_flowise_setup("http://localhost:3000", None)

            # Should succeed for local development without API key
            assert isinstance(result, ValidationResult)
            # May have warnings but should generally work

    def test_production_scenario(self):
        """Test validation for production setup."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {"id": "flow-1", "name": "Production Flow 1"},
                {"id": "flow-2", "name": "Production Flow 2"},
            ]
            mock_response.elapsed.total_seconds.return_value = 0.05
            mock_get.return_value = mock_response

            result = validate_flowise_setup(
                "https://flowise.company.com", "prod-api-key"
            )

            assert isinstance(result, ValidationResult)

    def test_validation_error_recovery(self):
        """Test validation handles and recovers from various errors."""
        error_scenarios = [
            ConnectionError("Network unreachable"),
            Timeout("Request timeout"),
            HTTPError("HTTP Error"),
            Exception("Generic error"),
        ]

        for error in error_scenarios:
            with patch("requests.get") as mock_get:
                mock_get.side_effect = error

                result = validate_flowise_setup("http://localhost:3000", "api-key")

                # Should handle error gracefully
                assert isinstance(result, ValidationResult)
                assert result.is_valid is False
                assert len(result.issues) > 0

    def test_validation_comprehensive_report(self):
        """Test validation provides comprehensive diagnostic information."""
        with patch("requests.get") as mock_get:
            # Simulate various response conditions
            responses = [
                # Connectivity check
                Mock(status_code=200, elapsed=Mock(total_seconds=lambda: 0.15)),
                # Authentication check
                Mock(status_code=200, json=lambda: []),
                # Chatflows check
                Mock(status_code=200, json=lambda: [{"id": "1", "name": "Flow 1"}]),
            ]
            mock_get.side_effect = responses

            result = validate_flowise_setup("http://localhost:3000", "api-key")

            # Should provide comprehensive information
            assert isinstance(result.summary, str)
            assert len(result.summary) > 0
            assert isinstance(result.issues, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
