"""Test suite for Dust validation utilities."""

import os
from unittest.mock import Mock, patch

import requests

from genops.providers.dust_validation import (
    ValidationIssue,
    ValidationResult,
    check_dependencies,
    check_dust_connectivity,
    check_environment_variables,
    check_workspace_access,
    print_validation_result,
    quick_validate,
    validate_setup,
)


class TestValidationIssue:
    """Test cases for ValidationIssue dataclass."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            level="error",
            component="environment",
            message="Missing API key",
            fix_suggestion="Set DUST_API_KEY environment variable",
        )

        assert issue.level == "error"
        assert issue.component == "environment"
        assert issue.message == "Missing API key"
        assert issue.fix_suggestion == "Set DUST_API_KEY environment variable"

    def test_validation_issue_without_fix(self):
        """Test ValidationIssue creation without fix suggestion."""
        issue = ValidationIssue(
            level="warning",
            component="configuration",
            message="Optional setting not configured",
        )

        assert issue.level == "warning"
        assert issue.component == "configuration"
        assert issue.message == "Optional setting not configured"
        assert issue.fix_suggestion is None


class TestValidationResult:
    """Test cases for ValidationResult namedtuple."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        issues = [
            ValidationIssue("error", "environment", "Missing API key"),
            ValidationIssue("warning", "config", "Optional setting"),
        ]

        summary = {
            "total_issues": 2,
            "errors": 1,
            "warnings": 1,
            "api_key_configured": False,
        }

        result = ValidationResult(is_valid=False, issues=issues, summary=summary)

        assert result.is_valid is False
        assert len(result.issues) == 2
        assert result.summary["errors"] == 1
        assert result.summary["warnings"] == 1


class TestCheckEnvironmentVariables:
    """Test cases for environment variable checking."""

    def test_check_environment_variables_all_present(self):
        """Test environment check when all required variables are present."""
        with patch.dict(
            os.environ,
            {
                "DUST_API_KEY": "test-api-key",
                "DUST_WORKSPACE_ID": "test-workspace",
                "OTEL_SERVICE_NAME": "test-service",
                "GENOPS_TEAM": "test-team",
                "GENOPS_PROJECT": "test-project",
            },
        ):
            issues = check_environment_variables()

            # Should have no error issues
            error_issues = [i for i in issues if i.level == "error"]
            assert len(error_issues) == 0

            # May have some warning issues for missing optional variables
            warning_issues = [i for i in issues if i.level == "warning"]
            assert len(warning_issues) >= 0  # Optional variables may still be missing

    def test_check_environment_variables_missing_required(self):
        """Test environment check when required variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            issues = check_environment_variables()

            error_issues = [i for i in issues if i.level == "error"]
            assert len(error_issues) == 2  # DUST_API_KEY and DUST_WORKSPACE_ID

            # Check specific error messages
            error_messages = [i.message for i in error_issues]
            assert any("DUST_API_KEY" in msg for msg in error_messages)
            assert any("DUST_WORKSPACE_ID" in msg for msg in error_messages)

    def test_check_environment_variables_partial(self):
        """Test environment check with partial configuration."""
        with patch.dict(
            os.environ,
            {
                "DUST_API_KEY": "test-key"
                # Missing DUST_WORKSPACE_ID
            },
            clear=True,
        ):
            issues = check_environment_variables()

            error_issues = [i for i in issues if i.level == "error"]
            assert len(error_issues) == 1  # Only DUST_WORKSPACE_ID missing

            warning_issues = [i for i in issues if i.level == "warning"]
            assert len(warning_issues) > 0  # Optional variables missing


class TestCheckDependencies:
    """Test cases for dependency checking."""

    def test_check_dependencies_all_available(self):
        """Test dependency check when all packages are available."""
        with patch("builtins.__import__", return_value=Mock()):
            issues = check_dependencies()

            error_issues = [i for i in issues if i.level == "error"]
            assert len(error_issues) == 0

    def test_check_dependencies_requests_missing(self):
        """Test dependency check when requests is missing."""

        def mock_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("No module named 'requests'")
            return Mock()

        with patch("builtins.__import__", side_effect=mock_import):
            issues = check_dependencies()

            error_issues = [i for i in issues if i.level == "error"]
            assert len(error_issues) == 1
            assert "requests" in error_issues[0].message

    def test_check_dependencies_optional_missing(self):
        """Test dependency check when optional packages are missing."""

        def mock_import(name, *args, **kwargs):
            if "opentelemetry" in name:
                raise ImportError(f"No module named '{name}'")
            return Mock()

        with patch("builtins.__import__", side_effect=mock_import):
            issues = check_dependencies()

            warning_issues = [i for i in issues if i.level == "warning"]
            assert len(warning_issues) >= 3  # OpenTelemetry packages


class TestCheckDustConnectivity:
    """Test cases for Dust API connectivity checking."""

    @patch("requests.get")
    def test_check_dust_connectivity_success(self, mock_get):
        """Test successful connectivity check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        issues = check_dust_connectivity("test-api-key", "test-workspace")

        info_issues = [i for i in issues if i.level == "info"]
        assert len(info_issues) == 1
        assert "Successfully connected" in info_issues[0].message

    @patch("requests.get")
    def test_check_dust_connectivity_unauthorized(self, mock_get):
        """Test connectivity check with invalid API key."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        issues = check_dust_connectivity("invalid-key", "test-workspace")

        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 1
        assert "Authentication failed" in error_issues[0].message
        assert "Invalid API key" in error_issues[0].message

    @patch("requests.get")
    def test_check_dust_connectivity_forbidden(self, mock_get):
        """Test connectivity check with insufficient permissions."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        issues = check_dust_connectivity("test-key", "test-workspace")

        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 1
        assert "Access denied" in error_issues[0].message

    @patch("requests.get")
    def test_check_dust_connectivity_not_found(self, mock_get):
        """Test connectivity check with invalid workspace ID."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        issues = check_dust_connectivity("test-key", "invalid-workspace")

        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 1
        assert "Workspace not found" in error_issues[0].message

    @patch("requests.get")
    def test_check_dust_connectivity_connection_error(self, mock_get):
        """Test connectivity check with connection error."""
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        issues = check_dust_connectivity("test-key", "test-workspace")

        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 1
        assert "Cannot connect to Dust API" in error_issues[0].message

    @patch("requests.get")
    def test_check_dust_connectivity_timeout(self, mock_get):
        """Test connectivity check with timeout."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        issues = check_dust_connectivity("test-key", "test-workspace")

        warning_issues = [i for i in issues if i.level == "warning"]
        assert len(warning_issues) == 1
        assert "timed out" in warning_issues[0].message

    def test_check_dust_connectivity_missing_credentials(self):
        """Test connectivity check without credentials."""
        issues = check_dust_connectivity(None, None)

        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 2  # API key and workspace ID missing


class TestCheckWorkspaceAccess:
    """Test cases for workspace access checking."""

    @patch("requests.get")
    def test_check_workspace_access_success(self, mock_get):
        """Test successful workspace access check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        issues = check_workspace_access("test-key", "test-workspace")

        info_issues = [i for i in issues if i.level == "info"]
        assert len(info_issues) == 1
        assert "Workspace access verified" in info_issues[0].message

    @patch("requests.get")
    def test_check_workspace_access_partial(self, mock_get):
        """Test workspace access with some restricted endpoints."""

        def mock_get_side_effect(url, **kwargs):
            mock_response = Mock()
            if "conversations" in url:
                mock_response.status_code = 200
            elif "agents" in url:
                mock_response.status_code = 200
            else:  # data_sources
                mock_response.status_code = 403
            return mock_response

        mock_get.side_effect = mock_get_side_effect

        issues = check_workspace_access("test-key", "test-workspace")

        info_issues = [i for i in issues if i.level == "info"]
        warning_issues = [i for i in issues if i.level == "warning"]

        assert len(info_issues) >= 1  # Some endpoints accessible
        assert len(warning_issues) >= 1  # Some endpoints restricted

    def test_check_workspace_access_missing_credentials(self):
        """Test workspace access check without credentials."""
        issues = check_workspace_access(None, None)

        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 1
        assert "Cannot check workspace access" in error_issues[0].message


class TestValidateSetup:
    """Test cases for comprehensive setup validation."""

    def test_validate_setup_success(self):
        """Test successful comprehensive validation."""
        with patch.dict(
            os.environ,
            {
                "DUST_API_KEY": "test-key",
                "DUST_WORKSPACE_ID": "test-workspace",
                "OTEL_SERVICE_NAME": "test-service",
            },
        ):
            with patch(
                "genops.providers.dust_validation.check_dust_connectivity"
            ) as mock_conn:
                with patch(
                    "genops.providers.dust_validation.check_workspace_access"
                ) as mock_access:
                    # Mock successful connectivity
                    mock_conn.return_value = [
                        ValidationIssue(
                            "info", "connectivity", "Successfully connected"
                        )
                    ]
                    mock_access.return_value = [
                        ValidationIssue("info", "workspace", "Access verified")
                    ]

                    result = validate_setup()

                    assert result.is_valid is True
                    assert result.summary["errors"] == 0
                    assert result.summary["api_key_configured"] is True
                    assert result.summary["workspace_configured"] is True

    def test_validate_setup_failure(self):
        """Test validation with errors."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_setup()

            assert result.is_valid is False
            assert result.summary["errors"] > 0
            assert result.summary["api_key_configured"] is False
            assert result.summary["workspace_configured"] is False

    def test_validate_setup_with_explicit_credentials(self):
        """Test validation with explicitly provided credentials."""
        with patch(
            "genops.providers.dust_validation.check_dust_connectivity"
        ) as mock_conn:
            with patch(
                "genops.providers.dust_validation.check_workspace_access"
            ) as mock_access:
                mock_conn.return_value = [
                    ValidationIssue("info", "connectivity", "Successfully connected")
                ]
                mock_access.return_value = [
                    ValidationIssue("info", "workspace", "Access verified")
                ]

                validate_setup(
                    api_key="explicit-key", workspace_id="explicit-workspace"
                )

                # Should call connectivity checks with explicit credentials
                mock_conn.assert_called_with(
                    "explicit-key", "explicit-workspace", "https://dust.tt"
                )
                mock_access.assert_called_with(
                    "explicit-key", "explicit-workspace", "https://dust.tt"
                )

    def test_validate_setup_custom_base_url(self):
        """Test validation with custom base URL."""
        with patch(
            "genops.providers.dust_validation.check_dust_connectivity"
        ) as mock_conn:
            with patch(
                "genops.providers.dust_validation.check_workspace_access"
            ) as mock_access:
                mock_conn.return_value = []
                mock_access.return_value = []

                validate_setup(
                    api_key="test-key",
                    workspace_id="test-workspace",
                    base_url="https://custom.dust.tt",
                )

                mock_conn.assert_called_with(
                    "test-key", "test-workspace", "https://custom.dust.tt"
                )
                mock_access.assert_called_with(
                    "test-key", "test-workspace", "https://custom.dust.tt"
                )


class TestPrintValidationResult:
    """Test cases for validation result printing."""

    @patch("builtins.print")
    def test_print_validation_result_success(self, mock_print):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True,
            issues=[ValidationIssue("info", "connectivity", "Successfully connected")],
            summary={
                "total_issues": 1,
                "errors": 0,
                "warnings": 0,
                "info": 1,
                "is_ready_for_production": True,
                "api_key_configured": True,
                "workspace_configured": True,
                "telemetry_configured": True,
                "governance_attributes_configured": True,
            },
        )

        print_validation_result(result)

        # Verify print was called (basic check)
        assert mock_print.called

        # Check that success indicators were printed
        all_print_calls = [call[0][0] for call in mock_print.call_args_list]
        printed_text = " ".join(str(call) for call in all_print_calls)

        assert "✅ READY" in printed_text or "READY" in printed_text

    @patch("builtins.print")
    def test_print_validation_result_failure(self, mock_print):
        """Test printing failed validation result."""
        result = ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    "error", "environment", "Missing API key", "Set DUST_API_KEY"
                ),
                ValidationIssue("warning", "config", "Optional setting missing"),
            ],
            summary={
                "total_issues": 2,
                "errors": 1,
                "warnings": 1,
                "info": 0,
                "is_ready_for_production": False,
                "api_key_configured": False,
                "workspace_configured": True,
                "telemetry_configured": False,
                "governance_attributes_configured": False,
            },
        )

        print_validation_result(result)

        # Verify print was called
        assert mock_print.called

        # Check that failure indicators were printed
        all_print_calls = [call[0][0] for call in mock_print.call_args_list]
        printed_text = " ".join(str(call) for call in all_print_calls)

        assert "❌ NEEDS ATTENTION" in printed_text or "NEEDS ATTENTION" in printed_text


class TestQuickValidate:
    """Test cases for quick validation function."""

    def test_quick_validate_success(self):
        """Test successful quick validation."""
        with patch("genops.providers.dust_validation.validate_setup") as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True, issues=[], summary={}
            )

            result = quick_validate()

            assert result is True
            mock_validate.assert_called_once()

    def test_quick_validate_failure(self):
        """Test failed quick validation."""
        with patch("genops.providers.dust_validation.validate_setup") as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=False, issues=[], summary={}
            )

            result = quick_validate()

            assert result is False
            mock_validate.assert_called_once()


class TestValidationIntegration:
    """Integration test cases combining multiple validation components."""

    def test_full_validation_flow_success(self):
        """Test complete validation flow with all components succeeding."""
        with patch.dict(
            os.environ,
            {
                "DUST_API_KEY": "test-key",
                "DUST_WORKSPACE_ID": "test-workspace",
                "OTEL_SERVICE_NAME": "test-service",
                "GENOPS_TEAM": "test-team",
                "GENOPS_PROJECT": "test-project",
            },
        ):
            with patch("requests.get") as mock_get:
                # Mock successful API responses
                mock_response = Mock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response

                result = validate_setup()

                assert result.is_valid is True
                assert result.summary["api_key_configured"] is True
                assert result.summary["workspace_configured"] is True
                assert result.summary["telemetry_configured"] is True
                assert result.summary["governance_attributes_configured"] is True

    def test_full_validation_flow_partial_failure(self):
        """Test validation flow with some components failing."""
        with patch.dict(
            os.environ,
            {
                "DUST_API_KEY": "test-key",
                # Missing DUST_WORKSPACE_ID
                "OTEL_SERVICE_NAME": "test-service",
            },
            clear=True,
        ):
            result = validate_setup()

            assert result.is_valid is False
            assert result.summary["errors"] > 0
            assert result.summary["api_key_configured"] is True
            assert result.summary["workspace_configured"] is False

    def test_validation_error_categorization(self):
        """Test that validation issues are properly categorized."""
        with patch.dict(os.environ, {}, clear=True):
            # This should generate both errors and warnings
            result = validate_setup()

            errors = [i for i in result.issues if i.level == "error"]
            warnings = [i for i in result.issues if i.level == "warning"]

            assert len(errors) >= 2  # At least missing API key and workspace ID
            assert len(warnings) > 0  # Optional environment variables

            assert result.summary["errors"] == len(errors)
            assert result.summary["warnings"] == len(warnings)

    def test_validation_with_network_errors(self):
        """Test validation handling of network connectivity issues."""
        with patch.dict(
            os.environ,
            {"DUST_API_KEY": "test-key", "DUST_WORKSPACE_ID": "test-workspace"},
        ):
            with patch("requests.get") as mock_get:
                mock_get.side_effect = requests.ConnectionError("Network error")

                result = validate_setup()

                # Should have connectivity errors but not fail completely
                # if other components are valid
                connectivity_errors = [
                    i
                    for i in result.issues
                    if i.component == "connectivity" and i.level == "error"
                ]
                assert len(connectivity_errors) > 0
