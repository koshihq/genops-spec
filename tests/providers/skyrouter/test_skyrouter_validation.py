"""
Comprehensive tests for SkyRouter validation functionality.

Tests setup validation, configuration checking, interactive setup,
and diagnostic capabilities.
"""

import os
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.skyrouter_validation import (
        IssueSeverity,
        IssueType,
        SkyRouterValidator,
        ValidationIssue,
        ValidationResult,
        print_validation_result,
        validate_skyrouter_setup,
    )

    SKYROUTER_VALIDATION_AVAILABLE = True
except ImportError:
    SKYROUTER_VALIDATION_AVAILABLE = False


@pytest.mark.skipif(
    not SKYROUTER_VALIDATION_AVAILABLE,
    reason="SkyRouter validation module not available",
)
class TestSkyRouterValidator:
    """Test suite for SkyRouter validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SkyRouterValidator()

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = SkyRouterValidator()
        assert validator is not None
        assert hasattr(validator, "validate_setup")
        assert hasattr(validator, "validate_configuration")

    def test_api_key_validation(self):
        """Test API key validation."""
        # Test missing API key
        with patch.dict(os.environ, {}, clear=True):
            issues = self.validator.validate_api_key()
            assert len(issues) > 0
            assert any(
                issue.issue_type == IssueType.MISSING_API_KEY for issue in issues
            )

        # Test present but invalid API key
        with patch.dict(os.environ, {"SKYROUTER_API_KEY": "invalid-key"}, clear=True):
            with patch(
                "genops.providers.skyrouter_validation.requests.get"
            ) as mock_get:
                mock_response = Mock()
                mock_response.status_code = 401
                mock_get.return_value = mock_response

                issues = self.validator.validate_api_key()
                assert len(issues) > 0
                assert any(
                    issue.issue_type == IssueType.INVALID_API_KEY for issue in issues
                )

        # Test valid API key
        with patch.dict(
            os.environ, {"SKYROUTER_API_KEY": "sk-valid-key-123"}, clear=True
        ):
            with patch(
                "genops.providers.skyrouter_validation.requests.get"
            ) as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "active"}
                mock_get.return_value = mock_response

                issues = self.validator.validate_api_key()
                assert (
                    len(
                        [
                            issue
                            for issue in issues
                            if issue.severity == IssueSeverity.ERROR
                        ]
                    )
                    == 0
                )

    def test_dependency_validation(self):
        """Test dependency validation."""
        # Test all dependencies present
        with patch(
            "genops.providers.skyrouter_validation.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = Mock()

            issues = self.validator.validate_dependencies()
            error_issues = [
                issue for issue in issues if issue.severity == IssueSeverity.ERROR
            ]
            assert len(error_issues) == 0

        # Test missing dependency
        with patch(
            "genops.providers.skyrouter_validation.importlib.import_module"
        ) as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            issues = self.validator.validate_dependencies()
            assert len(issues) > 0
            assert any(
                issue.issue_type == IssueType.MISSING_DEPENDENCY for issue in issues
            )

    def test_network_connectivity_validation(self):
        """Test network connectivity validation."""
        # Test successful connection
        with patch("genops.providers.skyrouter_validation.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 0.5
            mock_get.return_value = mock_response

            issues = self.validator.validate_network_connectivity()
            error_issues = [
                issue for issue in issues if issue.severity == IssueSeverity.ERROR
            ]
            assert len(error_issues) == 0

        # Test connection failure
        with patch("genops.providers.skyrouter_validation.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError("Network unreachable")

            issues = self.validator.validate_network_connectivity()
            assert len(issues) > 0
            assert any(issue.issue_type == IssueType.NETWORK_ERROR for issue in issues)

        # Test slow connection
        with patch("genops.providers.skyrouter_validation.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 10.0  # Very slow
            mock_get.return_value = mock_response

            issues = self.validator.validate_network_connectivity()
            warning_issues = [
                issue for issue in issues if issue.severity == IssueSeverity.WARNING
            ]
            assert len(warning_issues) > 0

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration
        valid_config = {
            "team": "test-team",
            "project": "test-project",
            "environment": "development",
            "daily_budget_limit": 50.0,
            "governance_policy": "advisory",
        }

        issues = self.validator.validate_configuration(valid_config)
        error_issues = [
            issue for issue in issues if issue.severity == IssueSeverity.ERROR
        ]
        assert len(error_issues) == 0

        # Invalid budget limit
        invalid_budget_config = valid_config.copy()
        invalid_budget_config["daily_budget_limit"] = -10.0

        issues = self.validator.validate_configuration(invalid_budget_config)
        assert len(issues) > 0
        assert any(
            issue.issue_type == IssueType.INVALID_CONFIGURATION for issue in issues
        )

        # Invalid governance policy
        invalid_policy_config = valid_config.copy()
        invalid_policy_config["governance_policy"] = "invalid-policy"

        issues = self.validator.validate_configuration(invalid_policy_config)
        assert len(issues) > 0
        assert any(
            issue.issue_type == IssueType.INVALID_CONFIGURATION for issue in issues
        )

    def test_model_availability_validation(self):
        """Test model availability validation."""
        # Test available models
        with patch("genops.providers.skyrouter_validation.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo", "gemini-pro"]
            }
            mock_get.return_value = mock_response

            issues = self.validator.validate_model_availability()
            error_issues = [
                issue for issue in issues if issue.severity == IssueSeverity.ERROR
            ]
            assert len(error_issues) == 0

        # Test API error
        with patch("genops.providers.skyrouter_validation.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            issues = self.validator.validate_model_availability()
            assert len(issues) > 0

    def test_permissions_validation(self):
        """Test permissions and access validation."""
        # Test with valid API key
        with patch.dict(os.environ, {"SKYROUTER_API_KEY": "sk-valid-key"}, clear=True):
            with patch(
                "genops.providers.skyrouter_validation.requests.get"
            ) as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "permissions": ["read", "write", "route"],
                    "rate_limits": {"rpm": 1000, "tpm": 100000},
                }
                mock_get.return_value = mock_response

                issues = self.validator.validate_permissions()
                error_issues = [
                    issue for issue in issues if issue.severity == IssueSeverity.ERROR
                ]
                assert len(error_issues) == 0

        # Test insufficient permissions
        with patch.dict(
            os.environ, {"SKYROUTER_API_KEY": "sk-limited-key"}, clear=True
        ):
            with patch(
                "genops.providers.skyrouter_validation.requests.get"
            ) as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "permissions": ["read"],  # Missing write and route permissions
                    "rate_limits": {"rpm": 10, "tpm": 1000},  # Low limits
                }
                mock_get.return_value = mock_response

                issues = self.validator.validate_permissions()
                assert len(issues) > 0

    def test_complete_validation_workflow(self):
        """Test complete validation workflow."""
        # Mock all validation methods to return success
        with patch.object(self.validator, "validate_api_key", return_value=[]):
            with patch.object(self.validator, "validate_dependencies", return_value=[]):
                with patch.object(
                    self.validator, "validate_network_connectivity", return_value=[]
                ):
                    with patch.object(
                        self.validator, "validate_configuration", return_value=[]
                    ):
                        with patch.object(
                            self.validator,
                            "validate_model_availability",
                            return_value=[],
                        ):
                            with patch.object(
                                self.validator, "validate_permissions", return_value=[]
                            ):
                                result = self.validator.validate_setup()

                                assert isinstance(result, ValidationResult)
                                assert result.is_valid is True
                                assert len(result.errors) == 0

    def test_validation_with_multiple_issues(self):
        """Test validation with multiple types of issues."""
        mock_issues = [
            ValidationIssue(
                issue_type=IssueType.MISSING_API_KEY,
                severity=IssueSeverity.ERROR,
                message="SKYROUTER_API_KEY not found",
                fix_suggestion="Set your API key: export SKYROUTER_API_KEY='your-key'",
            ),
            ValidationIssue(
                issue_type=IssueType.NETWORK_ERROR,
                severity=IssueSeverity.WARNING,
                message="Slow network connection detected",
                fix_suggestion="Check your internet connection",
            ),
        ]

        with patch.object(
            self.validator, "validate_api_key", return_value=[mock_issues[0]]
        ):
            with patch.object(self.validator, "validate_dependencies", return_value=[]):
                with patch.object(
                    self.validator,
                    "validate_network_connectivity",
                    return_value=[mock_issues[1]],
                ):
                    with patch.object(
                        self.validator, "validate_configuration", return_value=[]
                    ):
                        with patch.object(
                            self.validator,
                            "validate_model_availability",
                            return_value=[],
                        ):
                            with patch.object(
                                self.validator, "validate_permissions", return_value=[]
                            ):
                                result = self.validator.validate_setup()

                                assert result.is_valid is False
                                assert len(result.errors) == 1
                                assert len(result.warnings) == 1

    def test_interactive_setup_guidance(self):
        """Test interactive setup guidance."""
        # Mock user input for interactive setup
        with patch(
            "builtins.input",
            side_effect=["y", "test-api-key", "test-team", "test-project"],
        ):
            guidance = self.validator.provide_interactive_setup_guidance()

            assert isinstance(guidance, dict)
            assert "steps_completed" in guidance
            assert "configuration_generated" in guidance


@pytest.mark.skipif(
    not SKYROUTER_VALIDATION_AVAILABLE,
    reason="SkyRouter validation module not available",
)
class TestValidationResult:
    """Test suite for ValidationResult class."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation and properties."""
        issues = [
            ValidationIssue(
                issue_type=IssueType.MISSING_API_KEY,
                severity=IssueSeverity.ERROR,
                message="API key not found",
                fix_suggestion="Add API key",
            ),
            ValidationIssue(
                issue_type=IssueType.NETWORK_ERROR,
                severity=IssueSeverity.WARNING,
                message="Slow connection",
                fix_suggestion="Check network",
            ),
        ]

        result = ValidationResult(issues=issues)

        assert result.is_valid is False  # Has error
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.all_issues) == 2

    def test_validation_result_with_no_issues(self):
        """Test ValidationResult with no issues."""
        result = ValidationResult(issues=[])

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.all_issues) == 0

    def test_validation_result_summary(self):
        """Test ValidationResult summary generation."""
        issues = [
            ValidationIssue(
                issue_type=IssueType.INVALID_API_KEY,
                severity=IssueSeverity.ERROR,
                message="Invalid API key",
                fix_suggestion="Check your API key",
            )
        ]

        result = ValidationResult(issues=issues)
        summary = result.get_summary()

        assert isinstance(summary, dict)
        assert "is_valid" in summary
        assert "error_count" in summary
        assert "warning_count" in summary


@pytest.mark.skipif(
    not SKYROUTER_VALIDATION_AVAILABLE,
    reason="SkyRouter validation module not available",
)
class TestValidationIssue:
    """Test suite for ValidationIssue class."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            issue_type=IssueType.MISSING_DEPENDENCY,
            severity=IssueSeverity.ERROR,
            message="Required dependency not found",
            fix_suggestion="Install with: pip install skyrouter",
            context={"dependency": "skyrouter"},
        )

        assert issue.issue_type == IssueType.MISSING_DEPENDENCY
        assert issue.severity == IssueSeverity.ERROR
        assert issue.message == "Required dependency not found"
        assert "dependency" in issue.context

    def test_validation_issue_string_representation(self):
        """Test ValidationIssue string representation."""
        issue = ValidationIssue(
            issue_type=IssueType.NETWORK_ERROR,
            severity=IssueSeverity.WARNING,
            message="Network timeout",
            fix_suggestion="Retry operation",
        )

        str_repr = str(issue)
        assert "WARNING" in str_repr
        assert "Network timeout" in str_repr

    def test_issue_severity_ordering(self):
        """Test issue severity ordering."""
        assert IssueSeverity.ERROR > IssueSeverity.WARNING
        assert IssueSeverity.WARNING > IssueSeverity.INFO

    def test_issue_type_categorization(self):
        """Test issue type categorization."""
        config_issues = [
            IssueType.MISSING_API_KEY,
            IssueType.INVALID_API_KEY,
            IssueType.INVALID_CONFIGURATION,
        ]

        network_issues = [IssueType.NETWORK_ERROR, IssueType.API_UNREACHABLE]

        dependency_issues = [IssueType.MISSING_DEPENDENCY, IssueType.VERSION_MISMATCH]

        # All issue types should be categorizable
        all_types = config_issues + network_issues + dependency_issues
        assert len(all_types) > 0


@pytest.mark.skipif(
    not SKYROUTER_VALIDATION_AVAILABLE,
    reason="SkyRouter validation module not available",
)
class TestValidationFunctions:
    """Test suite for standalone validation functions."""

    @patch.dict(os.environ, {"SKYROUTER_API_KEY": "test-key"}, clear=True)
    def test_validate_skyrouter_setup_function(self):
        """Test standalone validate_skyrouter_setup function."""
        with patch(
            "genops.providers.skyrouter_validation.SkyRouterValidator"
        ) as mock_validator:
            mock_validator_instance = Mock()
            mock_validator_instance.validate_setup.return_value = ValidationResult(
                issues=[]
            )
            mock_validator.return_value = mock_validator_instance

            result = validate_skyrouter_setup()

            assert isinstance(result, ValidationResult)
            mock_validator_instance.validate_setup.assert_called_once()

    def test_print_validation_result_function(self):
        """Test print_validation_result function."""
        issues = [
            ValidationIssue(
                issue_type=IssueType.MISSING_API_KEY,
                severity=IssueSeverity.ERROR,
                message="API key not found",
                fix_suggestion="Add your API key",
            )
        ]

        result = ValidationResult(issues=issues)

        # Should not raise exception
        try:
            print_validation_result(result)
        except Exception as e:
            pytest.fail(f"print_validation_result raised an exception: {e}")

    def test_validation_with_custom_configuration(self):
        """Test validation with custom configuration parameters."""
        custom_config = {
            "team": "custom-team",
            "project": "custom-project",
            "environment": "production",
            "daily_budget_limit": 200.0,
            "governance_policy": "strict",
            "enable_cost_alerts": True,
        }

        with patch(
            "genops.providers.skyrouter_validation.SkyRouterValidator"
        ) as mock_validator:
            mock_validator_instance = Mock()
            mock_validator_instance.validate_setup.return_value = ValidationResult(
                issues=[]
            )
            mock_validator.return_value = mock_validator_instance

            result = validate_skyrouter_setup(config=custom_config)

            assert isinstance(result, ValidationResult)


@pytest.mark.skipif(
    not SKYROUTER_VALIDATION_AVAILABLE,
    reason="SkyRouter validation module not available",
)
class TestValidationDiagnostics:
    """Test suite for validation diagnostics and troubleshooting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SkyRouterValidator()

    def test_diagnostic_information_collection(self):
        """Test diagnostic information collection."""
        diagnostics = self.validator.collect_diagnostics()

        assert isinstance(diagnostics, dict)
        assert "system_info" in diagnostics
        assert "environment_variables" in diagnostics
        assert "installed_packages" in diagnostics

    def test_troubleshooting_guide_generation(self):
        """Test troubleshooting guide generation."""
        issues = [
            ValidationIssue(
                issue_type=IssueType.NETWORK_ERROR,
                severity=IssueSeverity.ERROR,
                message="Cannot connect to SkyRouter API",
                fix_suggestion="Check network connectivity",
            )
        ]

        guide = self.validator.generate_troubleshooting_guide(issues)

        assert isinstance(guide, dict)
        assert "common_solutions" in guide
        assert "step_by_step_fixes" in guide

    def test_environment_analysis(self):
        """Test environment analysis for common issues."""
        analysis = self.validator.analyze_environment()

        assert isinstance(analysis, dict)
        assert "python_version" in analysis
        assert "operating_system" in analysis
        assert "network_status" in analysis

    def test_configuration_recommendations(self):
        """Test configuration recommendations."""
        current_config = {
            "team": "test-team",
            "project": "test-project",
            "daily_budget_limit": 10.0,  # Low budget
        }

        recommendations = self.validator.generate_configuration_recommendations(
            current_config
        )

        assert isinstance(recommendations, list)
        # Should recommend increasing budget for production use
        budget_recommendations = [
            r for r in recommendations if "budget" in r.get("category", "")
        ]
        assert len(budget_recommendations) > 0

    def test_performance_validation(self):
        """Test performance-related validation."""
        with patch("genops.providers.skyrouter_validation.requests.get") as mock_get:
            # Simulate slow response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 5.0
            mock_get.return_value = mock_response

            performance_issues = self.validator.validate_performance()

            # Should detect performance issues
            assert len(performance_issues) > 0
            slow_response_issues = [
                issue for issue in performance_issues if "slow" in issue.message.lower()
            ]
            assert len(slow_response_issues) > 0

    def test_security_validation(self):
        """Test security-related validation."""
        # Test with API key in environment (good)
        with patch.dict(os.environ, {"SKYROUTER_API_KEY": "sk-secure-key"}, clear=True):
            security_issues = self.validator.validate_security()

            # Should not have major security issues
            critical_issues = [
                issue
                for issue in security_issues
                if issue.severity == IssueSeverity.ERROR
            ]
            assert len(critical_issues) == 0

        # Test with API key in code (bad - simulated)
        with patch(
            "genops.providers.skyrouter_validation.inspect.getsource"
        ) as mock_source:
            mock_source.return_value = "api_key = 'sk-hardcoded-key-123'"

            security_issues = self.validator.validate_security()

            # Should detect security issues
            assert len(security_issues) > 0


if __name__ == "__main__":
    pytest.main([__file__])
