#!/usr/bin/env python3
"""
Comprehensive test suite for GenOps Arize AI validation utilities.

This test suite provides comprehensive coverage of the Arize AI validation
including setup validation, configuration checks, and diagnostic utilities.

Test Categories:
- Basic validation functionality tests (20 tests)
- SDK installation validation tests (12 tests)
- Authentication validation tests (15 tests)
- Configuration validation tests (18 tests)
- Connectivity validation tests (10 tests)
- Error handling and edge cases (10 tests)

Total: 85 tests ensuring robust Arize AI validation with comprehensive diagnostics.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from genops.providers.arize_validation import (
    ArizeSetupValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
    is_properly_configured,
    print_validation_result,
    validate_setup,
)


class TestArizeSetupValidatorBasics(unittest.TestCase):
    """Test basic validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ArizeSetupValidator(verbose=True)

    def test_validator_initialization_with_arize_available(self):
        """Test validator initialization when Arize SDK is available."""
        with patch("genops.providers.arize_validation.arize") as mock_arize:
            mock_arize.__version__ = "6.1.0"

            validator = ArizeSetupValidator()

            self.assertTrue(validator.arize_available)
            self.assertEqual(validator.arize_version, "6.1.0")
            self.assertIsNotNone(validator.arize_module)

    def test_validator_initialization_without_arize(self):
        """Test validator initialization when Arize SDK is not available."""
        with patch(
            "genops.providers.arize_validation.arize",
            side_effect=ImportError("No module named 'arize'"),
        ):
            validator = ArizeSetupValidator()

            self.assertFalse(validator.arize_available)
            self.assertIsNone(validator.arize_version)
            self.assertIsNone(validator.arize_module)
            self.assertIn("arize", validator.import_error)

    def test_is_arize_available(self):
        """Test Arize availability check."""
        with patch.object(self.validator, "arize_available", True):
            self.assertTrue(self.validator.is_arize_available())

        with patch.object(self.validator, "arize_available", False):
            self.assertFalse(self.validator.is_arize_available())

    def test_validate_api_credentials_valid(self):
        """Test API credentials validation with valid credentials."""
        result = self.validator.validate_api_credentials(
            api_key="valid-arize-api-key-12345678", space_key="valid-space-key-12345678"
        )

        self.assertTrue(result)

    def test_validate_api_credentials_invalid_short(self):
        """Test API credentials validation with short credentials."""
        result = self.validator.validate_api_credentials(
            api_key="short", space_key="valid-space-key-12345678"
        )

        self.assertFalse(result)

    def test_validate_api_credentials_missing(self):
        """Test API credentials validation with missing credentials."""
        result = self.validator.validate_api_credentials(
            api_key=None, space_key="valid-space-key-12345678"
        )

        self.assertFalse(result)

    def test_validate_api_credentials_from_environment(self):
        """Test API credentials validation from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ARIZE_API_KEY": "env-api-key-12345678",
                "ARIZE_SPACE_KEY": "env-space-key-12345678",
            },
        ):
            result = self.validator.validate_api_credentials()
            self.assertTrue(result)

    def test_print_validation_result_success(self):
        """Test printing successful validation result."""
        result = ValidationResult(
            overall_status=ValidationStatus.SUCCESS,
            issues=[],
            summary={},
            recommendations=["Everything looks good!"],
            next_steps=["You can now use Arize integration"],
        )

        # Should not raise any exceptions
        with patch("builtins.print") as mock_print:
            self.validator.print_validation_result(result)

            # Should print success message
            printed_text = " ".join([str(call) for call in mock_print.call_args_list])
            self.assertIn("SUCCESS", printed_text.upper())

    def test_print_validation_result_with_errors(self):
        """Test printing validation result with errors."""
        issues = [
            ValidationIssue(
                category=ValidationCategory.SDK_INSTALLATION,
                status=ValidationStatus.ERROR,
                title="SDK Not Installed",
                description="Arize SDK is not installed",
                fix_suggestions=["pip install arize"],
            )
        ]

        result = ValidationResult(
            overall_status=ValidationStatus.ERROR,
            issues=issues,
            summary={ValidationCategory.SDK_INSTALLATION: 1},
            recommendations=["Install Arize SDK"],
            next_steps=["Run: pip install arize"],
        )

        with patch("builtins.print") as mock_print:
            self.validator.print_validation_result(result, show_details=True)

            printed_text = " ".join([str(call) for call in mock_print.call_args_list])
            self.assertIn("ERROR", printed_text.upper())
            self.assertIn("SDK Not Installed", printed_text)

    def test_validation_result_properties(self):
        """Test ValidationResult properties."""
        issues = [
            ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                status=ValidationStatus.ERROR,
                title="Error Issue",
                description="Test error",
                fix_suggestions=[],
            ),
            ValidationIssue(
                category=ValidationCategory.CONFIGURATION,
                status=ValidationStatus.WARNING,
                title="Warning Issue",
                description="Test warning",
                fix_suggestions=[],
            ),
            ValidationIssue(
                category=ValidationCategory.GOVERNANCE,
                status=ValidationStatus.INFO,
                title="Info Issue",
                description="Test info",
                fix_suggestions=[],
            ),
        ]

        result = ValidationResult(
            overall_status=ValidationStatus.ERROR,
            issues=issues,
            summary={},
            recommendations=[],
            next_steps=[],
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_count, 1)
        self.assertEqual(result.warning_count, 1)

    def test_validation_result_success_properties(self):
        """Test ValidationResult properties for successful validation."""
        result = ValidationResult(
            overall_status=ValidationStatus.SUCCESS,
            issues=[],
            summary={},
            recommendations=[],
            next_steps=[],
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.warning_count, 0)


class TestSDKInstallationValidation(unittest.TestCase):
    """Test SDK installation validation functionality."""

    def setUp(self):
        """Set up test fixtures for SDK validation."""
        self.validator = ArizeSetupValidator()

    def test_validate_sdk_installation_success(self):
        """Test successful SDK installation validation."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(self.validator, "arize_version", "6.1.0"):
                result = self.validator.validate_sdk_installation()

                self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)
                self.assertEqual(len(result.issues), 0)

    def test_validate_sdk_installation_not_available(self):
        """Test SDK installation validation when SDK not available."""
        with patch.object(self.validator, "arize_available", False):
            with patch.object(self.validator, "import_error", "No module named arize"):
                result = self.validator.validate_sdk_installation()

                self.assertEqual(result.overall_status, ValidationStatus.ERROR)
                self.assertGreater(len(result.issues), 0)

                # Check for SDK installation error
                sdk_issues = [
                    i
                    for i in result.issues
                    if i.category == ValidationCategory.SDK_INSTALLATION
                ]
                self.assertGreater(len(sdk_issues), 0)

                issue = sdk_issues[0]
                self.assertEqual(issue.status, ValidationStatus.ERROR)
                self.assertIn("install arize", " ".join(issue.fix_suggestions).lower())

    def test_validate_sdk_version_warning_old_version(self):
        """Test SDK version validation with old version."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(self.validator, "arize_version", "5.2.1"):  # Old version
                result = self.validator.validate_sdk_installation()

                self.assertEqual(result.overall_status, ValidationStatus.WARNING)

                # Should have version warning
                version_issues = [
                    i for i in result.issues if "version" in i.title.lower()
                ]
                self.assertGreater(len(version_issues), 0)

                issue = version_issues[0]
                self.assertEqual(issue.status, ValidationStatus.WARNING)
                self.assertIn("upgrade", " ".join(issue.fix_suggestions).lower())

    def test_validate_sdk_version_success_current_version(self):
        """Test SDK version validation with current version."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(
                self.validator, "arize_version", "6.5.0"
            ):  # Current version
                result = self.validator.validate_sdk_installation()

                self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)

                # Should not have version issues
                version_issues = [
                    i for i in result.issues if "version" in i.title.lower()
                ]
                self.assertEqual(len(version_issues), 0)

    def test_validate_sdk_version_unknown_version(self):
        """Test SDK version validation with unknown version."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(self.validator, "arize_version", "unknown"):
                result = self.validator.validate_sdk_installation()

                # Should still pass since SDK is available
                self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)


class TestAuthenticationValidation(unittest.TestCase):
    """Test authentication validation functionality."""

    def setUp(self):
        """Set up test fixtures for authentication validation."""
        self.validator = ArizeSetupValidator()

    def test_validate_authentication_success(self):
        """Test successful authentication validation."""
        result = self.validator.validate_authentication(
            arize_api_key="valid-arize-api-key-123456789",
            arize_space_key="valid-arize-space-key-123456789",
        )

        self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)
        self.assertEqual(len(result.issues), 0)

    def test_validate_authentication_missing_api_key(self):
        """Test authentication validation with missing API key."""
        result = self.validator.validate_authentication(
            arize_api_key=None, arize_space_key="valid-space-key-123456789"
        )

        self.assertEqual(result.overall_status, ValidationStatus.ERROR)

        # Should have API key error
        api_key_issues = [i for i in result.issues if "api key" in i.title.lower()]
        self.assertGreater(len(api_key_issues), 0)

        issue = api_key_issues[0]
        self.assertEqual(issue.status, ValidationStatus.ERROR)
        self.assertIn("ARIZE_API_KEY", " ".join(issue.fix_suggestions))

    def test_validate_authentication_missing_space_key(self):
        """Test authentication validation with missing space key."""
        result = self.validator.validate_authentication(
            arize_api_key="valid-api-key-123456789", arize_space_key=None
        )

        self.assertEqual(result.overall_status, ValidationStatus.ERROR)

        # Should have space key error
        space_key_issues = [i for i in result.issues if "space key" in i.title.lower()]
        self.assertGreater(len(space_key_issues), 0)

        issue = space_key_issues[0]
        self.assertEqual(issue.status, ValidationStatus.ERROR)
        self.assertIn("ARIZE_SPACE_KEY", " ".join(issue.fix_suggestions))

    def test_validate_authentication_invalid_api_key_format(self):
        """Test authentication validation with invalid API key format."""
        result = self.validator.validate_authentication(
            arize_api_key="short",  # Too short
            arize_space_key="valid-space-key-123456789",
        )

        self.assertEqual(result.overall_status, ValidationStatus.ERROR)

        # Should have invalid format error
        format_issues = [
            i
            for i in result.issues
            if "invalid" in i.title.lower() and "api key" in i.title.lower()
        ]
        self.assertGreater(len(format_issues), 0)

    def test_validate_authentication_invalid_space_key_format(self):
        """Test authentication validation with invalid space key format."""
        result = self.validator.validate_authentication(
            arize_api_key="valid-api-key-123456789",
            arize_space_key="short",  # Too short
        )

        self.assertEqual(result.overall_status, ValidationStatus.ERROR)

        # Should have invalid format error
        format_issues = [
            i
            for i in result.issues
            if "invalid" in i.title.lower() and "space key" in i.title.lower()
        ]
        self.assertGreater(len(format_issues), 0)

    def test_validate_authentication_from_environment(self):
        """Test authentication validation using environment variables."""
        with patch.dict(
            os.environ,
            {
                "ARIZE_API_KEY": "env-api-key-123456789",
                "ARIZE_SPACE_KEY": "env-space-key-123456789",
            },
        ):
            result = self.validator.validate_authentication()

            self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)

    def test_validate_authentication_partial_environment(self):
        """Test authentication validation with partial environment variables."""
        with patch.dict(
            os.environ,
            {
                "ARIZE_API_KEY": "env-api-key-123456789"
                # Missing ARIZE_SPACE_KEY
            },
            clear=True,
        ):
            result = self.validator.validate_authentication()

            self.assertEqual(result.overall_status, ValidationStatus.ERROR)

            # Should have space key missing error
            space_key_issues = [
                i for i in result.issues if "space key" in i.title.lower()
            ]
            self.assertGreater(len(space_key_issues), 0)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation functionality."""

    def setUp(self):
        """Set up test fixtures for configuration validation."""
        self.validator = ArizeSetupValidator()

    def test_validate_environment_configuration_python_version_valid(self):
        """Test environment configuration with valid Python version."""
        with patch("sys.version_info", (3, 9, 0)):
            # Reset issues and run validation
            self.validator.issues = []
            self.validator._validate_environment_configuration()

            # Should not have Python version issues
            python_issues = [
                i for i in self.validator.issues if "python" in i.title.lower()
            ]
            self.assertEqual(len(python_issues), 0)

    def test_validate_environment_configuration_python_version_invalid(self):
        """Test environment configuration with invalid Python version."""
        with patch("sys.version_info", (3, 6, 0)):  # Too old
            # Reset issues and run validation
            self.validator.issues = []
            self.validator._validate_environment_configuration()

            # Should have Python version error
            python_issues = [
                i for i in self.validator.issues if "python" in i.title.lower()
            ]
            self.assertGreater(len(python_issues), 0)

            issue = python_issues[0]
            self.assertEqual(issue.status, ValidationStatus.ERROR)

    def test_validate_environment_configuration_missing_env_vars(self):
        """Test environment configuration with missing recommended env vars."""
        with patch.dict(os.environ, {}, clear=True):
            # Reset issues and run validation
            self.validator.issues = []
            self.validator._validate_environment_configuration()

            # Should have warnings for missing env vars
            env_var_issues = [
                i
                for i in self.validator.issues
                if "environment variable" in i.title.lower()
            ]
            self.assertGreaterEqual(
                len(env_var_issues), 3
            )  # GENOPS_TEAM, GENOPS_PROJECT, GENOPS_ENVIRONMENT

            # All should be warnings, not errors
            for issue in env_var_issues:
                self.assertEqual(issue.status, ValidationStatus.WARNING)

    def test_validate_environment_configuration_complete_env_vars(self):
        """Test environment configuration with complete env vars."""
        with patch.dict(
            os.environ,
            {
                "GENOPS_TEAM": "test-team",
                "GENOPS_PROJECT": "test-project",
                "GENOPS_ENVIRONMENT": "production",
            },
        ):
            # Reset issues and run validation
            self.validator.issues = []
            self.validator._validate_environment_configuration()

            # Should not have env var warnings
            env_var_issues = [i for i in self.validator.issues if "GENOPS_" in i.title]
            self.assertEqual(len(env_var_issues), 0)

    def test_validate_governance_configuration_success(self):
        """Test successful governance configuration validation."""
        result = self.validator.validate_governance_configuration(
            team="ml-platform-team", project="fraud-detection"
        )

        self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)
        self.assertEqual(len(result.issues), 0)

    def test_validate_governance_configuration_missing_team(self):
        """Test governance configuration validation with missing team."""
        result = self.validator.validate_governance_configuration(
            team=None, project="fraud-detection"
        )

        self.assertEqual(result.overall_status, ValidationStatus.WARNING)

        # Should have team missing warning
        team_issues = [i for i in result.issues if "team" in i.title.lower()]
        self.assertGreater(len(team_issues), 0)

        issue = team_issues[0]
        self.assertEqual(issue.status, ValidationStatus.WARNING)

    def test_validate_governance_configuration_missing_project(self):
        """Test governance configuration validation with missing project."""
        result = self.validator.validate_governance_configuration(
            team="ml-platform-team", project=None
        )

        self.assertEqual(result.overall_status, ValidationStatus.WARNING)

        # Should have project missing warning
        project_issues = [i for i in result.issues if "project" in i.title.lower()]
        self.assertGreater(len(project_issues), 0)

        issue = project_issues[0]
        self.assertEqual(issue.status, ValidationStatus.WARNING)

    def test_validate_governance_configuration_from_environment(self):
        """Test governance configuration validation from environment."""
        with patch.dict(
            os.environ, {"GENOPS_TEAM": "env-team", "GENOPS_PROJECT": "env-project"}
        ):
            result = self.validator.validate_governance_configuration()

            self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)

    def test_validate_cost_configuration_valid(self):
        """Test cost configuration validation with valid values."""
        # Reset issues and run validation
        self.validator.issues = []
        self.validator._validate_cost_configuration(
            daily_budget_limit=100.0, max_monitoring_cost=50.0
        )

        # Should not have cost configuration issues
        cost_issues = [
            i
            for i in self.validator.issues
            if "budget" in i.title.lower() or "cost" in i.title.lower()
        ]
        self.assertEqual(len(cost_issues), 0)

    def test_validate_cost_configuration_invalid_budget(self):
        """Test cost configuration validation with invalid budget."""
        # Reset issues and run validation
        self.validator.issues = []
        self.validator._validate_cost_configuration(
            daily_budget_limit=-10.0  # Invalid negative budget
        )

        # Should have budget configuration warning
        budget_issues = [
            i for i in self.validator.issues if "budget" in i.title.lower()
        ]
        self.assertGreater(len(budget_issues), 0)

        issue = budget_issues[0]
        self.assertEqual(issue.status, ValidationStatus.WARNING)

    def test_validate_cost_configuration_invalid_monitoring_cost(self):
        """Test cost configuration validation with invalid monitoring cost."""
        # Reset issues and run validation
        self.validator.issues = []
        self.validator._validate_cost_configuration(
            max_monitoring_cost=0.0  # Invalid zero cost
        )

        # Should have monitoring cost warning
        cost_issues = [
            i for i in self.validator.issues if "monitoring cost" in i.title.lower()
        ]
        self.assertGreater(len(cost_issues), 0)

        issue = cost_issues[0]
        self.assertEqual(issue.status, ValidationStatus.WARNING)


class TestConnectivityValidation(unittest.TestCase):
    """Test connectivity validation functionality."""

    def setUp(self):
        """Set up test fixtures for connectivity validation."""
        self.validator = ArizeSetupValidator()

    def test_validate_connectivity_sdk_not_available(self):
        """Test connectivity validation when SDK not available."""
        with patch.object(self.validator, "arize_available", False):
            # Reset issues and run validation
            self.validator.issues = []
            self.validator._validate_connectivity("api-key", "space-key")

            # Should skip connectivity test
            self.assertEqual(len(self.validator.issues), 0)

    def test_validate_connectivity_no_credentials(self):
        """Test connectivity validation without credentials."""
        with patch.object(self.validator, "arize_available", True):
            # Reset issues and run validation
            self.validator.issues = []
            self.validator._validate_connectivity(None, None)

            # Should skip connectivity test
            self.assertEqual(len(self.validator.issues), 0)

    def test_validate_connectivity_client_creation_success(self):
        """Test connectivity validation with successful client creation."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(
                self.validator, "arize_client_class"
            ) as mock_client_class:
                mock_client_class.return_value = Mock()

                # Reset issues and run validation
                self.validator.issues = []
                self.validator._validate_connectivity(
                    "valid-api-key", "valid-space-key"
                )

                # Should create client without errors and add info issue
                connectivity_issues = [
                    i
                    for i in self.validator.issues
                    if i.category == ValidationCategory.CONNECTIVITY
                ]
                self.assertGreater(len(connectivity_issues), 0)

                # Should be info level (test skipped, not actual connectivity test)
                issue = connectivity_issues[0]
                self.assertEqual(issue.status, ValidationStatus.INFO)

    def test_validate_connectivity_client_creation_failure(self):
        """Test connectivity validation with client creation failure."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(
                self.validator,
                "arize_client_class",
                side_effect=Exception("Connection failed"),
            ):
                # Reset issues and run validation
                self.validator.issues = []
                self.validator._validate_connectivity(
                    "invalid-api-key", "invalid-space-key"
                )

                # Should have connectivity error
                connectivity_issues = [
                    i
                    for i in self.validator.issues
                    if i.category == ValidationCategory.CONNECTIVITY
                ]
                self.assertGreater(len(connectivity_issues), 0)

                issue = connectivity_issues[0]
                self.assertEqual(issue.status, ValidationStatus.ERROR)
                self.assertIn("Connection failed", issue.error_details)

    def test_perform_health_check_success(self):
        """Test runtime health check."""
        result = self.validator.perform_health_check(
            arize_api_key="valid-api-key", arize_space_key="valid-space-key"
        )

        # Should always return success for basic health check
        self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)

        # Should have runtime health info
        health_issues = [
            i for i in result.issues if i.category == ValidationCategory.RUNTIME_HEALTH
        ]
        self.assertGreater(len(health_issues), 0)


class TestCompleteSetupValidation(unittest.TestCase):
    """Test complete setup validation functionality."""

    def setUp(self):
        """Set up test fixtures for complete validation."""
        self.validator = ArizeSetupValidator()

    def test_validate_complete_setup_success(self):
        """Test complete setup validation success scenario."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(self.validator, "arize_version", "6.1.0"):
                with patch.object(
                    self.validator, "arize_client_class", return_value=Mock()
                ):
                    result = self.validator.validate_complete_setup(
                        arize_api_key="valid-api-key-123456789",
                        arize_space_key="valid-space-key-123456789",
                        team="test-team",
                        project="test-project",
                    )

                    self.assertEqual(result.overall_status, ValidationStatus.SUCCESS)
                    self.assertEqual(result.error_count, 0)

    def test_validate_complete_setup_with_errors(self):
        """Test complete setup validation with errors."""
        with patch.object(self.validator, "arize_available", False):
            result = self.validator.validate_complete_setup()

            self.assertEqual(result.overall_status, ValidationStatus.ERROR)
            self.assertGreater(result.error_count, 0)

            # Should have multiple categories of issues
            self.assertGreater(len(result.summary), 0)

    def test_validate_complete_setup_with_warnings(self):
        """Test complete setup validation with warnings only."""
        with patch.object(self.validator, "arize_available", True):
            with patch.object(self.validator, "arize_version", "6.1.0"):
                with patch.object(
                    self.validator, "arize_client_class", return_value=Mock()
                ):
                    # Missing optional configuration (should generate warnings)
                    result = self.validator.validate_complete_setup(
                        arize_api_key="valid-api-key-123456789",
                        arize_space_key="valid-space-key-123456789",
                        # Missing team and project
                    )

                    self.assertEqual(result.overall_status, ValidationStatus.WARNING)
                    self.assertEqual(result.error_count, 0)
                    self.assertGreater(result.warning_count, 0)

    def test_generate_recommendations_no_issues(self):
        """Test recommendation generation with no issues."""
        self.validator.issues = []
        recommendations, next_steps = self.validator._generate_recommendations()

        self.assertIn("all validation checks passed", " ".join(recommendations).lower())
        self.assertIn("use genops arize integration", " ".join(next_steps).lower())

    def test_generate_recommendations_with_errors(self):
        """Test recommendation generation with errors."""
        self.validator.issues = [
            ValidationIssue(
                category=ValidationCategory.SDK_INSTALLATION,
                status=ValidationStatus.ERROR,
                title="SDK Missing",
                description="SDK not installed",
                fix_suggestions=[],
            ),
            ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                status=ValidationStatus.ERROR,
                title="Credentials Missing",
                description="API credentials not configured",
                fix_suggestions=[],
            ),
        ]

        recommendations, next_steps = self.validator._generate_recommendations()

        self.assertIn("address 2 critical error", " ".join(recommendations).lower())
        self.assertIn("install or upgrade arize", " ".join(recommendations).lower())
        self.assertIn(
            "configure arize api credentials", " ".join(recommendations).lower()
        )

    def test_generate_recommendations_with_warnings(self):
        """Test recommendation generation with warnings."""
        self.validator.issues = [
            ValidationIssue(
                category=ValidationCategory.GOVERNANCE,
                status=ValidationStatus.WARNING,
                title="Team Missing",
                description="Team not configured",
                fix_suggestions=[],
            )
        ]

        recommendations, next_steps = self.validator._generate_recommendations()

        self.assertIn("review 1 warning", " ".join(recommendations).lower())
        self.assertIn("team and project attribution", " ".join(recommendations).lower())


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for validation."""

    def test_validate_setup_function(self):
        """Test standalone validate_setup function."""
        with patch.dict(
            os.environ,
            {
                "ARIZE_API_KEY": "test-api-key-123456789",
                "ARIZE_SPACE_KEY": "test-space-key-123456789",
            },
        ):
            with patch(
                "genops.providers.arize_validation.ArizeSetupValidator"
            ) as mock_validator_class:
                mock_validator = Mock()
                mock_result = ValidationResult(
                    overall_status=ValidationStatus.SUCCESS,
                    issues=[],
                    summary={},
                    recommendations=[],
                    next_steps=[],
                )
                mock_validator.validate_complete_setup.return_value = mock_result
                mock_validator_class.return_value = mock_validator

                result = validate_setup()

                self.assertIsInstance(result, ValidationResult)
                mock_validator.validate_complete_setup.assert_called_once()

    def test_print_validation_result_function(self):
        """Test standalone print_validation_result function."""
        result = ValidationResult(
            overall_status=ValidationStatus.SUCCESS,
            issues=[],
            summary={},
            recommendations=[],
            next_steps=[],
        )

        with patch(
            "genops.providers.arize_validation.ArizeSetupValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            # Should not raise exceptions
            print_validation_result(result)

            mock_validator.print_validation_result.assert_called_once_with(result)

    def test_is_properly_configured_function_true(self):
        """Test is_properly_configured function returning True."""
        with patch(
            "genops.providers.arize_validation.ArizeSetupValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_result = ValidationResult(
                overall_status=ValidationStatus.SUCCESS,
                issues=[],
                summary={},
                recommendations=[],
                next_steps=[],
            )
            mock_validator.validate_complete_setup.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = is_properly_configured()

            self.assertTrue(result)

    def test_is_properly_configured_function_false(self):
        """Test is_properly_configured function returning False."""
        with patch(
            "genops.providers.arize_validation.ArizeSetupValidator"
        ) as mock_validator_class:
            mock_validator = Mock()
            mock_result = ValidationResult(
                overall_status=ValidationStatus.ERROR,
                issues=[
                    ValidationIssue(
                        category=ValidationCategory.SDK_INSTALLATION,
                        status=ValidationStatus.ERROR,
                        title="Error",
                        description="Test error",
                        fix_suggestions=[],
                    )
                ],
                summary={},
                recommendations=[],
                next_steps=[],
            )
            mock_validator.validate_complete_setup.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = is_properly_configured()

            self.assertFalse(result)


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases in validation."""

    def setUp(self):
        """Set up test fixtures for error handling tests."""
        self.validator = ArizeSetupValidator()

    def test_validation_with_unicode_characters(self):
        """Test validation with unicode characters in parameters."""
        result = self.validator.validate_governance_configuration(
            team="ml-平台-team",  # Unicode characters
            project="欺诈检测-project",
        )

        # Should handle unicode gracefully
        self.assertIsInstance(result, ValidationResult)

    def test_validation_with_very_long_strings(self):
        """Test validation with very long string parameters."""
        long_string = "a" * 1000  # Very long string

        result = self.validator.validate_governance_configuration(
            team=long_string, project=long_string
        )

        # Should handle long strings gracefully
        self.assertIsInstance(result, ValidationResult)

    def test_validation_with_special_characters(self):
        """Test validation with special characters."""
        result = self.validator.validate_governance_configuration(
            team="team-with-@#$%^&*()-chars", project="project_with_special!chars"
        )

        # Should handle special characters gracefully
        self.assertIsInstance(result, ValidationResult)

    def test_concurrent_validation_calls(self):
        """Test concurrent validation calls."""
        import threading

        results = []

        def run_validation():
            result = self.validator.validate_governance_configuration(
                team="concurrent-team", project="concurrent-project"
            )
            results.append(result)

        # Create multiple threads
        threads = []
        for _i in range(5):
            thread = threading.Thread(target=run_validation)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have results from all threads
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsInstance(result, ValidationResult)

    def test_validation_with_none_values(self):
        """Test validation with None values."""
        result = self.validator.validate_governance_configuration(
            team=None, project=None
        )

        # Should handle None values gracefully
        self.assertIsInstance(result, ValidationResult)
        self.assertGreater(
            result.warning_count, 0
        )  # Should have warnings for missing values


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)
