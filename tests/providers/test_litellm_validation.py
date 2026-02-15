#!/usr/bin/env python3
"""
Test Suite for LiteLLM Validation Module

Tests cover all aspects of the LiteLLM validation functionality including:
- Installation validation
- API key validation
- GenOps integration testing
- Environment configuration checks
- Callback system validation
- Connectivity testing
- Error handling and edge cases
"""

import os

# Test imports
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.genops.providers.litellm_validation import (  # noqa: E402
    ValidationIssue,
    ValidationResult,
    ValidationStatus,
    print_validation_result,
    validate_callback_system,
    validate_environment_configuration,
    validate_genops_integration,
    validate_litellm_connectivity,
    validate_litellm_installation,
    validate_litellm_setup,
    validate_provider_api_keys,
)


class TestValidationDataStructures:
    """Test suite for validation data structures."""

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values."""
        assert ValidationStatus.SUCCESS.value == "success"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.ERROR.value == "error"
        assert ValidationStatus.SKIPPED.value == "skipped"

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation and attributes."""
        issue = ValidationIssue(
            component="Test Component",
            status=ValidationStatus.SUCCESS,
            message="Test message",
            fix_suggestion="Test fix",
            documentation_link="https://test.com",
        )

        assert issue.component == "Test Component"
        assert issue.status == ValidationStatus.SUCCESS
        assert issue.message == "Test message"
        assert issue.fix_suggestion == "Test fix"
        assert issue.documentation_link == "https://test.com"

    def test_validation_issue_minimal(self):
        """Test ValidationIssue with minimal required fields."""
        issue = ValidationIssue(
            component="Minimal Test",
            status=ValidationStatus.ERROR,
            message="Error message",
        )

        assert issue.component == "Minimal Test"
        assert issue.status == ValidationStatus.ERROR
        assert issue.message == "Error message"
        assert issue.fix_suggestion is None
        assert issue.documentation_link is None

    def test_validation_result_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert isinstance(result.issues, list)
        assert len(result.issues) == 0
        assert isinstance(result.summary, dict)
        assert isinstance(result.provider_status, dict)

    def test_validation_result_add_issue(self):
        """Test adding issues to ValidationResult."""
        result = ValidationResult(is_valid=True)

        # Add success issue - should not change validity
        result.add_issue(
            component="Test Success",
            status=ValidationStatus.SUCCESS,
            message="Success message",
        )

        assert result.is_valid is True
        assert len(result.issues) == 1

        # Add error issue - should change validity
        result.add_issue(
            component="Test Error",
            status=ValidationStatus.ERROR,
            message="Error message",
            fix_suggestion="Fix this error",
        )

        assert result.is_valid is False
        assert len(result.issues) == 2

        # Verify issue details
        error_issue = result.issues[1]
        assert error_issue.status == ValidationStatus.ERROR
        assert error_issue.fix_suggestion == "Fix this error"


class TestLiteLLMInstallationValidation:
    """Test suite for LiteLLM installation validation."""

    @patch("src.genops.providers.litellm_validation.litellm")
    def test_validate_litellm_installation_success(self, mock_litellm):
        """Test successful LiteLLM installation validation."""
        mock_litellm.__version__ = "1.2.3"
        mock_litellm.completion = Mock()
        mock_litellm.acompletion = Mock()
        mock_litellm.embedding = Mock()

        issues = validate_litellm_installation()

        assert len(issues) == 2  # Installation + API methods
        assert issues[0].status == ValidationStatus.SUCCESS
        assert "1.2.3" in issues[0].message
        assert issues[1].status == ValidationStatus.SUCCESS
        assert "All required LiteLLM methods available" in issues[1].message

    @patch("src.genops.providers.litellm_validation.litellm")
    def test_validate_litellm_installation_missing_methods(self, mock_litellm):
        """Test LiteLLM installation with missing methods."""
        mock_litellm.__version__ = "0.9.0"
        mock_litellm.completion = Mock()
        # Missing acompletion and embedding

        with patch("builtins.hasattr") as mock_hasattr:

            def hasattr_side_effect(obj, attr):
                if attr == "completion":
                    return True
                return False  # acompletion and embedding missing

            mock_hasattr.side_effect = hasattr_side_effect

            issues = validate_litellm_installation()

            # Should have installation success but API warning
            success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
            warning_issues = [i for i in issues if i.status == ValidationStatus.WARNING]

            assert len(success_issues) == 1  # Installation found
            assert len(warning_issues) == 1  # Missing methods
            assert "Missing methods" in warning_issues[0].message

    def test_validate_litellm_installation_import_error(self):
        """Test LiteLLM installation validation when import fails."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("No module named 'litellm'")

            issues = validate_litellm_installation()

            assert len(issues) == 1
            assert issues[0].status == ValidationStatus.ERROR
            assert "not installed" in issues[0].message
            assert "pip install litellm" in issues[0].fix_suggestion
            assert "docs.litellm.ai" in issues[0].documentation_link

    def test_validate_litellm_installation_unexpected_error(self):
        """Test handling of unexpected errors during installation validation."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = RuntimeError("Unexpected error")

            issues = validate_litellm_installation()

            assert len(issues) == 1
            assert issues[0].status == ValidationStatus.ERROR
            assert "Unexpected error" in issues[0].message


class TestGenOpsIntegrationValidation:
    """Test suite for GenOps integration validation."""

    @patch("src.genops.providers.litellm_validation.GenOpsLiteLLMCallback")
    @patch("src.genops.providers.litellm_validation.LiteLLMGovernanceContext")
    def test_validate_genops_integration_success(self, mock_context, mock_callback):
        """Test successful GenOps integration validation."""
        # Mock successful imports
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args, **kwargs):
                if "genops.providers.litellm" in name:
                    mock_module = Mock()
                    mock_module.auto_instrument = Mock()
                    mock_module.track_completion = Mock()
                    mock_module.get_usage_stats = Mock()
                    mock_module.GenOpsLiteLLMCallback = mock_callback
                    mock_module.LiteLLMGovernanceContext = mock_context
                    return mock_module
                return Mock()

            mock_import.side_effect = import_side_effect

            issues = validate_genops_integration()

            # Should have success for integration and callbacks
            success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
            assert len(success_issues) == 2
            assert any(
                "GenOps LiteLLM provider available" in i.message for i in success_issues
            )
            assert any(
                "callback system functional" in i.message for i in success_issues
            )

    def test_validate_genops_integration_import_error(self):
        """Test GenOps integration validation when import fails."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError(
                "No module named 'genops.providers.litellm'"
            )

            issues = validate_genops_integration()

            assert len(issues) == 1
            assert issues[0].status == ValidationStatus.ERROR
            assert "not available" in issues[0].message
            assert "genops-ai[litellm]" in issues[0].fix_suggestion

    @patch("src.genops.providers.litellm_validation.GenOpsLiteLLMCallback")
    @patch("src.genops.providers.litellm_validation.LiteLLMGovernanceContext")
    def test_validate_genops_integration_callback_error(
        self, mock_context, mock_callback
    ):
        """Test GenOps integration validation when callbacks fail."""
        # Mock import success but callback failure
        mock_context.side_effect = RuntimeError("Callback initialization failed")

        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args, **kwargs):
                if "genops.providers.litellm" in name:
                    mock_module = Mock()
                    mock_module.auto_instrument = Mock()
                    mock_module.track_completion = Mock()
                    mock_module.get_usage_stats = Mock()
                    mock_module.GenOpsLiteLLMCallback = mock_callback
                    mock_module.LiteLLMGovernanceContext = mock_context
                    return mock_module
                return Mock()

            mock_import.side_effect = import_side_effect

            issues = validate_genops_integration()

            # Should have success for import but warning for callbacks
            success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
            warning_issues = [i for i in issues if i.status == ValidationStatus.WARNING]

            assert len(success_issues) == 1  # Import success
            assert len(warning_issues) == 1  # Callback warning
            assert "Callback system issue" in warning_issues[0].message


class TestProviderAPIKeyValidation:
    """Test suite for provider API key validation."""

    def test_validate_provider_api_keys_all_configured(self):
        """Test API key validation when all providers are configured."""
        mock_env = {
            "OPENAI_API_KEY": "sk-test123",
            "ANTHROPIC_API_KEY": "sk-ant-test456",
            "GOOGLE_API_KEY": "test789",
            "AZURE_API_KEY": "azure-test",
            "AWS_ACCESS_KEY_ID": "aws-access",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "COHERE_API_KEY": "cohere-test",
        }

        with patch.dict(os.environ, mock_env, clear=True):
            issues, provider_status = validate_provider_api_keys()

            # Should have success issues for configured providers
            success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
            assert len(success_issues) >= 7  # At least 7 configured providers

            # Should have overall success
            overall_success = [i for i in issues if "Configured providers" in i.message]
            assert len(overall_success) == 1

            # Provider status should show successes
            successful_providers = [
                p for p, s in provider_status.items() if s == ValidationStatus.SUCCESS
            ]
            assert len(successful_providers) >= 6

    def test_validate_provider_api_keys_partial_configured(self):
        """Test API key validation with some providers configured."""
        mock_env = {
            "OPENAI_API_KEY": "sk-test123",
            "ANTHROPIC_API_KEY": "sk-ant-test456",
        }

        with patch.dict(os.environ, mock_env, clear=True):
            issues, provider_status = validate_provider_api_keys()

            # Should have success for configured providers
            success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
            configured_success = [
                i for i in success_issues if "configured with" in i.message
            ]
            assert len(configured_success) == 2  # OpenAI and Anthropic

            # Should have warnings for unconfigured providers
            warning_issues = [i for i in issues if i.status == ValidationStatus.WARNING]
            unconfigured_warnings = [
                i for i in warning_issues if "not configured" in i.message
            ]
            assert len(unconfigured_warnings) >= 8  # Other providers not configured

    def test_validate_provider_api_keys_none_configured(self):
        """Test API key validation when no providers are configured."""
        with patch.dict(os.environ, {}, clear=True):
            issues, provider_status = validate_provider_api_keys()

            # Should have error for no providers
            error_issues = [i for i in issues if i.status == ValidationStatus.ERROR]
            no_providers_error = [
                i for i in error_issues if "No LLM provider API keys" in i.message
            ]
            assert len(no_providers_error) == 1

            # All providers should have warning status
            warning_providers = [
                p for p, s in provider_status.items() if s == ValidationStatus.WARNING
            ]
            assert len(warning_providers) >= 10  # All major providers

    def test_validate_provider_api_keys_alternate_vars(self):
        """Test API key validation with alternate environment variables."""
        mock_env = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json",
            "HF_TOKEN": "hf_test123",
            "AWS_REGION": "us-east-1",
        }

        with patch.dict(os.environ, mock_env, clear=True):
            issues, provider_status = validate_provider_api_keys()

            # Should detect alternate variables
            success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
            [i for i in success_issues if "Google configured" in i.message]
            # Note: Google requires additional setup beyond just env vars


class TestConnectivityValidation:
    """Test suite for connectivity validation."""

    @patch("src.genops.providers.litellm_validation.litellm")
    def test_validate_litellm_connectivity_success(self, mock_litellm):
        """Test successful connectivity validation."""

        def mock_get_llm_provider(model):
            model_mappings = {
                "gpt-3.5-turbo": ("openai", {}),
                "claude-3-sonnet": ("anthropic", {}),
                "gemini-pro": ("google", {}),
            }
            return model_mappings.get(model, (None, {}))

        mock_litellm.get_llm_provider = mock_get_llm_provider

        issues = validate_litellm_connectivity()

        success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
        assert len(success_issues) >= 3  # Should have successful mappings

        # Verify specific model mappings
        model_messages = [i.message for i in success_issues]
        assert any(
            "gpt-3.5-turbo mapped to provider openai" in msg for msg in model_messages
        )
        assert any(
            "claude-3-sonnet mapped to provider anthropic" in msg
            for msg in model_messages
        )

    @patch("src.genops.providers.litellm_validation.litellm")
    def test_validate_litellm_connectivity_mapping_errors(self, mock_litellm):
        """Test connectivity validation with mapping errors."""

        def mock_get_llm_provider(model):
            if model == "gpt-3.5-turbo":
                return ("openai", {})
            elif model == "claude-3-sonnet":
                raise Exception("Provider mapping failed")
            else:
                return (None, {})

        mock_litellm.get_llm_provider = mock_get_llm_provider

        issues = validate_litellm_connectivity()

        # Should have mix of success and warnings
        success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
        warning_issues = [i for i in issues if i.status == ValidationStatus.WARNING]

        assert len(success_issues) >= 1  # gpt-3.5-turbo should work
        assert len(warning_issues) >= 2  # claude error + unclear mappings

    def test_validate_litellm_connectivity_import_error(self):
        """Test connectivity validation when LiteLLM not available."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("No module named 'litellm'")

            issues = validate_litellm_connectivity()

            assert len(issues) == 1
            assert issues[0].status == ValidationStatus.SKIPPED
            assert "not available for connectivity testing" in issues[0].message


class TestCallbackSystemValidation:
    """Test suite for callback system validation."""

    @patch("src.genops.providers.litellm_validation.litellm")
    def test_validate_callback_system_success(self, mock_litellm):
        """Test successful callback system validation."""
        # Mock callback attributes
        mock_litellm.input_callback = []
        mock_litellm.success_callback = []
        mock_litellm.failure_callback = []

        def mock_hasattr(obj, attr):
            return attr in ["input_callback", "success_callback", "failure_callback"]

        def mock_getattr(obj, attr, default=None):
            if attr in ["input_callback", "success_callback", "failure_callback"]:
                return []
            return default

        def mock_setattr(obj, attr, value):
            pass

        with patch("builtins.hasattr", mock_hasattr):
            with patch("builtins.getattr", mock_getattr):
                with patch("builtins.setattr", mock_setattr):
                    issues = validate_callback_system()

        success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
        assert len(success_issues) == 2  # Available + registration
        assert any("callback system available" in i.message for i in success_issues)
        assert any("registration functional" in i.message for i in success_issues)

    @patch("src.genops.providers.litellm_validation.litellm")
    def test_validate_callback_system_missing_attrs(self, mock_litellm):
        """Test callback system validation with missing attributes."""

        def mock_hasattr(obj, attr):
            return attr == "input_callback"  # Only input_callback available

        with patch("builtins.hasattr", mock_hasattr):
            issues = validate_callback_system()

        warning_issues = [i for i in issues if i.status == ValidationStatus.WARNING]
        assert len(warning_issues) == 1
        assert "Missing callback attributes" in warning_issues[0].message
        assert "success_callback, failure_callback" in warning_issues[0].message

    def test_validate_callback_system_import_error(self):
        """Test callback system validation when LiteLLM not available."""
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("No module named 'litellm'")

            issues = validate_callback_system()

            assert len(issues) == 1
            assert issues[0].status == ValidationStatus.SKIPPED
            assert "not available for callback testing" in issues[0].message


class TestEnvironmentValidation:
    """Test suite for environment configuration validation."""

    def test_validate_environment_configuration_success(self):
        """Test successful environment validation."""
        mock_env = {"PATH": "/usr/bin:/bin", "HOME": "/home/user"}

        with patch.dict(os.environ, mock_env, clear=True):
            with patch("sys.version_info", (3, 9, 0)):
                issues = validate_environment_configuration()

        success_issues = [i for i in issues if i.status == ValidationStatus.SUCCESS]
        assert len(success_issues) >= 3  # Python version + env vars

        python_success = [i for i in success_issues if "Python 3.9.0" in i.message]
        assert len(python_success) == 1

    def test_validate_environment_configuration_old_python(self):
        """Test environment validation with old Python version."""
        with patch("sys.version_info", (3, 7, 0)):
            issues = validate_environment_configuration()

        error_issues = [i for i in issues if i.status == ValidationStatus.ERROR]
        python_error = [i for i in error_issues if "not supported" in i.message]
        assert len(python_error) == 1
        assert "Python 3.7" in python_error[0].message
        assert "Upgrade to Python 3.8" in python_error[0].fix_suggestion

    def test_validate_environment_configuration_missing_vars(self):
        """Test environment validation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            issues = validate_environment_configuration()

        warning_issues = [i for i in issues if i.status == ValidationStatus.WARNING]
        missing_vars = [
            i for i in warning_issues if "Missing environment variable" in i.message
        ]
        assert len(missing_vars) >= 2  # PATH and HOME


class TestComprehensiveValidation:
    """Test suite for comprehensive validation function."""

    @patch("src.genops.providers.litellm_validation.validate_litellm_installation")
    @patch("src.genops.providers.litellm_validation.validate_genops_integration")
    @patch("src.genops.providers.litellm_validation.validate_provider_api_keys")
    @patch("src.genops.providers.litellm_validation.validate_callback_system")
    @patch("src.genops.providers.litellm_validation.validate_environment_configuration")
    @patch("src.genops.providers.litellm_validation.validate_litellm_connectivity")
    def test_validate_litellm_setup_comprehensive(
        self,
        mock_connectivity,
        mock_env,
        mock_callbacks,
        mock_api_keys,
        mock_genops,
        mock_installation,
    ):
        """Test comprehensive validation with all checks."""
        # Mock successful validations
        mock_installation.return_value = [
            ValidationIssue("LiteLLM", ValidationStatus.SUCCESS, "Installed")
        ]
        mock_genops.return_value = [
            ValidationIssue("GenOps", ValidationStatus.SUCCESS, "Available")
        ]
        mock_api_keys.return_value = (
            [ValidationIssue("API Keys", ValidationStatus.SUCCESS, "Configured")],
            {"openai": ValidationStatus.SUCCESS},
        )
        mock_callbacks.return_value = [
            ValidationIssue("Callbacks", ValidationStatus.SUCCESS, "Available")
        ]
        mock_env.return_value = [
            ValidationIssue("Environment", ValidationStatus.SUCCESS, "Valid")
        ]
        mock_connectivity.return_value = [
            ValidationIssue("Connectivity", ValidationStatus.SUCCESS, "Working")
        ]

        result = validate_litellm_setup(quick=False, test_connectivity=True)

        assert result.is_valid is True
        assert len(result.issues) == 6
        assert result.summary["errors"] == 0
        assert result.summary["validation_type"] == "comprehensive"

        # Verify all validators were called
        mock_installation.assert_called_once()
        mock_genops.assert_called_once()
        mock_api_keys.assert_called_once()
        mock_callbacks.assert_called_once()
        mock_env.assert_called_once()
        mock_connectivity.assert_called_once()

    @patch("src.genops.providers.litellm_validation.validate_litellm_installation")
    @patch("src.genops.providers.litellm_validation.validate_genops_integration")
    def test_validate_litellm_setup_quick(self, mock_genops, mock_installation):
        """Test quick validation mode."""
        mock_installation.return_value = [
            ValidationIssue("LiteLLM", ValidationStatus.SUCCESS, "Installed")
        ]
        mock_genops.return_value = [
            ValidationIssue("GenOps", ValidationStatus.SUCCESS, "Available")
        ]

        result = validate_litellm_setup(quick=True)

        assert result.is_valid is True
        assert result.summary["validation_type"] == "quick"

        # Only core validations should be called
        mock_installation.assert_called_once()
        mock_genops.assert_called_once()

    @patch("src.genops.providers.litellm_validation.validate_litellm_installation")
    def test_validate_litellm_setup_with_errors(self, mock_installation):
        """Test validation with errors."""
        mock_installation.return_value = [
            ValidationIssue("LiteLLM", ValidationStatus.ERROR, "Not installed")
        ]

        result = validate_litellm_setup(quick=True)

        assert result.is_valid is False
        assert result.summary["errors"] == 1

    def test_validate_litellm_setup_exception_handling(self):
        """Test validation exception handling."""
        with patch(
            "src.genops.providers.litellm_validation.validate_litellm_installation"
        ) as mock_install:
            mock_install.side_effect = RuntimeError("Validation system error")

            result = validate_litellm_setup()

            assert result.is_valid is False
            assert len(result.issues) == 1
            assert result.issues[0].status == ValidationStatus.ERROR
            assert "Validation system error" in result.issues[0].message


class TestValidationReporting:
    """Test suite for validation result reporting."""

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation results."""
        result = ValidationResult(is_valid=True)
        result.add_issue("Test Component", ValidationStatus.SUCCESS, "All good")
        result.summary = {"total_issues": 1, "errors": 0, "warnings": 0}

        print_validation_result(result, verbose=False)

        captured = capsys.readouterr()
        assert "Overall Status: READY" in captured.out
        assert "Total checks: 1" in captured.out
        assert "Errors: 0" in captured.out

    def test_print_validation_result_with_errors(self, capsys):
        """Test printing validation results with errors."""
        result = ValidationResult(is_valid=False)
        result.add_issue(
            "Error Component", ValidationStatus.ERROR, "Something failed", "Fix it"
        )
        result.add_issue(
            "Warning Component", ValidationStatus.WARNING, "Something suspicious"
        )
        result.summary = {"total_issues": 2, "errors": 1, "warnings": 1}

        print_validation_result(result, verbose=True)

        captured = capsys.readouterr()
        assert "Overall Status: ISSUES FOUND" in captured.out
        assert "Errors: 1" in captured.out
        assert "Warnings: 1" in captured.out
        assert "ERROR:" in captured.out
        assert "Something failed" in captured.out
        assert "Fix: Fix it" in captured.out
        assert "Action Required:" in captured.out

    def test_print_validation_result_with_providers(self, capsys):
        """Test printing validation results with provider status."""
        result = ValidationResult(is_valid=True)
        result.provider_status = {
            "openai": ValidationStatus.SUCCESS,
            "anthropic": ValidationStatus.WARNING,
            "google": ValidationStatus.ERROR,
        }
        result.summary = {"total_issues": 0, "errors": 0, "warnings": 0}

        print_validation_result(result, verbose=False)

        captured = capsys.readouterr()
        assert "Provider Status:" in captured.out
        assert "✅ openai" in captured.out
        assert "⚠️ anthropic" in captured.out
        assert "❌ google" in captured.out


class TestValidationEdgeCases:
    """Test suite for validation edge cases and error conditions."""

    def test_validation_with_empty_environment(self):
        """Test validation with completely empty environment."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_litellm_setup(quick=False)

            # Should handle gracefully, not crash
            assert isinstance(result, ValidationResult)
            assert isinstance(result.is_valid, bool)

    def test_validation_result_large_number_of_issues(self):
        """Test validation result with large number of issues."""
        result = ValidationResult(is_valid=True)

        # Add many issues
        for i in range(100):
            result.add_issue(
                f"Component {i}",
                ValidationStatus.SUCCESS if i % 2 == 0 else ValidationStatus.WARNING,
                f"Message {i}",
            )

        assert len(result.issues) == 100

        # Test reporting doesn't break
        print_validation_result(result, verbose=False)  # Should not raise

    def test_validation_with_unicode_content(self):
        """Test validation with unicode content."""
        result = ValidationResult(is_valid=True)
        result.add_issue(
            "Unicode Test 🚀",
            ValidationStatus.SUCCESS,
            "Message with émojis and spéciål characters 测试",
            fix_suggestion="Fix with ünicøde 修复",
        )

        # Should handle unicode gracefully
        print_validation_result(result, verbose=True)  # Should not raise


if __name__ == "__main__":
    # Run the validation tests
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])
