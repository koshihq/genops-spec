#!/usr/bin/env python3
"""
Test suite for GenOps Gemini validation.

This module tests the validation functionality including:
- Setup validation and diagnostics
- API connectivity testing  
- Error handling and recommendations
- Performance testing
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from genops.providers.gemini_validation import (
    GeminiValidationResult,
    ValidationCheck,
    ValidationLevel,
    print_validation_result,
    quick_validate,
    validate_gemini_quick,
    validate_gemini_setup,
)


class TestValidationLevel:
    """Test ValidationLevel enum."""

    def test_validation_levels(self):
        """Test validation level enum values."""
        assert ValidationLevel.SUCCESS.value == "success"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.ERROR.value == "error"
        assert ValidationLevel.CRITICAL.value == "critical"


class TestValidationCheck:
    """Test ValidationCheck data class."""

    def test_validation_check_creation(self):
        """Test creating a validation check."""
        check = ValidationCheck(
            name="test_check",
            level=ValidationLevel.SUCCESS,
            message="Test message",
            details="Test details",
            fix_suggestion="Fix this",
            documentation_link="https://example.com"
        )

        assert check.name == "test_check"
        assert check.level == ValidationLevel.SUCCESS
        assert check.message == "Test message"
        assert check.details == "Test details"
        assert check.fix_suggestion == "Fix this"
        assert check.documentation_link == "https://example.com"

    def test_validation_check_minimal(self):
        """Test creating validation check with minimal fields."""
        check = ValidationCheck(
            name="minimal_check",
            level=ValidationLevel.ERROR,
            message="Error message"
        )

        assert check.name == "minimal_check"
        assert check.level == ValidationLevel.ERROR
        assert check.message == "Error message"
        assert check.details is None
        assert check.fix_suggestion is None
        assert check.documentation_link is None


class TestGeminiValidationResult:
    """Test GeminiValidationResult functionality."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        checks = [
            ValidationCheck("check1", ValidationLevel.SUCCESS, "Success"),
            ValidationCheck("check2", ValidationLevel.WARNING, "Warning")
        ]

        result = GeminiValidationResult(
            success=True,
            checks=checks,
            errors=["Error 1"],
            warnings=["Warning 1"],
            recommendations=["Recommendation 1"],
            performance_metrics={"latency": 800},
            environment_info={"api_key_set": True}
        )

        assert result.success is True
        assert len(result.checks) == 2
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.recommendations) == 1
        assert result.performance_metrics["latency"] == 800
        assert result.environment_info["api_key_set"] is True

    def test_has_errors(self):
        """Test has_errors method."""
        # Test with errors list
        result_with_errors = GeminiValidationResult(
            success=False,
            errors=["Error 1"]
        )
        assert result_with_errors.has_errors() is True

        # Test with error check
        result_with_error_check = GeminiValidationResult(
            success=False,
            checks=[ValidationCheck("test", ValidationLevel.ERROR, "Error")]
        )
        assert result_with_error_check.has_errors() is True

        # Test without errors
        result_without_errors = GeminiValidationResult(
            success=True,
            checks=[ValidationCheck("test", ValidationLevel.SUCCESS, "Success")]
        )
        assert result_without_errors.has_errors() is False

    def test_has_warnings(self):
        """Test has_warnings method."""
        # Test with warnings list
        result_with_warnings = GeminiValidationResult(
            success=True,
            warnings=["Warning 1"]
        )
        assert result_with_warnings.has_warnings() is True

        # Test with warning check
        result_with_warning_check = GeminiValidationResult(
            success=True,
            checks=[ValidationCheck("test", ValidationLevel.WARNING, "Warning")]
        )
        assert result_with_warning_check.has_warnings() is True

        # Test without warnings
        result_without_warnings = GeminiValidationResult(
            success=True,
            checks=[ValidationCheck("test", ValidationLevel.SUCCESS, "Success")]
        )
        assert result_without_warnings.has_warnings() is False

    def test_get_error_count(self):
        """Test error count calculation."""
        result = GeminiValidationResult(
            success=False,
            checks=[
                ValidationCheck("check1", ValidationLevel.ERROR, "Error 1"),
                ValidationCheck("check2", ValidationLevel.SUCCESS, "Success"),
                ValidationCheck("check3", ValidationLevel.ERROR, "Error 2")
            ],
            errors=["Direct error"]
        )

        # Should count both direct errors and error checks
        assert result.get_error_count() == 3  # 1 direct + 2 check errors

    def test_get_warning_count(self):
        """Test warning count calculation."""
        result = GeminiValidationResult(
            success=True,
            checks=[
                ValidationCheck("check1", ValidationLevel.WARNING, "Warning 1"),
                ValidationCheck("check2", ValidationLevel.SUCCESS, "Success"),
                ValidationCheck("check3", ValidationLevel.WARNING, "Warning 2")
            ],
            warnings=["Direct warning"]
        )

        # Should count both direct warnings and warning checks
        assert result.get_warning_count() == 3  # 1 direct + 2 check warnings


class TestValidateGeminiSetup:
    """Test main validation function."""

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key_123"})
    def test_validation_all_success(self):
        """Test validation with all checks passing."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = validate_gemini_setup(test_connectivity=True)

            assert result.success is True
            assert result.get_error_count() == 0
            assert result.environment_info["gemini_sdk_available"] is True
            assert result.environment_info["genops_available"] is True
            assert result.environment_info["api_key_env_set"] is True

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', False)
    def test_validation_missing_gemini_sdk(self):
        """Test validation when Gemini SDK is missing."""
        result = validate_gemini_setup()

        assert result.success is False
        assert result.get_error_count() > 0
        assert any("Google Gemini SDK not installed" in error for error in result.errors)
        assert result.environment_info["gemini_sdk_available"] is False

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', False)
    def test_validation_missing_genops_core(self):
        """Test validation when GenOps core is missing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True):
            result = validate_gemini_setup()

            assert result.get_warning_count() > 0
            assert any("GenOps core not available" in warning for warning in result.warnings)
            assert result.environment_info["genops_available"] is False

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {}, clear=True)
    def test_validation_missing_api_key(self):
        """Test validation when API key is missing."""
        result = validate_gemini_setup()

        assert result.success is False
        assert result.get_error_count() > 0
        assert any("API key not configured" in error for error in result.errors)
        assert result.environment_info["api_key_env_set"] is False

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    def test_validation_with_explicit_api_key(self):
        """Test validation with explicitly provided API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Hello"
                mock_client.models.generate_content.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = validate_gemini_setup(api_key="explicit_key_123", test_connectivity=True)

                # Should pass even without environment variable
                assert result.success is True
                mock_client_class.assert_called_with(api_key="explicit_key_123")

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "invalid_key_format"})
    def test_validation_invalid_api_key_format(self):
        """Test validation with invalid API key format."""
        result = validate_gemini_setup(test_connectivity=False)

        # Should warn about unusual API key format
        assert result.get_warning_count() > 0
        assert any("format appears unusual" in check.message for check in result.checks
                  if check.level == ValidationLevel.WARNING)

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDVWsKuP8_correct_format_example"})
    def test_validation_correct_api_key_format(self):
        """Test validation with correct API key format."""
        result = validate_gemini_setup(test_connectivity=False)

        # Should pass format validation
        format_checks = [check for check in result.checks if check.name == "api_key_format"]
        assert len(format_checks) > 0
        assert format_checks[0].level == ValidationLevel.SUCCESS

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_validation_connectivity_success(self):
        """Test successful connectivity validation."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello response"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = validate_gemini_setup(test_connectivity=True)

            # Should have successful connectivity check
            connectivity_checks = [check for check in result.checks if check.name == "api_connectivity"]
            assert len(connectivity_checks) > 0
            assert connectivity_checks[0].level == ValidationLevel.SUCCESS

            # Should have performance metrics
            assert "connectivity_latency_ms" in result.performance_metrics
            assert result.performance_metrics["connectivity_latency_ms"] >= 0

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_validation_connectivity_auth_error(self):
        """Test connectivity validation with authentication error."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("API_KEY authentication failed")
            mock_client_class.return_value = mock_client

            result = validate_gemini_setup(test_connectivity=True)

            # Should have authentication error
            assert result.get_error_count() > 0
            assert any("API key authentication failed" in error for error in result.errors)

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_validation_connectivity_quota_error(self):
        """Test connectivity validation with quota error."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("quota exceeded")
            mock_client_class.return_value = mock_client

            result = validate_gemini_setup(test_connectivity=True)

            # Should have quota warning (not error)
            assert result.get_warning_count() > 0
            assert any("quota" in warning for warning in result.warnings)

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_validation_model_access_testing(self):
        """Test model access validation."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()

            # Mock successful responses for some models, failures for others
            def mock_generate_content(model, contents):
                if "flash" in model:
                    mock_response = MagicMock()
                    mock_response.text = "Response"
                    return mock_response
                else:
                    raise Exception("Model not accessible")

            mock_client.models.generate_content.side_effect = mock_generate_content
            mock_client_class.return_value = mock_client

            result = validate_gemini_setup(test_model_access=True, test_connectivity=False)

            # Should have accessible models in performance metrics
            assert "accessible_models" in result.performance_metrics
            accessible_models = result.performance_metrics["accessible_models"]
            assert any("flash" in model for model in accessible_models)

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_validation_performance_testing(self):
        """Test performance validation."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Response text for testing"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = validate_gemini_setup(
                test_connectivity=False,
                test_model_access=False,
                performance_test=True
            )

            # Should have performance metrics
            assert len([k for k in result.performance_metrics.keys() if "latency_ms" in k]) > 0
            assert len([k for k in result.performance_metrics.keys() if "tokens" in k]) > 0

    def test_validation_minimal_parameters(self):
        """Test validation with minimal parameters."""
        with patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', True):
                with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
                    result = validate_gemini_setup(
                        test_connectivity=False,
                        test_model_access=False,
                        performance_test=False
                    )

                    # Should still perform basic checks
                    assert len(result.checks) > 0
                    assert result.environment_info["gemini_sdk_available"] is True

    def test_validation_generates_recommendations(self):
        """Test that validation generates helpful recommendations."""
        with patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini_validation.GENOPS_AVAILABLE', False):
                with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
                    result = validate_gemini_setup()

                    # Should have recommendations
                    assert len(result.recommendations) > 0

                    # Should recommend GenOps core installation
                    assert any("GenOps core" in rec for rec in result.recommendations)


class TestValidateGeminiQuick:
    """Test quick validation function."""

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_quick_validation_success(self):
        """Test successful quick validation."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = validate_gemini_quick()

            assert result is True

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', False)
    def test_quick_validation_no_sdk(self):
        """Test quick validation when SDK is not available."""
        result = validate_gemini_quick()

        assert result is False

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch.dict(os.environ, {}, clear=True)
    def test_quick_validation_no_api_key(self):
        """Test quick validation when API key is missing."""
        result = validate_gemini_quick()

        assert result is False

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_quick_validation_api_error(self):
        """Test quick validation with API error."""
        with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client

            result = validate_gemini_quick()

            assert result is False

    @patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', True)
    def test_quick_validation_with_explicit_key(self):
        """Test quick validation with explicit API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genops.providers.gemini_validation.genai.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Hello"
                mock_client.models.generate_content.return_value = mock_response
                mock_client_class.return_value = mock_client

                result = validate_gemini_quick(api_key="explicit_key")

                assert result is True
                mock_client_class.assert_called_with(api_key="explicit_key")


class TestPrintValidationResult:
    """Test print validation result function."""

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation result."""
        result = GeminiValidationResult(
            success=True,
            checks=[
                ValidationCheck("check1", ValidationLevel.SUCCESS, "Success message")
            ],
            recommendations=["Recommendation 1"]
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "OVERALL STATUS: PASSED" in captured.out
        assert "Success message" in captured.out
        assert "Recommendation 1" in captured.out

    def test_print_validation_result_failure(self, capsys):
        """Test printing failed validation result."""
        result = GeminiValidationResult(
            success=False,
            checks=[
                ValidationCheck("check1", ValidationLevel.ERROR, "Error message", fix_suggestion="Fix this")
            ],
            errors=["Direct error"],
            warnings=["Warning message"]
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "OVERALL STATUS: FAILED" in captured.out
        assert "Error message" in captured.out
        assert "Fix this" in captured.out
        assert "Warning message" in captured.out

    def test_print_validation_result_detailed(self, capsys):
        """Test printing detailed validation result."""
        result = GeminiValidationResult(
            success=True,
            checks=[
                ValidationCheck(
                    "check1",
                    ValidationLevel.SUCCESS,
                    "Success",
                    details="Detailed info",
                    documentation_link="https://example.com"
                )
            ],
            performance_metrics={"latency": 800, "models": ["gemini-2.5-flash"]}
        )

        print_validation_result(result, detailed=True)

        captured = capsys.readouterr()
        assert "Detailed info" in captured.out
        assert "https://example.com" in captured.out
        assert "PERFORMANCE METRICS" in captured.out
        assert "latency: 800" in captured.out

    def test_print_validation_result_with_quick_fixes(self, capsys):
        """Test printing validation result with quick fixes."""
        with patch('genops.providers.gemini_validation.GEMINI_AVAILABLE', False):
            with patch.dict(os.environ, {}, clear=True):
                result = GeminiValidationResult(
                    success=False,
                    errors=["SDK not available", "API key missing"]
                )

                print_validation_result(result)

                captured = capsys.readouterr()
                assert "QUICK FIXES" in captured.out
                assert "pip install google-generativeai" in captured.out
                assert "export GEMINI_API_KEY" in captured.out


class TestQuickValidate:
    """Test quick_validate function."""

    @patch('genops.providers.gemini_validation.validate_gemini_quick')
    def test_quick_validate_success(self, mock_quick_validate, capsys):
        """Test quick_validate with successful validation."""
        mock_quick_validate.return_value = True

        quick_validate()

        captured = capsys.readouterr()
        assert "✅ Gemini setup appears to be working correctly!" in captured.out

    @patch('genops.providers.gemini_validation.validate_gemini_quick')
    def test_quick_validate_failure(self, mock_quick_validate, capsys):
        """Test quick_validate with failed validation."""
        mock_quick_validate.return_value = False

        quick_validate()

        captured = capsys.readouterr()
        assert "❌ Gemini setup validation failed" in captured.out
        assert "Run detailed validation" in captured.out


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
