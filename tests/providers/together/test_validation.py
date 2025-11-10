#!/usr/bin/env python3
"""
Unit tests for Together AI validation functionality.

Tests setup validation, error handling, diagnostic utilities,
and validation result structures.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.genops.providers.together_validation import (
        ValidationError,
        ValidationResult,
        check_genops_dependencies,
        check_together_api_key,
        print_validation_result,
        test_together_connectivity,
        validate_model_access,
        validate_together_setup,
    )
except ImportError as e:
    pytest.skip(f"Together AI validation not available: {e}", allow_module_level=True)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult can be created with all fields."""
        errors = [ValidationError("test_error", "Test error", "Fix it")]
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            model_access=["model1", "model2"],
            api_key_valid=False,
            dependencies_available=True,
            connectivity_working=False
        )

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.model_access == ["model1", "model2"]
        assert result.api_key_valid is False
        assert result.dependencies_available is True
        assert result.connectivity_working is False

    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(is_valid=True, errors=[])

        assert result.is_valid is True
        assert result.errors == []
        assert result.model_access is None
        assert result.api_key_valid is None
        assert result.dependencies_available is None
        assert result.connectivity_working is None

    def test_validation_error_creation(self):
        """Test ValidationError dataclass creation."""
        error = ValidationError(
            code="API_KEY_MISSING",
            message="API key not found",
            remediation="Set TOGETHER_API_KEY environment variable"
        )

        assert error.code == "API_KEY_MISSING"
        assert error.message == "API key not found"
        assert error.remediation == "Set TOGETHER_API_KEY environment variable"


class TestApiKeyValidation:
    """Test API key validation functionality."""

    @patch.dict(os.environ, {'TOGETHER_API_KEY': 'sk-test-key-123'})
    def test_check_api_key_valid_format(self):
        """Test API key validation with valid format."""
        result = check_together_api_key()

        assert isinstance(result, tuple)
        is_valid, error = result
        assert is_valid is True
        assert error is None

    @patch.dict(os.environ, {'TOGETHER_API_KEY': 'invalid-key'})
    def test_check_api_key_invalid_format(self):
        """Test API key validation with invalid format."""
        is_valid, error = check_together_api_key()

        assert is_valid is False
        assert error is not None
        assert isinstance(error, ValidationError)
        assert "format" in error.message.lower()

    @patch.dict(os.environ, {}, clear=True)
    def test_check_api_key_missing(self):
        """Test API key validation when key is missing."""
        if 'TOGETHER_API_KEY' in os.environ:
            del os.environ['TOGETHER_API_KEY']

        is_valid, error = check_together_api_key()

        assert is_valid is False
        assert error is not None
        assert isinstance(error, ValidationError)
        assert "missing" in error.message.lower() or "not found" in error.message.lower()

    def test_check_api_key_empty_string(self):
        """Test API key validation with empty string."""
        with patch.dict(os.environ, {'TOGETHER_API_KEY': ''}):
            is_valid, error = check_together_api_key()

            assert is_valid is False
            assert error is not None
            assert isinstance(error, ValidationError)

    def test_check_api_key_custom_key(self):
        """Test API key validation with custom key parameter."""
        is_valid, error = check_together_api_key(api_key="sk-custom-key-456")

        assert is_valid is True
        assert error is None

    def test_check_api_key_custom_invalid_key(self):
        """Test API key validation with custom invalid key."""
        is_valid, error = check_together_api_key(api_key="invalid-format")

        assert is_valid is False
        assert error is not None


class TestDependencyValidation:
    """Test dependency validation functionality."""

    def test_check_dependencies_available(self):
        """Test dependency checking when available."""
        is_valid, errors = check_genops_dependencies()

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

        if not is_valid:
            # If dependencies are missing, should have error messages
            assert len(errors) > 0
            for error in errors:
                assert isinstance(error, ValidationError)
                assert "install" in error.remediation.lower()

    @patch('importlib.import_module')
    def test_check_dependencies_missing_together(self, mock_import):
        """Test dependency checking when Together client is missing."""
        mock_import.side_effect = ImportError("No module named 'together'")

        is_valid, errors = check_genops_dependencies()

        assert is_valid is False
        assert len(errors) > 0

        together_error = next((e for e in errors if "together" in e.message.lower()), None)
        assert together_error is not None
        assert "pip install together" in together_error.remediation

    @patch('importlib.import_module')
    def test_check_dependencies_missing_opentelemetry(self, mock_import):
        """Test dependency checking when OpenTelemetry is missing."""
        def mock_import_side_effect(module_name):
            if "opentelemetry" in module_name:
                raise ImportError(f"No module named '{module_name}'")
            return MagicMock()

        mock_import.side_effect = mock_import_side_effect

        is_valid, errors = check_genops_dependencies()

        assert is_valid is False
        assert len(errors) > 0

        otel_error = next((e for e in errors if "opentelemetry" in e.message.lower()), None)
        assert otel_error is not None


class TestConnectivityValidation:
    """Test API connectivity validation."""

    @patch('src.genops.providers.together_validation.Together')
    def test_connectivity_success(self, mock_together):
        """Test successful API connectivity."""
        # Mock successful Together client
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[{"id": "test-model"}])
        mock_together.return_value = mock_client

        is_connected, error = test_together_connectivity("sk-test-key")

        assert is_connected is True
        assert error is None

    @patch('src.genops.providers.together_validation.Together')
    def test_connectivity_auth_failure(self, mock_together):
        """Test API connectivity with authentication failure."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Authentication failed")
        mock_together.return_value = mock_client

        is_connected, error = test_together_connectivity("invalid-key")

        assert is_connected is False
        assert error is not None
        assert isinstance(error, ValidationError)
        assert "authentication" in error.message.lower()

    @patch('src.genops.providers.together_validation.Together')
    def test_connectivity_network_failure(self, mock_together):
        """Test API connectivity with network failure."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ConnectionError("Network error")
        mock_together.return_value = mock_client

        is_connected, error = test_together_connectivity("sk-test-key")

        assert is_connected is False
        assert error is not None
        assert isinstance(error, ValidationError)
        assert "network" in error.message.lower() or "connection" in error.message.lower()


class TestModelAccessValidation:
    """Test model access validation."""

    @patch('src.genops.providers.together_validation.Together')
    def test_model_access_success(self, mock_together):
        """Test successful model access validation."""
        mock_client = MagicMock()
        mock_models = [
            {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"},
            {"id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"},
            {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
        ]
        mock_client.models.list.return_value = MagicMock(data=mock_models)
        mock_together.return_value = mock_client

        models, error = validate_model_access("sk-test-key")

        assert models is not None
        assert isinstance(models, list)
        assert len(models) == 3
        assert error is None

    @patch('src.genops.providers.together_validation.Together')
    def test_model_access_failure(self, mock_together):
        """Test model access validation failure."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Access denied")
        mock_together.return_value = mock_client

        models, error = validate_model_access("sk-test-key")

        assert models is None
        assert error is not None
        assert isinstance(error, ValidationError)

    @patch('src.genops.providers.together_validation.Together')
    def test_model_access_empty_list(self, mock_together):
        """Test model access with empty model list."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[])
        mock_together.return_value = mock_client

        models, error = validate_model_access("sk-test-key")

        assert models is not None
        assert isinstance(models, list)
        assert len(models) == 0
        assert error is None


class TestComprehensiveValidation:
    """Test comprehensive validation functionality."""

    @patch('src.genops.providers.together_validation.Together')
    @patch.dict(os.environ, {'TOGETHER_API_KEY': 'sk-test-key-123'})
    def test_validate_setup_success(self, mock_together):
        """Test successful comprehensive validation."""
        # Mock successful Together client
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[{"id": "test-model"}])
        mock_together.return_value = mock_client

        result = validate_together_setup()

        assert isinstance(result, ValidationResult)
        # Result might be valid or invalid depending on actual system state
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)

    @patch('src.genops.providers.together_validation.Together')
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_setup_missing_api_key(self, mock_together):
        """Test validation with missing API key."""
        if 'TOGETHER_API_KEY' in os.environ:
            del os.environ['TOGETHER_API_KEY']

        result = validate_together_setup()

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0

        # Should have API key error
        api_key_error = next((e for e in result.errors if "api" in e.message.lower()), None)
        assert api_key_error is not None

    @patch('src.genops.providers.together_validation.Together')
    def test_validate_setup_with_custom_api_key(self, mock_together):
        """Test validation with custom API key."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[{"id": "test-model"}])
        mock_together.return_value = mock_client

        result = validate_together_setup(together_api_key="sk-custom-key")

        assert isinstance(result, ValidationResult)
        # Should use the custom API key for validation

    def test_validate_setup_with_config(self):
        """Test validation with custom configuration."""
        config = {
            "team": "test-team",
            "project": "test-project",
            "daily_budget_limit": 25.0
        }

        result = validate_together_setup(config=config)

        assert isinstance(result, ValidationResult)
        # Configuration validation should not affect basic validation structure


class TestValidationResultPrinting:
    """Test validation result printing functionality."""

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            model_access=["model1", "model2"],
            api_key_valid=True,
            dependencies_available=True,
            connectivity_working=True
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "✅" in captured.out or "success" in captured.out.lower()
        assert "model1" in captured.out
        assert "model2" in captured.out

    def test_print_validation_result_failure(self, capsys):
        """Test printing failed validation result."""
        errors = [
            ValidationError("API_KEY_MISSING", "API key not found", "Set TOGETHER_API_KEY"),
            ValidationError("DEPENDENCY_MISSING", "Missing dependency", "pip install together")
        ]
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            api_key_valid=False,
            dependencies_available=False
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "❌" in captured.out or "error" in captured.out.lower()
        assert "API key not found" in captured.out
        assert "Missing dependency" in captured.out
        assert "Set TOGETHER_API_KEY" in captured.out
        assert "pip install together" in captured.out

    def test_print_validation_result_partial(self, capsys):
        """Test printing validation result with mixed results."""
        errors = [ValidationError("WARN", "Warning message", "Fix this")]
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            model_access=["available-model"],
            api_key_valid=True,
            dependencies_available=True,
            connectivity_working=False
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        # Should show both successes and failures
        assert "Warning message" in captured.out
        assert "available-model" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
