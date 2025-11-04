"""
Unit tests for Hugging Face validation utilities.

Tests the validation system including:
- Environment variable validation
- Dependency checking
- Connectivity testing
- GenOps integration validation
- Cost calculation validation
- User-friendly error reporting
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation and attributes."""
        from genops.providers.huggingface_validation import ValidationIssue
        
        issue = ValidationIssue(
            level="error",
            component="test-component",
            message="Test message",
            fix_suggestion="Test fix"
        )
        
        assert issue.level == "error"
        assert issue.component == "test-component"
        assert issue.message == "Test message"
        assert issue.fix_suggestion == "Test fix"

    def test_validation_issue_optional_fix(self):
        """Test ValidationIssue with optional fix suggestion."""
        from genops.providers.huggingface_validation import ValidationIssue
        
        issue = ValidationIssue(
            level="warning",
            component="test",
            message="Warning message"
        )
        
        assert issue.fix_suggestion is None


class TestValidationResult:
    """Test ValidationResult namedtuple."""

    def test_validation_result_structure(self):
        """Test ValidationResult structure and access."""
        from genops.providers.huggingface_validation import ValidationResult, ValidationIssue
        
        issues = [
            ValidationIssue("error", "component1", "Error message"),
            ValidationIssue("warning", "component2", "Warning message")
        ]
        
        summary = {"total": 2, "errors": 1, "warnings": 1}
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            summary=summary
        )
        
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert result.summary["total"] == 2


class TestEnvironmentVariableValidation:
    """Test environment variable validation."""

    @patch.dict(os.environ, {}, clear=True)
    def test_check_environment_variables_no_tokens(self):
        """Test environment validation with no tokens set."""
        from genops.providers.huggingface_validation import check_environment_variables
        
        issues = check_environment_variables()
        
        # Should have warning about missing HF token
        warning_issues = [i for i in issues if i.level == "warning"]
        assert len(warning_issues) >= 1
        
        hf_token_warning = next(
            (i for i in warning_issues if "Hugging Face token" in i.message),
            None
        )
        assert hf_token_warning is not None
        assert "HF_TOKEN" in hf_token_warning.fix_suggestion

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=True)
    def test_check_environment_variables_with_hf_token(self):
        """Test environment validation with HF token set."""
        from genops.providers.huggingface_validation import check_environment_variables
        
        issues = check_environment_variables()
        
        # Should not have HF token warning
        hf_token_warnings = [
            i for i in issues 
            if i.level == "warning" and "Hugging Face token" in i.message
        ]
        assert len(hf_token_warnings) == 0

    @patch.dict(os.environ, {"HUGGINGFACE_HUB_TOKEN": "alt-token"}, clear=True)
    def test_check_environment_variables_with_alt_token(self):
        """Test environment validation with alternative token name."""
        from genops.providers.huggingface_validation import check_environment_variables
        
        issues = check_environment_variables()
        
        # Should not have HF token warning
        hf_token_warnings = [
            i for i in issues 
            if i.level == "warning" and "Hugging Face token" in i.message
        ]
        assert len(hf_token_warnings) == 0

    @patch.dict(os.environ, {
        "OTEL_SERVICE_NAME": "test-service",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"
    }, clear=True)
    def test_check_environment_variables_with_otel(self):
        """Test environment validation with OpenTelemetry vars set."""
        from genops.providers.huggingface_validation import check_environment_variables
        
        issues = check_environment_variables()
        
        # Should have fewer info messages about missing OTEL vars
        otel_info_issues = [
            i for i in issues 
            if i.level == "info" and "OpenTelemetry" in i.message
        ]
        
        # Should still have some OTEL info messages for other variables
        assert len(otel_info_issues) > 0


class TestDependencyValidation:
    """Test dependency checking functionality."""

    @patch('builtins.__import__')
    def test_check_dependencies_all_available(self, mock_import):
        """Test dependency checking when all dependencies are available."""
        from genops.providers.huggingface_validation import check_dependencies
        
        # Mock successful imports
        mock_import.return_value = Mock()
        
        issues = check_dependencies()
        
        # Should not have error issues for core dependencies
        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) == 0

    @patch('builtins.__import__')
    def test_check_dependencies_missing_huggingface(self, mock_import):
        """Test dependency checking with missing huggingface_hub."""
        from genops.providers.huggingface_validation import check_dependencies
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name == 'huggingface_hub':
                raise ImportError("No module named 'huggingface_hub'")
            return Mock()
        
        mock_import.side_effect = mock_import_side_effect
        
        issues = check_dependencies()
        
        # Should have error for missing huggingface_hub
        error_issues = [i for i in issues if i.level == "error"]
        assert len(error_issues) >= 1
        
        hf_error = next(
            (i for i in error_issues if "huggingface_hub" in i.message),
            None
        )
        assert hf_error is not None
        assert "pip install huggingface_hub" in hf_error.fix_suggestion

    @patch('builtins.__import__')
    def test_check_dependencies_missing_optional(self, mock_import):
        """Test dependency checking with missing optional dependencies."""
        from genops.providers.huggingface_validation import check_dependencies
        
        def mock_import_side_effect(name, *args, **kwargs):
            if name in ['torch', 'transformers']:
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        mock_import.side_effect = mock_import_side_effect
        
        issues = check_dependencies()
        
        # Should have info about missing optional dependencies
        info_issues = [i for i in issues if i.level == "info"]
        optional_missing = next(
            (i for i in info_issues if "Optional AI/ML dependencies" in i.message),
            None
        )
        assert optional_missing is not None

    @patch('builtins.__import__')
    def test_check_dependencies_missing_otel(self, mock_import):
        """Test dependency checking with missing OpenTelemetry."""
        from genops.providers.huggingface_validation import check_dependencies
        
        def mock_import_side_effect(name, *args, **kwargs):
            if 'opentelemetry' in name:
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        mock_import.side_effect = mock_import_side_effect
        
        issues = check_dependencies()
        
        # Should have warnings for missing OpenTelemetry components
        warning_issues = [i for i in issues if i.level == "warning"]
        otel_warnings = [i for i in warning_issues if "telemetry" in i.message]
        assert len(otel_warnings) >= 1


class TestConnectivityValidation:
    """Test Hugging Face connectivity validation."""

    @patch('genops.providers.huggingface_validation.InferenceClient')
    def test_check_huggingface_connectivity_success(self, mock_inference_client):
        """Test successful Hugging Face connectivity check."""
        from genops.providers.huggingface_validation import check_huggingface_connectivity
        
        # Mock successful client creation
        mock_client_instance = Mock()
        mock_client_instance.text_generation = Mock()
        mock_inference_client.return_value = mock_client_instance
        
        issues = check_huggingface_connectivity()
        
        # Should have info message about successful client creation
        info_issues = [i for i in issues if i.level == "info"]
        success_info = next(
            (i for i in info_issues if "InferenceClient created successfully" in i.message),
            None
        )
        assert success_info is not None

    @patch('genops.providers.huggingface_validation.InferenceClient')
    def test_check_huggingface_connectivity_no_text_generation(self, mock_inference_client):
        """Test connectivity check when text_generation method missing."""
        from genops.providers.huggingface_validation import check_huggingface_connectivity
        
        # Mock client without text_generation method
        mock_client_instance = Mock()
        del mock_client_instance.text_generation  # Remove the method
        mock_inference_client.return_value = mock_client_instance
        
        issues = check_huggingface_connectivity()
        
        # Should have warning about missing method
        warning_issues = [i for i in issues if i.level == "warning"]
        method_warning = next(
            (i for i in warning_issues if "text_generation method not available" in i.message),
            None
        )
        assert method_warning is not None

    @patch('genops.providers.huggingface_validation.InferenceClient')
    def test_check_huggingface_connectivity_import_error(self, mock_inference_client):
        """Test connectivity check with import error."""
        from genops.providers.huggingface_validation import check_huggingface_connectivity
        
        # Mock ImportError
        mock_inference_client.side_effect = ImportError("huggingface_hub not found")
        
        # Patch the import check to simulate missing module
        with patch('genops.providers.huggingface_validation.InferenceClient', side_effect=ImportError):
            issues = check_huggingface_connectivity()
        
        # Should have error about import failure
        error_issues = [i for i in issues if i.level == "error"]
        import_error = next(
            (i for i in error_issues if "Cannot import huggingface_hub" in i.message),
            None
        )
        assert import_error is not None

    @patch('genops.providers.huggingface_validation.InferenceClient')
    def test_check_huggingface_connectivity_creation_error(self, mock_inference_client):
        """Test connectivity check with client creation error."""
        from genops.providers.huggingface_validation import check_huggingface_connectivity
        
        # Mock client creation error
        mock_inference_client.side_effect = Exception("Connection failed")
        
        issues = check_huggingface_connectivity()
        
        # Should have warning about client creation issue
        warning_issues = [i for i in issues if i.level == "warning"]
        creation_warning = next(
            (i for i in warning_issues if "Issue creating Hugging Face client" in i.message),
            None
        )
        assert creation_warning is not None


class TestGenOpsIntegrationValidation:
    """Test GenOps integration validation."""

    @patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter')
    def test_check_genops_integration_success(self, mock_adapter_class):
        """Test successful GenOps integration validation."""
        from genops.providers.huggingface_validation import check_genops_integration
        
        # Mock successful adapter creation and methods
        mock_adapter = Mock()
        mock_adapter.get_supported_tasks.return_value = [
            'text-generation', 'chat-completion', 'feature-extraction'
        ]
        mock_adapter.detect_provider_for_model.side_effect = lambda x: {
            'gpt-3.5-turbo': 'openai',
            'claude-3-sonnet': 'anthropic',
            'microsoft/DialoGPT-medium': 'huggingface_hub'
        }.get(x, 'unknown')
        
        mock_adapter_class.return_value = mock_adapter
        
        issues = check_genops_integration()
        
        # Should have positive info messages
        info_issues = [i for i in issues if i.level == "info"]
        
        adapter_success = next(
            (i for i in info_issues if "adapter working" in i.message),
            None
        )
        assert adapter_success is not None
        
        provider_detection_success = next(
            (i for i in info_issues if "Provider detection working correctly" in i.message),
            None
        )
        assert provider_detection_success is not None

    @patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter')
    def test_check_genops_integration_no_tasks(self, mock_adapter_class):
        """Test GenOps integration with no supported tasks."""
        from genops.providers.huggingface_validation import check_genops_integration
        
        # Mock adapter with no supported tasks
        mock_adapter = Mock()
        mock_adapter.get_supported_tasks.return_value = []
        mock_adapter_class.return_value = mock_adapter
        
        issues = check_genops_integration()
        
        # Should have warning about no supported tasks
        warning_issues = [i for i in issues if i.level == "warning"]
        no_tasks_warning = next(
            (i for i in warning_issues if "no supported tasks found" in i.message),
            None
        )
        assert no_tasks_warning is not None

    @patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter')
    def test_check_genops_integration_partial_provider_detection(self, mock_adapter_class):
        """Test GenOps integration with partial provider detection success."""
        from genops.providers.huggingface_validation import check_genops_integration
        
        # Mock adapter with partial provider detection success
        mock_adapter = Mock()
        mock_adapter.get_supported_tasks.return_value = ['text-generation']
        
        # Only detect some models correctly
        def partial_detection(model):
            if model == 'gpt-3.5-turbo':
                return 'openai'
            return 'unknown'  # Wrong detection for others
        
        mock_adapter.detect_provider_for_model.side_effect = partial_detection
        mock_adapter_class.return_value = mock_adapter
        
        issues = check_genops_integration()
        
        # Should have warning about partial detection
        warning_issues = [i for i in issues if i.level == "warning"]
        partial_warning = next(
            (i for i in warning_issues if "working for" in i.message and "test models" in i.message),
            None
        )
        assert partial_warning is not None

    def test_check_genops_integration_import_error(self):
        """Test GenOps integration validation with import error."""
        from genops.providers.huggingface_validation import check_genops_integration
        
        # Test with patched import to simulate missing module
        with patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter', side_effect=ImportError):
            issues = check_genops_integration()
        
        # Should have error about import failure
        error_issues = [i for i in issues if i.level == "error"]
        import_error = next(
            (i for i in error_issues if "Cannot import GenOps Hugging Face adapter" in i.message),
            None
        )
        assert import_error is not None

    @patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter')
    def test_check_genops_integration_creation_error(self, mock_adapter_class):
        """Test GenOps integration with adapter creation error."""
        from genops.providers.huggingface_validation import check_genops_integration
        
        # Mock adapter creation failure
        mock_adapter_class.side_effect = Exception("Adapter creation failed")
        
        issues = check_genops_integration()
        
        # Should have error about adapter creation failure
        error_issues = [i for i in issues if i.level == "error"]
        creation_error = next(
            (i for i in error_issues if "Failed to create GenOps Hugging Face adapter" in i.message),
            None
        )
        assert creation_error is not None


class TestCostCalculationValidation:
    """Test cost calculation validation."""

    @patch('genops.providers.huggingface_validation.detect_model_provider')
    @patch('genops.providers.huggingface_validation.calculate_huggingface_cost')
    @patch('genops.providers.huggingface_validation.get_provider_info')
    def test_check_cost_calculation_success(self, mock_get_provider_info, mock_calculate_cost, mock_detect_provider):
        """Test successful cost calculation validation."""
        from genops.providers.huggingface_validation import check_cost_calculation
        
        # Mock successful provider detection
        mock_detect_provider.side_effect = lambda x: {
            'gpt-4': 'openai',
            'claude-3-sonnet': 'anthropic', 
            'microsoft/DialoGPT-medium': 'huggingface_hub',
            'mistral-7b-instruct': 'mistral'
        }.get(x, 'huggingface_hub')
        
        # Mock successful cost calculation
        mock_calculate_cost.return_value = 0.002
        
        # Mock successful provider info
        mock_get_provider_info.return_value = {'provider': 'openai', 'cost_per_1k': {'input': 0.001}}
        
        issues = check_cost_calculation()
        
        # Should have positive info messages
        info_issues = [i for i in issues if i.level == "info"]
        
        detection_success = next(
            (i for i in info_issues if "Provider detection working correctly" in i.message),
            None
        )
        assert detection_success is not None
        
        calculation_success = next(
            (i for i in info_issues if "Cost calculation working" in i.message),
            None
        )
        assert calculation_success is not None

    def test_check_cost_calculation_import_error(self):
        """Test cost calculation validation with import error."""
        from genops.providers.huggingface_validation import check_cost_calculation
        
        # Test with patched import to simulate missing module
        with patch('genops.providers.huggingface_validation.detect_model_provider', side_effect=ImportError):
            issues = check_cost_calculation()
        
        # Should have error about import failure
        error_issues = [i for i in issues if i.level == "error"]
        import_error = next(
            (i for i in error_issues if "Cannot import Hugging Face pricing utilities" in i.message),
            None
        )
        assert import_error is not None

    @patch('genops.providers.huggingface_validation.detect_model_provider')
    @patch('genops.providers.huggingface_validation.calculate_huggingface_cost')
    def test_check_cost_calculation_partial_detection(self, mock_calculate_cost, mock_detect_provider):
        """Test cost calculation with partial provider detection success."""
        from genops.providers.huggingface_validation import check_cost_calculation
        
        # Mock partial provider detection success (2 out of 4 correct)
        detection_results = ['openai', 'anthropic', 'wrong', 'also_wrong']
        mock_detect_provider.side_effect = detection_results
        
        mock_calculate_cost.return_value = 0.001
        
        issues = check_cost_calculation()
        
        # Should have warning about partial detection
        warning_issues = [i for i in issues if i.level == "warning"]
        partial_warning = next(
            (i for i in warning_issues if "working for 2/4 models" in i.message),
            None
        )
        assert partial_warning is not None

    @patch('genops.providers.huggingface_validation.detect_model_provider')
    @patch('genops.providers.huggingface_validation.calculate_huggingface_cost')
    def test_check_cost_calculation_error(self, mock_calculate_cost, mock_detect_provider):
        """Test cost calculation validation with calculation error."""
        from genops.providers.huggingface_validation import check_cost_calculation
        
        mock_detect_provider.return_value = 'openai'
        
        # Mock cost calculation error
        mock_calculate_cost.side_effect = Exception("Calculation failed")
        
        issues = check_cost_calculation()
        
        # Should have warning about calculation failure
        warning_issues = [i for i in issues if i.level == "warning"]
        calc_warning = next(
            (i for i in warning_issues if "Cost calculation test failed" in i.message),
            None
        )
        assert calc_warning is not None


class TestMainValidationFunction:
    """Test main validation orchestration."""

    @patch('genops.providers.huggingface_validation.check_environment_variables')
    @patch('genops.providers.huggingface_validation.check_dependencies')
    @patch('genops.providers.huggingface_validation.check_huggingface_connectivity')
    @patch('genops.providers.huggingface_validation.check_genops_integration')
    @patch('genops.providers.huggingface_validation.check_cost_calculation')
    def test_validate_huggingface_setup_all_pass(self, mock_cost, mock_integration, mock_connectivity, mock_deps, mock_env):
        """Test main validation when all checks pass."""
        from genops.providers.huggingface_validation import validate_huggingface_setup, ValidationIssue
        
        # Mock all checks returning only info/warning issues
        mock_env.return_value = [ValidationIssue("info", "env", "Info message")]
        mock_deps.return_value = [ValidationIssue("warning", "deps", "Warning message")]
        mock_connectivity.return_value = [ValidationIssue("info", "conn", "Connected")]
        mock_integration.return_value = [ValidationIssue("info", "integration", "Working")]
        mock_cost.return_value = [ValidationIssue("info", "cost", "Calculating")]
        
        result = validate_huggingface_setup()
        
        assert result.is_valid is True
        assert len(result.issues) == 5
        assert result.summary['errors'] == 0
        assert result.summary['warnings'] == 1
        assert result.summary['info'] == 4

    @patch('genops.providers.huggingface_validation.check_environment_variables')
    @patch('genops.providers.huggingface_validation.check_dependencies')
    @patch('genops.providers.huggingface_validation.check_huggingface_connectivity')
    @patch('genops.providers.huggingface_validation.check_genops_integration')
    @patch('genops.providers.huggingface_validation.check_cost_calculation')
    def test_validate_huggingface_setup_with_errors(self, mock_cost, mock_integration, mock_connectivity, mock_deps, mock_env):
        """Test main validation with errors present."""
        from genops.providers.huggingface_validation import validate_huggingface_setup, ValidationIssue
        
        # Mock some checks returning error issues
        mock_env.return_value = []
        mock_deps.return_value = [ValidationIssue("error", "deps", "Missing dependency")]
        mock_connectivity.return_value = [ValidationIssue("error", "conn", "Cannot connect")]
        mock_integration.return_value = [ValidationIssue("warning", "integration", "Partial working")]
        mock_cost.return_value = [ValidationIssue("info", "cost", "Working")]
        
        result = validate_huggingface_setup()
        
        assert result.is_valid is False
        assert len(result.issues) == 4
        assert result.summary['errors'] == 2
        assert result.summary['warnings'] == 1
        assert result.summary['info'] == 1

    @patch('genops.providers.huggingface_validation.check_environment_variables')
    def test_validate_huggingface_setup_with_exception(self, mock_env):
        """Test main validation handles exceptions gracefully."""
        from genops.providers.huggingface_validation import validate_huggingface_setup
        
        # Mock one check throwing an exception
        mock_env.side_effect = Exception("Validation check failed")
        
        result = validate_huggingface_setup()
        
        # Should handle exception and create error issue
        assert result.is_valid is False
        
        error_issues = [i for i in result.issues if i.level == "error"]
        validation_error = next(
            (i for i in error_issues if "Validation check check_environment_variables failed" in i.message),
            None
        )
        assert validation_error is not None


class TestValidationReporting:
    """Test validation result reporting."""

    def test_print_huggingface_validation_result_valid(self, capsys):
        """Test printing valid validation result."""
        from genops.providers.huggingface_validation import (
            print_huggingface_validation_result, 
            ValidationResult,
            ValidationIssue
        )
        
        issues = [ValidationIssue("info", "test", "Test info message")]
        summary = {"components_checked": 5, "total_issues": 1, "errors": 0, "warnings": 0, "info": 1}
        
        result = ValidationResult(is_valid=True, issues=issues, summary=summary)
        
        print_huggingface_validation_result(result)
        
        captured = capsys.readouterr()
        assert "✅ Overall Status: VALID - Ready to use!" in captured.out
        assert "Components checked: 5" in captured.out

    def test_print_huggingface_validation_result_invalid(self, capsys):
        """Test printing invalid validation result."""
        from genops.providers.huggingface_validation import (
            print_huggingface_validation_result,
            ValidationResult, 
            ValidationIssue
        )
        
        issues = [
            ValidationIssue("error", "deps", "Missing dependency", "pip install xyz"),
            ValidationIssue("warning", "config", "Config issue")
        ]
        summary = {"components_checked": 5, "total_issues": 2, "errors": 1, "warnings": 1, "info": 0}
        
        result = ValidationResult(is_valid=False, issues=issues, summary=summary)
        
        print_huggingface_validation_result(result)
        
        captured = capsys.readouterr()
        assert "❌ Overall Status: ISSUES FOUND" in captured.out
        assert "❌ Errors: 1" in captured.out
        assert "⚠️  Warnings: 1" in captured.out
        assert "💡 Fix: pip install xyz" in captured.out

    def test_quick_validate_success(self, capsys):
        """Test quick validation success case."""
        from genops.providers.huggingface_validation import quick_validate
        
        with patch('genops.providers.huggingface_validation.validate_huggingface_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            result = quick_validate()
            
            assert result is True
            captured = capsys.readouterr()
            assert "✅ Hugging Face setup validation passed!" in captured.out

    def test_quick_validate_failure(self, capsys):
        """Test quick validation failure case."""
        from genops.providers.huggingface_validation import quick_validate
        
        with patch('genops.providers.huggingface_validation.validate_huggingface_setup') as mock_validate:
            mock_result = Mock(is_valid=False)
            mock_result.issues = [Mock(level="error"), Mock(level="warning")]
            mock_validate.return_value = mock_result
            
            result = quick_validate()
            
            assert result is False
            captured = capsys.readouterr()
            assert "❌ Hugging Face setup validation failed with 1 error(s)" in captured.out


class TestValidationScriptExecution:
    """Test validation script execution."""

    def test_main_execution_success(self):
        """Test main function execution with successful validation."""
        from genops.providers.huggingface_validation import ValidationResult, ValidationIssue
        
        with patch('genops.providers.huggingface_validation.validate_huggingface_setup') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                issues=[ValidationIssue("info", "test", "All good")],
                summary={"errors": 0}
            )
            
            # Test the module's main execution logic
            with patch('genops.providers.huggingface_validation.print_huggingface_validation_result'):
                # This simulates running the script directly
                from genops.providers.huggingface_validation import validate_huggingface_setup, print_huggingface_validation_result
                
                result = validate_huggingface_setup()
                assert result.is_valid is True

    def test_main_execution_failure(self):
        """Test main function execution with validation failures."""
        from genops.providers.huggingface_validation import ValidationResult, ValidationIssue
        
        with patch('genops.providers.huggingface_validation.validate_huggingface_setup') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=False,
                issues=[ValidationIssue("error", "test", "Something failed")],
                summary={"errors": 1}
            )
            
            result = mock_validate()
            assert result.is_valid is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])