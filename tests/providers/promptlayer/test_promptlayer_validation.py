"""
Comprehensive tests for GenOps PromptLayer Validation utilities.

Tests validation functionality including:
- Setup validation and diagnostics
- Environment variable checking
- API connectivity testing
- Dependency verification
- Error handling and suggestions
- Performance validation
"""

import os
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.promptlayer_validation import (
        ValidationCheck,
        ValidationResult,
        ValidationStatus,
        check_genops_configuration,
        check_genops_installation,
        check_genops_promptlayer_integration,
        check_governance_features,
        check_optional_dependencies,
        check_performance_overhead,
        check_promptlayer_api_key,
        check_promptlayer_connectivity,
        check_promptlayer_installation,
        check_python_version,
        print_validation_result,
        validate_setup,
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="PromptLayer validation not available")
class TestValidationDataClasses:
    """Test validation data classes and enums."""

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values."""
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.WARNING.value == "warning"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.SKIPPED.value == "skipped"

    def test_validation_check_creation(self):
        """Test ValidationCheck dataclass."""
        check = ValidationCheck(
            name="Test Check",
            status=ValidationStatus.PASSED,
            message="Test passed successfully",
            details="Additional test details",
            fix_suggestion="No fix needed",
            category="testing"
        )

        assert check.name == "Test Check"
        assert check.status == ValidationStatus.PASSED
        assert check.message == "Test passed successfully"
        assert check.details == "Additional test details"
        assert check.fix_suggestion == "No fix needed"
        assert check.category == "testing"

    def test_validation_result_properties(self):
        """Test ValidationResult properties and aggregation."""
        checks = [
            ValidationCheck("Check 1", ValidationStatus.PASSED, "Passed"),
            ValidationCheck("Check 2", ValidationStatus.WARNING, "Warning"),
            ValidationCheck("Check 3", ValidationStatus.FAILED, "Failed"),
            ValidationCheck("Check 4", ValidationStatus.PASSED, "Passed"),
        ]

        result = ValidationResult(
            overall_status=ValidationStatus.WARNING,
            checks=checks
        )

        assert result.passed_checks == 2
        assert result.warning_checks == 1
        assert result.failed_checks == 1
        assert len(result.checks) == 4


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="PromptLayer validation not available")
class TestIndividualValidationChecks:
    """Test individual validation check functions."""

    def test_check_python_version_success(self):
        """Test Python version check with compatible version."""
        with patch('sys.version_info', (3, 9, 0)):
            result = check_python_version()
            assert result.status == ValidationStatus.PASSED
            assert "3.9" in result.message
            assert result.category == "dependencies"

    def test_check_python_version_failure(self):
        """Test Python version check with incompatible version."""
        with patch('sys.version_info', (3, 7, 0)):
            result = check_python_version()
            assert result.status == ValidationStatus.FAILED
            assert "3.7" in result.message
            assert "Upgrade Python" in result.fix_suggestion

    def test_check_genops_installation_success(self):
        """Test GenOps installation check when available."""
        with patch('importlib.import_module') as mock_import:
            # Mock successful import
            mock_genops = Mock()
            mock_genops.__version__ = "1.0.0"
            mock_import.return_value = mock_genops

            result = check_genops_installation()
            assert result.status == ValidationStatus.PASSED
            assert "1.0.0" in result.message

    def test_check_genops_installation_failure(self):
        """Test GenOps installation check when not available."""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'genops'")):
            result = check_genops_installation()
            assert result.status == ValidationStatus.FAILED
            assert "not installed" in result.message.lower()
            assert "pip install genops" in result.fix_suggestion

    def test_check_promptlayer_installation_success(self):
        """Test PromptLayer SDK installation check when available."""
        with patch('importlib.import_module') as mock_import:
            mock_promptlayer = Mock()
            mock_promptlayer.__version__ = "0.15.0"
            mock_import.return_value = mock_promptlayer

            result = check_promptlayer_installation()
            assert result.status == ValidationStatus.PASSED
            assert "0.15.0" in result.message

    def test_check_promptlayer_installation_failure(self):
        """Test PromptLayer SDK installation check when not available."""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'promptlayer'")):
            result = check_promptlayer_installation()
            assert result.status == ValidationStatus.FAILED
            assert "not installed" in result.message.lower()
            assert "pip install promptlayer" in result.fix_suggestion

    def test_check_optional_dependencies_all_present(self):
        """Test optional dependencies check when all are present."""
        def mock_import(name):
            if name in ['openai', 'anthropic', 'requests']:
                return Mock()
            raise ImportError(f"No module named '{name}'")

        with patch('importlib.import_module', side_effect=mock_import):
            result = check_optional_dependencies()
            assert result.status == ValidationStatus.PASSED
            assert "All optional dependencies" in result.message

    def test_check_optional_dependencies_some_missing(self):
        """Test optional dependencies check when some are missing."""
        def mock_import(name):
            if name == 'requests':
                return Mock()
            raise ImportError(f"No module named '{name}'")

        with patch('importlib.import_module', side_effect=mock_import):
            result = check_optional_dependencies()
            assert result.status == ValidationStatus.WARNING
            assert "missing" in result.message.lower()
            assert "openai" in result.details
            assert "anthropic" in result.details

    def test_check_promptlayer_api_key_valid(self):
        """Test PromptLayer API key check with valid key."""
        result = check_promptlayer_api_key("pl-1234567890abcdef")
        assert result.status == ValidationStatus.PASSED
        assert "configured and format appears valid" in result.message
        assert "pl-" in result.details

    def test_check_promptlayer_api_key_missing(self):
        """Test PromptLayer API key check when missing."""
        result = check_promptlayer_api_key(None)
        assert result.status == ValidationStatus.FAILED
        assert "not found" in result.message.lower()
        assert "PROMPTLAYER_API_KEY" in result.fix_suggestion

    def test_check_promptlayer_api_key_invalid_format(self):
        """Test PromptLayer API key check with invalid format."""
        result = check_promptlayer_api_key("invalid-key-format")
        assert result.status == ValidationStatus.WARNING
        assert "format may be invalid" in result.message.lower()
        assert "pl-" in result.details

    def test_check_promptlayer_api_key_too_short(self):
        """Test PromptLayer API key check with too short key."""
        result = check_promptlayer_api_key("pl-123")
        assert result.status == ValidationStatus.WARNING
        assert "too short" in result.message.lower()
        assert "3 characters" in result.details

    def test_check_genops_configuration_complete(self):
        """Test GenOps configuration check when complete."""
        result = check_genops_configuration(team="test-team", project="test-project")
        assert result.status == ValidationStatus.PASSED
        assert "complete" in result.message.lower()
        assert "test-team" in result.details
        assert "test-project" in result.details

    def test_check_genops_configuration_incomplete(self):
        """Test GenOps configuration check when incomplete."""
        result = check_genops_configuration(team=None, project="test-project")
        assert result.status == ValidationStatus.WARNING
        assert "incomplete" in result.message.lower()
        assert "Team not specified" in result.details

    @patch('genops.providers.promptlayer_validation.PromptLayer')
    def test_check_promptlayer_connectivity_success(self, mock_promptlayer_class):
        """Test PromptLayer connectivity check when successful."""
        mock_client = Mock()
        mock_promptlayer_class.return_value = mock_client

        result = check_promptlayer_connectivity("pl-test-key")
        assert result.status == ValidationStatus.PASSED
        assert "Successfully connected" in result.message

    def test_check_promptlayer_connectivity_no_key(self):
        """Test PromptLayer connectivity check without API key."""
        result = check_promptlayer_connectivity(None)
        assert result.status == ValidationStatus.SKIPPED
        assert "no API key available" in result.message.lower()

    @patch('genops.providers.promptlayer_validation.PromptLayer')
    def test_check_promptlayer_connectivity_failure(self, mock_promptlayer_class):
        """Test PromptLayer connectivity check when connection fails."""
        mock_promptlayer_class.side_effect = Exception("Connection failed")

        result = check_promptlayer_connectivity("pl-test-key")
        assert result.status == ValidationStatus.FAILED
        assert "Failed to connect" in result.message
        assert "Connection failed" in result.details


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="PromptLayer validation not available")
class TestIntegrationValidationChecks:
    """Test integration-specific validation checks."""

    @patch('genops.providers.promptlayer_validation.GenOpsPromptLayerAdapter')
    def test_check_genops_promptlayer_integration_success(self, mock_adapter_class):
        """Test GenOps PromptLayer integration check when successful."""
        mock_adapter = Mock()
        mock_adapter.get_metrics.return_value = {
            'team': 'validation-test',
            'project': 'integration-check'
        }
        mock_adapter_class.return_value = mock_adapter

        result = check_genops_promptlayer_integration()
        assert result.status == ValidationStatus.PASSED
        assert "functional" in result.message.lower()
        assert "validation-test" in result.details

    @patch('genops.providers.promptlayer_validation.GenOpsPromptLayerAdapter')
    def test_check_genops_promptlayer_integration_import_failure(self, mock_adapter_class):
        """Test integration check when import fails."""
        with patch('genops.providers.promptlayer_validation.GenOpsPromptLayerAdapter',
                   side_effect=ImportError("Module not found")):
            result = check_genops_promptlayer_integration()
            assert result.status == ValidationStatus.FAILED
            assert "not available" in result.message.lower()
            assert "pip install genops[promptlayer]" in result.fix_suggestion

    @patch('genops.providers.promptlayer_validation.GenOpsPromptLayerAdapter')
    def test_check_governance_features_success(self, mock_adapter_class):
        """Test governance features check when successful."""
        mock_adapter = Mock()
        mock_adapter.get_metrics.return_value = {
            'daily_usage': 0.005,
            'operation_count': 1
        }
        mock_adapter_class.return_value = mock_adapter

        # Mock the track_prompt_operation context manager
        mock_span = Mock()
        mock_adapter.track_prompt_operation.return_value.__enter__ = Mock(return_value=mock_span)
        mock_adapter.track_prompt_operation.return_value.__exit__ = Mock(return_value=None)

        result = check_governance_features()
        assert result.status == ValidationStatus.PASSED
        assert "functional" in result.message.lower()

    @patch('genops.providers.promptlayer_validation.GenOpsPromptLayerAdapter')
    def test_check_performance_overhead_success(self, mock_adapter_class):
        """Test performance overhead check."""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        # Mock the track_prompt_operation context manager
        mock_span = Mock()
        mock_adapter.track_prompt_operation.return_value.__enter__ = Mock(return_value=mock_span)
        mock_adapter.track_prompt_operation.return_value.__exit__ = Mock(return_value=None)

        result = check_performance_overhead()
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        assert "overhead" in result.message.lower()
        assert "ms per operation" in result.details


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="PromptLayer validation not available")
class TestValidationSuite:
    """Test the complete validation suite."""

    @patch.dict(os.environ, {
        'PROMPTLAYER_API_KEY': 'pl-test-key-12345',
        'GENOPS_TEAM': 'test-team',
        'GENOPS_PROJECT': 'test-project'
    })
    def test_validate_setup_comprehensive(self):
        """Test comprehensive setup validation."""
        # Mock all the external dependencies
        with patch('sys.version_info', (3, 9, 0)), \
             patch('importlib.import_module') as mock_import, \
             patch('genops.providers.promptlayer_validation.GenOpsPromptLayerAdapter') as mock_adapter, \
             patch('genops.providers.promptlayer_validation.PromptLayer'):

            # Setup successful mocks
            mock_genops = Mock()
            mock_genops.__version__ = "1.0.0"
            mock_promptlayer = Mock()
            mock_promptlayer.__version__ = "0.15.0"

            def import_side_effect(name):
                if 'genops' in name:
                    return mock_genops
                elif 'promptlayer' in name:
                    return mock_promptlayer
                elif name in ['openai', 'anthropic', 'requests']:
                    return Mock()
                else:
                    raise ImportError(f"No module named '{name}'")

            mock_import.side_effect = import_side_effect

            # Mock adapter
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_metrics.return_value = {'team': 'test-team'}
            mock_adapter.return_value = mock_adapter_instance

            # Mock context manager
            mock_span = Mock()
            mock_adapter_instance.track_prompt_operation.return_value.__enter__ = Mock(return_value=mock_span)
            mock_adapter_instance.track_prompt_operation.return_value.__exit__ = Mock(return_value=None)

            result = validate_setup(
                include_connectivity_tests=True,
                include_performance_tests=True,
                include_governance_tests=True
            )

            assert result.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
            assert result.total_duration_ms > 0
            assert len(result.checks) >= 8  # Should have multiple checks

            # Verify summary
            assert 'total_checks' in result.summary
            assert 'passed' in result.summary
            assert 'categories' in result.summary

    def test_validate_setup_with_failures(self):
        """Test validation with some failing checks."""
        with patch('sys.version_info', (3, 7, 0)), \
             patch('importlib.import_module', side_effect=ImportError("Module not found")):

            result = validate_setup(
                include_connectivity_tests=False,
                include_performance_tests=False,
                include_governance_tests=False
            )

            assert result.overall_status == ValidationStatus.FAILED
            assert result.failed_checks > 0

    def test_validate_setup_custom_parameters(self):
        """Test validation with custom parameters."""
        result = validate_setup(
            promptlayer_api_key="pl-custom-key",
            team="custom-team",
            project="custom-project",
            include_connectivity_tests=False
        )

        # Should run basic checks even with failures
        assert isinstance(result, ValidationResult)
        assert len(result.checks) > 0


@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="PromptLayer validation not available")
class TestValidationResultFormatting:
    """Test validation result formatting and display."""

    def test_print_validation_result_success(self, capsys):
        """Test printing validation results for successful case."""
        checks = [
            ValidationCheck("Check 1", ValidationStatus.PASSED, "Success", category="test"),
            ValidationCheck("Check 2", ValidationStatus.PASSED, "Success", category="test"),
        ]

        result = ValidationResult(
            overall_status=ValidationStatus.PASSED,
            checks=checks,
            total_duration_ms=150.0,
            summary={
                'total_checks': 2,
                'passed': 2,
                'warnings': 0,
                'failed': 0,
                'skipped': 0
            }
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "✅" in captured.out
        assert "PASSED" in captured.out
        assert "150ms" in captured.out
        assert "All checks passed" in captured.out

    def test_print_validation_result_with_failures(self, capsys):
        """Test printing validation results with failures."""
        checks = [
            ValidationCheck("Check 1", ValidationStatus.PASSED, "Success", category="test"),
            ValidationCheck("Check 2", ValidationStatus.FAILED, "Failed",
                          fix_suggestion="Fix this issue", category="test"),
        ]

        result = ValidationResult(
            overall_status=ValidationStatus.FAILED,
            checks=checks,
            summary={
                'total_checks': 2,
                'passed': 1,
                'warnings': 0,
                'failed': 1,
                'skipped': 0
            }
        )

        print_validation_result(result, detailed=True)

        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "FAILED" in captured.out
        assert "Fix this issue" in captured.out
        assert "critical issues" in captured.out

    def test_print_validation_result_with_warnings(self, capsys):
        """Test printing validation results with warnings."""
        checks = [
            ValidationCheck("Check 1", ValidationStatus.PASSED, "Success", category="test"),
            ValidationCheck("Check 2", ValidationStatus.WARNING, "Warning message",
                          details="Warning details", category="test"),
        ]

        result = ValidationResult(
            overall_status=ValidationStatus.WARNING,
            checks=checks,
            summary={
                'total_checks': 2,
                'passed': 1,
                'warnings': 1,
                'failed': 0,
                'skipped': 0
            }
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "WARNING" in captured.out
        assert "optimizations are recommended" in captured.out


@pytest.mark.integration
@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="PromptLayer validation not available")
class TestRealValidation:
    """Integration tests for real validation scenarios."""

    def test_real_environment_validation(self):
        """Test validation against real environment."""
        # This test runs against the actual environment
        result = validate_setup(
            include_connectivity_tests=False,  # Don't test real API connections
            include_performance_tests=False,   # Skip performance tests
            include_governance_tests=False     # Skip governance tests
        )

        # Should at least validate Python and basic dependencies
        assert isinstance(result, ValidationResult)
        assert len(result.checks) > 0

        # Python version should pass
        python_checks = [c for c in result.checks if 'python' in c.name.lower()]
        assert len(python_checks) > 0
        assert python_checks[0].status == ValidationStatus.PASSED

    @pytest.mark.skipif(not os.getenv('PROMPTLAYER_API_KEY'),
                       reason="PROMPTLAYER_API_KEY not set")
    def test_real_api_connectivity(self):
        """Test real API connectivity if key is available."""
        api_key = os.getenv('PROMPTLAYER_API_KEY')

        result = check_promptlayer_connectivity(api_key)

        # Should either pass or provide meaningful error
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED]
        if result.status == ValidationStatus.FAILED:
            assert result.fix_suggestion is not None
