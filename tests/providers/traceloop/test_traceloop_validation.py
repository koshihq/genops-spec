#!/usr/bin/env python3
"""
Comprehensive test suite for Traceloop validation utilities.

Tests the validation framework that ensures proper setup of Traceloop + OpenLLMetry + GenOps
integration with comprehensive diagnostics and actionable error handling.
"""

import os
from unittest.mock import Mock, patch

import pytest

# Test imports with graceful handling for missing dependencies
try:
    from genops.providers.traceloop_validation import (
        ValidationCategory,
        ValidationResult,
        ValidationStatus,
        ValidationSummary,
        print_validation_result,
        validate_configuration,
        validate_connectivity,
        validate_dependencies,
        validate_governance,
        validate_performance,
        validate_setup,
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False


class TestValidationResults:
    """Unit tests for validation result structures."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation with all fields."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        result = ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="test_dependency",
            status=ValidationStatus.PASSED,
            message="Dependency check passed",
            details={"version": "1.0.0"},
            fix_suggestion="No action needed",
            execution_time_ms=25.5
        )

        assert result.category == ValidationCategory.DEPENDENCIES
        assert result.check_name == "test_dependency"
        assert result.status == ValidationStatus.PASSED
        assert result.message == "Dependency check passed"
        assert result.details["version"] == "1.0.0"
        assert result.fix_suggestion == "No action needed"
        assert result.execution_time_ms == 25.5

    def test_validation_summary_initialization(self):
        """Test ValidationSummary initialization."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        summary = ValidationSummary(
            overall_status=ValidationStatus.PASSED,
            total_checks=0,
            passed_checks=0,
            warning_checks=0,
            failed_checks=0,
            skipped_checks=0
        )

        assert summary.overall_status == ValidationStatus.PASSED
        assert summary.total_checks == 0
        assert len(summary.results) == 0
        assert summary.total_execution_time_ms == 0.0

    def test_validation_summary_add_results(self):
        """Test adding results to ValidationSummary."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        summary = ValidationSummary(
            overall_status=ValidationStatus.PASSED,
            total_checks=0,
            passed_checks=0,
            warning_checks=0,
            failed_checks=0,
            skipped_checks=0
        )

        # Add passed result
        passed_result = ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="test_passed",
            status=ValidationStatus.PASSED,
            message="Passed check",
            execution_time_ms=10.0
        )
        summary.add_result(passed_result)

        assert summary.total_checks == 1
        assert summary.passed_checks == 1
        assert summary.overall_status == ValidationStatus.PASSED

        # Add warning result
        warning_result = ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="test_warning",
            status=ValidationStatus.WARNING,
            message="Warning check",
            execution_time_ms=15.0
        )
        summary.add_result(warning_result)

        assert summary.total_checks == 2
        assert summary.passed_checks == 1
        assert summary.warning_checks == 1
        assert summary.overall_status == ValidationStatus.WARNING

        # Add failed result
        failed_result = ValidationResult(
            category=ValidationCategory.CONNECTIVITY,
            check_name="test_failed",
            status=ValidationStatus.FAILED,
            message="Failed check",
            execution_time_ms=5.0
        )
        summary.add_result(failed_result)

        assert summary.total_checks == 3
        assert summary.passed_checks == 1
        assert summary.warning_checks == 1
        assert summary.failed_checks == 1
        assert summary.overall_status == ValidationStatus.FAILED
        assert summary.total_execution_time_ms == 30.0


class TestDependencyValidation:
    """Tests for dependency validation functionality."""

    @patch('genops.providers.traceloop_validation.sys.version_info', (3, 9, 0))
    def test_python_version_validation_success(self):
        """Test Python version validation success."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        results = validate_dependencies()

        # Find Python version check
        python_results = [r for r in results if r.check_name == "python_version"]
        assert len(python_results) == 1
        assert python_results[0].status == ValidationStatus.PASSED

    @patch('genops.providers.traceloop_validation.sys.version_info', (3, 7, 0))
    def test_python_version_validation_failure(self):
        """Test Python version validation failure."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        results = validate_dependencies()

        # Find Python version check
        python_results = [r for r in results if r.check_name == "python_version"]
        assert len(python_results) == 1
        assert python_results[0].status == ValidationStatus.FAILED
        assert "Upgrade to Python 3.8" in python_results[0].fix_suggestion

    @patch('builtins.__import__', side_effect=lambda name, *args: Mock() if name == 'openllmetry' else __import__(name, *args))
    def test_openllmetry_availability_success(self):
        """Test OpenLLMetry availability check success."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        results = validate_dependencies()

        # Find OpenLLMetry check
        openllmetry_results = [r for r in results if r.check_name == "openllmetry_availability"]
        assert len(openllmetry_results) == 1
        assert openllmetry_results[0].status == ValidationStatus.PASSED

    def test_openllmetry_availability_failure(self):
        """Test OpenLLMetry availability check failure."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        with patch('builtins.__import__', side_effect=ImportError("No module named 'openllmetry'")):
            results = validate_dependencies()

            # Find OpenLLMetry check
            openllmetry_results = [r for r in results if r.check_name == "openllmetry_availability"]
            assert len(openllmetry_results) == 1
            assert openllmetry_results[0].status == ValidationStatus.FAILED
            assert "pip install openllmetry" in openllmetry_results[0].fix_suggestion

    @patch('builtins.__import__', side_effect=lambda name, *args: Mock() if name == 'traceloop.sdk' else __import__(name, *args))
    def test_traceloop_sdk_availability_success(self):
        """Test Traceloop SDK availability check success."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        results = validate_dependencies()

        # Find Traceloop SDK check
        traceloop_results = [r for r in results if r.check_name == "traceloop_sdk_availability"]
        assert len(traceloop_results) == 1
        assert traceloop_results[0].status == ValidationStatus.PASSED

    def test_genops_integration_availability(self):
        """Test GenOps integration availability check."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # This should pass since we're running the test
        results = validate_dependencies()

        # Find GenOps integration check
        genops_results = [r for r in results if r.check_name == "genops_traceloop_integration"]
        assert len(genops_results) == 1
        # Should pass if we got this far
        assert genops_results[0].status == ValidationStatus.PASSED


class TestConfigurationValidation:
    """Tests for configuration validation functionality."""

    def test_api_key_validation_openai_present(self):
        """Test API key validation when OpenAI key present."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'}):
            results = validate_configuration()

            # Find OpenAI key check
            openai_results = [r for r in results if r.check_name == "openai_api_key"]
            assert len(openai_results) == 1
            assert openai_results[0].status == ValidationStatus.PASSED

            # Find provider availability check
            provider_results = [r for r in results if r.check_name == "ai_provider_available"]
            assert len(provider_results) == 1
            assert provider_results[0].status == ValidationStatus.PASSED

    def test_api_key_validation_no_providers(self):
        """Test API key validation when no providers configured."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Clear all provider environment variables
        env_vars_to_clear = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GROQ_API_KEY']
        with patch.dict(os.environ, dict.fromkeys(env_vars_to_clear, ''), clear=True):
            results = validate_configuration()

            # Find provider availability check
            provider_results = [r for r in results if r.check_name == "ai_provider_available"]
            assert len(provider_results) == 1
            assert provider_results[0].status == ValidationStatus.FAILED

    def test_traceloop_platform_config_present(self):
        """Test Traceloop platform configuration when API key present."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        with patch.dict(os.environ, {'TRACELOOP_API_KEY': 'test-traceloop-key'}):
            results = validate_configuration()

            # Find Traceloop platform config check
            traceloop_results = [r for r in results if r.check_name == "traceloop_platform_config"]
            assert len(traceloop_results) == 1
            assert traceloop_results[0].status == ValidationStatus.PASSED

    def test_traceloop_platform_config_absent(self):
        """Test Traceloop platform configuration when API key absent."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        with patch.dict(os.environ, {}, clear=True):
            results = validate_configuration()

            # Find Traceloop platform config check
            traceloop_results = [r for r in results if r.check_name == "traceloop_platform_config"]
            assert len(traceloop_results) == 1
            assert traceloop_results[0].status == ValidationStatus.SKIPPED

    def test_genops_governance_config_complete(self):
        """Test GenOps governance configuration when complete."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        with patch.dict(os.environ, {
            'GENOPS_TEAM': 'test-team',
            'GENOPS_PROJECT': 'test-project'
        }):
            results = validate_configuration()

            # Find governance config check
            governance_results = [r for r in results if r.check_name == "genops_governance_config"]
            assert len(governance_results) == 1
            assert governance_results[0].status == ValidationStatus.PASSED

    def test_genops_governance_config_incomplete(self):
        """Test GenOps governance configuration when incomplete."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        with patch.dict(os.environ, {'GENOPS_TEAM': 'test-team'}, clear=True):
            results = validate_configuration()

            # Find governance config check
            governance_results = [r for r in results if r.check_name == "genops_governance_config"]
            assert len(governance_results) == 1
            assert governance_results[0].status == ValidationStatus.WARNING


class TestConnectivityValidation:
    """Tests for connectivity validation functionality."""

    @patch('genops.providers.traceloop_validation.openai.OpenAI')
    def test_openai_connectivity_success(self, mock_openai_class):
        """Test OpenAI connectivity validation success."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock successful OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            results = validate_connectivity()

            # Find OpenAI connectivity check
            openai_results = [r for r in results if r.check_name == "openai_connectivity"]
            assert len(openai_results) == 1
            assert openai_results[0].status == ValidationStatus.PASSED

    @patch('genops.providers.traceloop_validation.openai.OpenAI')
    def test_openai_connectivity_failure(self, mock_openai_class):
        """Test OpenAI connectivity validation failure."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock OpenAI connection failure
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        mock_openai_class.return_value = mock_client

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            results = validate_connectivity()

            # Find OpenAI connectivity check
            openai_results = [r for r in results if r.check_name == "openai_connectivity"]
            assert len(openai_results) == 1
            assert openai_results[0].status == ValidationStatus.FAILED

    @patch('genops.providers.traceloop_validation.anthropic.Anthropic')
    def test_anthropic_connectivity_success(self, mock_anthropic_class):
        """Test Anthropic connectivity validation success."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock successful Anthropic response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            results = validate_connectivity()

            # Find Anthropic connectivity check
            anthropic_results = [r for r in results if r.check_name == "anthropic_connectivity"]
            assert len(anthropic_results) == 1
            assert anthropic_results[0].status == ValidationStatus.PASSED


class TestGovernanceValidation:
    """Tests for governance validation functionality."""

    @patch('genops.providers.traceloop_validation.instrument_traceloop')
    def test_genops_adapter_creation_success(self, mock_instrument):
        """Test GenOps adapter creation validation success."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock successful adapter creation
        mock_adapter = Mock()
        mock_instrument.return_value = mock_adapter

        results = validate_governance()

        # Find adapter creation check
        adapter_results = [r for r in results if r.check_name == "genops_adapter_creation"]
        assert len(adapter_results) == 1
        assert adapter_results[0].status == ValidationStatus.PASSED

    @patch('genops.providers.traceloop_validation.instrument_traceloop')
    def test_genops_adapter_creation_failure(self, mock_instrument):
        """Test GenOps adapter creation validation failure."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock adapter creation failure
        mock_instrument.side_effect = Exception("Adapter creation failed")

        results = validate_governance()

        # Find adapter creation check
        adapter_results = [r for r in results if r.check_name == "genops_adapter_creation"]
        assert len(adapter_results) == 1
        assert adapter_results[0].status == ValidationStatus.FAILED

    @patch('genops.providers.traceloop_validation.auto_instrument')
    def test_auto_instrumentation_availability(self, mock_auto_instrument):
        """Test auto-instrumentation availability validation."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        results = validate_governance()

        # Find auto-instrumentation check
        auto_results = [r for r in results if r.check_name == "auto_instrumentation_available"]
        assert len(auto_results) == 1
        assert auto_results[0].status == ValidationStatus.PASSED


class TestPerformanceValidation:
    """Tests for performance validation functionality."""

    @patch('genops.providers.traceloop_validation.instrument_traceloop')
    def test_governance_overhead_acceptable(self, mock_instrument):
        """Test governance overhead is within acceptable limits."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock adapter with fast track_operation
        mock_adapter = Mock()
        mock_span = Mock()
        mock_span.update_cost = Mock()
        mock_span.get_metrics.return_value = {"estimated_cost": 0.001}

        @contextmanager
        def mock_track_operation(*args, **kwargs):
            yield mock_span

        mock_adapter.track_operation = mock_track_operation
        mock_instrument.return_value = mock_adapter

        results = validate_performance()

        # Find governance overhead check
        overhead_results = [r for r in results if r.check_name == "governance_overhead"]
        assert len(overhead_results) == 1
        # Should pass as mocked operation is fast
        assert overhead_results[0].status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    @patch('genops.providers.traceloop_validation.instrument_traceloop')
    def test_governance_overhead_slow(self, mock_instrument):
        """Test governance overhead detection when slow."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Mock adapter with slow track_operation
        mock_adapter = Mock()
        mock_span = Mock()
        mock_span.update_cost = Mock()
        mock_span.get_metrics.return_value = {"estimated_cost": 0.001}

        @contextmanager
        def mock_track_operation_slow(*args, **kwargs):
            import time
            time.sleep(0.1)  # Add 100ms delay
            yield mock_span

        mock_adapter.track_operation = mock_track_operation_slow
        mock_instrument.return_value = mock_adapter

        results = validate_performance()

        # Find governance overhead check
        overhead_results = [r for r in results if r.check_name == "governance_overhead"]
        assert len(overhead_results) == 1
        # Should warn about high overhead
        assert overhead_results[0].status == ValidationStatus.WARNING


class TestValidationIntegration:
    """Integration tests for complete validation workflows."""

    def test_validate_setup_minimal_config(self):
        """Test validate_setup with minimal configuration."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Run validation without connectivity/performance tests
        result = validate_setup(
            include_connectivity_tests=False,
            include_performance_tests=False
        )

        assert isinstance(result, ValidationSummary)
        assert result.total_checks > 0
        assert result.overall_status in [
            ValidationStatus.PASSED,
            ValidationStatus.WARNING,
            ValidationStatus.FAILED
        ]

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_validate_setup_with_provider(self):
        """Test validate_setup with provider configured."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        result = validate_setup(
            include_connectivity_tests=False,
            include_performance_tests=False
        )

        assert isinstance(result, ValidationSummary)
        # Should have better status with provider configured
        assert result.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    def test_print_validation_result_basic(self):
        """Test print_validation_result basic functionality."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Create test summary
        summary = ValidationSummary(
            overall_status=ValidationStatus.PASSED,
            total_checks=2,
            passed_checks=2,
            warning_checks=0,
            failed_checks=0,
            skipped_checks=0
        )

        test_result = ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="test_check",
            status=ValidationStatus.PASSED,
            message="Test passed"
        )
        summary.add_result(test_result)

        # Should not raise exception
        print_validation_result(summary, detailed=False)
        print_validation_result(summary, detailed=True)

    def test_validation_error_scenarios(self):
        """Test validation handles error scenarios gracefully."""
        if not HAS_VALIDATION:
            pytest.skip("Traceloop validation not available")

        # Test with ImportError scenarios
        with patch('builtins.__import__', side_effect=ImportError("Import failed")):
            results = validate_dependencies()

            # Should handle import errors gracefully
            assert len(results) > 0
            failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
            assert len(failed_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
