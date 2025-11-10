#!/usr/bin/env python3
"""
Comprehensive test suite for Haystack validation functionality.

Tests cover validation framework, environment checks, dependency validation,
and diagnostic systems as required by CLAUDE.md standards.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from typing import Dict, List, Optional

from genops.providers.haystack_validation import (
    ValidationResult,
    ValidationIssue,
    validate_haystack_setup,
    print_validation_result,
    validate_python_environment,
    validate_haystack_installation,
    validate_genops_installation,
    validate_ai_providers,
    validate_opentelemetry_setup,
    benchmark_performance
)


class TestValidationIssue:
    """Validation issue data structure tests."""

    def test_validation_issue_creation(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            severity="error",
            category="dependency",
            message="Test error message",
            fix_suggestion="Fix by doing X",
            documentation_link="https://docs.example.com"
        )
        
        assert issue.severity == "error"
        assert issue.category == "dependency"
        assert issue.message == "Test error message"
        assert issue.fix_suggestion == "Fix by doing X"
        assert issue.documentation_link == "https://docs.example.com"

    def test_validation_issue_without_docs_link(self):
        """Test validation issue without documentation link."""
        issue = ValidationIssue(
            severity="warning",
            category="configuration",
            message="Warning message",
            fix_suggestion="Fix warning"
        )
        
        assert issue.documentation_link is None


class TestValidationResult:
    """Validation result data structure tests."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            is_valid=True,
            overall_score=0.95,
            python_version="3.9.0",
            platform="linux",
            haystack_version="2.0.0",
            genops_version="1.0.0"
        )
        
        assert result.is_valid is True
        assert result.overall_score == 0.95
        assert result.python_version == "3.9.0"
        assert result.platform == "linux"
        assert result.haystack_version == "2.0.0"
        assert result.genops_version == "1.0.0"
        assert len(result.issues) == 0

    def test_validation_result_add_issue(self):
        """Test adding issues to validation result."""
        result = ValidationResult(is_valid=True, overall_score=1.0)
        
        result.add_issue(
            severity="error",
            category="dependency",
            message="Test error",
            fix_suggestion="Fix it"
        )
        
        assert len(result.issues) == 1
        assert result.dependencies_valid is False
        
        issue = result.issues[0]
        assert issue.severity == "error"
        assert issue.category == "dependency"
        assert issue.message == "Test error"

    def test_validation_result_issue_categorization(self):
        """Test validation result categorizes issues correctly."""
        result = ValidationResult(is_valid=True, overall_score=1.0)
        
        # Add configuration error
        result.add_issue("error", "configuration", "Config error", "Fix config")
        assert result.configuration_valid is False
        
        # Add connectivity error
        result.add_issue("error", "connectivity", "Connection error", "Fix connection")
        assert result.connectivity_valid is False
        
        # Add performance error
        result.add_issue("error", "performance", "Performance error", "Fix performance")
        assert result.performance_acceptable is False

    def test_validation_result_error_counts(self):
        """Test validation result error/warning counts."""
        result = ValidationResult(is_valid=True, overall_score=1.0)
        
        result.add_issue("error", "dependency", "Error 1", "Fix 1")
        result.add_issue("error", "configuration", "Error 2", "Fix 2")
        result.add_issue("warning", "dependency", "Warning 1", "Fix warning")
        result.add_issue("info", "configuration", "Info 1", "Just FYI")
        
        assert result.get_error_count() == 2
        assert result.get_warning_count() == 1


class TestPythonEnvironmentValidation:
    """Python environment validation tests."""

    def test_validate_python_version_current(self):
        """Test validation with current Python version."""
        valid, issues = validate_python_environment()
        
        # Should be valid for current environment
        assert valid is True
        # May have warnings but should not have errors for supported versions

    @patch('genops.providers.haystack_validation.sys.version_info', (3, 7, 0))
    def test_validate_python_version_too_old(self):
        """Test validation with old Python version."""
        valid, issues = validate_python_environment()
        
        assert valid is False
        assert len(issues) > 0
        
        error_issue = next(issue for issue in issues if issue.severity == "error")
        assert "too old" in error_issue.message
        assert "Upgrade to Python 3.8" in error_issue.fix_suggestion

    @patch('genops.providers.haystack_validation.sys.version_info', (3, 8, 5))
    def test_validate_python_version_minimum(self):
        """Test validation with minimum supported Python version."""
        valid, issues = validate_python_environment()
        
        assert valid is True
        # May have warning about upgrading to 3.9+
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        if warning_issues:
            assert "3.9+" in warning_issues[0].message


class TestHaystackInstallationValidation:
    """Haystack installation validation tests."""

    @patch('genops.providers.haystack_validation.importlib.import_module')
    def test_validate_haystack_installed(self, mock_import):
        """Test validation with Haystack installed."""
        # Mock Haystack module
        mock_haystack = Mock()
        mock_haystack.__version__ = "2.1.0"
        mock_import.return_value = mock_haystack
        
        with patch('genops.providers.haystack_validation.haystack', mock_haystack):
            valid, issues, version = validate_haystack_installation()
        
        assert valid is True
        assert version == "2.1.0"
        assert len([issue for issue in issues if issue.severity == "error"]) == 0

    def test_validate_haystack_not_installed(self):
        """Test validation without Haystack installed."""
        with patch('genops.providers.haystack_validation.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'haystack'")
            
            valid, issues, version = validate_haystack_installation()
        
        assert valid is False
        assert version is None
        assert len(issues) > 0
        
        error_issue = next(issue for issue in issues if issue.severity == "error")
        assert "not installed" in error_issue.message
        assert "pip install haystack-ai" in error_issue.fix_suggestion

    @patch('genops.providers.haystack_validation.importlib.import_module')
    def test_validate_haystack_old_version(self, mock_import):
        """Test validation with old Haystack version."""
        mock_haystack = Mock()
        mock_haystack.__version__ = "1.5.0"
        mock_import.return_value = mock_haystack
        
        with patch('genops.providers.haystack_validation.haystack', mock_haystack):
            valid, issues, version = validate_haystack_installation()
        
        assert valid is True  # Still valid but may have warnings
        assert version == "1.5.0"
        
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        if warning_issues:
            assert "older" in warning_issues[0].message

    @patch('genops.providers.haystack_validation.importlib.import_module')
    def test_validate_haystack_core_import_failure(self, mock_import):
        """Test validation when Haystack core imports fail."""
        # Mock Haystack available but core imports fail
        mock_haystack = Mock()
        mock_haystack.__version__ = "2.0.0"
        
        def side_effect(module_name):
            if module_name == 'haystack':
                return mock_haystack
            elif 'Pipeline' in module_name or 'Component' in module_name:
                raise ImportError("Core import failed")
            return Mock()
        
        mock_import.side_effect = side_effect
        
        with patch('genops.providers.haystack_validation.haystack', mock_haystack):
            valid, issues, version = validate_haystack_installation()
        
        assert valid is False
        assert len(issues) > 0
        
        error_issue = next(issue for issue in issues if issue.severity == "error")
        assert "core imports failed" in error_issue.message


class TestGenOpsInstallationValidation:
    """GenOps installation validation tests."""

    def test_validate_genops_installed(self):
        """Test validation with GenOps installed."""
        # This test runs against the actual installation
        valid, issues, version = validate_genops_installation()
        
        # Should be valid since we're testing the actual code
        assert valid is True
        # Version might be unknown but should not be None
        assert version is not None

    @patch('genops.providers.haystack_validation.importlib.import_module')
    def test_validate_genops_not_installed(self, mock_import):
        """Test validation without GenOps installed."""
        mock_import.side_effect = ImportError("No module named 'genops'")
        
        valid, issues, version = validate_genops_installation()
        
        assert valid is False
        assert version is None
        assert len(issues) > 0
        
        error_issue = next(issue for issue in issues if issue.severity == "error")
        assert "not installed" in error_issue.message
        assert "pip install genops-ai" in error_issue.fix_suggestion

    @patch('genops.providers.haystack_validation.importlib.import_module')
    def test_validate_genops_haystack_integration_missing(self, mock_import):
        """Test validation when GenOps Haystack integration is missing."""
        def side_effect(module_name):
            if module_name.startswith('genops.providers.haystack'):
                raise ImportError("Haystack integration not found")
            return Mock()
        
        mock_import.side_effect = side_effect
        
        valid, issues, version = validate_genops_installation()
        
        assert valid is False
        assert len(issues) > 0
        
        error_issue = next(issue for issue in issues if issue.severity == "error")
        assert "Haystack integration" in error_issue.message
        assert "genops-ai[haystack]" in error_issue.fix_suggestion


class TestAIProvidersValidation:
    """AI providers validation tests."""

    def test_validate_providers_no_keys(self):
        """Test provider validation with no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            provider_status, issues = validate_ai_providers()
        
        # All providers should be unavailable
        for provider, status in provider_status.items():
            assert status["status"] == "unavailable"
            assert status["api_key_configured"] is False
        
        # Should have warning about no providers configured
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        assert any("No AI providers" in issue.message for issue in warning_issues)

    def test_validate_providers_with_openai_key(self):
        """Test provider validation with OpenAI key."""
        env_vars = {"OPENAI_API_KEY": "sk-test123456789"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            with patch('genops.providers.haystack_validation.importlib.import_module') as mock_import:
                mock_import.return_value = Mock()  # Mock OpenAI library
                
                provider_status, issues = validate_ai_providers()
        
        openai_status = provider_status["openai"]
        assert openai_status["api_key_configured"] is True
        assert openai_status["key_format_valid"] is True
        assert openai_status["library_installed"] is True
        assert openai_status["status"] == "available"

    def test_validate_providers_invalid_key_format(self):
        """Test provider validation with invalid key format."""
        env_vars = {"OPENAI_API_KEY": "invalid-key-format"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            provider_status, issues = validate_ai_providers()
        
        openai_status = provider_status["openai"]
        assert openai_status["api_key_configured"] is True
        assert openai_status["key_format_valid"] is False
        
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        assert any("key format appears invalid" in issue.message for issue in warning_issues)

    def test_validate_providers_key_no_library(self):
        """Test provider validation with API key but missing library."""
        env_vars = {"ANTHROPIC_API_KEY": "test-anthropic-key"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            with patch('genops.providers.haystack_validation.importlib.import_module') as mock_import:
                mock_import.side_effect = ImportError("No module named 'anthropic'")
                
                provider_status, issues = validate_ai_providers()
        
        anthropic_status = provider_status["anthropic"]
        assert anthropic_status["api_key_configured"] is True
        assert anthropic_status["library_installed"] is False
        assert anthropic_status["status"] == "key_only"
        
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        assert any("API key found but library not installed" in issue.message for issue in warning_issues)

    def test_validate_providers_multiple_configured(self):
        """Test provider validation with multiple providers configured."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123",
            "ANTHROPIC_API_KEY": "test-anthropic",
            "COHERE_API_KEY": "test-cohere"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with patch('genops.providers.haystack_validation.importlib.import_module') as mock_import:
                mock_import.return_value = Mock()  # Mock all libraries
                
                provider_status, issues = validate_ai_providers()
        
        available_count = sum(1 for status in provider_status.values() if status["status"] == "available")
        assert available_count >= 3  # At least the three we configured


class TestOpenTelemetryValidation:
    """OpenTelemetry validation tests."""

    def test_validate_opentelemetry_success(self):
        """Test OpenTelemetry validation success."""
        # This should work in the actual environment
        valid, issues = validate_opentelemetry_setup()
        
        # Should be valid in our test environment
        assert valid is True
        assert len([issue for issue in issues if issue.severity == "error"]) == 0

    @patch('genops.providers.haystack_validation.importlib.import_module')
    def test_validate_opentelemetry_not_installed(self, mock_import):
        """Test OpenTelemetry validation when not installed."""
        mock_import.side_effect = ImportError("No module named 'opentelemetry'")
        
        valid, issues = validate_opentelemetry_setup()
        
        assert valid is False
        assert len(issues) > 0
        
        error_issue = next(issue for issue in issues if issue.severity == "error")
        assert "not properly installed" in error_issue.message
        assert "pip install opentelemetry" in error_issue.fix_suggestion

    @patch('genops.providers.haystack_validation.trace.get_tracer')
    def test_validate_opentelemetry_tracer_failure(self, mock_get_tracer):
        """Test OpenTelemetry validation when tracer fails."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(side_effect=Exception("Tracer failed"))
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        valid, issues = validate_opentelemetry_setup()
        
        assert valid is False
        assert len(issues) > 0
        
        warning_issue = next(issue for issue in issues if issue.severity == "warning")
        assert "basic test failed" in warning_issue.message


class TestPerformanceBenchmarking:
    """Performance benchmarking tests."""

    def test_benchmark_performance_success(self):
        """Test performance benchmarking success."""
        metrics, issues = benchmark_performance()
        
        assert isinstance(metrics, dict)
        assert "import_time_ms" in metrics
        assert metrics["import_time_ms"] > 0
        
        # Should not have performance errors for normal operation
        error_issues = [issue for issue in issues if issue.severity == "error"]
        assert len(error_issues) == 0

    @patch('genops.providers.haystack_validation.GenOpsHaystackAdapter')
    def test_benchmark_performance_slow_import(self, mock_adapter_class):
        """Test performance benchmarking with slow imports."""
        # Mock slow import
        import time
        original_import_time = time.perf_counter
        call_count = 0
        
        def mock_perf_counter():
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Start time
                return 0.0
            elif call_count == 2:  # After import
                return 0.6  # 600ms - should trigger warning
            else:
                return 0.7  # Subsequent calls
        
        with patch('genops.providers.haystack_validation.time.perf_counter', mock_perf_counter):
            metrics, issues = benchmark_performance()
        
        assert metrics["import_time_ms"] == 600.0
        
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        assert any("Slow import time" in issue.message for issue in warning_issues)

    @patch('genops.providers.haystack_validation.GenOpsHaystackAdapter')
    def test_benchmark_performance_import_failure(self, mock_adapter_class):
        """Test performance benchmarking with import failure."""
        mock_adapter_class.side_effect = ImportError("Import failed")
        
        metrics, issues = benchmark_performance()
        
        error_issues = [issue for issue in issues if issue.severity == "error"]
        assert any("Import benchmark failed" in issue.message for issue in error_issues)

    @patch('genops.providers.haystack_validation.GenOpsHaystackAdapter')
    def test_benchmark_performance_slow_instantiation(self, mock_adapter_class):
        """Test performance benchmarking with slow instantiation."""
        # Mock slow instantiation
        call_count = 0
        
        def mock_perf_counter():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Import phase
                return call_count * 0.05  # Fast import
            elif call_count == 3:  # Instantiation start
                return 0.1
            elif call_count == 4:  # Instantiation end
                return 0.25  # 150ms - should trigger warning
            else:
                return 0.3
        
        with patch('genops.providers.haystack_validation.time.perf_counter', mock_perf_counter):
            metrics, issues = benchmark_performance()
        
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        assert any("Slow adapter creation" in issue.message for issue in warning_issues)


class TestValidateHaystackSetup:
    """Main validation function tests."""

    def test_validate_haystack_setup_success(self):
        """Test main validation function with successful setup."""
        result = validate_haystack_setup()
        
        assert isinstance(result, ValidationResult)
        assert result.python_version is not None
        assert result.platform is not None
        
        # Should have some validation time
        assert result.validation_time_ms > 0

    @patch('genops.providers.haystack_validation.validate_python_environment')
    def test_validate_haystack_setup_python_failure(self, mock_validate_python):
        """Test main validation with Python environment failure."""
        mock_validate_python.return_value = (False, [
            ValidationIssue("error", "dependency", "Python too old", "Upgrade Python")
        ])
        
        result = validate_haystack_setup()
        
        assert result.is_valid is False
        assert result.dependencies_valid is False
        assert result.get_error_count() > 0

    @patch('genops.providers.haystack_validation.validate_haystack_installation')
    def test_validate_haystack_setup_haystack_failure(self, mock_validate_haystack):
        """Test main validation with Haystack installation failure."""
        mock_validate_haystack.return_value = (False, [
            ValidationIssue("error", "dependency", "Haystack not found", "Install Haystack")
        ], None)
        
        result = validate_haystack_setup()
        
        assert result.is_valid is False
        assert result.dependencies_valid is False
        assert result.haystack_version is None

    def test_validate_haystack_setup_score_calculation(self):
        """Test validation score calculation."""
        result = validate_haystack_setup()
        
        # Score should be between 0 and 1
        assert 0.0 <= result.overall_score <= 1.0
        
        # If valid, score should be high
        if result.is_valid:
            assert result.overall_score >= 0.7

    def test_validate_haystack_setup_recommendations(self):
        """Test validation generates appropriate recommendations."""
        result = validate_haystack_setup()
        
        assert isinstance(result.recommendations, list)
        
        if result.get_error_count() == 0 and result.get_warning_count() == 0:
            assert any("optimal" in rec.lower() for rec in result.recommendations)
        elif result.get_error_count() == 0:
            assert any("functional" in rec.lower() for rec in result.recommendations)


class TestPrintValidationResult:
    """Validation result printing tests."""

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True,
            overall_score=0.95,
            python_version="3.9.0",
            platform="linux",
            haystack_version="2.0.0",
            genops_version="1.0.0",
            available_providers=["OpenAI integration", "Anthropic integration"],
            import_time_ms=150.0,
            validation_time_ms=500.0
        )
        
        print_validation_result(result)
        
        captured = capsys.readouterr()
        assert "✅ Haystack + GenOps Setup Validation" in captured.out
        assert "95.0%" in captured.out
        assert "Python: 3.9.0" in captured.out
        assert "Haystack: 2.0.0" in captured.out
        assert "GenOps: 1.0.0" in captured.out
        assert "Available AI Providers" in captured.out

    def test_print_validation_result_with_errors(self, capsys):
        """Test printing validation result with errors."""
        result = ValidationResult(
            is_valid=False,
            overall_score=0.3,
            python_version="3.7.0",
            platform="linux"
        )
        
        result.add_issue(
            severity="error",
            category="dependency",
            message="Python version too old",
            fix_suggestion="Upgrade to Python 3.8+",
            documentation_link="https://docs.python.org"
        )
        
        print_validation_result(result)
        
        captured = capsys.readouterr()
        assert "❌ Haystack + GenOps Setup Issues Found" in captured.out
        assert "30.0%" in captured.out
        assert "🚨 Errors (1):" in captured.out
        assert "Python version too old" in captured.out
        assert "Fix: Upgrade to Python 3.8+" in captured.out
        assert "Docs: https://docs.python.org" in captured.out

    def test_print_validation_result_with_warnings(self, capsys):
        """Test printing validation result with warnings."""
        result = ValidationResult(
            is_valid=True,
            overall_score=0.8,
            python_version="3.8.5",
            platform="darwin"
        )
        
        result.add_issue(
            severity="warning",
            category="configuration",
            message="Consider upgrading Python",
            fix_suggestion="Upgrade to Python 3.9+"
        )
        
        print_validation_result(result)
        
        captured = capsys.readouterr()
        assert "⚠️ Warnings (1):" in captured.out
        assert "Consider upgrading Python" in captured.out
        assert "Suggestion: Upgrade to Python 3.9+" in captured.out

    def test_print_validation_result_ready_state(self, capsys):
        """Test printing ready validation result."""
        result = ValidationResult(
            is_valid=True,
            overall_score=1.0,
            python_version="3.9.0",
            available_providers=["OpenAI integration"]
        )
        
        result.recommendations = ["Setup is optimal! You're ready to build with Haystack + GenOps"]
        
        print_validation_result(result)
        
        captured = capsys.readouterr()
        assert "🚀 You're ready! Try:" in captured.out
        assert "from genops.providers.haystack import auto_instrument" in captured.out
        assert "auto_instrument()" in captured.out

    def test_print_validation_result_needs_fixes(self, capsys):
        """Test printing validation result that needs fixes."""
        result = ValidationResult(
            is_valid=False,
            overall_score=0.5
        )
        
        result.add_issue("error", "dependency", "Missing dependency", "Install it")
        
        print_validation_result(result)
        
        captured = capsys.readouterr()
        assert "🔧 Next steps:" in captured.out
        assert "1. Fix the errors listed above" in captured.out
        assert "2. Re-run validation" in captured.out