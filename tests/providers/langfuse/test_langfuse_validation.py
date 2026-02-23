"""Tests for GenOps Langfuse validation utilities."""

import os
import time
from unittest.mock import Mock, patch

import pytest

# Test if Langfuse is available
try:
    import langfuse  # noqa: F401

    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

pytestmark = pytest.mark.skipif(not HAS_LANGFUSE, reason="Langfuse not installed")

from genops.providers.langfuse_validation import (  # noqa: E402
    LangfuseValidationSuite,
    ValidationResult,
    ValidationStatus,
    print_validation_result,
    run_comprehensive_validation,
    validate_genops_integration,
    validate_langfuse_configuration,
    validate_langfuse_connectivity,
    validate_langfuse_installation,
    validate_performance_baseline,
    validate_setup,
)


class TestValidationDataClasses:
    """Test validation data structures."""

    def test_validation_result_creation(self):
        """Test ValidationResult data class."""
        result = ValidationResult(
            test_name="Test Installation",
            status=ValidationStatus.PASSED,
            message="Installation successful",
            details={"version": "2.0.0"},
            fix_suggestion="No action required",
            duration_ms=150.5,
        )

        assert result.test_name == "Test Installation"
        assert result.status == ValidationStatus.PASSED
        assert result.message == "Installation successful"
        assert result.details["version"] == "2.0.0"
        assert result.fix_suggestion == "No action required"
        assert result.duration_ms == 150.5

    def test_langfuse_validation_suite_creation(self):
        """Test LangfuseValidationSuite data class."""
        results = [
            ValidationResult("Test 1", ValidationStatus.PASSED, "Pass"),
            ValidationResult("Test 2", ValidationStatus.FAILED, "Fail"),
        ]

        suite = LangfuseValidationSuite(
            overall_status=ValidationStatus.FAILED,
            test_results=results,
            summary={"passed": 1, "failed": 1},
            recommendations=["Fix test 2"],
            total_duration_ms=300.0,
        )

        assert suite.overall_status == ValidationStatus.FAILED
        assert len(suite.test_results) == 2
        assert suite.summary["passed"] == 1
        assert suite.summary["failed"] == 1
        assert suite.recommendations[0] == "Fix test 2"
        assert suite.total_duration_ms == 300.0


class TestLangfuseInstallationValidation:
    """Test Langfuse installation validation."""

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    def test_validate_installation_success(self):
        """Test successful installation validation."""
        with (
            patch("genops.providers.langfuse_validation.observe"),
            patch("genops.providers.langfuse_validation.StatefulClient"),
        ):
            result = validate_langfuse_installation()

            assert result.status == ValidationStatus.PASSED
            assert "successfully imported" in result.message
            assert result.details["components"] == [
                "Langfuse",
                "observe",
                "StatefulClient",
            ]
            assert result.duration_ms > 0

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", False)
    def test_validate_installation_failure(self):
        """Test installation validation failure."""
        result = validate_langfuse_installation()

        assert result.status == ValidationStatus.FAILED
        assert "package not found" in result.message
        assert "pip install" in result.fix_suggestion
        assert result.duration_ms > 0

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    def test_validate_installation_import_error(self):
        """Test installation validation with import error."""
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            result = validate_langfuse_installation()

            assert result.status == ValidationStatus.FAILED
            assert "import failed" in result.message
            assert "Reinstall Langfuse" in result.fix_suggestion


class TestLangfuseConfigurationValidation:
    """Test Langfuse configuration validation."""

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-test-key-12345",
                "LANGFUSE_SECRET_KEY": "sk-lf-test-secret-67890",
                "LANGFUSE_BASE_URL": "https://test.langfuse.com",
            },
        ):
            result = validate_langfuse_configuration()

            assert result.status == ValidationStatus.PASSED
            assert "configuration valid" in result.message
            assert result.details["public_key_prefix"] == "pk-lf-te..."
            assert result.details["secret_key_prefix"] == "sk-lf-te..."
            assert result.details["base_url"] == "https://test.langfuse.com"

    def test_validate_configuration_missing_keys(self):
        """Test configuration validation with missing keys."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_langfuse_configuration()

            assert result.status == ValidationStatus.FAILED
            assert "Missing required environment variables" in result.message
            assert "LANGFUSE_PUBLIC_KEY" in result.message
            assert "LANGFUSE_SECRET_KEY" in result.message
            assert "export LANGFUSE_PUBLIC_KEY" in result.fix_suggestion

    def test_validate_configuration_missing_public_key(self):
        """Test configuration validation with missing public key only."""
        with patch.dict(
            os.environ, {"LANGFUSE_SECRET_KEY": "sk-lf-test-secret"}, clear=True
        ):
            result = validate_langfuse_configuration()

            assert result.status == ValidationStatus.FAILED
            assert "LANGFUSE_PUBLIC_KEY" in result.message
            assert (
                "LANGFUSE_SECRET_KEY" not in result.message.split(":")[-1]
            )  # Only public key missing

    def test_validate_configuration_wrong_format(self):
        """Test configuration validation with wrong key formats."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "wrong-format-key",
                "LANGFUSE_SECRET_KEY": "also-wrong-format",
            },
        ):
            result = validate_langfuse_configuration()

            assert result.status == ValidationStatus.WARNING
            assert "Configuration issues found" in result.message
            assert "should start with 'pk-lf-'" in result.details["issues"][0]
            assert "should start with 'sk-lf-'" in result.details["issues"][1]
            assert "API key formats" in result.fix_suggestion

    def test_validate_configuration_partial_correct_format(self):
        """Test configuration validation with one correct format."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-correct-format",
                "LANGFUSE_SECRET_KEY": "wrong-secret-format",
            },
        ):
            result = validate_langfuse_configuration()

            assert result.status == ValidationStatus.WARNING
            assert len(result.details["issues"]) == 1
            assert "should start with 'sk-lf-'" in result.details["issues"][0]


class TestLangfuseConnectivityValidation:
    """Test Langfuse connectivity validation."""

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", False)
    def test_validate_connectivity_no_langfuse(self):
        """Test connectivity validation when Langfuse not available."""
        result = validate_langfuse_connectivity()

        assert result.status == ValidationStatus.SKIPPED
        assert "not available for connectivity test" in result.message

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.Langfuse")
    def test_validate_connectivity_success(self, mock_langfuse):
        """Test successful connectivity validation."""
        mock_client = Mock()
        mock_trace = Mock()
        mock_trace.id = "trace-123"
        mock_client.trace.return_value = mock_trace
        mock_client.client.base_url = "https://cloud.langfuse.com"
        mock_langfuse.return_value = mock_client

        result = validate_langfuse_connectivity()

        assert result.status == ValidationStatus.PASSED
        assert "Successfully connected" in result.message
        assert result.details["trace_id"] == "trace-123"
        assert "cloud.langfuse.com" in result.details["host"]
        mock_client.trace.assert_called_once_with(name="genops_validation_test")

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.Langfuse")
    def test_validate_connectivity_unauthorized(self, mock_langfuse):
        """Test connectivity validation with unauthorized error."""
        mock_langfuse.side_effect = Exception("Unauthorized - 401")

        result = validate_langfuse_connectivity()

        assert result.status == ValidationStatus.FAILED
        assert "Failed to connect" in result.message
        assert "Check your Langfuse API keys" in result.fix_suggestion

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.Langfuse")
    def test_validate_connectivity_network_error(self, mock_langfuse):
        """Test connectivity validation with network error."""
        mock_langfuse.side_effect = Exception("Connection timeout")

        result = validate_langfuse_connectivity()

        assert result.status == ValidationStatus.FAILED
        assert "network connectivity" in result.fix_suggestion

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.Langfuse")
    def test_validate_connectivity_generic_error(self, mock_langfuse):
        """Test connectivity validation with generic error."""
        mock_langfuse.side_effect = Exception("Unknown error")

        result = validate_langfuse_connectivity()

        assert result.status == ValidationStatus.FAILED
        assert "Verify Langfuse configuration" in result.fix_suggestion


class TestGenOpsIntegrationValidation:
    """Test GenOps + Langfuse integration validation."""

    @patch("genops.providers.langfuse_validation.GenOpsLangfuseAdapter")
    @patch("genops.providers.langfuse_validation.instrument_langfuse")
    def test_validate_integration_success(self, mock_instrument, mock_adapter_class):
        """Test successful integration validation."""
        mock_adapter = Mock()
        mock_adapter.team = "validation-test"
        mock_adapter.project = "setup-check"
        mock_adapter.enable_governance = True
        mock_adapter_class.return_value = mock_adapter

        result = validate_genops_integration()

        assert result.status == ValidationStatus.PASSED
        assert "integration working correctly" in result.message
        assert result.details["adapter_initialized"] is True
        assert result.details["team"] == "validation-test"
        assert result.details["project"] == "setup-check"
        assert result.details["governance_enabled"] is True

    def test_validate_integration_import_error(self):
        """Test integration validation with import error."""
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            result = validate_genops_integration()

            assert result.status == ValidationStatus.FAILED
            assert "Failed to import" in result.message
            assert "GenOps is properly installed" in result.fix_suggestion

    @patch("genops.providers.langfuse_validation.GenOpsLangfuseAdapter")
    def test_validate_integration_runtime_error(self, mock_adapter_class):
        """Test integration validation with runtime error."""
        mock_adapter_class.side_effect = RuntimeError("Initialization failed")

        result = validate_genops_integration()

        assert result.status == ValidationStatus.FAILED
        assert "integration error" in result.message
        assert result.details["error"] == "Initialization failed"
        assert "GenOps and Langfuse configuration" in result.fix_suggestion


class TestPerformanceValidation:
    """Test performance baseline validation."""

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", False)
    def test_validate_performance_no_langfuse(self):
        """Test performance validation when Langfuse not available."""
        result = validate_performance_baseline()

        assert result.status == ValidationStatus.SKIPPED
        assert "not available for performance testing" in result.message

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.GenOpsLangfuseAdapter")
    def test_validate_performance_success(self, mock_adapter_class):
        """Test successful performance validation."""
        mock_adapter = Mock()
        mock_trace = Mock()

        # Mock context manager behavior
        mock_adapter.trace_with_governance.return_value.__enter__ = Mock(
            return_value=mock_trace
        )
        mock_adapter.trace_with_governance.return_value.__exit__ = Mock(
            return_value=None
        )

        mock_adapter_class.return_value = mock_adapter

        result = validate_performance_baseline()

        assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        assert result.details["initialization_ms"] > 0
        assert result.details["trace_creation_ms"] > 0

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.GenOpsLangfuseAdapter")
    def test_validate_performance_slow_initialization(self, mock_adapter_class):
        """Test performance validation with slow initialization."""

        # Simulate slow initialization
        def slow_init(*args, **kwargs):
            time.sleep(1.1)  # > 1 second
            return Mock()

        mock_adapter_class.side_effect = slow_init

        result = validate_performance_baseline()

        assert result.status == ValidationStatus.WARNING
        assert "Slow initialization" in result.details["issues"][0]

    @patch("genops.providers.langfuse_validation.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse_validation.GenOpsLangfuseAdapter")
    def test_validate_performance_error(self, mock_adapter_class):
        """Test performance validation with error."""
        mock_adapter_class.side_effect = Exception("Performance test failed")

        result = validate_performance_baseline()

        assert result.status == ValidationStatus.FAILED
        assert "Performance testing failed" in result.message


class TestComprehensiveValidation:
    """Test comprehensive validation suite."""

    @patch("genops.providers.langfuse_validation.validate_langfuse_installation")
    @patch("genops.providers.langfuse_validation.validate_langfuse_configuration")
    @patch("genops.providers.langfuse_validation.validate_genops_integration")
    @patch("genops.providers.langfuse_validation.validate_langfuse_connectivity")
    @patch("genops.providers.langfuse_validation.validate_performance_baseline")
    def test_comprehensive_validation_all_passed(
        self, mock_perf, mock_conn, mock_integration, mock_config, mock_install
    ):
        """Test comprehensive validation with all tests passing."""
        # Mock all tests to pass
        for mock_func in [
            mock_install,
            mock_config,
            mock_integration,
            mock_conn,
            mock_perf,
        ]:
            mock_func.return_value = ValidationResult(
                "Test", ValidationStatus.PASSED, "Success"
            )

        suite = run_comprehensive_validation(
            include_performance_tests=True, include_connectivity_tests=True
        )

        assert suite.overall_status == ValidationStatus.PASSED
        assert suite.summary["total_tests"] == 5
        assert suite.summary["passed"] == 5
        assert suite.summary["failed"] == 0
        assert suite.summary["warnings"] == 0
        assert suite.summary["success_rate"] == 1.0
        assert "integration is ready" in suite.recommendations[0]

    @patch("genops.providers.langfuse_validation.validate_langfuse_installation")
    @patch("genops.providers.langfuse_validation.validate_langfuse_configuration")
    @patch("genops.providers.langfuse_validation.validate_genops_integration")
    def test_comprehensive_validation_with_failure(
        self, mock_integration, mock_config, mock_install
    ):
        """Test comprehensive validation with one failure."""
        mock_install.return_value = ValidationResult(
            "Install", ValidationStatus.PASSED, "Success"
        )
        mock_config.return_value = ValidationResult(
            "Config", ValidationStatus.FAILED, "Failed"
        )
        mock_integration.return_value = ValidationResult(
            "Integration", ValidationStatus.PASSED, "Success"
        )

        suite = run_comprehensive_validation(
            include_performance_tests=False, include_connectivity_tests=False
        )

        assert suite.overall_status == ValidationStatus.FAILED
        assert suite.summary["total_tests"] == 3
        assert suite.summary["passed"] == 2
        assert suite.summary["failed"] == 1
        assert suite.summary["warnings"] == 0
        assert "Fix failed validation tests" in suite.recommendations[0]

    @patch("genops.providers.langfuse_validation.validate_langfuse_installation")
    @patch("genops.providers.langfuse_validation.validate_langfuse_configuration")
    @patch("genops.providers.langfuse_validation.validate_genops_integration")
    def test_comprehensive_validation_with_warning(
        self, mock_integration, mock_config, mock_install
    ):
        """Test comprehensive validation with warnings."""
        mock_install.return_value = ValidationResult(
            "Install", ValidationStatus.PASSED, "Success"
        )
        mock_config.return_value = ValidationResult(
            "Config", ValidationStatus.WARNING, "Warning"
        )
        mock_integration.return_value = ValidationResult(
            "Integration", ValidationStatus.PASSED, "Success"
        )

        suite = run_comprehensive_validation(
            include_performance_tests=False, include_connectivity_tests=False
        )

        assert suite.overall_status == ValidationStatus.WARNING
        assert suite.summary["warnings"] == 1
        assert "Review warnings" in suite.recommendations[0]

    @patch("genops.providers.langfuse_validation.validate_langfuse_installation")
    @patch("genops.providers.langfuse_validation.validate_langfuse_configuration")
    @patch("genops.providers.langfuse_validation.validate_genops_integration")
    @patch("genops.providers.langfuse_validation.validate_langfuse_connectivity")
    def test_comprehensive_validation_skipped_tests(
        self, mock_conn, mock_integration, mock_config, mock_install
    ):
        """Test comprehensive validation with skipped tests."""
        mock_install.return_value = ValidationResult(
            "Install", ValidationStatus.PASSED, "Success"
        )
        mock_config.return_value = ValidationResult(
            "Config", ValidationStatus.PASSED, "Success"
        )
        mock_integration.return_value = ValidationResult(
            "Integration", ValidationStatus.PASSED, "Success"
        )
        mock_conn.return_value = ValidationResult(
            "Connectivity", ValidationStatus.SKIPPED, "Skipped"
        )

        suite = run_comprehensive_validation(
            include_performance_tests=False, include_connectivity_tests=True
        )

        assert suite.summary["skipped"] == 1
        assert suite.summary["total_tests"] == 4

    def test_validate_setup_convenience_function(self):
        """Test the convenience validate_setup function."""
        with patch(
            "genops.providers.langfuse_validation.run_comprehensive_validation"
        ) as mock_run:
            mock_suite = LangfuseValidationSuite(overall_status=ValidationStatus.PASSED)
            mock_run.return_value = mock_suite

            result = validate_setup(include_performance_tests=True)

            assert result == mock_suite
            mock_run.assert_called_once_with(
                include_performance_tests=True, include_connectivity_tests=True
            )


class TestValidationPrinting:
    """Test validation result printing functionality."""

    def test_print_single_validation_result(self, capsys):
        """Test printing single validation result."""
        result = ValidationResult(
            test_name="Test Print",
            status=ValidationStatus.PASSED,
            message="Print test success",
            duration_ms=123.5,
            details={"key": "value"},
            fix_suggestion="No fixes needed",
        )

        print_validation_result(result, detailed=True)

        captured = capsys.readouterr()
        assert "✅ Test Print: Print test success (124ms)" in captured.out
        assert "💡 Fix: No fixes needed" in captured.out
        assert "📝 key: value" in captured.out

    def test_print_validation_suite(self, capsys):
        """Test printing validation suite."""
        results = [
            ValidationResult(
                "Test 1", ValidationStatus.PASSED, "Success", duration_ms=100.0
            ),
            ValidationResult(
                "Test 2", ValidationStatus.FAILED, "Failed", duration_ms=200.0
            ),
            ValidationResult(
                "Test 3", ValidationStatus.WARNING, "Warning", duration_ms=150.0
            ),
        ]

        suite = LangfuseValidationSuite(
            overall_status=ValidationStatus.FAILED,
            test_results=results,
            summary={
                "total_tests": 3,
                "passed": 1,
                "failed": 1,
                "warnings": 1,
                "skipped": 0,
                "success_rate": 0.33,
            },
            recommendations=["Fix the failed test", "Review warnings"],
            total_duration_ms=450.0,
        )

        print_validation_result(suite, detailed=False)

        captured = capsys.readouterr()
        assert "GenOps + Langfuse Integration Validation" in captured.out
        assert "❌ Overall Status: FAILED" in captured.out
        assert "Total Tests: 3" in captured.out
        assert "✅ Passed: 1" in captured.out
        assert "❌ Failed: 1" in captured.out
        assert "⚠️  Warnings: 1" in captured.out
        assert "📈 Success Rate: 33.0%" in captured.out
        assert "⏱️  Total Duration: 450ms" in captured.out
        assert "Fix the failed test" in captured.out

    def test_print_validation_suite_detailed(self, capsys):
        """Test printing validation suite with detailed output."""
        result_with_details = ValidationResult(
            test_name="Detailed Test",
            status=ValidationStatus.WARNING,
            message="Warning message",
            details={"detail1": "value1", "detail2": "value2"},
            fix_suggestion="Fix suggestion here",
        )

        suite = LangfuseValidationSuite(
            overall_status=ValidationStatus.WARNING,
            test_results=[result_with_details],
            summary={
                "total_tests": 1,
                "passed": 0,
                "failed": 0,
                "warnings": 1,
                "skipped": 0,
                "success_rate": 0.0,
            },
            recommendations=["Review warnings"],
            total_duration_ms=100.0,
        )

        print_validation_result(suite, detailed=True)

        captured = capsys.readouterr()
        assert "💡 Fix: Fix suggestion here" in captured.out
        assert "📝 detail1: value1" in captured.out
        assert "📝 detail2: value2" in captured.out


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions in validation."""

    def test_validation_with_none_duration(self):
        """Test validation result with None duration."""
        result = ValidationResult(
            test_name="No Duration Test",
            status=ValidationStatus.PASSED,
            message="Success",
            duration_ms=None,
        )

        # Should not crash when printing
        print_validation_result(result)

    def test_validation_with_empty_details(self):
        """Test validation result with empty details."""
        result = ValidationResult(
            test_name="Empty Details Test",
            status=ValidationStatus.PASSED,
            message="Success",
            details={},
        )

        # Should handle empty details gracefully
        print_validation_result(result, detailed=True)

    def test_validation_suite_with_empty_results(self):
        """Test validation suite with no test results."""
        suite = LangfuseValidationSuite(
            overall_status=ValidationStatus.PASSED,
            test_results=[],
            summary={
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "skipped": 0,
                "success_rate": 0.0,
            },
            recommendations=[],
            total_duration_ms=0.0,
        )

        # Should handle empty results gracefully
        print_validation_result(suite)

    def test_validation_with_very_long_duration(self):
        """Test validation with very long duration."""
        result = ValidationResult(
            test_name="Long Duration Test",
            status=ValidationStatus.PASSED,
            message="Success",
            duration_ms=999999.999,
        )

        # Should format large numbers correctly
        print_validation_result(result)

    def test_validation_status_enum_values(self):
        """Test all ValidationStatus enum values."""
        statuses = [
            ValidationStatus.PASSED,
            ValidationStatus.FAILED,
            ValidationStatus.WARNING,
            ValidationStatus.SKIPPED,
        ]

        for status in statuses:
            result = ValidationResult(
                test_name=f"Test {status.value}",
                status=status,
                message=f"Message for {status.value}",
            )

            # Should handle all status types
            assert result.status == status
            print_validation_result(result)
