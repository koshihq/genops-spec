"""Tests for Vercel AI SDK validation module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from genops.providers.vercel_ai_sdk_validation import (
    SetupValidationSummary,
    ValidationResult,
    VercelAISDKValidator,
    print_validation_result,
    quick_validation,
    validate_setup,
)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            check_name="Test Check", passed=True, message="Test passed successfully"
        )

        self.assertEqual(result.check_name, "Test Check")
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Test passed successfully")
        self.assertIsNone(result.details)
        self.assertIsNone(result.fix_suggestion)

    def test_validation_result_with_details(self):
        """Test ValidationResult with details and fix suggestion."""
        details = {"version": "1.0.0", "status": "active"}
        result = ValidationResult(
            check_name="Detailed Check",
            passed=False,
            message="Check failed",
            details=details,
            fix_suggestion="Run: npm install",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.details, details)
        self.assertEqual(result.fix_suggestion, "Run: npm install")


class TestSetupValidationSummary(unittest.TestCase):
    """Test SetupValidationSummary data class."""

    def test_summary_creation(self):
        """Test creating a validation summary."""
        results = [
            ValidationResult("Check 1", True, "Passed"),
            ValidationResult("Check 2", False, "Failed"),
        ]

        summary = SetupValidationSummary(
            all_passed=False,
            total_checks=2,
            passed_checks=1,
            failed_checks=1,
            results=results,
            overall_message="1 check failed",
        )

        self.assertFalse(summary.all_passed)
        self.assertEqual(summary.total_checks, 2)
        self.assertEqual(summary.passed_checks, 1)
        self.assertEqual(summary.failed_checks, 1)
        self.assertEqual(len(summary.results), 2)


class TestVercelAISDKValidator(unittest.TestCase):
    """Test the main validator class."""

    def setUp(self):
        """Set up test environment."""
        self.validator = VercelAISDKValidator()

    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsInstance(self.validator.validation_results, list)
        self.assertEqual(len(self.validator.validation_results), 0)

    @patch("subprocess.run")
    def test_validate_nodejs_success(self, mock_run):
        """Test successful Node.js validation."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "v18.15.0\n"

        self.validator._validate_nodejs()

        self.assertEqual(len(self.validator.validation_results), 1)
        result = self.validator.validation_results[0]
        self.assertEqual(result.check_name, "Node.js Installation")
        self.assertTrue(result.passed)
        self.assertIn("v18.15.0", result.message)

    @patch("subprocess.run")
    def test_validate_nodejs_old_version(self, mock_run):
        """Test Node.js validation with old version."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "v14.15.0\n"  # Too old

        self.validator._validate_nodejs()

        result = self.validator.validation_results[0]
        self.assertFalse(result.passed)
        self.assertIn("too old", result.message)
        self.assertIn("Update Node.js", result.fix_suggestion)

    @patch("subprocess.run")
    def test_validate_nodejs_not_found(self, mock_run):
        """Test Node.js validation when not found."""
        mock_run.side_effect = FileNotFoundError()

        self.validator._validate_nodejs()

        result = self.validator.validation_results[0]
        self.assertFalse(result.passed)
        self.assertIn("not found", result.message)
        self.assertIn("Install Node.js", result.fix_suggestion)

    @patch("subprocess.run")
    def test_validate_nodejs_timeout(self, mock_run):
        """Test Node.js validation timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("node", 10)

        self.validator._validate_nodejs()

        result = self.validator.validation_results[0]
        self.assertFalse(result.passed)
        self.assertIn("timed out", result.message)

    def test_validate_npm_packages_success(self):
        """Test successful npm package validation."""
        package_json_content = {
            "dependencies": {"ai": "^3.0.0", "@ai-sdk/openai": "^0.0.15"},
            "devDependencies": {"@ai-sdk/anthropic": "^0.0.10"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            package_json_path = Path(temp_dir) / "package.json"
            with open(package_json_path, "w") as f:
                json.dump(package_json_content, f)

            # Change to temp directory for test
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                self.validator._validate_npm_packages()
            finally:
                os.chdir(original_cwd)

        # Should have results for AI SDK and providers
        self.assertGreater(len(self.validator.validation_results), 0)

        # Find the Vercel AI SDK result
        ai_sdk_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "Vercel AI SDK Package"
            ),
            None,
        )
        self.assertIsNotNone(ai_sdk_result)
        self.assertTrue(ai_sdk_result.passed)

    def test_validate_npm_packages_no_package_json(self):
        """Test npm package validation without package.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                self.validator._validate_npm_packages()
            finally:
                os.chdir(original_cwd)

        result = self.validator.validation_results[0]
        self.assertFalse(result.passed)
        self.assertIn("No package.json found", result.message)

    def test_validate_npm_packages_invalid_json(self):
        """Test npm package validation with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            package_json_path = Path(temp_dir) / "package.json"
            with open(package_json_path, "w") as f:
                f.write("{ invalid json }")

            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                self.validator._validate_npm_packages()
            finally:
                os.chdir(original_cwd)

        result = self.validator.validation_results[0]
        self.assertFalse(result.passed)
        self.assertIn("Invalid package.json", result.message)

    def test_validate_python_dependencies(self):
        """Test Python dependencies validation."""
        with patch(
            "genops.providers.vercel_ai_sdk_validation.__import__"
        ) as mock_import:
            # Mock successful imports for required packages
            mock_import.return_value = Mock()

            self.validator._validate_python_dependencies()

        # Should have results for multiple packages
        self.assertGreater(len(self.validator.validation_results), 0)

        # Check that we tested required packages
        check_names = [r.check_name for r in self.validator.validation_results]
        self.assertTrue(any("genops" in name for name in check_names))
        self.assertTrue(any("opentelemetry-api" in name for name in check_names))

    def test_validate_python_dependencies_missing(self):
        """Test Python dependencies validation with missing packages."""
        with patch(
            "genops.providers.vercel_ai_sdk_validation.__import__"
        ) as mock_import:
            # Mock import errors for all packages
            mock_import.side_effect = ImportError("No module named 'test'")

            self.validator._validate_python_dependencies()

        # Should have results, some failed for required packages
        required_failures = [
            r
            for r in self.validator.validation_results
            if not r.passed and "(optional)" not in r.check_name
        ]
        self.assertGreater(len(required_failures), 0)

    def test_validate_environment_variables(self):
        """Test environment variables validation."""
        with patch.dict(
            os.environ,
            {
                "GENOPS_TEAM": "test-team",
                "GENOPS_PROJECT": "test-project",
                "GENOPS_ENVIRONMENT": "test",
            },
        ):
            self.validator._validate_environment_variables()

        # Should have results for governance variables
        self.assertGreater(len(self.validator.validation_results), 0)

        # Check governance configuration summary
        governance_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "Governance Configuration"
            ),
            None,
        )
        self.assertIsNotNone(governance_result)
        self.assertTrue(governance_result.passed)

    def test_validate_environment_variables_missing(self):
        """Test environment variables validation with missing variables."""
        # Clear relevant environment variables

        with patch.dict(os.environ, {}, clear=True):
            self.validator._validate_environment_variables()

        # Governance configuration should fail
        governance_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "Governance Configuration"
            ),
            None,
        )
        self.assertIsNotNone(governance_result)
        self.assertFalse(governance_result.passed)

    @patch("genops.providers.vercel_ai_sdk_validation.GenOpsTelemetry")
    def test_validate_genops_configuration_success(self, mock_telemetry):
        """Test successful GenOps configuration validation."""
        mock_telemetry.return_value = Mock()

        self.validator._validate_genops_configuration()

        # Should have telemetry result
        telemetry_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "GenOps Telemetry"
            ),
            None,
        )
        self.assertIsNotNone(telemetry_result)
        self.assertTrue(telemetry_result.passed)

    def test_validate_genops_configuration_import_error(self):
        """Test GenOps configuration validation with import error."""
        with patch(
            "genops.providers.vercel_ai_sdk_validation.GenOpsTelemetry"
        ) as mock_telemetry:
            mock_telemetry.side_effect = ImportError("Module not found")

            self.validator._validate_genops_configuration()

        telemetry_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "GenOps Telemetry"
            ),
            None,
        )
        self.assertIsNotNone(telemetry_result)
        self.assertFalse(telemetry_result.passed)

    def test_validate_provider_access(self):
        """Test AI provider access validation."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"}
        ):
            self.validator._validate_provider_access()

        # Should have provider access result
        provider_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "AI Provider Access"
            ),
            None,
        )
        self.assertIsNotNone(provider_result)
        self.assertTrue(provider_result.passed)
        self.assertIn("OpenAI", provider_result.message)
        self.assertIn("Anthropic", provider_result.message)

    def test_validate_provider_access_no_keys(self):
        """Test provider access validation without API keys."""
        # Clear all API key environment variables

        with patch.dict(os.environ, {}, clear=True):
            self.validator._validate_provider_access()

        provider_result = next(
            (
                r
                for r in self.validator.validation_results
                if r.check_name == "AI Provider Access"
            ),
            None,
        )
        self.assertIsNotNone(provider_result)
        self.assertFalse(provider_result.passed)

    def test_validate_setup_comprehensive(self):
        """Test comprehensive setup validation."""
        with patch.multiple(
            self.validator,
            _validate_nodejs=Mock(),
            _validate_npm_packages=Mock(),
            _validate_python_dependencies=Mock(),
            _validate_environment_variables=Mock(),
            _validate_genops_configuration=Mock(),
            _validate_provider_access=Mock(),
        ):
            summary = self.validator.validate_setup(verbose=False)

        # All validation methods should have been called
        self.validator._validate_nodejs.assert_called_once()
        self.validator._validate_npm_packages.assert_called_once()
        self.validator._validate_python_dependencies.assert_called_once()
        self.validator._validate_environment_variables.assert_called_once()
        self.validator._validate_genops_configuration.assert_called_once()
        self.validator._validate_provider_access.assert_called_once()

        self.assertIsInstance(summary, SetupValidationSummary)

    def test_validate_setup_selective(self):
        """Test selective setup validation."""
        with patch.multiple(
            self.validator,
            _validate_nodejs=Mock(),
            _validate_npm_packages=Mock(),
            _validate_python_dependencies=Mock(),
            _validate_environment_variables=Mock(),
            _validate_genops_configuration=Mock(),
            _validate_provider_access=Mock(),
        ):
            self.validator.validate_setup(
                check_nodejs=False,
                check_npm_packages=False,
                check_provider_access=False,
                verbose=False,
            )

        # Only selected methods should be called
        self.validator._validate_nodejs.assert_not_called()
        self.validator._validate_npm_packages.assert_not_called()
        self.validator._validate_python_dependencies.assert_called_once()
        self.validator._validate_environment_variables.assert_called_once()
        self.validator._validate_genops_configuration.assert_called_once()
        self.validator._validate_provider_access.assert_not_called()

    def test_generate_validation_summary_all_passed(self):
        """Test validation summary generation with all checks passed."""
        self.validator.validation_results = [
            ValidationResult("Check 1", True, "Passed"),
            ValidationResult("Check 2", True, "Passed"),
        ]

        summary = self.validator._generate_validation_summary()

        self.assertTrue(summary.all_passed)
        self.assertEqual(summary.total_checks, 2)
        self.assertEqual(summary.passed_checks, 2)
        self.assertEqual(summary.failed_checks, 0)
        self.assertIn("All validation checks passed", summary.overall_message)

    def test_generate_validation_summary_some_failed(self):
        """Test validation summary generation with some failures."""
        self.validator.validation_results = [
            ValidationResult("Check 1", True, "Passed"),
            ValidationResult("Check 2", False, "Failed"),
        ]

        summary = self.validator._generate_validation_summary()

        self.assertFalse(summary.all_passed)
        self.assertEqual(summary.total_checks, 2)
        self.assertEqual(summary.passed_checks, 1)
        self.assertEqual(summary.failed_checks, 1)
        self.assertIn("1 validation check(s) failed", summary.overall_message)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    @patch("genops.providers.vercel_ai_sdk_validation.validator")
    def test_validate_setup_function(self, mock_validator):
        """Test the validate_setup convenience function."""
        mock_summary = Mock()
        mock_validator.validate_setup.return_value = mock_summary

        result = validate_setup(check_nodejs=True, verbose=False)

        mock_validator.validate_setup.assert_called_once_with(
            check_nodejs=True,
            check_npm_packages=True,
            check_python_deps=True,
            check_environment=True,
            check_genops_config=True,
            check_provider_access=False,
            verbose=False,
        )
        self.assertEqual(result, mock_summary)

    @patch("genops.providers.vercel_ai_sdk_validation.validator")
    def test_quick_validation_function(self, mock_validator):
        """Test the quick_validation convenience function."""
        mock_summary = Mock()
        mock_summary.all_passed = True
        mock_validator.validate_setup.return_value = mock_summary

        result = quick_validation()

        self.assertTrue(result)
        mock_validator.validate_setup.assert_called_once()

    @patch("genops.providers.vercel_ai_sdk_validation.validator")
    def test_print_validation_result_function(self, mock_validator):
        """Test the print_validation_result convenience function."""
        mock_summary = Mock()
        mock_validator._print_validation_summary = Mock()

        print_validation_result(mock_summary)

        mock_validator._print_validation_summary.assert_called_once_with(mock_summary)


class TestValidationIntegration(unittest.TestCase):
    """Test validation integration scenarios."""

    def test_validation_with_mixed_results(self):
        """Test validation with mixed success/failure results."""
        validator = VercelAISDKValidator()

        # Mock some validations to pass, others to fail
        with patch.multiple(
            validator,
            _validate_nodejs=lambda: validator.validation_results.append(
                ValidationResult("Node.js", True, "Found")
            ),
            _validate_npm_packages=lambda: validator.validation_results.append(
                ValidationResult("NPM Packages", False, "Missing AI SDK")
            ),
            _validate_python_dependencies=lambda: validator.validation_results.append(
                ValidationResult("Python Deps", True, "All found")
            ),
            _validate_environment_variables=lambda: validator.validation_results.append(
                ValidationResult("Environment", False, "Missing GENOPS_TEAM")
            ),
            _validate_genops_configuration=lambda: validator.validation_results.append(
                ValidationResult("GenOps Config", True, "Working")
            ),
        ):
            summary = validator.validate_setup(
                check_provider_access=False, verbose=False
            )

        self.assertEqual(summary.total_checks, 5)
        self.assertEqual(summary.passed_checks, 3)
        self.assertEqual(summary.failed_checks, 2)
        self.assertFalse(summary.all_passed)

    def test_validation_error_handling(self):
        """Test validation handles unexpected errors gracefully."""
        validator = VercelAISDKValidator()

        # Mock a validation method to raise an exception
        def failing_validation():
            raise Exception("Unexpected error")

        with patch.object(validator, "_validate_nodejs", failing_validation):
            # Should not raise exception, but should handle gracefully
            try:
                validator.validate_setup(
                    check_npm_packages=False,
                    check_python_deps=False,
                    check_environment=False,
                    check_genops_config=False,
                    check_provider_access=False,
                    verbose=False,
                )
                # If we get here, the error was handled gracefully
            except Exception:
                self.fail("Validation should handle errors gracefully")


class TestValidatorPrintOutput(unittest.TestCase):
    """Test validator print output methods."""

    def test_print_validation_summary(self):
        """Test printing validation summary."""
        validator = VercelAISDKValidator()

        summary = SetupValidationSummary(
            all_passed=False,
            total_checks=3,
            passed_checks=2,
            failed_checks=1,
            results=[
                ValidationResult("Check 1", True, "Passed"),
                ValidationResult("Check 2", True, "Passed"),
                ValidationResult("Check 3", False, "Failed", fix_suggestion="Fix this"),
            ],
            overall_message="1 check failed",
        )

        # This is mainly a smoke test - ensure it doesn't crash
        try:
            # Capture stdout to avoid cluttering test output
            import io
            import sys

            captured_output = io.StringIO()
            sys.stdout = captured_output

            validator._print_validation_summary(summary)

            output = captured_output.getvalue()
            self.assertIn("Validation Summary", output)
            self.assertIn("Total checks: 3", output)
            self.assertIn("Passed: 2", output)
            self.assertIn("Failed: 1", output)

        finally:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    unittest.main()
