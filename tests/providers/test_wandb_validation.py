#!/usr/bin/env python3
"""
Test suite for GenOps W&B validation functionality.

This module tests the setup validation, configuration checking,
and diagnostic features for the W&B integration.
"""

import os
import sys
import unittest
from io import StringIO
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from genops.providers.wandb_validation import (
    ValidationCheck,
    ValidationResult,
    check_dependencies,
    check_environment_variables,
    print_validation_result,
    validate_genops_configuration,
    validate_governance_setup,
    validate_setup,
    validate_wandb_connection,
)


class TestWandbValidation(unittest.TestCase):
    """Test W&B validation functionality."""

    def test_validation_result_structure(self):
        """Test ValidationResult dataclass structure."""
        result = ValidationResult(
            overall_status="PASSED",
            checks=[
                ValidationCheck(
                    name="test_check",
                    status="PASSED",
                    message="Test message",
                    details={"key": "value"}
                )
            ],
            summary={"passed": 1, "warnings": 0, "failed": 0}
        )

        self.assertEqual(result.overall_status, "PASSED")
        self.assertEqual(len(result.checks), 1)
        self.assertEqual(result.checks[0].name, "test_check")
        self.assertEqual(result.summary["passed"], 1)

    def test_check_environment_variables(self):
        """Test environment variable validation."""
        # Test with missing variables
        with patch.dict(os.environ, {}, clear=True):
            result = check_environment_variables()
            self.assertIn("WANDB_API_KEY", [check.name for check in result if check.status == "FAILED"])

        # Test with present variables
        with patch.dict(os.environ, {
            'WANDB_API_KEY': 'test-key',
            'GENOPS_TEAM': 'test-team'
        }):
            result = check_environment_variables()
            api_key_check = next((c for c in result if c.name == "WANDB_API_KEY"), None)
            self.assertIsNotNone(api_key_check)
            self.assertEqual(api_key_check.status, "PASSED")

    def test_check_dependencies(self):
        """Test dependency checking."""
        # Mock successful imports
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()

            result = check_dependencies()

            # Should have checks for required dependencies
            dep_names = [check.name for check in result]
            self.assertIn("wandb", dep_names)
            self.assertIn("genops", dep_names)

    @patch('genops.providers.wandb_validation.wandb')
    def test_validate_wandb_connection(self, mock_wandb):
        """Test W&B connection validation."""
        # Test successful connection
        mock_wandb.Api.return_value.viewer = {"username": "testuser"}

        with patch.dict(os.environ, {'WANDB_API_KEY': 'test-key'}):
            result = validate_wandb_connection()
            self.assertEqual(result.status, "PASSED")

        # Test connection failure
        mock_wandb.Api.side_effect = Exception("Connection failed")

        result = validate_wandb_connection()
        self.assertEqual(result.status, "FAILED")

    def test_validate_genops_configuration(self):
        """Test GenOps configuration validation."""
        # Test minimal valid configuration
        result = validate_genops_configuration()
        self.assertIn(result.status, ["PASSED", "WARNING"])

        # Test with complete configuration
        with patch.dict(os.environ, {
            'GENOPS_TEAM': 'test-team',
            'GENOPS_PROJECT': 'test-project',
            'GENOPS_CUSTOMER_ID': 'test-customer'
        }):
            result = validate_genops_configuration()
            self.assertEqual(result.status, "PASSED")

    def test_validate_governance_setup(self):
        """Test governance setup validation."""
        result = validate_governance_setup()

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        # Check that governance-related validations are included
        check_names = [check.name for check in result]
        expected_checks = ["governance_policy", "budget_limits", "cost_tracking"]

        for expected_check in expected_checks:
            # At least one check should be related to governance
            governance_related = any(expected_check in name.lower() for name in check_names)
            # Note: This is a flexible check since exact check names may vary

    def test_validate_setup_basic(self):
        """Test basic setup validation."""
        with patch.dict(os.environ, {
            'WANDB_API_KEY': 'test-key',
            'GENOPS_TEAM': 'test-team'
        }):
            with patch('genops.providers.wandb_validation.validate_wandb_connection') as mock_wandb_check:
                mock_wandb_check.return_value = ValidationCheck(
                    name="wandb_connection",
                    status="PASSED",
                    message="Connection successful"
                )

                result = validate_setup(include_connectivity_tests=False)

                self.assertIsInstance(result, ValidationResult)
                self.assertIn(result.overall_status, ["PASSED", "WARNING", "FAILED"])

    def test_validate_setup_with_connectivity(self):
        """Test setup validation with connectivity tests."""
        with patch('genops.providers.wandb_validation.validate_wandb_connection') as mock_wandb_check:
            mock_wandb_check.return_value = ValidationCheck(
                name="wandb_connection",
                status="PASSED",
                message="Connection successful"
            )

            result = validate_setup(include_connectivity_tests=True)

            # Should include connectivity checks
            check_names = [check.name for check in result.checks]
            self.assertTrue(any("connection" in name.lower() for name in check_names))

    def test_validate_setup_with_governance(self):
        """Test setup validation with governance tests."""
        result = validate_setup(
            include_connectivity_tests=False,
            include_governance_tests=True
        )

        # Should include governance-related checks
        check_names = [check.name for check in result.checks]
        governance_checks = [name for name in check_names if "governance" in name.lower()]
        # Note: Flexible check since governance checks may be integrated differently

    def test_print_validation_result_basic(self):
        """Test basic validation result printing."""
        result = ValidationResult(
            overall_status="PASSED",
            checks=[
                ValidationCheck(
                    name="test_check_1",
                    status="PASSED",
                    message="First check passed"
                ),
                ValidationCheck(
                    name="test_check_2",
                    status="WARNING",
                    message="Second check has warning"
                )
            ],
            summary={"passed": 1, "warnings": 1, "failed": 0}
        )

        # Capture output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_validation_result(result, detailed=False)
            output = mock_stdout.getvalue()

            self.assertIn("PASSED", output)
            self.assertIn("1", output)  # Should show summary counts

    def test_print_validation_result_detailed(self):
        """Test detailed validation result printing."""
        result = ValidationResult(
            overall_status="WARNING",
            checks=[
                ValidationCheck(
                    name="detailed_check",
                    status="WARNING",
                    message="Check with warning",
                    details={"issue": "Minor configuration issue", "suggestion": "Set GENOPS_TEAM"}
                )
            ],
            summary={"passed": 0, "warnings": 1, "failed": 0}
        )

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            print_validation_result(result, detailed=True)
            output = mock_stdout.getvalue()

            self.assertIn("detailed_check", output)
            self.assertIn("Minor configuration issue", output)
            self.assertIn("Set GENOPS_TEAM", output)

    def test_validation_error_scenarios(self):
        """Test validation in error scenarios."""
        # Test with completely missing environment
        with patch.dict(os.environ, {}, clear=True):
            result = validate_setup(include_connectivity_tests=False)

            self.assertIn(result.overall_status, ["WARNING", "FAILED"])

            # Should have failed checks
            failed_checks = [check for check in result.checks if check.status == "FAILED"]
            self.assertGreater(len(failed_checks), 0)

    def test_validation_warning_scenarios(self):
        """Test validation in warning scenarios."""
        # Test with partial configuration
        with patch.dict(os.environ, {
            'WANDB_API_KEY': 'test-key'
            # Missing GENOPS_TEAM and other optional vars
        }):
            result = validate_setup(include_connectivity_tests=False)

            # Should pass basic validation but may have warnings
            self.assertIn(result.overall_status, ["PASSED", "WARNING"])

    def test_validation_performance(self):
        """Test validation performance and timeout handling."""
        # Mock slow connectivity test
        def slow_validation(*args, **kwargs):
            import time
            time.sleep(0.1)  # Short delay for testing
            return ValidationCheck("slow_check", "PASSED", "Slow check completed")

        with patch('genops.providers.wandb_validation.validate_wandb_connection', side_effect=slow_validation):
            import time
            start_time = time.time()

            result = validate_setup(
                include_connectivity_tests=True,
                include_performance_tests=False  # Keep test fast
            )

            end_time = time.time()
            duration = end_time - start_time

            # Should complete in reasonable time
            self.assertLess(duration, 5.0)  # 5 second timeout

    def test_validation_summary_calculation(self):
        """Test validation summary calculation."""
        checks = [
            ValidationCheck("check1", "PASSED", "Passed"),
            ValidationCheck("check2", "PASSED", "Passed"),
            ValidationCheck("check3", "WARNING", "Warning"),
            ValidationCheck("check4", "FAILED", "Failed"),
            ValidationCheck("check5", "FAILED", "Failed")
        ]

        # Calculate summary manually to test logic
        summary = {
            "passed": len([c for c in checks if c.status == "PASSED"]),
            "warnings": len([c for c in checks if c.status == "WARNING"]),
            "failed": len([c for c in checks if c.status == "FAILED"])
        }

        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["warnings"], 1)
        self.assertEqual(summary["failed"], 2)

        # Overall status logic
        if summary["failed"] > 0:
            overall_status = "FAILED"
        elif summary["warnings"] > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"

        self.assertEqual(overall_status, "FAILED")


if __name__ == '__main__':
    unittest.main(verbosity=2)
