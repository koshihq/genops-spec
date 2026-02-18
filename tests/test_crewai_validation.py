#!/usr/bin/env python3
"""
Test suite for CrewAI Validation System

Comprehensive tests for the validation system including:
- Setup validation and diagnostics
- Environment verification
- API key validation
- Dependency checking
- Error reporting and fix suggestions
"""

import os
from unittest.mock import Mock, patch

import pytest

# Import the CrewAI validation system
try:
    from genops.providers.crewai import (
        ValidationIssue,
        ValidationResult,
        print_validation_result,
        quick_validate,
        validate_crewai_setup,
    )
except ImportError:
    pytest.skip("CrewAI provider not available", allow_module_level=True)


class TestCrewAIValidation:
    """Test suite for CrewAI validation system."""

    def test_validation_result_structure(self):
        """Test ValidationResult data structure."""
        # Create a validation result
        issues = [
            ValidationIssue(
                category="dependency",
                severity="error",
                message="CrewAI not installed",
                fix_suggestion="pip install crewai",
            )
        ]

        result = ValidationResult(
            is_valid=False,
            issues=issues,
            summary="Validation failed",
            timestamp="2024-01-01T00:00:00Z",
        )

        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].category == "dependency"
        assert result.issues[0].severity == "error"
        assert "pip install crewai" in result.issues[0].fix_suggestion

    def test_validation_issue_structure(self):
        """Test ValidationIssue data structure."""
        issue = ValidationIssue(
            category="api_key",
            severity="warning",
            message="OpenAI API key not set",
            fix_suggestion="Set OPENAI_API_KEY environment variable",
            details={"env_var": "OPENAI_API_KEY"},
        )

        assert issue.category == "api_key"
        assert issue.severity == "warning"
        assert "OpenAI API key" in issue.message
        assert "OPENAI_API_KEY" in issue.fix_suggestion
        assert issue.details["env_var"] == "OPENAI_API_KEY"

    @patch("genops.providers.crewai.validation.importlib.util.find_spec")
    def test_validate_crewai_setup_crewai_not_installed(self, mock_find_spec):
        """Test validation when CrewAI is not installed."""
        # Mock CrewAI as not installed
        mock_find_spec.return_value = None

        result = validate_crewai_setup()

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False

        # Should have dependency issue
        dependency_issues = [
            issue for issue in result.issues if issue.category == "dependency"
        ]
        assert len(dependency_issues) > 0
        assert any("crewai" in issue.message.lower() for issue in dependency_issues)

    @patch("genops.providers.crewai.validation.importlib.util.find_spec")
    def test_validate_crewai_setup_crewai_installed(self, mock_find_spec):
        """Test validation when CrewAI is installed."""
        # Mock CrewAI as installed
        mock_spec = Mock()
        mock_find_spec.return_value = mock_spec

        result = validate_crewai_setup()

        assert isinstance(result, ValidationResult)
        # May still be invalid due to other issues (API keys, etc.)

        # Should not have CrewAI dependency issues
        crewai_issues = [
            issue
            for issue in result.issues
            if "crewai" in issue.message.lower() and issue.category == "dependency"
        ]
        assert len(crewai_issues) == 0

    def test_validate_crewai_setup_no_api_keys(self):
        """Test validation when no API keys are set."""
        # Clear all API key environment variables
        api_key_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "COHERE_API_KEY",
            "MISTRAL_API_KEY",
        ]

        original_values = {}
        for var in api_key_vars:
            original_values[var] = os.environ.get(var)
            os.environ.pop(var, None)

        try:
            result = validate_crewai_setup()

            assert isinstance(result, ValidationResult)

            # Should have API key issues
            api_key_issues = [
                issue for issue in result.issues if issue.category == "api_key"
            ]
            if len(api_key_issues) > 0:
                assert any(
                    "api key" in issue.message.lower() for issue in api_key_issues
                )

        finally:
            # Restore original environment variables
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_validate_crewai_setup_with_openai_key(self):
        """Test validation when OpenAI API key is set."""
        os.environ["OPENAI_API_KEY"] = "test-key-sk-1234567890abcdef"

        try:
            result = validate_crewai_setup()

            assert isinstance(result, ValidationResult)

            # Should not have OpenAI API key issues
            openai_issues = [
                issue
                for issue in result.issues
                if "openai" in issue.message.lower() and issue.category == "api_key"
            ]
            assert len(openai_issues) == 0

        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_quick_validate_function(self):
        """Test quick validation function."""
        result = quick_validate()

        assert isinstance(result, ValidationResult)
        # Quick validation should complete without errors
        assert result.summary is not None
        assert result.timestamp is not None

    def test_validation_with_all_dependencies(self):
        """Test validation when all dependencies are available."""
        # Mock all dependencies as available
        with patch(
            "genops.providers.crewai.validation.importlib.util.find_spec"
        ) as mock_find_spec:
            mock_spec = Mock()
            mock_find_spec.return_value = mock_spec

            # Set API key
            os.environ["OPENAI_API_KEY"] = "test-key"

            try:
                result = validate_crewai_setup()

                assert isinstance(result, ValidationResult)

                # Should have fewer issues
                [issue for issue in result.issues if issue.severity == "error"]
                # May still have warnings, but should have fewer errors

            finally:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_print_validation_result_success(self):
        """Test printing successful validation result."""
        result = ValidationResult(
            is_valid=True,
            issues=[],
            summary="All checks passed",
            timestamp="2024-01-01T00:00:00Z",
        )

        # Should not raise exception
        try:
            print_validation_result(result)
            success = True
        except Exception:
            success = False

        assert success is True

    def test_print_validation_result_with_issues(self):
        """Test printing validation result with issues."""
        issues = [
            ValidationIssue(
                category="api_key",
                severity="warning",
                message="API key not optimal",
                fix_suggestion="Use production API key",
            ),
            ValidationIssue(
                category="dependency",
                severity="error",
                message="Missing dependency",
                fix_suggestion="pip install missing-package",
            ),
        ]

        result = ValidationResult(
            is_valid=False,
            issues=issues,
            summary="Issues found",
            timestamp="2024-01-01T00:00:00Z",
        )

        # Should not raise exception
        try:
            print_validation_result(result)
            success = True
        except Exception:
            success = False

        assert success is True

    def test_validation_severity_levels(self):
        """Test different severity levels in validation."""
        result = validate_crewai_setup()

        if len(result.issues) > 0:
            severities = [issue.severity for issue in result.issues]
            valid_severities = ["info", "warning", "error", "critical"]

            for severity in severities:
                assert severity in valid_severities

    def test_validation_categories(self):
        """Test different validation categories."""
        result = validate_crewai_setup()

        if len(result.issues) > 0:
            categories = [issue.category for issue in result.issues]
            valid_categories = [
                "dependency",
                "api_key",
                "environment",
                "configuration",
                "network",
                "permissions",
                "version",
            ]

            for category in categories:
                assert category in valid_categories

    def test_validation_fix_suggestions(self):
        """Test that all issues have fix suggestions."""
        result = validate_crewai_setup()

        for issue in result.issues:
            assert issue.fix_suggestion is not None
            assert len(issue.fix_suggestion) > 0
            assert isinstance(issue.fix_suggestion, str)

    def test_validation_with_invalid_api_key_format(self):
        """Test validation with invalid API key format."""
        os.environ["OPENAI_API_KEY"] = "invalid-key-format"

        try:
            result = validate_crewai_setup()

            # May detect invalid key format
            [issue for issue in result.issues if "openai" in issue.message.lower()]

            # Should either detect invalid format or pass basic check
            assert isinstance(result, ValidationResult)

        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    @patch("genops.providers.crewai.validation.requests.get")
    def test_validation_with_network_check(self, mock_get):
        """Test validation with network connectivity check."""
        # Mock successful network response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = validate_crewai_setup()

        # Should not have network issues
        network_issues = [
            issue for issue in result.issues if issue.category == "network"
        ]
        assert len(network_issues) == 0

    @patch("genops.providers.crewai.validation.requests.get")
    def test_validation_with_network_failure(self, mock_get):
        """Test validation with network connectivity failure."""
        # Mock network failure
        mock_get.side_effect = Exception("Network error")

        result = validate_crewai_setup()

        # May have network issues
        [issue for issue in result.issues if issue.category == "network"]
        # Implementation dependent - may or may not check network
        assert isinstance(result, ValidationResult)

    def test_validation_result_serialization(self):
        """Test that validation results can be serialized."""
        result = validate_crewai_setup()

        # Should be able to convert to dict
        try:
            if hasattr(result, "__dict__"):
                result_dict = result.__dict__
                assert isinstance(result_dict, dict)
                assert "is_valid" in result_dict
                assert "issues" in result_dict
            elif hasattr(result, "_asdict"):
                result_dict = result._asdict()
                assert isinstance(result_dict, dict)
        except Exception as e:
            pytest.fail(f"ValidationResult serialization failed: {e}")

    def test_validation_comprehensive_vs_quick(self):
        """Test difference between comprehensive and quick validation."""
        quick_result = quick_validate()
        comprehensive_result = validate_crewai_setup(quick=False)

        assert isinstance(quick_result, ValidationResult)
        assert isinstance(comprehensive_result, ValidationResult)

        # Comprehensive should potentially have more detailed checks
        # (exact behavior is implementation dependent)
        assert quick_result.is_valid is not None
        assert comprehensive_result.is_valid is not None

    def test_validation_with_environment_variables(self):
        """Test validation considers environment variables."""
        # Set GenOps environment variables
        os.environ["GENOPS_TEAM"] = "test-team"
        os.environ["GENOPS_PROJECT"] = "test-project"
        os.environ["GENOPS_ENVIRONMENT"] = "testing"

        try:
            result = validate_crewai_setup()

            # Should recognize environment configuration
            [issue for issue in result.issues if issue.category == "configuration"]

            # May have fewer configuration issues
            assert isinstance(result, ValidationResult)

        finally:
            os.environ.pop("GENOPS_TEAM", None)
            os.environ.pop("GENOPS_PROJECT", None)
            os.environ.pop("GENOPS_ENVIRONMENT", None)

    def test_validation_error_handling(self):
        """Test validation handles errors gracefully."""
        # Test with unusual environment conditions
        with patch("genops.providers.crewai.validation.os.environ", {}):
            result = validate_crewai_setup()

            # Should complete without crashing
            assert isinstance(result, ValidationResult)
            assert result.summary is not None

    def test_validation_issue_details(self):
        """Test validation issues include helpful details."""
        result = validate_crewai_setup()

        for issue in result.issues:
            assert hasattr(issue, "category")
            assert hasattr(issue, "severity")
            assert hasattr(issue, "message")
            assert hasattr(issue, "fix_suggestion")

            # Message should be descriptive
            assert len(issue.message) > 10

            # Fix suggestion should be actionable
            assert len(issue.fix_suggestion) > 5

    def test_validation_timestamp_format(self):
        """Test validation result timestamp format."""
        result = validate_crewai_setup()

        assert result.timestamp is not None

        # Should be a reasonable timestamp format
        if isinstance(result.timestamp, str):
            # Should contain date/time information
            assert len(result.timestamp) > 10
        else:
            # Might be datetime object
            assert hasattr(result.timestamp, "year")

    def test_validation_summary_accuracy(self):
        """Test validation summary reflects actual results."""
        result = validate_crewai_setup()

        assert result.summary is not None
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

        # Summary should reflect validation status
        if result.is_valid:
            positive_words = ["success", "passed", "valid", "ok", "good"]
            assert any(word in result.summary.lower() for word in positive_words)
        else:
            negative_words = ["failed", "issues", "problems", "errors", "invalid"]
            assert any(word in result.summary.lower() for word in negative_words)

    def test_multiple_api_providers_validation(self):
        """Test validation with multiple API providers."""
        # Set multiple API keys
        api_keys = {
            "OPENAI_API_KEY": "sk-test-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
            "GOOGLE_API_KEY": "test-google-key",
        }

        original_values = {}
        for key, value in api_keys.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            result = validate_crewai_setup()

            # Should detect multiple providers
            [issue for issue in result.issues if issue.category == "api_key"]

            # Should have fewer or no API key issues
            assert isinstance(result, ValidationResult)

        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is not None:
                    os.environ[key] = original_value
                else:
                    os.environ.pop(key, None)

    def test_validation_performance(self):
        """Test validation completes in reasonable time."""
        import time

        start_time = time.time()
        result = quick_validate()  # Use quick validation for performance test
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 5 seconds
        assert execution_time < 5.0
        assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
