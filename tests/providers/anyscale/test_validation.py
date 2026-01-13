"""Tests for Anyscale validation functionality."""

import pytest
from unittest.mock import patch, Mock
import os

from genops.providers.anyscale.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationLevel,
    ValidationCategory,
    AnyscaleValidator,
    validate_setup,
    print_validation_result,
)


class TestValidationIssue:
    """Test ValidationIssue dataclass."""

    def test_validation_issue_creation(self):
        """Test basic validation issue creation."""
        issue = ValidationIssue(
            category=ValidationCategory.CONFIGURATION,
            level=ValidationLevel.ERROR,
            title="API key missing",
            description="ANYSCALE_API_KEY not set"
        )

        assert issue.category == ValidationCategory.CONFIGURATION
        assert issue.level == ValidationLevel.ERROR
        assert issue.title == "API key missing"

    def test_validation_issue_with_fix_suggestion(self):
        """Test validation issue with fix suggestion."""
        issue = ValidationIssue(
            category=ValidationCategory.CONFIGURATION,
            level=ValidationLevel.ERROR,
            title="Test",
            description="Test description",
            fix_suggestion="export ANYSCALE_API_KEY='your-key'"
        )

        assert issue.fix_suggestion == "export ANYSCALE_API_KEY='your-key'"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(
            success=True,
            total_checks=5,
            passed_checks=5,
            issues=[]
        )

        assert result.success is True
        assert result.score == 100.0

    def test_validation_result_partial_success(self):
        """Test partial validation result."""
        result = ValidationResult(
            success=False,
            total_checks=10,
            passed_checks=7,
            issues=[
                ValidationIssue(
                    ValidationCategory.CONNECTIVITY,
                    ValidationLevel.WARNING,
                    "Slow connection",
                    "API response time > 1s"
                )
            ]
        )

        assert result.success is False
        assert result.score == 70.0
        assert len(result.issues) == 1

    def test_validation_result_failure(self):
        """Test failed validation result."""
        result = ValidationResult(
            success=False,
            total_checks=5,
            passed_checks=0,
            issues=[]
        )

        assert result.success is False
        assert result.score == 0.0

    def test_validation_result_score_calculation(self):
        """Test score calculation."""
        result = ValidationResult(
            success=False,
            total_checks=8,
            passed_checks=6,
            issues=[]
        )

        assert result.score == 75.0


class TestValidationLevels:
    """Test ValidationLevel enum."""

    def test_validation_levels_exist(self):
        """Test all validation levels exist."""
        assert ValidationLevel.INFO
        assert ValidationLevel.WARNING
        assert ValidationLevel.ERROR
        assert ValidationLevel.CRITICAL


class TestValidationCategories:
    """Test ValidationCategory enum."""

    def test_validation_categories_exist(self):
        """Test all validation categories exist."""
        assert ValidationCategory.DEPENDENCIES
        assert ValidationCategory.CONFIGURATION
        assert ValidationCategory.CONNECTIVITY
        assert ValidationCategory.MODELS
        assert ValidationCategory.PRICING


class TestAnyscaleValidator:
    """Test AnyscaleValidator class."""

    @patch.dict('os.environ', {'ANYSCALE_API_KEY': 'test-key-123'})
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = AnyscaleValidator()

        assert validator.anyscale_api_key == 'test-key-123'

    def test_validator_with_custom_api_key(self):
        """Test validator with custom API key."""
        validator = AnyscaleValidator(anyscale_api_key="custom-key")

        assert validator.anyscale_api_key == "custom-key"

    @patch.dict('os.environ', {'ANYSCALE_API_KEY': 'test-key'})
    def test_check_configuration_success(self):
        """Test configuration check with valid API key."""
        validator = AnyscaleValidator()

        issues = validator._check_configuration()

        # Should have no critical issues with API key set
        critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
        assert len(critical_issues) == 0

    @patch.dict('os.environ', {}, clear=True)
    def test_check_configuration_missing_api_key(self):
        """Test configuration check with missing API key."""
        validator = AnyscaleValidator(anyscale_api_key=None)

        issues = validator._check_configuration()

        # Should have critical issue for missing API key
        api_key_issues = [
            i for i in issues
            if "API key" in i.title or "ANYSCALE_API_KEY" in i.description
        ]
        assert len(api_key_issues) > 0

    def test_check_dependencies(self):
        """Test dependency checking."""
        validator = AnyscaleValidator()

        issues = validator._check_dependencies()

        # Should check for required packages
        assert isinstance(issues, list)

    @patch('genops.providers.anyscale.validation.requests')
    def test_check_connectivity_success(self, mock_requests):
        """Test connectivity check with successful connection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": []}
        mock_requests.get.return_value = mock_response

        validator = AnyscaleValidator(anyscale_api_key="test-key")

        issues = validator._check_connectivity()

        # Should have no critical connectivity issues
        critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
        assert len(critical_issues) == 0

    @patch('genops.providers.anyscale.validation.requests')
    def test_check_connectivity_failure(self, mock_requests):
        """Test connectivity check with connection failure."""
        mock_requests.get.side_effect = Exception("Connection failed")

        validator = AnyscaleValidator(anyscale_api_key="test-key")

        issues = validator._check_connectivity()

        # Should have connectivity issues
        assert len(issues) > 0

    def test_check_pricing_database(self):
        """Test pricing database validation."""
        validator = AnyscaleValidator()

        issues = validator._check_pricing()

        # Pricing database should be valid
        critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
        assert len(critical_issues) == 0

    @patch.dict('os.environ', {'ANYSCALE_API_KEY': 'test-key'})
    @patch('genops.providers.anyscale.validation.requests')
    def test_validate_full_success(self, mock_requests):
        """Test full validation with all checks passing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": []}
        mock_requests.get.return_value = mock_response

        validator = AnyscaleValidator()

        result = validator.validate()

        assert isinstance(result, ValidationResult)
        assert result.total_checks > 0


class TestValidateSetup:
    """Test validate_setup function."""

    @patch.dict('os.environ', {'ANYSCALE_API_KEY': 'test-key'})
    @patch('genops.providers.anyscale.validation.requests')
    def test_validate_setup_basic(self, mock_requests):
        """Test basic setup validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": []}
        mock_requests.get.return_value = mock_response

        result = validate_setup()

        assert isinstance(result, ValidationResult)

    def test_validate_setup_with_api_key(self):
        """Test setup validation with custom API key."""
        result = validate_setup(anyscale_api_key="custom-key")

        assert isinstance(result, ValidationResult)

    def test_validate_setup_with_base_url(self):
        """Test setup validation with custom base URL."""
        result = validate_setup(
            anyscale_api_key="test-key",
            anyscale_base_url="https://custom.anyscale.com/v1"
        )

        assert isinstance(result, ValidationResult)


class TestPrintValidationResult:
    """Test print_validation_result function."""

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            success=True,
            total_checks=5,
            passed_checks=5,
            issues=[]
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "PASSED" in captured.out or "✅" in captured.out

    def test_print_validation_result_failure(self, capsys):
        """Test printing failed validation result."""
        result = ValidationResult(
            success=False,
            total_checks=5,
            passed_checks=2,
            issues=[
                ValidationIssue(
                    ValidationCategory.CONFIGURATION,
                    ValidationLevel.ERROR,
                    "API key invalid",
                    "Invalid API key format"
                )
            ]
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "FAILED" in captured.out or "❌" in captured.out

    def test_print_validation_result_with_issues(self, capsys):
        """Test printing validation result with issues."""
        result = ValidationResult(
            success=False,
            total_checks=3,
            passed_checks=2,
            issues=[
                ValidationIssue(
                    ValidationCategory.CONNECTIVITY,
                    ValidationLevel.WARNING,
                    "Slow API response",
                    "API response time exceeded 1s",
                    fix_suggestion="Check network connection"
                )
            ]
        )

        print_validation_result(result)

        captured = capsys.readouterr()
        output = captured.out
        assert len(output) > 0


class TestValidationIntegration:
    """Integration tests for validation system."""

    @patch.dict('os.environ', {'ANYSCALE_API_KEY': 'test-key-123'})
    @patch('genops.providers.anyscale.validation.requests')
    def test_end_to_end_validation(self, mock_requests):
        """Test end-to-end validation workflow."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": []}
        mock_requests.get.return_value = mock_response

        # Run validation
        result = validate_setup()

        # Print results
        print_validation_result(result)

        # Verify result structure
        assert isinstance(result, ValidationResult)
        assert result.total_checks > 0
        assert result.passed_checks >= 0
        assert isinstance(result.issues, list)

    @patch.dict('os.environ', {}, clear=True)
    def test_validation_catches_missing_api_key(self):
        """Test validation catches missing API key."""
        result = validate_setup(anyscale_api_key=None)

        assert result.success is False
        assert len(result.issues) > 0

        # Should have issue about missing API key
        api_key_issues = [
            i for i in result.issues
            if "API key" in i.title or "ANYSCALE_API_KEY" in i.description
        ]
        assert len(api_key_issues) > 0
