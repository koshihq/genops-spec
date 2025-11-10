"""Tests for GenOps Cohere validation system."""

import os
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

# Test imports
from genops.providers.cohere_validation import (
    CohereValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationLevel,
    ValidationResult,
    print_validation_result,
    quick_validate,
    validate_setup,
)


@dataclass
class MockCohereClient:
    """Mock Cohere client for testing."""
    api_key: str = "test-key"

    def __init__(self, api_key: str = "test-key"):
        self.api_key = api_key

    def check_api_key(self):
        """Mock API key check."""
        if self.api_key == "invalid-key":
            raise Exception("Unauthorized")
        return {"valid": True}


class TestCohereValidator:
    """Test suite for CohereValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return CohereValidator(api_key="test-api-key")

    def test_validator_initialization(self):
        """Test validator initialization with various configurations."""
        # Basic initialization
        validator = CohereValidator()
        assert validator.api_key is None
        assert validator.include_performance_tests is False

        # With API key
        validator = CohereValidator(api_key="test-key")
        assert validator.api_key == "test-key"

        # With performance tests
        validator = CohereValidator(include_performance_tests=True)
        assert validator.include_performance_tests is True

    def test_api_key_from_environment(self):
        """Test API key loading from environment variable."""
        with patch.dict(os.environ, {'CO_API_KEY': 'env-api-key'}):
            validator = CohereValidator()
            assert validator.api_key == "env-api-key"

    def test_validate_dependencies_success(self, validator):
        """Test successful dependency validation."""
        with patch('genops.providers.cohere_validation.HAS_COHERE', True):
            with patch('genops.providers.cohere_validation.ClientV2', MockCohereClient):
                result = validator.validate_all()

                # Should have no critical dependency issues
                dependency_issues = [
                    issue for issue in result.issues
                    if issue.category == ValidationCategory.DEPENDENCIES
                    and issue.level == ValidationLevel.CRITICAL
                ]
                assert len(dependency_issues) == 0

    def test_validate_dependencies_missing_client(self, validator):
        """Test dependency validation when Cohere client is missing."""
        with patch('genops.providers.cohere_validation.HAS_COHERE', False):
            result = validator.validate_all()

            # Should have critical dependency issue
            dependency_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.DEPENDENCIES
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(dependency_issues) > 0

            critical_issue = dependency_issues[0]
            assert "cohere" in critical_issue.title.lower()
            assert "pip install cohere" in critical_issue.fix_suggestion

    def test_validate_authentication_success(self, validator):
        """Test successful authentication validation."""
        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = MockCohereClient()
            mock_client_class.return_value = mock_client

            result = validator.validate_all()

            # Should have no critical auth issues
            auth_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.AUTHENTICATION
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(auth_issues) == 0

    def test_validate_authentication_invalid_key(self, validator):
        """Test authentication validation with invalid API key."""
        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = MockCohereClient(api_key="invalid-key")
            mock_client_class.return_value = mock_client
            mock_client.check_api_key.side_effect = Exception("Unauthorized")

            result = validator.validate_all()

            # Should have critical auth issue
            auth_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.AUTHENTICATION
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(auth_issues) > 0

            critical_issue = auth_issues[0]
            assert "api key" in critical_issue.title.lower()
            assert "CO_API_KEY" in critical_issue.fix_suggestion

    def test_validate_authentication_missing_key(self):
        """Test authentication validation when API key is missing."""
        validator = CohereValidator(api_key=None)

        with patch.dict(os.environ, {}, clear=True):
            result = validator.validate_all()

            # Should have critical auth issue
            auth_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.AUTHENTICATION
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(auth_issues) > 0

            critical_issue = auth_issues[0]
            assert "not found" in critical_issue.title.lower()

    def test_validate_connectivity_success(self, validator):
        """Test successful connectivity validation."""
        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = Mock()

            result = validator.validate_all()

            # Should have successful connectivity
            connectivity_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.CONNECTIVITY
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(connectivity_issues) == 0

    def test_validate_connectivity_network_error(self, validator):
        """Test connectivity validation with network error."""
        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.side_effect = Exception("Connection timeout")

            result = validator.validate_all()

            # Should have connectivity issue
            connectivity_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.CONNECTIVITY
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(connectivity_issues) > 0

            critical_issue = connectivity_issues[0]
            assert "connectivity" in critical_issue.title.lower() or "connection" in critical_issue.title.lower()

    def test_validate_models_success(self, validator):
        """Test successful model validation."""
        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = Mock()
            mock_client.embed.return_value = Mock()
            mock_client.rerank.return_value = Mock()

            result = validator.validate_all()

            # Should have no critical model issues
            model_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.MODELS
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(model_issues) == 0

    def test_validate_models_unsupported_model(self, validator):
        """Test model validation with unsupported model."""
        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.side_effect = Exception("Model not found")

            result = validator.validate_all()

            # Should have model access warning
            model_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.MODELS
            ]
            assert len(model_issues) > 0

    def test_validate_performance_tests(self):
        """Test performance validation when enabled."""
        validator = CohereValidator(
            api_key="test-key",
            include_performance_tests=True
        )

        with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock response with timing
            mock_response = Mock()
            mock_client.chat.return_value = mock_response

            with patch('time.time', side_effect=[0.0, 0.5]):  # 500ms response
                result = validator.validate_all()

            assert result.performance_metrics is not None
            assert "chat_latency" in result.performance_metrics

    def test_validate_pricing_calculator(self, validator):
        """Test pricing calculator validation."""
        with patch('genops.providers.cohere_validation.CohereCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.calculate_cost.return_value = (0.001, 0.002, 0.0)

            result = validator.validate_all()

            # Should have no critical pricing issues
            pricing_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.PRICING
                and issue.level == ValidationLevel.CRITICAL
            ]
            assert len(pricing_issues) == 0

    def test_validate_pricing_calculation_error(self, validator):
        """Test pricing validation with calculation error."""
        with patch('genops.providers.cohere_validation.CohereCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.calculate_cost.side_effect = Exception("Pricing error")

            result = validator.validate_all()

            # Should have pricing warning
            pricing_issues = [
                issue for issue in result.issues
                if issue.category == ValidationCategory.PRICING
            ]
            assert len(pricing_issues) > 0


class TestValidationStructures:
    """Test validation data structures."""

    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            title="Test Issue",
            description="Test description",
            level=ValidationLevel.WARNING,
            category=ValidationCategory.CONNECTIVITY,
            fix_suggestion="Fix it"
        )

        assert issue.title == "Test Issue"
        assert issue.level == ValidationLevel.WARNING
        assert issue.category == ValidationCategory.CONNECTIVITY
        assert issue.fix_suggestion == "Fix it"

    def test_validation_result_success(self):
        """Test ValidationResult for successful validation."""
        result = ValidationResult(
            success=True,
            issues=[],
            performance_metrics={"test": 100.0}
        )

        assert result.success is True
        assert len(result.issues) == 0
        assert result.has_critical_issues is False
        assert result.performance_metrics["test"] == 100.0

    def test_validation_result_with_critical_issues(self):
        """Test ValidationResult with critical issues."""
        critical_issue = ValidationIssue(
            title="Critical Issue",
            description="Critical problem",
            level=ValidationLevel.CRITICAL,
            category=ValidationCategory.AUTHENTICATION
        )

        result = ValidationResult(
            success=False,
            issues=[critical_issue]
        )

        assert result.success is False
        assert len(result.issues) == 1
        assert result.has_critical_issues is True

    def test_validation_result_with_warnings_only(self):
        """Test ValidationResult with warnings but no critical issues."""
        warning_issue = ValidationIssue(
            title="Warning Issue",
            description="Warning problem",
            level=ValidationLevel.WARNING,
            category=ValidationCategory.PERFORMANCE
        )

        result = ValidationResult(
            success=True,
            issues=[warning_issue]
        )

        assert result.success is True
        assert len(result.issues) == 1
        assert result.has_critical_issues is False


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_setup_success(self):
        """Test validate_setup function with successful validation."""
        with patch('genops.providers.cohere_validation.CohereValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            mock_validator.validate_all.return_value = ValidationResult(success=True, issues=[])

            result = validate_setup(api_key="test-key")

            assert result.success is True
            mock_validator_class.assert_called_once_with(
                api_key="test-key",
                include_performance_tests=False
            )

    def test_validate_setup_with_performance(self):
        """Test validate_setup with performance tests enabled."""
        with patch('genops.providers.cohere_validation.CohereValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            mock_validator.validate_all.return_value = ValidationResult(success=True, issues=[])

            result = validate_setup(
                api_key="test-key",
                include_performance_tests=True
            )

            mock_validator_class.assert_called_once_with(
                api_key="test-key",
                include_performance_tests=True
            )

    def test_quick_validate_success(self):
        """Test quick_validate function with successful validation."""
        with patch('genops.providers.cohere_validation.validate_setup') as mock_validate:
            mock_validate.return_value = ValidationResult(success=True, issues=[])

            result = quick_validate()

            assert result is True

    def test_quick_validate_failure(self):
        """Test quick_validate function with validation failure."""
        with patch('genops.providers.cohere_validation.validate_setup') as mock_validate:
            mock_validate.return_value = ValidationResult(success=False, issues=[])

            result = quick_validate()

            assert result is False

    def test_quick_validate_exception(self):
        """Test quick_validate function with exception."""
        with patch('genops.providers.cohere_validation.validate_setup') as mock_validate:
            mock_validate.side_effect = Exception("Validation error")

            result = quick_validate()

            assert result is False

    def test_print_validation_result_success(self, capsys):
        """Test print_validation_result with successful validation."""
        result = ValidationResult(success=True, issues=[])

        print_validation_result(result)

        captured = capsys.readouterr()
        assert "✅" in captured.out
        assert "validation successful" in captured.out.lower()

    def test_print_validation_result_with_issues(self, capsys):
        """Test print_validation_result with validation issues."""
        issue = ValidationIssue(
            title="Test Issue",
            description="Test description",
            level=ValidationLevel.CRITICAL,
            category=ValidationCategory.AUTHENTICATION,
            fix_suggestion="Fix suggestion"
        )

        result = ValidationResult(success=False, issues=[issue])

        print_validation_result(result, detailed=True)

        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "Test Issue" in captured.out
        assert "Fix suggestion" in captured.out

    def test_print_validation_result_with_performance_metrics(self, capsys):
        """Test print_validation_result with performance metrics."""
        result = ValidationResult(
            success=True,
            issues=[],
            performance_metrics={"chat_latency": 250.5, "embed_latency": 180.2}
        )

        print_validation_result(result, detailed=True)

        captured = capsys.readouterr()
        assert "Performance metrics" in captured.out
        assert "250.5ms" in captured.out
        assert "180.2ms" in captured.out


class TestValidationIntegration:
    """Test validation system integration."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        with patch('genops.providers.cohere_validation.HAS_COHERE', True):
            with patch('genops.providers.cohere_validation.ClientV2') as mock_client_class:
                with patch.dict(os.environ, {'CO_API_KEY': 'test-api-key'}):
                    mock_client = Mock()
                    mock_client_class.return_value = mock_client
                    mock_client.chat.return_value = Mock()
                    mock_client.embed.return_value = Mock()
                    mock_client.rerank.return_value = Mock()

                    # Run full validation
                    result = validate_setup(include_performance_tests=True)

                    # Should be successful with no critical issues
                    assert result.success is True
                    assert result.has_critical_issues is False

    def test_validation_with_multiple_issues(self):
        """Test validation handling multiple types of issues."""
        with patch('genops.providers.cohere_validation.HAS_COHERE', False):
            with patch.dict(os.environ, {}, clear=True):
                result = validate_setup()

                # Should have multiple critical issues
                assert result.success is False
                assert result.has_critical_issues is True
                assert len(result.issues) > 1

                # Should have both dependency and auth issues
                categories = {issue.category for issue in result.issues}
                assert ValidationCategory.DEPENDENCIES in categories
                assert ValidationCategory.AUTHENTICATION in categories

    def test_graceful_degradation_on_import_error(self):
        """Test graceful handling when validation modules can't be imported."""
        with patch('genops.providers.cohere_validation.CohereValidator') as mock_validator:
            mock_validator.side_effect = ImportError("Module not found")

            # Should not raise exception
            result = quick_validate()
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
