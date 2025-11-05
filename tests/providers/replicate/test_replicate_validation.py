#!/usr/bin/env python3
"""
Test Suite for ReplicateValidator and Validation Functions

Unit tests covering comprehensive validation functionality including:
- Environment setup validation  
- Dependencies and SDK version checking
- API authentication and connectivity testing
- Model availability verification across categories
- Performance benchmarking and diagnostics
- Error handling with actionable guidance

Target: ~33 tests covering all validation scenarios
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.genops.providers.replicate_validation import (
    ReplicateValidator,
    ValidationResult,
    ModelTestResult,
    validate_setup,
    print_validation_result,
    quick_validate
)

class TestValidationResult:
    """Test ValidationResult data structure."""
    
    def test_validation_result_initialization(self):
        """Test ValidationResult initialization with defaults."""
        result = ValidationResult(success=True)
        
        assert result.success is True
        assert result.errors == []
        assert result.warnings == []
        assert result.optimization_recommendations == []
    
    def test_validation_result_with_data(self):
        """Test ValidationResult with complete data."""
        result = ValidationResult(
            success=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            performance_metrics={"latency": 100},
            environment_info={"python_version": "3.9.0"},
            optimization_recommendations=["Use faster model"]
        )
        
        assert result.success is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.performance_metrics["latency"] == 100

class TestModelTestResult:
    """Test ModelTestResult data structure."""
    
    def test_model_test_result_success(self):
        """Test successful model test result."""
        result = ModelTestResult(
            model_name="meta/llama-2-7b-chat",
            available=True,
            latency_ms=1500.0,
            cost_estimate=0.001234,
            category="text"
        )
        
        assert result.model_name == "meta/llama-2-7b-chat"
        assert result.available is True
        assert result.latency_ms == 1500.0
        assert result.cost_estimate == 0.001234
        assert result.category == "text"
        assert result.error is None
    
    def test_model_test_result_failure(self):
        """Test failed model test result."""
        result = ModelTestResult(
            model_name="invalid/model",
            available=False,
            error="Model not found"
        )
        
        assert result.available is False
        assert result.error == "Model not found"
        assert result.latency_ms is None

class TestReplicateValidatorInitialization:
    """Test ReplicateValidator initialization."""
    
    def test_validator_initialization_with_token(self):
        """Test validator initialization with API token."""
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "r8_test_token"}):
            validator = ReplicateValidator()
            
            assert validator.api_token == "r8_test_token"
    
    def test_validator_initialization_without_token(self):
        """Test validator initialization without API token."""
        with patch.dict(os.environ, {}, clear=True):
            validator = ReplicateValidator()
            
            assert validator.api_token is None

class TestEnvironmentValidation:
    """Test environment configuration validation."""
    
    @pytest.fixture
    def validator(self):
        return ReplicateValidator()
    
    def test_validate_environment_with_valid_token(self, validator):
        """Test environment validation with valid API token."""
        validator.api_token = "r8_valid_token_format_12345678901234567890"
        result = ValidationResult(success=True)
        
        env_info = validator._validate_environment(result)
        
        assert env_info["replicate_token_set"] is True
        assert env_info["replicate_token_valid_format"] is True
        assert len(result.errors) == 0
    
    def test_validate_environment_with_invalid_token_format(self, validator):
        """Test environment validation with invalid token format."""
        validator.api_token = "invalid_token_format"
        result = ValidationResult(success=True)
        
        env_info = validator._validate_environment(result)
        
        assert env_info["replicate_token_set"] is True
        assert env_info["replicate_token_valid_format"] is False
        assert any("Invalid REPLICATE_API_TOKEN format" in error for error in result.errors)
        assert any("🔧 API TOKEN FORMAT FIX:" in error for error in result.errors)
    
    def test_validate_environment_without_token(self, validator):
        """Test environment validation without API token."""
        validator.api_token = None
        result = ValidationResult(success=True)
        
        env_info = validator._validate_environment(result)
        
        assert env_info["replicate_token_set"] is False
        assert any("REPLICATE_API_TOKEN environment variable not set" in error for error in result.errors)
        assert any("🔧 API TOKEN SETUP:" in error for error in result.errors)
    
    def test_validate_environment_python_version(self, validator):
        """Test Python version validation."""
        result = ValidationResult(success=True)
        
        with patch.object(sys, 'version_info', (3, 7, 0)):  # Python 3.7 (too old)
            env_info = validator._validate_environment(result)
            
            assert any("Python 3.8+ required" in error for error in result.errors)
    
    def test_validate_environment_optional_vars(self, validator):
        """Test validation of optional environment variables."""
        validator.api_token = "r8_valid_token_12345678901234567890"
        result = ValidationResult(success=True)
        
        with patch.dict(os.environ, {"GENOPS_ENVIRONMENT": "production"}, clear=True):
            env_info = validator._validate_environment(result)
            
            assert env_info["environment_variables"]["GENOPS_ENVIRONMENT"] == "production"
            # Should have warnings for other missing optional vars
            assert len(result.warnings) > 0

class TestDependenciesValidation:
    """Test dependencies validation."""
    
    @pytest.fixture
    def validator(self):
        return ReplicateValidator()
    
    def test_validate_dependencies_replicate_missing(self, validator):
        """Test validation when Replicate SDK is missing."""
        result = ValidationResult(success=True)
        
        with patch('src.genops.providers.replicate_validation.replicate', None):
            validator._validate_dependencies(result)
            
            assert any("Replicate Python SDK not installed" in error for error in result.errors)
            assert any("🔧 DEPENDENCY FIX:" in error for error in result.errors)
    
    def test_validate_dependencies_replicate_available(self, validator):
        """Test validation when Replicate SDK is available."""
        result = ValidationResult(success=True)
        
        mock_replicate = Mock()
        mock_replicate.__version__ = "0.25.0"
        
        with patch('src.genops.providers.replicate_validation.replicate', mock_replicate):
            validator._validate_dependencies(result)
            
            # Should not add errors for available SDK
            assert not any("not installed" in error for error in result.errors)
            assert result.environment_info["replicate_version"] == "0.25.0"
    
    def test_validate_dependencies_old_version_warning(self, validator):
        """Test warning for old Replicate SDK version."""
        result = ValidationResult(success=True)
        
        mock_replicate = Mock()
        mock_replicate.__version__ = "0.15.0"  # Old version
        
        with patch('src.genops.providers.replicate_validation.replicate', mock_replicate):
            validator._validate_dependencies(result)
            
            assert any("may be outdated" in warning for warning in result.warnings)
    
    def test_validate_dependencies_opentelemetry_missing(self, validator):
        """Test validation when OpenTelemetry is missing."""
        result = ValidationResult(success=True)
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'opentelemetry'")):
            validator._validate_dependencies(result)
            
            assert any("OpenTelemetry not available" in warning for warning in result.warnings)

class TestAuthenticationValidation:
    """Test API authentication validation."""
    
    @pytest.fixture
    def validator(self):
        validator = ReplicateValidator()
        validator.api_token = "r8_test_token_12345678901234567890"
        return validator
    
    @patch('requests.get')
    def test_validate_authentication_success(self, mock_get, validator):
        """Test successful authentication validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = ValidationResult(success=True)
        validator._validate_authentication(result)
        
        # Should not add errors for successful auth
        assert not any("Authentication failed" in error for error in result.errors)
    
    @patch('requests.get')
    def test_validate_authentication_invalid_token(self, mock_get, validator):
        """Test authentication validation with invalid token."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        result = ValidationResult(success=True)
        validator._validate_authentication(result)
        
        assert any("Authentication failed" in error for error in result.errors)
        assert any("🔧 AUTHENTICATION FIX:" in error for error in result.errors)
    
    @patch('requests.get')
    def test_validate_authentication_network_timeout(self, mock_get, validator):
        """Test authentication validation with network timeout."""
        mock_get.side_effect = requests.exceptions.ConnectTimeout("Connection timeout")
        
        result = ValidationResult(success=True)
        validator._validate_authentication(result)
        
        assert any("Connection timeout" in error for error in result.errors)
        assert any("🔧 CONNECTIVITY FIX:" in error for error in result.errors)
    
    @patch('requests.get')
    def test_validate_authentication_network_error(self, mock_get, validator):
        """Test authentication validation with general network error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        result = ValidationResult(success=True)
        validator._validate_authentication(result)
        
        assert any("Network error" in error for error in result.errors)
        assert any("🔧 NETWORK FIX:" in error for error in result.errors)
    
    def test_validate_authentication_no_token(self):
        """Test authentication validation without token."""
        validator = ReplicateValidator()
        validator.api_token = None
        result = ValidationResult(success=True)
        
        validator._validate_authentication(result)
        
        # Should skip validation without token (already handled in env validation)
        assert len(result.errors) == 0

class TestAPIConnectivityValidation:
    """Test API connectivity validation."""
    
    @pytest.fixture
    def validator(self):
        validator = ReplicateValidator()
        validator.api_token = "r8_test_token_12345678901234567890"
        return validator
    
    @patch('src.genops.providers.replicate_validation.replicate')
    def test_validate_api_connectivity_success(self, mock_replicate, validator):
        """Test successful API connectivity validation."""
        mock_client = Mock()
        mock_models = Mock()
        mock_models.list.return_value = ["model1", "model2"]
        mock_client.models = mock_models
        mock_replicate.Client.return_value = mock_client
        
        result = ValidationResult(success=True)
        
        with patch('time.time', side_effect=[1000, 1001]):  # 1 second latency
            validator._validate_api_connectivity(result)
        
        assert result.performance_metrics["api_latency_ms"] == 1000.0
        # Should not add errors for successful connectivity
        assert not any("connectivity test failed" in error.lower() for error in result.errors)
    
    @patch('src.genops.providers.replicate_validation.replicate')
    def test_validate_api_connectivity_high_latency(self, mock_replicate, validator):
        """Test API connectivity with high latency warning."""
        mock_client = Mock()
        mock_models = Mock()
        mock_models.list.return_value = ["model1"]
        mock_client.models = mock_models
        mock_replicate.Client.return_value = mock_client
        
        result = ValidationResult(success=True)
        
        with patch('time.time', side_effect=[1000, 1008]):  # 8 second latency (high)
            validator._validate_api_connectivity(result)
        
        assert result.performance_metrics["api_latency_ms"] == 8000.0
        assert any("High API latency" in warning for warning in result.warnings)
    
    @patch('src.genops.providers.replicate_validation.replicate')
    def test_validate_api_connectivity_failure(self, mock_replicate, validator):
        """Test API connectivity validation failure."""
        mock_replicate.Client.side_effect = Exception("Connection failed")
        
        result = ValidationResult(success=True)
        validator._validate_api_connectivity(result)
        
        assert any("API connectivity test failed" in error for error in result.errors)
    
    def test_validate_api_connectivity_skip_without_token(self):
        """Test API connectivity skipped without token."""
        validator = ReplicateValidator()
        validator.api_token = None
        result = ValidationResult(success=True, errors=["Previous error"])
        
        validator._validate_api_connectivity(result)
        
        # Should skip connectivity test
        assert result.performance_metrics is None

class TestModelAvailabilityValidation:
    """Test model availability validation."""
    
    @pytest.fixture
    def validator(self):
        validator = ReplicateValidator()
        validator.api_token = "r8_test_token_12345678901234567890"
        return validator
    
    @patch('src.genops.providers.replicate_validation.replicate')
    def test_validate_model_availability_success(self, mock_replicate, validator):
        """Test successful model availability validation."""
        mock_client = Mock()
        mock_model = Mock()
        mock_client.models.get.return_value = mock_model
        mock_replicate.Client.return_value = mock_client
        
        result = ValidationResult(success=True)
        
        with patch('time.time', side_effect=[1000, 1001, 1002, 1003, 1004, 1005]):
            availability = validator._validate_model_availability(result)
        
        # Should test multiple model categories
        assert len(availability) > 0
        # All test models should be available with mocked success
        assert all(availability.values())
    
    def test_validate_model_availability_skip_without_token(self):
        """Test model availability validation skipped without token."""
        validator = ReplicateValidator()
        validator.api_token = None
        result = ValidationResult(success=True)
        
        availability = validator._validate_model_availability(result)
        
        assert availability == {}
    
    @patch('src.genops.providers.replicate_validation.replicate')
    def test_test_model_availability_success(self, mock_replicate, validator):
        """Test individual model availability test."""
        mock_client = Mock()
        mock_model = Mock()
        mock_client.models.get.return_value = mock_model
        mock_replicate.Client.return_value = mock_client
        
        with patch('time.time', side_effect=[1000, 1001.5]):
            result = validator._test_model_availability("meta/llama-2-7b-chat", "text")
        
        assert result.model_name == "meta/llama-2-7b-chat"
        assert result.available is True
        assert result.latency_ms == 1500.0
        assert result.category == "text"
        assert result.error is None
    
    @patch('src.genops.providers.replicate_validation.replicate')
    def test_test_model_availability_failure(self, mock_replicate, validator):
        """Test individual model availability test failure."""
        mock_client = Mock()
        mock_replicate.exceptions = Mock()
        mock_replicate.exceptions.ReplicateError = Exception
        mock_client.models.get.side_effect = Exception("Model not found")
        mock_replicate.Client.return_value = mock_client
        
        result = validator._test_model_availability("invalid/model", "text")
        
        assert result.model_name == "invalid/model"
        assert result.available is False
        assert result.error == "Model not found"

class TestPerformanceBenchmarks:
    """Test performance benchmarking."""
    
    @pytest.fixture
    def validator(self):
        validator = ReplicateValidator()
        validator.api_token = "r8_test_token_12345678901234567890"
        return validator
    
    def test_run_performance_benchmarks_basic(self, validator):
        """Test basic performance benchmarking."""
        result = ValidationResult(success=True)
        result.environment_info = {"python_version": "3.9.0", "platform": "linux"}
        result.performance_metrics = {"api_latency_ms": 1500}
        
        metrics = validator._run_performance_benchmarks(result)
        
        assert "system" in metrics
        assert metrics["system"]["python_version"] == "3.9.0"
        assert metrics["system"]["platform"] == "linux"
        assert "timestamp" in metrics["system"]
    
    def test_run_performance_benchmarks_skip_on_errors(self, validator):
        """Test performance benchmarks skipped when there are setup errors."""
        result = ValidationResult(success=False, errors=["Setup error"])
        
        metrics = validator._run_performance_benchmarks(result)
        
        # Should return basic metrics even with errors
        assert "system" in metrics

class TestOptimizationRecommendations:
    """Test optimization recommendation generation."""
    
    @pytest.fixture
    def validator(self):
        return ReplicateValidator()
    
    def test_generate_recommendations_high_latency(self, validator):
        """Test recommendations for high API latency."""
        result = ValidationResult(success=True)
        result.performance_metrics = {"api_latency_ms": 3000}  # High latency
        
        recommendations = validator._generate_recommendations(result)
        
        assert any("high api latency" in rec.lower() for rec in recommendations)
        assert any("caching" in rec.lower() or "stream" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_good_performance(self, validator):
        """Test recommendations for good performance."""
        result = ValidationResult(success=True)
        result.performance_metrics = {"api_latency_ms": 300}  # Good latency
        
        recommendations = validator._generate_recommendations(result)
        
        assert any("good api performance" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_missing_telemetry(self, validator):
        """Test recommendations for missing telemetry configuration."""
        result = ValidationResult(success=True)
        result.environment_info = {
            "environment_variables": {
                "OTEL_EXPORTER_OTLP_ENDPOINT": None,
                "GENOPS_ENVIRONMENT": None
            }
        }
        
        recommendations = validator._generate_recommendations(result)
        
        assert any("otel_exporter_otlp_endpoint" in rec.lower() for rec in recommendations)
        assert any("genops_environment" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_all_models_available(self, validator):
        """Test recommendations when all models are available."""
        result = ValidationResult(success=True)
        result.model_availability = {
            "meta/llama-2-7b-chat": True,
            "black-forest-labs/flux-schnell": True,
            "openai/whisper": True
        }
        
        recommendations = validator._generate_recommendations(result)
        
        assert any("all test models available" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_setup_success(self, validator):
        """Test recommendations for successful setup."""
        result = ValidationResult(success=True)
        
        recommendations = validator._generate_recommendations(result)
        
        assert any("setup validation passed" in rec.lower() for rec in recommendations)
        assert any("hello_genops_minimal.py" in rec for rec in recommendations)
    
    def test_generate_recommendations_setup_failure(self, validator):
        """Test recommendations for failed setup."""
        result = ValidationResult(success=False, errors=["Setup error"])
        
        recommendations = validator._generate_recommendations(result)
        
        assert any("setup issues found" in rec.lower() for rec in recommendations)

class TestCompleteValidation:
    """Test complete validation workflow."""
    
    @pytest.fixture
    def validator(self):
        return ReplicateValidator()
    
    @patch.object(ReplicateValidator, '_validate_environment')
    @patch.object(ReplicateValidator, '_validate_dependencies')
    @patch.object(ReplicateValidator, '_validate_authentication')
    @patch.object(ReplicateValidator, '_validate_api_connectivity')
    @patch.object(ReplicateValidator, '_validate_model_availability')
    @patch.object(ReplicateValidator, '_run_performance_benchmarks')
    @patch.object(ReplicateValidator, '_generate_recommendations')
    def test_validate_complete_setup_success(
        self,
        mock_generate_recommendations,
        mock_run_performance_benchmarks,
        mock_validate_model_availability,
        mock_validate_api_connectivity,
        mock_validate_authentication,
        mock_validate_dependencies,
        mock_validate_environment,
        validator
    ):
        """Test complete validation workflow with success."""
        # Setup mocks for successful validation
        mock_validate_environment.return_value = {"python_version": "3.9.0"}
        mock_validate_model_availability.return_value = {"model1": True}
        mock_run_performance_benchmarks.return_value = {"latency": 100}
        mock_generate_recommendations.return_value = ["All good!"]
        
        result = validator.validate_complete_setup()
        
        assert result.success is True
        assert len(result.errors) == 0
        
        # Verify all validation methods were called
        mock_validate_environment.assert_called_once()
        mock_validate_dependencies.assert_called_once()
        mock_validate_authentication.assert_called_once()
        mock_validate_api_connectivity.assert_called_once()
        mock_validate_model_availability.assert_called_once()
        mock_run_performance_benchmarks.assert_called_once()
        mock_generate_recommendations.assert_called_once()
    
    @patch.object(ReplicateValidator, '_validate_environment')
    def test_validate_complete_setup_with_errors(self, mock_validate_environment, validator):
        """Test complete validation workflow with errors."""
        # Mock environment validation to add errors
        def add_errors(result):
            result.errors.append("Environment error")
            return {"error": True}
        
        mock_validate_environment.side_effect = add_errors
        
        result = validator.validate_complete_setup()
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "Environment error" in result.errors

class TestPublicFunctions:
    """Test public validation functions."""
    
    @patch('src.genops.providers.replicate_validation.ReplicateValidator')
    def test_validate_setup_function(self, mock_validator_class):
        """Test validate_setup public function."""
        mock_validator = Mock()
        mock_result = ValidationResult(success=True)
        mock_validator.validate_complete_setup.return_value = mock_result
        mock_validator_class.return_value = mock_validator
        
        result = validate_setup()
        
        assert result.success is True
        mock_validator_class.assert_called_once()
        mock_validator.validate_complete_setup.assert_called_once()
    
    @patch('builtins.print')
    def test_print_validation_result_success(self, mock_print):
        """Test print_validation_result with successful result."""
        result = ValidationResult(
            success=True,
            performance_metrics={"api_latency_ms": 500},
            environment_info={"python_version": "3.9.0"},
            optimization_recommendations=["Everything looks good!"]
        )
        
        print_validation_result(result)
        
        # Should print success message
        mock_print.assert_called()
        printed_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        assert "SUCCESS" in printed_output
    
    @patch('builtins.print')
    def test_print_validation_result_with_errors(self, mock_print):
        """Test print_validation_result with errors."""
        result = ValidationResult(
            success=False,
            errors=["Error 1", "🔧 QUICK FIX:", "   Fix command"],
            warnings=["Warning 1"]
        )
        
        print_validation_result(result)
        
        # Should print errors and warnings
        printed_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        assert "ISSUES FOUND" in printed_output
        assert "ERRORS TO FIX" in printed_output
        assert "WARNINGS" in printed_output
    
    @patch('src.genops.providers.replicate_validation.validate_setup')
    @patch('builtins.print')
    def test_quick_validate_success(self, mock_print, mock_validate_setup):
        """Test quick_validate with successful validation."""
        mock_validate_setup.return_value = ValidationResult(success=True)
        
        result = quick_validate()
        
        assert result is True
        assert any("validation passed" in str(call.args[0]) for call in mock_print.call_args_list)
    
    @patch('src.genops.providers.replicate_validation.validate_setup')
    @patch('builtins.print')
    def test_quick_validate_failure(self, mock_print, mock_validate_setup):
        """Test quick_validate with failed validation."""
        mock_validate_setup.return_value = ValidationResult(success=False)
        
        result = quick_validate()
        
        assert result is False
        assert any("validation failed" in str(call.args[0]) for call in mock_print.call_args_list)