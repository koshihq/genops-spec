"""
Comprehensive tests for Fireworks AI validation and setup testing.

Tests cover:
- Environment and configuration validation
- API key validation and connectivity testing
- Model accessibility and permissions
- Performance benchmarking and Fireattention optimization
- Diagnostic information collection
- Setup troubleshooting and error reporting
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from decimal import Decimal

from genops.providers.fireworks_validation import (
    ValidationResult,
    validate_fireworks_setup,
    check_api_key_validity,
    test_model_access,
    benchmark_performance,
    collect_diagnostics,
    generate_setup_report
)


class TestValidationResult:
    """Test ValidationResult data structure."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult object creation."""
        result = ValidationResult(
            is_valid=True,
            api_key_valid=True,
            connectivity_ok=True,
            model_access=["model1", "model2"],
            performance_metrics={"speed": 1.0},
            diagnostics={"test": "data"}
        )
        
        assert result.is_valid is True
        assert result.api_key_valid is True
        assert result.connectivity_ok is True
        assert len(result.model_access) == 2
        assert result.performance_metrics["speed"] == 1.0
        assert result.diagnostics["test"] == "data"
    
    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(is_valid=False)
        
        assert result.is_valid is False
        assert result.api_key_valid is False
        assert result.connectivity_ok is False
        assert result.model_access == []
        assert result.performance_metrics == {}
        assert result.diagnostics == {}


class TestAPIKeyValidation:
    """Test API key validation functionality."""
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_check_api_key_validity_success(self, mock_fireworks):
        """Test successful API key validation."""
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[
            Mock(id="accounts/fireworks/models/llama-v3p1-8b-instruct")
        ])
        mock_fireworks.return_value = mock_client
        
        is_valid, error_msg = check_api_key_validity("test-key")
        
        assert is_valid is True
        assert error_msg is None
        mock_fireworks.assert_called_once_with(api_key="test-key")
        mock_client.models.list.assert_called_once()
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_check_api_key_validity_invalid_key(self, mock_fireworks):
        """Test API key validation with invalid key."""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Invalid API key")
        mock_fireworks.return_value = mock_client
        
        is_valid, error_msg = check_api_key_validity("invalid-key")
        
        assert is_valid is False
        assert "Invalid API key" in error_msg
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_check_api_key_validity_network_error(self, mock_fireworks):
        """Test API key validation with network connectivity issues."""
        mock_client = Mock()
        mock_client.models.list.side_effect = ConnectionError("Network unreachable")
        mock_fireworks.return_value = mock_client
        
        is_valid, error_msg = check_api_key_validity("test-key")
        
        assert is_valid is False
        assert "Network" in error_msg or "connectivity" in error_msg.lower()
    
    def test_check_api_key_validity_empty_key(self):
        """Test API key validation with empty/None key."""
        is_valid, error_msg = check_api_key_validity("")
        
        assert is_valid is False
        assert "API key not provided" in error_msg
        
        is_valid, error_msg = check_api_key_validity(None)
        
        assert is_valid is False
        assert "API key not provided" in error_msg


class TestModelAccessValidation:
    """Test model accessibility validation."""
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_test_model_access_success(self, mock_fireworks):
        """Test successful model access validation."""
        mock_client = Mock()
        
        # Mock successful model list
        mock_client.models.list.return_value = Mock(data=[
            Mock(id="accounts/fireworks/models/llama-v3p1-8b-instruct"),
            Mock(id="accounts/fireworks/models/llama-v3p1-70b-instruct"),
            Mock(id="accounts/fireworks/models/nomic-embed-text-v1p5")
        ])
        
        # Mock successful chat completion test
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(total_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        mock_fireworks.return_value = mock_client
        
        accessible_models, failed_models = test_model_access("test-key")
        
        assert len(accessible_models) > 0
        assert len(failed_models) == 0
        assert "accounts/fireworks/models/llama-v3p1-8b-instruct" in accessible_models
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_test_model_access_partial_failure(self, mock_fireworks):
        """Test model access validation with some models failing."""
        mock_client = Mock()
        
        # Mock model list with multiple models
        mock_client.models.list.return_value = Mock(data=[
            Mock(id="accounts/fireworks/models/llama-v3p1-8b-instruct"),
            Mock(id="accounts/fireworks/models/llama-v3p1-70b-instruct")
        ])
        
        # Mock chat completion that succeeds for first model, fails for second
        def mock_create(**kwargs):
            if "8b" in kwargs["model"]:
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content="Success"))]
                mock_response.usage = Mock(total_tokens=25)
                return mock_response
            else:
                raise Exception("Model not accessible")
        
        mock_client.chat.completions.create.side_effect = mock_create
        mock_fireworks.return_value = mock_client
        
        accessible_models, failed_models = test_model_access("test-key")
        
        assert len(accessible_models) == 1
        assert len(failed_models) == 1
        assert "accounts/fireworks/models/llama-v3p1-8b-instruct" in accessible_models
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_test_model_access_no_models(self, mock_fireworks):
        """Test model access validation when no models are available."""
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[])
        mock_fireworks.return_value = mock_client
        
        accessible_models, failed_models = test_model_access("test-key")
        
        assert len(accessible_models) == 0
        assert len(failed_models) == 0
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_test_model_access_api_error(self, mock_fireworks):
        """Test model access validation with API errors."""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("API error")
        mock_fireworks.return_value = mock_client
        
        accessible_models, failed_models = test_model_access("test-key")
        
        assert len(accessible_models) == 0
        # Should handle error gracefully without crashing


class TestPerformanceBenchmarking:
    """Test performance benchmarking functionality."""
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_benchmark_performance_success(self, mock_fireworks):
        """Test successful performance benchmarking."""
        mock_client = Mock()
        
        # Mock a fast response (Fireattention optimization)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Benchmark response with 4x speed"))]
        mock_response.usage = Mock(
            prompt_tokens=20,
            completion_tokens=30,
            total_tokens=50
        )
        mock_client.chat.completions.create.return_value = mock_response
        
        mock_fireworks.return_value = mock_client
        
        # Mock time.time to control timing
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.85]  # 0.85s response time (4x faster)
            
            metrics = benchmark_performance(
                "test-key",
                "accounts/fireworks/models/llama-v3p1-8b-instruct"
            )
        
        assert "avg_response_time" in metrics
        assert "tokens_per_second" in metrics
        assert "fireattention_speedup" in metrics
        
        # Verify Fireattention optimization metrics
        assert metrics["avg_response_time"] < 1.0  # Should be fast
        assert metrics["tokens_per_second"] > 50   # Good throughput
        assert metrics["fireattention_speedup"] >= 3.0  # Should show speedup
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_benchmark_performance_multiple_runs(self, mock_fireworks):
        """Test performance benchmarking with multiple test runs."""
        mock_client = Mock()
        
        # Mock consistent fast responses
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Consistent fast response"))]
        mock_response.usage = Mock(total_tokens=40)
        mock_client.chat.completions.create.return_value = mock_response
        
        mock_fireworks.return_value = mock_client
        
        # Mock time progression for multiple calls
        with patch('time.time') as mock_time:
            # 3 runs: start, end1, start2, end2, start3, end3
            mock_time.side_effect = [0.0, 0.8, 1.0, 1.9, 2.0, 2.7]  
            
            metrics = benchmark_performance(
                "test-key",
                "accounts/fireworks/models/llama-v3p1-8b-instruct",
                num_runs=3
            )
        
        # Should average the results across runs
        assert metrics["avg_response_time"] > 0
        assert metrics["consistency_score"] >= 0  # Some consistency metric
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_benchmark_performance_error_handling(self, mock_fireworks):
        """Test performance benchmarking error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Benchmark failed")
        mock_fireworks.return_value = mock_client
        
        metrics = benchmark_performance(
            "test-key",
            "accounts/fireworks/models/llama-v3p1-8b-instruct"
        )
        
        # Should return empty metrics or error indicators
        assert isinstance(metrics, dict)
        assert metrics.get("error") is not None or len(metrics) == 0
    
    def test_benchmark_performance_invalid_model(self):
        """Test performance benchmarking with invalid model."""
        metrics = benchmark_performance("test-key", "invalid-model")
        
        # Should handle gracefully
        assert isinstance(metrics, dict)


class TestDiagnosticsCollection:
    """Test diagnostic information collection."""
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    @patch.dict(os.environ, {
        'FIREWORKS_API_KEY': 'test-key',
        'GENOPS_TEAM': 'test-team'
    })
    def test_collect_diagnostics_comprehensive(self, mock_fireworks):
        """Test comprehensive diagnostics collection."""
        mock_client = Mock()
        
        # Mock model capabilities
        mock_client.models.list.return_value = Mock(data=[
            Mock(
                id="accounts/fireworks/models/llama-v3p1-8b-instruct",
                object="model",
                created=1234567890
            ),
            Mock(
                id="accounts/fireworks/models/nomic-embed-text-v1p5",
                object="model", 
                created=1234567890
            )
        ])
        
        mock_fireworks.return_value = mock_client
        
        diagnostics = collect_diagnostics("test-key")
        
        assert isinstance(diagnostics, dict)
        assert "environment" in diagnostics
        assert "api_connectivity" in diagnostics
        assert "model_capabilities" in diagnostics
        assert "feature_support" in diagnostics
        
        # Check environment diagnostics
        env = diagnostics["environment"]
        assert "python_version" in env
        assert "platform" in env
        assert "dependencies" in env
        
        # Check feature support
        features = diagnostics["feature_support"]
        assert "chat_completions" in features
        assert "embeddings" in features
        assert "batch_processing" in features
        assert "streaming" in features
        assert "multimodal" in features
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_collect_diagnostics_fireattention_detection(self, mock_fireworks):
        """Test Fireattention optimization detection."""
        mock_client = Mock()
        mock_fireworks.return_value = mock_client
        
        diagnostics = collect_diagnostics("test-key")
        
        # Should detect Fireattention capabilities
        assert "fireattention_enabled" in diagnostics["feature_support"]
        assert "speed_optimization" in diagnostics["feature_support"]
    
    def test_collect_diagnostics_environment_variables(self):
        """Test environment variable diagnostics."""
        with patch.dict(os.environ, {
            'FIREWORKS_API_KEY': 'present',
            'GENOPS_TEAM': 'test-team',
            'GENOPS_PROJECT': 'test-project'
        }):
            diagnostics = collect_diagnostics("test-key")
            
            env_vars = diagnostics["environment"]["environment_variables"]
            assert env_vars["FIREWORKS_API_KEY"] == "✓ Set"
            assert env_vars["GENOPS_TEAM"] == "✓ Set (test-team)"
            assert env_vars["GENOPS_PROJECT"] == "✓ Set (test-project)"
    
    @patch('genops.providers.fireworks_validation.Fireworks')
    def test_collect_diagnostics_model_categorization(self, mock_fireworks):
        """Test model categorization in diagnostics."""
        mock_client = Mock()
        mock_client.models.list.return_value = Mock(data=[
            Mock(id="accounts/fireworks/models/llama-v3p1-8b-instruct"),
            Mock(id="accounts/fireworks/models/llama-v3p2-11b-vision-instruct"),
            Mock(id="accounts/fireworks/models/nomic-embed-text-v1p5")
        ])
        mock_fireworks.return_value = mock_client
        
        diagnostics = collect_diagnostics("test-key")
        
        capabilities = diagnostics["model_capabilities"]
        assert "text_models" in capabilities
        assert "vision_models" in capabilities  
        assert "embedding_models" in capabilities
        assert "code_models" in capabilities


class TestSetupValidation:
    """Test comprehensive setup validation."""
    
    @patch('genops.providers.fireworks_validation.check_api_key_validity')
    @patch('genops.providers.fireworks_validation.test_model_access')
    @patch('genops.providers.fireworks_validation.benchmark_performance')
    @patch('genops.providers.fireworks_validation.collect_diagnostics')
    def test_validate_fireworks_setup_success(
        self,
        mock_collect_diagnostics,
        mock_benchmark_performance,
        mock_test_model_access,
        mock_check_api_key_validity
    ):
        """Test successful comprehensive setup validation."""
        
        # Mock all validation steps as successful
        mock_check_api_key_validity.return_value = (True, None)
        mock_test_model_access.return_value = (
            ["accounts/fireworks/models/llama-v3p1-8b-instruct"], []
        )
        mock_benchmark_performance.return_value = {
            "avg_response_time": 0.85,
            "tokens_per_second": 120,
            "fireattention_speedup": 4.2
        }
        mock_collect_diagnostics.return_value = {
            "fireattention_enabled": True,
            "batch_processing_available": True
        }
        
        config = {
            "team": "test-team",
            "project": "test-project",
            "daily_budget_limit": 100.0
        }
        
        with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
            result = validate_fireworks_setup(config=config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.api_key_valid is True
        assert result.connectivity_ok is True
        assert len(result.model_access) > 0
        assert result.performance_metrics["fireattention_speedup"] > 4.0
    
    @patch('genops.providers.fireworks_validation.check_api_key_validity')
    def test_validate_fireworks_setup_invalid_api_key(self, mock_check_api_key_validity):
        """Test validation with invalid API key."""
        mock_check_api_key_validity.return_value = (False, "Invalid API key")
        
        config = {"team": "test-team"}
        
        with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'invalid-key'}):
            result = validate_fireworks_setup(config=config)
        
        assert result.is_valid is False
        assert result.api_key_valid is False
    
    def test_validate_fireworks_setup_missing_api_key(self):
        """Test validation with missing API key."""
        config = {"team": "test-team"}
        
        with patch.dict(os.environ, {}, clear=True):
            result = validate_fireworks_setup(config=config)
        
        assert result.is_valid is False
        assert result.api_key_valid is False
    
    @patch('genops.providers.fireworks_validation.check_api_key_validity')
    @patch('genops.providers.fireworks_validation.test_model_access')
    def test_validate_fireworks_setup_no_model_access(
        self, mock_test_model_access, mock_check_api_key_validity
    ):
        """Test validation when no models are accessible."""
        mock_check_api_key_validity.return_value = (True, None)
        mock_test_model_access.return_value = ([], ["model1", "model2"])  # All failed
        
        config = {"team": "test-team"}
        
        with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
            result = validate_fireworks_setup(config=config)
        
        assert result.is_valid is False  # No usable models
        assert result.api_key_valid is True
        assert len(result.model_access) == 0
    
    @patch('genops.providers.fireworks_validation.check_api_key_validity')
    @patch('genops.providers.fireworks_validation.test_model_access')
    @patch('genops.providers.fireworks_validation.benchmark_performance')
    def test_validate_fireworks_setup_performance_issues(
        self, mock_benchmark_performance, mock_test_model_access, mock_check_api_key_validity
    ):
        """Test validation with performance issues."""
        mock_check_api_key_validity.return_value = (True, None)
        mock_test_model_access.return_value = (["model1"], [])
        
        # Mock poor performance (no Fireattention optimization)
        mock_benchmark_performance.return_value = {
            "avg_response_time": 4.5,  # Slow
            "tokens_per_second": 20,   # Low throughput
            "fireattention_speedup": 1.0  # No speedup
        }
        
        config = {"team": "test-team"}
        
        with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
            result = validate_fireworks_setup(config=config)
        
        # Should still be valid but with performance warnings
        assert result.is_valid is True  # Basic functionality works
        assert result.performance_metrics["fireattention_speedup"] == 1.0


class TestSetupReportGeneration:
    """Test setup validation report generation."""
    
    def test_generate_setup_report_success(self, mock_validation_result):
        """Test setup report generation for successful validation."""
        report = generate_setup_report(mock_validation_result)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "✅" in report  # Success indicators
        assert "Fireattention" in report  # Performance optimization mentioned
        assert "4x faster" in report  # Speed benefit highlighted
    
    def test_generate_setup_report_failure(self):
        """Test setup report generation for failed validation."""
        failed_result = ValidationResult(
            is_valid=False,
            api_key_valid=False,
            connectivity_ok=False
        )
        
        report = generate_setup_report(failed_result)
        
        assert isinstance(report, str)
        assert "❌" in report  # Error indicators
        assert "API key" in report  # Error details
        assert "troubleshooting" in report.lower() or "fix" in report.lower()
    
    def test_generate_setup_report_partial_success(self):
        """Test setup report for partial success scenarios."""
        partial_result = ValidationResult(
            is_valid=True,
            api_key_valid=True,
            connectivity_ok=True,
            model_access=["model1"],  # Limited access
            performance_metrics={"avg_response_time": 2.0},  # Slower than expected
            diagnostics={"fireattention_enabled": False}  # No optimization
        )
        
        report = generate_setup_report(partial_result)
        
        assert "⚠️" in report or "warnings" in report.lower()  # Warning indicators
        assert "performance" in report.lower()  # Performance concerns mentioned


class TestValidationWithPrintOutput:
    """Test validation with print output enabled."""
    
    @patch('genops.providers.fireworks_validation.check_api_key_validity')
    @patch('genops.providers.fireworks_validation.test_model_access') 
    @patch('genops.providers.fireworks_validation.collect_diagnostics')
    @patch('builtins.print')
    def test_validate_fireworks_setup_with_print(
        self, mock_print, mock_collect_diagnostics, mock_test_model_access, mock_check_api_key_validity
    ):
        """Test validation with print_results=True."""
        mock_check_api_key_validity.return_value = (True, None)
        mock_test_model_access.return_value = (["model1"], [])
        mock_collect_diagnostics.return_value = {"test": "data"}
        
        config = {"team": "test-team"}
        
        with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
            result = validate_fireworks_setup(config=config, print_results=True)
        
        # Should have called print with validation progress
        assert mock_print.call_count > 0
        
        # Check that important information was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        printed_text = " ".join(print_calls)
        
        assert "Fireworks" in printed_text
        assert "validation" in printed_text.lower() or "testing" in printed_text.lower()


class TestValidationErrorScenarios:
    """Test various error scenarios during validation."""
    
    def test_validation_with_network_connectivity_issues(self):
        """Test validation handling network connectivity issues."""
        with patch('genops.providers.fireworks_validation.Fireworks') as mock_fireworks:
            mock_fireworks.side_effect = ConnectionError("Network unreachable")
            
            config = {"team": "test-team"}
            
            with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
                result = validate_fireworks_setup(config=config)
            
            assert result.is_valid is False
            assert result.connectivity_ok is False
    
    def test_validation_with_timeout_errors(self):
        """Test validation handling timeout errors."""
        with patch('genops.providers.fireworks_validation.Fireworks') as mock_fireworks:
            mock_client = Mock()
            mock_client.models.list.side_effect = TimeoutError("Request timeout")
            mock_fireworks.return_value = mock_client
            
            config = {"team": "test-team"}
            
            with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
                result = validate_fireworks_setup(config=config)
            
            assert result.is_valid is False
    
    def test_validation_with_rate_limiting(self):
        """Test validation handling rate limiting scenarios."""
        with patch('genops.providers.fireworks_validation.Fireworks') as mock_fireworks:
            mock_client = Mock()
            mock_client.models.list.side_effect = Exception("Rate limit exceeded")
            mock_fireworks.return_value = mock_client
            
            config = {"team": "test-team"}
            
            with patch.dict(os.environ, {'FIREWORKS_API_KEY': 'test-key'}):
                result = validate_fireworks_setup(config=config)
            
            assert result.is_valid is False