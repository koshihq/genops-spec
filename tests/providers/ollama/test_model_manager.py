"""Tests for Ollama model manager functionality."""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from genops.providers.ollama.model_manager import (
    OllamaModelManager,
    ModelInfo,
    ModelOptimizer,
    ModelComparison,
    ModelSize,
    ModelType,
    get_model_manager,
    set_model_manager,
    create_model_manager
)


class TestModelInfo:
    """Test ModelInfo dataclass functionality."""
    
    def test_model_info_creation(self):
        """Test basic model info creation."""
        model = ModelInfo(
            name="llama3.2:1b",
            size_gb=1.3,
            parameter_count="1.2B"
        )
        
        assert model.name == "llama3.2:1b"
        assert model.size_gb == 1.3
        assert model.parameter_count == "1.2B"
        assert model.total_inferences == 0
        assert model.success_rate == 100.0
    
    def test_model_categorization_by_size(self):
        """Test automatic model categorization by size."""
        # Test tiny model
        tiny_model = ModelInfo("tiny-model", size_gb=0.5)
        assert tiny_model.size_category == ModelSize.TINY
        
        # Test small model
        small_model = ModelInfo("small-model", size_gb=2.0)
        assert small_model.size_category == ModelSize.SMALL
        
        # Test medium model
        medium_model = ModelInfo("medium-model", size_gb=6.0)
        assert medium_model.size_category == ModelSize.MEDIUM
        
        # Test large model
        large_model = ModelInfo("large-model", size_gb=15.0)
        assert large_model.size_category == ModelSize.LARGE
        
        # Test xlarge model
        xlarge_model = ModelInfo("xlarge-model", size_gb=25.0)
        assert xlarge_model.size_category == ModelSize.XLARGE
    
    def test_model_type_detection(self):
        """Test automatic model type detection."""
        # Test code model
        code_model = ModelInfo("codellama:7b", size_gb=7.0)
        assert code_model.model_type == ModelType.CODE
        
        # Test chat model
        chat_model = ModelInfo("llama3.2:3b-chat", size_gb=3.0)
        assert chat_model.model_type == ModelType.CHAT
        
        # Test instruct model
        instruct_model = ModelInfo("mistral:7b-instruct", size_gb=7.0)
        assert instruct_model.model_type == ModelType.INSTRUCT
        
        # Test embedding model
        embed_model = ModelInfo("nomic-embed-text", size_gb=1.0)
        assert embed_model.model_type == ModelType.EMBEDDING
        
        # Test multimodal model
        vision_model = ModelInfo("llava:7b", size_gb=7.0)
        assert vision_model.model_type == ModelType.MULTIMODAL
    
    def test_performance_stats_update(self):
        """Test updating performance statistics."""
        model = ModelInfo("test-model", size_gb=1.0)
        
        # Add first inference
        model.update_performance_stats(
            inference_time_ms=1500.0,
            tokens=30,
            memory_mb=2048.0,
            cost=0.001
        )
        
        assert model.total_inferences == 1
        assert model.avg_inference_latency_ms == 1500.0
        assert model.avg_memory_usage_mb == 2048.0
        assert model.cost_per_inference == 0.001
        
        # Add second inference
        model.update_performance_stats(
            inference_time_ms=2000.0,
            tokens=50,
            memory_mb=2560.0,
            cost=0.0015
        )
        
        assert model.total_inferences == 2
        assert model.avg_inference_latency_ms == 1750.0  # (1500 + 2000) / 2
        assert model.avg_memory_usage_mb == 2304.0  # (2048 + 2560) / 2
        assert model.cost_per_inference == 0.00125  # (0.001 + 0.0015) / 2
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        model = ModelInfo("error-model", size_gb=1.0)
        
        # Start with 100% success rate
        assert model.success_rate == 100.0
        assert model.error_count == 0
        
        # Add successful inference
        model.update_performance_stats(1000.0, 20)
        assert model.total_inferences == 1
        assert model.success_rate == 100.0
        
        # Add error
        model.mark_error()
        assert model.error_count == 1
        assert model.success_rate == 50.0  # 1 success, 1 error
        
        # Add another success
        model.update_performance_stats(1200.0, 25)
        assert model.total_inferences == 2
        assert model.success_rate == 66.67  # 2 success, 1 error (rounded)


class TestModelOptimizer:
    """Test ModelOptimizer functionality."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = ModelOptimizer(
            model_name="test-model",
            current_performance={
                "latency_ms": 2000.0,
                "tokens_per_second": 15.0,
                "memory_usage_mb": 4096.0
            }
        )
        
        assert optimizer.model_name == "test-model"
        assert optimizer.current_performance["latency_ms"] == 2000.0
        assert len(optimizer.optimization_opportunities) == 0
    
    def test_add_recommendation(self):
        """Test adding optimization recommendations."""
        optimizer = ModelOptimizer("test", {})
        
        optimizer.add_recommendation(
            "Performance",
            "Use quantized model for faster inference",
            "high"
        )
        
        assert len(optimizer.optimization_opportunities) == 1
        assert "HIGH" in optimizer.optimization_opportunities[0]
        assert "Performance" in optimizer.optimization_opportunities[0]
    
    def test_suggest_alternative(self):
        """Test suggesting alternative models."""
        optimizer = ModelOptimizer("test", {})
        
        optimizer.suggest_alternative("llama3.2:1b", "faster and cheaper")
        
        assert len(optimizer.alternative_models) == 1
        assert "llama3.2:1b" in optimizer.alternative_models[0]
        assert "faster and cheaper" in optimizer.alternative_models[0]


class TestModelComparison:
    """Test ModelComparison functionality."""
    
    def test_comparison_creation(self):
        """Test comparison creation."""
        models = ["model1", "model2", "model3"]
        comparison = ModelComparison(models=models)
        
        assert comparison.models == models
        assert len(comparison.comparison_metrics) == 0
        assert comparison.best_for_cost is None
    
    def test_add_metric(self):
        """Test adding comparison metrics."""
        comparison = ModelComparison(models=["model1", "model2"])
        
        # Add cost metric
        comparison.add_metric("cost_per_inference", {
            "model1": 0.002,
            "model2": 0.001
        })
        
        assert "cost_per_inference" in comparison.comparison_metrics
        assert comparison.best_for_cost == "model2"  # Lower cost
        
        # Add speed metric
        comparison.add_metric("avg_tokens_per_second", {
            "model1": 20.0,
            "model2": 30.0
        })
        
        assert comparison.best_for_speed == "model2"  # Higher speed


class TestOllamaModelManager:
    """Test OllamaModelManager functionality."""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests for HTTP API."""
        with patch('genops.providers.ollama.model_manager.requests') as mock_req:
            yield mock_req
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client."""
        with patch('genops.providers.ollama.model_manager.ollama') as mock_ollama:
            mock_client = Mock()
            mock_ollama.Client.return_value = mock_client
            yield mock_client, mock_ollama
    
    @pytest.fixture
    def manager(self):
        """Create manager instance for testing."""
        with patch('genops.providers.ollama.model_manager.HAS_OLLAMA_CLIENT', False):
            return OllamaModelManager(
                enable_auto_optimization=True,
                track_performance_history=True
            )
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = OllamaModelManager(
            ollama_base_url="http://test:11434",
            enable_auto_optimization=False,
            history_size=500
        )
        
        assert manager.ollama_base_url == "http://test:11434"
        assert manager.enable_auto_optimization is False
        assert manager.history_size == 500
        assert len(manager.models) == 0
    
    def test_discover_models_http_api(self, manager, mock_requests):
        """Test model discovery via HTTP API."""
        mock_response_data = {
            "models": [
                {
                    "name": "llama3.2:1b",
                    "size": 1300000000,  # 1.3GB in bytes
                    "details": {
                        "parameter_size": "1.2B",
                        "family": "llama",
                        "format": "gguf"
                    }
                },
                {
                    "name": "mistral:7b",
                    "size": 7500000000,  # 7.5GB in bytes
                    "details": {
                        "parameter_size": "7B",
                        "family": "mistral"
                    }
                }
            ]
        }
        
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = mock_response_data
        mock_requests.get.return_value.raise_for_status.return_value = None
        
        models = manager.discover_models()
        
        assert len(models) == 2
        assert models[0].name == "llama3.2:1b"
        assert models[0].size_gb == pytest.approx(1.21, abs=0.01)  # 1.3GB
        assert models[0].parameter_count == "1.2B"
        
        assert models[1].name == "mistral:7b"
        assert models[1].size_gb == pytest.approx(6.98, abs=0.01)  # 7.5GB
        
        # Models should be stored in manager
        assert len(manager.models) == 2
        assert "llama3.2:1b" in manager.models
        assert "mistral:7b" in manager.models
    
    def test_discover_models_with_client(self, manager, mock_ollama_client):
        """Test model discovery with Ollama client."""
        mock_client, mock_ollama = mock_ollama_client
        
        mock_client.list.return_value = {
            "models": [
                {"name": "test-model", "size": 5000000000}
            ]
        }
        
        with patch('genops.providers.ollama.model_manager.HAS_OLLAMA_CLIENT', True):
            manager.client = mock_client
            models = manager.discover_models()
        
        assert len(models) == 1
        assert models[0].name == "test-model"
        mock_client.list.assert_called_once()
    
    def test_get_model_info(self, manager):
        """Test getting model information."""
        # Add a test model
        model = ModelInfo("test-model", size_gb=3.0)
        manager.models["test-model"] = model
        
        retrieved = manager.get_model_info("test-model")
        assert retrieved is model
        
        # Test non-existent model
        assert manager.get_model_info("nonexistent") is None
    
    def test_update_model_performance(self, manager):
        """Test updating model performance."""
        model_name = "performance-model"
        
        # Update performance for non-existent model (should create it)
        manager.update_model_performance(
            model_name,
            inference_time_ms=1800.0,
            tokens=45,
            memory_mb=3072.0,
            cost=0.0012
        )
        
        assert model_name in manager.models
        model = manager.models[model_name]
        assert model.total_inferences == 1
        assert model.avg_inference_latency_ms == 1800.0
        
        # Check performance history
        assert model_name in manager.performance_history
        assert len(manager.performance_history[model_name]) == 1
    
    def test_mark_model_error(self, manager):
        """Test marking model errors."""
        model_name = "error-model"
        model = ModelInfo(model_name, size_gb=1.0)
        manager.models[model_name] = model
        
        initial_error_count = model.error_count
        manager.mark_model_error(model_name)
        
        assert model.error_count == initial_error_count + 1
    
    def test_get_model_performance_summary(self, manager):
        """Test getting performance summary."""
        # Add test model with performance data
        model = ModelInfo("summary-model", size_gb=2.0)
        model.update_performance_stats(1500.0, 40, 2048.0, 0.001)
        manager.models["summary-model"] = model
        
        # Test specific model summary
        summary = manager.get_model_performance_summary("summary-model")
        assert summary["model_name"] == "summary-model"
        assert summary["total_inferences"] == 1
        assert summary["avg_inference_latency_ms"] == 1500.0
        
        # Test all models summary
        all_summaries = manager.get_model_performance_summary()
        assert "summary-model" in all_summaries
    
    def test_compare_models(self, manager):
        """Test model comparison functionality."""
        # Add test models
        model1 = ModelInfo("fast-model", size_gb=1.0)
        model1.update_performance_stats(1000.0, 50, 1024.0, 0.0005)
        manager.models["fast-model"] = model1
        
        model2 = ModelInfo("accurate-model", size_gb=5.0)
        model2.update_performance_stats(3000.0, 100, 4096.0, 0.002)
        manager.models["accurate-model"] = model2
        
        comparison = manager.compare_models(["fast-model", "accurate-model"])
        
        assert comparison.models == ["fast-model", "accurate-model"]
        assert "avg_inference_latency_ms" in comparison.comparison_metrics
        assert "cost_per_inference" in comparison.comparison_metrics
        
        # Fast model should be best for speed and cost
        assert comparison.best_for_cost == "fast-model"
        
        # Should have recommendations
        assert len(comparison.recommendations) > 0
    
    def test_optimization_recommendations_generation(self, manager):
        """Test optimization recommendations generation."""
        # Add model with performance issues
        slow_model = ModelInfo("slow-model", size_gb=15.0)
        slow_model.update_performance_stats(8000.0, 3, 12000.0, 0.02)  # Very slow, low throughput, high cost
        slow_model.success_rate = 85.0  # Low success rate
        manager.models["slow-model"] = slow_model
        
        recommendations = manager.get_optimization_recommendations("slow-model")
        
        assert "slow-model" in recommendations
        optimizer = recommendations["slow-model"]
        
        # Should have multiple recommendations for this problematic model
        assert len(optimizer.optimization_opportunities) > 3
        
        # Should recommend latency improvements
        latency_recs = [rec for rec in optimizer.optimization_opportunities if "latency" in rec.lower()]
        assert len(latency_recs) > 0
        
        # Should recommend cost improvements
        cost_recs = [rec for rec in optimizer.optimization_opportunities if "cost" in rec.lower()]
        assert len(cost_recs) > 0
    
    def test_model_usage_analytics(self, manager):
        """Test usage analytics generation."""
        # Add models with different usage patterns
        active_model = ModelInfo("active-model", size_gb=3.0)
        active_model.last_used = time.time() - 3600  # 1 hour ago
        active_model.total_inferences = 100
        active_model.cost_per_inference = 0.001
        manager.models["active-model"] = active_model
        
        inactive_model = ModelInfo("inactive-model", size_gb=7.0)
        inactive_model.last_used = time.time() - (48 * 3600)  # 2 days ago
        inactive_model.total_inferences = 10
        inactive_model.cost_per_inference = 0.003
        manager.models["inactive-model"] = inactive_model
        
        analytics = manager.get_model_usage_analytics(days=1)  # Last 24 hours
        
        assert analytics["total_models"] == 2
        assert analytics["active_models"] == 1  # Only active-model used recently
        assert analytics["total_inferences"] == 110  # 100 + 10
        
        # Should have usage and cost rankings
        assert len(analytics["models_by_usage"]) == 2
        assert len(analytics["models_by_cost"]) == 2
        
        # Most used model should be first
        assert analytics["models_by_usage"][0]["model"] == "active-model"
    
    def test_export_model_data_json(self, manager):
        """Test exporting model data as JSON."""
        # Add test model
        model = ModelInfo("export-model", size_gb=4.0)
        model.update_performance_stats(2000.0, 60, 3000.0, 0.0015)
        manager.models["export-model"] = model
        
        export_data = manager.export_model_data("json")
        
        assert isinstance(export_data, str)
        
        # Parse JSON to verify structure
        data = json.loads(export_data)
        assert "export_timestamp" in data
        assert "models" in data
        assert "export-model" in data["models"]
        
        model_data = data["models"]["export-model"]
        assert model_data["name"] == "export-model"
        assert model_data["size_gb"] == 4.0
        assert model_data["total_inferences"] == 1
    
    def test_export_unsupported_format(self, manager):
        """Test exporting with unsupported format."""
        with pytest.raises(ValueError):
            manager.export_model_data("csv")  # Not implemented
    
    def test_performance_history_tracking(self, manager):
        """Test performance history tracking."""
        model_name = "history-model"
        
        # Add multiple performance updates
        for i in range(5):
            manager.update_model_performance(
                model_name,
                inference_time_ms=1000.0 + i * 100,
                tokens=20 + i * 5,
                cost=0.001 * (i + 1)
            )
        
        # Should have history entries
        assert model_name in manager.performance_history
        history = manager.performance_history[model_name]
        assert len(history) == 5
        
        # Each entry should have required fields
        for entry in history:
            assert "timestamp" in entry
            assert "inference_time_ms" in entry
            assert "tokens" in entry
            assert "cost" in entry


class TestGlobalManagerFunctions:
    """Test global manager functions."""
    
    def test_get_model_manager_singleton(self):
        """Test global manager singleton behavior."""
        # Reset global state
        set_model_manager(None)
        
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        # Should be same instance
        assert manager1 is manager2
    
    def test_set_model_manager(self):
        """Test setting global manager instance."""
        custom_manager = create_model_manager(history_size=200)
        set_model_manager(custom_manager)
        
        retrieved_manager = get_model_manager()
        assert retrieved_manager is custom_manager
        assert retrieved_manager.history_size == 200
    
    def test_create_model_manager_factory(self):
        """Test manager factory function."""
        manager = create_model_manager(
            ollama_base_url="http://custom:11434",
            enable_auto_optimization=False,
            track_performance_history=False,
            history_size=100
        )
        
        assert manager.ollama_base_url == "http://custom:11434"
        assert manager.enable_auto_optimization is False
        assert manager.track_performance_history is False
        assert manager.history_size == 100


class TestModelManagerIntegration:
    """Integration tests for model manager."""
    
    def test_full_model_lifecycle(self, mock_requests):
        """Test complete model lifecycle management."""
        # Mock model discovery
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = {
            "models": [
                {"name": "lifecycle-model", "size": 3000000000}
            ]
        }
        
        manager = OllamaModelManager()
        
        # 1. Discover models
        models = manager.discover_models()
        assert len(models) == 1
        assert models[0].name == "lifecycle-model"
        
        # 2. Update performance multiple times
        for i in range(10):
            manager.update_model_performance(
                "lifecycle-model",
                inference_time_ms=1500.0 + i * 50,
                tokens=30 + i * 2,
                memory_mb=2048.0 + i * 100,
                cost=0.001 + i * 0.0001
            )
        
        # 3. Add some errors
        manager.mark_model_error("lifecycle-model")
        manager.mark_model_error("lifecycle-model")
        
        # 4. Get comprehensive analysis
        model_info = manager.get_model_info("lifecycle-model")
        assert model_info.total_inferences == 10
        assert model_info.error_count == 2
        assert model_info.success_rate == pytest.approx(83.33, abs=0.1)  # 10/(10+2) * 100
        
        # 5. Get optimization recommendations
        recommendations = manager.get_optimization_recommendations("lifecycle-model")
        assert "lifecycle-model" in recommendations
        
        # 6. Get usage analytics
        analytics = manager.get_model_usage_analytics()
        assert analytics["total_inferences"] == 10
        assert analytics["active_models"] == 1
        
        # 7. Export data
        export_data = manager.export_model_data("json")
        data = json.loads(export_data)
        assert "lifecycle-model" in data["models"]


if __name__ == "__main__":
    pytest.main([__file__])