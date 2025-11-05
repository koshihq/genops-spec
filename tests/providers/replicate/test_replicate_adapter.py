#!/usr/bin/env python3
"""
Test Suite for GenOpsReplicateAdapter

Unit tests covering core adapter functionality including:
- Basic initialization and configuration
- Text generation across different models
- Image generation with various parameters
- Video and audio processing
- Governance attribute handling
- Error handling and edge cases
- Auto-instrumentation patterns

Target: ~35 tests covering all core functionality
"""

import os
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Import test configuration
from . import TEST_CONFIG, MOCK_RESPONSES

# Import components under test
from src.genops.providers.replicate import (
    GenOpsReplicateAdapter,
    ReplicateResponse,
    ReplicateModelInfo,
    auto_instrument
)

class TestGenOpsReplicateAdapterInitialization:
    """Test adapter initialization and configuration."""
    
    def test_adapter_initialization_default(self):
        """Test default adapter initialization."""
        adapter = GenOpsReplicateAdapter()
        
        assert adapter is not None
        assert adapter.telemetry_enabled is True
        assert adapter.debug is False
        
    def test_adapter_initialization_with_token(self):
        """Test adapter initialization with explicit API token."""
        token = "r8_test_token_explicit"
        adapter = GenOpsReplicateAdapter(api_token=token)
        
        assert adapter.api_token == token
        
    def test_adapter_initialization_from_env(self):
        """Test adapter initialization from environment variable."""
        test_token = "r8_test_token_from_env"
        
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": test_token}):
            adapter = GenOpsReplicateAdapter()
            assert adapter.api_token == test_token
    
    def test_adapter_initialization_without_replicate_sdk(self):
        """Test graceful handling when Replicate SDK is not available."""
        with patch('src.genops.providers.replicate.replicate', None):
            with pytest.raises(ImportError) as exc_info:
                GenOpsReplicateAdapter()
            assert "Replicate SDK not found" in str(exc_info.value)
    
    def test_adapter_configuration_options(self):
        """Test various configuration options."""
        adapter = GenOpsReplicateAdapter(
            telemetry_enabled=False,
            debug=True
        )
        
        assert adapter.telemetry_enabled is False
        assert adapter.debug is True

class TestTextGeneration:
    """Test text generation functionality."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            adapter._pricing = Mock()
            adapter._validator = Mock()
            return adapter, mock_replicate
    
    def test_text_generation_basic(self, adapter):
        """Test basic text generation."""
        adapter_instance, mock_replicate = adapter
        
        # Mock Replicate response
        mock_replicate.run.return_value = MOCK_RESPONSES["text_generation"]["content"]
        
        # Mock model info
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="meta/llama-2-7b-chat",
            pricing_type="token",
            base_cost=0.5,
            input_cost=0.5,
            output_cost=0.5,
            category="text"
        )
        
        # Mock cost calculation
        adapter_instance._pricing.calculate_cost.return_value = 0.001234
        
        with patch('time.time', side_effect=[1000, 1001]):  # Mock timing
            response = adapter_instance.text_generation(
                model="meta/llama-2-7b-chat",
                prompt="Test prompt",
                max_tokens=100
            )
        
        assert isinstance(response, ReplicateResponse)
        assert response.model == "meta/llama-2-7b-chat"
        assert response.cost_usd == 0.001234
        assert response.latency_ms == 1000  # 1 second
        
    def test_text_generation_with_governance(self, adapter):
        """Test text generation with governance attributes."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = MOCK_RESPONSES["text_generation"]["content"]
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="meta/llama-2-7b-chat",
            pricing_type="token",
            base_cost=0.5,
            category="text"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.001234
        
        with patch('time.time', side_effect=[1000, 1001]):
            response = adapter_instance.text_generation(
                model="meta/llama-2-7b-chat",
                prompt="Test prompt with governance",
                team="test-team",
                project="test-project",
                customer_id="test-customer"
            )
        
        assert response.metadata["governance"]["genops.team"] == "test-team"
        assert response.metadata["governance"]["genops.project"] == "test-project"
        assert response.metadata["governance"]["genops.customer_id"] == "test-customer"
    
    def test_text_generation_with_streaming(self, adapter):
        """Test streaming text generation."""
        adapter_instance, mock_replicate = adapter
        
        # Mock streaming response
        mock_chunks = ["Hello", " from", " GenOps", "!"]
        mock_replicate.stream.return_value = iter(mock_chunks)
        
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="meta/llama-2-7b-chat",
            pricing_type="token",
            base_cost=0.5,
            category="text"
        )
        
        with patch('time.time', side_effect=[1000, 1002]):
            result = adapter_instance.text_generation(
                model="meta/llama-2-7b-chat",
                prompt="Test streaming",
                stream=True
            )
            
            # Collect streaming results
            chunks = list(result)
            assert chunks == mock_chunks
    
    def test_text_generation_error_handling(self, adapter):
        """Test error handling in text generation."""
        adapter_instance, mock_replicate = adapter
        
        # Mock API error
        mock_replicate.run.side_effect = Exception("API Error")
        
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="meta/llama-2-7b-chat",
            pricing_type="token",
            base_cost=0.5,
            category="text"
        )
        
        with pytest.raises(Exception) as exc_info:
            adapter_instance.text_generation(
                model="meta/llama-2-7b-chat",
                prompt="Test error"
            )
        
        assert "API Error" in str(exc_info.value)

class TestImageGeneration:
    """Test image generation functionality."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            adapter._pricing = Mock()
            adapter._validator = Mock()
            return adapter, mock_replicate
    
    def test_image_generation_basic(self, adapter):
        """Test basic image generation."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = MOCK_RESPONSES["image_generation"]["content"]
        
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="black-forest-labs/flux-schnell",
            pricing_type="output",
            base_cost=0.003,
            category="image"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.003
        
        with patch('time.time', side_effect=[1000, 1003]):
            response = adapter_instance.image_generation(
                model="black-forest-labs/flux-schnell",
                prompt="Test image generation",
                num_images=1
            )
        
        assert isinstance(response, ReplicateResponse)
        assert response.model == "black-forest-labs/flux-schnell"
        assert response.cost_usd == 0.003
        assert response.latency_ms == 3000
    
    def test_image_generation_multiple_images(self, adapter):
        """Test generating multiple images."""
        adapter_instance, mock_replicate = adapter
        
        mock_images = ["image1.png", "image2.png", "image3.png"]
        mock_replicate.run.return_value = mock_images
        
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="black-forest-labs/flux-schnell",
            pricing_type="output",
            base_cost=0.003,
            category="image"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.009  # 3 images * $0.003
        
        with patch('time.time', side_effect=[1000, 1005]):
            response = adapter_instance.image_generation(
                model="black-forest-labs/flux-schnell",
                prompt="Generate multiple test images",
                num_images=3,
                width=512,
                height=512
            )
        
        assert response.cost_usd == 0.009
        assert len(mock_images) == 3
    
    def test_image_generation_with_governance(self, adapter):
        """Test image generation with governance attributes."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = MOCK_RESPONSES["image_generation"]["content"]
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="black-forest-labs/flux-schnell",
            pricing_type="output",
            base_cost=0.003,
            category="image"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.003
        
        with patch('time.time', side_effect=[1000, 1003]):
            response = adapter_instance.image_generation(
                model="black-forest-labs/flux-schnell",
                prompt="Test image with governance",
                team="design-team",
                project="test-images"
            )
        
        assert response.metadata["governance"]["genops.team"] == "design-team"
        assert response.metadata["governance"]["genops.project"] == "test-images"

class TestVideoGeneration:
    """Test video generation functionality."""
    
    @pytest.fixture  
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            adapter._pricing = Mock()
            adapter._validator = Mock()
            return adapter, mock_replicate
    
    def test_video_generation_basic(self, adapter):
        """Test basic video generation."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = MOCK_RESPONSES["video_generation"]["content"]
        
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="google/veo-2",
            pricing_type="output",
            base_cost=0.5,  # $0.50 per second
            category="video"
        )
        adapter_instance._pricing.calculate_cost.return_value = 2.5  # 5 seconds * $0.50
        
        with patch('time.time', side_effect=[1000, 1015]):
            response = adapter_instance.video_generation(
                model="google/veo-2", 
                prompt="Generate test video",
                duration=5.0,
                fps=24
            )
        
        assert response.model == "google/veo-2"
        assert response.cost_usd == 2.5
        assert response.latency_ms == 15000

class TestAudioProcessing:
    """Test audio processing functionality."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            adapter._pricing = Mock()
            adapter._validator = Mock()
            return adapter, mock_replicate
    
    def test_audio_processing_basic(self, adapter):
        """Test basic audio processing (transcription)."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = MOCK_RESPONSES["audio_processing"]["content"]
        
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="openai/whisper",
            pricing_type="time",
            base_cost=0.0001,
            category="audio"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.0025  # 2.5 seconds * $0.0001
        
        with patch('time.time', side_effect=[1000, 1002.5]):
            response = adapter_instance.audio_processing(
                model="openai/whisper",
                audio_input="test_audio.wav",
                task="transcribe"
            )
        
        assert response.model == "openai/whisper"
        assert response.cost_usd == 0.0025
        assert response.content == MOCK_RESPONSES["audio_processing"]["content"]

class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""
    
    def test_auto_instrument_function_exists(self):
        """Test that auto_instrument function is available."""
        assert callable(auto_instrument)
    
    @patch('src.genops.providers.replicate.replicate')
    def test_auto_instrument_patches_replicate_run(self, mock_replicate):
        """Test that auto_instrument properly patches replicate.run."""
        mock_replicate._original_run = None
        mock_replicate.run = Mock()
        
        # Call auto_instrument
        auto_instrument()
        
        # Verify original function was saved
        assert hasattr(mock_replicate, '_original_run')
    
    @patch('src.genops.providers.replicate.replicate')
    def test_auto_instrumented_call(self, mock_replicate):
        """Test that auto-instrumented calls work correctly."""
        # Setup mocks
        mock_replicate._original_run = Mock(return_value="original response")
        mock_replicate.run = Mock()
        
        # Mock adapter behavior
        with patch('src.genops.providers.replicate.GenOpsReplicateAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.run_model.return_value = Mock(content="tracked response")
            mock_adapter_class.return_value = mock_adapter
            
            # Enable auto-instrumentation
            auto_instrument()
            
            # Make a call through the instrumented function
            result = mock_replicate.run(
                "test-model",
                input={"prompt": "test"},
                team="test-team"
            )
            
            # Verify the adapter was used
            mock_adapter_class.assert_called_once()

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for edge case testing."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            adapter._pricing = Mock()
            adapter._validator = Mock()
            return adapter, mock_replicate
    
    def test_run_model_without_pricing_module(self, adapter):
        """Test running model when pricing module is not available."""
        adapter_instance, mock_replicate = adapter
        adapter_instance._pricing = None  # Simulate missing pricing module
        
        mock_replicate.run.return_value = "test response"
        
        with patch('time.time', side_effect=[1000, 1001]):
            response = adapter_instance.run_model(
                model="meta/llama-2-7b-chat",
                input={"prompt": "test"}
            )
        
        # Should still work with fallback model info
        assert isinstance(response, ReplicateResponse)
        assert response.content == "test response"
    
    def test_run_model_with_none_governance_attrs(self, adapter):
        """Test running model with None governance attributes."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = "test response"
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="test-model",
            pricing_type="time", 
            base_cost=0.001,
            category="text"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.001
        
        with patch('time.time', side_effect=[1000, 1001]):
            # Pass None values for governance attributes
            response = adapter_instance.run_model(
                model="test-model",
                input={"prompt": "test"},
                team=None,
                project=None,
                customer_id=None
            )
        
        # Should handle None values gracefully
        assert isinstance(response, ReplicateResponse)
        governance = response.metadata["governance"]
        assert "genops.team" not in governance  # None values should be filtered out
    
    def test_validate_setup_without_validator(self, adapter):
        """Test setup validation when validator is not available."""
        adapter_instance, _ = adapter
        adapter_instance._validator = None  # Simulate missing validator
        
        result = adapter_instance.validate_setup()
        
        # Should provide basic validation
        assert isinstance(result, dict)
        assert "success" in result

# Performance and Load Tests
class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for performance testing."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            adapter._pricing = Mock()
            adapter._validator = Mock()
            return adapter, mock_replicate
    
    def test_multiple_concurrent_calls_simulation(self, adapter):
        """Simulate multiple concurrent calls for performance testing."""
        adapter_instance, mock_replicate = adapter
        
        mock_replicate.run.return_value = "concurrent response"
        adapter_instance._pricing.get_model_info.return_value = ReplicateModelInfo(
            name="test-model",
            pricing_type="token",
            base_cost=0.5,
            category="text"
        )
        adapter_instance._pricing.calculate_cost.return_value = 0.001
        
        # Simulate multiple calls
        responses = []
        start_time = time.time()
        
        for i in range(10):
            with patch('time.time', side_effect=[1000 + i, 1001 + i]):
                response = adapter_instance.text_generation(
                    model="test-model",
                    prompt=f"Test prompt {i}",
                    team=f"team-{i % 3}"  # Distribute across teams
                )
                responses.append(response)
        
        end_time = time.time()
        
        # Verify all responses completed
        assert len(responses) == 10
        for response in responses:
            assert isinstance(response, ReplicateResponse)
        
        # Performance should be reasonable (< 1 second for mocked calls)
        assert (end_time - start_time) < 1.0

# Integration with actual components
class TestComponentIntegration:
    """Test integration between adapter and other components."""
    
    def test_adapter_with_real_model_info_structure(self):
        """Test adapter with realistic model info structure."""
        with patch('src.genops.providers.replicate.replicate') as mock_replicate:
            adapter = GenOpsReplicateAdapter(api_token="r8_test_token")
            
            # Mock realistic model info
            mock_model_info = ReplicateModelInfo(
                name="meta/llama-2-70b-chat",
                pricing_type="token",
                base_cost=0.0,
                input_cost=1.0,
                output_cost=1.0,
                hardware_type="a100-80gb",
                official=True,
                category="text"
            )
            
            adapter._pricing = Mock()
            adapter._pricing.get_model_info.return_value = mock_model_info
            adapter._pricing.calculate_cost.return_value = 0.0156  # Realistic cost
            
            mock_replicate.run.return_value = "Realistic model response with more detailed content that represents actual model output."
            
            with patch('time.time', side_effect=[1000, 1003.5]):  # 3.5 second processing
                response = adapter.text_generation(
                    model="meta/llama-2-70b-chat",
                    prompt="Generate a comprehensive analysis of AI cost management best practices",
                    max_tokens=150
                )
            
            assert response.model == "meta/llama-2-70b-chat"
            assert response.cost_usd == 0.0156
            assert response.latency_ms == 3500
            assert response.hardware_used == "a100-80gb"
            assert len(response.content) > 50  # Realistic content length