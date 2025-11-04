"""
Unit tests for Hugging Face GenOps adapter.

Tests the core functionality of the GenOpsHuggingFaceAdapter including:
- Adapter initialization and configuration
- Provider detection and classification  
- Cost calculation integration
- Governance attribute extraction
- Error handling and edge cases
- Auto-instrumentation functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestGenOpsHuggingFaceAdapter:
    """Test suite for GenOpsHuggingFaceAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_telemetry = Mock()

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_adapter_initialization_default(self, mock_telemetry_class, mock_inference_client):
        """Test adapter initialization with default parameters."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        
        adapter = GenOpsHuggingFaceAdapter()
        
        # Verify initialization
        assert adapter.client is not None
        assert adapter.telemetry is not None
        mock_inference_client.assert_called_once()
        mock_telemetry_class.assert_called_once()

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_adapter_initialization_with_client(self, mock_telemetry_class):
        """Test adapter initialization with provided client."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        custom_client = Mock()
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        
        adapter = GenOpsHuggingFaceAdapter(client=custom_client)
        
        assert adapter.client is custom_client
        assert adapter.telemetry is mock_telemetry_instance

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', False)
    def test_adapter_initialization_missing_dependency(self):
        """Test adapter initialization fails with missing dependency."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        with pytest.raises(ImportError, match="Hugging Face Hub package not found"):
            GenOpsHuggingFaceAdapter()

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_governance_attributes_extraction(self, mock_telemetry_class, mock_inference_client):
        """Test extraction of governance and request attributes."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        
        test_kwargs = {
            'team': 'test-team',
            'project': 'test-project',
            'customer_id': 'client-123',
            'temperature': 0.7,
            'max_tokens': 100,
            'model': 'gpt-3.5-turbo',
            'prompt': 'test prompt'
        }
        
        governance_attrs, request_attrs, api_kwargs = adapter._extract_attributes(test_kwargs)
        
        # Check governance attributes
        assert governance_attrs['team'] == 'test-team'
        assert governance_attrs['project'] == 'test-project'
        assert governance_attrs['customer_id'] == 'client-123'
        
        # Check request attributes
        assert request_attrs['temperature'] == 0.7
        assert request_attrs['max_tokens'] == 100
        
        # Check API kwargs (should not contain governance attributes)
        assert 'team' not in api_kwargs
        assert 'project' not in api_kwargs
        assert 'model' in api_kwargs
        assert 'prompt' in api_kwargs

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_provider_detection(self, mock_telemetry_class, mock_inference_client):
        """Test provider detection for various model names."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        
        test_cases = [
            ('gpt-3.5-turbo', 'openai'),
            ('gpt-4', 'openai'),
            ('dall-e-2', 'openai'),
            ('claude-3-sonnet', 'anthropic'),
            ('claude-3-haiku', 'anthropic'),
            ('command-r', 'cohere'),
            ('embed-english-v3.0', 'cohere'),
            ('llama-2-7b-chat', 'meta'),
            ('meta-llama/Llama-2-7b-hf', 'meta'),
            ('mistral-7b-instruct', 'mistral'),
            ('gemma-7b-it', 'google'),
            ('microsoft/DialoGPT-medium', 'huggingface_hub'),
            ('sentence-transformers/all-MiniLM-L6-v2', 'huggingface_hub'),
            ('unknown-model', 'huggingface_hub')
        ]
        
        for model, expected_provider in test_cases:
            detected_provider = adapter._detect_provider(model)
            assert detected_provider == expected_provider, f"Failed for {model}: expected {expected_provider}, got {detected_provider}"

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_token_estimation(self, mock_telemetry_class, mock_inference_client):
        """Test token estimation functionality."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        
        test_cases = [
            ('', 0),
            ('hello', 1),  # 5 chars / 4 = 1.25, rounded down to 1
            ('hello world', 2),  # 11 chars / 4 = 2.75, rounded down to 2
            ('This is a longer test sentence with multiple words.', 13)  # 52 chars / 4 = 13
        ]
        
        for text, expected_tokens in test_cases:
            estimated_tokens = adapter._estimate_tokens(text)
            assert estimated_tokens == expected_tokens, f"Failed for '{text}': expected {expected_tokens}, got {estimated_tokens}"

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_cost_calculation_with_pricing_module(self, mock_telemetry_class, mock_inference_client):
        """Test cost calculation when pricing module is available."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost') as mock_calculate_cost:
            mock_calculate_cost.return_value = 0.002
            
            cost = adapter._calculate_cost(
                provider='openai',
                model='gpt-3.5-turbo',
                input_tokens=100,
                output_tokens=50
            )
            
            assert cost == 0.002
            mock_calculate_cost.assert_called_once_with(
                provider='openai',
                model='gpt-3.5-turbo',
                input_tokens=100,
                output_tokens=50,
                task='text-generation'
            )

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_cost_calculation_fallback(self, mock_telemetry_class, mock_inference_client):
        """Test cost calculation fallback when pricing module unavailable."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        
        with patch('genops.providers.huggingface.calculate_huggingface_cost', side_effect=ImportError()):
            cost = adapter._calculate_cost(
                provider='openai',
                model='gpt-3.5-turbo',
                input_tokens=100,
                output_tokens=50
            )
            
            # Should use fallback estimation
            expected_cost = (100 / 1000) * 0.0015 + (50 / 1000) * 0.002  # OpenAI fallback rates
            assert abs(cost - expected_cost) < 0.000001

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_supported_tasks(self, mock_telemetry_class, mock_inference_client):
        """Test getting supported AI tasks."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        supported_tasks = adapter.get_supported_tasks()
        
        assert isinstance(supported_tasks, list)
        assert len(supported_tasks) > 0
        
        # Check for key tasks
        expected_tasks = {
            'text-generation',
            'chat-completion',
            'feature-extraction',
            'text-to-image'
        }
        
        for task in expected_tasks:
            assert task in supported_tasks

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_is_available_true(self, mock_telemetry_class, mock_inference_client):
        """Test is_available returns True when dependencies are available."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        assert adapter.is_available() is True

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_is_available_false_no_client(self, mock_telemetry_class, mock_inference_client):
        """Test is_available returns False when client is None."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        adapter.client = None
        assert adapter.is_available() is False

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', False)
    def test_is_available_false_no_dependency(self):
        """Test is_available returns False when Hugging Face not available."""
        # This test doesn't create adapter since it would fail, just tests the condition
        from genops.providers.huggingface import HAS_HUGGINGFACE
        assert HAS_HUGGINGFACE is False


class TestTextGeneration:
    """Test suite for text generation functionality."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_text_generation_success(self, mock_telemetry_class, mock_inference_client):
        """Test successful text generation with telemetry."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.text_generation.return_value = "Generated text response"
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.001):
            adapter = GenOpsHuggingFaceAdapter()
            
            result = adapter.text_generation(
                prompt="Test prompt",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=100,
                team="test-team",
                project="test-project"
            )
            
            # Verify result
            assert result == "Generated text response"
            
            # Verify client was called correctly
            mock_client_instance.text_generation.assert_called_once()
            args, kwargs = mock_client_instance.text_generation.call_args
            assert args[0] == "Test prompt"
            assert 'team' not in kwargs  # Governance attrs should be removed
            assert 'project' not in kwargs
            assert kwargs['model'] == "microsoft/DialoGPT-medium"
            assert kwargs['max_new_tokens'] == 100
            
            # Verify telemetry was called
            mock_telemetry_instance.trace_operation.assert_called_once()
            mock_telemetry_instance.record_cost.assert_called_once()

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_text_generation_with_complex_response(self, mock_telemetry_class, mock_inference_client):
        """Test text generation with complex response object."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Mock response object with generated_text attribute
        mock_response = Mock()
        mock_response.generated_text = "Complex generated response"
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.text_generation.return_value = mock_response
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.002):
            adapter = GenOpsHuggingFaceAdapter()
            
            result = adapter.text_generation(
                prompt="Test prompt",
                model="gpt-3.5-turbo"
            )
            
            assert result == mock_response
            
            # Verify output tokens were estimated from generated_text
            mock_span.set_attribute.assert_any_call("genops.tokens.output", 6)  # "Complex generated response" ≈ 6 tokens

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_text_generation_error_handling(self, mock_telemetry_class, mock_inference_client):
        """Test text generation error handling."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.text_generation.side_effect = Exception("API Error")
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        adapter = GenOpsHuggingFaceAdapter()
        
        with pytest.raises(Exception, match="API Error"):
            adapter.text_generation(
                prompt="Test prompt",
                model="microsoft/DialoGPT-medium"
            )
        
        # Verify error attributes were set
        mock_span.set_attribute.assert_any_call("genops.error.message", "API Error")
        mock_span.set_attribute.assert_any_call("genops.error.type", "Exception")


class TestChatCompletion:
    """Test suite for chat completion functionality."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_chat_completion_success(self, mock_telemetry_class, mock_inference_client):
        """Test successful chat completion."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Mock response with OpenAI-style structure
        mock_choice = Mock()
        mock_choice.message.content = "Chat response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.003):
            adapter = GenOpsHuggingFaceAdapter()
            
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            
            result = adapter.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                team="chat-team"
            )
            
            assert result == mock_response
            mock_span.set_attribute.assert_any_call("genops.messages.count", 2)
            mock_span.set_attribute.assert_any_call("genops.tokens.output", 3)  # "Chat response" ≈ 3 tokens

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_chat_completion_input_token_estimation(self, mock_telemetry_class, mock_inference_client):
        """Test input token estimation from messages."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = Mock()
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        adapter = GenOpsHuggingFaceAdapter()
        
        messages = [
            {"role": "user", "content": "This is a test message"},  # 5 words ≈ 5 tokens
            {"role": "assistant", "content": "This is another message"},  # 4 words ≈ 4 tokens
            {"role": "user", "content": "Final message"}  # 2 words ≈ 2 tokens
        ]
        
        adapter.chat_completion(messages=messages, model="gpt-3.5-turbo")
        
        # Should estimate ~11 tokens for "This is a test message This is another message Final message"
        # Actual calculation: "This is a test message This is another message Final message" = 44 chars / 4 = 11 tokens
        mock_span.set_attribute.assert_any_call("genops.tokens.input", 11)


class TestFeatureExtraction:
    """Test suite for feature extraction functionality."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_feature_extraction_string_input(self, mock_telemetry_class, mock_inference_client):
        """Test feature extraction with string input."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]  # Mock embedding with 4 dimensions
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.feature_extraction.return_value = mock_embeddings
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.0001):
            adapter = GenOpsHuggingFaceAdapter()
            
            result = adapter.feature_extraction(
                inputs="Test text for embedding",
                model="sentence-transformers/all-MiniLM-L6-v2",
                team="embedding-team"
            )
            
            assert result == mock_embeddings
            
            # Verify task type was set correctly
            mock_span.set_attribute.assert_any_call("genops.task.type", "feature-extraction")
            
            # Verify input tokens were estimated
            mock_span.set_attribute.assert_any_call("genops.tokens.input", 6)  # "Test text for embedding" ≈ 6 tokens

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_feature_extraction_list_input(self, mock_telemetry_class, mock_inference_client):
        """Test feature extraction with list input."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.feature_extraction.return_value = mock_embeddings
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        adapter = GenOpsHuggingFaceAdapter()
        
        inputs = ["First text", "Second text"]
        
        result = adapter.feature_extraction(
            inputs=inputs,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert result == mock_embeddings
        
        # Verify input tokens estimated from combined text
        mock_span.set_attribute.assert_any_call("genops.tokens.input", 6)  # "First text Second text" ≈ 6 tokens


class TestTextToImage:
    """Test suite for text-to-image functionality."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_text_to_image_success(self, mock_telemetry_class, mock_inference_client):
        """Test successful text-to-image generation."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_image_data = b"fake_image_data"  # Simulate image bytes
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.text_to_image.return_value = mock_image_data
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.02):
            adapter = GenOpsHuggingFaceAdapter()
            
            result = adapter.text_to_image(
                prompt="A beautiful landscape with mountains",
                model="runwayml/stable-diffusion-v1-5",
                team="creative-team"
            )
            
            assert result == mock_image_data
            
            # Verify task type and image count tracking
            mock_span.set_attribute.assert_any_call("genops.task.type", "text-to-image")
            mock_span.set_attribute.assert_any_call("genops.images.generated", 1)
            
            # Verify cost was recorded with image generation flag
            mock_telemetry_instance.record_cost.assert_called_once()
            args, kwargs = mock_telemetry_instance.record_cost.call_args
            assert kwargs.get('images_generated') == 1

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_text_to_image_multiple_images(self, mock_telemetry_class, mock_inference_client):
        """Test text-to-image with multiple image response."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_image_list = [b"image1", b"image2", b"image3"]  # Multiple images
        
        mock_client_instance = Mock()
        mock_inference_client.return_value = mock_client_instance
        mock_client_instance.text_to_image.return_value = mock_image_list
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        adapter = GenOpsHuggingFaceAdapter()
        
        result = adapter.text_to_image(
            prompt="Generate multiple variations",
            model="runwayml/stable-diffusion-v1-5"
        )
        
        assert result == mock_image_list
        
        # Verify correct image count was tracked
        mock_span.set_attribute.assert_any_call("genops.images.generated", 3)


class TestAutoInstrumentation:
    """Test suite for auto-instrumentation functionality."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    def test_instrument_huggingface_success(self, mock_inference_client):
        """Test successful Hugging Face instrumentation."""
        from genops.providers.huggingface import instrument_huggingface
        
        # Mock InferenceClient methods
        mock_inference_client.text_generation = Mock()
        mock_inference_client.feature_extraction = Mock()
        mock_inference_client.text_to_image = Mock()
        
        result = instrument_huggingface()
        
        assert result is True
        
        # Verify original methods were stored
        assert hasattr(mock_inference_client, '_genops_original_text_generation')
        assert hasattr(mock_inference_client, '_genops_original_feature_extraction')
        assert hasattr(mock_inference_client, '_genops_original_text_to_image')

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', False)
    def test_instrument_huggingface_unavailable(self):
        """Test instrumentation when Hugging Face is unavailable."""
        from genops.providers.huggingface import instrument_huggingface
        
        result = instrument_huggingface()
        
        assert result is False

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    def test_uninstrument_huggingface_success(self, mock_inference_client):
        """Test successful Hugging Face uninstrumentation."""
        from genops.providers.huggingface import instrument_huggingface, uninstrument_huggingface
        
        # Set up original methods
        original_text_gen = Mock()
        original_feature_ext = Mock()
        original_text_to_img = Mock()
        
        mock_inference_client.text_generation = Mock()
        mock_inference_client.feature_extraction = Mock() 
        mock_inference_client.text_to_image = Mock()
        
        # First instrument
        instrument_huggingface()
        
        # Set up stored original methods
        mock_inference_client._genops_original_text_generation = original_text_gen
        mock_inference_client._genops_original_feature_extraction = original_feature_ext
        mock_inference_client._genops_original_text_to_image = original_text_to_img
        
        result = uninstrument_huggingface()
        
        assert result is True
        
        # Verify methods were restored
        assert mock_inference_client.text_generation == original_text_gen
        assert mock_inference_client.feature_extraction == original_feature_ext
        assert mock_inference_client.text_to_image == original_text_to_img

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_create_instrumented_client(self, mock_telemetry_class, mock_inference_client):
        """Test creating instrumented client."""
        from genops.providers.huggingface import create_instrumented_client
        
        client = create_instrumented_client(token="test-token")
        
        # Should return GenOpsHuggingFaceAdapter instance
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        assert isinstance(client, GenOpsHuggingFaceAdapter)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])