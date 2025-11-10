#!/usr/bin/env python3
"""
Test suite for GenOps Gemini adapter.

This module tests the core GenOpsGeminiAdapter functionality including:
- Adapter initialization and configuration
- Text generation with governance attributes
- Cost calculation and telemetry
- Error handling and validation
- Multi-model support
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock google.genai module before importing our adapter
genai_mock = MagicMock()
genai_mock.Client = MagicMock()

with patch.dict('sys.modules', {'google': MagicMock(), 'google.genai': genai_mock}):
    from genops.providers.gemini import (
        GeminiOperationResult,
        GenOpsGeminiAdapter,
    )


class TestGeminiAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_init_with_api_key(self):
        """Test adapter initialization with explicit API key."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                adapter = GenOpsGeminiAdapter(api_key="test_key_123")

                assert adapter.api_key == "test_key_123"
                assert adapter.default_model == "gemini-2.5-flash"
                assert adapter.enable_streaming is True

    def test_adapter_init_with_env_api_key(self):
        """Test adapter initialization using environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key_456"}):
            with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
                with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                    adapter = GenOpsGeminiAdapter()

                    assert adapter.api_key == "env_key_456"

    def test_adapter_init_without_api_key_raises_error(self):
        """Test that adapter raises error when no API key is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
                with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                    with pytest.raises(ValueError, match="Gemini API key required"):
                        GenOpsGeminiAdapter()

    def test_adapter_init_without_gemini_sdk_raises_error(self):
        """Test that adapter raises error when Gemini SDK is not available."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', False):
            with pytest.raises(ImportError, match="Google Gemini dependencies not available"):
                GenOpsGeminiAdapter(api_key="test_key")

    def test_adapter_init_with_custom_model(self):
        """Test adapter initialization with custom default model."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                adapter = GenOpsGeminiAdapter(
                    api_key="test_key",
                    default_model="gemini-2.5-pro"
                )

                assert adapter.default_model == "gemini-2.5-pro"

    def test_adapter_client_initialization(self):
        """Test that Gemini client is properly initialized."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                with patch('genops.providers.gemini.genai.Client') as mock_client:
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    mock_client.assert_called_once_with(api_key="test_key")
                    assert adapter.client == mock_client.return_value


class TestGeminiAdapterAvailability:
    """Test adapter availability checking."""

    def test_is_available_with_successful_api_call(self):
        """Test availability check with successful API call."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Hello"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    assert adapter.is_available() is True
                    mock_client.models.generate_content.assert_called_once_with(
                        model="gemini-2.5-flash",
                        contents="Hello"
                    )

    def test_is_available_with_failed_api_call(self):
        """Test availability check with failed API call."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                mock_client = MagicMock()
                mock_client.models.generate_content.side_effect = Exception("API Error")

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    assert adapter.is_available() is False

    def test_is_available_when_gemini_not_available(self):
        """Test availability check when Gemini SDK is not available."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', False):
            # This should not be called since adapter init would fail
            # But testing the method in isolation
            adapter = Mock()
            adapter.is_available = GenOpsGeminiAdapter.is_available.__get__(adapter, GenOpsGeminiAdapter)

            assert adapter.is_available() is False


class TestGeminiAdapterModelSupport:
    """Test model support functionality."""

    def test_get_supported_models_from_api(self):
        """Test getting supported models from API."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                mock_client = MagicMock()
                mock_model1 = MagicMock()
                mock_model1.name = "gemini-2.5-flash"
                mock_model2 = MagicMock()
                mock_model2.name = "gemini-2.5-pro"
                mock_client.models.list.return_value = [mock_model1, mock_model2]

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    models = adapter.get_supported_models()

                    assert "gemini-2.5-flash" in models
                    assert "gemini-2.5-pro" in models

    def test_get_supported_models_fallback(self):
        """Test getting supported models with API failure fallback."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                mock_client = MagicMock()
                mock_client.models.list.side_effect = Exception("API Error")

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    models = adapter.get_supported_models()

                    # Should return fallback list
                    assert "gemini-2.5-pro" in models
                    assert "gemini-2.5-flash" in models
                    assert "gemini-1.5-pro" in models

    def test_get_supported_tasks(self):
        """Test getting list of supported AI tasks."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                adapter = GenOpsGeminiAdapter(api_key="test_key")

                tasks = adapter.get_supported_tasks()

                assert "text-generation" in tasks
                assert "chat-completion" in tasks
                assert "content-generation" in tasks
                assert "streaming-generation" in tasks


class TestGeminiAdapterTextGeneration:
    """Test text generation functionality."""

    def test_text_generation_basic(self):
        """Test basic text generation without governance attributes."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):  # Test without telemetry
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Generated text response"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key")

                        result = adapter.text_generation(
                            prompt="Test prompt",
                            model="gemini-2.5-flash"
                        )

                        assert isinstance(result, GeminiOperationResult)
                        assert result.content == "Generated text response"
                        assert result.model_id == "gemini-2.5-flash"
                        assert result.latency_ms > 0
                        assert result.cost_usd >= 0

    def test_text_generation_with_governance_attributes(self):
        """Test text generation with governance attributes."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Generated text response"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key")

                        result = adapter.text_generation(
                            prompt="Test prompt",
                            model="gemini-2.5-flash",
                            team="test-team",
                            project="test-project",
                            customer_id="customer-123"
                        )

                        assert result.governance_attributes["team"] == "test-team"
                        assert result.governance_attributes["project"] == "test-project"
                        assert result.governance_attributes["customer_id"] == "customer-123"

    def test_text_generation_with_parameters(self):
        """Test text generation with various parameters."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Generated text response"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key")

                        result = adapter.text_generation(
                            prompt="Test prompt",
                            model="gemini-2.5-pro",
                            max_tokens=100,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=40
                        )

                        # Verify that parameters were passed to the API call
                        call_args = mock_client.models.generate_content.call_args
                        assert call_args[1]["model"] == "gemini-2.5-pro"

                        generation_config = call_args[1].get("generation_config", {})
                        assert generation_config.get("max_output_tokens") == 100
                        assert generation_config.get("temperature") == 0.7
                        assert generation_config.get("top_p") == 0.9
                        assert generation_config.get("top_k") == 40

    def test_text_generation_with_default_model(self):
        """Test text generation using default model."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Generated text response"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key", default_model="gemini-2.5-pro")

                        result = adapter.text_generation(prompt="Test prompt")

                        assert result.model_id == "gemini-2.5-pro"

    def test_text_generation_with_usage_metadata(self):
        """Test text generation with API response including usage metadata."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Generated text response"

                # Mock usage metadata
                mock_usage = MagicMock()
                mock_usage.prompt_token_count = 10
                mock_usage.candidates_token_count = 15
                mock_response.usage_metadata = mock_usage

                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key")

                        result = adapter.text_generation(prompt="Test prompt")

                        assert result.input_tokens == 10
                        assert result.output_tokens == 15

    def test_text_generation_api_error(self):
        """Test text generation with API error."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_client.models.generate_content.side_effect = Exception("API Error")

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    with pytest.raises(Exception, match="API Error"):
                        adapter.text_generation(prompt="Test prompt")


class TestGeminiAdapterChatCompletion:
    """Test chat completion functionality."""

    def test_chat_completion_basic(self):
        """Test basic chat completion."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Assistant response"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key")

                        messages = [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                            {"role": "user", "content": "How are you?"}
                        ]

                        result = adapter.chat_completion(messages=messages)

                        assert isinstance(result, GeminiOperationResult)
                        assert result.content == "Assistant response"

    def test_chat_completion_with_system_message(self):
        """Test chat completion with system message."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', False):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Assistant response"
                mock_client.models.generate_content.return_value = mock_response

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                        adapter = GenOpsGeminiAdapter(api_key="test_key")

                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello"}
                        ]

                        result = adapter.chat_completion(messages=messages)

                        # Should convert to combined prompt including system message
                        call_args = mock_client.models.generate_content.call_args
                        combined_prompt = call_args[1]["contents"]
                        assert "System: You are a helpful assistant." in combined_prompt
                        assert "User: Hello" in combined_prompt


class TestGeminiAdapterTelemetry:
    """Test telemetry integration."""

    def test_text_generation_with_telemetry(self):
        """Test text generation with full telemetry enabled."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Generated text response"
                mock_client.models.generate_content.return_value = mock_response

                mock_telemetry = MagicMock()
                mock_span = MagicMock()
                mock_telemetry.trace_operation.return_value.__enter__ = Mock(return_value=mock_span)
                mock_telemetry.trace_operation.return_value.__exit__ = Mock(return_value=None)

                with patch('genops.providers.gemini.genai.Client', return_value=mock_client):
                    with patch('genops.providers.gemini.GenOpsTelemetry', return_value=mock_telemetry):
                        with patch('genops.providers.gemini.calculate_gemini_cost', return_value=0.001234):
                            adapter = GenOpsGeminiAdapter(api_key="test_key")

                            result = adapter.text_generation(
                                prompt="Test prompt",
                                team="test-team",
                                project="test-project"
                            )

                            # Verify telemetry was called
                            mock_telemetry.trace_operation.assert_called_once()

                            # Verify span attributes were set
                            mock_span.set_attributes.assert_called()
                            call_args = mock_span.set_attributes.call_args[0][0]

                            assert call_args["genops.provider"] == "gemini"
                            assert call_args["genops.operation_type"] == "text_generation"
                            assert call_args["genops.cost.total"] == 0.001234
                            assert call_args["genops.cost.currency"] == "USD"


class TestGeminiAdapterUtilities:
    """Test utility methods."""

    def test_calculate_tokens(self):
        """Test token calculation utility."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                adapter = GenOpsGeminiAdapter(api_key="test_key")

                # Test with simple text
                tokens = adapter._calculate_tokens("Hello world")
                assert tokens >= 2  # At least 2 tokens for "Hello world"

                # Test with longer text
                long_text = "This is a longer text that should result in more tokens"
                long_tokens = adapter._calculate_tokens(long_text)
                assert long_tokens > tokens

                # Test with empty text
                empty_tokens = adapter._calculate_tokens("")
                assert empty_tokens == 1  # Should return at least 1

    def test_extract_response_content(self):
        """Test response content extraction."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                adapter = GenOpsGeminiAdapter(api_key="test_key")

                # Test with normal response
                mock_response = MagicMock()
                mock_response.text = "Test response"
                mock_usage = MagicMock()
                mock_usage.candidates_token_count = 10
                mock_response.usage_metadata = mock_usage

                content, tokens = adapter._extract_response_content(mock_response, "gemini-2.5-flash")

                assert content == "Test response"
                assert tokens == 10

    def test_create_operation_context(self):
        """Test operation context creation."""
        with patch('genops.providers.gemini.GEMINI_AVAILABLE', True):
            with patch('genops.providers.gemini.GENOPS_AVAILABLE', True):
                adapter = GenOpsGeminiAdapter(api_key="test_key")

                context = adapter._create_operation_context(
                    "test_operation",
                    "gemini-2.5-flash",
                    team="test-team",
                    project="test-project"
                )

                assert context.operation_name == "test_operation"
                assert context.provider == "gemini"
                assert context.model == "gemini-2.5-flash"
                assert hasattr(context, 'operation_id')
                assert len(context.operation_id) > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
