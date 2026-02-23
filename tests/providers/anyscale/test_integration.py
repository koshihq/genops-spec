"""Integration tests for Anyscale provider."""

import time
from unittest.mock import Mock, patch

import pytest

from genops.providers.anyscale import (
    auto_instrument,
    calculate_completion_cost,
    instrument_anyscale,
    validate_setup,
)


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_complete_completion_workflow(self, mock_requests):
        """Test complete workflow from adapter creation to response."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-123",
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "choices": [
                {"message": {"content": "Test response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_requests.post.return_value = mock_response

        # Create adapter
        adapter = instrument_anyscale(anyscale_api_key="test-key", team="test-team")

        # Make request
        response = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
        )

        # Verify response
        assert response["usage"]["total_tokens"] == 15

        # Calculate cost
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-70b-chat-hf", input_tokens=10, output_tokens=5
        )
        assert cost > 0

    @patch("genops.providers.anyscale.adapter.requests")
    def test_complete_embeddings_workflow(self, mock_requests):
        """Test complete embeddings workflow."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024}],
            "model": "thenlper/gte-large",
            "usage": {"total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        response = adapter.embeddings_create(
            model="thenlper/gte-large", input="Test text"
        )

        assert len(response["data"][0]["embedding"]) == 1024

    def test_validation_before_use(self):
        """Test validation can be run before using adapter."""
        # Run validation
        result = validate_setup(anyscale_api_key="test-key")

        # Validation should complete
        assert result is not None
        assert result.total_checks > 0

    @patch("genops.providers.anyscale.adapter.requests")
    def test_multi_request_workflow(self, mock_requests):
        """Test multiple requests with same adapter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        # Make multiple requests
        for i in range(3):
            response = adapter.completion_create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": f"test {i}"}],
            )
            assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_governance_context_workflow(self, mock_requests):
        """Test governance context manager workflow."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key", team="base-team")

        with adapter.governance_context(customer_id="customer-123"):
            response = adapter.completion_create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )
            assert response is not None


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration."""

    def test_auto_instrument_registration(self):
        """Test auto_instrument function exists and is callable."""
        assert callable(auto_instrument)

    def test_auto_instrument_with_governance(self):
        """Test auto_instrument with governance defaults."""
        result = auto_instrument(team="test-team", project="test-project")

        # Should return True if successful
        assert isinstance(result, bool)

    @patch("genops.providers.anyscale.registration.OpenAI")
    def test_auto_instrument_patches_openai(self, mock_openai):
        """Test auto_instrument attempts to patch OpenAI SDK."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        result = auto_instrument(team="test-team")

        # Should succeed
        assert result is True


class TestMultiModelIntegration:
    """Test integration across multiple models."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_switch_between_models(self, mock_requests):
        """Test switching between different models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "test-model",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        models = [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
        ]

        for model in models:
            response = adapter.completion_create(
                model=model, messages=[{"role": "user", "content": "test"}]
            )
            assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_chat_and_embeddings_mixed(self, mock_requests):
        """Test mixing chat and embedding requests."""

        def mock_response_factory(*args, **kwargs):
            response = Mock()
            response.status_code = 200
            if "embeddings" in kwargs.get("url", ""):
                response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1024}],
                    "usage": {"total_tokens": 5},
                }
            else:
                response.json.return_value = {
                    "id": "test",
                    "choices": [
                        {"message": {"content": "response"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                    },
                }
            return response

        mock_requests.post.side_effect = mock_response_factory

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        # Chat completion
        chat_response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
        )

        # Embeddings
        embed_response = adapter.embeddings_create(
            model="thenlper/gte-large", input="test"
        )

        assert chat_response is not None
        assert embed_response is not None


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_api_error_recovery(self, mock_requests):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        with pytest.raises(Exception):  # noqa: B017
            adapter.completion_create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("genops.providers.anyscale.adapter.requests")
    def test_network_timeout_handling(self, mock_requests):
        """Test handling of network timeouts."""
        import requests

        mock_requests.post.side_effect = requests.exceptions.Timeout("Timeout")

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        with pytest.raises(Exception):  # noqa: B017
            adapter.completion_create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("genops.providers.anyscale.adapter.requests")
    def test_invalid_response_handling(self, mock_requests):
        """Test handling of invalid API responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # Invalid response structure

        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        # Should handle gracefully or raise appropriate error
        try:
            adapter.completion_create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )
        except (KeyError, AttributeError, Exception):
            # Expected to raise error for invalid response
            pass


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_operation_timing(self, mock_requests):
        """Test operation timing is tracked."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        start_time = time.time()
        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
        )
        duration = time.time() - start_time

        # Operation should complete quickly (< 1s for mocked response)
        assert duration < 1.0
        assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_concurrent_requests_handling(self, mock_requests):
        """Test adapter can handle multiple requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "test-model",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        # Make rapid sequential requests
        for _ in range(10):
            response = adapter.completion_create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )
            assert response is not None


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    @patch.dict("os.environ", {"ANYSCALE_API_KEY": "env-key-123"})
    def test_env_var_configuration(self):
        """Test adapter uses environment variables."""
        adapter = instrument_anyscale()

        assert adapter.anyscale_api_key == "env-key-123"

    def test_explicit_configuration_override(self):
        """Test explicit configuration overrides defaults."""
        adapter = instrument_anyscale(
            anyscale_api_key="explicit-key", anyscale_base_url="https://custom.com/v1"
        )

        assert adapter.anyscale_api_key == "explicit-key"
        assert adapter.anyscale_base_url == "https://custom.com/v1"

    def test_telemetry_configuration(self):
        """Test telemetry can be configured."""
        adapter_enabled = instrument_anyscale(telemetry_enabled=True)
        adapter_disabled = instrument_anyscale(telemetry_enabled=False)

        assert adapter_enabled.telemetry_enabled is True
        assert adapter_disabled.telemetry_enabled is False
