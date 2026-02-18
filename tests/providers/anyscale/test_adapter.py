"""Tests for Anyscale adapter functionality."""

import time
from unittest.mock import Mock, patch

import pytest

from genops.providers.anyscale.adapter import (
    AnyscaleCostSummary,
    AnyscaleOperation,
    GenOpsAnyscaleAdapter,
    instrument_anyscale,
)


class TestAnyscaleOperation:
    """Test AnyscaleOperation dataclass."""

    def test_operation_creation(self):
        """Test basic operation creation."""
        operation = AnyscaleOperation(
            operation_id="test-123",
            operation_type="completion",
            model="meta-llama/Llama-2-70b-chat-hf",
            start_time=time.time(),
        )

        assert operation.operation_id == "test-123"
        assert operation.operation_type == "completion"
        assert operation.model == "meta-llama/Llama-2-70b-chat-hf"
        assert operation.governance_attributes == {}

    def test_operation_with_governance_attrs(self):
        """Test operation with governance attributes."""
        operation = AnyscaleOperation(
            operation_id="test-123",
            operation_type="completion",
            model="test-model",
            start_time=time.time(),
            governance_attributes={"team": "test-team", "customer_id": "customer-123"},
        )

        assert operation.governance_attributes["team"] == "test-team"
        assert operation.governance_attributes["customer_id"] == "customer-123"

    def test_operation_duration_calculation(self):
        """Test duration calculation for operation."""
        start = time.time()
        operation = AnyscaleOperation(
            operation_id="test",
            operation_type="completion",
            model="test-model",
            start_time=start,
        )

        # Test ongoing operation
        duration = operation.duration_ms
        assert duration > 0

        # Test completed operation
        operation.end_time = start + 2.0  # 2 seconds
        assert operation.duration_ms == 2000.0

    def test_operation_cost_tracking(self):
        """Test operation cost tracking."""
        operation = AnyscaleOperation(
            operation_id="test",
            operation_type="completion",
            model="meta-llama/Llama-2-70b-chat-hf",
            start_time=time.time(),
            input_tokens=100,
            output_tokens=50,
            total_cost_usd=0.00015,
        )

        assert operation.input_tokens == 100
        assert operation.output_tokens == 50
        assert operation.total_cost_usd == 0.00015


class TestAnyscaleCostSummary:
    """Test AnyscaleCostSummary dataclass."""

    def test_cost_summary_creation(self):
        """Test basic cost summary creation."""
        summary = AnyscaleCostSummary(
            total_cost=0.001, operation_count=5, total_tokens=1000
        )

        assert summary.total_cost == 0.001
        assert summary.operation_count == 5
        assert summary.total_tokens == 1000

    def test_cost_summary_with_breakdown(self):
        """Test cost summary with model breakdown."""
        summary = AnyscaleCostSummary(
            total_cost=0.002,
            operation_count=10,
            total_tokens=2000,
            cost_by_model={
                "meta-llama/Llama-2-70b-chat-hf": 0.0015,
                "meta-llama/Llama-2-7b-chat-hf": 0.0005,
            },
            cost_by_customer={"customer-A": 0.0012, "customer-B": 0.0008},
        )

        assert len(summary.cost_by_model) == 2
        assert len(summary.cost_by_customer) == 2
        assert sum(summary.cost_by_model.values()) == 0.002
        assert sum(summary.cost_by_customer.values()) == 0.002


class TestGenOpsAnyscaleAdapter:
    """Test GenOps Anyscale Adapter."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for Anyscale."""
        with patch("genops.providers.anyscale.adapter.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            yield mock_client

    def test_adapter_initialization_with_defaults(self):
        """Test adapter initialization with default values."""
        adapter = GenOpsAnyscaleAdapter()

        assert adapter.anyscale_base_url == "https://api.endpoints.anyscale.com/v1"
        assert adapter.telemetry_enabled is True
        assert adapter.cost_tracking_enabled is True
        assert adapter.debug is False

    def test_adapter_initialization_with_api_key(self):
        """Test adapter initialization with API key."""
        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="test-key-123")

        assert adapter.anyscale_api_key == "test-key-123"

    def test_adapter_initialization_with_governance_defaults(self):
        """Test adapter initialization with governance defaults."""
        adapter = GenOpsAnyscaleAdapter(
            team="test-team", project="test-project", environment="staging"
        )

        assert adapter.governance_defaults["team"] == "test-team"
        assert adapter.governance_defaults["project"] == "test-project"
        assert adapter.governance_defaults["environment"] == "staging"

    def test_adapter_initialization_custom_base_url(self):
        """Test adapter initialization with custom base URL."""
        adapter = GenOpsAnyscaleAdapter(
            anyscale_base_url="https://custom.anyscale.com/v1"
        )

        assert adapter.anyscale_base_url == "https://custom.anyscale.com/v1"

    def test_adapter_initialization_debug_mode(self):
        """Test adapter initialization with debug mode."""
        adapter = GenOpsAnyscaleAdapter(debug=True)

        assert adapter.debug is True

    @patch.dict("os.environ", {"ANYSCALE_API_KEY": "env-api-key"})
    def test_adapter_uses_env_var_api_key(self):
        """Test adapter uses ANYSCALE_API_KEY from environment."""
        adapter = GenOpsAnyscaleAdapter()

        assert adapter.anyscale_api_key == "env-api-key"

    def test_adapter_governance_context_manager(self):
        """Test adapter governance context manager."""
        adapter = GenOpsAnyscaleAdapter(team="base-team")

        with adapter.governance_context(customer_id="customer-123") as ctx:
            # Context should merge with defaults
            assert "customer_id" in ctx

    @patch("genops.providers.anyscale.adapter.requests")
    def test_completion_create_basic(self, mock_requests):
        """Test basic completion request."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Paris is the capital of France.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="test-key")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        assert response["model"] == "meta-llama/Llama-2-70b-chat-hf"
        assert response["usage"]["total_tokens"] == 25
        mock_requests.post.assert_called_once()

    @patch("genops.providers.anyscale.adapter.requests")
    def test_completion_create_with_governance(self, mock_requests):
        """Test completion with governance attributes."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="test-key", team="test-team")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            customer_id="customer-123",
        )

        assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_completion_create_with_parameters(self, mock_requests):
        """Test completion with various parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="test-key")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
        )

        assert response is not None
        # Verify parameters were passed in API call
        call_kwargs = mock_requests.post.call_args[1]["json"]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 500

    @patch("genops.providers.anyscale.adapter.requests")
    def test_embeddings_create_basic(self, mock_requests):
        """Test basic embeddings request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1] * 1024, "index": 0}],
            "model": "thenlper/gte-large",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="test-key")

        response = adapter.embeddings_create(
            model="thenlper/gte-large", input="Test text to embed"
        )

        assert response["model"] == "thenlper/gte-large"
        assert len(response["data"]) == 1
        assert len(response["data"][0]["embedding"]) == 1024

    @patch("genops.providers.anyscale.adapter.requests")
    def test_embeddings_create_with_list_input(self, mock_requests):
        """Test embeddings with list of strings."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1] * 1024, "index": 0},
                {"object": "embedding", "embedding": [0.2] * 1024, "index": 1},
            ],
            "model": "thenlper/gte-large",
            "usage": {"prompt_tokens": 20, "total_tokens": 20},
        }
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="test-key")

        response = adapter.embeddings_create(
            model="thenlper/gte-large", input=["Text 1", "Text 2"]
        )

        assert len(response["data"]) == 2

    @patch("genops.providers.anyscale.adapter.requests")
    def test_api_error_handling(self, mock_requests):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(anyscale_api_key="invalid-key")

        with pytest.raises(Exception):  # noqa: B017
            adapter.completion_create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("genops.providers.anyscale.adapter.requests")
    def test_cost_tracking_enabled(self, mock_requests):
        """Test cost tracking is performed when enabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        mock_requests.post.return_value = mock_response

        adapter = GenOpsAnyscaleAdapter(
            anyscale_api_key="test-key", cost_tracking_enabled=True
        )

        response = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
        )

        # Cost tracking should have been performed
        assert response["usage"]["total_tokens"] == 150

    def test_telemetry_disabled(self):
        """Test telemetry can be disabled."""
        adapter = GenOpsAnyscaleAdapter(telemetry_enabled=False)

        assert adapter.telemetry_enabled is False

    def test_get_framework_name(self):
        """Test framework name method."""
        adapter = GenOpsAnyscaleAdapter()

        assert adapter.get_framework_name() == "anyscale"

    def test_is_framework_available(self):
        """Test framework availability check."""
        adapter = GenOpsAnyscaleAdapter()

        # Should return True if requests is available
        assert isinstance(adapter.is_framework_available(), bool)


class TestInstrumentAnyscale:
    """Test instrument_anyscale factory function."""

    def test_instrument_anyscale_basic(self):
        """Test basic instrumentation."""
        adapter = instrument_anyscale()

        assert isinstance(adapter, GenOpsAnyscaleAdapter)

    def test_instrument_anyscale_with_governance(self):
        """Test instrumentation with governance defaults."""
        adapter = instrument_anyscale(
            team="test-team", project="test-project", environment="production"
        )

        assert adapter.governance_defaults["team"] == "test-team"
        assert adapter.governance_defaults["project"] == "test-project"
        assert adapter.governance_defaults["environment"] == "production"

    def test_instrument_anyscale_with_api_key(self):
        """Test instrumentation with API key."""
        adapter = instrument_anyscale(anyscale_api_key="custom-key")

        assert adapter.anyscale_api_key == "custom-key"

    def test_instrument_anyscale_returns_adapter(self):
        """Test instrument_anyscale returns adapter instance."""
        adapter = instrument_anyscale()

        assert hasattr(adapter, "completion_create")
        assert hasattr(adapter, "embeddings_create")
        assert hasattr(adapter, "governance_context")
