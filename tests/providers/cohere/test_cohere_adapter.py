"""Tests for GenOps Cohere adapter."""

import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

# Test imports
from genops.providers.cohere import (
    CohereOperation,
    CohereResponse,
    CohereUsageMetrics,
    GenOpsCohereAdapter,
    instrument_cohere,
)


@dataclass
class MockCohereResponse:
    """Mock Cohere API response."""
    message: Optional[Any] = None
    embeddings: Optional[list] = None
    results: Optional[list] = None
    usage: Optional[Any] = None

    def __post_init__(self):
        if self.message is None:
            self.message = Mock()
            self.message.content = [Mock(text="Mock response text")]

        if self.usage is None:
            self.usage = Mock()
            self.usage.input_tokens = 10
            self.usage.output_tokens = 20


@dataclass
class MockRerankResult:
    """Mock rerank result."""
    index: int
    relevance_score: float
    document: dict


class TestGenOpsCohereAdapter:
    """Test suite for GenOpsCohereAdapter."""

    @pytest.fixture
    def mock_cohere_client(self):
        """Mock Cohere client for testing."""
        with patch('genops.providers.cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Setup default mock responses
            mock_client.chat.return_value = MockCohereResponse()
            mock_client.embed.return_value = MockCohereResponse(
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            )
            mock_client.rerank.return_value = MockCohereResponse(
                results=[
                    MockRerankResult(0, 0.9, {"text": "First doc"}),
                    MockRerankResult(1, 0.7, {"text": "Second doc"})
                ]
            )

            yield mock_client

    @pytest.fixture
    def adapter(self, mock_cohere_client):
        """Create adapter instance for testing."""
        return GenOpsCohereAdapter(
            api_key="test-api-key",
            cost_tracking_enabled=True,
            default_team="test-team",
            default_project="test-project"
        )

    def test_adapter_initialization(self):
        """Test adapter initialization with various configurations."""
        # Basic initialization
        adapter = GenOpsCohereAdapter(api_key="test-key")
        assert adapter.api_key == "test-key"
        assert adapter.cost_tracking_enabled is True
        assert adapter.timeout == 60.0

        # Advanced configuration
        adapter = GenOpsCohereAdapter(
            api_key="test-key",
            timeout=30.0,
            cost_tracking_enabled=False,
            budget_limit=100.0,
            default_team="ml-team"
        )
        assert adapter.timeout == 30.0
        assert adapter.cost_tracking_enabled is False
        assert adapter.budget_limit == 100.0
        assert adapter.default_team == "ml-team"

    def test_api_key_from_environment(self):
        """Test API key loading from environment variable."""
        with patch.dict(os.environ, {'CO_API_KEY': 'env-api-key'}):
            adapter = GenOpsCohereAdapter()
            assert adapter.api_key == "env-api-key"

    def test_chat_operation(self, adapter, mock_cohere_client):
        """Test chat operation with cost tracking."""
        response = adapter.chat(
            message="Test message",
            model="command-r-plus-08-2024",
            team="test-team",
            project="test-project"
        )

        # Verify API call
        mock_cohere_client.chat.assert_called_once()
        call_args = mock_cohere_client.chat.call_args
        assert call_args[1]["model"] == "command-r-plus-08-2024"
        assert call_args[1]["messages"][0]["content"] == "Test message"

        # Verify response
        assert isinstance(response, CohereResponse)
        assert response.success is True
        assert response.content == "Mock response text"
        assert isinstance(response.usage, CohereUsageMetrics)
        assert response.usage.operation_type == CohereOperation.CHAT

    def test_chat_with_parameters(self, adapter, mock_cohere_client):
        """Test chat operation with various parameters."""
        response = adapter.chat(
            message="Test message",
            model="command-light",
            temperature=0.7,
            max_tokens=100,
            conversation_id="test-conv-123"
        )

        # Verify parameters passed correctly
        call_args = mock_cohere_client.chat.call_args[1]
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100
        assert call_args["conversation_id"] == "test-conv-123"

        assert response.success is True

    def test_generate_operation(self, adapter, mock_cohere_client):
        """Test text generation operation."""
        # Mock generate method if it exists
        mock_cohere_client.generate = Mock(return_value=MockCohereResponse())

        response = adapter.generate(
            prompt="Generate text about AI",
            model="command-r-08-2024",
            temperature=0.5,
            max_tokens=200
        )

        assert response.success is True
        assert isinstance(response.usage, CohereUsageMetrics)
        assert response.usage.operation_type == CohereOperation.GENERATE

    def test_embed_operation(self, adapter, mock_cohere_client):
        """Test embedding operation with cost tracking."""
        texts = ["Text 1", "Text 2", "Text 3"]

        response = adapter.embed(
            texts=texts,
            model="embed-english-v4.0",
            input_type="search_document",
            team="embed-team"
        )

        # Verify API call
        mock_cohere_client.embed.assert_called_once()
        call_args = mock_cohere_client.embed.call_args[1]
        assert call_args["texts"] == texts
        assert call_args["model"] == "embed-english-v4.0"
        assert call_args["input_type"] == "search_document"

        # Verify response
        assert response.success is True
        assert response.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.usage.operation_type == CohereOperation.EMBED
        assert response.usage.embedding_units == len(texts)

    def test_embed_single_string(self, adapter, mock_cohere_client):
        """Test embedding with single string input."""
        response = adapter.embed(
            texts="Single text",
            model="embed-english-v4.0"
        )

        # Should convert single string to list
        call_args = mock_cohere_client.embed.call_args[1]
        assert call_args["texts"] == ["Single text"]
        assert response.success is True

    def test_rerank_operation(self, adapter, mock_cohere_client):
        """Test document reranking operation."""
        query = "machine learning"
        documents = ["Doc 1 about ML", "Doc 2 about AI", "Doc 3 about data"]

        response = adapter.rerank(
            query=query,
            documents=documents,
            model="rerank-english-v3.0",
            top_n=2
        )

        # Verify API call
        mock_cohere_client.rerank.assert_called_once()
        call_args = mock_cohere_client.rerank.call_args[1]
        assert call_args["query"] == query
        assert call_args["documents"] == documents
        assert call_args["model"] == "rerank-english-v3.0"
        assert call_args["top_n"] == 2

        # Verify response
        assert response.success is True
        assert len(response.rankings) == 2
        assert response.rankings[0]["relevance_score"] == 0.9
        assert response.usage.operation_type == CohereOperation.RERANK
        assert response.usage.search_units == 1

    def test_cost_calculation(self, adapter):
        """Test cost calculation functionality."""
        # Mock the pricing calculator
        with patch('genops.providers.cohere.CohereCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.calculate_cost.return_value = (0.001, 0.002, 0.0)  # input, output, operation costs

            input_cost, output_cost, op_cost = adapter._calculate_cost(
                model="command-r-plus-08-2024",
                operation=CohereOperation.CHAT,
                input_tokens=100,
                output_tokens=150
            )

            assert input_cost == 0.001
            assert output_cost == 0.002
            assert op_cost == 0.0

    def test_governance_attributes(self, adapter, mock_cohere_client):
        """Test governance attribute handling."""
        governance_attrs = {
            "team": "ml-team",
            "project": "recommendation-engine",
            "customer_id": "enterprise-123",
            "environment": "production",
            "cost_center": "ai-infrastructure"
        }

        response = adapter.chat(
            message="Test message",
            model="command-light",
            **governance_attrs
        )

        # Verify governance attributes are captured
        assert response.usage.team == "ml-team"
        assert response.usage.project == "recommendation-engine"
        assert response.usage.customer_id == "enterprise-123"
        assert response.usage.environment == "production"

    def test_usage_statistics(self, adapter, mock_cohere_client):
        """Test usage statistics tracking."""
        # Perform multiple operations
        adapter.chat(message="Test 1", model="command-light")
        adapter.chat(message="Test 2", model="command-light")
        adapter.embed(texts=["Text 1"], model="embed-english-v4.0")

        # Check statistics
        summary = adapter.get_usage_summary()
        assert summary["total_operations"] == 3
        assert summary["total_cost"] > 0
        assert summary["cost_tracking_enabled"] is True

    def test_budget_controls(self, mock_cohere_client):
        """Test budget limit enforcement."""
        adapter = GenOpsCohereAdapter(
            api_key="test-key",
            budget_limit=0.001,  # Very low limit
            cost_alert_threshold=0.5
        )

        # Mock high cost calculation
        with patch.object(adapter, '_calculate_cost', return_value=(0.002, 0.0, 0.0)):
            # Should not block operation (just warning)
            response = adapter.chat(message="Test", model="command-light")
            assert response.success is True

    def test_error_handling_api_failure(self, adapter, mock_cohere_client):
        """Test error handling when API calls fail."""
        # Mock API failure
        mock_cohere_client.chat.side_effect = Exception("API Error")

        response = adapter.chat(message="Test", model="command-light")

        assert response.success is False
        assert "API Error" in response.error_message

    def test_error_handling_invalid_model(self, adapter, mock_cohere_client):
        """Test error handling for invalid model names."""
        mock_cohere_client.chat.side_effect = Exception("Model not found")

        response = adapter.chat(message="Test", model="invalid-model")

        assert response.success is False
        assert "Model not found" in response.error_message

    def test_operation_id_generation(self, adapter, mock_cohere_client):
        """Test unique operation ID generation."""
        response1 = adapter.chat(message="Test 1", model="command-light")
        response2 = adapter.chat(message="Test 2", model="command-light")

        assert response1.operation_id != response2.operation_id
        assert response1.operation_id.startswith("cohere-")
        assert response2.operation_id.startswith("cohere-")

    def test_telemetry_integration(self, adapter):
        """Test OpenTelemetry integration."""
        with patch('genops.providers.cohere.trace') as mock_trace:
            mock_tracer = Mock()
            mock_span = Mock()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            mock_trace.get_tracer.return_value = mock_tracer

            # Re-initialize adapter to pick up mocked tracer
            adapter.tracer = mock_tracer

            with patch.object(adapter, 'client') as mock_client:
                mock_client.chat.return_value = MockCohereResponse()
                adapter.chat(message="Test", model="command-light")

            # Verify span creation
            mock_tracer.start_as_current_span.assert_called()

    def test_streaming_support(self, adapter, mock_cohere_client):
        """Test streaming response handling."""
        # Mock streaming response
        mock_stream = [
            Mock(content="Hello"),
            Mock(content=" world"),
            Mock(content="!")
        ]
        mock_cohere_client.chat.return_value = mock_stream

        response = adapter.chat(
            message="Test streaming",
            model="command-r-08-2024",
            stream=True
        )

        # Should handle streaming (implementation dependent)
        assert response is not None

    def test_reset_usage_stats(self, adapter, mock_cohere_client):
        """Test usage statistics reset."""
        # Generate some usage
        adapter.chat(message="Test", model="command-light")

        summary_before = adapter.get_usage_summary()
        assert summary_before["total_operations"] > 0

        # Reset stats
        adapter.reset_usage_stats()

        summary_after = adapter.get_usage_summary()
        assert summary_after["total_operations"] == 0
        assert summary_after["total_cost"] == 0.0

    def test_context_manager_span_creation(self, adapter):
        """Test context manager for span creation."""
        with patch.object(adapter, 'tracer') as mock_tracer:
            mock_span = Mock()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

            with adapter._create_span("test_operation", team="test-team"):
                pass

            mock_tracer.start_as_current_span.assert_called_with("genops.cohere.test_operation")
            mock_span.set_attribute.assert_called()

    def test_performance_metrics(self, adapter, mock_cohere_client):
        """Test performance metrics calculation."""
        # Add delay to simulate API latency
        def delayed_response(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return MockCohereResponse()

        mock_cohere_client.chat.side_effect = delayed_response

        response = adapter.chat(message="Test", model="command-light")

        assert response.usage.latency_ms >= 100  # Should capture latency
        assert response.usage.latency_ms < 200   # But not too much overhead


class TestInstrumentCohere:
    """Test suite for instrument_cohere factory function."""

    def test_basic_instrumentation(self):
        """Test basic adapter creation."""
        with patch('genops.providers.cohere.GenOpsCohereAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter

            adapter = instrument_cohere(
                api_key="test-key",
                team="test-team"
            )

            mock_adapter_class.assert_called_once()
            call_kwargs = mock_adapter_class.call_args[1]
            assert "api_key" in call_kwargs
            assert "team" in call_kwargs

    def test_governance_defaults(self):
        """Test governance defaults passing."""
        with patch('genops.providers.cohere.GenOpsCohereAdapter') as mock_adapter_class:
            instrument_cohere(
                team="ml-team",
                project="ai-project",
                environment="production",
                customer_id="enterprise-123"
            )

            call_kwargs = mock_adapter_class.call_args[1]
            assert call_kwargs["team"] == "ml-team"
            assert call_kwargs["project"] == "ai-project"
            assert call_kwargs["environment"] == "production"
            assert call_kwargs["customer_id"] == "enterprise-123"


class TestCohereUsageMetrics:
    """Test suite for CohereUsageMetrics."""

    def test_metrics_initialization(self):
        """Test metrics object initialization."""
        metrics = CohereUsageMetrics(
            operation_id="test-op-123",
            operation_type=CohereOperation.CHAT,
            model="command-r-plus-08-2024",
            timestamp=time.time(),
            input_tokens=100,
            output_tokens=150,
            input_cost=0.001,
            output_cost=0.002
        )

        assert metrics.operation_id == "test-op-123"
        assert metrics.operation_type == CohereOperation.CHAT
        assert metrics.total_tokens == 250  # Auto-calculated
        assert metrics.total_cost == 0.003  # Auto-calculated

    def test_tokens_per_second_calculation(self):
        """Test tokens per second calculation."""
        metrics = CohereUsageMetrics(
            operation_id="test-op",
            operation_type=CohereOperation.CHAT,
            model="command-light",
            timestamp=time.time(),
            output_tokens=100,
            latency_ms=2000  # 2 seconds
        )

        assert metrics.tokens_per_second == 50.0  # 100 tokens / 2 seconds


class TestCohereResponse:
    """Test suite for CohereResponse."""

    def test_response_initialization(self):
        """Test response object initialization."""
        usage = CohereUsageMetrics(
            operation_id="test-op",
            operation_type=CohereOperation.CHAT,
            model="command-light",
            timestamp=time.time()
        )

        response = CohereResponse(
            content="Test response",
            usage=usage,
            model="command-light",
            operation_id="test-op-123",
            success=True
        )

        assert response.content == "Test response"
        assert response.success is True
        assert response.operation_id == "test-op-123"
        assert response.usage == usage

    def test_error_response(self):
        """Test error response creation."""
        response = CohereResponse(
            operation_id="failed-op",
            success=False,
            error_message="API call failed"
        )

        assert response.success is False
        assert response.error_message == "API call failed"
        assert response.content == ""  # Default empty content


class TestAutoInstrumentation:
    """Test suite for auto-instrumentation functionality."""

    @patch('genops.providers.cohere.HAS_COHERE', True)
    @patch('genops.providers.cohere.ClientV2')
    def test_auto_instrument_success(self, mock_client_class):
        """Test successful auto-instrumentation."""
        from genops.providers.cohere import auto_instrument

        result = auto_instrument()
        assert result is True

    @patch('genops.providers.cohere.HAS_COHERE', False)
    def test_auto_instrument_no_client(self):
        """Test auto-instrumentation when client not available."""
        from genops.providers.cohere import auto_instrument

        result = auto_instrument()
        assert result is False

    @patch('genops.providers.cohere.HAS_COHERE', True)
    @patch('genops.providers.cohere.ClientV2')
    def test_auto_instrument_error_handling(self, mock_client_class):
        """Test auto-instrumentation error handling."""
        from genops.providers.cohere import auto_instrument

        # Mock initialization error
        mock_client_class.side_effect = Exception("Initialization failed")

        result = auto_instrument()
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
