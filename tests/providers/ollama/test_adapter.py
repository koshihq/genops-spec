"""Tests for Ollama adapter functionality."""

import time
from unittest.mock import Mock, patch

import pytest

from genops.providers.ollama.adapter import (
    GenOpsOllamaAdapter,
    LocalModelMetrics,
    OllamaOperation,
    auto_instrument,
    instrument_ollama,
)


class TestOllamaOperation:
    """Test OllamaOperation dataclass."""

    def test_operation_creation(self):
        """Test basic operation creation."""
        operation = OllamaOperation(
            operation_id="test-123",
            operation_type="generate",
            model="llama3.2:1b",
            start_time=time.time()
        )

        assert operation.operation_id == "test-123"
        assert operation.operation_type == "generate"
        assert operation.model == "llama3.2:1b"
        assert operation.governance_attributes == {}

    def test_operation_duration_calculation(self):
        """Test duration calculation."""
        start = time.time()
        operation = OllamaOperation(
            operation_id="test-123",
            operation_type="generate",
            model="test-model",
            start_time=start
        )

        # Test ongoing operation
        duration = operation.duration_ms
        assert duration > 0

        # Test completed operation
        operation.end_time = start + 2.5  # 2.5 seconds
        assert operation.duration_ms == 2500.0

    def test_governance_attributes_initialization(self):
        """Test governance attributes are properly initialized."""
        operation = OllamaOperation(
            operation_id="test",
            operation_type="chat",
            model="test",
            start_time=time.time(),
            governance_attributes={"team": "test-team", "project": "test"}
        )

        assert operation.governance_attributes["team"] == "test-team"
        assert operation.governance_attributes["project"] == "test"


class TestLocalModelMetrics:
    """Test LocalModelMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics creation with defaults."""
        metrics = LocalModelMetrics(
            model_name="test-model",
            total_operations=10,
            total_inference_time_ms=5000.0
        )

        assert metrics.model_name == "test-model"
        assert metrics.total_operations == 10
        assert metrics.total_inference_time_ms == 5000.0
        assert metrics.success_rate == 100.0
        assert metrics.error_count == 0


class TestGenOpsOllamaAdapter:
    """Test GenOps Ollama Adapter."""

    @pytest.fixture
    def mock_requests(self):
        """Mock requests library."""
        with patch('genops.providers.ollama.adapter.requests') as mock_req:
            yield mock_req

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client."""
        with patch('genops.providers.ollama.adapter.ollama') as mock_ollama:
            mock_client = Mock()
            mock_ollama.Client.return_value = mock_client
            yield mock_client, mock_ollama

    @pytest.fixture
    def adapter(self, mock_requests):
        """Create adapter instance for testing."""
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = {"version": "0.1.0"}

        return GenOpsOllamaAdapter(
            ollama_base_url="http://localhost:11434",
            telemetry_enabled=True,
            cost_tracking_enabled=True,
            debug=True
        )

    def test_adapter_initialization(self, mock_requests):
        """Test adapter initialization."""
        mock_requests.get.return_value.status_code = 200

        adapter = GenOpsOllamaAdapter(
            ollama_base_url="http://localhost:11434",
            team="test-team",
            project="test-project"
        )

        assert adapter.ollama_base_url == "http://localhost:11434"
        assert adapter.telemetry_enabled is True
        assert adapter.cost_tracking_enabled is True
        assert adapter.governance_defaults["team"] == "test-team"
        assert adapter.governance_defaults["project"] == "test-project"

    def test_connection_test_success(self, mock_requests):
        """Test successful connection to Ollama server."""
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = {"version": "0.1.0"}

        # Should not raise exception
        adapter = GenOpsOllamaAdapter()
        assert len(adapter.operations) == 0

    def test_connection_test_failure(self, mock_requests):
        """Test failed connection to Ollama server."""
        mock_requests.get.side_effect = Exception("Connection failed")

        with pytest.raises(ConnectionError):
            GenOpsOllamaAdapter()

    def test_governance_context_manager(self, adapter):
        """Test governance context manager."""
        initial_context = adapter.get_current_governance_context()

        with adapter.governance_context(team="context-team", environment="test"):
            context_inside = adapter.get_current_governance_context()
            assert context_inside["team"] == "context-team"
            assert context_inside["environment"] == "test"

        final_context = adapter.get_current_governance_context()
        assert final_context == initial_context

    def test_list_models_success(self, adapter, mock_requests):
        """Test successful model listing."""
        mock_response = {
            "models": [
                {"name": "llama3.2:1b", "size": 1000000000},
                {"name": "llama3.2:3b", "size": 3000000000}
            ]
        }
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = mock_response

        models = adapter.list_models(team="test-team")

        assert len(models) == 2
        assert models[0]["name"] == "llama3.2:1b"
        assert len(adapter.operations) == 1
        assert adapter.operations[0].operation_type == "list_models"

    def test_list_models_with_client(self, adapter, mock_ollama_client):
        """Test model listing with Ollama client."""
        mock_client, mock_ollama = mock_ollama_client

        # Mock client response
        mock_client.list.return_value = {
            "models": [{"name": "test-model", "size": 1000000}]
        }

        # Re-initialize adapter with mocked client
        with patch('genops.providers.ollama.adapter.HAS_OLLAMA_CLIENT', True):
            adapter.client = mock_client
            models = adapter.list_models()

        assert len(models) == 1
        assert models[0]["name"] == "test-model"
        mock_client.list.assert_called_once()

    def test_generate_with_http_api(self, adapter, mock_requests):
        """Test text generation using HTTP API."""
        # Mock successful generation response
        mock_response = {
            "response": "Hello! I'm an AI assistant.",
            "eval_count": 10,
            "prompt_eval_count": 5
        }
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = mock_response

        response = adapter.generate(
            model="llama3.2:1b",
            prompt="Hello",
            team="test-team"
        )

        assert response["response"] == "Hello! I'm an AI assistant."
        assert len(adapter.operations) == 1

        operation = adapter.operations[0]
        assert operation.operation_type == "generate"
        assert operation.model == "llama3.2:1b"
        assert operation.prompt == "Hello"
        assert operation.output_tokens == 10
        assert operation.input_tokens == 5
        assert operation.governance_attributes["team"] == "test-team"

    def test_generate_with_client(self, adapter, mock_ollama_client):
        """Test text generation with Ollama client."""
        mock_client, mock_ollama = mock_ollama_client

        mock_response = {
            "response": "Generated text",
            "eval_count": 15
        }
        mock_client.generate.return_value = mock_response

        with patch('genops.providers.ollama.adapter.HAS_OLLAMA_CLIENT', True):
            adapter.client = mock_client
            response = adapter.generate(model="test-model", prompt="Test")

        assert response["response"] == "Generated text"
        mock_client.generate.assert_called_once_with(
            model="test-model",
            prompt="Test",
            stream=False
        )

    def test_chat_functionality(self, adapter, mock_requests):
        """Test chat functionality."""
        mock_response = {
            "message": {"content": "Chat response"},
            "eval_count": 12
        }
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = adapter.chat(
            model="llama3.2:1b",
            messages=messages,
            project="test-project"
        )

        assert response["message"]["content"] == "Chat response"
        assert len(adapter.operations) == 1

        operation = adapter.operations[0]
        assert operation.operation_type == "chat"
        assert operation.output_tokens == 12

    def test_cost_calculation(self, adapter):
        """Test infrastructure cost calculation."""
        operation = OllamaOperation(
            operation_id="test",
            operation_type="generate",
            model="llama3.2:3b",
            start_time=time.time() - 2.0,  # 2 seconds ago
            end_time=time.time()
        )

        cost = adapter._calculate_operation_cost(operation)

        assert cost > 0
        assert isinstance(cost, float)
        # Cost should be small for short operation
        assert cost < 0.01

    def test_model_size_cost_adjustment(self, adapter):
        """Test cost adjustment based on model size."""
        # Test with large model
        large_model_op = OllamaOperation(
            operation_id="test1",
            operation_type="generate",
            model="llama3.1:70b",  # Large model
            start_time=time.time() - 1.0,
            end_time=time.time()
        )

        # Test with small model
        small_model_op = OllamaOperation(
            operation_id="test2",
            operation_type="generate",
            model="llama3.2:1b",  # Small model
            start_time=time.time() - 1.0,
            end_time=time.time()
        )

        large_cost = adapter._calculate_operation_cost(large_model_op)
        small_cost = adapter._calculate_operation_cost(small_model_op)

        # Large model should cost more
        assert large_cost > small_cost

    def test_model_metrics_update(self, adapter):
        """Test model metrics updating."""
        model_name = "test-model"

        operation = OllamaOperation(
            operation_id="test",
            operation_type="generate",
            model=model_name,
            start_time=time.time(),
            end_time=time.time(),
            inference_time_ms=1500.0,
            input_tokens=10,
            output_tokens=20,
            infrastructure_cost=0.001
        )

        adapter._update_model_metrics(model_name, operation)

        assert model_name in adapter.model_metrics
        metrics = adapter.model_metrics[model_name]
        assert metrics.total_operations == 1
        assert metrics.avg_inference_latency_ms == 1500.0
        assert metrics.total_input_tokens == 10
        assert metrics.total_output_tokens == 20

    def test_get_model_metrics(self, adapter):
        """Test getting model metrics."""
        # Add some test operations first
        operation1 = OllamaOperation(
            operation_id="test1",
            operation_type="generate",
            model="model1",
            start_time=time.time(),
            inference_time_ms=1000.0
        )
        operation1.end_time = time.time()
        adapter._update_model_metrics("model1", operation1)

        # Test getting specific model metrics
        metrics = adapter.get_model_metrics("model1")
        assert metrics.model_name == "model1"
        assert metrics.total_operations == 1

        # Test getting all model metrics
        all_metrics = adapter.get_model_metrics()
        assert "model1" in all_metrics

    def test_operation_summary(self, adapter):
        """Test operation summary generation."""
        # Add some test operations
        for i in range(3):
            operation = OllamaOperation(
                operation_id=f"test{i}",
                operation_type="generate",
                model=f"model{i}",
                start_time=time.time(),
                infrastructure_cost=0.001,
                inference_time_ms=1000.0,
                response="test response"
            )
            operation.end_time = time.time()
            adapter.operations.append(operation)

        summary = adapter.get_operation_summary()

        assert summary["total_operations"] == 3
        assert summary["total_infrastructure_cost"] == 0.003
        assert len(summary["models_used"]) == 3
        assert summary["success_rate_percent"] == 100.0

    def test_error_handling_in_generate(self, adapter, mock_requests):
        """Test error handling during generation."""
        mock_requests.post.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            adapter.generate(model="test-model", prompt="test")

        # Should still record failed operation
        assert len(adapter.operations) == 1
        assert adapter.operations[0].end_time is not None

    def test_governance_attributes_propagation(self, adapter, mock_requests):
        """Test that governance attributes are properly propagated."""
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"response": "test"}

        response = adapter.generate(
            model="test-model",
            prompt="test",
            team="test-team",
            project="test-project",
            customer_id="customer-123",
            environment="staging"
        )

        operation = adapter.operations[0]
        attrs = operation.governance_attributes

        assert attrs["team"] == "test-team"
        assert attrs["project"] == "test-project"
        assert attrs["customer_id"] == "customer-123"
        assert attrs["environment"] == "staging"


class TestInstrumentationFunctions:
    """Test instrumentation helper functions."""

    def test_instrument_ollama_factory(self):
        """Test instrument_ollama factory function."""
        with patch('genops.providers.ollama.adapter.GenOpsOllamaAdapter') as mock_adapter:
            instrument_ollama(
                ollama_base_url="http://test:11434",
                team="factory-team",
                project="factory-project"
            )

            mock_adapter.assert_called_once_with(
                ollama_base_url="http://test:11434",
                telemetry_enabled=True,
                cost_tracking_enabled=True,
                team="factory-team",
                project="factory-project"
            )

    @patch('genops.providers.ollama.adapter.ollama')
    @patch('genops.providers.ollama.adapter.HAS_OLLAMA_CLIENT', True)
    def test_auto_instrument_patching(self, mock_ollama):
        """Test auto-instrumentation patching."""
        # Mock original methods
        mock_ollama.generate = Mock()
        mock_ollama.chat = Mock()

        # Store originals
        original_generate = mock_ollama.generate
        original_chat = mock_ollama.chat

        # Apply auto-instrumentation
        auto_instrument()

        # Methods should be different now (patched)
        assert mock_ollama.generate != original_generate
        assert mock_ollama.chat != original_chat

    @patch('genops.providers.ollama.adapter.HAS_OLLAMA_CLIENT', False)
    def test_auto_instrument_without_client(self):
        """Test auto-instrumentation when client not available."""
        # Should not raise exception, just warn
        result = auto_instrument()
        assert result is False


class TestTelemetryIntegration:
    """Test OpenTelemetry integration."""

    @patch('genops.providers.ollama.adapter.tracer')
    def test_telemetry_span_creation(self, mock_tracer, mock_requests):
        """Test that telemetry spans are created."""
        mock_requests.get.return_value.status_code = 200
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"response": "test"}

        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        adapter = GenOpsOllamaAdapter(telemetry_enabled=True)
        adapter.generate(model="test-model", prompt="test")

        # Should have created span
        mock_tracer.start_as_current_span.assert_called()
        mock_span.set_attributes.assert_called()

    @patch('genops.providers.ollama.adapter.tracer')
    def test_telemetry_attributes(self, mock_tracer, mock_requests):
        """Test telemetry attributes are set correctly."""
        mock_requests.get.return_value.status_code = 200
        mock_requests.post.return_value.status_code = 200
        mock_requests.post.return_value.json.return_value = {"response": "test"}

        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        adapter = GenOpsOllamaAdapter(telemetry_enabled=True)
        adapter.generate(
            model="test-model",
            prompt="test",
            team="telemetry-team"
        )

        # Check that attributes were set
        calls = mock_span.set_attributes.call_args_list

        # Should have multiple calls to set_attributes
        assert len(calls) > 0

        # Check some expected attributes in the calls
        all_attributes = {}
        for call in calls:
            all_attributes.update(call[0][0])

        assert "genops.operation_type" in all_attributes
        assert "genops.framework" in all_attributes
        assert "genops.model" in all_attributes
        assert all_attributes.get("genops.framework") == "ollama"


if __name__ == "__main__":
    pytest.main([__file__])
