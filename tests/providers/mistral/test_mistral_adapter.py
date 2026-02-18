#!/usr/bin/env python3
"""
Test suite for GenOps Mistral AI adapter

This test suite provides comprehensive coverage for the Mistral adapter,
including core functionality, cost tracking, European AI features,
error handling, and GDPR compliance scenarios.
"""

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

# Import the classes to test
from genops.providers.mistral import (
    GenOpsMistralAdapter,
    MistralModel,
    MistralOperation,
    MistralResponse,
    MistralUsage,
    instrument_mistral,
    mistral_workflow_context,
)


# Mock Mistral API response classes
@dataclass
class MockMistralUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150


@dataclass
class MockMistralMessage:
    content: str = "Mock response content"
    role: str = "assistant"


@dataclass
class MockMistralChoice:
    message: MockMistralMessage = None
    finish_reason: str = "stop"

    def __post_init__(self):
        if self.message is None:
            self.message = MockMistralMessage()


@dataclass
class MockMistralChatResponse:
    id: str = "chatcmpl-123"
    choices: list[MockMistralChoice] = None
    usage: MockMistralUsage = None
    model: str = "mistral-small-latest"

    def __post_init__(self):
        if self.choices is None:
            self.choices = [MockMistralChoice()]
        if self.usage is None:
            self.usage = MockMistralUsage()


@dataclass
class MockMistralEmbeddingData:
    embedding: list[float] = None
    index: int = 0

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = [0.1] * 1536  # Mock 1536-dimensional embedding


@dataclass
class MockMistralEmbeddingResponse:
    id: str = "embed-123"
    data: list[MockMistralEmbeddingData] = None
    model: str = "mistral-embed"

    def __post_init__(self):
        if self.data is None:
            self.data = [MockMistralEmbeddingData()]


class TestGenOpsMistralAdapter:
    """Test suite for GenOps Mistral adapter core functionality."""

    @pytest.fixture
    def mock_mistral_client(self):
        """Create a mock Mistral client."""
        with patch("genops.providers.mistral.Mistral") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock chat completion
            mock_client.chat.complete.return_value = MockMistralChatResponse()

            # Mock embeddings
            mock_client.embeddings.create.return_value = MockMistralEmbeddingResponse()

            yield mock_client

    @pytest.fixture
    def adapter(self, mock_mistral_client):
        """Create a test adapter instance."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-api-key"}):
            return GenOpsMistralAdapter(
                default_team="test-team", default_project="test-project"
            )

    def test_adapter_initialization(self, adapter):
        """Test adapter initialization with default values."""
        assert adapter.api_key == "test-api-key"
        assert adapter.cost_tracking_enabled is True
        assert adapter.default_team == "test-team"
        assert adapter.default_project == "test-project"
        assert adapter._total_cost == 0.0
        assert adapter._operation_count == 0
        assert len(adapter._session_id) > 0

    def test_initialization_without_api_key(self):
        """Test adapter initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Mistral API key is required"):
                GenOpsMistralAdapter()

    def test_initialization_with_custom_config(self, mock_mistral_client):
        """Test adapter initialization with custom configuration."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            adapter = GenOpsMistralAdapter(
                cost_tracking_enabled=False,
                budget_limit=100.0,
                cost_alert_threshold=0.9,
                default_environment="production",
                timeout=90.0,
            )

            assert adapter.cost_tracking_enabled is False
            assert adapter.budget_limit == 100.0
            assert adapter.cost_alert_threshold == 0.9
            assert adapter.default_environment == "production"
            assert adapter.timeout == 90.0

    def test_chat_operation_success(self, adapter, mock_mistral_client):
        """Test successful chat completion with cost tracking."""
        mock_response = MockMistralChatResponse()
        mock_mistral_client.chat.complete.return_value = mock_response

        response = adapter.chat(
            message="Test message",
            model="mistral-small-latest",
            team="ai-team",
            project="test-project",
        )

        # Verify API call
        mock_mistral_client.chat.complete.assert_called_once()
        call_args = mock_mistral_client.chat.complete.call_args
        assert call_args[1]["model"] == "mistral-small-latest"
        assert len(call_args[1]["messages"]) >= 1
        assert call_args[1]["messages"][-1]["content"] == "Test message"

        # Verify response structure
        assert isinstance(response, MistralResponse)
        assert response.success is True
        assert response.content == "Mock response content"
        assert response.model == "mistral-small-latest"
        assert response.operation == MistralOperation.CHAT.value

        # Verify usage tracking
        assert isinstance(response.usage, MistralUsage)
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150
        assert response.usage.model == "mistral-small-latest"

        # Verify session stats updated
        assert adapter._operation_count == 1
        assert adapter._total_cost > 0  # Should have some cost

    def test_chat_with_system_prompt(self, adapter, mock_mistral_client):
        """Test chat with system prompt."""
        adapter.chat(
            message="User message",
            system_prompt="You are a helpful assistant",
            model="mistral-medium-latest",
        )

        # Verify system prompt included
        call_args = mock_mistral_client.chat.complete.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User message"

    def test_chat_with_parameters(self, adapter, mock_mistral_client):
        """Test chat with additional parameters."""
        adapter.chat(
            message="Test",
            model="mistral-large-2407",
            temperature=0.3,
            max_tokens=200,
            stream=False,
        )

        call_args = mock_mistral_client.chat.complete.call_args
        assert call_args[1]["model"] == "mistral-large-2407"
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["max_tokens"] == 200
        assert call_args[1]["stream"] is False

    def test_embed_operation_success(self, adapter, mock_mistral_client):
        """Test successful embedding operation."""
        mock_response = MockMistralEmbeddingResponse()
        mock_mistral_client.embeddings.create.return_value = mock_response

        response = adapter.embed(
            texts=["Test text for embedding"], model="mistral-embed", team="data-team"
        )

        # Verify API call
        mock_mistral_client.embeddings.create.assert_called_once()
        call_args = mock_mistral_client.embeddings.create.call_args
        assert call_args[1]["model"] == "mistral-embed"
        assert call_args[1]["inputs"] == ["Test text for embedding"]

        # Verify response
        assert response.success is True
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 1536
        assert response.embedding_dimension == 1536
        assert response.operation == MistralOperation.EMBED.value

        # Verify session stats
        assert adapter._operation_count == 1

    def test_embed_multiple_texts(self, adapter, mock_mistral_client):
        """Test embedding multiple texts."""
        texts = ["Text one", "Text two", "Text three"]
        mock_response = MockMistralEmbeddingResponse()
        mock_response.data = [MockMistralEmbeddingData() for _ in texts]
        mock_mistral_client.embeddings.create.return_value = mock_response

        response = adapter.embed(texts=texts)

        # Verify API call
        call_args = mock_mistral_client.embeddings.create.call_args
        assert call_args[1]["inputs"] == texts

        # Verify response
        assert len(response.embeddings) == 3

    def test_embed_single_string(self, adapter, mock_mistral_client):
        """Test embedding single string (converted to list)."""
        adapter.embed(texts="Single text string")

        call_args = mock_mistral_client.embeddings.create.call_args
        assert call_args[1]["inputs"] == ["Single text string"]

    def test_generate_operation(self, adapter, mock_mistral_client):
        """Test generate operation (alias for chat)."""
        response = adapter.generate(
            prompt="Generate some text", model="mistral-small-latest"
        )

        # Should call chat internally
        mock_mistral_client.chat.complete.assert_called_once()
        assert response.success is True
        assert response.operation == MistralOperation.CHAT.value

    def test_error_handling_chat(self, adapter, mock_mistral_client):
        """Test error handling in chat operation."""
        # Mock an API error
        mock_mistral_client.chat.complete.side_effect = Exception("API Error")

        response = adapter.chat(message="Test", model="mistral-small-latest")

        assert response.success is False
        assert "API Error" in response.error_message
        assert response.usage.model == "mistral-small-latest"
        assert response.usage.total_cost == 0.0

    def test_error_handling_embed(self, adapter, mock_mistral_client):
        """Test error handling in embedding operation."""
        mock_mistral_client.embeddings.create.side_effect = Exception("Embedding Error")

        response = adapter.embed(texts=["Test text"])

        assert response.success is False
        assert "Embedding Error" in response.error_message
        assert len(response.embeddings) == 0

    def test_usage_summary(self, adapter, mock_mistral_client):
        """Test getting usage summary."""
        # Perform some operations
        adapter.chat(message="Test 1", model="mistral-small-latest")
        adapter.chat(message="Test 2", model="mistral-medium-latest")

        summary = adapter.get_usage_summary()

        assert summary["total_operations"] == 2
        assert summary["total_cost"] > 0
        assert summary["average_cost_per_operation"] > 0
        assert summary["cost_tracking_enabled"] is True
        assert "session_id" in summary

    def test_session_stats_reset(self, adapter, mock_mistral_client):
        """Test resetting session statistics."""
        # Perform operation
        adapter.chat(message="Test", model="mistral-small-latest")

        assert adapter._operation_count == 1
        assert adapter._total_cost > 0

        # Reset stats
        old_session_id = adapter._session_id
        adapter.reset_session_stats()

        assert adapter._operation_count == 0
        assert adapter._total_cost == 0.0
        assert adapter._session_id != old_session_id

    def test_cost_tracking_disabled(self, mock_mistral_client):
        """Test adapter with cost tracking disabled."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            adapter = GenOpsMistralAdapter(cost_tracking_enabled=False)

            response = adapter.chat(message="Test", model="mistral-small-latest")

            # Cost should be zero when tracking disabled
            assert response.usage.total_cost == 0.0
            assert response.usage.input_cost == 0.0
            assert response.usage.output_cost == 0.0

    def test_budget_limit_alert(self, adapter, mock_mistral_client):
        """Test budget limit alert functionality."""
        # Set a very low budget limit
        adapter.budget_limit = 0.001
        adapter.cost_alert_threshold = 0.5

        # Mock high-cost operation
        with patch.object(
            adapter, "_calculate_cost", return_value=(0.0005, 0.0005, 0.001)
        ):
            with patch("genops.providers.mistral.logger") as mock_logger:
                adapter.chat(message="Expensive test", model="mistral-large-2407")

                # Should log cost alert
                mock_logger.warning.assert_called()

    def test_governance_attributes(self, adapter, mock_mistral_client):
        """Test governance attributes are properly extracted."""
        response = adapter.chat(
            message="Test governance",
            model="mistral-small-latest",
            team="governance-team",
            project="compliance-project",
            customer_id="enterprise-123",
            environment="production",
        )

        # Governance attributes should be captured
        # This is mainly tested through integration with cost tracking
        assert response.success is True

    def test_european_ai_features(self, adapter, mock_mistral_client):
        """Test European AI specific features and models."""
        # Test with European-focused model selection
        response = adapter.chat(
            message="GDPR compliance question",
            model="mistral-medium-latest",  # European AI model
            team="compliance-eu",
            environment="eu-production",
        )

        assert response.success is True
        assert response.model == "mistral-medium-latest"

        # Test cost competitiveness (should have reasonable costs)
        assert response.usage.total_cost >= 0  # Should have cost tracking


class TestMistralInstrumentation:
    """Test suite for Mistral instrumentation functions."""

    @pytest.fixture
    def mock_mistral_client(self):
        """Create a mock Mistral client."""
        with patch("genops.providers.mistral.Mistral") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.complete.return_value = MockMistralChatResponse()
            yield mock_client

    def test_instrument_mistral(self, mock_mistral_client):
        """Test instrument_mistral function."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            adapter = instrument_mistral(
                team="test-team",
                project="test-project",
                customer_id="test-customer",
                environment="test",
            )

            assert isinstance(adapter, GenOpsMistralAdapter)
            assert adapter.default_team == "test-team"
            assert adapter.default_project == "test-project"
            assert adapter.default_customer_id == "test-customer"
            assert adapter.default_environment == "test"

    def test_mistral_workflow_context(self, mock_mistral_client):
        """Test Mistral workflow context manager."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            with mistral_workflow_context(
                "test-workflow", team="workflow-team", project="workflow-project"
            ) as (ctx, workflow_id):
                assert isinstance(ctx, GenOpsMistralAdapter)
                assert workflow_id.startswith("test-workflow-")
                assert len(workflow_id.split("-")) >= 2  # Should have UUID suffix

                # Test using the context
                response = ctx.chat(
                    message="Workflow test", model="mistral-small-latest"
                )
                assert response.success is True


class TestMistralModels:
    """Test suite for Mistral model enumeration and validation."""

    def test_mistral_model_enum(self):
        """Test Mistral model enumeration."""
        # Test core models exist
        assert MistralModel.MISTRAL_TINY.value == "mistral-tiny-2312"
        assert MistralModel.MISTRAL_SMALL.value == "mistral-small-latest"
        assert MistralModel.MISTRAL_MEDIUM.value == "mistral-medium-latest"
        assert MistralModel.MISTRAL_LARGE.value == "mistral-large-latest"
        assert MistralModel.MISTRAL_LARGE_2407.value == "mistral-large-2407"

        # Test specialized models
        assert MistralModel.MISTRAL_EMBED.value == "mistral-embed"
        assert MistralModel.CODESTRAL.value == "codestral-2405"
        assert MistralModel.MISTRAL_NEMO.value == "mistral-nemo-2407"

        # Test Mixtral models
        assert MistralModel.MIXTRAL_8X7B.value == "mixtral-8x7b-32768"
        assert MistralModel.MIXTRAL_8X22B.value == "mixtral-8x22b-32768"

    def test_mistral_operation_enum(self):
        """Test Mistral operation enumeration."""
        assert MistralOperation.CHAT.value == "chat"
        assert MistralOperation.EMBED.value == "embed"
        assert MistralOperation.COMPLETION.value == "completion"


class TestMistralDataClasses:
    """Test suite for Mistral data classes."""

    def test_mistral_usage_creation(self):
        """Test MistralUsage dataclass creation."""
        usage = MistralUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=0.001,
            model="mistral-small-latest",
            operation="chat",
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.total_cost == 0.001
        assert usage.model == "mistral-small-latest"
        assert usage.operation == "chat"

    def test_mistral_response_creation(self):
        """Test MistralResponse dataclass creation."""
        usage = MistralUsage()
        response = MistralResponse(
            content="Test response",
            success=True,
            model="mistral-medium-latest",
            operation="chat",
            usage=usage,
        )

        assert response.content == "Test response"
        assert response.success is True
        assert response.model == "mistral-medium-latest"
        assert response.operation == "chat"
        assert response.usage == usage
        assert response.embeddings == []  # Default empty list

    def test_mistral_response_with_embeddings(self):
        """Test MistralResponse with embeddings."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        response = MistralResponse(
            embeddings=embeddings, embedding_dimension=3, operation="embed"
        )

        assert response.embeddings == embeddings
        assert response.embedding_dimension == 3
        assert response.operation == "embed"


class TestMistralErrorScenarios:
    """Test suite for Mistral error scenarios and edge cases."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked client for error testing."""
        with patch("genops.providers.mistral.Mistral") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
                return GenOpsMistralAdapter()

    def test_chat_api_timeout(self, adapter):
        """Test handling of API timeout errors."""
        adapter.client.chat.complete.side_effect = TimeoutError("Request timed out")

        response = adapter.chat(message="Test timeout", model="mistral-small-latest")

        assert response.success is False
        assert "timed out" in response.error_message.lower()

    def test_chat_authentication_error(self, adapter):
        """Test handling of authentication errors."""
        adapter.client.chat.complete.side_effect = Exception("Unauthorized")

        response = adapter.chat(message="Test auth", model="mistral-small-latest")

        assert response.success is False
        assert "unauthorized" in response.error_message.lower()

    def test_embed_model_not_found(self, adapter):
        """Test handling of model not found errors."""
        adapter.client.embeddings.create.side_effect = Exception("Model not found")

        response = adapter.embed(texts=["Test"], model="invalid-model")

        assert response.success is False
        assert "model not found" in response.error_message.lower()

    def test_empty_message_handling(self, adapter):
        """Test handling of empty or invalid messages."""
        adapter.client.chat.complete.return_value = MockMistralChatResponse()

        # Empty message
        response = adapter.chat(message="", model="mistral-small-latest")
        assert response.success is True  # Should still work

        # Whitespace only
        response = adapter.chat(message="   ", model="mistral-small-latest")
        assert response.success is True

    def test_empty_texts_for_embedding(self, adapter):
        """Test handling of empty texts for embedding."""
        adapter.client.embeddings.create.return_value = MockMistralEmbeddingResponse()

        adapter.embed(texts=[])

        # Should handle gracefully
        call_args = adapter.client.embeddings.create.call_args
        assert call_args[1]["inputs"] == []


class TestMistralIntegration:
    """Integration tests for Mistral adapter with real-like scenarios."""

    @pytest.fixture
    def full_mock_setup(self):
        """Set up comprehensive mocks for integration testing."""
        with patch("genops.providers.mistral.Mistral") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock successful responses
            mock_client.chat.complete.return_value = MockMistralChatResponse()
            mock_client.embeddings.create.return_value = MockMistralEmbeddingResponse()

            with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-integration-key"}):
                yield mock_client

    def test_end_to_end_workflow(self, full_mock_setup):
        """Test complete end-to-end workflow."""
        # Initialize adapter
        adapter = GenOpsMistralAdapter(
            default_team="integration-team", default_project="e2e-test"
        )

        # Perform multiple operations
        operations = [
            ("chat", "mistral-tiny-2312", "Simple question"),
            ("chat", "mistral-small-latest", "Medium complexity task"),
            ("embed", "mistral-embed", ["Document 1", "Document 2"]),
        ]

        responses = []
        for op_type, model, content in operations:
            if op_type == "chat":
                response = adapter.chat(message=content, model=model)
            else:  # embed
                response = adapter.embed(texts=content, model=model)

            responses.append(response)
            assert response.success is True

        # Verify session tracking
        summary = adapter.get_usage_summary()
        assert summary["total_operations"] == 3
        assert summary["total_cost"] > 0

        # All responses should be successful
        assert all(r.success for r in responses)

    def test_european_ai_compliance_workflow(self, full_mock_setup):
        """Test European AI compliance workflow."""
        adapter = GenOpsMistralAdapter(
            default_team="eu-compliance",
            default_project="gdpr-workflow",
            default_environment="eu-production",
        )

        # GDPR compliance analysis
        gdpr_response = adapter.chat(
            message="Analyze this data for GDPR compliance",
            system_prompt="You are a GDPR compliance expert",
            model="mistral-medium-latest",
            customer_id="eu-enterprise",
            temperature=0.1,  # Low temperature for consistent compliance
        )

        assert gdpr_response.success is True
        assert gdpr_response.model == "mistral-medium-latest"

        # European AI cost tracking
        summary = adapter.get_usage_summary()
        assert summary["cost_tracking_enabled"] is True
        assert summary["total_cost"] >= 0

    def test_multi_model_cost_optimization(self, full_mock_setup):
        """Test cost optimization across multiple models."""
        adapter = GenOpsMistralAdapter(default_team="optimization-team")

        # Test different models for cost comparison
        models_to_test = [
            "mistral-tiny-2312",  # Ultra-low cost
            "mistral-small-latest",  # Cost-effective
            "mistral-medium-latest",  # Balanced performance
        ]

        simple_prompt = "What is 2+2?"
        costs_by_model = {}

        for model in models_to_test:
            response = adapter.chat(
                message=simple_prompt,
                model=model,
                max_tokens=10,  # Limit tokens for fair comparison
            )

            assert response.success is True
            costs_by_model[model] = response.usage.total_cost

        # Should have cost data for all models
        assert len(costs_by_model) == len(models_to_test)
        assert all(cost >= 0 for cost in costs_by_model.values())


# Performance and edge case tests
class TestMistralPerformance:
    """Test suite for performance and scaling scenarios."""

    @pytest.fixture
    def performance_adapter(self):
        """Create adapter optimized for performance testing."""
        with patch("genops.providers.mistral.Mistral") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.complete.return_value = MockMistralChatResponse()

            with patch.dict("os.environ", {"MISTRAL_API_KEY": "perf-test-key"}):
                return GenOpsMistralAdapter(
                    timeout=30.0, max_retries=2, cost_tracking_enabled=True
                )

    def test_multiple_concurrent_operations(self, performance_adapter):
        """Test handling multiple operations in sequence."""
        # Simulate multiple rapid operations
        results = []

        for i in range(10):
            response = performance_adapter.chat(
                message=f"Operation {i}", model="mistral-small-latest"
            )
            results.append(response.success)

        # All operations should succeed
        assert all(results)

        # Session should track all operations
        summary = performance_adapter.get_usage_summary()
        assert summary["total_operations"] == 10

    def test_large_text_handling(self, performance_adapter):
        """Test handling of large text inputs."""
        large_text = "This is a test. " * 1000  # ~15KB of text

        response = performance_adapter.chat(
            message=large_text, model="mistral-small-latest"
        )

        assert response.success is True
        assert response.usage.input_tokens > 0

    def test_memory_efficiency(self, performance_adapter):
        """Test memory efficiency with session reset."""
        # Perform operations
        for i in range(5):
            performance_adapter.chat(
                message=f"Memory test {i}", model="mistral-small-latest"
            )

        # Reset and verify cleanup
        performance_adapter.reset_session_stats()

        summary = performance_adapter.get_usage_summary()
        assert summary["total_operations"] == 0
        assert summary["total_cost"] == 0.0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
