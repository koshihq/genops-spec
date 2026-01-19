"""
Cross-provider integration tests: Elastic + OpenAI.

Tests cover:
- OpenAI operations tracked to Elasticsearch
- Multi-model cost aggregation
- Streaming and batch operations
- Error handling across both services
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from genops.providers.elastic import instrument_elastic
from genops.providers.elastic.event_exporter import ExportMode


class TestElasticOpenAIIntegration:
    """Test Elastic integration with OpenAI operations."""

    def test_track_openai_completion(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking OpenAI completion to Elasticsearch."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            # Simulate OpenAI completion
            with adapter.track_ai_operation(
                "openai-completion",
                operation_type="llm.completion",
                provider="openai"
            ) as span:
                # Record OpenAI cost
                adapter.record_cost(
                    span=span,
                    cost=0.05,
                    provider="openai",
                    model="gpt-4",
                    tokens_input=100,
                    tokens_output=200
                )

            # Verify attributes
            assert span.attributes.get("genops.cost.provider") == "openai"
            assert span.attributes.get("genops.cost.model") == "gpt-4"

    def test_track_openai_embedding(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking OpenAI embedding to Elasticsearch."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "openai-embedding",
                operation_type="llm.embedding"
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=0.0001,
                    provider="openai",
                    model="text-embedding-ada-002",
                    tokens_input=50,
                    tokens_output=0
                )

            assert span.attributes.get("genops.cost.model") == "text-embedding-ada-002"

    def test_track_multiple_openai_models(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking multiple OpenAI models in same session."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            models = [
                ("gpt-4", 0.05),
                ("gpt-3.5-turbo", 0.002),
                ("text-embedding-ada-002", 0.0001)
            ]

            for model, cost in models:
                with adapter.track_ai_operation(f"openai-{model}") as span:
                    adapter.record_cost(
                        span=span,
                        cost=cost,
                        provider="openai",
                        model=model,
                        tokens_input=100,
                        tokens_output=100
                    )

            adapter.exporter.flush()

    def test_openai_streaming_tracking(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking OpenAI streaming operations."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "openai-streaming",
                operation_type="llm.completion",
                streaming=True
            ) as span:
                # Simulate streaming chunks
                total_cost = 0.0
                for i in range(5):
                    chunk_cost = 0.01
                    total_cost += chunk_cost

                # Record total cost after streaming completes
                adapter.record_cost(
                    span=span,
                    cost=total_cost,
                    provider="openai",
                    model="gpt-4",
                    tokens_input=50,
                    tokens_output=250
                )

            assert span.attributes.get("genops.cost.total") == total_cost

    def test_openai_error_tracking(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking OpenAI errors to Elasticsearch."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with pytest.raises(Exception):
                with adapter.track_ai_operation("openai-error") as span:
                    # Simulate partial cost before error
                    adapter.record_cost(
                        span=span,
                        cost=0.02,
                        provider="openai",
                        model="gpt-4",
                        tokens_input=50,
                        tokens_output=0
                    )
                    raise Exception("OpenAI rate limit exceeded")

            # Error should be tracked in Elasticsearch

    def test_openai_cost_by_customer(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking OpenAI costs by customer."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            customers = ["customer-001", "customer-002", "customer-003"]

            for customer in customers:
                with adapter.track_ai_operation(
                    f"openai-{customer}",
                    customer_id=customer
                ) as span:
                    adapter.record_cost(
                        span=span,
                        cost=0.05,
                        provider="openai",
                        model="gpt-4"
                    )

            adapter.exporter.flush()

    def test_openai_batch_operations(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking batch OpenAI operations."""
        config = sample_elastic_config.copy()
        config["export_mode"] = "batch"
        config["batch_size"] = 5

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**config, auto_validate=False)

            # Simulate batch of OpenAI calls
            for i in range(10):
                with adapter.track_ai_operation(f"openai-batch-{i}") as span:
                    adapter.record_cost(
                        span=span,
                        cost=0.002,
                        provider="openai",
                        model="gpt-3.5-turbo",
                        tokens_input=100,
                        tokens_output=150
                    )

            # Should have flushed at least once
            assert mock_elasticsearch_client.bulk.called

    def test_openai_function_calling_tracking(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking OpenAI function calling."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "openai-function-call",
                operation_type="llm.function_call"
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=0.03,
                    provider="openai",
                    model="gpt-4",
                    tokens_input=150,
                    tokens_output=100
                )

                # Track function metadata
                span.set_attribute("genops.function_name", "get_weather")
                span.set_attribute("genops.function_result", "success")

            assert span.attributes.get("genops.function_name") == "get_weather"
