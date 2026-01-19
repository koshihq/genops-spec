"""
Cross-provider integration tests: Elastic + Anthropic.

Tests cover:
- Anthropic Claude operations tracked to Elasticsearch
- Token-based cost calculation
- Streaming responses
- Multiple Claude models
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from genops.providers.elastic import instrument_elastic


class TestElasticAnthropicIntegration:
    """Test Elastic integration with Anthropic Claude operations."""

    def test_track_claude_completion(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking Claude completion to Elasticsearch."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "claude-completion",
                operation_type="llm.completion",
                provider="anthropic"
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=0.03,
                    provider="anthropic",
                    model="claude-3-sonnet-20240229",
                    tokens_input=1000,
                    tokens_output=500
                )

            assert span.attributes.get("genops.cost.provider") == "anthropic"
            assert span.attributes.get("genops.cost.model") == "claude-3-sonnet-20240229"

    def test_track_claude_3_opus(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking Claude 3 Opus (premium model)."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("claude-opus") as span:
                adapter.record_cost(
                    span=span,
                    cost=0.15,  # Higher cost for Opus
                    provider="anthropic",
                    model="claude-3-opus-20240229",
                    tokens_input=1000,
                    tokens_output=500
                )

            assert span.attributes.get("genops.cost.total") == 0.15

    def test_track_claude_3_haiku(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking Claude 3 Haiku (economy model)."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("claude-haiku") as span:
                adapter.record_cost(
                    span=span,
                    cost=0.00025,  # Lower cost for Haiku
                    provider="anthropic",
                    model="claude-3-haiku-20240307",
                    tokens_input=1000,
                    tokens_output=500
                )

            assert span.attributes.get("genops.cost.model") == "claude-3-haiku-20240307"

    def test_track_multiple_claude_models(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking multiple Claude models in same session."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            models = [
                ("claude-3-opus-20240229", 0.15),
                ("claude-3-sonnet-20240229", 0.03),
                ("claude-3-haiku-20240307", 0.00025)
            ]

            for model, cost in models:
                with adapter.track_ai_operation(f"claude-{model}") as span:
                    adapter.record_cost(
                        span=span,
                        cost=cost,
                        provider="anthropic",
                        model=model,
                        tokens_input=1000,
                        tokens_output=500
                    )

            adapter.exporter.flush()

    def test_claude_streaming_tracking(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking Claude streaming responses."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "claude-streaming",
                operation_type="llm.completion",
                streaming=True
            ) as span:
                # Simulate streaming
                total_tokens_out = 0
                for i in range(10):
                    total_tokens_out += 50

                # Record final cost
                adapter.record_cost(
                    span=span,
                    cost=0.03,
                    provider="anthropic",
                    model="claude-3-sonnet-20240229",
                    tokens_input=100,
                    tokens_output=total_tokens_out
                )

            assert span.attributes.get("genops.cost.tokens_output") == total_tokens_out

    def test_claude_long_context_tracking(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking Claude with long context (200k tokens)."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "claude-long-context",
                context_length=200000
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=6.0,  # Significant cost for 200k tokens
                    provider="anthropic",
                    model="claude-3-opus-20240229",
                    tokens_input=180000,
                    tokens_output=20000
                )

            assert span.attributes.get("genops.cost.total") == 6.0

    def test_claude_error_tracking(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking Claude errors to Elasticsearch."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with pytest.raises(Exception):
                with adapter.track_ai_operation("claude-error") as span:
                    adapter.record_cost(
                        span=span,
                        cost=0.01,
                        provider="anthropic",
                        model="claude-3-sonnet-20240229",
                        tokens_input=100,
                        tokens_output=0
                    )
                    raise Exception("Anthropic rate limit")

    def test_claude_cost_split_calculation(self, sample_elastic_config, mock_elasticsearch_client):
        """Test tracking split input/output costs for Claude."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("claude-cost-split") as span:
                adapter.record_cost(
                    span=span,
                    cost=0.03,
                    provider="anthropic",
                    model="claude-3-sonnet-20240229",
                    tokens_input=1000,
                    tokens_output=500,
                    cost_input=0.02,  # Input cost component
                    cost_output=0.01  # Output cost component
                )

            assert span.attributes.get("genops.cost.input") == 0.02
            assert span.attributes.get("genops.cost.output") == 0.01
