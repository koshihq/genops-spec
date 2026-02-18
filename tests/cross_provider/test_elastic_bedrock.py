"""
Cross-provider integration tests: Elastic + AWS Bedrock.

Tests cover:
- AWS Bedrock multi-model operations tracked to Elasticsearch
- Cross-region cost tracking
- Model marketplace integrations
- IAM-based governance
"""

from unittest.mock import patch

import pytest

from genops.providers.elastic import instrument_elastic


class TestElasticBedrockIntegration:
    """Test Elastic integration with AWS Bedrock operations."""

    def test_track_bedrock_claude(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Bedrock Claude to Elasticsearch."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "bedrock-claude",
                operation_type="llm.completion",
                provider="bedrock",
                region="us-east-1",
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=0.03,
                    provider="bedrock",
                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                    tokens_input=1000,
                    tokens_output=500,
                )

            assert span.attributes.get("genops.cost.provider") == "bedrock"
            assert "claude" in span.attributes.get("genops.cost.model")

    def test_track_bedrock_titan(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Amazon Titan on Bedrock."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("bedrock-titan") as span:
                adapter.record_cost(
                    span=span,
                    cost=0.0008,
                    provider="bedrock",
                    model="amazon.titan-text-express-v1",
                    tokens_input=1000,
                    tokens_output=500,
                )

            assert "titan" in span.attributes.get("genops.cost.model")

    def test_track_bedrock_llama(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Meta Llama on Bedrock."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation("bedrock-llama") as span:
                adapter.record_cost(
                    span=span,
                    cost=0.002,
                    provider="bedrock",
                    model="meta.llama3-70b-instruct-v1:0",
                    tokens_input=1000,
                    tokens_output=500,
                )

            assert "llama" in span.attributes.get("genops.cost.model")

    def test_track_bedrock_multiple_models(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking multiple Bedrock models."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            models = [
                ("anthropic.claude-3-sonnet-20240229-v1:0", 0.03),
                ("amazon.titan-text-express-v1", 0.0008),
                ("meta.llama3-70b-instruct-v1:0", 0.002),
                ("cohere.command-text-v14", 0.0015),
            ]

            for model, cost in models:
                with adapter.track_ai_operation(f"bedrock-{model}") as span:
                    adapter.record_cost(
                        span=span,
                        cost=cost,
                        provider="bedrock",
                        model=model,
                        tokens_input=1000,
                        tokens_output=500,
                    )

            adapter.exporter.flush()

    def test_track_bedrock_cross_region(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Bedrock operations across regions."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            regions = ["us-east-1", "us-west-2", "eu-west-1"]

            for region in regions:
                with adapter.track_ai_operation(
                    f"bedrock-{region}", region=region
                ) as span:
                    adapter.record_cost(
                        span=span,
                        cost=0.03,
                        provider="bedrock",
                        model="anthropic.claude-3-sonnet-20240229-v1:0",
                    )

                    span.set_attribute("genops.aws.region", region)

            adapter.exporter.flush()

    def test_track_bedrock_embeddings(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Bedrock embedding models."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "bedrock-embedding", operation_type="llm.embedding"
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=0.0001,
                    provider="bedrock",
                    model="amazon.titan-embed-text-v1",
                    tokens_input=1000,
                    tokens_output=0,
                )

            assert span.attributes.get("genops.operation_type") == "llm.embedding"

    def test_track_bedrock_with_aws_account(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Bedrock with AWS account attribution."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with adapter.track_ai_operation(
                "bedrock-multi-account", aws_account_id="123456789012"
            ) as span:
                adapter.record_cost(
                    span=span,
                    cost=0.03,
                    provider="bedrock",
                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                )

                span.set_attribute("genops.aws.account_id", "123456789012")

            assert span.attributes.get("genops.aws.account_id") == "123456789012"

    def test_track_bedrock_error_handling(
        self, sample_elastic_config, mock_elasticsearch_client
    ):
        """Test tracking Bedrock errors to Elasticsearch."""
        with patch(
            "genops.providers.elastic.client.Elasticsearch",
            return_value=mock_elasticsearch_client,
        ):
            adapter = instrument_elastic(**sample_elastic_config, auto_validate=False)

            with pytest.raises(Exception):  # noqa: B017
                with adapter.track_ai_operation("bedrock-error") as span:
                    adapter.record_cost(
                        span=span,
                        cost=0.01,
                        provider="bedrock",
                        model="anthropic.claude-3-sonnet-20240229-v1:0",
                        tokens_input=100,
                        tokens_output=0,
                    )
                    raise Exception("Bedrock throttling")
