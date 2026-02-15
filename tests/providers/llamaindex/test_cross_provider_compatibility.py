"""
Cross-provider compatibility tests for GenOps LlamaIndex integration.

Tests compatibility and interoperability across different AI providers,
models, and deployment scenarios with unified GenOps tracking.
"""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from genops.providers.llamaindex.adapter import GenOpsLlamaIndexAdapter

# Import the modules under test
from genops.providers.llamaindex.cost_aggregator import (
    LlamaIndexCostAggregator,
    create_llamaindex_cost_context,
)


@dataclass
class ProviderConfig:
    """Configuration for a specific AI provider."""

    name: str
    embedding_model: str
    llm_model: str
    embedding_cost_per_1k: float
    llm_input_cost_per_1k: float
    llm_output_cost_per_1k: float
    typical_context_size: int


class TestProviderCostCompatibility:
    """Test cost calculation compatibility across providers."""

    @pytest.fixture
    def provider_configs(self):
        """Provider configurations for testing."""
        return [
            ProviderConfig(
                name="openai",
                embedding_model="text-embedding-ada-002",
                llm_model="gpt-4",
                embedding_cost_per_1k=0.0001,
                llm_input_cost_per_1k=0.03,
                llm_output_cost_per_1k=0.06,
                typical_context_size=8192,
            ),
            ProviderConfig(
                name="anthropic",
                embedding_model="voyage-large-2",  # Hypothetical
                llm_model="claude-3-sonnet-20240229",
                embedding_cost_per_1k=0.0001,
                llm_input_cost_per_1k=0.003,
                llm_output_cost_per_1k=0.015,
                typical_context_size=4096,
            ),
            ProviderConfig(
                name="google",
                embedding_model="textembedding-gecko",
                llm_model="gemini-pro",
                embedding_cost_per_1k=0.0001,
                llm_input_cost_per_1k=0.0005,
                llm_output_cost_per_1k=0.0015,
                typical_context_size=32768,
            ),
            ProviderConfig(
                name="cohere",
                embedding_model="embed-english-v3.0",
                llm_model="command-r",
                embedding_cost_per_1k=0.0001,
                llm_input_cost_per_1k=0.0005,
                llm_output_cost_per_1k=0.0015,
                typical_context_size=4096,
            ),
        ]

    def test_embedding_cost_calculation_across_providers(self, provider_configs):
        """Test embedding cost calculation consistency across providers."""
        aggregator = LlamaIndexCostAggregator("cross_provider_test")

        token_counts = [100, 1000, 5000, 10000]

        for config in provider_configs:
            for tokens in token_counts:
                cost = aggregator.calculate_embedding_cost(
                    config.name, config.embedding_model, tokens
                )

                expected_cost = (tokens / 1000) * config.embedding_cost_per_1k
                assert abs(cost - expected_cost) < 1e-10, (
                    f"Cost mismatch for {config.name} with {tokens} tokens"
                )

    def test_completion_cost_calculation_across_providers(self, provider_configs):
        """Test completion cost calculation consistency across providers."""
        aggregator = LlamaIndexCostAggregator("cross_provider_test")

        test_cases = [
            (500, 300),  # Small completion
            (2000, 1000),  # Medium completion
            (5000, 2000),  # Large completion
        ]

        for config in provider_configs:
            for input_tokens, output_tokens in test_cases:
                cost = aggregator.calculate_completion_cost(
                    config.name, config.llm_model, input_tokens, output_tokens
                )

                expected_cost = (input_tokens / 1000) * config.llm_input_cost_per_1k + (
                    output_tokens / 1000
                ) * config.llm_output_cost_per_1k

                assert abs(cost - expected_cost) < 1e-10, (
                    f"Cost mismatch for {config.name}"
                )

    def test_cost_aggregation_across_multiple_providers(self, provider_configs):
        """Test cost aggregation when using multiple providers simultaneously."""
        aggregator = LlamaIndexCostAggregator("multi_provider_test")

        # Use different providers for different operations
        operations = [
            ("openai", "embedding", 1000, 0.0001),
            ("anthropic", "completion", (1500, 800), 0.0165),  # 1.5K input, 0.8K output
            ("google", "embedding", 2000, 0.0002),
            ("cohere", "completion", (1000, 500), 0.001),
        ]

        total_expected_cost = 0.0

        for provider, operation_type, tokens, expected_cost in operations:
            if operation_type == "embedding":
                aggregator.add_embedding_cost(
                    provider, f"{provider}_embed_model", tokens, expected_cost
                )
            else:  # completion
                input_tokens, output_tokens = tokens
                aggregator.add_synthesis_cost(
                    provider,
                    f"{provider}_llm_model",
                    input_tokens,
                    output_tokens,
                    expected_cost,
                )

            total_expected_cost += expected_cost

        # Verify total cost aggregation
        assert abs(aggregator.total_cost - total_expected_cost) < 1e-10

        # Verify per-provider cost tracking
        assert len(aggregator.cost_by_provider) == 4
        assert "openai" in aggregator.cost_by_provider
        assert "anthropic" in aggregator.cost_by_provider
        assert "google" in aggregator.cost_by_provider
        assert "cohere" in aggregator.cost_by_provider


class TestProviderModelCompatibility:
    """Test model compatibility across different providers."""

    def test_unknown_model_fallback(self):
        """Test fallback behavior for unknown models."""
        aggregator = LlamaIndexCostAggregator("unknown_model_test")

        # Test unknown models for known providers
        known_providers = ["openai", "anthropic", "google"]

        for provider in known_providers:
            # Should use provider's default pricing for unknown model
            cost = aggregator.calculate_completion_cost(
                provider, "unknown-model-xyz", 1000, 500
            )
            assert cost > 0.0, f"Should calculate fallback cost for {provider}"

    def test_unknown_provider_fallback(self):
        """Test fallback behavior for unknown providers."""
        aggregator = LlamaIndexCostAggregator("unknown_provider_test")

        # Should use generic fallback pricing
        embedding_cost = aggregator.calculate_embedding_cost(
            "unknown_provider", "unknown_model", 1000
        )
        completion_cost = aggregator.calculate_completion_cost(
            "unknown_provider", "unknown_model", 1000, 500
        )

        assert embedding_cost > 0.0, "Should have fallback embedding cost"
        assert completion_cost > 0.0, "Should have fallback completion cost"

    def test_model_version_handling(self):
        """Test handling of different model versions."""
        aggregator = LlamaIndexCostAggregator("version_test")

        # Test different versions of the same base model
        model_versions = [
            ("openai", "gpt-4"),
            ("openai", "gpt-4-0314"),
            ("openai", "gpt-4-32k"),
            ("anthropic", "claude-3-sonnet-20240229"),
            ("anthropic", "claude-3-haiku-20240307"),
        ]

        for provider, model in model_versions:
            cost = aggregator.calculate_completion_cost(provider, model, 1000, 500)
            assert cost > 0.0, f"Should handle model version {provider}/{model}"


class TestCrossProviderRAGWorkflows:
    """Test RAG workflows using different provider combinations."""

    @pytest.fixture
    def mock_engines(self):
        """Create mock engines for different providers."""
        from llama_index.core.base.response.schema import Response
        from llama_index.core.query_engine.base import BaseQueryEngine

        engines = {}

        providers = ["openai", "anthropic", "google", "cohere"]

        for provider in providers:
            engine = Mock(spec=BaseQueryEngine)
            engine.query.return_value = Response(f"Response from {provider} model")
            engines[provider] = engine

        return engines

    def test_mixed_provider_rag_pipeline(self, mock_engines):
        """Test RAG pipeline using different providers for different components."""
        adapter = GenOpsLlamaIndexAdapter()

        with create_llamaindex_cost_context("mixed_provider_rag") as cost_context:
            # Simulate mixed provider usage
            provider_combinations = [
                ("openai", "embedding"),
                ("google", "retrieval"),
                ("anthropic", "synthesis"),
            ]

            for provider, operation_type in provider_combinations:
                if operation_type == "synthesis" and provider in mock_engines:
                    # Use different provider for synthesis
                    adapter.track_query(
                        mock_engines[provider],
                        f"Query using {provider}",
                        provider=provider,
                    )

                    # Record costs for this provider
                    if provider == "openai":
                        cost_context.add_synthesis_cost(
                            "openai", "gpt-4", 1000, 600, 0.048
                        )
                    elif provider == "anthropic":
                        cost_context.add_synthesis_cost(
                            "anthropic", "claude-3", 1000, 600, 0.012
                        )
                    elif provider == "google":
                        cost_context.add_synthesis_cost(
                            "google", "gemini-pro", 1000, 600, 0.0014
                        )

                elif operation_type == "embedding":
                    # Add embedding costs
                    cost_context.add_embedding_cost(
                        provider, f"{provider}_embed", 2000, 0.0002
                    )

                elif operation_type == "retrieval":
                    # Add retrieval costs
                    cost_context.add_retrieval_cost(0.001)

        # Verify mixed provider cost tracking
        summary = cost_context.get_current_summary()
        assert summary.total_cost > 0.0
        assert (
            len(summary.cost_breakdown.cost_by_provider) >= 2
        )  # Multiple providers used

    def test_provider_failover_scenarios(self, mock_engines):
        """Test failover between providers."""
        adapter = GenOpsLlamaIndexAdapter()

        # Configure primary provider to fail
        primary_engine = mock_engines["openai"]
        primary_engine.query.side_effect = Exception("OpenAI rate limit exceeded")

        # Fallback providers work normally
        fallback_engines = [mock_engines["anthropic"], mock_engines["google"]]

        with create_llamaindex_cost_context("failover_test") as cost_context:
            successful_queries = 0

            # Try primary provider first
            try:
                adapter.track_query(primary_engine, "Primary query")
            except Exception:
                # Failover to other providers
                for fallback_engine in fallback_engines:
                    try:
                        adapter.track_query(fallback_engine, "Fallback query")
                        cost_context.add_synthesis_cost(
                            "fallback", "fallback_model", 500, 300, 0.002
                        )
                        successful_queries += 1
                        break
                    except Exception:
                        continue

            assert successful_queries > 0, "Should have successful fallback"

    def test_cost_optimization_across_providers(self):
        """Test cost optimization strategies across providers."""

        with create_llamaindex_cost_context("cost_optimization") as cost_context:
            # Simulate different cost scenarios
            scenarios = [
                # (provider, model, input_tokens, output_tokens, description)
                ("openai", "gpt-3.5-turbo", 1000, 500, "cost_effective"),
                ("anthropic", "claude-3-haiku", 1000, 500, "balanced"),
                ("google", "gemini-pro", 1000, 500, "high_context"),
                ("openai", "gpt-4", 1000, 500, "high_quality"),
            ]

            costs = {}

            for provider, model, input_tokens, output_tokens, scenario in scenarios:
                cost = cost_context.calculate_completion_cost(
                    provider, model, input_tokens, output_tokens
                )
                costs[scenario] = cost

                # Record the operation
                cost_context.add_synthesis_cost(
                    provider, model, input_tokens, output_tokens, cost
                )

        # Verify cost differences reflect provider characteristics
        assert costs["cost_effective"] < costs["high_quality"]  # GPT-3.5 < GPT-4
        assert costs["balanced"] < costs["high_quality"]  # Claude-3-haiku < GPT-4

        # Verify optimization suggestions consider multiple providers
        summary = cost_context.get_current_summary()
        assert (
            len(summary.cost_breakdown.cost_by_provider) == 3
        )  # OpenAI, Anthropic, Google


class TestProviderSpecificFeatures:
    """Test provider-specific features and compatibility."""

    def test_context_window_handling(self):
        """Test handling of different context window sizes."""
        aggregator = LlamaIndexCostAggregator("context_window_test")

        # Test with different context window requirements
        context_scenarios = [
            ("openai", "gpt-4", 32000, "large_context"),  # 32K context
            ("anthropic", "claude-3", 100000, "very_large_context"),  # 100K context
            ("google", "gemini-pro", 1000000, "massive_context"),  # 1M context
        ]

        for provider, model, context_tokens, _scenario in context_scenarios:
            # Calculate cost for large context
            cost = aggregator.calculate_completion_cost(
                provider, model, context_tokens, 1000
            )
            assert cost > 0.0, f"Should handle large context for {provider}"

            # Record operation
            aggregator.add_synthesis_cost(provider, model, context_tokens, 1000, cost)

        # Verify all providers handled large contexts
        assert aggregator.operation_count == 3
        assert len(aggregator.cost_by_provider) == 3

    def test_multimodal_capability_compatibility(self):
        """Test multimodal capabilities across providers."""

        # Providers with multimodal capabilities
        multimodal_providers = [
            ("openai", "gpt-4-vision-preview", True),
            ("anthropic", "claude-3-sonnet", True),
            ("google", "gemini-pro-vision", True),
            ("cohere", "command-r", False),  # Text-only
        ]

        with create_llamaindex_cost_context("multimodal_test") as cost_context:
            for provider, model, has_vision in multimodal_providers:
                # Text operation (all providers support)
                text_cost = cost_context.calculate_completion_cost(
                    provider, model, 1000, 500
                )
                cost_context.add_synthesis_cost(provider, model, 1000, 500, text_cost)

                # Vision operation (only some providers support)
                if has_vision:
                    # Add extra cost for vision processing
                    vision_cost = 0.01  # Hypothetical vision cost
                    cost_context.add_synthesis_cost(
                        provider, f"{model}_vision", 0, 0, vision_cost
                    )

        summary = cost_context.get_current_summary()
        assert summary.total_cost > 0.0
        assert len(summary.cost_breakdown.cost_by_provider) == 4


class TestProviderInteroperability:
    """Test interoperability between GenOps and provider-specific features."""

    def test_provider_specific_metadata_handling(self):
        """Test handling of provider-specific metadata."""
        GenOpsLlamaIndexAdapter()

        with create_llamaindex_cost_context("metadata_test") as cost_context:
            # Add operations with provider-specific metadata
            metadata_scenarios = [
                {
                    "provider": "openai",
                    "metadata": {
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "frequency_penalty": 0.1,
                    },
                },
                {
                    "provider": "anthropic",
                    "metadata": {
                        "model": "claude-3-sonnet",
                        "temperature": 0.5,
                        "max_tokens": 2000,
                        "stop_sequences": ["Human:", "Assistant:"],
                    },
                },
                {
                    "provider": "google",
                    "metadata": {
                        "model": "gemini-pro",
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "top_k": 40,
                    },
                },
            ]

            for scenario in metadata_scenarios:
                provider = scenario["provider"]
                metadata = scenario["metadata"]

                # Record operation with provider-specific metadata
                cost_context.add_synthesis_cost(
                    provider,
                    metadata["model"],
                    1000,
                    500,
                    0.005,
                    **metadata,  # Pass metadata as additional attributes
                )

        # Verify all providers and metadata were handled
        summary = cost_context.get_current_summary()
        assert len(summary.cost_breakdown.cost_by_provider) == 3

    def test_governance_attributes_across_providers(self):
        """Test governance attributes work consistently across providers."""

        governance_attrs = {
            "team": "cross-provider-team",
            "project": "compatibility-test",
            "customer_id": "multi-provider-customer",
            "environment": "testing",
        }

        providers = ["openai", "anthropic", "google", "cohere"]

        with create_llamaindex_cost_context(
            "governance_test", **governance_attrs
        ) as cost_context:
            for provider in providers:
                # Each provider should respect governance attributes
                cost_context.add_synthesis_cost(
                    provider, f"{provider}_model", 1000, 500, 0.005, **governance_attrs
                )

        # Verify governance attributes were applied consistently
        summary = cost_context.get_current_summary()
        assert len(summary.cost_breakdown.cost_by_provider) == len(providers)

        # All operations should have governance context
        assert cost_context.governance_attrs == governance_attrs


class TestProviderCompatibilityEdgeCases:
    """Test edge cases in cross-provider compatibility."""

    def test_mixed_token_counting_standards(self):
        """Test compatibility with different token counting standards."""
        aggregator = LlamaIndexCostAggregator("token_counting_test")

        # Different providers may count tokens differently

        token_counts_by_provider = {
            "openai": 12,  # GPT tokenizer
            "anthropic": 11,  # Claude tokenizer
            "google": 13,  # Gemini tokenizer
            "cohere": 10,  # Command tokenizer
        }

        for provider, token_count in token_counts_by_provider.items():
            embedding_cost = aggregator.calculate_embedding_cost(
                provider, f"{provider}_embed", token_count
            )
            completion_cost = aggregator.calculate_completion_cost(
                provider, f"{provider}_llm", token_count, token_count // 2
            )

            assert embedding_cost > 0.0, (
                f"Embedding cost should be calculated for {provider}"
            )
            assert completion_cost > 0.0, (
                f"Completion cost should be calculated for {provider}"
            )

    def test_regional_pricing_variations(self):
        """Test handling of regional pricing variations."""
        aggregator = LlamaIndexCostAggregator("regional_pricing_test")

        # Simulate regional pricing differences
        regional_scenarios = [
            ("openai", "us-east-1", 1.0),  # Base pricing
            ("openai", "eu-west-1", 1.15),  # 15% markup for EU
            ("anthropic", "us-west-2", 1.0),  # Base pricing
            ("google", "asia-pacific", 1.1),  # 10% markup for APAC
        ]

        for provider, region, multiplier in regional_scenarios:
            base_cost = aggregator.calculate_completion_cost(
                provider, f"{provider}_model", 1000, 500
            )

            # In a full implementation, might apply regional multiplier
            regional_cost = base_cost * multiplier

            aggregator.add_synthesis_cost(
                provider,
                f"{provider}_model_{region}",
                1000,
                500,
                regional_cost,
                region=region,
            )

        # Verify all regional variations were tracked
        assert aggregator.operation_count == 4
        assert len(aggregator.cost_by_provider) >= 3  # At least 3 different providers


if __name__ == "__main__":
    pytest.main([__file__])
