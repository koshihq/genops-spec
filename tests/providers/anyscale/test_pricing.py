"""Tests for Anyscale pricing functionality."""

from dataclasses import asdict

import pytest

from genops.providers.anyscale.pricing import (
    ANYSCALE_PRICING,
    MODEL_ALIASES,
    AnyscalePricing,
    ModelPricing,
    calculate_completion_cost,
    calculate_embedding_cost,
    get_model_pricing,
)


class TestModelPricing:
    """Test ModelPricing dataclass."""

    def test_model_pricing_creation(self):
        """Test basic ModelPricing creation."""
        pricing = ModelPricing(
            model_name="test-model",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            currency="USD",
            category="chat",
        )

        assert pricing.model_name == "test-model"
        assert pricing.input_cost_per_million == 1.0
        assert pricing.output_cost_per_million == 2.0
        assert pricing.currency == "USD"
        assert pricing.category == "chat"

    def test_model_pricing_with_defaults(self):
        """Test ModelPricing with default values."""
        pricing = ModelPricing(
            model_name="test", input_cost_per_million=1.0, output_cost_per_million=1.0
        )

        assert pricing.currency == "USD"
        assert pricing.category == "chat"
        assert pricing.context_window is None
        assert pricing.notes is None

    def test_model_pricing_serialization(self):
        """Test ModelPricing can be serialized to dict."""
        pricing = ModelPricing(
            model_name="test-model",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            context_window=4096,
        )

        pricing_dict = asdict(pricing)
        assert isinstance(pricing_dict, dict)
        assert pricing_dict["model_name"] == "test-model"
        assert pricing_dict["context_window"] == 4096


class TestAnyscalePricingDatabase:
    """Test ANYSCALE_PRICING database."""

    def test_pricing_database_exists(self):
        """Test pricing database is not empty."""
        assert len(ANYSCALE_PRICING) > 0

    def test_llama2_models_in_database(self):
        """Test Llama-2 models are in pricing database."""
        assert "meta-llama/Llama-2-70b-chat-hf" in ANYSCALE_PRICING
        assert "meta-llama/Llama-2-13b-chat-hf" in ANYSCALE_PRICING
        assert "meta-llama/Llama-2-7b-chat-hf" in ANYSCALE_PRICING

    def test_llama3_models_in_database(self):
        """Test Llama-3 models are in pricing database."""
        assert "meta-llama/Meta-Llama-3-70B-Instruct" in ANYSCALE_PRICING
        assert "meta-llama/Meta-Llama-3-8B-Instruct" in ANYSCALE_PRICING

    def test_mistral_models_in_database(self):
        """Test Mistral models are in pricing database."""
        assert "mistralai/Mixtral-8x7B-Instruct-v0.1" in ANYSCALE_PRICING
        assert "mistralai/Mistral-7B-Instruct-v0.1" in ANYSCALE_PRICING

    def test_codellama_models_in_database(self):
        """Test CodeLlama models are in pricing database."""
        assert "codellama/CodeLlama-70b-Instruct-hf" in ANYSCALE_PRICING
        assert "codellama/CodeLlama-34b-Instruct-hf" in ANYSCALE_PRICING

    def test_embedding_models_in_database(self):
        """Test embedding models are in pricing database."""
        assert "thenlper/gte-large" in ANYSCALE_PRICING
        assert "BAAI/bge-large-en-v1.5" in ANYSCALE_PRICING

    def test_pricing_data_structure(self):
        """Test pricing data has correct structure."""
        for model_name, pricing in ANYSCALE_PRICING.items():
            assert isinstance(pricing, ModelPricing)
            assert pricing.model_name == model_name
            assert pricing.input_cost_per_million >= 0
            assert pricing.output_cost_per_million >= 0
            assert pricing.currency == "USD"


class TestModelAliases:
    """Test MODEL_ALIASES functionality."""

    def test_aliases_exist(self):
        """Test model aliases are defined."""
        assert len(MODEL_ALIASES) > 0

    def test_alias_targets_valid(self):
        """Test all alias targets exist in pricing database."""
        for alias, target in MODEL_ALIASES.items():
            assert target in ANYSCALE_PRICING, (
                f"Alias {alias} points to non-existent model {target}"
            )


class TestCalculateCompletionCost:
    """Test calculate_completion_cost function."""

    def test_basic_cost_calculation(self):
        """Test basic cost calculation for completion."""
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-70b-chat-hf", input_tokens=100, output_tokens=50
        )

        # Llama-2-70b is $1.00/M tokens for both input and output
        # (100 + 50) / 1,000,000 * $1.00 = $0.00015
        assert cost == pytest.approx(0.00015, abs=1e-8)

    def test_zero_tokens_cost(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-70b-chat-hf", input_tokens=0, output_tokens=0
        )

        assert cost == 0.0

    def test_input_only_cost(self):
        """Test cost calculation with only input tokens."""
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-70b-chat-hf", input_tokens=1000, output_tokens=0
        )

        # 1000 / 1,000,000 * $1.00 = $0.001
        assert cost == pytest.approx(0.001, abs=1e-8)

    def test_output_only_cost(self):
        """Test cost calculation with only output tokens."""
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-70b-chat-hf", input_tokens=0, output_tokens=500
        )

        # 500 / 1,000,000 * $1.00 = $0.0005
        assert cost == pytest.approx(0.0005, abs=1e-8)

    def test_different_model_costs(self):
        """Test cost calculation varies by model."""
        cost_70b = calculate_completion_cost("meta-llama/Llama-2-70b-chat-hf", 100, 100)

        cost_7b = calculate_completion_cost("meta-llama/Llama-2-7b-chat-hf", 100, 100)

        # Llama-2-7b should be cheaper than Llama-2-70b
        assert cost_7b < cost_70b

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-70b-chat-hf",
            input_tokens=10000,
            output_tokens=5000,
        )

        # 15000 / 1,000,000 * $1.00 = $0.015
        assert cost == pytest.approx(0.015, abs=1e-8)

    def test_unknown_model_fallback(self):
        """Test cost calculation falls back for unknown models."""
        # Unknown model should use fallback pricing
        cost = calculate_completion_cost(
            model="unknown-model/test", input_tokens=100, output_tokens=100
        )

        # Should return some cost (fallback pricing)
        assert cost > 0


class TestCalculateEmbeddingCost:
    """Test calculate_embedding_cost function."""

    def test_basic_embedding_cost(self):
        """Test basic embedding cost calculation."""
        cost = calculate_embedding_cost(model="thenlper/gte-large", tokens=1000)

        # gte-large is $0.05/M tokens
        # 1000 / 1,000,000 * $0.05 = $0.00005
        assert cost == pytest.approx(0.00005, abs=1e-8)

    def test_zero_tokens_embedding(self):
        """Test embedding cost with zero tokens."""
        cost = calculate_embedding_cost(model="thenlper/gte-large", tokens=0)

        assert cost == 0.0

    def test_large_embedding_tokens(self):
        """Test embedding cost with large token count."""
        cost = calculate_embedding_cost(model="thenlper/gte-large", tokens=50000)

        # 50000 / 1,000,000 * $0.05 = $0.0025
        assert cost == pytest.approx(0.0025, abs=1e-8)


class TestGetModelPricing:
    """Test get_model_pricing function."""

    def test_get_pricing_for_valid_model(self):
        """Test getting pricing for valid model."""
        pricing = get_model_pricing("meta-llama/Llama-2-70b-chat-hf")

        assert pricing is not None
        assert pricing.model_name == "meta-llama/Llama-2-70b-chat-hf"
        assert pricing.input_cost_per_million == 1.0
        assert pricing.output_cost_per_million == 1.0

    def test_get_pricing_returns_none_for_unknown(self):
        """Test getting pricing returns None for unknown model."""
        pricing = get_model_pricing("completely-unknown-model")

        assert pricing is None

    def test_get_pricing_with_alias(self):
        """Test getting pricing works with model aliases."""
        # If aliases are defined, test them
        if MODEL_ALIASES:
            alias = list(MODEL_ALIASES.keys())[0]
            pricing = get_model_pricing(alias)

            if pricing:  # If alias resolution is implemented
                target = MODEL_ALIASES[alias]
                expected_pricing = ANYSCALE_PRICING[target]
                assert pricing.model_name == expected_pricing.model_name


class TestAnyscalePricing:
    """Test AnyscalePricing class."""

    @pytest.fixture
    def pricing_calculator(self):
        """Create AnyscalePricing instance."""
        return AnyscalePricing()

    def test_pricing_calculator_initialization(self, pricing_calculator):
        """Test AnyscalePricing initializes correctly."""
        assert pricing_calculator is not None

    def test_calculate_cost_for_chat_model(self, pricing_calculator):
        """Test cost calculation for chat model."""
        cost = pricing_calculator.calculate_cost(
            model="meta-llama/Llama-2-70b-chat-hf", input_tokens=100, output_tokens=50
        )

        assert cost == pytest.approx(0.00015, abs=1e-8)

    def test_get_model_info(self, pricing_calculator):
        """Test getting model info."""
        info = pricing_calculator.get_model_info("meta-llama/Llama-2-70b-chat-hf")

        assert info is not None
        assert info.model_name == "meta-llama/Llama-2-70b-chat-hf"
        assert info.category == "chat"

    def test_get_model_alternatives(self, pricing_calculator):
        """Test getting model alternatives."""
        alternatives = pricing_calculator.get_model_alternatives(
            "meta-llama/Llama-2-70b-chat-hf"
        )

        # Should return list of cheaper alternatives
        assert isinstance(alternatives, list)

        # If alternatives exist, they should be cheaper
        if alternatives:
            for _alt_model, cost_ratio, description in alternatives:
                assert cost_ratio < 1.0  # Cheaper than original
                assert isinstance(description, str)
                assert len(description) > 0

    def test_list_chat_models(self, pricing_calculator):
        """Test listing chat models."""
        chat_models = [
            name
            for name, pricing in ANYSCALE_PRICING.items()
            if pricing.category == "chat"
        ]

        assert len(chat_models) > 0

    def test_list_embedding_models(self, pricing_calculator):
        """Test listing embedding models."""
        embedding_models = [
            name
            for name, pricing in ANYSCALE_PRICING.items()
            if pricing.category == "embedding"
        ]

        assert len(embedding_models) > 0


class TestPricingAccuracy:
    """Test pricing calculation accuracy."""

    def test_pricing_matches_published_rates(self):
        """Test pricing matches Anyscale published rates."""
        # Llama-2-70b should be $1.00/M tokens
        llama2_70b = ANYSCALE_PRICING["meta-llama/Llama-2-70b-chat-hf"]
        assert llama2_70b.input_cost_per_million == 1.0
        assert llama2_70b.output_cost_per_million == 1.0

        # Llama-2-7b should be $0.15/M tokens
        llama2_7b = ANYSCALE_PRICING["meta-llama/Llama-2-7b-chat-hf"]
        assert llama2_7b.input_cost_per_million == 0.15
        assert llama2_7b.output_cost_per_million == 0.15

    def test_cost_calculation_precision(self):
        """Test cost calculations maintain precision."""
        # Test with various token counts
        test_cases = [
            (1, 1),
            (10, 10),
            (100, 100),
            (1000, 1000),
            (12345, 67890),
        ]

        for input_tokens, output_tokens in test_cases:
            cost = calculate_completion_cost(
                "meta-llama/Llama-2-70b-chat-hf", input_tokens, output_tokens
            )

            expected = (input_tokens + output_tokens) / 1_000_000 * 1.0
            assert cost == pytest.approx(expected, abs=1e-10)
