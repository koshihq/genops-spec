"""Tests for Vercel AI SDK pricing calculation module."""

import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from genops.providers.vercel_ai_sdk_pricing import (
    CostBreakdown,
    ModelPricing,
    ProviderType,
    VercelAISDKPricingCalculator,
    calculate_cost,
    estimate_cost,
    get_model_info,
    get_supported_providers,
    pricing_calculator,
)


class TestModelPricing(unittest.TestCase):
    """Test ModelPricing data class."""

    def test_model_pricing_creation(self):
        """Test creating a ModelPricing object."""
        pricing = ModelPricing(
            input_price_per_1k=Decimal("0.01"),
            output_price_per_1k=Decimal("0.03"),
            provider="openai",
            model_name="gpt-4",
        )

        self.assertEqual(pricing.input_price_per_1k, Decimal("0.01"))
        self.assertEqual(pricing.output_price_per_1k, Decimal("0.03"))
        self.assertEqual(pricing.provider, "openai")
        self.assertEqual(pricing.model_name, "gpt-4")
        self.assertTrue(pricing.supports_streaming)  # Default value
        self.assertFalse(pricing.supports_tools)  # Default value
        self.assertEqual(pricing.context_length, 4096)  # Default value

    def test_model_pricing_with_features(self):
        """Test ModelPricing with advanced features."""
        pricing = ModelPricing(
            input_price_per_1k=Decimal("0.03"),
            output_price_per_1k=Decimal("0.06"),
            provider="openai",
            model_name="gpt-4",
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            context_length=8192,
        )

        self.assertTrue(pricing.supports_streaming)
        self.assertTrue(pricing.supports_tools)
        self.assertTrue(pricing.supports_vision)
        self.assertEqual(pricing.context_length, 8192)


class TestCostBreakdown(unittest.TestCase):
    """Test CostBreakdown data class."""

    def test_cost_breakdown_creation(self):
        """Test creating a CostBreakdown object."""
        breakdown = CostBreakdown(
            input_tokens=100,
            output_tokens=150,
            input_cost=Decimal("0.001"),
            output_cost=Decimal("0.0045"),
            total_cost=Decimal("0.0055"),
            provider="openai",
            model="gpt-4",
        )

        self.assertEqual(breakdown.input_tokens, 100)
        self.assertEqual(breakdown.output_tokens, 150)
        self.assertEqual(breakdown.input_cost, Decimal("0.001"))
        self.assertEqual(breakdown.output_cost, Decimal("0.0045"))
        self.assertEqual(breakdown.total_cost, Decimal("0.0055"))
        self.assertEqual(breakdown.provider, "openai")
        self.assertEqual(breakdown.model, "gpt-4")
        self.assertEqual(breakdown.currency, "USD")  # Default
        self.assertFalse(breakdown.estimated)  # Default


class TestProviderType(unittest.TestCase):
    """Test ProviderType enum."""

    def test_provider_types(self):
        """Test provider type enum values."""
        self.assertEqual(ProviderType.OPENAI.value, "openai")
        self.assertEqual(ProviderType.ANTHROPIC.value, "anthropic")
        self.assertEqual(ProviderType.GOOGLE.value, "google")
        self.assertEqual(ProviderType.UNKNOWN.value, "unknown")


class TestVercelAISDKPricingCalculator(unittest.TestCase):
    """Test the main pricing calculator."""

    def setUp(self):
        """Set up test environment."""
        self.calculator = VercelAISDKPricingCalculator()

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        self.assertIsInstance(self.calculator.DEFAULT_PRICING, dict)
        self.assertGreater(len(self.calculator.DEFAULT_PRICING), 0)
        self.assertIn("gpt-4", self.calculator.DEFAULT_PRICING)
        self.assertIn("claude-3-sonnet", self.calculator.DEFAULT_PRICING)

    def test_get_model_key(self):
        """Test model key generation for pricing lookup."""
        # Test exact match
        self.assertEqual(self.calculator._get_model_key("openai", "gpt-4"), "gpt-4")

        # Test with provider prefix
        self.assertEqual(
            self.calculator._get_model_key("openai", "openai/gpt-4"), "gpt-4"
        )

        # Test unknown model fallback
        unknown_key = self.calculator._get_model_key("unknown", "unknown-model")
        self.assertIn(unknown_key, ["unknown-small", "unknown-large"])

    def test_get_pricing_info(self):
        """Test getting pricing information for models."""
        # Test known model
        pricing = self.calculator._get_pricing_info("gpt-4", "openai", "gpt-4")
        self.assertEqual(pricing.provider, "openai")
        self.assertEqual(pricing.model_name, "gpt-4")
        self.assertGreater(pricing.input_price_per_1k, Decimal("0"))

        # Test unknown model
        pricing = self.calculator._get_pricing_info(
            "unknown-model", "test", "test-model"
        )
        self.assertEqual(pricing.provider, "test")
        self.assertEqual(pricing.model_name, "test-model")

    def test_calculate_cost_known_model(self):
        """Test cost calculation for known models."""
        breakdown = self.calculator.calculate_cost("openai", "gpt-4", 100, 150)

        self.assertIsInstance(breakdown, CostBreakdown)
        self.assertEqual(breakdown.input_tokens, 100)
        self.assertEqual(breakdown.output_tokens, 150)
        self.assertEqual(breakdown.provider, "openai")
        self.assertEqual(breakdown.model, "gpt-4")
        self.assertGreater(breakdown.total_cost, Decimal("0"))
        self.assertEqual(
            breakdown.input_cost + breakdown.output_cost, breakdown.total_cost
        )

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown models."""
        breakdown = self.calculator.calculate_cost("unknown", "unknown-model", 100, 150)

        self.assertIsInstance(breakdown, CostBreakdown)
        self.assertEqual(breakdown.provider, "unknown")
        self.assertEqual(breakdown.model, "unknown-model")
        self.assertTrue(breakdown.estimated)
        self.assertGreater(breakdown.total_cost, Decimal("0"))

    def test_calculate_cost_with_provider_calculator(self):
        """Test cost calculation using provider-specific calculator."""
        # Mock a provider calculator
        mock_calculator = Mock(return_value=Decimal("0.002"))
        self.calculator.provider_calculators = {"openai": mock_calculator}

        breakdown = self.calculator.calculate_cost("openai", "gpt-4", 100, 150)

        mock_calculator.assert_called_once_with("gpt-4", 100, 150)
        self.assertEqual(breakdown.total_cost, Decimal("0.002"))
        self.assertEqual(breakdown.pricing_source, "genops_provider")
        self.assertFalse(breakdown.estimated)

    def test_calculate_cost_provider_calculator_error(self):
        """Test fallback when provider calculator fails."""
        # Mock a failing provider calculator
        mock_calculator = Mock(side_effect=Exception("Calculator error"))
        self.calculator.provider_calculators = {"openai": mock_calculator}

        breakdown = self.calculator.calculate_cost("openai", "gpt-4", 100, 150)

        # Should fall back to default pricing
        self.assertEqual(breakdown.pricing_source, "default")
        self.assertTrue(breakdown.estimated)
        self.assertGreater(breakdown.total_cost, Decimal("0"))

    def test_get_model_info(self):
        """Test getting model information."""
        # Test known model
        info = self.calculator.get_model_info("openai", "gpt-4")
        self.assertIsInstance(info, ModelPricing)
        self.assertEqual(info.provider, "openai")
        self.assertEqual(info.model_name, "gpt-4")

        # Test unknown model
        info = self.calculator.get_model_info("unknown", "unknown-model")
        self.assertIsInstance(info, ModelPricing)
        self.assertEqual(info.provider, "unknown")
        self.assertEqual(info.model_name, "unknown-model")

    def test_estimate_cost(self):
        """Test cost estimation from prompt length."""
        min_cost, max_cost = self.calculator.estimate_cost(
            "openai",
            "gpt-4",
            prompt_length=400,  # ~100 tokens
            expected_response_length=800,  # ~200 tokens
        )

        self.assertIsInstance(min_cost, Decimal)
        self.assertIsInstance(max_cost, Decimal)
        self.assertGreater(min_cost, Decimal("0"))
        self.assertGreater(max_cost, min_cost)

    def test_estimate_cost_default_response_length(self):
        """Test cost estimation with default response length."""
        min_cost, max_cost = self.calculator.estimate_cost(
            "openai", "gpt-4", prompt_length=400
        )

        self.assertIsInstance(min_cost, Decimal)
        self.assertIsInstance(max_cost, Decimal)
        self.assertGreater(max_cost, min_cost)

    def test_get_supported_providers(self):
        """Test getting supported providers and models."""
        providers = self.calculator.get_supported_providers()

        self.assertIsInstance(providers, dict)
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)
        self.assertIsInstance(providers["openai"], list)
        self.assertGreater(len(providers["openai"]), 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    def test_calculate_cost_function(self):
        """Test the calculate_cost convenience function."""
        breakdown = calculate_cost("openai", "gpt-4", 100, 150)

        self.assertIsInstance(breakdown, CostBreakdown)
        self.assertEqual(breakdown.provider, "openai")
        self.assertEqual(breakdown.model, "gpt-4")

    def test_estimate_cost_function(self):
        """Test the estimate_cost convenience function."""
        min_cost, max_cost = estimate_cost("openai", "gpt-4", 400, 800)

        self.assertIsInstance(min_cost, Decimal)
        self.assertIsInstance(max_cost, Decimal)
        self.assertGreater(max_cost, min_cost)

    def test_get_model_info_function(self):
        """Test the get_model_info convenience function."""
        info = get_model_info("openai", "gpt-4")

        self.assertIsInstance(info, ModelPricing)
        self.assertEqual(info.provider, "openai")

    def test_get_supported_providers_function(self):
        """Test the get_supported_providers convenience function."""
        providers = get_supported_providers()

        self.assertIsInstance(providers, dict)
        self.assertGreater(len(providers), 0)


class TestProviderSpecificPricing(unittest.TestCase):
    """Test pricing for specific providers."""

    def setUp(self):
        """Set up test environment."""
        self.calculator = VercelAISDKPricingCalculator()

    def test_openai_models(self):
        """Test pricing for OpenAI models."""
        models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

        for model in models:
            breakdown = self.calculator.calculate_cost("openai", model, 100, 150)
            self.assertEqual(breakdown.provider, "openai")
            self.assertEqual(breakdown.model, model)
            self.assertGreater(breakdown.total_cost, Decimal("0"))

    def test_anthropic_models(self):
        """Test pricing for Anthropic models."""
        models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

        for model in models:
            breakdown = self.calculator.calculate_cost("anthropic", model, 100, 150)
            self.assertEqual(breakdown.provider, "anthropic")
            self.assertEqual(breakdown.model, model)
            self.assertGreater(breakdown.total_cost, Decimal("0"))

    def test_google_models(self):
        """Test pricing for Google models."""
        models = ["gemini-pro", "gemini-pro-vision"]

        for model in models:
            breakdown = self.calculator.calculate_cost("google", model, 100, 150)
            self.assertEqual(breakdown.provider, "google")
            self.assertEqual(breakdown.model, model)
            self.assertGreater(breakdown.total_cost, Decimal("0"))

    def test_cost_comparison(self):
        """Test cost comparison between different models."""
        # Generally, larger models should be more expensive
        gpt35_cost = self.calculator.calculate_cost("openai", "gpt-3.5-turbo", 100, 150)
        gpt4_cost = self.calculator.calculate_cost("openai", "gpt-4", 100, 150)

        # GPT-4 should be more expensive than GPT-3.5-turbo
        self.assertGreater(gpt4_cost.total_cost, gpt35_cost.total_cost)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test environment."""
        self.calculator = VercelAISDKPricingCalculator()

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        breakdown = self.calculator.calculate_cost("openai", "gpt-4", 0, 0)

        self.assertEqual(breakdown.input_tokens, 0)
        self.assertEqual(breakdown.output_tokens, 0)
        self.assertEqual(breakdown.total_cost, Decimal("0"))

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        breakdown = self.calculator.calculate_cost("openai", "gpt-4", 100000, 50000)

        self.assertEqual(breakdown.input_tokens, 100000)
        self.assertEqual(breakdown.output_tokens, 50000)
        self.assertGreater(breakdown.total_cost, Decimal("1"))  # Should be substantial

    def test_model_with_slash(self):
        """Test model names with provider prefixes."""
        breakdown = self.calculator.calculate_cost("openai", "openai/gpt-4", 100, 150)

        self.assertEqual(breakdown.provider, "openai")
        self.assertEqual(breakdown.model, "openai/gpt-4")
        self.assertGreater(breakdown.total_cost, Decimal("0"))

    def test_case_insensitive_providers(self):
        """Test that provider names are handled case-insensitively."""
        breakdown1 = self.calculator.calculate_cost("OpenAI", "gpt-4", 100, 150)
        breakdown2 = self.calculator.calculate_cost("openai", "gpt-4", 100, 150)

        # Both should use the same pricing (after normalization)
        self.assertEqual(breakdown1.provider, "openai")  # Normalized to lowercase
        self.assertEqual(breakdown2.provider, "openai")


class TestProviderCalculatorIntegration(unittest.TestCase):
    """Test integration with existing GenOps provider calculators."""

    def setUp(self):
        """Set up test environment."""
        self.calculator = VercelAISDKPricingCalculator()

    def test_initialize_provider_calculators(self):
        """Test initialization of provider calculators."""
        calculators = self.calculator._initialize_provider_calculators()

        self.assertIsInstance(calculators, dict)
        # The actual providers available depend on what's installed
        # but the structure should be correct

    @patch("genops.providers.vercel_ai_sdk_pricing.__import__")
    def test_provider_calculator_import_success(self, mock_import):
        """Test successful import of provider calculator."""
        mock_module = Mock()
        mock_module.calculate_cost = Mock(return_value=Decimal("0.002"))
        mock_import.return_value = mock_module

        VercelAISDKPricingCalculator()
        # Should have attempted to import provider modules
        mock_import.assert_called()

    @patch("genops.providers.vercel_ai_sdk_pricing.__import__")
    def test_provider_calculator_import_error(self, mock_import):
        """Test graceful handling of import errors."""
        mock_import.side_effect = ImportError("Module not found")

        # Should not raise exception, just log warning
        calculator = VercelAISDKPricingCalculator()
        self.assertIsInstance(calculator.provider_calculators, dict)


class TestGlobalPricingCalculatorInstance(unittest.TestCase):
    """Test the global pricing calculator instance."""

    def test_global_instance_exists(self):
        """Test that global pricing calculator instance exists."""
        self.assertIsInstance(pricing_calculator, VercelAISDKPricingCalculator)

    def test_global_functions_use_instance(self):
        """Test that global convenience functions use the global instance."""
        # This is mainly a smoke test to ensure functions work
        breakdown = calculate_cost("openai", "gpt-4", 100, 150)
        self.assertIsInstance(breakdown, CostBreakdown)

        min_cost, max_cost = estimate_cost("openai", "gpt-4", 400)
        self.assertIsInstance(min_cost, Decimal)
        self.assertIsInstance(max_cost, Decimal)


if __name__ == "__main__":
    unittest.main()
