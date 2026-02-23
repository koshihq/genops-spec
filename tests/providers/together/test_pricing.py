#!/usr/bin/env python3
"""
Unit tests for Together AI pricing calculator.

Tests cost calculation accuracy, model recommendations,
cost analysis, and pricing intelligence features.
"""

import os
import sys
from decimal import Decimal

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from src.genops.providers.together_pricing import (
        TOGETHER_CONTEXT_LENGTHS,
        TOGETHER_PRICING,
        TogetherPricingCalculator,
    )
except ImportError as e:
    pytest.skip(f"Together AI pricing not available: {e}", allow_module_level=True)


class TestTogetherPricingCalculator:
    """Unit tests for TogetherPricingCalculator class."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.calc = TogetherPricingCalculator()

    def test_calculator_initialization(self):
        """Test pricing calculator initializes correctly."""
        assert isinstance(self.calc, TogetherPricingCalculator)
        assert hasattr(self.calc, "pricing_data")
        assert len(self.calc.pricing_data) > 0

    def test_estimate_chat_cost_llama_8b(self):
        """Test cost estimation for Llama 3.1 8B model."""
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        cost = self.calc.estimate_chat_cost(model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Should be approximately $0.0001 for 1000 tokens at $0.10/M rate
        assert 0.00008 < float(cost) < 0.00012

    def test_estimate_chat_cost_llama_70b(self):
        """Test cost estimation for Llama 3.1 70B model."""
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        cost = self.calc.estimate_chat_cost(model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Should be approximately $0.00088 for 1000 tokens at $0.88/M rate
        assert 0.0008 < float(cost) < 0.001

    def test_estimate_chat_cost_llama_405b(self):
        """Test cost estimation for Llama 3.1 405B model."""
        model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        cost = self.calc.estimate_chat_cost(model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Should be approximately $0.005 for 1000 tokens at $5.00/M rate
        assert 0.004 < float(cost) < 0.006

    def test_estimate_chat_cost_deepseek_r1(self):
        """Test cost estimation for DeepSeek R1 model."""
        model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        cost = self.calc.estimate_chat_cost(model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0
        # DeepSeek models should have competitive pricing
        assert float(cost) < 0.002

    def test_estimate_chat_cost_unknown_model(self):
        """Test cost estimation for unknown model uses fallback."""
        unknown_model = "unknown/custom-model"
        cost = self.calc.estimate_chat_cost(unknown_model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Should use generic fallback pricing
        assert float(cost) > 0

    def test_estimate_chat_cost_with_input_output_tokens(self):
        """Test cost estimation with separate input/output tokens."""
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        cost = self.calc.estimate_chat_cost(model, input_tokens=500, output_tokens=300)

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Total 800 tokens should cost less than 1000 tokens
        cost_1000 = self.calc.estimate_chat_cost(model, tokens=1000)
        assert cost < cost_1000

    def test_calculate_detailed_cost_breakdown(self):
        """Test detailed cost calculation with breakdown."""
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        breakdown = self.calc.calculate_detailed_cost(
            model=model,
            input_tokens=200,
            output_tokens=150,
            session_context="test-session",
        )

        assert isinstance(breakdown, dict)
        assert "total_cost" in breakdown
        assert "input_cost" in breakdown
        assert "output_cost" in breakdown
        assert "model" in breakdown
        assert "tokens_breakdown" in breakdown

        assert isinstance(breakdown["total_cost"], Decimal)
        assert breakdown["total_cost"] > 0
        assert breakdown["model"] == model

    def test_recommend_model_for_simple_task(self):
        """Test model recommendation for simple tasks."""
        recommendation = self.calc.recommend_model(
            task_complexity="simple", budget_per_operation=0.001
        )

        assert isinstance(recommendation, dict)
        assert "recommended_model" in recommendation
        assert "estimated_cost" in recommendation
        assert "reasoning" in recommendation
        assert "budget_compliant" in recommendation

        # Should recommend a lite tier model for simple tasks
        assert recommendation["budget_compliant"] is True
        assert recommendation["estimated_cost"] <= 0.001

    def test_recommend_model_for_complex_task(self):
        """Test model recommendation for complex tasks."""
        recommendation = self.calc.recommend_model(
            task_complexity="complex",
            budget_per_operation=0.01,
            min_context_length=32768,
        )

        assert isinstance(recommendation, dict)
        assert recommendation["recommended_model"] is not None
        assert recommendation["budget_compliant"] is True
        assert recommendation["context_length"] >= 32768

    def test_recommend_model_with_tight_budget(self):
        """Test model recommendation with very tight budget."""
        recommendation = self.calc.recommend_model(
            task_complexity="moderate",
            budget_per_operation=0.0001,  # Very tight budget
        )

        # Should still recommend something, even if budget compliance is questionable
        assert isinstance(recommendation, dict)
        assert "recommended_model" in recommendation

    def test_compare_models_cost_effectiveness(self):
        """Test cost comparison across multiple models."""
        models = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ]

        comparisons = self.calc.compare_models(models, estimated_tokens=500)

        assert isinstance(comparisons, list)
        assert len(comparisons) == len(models)

        for comparison in comparisons:
            assert "model" in comparison
            assert "estimated_cost" in comparison
            assert "tier" in comparison
            assert "context_length" in comparison
            assert isinstance(comparison["estimated_cost"], (float, Decimal))

        # Should be sorted by cost-effectiveness
        costs = [comp["estimated_cost"] for comp in comparisons]
        assert costs == sorted(costs)

    def test_analyze_costs_daily_usage(self):
        """Test cost analysis for daily usage patterns."""
        analysis = self.calc.analyze_costs(
            operations_per_day=100,
            avg_tokens_per_operation=300,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            days_to_analyze=30,
        )

        assert isinstance(analysis, dict)
        assert "daily_cost" in analysis
        assert "monthly_cost" in analysis
        assert "yearly_cost" in analysis
        assert "operations_breakdown" in analysis
        assert "potential_savings" in analysis

        assert analysis["monthly_cost"] == analysis["daily_cost"] * 30
        assert analysis["yearly_cost"] == analysis["daily_cost"] * 365

    def test_analyze_costs_with_savings_opportunities(self):
        """Test cost analysis identifies savings opportunities."""
        # Use expensive model to trigger savings suggestions
        analysis = self.calc.analyze_costs(
            operations_per_day=1000,
            avg_tokens_per_operation=500,
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            days_to_analyze=30,
        )

        assert "potential_savings" in analysis
        savings = analysis["potential_savings"]

        if savings["best_alternative"]:
            assert "model" in savings["best_alternative"]
            assert "potential_monthly_savings" in savings
            assert isinstance(savings["potential_monthly_savings"], (int, float))

    def test_calculate_fine_tuning_cost(self):
        """Test fine-tuning cost calculation."""
        cost = self.calc.calculate_fine_tuning_cost(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            training_tokens=100_000,
            validation_tokens=10_000,
            epochs=3,
        )

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Fine-tuning should be more expensive than inference
        inference_cost = self.calc.estimate_chat_cost(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", tokens=100_000
        )
        assert cost > inference_cost

    def test_get_model_tier_classification(self):
        """Test model tier classification."""
        # Test lite tier
        tier_8b = self.calc.get_model_tier(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        assert tier_8b == "lite"

        # Test standard tier
        tier_70b = self.calc.get_model_tier(
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        )
        assert tier_70b == "standard"

        # Test large tier
        tier_405b = self.calc.get_model_tier(
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        )
        assert tier_405b == "large"

    def test_get_model_context_length(self):
        """Test model context length retrieval."""
        # Test known model
        context_8b = self.calc.get_model_context_length(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        assert isinstance(context_8b, int)
        assert context_8b >= 32768  # Llama 3.1 has extended context

        # Test unknown model
        context_unknown = self.calc.get_model_context_length("unknown/custom-model")
        assert isinstance(context_unknown, int)
        assert context_unknown > 0  # Should have fallback

    def test_pricing_data_consistency(self):
        """Test pricing data structure consistency."""
        assert isinstance(TOGETHER_PRICING, dict)
        assert len(TOGETHER_PRICING) > 0

        for model, pricing in TOGETHER_PRICING.items():
            assert isinstance(model, str)
            assert len(model) > 0
            assert isinstance(pricing, Decimal)
            assert pricing > 0

    def test_context_length_data_consistency(self):
        """Test context length data structure consistency."""
        assert isinstance(TOGETHER_CONTEXT_LENGTHS, dict)
        assert len(TOGETHER_CONTEXT_LENGTHS) > 0

        for model, context_length in TOGETHER_CONTEXT_LENGTHS.items():
            assert isinstance(model, str)
            assert len(model) > 0
            assert isinstance(context_length, int)
            assert context_length > 0

    def test_cost_calculation_precision(self):
        """Test cost calculations maintain precision."""
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

        # Calculate cost for small token counts
        cost_1 = self.calc.estimate_chat_cost(model, tokens=1)
        cost_10 = self.calc.estimate_chat_cost(model, tokens=10)
        cost_100 = self.calc.estimate_chat_cost(model, tokens=100)

        # Should maintain proportional relationship
        assert cost_10 == cost_1 * 10
        assert cost_100 == cost_1 * 100

        # Should maintain precision
        assert isinstance(cost_1, Decimal)
        assert len(str(cost_1).split(".")[-1]) >= 6  # At least 6 decimal places

    def test_batch_cost_estimation(self):
        """Test batch cost estimation for multiple operations."""
        operations = [
            {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "tokens": 100},
            {"model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "tokens": 200},
            {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "tokens": 150},
        ]

        total_cost = Decimal("0")
        individual_costs = []

        for op in operations:
            cost = self.calc.estimate_chat_cost(op["model"], tokens=op["tokens"])
            individual_costs.append(cost)
            total_cost += cost

        assert len(individual_costs) == len(operations)
        assert all(isinstance(cost, Decimal) for cost in individual_costs)
        assert total_cost == sum(individual_costs)
        assert total_cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
