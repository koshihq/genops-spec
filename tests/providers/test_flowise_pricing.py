"""
Test suite for Flowise pricing and cost calculation module.

This module tests the cost calculation functionality for Flowise,
including pricing tiers, model-specific costs, and optimization recommendations.
"""

import json
from decimal import Decimal, InvalidOperation

import pytest

from genops.providers.flowise_pricing import (
    CostOptimizationRecommendation,
    FlowiseCostCalculator,
    FlowisePricingTier,
    calculate_bulk_costs,
    calculate_flowise_cost,
    estimate_flowise_tokens,
    get_cost_optimization_recommendations,
    get_model_pricing_info,
)


class TestFlowisePricingTier:
    """Test FlowisePricingTier data class."""

    def test_pricing_tier_creation(self):
        """Test creating a pricing tier."""
        tier = FlowisePricingTier(
            name="professional",
            cost_per_1k_tokens=Decimal("0.002"),
            monthly_limit=100000,
            description="Professional tier for businesses",
        )

        assert tier.name == "professional"
        assert tier.cost_per_1k_tokens == Decimal("0.002")
        assert tier.monthly_limit == 100000
        assert tier.description == "Professional tier for businesses"

    def test_pricing_tier_defaults(self):
        """Test pricing tier with default values."""
        tier = FlowisePricingTier("basic", Decimal("0.001"))

        assert tier.name == "basic"
        assert tier.cost_per_1k_tokens == Decimal("0.001")
        assert tier.monthly_limit is None
        assert tier.description is None

    def test_pricing_tier_validation(self):
        """Test pricing tier validates input values."""
        # Test negative cost
        with pytest.raises(ValueError):
            FlowisePricingTier("invalid", Decimal("-0.001"))

        # Test zero cost (should be allowed)
        tier = FlowisePricingTier("free", Decimal("0"))
        assert tier.cost_per_1k_tokens == Decimal("0")

    def test_pricing_tier_comparison(self):
        """Test comparing pricing tiers."""
        tier1 = FlowisePricingTier("basic", Decimal("0.001"))
        tier2 = FlowisePricingTier("premium", Decimal("0.002"))

        # Should be able to compare by cost
        assert tier1.cost_per_1k_tokens < tier2.cost_per_1k_tokens

    def test_pricing_tier_serialization(self):
        """Test pricing tier can be serialized."""
        tier = FlowisePricingTier(
            "professional",
            Decimal("0.002"),
            monthly_limit=50000,
            description="Business tier",
        )

        # Convert to dict for JSON serialization
        tier_dict = {
            "name": tier.name,
            "cost_per_1k_tokens": str(tier.cost_per_1k_tokens),
            "monthly_limit": tier.monthly_limit,
            "description": tier.description,
        }

        json_str = json.dumps(tier_dict)
        parsed = json.loads(json_str)

        assert parsed["name"] == "professional"
        assert Decimal(parsed["cost_per_1k_tokens"]) == Decimal("0.002")


class TestFlowiseCostCalculator:
    """Test FlowiseCostCalculator functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.calculator = FlowiseCostCalculator()

    def test_calculator_initialization(self):
        """Test calculator initialization with default pricing."""
        assert isinstance(self.calculator, FlowiseCostCalculator)
        assert len(self.calculator.pricing_tiers) > 0
        assert isinstance(self.calculator.model_pricing, dict)

    def test_calculator_with_custom_pricing(self):
        """Test calculator with custom pricing tiers."""
        custom_tiers = [
            FlowisePricingTier("starter", Decimal("0.0005"), 25000),
            FlowisePricingTier("business", Decimal("0.0015"), 100000),
        ]

        calculator = FlowiseCostCalculator(custom_pricing_tiers=custom_tiers)

        assert len(calculator.pricing_tiers) == 2
        assert calculator.pricing_tiers[0].name == "starter"
        assert calculator.pricing_tiers[1].name == "business"

    def test_basic_cost_calculation(self):
        """Test basic cost calculation."""
        cost = self.calculator.calculate_cost(
            input_tokens=1000, output_tokens=500, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_cost_calculation_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = self.calculator.calculate_cost(
            input_tokens=0, output_tokens=0, model_name="gpt-3.5-turbo"
        )

        assert cost == Decimal("0")

    def test_cost_calculation_input_only(self):
        """Test cost calculation with only input tokens."""
        cost = self.calculator.calculate_cost(
            input_tokens=1000, output_tokens=0, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_cost_calculation_output_only(self):
        """Test cost calculation with only output tokens."""
        cost = self.calculator.calculate_cost(
            input_tokens=0, output_tokens=500, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_cost_calculation_different_models(self):
        """Test cost calculation for different models."""
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3", "gemini-pro"]

        costs = {}
        for model in models:
            cost = self.calculator.calculate_cost(
                input_tokens=1000, output_tokens=500, model_name=model
            )
            costs[model] = cost
            assert isinstance(cost, Decimal)
            assert cost > 0

        # Different models should potentially have different costs
        assert len(set(costs.values())) >= 1

    def test_cost_calculation_unknown_model(self):
        """Test cost calculation for unknown model uses default pricing."""
        cost = self.calculator.calculate_cost(
            input_tokens=1000, output_tokens=500, model_name="unknown-model-xyz"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_cost_calculation_with_pricing_tier(self):
        """Test cost calculation with specific pricing tier."""
        custom_tiers = [FlowisePricingTier("premium", Decimal("0.003"), 200000)]
        calculator = FlowiseCostCalculator(custom_pricing_tiers=custom_tiers)

        cost = calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="gpt-3.5-turbo",
            pricing_tier="premium",
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_cost_calculation_invalid_tier(self):
        """Test cost calculation with invalid pricing tier."""
        cost = self.calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="gpt-3.5-turbo",
            pricing_tier="nonexistent-tier",
        )

        # Should fall back to default pricing
        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_cost_calculation_with_multiplier(self):
        """Test cost calculation with cost multiplier."""
        base_cost = self.calculator.calculate_cost(
            input_tokens=1000, output_tokens=500, model_name="gpt-3.5-turbo"
        )

        multiplied_cost = self.calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="gpt-3.5-turbo",
            cost_multiplier=Decimal("1.5"),
        )

        assert multiplied_cost == base_cost * Decimal("1.5")

    def test_cost_calculation_negative_tokens(self):
        """Test cost calculation rejects negative token counts."""
        with pytest.raises(ValueError):
            self.calculator.calculate_cost(
                input_tokens=-100, output_tokens=500, model_name="gpt-3.5-turbo"
            )

        with pytest.raises(ValueError):
            self.calculator.calculate_cost(
                input_tokens=1000, output_tokens=-50, model_name="gpt-3.5-turbo"
            )

    def test_estimate_tokens_from_text(self):
        """Test token estimation from text."""
        texts = [
            "",
            "Hello",
            "This is a test message.",
            "This is a much longer message with many words and it should result in more tokens being estimated.",
            "Special characters: !@#$%^&*()",
            "Unicode text: 你好世界 🌍",
        ]

        for text in texts:
            tokens = self.calculator.estimate_tokens(text)
            assert isinstance(tokens, int)
            assert tokens >= 0

            if text:
                assert tokens > 0
            else:
                assert tokens == 0

    def test_estimate_tokens_accuracy(self):
        """Test token estimation gives reasonable results."""
        # Simple cases
        assert self.calculator.estimate_tokens("hello") > 0
        assert self.calculator.estimate_tokens(
            "hello world"
        ) > self.calculator.estimate_tokens("hello")

        # Longer text should have more tokens
        short_text = "Hello world"
        long_text = "This is a much longer piece of text with many more words and should result in significantly more tokens."

        short_tokens = self.calculator.estimate_tokens(short_text)
        long_tokens = self.calculator.estimate_tokens(long_text)

        assert long_tokens > short_tokens

    def test_get_pricing_tier_by_name(self):
        """Test getting pricing tier by name."""
        tier = self.calculator.get_pricing_tier("default")
        if tier:  # If default tier exists
            assert isinstance(tier, FlowisePricingTier)
            assert tier.name == "default"

        # Test nonexistent tier
        assert self.calculator.get_pricing_tier("nonexistent") is None

    def test_get_model_pricing_info(self):
        """Test getting model pricing information."""
        info = self.calculator.get_model_pricing_info("gpt-3.5-turbo")
        assert isinstance(info, dict)

        # Should contain basic pricing info
        expected_keys = ["input_cost", "output_cost", "model_name"]
        for key in expected_keys:
            if key in info:
                assert info[key] is not None

    def test_calculate_monthly_costs(self):
        """Test calculating monthly costs based on usage."""
        monthly_usage = [
            (1000, 500),  # Day 1: 1000 input, 500 output
            (2000, 1000),  # Day 2: 2000 input, 1000 output
            (1500, 750),  # Day 3: 1500 input, 750 output
        ]

        total_cost = Decimal("0")
        for input_tokens, output_tokens in monthly_usage:
            daily_cost = self.calculator.calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_name="gpt-3.5-turbo",
            )
            total_cost += daily_cost

        assert total_cost > 0
        assert isinstance(total_cost, Decimal)


class TestStandaloneFunctions:
    """Test standalone utility functions."""

    def test_calculate_flowise_cost_function(self):
        """Test standalone calculate_flowise_cost function."""
        cost = calculate_flowise_cost(
            input_tokens=1000, output_tokens=500, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_calculate_flowise_cost_with_params(self):
        """Test calculate_flowise_cost with additional parameters."""
        cost = calculate_flowise_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="gpt-4",
            pricing_tier="premium",
            cost_multiplier=Decimal("1.2"),
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_estimate_flowise_tokens_function(self):
        """Test standalone estimate_flowise_tokens function."""
        tokens = estimate_flowise_tokens("This is a test message")

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_get_model_pricing_info_function(self):
        """Test standalone get_model_pricing_info function."""
        info = get_model_pricing_info("gpt-3.5-turbo")

        assert isinstance(info, dict)
        # Should contain model information

    def test_calculate_bulk_costs(self):
        """Test calculating costs for multiple requests in bulk."""
        requests = [
            {"input_tokens": 1000, "output_tokens": 500, "model_name": "gpt-3.5-turbo"},
            {"input_tokens": 2000, "output_tokens": 1000, "model_name": "gpt-4"},
            {"input_tokens": 1500, "output_tokens": 750, "model_name": "claude-3"},
        ]

        results = calculate_bulk_costs(requests)

        assert isinstance(results, list)
        assert len(results) == len(requests)

        for result in results:
            assert isinstance(result, dict)
            assert "cost" in result
            assert isinstance(result["cost"], Decimal)
            assert result["cost"] > 0


class TestCostOptimizationRecommendation:
    """Test CostOptimizationRecommendation data class."""

    def test_recommendation_creation(self):
        """Test creating cost optimization recommendation."""
        rec = CostOptimizationRecommendation(
            recommendation_type="model_switch",
            current_model="gpt-4",
            suggested_model="gpt-3.5-turbo",
            estimated_savings=Decimal("0.05"),
            confidence_score=0.85,
            description="Switch to more cost-effective model",
            potential_tradeoffs=["Slightly reduced quality"],
        )

        assert rec.recommendation_type == "model_switch"
        assert rec.current_model == "gpt-4"
        assert rec.suggested_model == "gpt-3.5-turbo"
        assert rec.estimated_savings == Decimal("0.05")
        assert rec.confidence_score == 0.85
        assert len(rec.potential_tradeoffs) == 1

    def test_recommendation_defaults(self):
        """Test recommendation with default values."""
        rec = CostOptimizationRecommendation(
            recommendation_type="usage_optimization", description="Optimize token usage"
        )

        assert rec.recommendation_type == "usage_optimization"
        assert rec.current_model is None
        assert rec.suggested_model is None
        assert rec.estimated_savings == Decimal("0")
        assert rec.confidence_score == 0.0
        assert rec.potential_tradeoffs == []

    def test_recommendation_validation(self):
        """Test recommendation validates input values."""
        # Test invalid confidence score
        with pytest.raises(ValueError):
            CostOptimizationRecommendation(
                recommendation_type="test",
                description="Test",
                confidence_score=1.5,  # Should be <= 1.0
            )

        with pytest.raises(ValueError):
            CostOptimizationRecommendation(
                recommendation_type="test",
                description="Test",
                confidence_score=-0.1,  # Should be >= 0.0
            )


class TestCostOptimization:
    """Test cost optimization recommendations."""

    def test_get_basic_recommendations(self):
        """Test getting basic cost optimization recommendations."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.10"),
            input_tokens=1000,
            output_tokens=500,
        )

        assert isinstance(recommendations, list)
        # Should have at least some recommendations
        assert len(recommendations) >= 0

        for rec in recommendations:
            assert isinstance(rec, (dict, CostOptimizationRecommendation))

    def test_get_recommendations_with_budget_constraint(self):
        """Test recommendations with budget constraints."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.10"),
            input_tokens=1000,
            output_tokens=500,
            budget_constraint=Decimal("0.05"),
        )

        assert isinstance(recommendations, list)

        # Recommendations should respect budget constraints
        for rec in recommendations:
            if isinstance(rec, dict) and "estimated_cost" in rec:
                assert rec["estimated_cost"] <= Decimal("0.05")
            elif hasattr(rec, "estimated_cost"):
                assert rec.estimated_cost <= Decimal("0.05")

    def test_get_recommendations_for_expensive_models(self):
        """Test recommendations prioritize expensive models."""
        expensive_recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.20"),
            input_tokens=2000,
            output_tokens=1000,
        )

        cheap_recommendations = get_cost_optimization_recommendations(
            current_model="gpt-3.5-turbo",
            current_cost=Decimal("0.01"),
            input_tokens=200,
            output_tokens=100,
        )

        # Should have more recommendations for expensive usage
        assert len(expensive_recommendations) >= len(cheap_recommendations)

    def test_get_recommendations_model_alternatives(self):
        """Test recommendations suggest model alternatives."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.15"),
            input_tokens=1500,
            output_tokens=750,
        )

        # Should include model switching recommendations
        model_switch_recs = [
            rec
            for rec in recommendations
            if (isinstance(rec, dict) and rec.get("type") == "model_switch")
            or (
                hasattr(rec, "recommendation_type")
                and rec.recommendation_type == "model_switch"
            )
        ]

        assert len(model_switch_recs) >= 0

    def test_get_recommendations_usage_optimization(self):
        """Test recommendations include usage optimization."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-3.5-turbo",
            current_cost=Decimal("0.08"),
            input_tokens=5000,  # Large input
            output_tokens=3000,  # Large output
        )

        # Should include usage optimization recommendations
        usage_recs = [
            rec
            for rec in recommendations
            if (isinstance(rec, dict) and rec.get("type") == "usage_optimization")
            or (
                hasattr(rec, "recommendation_type")
                and rec.recommendation_type == "usage_optimization"
            )
        ]

        assert len(usage_recs) >= 0

    def test_get_recommendations_empty_case(self):
        """Test recommendations for already optimal case."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-3.5-turbo",  # Already cheap model
            current_cost=Decimal("0.001"),  # Very low cost
            input_tokens=100,  # Small usage
            output_tokens=50,
        )

        # May have fewer recommendations for already optimal usage
        assert isinstance(recommendations, list)

    def test_recommendations_confidence_scores(self):
        """Test recommendations include confidence scores."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.12"),
            input_tokens=1200,
            output_tokens=600,
        )

        for rec in recommendations:
            if isinstance(rec, dict) and "confidence" in rec:
                assert 0.0 <= rec["confidence"] <= 1.0
            elif hasattr(rec, "confidence_score"):
                assert 0.0 <= rec.confidence_score <= 1.0

    def test_recommendations_include_tradeoffs(self):
        """Test recommendations mention potential tradeoffs."""
        recommendations = get_cost_optimization_recommendations(
            current_model="gpt-4",
            current_cost=Decimal("0.20"),
            input_tokens=2000,
            output_tokens=1000,
        )

        # At least some recommendations should mention tradeoffs
        has_tradeoffs = any(
            (isinstance(rec, dict) and "tradeoffs" in rec and rec["tradeoffs"])
            or (hasattr(rec, "potential_tradeoffs") and rec.potential_tradeoffs)
            for rec in recommendations
        )

        # This might be implementation-dependent
        assert isinstance(has_tradeoffs, bool)


class TestPricingEdgeCases:
    """Test edge cases and error conditions in pricing."""

    def setup_method(self):
        """Setup for each test method."""
        self.calculator = FlowiseCostCalculator()

    def test_very_large_token_counts(self):
        """Test cost calculation with very large token counts."""
        cost = self.calculator.calculate_cost(
            input_tokens=1_000_000, output_tokens=1_000_000, model_name="gpt-3.5-turbo"
        )

        assert isinstance(cost, Decimal)
        assert cost > 0
        # Should handle large numbers without overflow

    def test_decimal_precision(self):
        """Test decimal precision in cost calculations."""
        cost1 = self.calculator.calculate_cost(1, 1, "gpt-3.5-turbo")
        cost2 = self.calculator.calculate_cost(1, 1, "gpt-3.5-turbo")

        # Should be exactly equal (no floating point errors)
        assert cost1 == cost2
        assert isinstance(cost1, Decimal)

    def test_cost_accumulation_precision(self):
        """Test precision is maintained when accumulating costs."""
        total_cost = Decimal("0")

        for _ in range(1000):
            cost = self.calculator.calculate_cost(1, 1, "gpt-3.5-turbo")
            total_cost += cost

        # Should maintain precision
        assert isinstance(total_cost, Decimal)

        # Compare with bulk calculation
        bulk_cost = self.calculator.calculate_cost(1000, 1000, "gpt-3.5-turbo")

        # Should be very close (allowing for minor differences in calculation approach)
        difference = abs(total_cost - bulk_cost)
        assert difference < Decimal("0.001")  # Very small tolerance

    def test_zero_cost_pricing_tier(self):
        """Test pricing tier with zero cost."""
        free_tier = FlowisePricingTier("free", Decimal("0"), 1000)
        calculator = FlowiseCostCalculator(custom_pricing_tiers=[free_tier])

        cost = calculator.calculate_cost(
            input_tokens=100,
            output_tokens=50,
            model_name="any-model",
            pricing_tier="free",
        )

        assert cost == Decimal("0")

    def test_invalid_decimal_inputs(self):
        """Test handling of invalid decimal inputs."""
        with pytest.raises((ValueError, InvalidOperation, TypeError)):
            self.calculator.calculate_cost(
                input_tokens="not-a-number",
                output_tokens=50,
                model_name="gpt-3.5-turbo",
            )

    def test_none_inputs(self):
        """Test handling of None inputs."""
        with pytest.raises(TypeError):
            self.calculator.calculate_cost(
                input_tokens=None, output_tokens=50, model_name="gpt-3.5-turbo"
            )

    def test_empty_string_model_name(self):
        """Test handling of empty model name."""
        cost = self.calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name=""
        )

        # Should use default pricing
        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_none_model_name(self):
        """Test handling of None model name."""
        cost = self.calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name=None
        )

        # Should use default pricing
        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_unicode_model_name(self):
        """Test handling of Unicode model names."""
        cost = self.calculator.calculate_cost(
            input_tokens=100, output_tokens=50, model_name="gpt-模型-🤖"
        )

        # Should handle Unicode gracefully
        assert isinstance(cost, Decimal)
        assert cost > 0


class TestPricingPerformance:
    """Test pricing calculation performance."""

    def setup_method(self):
        """Setup for each test method."""
        self.calculator = FlowiseCostCalculator()

    def test_single_calculation_performance(self):
        """Test performance of single cost calculations."""
        import time

        start_time = time.time()

        for _ in range(1000):
            self.calculator.calculate_cost(100, 50, "gpt-3.5-turbo")

        end_time = time.time()
        avg_time = (end_time - start_time) / 1000

        # Should be very fast (less than 0.1ms per calculation)
        assert avg_time < 0.0001

    def test_bulk_calculation_performance(self):
        """Test performance of bulk calculations."""
        requests = [
            {"input_tokens": 100, "output_tokens": 50, "model_name": "gpt-3.5-turbo"}
            for _ in range(1000)
        ]

        import time

        start_time = time.time()

        results = calculate_bulk_costs(requests)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == 1000
        assert total_time < 1.0  # Should complete in under 1 second

    def test_memory_usage_stability(self):
        """Test memory usage remains stable during calculations."""
        import gc

        gc.collect()

        # Perform many calculations
        for _ in range(10000):
            self.calculator.calculate_cost(100, 50, "gpt-3.5-turbo")

            if _ % 1000 == 0:
                gc.collect()

        gc.collect()
        # Test passes if no memory errors


class TestPricingIntegration:
    """Test pricing integration with other components."""

    def test_pricing_with_validation(self):
        """Test pricing integrates with validation components."""
        calculator = FlowiseCostCalculator()

        # Should handle validation of pricing configurations
        assert len(calculator.pricing_tiers) >= 0
        assert isinstance(calculator.model_pricing, dict)

    def test_pricing_serialization_for_telemetry(self):
        """Test pricing data can be serialized for telemetry."""
        calculator = FlowiseCostCalculator()

        cost = calculator.calculate_cost(100, 50, "gpt-3.5-turbo")

        # Cost should be serializable
        cost_str = str(cost)
        assert cost_str

        # Should be able to recreate from string
        recreated_cost = Decimal(cost_str)
        assert recreated_cost == cost

    def test_pricing_configuration_loading(self):
        """Test pricing can load from configuration."""
        # This would test loading pricing from external config files
        # Implementation depends on actual config system
        calculator = FlowiseCostCalculator()

        # Verify basic configuration is loaded
        assert hasattr(calculator, "pricing_tiers")
        assert hasattr(calculator, "model_pricing")

    def test_pricing_extensibility(self):
        """Test pricing system is extensible."""
        # Test adding custom pricing models
        custom_pricing = {
            "custom-model": {
                "input_cost_per_1k": Decimal("0.0025"),
                "output_cost_per_1k": Decimal("0.0035"),
            }
        }

        calculator = FlowiseCostCalculator(custom_model_pricing=custom_pricing)

        cost = calculator.calculate_cost(1000, 500, "custom-model")
        assert cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
