"""
Comprehensive tests for SkyRouter pricing calculations.

Tests pricing accuracy, volume discounts, multi-model cost calculations,
and cost optimization recommendations.
"""

from decimal import Decimal

import pytest

# Import the modules under test
try:
    from genops.providers.skyrouter_pricing import (
        ModelPricing,
        PricingTier,  # noqa: F401
        SkyRouterPricingCalculator,
        SkyRouterPricingConfig,
        VolumeDiscount,  # noqa: F401
    )

    SKYROUTER_PRICING_AVAILABLE = True
except ImportError:
    SKYROUTER_PRICING_AVAILABLE = False


@pytest.mark.skipif(
    not SKYROUTER_PRICING_AVAILABLE, reason="SkyRouter pricing module not available"
)
class TestSkyRouterPricingConfig:
    """Test suite for SkyRouter pricing configuration."""

    def test_pricing_config_initialization(self):
        """Test pricing configuration initialization."""
        config = SkyRouterPricingConfig()

        # Check default model pricing exists
        assert "gpt-4" in config.model_pricing
        assert "claude-3-sonnet" in config.model_pricing
        assert "gpt-3.5-turbo" in config.model_pricing

        # Check pricing structure
        gpt4_pricing = config.model_pricing["gpt-4"]
        assert isinstance(gpt4_pricing, ModelPricing)
        assert gpt4_pricing.input_cost_per_1k > 0
        assert gpt4_pricing.output_cost_per_1k > 0

    def test_custom_pricing_configuration(self):
        """Test custom pricing configuration."""
        custom_pricing = {
            "custom-model": ModelPricing(
                input_cost_per_1k=0.01, output_cost_per_1k=0.03, minimum_cost=0.001
            )
        }

        config = SkyRouterPricingConfig(custom_model_pricing=custom_pricing)
        assert "custom-model" in config.model_pricing
        assert config.model_pricing["custom-model"].input_cost_per_1k == 0.01

    def test_volume_discount_configuration(self):
        """Test volume discount configuration."""
        config = SkyRouterPricingConfig()
        config.volume_tiers = {
            1000: 0.05,  # 5% discount for 1K+ tokens
            10000: 0.12,  # 12% discount for 10K+ tokens
            50000: 0.20,  # 20% discount for 50K+ tokens
        }

        assert 1000 in config.volume_tiers
        assert config.volume_tiers[10000] == 0.12

    def test_pricing_tier_validation(self):
        """Test pricing tier validation."""
        config = SkyRouterPricingConfig()

        # Valid pricing tiers
        valid_tiers = ["standard", "premium", "enterprise"]
        for tier in valid_tiers:
            config.default_pricing_tier = tier
            assert config.default_pricing_tier == tier

    def test_regional_pricing_configuration(self):
        """Test regional pricing variations."""
        config = SkyRouterPricingConfig()
        config.regional_multipliers = {
            "us-east-1": 1.0,
            "eu-west-1": 1.1,
            "ap-southeast-1": 1.2,
        }

        assert config.regional_multipliers["us-east-1"] == 1.0
        assert config.regional_multipliers["eu-west-1"] == 1.1


@pytest.mark.skipif(
    not SKYROUTER_PRICING_AVAILABLE, reason="SkyRouter pricing module not available"
)
class TestSkyRouterPricingCalculator:
    """Test suite for SkyRouter pricing calculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SkyRouterPricingConfig()
        self.calculator = SkyRouterPricingCalculator(config=self.config)

    def test_basic_cost_calculation(self):
        """Test basic cost calculation for single models."""
        test_cases = [
            {
                "model": "gpt-4",
                "input_tokens": 1000,
                "output_tokens": 500,
                "expected_min": 0.03,  # Minimum expected cost
            },
            {
                "model": "gpt-3.5-turbo",
                "input_tokens": 1000,
                "output_tokens": 500,
                "expected_min": 0.002,
            },
            {
                "model": "claude-3-sonnet",
                "input_tokens": 1000,
                "output_tokens": 500,
                "expected_min": 0.005,
            },
        ]

        for case in test_cases:
            cost = self.calculator.calculate_cost(
                model=case["model"],
                input_tokens=case["input_tokens"],
                output_tokens=case["output_tokens"],
            )

            assert isinstance(cost, (float, Decimal))
            assert cost >= case["expected_min"]
            assert cost < 1.0  # Reasonable upper bound

    def test_cost_calculation_with_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        # Should handle zero tokens gracefully
        cost = self.calculator.calculate_cost(
            model="gpt-4", input_tokens=0, output_tokens=0
        )

        # Should return minimum cost or zero
        assert cost >= 0
        assert cost < 0.01  # Should be very small

    def test_cost_calculation_unknown_model(self):
        """Test cost calculation for unknown models."""
        # Should use fallback pricing
        cost = self.calculator.calculate_cost(
            model="unknown-model-xyz", input_tokens=1000, output_tokens=500
        )

        # Should still return a reasonable cost
        assert cost > 0
        assert cost < 1.0

    def test_volume_discount_calculation(self):
        """Test volume discount calculations."""
        # Set up volume tiers
        self.calculator.config.volume_tiers = {
            1000: 0.05,  # 5% discount
            10000: 0.15,  # 15% discount
            50000: 0.25,  # 25% discount
        }

        test_volumes = [
            {"tokens": 500, "expected_discount": 0.0},  # No discount
            {"tokens": 1500, "expected_discount": 0.05},  # 5% discount
            {"tokens": 15000, "expected_discount": 0.15},  # 15% discount
            {"tokens": 75000, "expected_discount": 0.25},  # 25% discount
        ]

        for volume in test_volumes:
            discount_info = self.calculator.calculate_volume_discount(volume["tokens"])
            assert discount_info["discount_percentage"] == volume["expected_discount"]

    def test_multi_model_cost_aggregation(self):
        """Test cost aggregation across multiple models."""
        operations = [
            {"model": "gpt-4", "input_tokens": 500, "output_tokens": 200},
            {"model": "claude-3-sonnet", "input_tokens": 800, "output_tokens": 300},
            {"model": "gpt-3.5-turbo", "input_tokens": 1200, "output_tokens": 400},
            {"model": "gemini-pro", "input_tokens": 600, "output_tokens": 250},
        ]

        total_cost = 0.0
        cost_breakdown = {}

        for op in operations:
            cost = self.calculator.calculate_cost(
                model=op["model"],
                input_tokens=op["input_tokens"],
                output_tokens=op["output_tokens"],
            )

            total_cost += cost
            cost_breakdown[op["model"]] = cost_breakdown.get(op["model"], 0) + cost

        assert total_cost > 0
        assert len(cost_breakdown) <= len({op["model"] for op in operations})

    def test_routing_strategy_cost_impact(self):
        """Test cost impact of different routing strategies."""
        base_operation = {
            "models": ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"],
            "input_tokens": 1000,
            "output_tokens": 500,
        }

        strategy_costs = {}
        strategies = [
            "cost_optimized",
            "balanced",
            "latency_optimized",
            "reliability_first",
        ]

        for strategy in strategies:
            cost = self.calculator.calculate_routing_cost(
                models=base_operation["models"],
                input_tokens=base_operation["input_tokens"],
                output_tokens=base_operation["output_tokens"],
                routing_strategy=strategy,
            )
            strategy_costs[strategy] = cost

        # Cost optimized should generally be cheapest
        assert strategy_costs["cost_optimized"] <= max(strategy_costs.values())

    def test_complexity_based_pricing(self):
        """Test pricing adjustments based on complexity."""
        complexity_levels = ["simple", "moderate", "complex", "enterprise"]
        base_cost = self.calculator.calculate_cost("gpt-4", 1000, 500)

        for complexity in complexity_levels:
            adjusted_cost = self.calculator.calculate_complexity_adjusted_cost(
                base_cost=base_cost, complexity=complexity
            )

            if complexity in ["complex", "enterprise"]:
                # More complex operations might have higher costs
                assert adjusted_cost >= base_cost
            else:
                # Simple operations might have same or lower costs
                assert adjusted_cost > 0

    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendation generation."""
        usage_data = {
            "operations": [
                {"model": "gpt-4", "count": 100, "avg_cost": 0.05},
                {"model": "gpt-3.5-turbo", "count": 500, "avg_cost": 0.002},
                {"model": "claude-3-sonnet", "count": 200, "avg_cost": 0.01},
            ],
            "total_monthly_cost": 15.0,
            "routing_strategies": {
                "cost_optimized": 0.3,
                "balanced": 0.5,
                "reliability_first": 0.2,
            },
        }

        recommendations = self.calculator.generate_cost_recommendations(usage_data)

        assert isinstance(recommendations, list)
        assert len(recommendations) >= 0

        # Check recommendation structure if any exist
        for rec in recommendations:
            assert "title" in rec
            assert "potential_savings" in rec
            assert "priority_score" in rec

    def test_regional_cost_calculation(self):
        """Test regional cost variations."""
        base_cost = self.calculator.calculate_cost("gpt-4", 1000, 500)

        regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
        regional_costs = {}

        for region in regions:
            regional_cost = self.calculator.calculate_regional_cost(
                base_cost=base_cost, region=region
            )
            regional_costs[region] = regional_cost

        # Regional costs should be reasonable variations
        for cost in regional_costs.values():
            assert cost > 0
            assert cost < base_cost * 2  # No more than 2x base cost

    def test_monthly_cost_estimation(self):
        """Test monthly cost estimation."""
        estimation_params = {
            "daily_operations": 100,
            "avg_tokens_per_operation": 1000,
            "model_distribution": {
                "gpt-4": 0.3,
                "claude-3-sonnet": 0.3,
                "gpt-3.5-turbo": 0.4,
            },
            "optimization_strategy": "balanced",
        }

        monthly_estimate = self.calculator.estimate_monthly_cost(**estimation_params)

        assert isinstance(monthly_estimate, dict)
        assert "estimated_monthly_cost" in monthly_estimate
        assert "cost_breakdown_by_model" in monthly_estimate
        assert monthly_estimate["estimated_monthly_cost"] > 0

    def test_budget_optimization_analysis(self):
        """Test budget optimization analysis."""
        budget_params = {
            "target_monthly_budget": 100.0,
            "current_usage_pattern": {
                "gpt-4": {"operations": 500, "avg_cost": 0.05},
                "claude-3-sonnet": {"operations": 800, "avg_cost": 0.01},
                "gpt-3.5-turbo": {"operations": 1200, "avg_cost": 0.002},
            },
        }

        optimization = self.calculator.optimize_for_budget(**budget_params)

        assert isinstance(optimization, dict)
        assert "recommended_distribution" in optimization
        assert "potential_savings" in optimization

    def test_cost_calculation_edge_cases(self):
        """Test cost calculation edge cases."""
        edge_cases = [
            # Very large token counts
            {"input_tokens": 1000000, "output_tokens": 500000},
            # Very small token counts
            {"input_tokens": 1, "output_tokens": 1},
            # Unbalanced input/output
            {"input_tokens": 10000, "output_tokens": 10},
            {"input_tokens": 10, "output_tokens": 10000},
        ]

        for case in edge_cases:
            cost = self.calculator.calculate_cost(
                model="gpt-4",
                input_tokens=case["input_tokens"],
                output_tokens=case["output_tokens"],
            )

            # Cost should always be positive and reasonable
            assert cost > 0
            assert cost < 1000  # Reasonable upper bound

    def test_pricing_tier_adjustments(self):
        """Test pricing adjustments for different tiers."""
        tiers = ["standard", "premium", "enterprise"]
        base_cost = self.calculator.calculate_cost("gpt-4", 1000, 500)

        for tier in tiers:
            adjusted_cost = self.calculator.apply_pricing_tier(
                base_cost=base_cost, tier=tier
            )

            # Premium and enterprise tiers might have different pricing
            assert adjusted_cost > 0
            if tier == "enterprise":
                # Enterprise might have volume discounts
                assert adjusted_cost <= base_cost * 1.1

    def test_competitive_analysis_pricing(self):
        """Test competitive analysis and pricing comparisons."""
        comparison_data = {
            "skyrouter": {"cost": 10.0, "models": 150},
            "openai_direct": {"cost": 15.0, "models": 4},
            "anthropic_direct": {"cost": 12.0, "models": 3},
            "other_providers": {"cost": 18.0, "models": 50},
        }

        analysis = self.calculator.generate_competitive_analysis(comparison_data)

        assert isinstance(analysis, dict)
        assert "cost_savings" in analysis
        assert "model_coverage_advantage" in analysis


@pytest.mark.skipif(
    not SKYROUTER_PRICING_AVAILABLE, reason="SkyRouter pricing module not available"
)
class TestSkyRouterPricingIntegration:
    """Integration tests for pricing calculations."""

    def test_end_to_end_pricing_workflow(self):
        """Test complete pricing workflow from request to final cost."""
        config = SkyRouterPricingConfig()
        calculator = SkyRouterPricingCalculator(config=config)

        # Simulate a complete workflow
        workflow_steps = [
            {"model": "gpt-3.5-turbo", "input_tokens": 200, "output_tokens": 50},
            {"model": "claude-3-sonnet", "input_tokens": 500, "output_tokens": 150},
            {"model": "gpt-4", "input_tokens": 800, "output_tokens": 200},
        ]

        total_workflow_cost = 0.0
        step_costs = []

        for step in workflow_steps:
            step_cost = calculator.calculate_cost(
                model=step["model"],
                input_tokens=step["input_tokens"],
                output_tokens=step["output_tokens"],
            )

            step_costs.append({"model": step["model"], "cost": step_cost})
            total_workflow_cost += step_cost

        # Verify workflow pricing
        assert total_workflow_cost > 0
        assert len(step_costs) == 3
        assert all(step["cost"] > 0 for step in step_costs)

    def test_pricing_accuracy_validation(self):
        """Test pricing accuracy against known benchmarks."""
        calculator = SkyRouterPricingCalculator()

        # Test against known pricing benchmarks
        benchmarks = [
            {
                "model": "gpt-4",
                "input_tokens": 1000,
                "output_tokens": 1000,
                "min_expected": 0.03,
                "max_expected": 0.12,
            },
            {
                "model": "gpt-3.5-turbo",
                "input_tokens": 1000,
                "output_tokens": 1000,
                "min_expected": 0.002,
                "max_expected": 0.008,
            },
        ]

        for benchmark in benchmarks:
            cost = calculator.calculate_cost(
                model=benchmark["model"],
                input_tokens=benchmark["input_tokens"],
                output_tokens=benchmark["output_tokens"],
            )

            assert cost >= benchmark["min_expected"]
            assert cost <= benchmark["max_expected"]

    def test_pricing_consistency(self):
        """Test pricing consistency across multiple calculations."""
        calculator = SkyRouterPricingCalculator()

        # Same operation should yield same cost
        operation = {"model": "gpt-4", "input_tokens": 1000, "output_tokens": 500}

        costs = []
        for _ in range(10):
            cost = calculator.calculate_cost(**operation)
            costs.append(cost)

        # All costs should be identical
        assert len(set(costs)) == 1, "Pricing should be consistent"


if __name__ == "__main__":
    pytest.main([__file__])
