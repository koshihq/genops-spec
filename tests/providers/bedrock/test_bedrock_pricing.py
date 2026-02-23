"""
Comprehensive tests for GenOps Bedrock Pricing Engine.

Tests the intelligent pricing functionality including:
- Cost calculations across all supported models
- Regional pricing differences
- Multi-model cost comparisons
- Optimization recommendations
- Edge cases and error handling
"""

import pytest

# Import the modules under test
try:
    from genops.providers.bedrock_pricing import (
        BEDROCK_MODELS,
        BedrockCostBreakdown,
        ModelComparison,
        OptimizationRecommendations,
        calculate_bedrock_cost,
        calculate_regional_costs,
        compare_bedrock_models,
        get_cost_optimization_recommendations,
    )

    PRICING_AVAILABLE = True
except ImportError:
    PRICING_AVAILABLE = False


@pytest.mark.skipif(
    not PRICING_AVAILABLE, reason="Bedrock pricing module not available"
)
class TestBedrockCostCalculation:
    """Test cost calculation functionality."""

    def test_claude_model_pricing(self):
        """Test cost calculation for Anthropic Claude models."""
        # Test Claude 3 Haiku (most cost-effective)
        cost = calculate_bedrock_cost(
            input_tokens=1000,
            output_tokens=500,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        assert isinstance(cost, BedrockCostBreakdown)
        assert cost.total_cost > 0
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total_cost == cost.input_cost + cost.output_cost
        assert cost.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
        assert cost.region == "us-east-1"
        assert cost.currency == "USD"

    def test_titan_model_pricing(self):
        """Test cost calculation for Amazon Titan models."""
        cost = calculate_bedrock_cost(
            input_tokens=2000,
            output_tokens=1000,
            model_id="amazon.titan-text-express-v1",
            region="us-east-1",
        )

        assert cost.total_cost > 0
        assert cost.model_id == "amazon.titan-text-express-v1"
        assert cost.input_tokens == 2000
        assert cost.output_tokens == 1000

    def test_ai21_model_pricing(self):
        """Test cost calculation for AI21 Jurassic models."""
        cost = calculate_bedrock_cost(
            input_tokens=1500,
            output_tokens=750,
            model_id="ai21.j2-ultra-v1",
            region="us-east-1",
        )

        assert cost.total_cost > 0
        assert cost.model_id == "ai21.j2-ultra-v1"

    def test_cohere_model_pricing(self):
        """Test cost calculation for Cohere Command models."""
        cost = calculate_bedrock_cost(
            input_tokens=800,
            output_tokens=400,
            model_id="cohere.command-text-v14",
            region="us-east-1",
        )

        assert cost.total_cost > 0
        assert cost.model_id == "cohere.command-text-v14"

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_bedrock_cost(
            input_tokens=0,
            output_tokens=0,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        assert cost.total_cost == 0
        assert cost.input_cost == 0
        assert cost.output_cost == 0

    def test_input_only_tokens(self):
        """Test cost calculation with only input tokens."""
        cost = calculate_bedrock_cost(
            input_tokens=1000,
            output_tokens=0,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        assert cost.input_cost > 0
        assert cost.output_cost == 0
        assert cost.total_cost == cost.input_cost

    def test_output_only_tokens(self):
        """Test cost calculation with only output tokens."""
        cost = calculate_bedrock_cost(
            input_tokens=0,
            output_tokens=500,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        assert cost.input_cost == 0
        assert cost.output_cost > 0
        assert cost.total_cost == cost.output_cost

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        cost = calculate_bedrock_cost(
            input_tokens=100000,
            output_tokens=50000,
            model_id="anthropic.claude-3-opus-20240229-v1:0",
            region="us-east-1",
        )

        assert cost.total_cost > 1.0  # Should be substantial cost
        assert cost.input_cost > cost.output_cost  # Opus has higher input cost

    def test_cost_precision(self):
        """Test that cost calculations maintain appropriate precision."""
        cost = calculate_bedrock_cost(
            input_tokens=1,
            output_tokens=1,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        # Should have at least 6 decimal places precision
        assert len(str(cost.total_cost).split(".")[-1]) >= 6 or cost.total_cost == 0

    def test_invalid_model_id(self):
        """Test handling of invalid model IDs."""
        with pytest.raises((ValueError, KeyError)):
            calculate_bedrock_cost(
                input_tokens=100,
                output_tokens=50,
                model_id="invalid-model-id-12345",
                region="us-east-1",
            )

    def test_invalid_region(self):
        """Test handling of invalid regions."""
        # Should either handle gracefully or raise appropriate error
        try:
            cost = calculate_bedrock_cost(
                input_tokens=100,
                output_tokens=50,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                region="invalid-region",
            )
            # If no error, should still return valid cost structure
            assert isinstance(cost, BedrockCostBreakdown)
        except ValueError:
            # Expected for invalid regions
            pass

    def test_negative_tokens(self):
        """Test handling of negative token counts."""
        with pytest.raises((ValueError, AssertionError)):
            calculate_bedrock_cost(
                input_tokens=-100,
                output_tokens=50,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                region="us-east-1",
            )


@pytest.mark.skipif(
    not PRICING_AVAILABLE, reason="Bedrock pricing module not available"
)
class TestModelComparison:
    """Test multi-model comparison functionality."""

    def test_basic_model_comparison(self):
        """Test basic comparison between different models."""
        models_to_compare = [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "amazon.titan-text-express-v1",
        ]

        comparison = compare_bedrock_models(
            prompt="Test prompt for comparison",
            models=models_to_compare,
            region="us-east-1",
            expected_output_tokens=200,
        )

        assert isinstance(comparison, ModelComparison)
        assert len(comparison.model_comparisons) == 3
        assert comparison.best_for_cost in models_to_compare
        assert comparison.best_for_speed in models_to_compare

        # Verify each model comparison has required fields
        for model_comp in comparison.model_comparisons:
            assert model_comp.model_id in models_to_compare
            assert model_comp.estimated_cost > 0
            assert model_comp.estimated_latency_ms > 0
            assert 0 <= model_comp.quality_score <= 10

    def test_single_model_comparison(self):
        """Test comparison with a single model."""
        comparison = compare_bedrock_models(
            prompt="Single model test",
            models=["anthropic.claude-3-haiku-20240307-v1:0"],
            region="us-east-1",
            expected_output_tokens=100,
        )

        assert len(comparison.model_comparisons) == 1
        assert comparison.best_for_cost == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_cost_ordering(self):
        """Test that models are correctly ordered by cost."""
        models_to_compare = [
            "anthropic.claude-3-opus-20240229-v1:0",  # Most expensive
            "anthropic.claude-3-sonnet-20240229-v1:0",  # Medium
            "anthropic.claude-3-haiku-20240307-v1:0",  # Least expensive
        ]

        comparison = compare_bedrock_models(
            prompt="Cost ordering test",
            models=models_to_compare,
            region="us-east-1",
            expected_output_tokens=300,
        )

        # Find costs for each model
        costs = {}
        for model_comp in comparison.model_comparisons:
            costs[model_comp.model_id] = model_comp.estimated_cost

        # Verify cost ordering (Opus > Sonnet > Haiku)
        assert (
            costs["anthropic.claude-3-opus-20240229-v1:0"]
            > costs["anthropic.claude-3-sonnet-20240229-v1:0"]
        )
        assert (
            costs["anthropic.claude-3-sonnet-20240229-v1:0"]
            > costs["anthropic.claude-3-haiku-20240307-v1:0"]
        )

    def test_quality_scoring(self):
        """Test that quality scores are reasonable."""
        comparison = compare_bedrock_models(
            prompt="Quality test prompt",
            models=[
                "anthropic.claude-3-opus-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
            ],
            region="us-east-1",
            expected_output_tokens=150,
        )

        opus_score = next(
            mc.quality_score
            for mc in comparison.model_comparisons
            if "opus" in mc.model_id
        )
        haiku_score = next(
            mc.quality_score
            for mc in comparison.model_comparisons
            if "haiku" in mc.model_id
        )

        # Opus should have higher quality score than Haiku
        assert opus_score > haiku_score

    def test_empty_model_list(self):
        """Test handling of empty model list."""
        with pytest.raises((ValueError, IndexError)):
            compare_bedrock_models(
                prompt="Empty list test",
                models=[],
                region="us-east-1",
                expected_output_tokens=100,
            )

    def test_duplicate_models(self):
        """Test handling of duplicate models in list."""
        comparison = compare_bedrock_models(
            prompt="Duplicate test",
            models=[
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",  # Duplicate
            ],
            region="us-east-1",
            expected_output_tokens=100,
        )

        # Should handle duplicates gracefully (either dedupe or allow)
        assert len(comparison.model_comparisons) >= 1


@pytest.mark.skipif(
    not PRICING_AVAILABLE, reason="Bedrock pricing module not available"
)
class TestOptimizationRecommendations:
    """Test cost optimization recommendation functionality."""

    def test_basic_optimization_recommendations(self):
        """Test basic optimization recommendations."""
        recommendations = get_cost_optimization_recommendations(
            prompt="Analyze this financial document for key metrics and trends",
            budget_constraint=0.01,  # $0.01 maximum
            quality_requirement="medium",
            region="us-east-1",
        )

        assert isinstance(recommendations, OptimizationRecommendations)
        assert recommendations.recommended_model is not None
        assert recommendations.estimated_cost <= 0.01  # Within budget
        assert recommendations.estimated_latency_ms > 0
        assert len(recommendations.reasoning) > 0

    def test_high_quality_requirement(self):
        """Test recommendations for high quality requirements."""
        recommendations = get_cost_optimization_recommendations(
            prompt="Perform detailed legal analysis of this contract",
            budget_constraint=0.10,  # Higher budget
            quality_requirement="high",
            region="us-east-1",
        )

        # Should recommend a high-quality model (likely Sonnet or Opus)
        assert (
            "sonnet" in recommendations.recommended_model.lower()
            or "opus" in recommendations.recommended_model.lower()
        )

    def test_low_budget_constraint(self):
        """Test recommendations for very low budget."""
        recommendations = get_cost_optimization_recommendations(
            prompt="Quick classification of this text",
            budget_constraint=0.001,  # Very low budget
            quality_requirement="low",
            region="us-east-1",
        )

        # Should recommend the most cost-effective model
        assert (
            "haiku" in recommendations.recommended_model.lower()
            or "titan" in recommendations.recommended_model.lower()
        )

    def test_no_budget_constraint(self):
        """Test recommendations without budget constraints."""
        recommendations = get_cost_optimization_recommendations(
            prompt="Complex analysis task",
            budget_constraint=None,
            quality_requirement="premium",
            region="us-east-1",
        )

        # Should recommend the highest quality model
        assert recommendations.recommended_model is not None
        assert recommendations.estimated_cost > 0

    def test_different_quality_levels(self):
        """Test different quality requirement levels."""
        quality_levels = ["low", "medium", "high", "premium"]

        for quality in quality_levels:
            recommendations = get_cost_optimization_recommendations(
                prompt="Standard analysis task",
                budget_constraint=0.05,
                quality_requirement=quality,
                region="us-east-1",
            )

            assert recommendations.recommended_model is not None
            assert recommendations.quality_score >= 0

    def test_complex_prompt_analysis(self):
        """Test that complex prompts get appropriate recommendations."""
        complex_prompt = """
        Please analyze this comprehensive financial report including:
        1. Revenue analysis across multiple segments
        2. Profitability trends and margin analysis
        3. Cash flow statement evaluation
        4. Risk assessment and recommendations
        5. Competitive positioning analysis
        """

        recommendations = get_cost_optimization_recommendations(
            prompt=complex_prompt,
            budget_constraint=0.05,
            quality_requirement="high",
            region="us-east-1",
        )

        # Complex tasks should prefer higher-quality models
        assert (
            "haiku" not in recommendations.recommended_model.lower()
        )  # Should use better than cheapest

    def test_simple_prompt_analysis(self):
        """Test that simple prompts get cost-effective recommendations."""
        simple_prompt = "Classify: positive or negative sentiment"

        recommendations = get_cost_optimization_recommendations(
            prompt=simple_prompt,
            budget_constraint=0.01,
            quality_requirement="medium",
            region="us-east-1",
        )

        # Simple tasks should be cost-effective
        assert recommendations.estimated_cost <= 0.01


@pytest.mark.skipif(
    not PRICING_AVAILABLE, reason="Bedrock pricing module not available"
)
class TestRegionalPricing:
    """Test regional pricing functionality."""

    def test_regional_cost_calculation(self):
        """Test cost calculation across different regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1"]

        regional_costs = calculate_regional_costs(
            prompt="Regional pricing test",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            regions=regions,
            expected_output_tokens=200,
        )

        assert len(regional_costs) == 3
        for cost_info in regional_costs:
            assert cost_info.region in regions
            assert cost_info.total_cost >= 0
            assert isinstance(cost_info.model_available, bool)

    def test_single_region_calculation(self):
        """Test calculation for a single region."""
        regional_costs = calculate_regional_costs(
            prompt="Single region test",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            regions=["us-east-1"],
            expected_output_tokens=100,
        )

        assert len(regional_costs) == 1
        assert regional_costs[0].region == "us-east-1"

    def test_model_availability_by_region(self):
        """Test that model availability varies by region."""
        # Test with a model that might not be available in all regions
        regional_costs = calculate_regional_costs(
            prompt="Availability test",
            model_id="anthropic.claude-3-opus-20240229-v1:0",
            regions=["us-east-1", "ap-south-1"],  # Some regions may not have all models
            expected_output_tokens=150,
        )

        # Should return results for all requested regions
        assert len(regional_costs) == 2

        # Check that availability is properly reported
        for cost_info in regional_costs:
            assert hasattr(cost_info, "model_available")

    def test_cost_differences_by_region(self):
        """Test that costs may differ by region."""
        regional_costs = calculate_regional_costs(
            prompt="Cost difference test",
            model_id="amazon.titan-text-express-v1",
            regions=["us-east-1", "eu-west-1"],
            expected_output_tokens=200,
        )

        us_cost = next(c.total_cost for c in regional_costs if c.region == "us-east-1")
        eu_cost = next(c.total_cost for c in regional_costs if c.region == "eu-west-1")

        # Costs might be the same or different - both are valid
        assert us_cost >= 0
        assert eu_cost >= 0


@pytest.mark.skipif(
    not PRICING_AVAILABLE, reason="Bedrock pricing module not available"
)
class TestPricingDataStructure:
    """Test the pricing data structures and models catalog."""

    def test_bedrock_models_catalog(self):
        """Test that the BEDROCK_MODELS catalog is properly structured."""
        assert isinstance(BEDROCK_MODELS, dict)
        assert len(BEDROCK_MODELS) > 0

        # Check that key models are present
        expected_models = [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "amazon.titan-text-express-v1",
        ]

        for model in expected_models:
            assert model in BEDROCK_MODELS

    def test_model_pricing_structure(self):
        """Test that each model has proper pricing structure."""
        for _model_id, model_info in BEDROCK_MODELS.items():
            assert "provider" in model_info
            assert "input_price_per_1k" in model_info
            assert "output_price_per_1k" in model_info

            # Prices should be positive numbers
            assert model_info["input_price_per_1k"] >= 0
            assert model_info["output_price_per_1k"] >= 0

            # Should have quality and speed estimates
            if "quality_score" in model_info:
                assert 0 <= model_info["quality_score"] <= 10
            if "speed_score" in model_info:
                assert 0 <= model_info["speed_score"] <= 10

    def test_provider_consistency(self):
        """Test that models are properly categorized by provider."""
        anthropic_models = [
            k for k, v in BEDROCK_MODELS.items() if v["provider"] == "anthropic"
        ]
        amazon_models = [
            k for k, v in BEDROCK_MODELS.items() if v["provider"] == "amazon"
        ]

        # Anthropic models should have 'anthropic' in their ID
        for model in anthropic_models:
            assert "anthropic" in model

        # Amazon models should have 'amazon' in their ID
        for model in amazon_models:
            assert "amazon" in model

    def test_cost_breakdown_structure(self):
        """Test BedrockCostBreakdown data structure."""
        # This tests the structure without making actual API calls
        expected_fields = [
            "total_cost",
            "input_cost",
            "output_cost",
            "input_tokens",
            "output_tokens",
            "model_id",
            "region",
            "currency",
            "cost_per_1k_input",
            "cost_per_1k_output",
        ]

        # Test with a sample calculation
        cost = calculate_bedrock_cost(
            input_tokens=100,
            output_tokens=50,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        for field in expected_fields:
            assert hasattr(cost, field)


@pytest.mark.skipif(
    not PRICING_AVAILABLE, reason="Bedrock pricing module not available"
)
class TestEdgeCases:
    """Test edge cases and boundary conditions for pricing."""

    def test_maximum_token_limits(self):
        """Test handling of maximum token limits."""
        # Test with very large token counts
        cost = calculate_bedrock_cost(
            input_tokens=200000,  # Very large input
            output_tokens=4096,  # Maximum typical output
            model_id="anthropic.claude-3-opus-20240229-v1:0",
            region="us-east-1",
        )

        assert cost.total_cost > 0
        assert cost.input_tokens == 200000
        assert cost.output_tokens == 4096

    def test_fractional_calculations(self):
        """Test that fractional token costs are handled correctly."""
        # Test with token counts that result in fractional costs
        cost = calculate_bedrock_cost(
            input_tokens=1,
            output_tokens=1,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        # Even tiny costs should be calculated precisely
        assert cost.total_cost > 0
        assert cost.input_cost > 0
        assert cost.output_cost > 0

    def test_all_supported_models(self):
        """Test cost calculation for all models in the catalog."""
        for model_id in BEDROCK_MODELS.keys():
            cost = calculate_bedrock_cost(
                input_tokens=100,
                output_tokens=50,
                model_id=model_id,
                region="us-east-1",
            )

            assert cost.total_cost > 0
            assert cost.model_id == model_id

    def test_pricing_consistency_across_calls(self):
        """Test that pricing calculations are consistent across multiple calls."""
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        input_tokens = 1000
        output_tokens = 500

        # Make multiple calls with same parameters
        costs = []
        for _ in range(5):
            cost = calculate_bedrock_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_id=model_id,
                region="us-east-1",
            )
            costs.append(cost.total_cost)

        # All costs should be identical
        assert all(c == costs[0] for c in costs)

    def test_currency_consistency(self):
        """Test that all costs are returned in consistent currency."""
        cost = calculate_bedrock_cost(
            input_tokens=100,
            output_tokens=50,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        assert cost.currency == "USD"

    def test_zero_cost_scenarios(self):
        """Test scenarios that should result in zero cost."""
        cost = calculate_bedrock_cost(
            input_tokens=0,
            output_tokens=0,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )

        assert cost.total_cost == 0.0
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0


@pytest.mark.performance
class TestPerformance:
    """Performance tests for pricing calculations."""

    def test_calculation_performance(self):
        """Test that pricing calculations are fast."""
        import time

        if not PRICING_AVAILABLE:
            pytest.skip("Bedrock pricing module not available")

        start_time = time.time()

        # Perform many calculations
        for _ in range(1000):
            calculate_bedrock_cost(
                input_tokens=100,
                output_tokens=50,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                region="us-east-1",
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 1000 calculations in under 1 second
        assert total_time < 1.0

    def test_comparison_performance(self):
        """Test that model comparisons are reasonably fast."""
        import time

        if not PRICING_AVAILABLE:
            pytest.skip("Bedrock pricing module not available")

        start_time = time.time()

        models_to_compare = list(BEDROCK_MODELS.keys())[:5]  # First 5 models

        compare_bedrock_models(
            prompt="Performance test prompt",
            models=models_to_compare,
            region="us-east-1",
            expected_output_tokens=200,
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete comparison in under 1 second
        assert total_time < 1.0
