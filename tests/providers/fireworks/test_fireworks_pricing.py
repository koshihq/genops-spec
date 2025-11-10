"""
Comprehensive tests for Fireworks AI pricing calculator.

Tests cover:
- Cost estimation across all pricing tiers ($0.10-$3.00 per 1M tokens)
- Model recommendations based on task complexity and budget
- Batch processing cost optimization (50% savings)
- Multi-model cost comparisons
- Cost analysis and projections
- Parameter-differentiated pricing (input/output rates)
"""

from decimal import Decimal

import pytest

from genops.providers.fireworks_pricing import (
    FireworksPricingCalculator,
    ModelRecommendation,
)


class TestFireworksPricingCalculator:
    """Test pricing calculator initialization and basic operations."""

    def test_pricing_calculator_initialization(self):
        """Test pricing calculator initialization."""
        calc = FireworksPricingCalculator()

        assert calc is not None
        assert hasattr(calc, 'model_pricing')
        assert len(calc.model_pricing) > 0

    def test_pricing_data_integrity(self):
        """Test pricing data structure and integrity."""
        calc = FireworksPricingCalculator()

        # Verify all models have required pricing fields
        for model_id, pricing in calc.model_pricing.items():
            assert 'input_price' in pricing
            assert 'output_price' in pricing
            assert 'context_length' in pricing
            assert 'tier' in pricing

            # Verify pricing values are valid
            assert isinstance(pricing['input_price'], Decimal)
            assert isinstance(pricing['output_price'], Decimal)
            assert pricing['input_price'] >= 0
            assert pricing['output_price'] >= 0
            assert pricing['context_length'] > 0

    def test_pricing_tiers_coverage(self):
        """Test that all pricing tiers are properly covered."""
        calc = FireworksPricingCalculator()

        expected_tiers = ['tiny', 'small', 'medium', 'large', 'premium', 'embedding']
        found_tiers = set()

        for pricing in calc.model_pricing.values():
            found_tiers.add(pricing['tier'])

        # Should have models across multiple tiers
        assert len(found_tiers) >= 4
        assert 'tiny' in found_tiers
        assert 'small' in found_tiers
        assert 'large' in found_tiers


class TestCostEstimation:
    """Test cost estimation for different operations."""

    def test_chat_cost_estimation_basic(self):
        """Test basic chat completion cost estimation."""
        calc = FireworksPricingCalculator()

        # Test with standard Llama model
        model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        tokens = 1000

        cost = calc.estimate_chat_cost(model, tokens=tokens)

        assert isinstance(cost, Decimal)
        assert cost > 0

        # Expected cost: 1000 tokens * $0.20/1M = $0.0002
        expected_cost = Decimal("0.0002")
        assert abs(cost - expected_cost) < Decimal("0.0001")

    def test_chat_cost_estimation_with_input_output_split(self):
        """Test cost estimation with separate input/output tokens."""
        calc = FireworksPricingCalculator()

        model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        input_tokens = 500
        output_tokens = 300

        cost = calc.estimate_chat_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # Expected cost: (500 + 300) * $0.20/1M = $0.00016
        expected_cost = Decimal("0.00016")
        assert abs(cost - expected_cost) < Decimal("0.00001")

    def test_chat_cost_estimation_batch_discount(self):
        """Test batch processing cost discount."""
        calc = FireworksPricingCalculator()

        model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        tokens = 1000

        standard_cost = calc.estimate_chat_cost(model, tokens=tokens, is_batch=False)
        batch_cost = calc.estimate_chat_cost(model, tokens=tokens, is_batch=True)

        # Batch cost should be 50% of standard
        expected_batch_cost = standard_cost * Decimal("0.5")
        assert abs(batch_cost - expected_batch_cost) < Decimal("0.00001")

    def test_embedding_cost_estimation(self):
        """Test embedding cost estimation."""
        calc = FireworksPricingCalculator()

        model = "accounts/fireworks/models/nomic-embed-text-v1p5"
        input_texts = ["Hello world", "Test embedding", "Fireworks AI is fast"]

        cost = calc.estimate_embedding_cost(model, input_texts)

        assert isinstance(cost, Decimal)
        assert cost > 0

        # Embedding models typically have lower costs
        assert cost < Decimal("0.01")

    def test_cost_estimation_different_tiers(self):
        """Test cost estimation across different pricing tiers."""
        calc = FireworksPricingCalculator()

        test_models = [
            ("accounts/fireworks/models/llama-v3p2-1b-instruct", "tiny"),      # $0.10/M
            ("accounts/fireworks/models/llama-v3p1-8b-instruct", "small"),     # $0.20/M
            ("accounts/fireworks/models/llama-v3p1-70b-instruct", "large"),    # $0.90/M
            ("accounts/fireworks/models/llama-v3p1-405b-instruct", "premium")  # $3.00/M
        ]

        tokens = 1000
        costs = []

        for model, tier in test_models:
            cost = calc.estimate_chat_cost(model, tokens=tokens)
            costs.append((tier, cost))

        # Costs should increase with tier
        for i in range(len(costs) - 1):
            assert costs[i][1] < costs[i + 1][1], f"{costs[i][0]} should cost less than {costs[i + 1][0]}"

    def test_cost_estimation_parameter_differentiated_pricing(self):
        """Test models with different input/output pricing."""
        calc = FireworksPricingCalculator()

        # DeepSeek R1 has differentiated pricing
        model = "accounts/fireworks/models/deepseek-r1-distill-llama-70b"

        # Test with more output tokens (should cost more due to higher output rate)
        cost_low_output = calc.estimate_chat_cost(
            model, input_tokens=800, output_tokens=200
        )
        cost_high_output = calc.estimate_chat_cost(
            model, input_tokens=500, output_tokens=500
        )

        # Higher output token count should result in higher cost
        assert cost_high_output > cost_low_output


class TestModelRecommendations:
    """Test model recommendation engine."""

    def test_recommend_model_simple_task(self):
        """Test model recommendation for simple tasks."""
        calc = FireworksPricingCalculator()

        recommendation = calc.recommend_model(
            task_complexity="simple",
            budget_per_operation=0.001
        )

        assert isinstance(recommendation, ModelRecommendation)
        assert recommendation.recommended_model is not None
        assert recommendation.estimated_cost <= Decimal("0.001")
        assert len(recommendation.reasoning) > 0
        assert len(recommendation.alternatives) > 0

        # Should recommend a smaller/cheaper model for simple tasks
        assert "1b" in recommendation.recommended_model.lower() or "8b" in recommendation.recommended_model.lower()

    def test_recommend_model_complex_task(self):
        """Test model recommendation for complex tasks."""
        calc = FireworksPricingCalculator()

        recommendation = calc.recommend_model(
            task_complexity="complex",
            budget_per_operation=0.01  # Higher budget for complex tasks
        )

        assert isinstance(recommendation, ModelRecommendation)
        assert recommendation.recommended_model is not None
        assert recommendation.estimated_cost <= Decimal("0.01")

        # Should recommend a larger model for complex tasks
        assert "70b" in recommendation.recommended_model.lower() or "405b" in recommendation.recommended_model.lower()

    def test_recommend_model_budget_constraints(self):
        """Test model recommendation with tight budget constraints."""
        calc = FireworksPricingCalculator()

        # Very tight budget
        recommendation = calc.recommend_model(
            task_complexity="moderate",
            budget_per_operation=0.0001
        )

        if recommendation.recommended_model:
            assert recommendation.estimated_cost <= Decimal("0.0001")
            # Should recommend the smallest available model
            assert "1b" in recommendation.recommended_model.lower()

    def test_recommend_model_with_preferences(self):
        """Test model recommendation with specific preferences."""
        calc = FireworksPricingCalculator()

        # Test batch preference
        recommendation = calc.recommend_model(
            task_complexity="moderate",
            budget_per_operation=0.005,
            prefer_batch=True
        )

        assert isinstance(recommendation, ModelRecommendation)
        assert "batch" in recommendation.reasoning.lower() or "50%" in recommendation.reasoning

    def test_recommend_model_context_length_requirements(self):
        """Test model recommendation with context length requirements."""
        calc = FireworksPricingCalculator()

        recommendation = calc.recommend_model(
            task_complexity="moderate",
            budget_per_operation=0.005,
            min_context_length=100000  # High context requirement
        )

        if recommendation.recommended_model:
            model_pricing = calc.model_pricing[recommendation.recommended_model]
            assert model_pricing['context_length'] >= 100000

    def test_recommend_model_no_suitable_options(self):
        """Test model recommendation when no model fits budget."""
        calc = FireworksPricingCalculator()

        recommendation = calc.recommend_model(
            task_complexity="complex",
            budget_per_operation=0.00001  # Impossibly low budget
        )

        # Should handle gracefully
        assert recommendation.recommended_model is None or len(recommendation.alternatives) > 0


class TestModelComparisons:
    """Test model comparison functionality."""

    def test_compare_models_basic(self):
        """Test basic model comparison."""
        calc = FireworksPricingCalculator()

        models = [
            "accounts/fireworks/models/llama-v3p2-1b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct"
        ]

        comparisons = calc.compare_models(models, estimated_tokens=1000)

        assert len(comparisons) == len(models)

        for comparison in comparisons:
            assert 'model' in comparison
            assert 'estimated_cost' in comparison
            assert 'tier' in comparison
            assert 'context_length' in comparison
            assert comparison['estimated_cost'] > 0

    def test_compare_models_with_batch_analysis(self):
        """Test model comparison including batch processing analysis."""
        calc = FireworksPricingCalculator()

        models = [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct"
        ]

        comparisons = calc.compare_models(
            models,
            estimated_tokens=1000,
            include_batch_analysis=True
        )

        for comparison in comparisons:
            assert 'batch_cost' in comparison
            assert 'batch_savings' in comparison

            # Batch cost should be 50% of standard
            expected_batch_cost = comparison['estimated_cost'] * Decimal("0.5")
            assert abs(comparison['batch_cost'] - expected_batch_cost) < Decimal("0.00001")

    def test_compare_models_sorting(self):
        """Test model comparison sorting by cost."""
        calc = FireworksPricingCalculator()

        models = [
            "accounts/fireworks/models/llama-v3p1-405b-instruct",  # Most expensive
            "accounts/fireworks/models/llama-v3p2-1b-instruct",   # Least expensive
            "accounts/fireworks/models/llama-v3p1-70b-instruct"   # Medium
        ]

        comparisons = calc.compare_models(
            models,
            estimated_tokens=1000,
            sort_by_cost=True
        )

        # Should be sorted by cost (ascending)
        for i in range(len(comparisons) - 1):
            assert comparisons[i]['estimated_cost'] <= comparisons[i + 1]['estimated_cost']


class TestCostAnalysis:
    """Test comprehensive cost analysis functionality."""

    def test_analyze_costs_basic(self):
        """Test basic cost analysis for a workload."""
        calc = FireworksPricingCalculator()

        analysis = calc.analyze_costs(
            operations_per_day=1000,
            avg_tokens_per_operation=500,
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            days_to_analyze=30
        )

        assert isinstance(analysis, dict)
        assert 'cost_analysis' in analysis
        assert 'optimization' in analysis
        assert 'current_model' in analysis

        cost_data = analysis['cost_analysis']
        assert 'daily_cost' in cost_data
        assert 'monthly_cost' in cost_data
        assert 'cost_per_operation' in cost_data

        # Verify calculations make sense
        expected_daily_cost = 1000 * 500 * Decimal("0.0002") / 1000  # $0.10 per day
        assert abs(cost_data['daily_cost'] - expected_daily_cost) < Decimal("0.01")

    def test_analyze_costs_with_batch_optimization(self):
        """Test cost analysis with batch processing optimization."""
        calc = FireworksPricingCalculator()

        analysis = calc.analyze_costs(
            operations_per_day=10000,  # High volume suitable for batching
            avg_tokens_per_operation=300,
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            days_to_analyze=30,
            batch_percentage=0.7  # 70% of operations are batched
        )

        optimization = analysis['optimization']
        assert 'batch_optimization_potential' in optimization
        assert optimization['batch_optimization_potential'] > 0

        # With 70% batching, should see significant savings
        assert optimization['batch_optimization_potential'] > Decimal("0.5")

    def test_analyze_costs_with_model_alternatives(self):
        """Test cost analysis with alternative model suggestions."""
        calc = FireworksPricingCalculator()

        analysis = calc.analyze_costs(
            operations_per_day=5000,
            avg_tokens_per_operation=200,
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",  # Expensive model
            days_to_analyze=30
        )

        optimization = analysis['optimization']

        if 'best_alternative' in optimization:
            alternative = optimization['best_alternative']
            assert 'model' in alternative
            assert 'monthly_savings' in alternative
            assert alternative['monthly_savings'] > 0

    def test_analyze_costs_different_volumes(self):
        """Test cost analysis across different operation volumes."""
        calc = FireworksPricingCalculator()

        volumes = [100, 1000, 10000, 100000]  # Low to high volume
        analyses = []

        for volume in volumes:
            analysis = calc.analyze_costs(
                operations_per_day=volume,
                avg_tokens_per_operation=400,
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                days_to_analyze=30
            )
            analyses.append(analysis)

        # Higher volumes should have lower cost per operation (efficiency gains)
        for i in range(len(analyses) - 1):
            current_cost_per_op = analyses[i]['cost_analysis']['cost_per_operation']
            next_cost_per_op = analyses[i + 1]['cost_analysis']['cost_per_operation']

            # With batch optimization, higher volumes should be more cost-effective
            if analyses[i + 1]['optimization'].get('batch_optimization_potential', 0) > 0:
                assert current_cost_per_op >= next_cost_per_op


class TestSpecializedPricing:
    """Test specialized pricing scenarios and edge cases."""

    def test_multimodal_model_pricing(self):
        """Test pricing for multimodal models."""
        calc = FireworksPricingCalculator()

        vision_model = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

        # Vision models might have different pricing structures
        cost = calc.estimate_chat_cost(vision_model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_embedding_model_specialized_pricing(self):
        """Test specialized pricing for embedding models."""
        calc = FireworksPricingCalculator()

        embedding_model = "accounts/fireworks/models/nomic-embed-text-v1p5"

        # Test with different input sizes
        small_inputs = ["Short text"]
        large_inputs = ["Long text " * 100]

        small_cost = calc.estimate_embedding_cost(embedding_model, small_inputs)
        large_cost = calc.estimate_embedding_cost(embedding_model, large_inputs)

        assert large_cost > small_cost

    def test_code_model_pricing(self):
        """Test pricing for code-specialized models."""
        calc = FireworksPricingCalculator()

        code_model = "accounts/fireworks/models/deepseek-coder-v2-lite"

        cost = calc.estimate_chat_cost(code_model, tokens=1000)

        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_premium_model_pricing(self):
        """Test pricing for premium/largest models."""
        calc = FireworksPricingCalculator()

        premium_model = "accounts/fireworks/models/llama-v3p1-405b-instruct"

        cost = calc.estimate_chat_cost(premium_model, tokens=1000)

        # Premium model should be the most expensive
        standard_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        standard_cost = calc.estimate_chat_cost(standard_model, tokens=1000)

        assert cost > standard_cost
        assert cost >= Decimal("0.003")  # Should be at premium tier pricing


class TestPricingEdgeCases:
    """Test edge cases and error handling in pricing."""

    def test_invalid_model_pricing(self):
        """Test handling of invalid model names."""
        calc = FireworksPricingCalculator()

        with pytest.raises((KeyError, ValueError)):
            calc.estimate_chat_cost("invalid-model-name", tokens=1000)

    def test_zero_tokens_pricing(self):
        """Test handling of zero token requests."""
        calc = FireworksPricingCalculator()

        cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=0
        )

        assert cost == Decimal("0")

    def test_negative_tokens_handling(self):
        """Test handling of negative token counts."""
        calc = FireworksPricingCalculator()

        with pytest.raises(ValueError):
            calc.estimate_chat_cost(
                "accounts/fireworks/models/llama-v3p1-8b-instruct",
                tokens=-100
            )

    def test_extremely_large_token_counts(self):
        """Test handling of very large token counts."""
        calc = FireworksPricingCalculator()

        # Test with 1M tokens
        cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=1000000
        )

        # Should be exactly the per-million rate
        expected_cost = Decimal("0.20")  # $0.20 per 1M tokens
        assert abs(cost - expected_cost) < Decimal("0.01")


class TestBatchOptimizationCalculations:
    """Test batch processing optimization calculations."""

    def test_batch_savings_calculation(self):
        """Test batch savings calculation accuracy."""
        calc = FireworksPricingCalculator()

        model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
        tokens = 10000  # Large batch

        standard_cost = calc.estimate_chat_cost(model, tokens=tokens, is_batch=False)
        batch_cost = calc.estimate_chat_cost(model, tokens=tokens, is_batch=True)

        savings = standard_cost - batch_cost
        savings_percentage = (savings / standard_cost) * 100

        # Should be exactly 50% savings
        assert abs(savings_percentage - 50.0) < 1.0

    def test_batch_threshold_recommendations(self):
        """Test batch processing threshold recommendations."""
        calc = FireworksPricingCalculator()

        # Low volume - batch not recommended
        low_volume_rec = calc.recommend_model(
            task_complexity="simple",
            budget_per_operation=0.001,
            operations_per_day=10
        )

        # High volume - batch should be recommended
        high_volume_rec = calc.recommend_model(
            task_complexity="simple",
            budget_per_operation=0.001,
            operations_per_day=10000,
            prefer_batch=True
        )

        # High volume recommendation should mention batching benefits
        assert "batch" in high_volume_rec.reasoning.lower() or "50%" in high_volume_rec.reasoning

    def test_batch_optimization_roi_calculation(self):
        """Test ROI calculation for batch processing optimization."""
        calc = FireworksPricingCalculator()

        analysis = calc.analyze_costs(
            operations_per_day=50000,  # Very high volume
            avg_tokens_per_operation=300,
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            days_to_analyze=30,
            batch_percentage=0.8  # 80% batching
        )

        batch_savings = analysis['optimization']['batch_optimization_potential']
        monthly_cost = analysis['cost_analysis']['monthly_cost']

        # ROI should be significant for high-volume workloads
        roi_percentage = (batch_savings / monthly_cost) * 100
        assert roi_percentage > 30  # Should see substantial ROI from batching
