#!/usr/bin/env python3
"""
Test suite for GenOps Gemini pricing calculations.

This module tests the Gemini pricing functionality including:
- Cost calculations for different models
- Model information and metadata
- Cost comparison across models
- Optimization recommendations
- Monthly cost estimation
"""

import pytest

from genops.providers.gemini_pricing import (
    GEMINI_MODELS,
    GeminiCostBreakdown,
    GeminiModelInfo,
    GeminiTier,
    calculate_gemini_cost,
    calculate_gemini_cost_breakdown,
    compare_gemini_models,
    estimate_monthly_cost,
    get_cost_optimization_recommendations,
    get_gemini_model_info,
    list_gemini_models,
)


class TestGeminiCostCalculation:
    """Test Gemini cost calculation functionality."""

    def test_calculate_cost_flash_model(self):
        """Test cost calculation for Gemini 2.5 Flash."""
        cost = calculate_gemini_cost(
            model_id="gemini-2.5-flash", input_tokens=1000, output_tokens=500
        )

        # Flash: $0.30/1M input, $2.50/1M output
        # Expected: (1000/1M * 0.30) + (500/1M * 2.50) = 0.0003 + 0.00125 = 0.00155
        expected = 0.00155
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_pro_model(self):
        """Test cost calculation for Gemini 2.5 Pro."""
        cost = calculate_gemini_cost(
            model_id="gemini-2.5-pro", input_tokens=1000, output_tokens=500
        )

        # Pro: $1.25/1M input, $10.00/1M output
        # Expected: (1000/1M * 1.25) + (500/1M * 10.00) = 0.00125 + 0.005 = 0.00625
        expected = 0.00625
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_flash_lite_model(self):
        """Test cost calculation for Gemini 2.5 Flash-Lite."""
        cost = calculate_gemini_cost(
            model_id="gemini-2.5-flash-lite", input_tokens=1000, output_tokens=500
        )

        # Flash-Lite: $0.15/1M input, $1.25/1M output (estimated)
        # Expected: (1000/1M * 0.15) + (500/1M * 1.25) = 0.00015 + 0.000625 = 0.000775
        expected = 0.000775
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_with_context_cache(self):
        """Test cost calculation including context caching."""
        cost = calculate_gemini_cost(
            model_id="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            context_cache_tokens=2000,
        )

        # Flash: $0.30/1M input, $2.50/1M output, $0.03/1M cache
        # Expected: (1000/1M * 0.30) + (500/1M * 2.50) + (2000/1M * 0.03)
        # = 0.0003 + 0.00125 + 0.00006 = 0.00161
        expected = 0.00161
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_free_tier(self):
        """Test cost calculation for free tier model."""
        cost = calculate_gemini_cost(
            model_id="gemini-1.5-flash-free", input_tokens=1000, output_tokens=500
        )

        # Free tier should return 0 cost
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model falls back to Flash pricing."""
        cost = calculate_gemini_cost(
            model_id="unknown-model-123", input_tokens=1000, output_tokens=500
        )

        # Should use Flash pricing as fallback
        # Flash: $0.30/1M input, $2.50/1M output
        expected = 0.00155
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_gemini_cost(
            model_id="gemini-2.5-flash", input_tokens=0, output_tokens=0
        )

        assert cost == 0.0

    def test_calculate_cost_large_numbers(self):
        """Test cost calculation with large token counts."""
        cost = calculate_gemini_cost(
            model_id="gemini-2.5-flash",
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=500_000,  # 0.5M tokens
        )

        # Flash: $0.30/1M input, $2.50/1M output
        # Expected: (1M/1M * 0.30) + (0.5M/1M * 2.50) = 0.30 + 1.25 = 1.55
        expected = 1.55
        assert abs(cost - expected) < 0.001


class TestGeminiCostBreakdown:
    """Test detailed cost breakdown functionality."""

    def test_cost_breakdown_detailed(self):
        """Test detailed cost breakdown calculation."""
        breakdown = calculate_gemini_cost_breakdown(
            model_id="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            context_cache_tokens=200,
        )

        assert isinstance(breakdown, GeminiCostBreakdown)
        assert breakdown.model_id == "gemini-2.5-flash"
        assert breakdown.input_tokens == 1000
        assert breakdown.output_tokens == 500
        assert breakdown.context_cache_tokens == 200

        # Check individual cost components
        assert breakdown.input_cost == 0.0003  # 1000/1M * 0.30
        assert breakdown.output_cost == 0.00125  # 500/1M * 2.50
        assert breakdown.context_cache_cost == 0.000006  # 200/1M * 0.03

        assert (
            breakdown.total_cost
            == breakdown.input_cost
            + breakdown.output_cost
            + breakdown.context_cache_cost
        )
        assert breakdown.currency == "USD"

        # Check derived metrics
        total_tokens = 1000 + 500 + 200
        expected_cost_per_1k = (breakdown.total_cost / total_tokens) * 1000
        assert abs(breakdown.cost_per_1k_tokens - expected_cost_per_1k) < 0.000001

    def test_cost_breakdown_no_context_cache(self):
        """Test cost breakdown without context caching."""
        breakdown = calculate_gemini_cost_breakdown(
            model_id="gemini-2.5-pro", input_tokens=500, output_tokens=300
        )

        assert breakdown.context_cache_tokens is None
        assert breakdown.context_cache_cost == 0.0

        # Pro model: $1.25/1M input, $10.00/1M output
        expected_input = 500 / 1_000_000 * 1.25
        expected_output = 300 / 1_000_000 * 10.00

        assert abs(breakdown.input_cost - expected_input) < 0.000001
        assert abs(breakdown.output_cost - expected_output) < 0.000001


class TestGeminiModelInfo:
    """Test model information functionality."""

    def test_get_model_info_existing(self):
        """Test getting info for existing model."""
        info = get_gemini_model_info("gemini-2.5-flash")

        assert isinstance(info, GeminiModelInfo)
        assert info.model_id == "gemini-2.5-flash"
        assert info.display_name == "Gemini 2.5 Flash"
        assert info.provider == "google"
        assert info.tier == GeminiTier.PAID
        assert info.supports_streaming is True
        assert info.supports_function_calling is True
        assert info.supports_multimodal is True

    def test_get_model_info_nonexistent(self):
        """Test getting info for non-existent model."""
        info = get_gemini_model_info("non-existent-model")

        assert info is None

    def test_list_all_models(self):
        """Test listing all available models."""
        models = list_gemini_models()

        assert len(models) > 0
        assert all(isinstance(model, GeminiModelInfo) for model in models)

        # Check that models are sorted by price
        prices = [model.input_price_per_1m_tokens for model in models]
        assert prices == sorted(prices)

    def test_list_models_by_tier(self):
        """Test listing models filtered by tier."""
        paid_models = list_gemini_models(tier=GeminiTier.PAID)
        free_models = list_gemini_models(tier=GeminiTier.FREE)

        assert all(model.tier == GeminiTier.PAID for model in paid_models)
        assert all(model.tier == GeminiTier.FREE for model in free_models)

        assert len(paid_models) > 0
        assert len(free_models) >= 0  # May be 0 or more free models

    def test_model_capabilities(self):
        """Test model capability information."""
        pro_info = get_gemini_model_info("gemini-2.5-pro")
        flash_info = get_gemini_model_info("gemini-2.5-flash")

        # Pro model should support code execution
        assert pro_info.supports_code_execution is True

        # Both should support basic capabilities
        assert pro_info.supports_streaming is True
        assert flash_info.supports_streaming is True
        assert pro_info.supports_function_calling is True
        assert flash_info.supports_function_calling is True


class TestGeminiModelComparison:
    """Test model comparison functionality."""

    def test_compare_models_basic(self):
        """Test basic model comparison."""
        models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        comparison = compare_gemini_models(
            models=models, input_tokens=1000, output_tokens=500
        )

        assert len(comparison) == 2
        assert all(isinstance(result, dict) for result in comparison)

        # Check required fields
        for result in comparison:
            assert "model_id" in result
            assert "total_cost" in result
            assert "display_name" in result
            assert "tier" in result
            assert "cost_per_1k_tokens" in result

        # Should be sorted by total cost (ascending)
        costs = [result["total_cost"] for result in comparison]
        assert costs == sorted(costs)

    def test_compare_models_sort_by_different_criteria(self):
        """Test model comparison with different sort criteria."""
        models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

        # Sort by model_id (ascending)
        comparison_by_id = compare_gemini_models(
            models=models, input_tokens=1000, output_tokens=500, sort_by="model_id"
        )

        model_ids = [result["model_id"] for result in comparison_by_id]
        assert model_ids == sorted(model_ids)

        # Sort by cost_per_1k_tokens (descending)
        comparison_by_cost = compare_gemini_models(
            models=models,
            input_tokens=1000,
            output_tokens=500,
            sort_by="cost_per_1k_tokens",
        )

        costs = [result["cost_per_1k_tokens"] for result in comparison_by_cost]
        assert costs == sorted(costs, reverse=True)

    def test_compare_models_with_context_cache(self):
        """Test model comparison including context caching."""
        models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        comparison = compare_gemini_models(
            models=models,
            input_tokens=1000,
            output_tokens=500,
            context_cache_tokens=1000,
        )

        # All results should include context cache costs
        for result in comparison:
            assert result["context_cache_cost"] > 0

    def test_compare_single_model(self):
        """Test comparison with single model."""
        comparison = compare_gemini_models(
            models=["gemini-2.5-flash"], input_tokens=1000, output_tokens=500
        )

        assert len(comparison) == 1
        assert comparison[0]["model_id"] == "gemini-2.5-flash"


class TestCostOptimization:
    """Test cost optimization recommendations."""

    def test_optimization_recommendations_general_use_case(self):
        """Test optimization recommendations for general use case."""
        recommendations = get_cost_optimization_recommendations(
            model_id="gemini-2.5-pro",  # Start with most expensive
            input_tokens=1000,
            output_tokens=500,
            use_case="general",
        )

        assert len(recommendations) > 0

        for rec in recommendations:
            assert "model_id" in rec
            assert "savings" in rec
            assert "savings_percent" in rec
            assert "recommendation_type" in rec

            # Should recommend cheaper alternatives
            assert rec["savings"] > 0
            assert rec["savings_percent"] > 0

    def test_optimization_recommendations_code_use_case(self):
        """Test optimization recommendations for code use case."""
        recommendations = get_cost_optimization_recommendations(
            model_id="gemini-2.5-pro",
            input_tokens=1000,
            output_tokens=500,
            use_case="code",
        )

        # Should include Pro and Flash models for code tasks
        recommended_models = [rec["model_id"] for rec in recommendations]
        assert any("flash" in model for model in recommended_models)

    def test_optimization_recommendations_with_budget_constraint(self):
        """Test optimization recommendations with budget constraints."""
        budget_limit = 0.001  # Very low budget

        recommendations = get_cost_optimization_recommendations(
            model_id="gemini-2.5-pro",
            input_tokens=1000,
            output_tokens=500,
            budget_constraint=budget_limit,
        )

        # All recommendations should be within budget
        for rec in recommendations:
            assert rec["alternative_cost"] <= budget_limit

    def test_optimization_no_savings_available(self):
        """Test optimization when no savings are available."""
        recommendations = get_cost_optimization_recommendations(
            model_id="gemini-2.5-flash-lite",  # Already cheapest
            input_tokens=1000,
            output_tokens=500,
        )

        # Should return empty or minimal recommendations
        # since we're already using the cheapest model
        meaningful_savings = [
            rec for rec in recommendations if rec["savings"] > 0.000001
        ]
        assert len(meaningful_savings) == 0


class TestMonthlyEstimation:
    """Test monthly cost estimation."""

    def test_estimate_monthly_cost_basic(self):
        """Test basic monthly cost estimation."""
        estimate = estimate_monthly_cost(
            model_id="gemini-2.5-flash",
            daily_operations=100,
            avg_input_tokens=1000,
            avg_output_tokens=500,
        )

        assert isinstance(estimate, dict)
        assert "monthly_cost" in estimate
        assert "daily_cost" in estimate
        assert "cost_per_operation" in estimate
        assert "monthly_operations" in estimate

        # Verify calculations
        expected_ops_per_month = 100 * 30
        assert estimate["monthly_operations"] == expected_ops_per_month

        expected_daily_cost = estimate["cost_per_operation"] * 100
        assert abs(estimate["daily_cost"] - expected_daily_cost) < 0.000001

        expected_monthly_cost = estimate["daily_cost"] * 30
        assert abs(estimate["monthly_cost"] - expected_monthly_cost) < 0.000001

    def test_estimate_monthly_cost_custom_days(self):
        """Test monthly cost estimation with custom days per month."""
        estimate = estimate_monthly_cost(
            model_id="gemini-2.5-flash",
            daily_operations=50,
            avg_input_tokens=500,
            avg_output_tokens=300,
            days_per_month=31,  # Custom month length
        )

        expected_monthly_ops = 50 * 31
        assert estimate["monthly_operations"] == expected_monthly_ops

        expected_monthly_cost = estimate["daily_cost"] * 31
        assert abs(estimate["monthly_cost"] - expected_monthly_cost) < 0.000001

    def test_estimate_monthly_cost_high_volume(self):
        """Test monthly cost estimation for high volume usage."""
        estimate = estimate_monthly_cost(
            model_id="gemini-2.5-pro",
            daily_operations=10000,  # High volume
            avg_input_tokens=2000,  # Larger requests
            avg_output_tokens=1000,
        )

        # High volume should result in significant monthly cost
        assert estimate["monthly_cost"] > 10.0  # Should be substantial
        assert estimate["monthly_operations"] == 300000  # 10k * 30

    def test_estimate_monthly_cost_different_models(self):
        """Test that different models produce different cost estimates."""
        flash_estimate = estimate_monthly_cost(
            model_id="gemini-2.5-flash",
            daily_operations=100,
            avg_input_tokens=1000,
            avg_output_tokens=500,
        )

        pro_estimate = estimate_monthly_cost(
            model_id="gemini-2.5-pro",
            daily_operations=100,
            avg_input_tokens=1000,
            avg_output_tokens=500,
        )

        # Pro should be more expensive than Flash
        assert pro_estimate["monthly_cost"] > flash_estimate["monthly_cost"]
        assert pro_estimate["cost_per_operation"] > flash_estimate["cost_per_operation"]


class TestPricingConstants:
    """Test pricing data and constants."""

    def test_gemini_models_data_integrity(self):
        """Test that GEMINI_MODELS data is complete and valid."""
        assert len(GEMINI_MODELS) > 0

        for model_id, model_info in GEMINI_MODELS.items():
            assert isinstance(model_info, GeminiModelInfo)
            assert model_info.model_id == model_id
            assert model_info.display_name
            assert model_info.provider == "google"
            assert isinstance(model_info.tier, GeminiTier)
            assert model_info.input_price_per_1m_tokens >= 0
            assert model_info.output_price_per_1m_tokens >= 0
            assert model_info.max_context_length > 0
            assert model_info.max_output_tokens > 0

    def test_pricing_consistency(self):
        """Test pricing consistency across models."""
        # Free tier models should have zero cost
        free_models = [
            info for info in GEMINI_MODELS.values() if info.tier == GeminiTier.FREE
        ]
        for model in free_models:
            assert model.input_price_per_1m_tokens == 0
            assert model.output_price_per_1m_tokens == 0

        # Paid models should have non-zero cost
        paid_models = [
            info for info in GEMINI_MODELS.values() if info.tier == GeminiTier.PAID
        ]
        for model in paid_models:
            assert model.input_price_per_1m_tokens > 0
            assert model.output_price_per_1m_tokens > 0

    def test_model_hierarchy_pricing(self):
        """Test that model pricing follows expected hierarchy."""
        flash_lite = GEMINI_MODELS.get("gemini-2.5-flash-lite")
        flash = GEMINI_MODELS.get("gemini-2.5-flash")
        pro = GEMINI_MODELS.get("gemini-2.5-pro")

        if flash_lite and flash and pro:
            # Pro should be most expensive
            assert pro.input_price_per_1m_tokens >= flash.input_price_per_1m_tokens
            assert pro.output_price_per_1m_tokens >= flash.output_price_per_1m_tokens

            # Flash-Lite should be cheapest
            assert (
                flash_lite.input_price_per_1m_tokens <= flash.input_price_per_1m_tokens
            )
            assert (
                flash_lite.output_price_per_1m_tokens
                <= flash.output_price_per_1m_tokens
            )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
