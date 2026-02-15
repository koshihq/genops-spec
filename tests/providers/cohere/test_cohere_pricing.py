"""Tests for GenOps Cohere pricing calculator."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Test imports
from genops.providers.cohere_pricing import (
    CohereCalculator,
    CohereOperation,
    CostBreakdown,
    ModelPricingTier,
    PricingPeriod,
)


class TestCohereCalculator:
    """Test suite for CohereCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return CohereCalculator()

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = CohereCalculator()
        assert calculator is not None

        # Test with custom pricing date
        custom_date = datetime(2024, 11, 1)
        calculator = CohereCalculator(pricing_date=custom_date)
        assert calculator.pricing_date == custom_date

    def test_chat_model_cost_calculation(self, calculator):
        """Test cost calculation for chat models."""
        # Test command-r-plus-08-2024 (premium model)
        input_cost, output_cost, op_cost = calculator.calculate_cost(
            model="command-r-plus-08-2024",
            operation="CHAT",
            input_tokens=1000,
            output_tokens=500,
            operation_units=1,
        )

        # Premium model should have higher costs
        assert input_cost > 0
        assert output_cost > 0
        assert op_cost == 0  # Chat operations don't have operation costs
        assert output_cost > input_cost  # Output typically costs more

    def test_embedding_model_cost_calculation(self, calculator):
        """Test cost calculation for embedding models."""
        input_cost, output_cost, op_cost = calculator.calculate_cost(
            model="embed-english-v4.0",
            operation="EMBED",
            input_tokens=1000,
            output_tokens=0,
            operation_units=3,  # 3 texts embedded
        )

        # Embedding models should have input costs and operation costs
        assert input_cost > 0
        assert output_cost == 0  # Embeddings don't have output tokens
        assert op_cost > 0  # Should have per-embedding operation cost

    def test_rerank_model_cost_calculation(self, calculator):
        """Test cost calculation for rerank models."""
        input_cost, output_cost, op_cost = calculator.calculate_cost(
            model="rerank-english-v3.0",
            operation="RERANK",
            input_tokens=0,  # Rerank typically doesn't count tokens
            output_tokens=0,
            operation_units=1,  # 1 rerank operation
        )

        # Rerank models primarily use operation-based pricing
        assert op_cost > 0  # Should have per-search operation cost
        # Input/output costs may or may not be present depending on model

    def test_model_not_found_error(self, calculator):
        """Test error handling for unknown models."""
        with pytest.raises(ValueError) as exc_info:
            calculator.calculate_cost(
                model="unknown-model",
                operation="CHAT",
                input_tokens=100,
                output_tokens=50,
            )

        assert "unknown model" in str(exc_info.value).lower()

    def test_invalid_operation_error(self, calculator):
        """Test error handling for invalid operations."""
        with pytest.raises(ValueError) as exc_info:
            calculator.calculate_cost(
                model="command-light",
                operation="INVALID_OPERATION",
                input_tokens=100,
                output_tokens=50,
            )

        assert "invalid operation" in str(exc_info.value).lower()

    def test_cost_breakdown_creation(self, calculator):
        """Test CostBreakdown object creation."""
        breakdown = calculator.get_cost_breakdown(
            model="command-r-08-2024",
            operation="CHAT",
            input_tokens=200,
            output_tokens=150,
        )

        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.model == "command-r-08-2024"
        assert breakdown.operation == CohereOperation.CHAT
        assert breakdown.input_tokens == 200
        assert breakdown.output_tokens == 150
        assert (
            breakdown.total_cost
            == breakdown.input_cost + breakdown.output_cost + breakdown.operation_cost
        )

    def test_compare_model_costs(self, calculator):
        """Test model cost comparison functionality."""
        models = ["command-light", "command-r-08-2024", "command-r-plus-08-2024"]

        comparison = calculator.compare_model_costs(
            models=models, operation="CHAT", input_tokens=1000, output_tokens=500
        )

        assert len(comparison) == 3
        for model in models:
            assert model in comparison
            assert isinstance(comparison[model], CostBreakdown)

        # Premium model should cost more than light model
        light_cost = comparison["command-light"].total_cost
        plus_cost = comparison["command-r-plus-08-2024"].total_cost
        assert plus_cost > light_cost

    def test_get_cheapest_model(self, calculator):
        """Test cheapest model selection."""
        models = ["command-light", "command-r-08-2024", "command-r-plus-08-2024"]

        cheapest = calculator.get_cheapest_model(
            models=models, operation="CHAT", input_tokens=1000, output_tokens=500
        )

        # Light model should be cheapest
        assert cheapest == "command-light"

    def test_get_most_expensive_model(self, calculator):
        """Test most expensive model selection."""
        models = ["command-light", "command-r-08-2024", "command-r-plus-08-2024"]

        most_expensive = calculator.get_most_expensive_model(
            models=models, operation="CHAT", input_tokens=1000, output_tokens=500
        )

        # Plus model should be most expensive
        assert most_expensive == "command-r-plus-08-2024"

    def test_estimate_cost_from_text_length(self, calculator):
        """Test cost estimation from text length."""
        text = "This is a sample text for cost estimation testing." * 20  # ~200 chars

        estimated_cost = calculator.estimate_cost_from_text(
            text=text,
            model="command-r-08-2024",
            operation="CHAT",
            expected_output_ratio=0.5,  # 50% output length
        )

        assert estimated_cost > 0
        assert isinstance(estimated_cost, float)

    def test_bulk_cost_calculation(self, calculator):
        """Test bulk cost calculations for multiple operations."""
        operations = [
            {
                "model": "command-light",
                "operation": "CHAT",
                "input_tokens": 100,
                "output_tokens": 50,
            },
            {
                "model": "embed-english-v4.0",
                "operation": "EMBED",
                "input_tokens": 200,
                "operation_units": 2,
            },
            {
                "model": "rerank-english-v3.0",
                "operation": "RERANK",
                "operation_units": 1,
            },
        ]

        total_cost = calculator.calculate_bulk_cost(operations)

        assert total_cost > 0
        assert isinstance(total_cost, float)

    def test_pricing_tier_handling(self, calculator):
        """Test pricing tier calculations."""
        # Test volume pricing (if implemented)
        high_volume_cost = calculator.calculate_cost(
            model="command-r-08-2024",
            operation="CHAT",
            input_tokens=100000,  # High volume
            output_tokens=50000,
            pricing_tier=ModelPricingTier.ENTERPRISE,
        )

        standard_cost = calculator.calculate_cost(
            model="command-r-08-2024",
            operation="CHAT",
            input_tokens=100000,
            output_tokens=50000,
            pricing_tier=ModelPricingTier.STANDARD,
        )

        # Enterprise tier might have volume discounts
        input_cost_enterprise, output_cost_enterprise, _ = high_volume_cost
        input_cost_standard, output_cost_standard, _ = standard_cost
        total_enterprise = input_cost_enterprise + output_cost_enterprise
        total_standard = input_cost_standard + output_cost_standard

        # Note: This assumes volume discounts exist, test may need adjustment
        assert total_enterprise <= total_standard or total_enterprise == total_standard

    def test_time_based_pricing(self, calculator):
        """Test pricing calculations for different time periods."""
        # Calculate costs for different periods
        daily_cost = calculator.calculate_period_cost(
            operations_per_period=100,
            model="command-r-08-2024",
            operation="CHAT",
            avg_input_tokens=500,
            avg_output_tokens=300,
            period=PricingPeriod.DAILY,
        )

        monthly_cost = calculator.calculate_period_cost(
            operations_per_period=3000,  # 100/day * 30 days
            model="command-r-08-2024",
            operation="CHAT",
            avg_input_tokens=500,
            avg_output_tokens=300,
            period=PricingPeriod.MONTHLY,
        )

        assert daily_cost > 0
        assert monthly_cost > daily_cost
        assert monthly_cost >= daily_cost * 30  # Should be at least 30x daily

    def test_multi_operation_workflow_costing(self, calculator):
        """Test costing for multi-operation workflows."""
        workflow = {
            "embed_query": {
                "model": "embed-english-v4.0",
                "operation": "EMBED",
                "input_tokens": 50,
                "operation_units": 1,
            },
            "embed_docs": {
                "model": "embed-english-v4.0",
                "operation": "EMBED",
                "input_tokens": 1000,
                "operation_units": 10,
            },
            "rerank": {
                "model": "rerank-english-v3.0",
                "operation": "RERANK",
                "operation_units": 1,
            },
            "summarize": {
                "model": "command-r-08-2024",
                "operation": "CHAT",
                "input_tokens": 800,
                "output_tokens": 200,
            },
        }

        total_workflow_cost = calculator.calculate_workflow_cost(workflow)

        assert total_workflow_cost > 0

        # Verify individual component costs sum up correctly
        component_costs = []
        for _step_name, step_config in workflow.items():
            step_cost = sum(calculator.calculate_cost(**step_config))
            component_costs.append(step_cost)

        expected_total = sum(component_costs)
        assert (
            abs(total_workflow_cost - expected_total) < 0.000001
        )  # Allow for float precision

    def test_cost_optimization_recommendations(self, calculator):
        """Test cost optimization recommendations."""
        current_config = {
            "model": "command-r-plus-08-2024",
            "operation": "CHAT",
            "input_tokens": 500,
            "output_tokens": 300,
        }

        recommendations = calculator.get_optimization_recommendations(
            current_config=current_config,
            quality_threshold=0.8,  # Maintain 80% quality
            cost_reduction_target=0.3,  # Reduce costs by 30%
        )

        assert len(recommendations) > 0

        # Should recommend cheaper models
        for rec in recommendations:
            assert "model" in rec
            assert rec["estimated_savings"] > 0
            assert rec["quality_impact"] <= 0.2  # Within threshold

    def test_pricing_data_freshness(self, calculator):
        """Test pricing data age and freshness warnings."""
        # Test with old pricing date
        old_date = datetime.now() - timedelta(days=60)
        old_calculator = CohereCalculator(pricing_date=old_date)

        # Should warn about stale pricing
        with patch("warnings.warn") as mock_warn:
            old_calculator.calculate_cost(
                model="command-light",
                operation="CHAT",
                input_tokens=100,
                output_tokens=50,
            )

            # Should warn about stale pricing data
            mock_warn.assert_called()
            warning_message = mock_warn.call_args[0][0]
            assert "pricing data" in warning_message.lower()
            assert (
                "outdated" in warning_message.lower()
                or "stale" in warning_message.lower()
            )


class TestCostBreakdown:
    """Test CostBreakdown data structure."""

    def test_cost_breakdown_initialization(self):
        """Test CostBreakdown creation and properties."""
        breakdown = CostBreakdown(
            model="command-r-08-2024",
            operation=CohereOperation.CHAT,
            input_tokens=100,
            output_tokens=75,
            input_cost=0.001,
            output_cost=0.002,
            operation_cost=0.0,
        )

        assert breakdown.model == "command-r-08-2024"
        assert breakdown.operation == CohereOperation.CHAT
        assert breakdown.total_tokens == 175
        assert breakdown.total_cost == 0.003

    def test_cost_breakdown_comparison(self):
        """Test CostBreakdown comparison methods."""
        breakdown1 = CostBreakdown(
            model="command-light", operation=CohereOperation.CHAT, total_cost=0.001
        )

        breakdown2 = CostBreakdown(
            model="command-r-plus-08-2024",
            operation=CohereOperation.CHAT,
            total_cost=0.005,
        )

        assert breakdown1 < breakdown2
        assert breakdown2 > breakdown1
        assert breakdown1 != breakdown2

    def test_cost_breakdown_serialization(self):
        """Test CostBreakdown serialization to dict."""
        breakdown = CostBreakdown(
            model="embed-english-v4.0",
            operation=CohereOperation.EMBED,
            input_tokens=200,
            operation_units=5,
            input_cost=0.002,
            operation_cost=0.003,
        )

        as_dict = breakdown.to_dict()

        assert isinstance(as_dict, dict)
        assert as_dict["model"] == "embed-english-v4.0"
        assert as_dict["operation"] == "EMBED"
        assert as_dict["total_cost"] == 0.005


class TestPricingEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def calculator(self):
        return CohereCalculator()

    def test_zero_token_calculation(self, calculator):
        """Test cost calculation with zero tokens."""
        input_cost, output_cost, op_cost = calculator.calculate_cost(
            model="command-light", operation="CHAT", input_tokens=0, output_tokens=0
        )

        # Should handle zero tokens gracefully
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_negative_token_calculation(self, calculator):
        """Test cost calculation with negative tokens."""
        with pytest.raises(ValueError) as exc_info:
            calculator.calculate_cost(
                model="command-light",
                operation="CHAT",
                input_tokens=-100,
                output_tokens=50,
            )

        assert "negative" in str(exc_info.value).lower()

    def test_extremely_large_token_count(self, calculator):
        """Test cost calculation with very large token counts."""
        large_tokens = 10**9  # 1 billion tokens

        input_cost, output_cost, op_cost = calculator.calculate_cost(
            model="command-light",
            operation="CHAT",
            input_tokens=large_tokens,
            output_tokens=large_tokens // 2,
        )

        # Should handle large numbers without overflow
        assert input_cost > 0
        assert output_cost > 0
        assert isinstance(input_cost, float)
        assert isinstance(output_cost, float)

    def test_mixed_case_model_names(self, calculator):
        """Test cost calculation with mixed case model names."""
        # Should be case-insensitive
        cost1 = calculator.calculate_cost(
            model="command-light", operation="CHAT", input_tokens=100, output_tokens=50
        )

        cost2 = calculator.calculate_cost(
            model="COMMAND-LIGHT", operation="CHAT", input_tokens=100, output_tokens=50
        )

        # Should produce same results
        assert cost1 == cost2

    def test_pricing_calculation_precision(self, calculator):
        """Test pricing calculation precision for small amounts."""
        input_cost, output_cost, op_cost = calculator.calculate_cost(
            model="command-light",
            operation="CHAT",
            input_tokens=1,  # Single token
            output_tokens=1,
        )

        # Should maintain precision for small calculations
        assert input_cost > 0
        assert output_cost > 0

        # Should be appropriately small
        assert input_cost < 0.01  # Less than 1 cent
        assert output_cost < 0.01


class TestPricingIntegration:
    """Test pricing calculator integration scenarios."""

    def test_pricing_with_adapter_integration(self):
        """Test pricing calculator integration with adapter."""
        with patch(
            "genops.providers.cohere_pricing.CohereCalculator"
        ) as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.calculate_cost.return_value = (0.001, 0.002, 0.0)

            # Simulate adapter using calculator
            calculator = mock_calc_class()
            cost = calculator.calculate_cost(
                model="command-r-08-2024",
                operation="CHAT",
                input_tokens=100,
                output_tokens=150,
            )

            assert cost == (0.001, 0.002, 0.0)
            mock_calc.calculate_cost.assert_called_once_with(
                model="command-r-08-2024",
                operation="CHAT",
                input_tokens=100,
                output_tokens=150,
            )

    def test_pricing_data_validation(self):
        """Test that pricing data is properly validated."""
        calculator = CohereCalculator()

        # Should have pricing data for all major models
        major_models = [
            "command-light",
            "command-r-08-2024",
            "command-r-plus-08-2024",
            "embed-english-v4.0",
            "rerank-english-v3.0",
        ]

        for model in major_models:
            try:
                calculator.calculate_cost(
                    model=model,
                    operation="CHAT"
                    if "embed" not in model and "rerank" not in model
                    else "EMBED"
                    if "embed" in model
                    else "RERANK",
                    input_tokens=100,
                    output_tokens=50
                    if "rerank" not in model and "embed" not in model
                    else 0,
                    operation_units=1 if "embed" in model or "rerank" in model else 0,
                )
            except ValueError:
                pytest.fail(f"Pricing data missing for major model: {model}")


if __name__ == "__main__":
    pytest.main([__file__])
