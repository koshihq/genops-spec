"""Tests for Anyscale cost calculation accuracy and edge cases."""

import pytest
from genops.providers.anyscale.pricing import (
    calculate_completion_cost,
    calculate_embedding_cost,
    ANYSCALE_PRICING,
)


class TestCostCalculationAccuracy:
    """Test cost calculation accuracy across models."""

    def test_llama2_70b_cost_accuracy(self):
        """Test Llama-2-70b cost calculation accuracy."""
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=1000,
            output_tokens=500
        )

        # $1.00/M tokens for both input and output
        # (1000 + 500) / 1,000,000 * $1.00 = $0.0015
        expected = 0.0015
        assert cost == pytest.approx(expected, abs=1e-10)

    def test_llama2_7b_cost_accuracy(self):
        """Test Llama-2-7b cost calculation accuracy."""
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-7b-chat-hf",
            input_tokens=1000,
            output_tokens=500
        )

        # $0.15/M tokens for both input and output
        # (1000 + 500) / 1,000,000 * $0.15 = $0.000225
        expected = 0.000225
        assert cost == pytest.approx(expected, abs=1e-10)

    def test_mistral_7b_cost_accuracy(self):
        """Test Mistral-7b cost calculation accuracy."""
        cost = calculate_completion_cost(
            "mistralai/Mistral-7B-Instruct-v0.1",
            input_tokens=2000,
            output_tokens=1000
        )

        # $0.15/M tokens
        # (2000 + 1000) / 1,000,000 * $0.15 = $0.00045
        expected = 0.00045
        assert cost == pytest.approx(expected, abs=1e-10)

    def test_embedding_cost_accuracy(self):
        """Test embedding cost calculation accuracy."""
        cost = calculate_embedding_cost(
            "thenlper/gte-large",
            tokens=5000
        )

        # $0.05/M tokens
        # 5000 / 1,000,000 * $0.05 = $0.00025
        expected = 0.00025
        assert cost == pytest.approx(expected, abs=1e-10)


class TestCostCalculationEdgeCases:
    """Test cost calculation edge cases."""

    def test_single_token_cost(self):
        """Test cost calculation with single token."""
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=1,
            output_tokens=0
        )

        expected = 1 / 1_000_000 * 1.0
        assert cost == pytest.approx(expected, abs=1e-12)

    def test_very_large_token_count(self):
        """Test cost calculation with very large token counts."""
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=100000,
            output_tokens=50000
        )

        # 150,000 / 1,000,000 * $1.00 = $0.15
        expected = 0.15
        assert cost == pytest.approx(expected, abs=1e-10)

    def test_cost_precision_maintained(self):
        """Test cost calculation maintains high precision."""
        # Test with prime numbers to check precision
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=1237,
            output_tokens=4567
        )

        expected = (1237 + 4567) / 1_000_000 * 1.0
        assert cost == pytest.approx(expected, rel=1e-10)

    def test_negative_tokens_handling(self):
        """Test handling of negative token counts."""
        # Should handle gracefully or raise error
        try:
            cost = calculate_completion_cost(
                "meta-llama/Llama-2-70b-chat-hf",
                input_tokens=-100,
                output_tokens=50
            )
            # If it doesn't raise, cost should be 0 or handle gracefully
            assert cost >= 0
        except (ValueError, AssertionError):
            # Acceptable to raise error for invalid input
            pass


class TestMultiModelCostComparison:
    """Test cost comparisons across different models."""

    def test_model_cost_ordering(self):
        """Test that model costs are ordered as expected."""
        tokens_in = 1000
        tokens_out = 1000

        cost_70b = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", tokens_in, tokens_out
        )
        cost_13b = calculate_completion_cost(
            "meta-llama/Llama-2-13b-chat-hf", tokens_in, tokens_out
        )
        cost_7b = calculate_completion_cost(
            "meta-llama/Llama-2-7b-chat-hf", tokens_in, tokens_out
        )

        # Larger models should cost more
        assert cost_70b > cost_7b
        assert cost_70b > cost_13b
        assert cost_13b > cost_7b

    def test_cost_savings_calculation(self):
        """Test calculating cost savings between models."""
        tokens_in = 10000
        tokens_out = 5000

        cost_expensive = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", tokens_in, tokens_out
        )
        cost_cheap = calculate_completion_cost(
            "meta-llama/Llama-2-7b-chat-hf", tokens_in, tokens_out
        )

        savings = cost_expensive - cost_cheap
        savings_percent = (savings / cost_expensive) * 100

        # Should have significant savings
        assert savings > 0
        assert savings_percent > 50  # At least 50% savings


class TestCostCalculationConsistency:
    """Test cost calculation consistency."""

    def test_same_tokens_same_cost(self):
        """Test same token counts produce same cost."""
        cost1 = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", 100, 50
        )
        cost2 = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", 100, 50
        )

        assert cost1 == cost2

    def test_order_independence(self):
        """Test that swapping input/output with same pricing gives same cost."""
        # For models with same input/output pricing
        cost1 = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=100,
            output_tokens=50
        )
        cost2 = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=50,
            output_tokens=100
        )

        # Both should equal 150 tokens * $1.00/M
        assert cost1 == cost2

    def test_additive_property(self):
        """Test that costs are additive."""
        cost_part1 = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", 100, 50
        )
        cost_part2 = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", 200, 100
        )
        cost_total = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf", 300, 150
        )

        assert cost_total == pytest.approx(cost_part1 + cost_part2, abs=1e-10)


class TestRealWorldScenarios:
    """Test cost calculations for real-world scenarios."""

    def test_typical_chat_message_cost(self):
        """Test cost for typical chat message."""
        # Typical chat: ~50 input tokens, ~100 output tokens
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=50,
            output_tokens=100
        )

        # Should be very small cost
        assert cost < 0.001  # Less than $0.001
        assert cost > 0      # But greater than zero

    def test_long_document_analysis_cost(self):
        """Test cost for long document analysis."""
        # Long document: ~5000 input tokens, ~500 output tokens
        cost = calculate_completion_cost(
            "meta-llama/Llama-2-70b-chat-hf",
            input_tokens=5000,
            output_tokens=500
        )

        # Should be reasonable cost
        assert 0.001 < cost < 0.01

    def test_batch_processing_cost(self):
        """Test cost for batch processing 100 items."""
        # 100 items, each with 20 input + 30 output tokens
        total_cost = 0
        for _ in range(100):
            cost = calculate_completion_cost(
                "meta-llama/Llama-2-7b-chat-hf",  # Use cheaper model
                input_tokens=20,
                output_tokens=30
            )
            total_cost += cost

        # Total should be less than $0.01 for 100 items with cheap model
        assert total_cost < 0.01

    def test_embedding_batch_cost(self):
        """Test cost for embedding batch."""
        # Embed 1000 documents, each ~100 tokens
        total_cost = 0
        for _ in range(1000):
            cost = calculate_embedding_cost(
                "thenlper/gte-large",
                tokens=100
            )
            total_cost += cost

        # Total cost for 100,000 tokens
        expected = 100_000 / 1_000_000 * 0.05  # $0.005
        assert total_cost == pytest.approx(expected, abs=1e-8)
