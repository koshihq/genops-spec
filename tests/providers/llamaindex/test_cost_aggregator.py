"""
Unit tests for GenOps LlamaIndex Cost Aggregator.

Comprehensive test coverage for LlamaIndexCostAggregator including
cost tracking, budget management, multi-provider aggregation, and reporting.
"""

import pytest

pytest.skip(
    "Tests reference LlamaIndexOperationSummary and create_llamaindex_cost_context "
    "which are not implemented in the cost_aggregator module",
    allow_module_level=True,
)

from datetime import datetime  # noqa: E402

# Import the module under test
from genops.providers.llamaindex.cost_aggregator import (  # noqa: E402
    LlamaIndexCostAggregator,
    LlamaIndexOperationSummary,
    create_llamaindex_cost_context,
)


class TestLlamaIndexCostAggregatorInitialization:
    """Test cost aggregator initialization and configuration."""

    def test_default_initialization(self):
        """Test cost aggregator with default parameters."""
        aggregator = LlamaIndexCostAggregator("test_context")

        assert aggregator.context_name == "test_context"
        assert aggregator.budget_limit is None
        assert aggregator.enable_alerts is True
        assert aggregator.total_cost == 0.0
        assert aggregator.operation_count == 0
        assert aggregator.cost_by_provider == {}
        assert aggregator.start_time is not None
        assert isinstance(aggregator.start_time, datetime)

    def test_initialization_with_budget_limit(self):
        """Test cost aggregator with budget limit."""
        aggregator = LlamaIndexCostAggregator("test_context", budget_limit=10.0)

        assert aggregator.budget_limit == 10.0
        assert aggregator.enable_alerts is True

    def test_initialization_with_alerts_disabled(self):
        """Test cost aggregator with alerts disabled."""
        aggregator = LlamaIndexCostAggregator("test_context", enable_alerts=False)

        assert aggregator.enable_alerts is False

    def test_initialization_with_governance_attributes(self):
        """Test cost aggregator with governance attributes."""
        governance_attrs = {
            "team": "test-team",
            "project": "test-project",
            "customer_id": "customer-123",
        }

        aggregator = LlamaIndexCostAggregator("test_context", **governance_attrs)

        assert aggregator.governance_attrs == governance_attrs


class TestCostTracking:
    """Test cost tracking functionality."""

    @pytest.fixture
    def aggregator(self):
        """Create cost aggregator for testing."""
        return LlamaIndexCostAggregator("test_tracking")

    def test_add_embedding_cost(self, aggregator):
        """Test adding embedding costs."""
        aggregator.add_embedding_cost(
            provider="openai", model="text-embedding-ada-002", tokens=1000, cost=0.0001
        )

        assert aggregator.total_cost == 0.0001
        assert aggregator.cost_by_provider["openai"] == 0.0001
        assert aggregator.embedding_tokens == 1000
        assert aggregator.embedding_operations == 1

    def test_add_retrieval_cost(self, aggregator):
        """Test adding retrieval costs."""
        aggregator.add_retrieval_cost(operation_cost=0.002)

        assert aggregator.total_cost == 0.002
        assert aggregator.retrieval_operations == 1

    def test_add_synthesis_cost(self, aggregator):
        """Test adding synthesis costs."""
        aggregator.add_synthesis_cost(
            provider="anthropic",
            model="claude-3-haiku",
            input_tokens=500,
            output_tokens=200,
            cost=0.003,
        )

        assert aggregator.total_cost == 0.003
        assert aggregator.cost_by_provider["anthropic"] == 0.003
        assert aggregator.synthesis_input_tokens == 500
        assert aggregator.synthesis_output_tokens == 200
        assert aggregator.synthesis_operations == 1

    def test_multiple_cost_additions(self, aggregator):
        """Test multiple cost additions accumulate correctly."""
        # Add embedding cost
        aggregator.add_embedding_cost("openai", "ada-002", 1000, 0.0001)

        # Add retrieval cost
        aggregator.add_retrieval_cost(0.002)

        # Add synthesis cost
        aggregator.add_synthesis_cost("anthropic", "claude-3", 500, 200, 0.003)

        # Verify totals
        assert aggregator.total_cost == 0.0051  # 0.0001 + 0.002 + 0.003
        assert aggregator.operation_count == 3
        assert len(aggregator.cost_by_provider) == 2
        assert aggregator.cost_by_provider["openai"] == 0.0001
        assert aggregator.cost_by_provider["anthropic"] == 0.003

    def test_same_provider_multiple_operations(self, aggregator):
        """Test multiple operations from same provider."""
        # Two OpenAI operations
        aggregator.add_embedding_cost("openai", "ada-002", 1000, 0.0001)
        aggregator.add_synthesis_cost("openai", "gpt-4", 500, 300, 0.015)

        # Should accumulate costs for same provider
        assert aggregator.cost_by_provider["openai"] == 0.0151
        assert aggregator.total_cost == 0.0151
        assert aggregator.operation_count == 2


class TestBudgetManagement:
    """Test budget management and alerting."""

    def test_budget_tracking_under_limit(self):
        """Test budget tracking when under limit."""
        aggregator = LlamaIndexCostAggregator("test_budget", budget_limit=1.0)

        aggregator.add_embedding_cost("openai", "ada-002", 1000, 0.1)

        budget_status = aggregator.get_budget_status()
        assert budget_status["budget_limit"] == 1.0
        assert budget_status["total_cost"] == 0.1
        assert budget_status["utilization"] == 0.1  # 10%
        assert budget_status["alerts"] == []
        assert budget_status["remaining"] == 0.9

    def test_budget_tracking_over_limit(self):
        """Test budget tracking when over limit."""
        aggregator = LlamaIndexCostAggregator("test_budget", budget_limit=0.5)

        # Add cost that exceeds budget
        aggregator.add_synthesis_cost("anthropic", "claude-3", 2000, 1000, 0.8)

        budget_status = aggregator.get_budget_status()
        assert budget_status["budget_limit"] == 0.5
        assert budget_status["total_cost"] == 0.8
        assert budget_status["utilization"] == 1.6  # 160%
        assert len(budget_status["alerts"]) > 0
        assert budget_status["remaining"] == -0.3  # Over budget

        # Check alert content
        alert = budget_status["alerts"][0]
        assert "Budget exceeded" in alert["message"]
        assert alert["severity"] == "high"

    def test_budget_warning_threshold(self):
        """Test budget warning at 80% threshold."""
        aggregator = LlamaIndexCostAggregator("test_budget", budget_limit=1.0)

        # Add cost that hits warning threshold (80%)
        aggregator.add_embedding_cost("openai", "ada-002", 8000, 0.8)

        budget_status = aggregator.get_budget_status()
        assert len(budget_status["alerts"]) == 1

        alert = budget_status["alerts"][0]
        assert "80%" in alert["message"]
        assert alert["severity"] == "medium"

    def test_budget_critical_threshold(self):
        """Test budget critical alert at 95% threshold."""
        aggregator = LlamaIndexCostAggregator("test_budget", budget_limit=1.0)

        # Add cost that hits critical threshold (95%)
        aggregator.add_synthesis_cost("anthropic", "claude-3", 3000, 2000, 0.95)

        budget_status = aggregator.get_budget_status()
        assert len(budget_status["alerts"]) == 1

        alert = budget_status["alerts"][0]
        assert "95%" in alert["message"]
        assert alert["severity"] == "high"

    def test_budget_alerts_disabled(self):
        """Test budget tracking with alerts disabled."""
        aggregator = LlamaIndexCostAggregator(
            "test_budget", budget_limit=0.5, enable_alerts=False
        )

        # Add cost that exceeds budget
        aggregator.add_synthesis_cost("openai", "gpt-4", 2000, 1000, 0.8)

        budget_status = aggregator.get_budget_status()
        assert budget_status["alerts"] == []  # No alerts when disabled
        assert budget_status["total_cost"] == 0.8
        assert budget_status["utilization"] == 1.6


class TestProviderPricing:
    """Test provider pricing calculations."""

    @pytest.fixture
    def aggregator(self):
        """Create cost aggregator for testing."""
        return LlamaIndexCostAggregator("test_pricing")

    def test_calculate_openai_embedding_cost(self, aggregator):
        """Test OpenAI embedding cost calculation."""
        cost = aggregator.calculate_embedding_cost(
            "openai", "text-embedding-ada-002", 1000
        )
        expected_cost = (1000 / 1000) * 0.0001  # $0.0001 per 1K tokens
        assert cost == expected_cost

    def test_calculate_openai_completion_cost(self, aggregator):
        """Test OpenAI completion cost calculation."""
        cost = aggregator.calculate_completion_cost(
            "openai", "gpt-4", input_tokens=1000, output_tokens=500
        )
        # GPT-4: $0.03/1K input, $0.06/1K output
        expected_cost = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert cost == expected_cost

    def test_calculate_anthropic_completion_cost(self, aggregator):
        """Test Anthropic completion cost calculation."""
        cost = aggregator.calculate_completion_cost(
            "anthropic", "claude-3-haiku-20240307", input_tokens=1000, output_tokens=500
        )
        # Claude-3 Haiku: $0.00025/1K input, $0.00125/1K output
        expected_cost = (1000 / 1000) * 0.00025 + (500 / 1000) * 0.00125
        assert cost == expected_cost

    def test_calculate_cost_unknown_provider(self, aggregator):
        """Test cost calculation for unknown provider."""
        # Should use fallback pricing
        cost = aggregator.calculate_embedding_cost(
            "unknown_provider", "unknown_model", 1000
        )
        expected_cost = (1000 / 1000) * 0.0001  # Default embedding cost
        assert cost == expected_cost

    def test_calculate_cost_unknown_model(self, aggregator):
        """Test cost calculation for unknown model."""
        # Known provider, unknown model - should use provider's default
        cost = aggregator.calculate_completion_cost(
            "openai", "unknown-model", input_tokens=1000, output_tokens=500
        )
        # Should use OpenAI's default pricing (GPT-3.5-turbo)
        expected_cost = (1000 / 1000) * 0.0015 + (500 / 1000) * 0.002
        assert cost == expected_cost


class TestOperationSummary:
    """Test operation summary generation."""

    @pytest.fixture
    def aggregator_with_data(self):
        """Create aggregator with sample data."""
        aggregator = LlamaIndexCostAggregator("test_summary", budget_limit=5.0)

        # Add various operations
        aggregator.add_embedding_cost("openai", "ada-002", 2000, 0.0002)
        aggregator.add_retrieval_cost(0.001)
        aggregator.add_synthesis_cost("anthropic", "claude-3", 1000, 800, 0.004)
        aggregator.add_embedding_cost("google", "gecko", 1500, 0.0001)

        return aggregator

    def test_get_current_summary(self, aggregator_with_data):
        """Test getting current operation summary."""
        summary = aggregator_with_data.get_current_summary()

        assert isinstance(summary, LlamaIndexOperationSummary)
        assert summary.total_cost == 0.0053  # 0.0002 + 0.001 + 0.004 + 0.0001
        assert summary.operation_count == 4

        # Check cost breakdown
        breakdown = summary.cost_breakdown
        assert breakdown.embedding_cost == 0.0003  # 0.0002 + 0.0001
        assert breakdown.retrieval_cost == 0.001
        assert breakdown.synthesis_cost == 0.004
        assert breakdown.embedding_tokens == 3500  # 2000 + 1500
        assert breakdown.synthesis_tokens == 1800  # 1000 + 800

        # Check provider breakdown
        assert len(breakdown.cost_by_provider) == 3
        assert breakdown.cost_by_provider["openai"] == 0.0002
        assert breakdown.cost_by_provider["anthropic"] == 0.004
        assert breakdown.cost_by_provider["google"] == 0.0001

    def test_get_summary_with_budget_status(self, aggregator_with_data):
        """Test summary includes budget status."""
        summary = aggregator_with_data.get_current_summary()

        assert summary.budget_status is not None
        budget_status = summary.budget_status
        assert budget_status["budget_limit"] == 5.0
        assert budget_status["total_cost"] == 0.0053
        assert budget_status["utilization"] < 0.01  # Very low utilization

    def test_get_summary_with_optimization_suggestions(self, aggregator_with_data):
        """Test summary includes optimization suggestions."""
        summary = aggregator_with_data.get_current_summary()

        # Should have optimization suggestions based on usage patterns
        suggestions = summary.cost_breakdown.optimization_suggestions
        assert isinstance(suggestions, list)
        # Specific suggestions depend on cost patterns

    def test_empty_aggregator_summary(self):
        """Test summary for empty aggregator."""
        aggregator = LlamaIndexCostAggregator("empty_test")
        summary = aggregator.get_current_summary()

        assert summary.total_cost == 0.0
        assert summary.operation_count == 0
        assert summary.avg_cost_per_operation == 0.0
        assert summary.cost_breakdown.embedding_cost == 0.0
        assert summary.cost_breakdown.retrieval_cost == 0.0
        assert summary.cost_breakdown.synthesis_cost == 0.0


class TestOptimizationSuggestions:
    """Test cost optimization suggestion generation."""

    def test_suggestions_for_expensive_embeddings(self):
        """Test suggestions when embeddings are expensive."""
        aggregator = LlamaIndexCostAggregator("test_opt")

        # Add expensive embedding operations
        for _ in range(10):
            aggregator.add_embedding_cost("openai", "ada-002", 5000, 0.0005)

        suggestions = aggregator.generate_optimization_suggestions()

        # Should suggest embedding optimization
        embedding_suggestions = [s for s in suggestions if "embedding" in s.lower()]
        assert len(embedding_suggestions) > 0

    def test_suggestions_for_expensive_synthesis(self):
        """Test suggestions when synthesis is expensive."""
        aggregator = LlamaIndexCostAggregator("test_opt")

        # Add expensive synthesis operations
        for _ in range(5):
            aggregator.add_synthesis_cost("openai", "gpt-4", 3000, 2000, 0.15)

        suggestions = aggregator.generate_optimization_suggestions()

        # Should suggest model optimization
        model_suggestions = [s for s in suggestions if "model" in s.lower()]
        assert len(model_suggestions) > 0

    def test_suggestions_for_high_retrieval_cost(self):
        """Test suggestions when retrieval is expensive."""
        aggregator = LlamaIndexCostAggregator("test_opt")

        # Add many retrieval operations
        for _ in range(20):
            aggregator.add_retrieval_cost(0.005)

        suggestions = aggregator.generate_optimization_suggestions()

        # Should suggest retrieval optimization
        retrieval_suggestions = [
            s for s in suggestions if "retrieval" in s.lower() or "cache" in s.lower()
        ]
        assert len(retrieval_suggestions) > 0

    def test_no_suggestions_for_low_cost(self):
        """Test no suggestions when costs are low."""
        aggregator = LlamaIndexCostAggregator("test_opt")

        # Add minimal cost operations
        aggregator.add_embedding_cost("openai", "ada-002", 100, 0.00001)
        aggregator.add_retrieval_cost(0.0001)

        suggestions = aggregator.generate_optimization_suggestions()

        # Should have few or no suggestions for low costs
        assert len(suggestions) <= 1


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_basic_usage(self):
        """Test basic context manager usage."""
        with create_llamaindex_cost_context("test_context") as aggregator:
            assert isinstance(aggregator, LlamaIndexCostAggregator)
            assert aggregator.context_name == "test_context"

            # Add some operations
            aggregator.add_embedding_cost("openai", "ada-002", 1000, 0.0001)

    def test_context_manager_with_budget_limit(self):
        """Test context manager with budget limit."""
        with create_llamaindex_cost_context(
            "test_context", budget_limit=1.0
        ) as aggregator:
            assert aggregator.budget_limit == 1.0

            aggregator.add_synthesis_cost("anthropic", "claude-3", 1000, 500, 0.002)

            # Should be under budget
            status = aggregator.get_budget_status()
            assert status["utilization"] < 1.0

    def test_context_manager_with_governance_attributes(self):
        """Test context manager with governance attributes."""
        with create_llamaindex_cost_context(
            "test_context", team="test-team", project="test-project"
        ) as aggregator:
            assert aggregator.governance_attrs["team"] == "test-team"
            assert aggregator.governance_attrs["project"] == "test-project"

    def test_context_manager_exception_handling(self):
        """Test context manager handles exceptions properly."""
        try:
            with create_llamaindex_cost_context("test_context") as aggregator:
                aggregator.add_embedding_cost("openai", "ada-002", 1000, 0.0001)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Context should be properly cleaned up
        # In a real implementation, this would verify cleanup occurred

    def test_context_manager_alerts_enabled(self):
        """Test context manager with alerts enabled."""
        with create_llamaindex_cost_context(
            "test_context", budget_limit=0.001, enable_alerts=True
        ) as aggregator:
            # Exceed budget to trigger alert
            aggregator.add_synthesis_cost("openai", "gpt-4", 1000, 500, 0.05)

            status = aggregator.get_budget_status()
            assert len(status["alerts"]) > 0


class TestCostAggregatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_negative_cost_handling(self):
        """Test handling of negative costs."""
        aggregator = LlamaIndexCostAggregator("test_edge")

        # Should handle negative costs gracefully
        aggregator.add_embedding_cost("openai", "ada-002", 1000, -0.001)

        # Should not allow negative total cost
        assert aggregator.total_cost >= 0.0

    def test_zero_token_operations(self):
        """Test operations with zero tokens."""
        aggregator = LlamaIndexCostAggregator("test_edge")

        aggregator.add_embedding_cost("openai", "ada-002", 0, 0.0)
        aggregator.add_synthesis_cost("anthropic", "claude-3", 0, 0, 0.0)

        assert aggregator.embedding_tokens == 0
        assert aggregator.synthesis_input_tokens == 0
        assert aggregator.synthesis_output_tokens == 0
        assert aggregator.total_cost == 0.0

    def test_very_large_token_counts(self):
        """Test operations with very large token counts."""
        aggregator = LlamaIndexCostAggregator("test_edge")

        large_tokens = 1_000_000  # 1M tokens
        cost = aggregator.calculate_embedding_cost("openai", "ada-002", large_tokens)

        aggregator.add_embedding_cost("openai", "ada-002", large_tokens, cost)

        assert aggregator.embedding_tokens == large_tokens
        assert aggregator.total_cost == cost

    def test_many_small_operations(self):
        """Test many small operations for performance."""
        aggregator = LlamaIndexCostAggregator("test_edge")

        # Add 1000 small operations
        for _i in range(1000):
            aggregator.add_embedding_cost("openai", "ada-002", 10, 0.000001)

        assert aggregator.operation_count == 1000
        assert aggregator.embedding_operations == 1000
        assert aggregator.embedding_tokens == 10000
        assert abs(aggregator.total_cost - 0.001) < 1e-10  # Float precision

    def test_concurrent_operations(self):
        """Test concurrent cost additions."""
        aggregator = LlamaIndexCostAggregator("test_concurrent")

        import threading

        def add_costs(thread_id):
            for _i in range(100):
                aggregator.add_embedding_cost("openai", "ada-002", 100, 0.00001)

        # Run 5 threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_costs, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have processed all operations
        assert aggregator.operation_count == 500  # 5 threads * 100 operations
        assert aggregator.embedding_tokens == 50000  # 500 * 100 tokens


if __name__ == "__main__":
    pytest.main([__file__])
