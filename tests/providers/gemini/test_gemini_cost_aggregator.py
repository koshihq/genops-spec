#!/usr/bin/env python3
"""
Test suite for GenOps Gemini cost aggregation.

This module tests the cost aggregation functionality including:
- Context manager patterns for multi-operation tracking
- Budget monitoring and alerts
- Cost optimization recommendations
- Performance metrics and analysis
"""

import time
from unittest.mock import patch

import pytest

from genops.providers.gemini_cost_aggregator import (
    CostAlertLevel,
    GeminiCostContext,
    GeminiCostSummary,
    GeminiOperation,
    aggregate_multiple_contexts,
    create_gemini_cost_context,
)


class TestGeminiOperation:
    """Test GeminiOperation data class."""

    def test_operation_creation(self):
        """Test creating a GeminiOperation."""
        operation = GeminiOperation(
            operation_id="test-op-1",
            model_id="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=800.5,
            cost_usd=0.00155,
            timestamp=time.time(),
            governance_attributes={"team": "test-team", "project": "test-project"},
            operation_type="text_generation"
        )

        assert operation.operation_id == "test-op-1"
        assert operation.model_id == "gemini-2.5-flash"
        assert operation.input_tokens == 1000
        assert operation.output_tokens == 500
        assert operation.latency_ms == 800.5
        assert operation.cost_usd == 0.00155
        assert operation.governance_attributes["team"] == "test-team"
        assert operation.operation_type == "text_generation"

    def test_operation_with_context_cache(self):
        """Test creating operation with context cache tokens."""
        operation = GeminiOperation(
            operation_id="test-op-2",
            model_id="gemini-2.5-pro",
            input_tokens=1500,
            output_tokens=750,
            latency_ms=1200.0,
            cost_usd=0.00625,
            timestamp=time.time(),
            context_cache_tokens=2000
        )

        assert operation.context_cache_tokens == 2000


class TestGeminiCostSummary:
    """Test GeminiCostSummary functionality."""

    def test_cost_summary_creation(self):
        """Test creating a cost summary."""
        operations = [
            GeminiOperation(
                operation_id="op-1",
                model_id="gemini-2.5-flash",
                input_tokens=1000,
                output_tokens=500,
                latency_ms=800.0,
                cost_usd=0.00155,
                timestamp=time.time()
            ),
            GeminiOperation(
                operation_id="op-2",
                model_id="gemini-2.5-pro",
                input_tokens=800,
                output_tokens=400,
                latency_ms=1200.0,
                cost_usd=0.005,
                timestamp=time.time()
            )
        ]

        summary = GeminiCostSummary(
            total_cost=0.00655,
            currency="USD",
            total_operations=2,
            unique_models={"gemini-2.5-flash", "gemini-2.5-pro"},
            cost_by_model={"gemini-2.5-flash": 0.00155, "gemini-2.5-pro": 0.005},
            cost_by_operation_type={"text_generation": 0.00655},
            total_input_tokens=1800,
            total_output_tokens=900,
            total_latency_ms=2000.0,
            operations=operations,
            governance_attributes={"team": "test-team"}
        )

        assert summary.total_cost == 0.00655
        assert summary.total_operations == 2
        assert len(summary.unique_models) == 2
        assert "gemini-2.5-flash" in summary.unique_models
        assert "gemini-2.5-pro" in summary.unique_models

    def test_average_cost_calculation(self):
        """Test average cost per operation calculation."""
        summary = GeminiCostSummary(
            total_cost=0.01,
            currency="USD",
            total_operations=5,
            unique_models=set(),
            cost_by_model={},
            cost_by_operation_type={},
            total_input_tokens=0,
            total_output_tokens=0,
            total_latency_ms=0.0,
            operations=[],
            governance_attributes={}
        )

        assert summary.get_average_cost_per_operation() == 0.002  # 0.01 / 5

    def test_average_cost_with_zero_operations(self):
        """Test average cost calculation with zero operations."""
        summary = GeminiCostSummary(
            total_cost=0.0,
            currency="USD",
            total_operations=0,
            unique_models=set(),
            cost_by_model={},
            cost_by_operation_type={},
            total_input_tokens=0,
            total_output_tokens=0,
            total_latency_ms=0.0,
            operations=[],
            governance_attributes={}
        )

        assert summary.get_average_cost_per_operation() == 0.0

    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        summary = GeminiCostSummary(
            total_cost=0.0,
            currency="USD",
            total_operations=3,
            unique_models=set(),
            cost_by_model={},
            cost_by_operation_type={},
            total_input_tokens=0,
            total_output_tokens=0,
            total_latency_ms=3000.0,
            operations=[],
            governance_attributes={}
        )

        assert summary.get_average_latency_ms() == 1000.0  # 3000 / 3

    def test_cost_efficiency_score(self):
        """Test cost efficiency score calculation."""
        summary = GeminiCostSummary(
            total_cost=0.001,
            currency="USD",
            total_operations=1,
            unique_models=set(),
            cost_by_model={},
            cost_by_operation_type={},
            total_input_tokens=1000,
            total_output_tokens=500,
            total_latency_ms=0.0,
            operations=[],
            governance_attributes={}
        )

        # Cost per 1k tokens: (0.001 / 1500) * 1000 = 0.667
        expected_score = (0.001 / 1500) * 1000
        assert abs(summary.get_cost_efficiency_score() - expected_score) < 0.001


class TestGeminiCostContext:
    """Test GeminiCostContext functionality."""

    def test_context_initialization(self):
        """Test cost context initialization."""
        context = GeminiCostContext(
            context_id="test-context",
            budget_limit=5.0,
            enable_optimization=True,
            enable_alerts=True,
            governance_attributes={"team": "test-team"}
        )

        assert context.context_id == "test-context"
        assert context.budget_limit == 5.0
        assert context.enable_optimization is True
        assert context.enable_alerts is True
        assert context.governance_attributes["team"] == "test-team"
        assert len(context.operations) == 0
        assert context.total_cost == 0.0

    def test_context_manager_protocol(self):
        """Test context manager enter/exit protocol."""
        with patch('genops.providers.gemini_cost_aggregator.GENOPS_AVAILABLE', False):
            context = GeminiCostContext("test-context")

            # Test __enter__
            entered_context = context.__enter__()
            assert entered_context is context
            assert context.start_time is not None

            # Test __exit__
            context.__exit__(None, None, None)
            assert context.end_time is not None
            assert context.end_time >= context.start_time

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_add_operation(self, mock_calculate_cost):
        """Test adding an operation to the context."""
        mock_calculate_cost.return_value = 0.00155

        context = GeminiCostContext("test-context")

        operation = context.add_operation(
            operation_id="test-op-1",
            model_id="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=800.0,
            operation_type="text_generation",
            governance_attributes={"project": "test-project"}
        )

        assert isinstance(operation, GeminiOperation)
        assert operation.operation_id == "test-op-1"
        assert operation.model_id == "gemini-2.5-flash"
        assert operation.cost_usd == 0.00155

        # Check that operation was added to context
        assert len(context.operations) == 1
        assert context.total_cost == 0.00155

        # Check governance attributes merging
        assert operation.governance_attributes["project"] == "test-project"

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_multiple_operations(self, mock_calculate_cost):
        """Test adding multiple operations."""
        mock_calculate_cost.side_effect = [0.001, 0.002, 0.003]

        context = GeminiCostContext("test-context")

        # Add three operations
        context.add_operation("op-1", "gemini-2.5-flash", 1000, 500, 800.0)
        context.add_operation("op-2", "gemini-2.5-pro", 800, 400, 1200.0)
        context.add_operation("op-3", "gemini-2.5-flash", 1200, 600, 900.0)

        assert len(context.operations) == 3
        assert context.total_cost == 0.006  # 0.001 + 0.002 + 0.003

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_get_current_summary(self, mock_calculate_cost):
        """Test getting current summary from context."""
        mock_calculate_cost.side_effect = [0.001, 0.002]

        context = GeminiCostContext("test-context", governance_attributes={"team": "test-team"})

        # Add operations
        context.add_operation("op-1", "gemini-2.5-flash", 1000, 500, 800.0)
        context.add_operation("op-2", "gemini-2.5-pro", 800, 400, 1200.0)

        summary = context.get_current_summary()

        assert isinstance(summary, GeminiCostSummary)
        assert summary.total_cost == 0.003
        assert summary.total_operations == 2
        assert len(summary.unique_models) == 2
        assert "gemini-2.5-flash" in summary.unique_models
        assert "gemini-2.5-pro" in summary.unique_models

        # Check cost by model aggregation
        assert summary.cost_by_model["gemini-2.5-flash"] == 0.001
        assert summary.cost_by_model["gemini-2.5-pro"] == 0.002

        # Check token aggregation
        assert summary.total_input_tokens == 1800  # 1000 + 800
        assert summary.total_output_tokens == 900   # 500 + 400
        assert summary.total_latency_ms == 2000.0   # 800 + 1200

    def test_get_current_summary_empty(self):
        """Test getting summary from empty context."""
        context = GeminiCostContext("test-context")

        summary = context.get_current_summary()

        assert summary.total_cost == 0.0
        assert summary.total_operations == 0
        assert len(summary.unique_models) == 0
        assert len(summary.operations) == 0

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_budget_alerts(self, mock_calculate_cost):
        """Test budget alert functionality."""
        # Set up costs that will trigger alerts
        mock_calculate_cost.side_effect = [2.5, 1.25, 1.25]  # Total = 5.0

        context = GeminiCostContext(
            context_id="budget-test",
            budget_limit=5.0,
            enable_alerts=True
        )

        # Add operations that gradually approach budget
        context.add_operation("op-1", "gemini-2.5-pro", 1000, 500, 800.0)  # 50% budget
        assert CostAlertLevel.INFO in context.budget_warnings_sent

        context.add_operation("op-2", "gemini-2.5-pro", 800, 400, 900.0)   # 75% budget
        assert CostAlertLevel.WARNING in context.budget_warnings_sent

        context.add_operation("op-3", "gemini-2.5-pro", 800, 400, 900.0)   # 100% budget
        assert CostAlertLevel.BUDGET_EXCEEDED in context.budget_warnings_sent

        # Check that alerts were created
        assert len(context.cost_alerts) >= 3

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_budget_alerts_disabled(self, mock_calculate_cost):
        """Test that budget alerts can be disabled."""
        mock_calculate_cost.return_value = 6.0  # Over budget

        context = GeminiCostContext(
            context_id="no-alerts-test",
            budget_limit=5.0,
            enable_alerts=False  # Disabled
        )

        context.add_operation("op-1", "gemini-2.5-pro", 1000, 500, 800.0)

        # No alerts should be created
        assert len(context.cost_alerts) == 0
        assert len(context.budget_warnings_sent) == 0

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_model_performance_comparison(self, mock_calculate_cost):
        """Test model performance comparison."""
        mock_calculate_cost.side_effect = [0.001, 0.005, 0.001]

        context = GeminiCostContext("perf-test")

        # Add operations with different models
        context.add_operation("op-1", "gemini-2.5-flash", 1000, 500, 800.0)
        context.add_operation("op-2", "gemini-2.5-pro", 1000, 500, 1200.0)
        context.add_operation("op-3", "gemini-2.5-flash", 1000, 500, 900.0)

        performance = context.get_model_performance_comparison()

        assert "gemini-2.5-flash" in performance
        assert "gemini-2.5-pro" in performance

        flash_perf = performance["gemini-2.5-flash"]
        pro_perf = performance["gemini-2.5-pro"]

        assert flash_perf["operations_count"] == 2
        assert pro_perf["operations_count"] == 1

        assert flash_perf["total_cost"] == 0.002  # 0.001 + 0.001
        assert pro_perf["total_cost"] == 0.005

        assert flash_perf["average_latency_ms"] == 850.0  # (800 + 900) / 2
        assert pro_perf["average_latency_ms"] == 1200.0


class TestCreateGeminiCostContext:
    """Test create_gemini_cost_context function."""

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_context_manager_function(self, mock_calculate_cost):
        """Test create_gemini_cost_context as context manager."""
        mock_calculate_cost.return_value = 0.001

        with create_gemini_cost_context(
            "test-context",
            budget_limit=1.0,
            team="test-team",
            project="test-project"
        ) as context:
            assert isinstance(context, GeminiCostContext)
            assert context.context_id == "test-context"
            assert context.budget_limit == 1.0
            assert context.governance_attributes["team"] == "test-team"
            assert context.governance_attributes["project"] == "test-project"

            # Add operation within context
            context.add_operation("op-1", "gemini-2.5-flash", 1000, 500, 800.0)

    def test_context_manager_exception_handling(self):
        """Test context manager with exception."""
        with patch('genops.providers.gemini_cost_aggregator.GENOPS_AVAILABLE', False):
            try:
                with create_gemini_cost_context("test-context") as context:
                    raise ValueError("Test exception")
            except ValueError as e:
                assert str(e) == "Test exception"

            # Context should still be properly finalized


class TestAggregateMultipleContexts:
    """Test aggregating multiple cost contexts."""

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_aggregate_two_contexts(self, mock_calculate_cost):
        """Test aggregating two contexts."""
        mock_calculate_cost.side_effect = [0.001, 0.002, 0.003, 0.004]

        # Create first context
        context1 = GeminiCostContext("context-1", governance_attributes={"team": "team-1"})
        context1.add_operation("op-1", "gemini-2.5-flash", 1000, 500, 800.0)
        context1.add_operation("op-2", "gemini-2.5-pro", 800, 400, 1200.0)

        # Create second context
        context2 = GeminiCostContext("context-2", governance_attributes={"team": "team-2"})
        context2.add_operation("op-3", "gemini-2.5-flash", 1200, 600, 900.0)
        context2.add_operation("op-4", "gemini-2.5-pro", 900, 450, 1000.0)

        # Aggregate
        aggregated = aggregate_multiple_contexts([context1, context2])

        assert isinstance(aggregated, GeminiCostSummary)
        assert aggregated.total_cost == 0.01  # 0.001 + 0.002 + 0.003 + 0.004
        assert aggregated.total_operations == 4
        assert len(aggregated.unique_models) == 2

        # Check token aggregation
        assert aggregated.total_input_tokens == 3900   # 1000+800+1200+900
        assert aggregated.total_output_tokens == 1950  # 500+400+600+450

        # Check cost by model
        expected_flash_cost = 0.001 + 0.003  # op-1 + op-3
        expected_pro_cost = 0.002 + 0.004    # op-2 + op-4
        assert aggregated.cost_by_model["gemini-2.5-flash"] == expected_flash_cost
        assert aggregated.cost_by_model["gemini-2.5-pro"] == expected_pro_cost

    def test_aggregate_empty_contexts(self):
        """Test aggregating empty contexts."""
        context1 = GeminiCostContext("context-1")
        context2 = GeminiCostContext("context-2")

        aggregated = aggregate_multiple_contexts([context1, context2])

        assert aggregated.total_cost == 0.0
        assert aggregated.total_operations == 0
        assert len(aggregated.unique_models) == 0
        assert len(aggregated.operations) == 0

    def test_aggregate_no_contexts(self):
        """Test aggregating empty context list."""
        aggregated = aggregate_multiple_contexts([])

        assert aggregated.total_cost == 0.0
        assert aggregated.total_operations == 0
        assert len(aggregated.unique_models) == 0


class TestCostContextOptimization:
    """Test optimization features in cost context."""

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_optimization_recommendations_generation(self, mock_calculate_cost):
        """Test generation of optimization recommendations."""
        # Set up costs to trigger recommendations
        mock_calculate_cost.side_effect = [0.005, 0.001]  # Pro expensive, Flash cheap

        with create_gemini_cost_context("opt-test", enable_optimization=True) as context:
            # Use expensive Pro model
            context.add_operation("op-1", "gemini-2.5-pro", 1000, 500, 1200.0)
            # Use cheaper Flash model
            context.add_operation("op-2", "gemini-2.5-flash", 1000, 500, 800.0)

        summary = context.get_current_summary()

        # Should have generated optimization recommendations
        assert len(summary.optimization_recommendations) > 0

        # Should recommend using Flash instead of Pro
        recommendations_text = " ".join(summary.optimization_recommendations)
        assert "Flash" in recommendations_text or "flash" in recommendations_text

    @patch('genops.providers.gemini_cost_aggregator.calculate_gemini_cost')
    def test_high_token_usage_recommendation(self, mock_calculate_cost):
        """Test recommendation for high token usage."""
        mock_calculate_cost.return_value = 0.01

        with create_gemini_cost_context("token-test", enable_optimization=True) as context:
            # Add operation with high token usage
            context.add_operation("op-1", "gemini-2.5-flash", 3000, 2000, 1500.0)  # High tokens

        summary = context.get_current_summary()

        # Should recommend prompt optimization or context caching
        recommendations_text = " ".join(summary.optimization_recommendations)
        assert "token" in recommendations_text.lower() or "optimization" in recommendations_text.lower()

    def test_optimization_disabled(self):
        """Test that optimization can be disabled."""
        with create_gemini_cost_context("no-opt-test", enable_optimization=False) as context:
            pass

        summary = context.get_current_summary()

        # Should have no optimization recommendations when disabled
        assert len(summary.optimization_recommendations) == 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
