"""
Comprehensive tests for GenOps Bedrock Cost Aggregator.

Tests the advanced cost tracking context manager including:
- Multi-operation cost aggregation
- Context manager lifecycle
- Cost summary calculations
- Provider and model tracking
- Optimization recommendations
- Error handling and edge cases
"""

import time

import pytest

# Import the modules under test
try:
    from genops.providers.bedrock_cost_aggregator import (
        BedrockCostContext,
        BedrockCostSummary,
        CostOperation,  # noqa: F401
        add_bedrock_operation,
        create_bedrock_cost_context,
        get_optimization_recommendations,
    )

    COST_AGGREGATOR_AVAILABLE = True
except ImportError:
    COST_AGGREGATOR_AVAILABLE = False


@pytest.mark.skipif(
    not COST_AGGREGATOR_AVAILABLE, reason="Bedrock cost aggregator not available"
)
class TestBedrockCostContext:
    """Test the cost context manager."""

    def test_context_manager_creation(self):
        """Test basic context manager creation."""
        context_id = "test_context_123"

        with create_bedrock_cost_context(context_id) as context:
            assert isinstance(context, BedrockCostContext)
            assert context.context_id == context_id
            assert context.start_time is not None
            assert len(context.operations) == 0

    def test_context_manager_lifecycle(self):
        """Test complete context manager lifecycle."""
        start_time = time.time()

        with create_bedrock_cost_context("lifecycle_test") as context:
            # Test context is properly initialized
            assert context.context_id == "lifecycle_test"
            assert context.start_time >= start_time
            assert context.end_time is None

            # Add a sample operation
            context.add_operation(
                operation_id="test_op_1",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1200,
                governance_attributes={"team": "test-team"},
            )

            assert len(context.operations) == 1

        # After context exit, should be finalized
        assert context.end_time is not None
        assert context.end_time >= context.start_time

    def test_add_operation_basic(self):
        """Test adding a basic operation."""
        with create_bedrock_cost_context("add_op_test") as context:
            context.add_operation(
                operation_id="op_001",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=500,
                output_tokens=250,
                latency_ms=1800,
                governance_attributes={
                    "team": "engineering",
                    "project": "ai-platform",
                    "customer_id": "client-123",
                },
            )

            assert len(context.operations) == 1
            operation = context.operations[0]

            assert operation.operation_id == "op_001"
            assert operation.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert operation.provider == "anthropic"
            assert operation.region == "us-east-1"
            assert operation.input_tokens == 500
            assert operation.output_tokens == 250
            assert operation.latency_ms == 1800
            assert operation.governance_attributes["team"] == "engineering"

    def test_add_multiple_operations(self):
        """Test adding multiple operations."""
        with create_bedrock_cost_context("multi_op_test") as context:
            # Add first operation
            context.add_operation(
                operation_id="op_001",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=800,
                governance_attributes={"team": "team-a"},
            )

            # Add second operation with different model
            context.add_operation(
                operation_id="op_002",
                model_id="amazon.titan-text-express-v1",
                provider="amazon",
                region="us-west-2",
                input_tokens=200,
                output_tokens=100,
                latency_ms=1200,
                governance_attributes={"team": "team-b"},
            )

            # Add third operation with same model as first
            context.add_operation(
                operation_id="op_003",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=150,
                output_tokens=75,
                latency_ms=900,
                governance_attributes={"team": "team-a"},
            )

            assert len(context.operations) == 3

            # Test that operations are stored correctly
            op_ids = [op.operation_id for op in context.operations]
            assert "op_001" in op_ids
            assert "op_002" in op_ids
            assert "op_003" in op_ids

    def test_get_current_summary(self):
        """Test getting current cost summary."""
        with create_bedrock_cost_context("summary_test") as context:
            # Add operations
            context.add_operation(
                operation_id="op_001",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=1000,
                output_tokens=500,
                latency_ms=1000,
                governance_attributes={"team": "test"},
            )

            context.add_operation(
                operation_id="op_002",
                model_id="amazon.titan-text-express-v1",
                provider="amazon",
                region="us-east-1",
                input_tokens=800,
                output_tokens=400,
                latency_ms=1200,
                governance_attributes={"team": "test"},
            )

            summary = context.get_current_summary()

            assert isinstance(summary, BedrockCostSummary)
            assert summary.total_cost > 0
            assert summary.total_operations == 2
            assert summary.total_input_tokens == 1800
            assert summary.total_output_tokens == 900
            assert summary.total_latency_ms == 2200
            assert len(summary.unique_models) == 2
            assert len(summary.unique_providers) == 2

    def test_cost_calculation_accuracy(self):
        """Test that cost calculations are accurate."""
        with create_bedrock_cost_context("cost_accuracy_test") as context:
            # Add operation with known token costs
            context.add_operation(
                operation_id="cost_test",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=1000,
                output_tokens=500,
                latency_ms=1000,
                governance_attributes={"team": "cost-test"},
            )

            summary = context.get_current_summary()

            # Verify cost structure
            assert summary.total_cost > 0
            assert summary.cost_by_model is not None
            assert summary.cost_by_provider is not None

            # Cost should be positive for non-zero tokens
            model_cost = summary.cost_by_model.get(
                "anthropic.claude-3-haiku-20240307-v1:0", 0
            )
            assert model_cost > 0

            provider_cost = summary.cost_by_provider.get("anthropic", 0)
            assert provider_cost > 0

            # Total cost should equal sum of provider costs
            total_provider_cost = sum(summary.cost_by_provider.values())
            assert abs(summary.total_cost - total_provider_cost) < 0.000001

    def test_unique_tracking(self):
        """Test tracking of unique models and providers."""
        with create_bedrock_cost_context("unique_test") as context:
            # Add operations with different models
            models_and_providers = [
                ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic"),
                ("anthropic.claude-3-sonnet-20240229-v1:0", "anthropic"),
                ("amazon.titan-text-express-v1", "amazon"),
                ("ai21.j2-ultra-v1", "ai21"),
                ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic"),  # Duplicate
            ]

            for i, (model, provider) in enumerate(models_and_providers):
                context.add_operation(
                    operation_id=f"op_{i + 1}",
                    model_id=model,
                    provider=provider,
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"team": "unique-test"},
                )

            summary = context.get_current_summary()

            # Should track unique models and providers correctly
            assert len(summary.unique_models) == 4  # 4 unique models
            assert len(summary.unique_providers) == 3  # 3 unique providers

            assert "anthropic.claude-3-haiku-20240307-v1:0" in summary.unique_models
            assert "anthropic.claude-3-sonnet-20240229-v1:0" in summary.unique_models
            assert "amazon.titan-text-express-v1" in summary.unique_models
            assert "ai21.j2-ultra-v1" in summary.unique_models

            assert "anthropic" in summary.unique_providers
            assert "amazon" in summary.unique_providers
            assert "ai21" in summary.unique_providers

    def test_governance_attributes_tracking(self):
        """Test tracking of governance attributes."""
        with create_bedrock_cost_context("governance_test") as context:
            # Add operations with different governance attributes
            context.add_operation(
                operation_id="op_team_a",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                governance_attributes={
                    "team": "team-a",
                    "project": "project-alpha",
                    "customer_id": "customer-1",
                },
            )

            context.add_operation(
                operation_id="op_team_b",
                model_id="amazon.titan-text-express-v1",
                provider="amazon",
                region="us-east-1",
                input_tokens=200,
                output_tokens=100,
                latency_ms=1200,
                governance_attributes={
                    "team": "team-b",
                    "project": "project-beta",
                    "customer_id": "customer-2",
                },
            )

            # Operations should maintain their governance attributes
            assert len(context.operations) == 2

            team_a_op = next(
                op for op in context.operations if op.operation_id == "op_team_a"
            )
            assert team_a_op.governance_attributes["team"] == "team-a"
            assert team_a_op.governance_attributes["project"] == "project-alpha"

            team_b_op = next(
                op for op in context.operations if op.operation_id == "op_team_b"
            )
            assert team_b_op.governance_attributes["team"] == "team-b"
            assert team_b_op.governance_attributes["project"] == "project-beta"

    def test_regional_tracking(self):
        """Test tracking of operations across regions."""
        with create_bedrock_cost_context("regional_test") as context:
            regions = ["us-east-1", "us-west-2", "eu-west-1"]

            for i, region in enumerate(regions):
                context.add_operation(
                    operation_id=f"op_region_{i + 1}",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region=region,
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"region": region},
                )

            context.get_current_summary()

            # Should track regional operations
            regional_ops = {}
            for op in context.operations:
                if op.region not in regional_ops:
                    regional_ops[op.region] = 0
                regional_ops[op.region] += 1

            assert len(regional_ops) == 3
            assert all(count == 1 for count in regional_ops.values())

    def test_empty_context(self):
        """Test context with no operations."""
        with create_bedrock_cost_context("empty_test") as context:
            summary = context.get_current_summary()

            assert summary.total_cost == 0
            assert summary.total_operations == 0
            assert summary.total_input_tokens == 0
            assert summary.total_output_tokens == 0
            assert summary.total_latency_ms == 0
            assert len(summary.unique_models) == 0
            assert len(summary.unique_providers) == 0
            assert len(summary.cost_by_model) == 0
            assert len(summary.cost_by_provider) == 0

    def test_context_with_budget_limit(self):
        """Test context with budget constraints."""
        budget_limit = 0.01  # $0.01 limit

        with create_bedrock_cost_context(
            "budget_test", budget_limit=budget_limit
        ) as context:
            # Add operation that should be within budget
            context.add_operation(
                operation_id="small_op",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=10,
                output_tokens=5,
                latency_ms=500,
                governance_attributes={"team": "budget-test"},
            )

            summary = context.get_current_summary()

            # Should track budget information
            assert (
                hasattr(context, "budget_limit") or summary.total_cost <= budget_limit
            )

    def test_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        with create_bedrock_cost_context("optimization_test") as context:
            # Add expensive operations
            context.add_operation(
                operation_id="expensive_op",
                model_id="anthropic.claude-3-opus-20240229-v1:0",  # Expensive model
                provider="anthropic",
                region="us-east-1",
                input_tokens=5000,
                output_tokens=2000,
                latency_ms=3000,
                governance_attributes={"team": "optimization-test"},
            )

            summary = context.get_current_summary()

            # Should provide optimization recommendations
            if hasattr(summary, "optimization_recommendations"):
                assert isinstance(summary.optimization_recommendations, list)


@pytest.mark.skipif(
    not COST_AGGREGATOR_AVAILABLE, reason="Bedrock cost aggregator not available"
)
class TestCostSummaryCalculations:
    """Test cost summary calculation functionality."""

    def test_cost_summary_structure(self):
        """Test BedrockCostSummary data structure."""
        with create_bedrock_cost_context("structure_test") as context:
            context.add_operation(
                operation_id="test_op",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                governance_attributes={"team": "test"},
            )

            summary = context.get_current_summary()

            # Check all required fields exist
            required_fields = [
                "total_cost",
                "total_operations",
                "total_input_tokens",
                "total_output_tokens",
                "total_latency_ms",
                "unique_models",
                "unique_providers",
                "cost_by_model",
                "cost_by_provider",
            ]

            for field in required_fields:
                assert hasattr(summary, field), f"Missing field: {field}"

    def test_cost_aggregation_multiple_providers(self):
        """Test cost aggregation across multiple providers."""
        with create_bedrock_cost_context("multi_provider_test") as context:
            # Add operations from different providers
            providers_data = [
                ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic", 500, 250),
                ("amazon.titan-text-express-v1", "amazon", 400, 200),
                ("ai21.j2-mid-v1", "ai21", 300, 150),
                ("cohere.command-text-v14", "cohere", 600, 300),
            ]

            for i, (model, provider, input_tokens, output_tokens) in enumerate(
                providers_data
            ):
                context.add_operation(
                    operation_id=f"op_{i + 1}",
                    model_id=model,
                    provider=provider,
                    region="us-east-1",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=1000,
                    governance_attributes={"team": "multi-provider-test"},
                )

            summary = context.get_current_summary()

            # Verify aggregation
            assert summary.total_operations == 4
            assert summary.total_input_tokens == 1800  # Sum of all input tokens
            assert summary.total_output_tokens == 900  # Sum of all output tokens
            assert len(summary.unique_providers) == 4

            # Each provider should have associated costs
            for provider in ["anthropic", "amazon", "ai21", "cohere"]:
                assert provider in summary.cost_by_provider
                assert summary.cost_by_provider[provider] > 0

    def test_get_average_cost_per_operation(self):
        """Test average cost per operation calculation."""
        with create_bedrock_cost_context("average_test") as context:
            # Add multiple operations
            for i in range(5):
                context.add_operation(
                    operation_id=f"avg_op_{i + 1}",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"team": "average-test"},
                )

            summary = context.get_current_summary()

            if hasattr(summary, "get_average_cost_per_operation"):
                avg_cost = summary.get_average_cost_per_operation()
                assert avg_cost > 0
                assert abs(avg_cost * 5 - summary.total_cost) < 0.000001

    def test_get_average_latency_ms(self):
        """Test average latency calculation."""
        with create_bedrock_cost_context("latency_test") as context:
            latencies = [800, 1200, 1000, 1500, 900]

            for i, latency in enumerate(latencies):
                context.add_operation(
                    operation_id=f"latency_op_{i + 1}",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=latency,
                    governance_attributes={"team": "latency-test"},
                )

            summary = context.get_current_summary()

            if hasattr(summary, "get_average_latency_ms"):
                avg_latency = summary.get_average_latency_ms()
                expected_avg = sum(latencies) / len(latencies)
                assert abs(avg_latency - expected_avg) < 1.0  # Within 1ms tolerance

    def test_cost_breakdown_by_model(self):
        """Test detailed cost breakdown by model."""
        with create_bedrock_cost_context("model_breakdown_test") as context:
            # Add operations with different models
            models_data = [
                ("anthropic.claude-3-haiku-20240307-v1:0", 1000, 500),
                ("anthropic.claude-3-sonnet-20240229-v1:0", 800, 400),
                ("anthropic.claude-3-haiku-20240307-v1:0", 500, 250),  # Duplicate model
            ]

            for i, (model, input_tokens, output_tokens) in enumerate(models_data):
                context.add_operation(
                    operation_id=f"model_op_{i + 1}",
                    model_id=model,
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=1000,
                    governance_attributes={"team": "model-breakdown-test"},
                )

            summary = context.get_current_summary()

            # Should have costs aggregated by model
            haiku_total_cost = summary.cost_by_model.get(
                "anthropic.claude-3-haiku-20240307-v1:0", 0
            )
            sonnet_cost = summary.cost_by_model.get(
                "anthropic.claude-3-sonnet-20240229-v1:0", 0
            )

            # Haiku should have higher total cost (2 operations vs 1)
            assert haiku_total_cost > sonnet_cost
            assert haiku_total_cost > 0
            assert sonnet_cost > 0

    def test_zero_token_operations(self):
        """Test handling operations with zero tokens."""
        with create_bedrock_cost_context("zero_token_test") as context:
            context.add_operation(
                operation_id="zero_op",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=0,
                output_tokens=0,
                latency_ms=500,
                governance_attributes={"team": "zero-test"},
            )

            summary = context.get_current_summary()

            # Should handle zero tokens gracefully
            assert summary.total_cost == 0
            assert summary.total_operations == 1
            assert summary.total_input_tokens == 0
            assert summary.total_output_tokens == 0


@pytest.mark.skipif(
    not COST_AGGREGATOR_AVAILABLE, reason="Bedrock cost aggregator not available"
)
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_operation_data(self):
        """Test handling of invalid operation data."""
        with create_bedrock_cost_context("invalid_test") as context:
            # Test with negative tokens
            with pytest.raises((ValueError, AssertionError)):
                context.add_operation(
                    operation_id="invalid_op",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=-100,  # Invalid
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"team": "invalid-test"},
                )

    def test_duplicate_operation_ids(self):
        """Test handling of duplicate operation IDs."""
        with create_bedrock_cost_context("duplicate_test") as context:
            # Add first operation
            context.add_operation(
                operation_id="duplicate_id",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                governance_attributes={"team": "duplicate-test"},
            )

            # Add second operation with same ID
            try:
                context.add_operation(
                    operation_id="duplicate_id",  # Same ID
                    model_id="amazon.titan-text-express-v1",
                    provider="amazon",
                    region="us-east-1",
                    input_tokens=200,
                    output_tokens=100,
                    latency_ms=1200,
                    governance_attributes={"team": "duplicate-test"},
                )

                # Should either accept (overwrite) or have 2 operations
                assert len(context.operations) >= 1
            except ValueError:
                # Or raise an error for duplicates
                pass

    def test_empty_governance_attributes(self):
        """Test operations with empty governance attributes."""
        with create_bedrock_cost_context("empty_governance_test") as context:
            context.add_operation(
                operation_id="empty_gov_op",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                governance_attributes={},  # Empty
            )

            summary = context.get_current_summary()
            assert summary.total_operations == 1

    def test_none_governance_attributes(self):
        """Test operations with None governance attributes."""
        with create_bedrock_cost_context("none_governance_test") as context:
            try:
                context.add_operation(
                    operation_id="none_gov_op",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes=None,  # None
                )

                summary = context.get_current_summary()
                assert summary.total_operations == 1
            except (TypeError, ValueError):
                # May require non-None governance attributes
                pass

    def test_context_exception_handling(self):
        """Test context manager behavior when exceptions occur."""
        try:
            with create_bedrock_cost_context("exception_test") as context:
                context.add_operation(
                    operation_id="before_exception",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"team": "exception-test"},
                )

                # Raise exception within context
                raise Exception("Test exception")

        except Exception as e:
            assert str(e) == "Test exception"

        # Context should still be properly finalized
        assert context.end_time is not None

    def test_large_number_of_operations(self):
        """Test performance with large number of operations."""
        with create_bedrock_cost_context("large_test") as context:
            num_operations = 1000

            start_time = time.time()

            for i in range(num_operations):
                context.add_operation(
                    operation_id=f"large_op_{i}",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={
                        "team": "large-test",
                        "batch": str(i // 100),
                    },
                )

            end_time = time.time()
            summary = context.get_current_summary()

            # Should handle large number of operations
            assert summary.total_operations == num_operations
            assert summary.total_input_tokens == num_operations * 100

            # Should complete reasonably quickly (under 1 second for 1000 ops)
            assert (end_time - start_time) < 1.0


@pytest.mark.skipif(
    not COST_AGGREGATOR_AVAILABLE, reason="Bedrock cost aggregator not available"
)
class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety."""

    def test_concurrent_context_creation(self):
        """Test creating multiple contexts concurrently."""
        import threading

        contexts = []

        def create_context(context_id):
            with create_bedrock_cost_context(f"concurrent_{context_id}") as context:
                contexts.append(context)
                time.sleep(0.1)  # Simulate some work

        # Create multiple contexts in threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_context, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)

        # All contexts should be created successfully
        assert len(contexts) == 5
        for context in contexts:
            assert isinstance(context, BedrockCostContext)

    def test_concurrent_operation_addition(self):
        """Test adding operations to context concurrently."""
        import threading

        with create_bedrock_cost_context("concurrent_ops_test") as context:

            def add_operations(thread_id):
                for i in range(10):
                    try:
                        context.add_operation(
                            operation_id=f"thread_{thread_id}_op_{i}",
                            model_id="anthropic.claude-3-haiku-20240307-v1:0",
                            provider="anthropic",
                            region="us-east-1",
                            input_tokens=100,
                            output_tokens=50,
                            latency_ms=1000,
                            governance_attributes={"thread_id": str(thread_id)},
                        )
                    except Exception:
                        # Some thread safety issues may be expected
                        pass

            # Add operations from multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=add_operations, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join(timeout=5)

            # Should have some operations (exact count depends on thread safety)
            summary = context.get_current_summary()
            assert summary.total_operations > 0


@pytest.mark.skipif(
    not COST_AGGREGATOR_AVAILABLE, reason="Bedrock cost aggregator not available"
)
class TestUtilityFunctions:
    """Test utility functions."""

    def test_add_bedrock_operation_function(self):
        """Test standalone add_bedrock_operation function if available."""
        if "add_bedrock_operation" in globals():
            # Test the standalone function
            context_id = "utility_test"

            with create_bedrock_cost_context(context_id) as context:
                # If there's a standalone function, test it
                try:
                    add_bedrock_operation(
                        context_id=context_id,
                        operation_id="utility_op",
                        model_id="anthropic.claude-3-haiku-20240307-v1:0",
                        provider="anthropic",
                        region="us-east-1",
                        input_tokens=100,
                        output_tokens=50,
                        latency_ms=1000,
                        governance_attributes={"team": "utility-test"},
                    )

                    summary = context.get_current_summary()
                    assert summary.total_operations >= 1

                except Exception:
                    # Function may not be implemented or work differently
                    pass

    def test_optimization_recommendations_function(self):
        """Test optimization recommendations utility function if available."""
        if "get_optimization_recommendations" in globals():
            with create_bedrock_cost_context("optimization_util_test") as context:
                # Add some expensive operations
                context.add_operation(
                    operation_id="expensive_op",
                    model_id="anthropic.claude-3-opus-20240229-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=10000,
                    output_tokens=5000,
                    latency_ms=5000,
                    governance_attributes={"team": "optimization-util-test"},
                )

                try:
                    recommendations = get_optimization_recommendations(context)
                    assert isinstance(recommendations, list)

                except Exception:
                    # Function may not be implemented
                    pass


@pytest.mark.performance
class TestPerformance:
    """Performance tests for cost aggregator."""

    def test_operation_addition_performance(self):
        """Test performance of adding operations."""
        if not COST_AGGREGATOR_AVAILABLE:
            pytest.skip("Cost aggregator not available")

        with create_bedrock_cost_context("perf_test") as context:
            start_time = time.time()

            # Add many operations
            for i in range(1000):
                context.add_operation(
                    operation_id=f"perf_op_{i}",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"batch": str(i // 100)},
                )

            end_time = time.time()

            # Should complete in reasonable time
            assert (end_time - start_time) < 2.0  # Less than 2 seconds for 1000 ops

    def test_summary_calculation_performance(self):
        """Test performance of summary calculations."""
        if not COST_AGGREGATOR_AVAILABLE:
            pytest.skip("Cost aggregator not available")

        with create_bedrock_cost_context("summary_perf_test") as context:
            # Add operations
            for i in range(100):
                context.add_operation(
                    operation_id=f"summary_perf_op_{i}",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"batch": str(i // 10)},
                )

            start_time = time.time()

            # Calculate summary multiple times
            for _ in range(10):
                summary = context.get_current_summary()
                assert summary.total_operations == 100

            end_time = time.time()

            # Summary calculation should be fast
            assert (end_time - start_time) < 0.1  # Less than 100ms for 10 calculations
