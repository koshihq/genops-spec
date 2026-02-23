"""
Unit tests for Kubetorch cost aggregation module.

Tests cover:
- ComputeResourceCost dataclass
- ComputeCostSummary aggregation
- KubetorchCostAggregator functionality
- Context manager pattern
- Multi-resource cost tracking
"""

import time

import pytest
from src.genops.providers.kubetorch.cost_aggregator import (
    ComputeCostSummary,
    ComputeResourceCost,
    KubetorchCostAggregator,
    create_compute_cost_context,
    get_cost_aggregator,
    reset_cost_aggregator,
)


class TestComputeResourceCost:
    """Test ComputeResourceCost dataclass."""

    def test_dataclass_creation(self):
        """Test creating ComputeResourceCost instance."""
        cost = ComputeResourceCost(
            resource_type="gpu", instance_type="a100", quantity=8.0, cost=262.16
        )
        assert cost.resource_type == "gpu"
        assert cost.instance_type == "a100"
        assert cost.quantity == 8.0
        assert cost.cost == 262.16
        assert cost.currency == "USD"

    def test_str_representation(self):
        """Test string representation."""
        cost = ComputeResourceCost(
            resource_type="gpu", instance_type="a100", quantity=8.0, cost=262.16
        )
        str_repr = str(cost)
        assert "gpu" in str_repr
        assert "a100" in str_repr
        assert "8.00" in str_repr
        assert "262.16" in str_repr


class TestComputeCostSummary:
    """Test ComputeCostSummary aggregation."""

    def test_summary_initialization(self):
        """Test default summary initialization."""
        summary = ComputeCostSummary()
        assert summary.total_cost == 0.0
        assert summary.total_gpu_hours == 0.0
        assert len(summary.resource_costs) == 0

    def test_add_single_resource(self):
        """Test adding single resource cost."""
        summary = ComputeCostSummary()
        gpu_cost = ComputeResourceCost(
            resource_type="gpu", instance_type="a100", quantity=8.0, cost=262.16
        )
        summary.add_resource_cost(gpu_cost)

        assert summary.total_cost == 262.16
        assert summary.total_gpu_hours == 8.0
        assert len(summary.resource_costs) == 1
        assert summary.cost_by_resource_type["gpu"] == 262.16

    def test_add_multiple_resources(self):
        """Test adding multiple resource costs."""
        summary = ComputeCostSummary()

        gpu_cost = ComputeResourceCost("gpu", "a100", 8.0, 262.16)
        storage_cost = ComputeResourceCost("storage", "storage", 2400.0, 0.08)
        network_cost = ComputeResourceCost("network", "network", 50.0, 4.50)

        summary.add_resource_cost(gpu_cost)
        summary.add_resource_cost(storage_cost)
        summary.add_resource_cost(network_cost)

        # Verify totals
        assert abs(summary.total_cost - 266.74) < 0.01
        assert summary.total_gpu_hours == 8.0
        assert summary.total_storage_gb_hours == 2400.0
        assert summary.total_network_gb == 50.0
        assert len(summary.resource_costs) == 3

        # Verify breakdowns
        assert summary.cost_by_resource_type["gpu"] == 262.16
        assert abs(summary.cost_by_resource_type["storage"] - 0.08) < 0.01
        assert summary.cost_by_resource_type["network"] == 4.50

    def test_duration_calculation(self):
        """Test duration calculation."""
        summary = ComputeCostSummary()
        summary.start_time = time.time()
        time.sleep(0.1)
        summary.end_time = time.time()

        duration = summary.duration_seconds
        assert duration >= 0.1
        assert duration < 0.2

    def test_get_summary_dict(self):
        """Test summary dictionary serialization."""
        summary = ComputeCostSummary()
        summary.add_resource_cost(ComputeResourceCost("gpu", "a100", 8.0, 262.16))

        summary_dict = summary.get_summary_dict()
        assert "total_cost" in summary_dict
        assert "total_gpu_hours" in summary_dict
        assert "cost_by_resource_type" in summary_dict
        assert summary_dict["total_cost"] == 262.16


class TestKubetorchCostAggregator:
    """Test KubetorchCostAggregator functionality."""

    def setup_method(self):
        """Reset aggregator before each test."""
        reset_cost_aggregator()

    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = KubetorchCostAggregator()
        assert aggregator is not None
        assert len(aggregator.active_operations) == 0

    def test_start_operation_tracking(self):
        """Test starting operation tracking."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")

        assert "job-001" in aggregator.active_operations
        assert len(aggregator.get_active_operations()) == 1

    def test_start_duplicate_operation(self):
        """Test starting duplicate operation tracking."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")
        aggregator.start_operation_tracking("job-001")  # Duplicate

        # Should still only have one operation
        assert len(aggregator.get_active_operations()) == 1

    def test_add_gpu_cost(self):
        """Test adding GPU cost."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")

        cost = aggregator.add_gpu_cost("job-001", "a100", 8.0)

        assert cost is not None
        assert cost.resource_type == "gpu"
        assert cost.instance_type == "a100"
        assert cost.quantity == 8.0
        assert cost.cost > 0

    def test_add_storage_cost(self):
        """Test adding storage cost."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")

        cost = aggregator.add_storage_cost("job-001", 2400.0)

        assert cost is not None
        assert cost.resource_type == "storage"
        assert cost.quantity == 2400.0

    def test_add_network_cost(self):
        """Test adding network cost."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")

        cost = aggregator.add_network_cost("job-001", 50.0)

        assert cost is not None
        assert cost.resource_type == "network"
        assert cost.quantity == 50.0

    def test_add_cost_to_nonexistent_operation(self):
        """Test adding cost to non-existent operation."""
        aggregator = KubetorchCostAggregator()

        cost = aggregator.add_gpu_cost("nonexistent", "a100", 8.0)

        assert cost is None

    def test_finalize_operation_tracking(self):
        """Test finalizing operation tracking."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")
        aggregator.add_gpu_cost("job-001", "a100", 8.0)

        summary = aggregator.finalize_operation_tracking("job-001")

        assert summary is not None
        assert summary.total_cost > 0
        assert summary.total_gpu_hours == 8.0
        assert "job-001" not in aggregator.active_operations

    def test_finalize_nonexistent_operation(self):
        """Test finalizing non-existent operation."""
        aggregator = KubetorchCostAggregator()

        summary = aggregator.finalize_operation_tracking("nonexistent")

        assert summary is None

    def test_multiple_operations(self):
        """Test tracking multiple operations concurrently."""
        aggregator = KubetorchCostAggregator()

        aggregator.start_operation_tracking("job-001")
        aggregator.start_operation_tracking("job-002")
        aggregator.start_operation_tracking("job-003")

        aggregator.add_gpu_cost("job-001", "a100", 8.0)
        aggregator.add_gpu_cost("job-002", "h100", 4.0)
        aggregator.add_gpu_cost("job-003", "v100", 16.0)

        assert len(aggregator.get_active_operations()) == 3

        summary1 = aggregator.finalize_operation_tracking("job-001")
        summary2 = aggregator.finalize_operation_tracking("job-002")
        summary3 = aggregator.finalize_operation_tracking("job-003")

        assert summary1.total_gpu_hours == 8.0
        assert summary2.total_gpu_hours == 4.0
        assert summary3.total_gpu_hours == 16.0

    def test_clear_all_operations(self):
        """Test clearing all operations."""
        aggregator = KubetorchCostAggregator()
        aggregator.start_operation_tracking("job-001")
        aggregator.start_operation_tracking("job-002")

        aggregator.clear_all_operations()

        assert len(aggregator.get_active_operations()) == 0


class TestContextManager:
    """Test context manager functionality."""

    def setup_method(self):
        """Reset aggregator before each test."""
        reset_cost_aggregator()

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        with create_compute_cost_context("job-001") as ctx:
            ctx.add_gpu_cost("a100", 8.0)

        assert ctx.summary is not None
        assert ctx.summary.total_cost > 0
        assert ctx.summary.total_gpu_hours == 8.0

    def test_context_manager_multiple_costs(self):
        """Test context manager with multiple cost types."""
        with create_compute_cost_context("job-002") as ctx:
            ctx.add_gpu_cost("a100", 8.0)
            ctx.add_storage_cost(2400.0)
            ctx.add_network_cost(50.0)

        assert ctx.summary is not None
        assert ctx.summary.total_cost > 0
        assert ctx.summary.total_gpu_hours == 8.0
        assert ctx.summary.total_storage_gb_hours == 2400.0
        assert ctx.summary.total_network_gb == 50.0

    def test_context_manager_with_exception(self):
        """Test context manager finalization with exception."""
        try:
            with create_compute_cost_context("job-003") as ctx:
                ctx.add_gpu_cost("a100", 8.0)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Summary should still be finalized
        assert ctx.summary is not None
        assert ctx.summary.total_gpu_hours == 8.0

    def test_context_manager_nested(self):
        """Test nested context managers."""
        with create_compute_cost_context("job-outer") as ctx_outer:
            ctx_outer.add_gpu_cost("a100", 4.0)

            with create_compute_cost_context("job-inner") as ctx_inner:
                ctx_inner.add_gpu_cost("v100", 8.0)

            assert ctx_inner.summary.total_gpu_hours == 8.0

        assert ctx_outer.summary.total_gpu_hours == 4.0


class TestGlobalAggregator:
    """Test global aggregator singleton pattern."""

    def setup_method(self):
        """Reset aggregator before each test."""
        reset_cost_aggregator()

    def test_get_cost_aggregator(self):
        """Test getting global aggregator."""
        aggregator1 = get_cost_aggregator()
        aggregator2 = get_cost_aggregator()

        # Should be the same instance (singleton)
        assert aggregator1 is aggregator2

    def test_reset_cost_aggregator(self):
        """Test resetting global aggregator."""
        aggregator1 = get_cost_aggregator()
        aggregator1.start_operation_tracking("job-001")

        reset_cost_aggregator()

        aggregator2 = get_cost_aggregator()
        # Should be a new instance after reset
        assert len(aggregator2.get_active_operations()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
