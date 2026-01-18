"""
Integration tests for Kubetorch provider.

Tests cover end-to-end workflows and cross-module functionality.
"""

import pytest
import time
from src.genops.providers.kubetorch import (
    instrument_kubetorch,
    calculate_gpu_cost,
    get_pricing_info,
    create_compute_cost_context,
    get_cost_aggregator,
    reset_cost_aggregator,
    auto_instrument_kubetorch,
    uninstrument_kubetorch,
    is_kubetorch_instrumented,
    validate_kubetorch_setup,
    get_module_status,
)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_basic_cost_tracking_workflow(self):
        """Test basic cost tracking from start to finish."""
        # Reset state
        reset_cost_aggregator()

        # Track a training job
        with create_compute_cost_context("train-job-001") as ctx:
            # Simulate training with 8 A100 GPUs for 1 hour worth of operations
            ctx.add_gpu_cost("a100", 8.0, operation_name="training")
            ctx.add_storage_cost(100 * 24, operation_name="checkpoints")
            ctx.add_network_cost(50, operation_name="data_transfer")

        # Verify results
        assert ctx.summary is not None
        assert ctx.summary.total_cost > 0
        assert ctx.summary.total_gpu_hours == 8.0
        assert ctx.summary.total_storage_gb_hours == 2400.0
        assert ctx.summary.total_network_gb == 50.0

    def test_multiple_concurrent_jobs(self):
        """Test tracking multiple jobs concurrently."""
        reset_cost_aggregator()

        aggregator = get_cost_aggregator()

        # Start multiple jobs
        aggregator.start_operation_tracking("job-1")
        aggregator.start_operation_tracking("job-2")
        aggregator.start_operation_tracking("job-3")

        # Track different resources for each job
        aggregator.add_gpu_cost("job-1", "a100", 8.0)
        aggregator.add_gpu_cost("job-2", "h100", 4.0)
        aggregator.add_gpu_cost("job-3", "v100", 16.0)

        # Finalize jobs
        summary1 = aggregator.finalize_operation_tracking("job-1")
        summary2 = aggregator.finalize_operation_tracking("job-2")
        summary3 = aggregator.finalize_operation_tracking("job-3")

        # Verify each job tracked correctly
        assert summary1.total_gpu_hours == 8.0
        assert summary2.total_gpu_hours == 4.0
        assert summary3.total_gpu_hours == 16.0

    def test_adapter_with_cost_tracking(self):
        """Test adapter integration with cost tracking."""
        adapter = instrument_kubetorch(
            team="ml-research",
            project="llm-training",
            cost_tracking_enabled=True,
        )

        result = adapter.track_compute_deployment(
            instance_type="a100",
            num_devices=8,
            workload_type="training",
            duration_seconds=3600,
        )

        assert result is not None
        assert 'operation_id' in result
        assert 'cost_total' in result
        assert result['cost_total'] > 0
        assert result['gpu_hours'] == 8.0


class TestPricingIntegration:
    """Test pricing module integration."""

    def test_pricing_info_retrieval(self):
        """Test retrieving pricing information."""
        # Test all major GPU types
        gpu_types = ["a100", "h100", "v100", "a10g", "t4"]

        for gpu_type in gpu_types:
            info = get_pricing_info(gpu_type)
            assert info is not None
            assert info.cost_per_hour > 0
            assert info.gpu_memory_gb > 0

    def test_cost_calculation_consistency(self):
        """Test that cost calculations are consistent."""
        # Calculate cost using different methods
        cost1 = calculate_gpu_cost("a100", num_devices=8, duration_seconds=3600)

        # Calculate using aggregator
        reset_cost_aggregator()
        aggregator = get_cost_aggregator()
        aggregator.start_operation_tracking("test")
        aggregator.add_gpu_cost("test", "a100", 8.0)
        summary = aggregator.finalize_operation_tracking("test")
        cost2 = summary.total_cost

        # Both methods should give same result
        assert abs(cost1 - cost2) < 0.01

    def test_multi_resource_cost_breakdown(self):
        """Test detailed cost breakdown for multiple resources."""
        reset_cost_aggregator()

        with create_compute_cost_context("full-job") as ctx:
            ctx.add_gpu_cost("a100", 8.0, operation_name="training")
            ctx.add_storage_cost(1000 * 24, operation_name="storage")
            ctx.add_network_cost(100, operation_name="network")

        # Verify breakdown
        assert "gpu" in ctx.summary.cost_by_resource_type
        assert "storage" in ctx.summary.cost_by_resource_type
        assert "network" in ctx.summary.cost_by_resource_type

        # GPU should be the largest cost component
        assert ctx.summary.cost_by_resource_type["gpu"] > ctx.summary.cost_by_resource_type["storage"]
        assert ctx.summary.cost_by_resource_type["gpu"] > ctx.summary.cost_by_resource_type["network"]


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def teardown_method(self):
        """Clean up instrumentation after each test."""
        if is_kubetorch_instrumented():
            uninstrument_kubetorch()

    def test_auto_instrumentation_lifecycle(self):
        """Test complete instrumentation lifecycle."""
        # Initially not instrumented
        assert not is_kubetorch_instrumented()

        # Enable instrumentation
        result = auto_instrument_kubetorch(
            team="ml-research",
            project="test-project",
        )

        # Should be instrumented now
        assert is_kubetorch_instrumented()

        # Disable instrumentation
        uninstrument_result = uninstrument_kubetorch()
        assert uninstrument_result is True

        # Should not be instrumented anymore
        assert not is_kubetorch_instrumented()

    def test_auto_instrumentation_idempotent(self):
        """Test that auto-instrumentation is idempotent."""
        # First call should succeed
        result1 = auto_instrument_kubetorch(team="test")
        assert is_kubetorch_instrumented()

        # Second call should be no-op
        result2 = auto_instrument_kubetorch(team="test")
        assert result2 is False  # Already instrumented

        # Clean up
        uninstrument_kubetorch()


class TestValidation:
    """Test validation functionality."""

    def test_validation_runs_successfully(self):
        """Test that validation completes without errors."""
        result = validate_kubetorch_setup()

        assert result is not None
        assert result.total_checks > 0

    def test_validation_checks_modules(self):
        """Test that validation checks all modules."""
        result = validate_kubetorch_setup()

        # Should check all major modules
        module_checks = [
            issue for issue in result.issues
            if issue.component.startswith("Module:")
        ]

        # Should have checks for all 6 modules
        assert len(module_checks) >= 6

    def test_module_status_reporting(self):
        """Test module status reporting."""
        status = get_module_status()

        assert "pricing" in status
        assert "adapter" in status
        assert "cost_aggregator" in status
        assert "compute_monitor" in status
        assert "validation" in status
        assert "registration" in status

        # Pricing and implemented modules should be available
        assert status["pricing"] is True
        assert status["adapter"] is True
        assert status["cost_aggregator"] is True


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_cost_with_zero_duration(self):
        """Test cost calculation with zero duration."""
        cost = calculate_gpu_cost("a100", num_devices=8, duration_seconds=0)
        assert cost == 0.0

    def test_cost_with_zero_devices(self):
        """Test cost calculation with zero devices."""
        cost = calculate_gpu_cost("a100", num_devices=0, duration_seconds=3600)
        assert cost == 0.0

    def test_finalize_nonexistent_operation(self):
        """Test finalizing operation that doesn't exist."""
        reset_cost_aggregator()
        aggregator = get_cost_aggregator()

        summary = aggregator.finalize_operation_tracking("nonexistent")
        assert summary is None

    def test_add_cost_to_nonexistent_operation(self):
        """Test adding cost to non-existent operation."""
        reset_cost_aggregator()
        aggregator = get_cost_aggregator()

        result = aggregator.add_gpu_cost("nonexistent", "a100", 8.0)
        assert result is None


class TestPerformance:
    """Test performance characteristics."""

    def test_high_volume_cost_tracking(self):
        """Test tracking many operations."""
        reset_cost_aggregator()
        aggregator = get_cost_aggregator()

        start_time = time.time()

        # Track 100 operations
        for i in range(100):
            op_id = f"job-{i}"
            aggregator.start_operation_tracking(op_id)
            aggregator.add_gpu_cost(op_id, "a100", 1.0)
            aggregator.finalize_operation_tracking(op_id)

        duration = time.time() - start_time

        # Should complete in reasonable time (< 1 second)
        assert duration < 1.0

    def test_context_manager_overhead(self):
        """Test context manager overhead."""
        reset_cost_aggregator()

        start_time = time.time()

        # Run 50 context manager operations
        for i in range(50):
            with create_compute_cost_context(f"job-{i}") as ctx:
                ctx.add_gpu_cost("a100", 1.0)

        duration = time.time() - start_time

        # Should complete in reasonable time (< 1 second)
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
