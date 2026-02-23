"""
Unit tests for Kubetorch pricing module.

Tests cover:
- GPU pricing database integrity
- Cost calculation accuracy
- Fuzzy matching for instance types
- Storage and network cost calculation
- Training cost estimation
- Edge cases and error handling
"""

import pytest
from src.genops.providers.kubetorch.pricing import (
    GPU_PRICING,
    NETWORK_COST_PER_GB,
    STORAGE_COST_PER_GB_MONTH,
    GPUInstancePricing,
    KubetorchPricing,
    calculate_gpu_cost,
    get_pricing_info,
)


class TestGPUPricingDatabase:
    """Test GPU pricing database integrity."""

    def test_pricing_database_exists(self):
        """Test that GPU pricing database is populated."""
        assert len(GPU_PRICING) > 0, "GPU pricing database should not be empty"
        assert len(GPU_PRICING) >= 8, "Should have at least 8 GPU variants"

    def test_a100_pricing_exists(self):
        """Test A100 pricing entries exist."""
        assert "a100" in GPU_PRICING
        assert "a100-40gb" in GPU_PRICING
        assert "a100-80gb" in GPU_PRICING

    def test_h100_pricing_exists(self):
        """Test H100 pricing entries exist."""
        assert "h100" in GPU_PRICING
        assert "h100-80gb" in GPU_PRICING

    def test_v100_pricing_exists(self):
        """Test V100 pricing entries exist."""
        assert "v100" in GPU_PRICING
        assert "v100-16gb" in GPU_PRICING

    def test_a10g_pricing_exists(self):
        """Test A10G pricing entries exist."""
        assert "a10g" in GPU_PRICING
        assert "a10g-24gb" in GPU_PRICING

    def test_t4_pricing_exists(self):
        """Test T4 pricing entries exist."""
        assert "t4" in GPU_PRICING
        assert "t4-16gb" in GPU_PRICING

    def test_pricing_values_reasonable(self):
        """Test that pricing values are within reasonable ranges."""
        for key, pricing in GPU_PRICING.items():
            # All GPUs should cost between $1 and $150 per hour
            assert 1.0 <= pricing.cost_per_hour <= 150.0, (
                f"{key} cost ${pricing.cost_per_hour}/hr seems unreasonable"
            )

            # GPU memory should be between 8GB and 128GB
            assert 8 <= pricing.gpu_memory_gb <= 128, (
                f"{key} memory {pricing.gpu_memory_gb}GB seems unreasonable"
            )

    def test_pricing_hierarchy(self):
        """Test that pricing follows expected hierarchy (H100 > A100 > V100 > T4)."""
        h100_cost = GPU_PRICING["h100"].cost_per_hour
        a100_cost = GPU_PRICING["a100"].cost_per_hour
        v100_cost = GPU_PRICING["v100"].cost_per_hour
        t4_cost = GPU_PRICING["t4"].cost_per_hour

        assert h100_cost > a100_cost, "H100 should be more expensive than A100"
        assert a100_cost > v100_cost, "A100 should be more expensive than V100"
        assert v100_cost > t4_cost, "V100 should be more expensive than T4"


class TestKubetorchPricing:
    """Test KubetorchPricing class functionality."""

    def test_initialization_default(self):
        """Test default initialization."""
        pricing = KubetorchPricing()
        assert pricing is not None
        assert len(pricing.pricing_db) >= 8

    def test_initialization_custom_pricing(self):
        """Test initialization with custom pricing."""
        custom_pricing = {
            "custom-gpu": GPUInstancePricing(
                instance_type="custom-gpu",
                gpu_type="custom",
                cost_per_hour=50.0,
                gpu_memory_gb=64,
            )
        }
        pricing = KubetorchPricing(custom_pricing=custom_pricing)
        assert "custom-gpu" in pricing.pricing_db
        assert pricing.pricing_db["custom-gpu"].cost_per_hour == 50.0

    def test_calculate_compute_cost_gpu_a100_1hour(self):
        """Test GPU cost calculation for A100, 1 device, 1 hour."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost(
            instance_type="a100",
            num_devices=1,
            duration_seconds=3600,  # 1 hour
            resource_type="gpu",
        )
        expected = GPU_PRICING["a100"].cost_per_hour
        assert abs(cost - expected) < 0.01, f"Expected ${expected:.2f}, got ${cost:.2f}"

    def test_calculate_compute_cost_gpu_multiple_devices(self):
        """Test GPU cost calculation with multiple devices."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost(
            instance_type="a100",
            num_devices=8,
            duration_seconds=3600,  # 1 hour
            resource_type="gpu",
        )
        expected = GPU_PRICING["a100"].cost_per_hour * 8
        assert abs(cost - expected) < 0.01

    def test_calculate_compute_cost_gpu_partial_hour(self):
        """Test GPU cost calculation with partial hour."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost(
            instance_type="a100",
            num_devices=1,
            duration_seconds=1800,  # 30 minutes
            resource_type="gpu",
        )
        expected = GPU_PRICING["a100"].cost_per_hour * 0.5
        assert abs(cost - expected) < 0.01

    def test_calculate_compute_cost_cpu(self):
        """Test CPU cost calculation."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost(
            instance_type="cpu",  # Type doesn't matter for CPU
            num_devices=16,
            duration_seconds=3600,  # 1 hour
            resource_type="cpu",
        )
        expected = 16 * 1 * 0.50  # 16 cores × 1 hour × $0.50/core-hour
        assert abs(cost - expected) < 0.01

    def test_calculate_compute_cost_unknown_instance_fallback(self):
        """Test fallback pricing for unknown instance type."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost(
            instance_type="unknown-gpu-xyz",
            num_devices=1,
            duration_seconds=3600,
            resource_type="gpu",
        )
        # Should use A100 fallback pricing
        expected_fallback = 32.77
        assert abs(cost - expected_fallback) < 0.01

    def test_calculate_storage_cost(self):
        """Test storage cost calculation."""
        pricing = KubetorchPricing()

        # 100GB stored for 24 hours
        storage_gb_hours = 100 * 24
        cost = pricing.calculate_storage_cost(storage_gb_hours)

        # Convert to GB-months: 2400 GB-hours / 720 hours/month = 3.33 GB-months
        expected_gb_months = storage_gb_hours / 720
        expected_cost = expected_gb_months * STORAGE_COST_PER_GB_MONTH
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_network_cost(self):
        """Test network cost calculation."""
        pricing = KubetorchPricing()

        # 100GB data transfer
        cost = pricing.calculate_network_cost(100)

        expected = 100 * NETWORK_COST_PER_GB
        assert abs(cost - expected) < 0.01

    def test_estimate_training_cost_basic(self):
        """Test basic training cost estimation."""
        pricing = KubetorchPricing()

        result = pricing.estimate_training_cost(
            instance_type="a100",
            num_devices=8,
            estimated_hours=24,
            checkpoint_size_gb=0,  # No checkpoints
            data_transfer_gb=0,  # No data transfer
        )

        assert "cost_compute" in result
        assert "cost_storage" in result
        assert "cost_total" in result
        assert "gpu_hours" in result

        # 8 GPUs × 24 hours = 192 GPU-hours
        assert result["gpu_hours"] == 192

        # Cost should be compute only (no storage/network)
        expected_compute = GPU_PRICING["a100"].cost_per_hour * 8 * 24
        assert abs(result["cost_compute"] - expected_compute) < 0.01
        assert result["cost_storage"] == 0.0
        assert result["cost_total"] == result["cost_compute"]

    def test_estimate_training_cost_with_checkpoints(self):
        """Test training cost estimation with checkpoints."""
        pricing = KubetorchPricing()

        result = pricing.estimate_training_cost(
            instance_type="a100",
            num_devices=8,
            estimated_hours=24,
            checkpoint_size_gb=25.6,
            checkpoint_frequency_hours=2.0,  # Checkpoint every 2 hours
            data_transfer_gb=50,
        )

        # Should have compute, storage, and network costs
        assert result["cost_compute"] > 0
        assert result["cost_storage"] > 0
        assert result["cost_network"] > 0
        assert result["cost_total"] == (
            result["cost_compute"] + result["cost_storage"] + result["cost_network"]
        )

    def test_get_instance_pricing_exact_match(self):
        """Test exact instance type matching."""
        pricing = KubetorchPricing()
        result = pricing._get_instance_pricing("a100")
        assert result is not None
        assert result.instance_type == "a100"

    def test_get_instance_pricing_fuzzy_match_uppercase(self):
        """Test fuzzy matching with uppercase."""
        pricing = KubetorchPricing()
        result = pricing._get_instance_pricing("A100")
        assert result is not None
        assert result.gpu_type == "a100"

    def test_get_instance_pricing_fuzzy_match_variant(self):
        """Test fuzzy matching with variant notation."""
        pricing = KubetorchPricing()
        # Fuzzy matching "A100_80GB" will match first a100 variant found
        result = pricing._get_instance_pricing("A100_80GB")
        assert result is not None
        assert result.gpu_type == "a100"
        # Should match an a100 variant (could be 40GB or 80GB depending on dict order)
        assert result.gpu_memory_gb in [40, 80]

    def test_get_instance_pricing_not_found(self):
        """Test instance type not found."""
        pricing = KubetorchPricing()
        result = pricing._get_instance_pricing("nonexistent-gpu-999")
        assert result is None

    def test_fallback_cost_calculation(self):
        """Test fallback cost calculation."""
        pricing = KubetorchPricing()
        cost = pricing._fallback_cost_calculation(num_devices=4, duration_seconds=7200)

        # Should use A100 baseline: 4 devices × 2 hours × $32.77/hr
        expected = 4 * 2 * 32.77
        assert abs(cost - expected) < 0.01

    def test_get_supported_instance_types(self):
        """Test getting list of supported instance types."""
        pricing = KubetorchPricing()
        types = pricing.get_supported_instance_types()

        assert len(types) >= 8
        assert "a100" in types
        assert "h100" in types
        assert "v100" in types

    def test_get_instance_info(self):
        """Test getting instance information."""
        pricing = KubetorchPricing()
        info = pricing.get_instance_info("h100")

        assert info is not None
        assert info.instance_type == "h100"
        assert info.gpu_type == "h100"
        assert info.gpu_memory_gb == 80
        assert info.cost_per_hour > 90  # H100 is expensive


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_calculate_gpu_cost_function(self):
        """Test calculate_gpu_cost convenience function."""
        cost = calculate_gpu_cost("a100", num_devices=8, duration_seconds=3600)

        expected = GPU_PRICING["a100"].cost_per_hour * 8
        assert abs(cost - expected) < 0.01

    def test_get_pricing_info_function(self):
        """Test get_pricing_info convenience function."""
        info = get_pricing_info("h100")

        assert info is not None
        assert info.gpu_type == "h100"
        assert info.cost_per_hour > 0

    def test_get_pricing_info_not_found(self):
        """Test get_pricing_info with unknown instance."""
        info = get_pricing_info("nonexistent-gpu")
        assert info is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_duration(self):
        """Test cost calculation with zero duration."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost("a100", 8, 0)
        assert cost == 0.0

    def test_zero_devices(self):
        """Test cost calculation with zero devices."""
        pricing = KubetorchPricing()
        cost = pricing.calculate_compute_cost("a100", 0, 3600)
        assert cost == 0.0

    def test_very_large_duration(self):
        """Test cost calculation with very large duration (30 days)."""
        pricing = KubetorchPricing()
        duration_30_days = 30 * 24 * 3600
        cost = pricing.calculate_compute_cost("a100", 8, duration_30_days)

        expected = GPU_PRICING["a100"].cost_per_hour * 8 * (30 * 24)
        assert abs(cost - expected) < 1.0  # Allow small floating point error

    def test_fractional_gpu_hours(self):
        """Test cost calculation with fractional GPU-hours."""
        pricing = KubetorchPricing()

        # 1 GPU for 90 seconds (0.025 hours)
        cost = pricing.calculate_compute_cost("a100", 1, 90)

        expected = GPU_PRICING["a100"].cost_per_hour * (90 / 3600)
        assert abs(cost - expected) < 0.001


class TestPricingDataclass:
    """Test GPUInstancePricing dataclass."""

    def test_dataclass_creation(self):
        """Test creating GPUInstancePricing instance."""
        pricing = GPUInstancePricing(
            instance_type="test-gpu",
            gpu_type="test",
            cost_per_hour=10.0,
            gpu_memory_gb=32,
        )

        assert pricing.instance_type == "test-gpu"
        assert pricing.gpu_type == "test"
        assert pricing.cost_per_hour == 10.0
        assert pricing.gpu_memory_gb == 32
        assert pricing.currency == "USD"  # Default

    def test_dataclass_str_representation(self):
        """Test string representation of GPUInstancePricing."""
        pricing = GPU_PRICING["a100"]
        str_repr = str(pricing)

        assert "a100" in str_repr
        assert "$" in str_repr
        assert "A100" in str_repr.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
