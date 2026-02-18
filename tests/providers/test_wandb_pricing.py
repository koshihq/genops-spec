#!/usr/bin/env python3
"""
Test suite for GenOps W&B pricing model functionality.

This module tests the pricing calculations, cost estimation,
and pricing model customization features.
"""

import os
import sys
import unittest
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from genops.providers.wandb_pricing import (
    WandbPricingModel,
    calculate_compute_cost,
    calculate_data_transfer_cost,
    calculate_storage_cost,
    estimate_experiment_cost,
    get_gpu_pricing,
    get_storage_pricing,
)


class TestWandbPricingModel(unittest.TestCase):
    """Test W&B pricing model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.pricing_model = WandbPricingModel()

    def test_pricing_model_initialization(self):
        """Test pricing model initialization with defaults."""
        self.assertIsInstance(self.pricing_model.compute_rates, dict)
        self.assertIsInstance(self.pricing_model.storage_rates, dict)
        self.assertIsInstance(self.pricing_model.data_transfer_rates, dict)

        # Check some expected GPU types
        self.assertIn("v100", self.pricing_model.compute_rates)
        self.assertIn("a100", self.pricing_model.compute_rates)

    def test_custom_pricing_model(self):
        """Test custom pricing model initialization."""
        custom_rates = {
            "custom_gpu": 5.00,
            "v100": 2.50,  # Override default
        }

        custom_model = WandbPricingModel(compute_rates=custom_rates)

        self.assertEqual(custom_model.compute_rates["custom_gpu"], 5.00)
        self.assertEqual(custom_model.compute_rates["v100"], 2.50)

    def test_get_gpu_pricing(self):
        """Test GPU pricing retrieval."""
        v100_price = get_gpu_pricing("v100", region="us-east-1")
        self.assertGreater(v100_price, 0)
        self.assertIsInstance(v100_price, (int, float))

        # Test unknown GPU type
        unknown_price = get_gpu_pricing("unknown_gpu", region="us-east-1")
        self.assertGreater(unknown_price, 0)  # Should return default

    def test_get_storage_pricing(self):
        """Test storage pricing retrieval."""
        ssd_price = get_storage_pricing("ssd", region="us-east-1")
        self.assertGreater(ssd_price, 0)

        hdd_price = get_storage_pricing("hdd", region="us-east-1")
        self.assertGreater(hdd_price, 0)

        # SSD should be more expensive than HDD
        self.assertGreater(ssd_price, hdd_price)

    def test_calculate_compute_cost_basic(self):
        """Test basic compute cost calculation."""
        cost = calculate_compute_cost(
            instance_type="p3.2xlarge", hours=2.0, region="us-east-1"
        )

        self.assertGreater(cost, 0)
        self.assertIsInstance(cost, (int, float))

    def test_calculate_compute_cost_with_custom_model(self):
        """Test compute cost calculation with custom pricing model."""
        custom_model = WandbPricingModel(compute_rates={"p3.2xlarge": 5.00})

        cost = calculate_compute_cost(
            instance_type="p3.2xlarge",
            hours=2.0,
            region="us-east-1",
            pricing_model=custom_model,
        )

        expected_cost = 2.0 * 5.00  # 2 hours * $5.00/hour
        self.assertEqual(cost, expected_cost)

    def test_calculate_storage_cost_basic(self):
        """Test basic storage cost calculation."""
        cost = calculate_storage_cost(
            storage_type="ssd", size_gb=100.0, duration_days=30, region="us-east-1"
        )

        self.assertGreater(cost, 0)

        # Test different durations
        cost_15_days = calculate_storage_cost("ssd", 100.0, 15, "us-east-1")
        cost_30_days = calculate_storage_cost("ssd", 100.0, 30, "us-east-1")

        # 30 days should cost more than 15 days
        self.assertGreater(cost_30_days, cost_15_days)

    def test_calculate_data_transfer_cost(self):
        """Test data transfer cost calculation."""
        # Internal transfer (should be free or cheap)
        internal_cost = calculate_data_transfer_cost(
            transfer_gb=100.0, transfer_type="internal", region="us-east-1"
        )

        # External transfer (should cost more)
        external_cost = calculate_data_transfer_cost(
            transfer_gb=100.0, transfer_type="external", region="us-east-1"
        )

        self.assertGreaterEqual(internal_cost, 0)
        self.assertGreater(external_cost, internal_cost)

    def test_estimate_experiment_cost_comprehensive(self):
        """Test comprehensive experiment cost estimation."""
        config = {
            "instance_type": "p3.2xlarge",
            "duration_hours": 3.0,
            "storage_gb": 50.0,
            "storage_duration_days": 7,
            "data_transfer_gb": 25.0,
            "transfer_type": "external",
            "region": "us-east-1",
        }

        total_cost = estimate_experiment_cost(config)

        # Calculate components separately
        compute_cost = calculate_compute_cost(
            config["instance_type"], config["duration_hours"], config["region"]
        )

        storage_cost = calculate_storage_cost(
            "ssd",  # Default storage type
            config["storage_gb"],
            config["storage_duration_days"],
            config["region"],
        )

        transfer_cost = calculate_data_transfer_cost(
            config["data_transfer_gb"], config["transfer_type"], config["region"]
        )

        expected_total = compute_cost + storage_cost + transfer_cost
        self.assertAlmostEqual(total_cost, expected_total, places=2)

    def test_estimate_experiment_cost_minimal_config(self):
        """Test experiment cost estimation with minimal configuration."""
        config = {"instance_type": "p3.2xlarge", "duration_hours": 1.0}

        cost = estimate_experiment_cost(config)

        # Should at least include compute cost
        min_expected_cost = calculate_compute_cost(
            config["instance_type"],
            config["duration_hours"],
            "us-east-1",  # Default region
        )

        self.assertGreaterEqual(cost, min_expected_cost)

    def test_regional_pricing_differences(self):
        """Test pricing differences across regions."""
        regions = ["us-east-1", "us-west-2", "eu-west-1"]
        costs = []

        for region in regions:
            cost = calculate_compute_cost(
                instance_type="p3.2xlarge", hours=1.0, region=region
            )
            costs.append(cost)

        # All costs should be positive
        for cost in costs:
            self.assertGreater(cost, 0)

        # There might be regional differences (but not required)
        self.assertEqual(len(costs), len(regions))

    def test_precision_and_rounding(self):
        """Test pricing precision and rounding behavior."""
        # Test with small amounts
        small_cost = calculate_compute_cost(
            instance_type="p3.2xlarge",
            hours=0.001,  # 3.6 seconds
            region="us-east-1",
        )

        self.assertGreater(small_cost, 0)
        self.assertLess(small_cost, 1.0)

        # Test precision
        self.assertIsInstance(small_cost, (int, float, Decimal))

    def test_cost_scaling_linearity(self):
        """Test that cost scaling is linear for compute resources."""
        base_cost = calculate_compute_cost("p3.2xlarge", 1.0, "us-east-1")
        double_cost = calculate_compute_cost("p3.2xlarge", 2.0, "us-east-1")

        # Should scale linearly
        self.assertAlmostEqual(double_cost, base_cost * 2, places=2)

    def test_storage_cost_monthly_calculation(self):
        """Test monthly storage cost calculation."""
        # Test different month lengths
        monthly_cost_30 = calculate_storage_cost("ssd", 100.0, 30, "us-east-1")
        monthly_cost_31 = calculate_storage_cost("ssd", 100.0, 31, "us-east-1")

        # 31 days should cost slightly more than 30 days
        self.assertGreater(monthly_cost_31, monthly_cost_30)

        # But difference should be small (1/30th more)
        expected_ratio = 31.0 / 30.0
        actual_ratio = monthly_cost_31 / monthly_cost_30
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=2)

    def test_pricing_model_edge_cases(self):
        """Test pricing model edge cases."""
        # Test zero costs
        zero_compute = calculate_compute_cost("p3.2xlarge", 0.0, "us-east-1")
        self.assertEqual(zero_compute, 0.0)

        zero_storage = calculate_storage_cost("ssd", 0.0, 30, "us-east-1")
        self.assertEqual(zero_storage, 0.0)

        zero_transfer = calculate_data_transfer_cost(0.0, "external", "us-east-1")
        self.assertEqual(zero_transfer, 0.0)

    def test_invalid_configurations(self):
        """Test handling of invalid configurations."""
        # Test negative values - should handle gracefully
        try:
            negative_cost = calculate_compute_cost("p3.2xlarge", -1.0, "us-east-1")
            # If it doesn't raise an exception, should return 0 or positive
            self.assertGreaterEqual(negative_cost, 0)
        except ValueError:
            # Acceptable to raise ValueError for invalid input
            pass

    def test_pricing_model_serialization(self):
        """Test pricing model can be serialized/deserialized."""
        # Test that pricing model data structures are JSON-serializable
        import json

        pricing_data = {
            "compute_rates": self.pricing_model.compute_rates,
            "storage_rates": self.pricing_model.storage_rates,
            "data_transfer_rates": self.pricing_model.data_transfer_rates,
        }

        # Should be able to serialize to JSON
        json_str = json.dumps(pricing_data)
        self.assertIsInstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        self.assertEqual(
            deserialized["compute_rates"], self.pricing_model.compute_rates
        )

    def test_bulk_cost_calculation(self):
        """Test bulk cost calculations for multiple experiments."""
        experiment_configs = [
            {"instance_type": "p3.2xlarge", "duration_hours": 1.0},
            {"instance_type": "p3.2xlarge", "duration_hours": 2.0},
            {"instance_type": "p3.8xlarge", "duration_hours": 1.0},
        ]

        total_cost = 0.0
        individual_costs = []

        for config in experiment_configs:
            cost = estimate_experiment_cost(config)
            individual_costs.append(cost)
            total_cost += cost

        # All costs should be positive
        for cost in individual_costs:
            self.assertGreater(cost, 0)

        # Total should equal sum of individuals
        self.assertEqual(total_cost, sum(individual_costs))

        # Different configs should have different costs
        self.assertNotEqual(individual_costs[0], individual_costs[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
