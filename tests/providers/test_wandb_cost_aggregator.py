#!/usr/bin/env python3
"""
Test suite for GenOps W&B Cost Aggregator functionality.

This module tests the cost aggregation, forecasting, and optimization
recommendation features for the W&B integration.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from genops.providers.wandb_cost_aggregator import (
    WandbCostAggregator,
    calculate_simple_experiment_cost,
    forecast_experiment_costs,
    generate_cost_optimization_recommendations,
)


class TestWandbCostAggregator(unittest.TestCase):
    """Test cost aggregation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = WandbCostAggregator(
            team="test-team", project="test-project", customer_id="test-customer"
        )

    def test_aggregator_initialization(self):
        """Test cost aggregator initialization."""
        self.assertEqual(self.aggregator.team, "test-team")
        self.assertEqual(self.aggregator.project, "test-project")
        self.assertEqual(self.aggregator.customer_id, "test-customer")

    def test_simple_cost_summary(self):
        """Test simple cost summary generation."""
        mock_data = [
            {"experiment_id": "exp1", "cost": 10.0, "duration_hours": 1.0},
            {"experiment_id": "exp2", "cost": 20.0, "duration_hours": 2.0},
            {"experiment_id": "exp3", "cost": 30.0, "duration_hours": 3.0},
        ]

        with patch.object(
            self.aggregator, "_get_experiment_data", return_value=mock_data
        ):
            summary = self.aggregator.get_simple_cost_summary(time_period_days=7)

            self.assertEqual(summary["total_cost"], 60.0)
            self.assertEqual(summary["experiment_count"], 3)
            self.assertEqual(summary["average_cost"], 20.0)
            self.assertEqual(summary["min_cost"], 10.0)
            self.assertEqual(summary["max_cost"], 30.0)

    def test_comprehensive_cost_summary(self):
        """Test comprehensive cost summary with forecasting."""
        mock_data = [
            {
                "experiment_id": f"exp{i}",
                "cost": i * 10.0,
                "duration_hours": i * 0.5,
                "experiment_type": "training",
                "timestamp": datetime.utcnow() - timedelta(days=i),
            }
            for i in range(1, 6)
        ]

        with patch.object(
            self.aggregator, "_get_experiment_data", return_value=mock_data
        ):
            summary = self.aggregator.get_comprehensive_cost_summary(
                time_period_days=30, include_forecasting=True
            )

            self.assertIn("total_cost", summary)
            self.assertIn("cost_by_experiment_type", summary)
            self.assertIn("cost_trend", summary)
            self.assertIn("forecasted_cost", summary)

    def test_team_cost_breakdown(self):
        """Test cost breakdown by team."""
        with patch.object(self.aggregator, "_get_team_experiments") as mock_team_data:
            mock_team_data.return_value = {
                "team-a": [
                    {"cost": 15.0, "experiment_type": "training"},
                    {"cost": 25.0, "experiment_type": "evaluation"},
                ],
                "team-b": [{"cost": 35.0, "experiment_type": "training"}],
            }

            breakdown = self.aggregator.get_team_cost_breakdown(time_period_days=7)

            self.assertEqual(breakdown["team-a"]["total_cost"], 40.0)
            self.assertEqual(breakdown["team-b"]["total_cost"], 35.0)

    def test_cost_forecasting(self):
        """Test cost forecasting functionality."""
        historical_data = [
            {"date": datetime.utcnow() - timedelta(days=i), "cost": 10.0 + i}
            for i in range(7)
        ]

        with patch.object(
            self.aggregator, "_get_historical_costs", return_value=historical_data
        ):
            forecast = self.aggregator.forecast_costs(days_ahead=7)

            self.assertIn("forecasted_cost", forecast)
            self.assertIn("confidence_interval", forecast)
            self.assertIn("trend", forecast)
            self.assertGreater(forecast["forecasted_cost"], 0)

    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendation generation."""
        mock_experiments = [
            {"cost": 100.0, "accuracy": 0.90, "duration": 2.0, "model_type": "large"},
            {"cost": 50.0, "accuracy": 0.85, "duration": 1.5, "model_type": "medium"},
            {"cost": 25.0, "accuracy": 0.80, "duration": 1.0, "model_type": "small"},
        ]

        with patch.object(
            self.aggregator, "_get_experiment_data", return_value=mock_experiments
        ):
            recommendations = self.aggregator.generate_optimization_recommendations()

            self.assertIsInstance(recommendations, list)
            if recommendations:
                rec = recommendations[0]
                self.assertIn("recommendation", rec)
                self.assertIn("estimated_savings", rec)
                self.assertIn("confidence", rec)


class TestCostCalculationFunctions(unittest.TestCase):
    """Test standalone cost calculation functions."""

    def test_calculate_simple_experiment_cost(self):
        """Test simple experiment cost calculation."""
        cost = calculate_simple_experiment_cost(
            compute_hours=2.0, gpu_type="v100", storage_gb=10.0, data_transfer_gb=5.0
        )

        self.assertGreater(cost, 0)
        self.assertIsInstance(cost, float)

    def test_cost_calculation_with_different_gpu_types(self):
        """Test cost calculation with different GPU types."""
        v100_cost = calculate_simple_experiment_cost(
            compute_hours=1.0, gpu_type="v100", storage_gb=0.0
        )

        a100_cost = calculate_simple_experiment_cost(
            compute_hours=1.0, gpu_type="a100", storage_gb=0.0
        )

        # A100 should typically be more expensive
        self.assertGreater(a100_cost, v100_cost)

    def test_forecast_experiment_costs(self):
        """Test experiment cost forecasting."""
        historical_costs = [10.0, 12.0, 11.0, 15.0, 13.0, 16.0, 14.0]

        forecast = forecast_experiment_costs(
            historical_costs=historical_costs, forecast_periods=5
        )

        self.assertIn("forecasted_costs", forecast)
        self.assertIn("trend", forecast)
        self.assertEqual(len(forecast["forecasted_costs"]), 5)

    def test_generate_cost_optimization_recommendations(self):
        """Test global cost optimization recommendations."""
        recommendations = generate_cost_optimization_recommendations(
            team="optimization-team", lookback_days=30, target_savings_percentage=20.0
        )

        self.assertIsInstance(recommendations, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
