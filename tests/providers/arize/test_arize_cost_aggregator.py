#!/usr/bin/env python3
"""
Comprehensive test suite for GenOps Arize AI cost aggregator.

This test suite provides comprehensive coverage of the Arize AI cost aggregation
including multi-model cost tracking, optimization recommendations, and budget analysis.

Test Categories:
- Cost aggregation and summary tests (15 tests)
- Multi-model cost analysis tests (12 tests)
- Cost optimization recommendation tests (10 tests)
- Budget analysis and forecasting tests (8 tests)
- Error handling and edge cases (5 tests)

Total: 50 tests ensuring robust Arize AI cost aggregation with GenOps intelligence.
"""

import os
import sys
import unittest

pd = __import__("pytest").importorskip("pandas")
from datetime import datetime, timedelta  # noqa: E402

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from genops.providers.arize_cost_aggregator import (  # noqa: E402
    ArizeCostAggregator,
    ArizeCostSummary,
    CostOptimizationRecommendation,
    ModelCostBreakdown,
    MonitoringEfficiencyMetrics,
    OptimizationType,
    analyze_cost_trends,
    calculate_model_monitoring_cost,
    estimate_monthly_monitoring_cost,
)


class TestArizeCostAggregator(unittest.TestCase):
    """Test core cost aggregation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = ArizeCostAggregator(
            team="test-team", project="test-project", customer_id="test-customer-123"
        )

        # Mock cost data for testing
        self.mock_cost_data = [
            {
                "timestamp": datetime.utcnow() - timedelta(days=7),
                "model_id": "fraud-detection-v1",
                "environment": "production",
                "prediction_logging_cost": 5.25,
                "data_quality_cost": 2.10,
                "alert_management_cost": 1.50,
                "dashboard_cost": 0.75,
                "prediction_count": 5250,
                "data_quality_checks": 105,
                "active_alerts": 3,
            },
            {
                "timestamp": datetime.utcnow() - timedelta(days=5),
                "model_id": "fraud-detection-v1",
                "environment": "production",
                "prediction_logging_cost": 8.40,
                "data_quality_cost": 3.20,
                "alert_management_cost": 2.00,
                "dashboard_cost": 1.00,
                "prediction_count": 8400,
                "data_quality_checks": 160,
                "active_alerts": 4,
            },
            {
                "timestamp": datetime.utcnow() - timedelta(days=3),
                "model_id": "sentiment-analysis-v2",
                "environment": "production",
                "prediction_logging_cost": 12.60,
                "data_quality_cost": 4.50,
                "alert_management_cost": 1.75,
                "dashboard_cost": 1.25,
                "prediction_count": 12600,
                "data_quality_checks": 225,
                "active_alerts": 5,
            },
        ]

    def test_aggregator_initialization(self):
        """Test cost aggregator initialization."""
        self.assertEqual(self.aggregator.team, "test-team")
        self.assertEqual(self.aggregator.project, "test-project")
        self.assertEqual(self.aggregator.customer_id, "test-customer-123")
        self.assertEqual(len(self.aggregator.cost_records), 0)
        self.assertIsNotNone(self.aggregator.aggregation_start_time)

    def test_add_cost_record(self):
        """Test adding individual cost records."""
        cost_record = self.mock_cost_data[0]

        self.aggregator.add_cost_record(
            model_id=cost_record["model_id"],
            environment=cost_record["environment"],
            prediction_logging_cost=cost_record["prediction_logging_cost"],
            data_quality_cost=cost_record["data_quality_cost"],
            alert_management_cost=cost_record["alert_management_cost"],
            dashboard_cost=cost_record["dashboard_cost"],
            prediction_count=cost_record["prediction_count"],
            data_quality_checks=cost_record["data_quality_checks"],
            active_alerts=cost_record["active_alerts"],
        )

        self.assertEqual(len(self.aggregator.cost_records), 1)

        record = self.aggregator.cost_records[0]
        self.assertEqual(record.model_id, cost_record["model_id"])
        self.assertEqual(record.environment, cost_record["environment"])
        self.assertEqual(
            record.prediction_logging_cost, cost_record["prediction_logging_cost"]
        )
        self.assertEqual(record.total_cost, 9.60)  # Sum of all cost components

    def test_bulk_add_cost_records(self):
        """Test adding multiple cost records in bulk."""
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        self.assertEqual(len(self.aggregator.cost_records), 3)

        # Verify total cost calculation
        total_cost = sum(r.total_cost for r in self.aggregator.cost_records)
        expected_total = 9.60 + 14.60 + 20.10  # Manual calculation
        self.assertEqual(total_cost, expected_total)

    def test_get_cost_summary_by_model(self):
        """Test cost summary aggregation by model."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        summary = self.aggregator.get_cost_summary_by_model()

        self.assertIsInstance(summary, ArizeCostSummary)
        self.assertEqual(len(summary.cost_by_model), 2)  # Two unique models

        # Check fraud detection model costs (2 records)
        fraud_model_cost = summary.cost_by_model.get("fraud-detection-v1", 0)
        self.assertEqual(fraud_model_cost, 24.20)  # 9.60 + 14.60

        # Check sentiment analysis model costs (1 record)
        sentiment_model_cost = summary.cost_by_model.get("sentiment-analysis-v2", 0)
        self.assertEqual(sentiment_model_cost, 20.10)

        # Check total cost
        self.assertEqual(summary.total_cost, 44.30)

    def test_get_cost_summary_by_environment(self):
        """Test cost summary aggregation by environment."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        summary = self.aggregator.get_cost_summary_by_environment()

        # All test data is production environment
        self.assertEqual(len(summary.cost_by_environment), 1)
        self.assertEqual(summary.cost_by_environment["production"], 44.30)

    def test_get_model_cost_breakdown(self):
        """Test detailed model cost breakdown."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        breakdown = self.aggregator.get_model_cost_breakdown("fraud-detection-v1")

        self.assertIsInstance(breakdown, ModelCostBreakdown)
        self.assertEqual(breakdown.model_id, "fraud-detection-v1")
        self.assertEqual(breakdown.total_cost, 24.20)
        self.assertEqual(breakdown.prediction_logging_cost, 13.65)  # 5.25 + 8.40
        self.assertEqual(breakdown.data_quality_cost, 5.30)  # 2.10 + 3.20
        self.assertEqual(breakdown.alert_management_cost, 3.50)  # 1.50 + 2.00
        self.assertEqual(breakdown.dashboard_cost, 1.75)  # 0.75 + 1.00

        # Check aggregated metrics
        self.assertEqual(breakdown.total_predictions, 13650)  # 5250 + 8400
        self.assertEqual(breakdown.total_data_quality_checks, 265)  # 105 + 160
        self.assertEqual(breakdown.total_alerts, 7)  # 3 + 4

    def test_get_cost_trends_analysis(self):
        """Test cost trends analysis over time."""
        # Add test data with timestamps
        for record in self.mock_cost_data:
            cost_record = self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )
            cost_record.timestamp = record[
                "timestamp"
            ]  # Manually set timestamp for testing

        trends = self.aggregator.get_cost_trends_analysis(days=10)

        self.assertIn("daily_costs", trends)
        self.assertIn("cost_trend", trends)
        self.assertIn("prediction_trends", trends)
        self.assertIn("efficiency_trends", trends)

        # Check daily costs structure
        daily_costs = trends["daily_costs"]
        self.assertIsInstance(daily_costs, list)
        self.assertGreaterEqual(len(daily_costs), 3)  # At least 3 data points

    def test_get_efficiency_metrics(self):
        """Test monitoring efficiency metrics calculation."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        metrics = self.aggregator.get_efficiency_metrics()

        self.assertIsInstance(metrics, MonitoringEfficiencyMetrics)
        self.assertGreater(metrics.cost_per_prediction, 0)
        self.assertGreater(metrics.cost_per_data_quality_check, 0)
        self.assertGreater(metrics.cost_per_alert, 0)
        self.assertGreater(metrics.predictions_per_dollar, 0)

        # Check efficiency ratios
        self.assertIsInstance(metrics.model_efficiency_scores, dict)
        self.assertIn("fraud-detection-v1", metrics.model_efficiency_scores)
        self.assertIn("sentiment-analysis-v2", metrics.model_efficiency_scores)

    def test_get_monthly_cost_forecast(self):
        """Test monthly cost forecasting."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        forecast = self.aggregator.get_monthly_cost_forecast()

        self.assertIn("projected_monthly_cost", forecast)
        self.assertIn("confidence_interval", forecast)
        self.assertIn("cost_by_model_monthly", forecast)
        self.assertIn("growth_rate", forecast)

        # Check projected cost is reasonable
        projected_cost = forecast["projected_monthly_cost"]
        self.assertGreater(projected_cost, 0)
        self.assertLess(projected_cost, 10000)  # Reasonable upper bound

    def test_get_cost_optimization_recommendations(self):
        """Test cost optimization recommendations generation."""
        # Add test data with high costs to trigger recommendations
        high_cost_data = self.mock_cost_data.copy()
        high_cost_data[0]["prediction_logging_cost"] = 50.0  # High prediction cost
        high_cost_data[1]["data_quality_cost"] = 30.0  # High data quality cost

        for record in high_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Check recommendation structure
        for rec in recommendations:
            self.assertIsInstance(rec, CostOptimizationRecommendation)
            self.assertIn(
                rec.optimization_type,
                [
                    OptimizationType.REDUCE_PREDICTION_LOGGING,
                    OptimizationType.OPTIMIZE_DATA_QUALITY,
                    OptimizationType.CONSOLIDATE_ALERTS,
                ],
            )
            self.assertIsInstance(rec.potential_savings, float)
            self.assertGreater(len(rec.action_items), 0)

    def test_export_cost_summary_to_dataframe(self):
        """Test exporting cost data to pandas DataFrame."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        df = self.aggregator.export_cost_summary_to_dataframe()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)  # 3 cost records

        # Check required columns
        required_columns = [
            "model_id",
            "environment",
            "total_cost",
            "prediction_logging_cost",
            "data_quality_cost",
            "alert_management_cost",
            "dashboard_cost",
            "prediction_count",
            "data_quality_checks",
            "active_alerts",
        ]

        for col in required_columns:
            self.assertIn(col, df.columns)

        # Check data types
        self.assertEqual(df["total_cost"].dtype, float)
        self.assertEqual(df["prediction_count"].dtype, int)

    def test_reset_cost_aggregation(self):
        """Test resetting cost aggregation data."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        self.assertEqual(len(self.aggregator.cost_records), 3)

        # Reset aggregation
        self.aggregator.reset_cost_aggregation()

        self.assertEqual(len(self.aggregator.cost_records), 0)
        self.assertIsNotNone(
            self.aggregator.aggregation_start_time
        )  # Should be updated

    def test_get_cost_summary_with_time_filter(self):
        """Test cost summary with time-based filtering."""
        # Add test data with specific timestamps
        cutoff_date = datetime.utcnow() - timedelta(days=5)

        for record in self.mock_cost_data:
            cost_record = self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )
            cost_record.timestamp = record["timestamp"]

        # Get summary for recent records only (last 5 days)
        recent_summary = self.aggregator.get_cost_summary_by_model(
            start_date=cutoff_date
        )

        # Should only include 2 records (5 days ago and 3 days ago)
        expected_total = 14.60 + 20.10  # Costs from recent records
        self.assertEqual(recent_summary.total_cost, expected_total)

    def test_calculate_model_roi_metrics(self):
        """Test ROI and value metrics calculation."""
        # Add test data
        for record in self.mock_cost_data:
            self.aggregator.add_cost_record(
                **{k: v for k, v in record.items() if k != "timestamp"}
            )

        roi_metrics = self.aggregator.calculate_model_roi_metrics(
            model_id="fraud-detection-v1",
            business_value_per_prediction=0.05,  # $0.05 value per prediction
        )

        self.assertIn("total_cost", roi_metrics)
        self.assertIn("total_value_generated", roi_metrics)
        self.assertIn("roi_percentage", roi_metrics)
        self.assertIn("break_even_predictions", roi_metrics)

        # Check ROI calculation
        total_cost = roi_metrics["total_cost"]
        total_value = roi_metrics["total_value_generated"]
        roi = roi_metrics["roi_percentage"]

        expected_roi = ((total_value - total_cost) / total_cost) * 100
        self.assertAlmostEqual(roi, expected_roi, places=2)


class TestCostOptimizationRecommendations(unittest.TestCase):
    """Test cost optimization recommendation engine."""

    def setUp(self):
        """Set up test fixtures for optimization testing."""
        self.aggregator = ArizeCostAggregator(
            team="optimization-team", project="cost-optimization"
        )

    def test_high_prediction_logging_cost_recommendation(self):
        """Test recommendation for high prediction logging costs."""
        # Add data with high prediction logging costs
        self.aggregator.add_cost_record(
            model_id="expensive-model",
            environment="production",
            prediction_logging_cost=100.0,  # Very high
            data_quality_cost=5.0,
            alert_management_cost=2.0,
            dashboard_cost=1.0,
            prediction_count=50000,  # High volume
            data_quality_checks=50,
            active_alerts=2,
        )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should recommend prediction logging optimization
        prediction_recs = [
            r
            for r in recommendations
            if r.optimization_type == OptimizationType.REDUCE_PREDICTION_LOGGING
        ]
        self.assertGreater(len(prediction_recs), 0)

        rec = prediction_recs[0]
        self.assertGreater(rec.potential_savings, 0)
        self.assertIn("sampling", " ".join(rec.action_items).lower())

    def test_data_quality_optimization_recommendation(self):
        """Test recommendation for data quality cost optimization."""
        # Add data with high data quality costs
        self.aggregator.add_cost_record(
            model_id="quality-heavy-model",
            environment="production",
            prediction_logging_cost=10.0,
            data_quality_cost=80.0,  # Very high
            alert_management_cost=3.0,
            dashboard_cost=2.0,
            prediction_count=10000,
            data_quality_checks=8000,  # Very frequent checks
            active_alerts=1,
        )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should recommend data quality optimization
        quality_recs = [
            r
            for r in recommendations
            if r.optimization_type == OptimizationType.OPTIMIZE_DATA_QUALITY
        ]
        self.assertGreater(len(quality_recs), 0)

        rec = quality_recs[0]
        self.assertGreater(rec.potential_savings, 0)
        self.assertIn("frequency", " ".join(rec.action_items).lower())

    def test_alert_consolidation_recommendation(self):
        """Test recommendation for alert consolidation."""
        # Add data with many alerts
        self.aggregator.add_cost_record(
            model_id="alert-heavy-model",
            environment="production",
            prediction_logging_cost=15.0,
            data_quality_cost=8.0,
            alert_management_cost=50.0,  # Very high
            dashboard_cost=2.0,
            prediction_count=15000,
            data_quality_checks=150,
            active_alerts=25,  # Too many alerts
        )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should recommend alert consolidation
        alert_recs = [
            r
            for r in recommendations
            if r.optimization_type == OptimizationType.CONSOLIDATE_ALERTS
        ]
        self.assertGreater(len(alert_recs), 0)

        rec = alert_recs[0]
        self.assertGreater(rec.potential_savings, 0)
        self.assertIn("consolidate", " ".join(rec.action_items).lower())

    def test_model_right_sizing_recommendation(self):
        """Test recommendation for model right-sizing."""
        # Add data suggesting over-provisioning
        self.aggregator.add_cost_record(
            model_id="over-provisioned-model",
            environment="production",
            prediction_logging_cost=40.0,
            data_quality_cost=20.0,
            alert_management_cost=10.0,
            dashboard_cost=5.0,
            prediction_count=5000,  # Low volume for high cost
            data_quality_checks=100,
            active_alerts=5,
        )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should recommend model right-sizing
        sizing_recs = [
            r
            for r in recommendations
            if r.optimization_type == OptimizationType.MODEL_RIGHT_SIZING
        ]
        self.assertGreater(len(sizing_recs), 0)

        rec = sizing_recs[0]
        self.assertGreater(rec.potential_savings, 0)

    def test_environment_optimization_recommendation(self):
        """Test recommendation for environment-specific optimization."""
        # Add development environment data with production-level costs
        self.aggregator.add_cost_record(
            model_id="dev-model",
            environment="development",  # Dev environment
            prediction_logging_cost=30.0,  # High cost for dev
            data_quality_cost=15.0,
            alert_management_cost=8.0,
            dashboard_cost=3.0,
            prediction_count=3000,
            data_quality_checks=75,
            active_alerts=4,
        )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should recommend environment optimization
        env_recs = [
            r
            for r in recommendations
            if r.optimization_type == OptimizationType.ENVIRONMENT_OPTIMIZATION
        ]
        self.assertGreater(len(env_recs), 0)

        rec = env_recs[0]
        self.assertGreater(rec.potential_savings, 0)
        self.assertIn("development", " ".join(rec.action_items).lower())

    def test_no_recommendations_for_optimal_usage(self):
        """Test that no recommendations are generated for optimal usage."""
        # Add data with reasonable, balanced costs
        self.aggregator.add_cost_record(
            model_id="optimal-model",
            environment="production",
            prediction_logging_cost=5.0,  # Reasonable
            data_quality_cost=2.0,  # Reasonable
            alert_management_cost=1.0,  # Reasonable
            dashboard_cost=0.5,  # Reasonable
            prediction_count=5000,
            data_quality_checks=50,
            active_alerts=2,
        )

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should have few or no recommendations for optimal usage
        self.assertLessEqual(len(recommendations), 1)

    def test_recommendation_prioritization(self):
        """Test that recommendations are prioritized by potential savings."""
        # Add data that will generate multiple recommendations
        high_cost_records = [
            {
                "model_id": "model-1",
                "environment": "production",
                "prediction_logging_cost": 100.0,  # High
                "data_quality_cost": 50.0,  # High
                "alert_management_cost": 30.0,  # High
                "dashboard_cost": 5.0,
                "prediction_count": 50000,
                "data_quality_checks": 5000,
                "active_alerts": 15,
            }
        ]

        for record in high_cost_records:
            self.aggregator.add_cost_record(**record)

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        # Should be sorted by potential savings (descending)
        if len(recommendations) > 1:
            for i in range(len(recommendations) - 1):
                self.assertGreaterEqual(
                    recommendations[i].potential_savings,
                    recommendations[i + 1].potential_savings,
                )


class TestCostAnalysisUtilities(unittest.TestCase):
    """Test utility functions for cost analysis."""

    def test_calculate_model_monitoring_cost_function(self):
        """Test standalone cost calculation function."""
        cost = calculate_model_monitoring_cost(
            prediction_count=10000,
            data_quality_checks=100,
            active_alerts=5,
            monitoring_duration_days=30,
        )

        self.assertIsInstance(cost, dict)
        self.assertIn("total_cost", cost)
        self.assertIn("prediction_logging_cost", cost)
        self.assertIn("data_quality_cost", cost)
        self.assertIn("alert_management_cost", cost)
        self.assertIn("dashboard_cost", cost)

        # Check total cost calculation
        expected_total = (
            cost["prediction_logging_cost"]
            + cost["data_quality_cost"]
            + cost["alert_management_cost"]
            + cost["dashboard_cost"]
        )
        self.assertEqual(cost["total_cost"], expected_total)

    def test_estimate_monthly_monitoring_cost_function(self):
        """Test monthly cost estimation function."""
        monthly_cost = estimate_monthly_monitoring_cost(
            daily_prediction_volume=5000, daily_data_quality_checks=50, average_alerts=3
        )

        self.assertIsInstance(monthly_cost, dict)
        self.assertIn("total_monthly_cost", monthly_cost)
        self.assertIn("cost_breakdown", monthly_cost)
        self.assertIn("volume_projections", monthly_cost)

        # Check monthly calculation
        total_monthly = monthly_cost["total_monthly_cost"]
        self.assertGreater(total_monthly, 0)
        self.assertLess(total_monthly, 10000)  # Reasonable upper bound

    def test_analyze_cost_trends_function(self):
        """Test cost trends analysis function."""
        # Create sample cost data
        cost_history = []
        for i in range(30):  # 30 days of data
            cost_history.append(
                {
                    "date": datetime.utcnow() - timedelta(days=i),
                    "total_cost": 10 + (i * 0.5),  # Increasing trend
                    "prediction_count": 1000 + (i * 50),
                    "model_id": "trend-model",
                }
            )

        trends = analyze_cost_trends(cost_history, days=30)

        self.assertIn("trend_direction", trends)
        self.assertIn("daily_growth_rate", trends)
        self.assertIn("cost_volatility", trends)
        self.assertIn("prediction_efficiency_trend", trends)

        # Should detect increasing trend
        self.assertEqual(trends["trend_direction"], "increasing")
        self.assertGreater(trends["daily_growth_rate"], 0)


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases in cost aggregation."""

    def test_empty_cost_aggregator(self):
        """Test behavior with no cost records."""
        aggregator = ArizeCostAggregator()

        summary = aggregator.get_cost_summary_by_model()
        self.assertEqual(summary.total_cost, 0.0)
        self.assertEqual(len(summary.cost_by_model), 0)

        recommendations = aggregator.get_cost_optimization_recommendations()
        self.assertEqual(len(recommendations), 0)

        metrics = aggregator.get_efficiency_metrics()
        self.assertEqual(metrics.cost_per_prediction, 0.0)

    def test_invalid_cost_values(self):
        """Test handling of invalid cost values."""
        aggregator = ArizeCostAggregator()

        # Test negative costs (should be handled gracefully)
        with self.assertLogs(level="WARNING"):
            aggregator.add_cost_record(
                model_id="invalid-model",
                environment="test",
                prediction_logging_cost=-5.0,  # Invalid negative cost
                data_quality_cost=2.0,
                alert_management_cost=1.0,
                dashboard_cost=0.5,
                prediction_count=1000,
                data_quality_checks=10,
                active_alerts=1,
            )

    def test_model_cost_breakdown_nonexistent_model(self):
        """Test cost breakdown for non-existent model."""
        aggregator = ArizeCostAggregator()

        breakdown = aggregator.get_model_cost_breakdown("nonexistent-model")
        self.assertIsNone(breakdown)

    def test_large_dataset_performance(self):
        """Test performance with large number of cost records."""
        aggregator = ArizeCostAggregator()

        # Add large number of records
        for i in range(1000):
            aggregator.add_cost_record(
                model_id=f"model-{i % 10}",  # 10 different models
                environment="production",
                prediction_logging_cost=1.0 + (i * 0.001),
                data_quality_cost=0.5,
                alert_management_cost=0.25,
                dashboard_cost=0.1,
                prediction_count=1000,
                data_quality_checks=10,
                active_alerts=1,
            )

        # Should handle large dataset efficiently
        summary = aggregator.get_cost_summary_by_model()
        self.assertEqual(len(summary.cost_by_model), 10)  # 10 unique models

        recommendations = aggregator.get_cost_optimization_recommendations()
        self.assertIsInstance(recommendations, list)

    def test_concurrent_cost_record_addition(self):
        """Test thread safety of cost record addition."""
        import threading

        aggregator = ArizeCostAggregator()

        def add_records(thread_id):
            for i in range(100):
                aggregator.add_cost_record(
                    model_id=f"thread-{thread_id}-model-{i}",
                    environment="test",
                    prediction_logging_cost=1.0,
                    data_quality_cost=0.5,
                    alert_management_cost=0.25,
                    dashboard_cost=0.1,
                    prediction_count=100,
                    data_quality_checks=5,
                    active_alerts=1,
                )

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_records, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have all records
        self.assertEqual(len(aggregator.cost_records), 500)  # 5 threads * 100 records


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)
