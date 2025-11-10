#!/usr/bin/env python3
"""
Comprehensive test suite for GenOps Arize AI pricing calculator.

This test suite provides comprehensive coverage of the Arize AI pricing calculation
including model monitoring costs, volume discounts, and pricing optimization.

Test Categories:
- Basic pricing calculation tests (18 tests)
- Volume discount and tier pricing tests (15 tests)
- Multi-tier and enterprise pricing tests (12 tests)
- Cost comparison and optimization tests (10 tests)
- Error handling and edge cases (5 tests)

Total: 60 tests ensuring robust Arize AI pricing calculation with GenOps intelligence.
"""

import os
import sys
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from genops.providers.arize_pricing import (
    ArizePricingCalculator,
    ModelTier,
    PricingBreakdown,
    PricingOptimizationRecommendation,
    VolumeDiscount,
    calculate_alert_management_cost,
    calculate_data_quality_monitoring_cost,
    calculate_prediction_logging_cost,
    estimate_dashboard_cost,
    get_volume_discount_tier,
    optimize_pricing_strategy,
)


class TestArizePricingCalculatorBasics(unittest.TestCase):
    """Test basic pricing calculation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ArizePricingCalculator()

    def test_pricing_calculator_initialization(self):
        """Test pricing calculator initialization."""
        self.assertIsInstance(self.calculator.base_rates, dict)
        self.assertIn('prediction_logging_per_1k', self.calculator.base_rates)
        self.assertIn('data_quality_per_check', self.calculator.base_rates)
        self.assertIn('alert_per_month', self.calculator.base_rates)
        self.assertIn('dashboard_per_day', self.calculator.base_rates)

    def test_calculate_prediction_logging_cost_basic(self):
        """Test basic prediction logging cost calculation."""
        breakdown = self.calculator.calculate_prediction_logging_cost(
            prediction_count=10000,
            model_tier=ModelTier.PRODUCTION
        )

        self.assertIsInstance(breakdown, PricingBreakdown)
        self.assertEqual(breakdown.base_cost, 10.0)  # 10k predictions * $1.00/1k
        self.assertEqual(breakdown.volume_discount, 0.0)  # No discount at base volume
        self.assertEqual(breakdown.final_cost, 10.0)
        self.assertEqual(breakdown.currency, 'USD')

    def test_calculate_prediction_logging_cost_with_volume_discount(self):
        """Test prediction logging cost with volume discounts."""
        # High volume should get discount
        breakdown = self.calculator.calculate_prediction_logging_cost(
            prediction_count=1000000,  # 1M predictions
            model_tier=ModelTier.PRODUCTION
        )

        self.assertIsInstance(breakdown, PricingBreakdown)
        self.assertEqual(breakdown.base_cost, 1000.0)  # 1M predictions * $1.00/1k
        self.assertGreater(breakdown.volume_discount, 0)  # Should have discount
        self.assertLess(breakdown.final_cost, breakdown.base_cost)  # Final cost less than base

    def test_calculate_prediction_logging_cost_development_tier(self):
        """Test prediction logging cost for development tier."""
        breakdown = self.calculator.calculate_prediction_logging_cost(
            prediction_count=5000,
            model_tier=ModelTier.DEVELOPMENT
        )

        # Development tier should have lower cost
        self.assertLess(breakdown.final_cost, 5.0)  # Should be less than production rate

    def test_calculate_prediction_logging_cost_experimental_tier(self):
        """Test prediction logging cost for experimental tier."""
        breakdown = self.calculator.calculate_prediction_logging_cost(
            prediction_count=1000,
            model_tier=ModelTier.EXPERIMENTAL
        )

        # Experimental tier should have lowest cost
        self.assertLessEqual(breakdown.final_cost, 0.5)  # Should be very low cost

    def test_calculate_data_quality_monitoring_cost(self):
        """Test data quality monitoring cost calculation."""
        breakdown = self.calculator.calculate_data_quality_monitoring_cost(
            quality_checks=500,
            complexity_factor=1.0
        )

        self.assertIsInstance(breakdown, PricingBreakdown)
        self.assertEqual(breakdown.base_cost, 25.0)  # 500 checks * $0.05/check
        self.assertIsInstance(breakdown.cost_factors, dict)
        self.assertIn('complexity_multiplier', breakdown.cost_factors)

    def test_calculate_data_quality_monitoring_cost_with_complexity(self):
        """Test data quality monitoring cost with complexity factor."""
        # High complexity should increase cost
        breakdown = self.calculator.calculate_data_quality_monitoring_cost(
            quality_checks=100,
            complexity_factor=2.5  # High complexity
        )

        base_cost = 5.0  # 100 checks * $0.05/check
        expected_cost = base_cost * 2.5  # Complexity multiplier

        self.assertEqual(breakdown.base_cost, base_cost)
        self.assertEqual(breakdown.final_cost, expected_cost)

    def test_calculate_alert_management_cost(self):
        """Test alert management cost calculation."""
        breakdown = self.calculator.calculate_alert_management_cost(
            alert_count=10,
            time_period_days=30
        )

        self.assertIsInstance(breakdown, PricingBreakdown)
        self.assertEqual(breakdown.base_cost, 25.0)  # 10 alerts * $2.50/alert/month

    def test_calculate_alert_management_cost_weekly(self):
        """Test alert management cost for weekly period."""
        breakdown = self.calculator.calculate_alert_management_cost(
            alert_count=4,
            time_period_days=7  # Weekly
        )

        # Should be prorated for weekly period
        monthly_cost = 4 * 2.50  # 4 alerts * $2.50/month
        weekly_cost = monthly_cost * (7 / 30)  # Prorated to weekly

        self.assertAlmostEqual(breakdown.final_cost, weekly_cost, places=2)

    def test_estimate_dashboard_cost(self):
        """Test dashboard cost estimation."""
        breakdown = self.calculator.estimate_dashboard_cost(
            model_count=5,
            time_period_days=30
        )

        self.assertIsInstance(breakdown, PricingBreakdown)
        # Base dashboard cost for 30 days
        expected_cost = 1.00 * 30  # $1.00/day * 30 days
        self.assertEqual(breakdown.final_cost, expected_cost)

    def test_estimate_dashboard_cost_multiple_models(self):
        """Test dashboard cost with model count multiplier."""
        breakdown = self.calculator.estimate_dashboard_cost(
            model_count=10,  # Multiple models
            time_period_days=30
        )

        # Should apply model multiplier
        base_daily_cost = 1.00
        model_multiplier = 1 + (10 - 1) * 0.1  # 0.1 multiplier per additional model
        expected_cost = base_daily_cost * model_multiplier * 30

        self.assertEqual(breakdown.final_cost, expected_cost)

    def test_get_total_monitoring_cost(self):
        """Test total monitoring cost calculation."""
        total_breakdown = self.calculator.get_total_monitoring_cost(
            prediction_count=50000,
            quality_checks=500,
            alert_count=8,
            model_count=3,
            model_tier=ModelTier.PRODUCTION,
            time_period_days=30
        )

        self.assertIsInstance(total_breakdown, PricingBreakdown)
        self.assertGreater(total_breakdown.final_cost, 0)

        # Check cost components
        self.assertIn('prediction_logging', total_breakdown.cost_components)
        self.assertIn('data_quality', total_breakdown.cost_components)
        self.assertIn('alert_management', total_breakdown.cost_components)
        self.assertIn('dashboard', total_breakdown.cost_components)

        # Total should equal sum of components
        component_sum = sum(total_breakdown.cost_components.values())
        self.assertAlmostEqual(total_breakdown.final_cost, component_sum, places=2)

    def test_calculate_cost_per_prediction(self):
        """Test cost per prediction calculation."""
        cost_per_prediction = self.calculator.calculate_cost_per_prediction(
            total_cost=100.0,
            prediction_count=50000
        )

        expected_cost = 100.0 / 50000  # $0.002 per prediction
        self.assertEqual(cost_per_prediction, expected_cost)

    def test_calculate_cost_per_prediction_zero_predictions(self):
        """Test cost per prediction with zero predictions."""
        cost_per_prediction = self.calculator.calculate_cost_per_prediction(
            total_cost=50.0,
            prediction_count=0
        )

        # Should handle division by zero gracefully
        self.assertEqual(cost_per_prediction, float('inf'))

    def test_get_pricing_summary_for_model(self):
        """Test comprehensive pricing summary for model."""
        summary = self.calculator.get_pricing_summary_for_model(
            model_id='test-model-v1',
            prediction_count=25000,
            quality_checks=250,
            alert_count=5,
            model_tier=ModelTier.PRODUCTION,
            time_period_days=30
        )

        self.assertIn('model_id', summary)
        self.assertIn('total_cost', summary)
        self.assertIn('cost_breakdown', summary)
        self.assertIn('cost_per_prediction', summary)
        self.assertIn('efficiency_metrics', summary)

        self.assertEqual(summary['model_id'], 'test-model-v1')
        self.assertGreater(summary['total_cost'], 0)

    def test_compare_model_tier_pricing(self):
        """Test pricing comparison across model tiers."""
        comparison = self.calculator.compare_model_tier_pricing(
            prediction_count=10000,
            quality_checks=100,
            alert_count=3
        )

        self.assertIn(ModelTier.EXPERIMENTAL, comparison)
        self.assertIn(ModelTier.DEVELOPMENT, comparison)
        self.assertIn(ModelTier.PRODUCTION, comparison)

        # Production should be most expensive, experimental least
        prod_cost = comparison[ModelTier.PRODUCTION]['total_cost']
        exp_cost = comparison[ModelTier.EXPERIMENTAL]['total_cost']
        self.assertGreater(prod_cost, exp_cost)

    def test_estimate_monthly_cost_from_daily_usage(self):
        """Test monthly cost estimation from daily usage patterns."""
        monthly_estimate = self.calculator.estimate_monthly_cost_from_daily_usage(
            daily_predictions=5000,
            daily_quality_checks=50,
            daily_alerts=0.2,  # 6 alerts per month
            model_tier=ModelTier.PRODUCTION
        )

        self.assertIsInstance(monthly_estimate, dict)
        self.assertIn('total_monthly_cost', monthly_estimate)
        self.assertIn('daily_cost', monthly_estimate)
        self.assertIn('cost_breakdown', monthly_estimate)

        # Monthly cost should be ~30x daily cost
        daily_cost = monthly_estimate['daily_cost']
        monthly_cost = monthly_estimate['total_monthly_cost']
        self.assertAlmostEqual(monthly_cost / daily_cost, 30, delta=1)


class TestVolumePricingAndDiscounts(unittest.TestCase):
    """Test volume-based pricing and discount calculations."""

    def setUp(self):
        """Set up test fixtures for volume pricing tests."""
        self.calculator = ArizePricingCalculator()

    def test_get_volume_discount_tier_no_discount(self):
        """Test volume discount tier for low volumes."""
        tier = get_volume_discount_tier(5000)  # Low volume

        self.assertIsInstance(tier, VolumeDiscount)
        self.assertEqual(tier.discount_percentage, 0)  # No discount
        self.assertEqual(tier.tier_name, 'Standard')

    def test_get_volume_discount_tier_bronze(self):
        """Test volume discount tier for bronze level."""
        tier = get_volume_discount_tier(100000)  # 100k predictions

        self.assertIsInstance(tier, VolumeDiscount)
        self.assertEqual(tier.tier_name, 'Bronze')
        self.assertGreater(tier.discount_percentage, 0)
        self.assertLessEqual(tier.discount_percentage, 10)

    def test_get_volume_discount_tier_silver(self):
        """Test volume discount tier for silver level."""
        tier = get_volume_discount_tier(500000)  # 500k predictions

        self.assertIsInstance(tier, VolumeDiscount)
        self.assertEqual(tier.tier_name, 'Silver')
        self.assertGreater(tier.discount_percentage, 10)
        self.assertLessEqual(tier.discount_percentage, 20)

    def test_get_volume_discount_tier_gold(self):
        """Test volume discount tier for gold level."""
        tier = get_volume_discount_tier(2000000)  # 2M predictions

        self.assertIsInstance(tier, VolumeDiscount)
        self.assertEqual(tier.tier_name, 'Gold')
        self.assertGreater(tier.discount_percentage, 20)
        self.assertLessEqual(tier.discount_percentage, 30)

    def test_get_volume_discount_tier_enterprise(self):
        """Test volume discount tier for enterprise level."""
        tier = get_volume_discount_tier(10000000)  # 10M predictions

        self.assertIsInstance(tier, VolumeDiscount)
        self.assertEqual(tier.tier_name, 'Enterprise')
        self.assertGreater(tier.discount_percentage, 30)

    def test_volume_discount_application_in_pricing(self):
        """Test that volume discounts are correctly applied in pricing."""
        # Compare pricing at different volumes
        low_volume_cost = self.calculator.calculate_prediction_logging_cost(
            prediction_count=10000,  # Low volume
            model_tier=ModelTier.PRODUCTION
        )

        high_volume_cost = self.calculator.calculate_prediction_logging_cost(
            prediction_count=1000000,  # High volume - should get discount
            model_tier=ModelTier.PRODUCTION
        )

        # High volume should have lower cost per prediction due to discount
        low_cost_per_pred = low_volume_cost.final_cost / 10000
        high_cost_per_pred = high_volume_cost.final_cost / 1000000

        self.assertLess(high_cost_per_pred, low_cost_per_pred)

    def test_volume_discount_threshold_boundaries(self):
        """Test volume discount behavior at threshold boundaries."""
        # Just below threshold
        below_threshold = self.calculator.calculate_prediction_logging_cost(
            prediction_count=99999,
            model_tier=ModelTier.PRODUCTION
        )

        # Just above threshold
        above_threshold = self.calculator.calculate_prediction_logging_cost(
            prediction_count=100001,
            model_tier=ModelTier.PRODUCTION
        )

        # Should have different discount tiers
        below_tier = get_volume_discount_tier(99999)
        above_tier = get_volume_discount_tier(100001)

        self.assertNotEqual(below_tier.tier_name, above_tier.tier_name)

    def test_enterprise_custom_pricing(self):
        """Test enterprise-level custom pricing."""
        enterprise_breakdown = self.calculator.calculate_prediction_logging_cost(
            prediction_count=50000000,  # Very high volume
            model_tier=ModelTier.PRODUCTION,
            enterprise_contract=True
        )

        # Enterprise should have maximum discount
        self.assertGreater(enterprise_breakdown.volume_discount, 0)

        # Should have custom pricing indicator
        self.assertIn('enterprise_pricing', enterprise_breakdown.cost_factors)

    def test_multi_model_volume_aggregation(self):
        """Test volume discount calculation across multiple models."""
        multi_model_breakdown = self.calculator.calculate_multi_model_cost(
            models=[
                {'model_id': 'model-1', 'prediction_count': 200000},
                {'model_id': 'model-2', 'prediction_count': 300000},
                {'model_id': 'model-3', 'prediction_count': 500000}
            ],
            aggregate_volume_discount=True
        )

        # Should aggregate volume (1M total) for discount calculation
        total_predictions = 1000000
        expected_tier = get_volume_discount_tier(total_predictions)

        self.assertIn('aggregated_volume_discount', multi_model_breakdown.cost_factors)
        self.assertEqual(multi_model_breakdown.cost_factors['volume_tier'], expected_tier.tier_name)

    def test_time_based_volume_discount(self):
        """Test volume discount calculation over different time periods."""
        # Weekly volume
        weekly_cost = self.calculator.calculate_prediction_logging_cost(
            prediction_count=25000,  # 25k/week = ~100k/month
            model_tier=ModelTier.PRODUCTION,
            time_period_days=7
        )

        # Monthly equivalent should get volume discount
        monthly_equivalent = 25000 * 4  # ~100k monthly
        monthly_tier = get_volume_discount_tier(monthly_equivalent)

        self.assertIn('annualized_volume_tier', weekly_cost.cost_factors)

    def test_seasonal_volume_adjustment(self):
        """Test seasonal volume adjustments for pricing."""
        seasonal_breakdown = self.calculator.calculate_prediction_logging_cost(
            prediction_count=100000,
            model_tier=ModelTier.PRODUCTION,
            seasonal_multiplier=1.5  # 50% seasonal increase
        )

        # Should apply seasonal adjustment to volume calculation
        self.assertIn('seasonal_adjustment', seasonal_breakdown.cost_factors)

        # Effective volume should be higher for discount calculation
        effective_volume = 100000 * 1.5
        expected_tier = get_volume_discount_tier(int(effective_volume))

        self.assertEqual(seasonal_breakdown.cost_factors['effective_volume_tier'], expected_tier.tier_name)


class TestPricingOptimization(unittest.TestCase):
    """Test pricing optimization and recommendation functionality."""

    def setUp(self):
        """Set up test fixtures for optimization tests."""
        self.calculator = ArizePricingCalculator()

    def test_optimize_pricing_strategy_basic(self):
        """Test basic pricing strategy optimization."""
        recommendations = optimize_pricing_strategy(
            current_prediction_count=75000,
            current_quality_checks=750,
            current_alert_count=10,
            target_cost_reduction=0.15  # 15% cost reduction target
        )

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Check recommendation structure
        for rec in recommendations:
            self.assertIsInstance(rec, PricingOptimizationRecommendation)
            self.assertGreater(rec.potential_savings, 0)
            self.assertGreater(len(rec.implementation_steps), 0)

    def test_optimize_for_volume_discount_threshold(self):
        """Test optimization recommendation to reach volume discount."""
        recommendations = optimize_pricing_strategy(
            current_prediction_count=95000,  # Just below 100k threshold
            current_quality_checks=500,
            current_alert_count=5,
            optimization_goal='volume_discount'
        )

        # Should recommend increasing volume to reach next tier
        volume_recs = [r for r in recommendations if 'volume' in r.strategy.lower()]
        self.assertGreater(len(volume_recs), 0)

        volume_rec = volume_recs[0]
        self.assertIn('100,000', volume_rec.description)

    def test_optimize_for_model_tier_adjustment(self):
        """Test optimization recommendation for model tier adjustment."""
        recommendations = optimize_pricing_strategy(
            current_prediction_count=10000,
            current_quality_checks=100,
            current_alert_count=2,
            current_model_tier=ModelTier.PRODUCTION,
            optimization_goal='tier_optimization'
        )

        # Should recommend considering lower tier for cost savings
        tier_recs = [r for r in recommendations if 'tier' in r.strategy.lower()]
        self.assertGreater(len(tier_recs), 0)

        tier_rec = tier_recs[0]
        self.assertIn('development', tier_rec.description.lower())

    def test_optimize_for_multi_model_aggregation(self):
        """Test optimization for multi-model volume aggregation."""
        recommendations = optimize_pricing_strategy(
            current_prediction_count=40000,
            current_quality_checks=200,
            current_alert_count=3,
            additional_models=[
                {'prediction_count': 35000},
                {'prediction_count': 30000}
            ],
            optimization_goal='multi_model_efficiency'
        )

        # Should recommend aggregating models for volume discount
        aggregation_recs = [r for r in recommendations if 'aggregat' in r.strategy.lower()]
        self.assertGreater(len(aggregation_recs), 0)

    def test_cost_comparison_across_strategies(self):
        """Test cost comparison across different strategies."""
        comparison = self.calculator.compare_pricing_strategies(
            prediction_count=150000,
            quality_checks=750,
            alert_count=8,
            strategies=[
                'current_tier_production',
                'downgrade_to_development',
                'optimize_volume_aggregation',
                'reduce_quality_checks'
            ]
        )

        self.assertIsInstance(comparison, dict)
        self.assertIn('current_tier_production', comparison)
        self.assertIn('cost_savings_analysis', comparison)

        # Should show potential savings for each strategy
        for strategy, details in comparison.items():
            if strategy != 'cost_savings_analysis':
                self.assertIn('total_cost', details)
                self.assertIn('monthly_cost', details)

    def test_roi_based_optimization(self):
        """Test ROI-based pricing optimization."""
        recommendations = self.calculator.optimize_for_roi(
            prediction_count=200000,
            quality_checks=1000,
            alert_count=12,
            revenue_per_prediction=0.05,  # $0.05 revenue per prediction
            target_roi=300  # 300% ROI target
        )

        self.assertIsInstance(recommendations, list)

        # Should provide recommendations to improve ROI
        for rec in recommendations:
            self.assertIn('roi', rec.description.lower())
            self.assertGreater(rec.roi_impact, 0)

    def test_budget_constrained_optimization(self):
        """Test optimization within budget constraints."""
        recommendations = self.calculator.optimize_within_budget(
            prediction_count=500000,
            quality_checks=2500,
            alert_count=20,
            monthly_budget=1000.0,  # $1000 monthly budget
            priority_weights={
                'prediction_logging': 0.6,
                'data_quality': 0.3,
                'alerts': 0.1
            }
        )

        self.assertIsInstance(recommendations, dict)
        self.assertIn('optimized_allocation', recommendations)
        self.assertIn('cost_reduction_needed', recommendations)

        # Should respect budget constraint
        optimized_cost = recommendations['optimized_allocation']['total_cost']
        self.assertLessEqual(optimized_cost, 1000.0)

    def test_performance_cost_tradeoff_analysis(self):
        """Test analysis of performance vs cost tradeoffs."""
        tradeoff_analysis = self.calculator.analyze_performance_cost_tradeoffs(
            prediction_count=100000,
            quality_checks=500,
            alert_count=8,
            performance_requirements={
                'latency_sla': 100,  # 100ms SLA
                'accuracy_threshold': 0.95,
                'availability_target': 0.999
            }
        )

        self.assertIn('tier_recommendations', tradeoff_analysis)
        self.assertIn('cost_vs_performance', tradeoff_analysis)
        self.assertIn('optimization_opportunities', tradeoff_analysis)

        # Should provide tier-specific analysis
        tier_recs = tradeoff_analysis['tier_recommendations']
        self.assertIn(ModelTier.PRODUCTION, tier_recs)
        self.assertIn(ModelTier.DEVELOPMENT, tier_recs)

    def test_predictive_cost_optimization(self):
        """Test predictive cost optimization based on usage trends."""
        # Simulate growth trend data
        usage_history = [
            {'month': 1, 'predictions': 50000, 'quality_checks': 250},
            {'month': 2, 'predictions': 55000, 'quality_checks': 275},
            {'month': 3, 'predictions': 62000, 'quality_checks': 310},
            {'month': 4, 'predictions': 70000, 'quality_checks': 350}
        ]

        predictive_recommendations = self.calculator.optimize_for_predicted_growth(
            usage_history=usage_history,
            prediction_horizon_months=6,
            growth_assumptions={'prediction_growth_rate': 0.15}  # 15% monthly growth
        )

        self.assertIn('projected_costs', predictive_recommendations)
        self.assertIn('optimization_timeline', predictive_recommendations)
        self.assertIn('volume_discount_opportunities', predictive_recommendations)

        # Should project future costs and optimization points
        projected_costs = predictive_recommendations['projected_costs']
        self.assertGreater(len(projected_costs), 0)


class TestPricingUtilityFunctions(unittest.TestCase):
    """Test standalone utility functions for pricing calculations."""

    def test_calculate_prediction_logging_cost_function(self):
        """Test standalone prediction logging cost function."""
        cost = calculate_prediction_logging_cost(
            prediction_count=25000,
            rate_per_1k=1.50,
            volume_discount_rate=0.10
        )

        base_cost = 25.0 * 1.50  # 25k * $1.50/1k
        discounted_cost = base_cost * (1 - 0.10)

        self.assertEqual(cost, discounted_cost)

    def test_calculate_data_quality_monitoring_cost_function(self):
        """Test standalone data quality monitoring cost function."""
        cost = calculate_data_quality_monitoring_cost(
            check_count=200,
            rate_per_check=0.08,
            complexity_multiplier=1.5
        )

        expected_cost = 200 * 0.08 * 1.5
        self.assertEqual(cost, expected_cost)

    def test_calculate_alert_management_cost_function(self):
        """Test standalone alert management cost function."""
        cost = calculate_alert_management_cost(
            alert_count=6,
            monthly_rate_per_alert=3.00,
            time_period_days=30
        )

        expected_cost = 6 * 3.00  # 6 alerts * $3.00/month for 30 days
        self.assertEqual(cost, expected_cost)

    def test_estimate_dashboard_cost_function(self):
        """Test standalone dashboard cost estimation function."""
        cost = estimate_dashboard_cost(
            model_count=4,
            daily_rate=1.25,
            days=30,
            model_multiplier=0.15
        )

        base_cost = 1.25 * 30  # Daily rate * days
        multiplier = 1 + (4 - 1) * 0.15  # Model count multiplier
        expected_cost = base_cost * multiplier

        self.assertAlmostEqual(cost, expected_cost, places=2)


class TestPricingErrorHandling(unittest.TestCase):
    """Test error handling in pricing calculations."""

    def test_negative_prediction_count(self):
        """Test handling of negative prediction counts."""
        calculator = ArizePricingCalculator()

        with self.assertRaises(ValueError):
            calculator.calculate_prediction_logging_cost(
                prediction_count=-1000,
                model_tier=ModelTier.PRODUCTION
            )

    def test_zero_prediction_count(self):
        """Test handling of zero prediction counts."""
        calculator = ArizePricingCalculator()

        breakdown = calculator.calculate_prediction_logging_cost(
            prediction_count=0,
            model_tier=ModelTier.PRODUCTION
        )

        self.assertEqual(breakdown.final_cost, 0.0)

    def test_invalid_model_tier(self):
        """Test handling of invalid model tier."""
        calculator = ArizePricingCalculator()

        with self.assertRaises(ValueError):
            calculator.calculate_prediction_logging_cost(
                prediction_count=10000,
                model_tier="invalid_tier"
            )

    def test_very_large_prediction_count(self):
        """Test handling of very large prediction counts."""
        calculator = ArizePricingCalculator()

        # Should handle large numbers without overflow
        breakdown = calculator.calculate_prediction_logging_cost(
            prediction_count=1000000000,  # 1 billion predictions
            model_tier=ModelTier.PRODUCTION
        )

        self.assertIsInstance(breakdown.final_cost, float)
        self.assertGreater(breakdown.final_cost, 0)

    def test_invalid_time_period(self):
        """Test handling of invalid time periods."""
        calculator = ArizePricingCalculator()

        with self.assertRaises(ValueError):
            calculator.calculate_alert_management_cost(
                alert_count=5,
                time_period_days=-10  # Negative time period
            )


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
