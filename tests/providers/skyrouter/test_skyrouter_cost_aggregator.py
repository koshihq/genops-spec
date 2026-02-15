"""
Comprehensive tests for SkyRouter cost aggregation functionality.

Tests cost tracking, aggregation, budget monitoring, optimization
recommendations, and multi-dimensional cost attribution.
"""

from datetime import datetime, timedelta

import pytest

# Import the modules under test
try:
    from genops.providers.skyrouter_cost_aggregator import (
        BudgetStatus,  # noqa: F401
        CostBreakdown,  # noqa: F401
        CostSummary,
        OptimizationRecommendation,
        SkyRouterCostAggregator,
        UsageMetrics,
    )

    SKYROUTER_COST_AGGREGATOR_AVAILABLE = True
except ImportError:
    SKYROUTER_COST_AGGREGATOR_AVAILABLE = False


@pytest.mark.skipif(
    not SKYROUTER_COST_AGGREGATOR_AVAILABLE,
    reason="SkyRouter cost aggregator not available",
)
class TestSkyRouterCostAggregator:
    """Test suite for SkyRouter cost aggregator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = SkyRouterCostAggregator(
            team="test-team", project="test-project"
        )

    def test_cost_aggregator_initialization(self):
        """Test cost aggregator initialization."""
        aggregator = SkyRouterCostAggregator(
            team="init-test", project="aggregator-test", daily_budget_limit=100.0
        )

        assert aggregator.team == "init-test"
        assert aggregator.project == "aggregator-test"
        assert aggregator.daily_budget_limit == 100.0

    def test_add_operation_cost(self):
        """Test adding operation costs."""
        operation_data = {
            "model": "gpt-4",
            "cost": 0.05,
            "input_tokens": 1000,
            "output_tokens": 500,
            "routing_strategy": "balanced",
            "complexity": "moderate",
        }

        self.aggregator.add_operation_cost(**operation_data)

        # Verify cost was added
        summary = self.aggregator.get_summary()
        assert summary.total_cost >= 0.05
        assert summary.total_operations >= 1

    def test_multiple_operation_cost_tracking(self):
        """Test tracking multiple operation costs."""
        operations = [
            {
                "model": "gpt-4",
                "cost": 0.05,
                "input_tokens": 1000,
                "output_tokens": 500,
            },
            {
                "model": "claude-3-sonnet",
                "cost": 0.02,
                "input_tokens": 800,
                "output_tokens": 300,
            },
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.001,
                "input_tokens": 500,
                "output_tokens": 200,
            },
        ]

        for op in operations:
            self.aggregator.add_operation_cost(**op)

        summary = self.aggregator.get_summary()
        expected_total = sum(op["cost"] for op in operations)

        assert abs(summary.total_cost - expected_total) < 0.001
        assert summary.total_operations == len(operations)

    def test_cost_breakdown_by_model(self):
        """Test cost breakdown by model."""
        operations = [
            {"model": "gpt-4", "cost": 0.05},
            {"model": "gpt-4", "cost": 0.03},
            {"model": "claude-3-sonnet", "cost": 0.02},
            {"model": "gpt-3.5-turbo", "cost": 0.001},
        ]

        for op in operations:
            self.aggregator.add_operation_cost(**op)

        summary = self.aggregator.get_summary()

        assert "gpt-4" in summary.cost_by_model
        assert "claude-3-sonnet" in summary.cost_by_model
        assert "gpt-3.5-turbo" in summary.cost_by_model

        # GPT-4 should have highest cost (0.05 + 0.03 = 0.08)
        assert summary.cost_by_model["gpt-4"] >= 0.08

    def test_cost_breakdown_by_routing_strategy(self):
        """Test cost breakdown by routing strategy."""
        operations = [
            {"model": "gpt-4", "cost": 0.05, "routing_strategy": "cost_optimized"},
            {"model": "claude-3-sonnet", "cost": 0.02, "routing_strategy": "balanced"},
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.001,
                "routing_strategy": "latency_optimized",
            },
            {"model": "gpt-4", "cost": 0.04, "routing_strategy": "reliability_first"},
        ]

        for op in operations:
            self.aggregator.add_operation_cost(**op)

        summary = self.aggregator.get_summary()

        assert "cost_optimized" in summary.cost_by_route
        assert "balanced" in summary.cost_by_route
        assert "latency_optimized" in summary.cost_by_route
        assert "reliability_first" in summary.cost_by_route

    def test_budget_status_checking(self):
        """Test budget status checking and monitoring."""
        # Set a small budget for testing
        self.aggregator.daily_budget_limit = 0.10

        # Add operations within budget
        self.aggregator.add_operation_cost(model="gpt-3.5-turbo", cost=0.05)

        budget_status = self.aggregator.check_budget_status()

        assert budget_status["current_daily_cost"] == 0.05
        assert budget_status["daily_budget_limit"] == 0.10
        assert budget_status["budget_utilization"] == 50.0
        assert budget_status["budget_remaining"] == 0.05

    def test_budget_limit_exceeded(self):
        """Test budget limit exceeded detection."""
        # Set a very small budget
        self.aggregator.daily_budget_limit = 0.01

        # Add operation that exceeds budget
        self.aggregator.add_operation_cost(model="gpt-4", cost=0.05)

        budget_status = self.aggregator.check_budget_status()

        assert budget_status["budget_exceeded"] is True
        assert budget_status["budget_utilization"] > 100.0

    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendation generation."""
        # Add operations with different characteristics
        expensive_operations = [
            {
                "model": "gpt-4",
                "cost": 0.08,
                "routing_strategy": "reliability_first",
                "complexity": "enterprise",
            },
            {
                "model": "gpt-4",
                "cost": 0.07,
                "routing_strategy": "reliability_first",
                "complexity": "complex",
            },
            {
                "model": "claude-3-opus",
                "cost": 0.06,
                "routing_strategy": "reliability_first",
                "complexity": "complex",
            },
        ]

        cheap_operations = [
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.002,
                "routing_strategy": "cost_optimized",
                "complexity": "simple",
            },
            {
                "model": "claude-3-haiku",
                "cost": 0.001,
                "routing_strategy": "cost_optimized",
                "complexity": "simple",
            },
        ]

        all_operations = expensive_operations + cheap_operations

        for op in all_operations:
            self.aggregator.add_operation_cost(**op)

        recommendations = self.aggregator.get_cost_optimization_recommendations()

        assert isinstance(recommendations, list)
        # Should have recommendations due to expensive operations
        assert len(recommendations) > 0

        # Check recommendation structure
        for rec in recommendations:
            assert "title" in rec
            assert "potential_savings" in rec
            assert "priority_score" in rec
            assert "optimization_type" in rec

    def test_time_based_cost_tracking(self):
        """Test time-based cost tracking and analysis."""
        # Add operations with different timestamps
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Today's operations
        self.aggregator.add_operation_cost(model="gpt-4", cost=0.05, timestamp=now)

        # Yesterday's operations (should not count toward daily budget)
        self.aggregator.add_operation_cost(
            model="gpt-4", cost=0.10, timestamp=yesterday
        )

        # Daily cost should only include today's operations
        budget_status = self.aggregator.check_budget_status()
        assert budget_status["current_daily_cost"] == 0.05

    def test_team_and_project_attribution(self):
        """Test team and project cost attribution."""
        # Create aggregators for different teams/projects
        team1_aggregator = SkyRouterCostAggregator(team="team1", project="proj1")
        team2_aggregator = SkyRouterCostAggregator(team="team2", project="proj2")

        # Add operations to each
        team1_aggregator.add_operation_cost(model="gpt-4", cost=0.05)
        team2_aggregator.add_operation_cost(model="claude-3-sonnet", cost=0.02)

        # Costs should be attributed separately
        team1_summary = team1_aggregator.get_summary()
        team2_summary = team2_aggregator.get_summary()

        assert team1_summary.total_cost == 0.05
        assert team2_summary.total_cost == 0.02
        assert team1_summary.team_attribution == "team1"
        assert team2_summary.team_attribution == "team2"

    def test_customer_specific_cost_tracking(self):
        """Test customer-specific cost tracking."""
        customer_aggregator = SkyRouterCostAggregator(
            team="customer-team", project="customer-project", customer_id="customer-123"
        )

        customer_aggregator.add_operation_cost(
            model="gpt-4", cost=0.05, customer_id="customer-123"
        )

        summary = customer_aggregator.get_summary()
        assert summary.customer_attribution == "customer-123"
        assert "customer-123" in summary.cost_by_customer

    def test_usage_metrics_calculation(self):
        """Test usage metrics calculation."""
        operations = [
            {
                "model": "gpt-4",
                "cost": 0.05,
                "input_tokens": 1000,
                "output_tokens": 500,
            },
            {
                "model": "claude-3-sonnet",
                "cost": 0.02,
                "input_tokens": 800,
                "output_tokens": 300,
            },
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.001,
                "input_tokens": 500,
                "output_tokens": 200,
            },
        ]

        for op in operations:
            self.aggregator.add_operation_cost(**op)

        metrics = self.aggregator.get_usage_metrics()

        assert isinstance(metrics, UsageMetrics)
        assert metrics.total_input_tokens == 2300  # 1000 + 800 + 500
        assert metrics.total_output_tokens == 1000  # 500 + 300 + 200
        assert metrics.average_cost_per_operation > 0
        assert metrics.average_tokens_per_operation > 0

    def test_cost_trend_analysis(self):
        """Test cost trend analysis over time."""
        # Add operations over multiple days
        base_time = datetime.now() - timedelta(days=7)

        for day in range(7):
            operation_time = base_time + timedelta(days=day)
            daily_cost = 0.01 * (day + 1)  # Increasing cost trend

            self.aggregator.add_operation_cost(
                model="gpt-4", cost=daily_cost, timestamp=operation_time
            )

        trend_analysis = self.aggregator.analyze_cost_trends()

        assert isinstance(trend_analysis, dict)
        assert "daily_costs" in trend_analysis
        assert "trend_direction" in trend_analysis
        assert trend_analysis["trend_direction"] == "increasing"

    def test_cost_alerts_configuration(self):
        """Test cost alerts configuration and triggering."""
        alert_aggregator = SkyRouterCostAggregator(
            team="alert-team",
            project="alert-project",
            daily_budget_limit=0.10,
            enable_cost_alerts=True,
            cost_alert_thresholds=[0.50, 0.75, 0.90],  # 50%, 75%, 90% thresholds
        )

        # Add cost that triggers 50% threshold
        alert_aggregator.add_operation_cost(model="gpt-4", cost=0.05)

        alerts = alert_aggregator.check_cost_alerts()

        assert isinstance(alerts, list)
        # Should trigger 50% threshold alert
        threshold_alerts = [
            alert for alert in alerts if "50%" in alert.get("message", "")
        ]
        assert len(threshold_alerts) > 0

    def test_cost_aggregation_performance(self):
        """Test cost aggregation performance with large datasets."""
        # Add many operations
        num_operations = 1000

        for _i in range(num_operations):
            self.aggregator.add_operation_cost(
                model="gpt-3.5-turbo", cost=0.001, input_tokens=100, output_tokens=50
            )

        # Getting summary should be fast
        summary = self.aggregator.get_summary()

        assert summary.total_operations == num_operations
        assert abs(summary.total_cost - 1.0) < 0.01  # 1000 * 0.001

    def test_cost_export_and_reporting(self):
        """Test cost data export and reporting capabilities."""
        operations = [
            {"model": "gpt-4", "cost": 0.05, "routing_strategy": "balanced"},
            {
                "model": "claude-3-sonnet",
                "cost": 0.02,
                "routing_strategy": "cost_optimized",
            },
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.001,
                "routing_strategy": "latency_optimized",
            },
        ]

        for op in operations:
            self.aggregator.add_operation_cost(**op)

        # Test different export formats
        csv_export = self.aggregator.export_costs(format="csv")
        json_export = self.aggregator.export_costs(format="json")

        assert isinstance(csv_export, str)
        assert isinstance(json_export, str)

        # JSON should be parseable
        import json

        parsed_json = json.loads(json_export)
        assert "total_cost" in parsed_json
        assert "operations" in parsed_json

    def test_budget_forecasting(self):
        """Test budget forecasting capabilities."""
        # Add historical operations
        for day in range(30):  # 30 days of history
            operation_time = datetime.now() - timedelta(days=day)
            self.aggregator.add_operation_cost(
                model="gpt-4",
                cost=0.02,  # Consistent daily cost
                timestamp=operation_time,
            )

        forecast = self.aggregator.forecast_monthly_cost()

        assert isinstance(forecast, dict)
        assert "projected_monthly_cost" in forecast
        assert "confidence_interval" in forecast
        assert forecast["projected_monthly_cost"] > 0

    def test_multi_dimensional_cost_analysis(self):
        """Test multi-dimensional cost analysis."""
        # Add operations with multiple dimensions
        operations = [
            {
                "model": "gpt-4",
                "cost": 0.05,
                "routing_strategy": "balanced",
                "environment": "production",
                "feature": "chat",
                "customer_id": "cust1",
            },
            {
                "model": "claude-3-sonnet",
                "cost": 0.02,
                "routing_strategy": "cost_optimized",
                "environment": "staging",
                "feature": "search",
                "customer_id": "cust2",
            },
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.001,
                "routing_strategy": "latency_optimized",
                "environment": "development",
                "feature": "chat",
                "customer_id": "cust1",
            },
        ]

        for op in operations:
            self.aggregator.add_operation_cost(**op)

        analysis = self.aggregator.get_multi_dimensional_analysis()

        assert isinstance(analysis, dict)
        assert "cost_by_environment" in analysis
        assert "cost_by_feature" in analysis
        assert "cost_by_customer" in analysis

        # Verify specific breakdowns
        assert "production" in analysis["cost_by_environment"]
        assert "chat" in analysis["cost_by_feature"]
        assert "cust1" in analysis["cost_by_customer"]


@pytest.mark.skipif(
    not SKYROUTER_COST_AGGREGATOR_AVAILABLE,
    reason="SkyRouter cost aggregator not available",
)
class TestCostSummary:
    """Test suite for CostSummary class."""

    def test_cost_summary_creation(self):
        """Test CostSummary creation and properties."""
        summary = CostSummary(
            total_cost=10.50,
            total_operations=100,
            cost_by_model={"gpt-4": 8.0, "gpt-3.5-turbo": 2.5},
            cost_by_route={"balanced": 6.0, "cost_optimized": 4.5},
            team_attribution="test-team",
            project_attribution="test-project",
        )

        assert summary.total_cost == 10.50
        assert summary.total_operations == 100
        assert summary.average_cost_per_operation == 0.105
        assert "gpt-4" in summary.cost_by_model
        assert summary.team_attribution == "test-team"

    def test_cost_summary_calculations(self):
        """Test CostSummary calculated properties."""
        summary = CostSummary(
            total_cost=5.25,
            total_operations=75,
            cost_by_model={"model1": 3.0, "model2": 2.25},
        )

        assert abs(summary.average_cost_per_operation - 0.07) < 0.001

        # Most expensive model
        most_expensive = summary.get_most_expensive_model()
        assert most_expensive == "model1"

        # Cost distribution
        distribution = summary.get_cost_distribution()
        assert distribution["model1"] == 3.0 / 5.25  # ~57.14%

    def test_cost_summary_comparison(self):
        """Test CostSummary comparison capabilities."""
        summary1 = CostSummary(total_cost=10.0, total_operations=100)
        summary2 = CostSummary(total_cost=15.0, total_operations=120)

        comparison = summary1.compare_with(summary2)

        assert isinstance(comparison, dict)
        assert "cost_difference" in comparison
        assert "operations_difference" in comparison
        assert comparison["cost_difference"] == -5.0  # summary1 is $5 cheaper


@pytest.mark.skipif(
    not SKYROUTER_COST_AGGREGATOR_AVAILABLE,
    reason="SkyRouter cost aggregator not available",
)
class TestOptimizationRecommendation:
    """Test suite for OptimizationRecommendation class."""

    def test_optimization_recommendation_creation(self):
        """Test OptimizationRecommendation creation."""
        recommendation = OptimizationRecommendation(
            title="Switch to cost-optimized routing",
            description="Use cost-optimized routing for non-critical operations",
            potential_savings=2.50,
            effort_level="low",
            priority_score=85,
            optimization_type="route_optimization",
        )

        assert recommendation.title == "Switch to cost-optimized routing"
        assert recommendation.potential_savings == 2.50
        assert recommendation.priority_score == 85
        assert recommendation.optimization_type == "route_optimization"

    def test_recommendation_prioritization(self):
        """Test recommendation prioritization logic."""
        recommendations = [
            OptimizationRecommendation(
                title="High impact, low effort",
                potential_savings=10.0,
                effort_level="low",
                priority_score=95,
            ),
            OptimizationRecommendation(
                title="Medium impact, medium effort",
                potential_savings=5.0,
                effort_level="medium",
                priority_score=70,
            ),
            OptimizationRecommendation(
                title="Low impact, high effort",
                potential_savings=1.0,
                effort_level="high",
                priority_score=30,
            ),
        ]

        # Sort by priority score
        sorted_recs = sorted(
            recommendations, key=lambda r: r.priority_score, reverse=True
        )

        assert sorted_recs[0].title == "High impact, low effort"
        assert sorted_recs[-1].title == "Low impact, high effort"


@pytest.mark.skipif(
    not SKYROUTER_COST_AGGREGATOR_AVAILABLE,
    reason="SkyRouter cost aggregator not available",
)
class TestCostAggregatorIntegration:
    """Integration tests for cost aggregator functionality."""

    def test_end_to_end_cost_tracking_workflow(self):
        """Test complete cost tracking workflow."""
        aggregator = SkyRouterCostAggregator(
            team="integration-team", project="workflow-test", daily_budget_limit=50.0
        )

        # Simulate a day of operations
        morning_operations = [
            {
                "model": "gpt-3.5-turbo",
                "cost": 0.01,
                "routing_strategy": "cost_optimized",
            },
            {"model": "claude-3-sonnet", "cost": 0.02, "routing_strategy": "balanced"},
        ] * 10  # 10 operations each

        afternoon_operations = [
            {"model": "gpt-4", "cost": 0.05, "routing_strategy": "reliability_first"},
            {"model": "claude-3-opus", "cost": 0.06, "routing_strategy": "balanced"},
        ] * 5  # 5 operations each

        all_operations = morning_operations + afternoon_operations

        # Add all operations
        for op in all_operations:
            aggregator.add_operation_cost(**op)

        # Get comprehensive analysis
        summary = aggregator.get_summary()
        budget_status = aggregator.check_budget_status()
        recommendations = aggregator.get_cost_optimization_recommendations()
        metrics = aggregator.get_usage_metrics()

        # Verify results
        assert summary.total_operations == 30  # 20 + 10
        assert budget_status["current_daily_cost"] > 0
        assert isinstance(recommendations, list)
        assert metrics.total_operations == 30

    def test_real_time_cost_monitoring(self):
        """Test real-time cost monitoring capabilities."""
        aggregator = SkyRouterCostAggregator(
            team="monitoring-team",
            project="realtime-test",
            daily_budget_limit=10.0,
            enable_cost_alerts=True,
        )

        # Simulate operations that gradually increase cost
        costs = [1.0, 2.0, 3.0, 5.0]  # Cumulative: 1, 3, 6, 11 (exceeds budget)

        alert_triggered = False

        for cost in costs:
            aggregator.add_operation_cost(model="gpt-4", cost=cost)

            budget_status = aggregator.check_budget_status()
            if budget_status.get("budget_exceeded"):
                alert_triggered = True
                break

        assert alert_triggered is True

    def test_multi_team_cost_isolation(self):
        """Test cost isolation between multiple teams."""
        team_configs = [
            {"team": "team-a", "project": "project-1", "budget": 20.0},
            {"team": "team-b", "project": "project-2", "budget": 30.0},
            {"team": "team-c", "project": "project-3", "budget": 40.0},
        ]

        aggregators = []
        for config in team_configs:
            aggregator = SkyRouterCostAggregator(
                team=config["team"],
                project=config["project"],
                daily_budget_limit=config["budget"],
            )
            aggregators.append(aggregator)

        # Add different operations to each team
        team_operations = [
            [{"model": "gpt-3.5-turbo", "cost": 0.01}]
            * 100,  # Team A: many cheap operations
            [{"model": "gpt-4", "cost": 0.05}]
            * 20,  # Team B: fewer expensive operations
            [{"model": "claude-3-sonnet", "cost": 0.02}]
            * 50,  # Team C: medium operations
        ]

        for aggregator, operations in zip(aggregators, team_operations):
            for op in operations:
                aggregator.add_operation_cost(**op)

        # Verify isolation
        summaries = [agg.get_summary() for agg in aggregators]

        # Each team should have different costs
        costs = [summary.total_cost for summary in summaries]
        assert len(set(costs)) == 3  # All different costs

        # Team attributions should be correct
        for i, summary in enumerate(summaries):
            assert summary.team_attribution == team_configs[i]["team"]


if __name__ == "__main__":
    pytest.main([__file__])
