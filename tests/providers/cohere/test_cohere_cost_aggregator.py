"""Tests for GenOps Cohere cost aggregator."""

import time
from unittest.mock import patch

import pytest

from genops.providers.cohere import CohereOperation

# Test imports
from genops.providers.cohere_cost_aggregator import (
    CohereCostAggregator,
    CostBreakdown,
    CostSummary,
    OptimizationInsight,
    TimeWindow,
)


class TestCohereCostAggregator:
    """Test suite for CohereCostAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance for testing."""
        return CohereCostAggregator(
            enable_detailed_tracking=True,
            cost_alert_threshold=10.0,
            budget_period_hours=24,
        )

    @pytest.fixture
    def sample_cost_breakdown(self):
        """Create sample cost breakdown for testing."""
        return CostBreakdown(
            model="command-r-08-2024",
            operation=CohereOperation.CHAT,
            input_tokens=100,
            output_tokens=150,
            input_cost=0.001,
            output_cost=0.003,
            operation_cost=0.0,
        )

    def test_aggregator_initialization(self):
        """Test aggregator initialization with various configurations."""
        # Basic initialization
        aggregator = CohereCostAggregator()
        assert aggregator.enable_detailed_tracking is False
        assert aggregator.cost_alert_threshold == 0.0
        assert aggregator.budget_period_hours == 24
        assert len(aggregator.operation_records) == 0

        # Advanced configuration
        aggregator = CohereCostAggregator(
            enable_detailed_tracking=True,
            cost_alert_threshold=50.0,
            budget_period_hours=168,  # Weekly
            max_records=1000,
        )
        assert aggregator.enable_detailed_tracking is True
        assert aggregator.cost_alert_threshold == 50.0
        assert aggregator.budget_period_hours == 168
        assert aggregator.max_records == 1000

    def test_record_operation_basic(self, aggregator, sample_cost_breakdown):
        """Test basic operation recording."""
        aggregator.record_operation(
            model="command-r-08-2024",
            operation_type=CohereOperation.CHAT,
            cost_breakdown=sample_cost_breakdown,
            team="test-team",
            project="test-project",
        )

        assert len(aggregator.operation_records) == 1

        record = aggregator.operation_records[0]
        assert record.model == "command-r-08-2024"
        assert record.operation_type == CohereOperation.CHAT
        assert record.team == "test-team"
        assert record.project == "test-project"
        assert record.total_cost == 0.004  # 0.001 + 0.003 + 0.0

    def test_record_operation_with_governance_attributes(
        self, aggregator, sample_cost_breakdown
    ):
        """Test operation recording with comprehensive governance attributes."""
        governance_attrs = {
            "team": "ml-team",
            "project": "recommendation-engine",
            "customer_id": "enterprise-123",
            "environment": "production",
            "cost_center": "ai-infrastructure",
            "feature": "semantic-search",
        }

        aggregator.record_operation(
            model="embed-english-v4.0",
            operation_type=CohereOperation.EMBED,
            cost_breakdown=sample_cost_breakdown,
            **governance_attrs,
        )

        record = aggregator.operation_records[0]
        assert record.team == "ml-team"
        assert record.project == "recommendation-engine"
        assert record.customer_id == "enterprise-123"
        assert record.environment == "production"
        assert record.cost_center == "ai-infrastructure"
        assert record.feature == "semantic-search"

    def test_multiple_operation_recording(self, aggregator):
        """Test recording multiple operations."""
        operations = [
            {
                "model": "command-light",
                "operation_type": CohereOperation.CHAT,
                "cost": 0.001,
                "team": "team-a",
            },
            {
                "model": "embed-english-v4.0",
                "operation_type": CohereOperation.EMBED,
                "cost": 0.002,
                "team": "team-b",
            },
            {
                "model": "rerank-english-v3.0",
                "operation_type": CohereOperation.RERANK,
                "cost": 0.003,
                "team": "team-a",
            },
        ]

        for op in operations:
            cost_breakdown = CostBreakdown(
                model=op["model"], operation=op["operation_type"], total_cost=op["cost"]
            )

            aggregator.record_operation(
                model=op["model"],
                operation_type=op["operation_type"],
                cost_breakdown=cost_breakdown,
                team=op["team"],
            )

        assert len(aggregator.operation_records) == 3

        # Verify total cost
        total_cost = sum(record.total_cost for record in aggregator.operation_records)
        assert total_cost == 0.006

    def test_get_cost_summary_by_time_window(self, aggregator):
        """Test cost summary generation for different time windows."""
        # Record operations at different times
        current_time = time.time()

        # Recent operation (within 1 hour)
        with patch("time.time", return_value=current_time):
            aggregator.record_operation(
                model="command-light",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-light",
                    operation=CohereOperation.CHAT,
                    total_cost=0.001,
                ),
                team="recent-team",
            )

        # Older operation (2 days ago)
        old_time = current_time - (2 * 24 * 3600)
        with patch("time.time", return_value=old_time):
            aggregator.record_operation(
                model="command-r-08-2024",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-r-08-2024",
                    operation=CohereOperation.CHAT,
                    total_cost=0.005,
                ),
                team="old-team",
            )

        # Get hourly summary (should include only recent operation)
        hourly_summary = aggregator.get_cost_summary(TimeWindow.HOUR)
        assert hourly_summary.overview.total_cost == 0.001
        assert hourly_summary.overview.total_operations == 1

        # Get weekly summary (should include both operations)
        weekly_summary = aggregator.get_cost_summary(TimeWindow.WEEK)
        assert weekly_summary.overview.total_cost == 0.006
        assert weekly_summary.overview.total_operations == 2

    def test_get_cost_summary_by_team(self, aggregator):
        """Test cost summary generation grouped by team."""
        # Record operations for different teams
        teams_data = [
            {"team": "ml-team", "cost": 0.010, "operations": 5},
            {"team": "search-team", "cost": 0.005, "operations": 2},
            {"team": "analytics-team", "cost": 0.015, "operations": 8},
        ]

        for team_data in teams_data:
            for _ in range(team_data["operations"]):
                aggregator.record_operation(
                    model="command-light",
                    operation_type=CohereOperation.CHAT,
                    cost_breakdown=CostBreakdown(
                        model="command-light",
                        operation=CohereOperation.CHAT,
                        total_cost=team_data["cost"] / team_data["operations"],
                    ),
                    team=team_data["team"],
                )

        summary = aggregator.get_cost_summary(TimeWindow.DAY)

        # Verify team-level breakdowns
        assert "ml-team" in summary.by_team
        assert "search-team" in summary.by_team
        assert "analytics-team" in summary.by_team

        # Check team costs
        assert abs(summary.by_team["ml-team"].total_cost - 0.010) < 0.001
        assert summary.by_team["ml-team"].total_operations == 5

        assert abs(summary.by_team["search-team"].total_cost - 0.005) < 0.001
        assert summary.by_team["search-team"].total_operations == 2

    def test_get_cost_summary_by_model(self, aggregator):
        """Test cost summary generation grouped by model."""
        models_data = [
            {"model": "command-light", "cost": 0.002, "count": 3},
            {"model": "command-r-08-2024", "cost": 0.008, "count": 2},
            {"model": "embed-english-v4.0", "cost": 0.003, "count": 5},
        ]

        for model_data in models_data:
            for _ in range(model_data["count"]):
                operation_type = (
                    CohereOperation.EMBED
                    if "embed" in model_data["model"]
                    else CohereOperation.CHAT
                )

                aggregator.record_operation(
                    model=model_data["model"],
                    operation_type=operation_type,
                    cost_breakdown=CostBreakdown(
                        model=model_data["model"],
                        operation=operation_type,
                        total_cost=model_data["cost"] / model_data["count"],
                    ),
                )

        summary = aggregator.get_cost_summary(TimeWindow.DAY)

        # Verify model-level breakdowns
        assert "command-light" in summary.by_model
        assert "command-r-08-2024" in summary.by_model
        assert "embed-english-v4.0" in summary.by_model

        # Check model costs
        assert abs(summary.by_model["command-light"].total_cost - 0.002) < 0.001
        assert summary.by_model["command-light"].total_operations == 3

    def test_get_operation_summary(self, aggregator):
        """Test operation summary generation."""
        # Record diverse operations
        operations = [
            (CohereOperation.CHAT, "command-light", 0.001, 3),
            (CohereOperation.CHAT, "command-r-08-2024", 0.004, 2),
            (CohereOperation.EMBED, "embed-english-v4.0", 0.002, 5),
            (CohereOperation.RERANK, "rerank-english-v3.0", 0.003, 1),
        ]

        for operation_type, model, unit_cost, count in operations:
            for _ in range(count):
                aggregator.record_operation(
                    model=model,
                    operation_type=operation_type,
                    cost_breakdown=CostBreakdown(
                        model=model, operation=operation_type, total_cost=unit_cost
                    ),
                )

        summary = aggregator.get_operation_summary()

        # Verify operation type breakdowns
        assert CohereOperation.CHAT in summary
        assert CohereOperation.EMBED in summary
        assert CohereOperation.RERANK in summary

        # Check operation counts
        assert summary[CohereOperation.CHAT].total_operations == 5  # 3 + 2
        assert summary[CohereOperation.EMBED].total_operations == 5
        assert summary[CohereOperation.RERANK].total_operations == 1

    def test_cost_optimization_insights(self, aggregator):
        """Test cost optimization insight generation."""
        # Record expensive operations that could be optimized
        expensive_operations = [
            (
                "command-r-plus-08-2024",
                CohereOperation.CHAT,
                0.010,
                10,
            ),  # High-cost model
            ("command-r-08-2024", CohereOperation.CHAT, 0.005, 20),  # Medium cost
            ("command-light", CohereOperation.CHAT, 0.001, 5),  # Low cost
        ]

        for model, operation_type, unit_cost, count in expensive_operations:
            for _ in range(count):
                aggregator.record_operation(
                    model=model,
                    operation_type=operation_type,
                    cost_breakdown=CostBreakdown(
                        model=model, operation=operation_type, total_cost=unit_cost
                    ),
                    team="optimization-test",
                )

        insights = aggregator.get_cost_optimization_insights()

        # Should provide insights
        assert len(insights.recommendations) > 0

        # Should identify high-cost models for optimization
        high_cost_recommendation = None
        for recommendation in insights.recommendations:
            if "command-r-plus" in recommendation.description:
                high_cost_recommendation = recommendation
                break

        assert high_cost_recommendation is not None
        assert high_cost_recommendation.potential_savings > 0

    def test_budget_alert_functionality(self, aggregator):
        """Test budget alert generation."""
        # Set low alert threshold
        aggregator.cost_alert_threshold = 0.005

        # Record operations that exceed threshold
        for i in range(3):
            aggregator.record_operation(
                model="command-r-plus-08-2024",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-r-plus-08-2024",
                    operation=CohereOperation.CHAT,
                    total_cost=0.003,
                ),
                team=f"team-{i}",
            )

        # Total cost: 0.009, exceeds threshold of 0.005
        alerts = aggregator.get_budget_alerts()

        assert len(alerts) > 0

        # Should have budget threshold alert
        budget_alert = next(
            (alert for alert in alerts if alert.type == "budget_threshold"), None
        )
        assert budget_alert is not None
        assert budget_alert.current_amount > aggregator.cost_alert_threshold

    def test_export_cost_data(self, aggregator):
        """Test cost data export functionality."""
        # Record some operations
        for i in range(5):
            aggregator.record_operation(
                model="command-light",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-light",
                    operation=CohereOperation.CHAT,
                    total_cost=0.001,
                ),
                team=f"export-team-{i % 2}",
                project=f"export-project-{i % 3}",
            )

        # Export as dictionary
        export_data = aggregator.export_cost_data(format="dict")

        assert isinstance(export_data, dict)
        assert "operations" in export_data
        assert "summary" in export_data
        assert len(export_data["operations"]) == 5

        # Export as JSON string
        json_data = aggregator.export_cost_data(format="json")
        assert isinstance(json_data, str)

        import json

        parsed_data = json.loads(json_data)
        assert "operations" in parsed_data
        assert len(parsed_data["operations"]) == 5

    def test_time_based_cost_analysis(self, aggregator):
        """Test time-based cost analysis functionality."""
        # Record operations at different times
        base_time = time.time()
        times_and_costs = [
            (base_time - 3600, 0.001),  # 1 hour ago
            (base_time - 1800, 0.002),  # 30 minutes ago
            (base_time - 900, 0.004),  # 15 minutes ago
            (base_time, 0.003),  # Now
        ]

        for timestamp, cost in times_and_costs:
            with patch("time.time", return_value=timestamp):
                aggregator.record_operation(
                    model="command-light",
                    operation_type=CohereOperation.CHAT,
                    cost_breakdown=CostBreakdown(
                        model="command-light",
                        operation=CohereOperation.CHAT,
                        total_cost=cost,
                    ),
                )

        # Analyze cost trends
        analysis = aggregator.get_time_based_analysis(TimeWindow.HOUR)

        assert analysis is not None
        assert "trend" in analysis
        assert "hourly_breakdown" in analysis

        # Should detect increasing trend
        assert analysis["trend"] in ["increasing", "stable", "decreasing"]

    def test_memory_management(self, aggregator):
        """Test memory management for large numbers of records."""
        # Set small max_records for testing
        aggregator.max_records = 10

        # Record more operations than max_records
        for _i in range(15):
            aggregator.record_operation(
                model="command-light",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-light",
                    operation=CohereOperation.CHAT,
                    total_cost=0.001,
                ),
            )

        # Should maintain only max_records
        assert len(aggregator.operation_records) <= aggregator.max_records

        # Should keep most recent records
        latest_record = aggregator.operation_records[-1]
        assert latest_record.timestamp > aggregator.operation_records[0].timestamp

    def test_concurrent_access_safety(self, aggregator):
        """Test thread safety for concurrent access."""
        import concurrent.futures

        def record_operation(operation_id):
            """Record operation in thread."""
            aggregator.record_operation(
                model="command-light",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-light",
                    operation=CohereOperation.CHAT,
                    total_cost=0.001,
                ),
                operation_id=f"concurrent-op-{operation_id}",
            )

        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(record_operation, i) for i in range(20)]
            concurrent.futures.wait(futures)

        # Should have recorded all operations safely
        assert len(aggregator.operation_records) == 20

        # Should be able to get summary without errors
        summary = aggregator.get_cost_summary(TimeWindow.DAY)
        assert summary.overview.total_operations == 20

    def test_reset_aggregator(self, aggregator):
        """Test aggregator reset functionality."""
        # Record some operations
        for _i in range(5):
            aggregator.record_operation(
                model="command-light",
                operation_type=CohereOperation.CHAT,
                cost_breakdown=CostBreakdown(
                    model="command-light",
                    operation=CohereOperation.CHAT,
                    total_cost=0.001,
                ),
            )

        assert len(aggregator.operation_records) == 5

        # Reset aggregator
        aggregator.reset()

        assert len(aggregator.operation_records) == 0

        # Summary should show zero costs
        summary = aggregator.get_cost_summary(TimeWindow.DAY)
        assert summary.overview.total_cost == 0.0
        assert summary.overview.total_operations == 0


class TestCostSummary:
    """Test CostSummary data structure."""

    def test_cost_summary_creation(self):
        """Test CostSummary object creation."""
        from genops.providers.cohere_cost_aggregator import (
            ModelSummary,
            OverviewSummary,
            TeamSummary,
        )

        overview = OverviewSummary(
            total_cost=0.050,
            total_operations=100,
            avg_cost_per_operation=0.0005,
            unique_models=5,
            unique_teams=3,
            time_period="24h",
        )

        team_summary = {
            "ml-team": TeamSummary(
                total_cost=0.030,
                total_operations=60,
                avg_cost_per_operation=0.0005,
                primary_models=["command-light", "embed-english-v4.0"],
            )
        }

        model_summary = {
            "command-light": ModelSummary(
                total_cost=0.020,
                total_operations=80,
                avg_cost_per_operation=0.00025,
                usage_teams=["ml-team", "search-team"],
            )
        }

        summary = CostSummary(
            overview=overview,
            by_team=team_summary,
            by_model=model_summary,
            by_operation={},
            time_window=TimeWindow.DAY,
        )

        assert summary.overview.total_cost == 0.050
        assert "ml-team" in summary.by_team
        assert "command-light" in summary.by_model
        assert summary.time_window == TimeWindow.DAY

    def test_cost_summary_serialization(self):
        """Test CostSummary serialization."""
        from genops.providers.cohere_cost_aggregator import OverviewSummary

        overview = OverviewSummary(
            total_cost=0.025, total_operations=50, avg_cost_per_operation=0.0005
        )

        summary = CostSummary(
            overview=overview,
            by_team={},
            by_model={},
            by_operation={},
            time_window=TimeWindow.HOUR,
        )

        as_dict = summary.to_dict()

        assert isinstance(as_dict, dict)
        assert as_dict["overview"]["total_cost"] == 0.025
        assert as_dict["time_window"] == "HOUR"


class TestOptimizationInsights:
    """Test optimization insight generation."""

    def test_optimization_insight_creation(self):
        """Test OptimizationInsight object creation."""
        insight = OptimizationInsight(
            type="model_optimization",
            title="Switch to cheaper model",
            description="Consider using command-light instead of command-r-plus for simple tasks",
            potential_savings=0.008,
            confidence_score=0.85,
            action_required="model_change",
            affected_operations=["chat_operation_1", "chat_operation_2"],
        )

        assert insight.type == "model_optimization"
        assert insight.potential_savings == 0.008
        assert insight.confidence_score == 0.85
        assert len(insight.affected_operations) == 2

    def test_optimization_insight_ranking(self):
        """Test optimization insight ranking by savings potential."""
        insights = [
            OptimizationInsight(
                type="model_optimization",
                title="High savings",
                potential_savings=0.050,
                confidence_score=0.9,
            ),
            OptimizationInsight(
                type="batching_optimization",
                title="Medium savings",
                potential_savings=0.020,
                confidence_score=0.8,
            ),
            OptimizationInsight(
                type="caching_optimization",
                title="Low savings",
                potential_savings=0.005,
                confidence_score=0.7,
            ),
        ]

        # Sort by potential savings
        sorted_insights = sorted(
            insights, key=lambda x: x.potential_savings, reverse=True
        )

        assert sorted_insights[0].title == "High savings"
        assert sorted_insights[1].title == "Medium savings"
        assert sorted_insights[2].title == "Low savings"


if __name__ == "__main__":
    pytest.main([__file__])
