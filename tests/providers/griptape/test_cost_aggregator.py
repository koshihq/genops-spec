#!/usr/bin/env python3
"""
Test suite for Griptape Cost Aggregator

Tests multi-provider cost calculation, aggregation, and reporting functionality
for Griptape framework operations.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from genops.providers.griptape.cost_aggregator import (
    GriptapeCostAggregator,
    GriptapeCostBreakdown,
    GriptapeCostSummary,
)


class TestGriptapeCostBreakdown:
    """Test GriptapeCostBreakdown data class."""

    def test_cost_breakdown_initialization(self):
        """Test cost breakdown creation."""
        timestamp = datetime.now()
        breakdown = GriptapeCostBreakdown(
            structure_id="agent-123",
            structure_type="agent",
            provider="openai",
            model="gpt-4",
            input_tokens=150,
            output_tokens=300,
            total_tokens=450,
            input_cost=Decimal("0.003"),
            output_cost=Decimal("0.009"),
            total_cost=Decimal("0.012"),
            timestamp=timestamp,
            team="test-team",
            project="test-project",
        )

        assert breakdown.structure_id == "agent-123"
        assert breakdown.structure_type == "agent"
        assert breakdown.provider == "openai"
        assert breakdown.model == "gpt-4"
        assert breakdown.input_tokens == 150
        assert breakdown.output_tokens == 300
        assert breakdown.total_tokens == 450
        assert breakdown.total_cost == Decimal("0.012")
        assert breakdown.team == "test-team"
        assert breakdown.project == "test-project"

    def test_cost_breakdown_to_dict(self):
        """Test cost breakdown serialization."""
        timestamp = datetime.now()
        breakdown = GriptapeCostBreakdown(
            structure_id="pipeline-456",
            structure_type="pipeline",
            provider="anthropic",
            model="claude-3",
            input_tokens=200,
            output_tokens=400,
            total_tokens=600,
            input_cost=Decimal("0.004"),
            output_cost=Decimal("0.008"),
            total_cost=Decimal("0.012"),
            timestamp=timestamp,
        )

        data = breakdown.to_dict()

        assert data["structure_id"] == "pipeline-456"
        assert data["structure_type"] == "pipeline"
        assert data["provider"] == "anthropic"
        assert data["model"] == "claude-3"
        assert data["total_cost"] == 0.012
        assert data["timestamp"] == timestamp.isoformat()


class TestGriptapeCostSummary:
    """Test GriptapeCostSummary data class."""

    def test_cost_summary_initialization(self):
        """Test cost summary creation."""
        summary = GriptapeCostSummary()

        assert summary.total_cost == Decimal("0")
        assert len(summary.cost_by_provider) == 0
        assert len(summary.cost_by_model) == 0
        assert summary.total_requests == 0
        assert summary.total_tokens == 0
        assert len(summary.unique_providers) == 0

    def test_get_top_providers(self):
        """Test top providers ranking."""
        summary = GriptapeCostSummary()
        summary.cost_by_provider = {
            "openai": Decimal("10.50"),
            "anthropic": Decimal("7.25"),
            "google": Decimal("3.80"),
            "cohere": Decimal("1.20"),
        }

        top_providers = summary.get_top_providers(limit=3)

        assert len(top_providers) == 3
        assert top_providers[0] == ("openai", Decimal("10.50"))
        assert top_providers[1] == ("anthropic", Decimal("7.25"))
        assert top_providers[2] == ("google", Decimal("3.80"))

    def test_get_cost_efficiency(self):
        """Test cost efficiency metrics calculation."""
        summary = GriptapeCostSummary()
        summary.total_cost = Decimal("5.50")
        summary.total_tokens = 11000
        summary.total_requests = 25

        efficiency = summary.get_cost_efficiency()

        assert efficiency["cost_per_token"] == 0.0005  # 5.50 / 11000
        assert efficiency["cost_per_request"] == 0.22  # 5.50 / 25
        assert efficiency["tokens_per_request"] == 440.0  # 11000 / 25

    def test_get_cost_efficiency_zero_division(self):
        """Test cost efficiency with zero values."""
        summary = GriptapeCostSummary()

        efficiency = summary.get_cost_efficiency()

        assert efficiency["cost_per_token"] == 0.0
        assert efficiency["cost_per_request"] == 0.0


class TestGriptapeCostAggregator:
    """Test GriptapeCostAggregator functionality."""

    @pytest.fixture
    def aggregator(self):
        """Create test cost aggregator."""
        return GriptapeCostAggregator()

    def test_aggregator_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert len(aggregator.cost_breakdowns) == 0
        assert "openai" in aggregator.calculators
        assert "anthropic" in aggregator.calculators
        assert "cohere" in aggregator.fallback_pricing
        assert "mistral" in aggregator.fallback_pricing

    @patch("genops.providers.griptape.cost_aggregator.OpenAICostCalculator")
    def test_calculate_cost_with_provider_calculator(
        self, mock_openai_calc, aggregator
    ):
        """Test cost calculation using provider-specific calculator."""
        # Mock OpenAI calculator
        mock_calc_instance = Mock()
        mock_calc_instance.calculate_cost.return_value = {
            "input_cost": Decimal("0.003"),
            "output_cost": Decimal("0.009"),
            "total_cost": Decimal("0.012"),
        }
        mock_openai_calc.return_value = mock_calc_instance

        # Recalculate to use mocked calculator
        aggregator.calculators["openai"] = mock_calc_instance

        result = aggregator.calculate_cost("openai", "gpt-4", 150, 300)

        assert result["input_cost"] == Decimal("0.003")
        assert result["output_cost"] == Decimal("0.009")
        assert result["total_cost"] == Decimal("0.012")
        mock_calc_instance.calculate_cost.assert_called_once_with("gpt-4", 150, 300)

    def test_calculate_cost_with_fallback_pricing(self, aggregator):
        """Test cost calculation using fallback pricing."""
        result = aggregator.calculate_cost("cohere", "command", 1000, 500)

        # Fallback pricing: cohere input=0.0015, output=0.002 per 1K tokens
        expected_input = Decimal("0.0015")  # 1000 tokens / 1000 * 0.0015
        expected_output = Decimal("0.001")  # 500 tokens / 1000 * 0.002
        expected_total = expected_input + expected_output

        assert result["input_cost"] == expected_input
        assert result["output_cost"] == expected_output
        assert result["total_cost"] == expected_total

    def test_calculate_cost_generic_fallback(self, aggregator):
        """Test cost calculation with generic fallback."""
        # Use unknown provider
        result = aggregator.calculate_cost(
            "unknown-provider", "unknown-model", 500, 250
        )

        # Generic fallback: $0.002 per 1K tokens
        expected_total = (Decimal("750") / 1000) * Decimal("0.002")

        assert result["total_cost"] == expected_total
        assert result["input_cost"] > Decimal("0")
        assert result["output_cost"] > Decimal("0")

    def test_add_structure_cost(self, aggregator):
        """Test adding structure cost breakdown."""
        breakdown = aggregator.add_structure_cost(
            structure_id="agent-123",
            structure_type="agent",
            provider="openai",
            model="gpt-4",
            input_tokens=150,
            output_tokens=300,
            operation_type="run",
            governance_attrs={
                "team": "test-team",
                "project": "test-project",
                "customer_id": "customer-123",
            },
        )

        assert breakdown.structure_id == "agent-123"
        assert breakdown.structure_type == "agent"
        assert breakdown.provider == "openai"
        assert breakdown.model == "gpt-4"
        assert breakdown.team == "test-team"
        assert breakdown.project == "test-project"
        assert breakdown.customer_id == "customer-123"
        assert breakdown.total_cost > Decimal("0")

        # Check it was added to storage
        assert len(aggregator.cost_breakdowns) == 1
        assert aggregator.cost_breakdowns[0] == breakdown

    def test_get_cost_summary_no_filters(self, aggregator):
        """Test cost summary generation without filters."""
        # Add multiple cost breakdowns
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)
        aggregator.add_structure_cost(
            "pipeline-1", "pipeline", "anthropic", "claude-3", 150, 300
        )
        aggregator.add_structure_cost(
            "workflow-1", "workflow", "google", "gemini-pro", 80, 160
        )

        summary = aggregator.get_cost_summary()

        assert summary.total_requests == 3
        assert summary.total_cost > Decimal("0")
        assert len(summary.unique_providers) == 3
        assert "openai" in summary.unique_providers
        assert "anthropic" in summary.unique_providers
        assert "google" in summary.unique_providers

        # Check structure type breakdown
        assert "agent" in summary.cost_by_structure_type
        assert "pipeline" in summary.cost_by_structure_type
        assert "workflow" in summary.cost_by_structure_type

    def test_get_cost_summary_with_filters(self, aggregator):
        """Test cost summary with filtering."""
        # Add breakdowns with different attributes
        aggregator.add_structure_cost(
            "agent-1",
            "agent",
            "openai",
            "gpt-4",
            100,
            200,
            governance_attrs={"team": "team-a", "project": "project-1"},
        )
        aggregator.add_structure_cost(
            "agent-2",
            "agent",
            "anthropic",
            "claude-3",
            150,
            300,
            governance_attrs={"team": "team-b", "project": "project-1"},
        )
        aggregator.add_structure_cost(
            "pipeline-1",
            "pipeline",
            "openai",
            "gpt-4",
            80,
            160,
            governance_attrs={"team": "team-a", "project": "project-2"},
        )

        # Filter by structure type
        agent_summary = aggregator.get_cost_summary(structure_type="agent")
        assert agent_summary.total_requests == 2
        assert len(agent_summary.cost_by_structure_type) == 1
        assert "agent" in agent_summary.cost_by_structure_type

        # Filter by provider
        openai_summary = aggregator.get_cost_summary(provider="openai")
        assert openai_summary.total_requests == 2
        assert len(openai_summary.unique_providers) == 1
        assert "openai" in openai_summary.unique_providers

        # Filter by team
        team_a_summary = aggregator.get_cost_summary(team="team-a")
        assert team_a_summary.total_requests == 2
        assert "team-a" in team_a_summary.cost_by_team

    def test_get_cost_summary_with_time_filter(self, aggregator):
        """Test cost summary with time-based filtering."""
        # Add breakdown with specific timestamp
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Manually create and add breakdown with yesterday timestamp
        breakdown = GriptapeCostBreakdown(
            structure_id="old-agent",
            structure_type="agent",
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_cost=Decimal("0.002"),
            output_cost=Decimal("0.006"),
            total_cost=Decimal("0.008"),
            timestamp=yesterday,
        )
        aggregator.cost_breakdowns.append(breakdown)

        # Add recent breakdown
        aggregator.add_structure_cost("new-agent", "agent", "openai", "gpt-4", 100, 200)

        # Filter to only recent breakdowns
        recent_summary = aggregator.get_cost_summary(
            start_time=now - timedelta(minutes=5)
        )
        assert recent_summary.total_requests == 1  # Only recent one

        # Filter to include old breakdown
        all_summary = aggregator.get_cost_summary(
            start_time=yesterday - timedelta(hours=1)
        )
        assert all_summary.total_requests == 2  # Both breakdowns

    def test_get_daily_costs(self, aggregator):
        """Test daily cost calculation."""
        # Add some costs for today
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)
        aggregator.add_structure_cost(
            "agent-2", "agent", "anthropic", "claude-3", 150, 300
        )

        daily_costs = aggregator.get_daily_costs()
        assert daily_costs > Decimal("0")

        # Test specific date
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_costs = aggregator.get_daily_costs(yesterday)
        assert yesterday_costs == Decimal("0")  # No costs for yesterday

    def test_get_weekly_costs(self, aggregator):
        """Test weekly cost calculation."""
        # Add some costs
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)

        weekly_costs = aggregator.get_weekly_costs()
        assert weekly_costs > Decimal("0")

    def test_get_monthly_costs(self, aggregator):
        """Test monthly cost calculation."""
        # Add some costs
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)

        monthly_costs = aggregator.get_monthly_costs()
        assert monthly_costs > Decimal("0")

    def test_export_cost_data_json(self, aggregator):
        """Test cost data export in JSON format."""
        # Add breakdown
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)

        json_data = aggregator.export_cost_data(format="json")

        assert isinstance(json_data, str)
        assert "agent-1" in json_data
        assert "openai" in json_data
        assert "gpt-4" in json_data

    def test_export_cost_data_csv(self, aggregator):
        """Test cost data export in CSV format."""
        # Add breakdown
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)

        csv_data = aggregator.export_cost_data(format="csv")

        assert isinstance(csv_data, str)
        assert "structure_id,structure_type,provider,model" in csv_data
        assert "agent-1,agent,openai,gpt-4" in csv_data

    def test_export_cost_data_dict(self, aggregator):
        """Test cost data export as dictionary list."""
        # Add breakdown
        aggregator.add_structure_cost("agent-1", "agent", "openai", "gpt-4", 100, 200)

        dict_data = aggregator.export_cost_data(format="dict")

        assert isinstance(dict_data, list)
        assert len(dict_data) == 1
        assert dict_data[0]["structure_id"] == "agent-1"

    def test_clear_old_data(self, aggregator):
        """Test clearing old cost data."""
        # Add current breakdown
        aggregator.add_structure_cost(
            "recent-agent", "agent", "openai", "gpt-4", 100, 200
        )

        # Add old breakdown manually
        old_timestamp = datetime.now() - timedelta(days=35)
        old_breakdown = GriptapeCostBreakdown(
            structure_id="old-agent",
            structure_type="agent",
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            input_cost=Decimal("0.002"),
            output_cost=Decimal("0.006"),
            total_cost=Decimal("0.008"),
            timestamp=old_timestamp,
        )
        aggregator.cost_breakdowns.append(old_breakdown)

        assert len(aggregator.cost_breakdowns) == 2

        # Clear data older than 30 days
        removed_count = aggregator.clear_old_data(days_to_keep=30)

        assert removed_count == 1
        assert len(aggregator.cost_breakdowns) == 1
        assert aggregator.cost_breakdowns[0].structure_id == "recent-agent"

    def test_thread_safety(self, aggregator):
        """Test thread safety of cost aggregator operations."""
        import threading
        import time

        def add_costs(thread_id):
            for i in range(10):
                aggregator.add_structure_cost(
                    f"agent-{thread_id}-{i}", "agent", "openai", "gpt-4", 100, 200
                )
                time.sleep(0.001)  # Small delay to increase chance of race condition

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_costs, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have 50 breakdowns total (5 threads * 10 each)
        assert len(aggregator.cost_breakdowns) == 50

        # All should be unique
        structure_ids = [b.structure_id for b in aggregator.cost_breakdowns]
        assert len(set(structure_ids)) == 50


if __name__ == "__main__":
    pytest.main([__file__])
