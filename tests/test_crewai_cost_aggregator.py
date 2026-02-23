#!/usr/bin/env python3
"""
Test suite for CrewAI Cost Aggregator

Comprehensive tests for the CrewAICostAggregator class including:
- Multi-provider cost tracking
- Cost optimization recommendations
- Provider comparison and analysis
- Real-time cost calculation
- Budget management and alerts
"""

from datetime import datetime, timedelta

import pytest

# Import the CrewAI cost aggregator and related classes
try:
    from genops.providers.crewai import (
        AgentCostEntry,
        CostAnalysisResult,
        CostOptimizationRecommendation,  # noqa: F401
        CrewAICostAggregator,
        CrewCostSummary,  # noqa: F401
        ProviderCostSummary,
        ProviderType,  # noqa: F401
    )
except ImportError:
    pytest.skip("CrewAI provider not available", allow_module_level=True)


class TestCrewAICostAggregator:
    """Test suite for CrewAICostAggregator."""

    def test_aggregator_initialization(self):
        """Test cost aggregator initialization."""
        aggregator = CrewAICostAggregator()

        assert aggregator is not None
        assert hasattr(aggregator, "cost_entries")
        assert hasattr(aggregator, "provider_summaries")

    def test_add_cost_entry_single_provider(self):
        """Test adding a cost entry for a single provider."""
        aggregator = CrewAICostAggregator()

        entry = AgentCostEntry(
            provider="openai",
            model="gpt-4",
            agent_id="agent_1",
            tokens_in=150,
            tokens_out=50,
            cost=0.045,
            timestamp=datetime.now(),
        )

        aggregator.add_cost_entry(entry)

        assert len(aggregator.cost_entries) == 1
        assert aggregator.cost_entries[0].provider == "openai"
        assert aggregator.cost_entries[0].cost == 0.045

    def test_add_cost_entry_multiple_providers(self):
        """Test adding cost entries for multiple providers."""
        aggregator = CrewAICostAggregator()

        providers_data = [
            ("openai", "gpt-4", 0.045),
            ("anthropic", "claude-2", 0.032),
            ("google", "gemini-pro", 0.028),
        ]

        for provider, model, cost in providers_data:
            entry = AgentCostEntry(
                provider=provider,
                model=model,
                agent_id=f"agent_{provider}",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        assert len(aggregator.cost_entries) == 3
        providers = [entry.provider for entry in aggregator.cost_entries]
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers

    def test_get_total_cost(self):
        """Test getting total cost across all providers."""
        aggregator = CrewAICostAggregator()

        costs = [0.045, 0.032, 0.028, 0.015]
        for i, cost in enumerate(costs):
            entry = AgentCostEntry(
                provider=f"provider_{i}",
                model=f"model_{i}",
                agent_id=f"agent_{i}",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        total_cost = aggregator.get_total_cost()
        expected_total = sum(costs)

        assert (
            abs(total_cost - expected_total) < 0.001
        )  # Account for floating point precision

    def test_get_cost_by_provider(self):
        """Test getting cost breakdown by provider."""
        aggregator = CrewAICostAggregator()

        # Add multiple entries for same provider
        openai_costs = [0.045, 0.023, 0.067]
        for cost in openai_costs:
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id="agent_openai",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        # Add entry for different provider
        entry = AgentCostEntry(
            provider="anthropic",
            model="claude-2",
            agent_id="agent_anthropic",
            tokens_in=100,
            tokens_out=50,
            cost=0.038,
            timestamp=datetime.now(),
        )
        aggregator.add_cost_entry(entry)

        cost_by_provider = aggregator.get_cost_by_provider()

        assert "openai" in cost_by_provider
        assert "anthropic" in cost_by_provider
        assert abs(cost_by_provider["openai"] - sum(openai_costs)) < 0.001
        assert abs(cost_by_provider["anthropic"] - 0.038) < 0.001

    def test_get_cost_by_agent(self):
        """Test getting cost breakdown by agent."""
        aggregator = CrewAICostAggregator()

        agents_data = [
            ("agent_1", 0.045),
            ("agent_1", 0.023),  # Same agent, multiple calls
            ("agent_2", 0.067),
            ("agent_3", 0.015),
        ]

        for agent_id, cost in agents_data:
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id=agent_id,
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        cost_by_agent = aggregator.get_cost_by_agent()

        assert "agent_1" in cost_by_agent
        assert "agent_2" in cost_by_agent
        assert "agent_3" in cost_by_agent
        assert abs(cost_by_agent["agent_1"] - (0.045 + 0.023)) < 0.001
        assert abs(cost_by_agent["agent_2"] - 0.067) < 0.001
        assert abs(cost_by_agent["agent_3"] - 0.015) < 0.001

    def test_get_cost_by_model(self):
        """Test getting cost breakdown by model."""
        aggregator = CrewAICostAggregator()

        models_data = [
            ("gpt-4", 0.045),
            ("gpt-4", 0.023),  # Same model, multiple calls
            ("gpt-3.5-turbo", 0.008),
            ("claude-2", 0.032),
        ]

        for model, cost in models_data:
            entry = AgentCostEntry(
                provider="openai" if "gpt" in model else "anthropic",
                model=model,
                agent_id="test_agent",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        cost_by_model = aggregator.get_cost_by_model()

        assert "gpt-4" in cost_by_model
        assert "gpt-3.5-turbo" in cost_by_model
        assert "claude-2" in cost_by_model
        assert abs(cost_by_model["gpt-4"] - (0.045 + 0.023)) < 0.001

    def test_get_provider_summary(self):
        """Test getting comprehensive provider summary."""
        aggregator = CrewAICostAggregator()

        # Add multiple entries for OpenAI
        for i in range(5):
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id=f"agent_{i}",
                tokens_in=100,
                tokens_out=50,
                cost=0.030 + i * 0.005,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        provider_summary = aggregator.get_provider_summary("openai")

        assert isinstance(provider_summary, ProviderCostSummary)
        assert provider_summary.provider == "openai"
        assert provider_summary.total_operations == 5
        assert provider_summary.total_cost > 0
        assert len(provider_summary.agents_used) == 5
        assert "gpt-4" in provider_summary.models_used

    def test_time_range_filtering(self):
        """Test filtering cost entries by time range."""
        aggregator = CrewAICostAggregator()

        # Add entries with different timestamps
        now = datetime.now()
        entries_data = [
            (now - timedelta(hours=2), 0.045),  # 2 hours ago
            (now - timedelta(hours=1), 0.032),  # 1 hour ago
            (now - timedelta(minutes=30), 0.028),  # 30 minutes ago
            (now, 0.015),  # Now
        ]

        for timestamp, cost in entries_data:
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id="test_agent",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=timestamp,
            )
            aggregator.add_cost_entry(entry)

        # Get cost for last 1 hour
        recent_cost = aggregator.get_cost_by_time_range(hours=1)
        expected_recent = 0.028 + 0.015  # Last 2 entries

        assert abs(recent_cost - expected_recent) < 0.001

    def test_cost_optimization_recommendations(self):
        """Test generating cost optimization recommendations."""
        aggregator = CrewAICostAggregator()

        # Add high-cost entries for expensive model
        expensive_entries = [
            ("openai", "gpt-4", "agent_1", 0.080),
            ("openai", "gpt-4", "agent_1", 0.075),
            ("openai", "gpt-4", "agent_1", 0.090),
        ]

        # Add low-cost entries for cheaper model
        cheap_entries = [
            ("openai", "gpt-3.5-turbo", "agent_2", 0.008),
            ("openai", "gpt-3.5-turbo", "agent_2", 0.012),
            ("openai", "gpt-3.5-turbo", "agent_2", 0.006),
        ]

        all_entries = expensive_entries + cheap_entries

        for provider, model, agent_id, cost in all_entries:
            entry = AgentCostEntry(
                provider=provider,
                model=model,
                agent_id=agent_id,
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        recommendations = aggregator.get_optimization_recommendations()

        assert len(recommendations) > 0
        # Should recommend switching from expensive to cheap model
        high_cost_recs = [r for r in recommendations if r.agent_name == "agent_1"]
        assert len(high_cost_recs) > 0

    def test_cost_analysis_result(self):
        """Test comprehensive cost analysis result."""
        aggregator = CrewAICostAggregator()

        # Add diverse cost entries
        providers_data = [
            ("openai", "gpt-4", "researcher", 0.080),
            ("openai", "gpt-3.5-turbo", "writer", 0.012),
            ("anthropic", "claude-2", "analyst", 0.045),
            ("google", "gemini-pro", "reviewer", 0.025),
        ]

        for provider, model, agent, cost in providers_data:
            entry = AgentCostEntry(
                provider=provider,
                model=model,
                agent_id=agent,
                tokens_in=150,
                tokens_out=75,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        analysis = aggregator.get_cost_analysis()

        assert isinstance(analysis, CostAnalysisResult)
        assert analysis.total_cost > 0
        assert len(analysis.cost_by_provider) > 0
        assert len(analysis.cost_by_agent) > 0
        assert len(analysis.provider_summaries) > 0

    def test_budget_limit_checking(self):
        """Test budget limit checking and alerts."""
        daily_limit = 10.0
        aggregator = CrewAICostAggregator(daily_budget_limit=daily_limit)

        # Add costs approaching the limit
        costs = [3.0, 3.5, 2.8, 1.2]  # Total = 10.5 (over limit)

        for i, cost in enumerate(costs):
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id=f"agent_{i}",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

            # Check if budget exceeded
            is_over_budget = aggregator.is_over_budget()
            total_cost = aggregator.get_total_cost()

            if total_cost > daily_limit:
                assert is_over_budget is True
            else:
                assert is_over_budget is False

    def test_cost_trend_analysis(self):
        """Test cost trend analysis over time."""
        aggregator = CrewAICostAggregator()

        # Add entries with increasing cost trend
        base_time = datetime.now() - timedelta(hours=5)

        for hour in range(5):
            cost = 0.020 + (hour * 0.010)  # Increasing cost trend
            timestamp = base_time + timedelta(hours=hour)

            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id="trend_agent",
                tokens_in=100,
                tokens_out=50,
                cost=cost,
                timestamp=timestamp,
            )
            aggregator.add_cost_entry(entry)

        # Analyze trend (this would need trend analysis method)
        hourly_costs = {}
        for entry in aggregator.cost_entries:
            hour = entry.timestamp.hour
            if hour not in hourly_costs:
                hourly_costs[hour] = 0
            hourly_costs[hour] += entry.cost

        # Should show increasing trend
        costs_list = list(hourly_costs.values())
        assert len(costs_list) >= 2
        # Verify trend is generally increasing (allowing for some variation)
        trend_positive = costs_list[-1] > costs_list[0]
        assert trend_positive

    def test_concurrent_cost_tracking(self):
        """Test concurrent cost tracking from multiple threads."""
        import threading

        aggregator = CrewAICostAggregator()
        results = []

        def add_costs(thread_id):
            for i in range(10):
                entry = AgentCostEntry(
                    provider="openai",
                    model="gpt-4",
                    agent_id=f"thread_{thread_id}_agent_{i}",
                    tokens_in=100,
                    tokens_out=50,
                    cost=0.01 + (thread_id * 0.01) + (i * 0.001),
                    timestamp=datetime.now(),
                )
                aggregator.add_cost_entry(entry)
            results.append(thread_id)

        # Start multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=add_costs, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert len(aggregator.cost_entries) == 30  # 3 threads * 10 entries each
        total_cost = aggregator.get_total_cost()
        assert total_cost > 0

    def test_provider_comparison(self):
        """Test provider cost and performance comparison."""
        aggregator = CrewAICostAggregator()

        # Add entries for different providers with different characteristics
        providers_data = [
            ("openai", "gpt-4", 0.080, 95),  # Expensive but high quality
            ("openai", "gpt-3.5-turbo", 0.012, 85),  # Cheaper, lower quality
            ("anthropic", "claude-2", 0.045, 90),  # Mid-range
            ("google", "gemini-pro", 0.025, 88),  # Good value
        ]

        for provider, model, cost, quality_score in providers_data:
            entry = AgentCostEntry(
                provider=provider,
                model=model,
                agent_id=f"agent_{provider}",
                tokens_in=150,
                tokens_out=75,
                cost=cost,
                timestamp=datetime.now(),
            )
            # Add quality score as metadata if supported
            if hasattr(entry, "metadata"):
                entry.metadata = {"quality_score": quality_score}
            aggregator.add_cost_entry(entry)

        cost_by_provider = aggregator.get_cost_by_provider()

        # Verify all providers are tracked
        assert "openai" in cost_by_provider
        assert "anthropic" in cost_by_provider
        assert "google" in cost_by_provider

        # OpenAI should have highest cost (due to gpt-4)
        assert cost_by_provider["openai"] > cost_by_provider["google"]

    def test_cost_entry_validation(self):
        """Test validation of cost entry data."""
        aggregator = CrewAICostAggregator()

        # Test valid entry
        valid_entry = AgentCostEntry(
            provider="openai",
            model="gpt-4",
            agent_id="valid_agent",
            tokens_in=100,
            tokens_out=50,
            cost=0.045,
            timestamp=datetime.now(),
        )

        aggregator.add_cost_entry(valid_entry)
        assert len(aggregator.cost_entries) == 1

        # Test invalid entries (should be handled gracefully)
        try:
            invalid_entry = AgentCostEntry(
                provider="",  # Empty provider
                model="gpt-4",
                agent_id="test_agent",
                tokens_in=100,
                tokens_out=50,
                cost=0.045,
                timestamp=datetime.now(),
            )
            # Depending on implementation, this might raise an error or be handled
            aggregator.add_cost_entry(invalid_entry)
        except (ValueError, TypeError):
            pass  # Expected for invalid entry

    def test_cost_aggregator_reset(self):
        """Test resetting cost aggregator data."""
        aggregator = CrewAICostAggregator()

        # Add some entries
        for i in range(5):
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id=f"agent_{i}",
                tokens_in=100,
                tokens_out=50,
                cost=0.030,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        assert len(aggregator.cost_entries) == 5

        # Reset aggregator
        if hasattr(aggregator, "reset"):
            aggregator.reset()
            assert len(aggregator.cost_entries) == 0
            assert aggregator.get_total_cost() == 0

    def test_export_cost_data(self):
        """Test exporting cost data for external analysis."""
        aggregator = CrewAICostAggregator()

        # Add sample data
        for i in range(3):
            entry = AgentCostEntry(
                provider="openai",
                model="gpt-4",
                agent_id=f"agent_{i}",
                tokens_in=100 + i * 10,
                tokens_out=50 + i * 5,
                cost=0.030 + i * 0.005,
                timestamp=datetime.now(),
            )
            aggregator.add_cost_entry(entry)

        # Export data (implementation dependent)
        if hasattr(aggregator, "export_data"):
            exported_data = aggregator.export_data()
            assert isinstance(exported_data, (dict, list))
            assert len(exported_data) > 0
        else:
            # Test getting raw entries
            entries = aggregator.cost_entries
            assert len(entries) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
