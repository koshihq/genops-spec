#!/usr/bin/env python3
"""
Comprehensive test suite for Haystack cost aggregator functionality.

Tests cover cost calculation, multi-provider aggregation, optimization recommendations,
and analysis scenarios as required by CLAUDE.md standards.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from genops.providers.haystack_cost_aggregator import (
    ComponentCostEntry,
    CostAnalysisResult,
    CostOptimizationRecommendation,
    HaystackCostAggregator,
    ProviderCostSummary,
)


class TestComponentCostEntry:
    """Component cost entry data structure tests."""

    def test_cost_entry_creation(self):
        """Test component cost entry creation."""
        entry = ComponentCostEntry(
            component_name="test-generator",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            cost=Decimal("0.005"),
            timestamp=datetime.now()
        )

        assert entry.component_name == "test-generator"
        assert entry.component_type == "Generator"
        assert entry.provider_name == "OpenAI"
        assert entry.model_name == "gpt-3.5-turbo"
        assert entry.cost == Decimal("0.005")

    def test_cost_entry_with_optional_fields(self):
        """Test cost entry with optional fields."""
        entry = ComponentCostEntry(
            component_name="test-retriever",
            component_type="Retriever",
            provider_name="HuggingFace",
            model_name="sentence-transformers",
            cost=Decimal("0.001"),
            timestamp=datetime.now(),
            tokens_used=100,
            operation_type="embedding"
        )

        assert entry.tokens_used == 100
        assert entry.operation_type == "embedding"


class TestProviderCostSummary:
    """Provider cost summary data structure tests."""

    def test_provider_summary_creation(self):
        """Test provider cost summary creation."""
        summary = ProviderCostSummary(
            provider_name="OpenAI",
            total_cost=Decimal("0.025"),
            total_operations=15,
            components_used={"Generator", "Embedder"},
            models_used={"gpt-3.5-turbo", "text-embedding-ada-002"}
        )

        assert summary.provider_name == "OpenAI"
        assert summary.total_cost == Decimal("0.025")
        assert summary.total_operations == 15
        assert len(summary.components_used) == 2
        assert len(summary.models_used) == 2


class TestCostOptimizationRecommendation:
    """Cost optimization recommendation tests."""

    def test_recommendation_creation(self):
        """Test optimization recommendation creation."""
        recommendation = CostOptimizationRecommendation(
            component_name="expensive-generator",
            current_provider="OpenAI",
            recommended_provider="Anthropic",
            current_cost=Decimal("0.020"),
            recommended_cost=Decimal("0.015"),
            potential_savings=Decimal("0.005"),
            reasoning="Anthropic offers better cost-performance for this use case"
        )

        assert recommendation.component_name == "expensive-generator"
        assert recommendation.current_provider == "OpenAI"
        assert recommendation.recommended_provider == "Anthropic"
        assert recommendation.potential_savings == Decimal("0.005")

    def test_recommendation_savings_calculation(self):
        """Test recommendation calculates savings correctly."""
        recommendation = CostOptimizationRecommendation(
            component_name="test-component",
            current_provider="Provider A",
            recommended_provider="Provider B",
            current_cost=Decimal("0.100"),
            recommended_cost=Decimal("0.075"),
            potential_savings=Decimal("0.025"),
            reasoning="Test reason"
        )

        savings_percentage = (recommendation.potential_savings / recommendation.current_cost) * 100
        assert savings_percentage == 25.0


class TestHaystackCostAggregator:
    """Core cost aggregator functionality tests."""

    def test_aggregator_initialization(self):
        """Test cost aggregator initializes properly."""
        aggregator = HaystackCostAggregator()

        assert aggregator.cost_entries == []
        assert hasattr(aggregator, 'provider_pricing')
        assert hasattr(aggregator, 'add_component_cost')

    def test_aggregator_add_component_cost(self):
        """Test adding component cost to aggregator."""
        aggregator = HaystackCostAggregator()

        aggregator.add_component_cost(
            component_name="test-generator",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            cost=Decimal("0.005")
        )

        assert len(aggregator.cost_entries) == 1
        entry = aggregator.cost_entries[0]
        assert entry.component_name == "test-generator"
        assert entry.cost == Decimal("0.005")

    def test_aggregator_calculate_cost_openai_gpt35(self):
        """Test OpenAI GPT-3.5 cost calculation."""
        aggregator = HaystackCostAggregator()

        cost = aggregator._calculate_component_cost(
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            component_type="Generator",
            tokens_used=1000
        )

        # GPT-3.5-turbo: $0.002 per 1K tokens
        expected_cost = Decimal("0.002")
        assert cost == expected_cost

    def test_aggregator_calculate_cost_openai_gpt4(self):
        """Test OpenAI GPT-4 cost calculation."""
        aggregator = HaystackCostAggregator()

        cost = aggregator._calculate_component_cost(
            provider_name="OpenAI",
            model_name="gpt-4",
            component_type="Generator",
            tokens_used=1000
        )

        # GPT-4: $0.06 per 1K tokens (input)
        expected_cost = Decimal("0.06")
        assert cost == expected_cost

    def test_aggregator_calculate_cost_anthropic_claude(self):
        """Test Anthropic Claude cost calculation."""
        aggregator = HaystackCostAggregator()

        cost = aggregator._calculate_component_cost(
            provider_name="Anthropic",
            model_name="claude-3-haiku",
            component_type="Generator",
            tokens_used=1000
        )

        # Claude-3-haiku: $0.00025 per 1K tokens (input)
        expected_cost = Decimal("0.00025")
        assert cost == expected_cost

    def test_aggregator_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model uses generic pricing."""
        aggregator = HaystackCostAggregator()

        cost = aggregator._calculate_component_cost(
            provider_name="UnknownProvider",
            model_name="unknown-model",
            component_type="Generator",
            tokens_used=1000
        )

        # Generic pricing: $0.001 per 1K tokens
        expected_cost = Decimal("0.001")
        assert cost == expected_cost

    def test_aggregator_get_cost_summary(self):
        """Test cost aggregator summary calculation."""
        aggregator = HaystackCostAggregator()

        # Add multiple cost entries
        aggregator.add_component_cost(
            component_name="generator1",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            cost=Decimal("0.005")
        )

        aggregator.add_component_cost(
            component_name="retriever1",
            component_type="Retriever",
            provider_name="HuggingFace",
            model_name="sentence-transformers",
            cost=Decimal("0.001")
        )

        aggregator.add_component_cost(
            component_name="generator2",
            component_type="Generator",
            provider_name="Anthropic",
            model_name="claude-3-haiku",
            cost=Decimal("0.003")
        )

        summary = aggregator.get_cost_summary()

        assert summary["total_cost"] == Decimal("0.009")
        assert summary["cost_by_provider"]["OpenAI"] == Decimal("0.005")
        assert summary["cost_by_provider"]["HuggingFace"] == Decimal("0.001")
        assert summary["cost_by_provider"]["Anthropic"] == Decimal("0.003")
        assert summary["cost_by_component"]["generator1"] == Decimal("0.005")

    def test_aggregator_get_daily_costs(self):
        """Test daily cost calculation."""
        aggregator = HaystackCostAggregator()

        # Add costs from different days
        today = datetime.now()
        yesterday = today - timedelta(days=1)

        aggregator.cost_entries = [
            ComponentCostEntry(
                component_name="today1",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-3.5-turbo",
                cost=Decimal("0.005"),
                timestamp=today
            ),
            ComponentCostEntry(
                component_name="today2",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-3.5-turbo",
                cost=Decimal("0.003"),
                timestamp=today
            ),
            ComponentCostEntry(
                component_name="yesterday1",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-3.5-turbo",
                cost=Decimal("0.010"),
                timestamp=yesterday
            )
        ]

        daily_costs = aggregator.get_daily_costs()
        assert daily_costs == Decimal("0.008")  # Only today's costs

    def test_aggregator_get_cost_analysis(self):
        """Test comprehensive cost analysis."""
        aggregator = HaystackCostAggregator()

        # Add various cost entries
        aggregator.add_component_cost(
            component_name="expensive-generator",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-4",
            cost=Decimal("0.050")
        )

        aggregator.add_component_cost(
            component_name="cheap-generator",
            component_type="Generator",
            provider_name="Anthropic",
            model_name="claude-3-haiku",
            cost=Decimal("0.005")
        )

        analysis = aggregator.get_cost_analysis(time_period_hours=24)

        assert isinstance(analysis, CostAnalysisResult)
        assert analysis.total_cost == Decimal("0.055")
        assert len(analysis.cost_by_provider) == 2
        assert analysis.cost_by_provider["OpenAI"] == Decimal("0.050")
        assert analysis.cost_by_provider["Anthropic"] == Decimal("0.005")

    def test_aggregator_optimization_recommendations(self):
        """Test cost optimization recommendations."""
        aggregator = HaystackCostAggregator()

        # Add expensive OpenAI operations
        for i in range(5):
            aggregator.add_component_cost(
                component_name=f"generator{i}",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-4",
                cost=Decimal("0.060")
            )

        analysis = aggregator.get_cost_analysis(time_period_hours=24)

        # Should generate recommendations due to high OpenAI costs
        assert len(analysis.optimization_recommendations) > 0

        # Check recommendation properties
        rec = analysis.optimization_recommendations[0]
        assert rec.current_provider == "OpenAI"
        assert rec.potential_savings > Decimal("0")


class TestProviderPricingModels:
    """Provider pricing model tests."""

    def test_openai_pricing_models(self):
        """Test OpenAI pricing models are defined."""
        aggregator = HaystackCostAggregator()
        pricing = aggregator.provider_pricing

        assert "OpenAI" in pricing
        assert "gpt-3.5-turbo" in pricing["OpenAI"]
        assert "gpt-4" in pricing["OpenAI"]
        assert "text-embedding-ada-002" in pricing["OpenAI"]

    def test_anthropic_pricing_models(self):
        """Test Anthropic pricing models are defined."""
        aggregator = HaystackCostAggregator()
        pricing = aggregator.provider_pricing

        assert "Anthropic" in pricing
        assert "claude-3-haiku" in pricing["Anthropic"]
        assert "claude-3-sonnet" in pricing["Anthropic"]
        assert "claude-3-opus" in pricing["Anthropic"]

    def test_huggingface_pricing_models(self):
        """Test HuggingFace pricing models are defined."""
        aggregator = HaystackCostAggregator()
        pricing = aggregator.provider_pricing

        assert "HuggingFace" in pricing
        # Generic pricing for HuggingFace models

    def test_cohere_pricing_models(self):
        """Test Cohere pricing models are defined."""
        aggregator = HaystackCostAggregator()
        pricing = aggregator.provider_pricing

        assert "Cohere" in pricing


class TestCostAnalysisResult:
    """Cost analysis result tests."""

    def test_analysis_result_creation(self):
        """Test cost analysis result creation."""
        result = CostAnalysisResult(
            total_cost=Decimal("0.100"),
            cost_by_provider={"OpenAI": Decimal("0.060"), "Anthropic": Decimal("0.040")},
            cost_by_component={"gen1": Decimal("0.060"), "gen2": Decimal("0.040")},
            provider_summaries={
                "OpenAI": ProviderCostSummary(
                    provider_name="OpenAI",
                    total_cost=Decimal("0.060"),
                    total_operations=10,
                    components_used={"Generator"},
                    models_used={"gpt-4"}
                )
            },
            optimization_recommendations=[],
            time_period_hours=24
        )

        assert result.total_cost == Decimal("0.100")
        assert len(result.cost_by_provider) == 2
        assert result.time_period_hours == 24


class TestMultiProviderScenarios:
    """Multi-provider cost aggregation tests."""

    def test_multi_provider_cost_tracking(self):
        """Test tracking costs across multiple providers."""
        aggregator = HaystackCostAggregator()

        # OpenAI generator
        aggregator.add_component_cost(
            component_name="openai-gen",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            cost=Decimal("0.010")
        )

        # Anthropic generator
        aggregator.add_component_cost(
            component_name="anthropic-gen",
            component_type="Generator",
            provider_name="Anthropic",
            model_name="claude-3-haiku",
            cost=Decimal("0.005")
        )

        # HuggingFace embedder
        aggregator.add_component_cost(
            component_name="hf-embed",
            component_type="Embedder",
            provider_name="HuggingFace",
            model_name="sentence-transformers",
            cost=Decimal("0.001")
        )

        summary = aggregator.get_cost_summary()

        assert summary["total_cost"] == Decimal("0.016")
        assert len(summary["cost_by_provider"]) == 3
        assert "OpenAI" in summary["cost_by_provider"]
        assert "Anthropic" in summary["cost_by_provider"]
        assert "HuggingFace" in summary["cost_by_provider"]

    def test_cross_provider_optimization(self):
        """Test optimization recommendations across providers."""
        aggregator = HaystackCostAggregator()

        # Add many expensive OpenAI calls
        for i in range(10):
            aggregator.add_component_cost(
                component_name=f"openai-gen-{i}",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-4",
                cost=Decimal("0.060")
            )

        analysis = aggregator.get_cost_analysis(time_period_hours=1)

        # Should recommend switching to cheaper alternatives
        assert len(analysis.optimization_recommendations) > 0
        rec = analysis.optimization_recommendations[0]
        assert rec.current_provider == "OpenAI"
        assert rec.recommended_provider in ["Anthropic", "Cohere"]


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_aggregator_empty_cost_entries(self):
        """Test aggregator with no cost entries."""
        aggregator = HaystackCostAggregator()

        summary = aggregator.get_cost_summary()
        assert summary["total_cost"] == Decimal("0.0")
        assert len(summary["cost_by_provider"]) == 0

        daily_costs = aggregator.get_daily_costs()
        assert daily_costs == Decimal("0.0")

    def test_aggregator_zero_cost_entries(self):
        """Test aggregator with zero-cost entries."""
        aggregator = HaystackCostAggregator()

        aggregator.add_component_cost(
            component_name="free-component",
            component_type="Preprocessor",
            provider_name="Local",
            model_name="custom-model",
            cost=Decimal("0.0")
        )

        summary = aggregator.get_cost_summary()
        assert summary["total_cost"] == Decimal("0.0")

    def test_aggregator_very_small_costs(self):
        """Test aggregator with very small cost values."""
        aggregator = HaystackCostAggregator()

        aggregator.add_component_cost(
            component_name="micro-cost",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            cost=Decimal("0.000001")
        )

        summary = aggregator.get_cost_summary()
        assert summary["total_cost"] == Decimal("0.000001")

    def test_aggregator_invalid_cost_values(self):
        """Test aggregator handles invalid cost values."""
        aggregator = HaystackCostAggregator()

        # Negative costs should be handled gracefully
        with pytest.raises(ValueError):
            aggregator.add_component_cost(
                component_name="invalid-cost",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-3.5-turbo",
                cost=Decimal("-0.001")
            )

    def test_aggregator_missing_provider_pricing(self):
        """Test aggregator handles unknown providers gracefully."""
        aggregator = HaystackCostAggregator()

        cost = aggregator._calculate_component_cost(
            provider_name="UnknownProvider",
            model_name="unknown-model",
            component_type="Generator",
            tokens_used=1000
        )

        # Should fall back to generic pricing
        assert cost == Decimal("0.001")  # Generic rate
