"""Tests for LangChain cost aggregation functionality."""

import uuid
from unittest.mock import patch

import pytest

from genops.providers.langchain.cost_aggregator import (
    ChainCostContext,
    ChainCostSummary,
    LangChainCostAggregator,
    LLMCallCost,
    create_chain_cost_context,
    get_cost_aggregator,
)


class TestLLMCallCost:
    """Test LLMCallCost dataclass."""

    def test_llm_call_cost_creation(self):
        """Test creating LLMCallCost instance."""
        cost = LLMCallCost(
            provider="openai",
            model="gpt-4",
            tokens_input=100,
            tokens_output=50,
            cost=0.015,
            operation_name="test_operation"
        )

        assert cost.provider == "openai"
        assert cost.model == "gpt-4"
        assert cost.tokens_input == 100
        assert cost.tokens_output == 50
        assert cost.cost == 0.015
        assert cost.currency == "USD"
        assert cost.operation_name == "test_operation"
        assert isinstance(cost.metadata, dict)


class TestChainCostSummary:
    """Test ChainCostSummary dataclass."""

    def test_empty_chain_cost_summary(self):
        """Test creating empty ChainCostSummary."""
        summary = ChainCostSummary()

        assert summary.total_cost == 0.0
        assert summary.currency == "USD"
        assert len(summary.llm_calls) == 0
        assert len(summary.cost_by_provider) == 0
        assert len(summary.cost_by_model) == 0
        assert summary.total_tokens_input == 0
        assert summary.total_tokens_output == 0
        assert len(summary.unique_providers) == 0
        assert len(summary.unique_models) == 0

    def test_chain_cost_summary_with_calls(self):
        """Test ChainCostSummary with LLM calls."""
        calls = [
            LLMCallCost("openai", "gpt-4", 100, 50, 0.015),
            LLMCallCost("anthropic", "claude-3", 80, 40, 0.012),
            LLMCallCost("openai", "gpt-3.5", 60, 30, 0.003),
        ]

        summary = ChainCostSummary(llm_calls=calls)

        assert summary.total_cost == 0.030
        assert summary.cost_by_provider["openai"] == 0.018
        assert summary.cost_by_provider["anthropic"] == 0.012
        assert summary.cost_by_model["gpt-4"] == 0.015
        assert summary.cost_by_model["claude-3"] == 0.012
        assert summary.cost_by_model["gpt-3.5"] == 0.003
        assert summary.total_tokens_input == 240
        assert summary.total_tokens_output == 120
        assert summary.unique_providers == {"openai", "anthropic"}
        assert summary.unique_models == {"gpt-4", "claude-3", "gpt-3.5"}

    def test_add_llm_call(self):
        """Test adding LLM call to summary."""
        summary = ChainCostSummary()

        call = LLMCallCost("openai", "gpt-4", 100, 50, 0.015)
        summary.add_llm_call(call)

        assert len(summary.llm_calls) == 1
        assert summary.total_cost == 0.015
        assert summary.unique_providers == {"openai"}

    def test_to_dict(self):
        """Test converting summary to dictionary."""
        calls = [LLMCallCost("openai", "gpt-4", 100, 50, 0.015)]
        summary = ChainCostSummary(llm_calls=calls)

        result = summary.to_dict()

        assert isinstance(result, dict)
        assert result["total_cost"] == 0.015
        assert result["llm_calls_count"] == 1
        assert result["total_tokens_input"] == 100
        assert result["total_tokens_output"] == 50
        assert result["provider_count"] == 1
        assert result["model_count"] == 1
        assert result["unique_providers"] == ["openai"]
        assert result["unique_models"] == ["gpt-4"]


class TestLangChainCostAggregator:
    """Test LangChainCostAggregator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = LangChainCostAggregator()

    def test_aggregator_initialization(self):
        """Test aggregator initializes correctly."""
        assert isinstance(self.aggregator.active_chains, dict)
        assert len(self.aggregator.active_chains) == 0
        assert isinstance(self.aggregator.provider_cost_calculators, dict)

    def test_start_chain_tracking(self):
        """Test starting chain tracking."""
        chain_id = "test_chain_123"

        self.aggregator.start_chain_tracking(chain_id)

        assert chain_id in self.aggregator.active_chains
        assert isinstance(self.aggregator.active_chains[chain_id], ChainCostSummary)

    def test_add_llm_call_cost_success(self):
        """Test adding LLM call cost successfully."""
        chain_id = "test_chain_123"
        self.aggregator.start_chain_tracking(chain_id)

        result = self.aggregator.add_llm_call_cost(
            chain_id=chain_id,
            provider="openai",
            model="gpt-4",
            tokens_input=100,
            tokens_output=50,
            operation_name="test_op"
        )

        assert result is not None
        assert isinstance(result, LLMCallCost)
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert result.tokens_input == 100
        assert result.tokens_output == 50
        assert result.cost > 0  # Should calculate some cost

        # Check that it was added to the chain
        summary = self.aggregator.active_chains[chain_id]
        assert len(summary.llm_calls) == 1
        assert summary.total_cost > 0

    def test_add_llm_call_cost_nonexistent_chain(self):
        """Test adding LLM call cost to nonexistent chain."""
        result = self.aggregator.add_llm_call_cost(
            chain_id="nonexistent",
            provider="openai",
            model="gpt-4",
            tokens_input=100,
            tokens_output=50
        )

        assert result is None

    @patch('genops.providers.langchain.cost_aggregator.LangChainCostAggregator._calculate_provider_cost')
    def test_add_llm_call_cost_with_mocked_calculation(self, mock_calc):
        """Test adding LLM call cost with mocked cost calculation."""
        mock_calc.return_value = 0.025

        chain_id = "test_chain"
        self.aggregator.start_chain_tracking(chain_id)

        result = self.aggregator.add_llm_call_cost(
            chain_id=chain_id,
            provider="custom",
            model="custom-model",
            tokens_input=200,
            tokens_output=100
        )

        assert result.cost == 0.025
        mock_calc.assert_called_once_with("custom", "custom-model", 200, 100)

    def test_generic_cost_calculation(self):
        """Test generic cost calculation fallback."""
        cost = self.aggregator._generic_cost_calculation("unknown-model", 1000, 500)

        assert cost > 0
        assert isinstance(cost, float)

    def test_generic_cost_calculation_known_patterns(self):
        """Test generic cost calculation with known model patterns."""
        gpt4_cost = self.aggregator._generic_cost_calculation("gpt-4-turbo", 1000, 500)
        claude_cost = self.aggregator._generic_cost_calculation("claude-3-sonnet", 1000, 500)

        assert gpt4_cost > 0
        assert claude_cost > 0
        # Claude should be more expensive per token (different pricing structure)

    def test_finalize_chain_tracking(self):
        """Test finalizing chain tracking."""
        chain_id = "test_chain"
        self.aggregator.start_chain_tracking(chain_id)

        # Add some costs
        self.aggregator.add_llm_call_cost(
            chain_id=chain_id,
            provider="openai",
            model="gpt-4",
            tokens_input=100,
            tokens_output=50
        )

        summary = self.aggregator.finalize_chain_tracking(chain_id, total_time=2.5)

        assert summary is not None
        assert isinstance(summary, ChainCostSummary)
        assert summary.total_time == 2.5
        assert chain_id not in self.aggregator.active_chains  # Should be removed

    def test_finalize_nonexistent_chain(self):
        """Test finalizing nonexistent chain."""
        summary = self.aggregator.finalize_chain_tracking("nonexistent", 1.0)

        assert summary is None

    def test_get_chain_summary(self):
        """Test getting chain summary."""
        chain_id = "test_chain"
        self.aggregator.start_chain_tracking(chain_id)

        summary = self.aggregator.get_chain_summary(chain_id)

        assert summary is not None
        assert isinstance(summary, ChainCostSummary)

    def test_get_active_chains(self):
        """Test getting active chains."""
        chain1 = "chain1"
        chain2 = "chain2"

        self.aggregator.start_chain_tracking(chain1)
        self.aggregator.start_chain_tracking(chain2)

        active = self.aggregator.get_active_chains()

        assert len(active) == 2
        assert chain1 in active
        assert chain2 in active

    def test_clear_all_tracking(self):
        """Test clearing all tracking."""
        self.aggregator.start_chain_tracking("chain1")
        self.aggregator.start_chain_tracking("chain2")

        assert len(self.aggregator.active_chains) == 2

        self.aggregator.clear_all_tracking()

        assert len(self.aggregator.active_chains) == 0


class TestChainCostContext:
    """Test ChainCostContext context manager."""

    def test_chain_cost_context_creation(self):
        """Test creating chain cost context."""
        context = ChainCostContext("test_chain_id")

        assert context.chain_id == "test_chain_id"
        assert context.start_time is None
        assert context.summary is None

    def test_chain_cost_context_manager(self):
        """Test using ChainCostContext as context manager."""
        chain_id = str(uuid.uuid4())

        with ChainCostContext(chain_id) as context:
            assert context.chain_id == chain_id
            assert context.start_time is not None
            assert context.operation_id == chain_id  # Should be set to chain_id

            # Add a cost within the context
            context.add_llm_call(
                provider="openai",
                model="gpt-4",
                tokens_input=100,
                tokens_output=50
            )

            current_summary = context.get_current_summary()
            assert current_summary is not None

        # After exiting context
        final_summary = context.get_final_summary()
        assert final_summary is not None
        assert final_summary.total_time > 0

    def test_chain_cost_context_with_exception(self):
        """Test ChainCostContext handles exceptions properly."""
        chain_id = str(uuid.uuid4())

        try:
            with ChainCostContext(chain_id) as context:
                assert context.operation_id is not None
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still finalize properly
        assert context.get_final_summary() is not None


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_cost_aggregator_singleton(self):
        """Test that get_cost_aggregator returns singleton."""
        aggregator1 = get_cost_aggregator()
        aggregator2 = get_cost_aggregator()

        assert aggregator1 is aggregator2
        assert isinstance(aggregator1, LangChainCostAggregator)

    def test_create_chain_cost_context(self):
        """Test creating chain cost context."""
        chain_id = "test_chain"

        context = create_chain_cost_context(chain_id)

        assert isinstance(context, ChainCostContext)
        assert context.chain_id == chain_id


@pytest.fixture
def mock_openai_calculator():
    """Mock OpenAI cost calculator."""
    def calculator(model, input_tokens, output_tokens):
        # Simple mock calculation
        return (input_tokens * 0.00003) + (output_tokens * 0.00006)
    return calculator


@pytest.fixture
def mock_anthropic_calculator():
    """Mock Anthropic cost calculator."""
    def calculator(model, input_tokens, output_tokens):
        # Simple mock calculation
        return (input_tokens * 0.000003) + (output_tokens * 0.000015)
    return calculator


class TestCostAggregatorIntegration:
    """Integration tests for cost aggregator."""

    def test_multi_provider_cost_aggregation(self, mock_openai_calculator, mock_anthropic_calculator):
        """Test aggregating costs from multiple providers."""
        aggregator = LangChainCostAggregator()
        aggregator.provider_cost_calculators["openai"] = mock_openai_calculator
        aggregator.provider_cost_calculators["anthropic"] = mock_anthropic_calculator

        chain_id = "multi_provider_chain"
        aggregator.start_chain_tracking(chain_id)

        # Add OpenAI call
        openai_call = aggregator.add_llm_call_cost(
            chain_id=chain_id,
            provider="openai",
            model="gpt-4",
            tokens_input=1000,
            tokens_output=500,
            operation_name="openai_completion"
        )

        # Add Anthropic call
        anthropic_call = aggregator.add_llm_call_cost(
            chain_id=chain_id,
            provider="anthropic",
            model="claude-3",
            tokens_input=800,
            tokens_output=400,
            operation_name="anthropic_completion"
        )

        summary = aggregator.finalize_chain_tracking(chain_id, total_time=3.5)

        assert summary is not None
        assert len(summary.llm_calls) == 2
        assert len(summary.unique_providers) == 2
        assert len(summary.unique_models) == 2
        assert summary.cost_by_provider["openai"] == openai_call.cost
        assert summary.cost_by_provider["anthropic"] == anthropic_call.cost
        assert summary.total_cost == openai_call.cost + anthropic_call.cost
        assert summary.total_tokens_input == 1800
        assert summary.total_tokens_output == 900

    def test_end_to_end_cost_tracking(self):
        """Test end-to-end cost tracking workflow."""
        chain_id = str(uuid.uuid4())

        with create_chain_cost_context(chain_id) as context:
            # Simulate multiple LLM calls in a chain
            context.add_llm_call("openai", "gpt-4", 500, 250)
            context.add_llm_call("anthropic", "claude-3", 300, 150)
            context.add_llm_call("openai", "gpt-3.5", 200, 100)

            # Record generation cost
            context.record_generation_cost(0.05)

        final_summary = context.get_final_summary()

        assert final_summary is not None
        assert len(final_summary.llm_calls) == 3
        assert final_summary.generation_cost == 0.05
        assert final_summary.total_cost > 0
        assert final_summary.total_time > 0
        assert len(final_summary.unique_providers) == 2
        assert len(final_summary.unique_models) == 3
