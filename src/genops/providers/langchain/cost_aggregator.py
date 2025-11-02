"""Multi-provider cost aggregation for LangChain operations."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMCallCost:
    """Represents cost information for a single LLM call."""
    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    cost: float
    currency: str = "USD"
    operation_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainCostSummary:
    """Aggregated cost summary for a LangChain operation."""
    total_cost: float = 0.0
    currency: str = "USD"
    llm_calls: list[LLMCallCost] = field(default_factory=list)
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    unique_providers: set[str] = field(default_factory=set)
    unique_models: set[str] = field(default_factory=set)
    total_time: float = 0.0
    generation_cost: float = 0.0

    def __post_init__(self):
        """Calculate aggregated values after initialization."""
        self._calculate_aggregates()

    def _calculate_aggregates(self) -> None:
        """Calculate aggregate cost and token values."""
        self.cost_by_provider = defaultdict(float)
        self.cost_by_model = defaultdict(float)
        self.unique_providers = set()
        self.unique_models = set()
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        for call in self.llm_calls:
            self.cost_by_provider[call.provider] += call.cost
            self.cost_by_model[call.model] += call.cost
            self.unique_providers.add(call.provider)
            self.unique_models.add(call.model)
            self.total_tokens_input += call.tokens_input
            self.total_tokens_output += call.tokens_output

        self.total_cost = sum(call.cost for call in self.llm_calls)

    def add_llm_call(self, llm_call: LLMCallCost) -> None:
        """Add an LLM call to the summary."""
        self.llm_calls.append(llm_call)
        self._calculate_aggregates()

    def calculate_total_cost(self) -> float:
        """Calculate total cost from all LLM calls and generation cost."""
        llm_cost = sum(call.cost for call in self.llm_calls)
        return llm_cost + self.generation_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "total_cost": self.total_cost,
            "currency": self.currency,
            "llm_calls_count": len(self.llm_calls),
            "cost_by_provider": dict(self.cost_by_provider),
            "cost_by_model": dict(self.cost_by_model),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "unique_providers": list(self.unique_providers),
            "unique_models": list(self.unique_models),
            "provider_count": len(self.unique_providers),
            "model_count": len(self.unique_models)
        }


class LangChainCostAggregator:
    """Aggregates costs across multiple providers in LangChain operations."""

    def __init__(self):
        self.active_chains: dict[str, ChainCostSummary] = {}
        self.provider_cost_calculators = {}
        self._setup_provider_calculators()

    def _setup_provider_calculators(self) -> None:
        """Setup cost calculators for different providers."""
        try:
            from genops.providers.openai import GenOpsOpenAIAdapter
            # Create adapter without client to avoid requiring API keys
            adapter = GenOpsOpenAIAdapter.__new__(GenOpsOpenAIAdapter)
            self.provider_cost_calculators["openai"] = adapter._calculate_cost
        except (ImportError, Exception) as e:
            logger.debug(f"OpenAI provider not available for cost calculation: {e}")

        try:
            from genops.providers.anthropic import GenOpsAnthropicAdapter
            # Create adapter without client to avoid requiring API keys
            adapter = GenOpsAnthropicAdapter.__new__(GenOpsAnthropicAdapter)
            self.provider_cost_calculators["anthropic"] = adapter._calculate_cost
        except (ImportError, Exception) as e:
            logger.debug(f"Anthropic provider not available for cost calculation: {e}")

    def start_chain_tracking(self, chain_id: str) -> None:
        """Start tracking costs for a chain execution."""
        self.active_chains[chain_id] = ChainCostSummary()
        logger.debug(f"Started cost tracking for chain: {chain_id}")

    def add_llm_call_cost(
        self,
        chain_id: str,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        operation_name: str | None = None,
        **metadata
    ) -> LLMCallCost | None:
        """
        Add an LLM call cost to a chain's tracking.

        Args:
            chain_id: Unique identifier for the chain
            provider: Provider name (openai, anthropic, etc.)
            model: Model name
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
            operation_name: Name of the operation
            **metadata: Additional metadata

        Returns:
            LLMCallCost object if successful, None otherwise
        """
        if chain_id not in self.active_chains:
            logger.warning(f"Chain {chain_id} not found in active tracking")
            return None

        # Calculate cost using provider-specific calculator
        cost = self._calculate_provider_cost(provider, model, tokens_input, tokens_output)

        llm_call = LLMCallCost(
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            operation_name=operation_name,
            metadata=metadata
        )

        self.active_chains[chain_id].add_llm_call(llm_call)
        logger.debug(f"Added LLM call cost: ${cost:.4f} ({provider}/{model}) to chain {chain_id}")

        return llm_call

    def _calculate_provider_cost(
        self,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int
    ) -> float:
        """Calculate cost using provider-specific logic."""
        provider_key = provider.lower()

        if provider_key in self.provider_cost_calculators:
            try:
                return self.provider_cost_calculators[provider_key](
                    model, tokens_input, tokens_output
                )
            except Exception as e:
                logger.warning(f"Failed to calculate cost for {provider}: {e}")

        # Fallback to generic pricing
        return self._generic_cost_calculation(model, tokens_input, tokens_output)

    def _generic_cost_calculation(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int
    ) -> float:
        """Generic cost calculation for unknown providers."""
        # Very rough estimates - should be configured per deployment
        generic_pricing = {
            # OpenAI-style models
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-3.5": {"input": 0.001 / 1000, "output": 0.002 / 1000},
            # Anthropic-style models
            "claude-3": {"input": 3.0 / 1000000, "output": 15.0 / 1000000},
            "claude-2": {"input": 8.0 / 1000000, "output": 24.0 / 1000000},
            # Default fallback
            "default": {"input": 0.001 / 1000, "output": 0.002 / 1000}
        }

        # Find matching pricing
        model_pricing = None
        for key, pricing in generic_pricing.items():
            if key.lower() in model.lower():
                model_pricing = pricing
                break

        if not model_pricing:
            model_pricing = generic_pricing["default"]

        input_cost = tokens_input * model_pricing["input"]
        output_cost = tokens_output * model_pricing["output"]

        return input_cost + output_cost

    def finalize_chain_tracking(self, chain_id: str, total_time: float = 0.0) -> ChainCostSummary | None:
        """
        Finalize cost tracking for a chain and return summary.

        Args:
            chain_id: Chain identifier
            total_time: Total time for the chain execution

        Returns:
            ChainCostSummary if chain was being tracked, None otherwise
        """
        if chain_id not in self.active_chains:
            logger.warning(f"Chain {chain_id} not found in active tracking")
            return None

        summary = self.active_chains.pop(chain_id)
        summary.total_time = total_time
        summary.total_cost = summary.calculate_total_cost()
        logger.debug(f"Finalized cost tracking for chain {chain_id}: ${summary.total_cost:.4f}")

        return summary

    def get_chain_summary(self, chain_id: str) -> ChainCostSummary | None:
        """Get current cost summary for an active chain."""
        return self.active_chains.get(chain_id)

    def get_active_chains(self) -> list[str]:
        """Get list of currently tracked chain IDs."""
        return list(self.active_chains.keys())

    def clear_all_tracking(self) -> None:
        """Clear all active chain tracking."""
        cleared_count = len(self.active_chains)
        self.active_chains.clear()
        logger.debug(f"Cleared {cleared_count} active chain trackings")


# Global cost aggregator instance
_cost_aggregator: LangChainCostAggregator | None = None


def get_cost_aggregator() -> LangChainCostAggregator:
    """Get the global LangChain cost aggregator instance."""
    global _cost_aggregator
    if _cost_aggregator is None:
        _cost_aggregator = LangChainCostAggregator()
    return _cost_aggregator


def create_chain_cost_context(chain_id: str) -> 'ChainCostContext':
    """Create a context manager for chain cost tracking."""
    return ChainCostContext(chain_id)


class ChainCostContext:
    """Context manager for chain cost tracking."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.aggregator = get_cost_aggregator()
        self.summary: ChainCostSummary | None = None
        self.start_time = None
        self.operation_id = None

    def __enter__(self) -> 'ChainCostContext':
        import time
        self.start_time = time.time()
        self.operation_id = self.chain_id  # Use chain_id as operation_id
        self.aggregator.start_chain_tracking(self.chain_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        total_time = time.time() - self.start_time if self.start_time else 0.0
        self.summary = self.aggregator.finalize_chain_tracking(self.chain_id, total_time)

    def add_llm_call(
        self,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        operation_name: str | None = None,
        **metadata
    ) -> LLMCallCost | None:
        """Add an LLM call cost within this context."""
        return self.aggregator.add_llm_call_cost(
            self.chain_id, provider, model, tokens_input, tokens_output,
            operation_name, **metadata
        )

    def get_current_summary(self) -> ChainCostSummary | None:
        """Get the current cost summary."""
        return self.aggregator.get_chain_summary(self.chain_id)

    def get_final_summary(self) -> ChainCostSummary | None:
        """Get the final cost summary (available after context exit)."""
        return self.summary

    def record_generation_cost(self, cost: float) -> None:
        """Record generation cost within this context."""
        # For now, just store it on the current summary
        current_summary = self.get_current_summary()
        if current_summary and hasattr(current_summary, 'generation_cost'):
            current_summary.generation_cost = cost
