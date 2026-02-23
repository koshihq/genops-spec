"""Multi-provider cost aggregation for Hugging Face operations."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceCallCost:
    """Represents cost information for a single Hugging Face operation call."""

    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    cost: float
    currency: str = "USD"
    task: str | None = None
    operation_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HuggingFaceCostSummary:
    """
    Framework-agnostic cost summary for Hugging Face operations.

    This follows the standardized cost structure specified in CLAUDE.md
    for consistency across all GenOps provider adapters.
    """

    total_cost: float = 0.0
    currency: str = "USD"
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    unique_providers: set[str] = field(default_factory=set)
    total_time: float = 0.0
    governance_attributes: dict[str, str] = field(default_factory=dict)

    # Hugging Face specific attributes
    hf_calls: list[HuggingFaceCallCost] = field(default_factory=list)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    unique_models: set[str] = field(default_factory=set)
    tasks_performed: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Calculate aggregated values after initialization."""
        self._calculate_aggregates()

    def _calculate_aggregates(self) -> None:
        """Calculate aggregate cost and token values."""
        self.cost_by_provider = defaultdict(float)
        self.cost_by_model = defaultdict(float)
        self.unique_providers = set()
        self.unique_models = set()
        self.tasks_performed = set()

        self.total_cost = 0.0
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        for call in self.hf_calls:
            # Aggregate costs
            self.cost_by_provider[call.provider] += call.cost
            self.cost_by_model[call.model] += call.cost
            self.total_cost += call.cost

            # Aggregate tokens
            self.total_tokens_input += call.tokens_input
            self.total_tokens_output += call.tokens_output

            # Track unique values
            self.unique_providers.add(call.provider)
            self.unique_models.add(call.model)
            if call.task:
                self.tasks_performed.add(call.task)

        # Convert defaultdict to regular dict for serialization
        self.cost_by_provider = dict(self.cost_by_provider)
        self.cost_by_model = dict(self.cost_by_model)

    def add_call(self, call: HuggingFaceCallCost) -> None:
        """Add a new call and recalculate aggregates."""
        self.hf_calls.append(call)
        self._calculate_aggregates()

    def calculate_total_cost(self) -> float:
        """Calculate total cost across all calls."""
        return sum(call.cost for call in self.hf_calls)

    def get_provider_breakdown(self) -> dict[str, dict]:
        """Get detailed breakdown by provider."""
        breakdown = {}
        for provider in self.unique_providers:
            provider_calls = [
                call for call in self.hf_calls if call.provider == provider
            ]
            breakdown[provider] = {
                "cost": self.cost_by_provider[provider],
                "calls": len(provider_calls),
                "tokens_input": sum(call.tokens_input for call in provider_calls),
                "tokens_output": sum(call.tokens_output for call in provider_calls),
                "models_used": list({call.model for call in provider_calls}),
            }
        return breakdown

    def get_model_breakdown(self) -> dict[str, dict]:
        """Get detailed breakdown by model."""
        breakdown = {}
        for model in self.unique_models:
            model_calls = [call for call in self.hf_calls if call.model == model]
            breakdown[model] = {
                "cost": self.cost_by_model[model],
                "calls": len(model_calls),
                "tokens_input": sum(call.tokens_input for call in model_calls),
                "tokens_output": sum(call.tokens_output for call in model_calls),
                "provider": model_calls[0].provider if model_calls else "unknown",
            }
        return breakdown

    def get_task_breakdown(self) -> dict[str, dict]:
        """Get detailed breakdown by task type."""
        breakdown = {}
        for task in self.tasks_performed:
            task_calls = [call for call in self.hf_calls if call.task == task]
            total_cost = sum(call.cost for call in task_calls)
            breakdown[task] = {
                "cost": total_cost,
                "calls": len(task_calls),
                "tokens_input": sum(call.tokens_input for call in task_calls),
                "tokens_output": sum(call.tokens_output for call in task_calls),
                "models_used": list({call.model for call in task_calls}),
            }
        return breakdown


class HuggingFaceCostAggregator:
    """
    Aggregates costs across multiple Hugging Face operations and providers.

    This follows the exact same pattern as LangChain's cost aggregator
    to maintain consistency across GenOps provider adapters.
    """

    def __init__(self):
        self.active_operations: dict[str, HuggingFaceCostSummary] = {}
        self.provider_cost_calculators = {}
        self._setup_provider_calculators()

    def _setup_provider_calculators(self) -> None:
        """Setup cost calculators for different providers."""
        try:
            from genops.providers.huggingface_pricing import calculate_huggingface_cost

            self.calculate_cost_func = calculate_huggingface_cost
        except ImportError:
            logger.debug(
                "Hugging Face pricing module not available, using fallback cost calculation"
            )
            self.calculate_cost_func = self._fallback_cost_calculation  # type: ignore[assignment]

    def _fallback_cost_calculation(self, **kwargs) -> float:
        """Fallback cost calculation when pricing module is unavailable."""
        provider = kwargs.get("provider", "huggingface_hub")
        tokens_input = kwargs.get("input_tokens", 0)
        tokens_output = kwargs.get("output_tokens", 0)

        # Basic fallback pricing
        generic_pricing = {
            "openai": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
            "anthropic": {"input": 3.0 / 1000000, "output": 15.0 / 1000000},
            "huggingface_hub": {"input": 0.00005 / 1000, "output": 0.0001 / 1000},
            "cohere": {"input": 0.001 / 1000, "output": 0.002 / 1000},
            "meta": {"input": 0.0002 / 1000, "output": 0.0002 / 1000},
            "mistral": {"input": 0.0004 / 1000, "output": 0.0004 / 1000},
            "google": {"input": 0.0001 / 1000, "output": 0.0003 / 1000},
        }

        pricing = generic_pricing.get(provider, generic_pricing["huggingface_hub"])
        input_cost = tokens_input * pricing["input"]
        output_cost = tokens_output * pricing["output"]

        return input_cost + output_cost

    def start_operation_tracking(
        self,
        operation_id: str,
        governance_attributes: dict[str, str] = None,  # type: ignore[assignment]
    ) -> None:
        """Start tracking costs for a Hugging Face operation."""
        summary = HuggingFaceCostSummary()
        if governance_attributes:
            summary.governance_attributes = governance_attributes.copy()

        self.active_operations[operation_id] = summary
        logger.debug(
            f"Started cost tracking for Hugging Face operation: {operation_id}"
        )

    def add_operation_call_cost(
        self,
        operation_id: str,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        task: str = None,  # type: ignore[assignment]
        operation_name: str | None = None,
        **metadata,
    ) -> HuggingFaceCallCost | None:
        """
        Add a Hugging Face call cost to an operation's tracking.

        Args:
            operation_id: Unique identifier for the operation
            provider: Provider name (openai, anthropic, huggingface_hub, etc.)
            model: Model name
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
            task: Task type (text-generation, feature-extraction, etc.)
            operation_name: Name of the specific operation
            **metadata: Additional metadata

        Returns:
            HuggingFaceCallCost object if successful, None otherwise
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active tracking")
            return None

        # Calculate cost using provider-specific logic
        try:
            cost = self.calculate_cost_func(
                provider=provider,
                model=model,
                input_tokens=tokens_input,
                output_tokens=tokens_output,
                task=task or "text-generation",
            )
        except Exception as e:
            logger.warning(f"Cost calculation failed for {provider}/{model}: {e}")
            cost = self._fallback_cost_calculation(
                provider=provider,
                input_tokens=tokens_input,
                output_tokens=tokens_output,
            )

        # Create call cost object
        call_cost = HuggingFaceCallCost(
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            task=task,
            operation_name=operation_name,
            metadata=metadata,
        )

        # Add to operation tracking
        self.active_operations[operation_id].add_call(call_cost)
        logger.debug(
            f"Added call cost to operation {operation_id}: ${cost:.4f} ({provider}/{model})"
        )

        return call_cost

    def finalize_operation_tracking(
        self, operation_id: str, total_time: float = 0.0
    ) -> HuggingFaceCostSummary | None:
        """
        Finalize cost tracking for an operation and return summary.

        Args:
            operation_id: Operation identifier
            total_time: Total time for the operation execution

        Returns:
            HuggingFaceCostSummary if operation was being tracked, None otherwise
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active tracking")
            return None

        summary = self.active_operations.pop(operation_id)
        summary.total_time = total_time
        summary.total_cost = summary.calculate_total_cost()
        logger.debug(
            f"Finalized cost tracking for operation {operation_id}: ${summary.total_cost:.4f}"
        )

        return summary

    def get_operation_summary(self, operation_id: str) -> HuggingFaceCostSummary | None:
        """Get current cost summary for an active operation."""
        return self.active_operations.get(operation_id)

    def get_active_operations(self) -> list[str]:
        """Get list of currently tracked operation IDs."""
        return list(self.active_operations.keys())

    def clear_all_tracking(self) -> None:
        """Clear all active operation tracking."""
        cleared_count = len(self.active_operations)
        self.active_operations.clear()
        logger.debug(f"Cleared {cleared_count} active Hugging Face operation trackings")


# Global cost aggregator instance
_cost_aggregator: HuggingFaceCostAggregator | None = None


def get_cost_aggregator() -> HuggingFaceCostAggregator:
    """Get the global Hugging Face cost aggregator instance."""
    global _cost_aggregator
    if _cost_aggregator is None:
        _cost_aggregator = HuggingFaceCostAggregator()
    return _cost_aggregator


def create_huggingface_cost_context(operation_id: str) -> "HuggingFaceCostContext":
    """
    Create a context manager for Hugging Face cost tracking.

    This follows the exact pattern specified in CLAUDE.md:

    with create_huggingface_cost_context("operation_id") as context:
        # Multiple providers automatically aggregated
        result1 = provider1_operation()
        result2 = provider2_operation()
        summary = context.get_final_summary()
    """
    return HuggingFaceCostContext(operation_id)


class HuggingFaceCostContext:
    """
    Context manager for Hugging Face cost tracking.

    This enables the standardized multi-provider cost aggregation pattern
    specified in CLAUDE.md for all GenOps framework adapters.
    """

    def __init__(self, operation_id: str, governance_attributes: dict[str, str] = None):  # type: ignore[assignment]
        self.operation_id = operation_id
        self.governance_attributes = governance_attributes or {}
        self.aggregator = get_cost_aggregator()
        self.summary: HuggingFaceCostSummary | None = None
        self.start_time = None

    def __enter__(self) -> "HuggingFaceCostContext":
        import time

        self.start_time = time.time()  # type: ignore[assignment]
        self.aggregator.start_operation_tracking(
            self.operation_id, self.governance_attributes
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time

        total_time = time.time() - self.start_time if self.start_time else 0.0
        self.summary = self.aggregator.finalize_operation_tracking(
            self.operation_id, total_time
        )

    def add_hf_call(
        self,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        task: str = None,  # type: ignore[assignment]
        operation_name: str | None = None,
        **metadata,
    ) -> HuggingFaceCallCost | None:
        """Add a Hugging Face call cost within this context."""
        return self.aggregator.add_operation_call_cost(
            self.operation_id,
            provider,
            model,
            tokens_input,
            tokens_output,
            task,
            operation_name,
            **metadata,
        )

    def get_current_summary(self) -> HuggingFaceCostSummary | None:
        """Get the current cost summary."""
        return self.aggregator.get_operation_summary(self.operation_id)

    def get_final_summary(self) -> HuggingFaceCostSummary | None:
        """Get the final cost summary (available after context exit)."""
        return self.summary

    def record_operation_cost(self, cost: float, provider: str = "manual") -> None:
        """Record additional operation cost within this context."""
        # Create a manual call entry for additional costs
        self.add_hf_call(
            provider=provider,
            model="manual_cost_entry",
            tokens_input=0,
            tokens_output=0,
            operation_name="manual_cost",
            manual_cost=cost,
        )
