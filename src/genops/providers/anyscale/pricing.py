#!/usr/bin/env python3
"""
GenOps Anyscale Endpoints Pricing

Comprehensive pricing for Anyscale managed LLM endpoints including chat completion
and embedding models. Based on official Anyscale Endpoints pricing as of January 2026.

Features:
- Official Anyscale Endpoints pricing database
- Token-based cost calculation
- Model alias resolution (handles various model name formats)
- Fallback pricing estimation for new/unknown models
- Cost optimization recommendations

Usage:
    from genops.providers.anyscale.pricing import AnyscalePricing, calculate_completion_cost

    cost = calculate_completion_cost(
        model="meta-llama/Llama-2-70b-chat-hf",
        input_tokens=100,
        output_tokens=50
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
    """Pricing information for an Anyscale model."""

    model_name: str
    input_cost_per_million: float  # Cost per 1M input tokens in USD
    output_cost_per_million: float  # Cost per 1M output tokens in USD
    currency: str = "USD"
    category: str = "chat"  # 'chat' or 'embedding'
    context_window: Optional[int] = None
    notes: Optional[str] = None

    @property
    def input_cost_per_1k(self) -> float:
        """Cost per 1K input tokens."""
        return self.input_cost_per_million / 1000

    @property
    def output_cost_per_1k(self) -> float:
        """Cost per 1K output tokens."""
        return self.output_cost_per_million / 1000


# Official Anyscale Endpoints Pricing (as of January 2026)
# Source: https://www.anyscale.com/endpoints pricing page
ANYSCALE_PRICING: dict[str, ModelPricing] = {
    # Meta Llama Models
    "meta-llama/Llama-2-70b-chat-hf": ModelPricing(
        model_name="meta-llama/Llama-2-70b-chat-hf",
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
        context_window=4096,
        notes="50% lower than GPT-3.5 Turbo",
    ),
    "meta-llama/Llama-2-13b-chat-hf": ModelPricing(
        model_name="meta-llama/Llama-2-13b-chat-hf",
        input_cost_per_million=0.25,
        output_cost_per_million=0.25,
        context_window=4096,
        notes="Cost-effective for smaller tasks",
    ),
    "meta-llama/Llama-2-7b-chat-hf": ModelPricing(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        input_cost_per_million=0.15,
        output_cost_per_million=0.15,
        context_window=4096,
        notes="Optimized for speed and cost",
    ),
    # Meta Llama 3 Models (newer generation)
    "meta-llama/Meta-Llama-3-70B-Instruct": ModelPricing(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
        context_window=8192,
        notes="Improved context and capabilities vs Llama-2",
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": ModelPricing(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        input_cost_per_million=0.15,
        output_cost_per_million=0.15,
        context_window=8192,
        notes="Llama 3 efficiency at lower cost",
    ),
    # Mistral AI Models
    "mistralai/Mistral-7B-Instruct-v0.1": ModelPricing(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        input_cost_per_million=0.15,
        output_cost_per_million=0.15,
        context_window=8192,
        notes="European AI provider with strong performance",
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        input_cost_per_million=0.50,
        output_cost_per_million=0.50,
        context_window=32768,
        notes="Mixture of experts with large context window",
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelPricing(
        model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        input_cost_per_million=0.90,
        output_cost_per_million=0.90,
        context_window=65536,
        notes="Large mixture of experts model",
    ),
    # Embedding Models
    "thenlper/gte-large": ModelPricing(
        model_name="thenlper/gte-large",
        input_cost_per_million=0.05,
        output_cost_per_million=0.00,  # No output tokens for embeddings
        category="embedding",
        notes="High-quality embeddings at low cost",
    ),
    "BAAI/bge-large-en-v1.5": ModelPricing(
        model_name="BAAI/bge-large-en-v1.5",
        input_cost_per_million=0.05,
        output_cost_per_million=0.00,
        category="embedding",
        notes="State-of-the-art English embeddings",
    ),
    # Code Models
    "codellama/CodeLlama-34b-Instruct-hf": ModelPricing(
        model_name="codellama/CodeLlama-34b-Instruct-hf",
        input_cost_per_million=0.75,
        output_cost_per_million=0.75,
        context_window=16384,
        notes="Specialized for code generation",
    ),
    "codellama/CodeLlama-70b-Instruct-hf": ModelPricing(
        model_name="codellama/CodeLlama-70b-Instruct-hf",
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
        context_window=16384,
        notes="Large code model for complex tasks",
    ),
}

# Model aliases for flexible name matching
MODEL_ALIASES: dict[str, str] = {
    # Llama 2 aliases
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    # Llama 3 aliases
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    # Mistral aliases
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    # CodeLlama aliases
    "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    "codellama-70b": "codellama/CodeLlama-70b-Instruct-hf",
    # Embedding aliases
    "gte-large": "thenlper/gte-large",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


class AnyscalePricing:
    """Anyscale pricing calculator with fallback estimation."""

    def __init__(self):
        """Initialize pricing calculator."""
        self.pricing_db = ANYSCALE_PRICING
        self.aliases = MODEL_ALIASES

    def resolve_model_name(self, model: str) -> str:
        """
        Resolve model name using aliases.

        Args:
            model: Model name (may be alias or full name)

        Returns:
            Canonical model name
        """
        # Try exact match first
        if model in self.pricing_db:
            return model

        # Try aliases
        if model in self.aliases:
            return self.aliases[model]

        # Try case-insensitive alias match
        model_lower = model.lower()
        for alias, canonical in self.aliases.items():
            if model_lower == alias.lower():
                return canonical

        # Return original if no match found
        return model

    def get_model_pricing(self, model: str) -> Optional[ModelPricing]:
        """
        Get pricing for a specific model.

        Args:
            model: Model name

        Returns:
            ModelPricing if found, None otherwise
        """
        canonical_name = self.resolve_model_name(model)
        return self.pricing_db.get(canonical_name)

    def get_fallback_pricing(self, model: str) -> ModelPricing:
        """
        Get fallback pricing estimate for unknown models.

        Args:
            model: Model name

        Returns:
            Estimated ModelPricing
        """
        model_lower = model.lower()

        # Estimate based on model size/type
        if any(term in model_lower for term in ["70b", "large", "xl"]):
            input_cost = 1.00
            output_cost = 1.00
            notes = "Estimated pricing for large model (~70B parameters)"
        elif any(term in model_lower for term in ["13b", "34b", "medium"]):
            input_cost = 0.50
            output_cost = 0.50
            notes = "Estimated pricing for medium model (~13-34B parameters)"
        elif any(term in model_lower for term in ["7b", "8b", "small"]):
            input_cost = 0.15
            output_cost = 0.15
            notes = "Estimated pricing for small model (~7-8B parameters)"
        elif "embed" in model_lower:
            input_cost = 0.05
            output_cost = 0.00
            notes = "Estimated pricing for embedding model"
        else:
            # Default to medium model pricing
            input_cost = 0.50
            output_cost = 0.50
            notes = "Default estimated pricing (unknown model size)"

        logger.warning(
            f"Model '{model}' not in pricing database. Using fallback estimate: "
            f"${input_cost}/M input, ${output_cost}/M output. "
            f"Actual costs may differ."
        )

        return ModelPricing(
            model_name=model,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
            notes=notes,
        )

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Calculate total cost for a completion.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        pricing = self.get_model_pricing(model)
        if not pricing:
            pricing = self.get_fallback_pricing(model)

        input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_million

        return input_cost + output_cost

    def get_optimization_suggestions(
        self, model: str, input_tokens: int, output_tokens: int, cost: float
    ) -> list[str]:
        """
        Get cost optimization suggestions.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Calculated cost

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Large prompt suggestions
        if input_tokens > 2000:
            suggestions.append(
                "Consider breaking large prompts into smaller chunks or using prompt compression"
            )

        # Large output suggestions
        if output_tokens > 1000:
            suggestions.append(
                "Use max_tokens parameter to limit response length if full output not needed"
            )

        # High cost suggestions
        if cost > 0.01:
            suggestions.append(
                f"High cost operation (${cost:.4f}). Consider using a smaller model if appropriate"
            )

        # Model-specific suggestions
        pricing = self.get_model_pricing(model)
        if pricing and "70b" in model.lower():
            # Check if smaller model might work
            suggestions.append(
                "Consider using Llama-2-13b or Llama-2-7b for simpler tasks (3-7x cost reduction)"
            )

        return suggestions

    def get_model_alternatives(self, model: str) -> list[tuple[str, float, str]]:
        """
        Get alternative models for cost optimization.

        Args:
            model: Current model name

        Returns:
            List of (model_name, cost_ratio, description) tuples
        """
        alternatives = []
        current_pricing = self.get_model_pricing(model)

        if not current_pricing:
            return alternatives

        current_avg_cost = (
            current_pricing.input_cost_per_million
            + current_pricing.output_cost_per_million
        ) / 2

        # Find cheaper alternatives in same category
        for alt_model, alt_pricing in self.pricing_db.items():
            if alt_model == current_pricing.model_name:
                continue

            if alt_pricing.category != current_pricing.category:
                continue

            alt_avg_cost = (
                alt_pricing.input_cost_per_million + alt_pricing.output_cost_per_million
            ) / 2

            if alt_avg_cost < current_avg_cost:
                cost_ratio = alt_avg_cost / current_avg_cost
                savings_pct = int((1 - cost_ratio) * 100)
                description = f"~{savings_pct}% cost reduction"

                alternatives.append((alt_model, cost_ratio, description))

        # Sort by cost (cheapest first)
        alternatives.sort(key=lambda x: x[1])

        return alternatives[:3]  # Return top 3


# Convenience functions for direct use
_pricing_calculator = AnyscalePricing()


def calculate_completion_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float:
    """
    Calculate cost for a chat completion.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    return _pricing_calculator.calculate_cost(model, input_tokens, output_tokens)


def calculate_embedding_cost(model: str, tokens: int) -> float:
    """
    Calculate cost for embeddings.

    Args:
        model: Embedding model name
        tokens: Number of tokens

    Returns:
        Total cost in USD
    """
    return _pricing_calculator.calculate_cost(model, tokens, 0)


def get_model_pricing(model: str) -> Optional[ModelPricing]:
    """
    Get pricing information for a model.

    Args:
        model: Model name

    Returns:
        ModelPricing if found, None otherwise
    """
    return _pricing_calculator.get_model_pricing(model)


# Export public API
__all__ = [
    "ModelPricing",
    "AnyscalePricing",
    "ANYSCALE_PRICING",
    "MODEL_ALIASES",
    "calculate_completion_cost",
    "calculate_embedding_cost",
    "get_model_pricing",
]
