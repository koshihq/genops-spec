"""Vercel AI SDK pricing and cost calculation module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Enum for different provider types supported by Vercel AI SDK."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    UNKNOWN = "unknown"


@dataclass
class ModelPricing:
    """Data class for model pricing information."""

    input_price_per_1k: Decimal
    output_price_per_1k: Decimal
    provider: str
    model_name: str
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    context_length: int = 4096


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a Vercel AI SDK request."""

    input_tokens: int
    output_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    provider: str
    model: str
    currency: str = "USD"
    pricing_source: str = "genops"
    estimated: bool = False


class VercelAISDKPricingCalculator:
    """
    Pricing calculator for Vercel AI SDK requests across multiple providers.

    Leverages existing GenOps provider pricing modules where available,
    and provides fallback pricing for unsupported providers.
    """

    # Default pricing for common models (per 1K tokens)
    DEFAULT_PRICING: dict[str, ModelPricing] = {
        # OpenAI models
        "gpt-4": ModelPricing(
            input_price_per_1k=Decimal("0.03"),
            output_price_per_1k=Decimal("0.06"),
            provider="openai",
            model_name="gpt-4",
            supports_tools=True,
            supports_vision=True,
            context_length=8192,
        ),
        "gpt-4-turbo": ModelPricing(
            input_price_per_1k=Decimal("0.01"),
            output_price_per_1k=Decimal("0.03"),
            provider="openai",
            model_name="gpt-4-turbo",
            supports_tools=True,
            supports_vision=True,
            context_length=128000,
        ),
        "gpt-3.5-turbo": ModelPricing(
            input_price_per_1k=Decimal("0.001"),
            output_price_per_1k=Decimal("0.002"),
            provider="openai",
            model_name="gpt-3.5-turbo",
            supports_tools=True,
            context_length=4096,
        ),
        # Anthropic models
        "claude-3-opus": ModelPricing(
            input_price_per_1k=Decimal("0.015"),
            output_price_per_1k=Decimal("0.075"),
            provider="anthropic",
            model_name="claude-3-opus",
            supports_tools=True,
            supports_vision=True,
            context_length=200000,
        ),
        "claude-3-sonnet": ModelPricing(
            input_price_per_1k=Decimal("0.003"),
            output_price_per_1k=Decimal("0.015"),
            provider="anthropic",
            model_name="claude-3-sonnet",
            supports_tools=True,
            supports_vision=True,
            context_length=200000,
        ),
        "claude-3-haiku": ModelPricing(
            input_price_per_1k=Decimal("0.00025"),
            output_price_per_1k=Decimal("0.00125"),
            provider="anthropic",
            model_name="claude-3-haiku",
            supports_tools=True,
            supports_vision=True,
            context_length=200000,
        ),
        # Google models
        "gemini-pro": ModelPricing(
            input_price_per_1k=Decimal("0.000125"),
            output_price_per_1k=Decimal("0.000375"),
            provider="google",
            model_name="gemini-pro",
            supports_tools=True,
            supports_vision=True,
            context_length=32768,
        ),
        "gemini-pro-vision": ModelPricing(
            input_price_per_1k=Decimal("0.00025"),
            output_price_per_1k=Decimal("0.00075"),
            provider="google",
            model_name="gemini-pro-vision",
            supports_tools=True,
            supports_vision=True,
            context_length=16384,
        ),
        # Cohere models
        "command": ModelPricing(
            input_price_per_1k=Decimal("0.0015"),
            output_price_per_1k=Decimal("0.002"),
            provider="cohere",
            model_name="command",
            context_length=4096,
        ),
        "command-nightly": ModelPricing(
            input_price_per_1k=Decimal("0.0015"),
            output_price_per_1k=Decimal("0.002"),
            provider="cohere",
            model_name="command-nightly",
            context_length=4096,
        ),
        # Mistral models
        "mistral-tiny": ModelPricing(
            input_price_per_1k=Decimal("0.00025"),
            output_price_per_1k=Decimal("0.00025"),
            provider="mistral",
            model_name="mistral-tiny",
            context_length=32000,
        ),
        "mistral-small": ModelPricing(
            input_price_per_1k=Decimal("0.002"),
            output_price_per_1k=Decimal("0.006"),
            provider="mistral",
            model_name="mistral-small",
            context_length=32000,
        ),
        "mistral-medium": ModelPricing(
            input_price_per_1k=Decimal("0.0027"),
            output_price_per_1k=Decimal("0.0081"),
            provider="mistral",
            model_name="mistral-medium",
            context_length=32000,
        ),
        # Generic fallbacks for unknown models
        "unknown-small": ModelPricing(
            input_price_per_1k=Decimal("0.001"),
            output_price_per_1k=Decimal("0.002"),
            provider="unknown",
            model_name="unknown-small",
            context_length=4096,
        ),
        "unknown-large": ModelPricing(
            input_price_per_1k=Decimal("0.01"),
            output_price_per_1k=Decimal("0.03"),
            provider="unknown",
            model_name="unknown-large",
            context_length=8192,
        ),
    }

    def __init__(self):
        """Initialize the pricing calculator."""
        self.provider_calculators = self._initialize_provider_calculators()

    def _initialize_provider_calculators(self) -> dict[str, Any]:
        """Initialize provider-specific cost calculators from existing GenOps modules."""
        calculators = {}

        # Try to import existing GenOps provider calculators
        providers_to_try = [
            ("openai", "genops.providers.openai"),
            ("anthropic", "genops.providers.anthropic"),
            ("google", "genops.providers.gemini"),
            ("cohere", "genops.providers.cohere"),
            ("mistral", "genops.providers.mistral"),
            ("replicate", "genops.providers.replicate"),
            ("huggingface", "genops.providers.huggingface"),
            ("perplexity", "genops.providers.perplexity"),
            ("fireworks", "genops.providers.fireworks"),
            ("together", "genops.providers.together"),
        ]

        for provider, module_name in providers_to_try:
            try:
                module = __import__(module_name, fromlist=["calculate_cost"])
                if hasattr(module, "calculate_cost"):
                    calculators[provider] = module.calculate_cost
                    logger.debug(f"Loaded cost calculator for {provider}")
            except ImportError:
                logger.debug(f"No cost calculator available for {provider}")
            except Exception as e:
                logger.warning(f"Error loading cost calculator for {provider}: {e}")

        return calculators

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation_type: str = "generateText",
    ) -> CostBreakdown:
        """
        Calculate the cost for a Vercel AI SDK request.

        Args:
            provider: AI provider (openai, anthropic, etc.)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation_type: Type of operation (generateText, embed, etc.)

        Returns:
            CostBreakdown: Detailed cost breakdown
        """
        # Normalize provider name
        provider = provider.lower()
        model_key = self._get_model_key(provider, model)

        # Try to use provider-specific calculator first
        if provider in self.provider_calculators:
            try:
                total_cost = self.provider_calculators[provider](
                    model, input_tokens, output_tokens
                )
                if total_cost is not None:
                    # Estimate input/output split (typically 25/75 for output-heavy operations)
                    total_tokens = input_tokens + output_tokens
                    if total_tokens > 0:
                        input_ratio = input_tokens / total_tokens
                        output_ratio = output_tokens / total_tokens
                        input_cost = total_cost * Decimal(str(input_ratio))
                        output_cost = total_cost * Decimal(str(output_ratio))
                    else:
                        input_cost = output_cost = Decimal("0")

                    return CostBreakdown(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        input_cost=input_cost,
                        output_cost=output_cost,
                        total_cost=total_cost,
                        provider=provider,
                        model=model,
                        pricing_source="genops_provider",
                        estimated=False,
                    )
            except Exception as e:
                logger.warning(f"Error using provider calculator for {provider}: {e}")

        # Fall back to default pricing
        return self._calculate_cost_with_defaults(
            provider, model, input_tokens, output_tokens, model_key
        )

    def _calculate_cost_with_defaults(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        model_key: str,
    ) -> CostBreakdown:
        """Calculate cost using default pricing information."""
        # Get pricing info
        pricing_info = self._get_pricing_info(model_key, provider, model)

        # Calculate costs
        input_cost = (
            Decimal(str(input_tokens)) / Decimal("1000")
        ) * pricing_info.input_price_per_1k
        output_cost = (
            Decimal(str(output_tokens)) / Decimal("1000")
        ) * pricing_info.output_price_per_1k
        total_cost = input_cost + output_cost

        return CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            provider=provider,
            model=model,
            pricing_source="default",
            estimated=True,
        )

    def _get_model_key(self, provider: str, model: str) -> str:
        """Get the model key for pricing lookup."""
        # Handle model strings that might include provider prefix
        if "/" in model:
            provider_prefix, model_name = model.split("/", 1)
            if not provider or provider == "unknown":
                provider = provider_prefix
            model = model_name

        # Try exact match first
        model_key = model
        if model_key in self.DEFAULT_PRICING:
            return model_key

        # Try provider-prefixed version
        model_key = f"{provider}-{model}"
        if model_key in self.DEFAULT_PRICING:
            return model_key

        # Try to match by model family
        model_lower = model.lower()
        for key in self.DEFAULT_PRICING.keys():
            if model_lower in key or key in model_lower:
                return key

        # Default to unknown models
        if (
            "gpt-4" in model_lower
            or "claude-3-opus" in model_lower
            or "large" in model_lower
        ):
            return "unknown-large"
        else:
            return "unknown-small"

    def _get_pricing_info(
        self, model_key: str, provider: str, model: str
    ) -> ModelPricing:
        """Get pricing information for a model."""
        if model_key in self.DEFAULT_PRICING:
            return self.DEFAULT_PRICING[model_key]

        # Return generic pricing based on model characteristics
        if "large" in model.lower() or "4" in model or "opus" in model.lower():
            pricing = self.DEFAULT_PRICING["unknown-large"]
        else:
            pricing = self.DEFAULT_PRICING["unknown-small"]

        # Update provider and model name
        return ModelPricing(
            input_price_per_1k=pricing.input_price_per_1k,
            output_price_per_1k=pricing.output_price_per_1k,
            provider=provider,
            model_name=model,
            supports_streaming=pricing.supports_streaming,
            supports_tools=pricing.supports_tools,
            supports_vision=pricing.supports_vision,
            context_length=pricing.context_length,
        )

    def get_model_info(self, provider: str, model: str) -> ModelPricing | None:
        """
        Get detailed information about a model.

        Args:
            provider: AI provider
            model: Model name

        Returns:
            ModelPricing: Model information or None if not found
        """
        model_key = self._get_model_key(provider, model)
        return self._get_pricing_info(model_key, provider, model)

    def estimate_cost(
        self,
        provider: str,
        model: str,
        prompt_length: int,
        expected_response_length: int = None,  # type: ignore[assignment]
    ) -> tuple[Decimal, Decimal]:
        """
        Estimate cost for a request before making it.

        Args:
            provider: AI provider
            model: Model name
            prompt_length: Estimated prompt length in characters
            expected_response_length: Expected response length in characters

        Returns:
            Tuple of (minimum_cost, maximum_cost)
        """
        # Rough character to token conversion (varies by model, ~4 chars per token average)
        chars_per_token = 4
        input_tokens = max(1, prompt_length // chars_per_token)

        # Estimate output tokens (default to reasonable response length)
        if expected_response_length is None:
            output_tokens = min(input_tokens * 2, 1000)  # Default response
        else:
            output_tokens = max(1, expected_response_length // chars_per_token)

        # Calculate minimum cost (exact estimate)
        min_breakdown = self.calculate_cost(
            provider, model, input_tokens, output_tokens
        )
        min_cost = min_breakdown.total_cost

        # Calculate maximum cost (with 50% buffer for uncertainty)
        max_input_tokens = int(input_tokens * 1.5)
        max_output_tokens = int(output_tokens * 1.5)
        max_breakdown = self.calculate_cost(
            provider, model, max_input_tokens, max_output_tokens
        )
        max_cost = max_breakdown.total_cost

        return min_cost, max_cost

    def get_supported_providers(self) -> dict[str, list[str]]:
        """Get list of supported providers and their models."""
        providers = {}

        for _model_key, pricing_info in self.DEFAULT_PRICING.items():
            provider = pricing_info.provider
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(pricing_info.model_name)

        return providers


# Global pricing calculator instance
pricing_calculator = VercelAISDKPricingCalculator()


# Convenience functions
def calculate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> CostBreakdown:
    """Calculate cost for a Vercel AI SDK request."""
    return pricing_calculator.calculate_cost(
        provider, model, input_tokens, output_tokens
    )


def estimate_cost(
    provider: str,
    model: str,
    prompt_length: int,
    response_length: int = None,  # type: ignore[assignment]
) -> tuple[Decimal, Decimal]:
    """Estimate cost for a request before making it."""
    return pricing_calculator.estimate_cost(
        provider, model, prompt_length, response_length
    )


def get_model_info(provider: str, model: str) -> ModelPricing | None:
    """Get detailed information about a model."""
    return pricing_calculator.get_model_info(provider, model)


def get_supported_providers() -> dict[str, list[str]]:
    """Get list of supported providers and their models."""
    return pricing_calculator.get_supported_providers()
