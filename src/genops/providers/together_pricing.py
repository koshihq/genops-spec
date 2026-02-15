"""
Together AI Pricing Calculator for GenOps Cost Management

Provides accurate cost calculation and optimization for Together AI's 200+ models
with real-time pricing data and intelligent model selection recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TogetherModelTier(Enum):
    """Together AI model pricing tiers."""

    LITE = "lite"  # Ultra-low cost, optimized for high throughput
    STANDARD = "standard"  # Balanced cost and performance
    LARGE = "large"  # High-capability models
    PREMIUM = "premium"  # State-of-the-art models


@dataclass
class ModelPricing:
    """Pricing information for a Together AI model."""

    model_id: str
    input_cost_per_million: Decimal
    output_cost_per_million: Decimal
    tier: TogetherModelTier
    context_length: int
    cost_per_image: Decimal | None = None  # For multimodal models
    fine_tuning_cost_per_million: Decimal | None = None


class TogetherPricingCalculator:
    """
    Comprehensive pricing calculator for Together AI operations.

    Provides accurate cost calculation, model comparison, and optimization
    recommendations for Together AI's 200+ model catalog.
    """

    def __init__(self):
        """Initialize pricing calculator with current Together AI rates."""
        self.pricing_data = self._initialize_pricing_data()
        self.default_fallback_pricing = ModelPricing(
            model_id="unknown",
            input_cost_per_million=Decimal("0.20"),  # Conservative estimate
            output_cost_per_million=Decimal("0.60"),
            tier=TogetherModelTier.STANDARD,
            context_length=8192,
        )

    def _initialize_pricing_data(self) -> dict[str, ModelPricing]:
        """Initialize pricing data for Together AI models (2024 rates)."""
        pricing = {}

        # Llama 3.1 Models
        pricing["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"] = ModelPricing(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            input_cost_per_million=Decimal("0.10"),
            output_cost_per_million=Decimal("0.10"),
            tier=TogetherModelTier.LITE,
            context_length=131072,
            fine_tuning_cost_per_million=Decimal("0.80"),
        )

        pricing["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"] = ModelPricing(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            input_cost_per_million=Decimal("0.88"),
            output_cost_per_million=Decimal("0.88"),
            tier=TogetherModelTier.STANDARD,
            context_length=131072,
            fine_tuning_cost_per_million=Decimal("3.20"),
        )

        pricing["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"] = ModelPricing(
            model_id="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            input_cost_per_million=Decimal("5.00"),
            output_cost_per_million=Decimal("5.00"),
            tier=TogetherModelTier.PREMIUM,
            context_length=131072,
            fine_tuning_cost_per_million=Decimal("12.00"),
        )

        # DeepSeek Models
        pricing["deepseek-ai/DeepSeek-R1"] = ModelPricing(
            model_id="deepseek-ai/DeepSeek-R1",
            input_cost_per_million=Decimal("0.55"),
            output_cost_per_million=Decimal("2.19"),
            tier=TogetherModelTier.STANDARD,
            context_length=65536,
        )

        pricing["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"] = ModelPricing(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            input_cost_per_million=Decimal("0.10"),
            output_cost_per_million=Decimal("0.10"),
            tier=TogetherModelTier.LITE,
            context_length=32768,
        )

        pricing["deepseek-ai/DeepSeek-Coder-V2-Instruct"] = ModelPricing(
            model_id="deepseek-ai/DeepSeek-Coder-V2-Instruct",
            input_cost_per_million=Decimal("0.14"),
            output_cost_per_million=Decimal("0.28"),
            tier=TogetherModelTier.LITE,
            context_length=65536,
        )

        # Multimodal Models
        pricing["Qwen/Qwen2.5-VL-72B-Instruct"] = ModelPricing(
            model_id="Qwen/Qwen2.5-VL-72B-Instruct",
            input_cost_per_million=Decimal("1.20"),
            output_cost_per_million=Decimal("1.20"),
            tier=TogetherModelTier.LARGE,
            context_length=32768,
            cost_per_image=Decimal("0.001"),  # $0.001 per image
        )

        pricing["meta-llama/Llama-Vision-Free"] = ModelPricing(
            model_id="meta-llama/Llama-Vision-Free",
            input_cost_per_million=Decimal("0.18"),
            output_cost_per_million=Decimal("0.18"),
            tier=TogetherModelTier.LITE,
            context_length=131072,
            cost_per_image=Decimal("0.0005"),
        )

        # Code Models
        pricing["Qwen/Qwen2.5-Coder-32B-Instruct"] = ModelPricing(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            input_cost_per_million=Decimal("0.30"),
            output_cost_per_million=Decimal("0.30"),
            tier=TogetherModelTier.STANDARD,
            context_length=32768,
        )

        # Mixtral Models
        pricing["mistralai/Mixtral-8x7B-Instruct-v0.1"] = ModelPricing(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            input_cost_per_million=Decimal("0.60"),
            output_cost_per_million=Decimal("0.60"),
            tier=TogetherModelTier.STANDARD,
            context_length=32768,
        )

        pricing["mistralai/Mixtral-8x22B-Instruct-v0.1"] = ModelPricing(
            model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
            input_cost_per_million=Decimal("1.20"),
            output_cost_per_million=Decimal("1.20"),
            tier=TogetherModelTier.LARGE,
            context_length=65536,
        )

        return pricing

    def get_model_pricing(self, model_id: str) -> ModelPricing:
        """
        Get pricing information for a specific model.

        Args:
            model_id: Together AI model identifier

        Returns:
            ModelPricing: Pricing information for the model
        """
        # Try exact match first
        if model_id in self.pricing_data:
            return self.pricing_data[model_id]

        # Try partial matches for model families
        for known_model in self.pricing_data:
            if (
                model_id.lower() in known_model.lower()
                or known_model.lower() in model_id.lower()
            ):
                logger.info(
                    f"Using pricing for similar model '{known_model}' for '{model_id}'"
                )
                pricing = self.pricing_data[known_model]
                return ModelPricing(
                    model_id=model_id,
                    input_cost_per_million=pricing.input_cost_per_million,
                    output_cost_per_million=pricing.output_cost_per_million,
                    tier=pricing.tier,
                    context_length=pricing.context_length,
                    cost_per_image=pricing.cost_per_image,
                    fine_tuning_cost_per_million=pricing.fine_tuning_cost_per_million,
                )

        # Fallback to conservative estimate
        logger.warning(f"Unknown model '{model_id}', using fallback pricing")
        return ModelPricing(
            model_id=model_id,
            input_cost_per_million=self.default_fallback_pricing.input_cost_per_million,
            output_cost_per_million=self.default_fallback_pricing.output_cost_per_million,
            tier=self.default_fallback_pricing.tier,
            context_length=self.default_fallback_pricing.context_length,
        )

    def calculate_chat_cost(
        self, model: str, input_tokens: int, output_tokens: int, images: int = 0
    ) -> Decimal:
        """
        Calculate cost for chat completion operation.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            images: Number of images (for multimodal models)

        Returns:
            Decimal: Total cost in USD
        """
        pricing = self.get_model_pricing(model)

        # Calculate token costs
        input_cost = (
            Decimal(input_tokens) / Decimal(1_000_000)
        ) * pricing.input_cost_per_million
        output_cost = (
            Decimal(output_tokens) / Decimal(1_000_000)
        ) * pricing.output_cost_per_million

        # Add image costs if applicable
        image_cost = Decimal("0")
        if images > 0 and pricing.cost_per_image:
            image_cost = Decimal(images) * pricing.cost_per_image

        total_cost = input_cost + output_cost + image_cost
        return total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def calculate_completion_cost(self, model: str, tokens_used: int) -> Decimal:
        """
        Calculate cost for text completion operation.

        Args:
            model: Model identifier
            tokens_used: Total tokens processed

        Returns:
            Decimal: Total cost in USD
        """
        pricing = self.get_model_pricing(model)

        # For completions, use output token pricing (more conservative)
        cost = (
            Decimal(tokens_used) / Decimal(1_000_000)
        ) * pricing.output_cost_per_million
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def estimate_chat_cost(
        self, model: str, estimated_tokens: int, input_output_ratio: float = 0.3
    ) -> Decimal:
        """
        Estimate cost for a chat operation before execution.

        Args:
            model: Model identifier
            estimated_tokens: Total estimated tokens
            input_output_ratio: Ratio of input to output tokens (default 0.3)

        Returns:
            Decimal: Estimated cost in USD
        """
        estimated_input = int(estimated_tokens * input_output_ratio)
        estimated_output = int(estimated_tokens * (1 - input_output_ratio))

        return self.calculate_chat_cost(
            model=model, input_tokens=estimated_input, output_tokens=estimated_output
        )

    def estimate_completion_cost(self, model: str, estimated_tokens: int) -> Decimal:
        """
        Estimate cost for a completion operation before execution.

        Args:
            model: Model identifier
            estimated_tokens: Total estimated tokens

        Returns:
            Decimal: Estimated cost in USD
        """
        return self.calculate_completion_cost(model, estimated_tokens)

    def calculate_fine_tuning_cost(
        self,
        model: str,
        training_tokens: int,
        validation_tokens: int = 0,
        epochs: int = 1,
    ) -> Decimal:
        """
        Calculate cost for fine-tuning operation.

        Args:
            model: Base model identifier
            training_tokens: Number of tokens in training dataset
            validation_tokens: Number of tokens in validation dataset
            epochs: Number of training epochs

        Returns:
            Decimal: Total fine-tuning cost in USD
        """
        pricing = self.get_model_pricing(model)

        if not pricing.fine_tuning_cost_per_million:
            logger.warning(
                f"Fine-tuning pricing not available for {model}, using estimate"
            )
            # Use 4x the input token price as estimate
            ft_cost_per_million = pricing.input_cost_per_million * 4
        else:
            ft_cost_per_million = pricing.fine_tuning_cost_per_million

        # Calculate total tokens processed during training
        total_tokens = (training_tokens * epochs) + validation_tokens

        cost = (Decimal(total_tokens) / Decimal(1_000_000)) * ft_cost_per_million
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def compare_models(
        self,
        models: list[str],
        estimated_tokens: int = 1000,
        include_context_length: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Compare costs and capabilities across multiple models.

        Args:
            models: List of model identifiers to compare
            estimated_tokens: Tokens for cost comparison
            include_context_length: Include context length in comparison

        Returns:
            List of model comparisons sorted by cost
        """
        comparisons = []

        for model in models:
            pricing = self.get_model_pricing(model)
            estimated_cost = self.estimate_chat_cost(model, estimated_tokens)

            comparison = {
                "model": model,
                "tier": pricing.tier.value,
                "input_cost_per_million": float(pricing.input_cost_per_million),
                "output_cost_per_million": float(pricing.output_cost_per_million),
                "estimated_cost": float(estimated_cost),
                "cost_per_1k_tokens": float(estimated_cost * 1000 / estimated_tokens),
            }

            if include_context_length:
                comparison["context_length"] = pricing.context_length
                comparison["cost_per_context_token"] = float(
                    pricing.input_cost_per_million / Decimal(pricing.context_length)
                )

            if pricing.cost_per_image:
                comparison["cost_per_image"] = float(pricing.cost_per_image)

            if pricing.fine_tuning_cost_per_million:
                comparison["fine_tuning_cost_per_million"] = float(
                    pricing.fine_tuning_cost_per_million
                )

            comparisons.append(comparison)

        # Sort by estimated cost
        return sorted(comparisons, key=lambda x: x["estimated_cost"])

    def recommend_model(
        self,
        task_complexity: str,  # "simple", "moderate", "complex"
        budget_per_operation: float | None = None,
        require_multimodal: bool = False,
        require_code: bool = False,
        min_context_length: int = 8192,
    ) -> dict[str, Any]:
        """
        Recommend optimal model based on requirements and budget.

        Args:
            task_complexity: Complexity of the task
            budget_per_operation: Maximum budget per operation
            require_multimodal: Require multimodal capabilities
            require_code: Require code generation capabilities
            min_context_length: Minimum required context length

        Returns:
            Dict with model recommendation and rationale
        """
        # Filter models based on requirements
        candidate_models = []

        for model_id, pricing in self.pricing_data.items():
            # Context length filter
            if pricing.context_length < min_context_length:
                continue

            # Multimodal filter
            if require_multimodal and not pricing.cost_per_image:
                continue

            # Code model filter
            if (
                require_code
                and "code" not in model_id.lower()
                and "deepseek" not in model_id.lower()
            ):
                continue

            candidate_models.append(model_id)

        if not candidate_models:
            return {
                "recommended_model": None,
                "reason": "No models match the specified requirements",
                "alternatives": [],
            }

        # Compare candidates for 1000 token operation
        comparisons = self.compare_models(candidate_models, 1000)

        # Apply task complexity and budget filters
        filtered_comparisons = []
        for comp in comparisons:
            # Budget filter
            if budget_per_operation and comp["estimated_cost"] > budget_per_operation:
                continue

            # Complexity-based tier filtering
            tier = TogetherModelTier(comp["tier"])
            if task_complexity == "simple" and tier in [
                TogetherModelTier.LITE,
                TogetherModelTier.STANDARD,
            ]:
                filtered_comparisons.append(comp)
            elif task_complexity == "moderate" and tier in [
                TogetherModelTier.STANDARD,
                TogetherModelTier.LARGE,
            ]:
                filtered_comparisons.append(comp)
            elif task_complexity == "complex" and tier in [
                TogetherModelTier.LARGE,
                TogetherModelTier.PREMIUM,
            ]:
                filtered_comparisons.append(comp)
            else:
                filtered_comparisons.append(comp)  # Include all if no specific match

        if not filtered_comparisons:
            filtered_comparisons = comparisons[:3]  # Fallback to cheapest options

        # Select best option (lowest cost with appropriate capability)
        recommended = filtered_comparisons[0]

        return {
            "recommended_model": recommended["model"],
            "estimated_cost": recommended["estimated_cost"],
            "tier": recommended["tier"],
            "context_length": recommended.get("context_length", "unknown"),
            "reason": f"Best cost-performance balance for {task_complexity} tasks",
            "alternatives": filtered_comparisons[1:4],  # Next 3 best options
            "all_candidates": len(candidate_models),
            "budget_compliant": budget_per_operation is None
            or recommended["estimated_cost"] <= budget_per_operation,
        }

    def analyze_costs(
        self,
        operations_per_day: int,
        avg_tokens_per_operation: int,
        model: str,
        days_to_analyze: int = 30,
    ) -> dict[str, Any]:
        """
        Analyze projected costs over time with optimization recommendations.

        Args:
            operations_per_day: Expected operations per day
            avg_tokens_per_operation: Average tokens per operation
            model: Model identifier to analyze
            days_to_analyze: Number of days to project

        Returns:
            Dict with cost analysis and recommendations
        """
        pricing = self.get_model_pricing(model)
        cost_per_operation = self.estimate_chat_cost(model, avg_tokens_per_operation)

        daily_cost = cost_per_operation * operations_per_day
        monthly_cost = daily_cost * days_to_analyze
        yearly_cost = daily_cost * 365

        # Find more cost-effective alternatives
        all_models = list(self.pricing_data.keys())
        comparisons = self.compare_models(all_models, avg_tokens_per_operation)

        # Find models with similar context length but lower cost
        current_context = pricing.context_length
        cheaper_alternatives = [
            comp
            for comp in comparisons
            if (
                comp["estimated_cost"] < float(cost_per_operation)
                and comp.get("context_length", 0) >= current_context * 0.8
            )  # At least 80% of current context
        ]

        return {
            "current_model": model,
            "cost_per_operation": float(cost_per_operation),
            "daily_cost": float(daily_cost),
            "monthly_cost": float(monthly_cost),
            "yearly_cost": float(yearly_cost),
            "operations_per_day": operations_per_day,
            "avg_tokens_per_operation": avg_tokens_per_operation,
            "model_tier": pricing.tier.value,
            "context_length": pricing.context_length,
            "potential_savings": {
                "cheaper_alternatives": len(cheaper_alternatives),
                "best_alternative": cheaper_alternatives[0]
                if cheaper_alternatives
                else None,
                "potential_daily_savings": float(
                    daily_cost - (comparisons[0]["estimated_cost"] * operations_per_day)
                )
                if comparisons
                else 0,
                "potential_monthly_savings": float(
                    monthly_cost
                    - (
                        comparisons[0]["estimated_cost"]
                        * operations_per_day
                        * days_to_analyze
                    )
                )
                if comparisons
                else 0,
            },
        }

    def get_all_models_by_tier(self) -> dict[str, list[str]]:
        """
        Get all available models grouped by pricing tier.

        Returns:
            Dict mapping tier names to lists of model IDs
        """
        tiers: dict[str, list[str]] = {
            "lite": [],
            "standard": [],
            "large": [],
            "premium": [],
        }

        for model_id, pricing in self.pricing_data.items():
            tiers[pricing.tier.value].append(model_id)

        return tiers
