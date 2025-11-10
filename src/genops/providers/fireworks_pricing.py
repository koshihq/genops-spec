"""
Fireworks AI Pricing Calculator for GenOps Cost Tracking

Provides accurate cost calculations for Fireworks AI's 100+ models across all modalities,
with parameter-based pricing tiers and intelligent cost optimization features.
"""

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FireworksPricingTier(Enum):
    """Fireworks AI pricing tiers based on model parameters."""
    TINY = "tiny"              # < 1B parameters
    SMALL = "small"            # 1B - 4B parameters
    MEDIUM = "medium"          # 4B - 16B parameters
    LARGE = "large"            # 16B+ parameters
    MIXTURE_OF_EXPERTS = "moe" # MoE models
    SPECIALIZED = "specialized" # Custom pricing models


@dataclass
class ModelInfo:
    """Information about a Fireworks AI model."""
    name: str
    parameters: str
    pricing_tier: FireworksPricingTier
    cost_per_million_tokens: Decimal
    context_length: int
    modalities: List[str]
    specialized_pricing: Optional[Dict[str, Decimal]] = None
    batch_discount: float = 0.5  # 50% discount for batch inference


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an operation."""
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    pricing_tier: str
    batch_discount_applied: bool = False


@dataclass
class ModelRecommendation:
    """Model recommendation with cost analysis."""
    recommended_model: str
    estimated_cost: Decimal
    reasoning: str
    alternatives: List[Dict[str, Any]]
    cost_comparison: Dict[str, Decimal]


class FireworksPricingCalculator:
    """
    Comprehensive pricing calculator for Fireworks AI models with intelligent
    cost optimization and multi-model comparison capabilities.
    """

    def __init__(self):
        """Initialize the pricing calculator with current Fireworks AI pricing."""
        self.model_catalog = self._initialize_model_catalog()
        self.default_batch_discount = 0.5  # 50% discount for batch inference

    def _initialize_model_catalog(self) -> Dict[str, ModelInfo]:
        """Initialize comprehensive model catalog with current Fireworks pricing."""
        catalog = {}

        # Tiny Models (< 1B parameters) - $0.10 per 1M tokens
        tiny_models = [
            ("accounts/fireworks/models/llama-v3p2-1b-instruct", "1B", 128000),
        ]

        for model, params, context in tiny_models:
            catalog[model] = ModelInfo(
                name=model,
                parameters=params,
                pricing_tier=FireworksPricingTier.TINY,
                cost_per_million_tokens=Decimal("0.10"),
                context_length=context,
                modalities=["text"]
            )

        # Small Models (1B - 4B parameters) - $0.20 per 1M tokens
        small_models = [
            ("accounts/fireworks/models/llama-v3p2-3b-instruct", "3B", 128000),
            ("accounts/deepseek-ai/models/deepseek-coder-v2-lite-instruct", "16B", 65536),
        ]

        for model, params, context in small_models:
            catalog[model] = ModelInfo(
                name=model,
                parameters=params,
                pricing_tier=FireworksPricingTier.SMALL,
                cost_per_million_tokens=Decimal("0.20"),
                context_length=context,
                modalities=["text"]
            )

        # Medium Models (4B - 16B parameters) - $0.20 per 1M tokens
        medium_models = [
            ("accounts/fireworks/models/llama-v3p1-8b-instruct", "8B", 128000),
            ("accounts/fireworks/models/llama-v3p2-11b-vision-instruct", "11B", 32768),
            ("accounts/qwen/models/qwen2p5-coder-32b-instruct", "32B", 32768),
        ]

        for model, params, context in medium_models:
            catalog[model] = ModelInfo(
                name=model,
                parameters=params,
                pricing_tier=FireworksPricingTier.MEDIUM,
                cost_per_million_tokens=Decimal("0.20"),
                context_length=context,
                modalities=["text"] if "vision" not in model else ["text", "image"]
            )

        # Large Models (16B+ parameters) - $0.90 per 1M tokens
        large_models = [
            ("accounts/fireworks/models/llama-v3p1-70b-instruct", "70B", 128000),
            ("accounts/qwen/models/qwen2-vl-72b-instruct", "72B", 32768),
            ("accounts/codellama/models/codellama-70b-instruct", "70B", 4096),
        ]

        for model, params, context in large_models:
            catalog[model] = ModelInfo(
                name=model,
                parameters=params,
                pricing_tier=FireworksPricingTier.LARGE,
                cost_per_million_tokens=Decimal("0.90"),
                context_length=context,
                modalities=["text"] if "vl" not in model else ["text", "image"]
            )

        # Mixture of Experts Models - Variable pricing ($0.50 - $1.20 per 1M tokens)
        moe_models = [
            ("accounts/fireworks/models/mixtral-8x7b-instruct", "8x7B", Decimal("0.50"), 32768),
            ("accounts/fireworks/models/mixtral-8x22b-instruct", "8x22B", Decimal("1.20"), 65536),
        ]

        for model, params, cost, context in moe_models:
            catalog[model] = ModelInfo(
                name=model,
                parameters=params,
                pricing_tier=FireworksPricingTier.MIXTURE_OF_EXPERTS,
                cost_per_million_tokens=cost,
                context_length=context,
                modalities=["text"]
            )

        # Specialized Models with Custom Pricing
        specialized_models = [
            # Llama 405B - Premium pricing
            ("accounts/fireworks/models/llama-v3p1-405b-instruct", "405B", Decimal("3.00"), 128000, ["text"]),

            # DeepSeek R1 - Input/Output differentiated pricing
            ("accounts/deepseek-ai/models/deepseek-r1", "70B", None, 32768, ["text"], {
                "input": Decimal("1.35"),
                "output": Decimal("5.40")
            }),
            ("accounts/deepseek-ai/models/deepseek-r1-distill-llama-70b", "70B", None, 32768, ["text"], {
                "input": Decimal("0.14"),
                "output": Decimal("0.56")
            }),

            # Multimodal Models
            ("accounts/mistral/models/pixtral-12b-2409", "12B", Decimal("0.15"), 128000, ["text", "image"]),

            # Embedding Models - Lower cost
            ("accounts/fireworks/models/nomic-embed-text-v1p5", "137M", Decimal("0.02"), 8192, ["text"]),
            ("accounts/fireworks/models/bge-base-en-v1p5", "109M", Decimal("0.02"), 512, ["text"]),

            # Audio Models
            ("accounts/fireworks/models/whisper-v3", "1.5B", Decimal("0.006"), None, ["audio"]),  # per minute
        ]

        for model_data in specialized_models:
            model, params, base_cost, context, modalities = model_data[:5]
            specialized_pricing = model_data[5] if len(model_data) > 5 else None

            catalog[model] = ModelInfo(
                name=model,
                parameters=params,
                pricing_tier=FireworksPricingTier.SPECIALIZED,
                cost_per_million_tokens=base_cost or Decimal("0.00"),
                context_length=context or 0,
                modalities=modalities,
                specialized_pricing=specialized_pricing
            )

        return catalog

    def estimate_chat_cost(
        self,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        tokens: Optional[int] = None,
        is_batch: bool = False,
        **kwargs
    ) -> Decimal:
        """
        Estimate cost for chat completion.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens  
            tokens: Total tokens (if input/output not differentiated)
            is_batch: Whether this is batch inference (50% discount)
            **kwargs: Additional parameters
        
        Returns:
            Estimated cost in USD
        """
        if model not in self.model_catalog:
            logger.warning(f"Model {model} not in catalog, using default pricing")
            return self._estimate_unknown_model_cost(tokens or (input_tokens or 0) + (output_tokens or 0))

        model_info = self.model_catalog[model]

        # Handle specialized pricing (e.g., DeepSeek R1 with input/output rates)
        if model_info.specialized_pricing:
            return self._calculate_specialized_cost(model_info, input_tokens, output_tokens, tokens, is_batch)

        # Standard token-based pricing
        total_tokens = tokens or ((input_tokens or 0) + (output_tokens or 0))

        if total_tokens == 0:
            return Decimal("0.00")

        # Calculate base cost
        cost = (Decimal(str(total_tokens)) / Decimal("1000000")) * model_info.cost_per_million_tokens

        # Apply batch discount if applicable
        if is_batch:
            cost *= Decimal(str(1 - self.default_batch_discount))

        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def _calculate_specialized_cost(
        self,
        model_info: ModelInfo,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        tokens: Optional[int],
        is_batch: bool
    ) -> Decimal:
        """Calculate cost for models with specialized pricing."""
        if not model_info.specialized_pricing:
            return Decimal("0.00")

        total_cost = Decimal("0.00")

        # Handle input/output differentiated pricing
        if "input" in model_info.specialized_pricing and "output" in model_info.specialized_pricing:
            if input_tokens:
                input_cost = (Decimal(str(input_tokens)) / Decimal("1000000")) * model_info.specialized_pricing["input"]
                total_cost += input_cost

            if output_tokens:
                output_cost = (Decimal(str(output_tokens)) / Decimal("1000000")) * model_info.specialized_pricing["output"]
                total_cost += output_cost

            # If only total tokens provided, assume 50/50 split
            if tokens and not (input_tokens and output_tokens):
                half_tokens = tokens // 2
                input_cost = (Decimal(str(half_tokens)) / Decimal("1000000")) * model_info.specialized_pricing["input"]
                output_cost = (Decimal(str(tokens - half_tokens)) / Decimal("1000000")) * model_info.specialized_pricing["output"]
                total_cost = input_cost + output_cost

        # Apply batch discount
        if is_batch:
            total_cost *= Decimal(str(1 - self.default_batch_discount))

        return total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def estimate_embedding_cost(
        self,
        model: str,
        tokens: int,
        **kwargs
    ) -> Decimal:
        """
        Estimate cost for embedding generation.
        
        Args:
            model: Embedding model name
            tokens: Number of input tokens
            **kwargs: Additional parameters
        
        Returns:
            Estimated cost in USD
        """
        if model not in self.model_catalog:
            logger.warning(f"Embedding model {model} not in catalog, using default pricing")
            return Decimal(str(tokens)) * Decimal("0.00002")  # Default $0.02 per 1M tokens

        model_info = self.model_catalog[model]

        if tokens == 0:
            return Decimal("0.00")

        cost = (Decimal(str(tokens)) / Decimal("1000000")) * model_info.cost_per_million_tokens
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def estimate_audio_cost(
        self,
        model: str,
        duration_minutes: float,
        **kwargs
    ) -> Decimal:
        """
        Estimate cost for audio processing (e.g., Whisper transcription).
        
        Args:
            model: Audio model name  
            duration_minutes: Audio duration in minutes
            **kwargs: Additional parameters
        
        Returns:
            Estimated cost in USD
        """
        if model not in self.model_catalog:
            logger.warning(f"Audio model {model} not in catalog, using default pricing")
            return Decimal(str(duration_minutes)) * Decimal("0.006")  # Default $0.006 per minute

        model_info = self.model_catalog[model]

        if duration_minutes == 0:
            return Decimal("0.00")

        # Audio models typically charge per minute
        cost = Decimal(str(duration_minutes)) * model_info.cost_per_million_tokens
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def compare_models(
        self,
        models: List[str],
        estimated_tokens: int = 1000,
        include_batch_pricing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compare costs across multiple models.
        
        Args:
            models: List of model names to compare
            estimated_tokens: Estimated tokens for comparison
            include_batch_pricing: Include batch pricing in comparison
        
        Returns:
            List of model comparisons with cost analysis
        """
        comparisons = []

        for model in models:
            if model not in self.model_catalog:
                logger.warning(f"Model {model} not in catalog, skipping")
                continue

            model_info = self.model_catalog[model]

            # Calculate standard cost
            standard_cost = self.estimate_chat_cost(model, tokens=estimated_tokens)

            # Calculate batch cost if applicable
            batch_cost = None
            if include_batch_pricing:
                batch_cost = self.estimate_chat_cost(model, tokens=estimated_tokens, is_batch=True)

            comparison = {
                "model": model,
                "parameters": model_info.parameters,
                "pricing_tier": model_info.pricing_tier.value,
                "cost_per_million_tokens": float(model_info.cost_per_million_tokens),
                "estimated_cost": float(standard_cost),
                "context_length": model_info.context_length,
                "modalities": model_info.modalities,
            }

            if batch_cost is not None:
                comparison["batch_cost"] = float(batch_cost)
                comparison["batch_savings"] = float(standard_cost - batch_cost)

            comparisons.append(comparison)

        # Sort by estimated cost
        comparisons.sort(key=lambda x: x["estimated_cost"])

        return comparisons

    def recommend_model(
        self,
        task_complexity: str = "moderate",  # simple, moderate, complex
        budget_per_operation: float = 0.01,
        min_context_length: int = 4096,
        required_modalities: Optional[List[str]] = None,
        prefer_batch: bool = False,
        **kwargs
    ) -> ModelRecommendation:
        """
        Recommend optimal model based on requirements and budget.
        
        Args:
            task_complexity: Task complexity level
            budget_per_operation: Budget per operation in USD
            min_context_length: Minimum required context length
            required_modalities: Required modalities (text, image, audio)
            prefer_batch: Prefer models with good batch pricing
            **kwargs: Additional parameters
        
        Returns:
            ModelRecommendation with analysis and alternatives
        """
        required_modalities = required_modalities or ["text"]

        # Filter models by requirements
        suitable_models = []

        for model_name, model_info in self.model_catalog.items():
            # Check context length requirement
            if model_info.context_length and model_info.context_length < min_context_length:
                continue

            # Check modality requirements
            if not all(modality in model_info.modalities for modality in required_modalities):
                continue

            # Estimate cost for typical operation
            estimated_tokens = self._get_tokens_for_complexity(task_complexity)
            estimated_cost = self.estimate_chat_cost(model_name, tokens=estimated_tokens, is_batch=prefer_batch)

            # Check budget constraint
            if float(estimated_cost) > budget_per_operation:
                continue

            suitable_models.append({
                "model": model_name,
                "info": model_info,
                "estimated_cost": estimated_cost,
                "estimated_tokens": estimated_tokens
            })

        if not suitable_models:
            # No models meet all requirements, find closest alternatives
            alternatives = self._find_alternative_models(
                task_complexity, budget_per_operation, min_context_length, required_modalities
            )

            return ModelRecommendation(
                recommended_model=None,
                estimated_cost=Decimal("0.00"),
                reasoning="No models meet all requirements. Consider increasing budget or reducing requirements.",
                alternatives=alternatives,
                cost_comparison={}
            )

        # Sort by cost and select best option
        suitable_models.sort(key=lambda x: (float(x["estimated_cost"]), self._get_model_score(x["info"])))

        best_model = suitable_models[0]
        alternatives = suitable_models[1:5]  # Top 5 alternatives

        # Build cost comparison
        cost_comparison = {}
        for model_data in suitable_models[:5]:
            cost_comparison[model_data["model"]] = model_data["estimated_cost"]

        # Generate reasoning
        reasoning = self._generate_recommendation_reasoning(
            best_model, task_complexity, budget_per_operation, prefer_batch
        )

        return ModelRecommendation(
            recommended_model=best_model["model"],
            estimated_cost=best_model["estimated_cost"],
            reasoning=reasoning,
            alternatives=[
                {
                    "model": alt["model"],
                    "cost": float(alt["estimated_cost"]),
                    "parameters": alt["info"].parameters,
                    "tier": alt["info"].pricing_tier.value
                }
                for alt in alternatives
            ],
            cost_comparison={k: float(v) for k, v in cost_comparison.items()}
        )

    def analyze_costs(
        self,
        operations_per_day: int,
        avg_tokens_per_operation: int,
        model: str,
        days_to_analyze: int = 30,
        batch_percentage: float = 0.0
    ) -> Dict[str, Any]:
        """
        Analyze costs for projected usage patterns.
        
        Args:
            operations_per_day: Expected operations per day
            avg_tokens_per_operation: Average tokens per operation
            model: Model to analyze
            days_to_analyze: Number of days to project
            batch_percentage: Percentage of operations that use batch pricing
        
        Returns:
            Detailed cost analysis and projections
        """
        if model not in self.model_catalog:
            return {"error": f"Model {model} not found in catalog"}

        model_info = self.model_catalog[model]

        # Calculate standard and batch costs
        standard_cost_per_op = self.estimate_chat_cost(model, tokens=avg_tokens_per_operation)
        batch_cost_per_op = self.estimate_chat_cost(model, tokens=avg_tokens_per_operation, is_batch=True)

        # Calculate blended cost per operation
        standard_ops = operations_per_day * (1 - batch_percentage)
        batch_ops = operations_per_day * batch_percentage

        daily_cost = (standard_ops * float(standard_cost_per_op)) + (batch_ops * float(batch_cost_per_op))
        monthly_cost = daily_cost * days_to_analyze

        # Find potential savings with alternative models
        alternatives = self._find_cost_alternatives(model, avg_tokens_per_operation)
        best_alternative = alternatives[0] if alternatives else None

        potential_savings = 0.0
        if best_alternative:
            alt_daily_cost = operations_per_day * float(best_alternative["estimated_cost"])
            potential_savings = (daily_cost - alt_daily_cost) * days_to_analyze

        return {
            "current_model": model,
            "model_info": {
                "parameters": model_info.parameters,
                "tier": model_info.pricing_tier.value,
                "cost_per_million": float(model_info.cost_per_million_tokens)
            },
            "usage_projection": {
                "operations_per_day": operations_per_day,
                "avg_tokens_per_operation": avg_tokens_per_operation,
                "batch_percentage": batch_percentage * 100,
                "analysis_days": days_to_analyze
            },
            "cost_analysis": {
                "cost_per_operation": daily_cost / operations_per_day,
                "daily_cost": daily_cost,
                "monthly_cost": monthly_cost,
                "standard_cost_per_op": float(standard_cost_per_op),
                "batch_cost_per_op": float(batch_cost_per_op),
                "batch_savings_per_op": float(standard_cost_per_op - batch_cost_per_op)
            },
            "optimization": {
                "best_alternative": best_alternative,
                "potential_monthly_savings": potential_savings,
                "batch_optimization_potential": float(standard_cost_per_op - batch_cost_per_op) * operations_per_day * days_to_analyze
            }
        }

    def _estimate_unknown_model_cost(self, tokens: int) -> Decimal:
        """Estimate cost for unknown models using medium tier pricing."""
        return (Decimal(str(tokens)) / Decimal("1000000")) * Decimal("0.20")

    def _get_tokens_for_complexity(self, complexity: str) -> int:
        """Get typical token count for task complexity."""
        complexity_tokens = {
            "simple": 500,      # Simple Q&A, basic chat
            "moderate": 1500,   # Analysis, explanations, code review
            "complex": 4000     # Complex reasoning, long-form content
        }
        return complexity_tokens.get(complexity, 1500)

    def _get_model_score(self, model_info: ModelInfo) -> float:
        """Get quality score for model (higher = better)."""
        # Score based on parameters and tier
        param_scores = {
            "1B": 1, "3B": 2, "8B": 3, "11B": 4, "16B": 5,
            "32B": 6, "70B": 8, "405B": 10
        }

        base_score = param_scores.get(model_info.parameters.replace("B", "B"), 3)

        # Bonus for multimodal capabilities
        if len(model_info.modalities) > 1:
            base_score += 1

        # Tier adjustments
        if model_info.pricing_tier == FireworksPricingTier.SPECIALIZED:
            base_score += 2

        return base_score

    def _find_alternative_models(
        self,
        task_complexity: str,
        budget: float,
        context_length: int,
        modalities: List[str]
    ) -> List[Dict[str, Any]]:
        """Find alternative models when no perfect match exists."""
        alternatives = []

        for model_name, model_info in self.model_catalog.items():
            estimated_tokens = self._get_tokens_for_complexity(task_complexity)
            estimated_cost = self.estimate_chat_cost(model_name, tokens=estimated_tokens)

            # Rate how well this model fits requirements
            fit_score = 0

            # Modality fit
            modality_match = sum(1 for mod in modalities if mod in model_info.modalities) / len(modalities)
            fit_score += modality_match * 5

            # Context length fit
            if model_info.context_length and model_info.context_length >= context_length:
                fit_score += 3
            elif model_info.context_length and model_info.context_length >= context_length * 0.5:
                fit_score += 1

            # Budget fit
            if float(estimated_cost) <= budget:
                fit_score += 5
            elif float(estimated_cost) <= budget * 1.5:
                fit_score += 2

            alternatives.append({
                "model": model_name,
                "estimated_cost": float(estimated_cost),
                "fit_score": fit_score,
                "parameters": model_info.parameters,
                "tier": model_info.pricing_tier.value,
                "context_length": model_info.context_length,
                "modalities": model_info.modalities
            })

        # Sort by fit score descending
        alternatives.sort(key=lambda x: x["fit_score"], reverse=True)

        return alternatives[:5]

    def _find_cost_alternatives(self, current_model: str, tokens: int) -> List[Dict[str, Any]]:
        """Find cheaper alternatives to current model."""
        current_cost = self.estimate_chat_cost(current_model, tokens=tokens)

        alternatives = []

        for model_name, model_info in self.model_catalog.items():
            if model_name == current_model:
                continue

            model_cost = self.estimate_chat_cost(model_name, tokens=tokens)

            if model_cost < current_cost:
                savings = float(current_cost - model_cost)
                alternatives.append({
                    "model": model_name,
                    "estimated_cost": float(model_cost),
                    "savings_per_operation": savings,
                    "parameters": model_info.parameters,
                    "tier": model_info.pricing_tier.value
                })

        # Sort by savings (highest first)
        alternatives.sort(key=lambda x: x["savings_per_operation"], reverse=True)

        return alternatives

    def _generate_recommendation_reasoning(
        self,
        model_data: Dict[str, Any],
        task_complexity: str,
        budget: float,
        prefer_batch: bool
    ) -> str:
        """Generate human-readable reasoning for model recommendation."""
        model_info = model_data["info"]
        cost = float(model_data["estimated_cost"])

        reasoning_parts = []

        # Cost efficiency
        if cost <= budget * 0.5:
            reasoning_parts.append(f"Excellent cost efficiency at ${cost:.4f} (well under ${budget:.3f} budget)")
        elif cost <= budget * 0.8:
            reasoning_parts.append(f"Good cost efficiency at ${cost:.4f} (within ${budget:.3f} budget)")
        else:
            reasoning_parts.append(f"Cost-effective at ${cost:.4f} (fits ${budget:.3f} budget)")

        # Model capabilities
        if "image" in model_info.modalities:
            reasoning_parts.append("supports multimodal (vision) capabilities")

        if model_info.context_length > 100000:
            reasoning_parts.append("offers large context window for complex tasks")

        # Tier explanation
        if model_info.pricing_tier == FireworksPricingTier.TINY:
            reasoning_parts.append("optimized for high-throughput, simple tasks")
        elif model_info.pricing_tier == FireworksPricingTier.SMALL:
            reasoning_parts.append("balanced for cost and performance")
        elif model_info.pricing_tier == FireworksPricingTier.LARGE:
            reasoning_parts.append("provides high-quality responses for complex tasks")
        elif model_info.pricing_tier == FireworksPricingTier.SPECIALIZED:
            reasoning_parts.append("specialized model with advanced capabilities")

        if prefer_batch:
            reasoning_parts.append("optimized for batch processing with 50% cost savings")

        return "; ".join(reasoning_parts)


# Export key classes
__all__ = [
    "FireworksPricingCalculator",
    "FireworksPricingTier",
    "ModelInfo",
    "CostBreakdown",
    "ModelRecommendation"
]
