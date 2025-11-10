#!/usr/bin/env python3
"""
GenOps Mistral AI Pricing Calculator

This module provides accurate cost calculation for all Mistral AI models and operations.
It maintains up-to-date pricing information and provides cost optimization insights
for European AI workloads with GDPR compliance benefits.

Features:
- Current Mistral model pricing (November 2024)
- Token-based cost calculation for chat and completion operations
- Embedding cost calculation with performance metrics
- Cost optimization recommendations and model comparisons
- European AI provider pricing advantages analysis
- Enterprise pricing support for custom rates

Usage:
    from genops.providers.mistral_pricing import MistralPricingCalculator
    
    calc = MistralPricingCalculator()
    input_cost, output_cost, total_cost = calc.calculate_cost(
        model="mistral-small-latest",
        operation="chat", 
        input_tokens=100,
        output_tokens=50
    )
    
    print(f"Total cost: ${total_cost:.6f}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MistralPricingTier(Enum):
    """Mistral pricing tiers with different rate structures."""
    PAY_AS_YOU_GO = "pay_as_you_go"
    ENTERPRISE = "enterprise"
    VOLUME_DISCOUNT = "volume_discount"

@dataclass
class ModelPricing:
    """Pricing information for a specific Mistral model."""
    model_name: str
    input_price_per_million: float  # USD per million input tokens
    output_price_per_million: float  # USD per million output tokens
    context_window: int  # Maximum context length
    description: str
    model_family: str
    recommended_use_cases: List[str] = field(default_factory=list)
    performance_tier: str = "standard"  # standard, premium, enterprise
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for transparency and optimization."""
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cost_per_token: float
    pricing_tier: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Cost efficiency metrics
    tokens_per_dollar: float = 0.0
    cost_per_1k_tokens: float = 0.0
    relative_cost_vs_baseline: float = 1.0  # Relative to mistral-small-latest

@dataclass
class PricingInsight:
    """Cost optimization insight and recommendation."""
    category: str  # "model_selection", "token_optimization", "cost_efficiency"
    insight: str
    potential_savings: float
    recommended_action: str
    confidence: str  # "high", "medium", "low"

class MistralPricingCalculator:
    """Comprehensive pricing calculator for Mistral AI models."""

    def __init__(self, pricing_tier: MistralPricingTier = MistralPricingTier.PAY_AS_YOU_GO):
        """
        Initialize pricing calculator with current rates.
        
        Args:
            pricing_tier: Pricing tier for rate calculations
        """
        self.pricing_tier = pricing_tier
        self.pricing_data = self._load_current_pricing()
        self.baseline_model = "mistral-small-latest"  # Reference for comparisons

        logger.info(f"Mistral pricing calculator initialized with {len(self.pricing_data)} models")

    def _load_current_pricing(self) -> Dict[str, ModelPricing]:
        """Load current Mistral AI pricing (November 2024)."""
        pricing = {
            # Core models - latest pricing
            "mistral-large-2407": ModelPricing(
                model_name="mistral-large-2407",
                input_price_per_million=8.0,
                output_price_per_million=24.0,
                context_window=128000,
                description="Flagship model with advanced reasoning",
                model_family="mistral-large",
                recommended_use_cases=[
                    "Complex reasoning", "Code generation", "Analysis",
                    "Research", "Enterprise applications"
                ],
                performance_tier="premium"
            ),

            "mistral-large-latest": ModelPricing(
                model_name="mistral-large-latest",
                input_price_per_million=8.0,
                output_price_per_million=24.0,
                context_window=128000,
                description="Latest large model with frontier capabilities",
                model_family="mistral-large",
                recommended_use_cases=[
                    "Advanced reasoning", "Complex analysis", "Enterprise AI",
                    "Research applications", "Multi-step workflows"
                ],
                performance_tier="premium"
            ),

            "mistral-medium-latest": ModelPricing(
                model_name="mistral-medium-latest",
                input_price_per_million=2.75,
                output_price_per_million=8.10,
                context_window=32000,
                description="Balanced performance and cost",
                model_family="mistral-medium",
                recommended_use_cases=[
                    "General chat", "Content generation", "Analysis",
                    "Customer service", "Document processing"
                ],
                performance_tier="standard"
            ),

            "mistral-small-latest": ModelPricing(
                model_name="mistral-small-latest",
                input_price_per_million=1.0,
                output_price_per_million=3.0,
                context_window=32000,
                description="Cost-effective for most tasks",
                model_family="mistral-small",
                recommended_use_cases=[
                    "Simple chat", "Basic generation", "Classification",
                    "Summarization", "Q&A"
                ],
                performance_tier="standard"
            ),

            "mistral-tiny-2312": ModelPricing(
                model_name="mistral-tiny-2312",
                input_price_per_million=0.25,
                output_price_per_million=0.25,
                context_window=32000,
                description="Ultra-low cost for simple tasks",
                model_family="mistral-tiny",
                recommended_use_cases=[
                    "Simple classification", "Basic Q&A", "Testing",
                    "High-volume simple tasks", "Development"
                ],
                performance_tier="basic"
            ),

            # Mixtral models
            "mixtral-8x7b-32768": ModelPricing(
                model_name="mixtral-8x7b-32768",
                input_price_per_million=0.7,
                output_price_per_million=0.7,
                context_window=32000,
                description="Mixture of experts model",
                model_family="mixtral",
                recommended_use_cases=[
                    "Code generation", "Multi-domain tasks", "Efficient processing",
                    "Specialized workflows", "Performance-cost balance"
                ],
                performance_tier="standard"
            ),

            "mixtral-8x22b-32768": ModelPricing(
                model_name="mixtral-8x22b-32768",
                input_price_per_million=2.0,
                output_price_per_million=6.0,
                context_window=64000,
                description="Large mixture of experts model",
                model_family="mixtral",
                recommended_use_cases=[
                    "Advanced code generation", "Complex reasoning",
                    "Multi-domain expertise", "Large context tasks"
                ],
                performance_tier="premium"
            ),

            # Specialized models
            "mistral-nemo-2407": ModelPricing(
                model_name="mistral-nemo-2407",
                input_price_per_million=1.0,
                output_price_per_million=1.0,
                context_window=128000,
                description="Long context specialized model",
                model_family="mistral-nemo",
                recommended_use_cases=[
                    "Long document analysis", "Extended context",
                    "Research", "Document processing", "Large context tasks"
                ],
                performance_tier="specialized"
            ),

            "codestral-2405": ModelPricing(
                model_name="codestral-2405",
                input_price_per_million=3.0,
                output_price_per_million=3.0,
                context_window=32000,
                description="Code generation and analysis specialist",
                model_family="codestral",
                recommended_use_cases=[
                    "Code generation", "Code review", "Programming assistance",
                    "Technical documentation", "Software development"
                ],
                performance_tier="specialized"
            ),

            # Embedding models
            "mistral-embed": ModelPricing(
                model_name="mistral-embed",
                input_price_per_million=0.1,  # Embedding models typically charged per input
                output_price_per_million=0.0,  # No output tokens for embeddings
                context_window=8192,
                description="Text embedding model",
                model_family="mistral-embed",
                recommended_use_cases=[
                    "Semantic search", "Document similarity", "Clustering",
                    "Classification", "RAG applications"
                ],
                performance_tier="specialized"
            )
        }

        return pricing

    def get_model_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing information for a specific model."""
        return self.pricing_data.get(model)

    def calculate_cost(
        self,
        model: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **kwargs
    ) -> Tuple[float, float, float]:
        """
        Calculate costs for a Mistral operation.
        
        Args:
            model: Mistral model name
            operation: Operation type (chat, embed, completion)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (0 for embeddings)
            **kwargs: Additional parameters (pricing_tier, volume_discount, etc.)
            
        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD
        """
        pricing = self.get_model_pricing(model)
        if not pricing:
            logger.warning(f"Pricing not available for model: {model}")
            return 0.0, 0.0, 0.0

        # Calculate base costs
        input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_million
        total_cost = input_cost + output_cost

        # Apply pricing tier adjustments
        tier_multiplier = self._get_tier_multiplier(kwargs.get('pricing_tier', self.pricing_tier))

        input_cost *= tier_multiplier
        output_cost *= tier_multiplier
        total_cost *= tier_multiplier

        return input_cost, output_cost, total_cost

    def _get_tier_multiplier(self, tier: MistralPricingTier) -> float:
        """Get pricing multiplier for different tiers."""
        multipliers = {
            MistralPricingTier.PAY_AS_YOU_GO: 1.0,
            MistralPricingTier.ENTERPRISE: 0.85,  # 15% enterprise discount
            MistralPricingTier.VOLUME_DISCOUNT: 0.75  # 25% volume discount
        }
        return multipliers.get(tier, 1.0)

    def get_cost_breakdown(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        **kwargs
    ) -> CostBreakdown:
        """Get detailed cost breakdown with efficiency metrics."""
        input_cost, output_cost, total_cost = self.calculate_cost(
            model, operation, input_tokens, output_tokens, **kwargs
        )

        total_tokens = input_tokens + output_tokens
        cost_per_token = total_cost / max(total_tokens, 1)
        tokens_per_dollar = max(total_tokens, 1) / max(total_cost, 0.000001)
        cost_per_1k_tokens = cost_per_token * 1000

        # Calculate relative cost vs baseline
        baseline_cost = self.calculate_cost(
            self.baseline_model, operation, input_tokens, output_tokens, **kwargs
        )[2]
        relative_cost = total_cost / max(baseline_cost, 0.000001)

        return CostBreakdown(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            cost_per_token=cost_per_token,
            pricing_tier=self.pricing_tier.value,
            tokens_per_dollar=tokens_per_dollar,
            cost_per_1k_tokens=cost_per_1k_tokens,
            relative_cost_vs_baseline=relative_cost
        )

    def compare_models(
        self,
        models: List[str],
        operation: str = "chat",
        input_tokens: int = 1000,
        output_tokens: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Compare costs across multiple Mistral models.
        
        Args:
            models: List of model names to compare
            operation: Operation type
            input_tokens: Input tokens for comparison
            output_tokens: Output tokens for comparison
            
        Returns:
            List of model comparisons sorted by cost efficiency
        """
        comparisons = []

        for model in models:
            breakdown = self.get_cost_breakdown(model, operation, input_tokens, output_tokens)
            pricing = self.get_model_pricing(model)

            if pricing:
                comparisons.append({
                    "model": model,
                    "total_cost": breakdown.total_cost,
                    "cost_per_1k_tokens": breakdown.cost_per_1k_tokens,
                    "tokens_per_dollar": breakdown.tokens_per_dollar,
                    "relative_cost": breakdown.relative_cost_vs_baseline,
                    "context_window": pricing.context_window,
                    "performance_tier": pricing.performance_tier,
                    "model_family": pricing.model_family,
                    "recommended_use_cases": pricing.recommended_use_cases
                })

        # Sort by cost efficiency (tokens per dollar)
        comparisons.sort(key=lambda x: x["tokens_per_dollar"], reverse=True)

        return comparisons

    def get_optimization_insights(
        self,
        current_model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        use_case: Optional[str] = None
    ) -> List[PricingInsight]:
        """
        Generate cost optimization insights and recommendations.
        
        Args:
            current_model: Currently used model
            operation: Operation type
            input_tokens: Average input tokens
            output_tokens: Average output tokens
            use_case: Specific use case for tailored recommendations
            
        Returns:
            List of optimization insights and recommendations
        """
        insights = []
        current_breakdown = self.get_cost_breakdown(current_model, operation, input_tokens, output_tokens)

        # Model selection insights
        all_models = list(self.pricing_data.keys())
        comparisons = self.compare_models(all_models, operation, input_tokens, output_tokens)

        # Find more cost-effective alternatives
        current_cost = current_breakdown.total_cost
        for comp in comparisons:
            if (comp["model"] != current_model and
                comp["total_cost"] < current_cost * 0.8):  # 20%+ savings

                savings = current_cost - comp["total_cost"]
                savings_percent = (savings / current_cost) * 100

                insight = PricingInsight(
                    category="model_selection",
                    insight=f"Switch from {current_model} to {comp['model']} could save ${savings:.6f} ({savings_percent:.1f}%) per operation",
                    potential_savings=savings,
                    recommended_action=f"Test {comp['model']} for your use case - it's {comp['performance_tier']} tier",
                    confidence="medium" if comp["performance_tier"] == "standard" else "low"
                )
                insights.append(insight)
                break  # Only suggest the best alternative

        # Token optimization insights
        if output_tokens > input_tokens * 2:  # High output ratio
            potential_savings = current_breakdown.output_cost * 0.3  # 30% reduction potential
            insights.append(PricingInsight(
                category="token_optimization",
                insight=f"High output token ratio ({output_tokens}/{input_tokens}). Reducing output length could save ~${potential_savings:.6f}",
                potential_savings=potential_savings,
                recommended_action="Use max_tokens parameter to limit response length for simple tasks",
                confidence="high"
            ))

        # European AI provider advantages
        if current_breakdown.total_cost > 0.001:  # For significant costs
            insights.append(PricingInsight(
                category="cost_efficiency",
                insight="Mistral provides GDPR-compliant EU-based AI at competitive rates vs US providers",
                potential_savings=current_breakdown.total_cost * 0.2,  # Estimated 20% vs OpenAI equivalent
                recommended_action="Leverage Mistral's European data residency and cost advantages for compliance",
                confidence="high"
            ))

        # Volume pricing insights
        if current_breakdown.total_cost > 0.01:  # For high-volume usage
            volume_savings = current_breakdown.total_cost * 0.25  # 25% volume discount
            insights.append(PricingInsight(
                category="cost_efficiency",
                insight="Volume discounts available for enterprise usage",
                potential_savings=volume_savings,
                recommended_action="Contact Mistral for enterprise pricing on high-volume workloads",
                confidence="medium"
            ))

        return insights

    def estimate_monthly_cost(
        self,
        model: str,
        operations_per_day: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        operation: str = "chat"
    ) -> Dict[str, Any]:
        """
        Estimate monthly costs for regular usage patterns.
        
        Args:
            model: Mistral model name
            operations_per_day: Average operations per day
            avg_input_tokens: Average input tokens per operation
            avg_output_tokens: Average output tokens per operation
            operation: Operation type
            
        Returns:
            Monthly cost estimate with breakdown
        """
        daily_cost = self.calculate_cost(
            model, operation,
            avg_input_tokens * operations_per_day,
            avg_output_tokens * operations_per_day
        )[2]

        monthly_cost = daily_cost * 30
        annual_cost = daily_cost * 365

        # Get efficiency metrics
        breakdown = self.get_cost_breakdown(model, operation, avg_input_tokens, avg_output_tokens)

        return {
            "model": model,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "annual_cost": annual_cost,
            "operations_per_day": operations_per_day,
            "cost_per_operation": breakdown.total_cost,
            "tokens_per_operation": breakdown.total_tokens,
            "cost_efficiency": {
                "cost_per_1k_tokens": breakdown.cost_per_1k_tokens,
                "tokens_per_dollar": breakdown.tokens_per_dollar,
                "relative_to_baseline": breakdown.relative_cost_vs_baseline
            }
        }

    def get_model_recommendations(self, use_case: str) -> List[Dict[str, Any]]:
        """
        Get model recommendations for specific use cases.
        
        Args:
            use_case: Description of the use case
            
        Returns:
            List of recommended models with rationale
        """
        recommendations = []
        use_case_lower = use_case.lower()

        # Analyze use case and match to models
        for model_name, pricing in self.pricing_data.items():
            relevance_score = 0
            rationale = []

            # Check use case alignment
            for rec_use_case in pricing.recommended_use_cases:
                if any(keyword in use_case_lower for keyword in rec_use_case.lower().split()):
                    relevance_score += 2
                    rationale.append(f"Optimized for {rec_use_case}")

            # Performance tier matching
            if "complex" in use_case_lower or "advanced" in use_case_lower:
                if pricing.performance_tier in ["premium", "specialized"]:
                    relevance_score += 1
                    rationale.append("High-performance capabilities")
            elif "simple" in use_case_lower or "basic" in use_case_lower:
                if pricing.performance_tier == "basic":
                    relevance_score += 2
                    rationale.append("Cost-optimized for simple tasks")

            # Context window requirements
            if "long" in use_case_lower or "document" in use_case_lower:
                if pricing.context_window >= 64000:
                    relevance_score += 1
                    rationale.append("Large context window support")

            if relevance_score > 0:
                # Calculate cost efficiency for typical use case
                breakdown = self.get_cost_breakdown(model_name, "chat", 500, 200)  # Typical tokens

                recommendations.append({
                    "model": model_name,
                    "relevance_score": relevance_score,
                    "rationale": rationale,
                    "cost_per_operation": breakdown.total_cost,
                    "performance_tier": pricing.performance_tier,
                    "context_window": pricing.context_window,
                    "model_family": pricing.model_family
                })

        # Sort by relevance score, then by cost efficiency
        recommendations.sort(key=lambda x: (x["relevance_score"], -x["cost_per_operation"]), reverse=True)

        return recommendations[:5]  # Top 5 recommendations

    def export_pricing_data(self) -> Dict[str, Any]:
        """Export current pricing data for external use."""
        exported = {
            "pricing_tier": self.pricing_tier.value,
            "last_updated": datetime.now().isoformat(),
            "models": {}
        }

        for model_name, pricing in self.pricing_data.items():
            exported["models"][model_name] = {
                "input_price_per_million": pricing.input_price_per_million,
                "output_price_per_million": pricing.output_price_per_million,
                "context_window": pricing.context_window,
                "description": pricing.description,
                "model_family": pricing.model_family,
                "performance_tier": pricing.performance_tier,
                "recommended_use_cases": pricing.recommended_use_cases
            }

        return exported

# Convenience functions
def calculate_mistral_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    operation: str = "chat"
) -> float:
    """Quick cost calculation for Mistral operations."""
    calc = MistralPricingCalculator()
    return calc.calculate_cost(model, operation, input_tokens, output_tokens)[2]

def compare_mistral_models(models: List[str], tokens: Tuple[int, int] = (1000, 500)) -> List[Dict[str, Any]]:
    """Quick model comparison for cost optimization."""
    calc = MistralPricingCalculator()
    return calc.compare_models(models, "chat", tokens[0], tokens[1])

if __name__ == "__main__":
    # Demo and testing
    print("Mistral AI Pricing Calculator Demo")
    print("=" * 40)

    calc = MistralPricingCalculator()

    # Test cost calculation
    model = "mistral-small-latest"
    input_tokens, output_tokens = 1000, 500

    input_cost, output_cost, total_cost = calc.calculate_cost(model, "chat", input_tokens, output_tokens)
    print(f"Cost for {model}:")
    print(f"  Input: ${input_cost:.6f}")
    print(f"  Output: ${output_cost:.6f}")
    print(f"  Total: ${total_cost:.6f}")

    # Test model comparison
    models = ["mistral-tiny-2312", "mistral-small-latest", "mistral-medium-latest"]
    comparisons = calc.compare_models(models)

    print("\nModel Comparison (1000 in, 500 out tokens):")
    for comp in comparisons:
        print(f"  {comp['model']}: ${comp['total_cost']:.6f} ({comp['cost_per_1k_tokens']:.4f}/1k tokens)")

    # Test insights
    insights = calc.get_optimization_insights("mistral-large-latest", "chat", 1000, 500)
    if insights:
        print("\nOptimization Insights:")
        for insight in insights[:2]:  # Top 2
            print(f"  • {insight.insight}")
            print(f"    Action: {insight.recommended_action}")
