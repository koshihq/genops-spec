#!/usr/bin/env python3
"""
Gemini pricing calculations and cost intelligence for GenOps.

This module provides comprehensive pricing information and cost calculation
utilities for Google Gemini models. It supports all major Gemini model
variants and pricing tiers.

Features:
- Real-time cost calculation for all Gemini models
- Multi-tier pricing support (free, paid, enterprise)
- Context caching cost calculations
- Model comparison and optimization recommendations
- Regional pricing variations (where applicable)

Usage:
    from genops.providers.gemini_pricing import calculate_gemini_cost, compare_gemini_models
    
    # Calculate cost for a specific operation
    cost = calculate_gemini_cost(
        model_id="gemini-2.5-flash",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Compare models for cost optimization
    comparison = compare_gemini_models(
        models=["gemini-2.5-pro", "gemini-2.5-flash"],
        input_tokens=1000,
        output_tokens=500
    )
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GeminiTier(Enum):
    """Gemini pricing tiers."""
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"


@dataclass
class GeminiModelInfo:
    """Information about a Gemini model including pricing and capabilities."""
    model_id: str
    display_name: str
    provider: str
    tier: GeminiTier
    input_price_per_1m_tokens: float  # USD per 1M input tokens
    output_price_per_1m_tokens: float  # USD per 1M output tokens
    context_cache_price_per_1m_tokens: Optional[float]  # USD per 1M cached tokens
    max_context_length: int
    max_output_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_multimodal: bool
    supports_code_execution: bool
    knowledge_cutoff: str
    description: str


@dataclass
class GeminiCostBreakdown:
    """Detailed cost breakdown for a Gemini operation."""
    model_id: str
    input_tokens: int
    output_tokens: int
    context_cache_tokens: Optional[int]
    input_cost: float
    output_cost: float
    context_cache_cost: float
    total_cost: float
    currency: str
    tier: GeminiTier
    cost_per_1k_tokens: float
    estimated_cost_1k_requests: float


# Comprehensive Gemini model pricing data
# Pricing as of January 2025 - update regularly
GEMINI_MODELS: Dict[str, GeminiModelInfo] = {
    "gemini-2.5-pro": GeminiModelInfo(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        provider="google",
        tier=GeminiTier.PAID,
        input_price_per_1m_tokens=1.25,  # $1.25 per 1M input tokens (≤200k)
        output_price_per_1m_tokens=10.00,  # $10.00 per 1M output tokens
        context_cache_price_per_1m_tokens=0.125,  # $0.125 per 1M cached tokens
        max_context_length=1_048_576,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_multimodal=True,
        supports_code_execution=True,
        knowledge_cutoff="January 2025",
        description="Most capable reasoning model for complex problem solving"
    ),

    "gemini-2.5-flash": GeminiModelInfo(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        provider="google",
        tier=GeminiTier.PAID,
        input_price_per_1m_tokens=0.30,  # $0.30 per 1M input tokens
        output_price_per_1m_tokens=2.50,  # $2.50 per 1M output tokens
        context_cache_price_per_1m_tokens=0.03,  # $0.03 per 1M cached tokens
        max_context_length=1_048_576,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_multimodal=True,
        supports_code_execution=True,
        knowledge_cutoff="January 2025",
        description="Best price-performance model for large-scale processing"
    ),

    "gemini-2.5-flash-lite": GeminiModelInfo(
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        provider="google",
        tier=GeminiTier.PAID,
        input_price_per_1m_tokens=0.15,  # Estimated - most cost-efficient
        output_price_per_1m_tokens=1.25,  # Estimated
        context_cache_price_per_1m_tokens=0.015,  # Estimated
        max_context_length=1_048_576,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_multimodal=True,
        supports_code_execution=False,
        knowledge_cutoff="January 2025",
        description="Most cost-efficient model optimized for low latency"
    ),

    "gemini-1.5-pro": GeminiModelInfo(
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        provider="google",
        tier=GeminiTier.PAID,
        input_price_per_1m_tokens=1.25,  # Same as 2.5 Pro
        output_price_per_1m_tokens=10.00,
        context_cache_price_per_1m_tokens=0.125,
        max_context_length=2_097_152,  # 2M context
        max_output_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_multimodal=True,
        supports_code_execution=True,
        knowledge_cutoff="April 2024",
        description="Previous generation Pro model with extended context"
    ),

    "gemini-1.5-flash": GeminiModelInfo(
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        provider="google",
        tier=GeminiTier.PAID,
        input_price_per_1m_tokens=0.30,
        output_price_per_1m_tokens=2.50,
        context_cache_price_per_1m_tokens=0.03,
        max_context_length=1_048_576,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_multimodal=True,
        supports_code_execution=True,
        knowledge_cutoff="April 2024",
        description="Previous generation Flash model"
    ),

    # Free tier models (limited capabilities and rate limits)
    "gemini-1.5-flash-free": GeminiModelInfo(
        model_id="gemini-1.5-flash-free",
        display_name="Gemini 1.5 Flash (Free)",
        provider="google",
        tier=GeminiTier.FREE,
        input_price_per_1m_tokens=0.0,  # Free tier
        output_price_per_1m_tokens=0.0,
        context_cache_price_per_1m_tokens=0.0,
        max_context_length=1_048_576,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_function_calling=True,
        supports_multimodal=True,
        supports_code_execution=False,
        knowledge_cutoff="April 2024",
        description="Free tier with rate limits and usage restrictions"
    )
}


def calculate_gemini_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    context_cache_tokens: Optional[int] = None,
    tier: Optional[GeminiTier] = None
) -> float:
    """
    Calculate the cost of a Gemini API operation.
    
    Args:
        model_id: Gemini model identifier (e.g., "gemini-2.5-flash")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens  
        context_cache_tokens: Number of context cache tokens (optional)
        tier: Pricing tier override (optional)
    
    Returns:
        Total cost in USD
    """
    if model_id not in GEMINI_MODELS:
        logger.warning(f"Unknown Gemini model: {model_id}, using default pricing")
        # Use Flash pricing as fallback
        model_info = GEMINI_MODELS["gemini-2.5-flash"]
    else:
        model_info = GEMINI_MODELS[model_id]

    # Override tier if specified
    if tier:
        if tier == GeminiTier.FREE:
            return 0.0  # Free tier has no cost
        # For other tiers, use the model's base pricing

    # Calculate input cost (per million tokens)
    input_cost = (input_tokens / 1_000_000) * model_info.input_price_per_1m_tokens

    # Calculate output cost (per million tokens)
    output_cost = (output_tokens / 1_000_000) * model_info.output_price_per_1m_tokens

    # Calculate context cache cost if applicable
    context_cache_cost = 0.0
    if context_cache_tokens and model_info.context_cache_price_per_1m_tokens:
        context_cache_cost = (context_cache_tokens / 1_000_000) * model_info.context_cache_price_per_1m_tokens

    total_cost = input_cost + output_cost + context_cache_cost

    return round(total_cost, 8)  # Round to 8 decimal places for precision


def calculate_gemini_cost_breakdown(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    context_cache_tokens: Optional[int] = None
) -> GeminiCostBreakdown:
    """
    Calculate detailed cost breakdown for a Gemini operation.
    
    Args:
        model_id: Gemini model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        context_cache_tokens: Number of context cache tokens
    
    Returns:
        GeminiCostBreakdown with detailed cost information
    """
    model_info = GEMINI_MODELS.get(model_id, GEMINI_MODELS["gemini-2.5-flash"])

    # Calculate individual cost components
    input_cost = (input_tokens / 1_000_000) * model_info.input_price_per_1m_tokens
    output_cost = (output_tokens / 1_000_000) * model_info.output_price_per_1m_tokens

    context_cache_cost = 0.0
    if context_cache_tokens and model_info.context_cache_price_per_1m_tokens:
        context_cache_cost = (context_cache_tokens / 1_000_000) * model_info.context_cache_price_per_1m_tokens

    total_cost = input_cost + output_cost + context_cache_cost

    # Calculate cost per 1k tokens for comparison
    total_tokens = input_tokens + output_tokens + (context_cache_tokens or 0)
    cost_per_1k_tokens = (total_cost / total_tokens) * 1000 if total_tokens > 0 else 0.0

    # Estimate cost for 1k requests of similar size
    estimated_cost_1k_requests = total_cost * 1000

    return GeminiCostBreakdown(
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_cache_tokens=context_cache_tokens,
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        context_cache_cost=round(context_cache_cost, 8),
        total_cost=round(total_cost, 8),
        currency="USD",
        tier=model_info.tier,
        cost_per_1k_tokens=round(cost_per_1k_tokens, 8),
        estimated_cost_1k_requests=round(estimated_cost_1k_requests, 4)
    )


def get_gemini_model_info(model_id: str) -> Optional[GeminiModelInfo]:
    """
    Get detailed information about a Gemini model.
    
    Args:
        model_id: Gemini model identifier
    
    Returns:
        GeminiModelInfo object or None if model not found
    """
    return GEMINI_MODELS.get(model_id)


def list_gemini_models(tier: Optional[GeminiTier] = None) -> List[GeminiModelInfo]:
    """
    List all available Gemini models, optionally filtered by tier.
    
    Args:
        tier: Optional tier filter (FREE, PAID, ENTERPRISE)
    
    Returns:
        List of GeminiModelInfo objects
    """
    models = list(GEMINI_MODELS.values())

    if tier:
        models = [model for model in models if model.tier == tier]

    return sorted(models, key=lambda m: m.input_price_per_1m_tokens)


def compare_gemini_models(
    models: List[str],
    input_tokens: int,
    output_tokens: int,
    context_cache_tokens: Optional[int] = None,
    sort_by: str = "total_cost"
) -> List[Dict[str, Any]]:
    """
    Compare costs across multiple Gemini models for the same operation.
    
    Args:
        models: List of Gemini model IDs to compare
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        context_cache_tokens: Number of context cache tokens
        sort_by: Sort key ("total_cost", "cost_per_1k_tokens", "model_id")
    
    Returns:
        List of comparison results sorted by specified criteria
    """
    comparisons = []

    for model_id in models:
        breakdown = calculate_gemini_cost_breakdown(
            model_id, input_tokens, output_tokens, context_cache_tokens
        )
        model_info = get_gemini_model_info(model_id)

        comparison = {
            "model_id": model_id,
            "display_name": model_info.display_name if model_info else model_id,
            "tier": model_info.tier if model_info else GeminiTier.PAID,
            "total_cost": breakdown.total_cost,
            "input_cost": breakdown.input_cost,
            "output_cost": breakdown.output_cost,
            "context_cache_cost": breakdown.context_cache_cost,
            "cost_per_1k_tokens": breakdown.cost_per_1k_tokens,
            "supports_streaming": model_info.supports_streaming if model_info else False,
            "supports_function_calling": model_info.supports_function_calling if model_info else False,
            "max_context_length": model_info.max_context_length if model_info else 0,
            "description": model_info.description if model_info else "Unknown model"
        }
        comparisons.append(comparison)

    # Sort by specified criteria
    reverse = sort_by != "model_id"  # Sort ascending for model_id, descending for costs
    comparisons.sort(key=lambda x: x[sort_by], reverse=reverse)

    return comparisons


def get_cost_optimization_recommendations(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    use_case: str = "general",
    budget_constraint: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Get cost optimization recommendations for Gemini usage.
    
    Args:
        model_id: Current model being used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        use_case: Use case category ("general", "code", "analysis", "creative")
        budget_constraint: Maximum cost per operation (optional)
    
    Returns:
        List of optimization recommendations
    """
    recommendations = []
    current_cost = calculate_gemini_cost(model_id, input_tokens, output_tokens)

    # Get alternative models based on use case
    if use_case == "code":
        alternatives = ["gemini-2.5-pro", "gemini-2.5-flash"]
    elif use_case == "creative":
        alternatives = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    elif use_case == "analysis":
        alternatives = ["gemini-2.5-pro", "gemini-2.5-flash"]
    else:  # general
        alternatives = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

    # Remove current model from alternatives
    alternatives = [m for m in alternatives if m != model_id]

    for alt_model in alternatives:
        alt_cost = calculate_gemini_cost(alt_model, input_tokens, output_tokens)
        alt_info = get_gemini_model_info(alt_model)

        if not alt_info:
            continue

        savings = current_cost - alt_cost
        savings_percent = (savings / current_cost) * 100 if current_cost > 0 else 0

        # Skip if no meaningful savings or if over budget
        if savings <= 0.000001:  # Less than $0.000001 savings
            continue

        if budget_constraint and alt_cost > budget_constraint:
            continue

        recommendation = {
            "model_id": alt_model,
            "display_name": alt_info.display_name,
            "current_cost": current_cost,
            "alternative_cost": alt_cost,
            "savings": abs(savings),
            "savings_percent": abs(savings_percent),
            "recommendation_type": "cost_reduction" if savings > 0 else "capability_upgrade",
            "description": alt_info.description,
            "tier": alt_info.tier,
            "trade_offs": []
        }

        # Add trade-off analysis
        current_info = get_gemini_model_info(model_id)
        if current_info:
            if alt_info.max_context_length < current_info.max_context_length:
                recommendation["trade_offs"].append("Smaller context window")
            if not alt_info.supports_code_execution and current_info.supports_code_execution:
                recommendation["trade_offs"].append("No code execution support")

        recommendations.append(recommendation)

    # Sort by savings potential
    recommendations.sort(key=lambda x: x["savings_percent"], reverse=True)

    return recommendations


def estimate_monthly_cost(
    model_id: str,
    daily_operations: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    days_per_month: int = 30
) -> Dict[str, Any]:
    """
    Estimate monthly costs for Gemini usage patterns.
    
    Args:
        model_id: Gemini model identifier
        daily_operations: Average number of operations per day
        avg_input_tokens: Average input tokens per operation
        avg_output_tokens: Average output tokens per operation
        days_per_month: Days per month for calculation
    
    Returns:
        Dictionary with monthly cost estimates and breakdowns
    """
    cost_per_operation = calculate_gemini_cost(model_id, avg_input_tokens, avg_output_tokens)

    daily_cost = cost_per_operation * daily_operations
    monthly_cost = daily_cost * days_per_month

    model_info = get_gemini_model_info(model_id)

    return {
        "model_id": model_id,
        "model_name": model_info.display_name if model_info else model_id,
        "cost_per_operation": cost_per_operation,
        "daily_operations": daily_operations,
        "daily_cost": daily_cost,
        "monthly_cost": monthly_cost,
        "monthly_operations": daily_operations * days_per_month,
        "avg_tokens_per_operation": avg_input_tokens + avg_output_tokens,
        "monthly_tokens": (avg_input_tokens + avg_output_tokens) * daily_operations * days_per_month,
        "tier": model_info.tier if model_info else GeminiTier.PAID,
        "currency": "USD"
    }


# Export main functions and classes
__all__ = [
    'GeminiModelInfo',
    'GeminiCostBreakdown',
    'GeminiTier',
    'GEMINI_MODELS',
    'calculate_gemini_cost',
    'calculate_gemini_cost_breakdown',
    'get_gemini_model_info',
    'list_gemini_models',
    'compare_gemini_models',
    'get_cost_optimization_recommendations',
    'estimate_monthly_cost'
]
