#!/usr/bin/env python3
"""
SkyRouter Cost Calculation Engine

This module provides sophisticated cost calculation capabilities for SkyRouter
multi-model routing operations, including volume discounts, route optimization
savings, and multi-modal pricing across 150+ supported models.

Features:
- Multi-model pricing across 150+ models
- Route optimization cost calculations
- Volume discount tiers and optimization
- Multi-modal operation pricing (text, vision, audio)
- Agent workflow cost modeling
- Currency conversion and regional pricing
- Complex routing scenario cost analysis

Author: GenOps AI Contributors
License: Apache 2.0
"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RouteOptimization(Enum):
    """SkyRouter optimization strategies."""

    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    BALANCED = "balanced"
    RELIABILITY_FIRST = "reliability_first"
    CUSTOM = "custom"


class OperationType(Enum):
    """Types of SkyRouter operations."""

    MODEL_CALL = "model_call"
    MULTI_MODEL_ROUTING = "multi_model_routing"
    AGENT_WORKFLOW = "agent_workflow"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"


@dataclass
class SkyRouterPricingConfig:
    """Configuration for SkyRouter pricing calculations."""

    # Base pricing per 1K tokens for different model tiers
    tier_pricing: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "premium": {"input": 0.030, "output": 0.060},  # GPT-4, Claude-3-Opus
            "standard": {"input": 0.010, "output": 0.020},  # GPT-3.5, Claude-3-Sonnet
            "efficient": {"input": 0.002, "output": 0.004},  # Gemini-Pro, Llama-2
            "local": {"input": 0.000, "output": 0.000},  # Local/open models
        }
    )

    # Route optimization multipliers
    optimization_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            RouteOptimization.COST_OPTIMIZED.value: 0.85,  # 15% savings
            RouteOptimization.LATENCY_OPTIMIZED.value: 1.20,  # 20% premium
            RouteOptimization.BALANCED.value: 1.00,  # Standard pricing
            RouteOptimization.RELIABILITY_FIRST.value: 1.30,  # 30% premium
            RouteOptimization.CUSTOM.value: 1.00,
        }
    )

    # Volume discount tiers (monthly volume)
    volume_tiers: dict[int, float] = field(
        default_factory=lambda: {
            1000: 0.05,  # 5% discount for 1K+ tokens
            10000: 0.12,  # 12% discount for 10K+ tokens
            100000: 0.20,  # 20% discount for 100K+ tokens
            1000000: 0.30,  # 30% discount for 1M+ tokens
        }
    )

    # Complexity multipliers
    complexity_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "simple": 0.7,
            "moderate": 1.0,
            "complex": 1.5,
            "enterprise": 2.0,
        }
    )

    # Multi-modal pricing multipliers
    modal_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "text": 1.0,
            "vision": 1.8,
            "audio": 1.5,
            "multimodal": 2.2,
        }
    )

    # Regional pricing adjustments
    regional_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "us-east": 1.0,
            "us-west": 1.05,
            "eu-central": 1.15,
            "asia-pacific": 1.25,
            "global": 1.10,
        }
    )

    # Currency conversion rates (to USD)
    currency_rates: dict[str, float] = field(
        default_factory=lambda: {
            "USD": 1.0,
            "EUR": 1.08,
            "GBP": 1.25,
            "JPY": 0.0067,
            "CAD": 0.73,
        }
    )


@dataclass
class SkyRouterCostBreakdown:
    """Detailed cost breakdown for SkyRouter operations."""

    base_cost: Decimal
    optimization_adjustment: Decimal
    volume_discount: Decimal
    complexity_adjustment: Decimal
    modal_adjustment: Decimal
    regional_adjustment: Decimal
    final_cost: Decimal

    # Token usage details
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Route details
    selected_route: str
    optimization_strategy: str
    potential_savings: Decimal
    efficiency_score: float

    # Metadata
    currency: str = "USD"
    region: str = "us-east"
    timestamp: float = field(default_factory=time.time)


class SkyRouterPricingCalculator:
    """Advanced cost calculator for SkyRouter multi-model operations."""

    def __init__(self, config: Optional[SkyRouterPricingConfig] = None):
        """Initialize pricing calculator with configuration."""
        self.config = config or SkyRouterPricingConfig()
        self.monthly_volume = 0
        self.current_discount = 0.0

        # Model tier mapping
        self._initialize_model_tiers()

        logger.info("SkyRouter pricing calculator initialized")

    def _initialize_model_tiers(self):
        """Initialize model to tier mapping."""
        self.model_tiers = {
            # Premium tier models
            "gpt-4": "premium",
            "gpt-4-turbo": "premium",
            "claude-3-opus": "premium",
            "claude-3-sonnet": "premium",
            # Standard tier models
            "gpt-3.5-turbo": "standard",
            "claude-3-haiku": "standard",
            "gemini-pro": "standard",
            # Efficient tier models
            "llama-2": "efficient",
            "mistral-7b": "efficient",
            "codellama": "efficient",
            # Local models
            "ollama": "local",
            "local-model": "local",
        }

    def calculate_model_call_cost(
        self,
        model: str,
        input_data: dict[str, Any],
        route_optimization: str = "balanced",
        complexity: str = "moderate",
        region: str = "us-east",
        currency: str = "USD",
    ) -> "SkyRouterCostResult":  # type: ignore  # noqa: F821
        """Calculate cost for a single model call through SkyRouter."""

        # Determine model tier
        tier = self._get_model_tier(model)

        # Estimate token usage
        input_tokens = self._estimate_input_tokens(input_data)
        output_tokens = self._estimate_output_tokens(input_data, complexity)
        total_tokens = input_tokens + output_tokens

        # Base cost calculation
        tier_pricing = self.config.tier_pricing[tier]
        input_cost = Decimal(str((input_tokens / 1000) * tier_pricing["input"]))
        output_cost = Decimal(str((output_tokens / 1000) * tier_pricing["output"]))
        base_cost = input_cost + output_cost

        # Apply optimization multiplier
        optimization_multiplier = self.config.optimization_multipliers.get(
            route_optimization, 1.0
        )
        optimization_adjustment = base_cost * (
            Decimal(str(optimization_multiplier)) - 1
        )

        # Apply volume discount
        volume_discount_rate = self._get_volume_discount(total_tokens)
        volume_discount = base_cost * Decimal(str(volume_discount_rate))

        # Apply complexity multiplier
        complexity_multiplier = self.config.complexity_multipliers.get(complexity, 1.0)
        complexity_adjustment = base_cost * (Decimal(str(complexity_multiplier)) - 1)

        # Apply modal multiplier (detect if multimodal)
        modal_type = self._detect_modal_type(input_data)
        modal_multiplier = self.config.modal_multipliers.get(modal_type, 1.0)
        modal_adjustment = base_cost * (Decimal(str(modal_multiplier)) - 1)

        # Apply regional multiplier
        regional_multiplier = self.config.regional_multipliers.get(region, 1.0)
        regional_adjustment = base_cost * (Decimal(str(regional_multiplier)) - 1)

        # Calculate final cost
        final_cost = (
            base_cost
            + optimization_adjustment
            + complexity_adjustment
            + modal_adjustment
            + regional_adjustment
            - volume_discount
        )

        # Apply currency conversion
        if currency != "USD":
            currency_rate = self.config.currency_rates.get(currency, 1.0)
            final_cost = final_cost * Decimal(str(currency_rate))

        # Calculate potential savings
        standard_cost = base_cost * Decimal(
            str(self.config.complexity_multipliers["moderate"])
        )
        potential_savings = max(standard_cost - final_cost, Decimal("0"))

        # Create cost breakdown
        breakdown = SkyRouterCostBreakdown(
            base_cost=base_cost,
            optimization_adjustment=optimization_adjustment,
            volume_discount=volume_discount,
            complexity_adjustment=complexity_adjustment,
            modal_adjustment=modal_adjustment,
            regional_adjustment=regional_adjustment,
            final_cost=final_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            selected_route=f"{tier}_{route_optimization}",
            optimization_strategy=route_optimization,
            potential_savings=potential_savings,
            efficiency_score=float(potential_savings / standard_cost)
            if standard_cost > 0
            else 0.0,
            currency=currency,
            region=region,
        )

        # Import here to avoid circular import
        from .skyrouter import SkyRouterCostResult

        return SkyRouterCostResult(
            operation_type="model_call",
            model=model,
            route=f"skyrouter_{route_optimization}",
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=final_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            optimization_savings=potential_savings,
            route_efficiency_score=breakdown.efficiency_score,
            metadata={
                "breakdown": breakdown,
                "tier": tier,
                "modal_type": modal_type,
                "region": region,
                "currency": currency,
            },
        )

    def calculate_multi_model_cost(
        self,
        models: list[str],
        input_data: dict[str, Any],
        routing_strategy: str = "cost_optimized",
        region: str = "us-east",
    ) -> "SkyRouterCostResult":  # type: ignore  # noqa: F821
        """Calculate cost for multi-model routing operation."""

        # Calculate cost for each candidate model
        model_costs = []
        for model in models:
            cost_result = self.calculate_model_call_cost(
                model=model,
                input_data=input_data,
                route_optimization=routing_strategy,
                region=region,
            )
            model_costs.append((model, cost_result))

        # Select optimal model based on strategy
        if routing_strategy == "cost_optimized":
            selected_model, selected_result = min(
                model_costs, key=lambda x: x[1].total_cost
            )
        elif routing_strategy == "latency_optimized":
            # Use model priority (first in list for latency)
            selected_model, selected_result = (
                model_costs[0] if model_costs else (models[0], None)
            )
        else:
            # Balanced approach - weighted score
            selected_model, selected_result = self._select_balanced_model(model_costs)

        # Calculate savings from optimization
        if len(model_costs) > 1:
            costs_only = [result.total_cost for _, result in model_costs]
            max_cost = max(costs_only)
            savings = max_cost - selected_result.total_cost
        else:
            savings = Decimal("0")

        # Update result for multi-model operation
        selected_result.operation_type = "multi_model_routing"
        selected_result.route = f"multi_model_{routing_strategy}"
        selected_result.optimization_savings = savings
        selected_result.metadata.update(
            {
                "candidate_models": models,
                "routing_strategy": routing_strategy,
                "model_costs": {
                    model: float(result.total_cost) for model, result in model_costs
                },
            }
        )

        return selected_result

    def calculate_agent_workflow_cost(
        self,
        workflow_name: str,
        agent_steps: list[dict[str, Any]],
        region: str = "us-east",
    ) -> "SkyRouterCostResult":  # type: ignore  # noqa: F821
        """Calculate cost for complete agent workflow."""

        total_input_cost = Decimal("0")
        total_output_cost = Decimal("0")
        total_input_tokens = 0
        total_output_tokens = 0
        total_savings = Decimal("0")

        step_costs = []
        primary_model = "unknown"

        for i, step in enumerate(agent_steps):
            model = step.get("model", "gpt-3.5-turbo")
            if i == 0:
                primary_model = model

            input_data = step.get("input", {})
            complexity = step.get("complexity", "moderate")
            optimization = step.get("optimization", "balanced")

            step_result = self.calculate_model_call_cost(
                model=model,
                input_data=input_data,
                route_optimization=optimization,
                complexity=complexity,
                region=region,
            )

            total_input_cost += step_result.input_cost
            total_output_cost += step_result.output_cost
            total_input_tokens += step_result.input_tokens
            total_output_tokens += step_result.output_tokens
            total_savings += step_result.optimization_savings

            step_costs.append(
                {
                    "step": i + 1,
                    "model": model,
                    "cost": float(step_result.total_cost),
                    "optimization": optimization,
                }
            )

        # Apply workflow-level volume discount
        total_tokens = total_input_tokens + total_output_tokens
        workflow_discount = self._get_volume_discount(total_tokens) * (
            total_input_cost + total_output_cost
        )

        final_cost = total_input_cost + total_output_cost - workflow_discount

        # Import here to avoid circular import
        from .skyrouter import SkyRouterCostResult

        return SkyRouterCostResult(
            operation_type="agent_workflow",
            model=primary_model,
            route=f"agent_workflow_{workflow_name}",
            input_cost=total_input_cost,
            output_cost=total_output_cost,
            total_cost=final_cost,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            optimization_savings=total_savings + workflow_discount,
            route_efficiency_score=float(total_savings / final_cost)
            if final_cost > 0
            else 0.0,
            metadata={
                "workflow_name": workflow_name,
                "step_count": len(agent_steps),
                "step_costs": step_costs,
                "workflow_discount": float(workflow_discount),
                "region": region,
            },
        )

    def _get_model_tier(self, model: str) -> str:
        """Determine tier for a model."""
        model_lower = model.lower().replace("-", "").replace("_", "")

        for known_model, tier in self.model_tiers.items():
            known_lower = known_model.lower().replace("-", "").replace("_", "")
            if known_lower in model_lower:
                return tier

        # Default to standard tier for unknown models
        return "standard"

    def _estimate_input_tokens(self, input_data: dict[str, Any]) -> int:
        """Estimate input tokens from input data."""
        if not input_data:
            return 50

        # Handle common input formats
        if "messages" in input_data:
            # Chat format
            messages = input_data["messages"]
            if isinstance(messages, list):
                text_content = " ".join(str(msg.get("content", "")) for msg in messages)
            else:
                text_content = str(messages)
        elif "prompt" in input_data:
            # Direct prompt format
            text_content = str(input_data["prompt"])
        else:
            # Generic text extraction
            text_content = str(input_data)

        # Estimate tokens (roughly 0.75 tokens per word)
        words = len(text_content.split())
        estimated_tokens = int(words * 1.3)

        return max(estimated_tokens, 10)

    def _estimate_output_tokens(
        self, input_data: dict[str, Any], complexity: str
    ) -> int:
        """Estimate output tokens based on input and complexity."""
        input_tokens = self._estimate_input_tokens(input_data)

        # Base output ratio by complexity
        complexity_ratios = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 2.0,
            "enterprise": 3.0,
        }

        ratio = complexity_ratios.get(complexity, 1.0)
        base_output = max(int(input_tokens * ratio), 20)

        # Add some variability based on input size
        if input_tokens > 1000:
            base_output = int(
                base_output * 1.2
            )  # Longer inputs often need longer outputs

        return base_output

    def _detect_modal_type(self, input_data: dict[str, Any]) -> str:
        """Detect the modal type of input data."""
        if not isinstance(input_data, dict):
            return "text"

        # Check for vision/image data
        if any(key in input_data for key in ["image", "images", "vision", "visual"]):
            return "vision"

        # Check for audio data
        if any(key in input_data for key in ["audio", "speech", "voice"]):
            return "audio"

        # Check for multimodal indicators
        if any(key in input_data for key in ["multimodal", "multimedia", "mixed"]):
            return "multimodal"

        return "text"

    def _get_volume_discount(self, token_count: int) -> float:
        """Calculate volume discount based on token usage."""
        # Use cumulative monthly volume
        cumulative_volume = self.monthly_volume + token_count

        # Find applicable discount tier
        applicable_discount = 0.0
        for threshold, discount in sorted(self.config.volume_tiers.items()):
            if cumulative_volume >= threshold:
                applicable_discount = discount
            else:
                break

        return applicable_discount

    def _select_balanced_model(
        self, model_costs: list[tuple[str, Any]]
    ) -> tuple[str, Any]:
        """Select model based on balanced cost/performance score."""
        if not model_costs:
            return "unknown", None

        # Simple balanced selection - could be enhanced with performance metrics
        scores = []
        for model, result in model_costs:
            # Weight: 70% cost, 30% efficiency
            cost_score = 1.0 / float(result.total_cost) if result.total_cost > 0 else 0
            efficiency_score = result.route_efficiency_score
            balanced_score = 0.7 * cost_score + 0.3 * efficiency_score
            scores.append((balanced_score, model, result))

        # Select highest scoring model
        best_score, selected_model, selected_result = max(scores, key=lambda x: x[0])
        return selected_model, selected_result

    def update_monthly_volume(self, token_count: int):
        """Update monthly volume for discount calculations."""
        self.monthly_volume += token_count
        self.current_discount = self._get_volume_discount(self.monthly_volume)
        logger.debug(
            f"Updated monthly volume: {self.monthly_volume}, current discount: {self.current_discount:.1%}"
        )

    def get_volume_discount_info(self) -> dict[str, Any]:
        """Get current volume discount information."""
        current_discount = self._get_volume_discount(self.monthly_volume)

        # Find next discount tier
        next_threshold = None
        next_discount = None

        for threshold, discount in sorted(self.config.volume_tiers.items()):
            if self.monthly_volume < threshold:
                next_threshold = threshold
                next_discount = discount
                break

        return {
            "monthly_volume": self.monthly_volume,
            "current_discount_percentage": current_discount * 100,
            "next_threshold": next_threshold,
            "next_discount_percentage": next_discount * 100 if next_discount else None,
            "tokens_to_next_tier": next_threshold - self.monthly_volume
            if next_threshold
            else 0,
        }

    def estimate_monthly_cost(
        self,
        daily_operations: int,
        avg_tokens_per_operation: int,
        model_distribution: dict[str, float],
        optimization_strategy: str = "balanced",
    ) -> dict[str, Any]:
        """Estimate monthly cost based on usage patterns."""

        monthly_operations = daily_operations * 30
        monthly_tokens = monthly_operations * avg_tokens_per_operation

        # Calculate weighted cost per operation
        weighted_cost = Decimal("0")
        for model, percentage in model_distribution.items():
            sample_input = {"prompt": "Sample prompt for cost estimation"}
            cost_result = self.calculate_model_call_cost(
                model=model,
                input_data=sample_input,
                route_optimization=optimization_strategy,
            )
            weighted_cost += cost_result.total_cost * Decimal(str(percentage))

        base_monthly_cost = weighted_cost * monthly_operations

        # Apply volume discount
        volume_discount_rate = self._get_volume_discount(monthly_tokens)
        volume_discount_amount = base_monthly_cost * Decimal(str(volume_discount_rate))

        final_monthly_cost = base_monthly_cost - volume_discount_amount

        return {
            "monthly_operations": monthly_operations,
            "monthly_tokens": monthly_tokens,
            "base_monthly_cost": float(base_monthly_cost),
            "volume_discount_amount": float(volume_discount_amount),
            "final_monthly_cost": float(final_monthly_cost),
            "cost_per_operation": float(final_monthly_cost / monthly_operations),
            "cost_per_token": float(final_monthly_cost / monthly_tokens)
            if monthly_tokens > 0
            else 0,
            "volume_discount_percentage": volume_discount_rate * 100,
        }
