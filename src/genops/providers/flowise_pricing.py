"""Flowise pricing and cost calculation for GenOps AI governance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class FlowiseExecutionCost:
    """Represents the cost of a single Flowise flow execution."""

    flow_id: str
    flow_name: str
    execution_id: str | None = None
    base_execution_cost: Decimal = Decimal("0.001")  # Base cost per execution
    token_costs: dict[str, Decimal] = None  # type: ignore  # Costs from underlying LLM providers
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost: Decimal = Decimal("0.0")
    provider_costs: dict[str, Decimal] = None  # type: ignore  # Costs by provider (OpenAI, Anthropic, etc.)
    execution_duration_ms: int = 0

    def __post_init__(self):
        if self.token_costs is None:
            self.token_costs = {}
        if self.provider_costs is None:
            self.provider_costs = {}

    def add_provider_cost(self, provider: str, cost: Decimal) -> None:
        """Add cost from an underlying LLM provider."""
        self.provider_costs[provider] = (
            self.provider_costs.get(provider, Decimal("0.0")) + cost
        )
        self._recalculate_total()

    def add_token_cost(
        self, model: str, input_tokens: int, output_tokens: int, cost: Decimal
    ) -> None:
        """Add token-based cost from a model."""
        self.token_costs[model] = cost
        self.total_tokens_input += input_tokens
        self.total_tokens_output += output_tokens
        self._recalculate_total()

    def _recalculate_total(self) -> None:
        """Recalculate total cost from all components."""
        provider_total = sum(self.provider_costs.values())
        token_total = sum(self.token_costs.values())
        self.total_cost = self.base_execution_cost + provider_total + token_total


@dataclass
class FlowisePricingTier:
    """Represents a Flowise pricing tier or deployment model."""

    name: str
    base_cost_per_execution: Decimal
    included_executions_per_month: int
    overage_cost_per_execution: Decimal
    max_executions_per_month: int | None = None
    description: str = ""


# Common Flowise pricing tiers
FLOWISE_PRICING_TIERS = {
    "self_hosted": FlowisePricingTier(
        name="Self-Hosted",
        base_cost_per_execution=Decimal("0.0"),  # No Flowise platform costs
        included_executions_per_month=999999999,  # Unlimited executions
        overage_cost_per_execution=Decimal("0.0"),
        description="Self-hosted Flowise instance - only underlying provider costs apply",
    ),
    "cloud_free": FlowisePricingTier(
        name="Flowise Cloud Free",
        base_cost_per_execution=Decimal("0.0"),
        included_executions_per_month=200,
        overage_cost_per_execution=Decimal("0.01"),  # Example pricing
        max_executions_per_month=200,
        description="Flowise Cloud free tier with execution limits",
    ),
    "cloud_starter": FlowisePricingTier(
        name="Flowise Cloud Starter",
        base_cost_per_execution=Decimal("0.001"),
        included_executions_per_month=10000,
        overage_cost_per_execution=Decimal("0.001"),
        description="Flowise Cloud starter plan for small applications",
    ),
    "cloud_pro": FlowisePricingTier(
        name="Flowise Cloud Pro",
        base_cost_per_execution=Decimal("0.0008"),
        included_executions_per_month=100000,
        overage_cost_per_execution=Decimal("0.0008"),
        description="Flowise Cloud professional plan for production applications",
    ),
    "cloud_enterprise": FlowisePricingTier(
        name="Flowise Cloud Enterprise",
        base_cost_per_execution=Decimal("0.0005"),
        included_executions_per_month=1000000,
        overage_cost_per_execution=Decimal("0.0005"),
        description="Flowise Cloud enterprise plan with volume discounts",
    ),
}


class FlowiseCostCalculator:
    """Cost calculator for Flowise flow executions with multi-provider support."""

    def __init__(
        self, pricing_tier: str = "self_hosted", monthly_execution_count: int = 0
    ):
        """
        Initialize cost calculator with pricing tier.

        Args:
            pricing_tier: Pricing tier name from FLOWISE_PRICING_TIERS
            monthly_execution_count: Current execution count for the month (for overage calculation)
        """
        if pricing_tier not in FLOWISE_PRICING_TIERS:
            logger.warning(
                f"Unknown pricing tier '{pricing_tier}', defaulting to 'self_hosted'"
            )
            pricing_tier = "self_hosted"

        self.pricing_tier = FLOWISE_PRICING_TIERS[pricing_tier]
        self.monthly_execution_count = monthly_execution_count

        # Import provider cost calculators as needed
        self._provider_calculators = {}
        self._load_provider_calculators()

    def _load_provider_calculators(self):
        """Load cost calculators for supported providers."""
        try:
            from genops.providers.openai_pricing import OpenAICostCalculator

            self._provider_calculators["openai"] = OpenAICostCalculator()
        except ImportError:
            logger.debug("OpenAI pricing not available")

        try:
            from genops.providers.anthropic_pricing import AnthropicCostCalculator

            self._provider_calculators["anthropic"] = AnthropicCostCalculator()
        except ImportError:
            logger.debug("Anthropic pricing not available")

        try:
            from genops.providers.gemini_pricing import GeminiCostCalculator

            self._provider_calculators["gemini"] = GeminiCostCalculator()
        except ImportError:
            logger.debug("Gemini pricing not available")

    def calculate_execution_cost(
        self,
        flow_id: str,
        flow_name: str,
        underlying_provider_calls: list[dict] | None = None,
        execution_id: str | None = None,
        execution_duration_ms: int = 0,
    ) -> FlowiseExecutionCost:
        """
        Calculate the cost of a single flow execution.

        Args:
            flow_id: Unique identifier for the flow
            flow_name: Human-readable flow name
            underlying_provider_calls: List of provider API calls made during execution
            execution_id: Optional execution identifier
            execution_duration_ms: Execution duration in milliseconds

        Returns:
            FlowiseExecutionCost: Detailed cost breakdown

        Example:
            provider_calls = [
                {
                    'provider': 'openai',
                    'model': 'gpt-4',
                    'input_tokens': 100,
                    'output_tokens': 50,
                    'cost': 0.006  # Pre-calculated or to be calculated
                }
            ]
            cost = calculator.calculate_execution_cost('flow-123', 'Customer Support Bot', provider_calls)
        """

        # Determine base execution cost based on tier and usage
        base_cost = self._calculate_base_execution_cost()

        # Initialize cost object
        execution_cost = FlowiseExecutionCost(
            flow_id=flow_id,
            flow_name=flow_name,
            execution_id=execution_id,
            base_execution_cost=base_cost,
            execution_duration_ms=execution_duration_ms,
        )

        # Calculate costs from underlying provider calls
        if underlying_provider_calls:
            for call in underlying_provider_calls:
                self._add_provider_call_cost(execution_cost, call)

        return execution_cost

    def _calculate_base_execution_cost(self) -> Decimal:
        """Calculate base Flowise execution cost based on pricing tier and usage."""
        # Check if we're in the included executions range
        if (
            self.monthly_execution_count
            < self.pricing_tier.included_executions_per_month
        ):
            return self.pricing_tier.base_cost_per_execution
        else:
            # Using overage pricing
            return self.pricing_tier.overage_cost_per_execution

    def _add_provider_call_cost(
        self, execution_cost: FlowiseExecutionCost, provider_call: dict
    ) -> None:
        """Add cost from an underlying provider API call."""
        provider = provider_call.get("provider", "").lower()
        model = provider_call.get("model", "unknown")
        input_tokens = provider_call.get("input_tokens", 0)
        output_tokens = provider_call.get("output_tokens", 0)

        # Try to calculate cost if not provided
        if "cost" in provider_call:
            cost = Decimal(str(provider_call["cost"]))
        else:
            cost = self._calculate_provider_cost(
                provider, model, input_tokens, output_tokens
            )

        # Add to execution cost
        execution_cost.add_provider_cost(provider, cost)
        execution_cost.add_token_cost(
            f"{provider}-{model}", input_tokens, output_tokens, cost
        )

    def _calculate_provider_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Calculate cost for a provider API call."""
        if provider in self._provider_calculators:
            try:
                calculator = self._provider_calculators[provider]
                # Different providers have different interfaces - adapt as needed
                if hasattr(calculator, "calculate_cost"):
                    return Decimal(
                        str(
                            calculator.calculate_cost(
                                model, input_tokens, output_tokens
                            )
                        )
                    )
                else:
                    logger.debug(
                        f"Provider calculator for {provider} doesn't have calculate_cost method"
                    )
            except Exception as e:
                logger.debug(f"Error calculating {provider} cost: {e}")

        # Fallback to generic cost estimation
        return self._estimate_generic_cost(model, input_tokens, output_tokens)

    def _estimate_generic_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        """Generic cost estimation when provider-specific calculators aren't available."""
        # Generic pricing estimates based on common model patterns
        model_lower = model.lower()

        if "gpt-4" in model_lower:
            # GPT-4 family pricing estimate
            input_rate = Decimal("0.00003")  # $0.03 per 1k tokens
            output_rate = Decimal("0.00006")  # $0.06 per 1k tokens
        elif "gpt-3.5" in model_lower:
            # GPT-3.5 family pricing estimate
            input_rate = Decimal("0.000001")  # $0.001 per 1k tokens
            output_rate = Decimal("0.000002")  # $0.002 per 1k tokens
        elif "claude" in model_lower:
            # Claude family pricing estimate
            if "opus" in model_lower:
                input_rate = Decimal("0.000015")  # $0.015 per 1k tokens
                output_rate = Decimal("0.000075")  # $0.075 per 1k tokens
            elif "sonnet" in model_lower:
                input_rate = Decimal("0.000003")  # $0.003 per 1k tokens
                output_rate = Decimal("0.000015")  # $0.015 per 1k tokens
            else:  # haiku
                input_rate = Decimal("0.00000025")  # $0.00025 per 1k tokens
                output_rate = Decimal("0.00000125")  # $0.00125 per 1k tokens
        elif "gemini" in model_lower:
            # Gemini family pricing estimate
            input_rate = Decimal("0.000001")  # $0.001 per 1k tokens (example)
            output_rate = Decimal("0.000002")  # $0.002 per 1k tokens (example)
        else:
            # Generic fallback
            logger.debug(f"Unknown model {model}, using generic pricing")
            input_rate = Decimal("0.000002")  # $0.002 per 1k tokens
            output_rate = Decimal("0.000004")  # $0.004 per 1k tokens

        input_cost = (Decimal(input_tokens) / 1000) * input_rate
        output_cost = (Decimal(output_tokens) / 1000) * output_rate

        return input_cost + output_cost

    def calculate_monthly_costs(
        self, execution_costs: list[FlowiseExecutionCost]
    ) -> dict[str, Decimal | dict[str, Decimal]]:
        """
        Calculate monthly cost summary from a list of execution costs.

        Args:
            execution_costs: List of FlowiseExecutionCost objects

        Returns:
            Dict with cost breakdown by various dimensions
        """
        total_cost = Decimal("0.0")
        total_executions = len(execution_costs)
        costs_by_flow = {}
        costs_by_provider = {}
        total_tokens_input = 0
        total_tokens_output = 0

        for cost in execution_costs:
            total_cost += cost.total_cost
            total_tokens_input += cost.total_tokens_input
            total_tokens_output += cost.total_tokens_output

            # Group by flow
            flow_key = f"{cost.flow_name} ({cost.flow_id})"
            costs_by_flow[flow_key] = (
                costs_by_flow.get(flow_key, Decimal("0.0")) + cost.total_cost
            )

            # Group by provider
            for provider, provider_cost in cost.provider_costs.items():
                costs_by_provider[provider] = (
                    costs_by_provider.get(provider, Decimal("0.0")) + provider_cost
                )

        return {
            "total_cost": total_cost,
            "total_executions": total_executions,
            "average_cost_per_execution": total_cost / total_executions
            if total_executions > 0
            else Decimal("0.0"),
            "costs_by_flow": costs_by_flow,
            "costs_by_provider": costs_by_provider,
            "total_tokens_input": total_tokens_input,
            "total_tokens_output": total_tokens_output,
            "total_tokens": total_tokens_input + total_tokens_output,
            "pricing_tier": self.pricing_tier.name,
        }

    def estimate_monthly_spend(
        self,
        expected_executions_per_month: int,
        average_tokens_per_execution: int = 1000,
        provider_distribution: dict[str, float] | None = None,
    ) -> dict[str, Decimal]:
        """
        Estimate monthly spending based on expected usage.

        Args:
            expected_executions_per_month: Expected number of flow executions
            average_tokens_per_execution: Average tokens per execution
            provider_distribution: Distribution of usage across providers (e.g., {'openai': 0.7, 'anthropic': 0.3})

        Returns:
            Dict with cost estimates
        """
        if provider_distribution is None:
            provider_distribution = {"generic": 1.0}

        # Calculate base Flowise platform costs
        if (
            expected_executions_per_month
            <= self.pricing_tier.included_executions_per_month
        ):
            flowise_cost = (
                Decimal(expected_executions_per_month)
                * self.pricing_tier.base_cost_per_execution
            )
        else:
            included_cost = (
                Decimal(self.pricing_tier.included_executions_per_month)
                * self.pricing_tier.base_cost_per_execution
            )
            overage_executions = (
                expected_executions_per_month
                - self.pricing_tier.included_executions_per_month
            )
            overage_cost = (
                Decimal(overage_executions)
                * self.pricing_tier.overage_cost_per_execution
            )
            flowise_cost = included_cost + overage_cost

        # Estimate provider costs
        provider_costs = {}
        total_provider_cost = Decimal("0.0")

        for provider, distribution in provider_distribution.items():
            provider_executions = int(expected_executions_per_month * distribution)
            provider_tokens = provider_executions * average_tokens_per_execution

            # Estimate cost per token for provider (rough estimates)
            if provider == "openai":
                cost_per_token = Decimal("0.000002")  # Rough average
            elif provider == "anthropic":
                cost_per_token = Decimal("0.000008")  # Rough average
            elif provider == "gemini":
                cost_per_token = Decimal("0.0000015")  # Rough average
            else:
                cost_per_token = Decimal("0.000003")  # Generic estimate

            provider_cost = Decimal(provider_tokens) * cost_per_token
            provider_costs[provider] = provider_cost
            total_provider_cost += provider_cost

        total_estimated_cost = flowise_cost + total_provider_cost

        return {
            "total_estimated_cost": total_estimated_cost,
            "flowise_platform_cost": flowise_cost,
            "total_provider_costs": total_provider_cost,
            "provider_cost_breakdown": provider_costs,
            "expected_executions": expected_executions_per_month,
            "pricing_tier": self.pricing_tier.name,
        }

    def get_pricing_tier_info(self) -> dict[str, any]:
        """Get information about the current pricing tier."""
        return {
            "name": self.pricing_tier.name,
            "base_cost_per_execution": float(self.pricing_tier.base_cost_per_execution),
            "included_executions_per_month": self.pricing_tier.included_executions_per_month,
            "overage_cost_per_execution": float(
                self.pricing_tier.overage_cost_per_execution
            ),
            "max_executions_per_month": self.pricing_tier.max_executions_per_month,
            "description": self.pricing_tier.description,
            "current_monthly_execution_count": self.monthly_execution_count,
        }


# Convenience function for quick cost calculations
def calculate_flow_execution_cost(
    flow_id: str,
    flow_name: str,
    provider_calls: list[dict] | None = None,
    pricing_tier: str = "self_hosted",
) -> FlowiseExecutionCost:
    """
    Quick cost calculation for a single flow execution.

    Args:
        flow_id: Flow identifier
        flow_name: Flow name
        provider_calls: List of underlying provider calls
        pricing_tier: Flowise pricing tier

    Returns:
        FlowiseExecutionCost: Calculated cost

    Example:
        cost = calculate_flow_execution_cost(
            'chatbot-v1',
            'Customer Support Chatbot',
            [{'provider': 'openai', 'model': 'gpt-4', 'input_tokens': 100, 'output_tokens': 50}]
        )
        print(f"Total cost: ${cost.total_cost:.6f}")
    """
    calculator = FlowiseCostCalculator(pricing_tier=pricing_tier)
    return calculator.calculate_execution_cost(flow_id, flow_name, provider_calls)


# Cost optimization utilities
def analyze_cost_optimization_opportunities(
    execution_costs: list[FlowiseExecutionCost],
) -> dict[str, any]:
    """
    Analyze execution costs to identify optimization opportunities.

    Args:
        execution_costs: List of execution costs to analyze

    Returns:
        Dict with optimization recommendations
    """
    if not execution_costs:
        return {"recommendations": [], "total_potential_savings": Decimal("0.0")}

    # Analyze cost patterns
    total_cost = sum(cost.total_cost for cost in execution_costs)
    provider_costs = {}
    flow_costs = {}

    for cost in execution_costs:
        # Aggregate by provider
        for provider, provider_cost in cost.provider_costs.items():
            provider_costs[provider] = (
                provider_costs.get(provider, Decimal("0.0")) + provider_cost
            )

        # Aggregate by flow
        flow_costs[cost.flow_id] = (
            flow_costs.get(cost.flow_id, Decimal("0.0")) + cost.total_cost
        )

    recommendations = []
    potential_savings = Decimal("0.0")

    # Identify expensive providers
    if provider_costs:
        most_expensive_provider = max(provider_costs.items(), key=lambda x: x[1])
        provider_name, provider_cost = most_expensive_provider

        if provider_cost > total_cost * Decimal("0.6"):  # More than 60% of total cost
            recommendations.append(
                {
                    "type": "provider_optimization",
                    "provider": provider_name,
                    "current_cost": float(provider_cost),
                    "suggestion": f"Consider switching some workloads from {provider_name} to a more cost-effective provider",
                    "potential_savings_percent": 20,  # Estimate
                }
            )
            potential_savings += provider_cost * Decimal("0.2")

    # Identify expensive flows
    if flow_costs:
        most_expensive_flow = max(flow_costs.items(), key=lambda x: x[1])
        flow_id, flow_cost = most_expensive_flow

        if flow_cost > total_cost * Decimal("0.4"):  # More than 40% of total cost
            recommendations.append(
                {
                    "type": "flow_optimization",
                    "flow_id": flow_id,
                    "current_cost": float(flow_cost),
                    "suggestion": f"Flow {flow_id} is consuming a large portion of budget - consider optimizing prompts or model selection",
                    "potential_savings_percent": 15,  # Estimate
                }
            )
            potential_savings += flow_cost * Decimal("0.15")

    # Suggest token optimization
    avg_tokens = sum(
        cost.total_tokens_input + cost.total_tokens_output for cost in execution_costs
    ) / len(execution_costs)
    if avg_tokens > 2000:  # High token usage
        recommendations.append(
            {
                "type": "token_optimization",
                "average_tokens_per_execution": int(avg_tokens),
                "suggestion": "High token usage detected - consider prompt optimization or response length limits",
                "potential_savings_percent": 25,
            }
        )
        potential_savings += total_cost * Decimal("0.25")

    return {
        "recommendations": recommendations,
        "total_potential_savings": potential_savings,
        "total_analyzed_cost": total_cost,
        "analysis_period_executions": len(execution_costs),
        "cost_breakdown": {
            "by_provider": {k: float(v) for k, v in provider_costs.items()},
            "by_flow": {k: float(v) for k, v in flow_costs.items()},
        },
    }


@dataclass
class CostOptimizationRecommendation:
    """A cost optimization recommendation."""

    recommendation_type: str
    description: str
    current_model: str | None = None
    suggested_model: str | None = None
    estimated_savings: Decimal = Decimal("0")
    confidence_score: float = 0.0
    potential_tradeoffs: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.confidence_score < 0.0 or self.confidence_score > 1.0:
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )


def calculate_flowise_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str = "gpt-3.5-turbo",
    pricing_tier: str | None = None,
    cost_multiplier: Decimal = Decimal("1"),
) -> Decimal:
    """Calculate cost for a Flowise flow execution."""
    calculator = FlowiseCostCalculator(pricing_tier=pricing_tier or "self_hosted")
    cost = calculator._estimate_generic_cost(model_name, input_tokens, output_tokens)
    return cost * cost_multiplier


def estimate_flowise_tokens(text: str) -> int:
    """Estimate token count for a text string (rough approximation)."""
    # Rough heuristic: ~4 chars per token for English
    return max(1, len(text) // 4)


def get_model_pricing_info(model_name: str) -> dict:
    """Get pricing information for a model."""
    calculator = FlowiseCostCalculator()
    # Use generic estimation to get approximate rates
    input_cost = float(calculator._estimate_generic_cost(model_name, 1000, 0))
    output_cost = float(calculator._estimate_generic_cost(model_name, 0, 1000))
    return {
        "model": model_name,
        "input_cost_per_1k": input_cost,
        "output_cost_per_1k": output_cost,
    }


def calculate_bulk_costs(requests_list: list[dict]) -> list[dict]:
    """Calculate costs for multiple requests at once."""
    results = []
    for req in requests_list:
        cost = calculate_flowise_cost(
            input_tokens=req.get("input_tokens", 0),
            output_tokens=req.get("output_tokens", 0),
            model_name=req.get("model_name", "gpt-3.5-turbo"),
        )
        results.append(
            {
                "cost": cost,
                "model": req.get("model_name", "gpt-3.5-turbo"),
                "input_tokens": req.get("input_tokens", 0),
                "output_tokens": req.get("output_tokens", 0),
            }
        )
    return results


def get_cost_optimization_recommendations(
    current_model: str = "gpt-4",
    current_cost: Decimal = Decimal("0"),
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> list[CostOptimizationRecommendation]:
    """Generate cost optimization recommendations."""
    recs = []
    if current_model in ("gpt-4", "gpt-4-turbo"):
        recs.append(
            CostOptimizationRecommendation(
                recommendation_type="model_switch",
                current_model=current_model,
                suggested_model="gpt-3.5-turbo",
                estimated_savings=current_cost * Decimal("0.5"),
                confidence_score=0.8,
                description="Consider using gpt-3.5-turbo for simpler tasks",
                potential_tradeoffs=["Slightly reduced quality for complex reasoning"],
            )
        )
    if input_tokens > 2000:
        recs.append(
            CostOptimizationRecommendation(
                recommendation_type="token_optimization",
                description="Optimize input prompts to reduce token usage",
                estimated_savings=current_cost * Decimal("0.2"),
                confidence_score=0.7,
            )
        )
    return recs
