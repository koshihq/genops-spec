"""
Perplexity AI Pricing Calculator

Implements Perplexity's unique dual pricing model:
1. Token costs (per 1M tokens) - varies by model and token type
2. Request costs (per 1K requests) - varies by search context depth

This calculator handles:
- Complex model-specific pricing tiers
- Search context-dependent request fees
- Volume optimization analysis
- Cost forecasting and budget planning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PerplexityModel(Enum):
    """Perplexity AI models with pricing characteristics."""

    SONAR = "sonar"
    SONAR_PRO = "sonar-pro"
    SONAR_REASONING = "sonar-reasoning"
    SONAR_REASONING_PRO = "sonar-reasoning-pro"
    SONAR_DEEP_RESEARCH = "sonar-deep-research"


class SearchContext(Enum):
    """Search context depth levels affecting request pricing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TokenPricing:
    """Token pricing structure for a model."""

    input_per_million: Decimal
    output_per_million: Decimal
    citation_per_million: Decimal | None = None
    reasoning_per_million: Decimal | None = None


@dataclass
class RequestPricing:
    """Request pricing structure by search context."""

    low_context_per_thousand: Decimal
    medium_context_per_thousand: Decimal
    high_context_per_thousand: Decimal


@dataclass
class SearchCostBreakdown:
    """Detailed cost breakdown for a search operation."""

    model: str
    tokens_used: int
    search_context: str
    token_cost: Decimal
    request_cost: Decimal
    total_cost: Decimal
    cost_per_token: Decimal
    pricing_details: dict[str, Any]


@dataclass
class CostAnalysis:
    """Cost analysis with optimization recommendations."""

    current_cost_structure: dict[str, Any]
    projected_costs: dict[str, Decimal]
    optimization_opportunities: list[dict[str, Any]]
    budget_analysis: dict[str, Any]
    recommendations: list[str]


class PerplexityPricingCalculator:
    """
    Comprehensive pricing calculator for Perplexity AI operations.

    Handles Perplexity's unique dual pricing model:
    - Token costs that vary by model and token type
    - Request fees that depend on search context depth
    - Volume discounts and optimization analysis
    """

    def __init__(self):
        """Initialize the pricing calculator with current Perplexity rates."""

        # Token pricing by model (per 1M tokens)
        self.token_pricing = {
            PerplexityModel.SONAR.value: TokenPricing(
                input_per_million=Decimal("1.00"), output_per_million=Decimal("1.00")
            ),
            PerplexityModel.SONAR_PRO.value: TokenPricing(
                input_per_million=Decimal("3.00"), output_per_million=Decimal("15.00")
            ),
            PerplexityModel.SONAR_REASONING.value: TokenPricing(
                input_per_million=Decimal("1.00"),
                output_per_million=Decimal("5.00"),
                reasoning_per_million=Decimal("5.00"),
            ),
            PerplexityModel.SONAR_REASONING_PRO.value: TokenPricing(
                input_per_million=Decimal("2.00"),
                output_per_million=Decimal("8.00"),
                reasoning_per_million=Decimal("8.00"),
            ),
            PerplexityModel.SONAR_DEEP_RESEARCH.value: TokenPricing(
                input_per_million=Decimal("5.00"),
                output_per_million=Decimal("20.00"),
                citation_per_million=Decimal("1.00"),
                reasoning_per_million=Decimal("10.00"),
            ),
        }

        # Request pricing by model and search context (per 1K requests)
        self.request_pricing = {
            PerplexityModel.SONAR.value: RequestPricing(
                low_context_per_thousand=Decimal("5.00"),
                medium_context_per_thousand=Decimal("8.00"),
                high_context_per_thousand=Decimal("12.00"),
            ),
            PerplexityModel.SONAR_PRO.value: RequestPricing(
                low_context_per_thousand=Decimal("7.00"),
                medium_context_per_thousand=Decimal("10.00"),
                high_context_per_thousand=Decimal("14.00"),
            ),
            PerplexityModel.SONAR_REASONING.value: RequestPricing(
                low_context_per_thousand=Decimal("6.00"),
                medium_context_per_thousand=Decimal("9.00"),
                high_context_per_thousand=Decimal("13.00"),
            ),
            PerplexityModel.SONAR_REASONING_PRO.value: RequestPricing(
                low_context_per_thousand=Decimal("8.00"),
                medium_context_per_thousand=Decimal("11.00"),
                high_context_per_thousand=Decimal("15.00"),
            ),
            PerplexityModel.SONAR_DEEP_RESEARCH.value: RequestPricing(
                low_context_per_thousand=Decimal("10.00"),
                medium_context_per_thousand=Decimal("15.00"),
                high_context_per_thousand=Decimal("20.00"),
            ),
        }

        # Search API pricing (separate from chat completions)
        self.search_api_flat_rate = Decimal("5.00")  # per 1K requests, no token costs

        logger.info("Perplexity pricing calculator initialized with current rates")

    def calculate_token_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        citation_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> Decimal:
        """
        Calculate token costs for a given model and token usage.

        Args:
            model: Perplexity model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            citation_tokens: Number of citation tokens (for supported models)
            reasoning_tokens: Number of reasoning tokens (for supported models)

        Returns:
            Total token cost as Decimal
        """
        if model not in self.token_pricing:
            logger.warning(f"Unknown model {model}, using Sonar pricing")
            model = PerplexityModel.SONAR.value

        pricing = self.token_pricing[model]
        total_cost = Decimal("0")

        # Input tokens
        if input_tokens > 0:
            input_cost = (
                Decimal(str(input_tokens)) / Decimal("1000000")
            ) * pricing.input_per_million
            total_cost += input_cost

        # Output tokens
        if output_tokens > 0:
            output_cost = (
                Decimal(str(output_tokens)) / Decimal("1000000")
            ) * pricing.output_per_million
            total_cost += output_cost

        # Citation tokens (for supported models)
        if citation_tokens > 0 and pricing.citation_per_million:
            citation_cost = (
                Decimal(str(citation_tokens)) / Decimal("1000000")
            ) * pricing.citation_per_million
            total_cost += citation_cost

        # Reasoning tokens (for supported models)
        if reasoning_tokens > 0 and pricing.reasoning_per_million:
            reasoning_cost = (
                Decimal(str(reasoning_tokens)) / Decimal("1000000")
            ) * pricing.reasoning_per_million
            total_cost += reasoning_cost

        return total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def calculate_request_cost(
        self,
        model: str,
        search_context: str | SearchContext,
        request_count: int = 1,
    ) -> Decimal:
        """
        Calculate request costs based on model and search context.

        Args:
            model: Perplexity model name
            search_context: Search context depth (low/medium/high)
            request_count: Number of requests

        Returns:
            Total request cost as Decimal
        """
        if model not in self.request_pricing:
            logger.warning(f"Unknown model {model}, using Sonar pricing")
            model = PerplexityModel.SONAR.value

        pricing = self.request_pricing[model]

        # Normalize search context
        if isinstance(search_context, SearchContext):
            context = search_context.value
        else:
            context = str(search_context).lower()

        # Get request rate based on context
        if context == SearchContext.LOW.value:
            rate_per_thousand = pricing.low_context_per_thousand
        elif context == SearchContext.MEDIUM.value:
            rate_per_thousand = pricing.medium_context_per_thousand
        elif context == SearchContext.HIGH.value:
            rate_per_thousand = pricing.high_context_per_thousand
        else:
            logger.warning(f"Unknown search context {context}, using medium")
            rate_per_thousand = pricing.medium_context_per_thousand

        # Calculate cost
        request_cost = (
            Decimal(str(request_count)) / Decimal("1000")
        ) * rate_per_thousand
        return request_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def calculate_search_cost(
        self,
        model: str,
        tokens_used: int,
        search_context: str | SearchContext,
        input_token_ratio: float = 0.3,  # Approximate input/output ratio
        citation_token_ratio: float = 0.1,  # Citations as % of total tokens
        reasoning_token_ratio: float = 0.0,  # Reasoning tokens (model-dependent)
    ) -> Decimal:
        """
        Calculate total cost for a search operation (tokens + request fee).

        Args:
            model: Perplexity model name
            tokens_used: Total tokens used in the operation
            search_context: Search context depth
            input_token_ratio: Ratio of tokens that are input tokens
            citation_token_ratio: Ratio of tokens that are citation tokens
            reasoning_token_ratio: Ratio of tokens that are reasoning tokens

        Returns:
            Total search cost as Decimal
        """
        # Estimate token breakdown
        input_tokens = int(tokens_used * input_token_ratio)
        output_tokens = int(
            tokens_used
            * (1 - input_token_ratio - citation_token_ratio - reasoning_token_ratio)
        )
        citation_tokens = int(tokens_used * citation_token_ratio)
        reasoning_tokens = int(tokens_used * reasoning_token_ratio)

        # Calculate token cost
        token_cost = self.calculate_token_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            citation_tokens=citation_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        # Calculate request cost
        request_cost = self.calculate_request_cost(
            model=model, search_context=search_context, request_count=1
        )

        total_cost = token_cost + request_cost
        return total_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def estimate_search_cost(
        self,
        model: str,
        estimated_tokens: int,
        search_context: str | SearchContext,
    ) -> Decimal:
        """
        Estimate cost for a search operation before execution.

        Args:
            model: Perplexity model name
            estimated_tokens: Estimated token usage
            search_context: Search context depth

        Returns:
            Estimated cost as Decimal
        """
        return self.calculate_search_cost(
            model=model, tokens_used=estimated_tokens, search_context=search_context
        )

    def get_detailed_cost_breakdown(
        self, model: str, tokens_used: int, search_context: str | SearchContext
    ) -> SearchCostBreakdown:
        """
        Get detailed cost breakdown for analysis and reporting.

        Args:
            model: Perplexity model name
            tokens_used: Total tokens used
            search_context: Search context depth

        Returns:
            Detailed cost breakdown
        """
        # Calculate component costs
        token_cost = self.calculate_token_cost(
            model=model,
            input_tokens=int(tokens_used * 0.3),  # Estimated input ratio
            output_tokens=int(tokens_used * 0.6),  # Estimated output ratio
            citation_tokens=int(tokens_used * 0.1),  # Estimated citation ratio
        )

        request_cost = self.calculate_request_cost(
            model=model, search_context=search_context, request_count=1
        )

        total_cost = token_cost + request_cost
        cost_per_token = (
            total_cost / Decimal(str(tokens_used)) if tokens_used > 0 else Decimal("0")
        )

        # Get pricing details
        token_pricing = self.token_pricing.get(
            model, self.token_pricing[PerplexityModel.SONAR.value]
        )
        request_pricing = self.request_pricing.get(
            model, self.request_pricing[PerplexityModel.SONAR.value]
        )

        context_str = (
            search_context.value
            if isinstance(search_context, SearchContext)
            else str(search_context)
        )

        pricing_details = {
            "model": model,
            "token_pricing": {
                "input_per_million": float(token_pricing.input_per_million),
                "output_per_million": float(token_pricing.output_per_million),
                "citation_per_million": float(token_pricing.citation_per_million)
                if token_pricing.citation_per_million
                else None,
                "reasoning_per_million": float(token_pricing.reasoning_per_million)
                if token_pricing.reasoning_per_million
                else None,
            },
            "request_pricing": {
                "low_context": float(request_pricing.low_context_per_thousand),
                "medium_context": float(request_pricing.medium_context_per_thousand),
                "high_context": float(request_pricing.high_context_per_thousand),
            },
            "context_used": context_str,
        }

        return SearchCostBreakdown(
            model=model,
            tokens_used=tokens_used,
            search_context=context_str,
            token_cost=token_cost,
            request_cost=request_cost,
            total_cost=total_cost,
            cost_per_token=cost_per_token,
            pricing_details=pricing_details,
        )

    def analyze_search_costs(
        self,
        projected_queries: int,
        model: str = "sonar",
        average_tokens_per_query: int = 1000,
        search_context: str | SearchContext = SearchContext.MEDIUM,
        current_daily_costs: Decimal | None = None,
        daily_budget_limit: Decimal | None = None,
    ) -> CostAnalysis:
        """
        Analyze projected search costs and provide optimization recommendations.

        Args:
            projected_queries: Number of queries to analyze
            model: Model to analyze
            average_tokens_per_query: Average tokens per query
            search_context: Search context depth
            current_daily_costs: Current daily costs
            daily_budget_limit: Daily budget limit

        Returns:
            Comprehensive cost analysis with recommendations
        """
        # Calculate base costs
        cost_per_query = self.calculate_search_cost(
            model=model,
            tokens_used=average_tokens_per_query,
            search_context=search_context,
        )

        total_projected_cost = cost_per_query * Decimal(str(projected_queries))

        # Analyze different optimization scenarios
        optimization_opportunities = []

        # Model optimization
        if model != PerplexityModel.SONAR.value:
            sonar_cost = self.calculate_search_cost(
                model=PerplexityModel.SONAR.value,
                tokens_used=average_tokens_per_query,
                search_context=search_context,
            )
            sonar_total = sonar_cost * Decimal(str(projected_queries))
            savings = total_projected_cost - sonar_total

            if savings > 0:
                optimization_opportunities.append(
                    {
                        "optimization_type": "model_downgrade",
                        "description": f"Switch from {model} to sonar model",
                        "potential_savings_per_query": float(
                            cost_per_query - sonar_cost
                        ),
                        "potential_savings_total": float(savings),
                        "trade_offs": "Lower accuracy, fewer features",
                        "priority_score": min(
                            100, float(savings / total_projected_cost * 100)
                        ),
                    }
                )

        # Search context optimization
        if search_context != SearchContext.LOW:
            low_context_cost = self.calculate_search_cost(
                model=model,
                tokens_used=average_tokens_per_query,
                search_context=SearchContext.LOW,
            )
            low_context_total = low_context_cost * Decimal(str(projected_queries))
            context_savings = total_projected_cost - low_context_total

            if context_savings > 0:
                optimization_opportunities.append(
                    {
                        "optimization_type": "search_context_reduction",
                        "description": f"Reduce search context from {search_context.value if isinstance(search_context, SearchContext) else search_context} to low",
                        "potential_savings_per_query": float(
                            cost_per_query - low_context_cost
                        ),
                        "potential_savings_total": float(context_savings),
                        "trade_offs": "Less comprehensive search results",
                        "priority_score": min(
                            100, float(context_savings / total_projected_cost * 50)
                        ),
                    }
                )

        # Token optimization
        reduced_tokens = int(average_tokens_per_query * 0.7)  # 30% reduction
        reduced_token_cost = self.calculate_search_cost(
            model=model, tokens_used=reduced_tokens, search_context=search_context
        )
        reduced_token_total = reduced_token_cost * Decimal(str(projected_queries))
        token_savings = total_projected_cost - reduced_token_total

        if token_savings > 0:
            optimization_opportunities.append(
                {
                    "optimization_type": "token_optimization",
                    "description": f"Reduce average tokens per query from {average_tokens_per_query} to {reduced_tokens}",
                    "potential_savings_per_query": float(
                        cost_per_query - reduced_token_cost
                    ),
                    "potential_savings_total": float(token_savings),
                    "trade_offs": "Shorter responses, less detail",
                    "priority_score": min(
                        100, float(token_savings / total_projected_cost * 75)
                    ),
                }
            )

        # Budget analysis
        budget_analysis = {}
        if current_daily_costs is not None and daily_budget_limit is not None:
            remaining_budget = daily_budget_limit - current_daily_costs
            budget_utilization = (
                (current_daily_costs / daily_budget_limit * 100)
                if daily_budget_limit > 0
                else 0
            )

            budget_analysis = {
                "current_daily_costs": float(current_daily_costs),
                "daily_budget_limit": float(daily_budget_limit),
                "remaining_budget": float(remaining_budget),
                "budget_utilization_percent": float(budget_utilization),
                "projected_cost_fits_budget": total_projected_cost <= remaining_budget,
            }

        # Generate recommendations
        recommendations = []

        if optimization_opportunities:
            top_opportunity = max(
                optimization_opportunities,
                key=lambda x: x["priority_score"],  # type: ignore
            )
            recommendations.append(
                f"Consider {top_opportunity['optimization_type']}: {top_opportunity['description']} "
                f"(${top_opportunity['potential_savings_total']:.4f} potential savings)"
            )

        if (
            budget_analysis
            and budget_analysis.get("budget_utilization_percent", 0) > 80
        ):
            recommendations.append(
                "High budget utilization detected. Consider implementing cost controls."
            )

        if projected_queries > 1000:
            recommendations.append(
                "High query volume detected. Consider batch processing or query caching."
            )

        return CostAnalysis(
            current_cost_structure={
                "cost_per_query": float(cost_per_query),
                "projected_total_cost": float(total_projected_cost),
                "model": model,
                "search_context": search_context.value
                if isinstance(search_context, SearchContext)
                else str(search_context),
                "average_tokens": average_tokens_per_query,
                "query_count": projected_queries,
            },
            projected_costs={
                "total": total_projected_cost,
                "per_query": cost_per_query,
                "daily_if_spread_evenly": total_projected_cost
                / Decimal("30"),  # Assume monthly spread
                "monthly": total_projected_cost,
            },
            optimization_opportunities=sorted(
                optimization_opportunities,
                key=lambda x: x["priority_score"],
                reverse=True,
            ),
            budget_analysis=budget_analysis,
            recommendations=recommendations,
        )

    def calculate_search_api_cost(self, request_count: int) -> Decimal:
        """
        Calculate cost for Search API usage (flat rate, no token costs).

        Args:
            request_count: Number of Search API requests

        Returns:
            Total cost for Search API requests
        """
        cost = (
            Decimal(str(request_count)) / Decimal("1000")
        ) * self.search_api_flat_rate
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# Convenience functions for common calculations
def calculate_perplexity_search_cost(
    model: str, tokens_used: int, search_context: str = "medium"
) -> float:
    """
    Quick calculation of Perplexity search cost.

    Args:
        model: Perplexity model name
        tokens_used: Number of tokens used
        search_context: Search context depth

    Returns:
        Total cost as float
    """
    calculator = PerplexityPricingCalculator()
    cost = calculator.calculate_search_cost(model, tokens_used, search_context)
    return float(cost)


def estimate_monthly_perplexity_costs(
    daily_queries: int,
    model: str = "sonar",
    average_tokens: int = 1000,
    search_context: str = "medium",
) -> dict[str, float]:
    """
    Estimate monthly Perplexity costs based on daily usage patterns.

    Args:
        daily_queries: Average queries per day
        model: Perplexity model
        average_tokens: Average tokens per query
        search_context: Search context depth

    Returns:
        Monthly cost estimates
    """
    calculator = PerplexityPricingCalculator()
    cost_per_query = calculator.calculate_search_cost(
        model, average_tokens, search_context
    )

    daily_cost = cost_per_query * Decimal(str(daily_queries))
    monthly_cost = daily_cost * Decimal("30")

    return {
        "cost_per_query": float(cost_per_query),
        "daily_cost": float(daily_cost),
        "monthly_cost": float(monthly_cost),
        "annual_cost": float(monthly_cost * 12),
    }


# Export key classes and functions
__all__ = [
    "PerplexityPricingCalculator",
    "SearchCostBreakdown",
    "CostAnalysis",
    "TokenPricing",
    "RequestPricing",
    "PerplexityModel",
    "SearchContext",
    "calculate_perplexity_search_cost",
    "estimate_monthly_perplexity_costs",
]
