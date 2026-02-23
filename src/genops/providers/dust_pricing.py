"""Dust pricing engine for cost calculation and optimization insights."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DustPricing:
    """Dust pricing information."""

    pro_monthly_per_user: float
    enterprise_monthly_per_user: float | None
    currency: str = "EUR"
    billing_model: str = "per_user"


@dataclass
class DustCostBreakdown:
    """Detailed cost breakdown for Dust usage."""

    operation_type: str
    operation_count: int
    estimated_tokens: int
    user_count: int
    monthly_subscription_cost: float
    estimated_api_cost: float = 0.0  # For enterprise custom pricing
    total_cost: float = 0.0
    currency: str = "EUR"
    billing_period: str = "monthly"


class DustPricingEngine:
    """Dust pricing engine with subscription and usage-based cost tracking."""

    def __init__(self):
        # Dust uses a subscription-based model with fixed pricing per user
        self.pricing = self._initialize_pricing()

    def _initialize_pricing(self) -> DustPricing:
        """Initialize Dust pricing information."""
        return DustPricing(
            pro_monthly_per_user=29.0,  # €29 per user per month
            enterprise_monthly_per_user=None,  # Custom pricing
            currency="EUR",
            billing_model="per_user",
        )

    def calculate_subscription_cost(
        self, user_count: int, plan_type: str = "pro", billing_period: str = "monthly"
    ) -> float:
        """Calculate subscription cost based on user count and plan."""
        if plan_type.lower() == "pro":
            monthly_cost = self.pricing.pro_monthly_per_user * user_count

            if billing_period.lower() == "annual":
                # Assume 10% discount for annual billing (common practice)
                return monthly_cost * 12 * 0.9

            return monthly_cost

        elif plan_type.lower() == "enterprise":
            # Enterprise pricing is custom, return 0 as placeholder
            logger.warning(
                "Enterprise pricing is custom. Contact Dust for specific rates."
            )
            return 0.0

        else:
            raise ValueError(f"Unknown plan type: {plan_type}")

    def calculate_operation_cost(
        self,
        operation_type: str,
        operation_count: int = 1,
        estimated_tokens: int = 0,
        user_count: int = 1,
        plan_type: str = "pro",
        **kwargs,
    ) -> DustCostBreakdown:
        """
        Calculate cost for Dust operations.

        Since Dust uses subscription pricing, most operations are "included"
        in the subscription cost, but we track usage for optimization.
        """

        # Calculate base subscription cost
        monthly_subscription = self.calculate_subscription_cost(user_count, plan_type)

        # For Pro plan, API usage is included under fair-use
        # Enterprise plans may have custom API pricing
        estimated_api_cost = 0.0

        if plan_type.lower() == "enterprise":
            # Enterprise may have custom API pricing
            # This is a placeholder - actual pricing would need to be configured
            estimated_api_cost = self._estimate_enterprise_api_cost(
                operation_type, operation_count, estimated_tokens
            )

        total_cost = monthly_subscription + estimated_api_cost

        return DustCostBreakdown(
            operation_type=operation_type,
            operation_count=operation_count,
            estimated_tokens=estimated_tokens,
            user_count=user_count,
            monthly_subscription_cost=monthly_subscription,
            estimated_api_cost=estimated_api_cost,
            total_cost=total_cost,
            currency=self.pricing.currency,
            billing_period="monthly",
        )

    def _estimate_enterprise_api_cost(
        self, operation_type: str, operation_count: int, estimated_tokens: int
    ) -> float:
        """
        Estimate enterprise API costs (placeholder implementation).

        Enterprise customers should configure actual rates based on their
        custom pricing agreements with Dust.
        """

        # Placeholder rates - these should be configured per enterprise customer
        base_rates = {
            "conversation": 0.01,  # €0.01 per conversation
            "message": 0.005,  # €0.005 per message
            "agent_execution": 0.02,  # €0.02 per agent run
            "datasource_search": 0.001,  # €0.001 per search
            "datasource_creation": 0.05,  # €0.05 per datasource
        }

        base_rate = base_rates.get(operation_type.lower(), 0.001)

        # Token-based adjustment (very rough estimate)
        token_multiplier = max(1.0, estimated_tokens / 1000)

        return base_rate * operation_count * token_multiplier

    def get_cost_optimization_insights(
        self, usage_stats: dict[str, Any]
    ) -> dict[str, str]:
        """Provide cost optimization recommendations for Dust usage."""
        insights = {}

        # Analyze user utilization
        active_users = usage_stats.get("active_users", 0)
        total_users = usage_stats.get("total_users", active_users)

        if total_users > 0:
            utilization_rate = active_users / total_users

            if utilization_rate < 0.5:
                insights["user_optimization"] = (
                    f"Low user utilization ({utilization_rate:.1%}). "
                    "Consider reviewing user licenses or increasing adoption."
                )
            elif utilization_rate > 0.9:
                insights["user_optimization"] = (
                    f"High user utilization ({utilization_rate:.1%}). "
                    "Well-optimized user base."
                )

        # Analyze operation patterns
        total_operations = usage_stats.get("total_operations", 0)
        if total_operations > 0:
            usage_stats.get("conversations", 0)
            agent_runs = usage_stats.get("agent_runs", 0)
            searches = usage_stats.get("searches", 0)

            if agent_runs / total_operations > 0.7:
                insights["usage_pattern"] = (
                    "Heavy agent usage detected. Ensure agents are optimized "
                    "for efficiency and consider batch processing where possible."
                )

            if searches / total_operations > 0.5:
                insights["search_optimization"] = (
                    "High search volume. Consider optimizing datasources "
                    "and implementing search result caching."
                )

        # Plan recommendations
        if total_users >= 50:
            insights["plan_recommendation"] = (
                "Consider Enterprise plan for teams over 50 users to get "
                "SSO, SCIM provisioning, and custom pricing."
            )

        return insights

    def estimate_monthly_cost(
        self, user_count: int, usage_forecast: dict[str, int], plan_type: str = "pro"
    ) -> dict[str, Any]:
        """Estimate total monthly cost based on user count and usage patterns."""

        base_subscription = self.calculate_subscription_cost(user_count, plan_type)

        # Calculate operation-based costs
        total_api_cost = 0.0
        operation_breakdown = {}

        for operation_type, operation_count in usage_forecast.items():
            estimated_tokens = operation_count * 100  # Rough estimate

            cost_breakdown = self.calculate_operation_cost(
                operation_type=operation_type,
                operation_count=operation_count,
                estimated_tokens=estimated_tokens,
                user_count=user_count,
                plan_type=plan_type,
            )

            operation_breakdown[operation_type] = {
                "operations": operation_count,
                "estimated_cost": cost_breakdown.estimated_api_cost,
            }

            total_api_cost += cost_breakdown.estimated_api_cost

        total_monthly_cost = base_subscription + total_api_cost

        return {
            "user_count": user_count,
            "plan_type": plan_type,
            "base_subscription": base_subscription,
            "api_costs": total_api_cost,
            "total_monthly_cost": total_monthly_cost,
            "currency": self.pricing.currency,
            "operation_breakdown": operation_breakdown,
            "cost_per_user": total_monthly_cost / user_count if user_count > 0 else 0,
        }


def calculate_dust_cost(
    operation_type: str,
    operation_count: int = 1,
    estimated_tokens: int = 0,
    user_count: int = 1,
    plan_type: str = "pro",
    **kwargs,
) -> DustCostBreakdown:
    """Calculate cost for Dust operations using the pricing engine."""
    engine = DustPricingEngine()
    return engine.calculate_operation_cost(
        operation_type=operation_type,
        operation_count=operation_count,
        estimated_tokens=estimated_tokens,
        user_count=user_count,
        plan_type=plan_type,
        **kwargs,
    )


def get_dust_pricing_info() -> DustPricing:
    """Get current Dust pricing information."""
    engine = DustPricingEngine()
    return engine.pricing
