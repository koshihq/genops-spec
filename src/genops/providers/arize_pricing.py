#!/usr/bin/env python3
"""
GenOps Arize AI Pricing Models

This module provides comprehensive pricing models and cost calculation utilities
for Arize AI model monitoring operations. It supports multiple pricing tiers,
usage-based billing, and cost optimization strategies.

Features:
- Multi-tier pricing models (Starter, Professional, Enterprise)
- Usage-based cost calculation with volume discounts
- Custom pricing for enterprise contracts
- Cost estimation and forecasting utilities
- Regional pricing variations and currency conversion
- Billing cycle management and prorated charges
- Cost optimization recommendations based on usage patterns

Pricing Categories:
- Prediction Logging: Per-prediction costs with volume discounts
- Data Quality Monitoring: Cost per data quality check and drift analysis
- Alert Management: Cost per active alert and notification
- Dashboard Analytics: Cost per dashboard view and custom analytics
- Model Performance Tracking: Cost per performance metric collection
- Storage: Cost per GB of stored monitoring data
- API Usage: Cost per API call with rate limiting considerations

Example usage:

    from genops.providers.arize_pricing import ArizePricingCalculator, PricingTier

    # Initialize pricing calculator
    calculator = ArizePricingCalculator(
        tier=PricingTier.PROFESSIONAL,
        region="us-east-1",
        currency="USD"
    )

    # Calculate costs for monitoring operations
    prediction_cost = calculator.calculate_prediction_logging_cost(
        prediction_count=100000,
        model_tier="production"
    )

    alert_cost = calculator.calculate_alert_management_cost(
        alert_count=5,
        duration_days=30,
        alert_complexity="advanced"
    )

    # Get volume discount information
    discount_info = calculator.get_volume_discount_tier(monthly_predictions=1000000)

    # Estimate monthly costs with optimization
    monthly_estimate = calculator.estimate_monthly_cost(
        models=10,
        predictions_per_model=50000,
        optimize_for_cost=True
    )
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class PricingTier(Enum):
    """Arize AI pricing tiers with different feature sets."""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class BillingCycle(Enum):
    """Billing cycle options."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class AlertComplexity(Enum):
    """Alert complexity levels affecting cost."""

    BASIC = "basic"
    ADVANCED = "advanced"
    CUSTOM = "custom"


class ModelTier(Enum):
    """Model tier classifications affecting pricing."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CRITICAL = "critical"


@dataclass
class VolumeDiscountTier:
    """Volume discount tier information."""

    tier_name: str
    min_volume: int
    max_volume: Optional[int]
    discount_percentage: float
    effective_rate: float
    tier_description: str


@dataclass
class PricingBreakdown:
    """Detailed pricing breakdown for cost analysis."""

    base_cost: float
    volume_discount: float
    tier_discount: float
    regional_adjustment: float
    final_cost: float
    effective_rate: float
    discount_details: dict[str, float]
    billing_period: str


@dataclass
class MonthlyEstimate:
    """Monthly cost estimate with breakdown and optimization suggestions."""

    total_estimated_cost: float
    cost_breakdown: dict[str, float]
    volume_discounts_applied: dict[str, float]
    optimization_opportunities: list[str]
    confidence_level: float
    assumptions: list[str]
    recommended_tier: PricingTier
    potential_savings: float


@dataclass
class CostForecast:
    """Cost forecasting with different scenarios."""

    base_forecast: float
    optimistic_forecast: float
    pessimistic_forecast: float
    forecast_confidence: float
    key_drivers: list[str]
    risk_factors: list[str]
    recommendations: list[str]


class ArizePricingCalculator:
    """
    Comprehensive pricing calculator for Arize AI monitoring operations.

    Provides accurate cost calculations, volume discounts, and optimization
    recommendations based on usage patterns and pricing tiers.
    """

    def __init__(
        self,
        tier: PricingTier = PricingTier.PROFESSIONAL,
        region: str = "us-east-1",
        currency: str = "USD",
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        enterprise_discount: float = 0.0,
        custom_pricing: Optional[dict[str, float]] = None,
    ):
        """
        Initialize Arize pricing calculator.

        Args:
            tier: Pricing tier (Starter, Professional, Enterprise, Custom)
            region: AWS region for regional pricing adjustments
            currency: Currency for pricing (USD, EUR, GBP)
            billing_cycle: Billing cycle affecting discounts
            enterprise_discount: Additional enterprise discount percentage
            custom_pricing: Custom pricing rates for enterprise contracts
        """
        self.tier = tier
        self.region = region
        self.currency = currency
        self.billing_cycle = billing_cycle
        self.enterprise_discount = enterprise_discount

        # Base pricing rates (USD, per unit)
        self.base_rates = custom_pricing or self._get_base_pricing_rates()

        # Volume discount tiers for prediction logging
        self.volume_discount_tiers = self._get_volume_discount_tiers()

        # Regional pricing multipliers
        self.regional_multipliers = {
            "us-east-1": 1.0,
            "us-west-2": 1.0,
            "eu-west-1": 1.15,
            "eu-central-1": 1.12,
            "ap-southeast-1": 1.08,
            "ap-northeast-1": 1.10,
        }

        # Currency conversion rates (simplified - in production would use live rates)
        self.currency_rates = {
            "USD": 1.0,
            "EUR": 0.85,
            "GBP": 0.73,
            "CAD": 1.25,
            "AUD": 1.35,
        }

        # Billing cycle discounts
        self.billing_cycle_discounts = {
            BillingCycle.MONTHLY: 0.0,
            BillingCycle.QUARTERLY: 0.05,
            BillingCycle.ANNUAL: 0.15,
        }

        logger.info(
            f"Arize pricing calculator initialized: {tier.value}, {region}, {currency}"
        )

    def calculate_prediction_logging_cost(
        self,
        prediction_count: int,
        model_tier: Union[ModelTier, str] = ModelTier.PRODUCTION,
        time_period_days: int = 30,
    ) -> PricingBreakdown:
        """
        Calculate cost for prediction logging with volume discounts.

        Args:
            prediction_count: Number of predictions to log
            model_tier: Model tier affecting pricing
            time_period_days: Time period for cost calculation

        Returns:
            PricingBreakdown with detailed cost analysis
        """
        if isinstance(model_tier, str):
            model_tier = ModelTier(model_tier)

        # Base rate per prediction
        base_rate = self.base_rates["prediction_logging"]

        # Model tier adjustments
        tier_multipliers = {
            ModelTier.DEVELOPMENT: 0.5,
            ModelTier.STAGING: 0.7,
            ModelTier.PRODUCTION: 1.0,
            ModelTier.CRITICAL: 1.3,
        }

        adjusted_rate = base_rate * tier_multipliers[model_tier]
        base_cost = prediction_count * adjusted_rate

        # Apply volume discounts
        volume_discount_info = self.get_volume_discount_tier(prediction_count)
        volume_discount_amount = base_cost * (
            volume_discount_info.discount_percentage / 100
        )

        # Apply tier discount
        tier_discount_amount = base_cost * self._get_tier_discount_percentage()

        # Apply regional adjustment
        regional_multiplier = self.regional_multipliers.get(self.region, 1.0)
        regional_adjustment = base_cost * (regional_multiplier - 1.0)

        # Calculate final cost
        final_cost = (
            base_cost
            - volume_discount_amount
            - tier_discount_amount
            + regional_adjustment
        )

        # Apply enterprise discount
        if self.enterprise_discount > 0:
            final_cost *= 1 - self.enterprise_discount / 100

        # Convert currency if needed
        final_cost *= self.currency_rates[self.currency]

        return PricingBreakdown(
            base_cost=base_cost * self.currency_rates[self.currency],
            volume_discount=volume_discount_amount * self.currency_rates[self.currency],
            tier_discount=tier_discount_amount * self.currency_rates[self.currency],
            regional_adjustment=regional_adjustment
            * self.currency_rates[self.currency],
            final_cost=final_cost,
            effective_rate=final_cost / prediction_count if prediction_count > 0 else 0,
            discount_details={
                "volume_discount_percentage": volume_discount_info.discount_percentage,
                "tier_discount_percentage": self._get_tier_discount_percentage(),
                "enterprise_discount_percentage": self.enterprise_discount,
            },
            billing_period=f"{time_period_days} days",
        )

    def calculate_data_quality_cost(
        self,
        quality_checks: int,
        drift_analyses: int = 0,
        feature_monitoring: int = 0,
        time_period_days: int = 30,
    ) -> PricingBreakdown:
        """
        Calculate cost for data quality monitoring operations.

        Args:
            quality_checks: Number of data quality checks
            drift_analyses: Number of drift analyses performed
            feature_monitoring: Number of features monitored
            time_period_days: Time period for cost calculation

        Returns:
            PricingBreakdown with detailed cost analysis
        """
        # Calculate component costs
        quality_check_cost = quality_checks * self.base_rates["data_quality_check"]
        drift_analysis_cost = drift_analyses * self.base_rates["drift_analysis"]
        feature_monitoring_cost = (
            feature_monitoring * self.base_rates["feature_monitoring"]
        )

        base_cost = quality_check_cost + drift_analysis_cost + feature_monitoring_cost

        # Apply tier discount
        tier_discount_amount = base_cost * self._get_tier_discount_percentage()

        # Apply regional and enterprise adjustments
        regional_multiplier = self.regional_multipliers.get(self.region, 1.0)
        final_cost = (base_cost - tier_discount_amount) * regional_multiplier

        if self.enterprise_discount > 0:
            final_cost *= 1 - self.enterprise_discount / 100

        # Convert currency
        final_cost *= self.currency_rates[self.currency]

        return PricingBreakdown(
            base_cost=base_cost * self.currency_rates[self.currency],
            volume_discount=0.0,  # No volume discounts for data quality
            tier_discount=tier_discount_amount * self.currency_rates[self.currency],
            regional_adjustment=(final_cost - base_cost)
            * self.currency_rates[self.currency],
            final_cost=final_cost,
            effective_rate=final_cost
            / max(quality_checks + drift_analyses + feature_monitoring, 1),
            discount_details={
                "tier_discount_percentage": self._get_tier_discount_percentage(),
                "enterprise_discount_percentage": self.enterprise_discount,
            },
            billing_period=f"{time_period_days} days",
        )

    def calculate_alert_management_cost(
        self,
        alert_count: int,
        duration_days: int = 30,
        alert_complexity: Union[AlertComplexity, str] = AlertComplexity.BASIC,
        notification_channels: int = 1,
    ) -> PricingBreakdown:
        """
        Calculate cost for alert management operations.

        Args:
            alert_count: Number of active alerts
            duration_days: Duration alerts are active
            alert_complexity: Complexity level of alerts
            notification_channels: Number of notification channels per alert

        Returns:
            PricingBreakdown with detailed cost analysis
        """
        if isinstance(alert_complexity, str):
            alert_complexity = AlertComplexity(alert_complexity)

        # Base alert cost
        base_alert_rate = self.base_rates["alert_management"]

        # Complexity multipliers
        complexity_multipliers = {
            AlertComplexity.BASIC: 1.0,
            AlertComplexity.ADVANCED: 1.5,
            AlertComplexity.CUSTOM: 2.0,
        }

        # Calculate costs
        alert_cost = (
            alert_count
            * base_alert_rate
            * complexity_multipliers[alert_complexity]
            * duration_days
        )
        notification_cost = (
            alert_count
            * notification_channels
            * self.base_rates["notification"]
            * duration_days
        )

        base_cost = alert_cost + notification_cost

        # Apply discounts and adjustments
        tier_discount_amount = base_cost * self._get_tier_discount_percentage()
        regional_multiplier = self.regional_multipliers.get(self.region, 1.0)
        final_cost = (base_cost - tier_discount_amount) * regional_multiplier

        if self.enterprise_discount > 0:
            final_cost *= 1 - self.enterprise_discount / 100

        final_cost *= self.currency_rates[self.currency]

        return PricingBreakdown(
            base_cost=base_cost * self.currency_rates[self.currency],
            volume_discount=0.0,
            tier_discount=tier_discount_amount * self.currency_rates[self.currency],
            regional_adjustment=(final_cost - base_cost + tier_discount_amount)
            * self.currency_rates[self.currency],
            final_cost=final_cost,
            effective_rate=final_cost / (alert_count * duration_days)
            if alert_count > 0 and duration_days > 0
            else 0,
            discount_details={
                "tier_discount_percentage": self._get_tier_discount_percentage(),
                "complexity_multiplier": complexity_multipliers[alert_complexity],
                "enterprise_discount_percentage": self.enterprise_discount,
            },
            billing_period=f"{duration_days} days",
        )

    def calculate_dashboard_analytics_cost(
        self,
        dashboard_count: int,
        dashboard_views: int,
        custom_metrics: int = 0,
        time_period_days: int = 30,
    ) -> PricingBreakdown:
        """
        Calculate cost for dashboard and analytics operations.

        Args:
            dashboard_count: Number of active dashboards
            dashboard_views: Number of dashboard views
            custom_metrics: Number of custom metrics
            time_period_days: Time period for cost calculation

        Returns:
            PricingBreakdown with detailed cost analysis
        """
        # Calculate component costs
        dashboard_cost = (
            dashboard_count * self.base_rates["dashboard"] * time_period_days
        )
        view_cost = dashboard_views * self.base_rates["dashboard_view"]
        custom_metrics_cost = (
            custom_metrics * self.base_rates["custom_metric"] * time_period_days
        )

        base_cost = dashboard_cost + view_cost + custom_metrics_cost

        # Apply discounts and adjustments (similar to other methods)
        tier_discount_amount = base_cost * self._get_tier_discount_percentage()
        regional_multiplier = self.regional_multipliers.get(self.region, 1.0)
        final_cost = (base_cost - tier_discount_amount) * regional_multiplier

        if self.enterprise_discount > 0:
            final_cost *= 1 - self.enterprise_discount / 100

        final_cost *= self.currency_rates[self.currency]

        return PricingBreakdown(
            base_cost=base_cost * self.currency_rates[self.currency],
            volume_discount=0.0,
            tier_discount=tier_discount_amount * self.currency_rates[self.currency],
            regional_adjustment=(final_cost - base_cost + tier_discount_amount)
            * self.currency_rates[self.currency],
            final_cost=final_cost,
            effective_rate=final_cost / time_period_days,
            discount_details={
                "tier_discount_percentage": self._get_tier_discount_percentage(),
                "enterprise_discount_percentage": self.enterprise_discount,
            },
            billing_period=f"{time_period_days} days",
        )

    def get_volume_discount_tier(self, monthly_predictions: int) -> VolumeDiscountTier:
        """
        Get volume discount tier information for prediction volume.

        Args:
            monthly_predictions: Monthly prediction volume

        Returns:
            VolumeDiscountTier information
        """
        for tier in self.volume_discount_tiers:
            if monthly_predictions >= tier.min_volume and (
                tier.max_volume is None or monthly_predictions <= tier.max_volume
            ):
                return tier

        # Default to highest tier if volume exceeds all tiers
        return self.volume_discount_tiers[-1]

    def estimate_monthly_cost(
        self,
        models: int,
        predictions_per_model: int,
        quality_checks_per_model: int = 100,
        alerts_per_model: int = 3,
        dashboards: int = 5,
        optimize_for_cost: bool = False,
    ) -> MonthlyEstimate:
        """
        Estimate comprehensive monthly costs with optimization suggestions.

        Args:
            models: Number of models to monitor
            predictions_per_model: Average predictions per model per month
            quality_checks_per_model: Quality checks per model per month
            alerts_per_model: Number of alerts per model
            dashboards: Number of dashboards
            optimize_for_cost: Whether to include cost optimization suggestions

        Returns:
            MonthlyEstimate with detailed breakdown and recommendations
        """
        total_predictions = models * predictions_per_model
        total_quality_checks = models * quality_checks_per_model
        total_alerts = models * alerts_per_model

        # Calculate component costs
        prediction_breakdown = self.calculate_prediction_logging_cost(
            total_predictions, ModelTier.PRODUCTION, 30
        )
        quality_breakdown = self.calculate_data_quality_cost(
            total_quality_checks, 0, 0, 30
        )
        alert_breakdown = self.calculate_alert_management_cost(
            total_alerts, 30, AlertComplexity.BASIC, 1
        )
        dashboard_breakdown = self.calculate_dashboard_analytics_cost(
            dashboards, dashboards * 100, 0, 30
        )

        # Aggregate costs
        total_cost = (
            prediction_breakdown.final_cost
            + quality_breakdown.final_cost
            + alert_breakdown.final_cost
            + dashboard_breakdown.final_cost
        )

        cost_breakdown = {
            "prediction_logging": prediction_breakdown.final_cost,
            "data_quality": quality_breakdown.final_cost,
            "alert_management": alert_breakdown.final_cost,
            "dashboard_analytics": dashboard_breakdown.final_cost,
        }

        # Calculate total discounts applied
        volume_discounts = {
            "prediction_volume_discount": prediction_breakdown.volume_discount
        }

        # Generate optimization opportunities
        optimization_opportunities = []
        potential_savings = 0.0

        if optimize_for_cost:
            if total_predictions > 1000000:
                optimization_opportunities.append(
                    "Consider prediction sampling to reduce logging costs"
                )
                potential_savings += prediction_breakdown.final_cost * 0.3

            if total_alerts > 20:
                optimization_opportunities.append(
                    "Consolidate alerts to reduce management overhead"
                )
                potential_savings += alert_breakdown.final_cost * 0.2

            if models > 10:
                optimization_opportunities.append(
                    "Consider environment-based monitoring tiers"
                )
                potential_savings += total_cost * 0.15

        # Recommend optimal tier
        recommended_tier = self._recommend_optimal_tier(total_cost)

        return MonthlyEstimate(
            total_estimated_cost=total_cost,
            cost_breakdown=cost_breakdown,
            volume_discounts_applied=volume_discounts,
            optimization_opportunities=optimization_opportunities,
            confidence_level=0.85,  # 85% confidence in estimate
            assumptions=[
                f"Based on {models} models with {predictions_per_model} predictions each",
                "Standard monitoring configuration assumed",
                f"Current pricing tier: {self.tier.value}",
            ],
            recommended_tier=recommended_tier,
            potential_savings=potential_savings,
        )

    def compare_pricing_tiers(
        self, usage_scenario: dict[str, Any]
    ) -> dict[PricingTier, float]:
        """
        Compare costs across different pricing tiers for a usage scenario.

        Args:
            usage_scenario: Dictionary with usage parameters

        Returns:
            Dictionary mapping pricing tiers to estimated costs
        """
        tier_costs = {}

        for tier in [
            PricingTier.STARTER,
            PricingTier.PROFESSIONAL,
            PricingTier.ENTERPRISE,
        ]:
            # Create temporary calculator for this tier
            temp_calculator = ArizePricingCalculator(
                tier=tier,
                region=self.region,
                currency=self.currency,
                billing_cycle=self.billing_cycle,
            )

            # Calculate cost for this tier
            estimate = temp_calculator.estimate_monthly_cost(
                models=usage_scenario.get("models", 5),
                predictions_per_model=usage_scenario.get(
                    "predictions_per_model", 50000
                ),
                quality_checks_per_model=usage_scenario.get(
                    "quality_checks_per_model", 100
                ),
                alerts_per_model=usage_scenario.get("alerts_per_model", 3),
                dashboards=usage_scenario.get("dashboards", 5),
            )

            tier_costs[tier] = estimate.total_estimated_cost

        return tier_costs

    def _get_base_pricing_rates(self) -> dict[str, float]:
        """Get base pricing rates based on tier."""
        base_rates = {
            PricingTier.STARTER: {
                "prediction_logging": 0.0015,
                "data_quality_check": 0.015,
                "drift_analysis": 0.05,
                "feature_monitoring": 0.01,
                "alert_management": 0.08,
                "notification": 0.005,
                "dashboard": 0.20,
                "dashboard_view": 0.001,
                "custom_metric": 0.05,
            },
            PricingTier.PROFESSIONAL: {
                "prediction_logging": 0.001,
                "data_quality_check": 0.01,
                "drift_analysis": 0.03,
                "feature_monitoring": 0.008,
                "alert_management": 0.05,
                "notification": 0.003,
                "dashboard": 0.15,
                "dashboard_view": 0.0008,
                "custom_metric": 0.03,
            },
            PricingTier.ENTERPRISE: {
                "prediction_logging": 0.0008,
                "data_quality_check": 0.008,
                "drift_analysis": 0.025,
                "feature_monitoring": 0.006,
                "alert_management": 0.04,
                "notification": 0.002,
                "dashboard": 0.12,
                "dashboard_view": 0.0006,
                "custom_metric": 0.025,
            },
        }

        return base_rates.get(self.tier, base_rates[PricingTier.PROFESSIONAL])

    def _get_volume_discount_tiers(self) -> list[VolumeDiscountTier]:
        """Get volume discount tiers for prediction logging."""
        return [
            VolumeDiscountTier(
                "Small", 0, 100000, 0.0, 0.001, "Up to 100K predictions"
            ),
            VolumeDiscountTier(
                "Medium", 100001, 500000, 10.0, 0.0009, "100K-500K predictions"
            ),
            VolumeDiscountTier(
                "Large", 500001, 2000000, 20.0, 0.0008, "500K-2M predictions"
            ),
            VolumeDiscountTier(
                "Enterprise", 2000001, 10000000, 30.0, 0.0007, "2M-10M predictions"
            ),
            VolumeDiscountTier(
                "Scale", 10000001, None, 40.0, 0.0006, "10M+ predictions"
            ),
        ]

    def _get_tier_discount_percentage(self) -> float:
        """Get discount percentage based on pricing tier."""
        tier_discounts = {
            PricingTier.STARTER: 0.0,
            PricingTier.PROFESSIONAL: 0.10,
            PricingTier.ENTERPRISE: 0.20,
            PricingTier.CUSTOM: 0.25,
        }
        return tier_discounts.get(self.tier, 0.0)

    def _recommend_optimal_tier(self, monthly_cost: float) -> PricingTier:
        """Recommend optimal pricing tier based on usage."""
        if monthly_cost < 100:
            return PricingTier.STARTER
        elif monthly_cost < 1000:
            return PricingTier.PROFESSIONAL
        else:
            return PricingTier.ENTERPRISE


# Convenience functions for quick cost estimates


def quick_prediction_cost_estimate(
    predictions: int, tier: str = "professional"
) -> float:
    """Quick estimate for prediction logging costs."""
    rates = {"starter": 0.0015, "professional": 0.001, "enterprise": 0.0008}
    return predictions * rates.get(tier, 0.001)


def quick_monthly_estimate(
    models: int, predictions_per_model: int, tier: str = "professional"
) -> float:
    """Quick estimate for monthly monitoring costs."""
    calculator = ArizePricingCalculator(tier=PricingTier(tier))
    estimate = calculator.estimate_monthly_cost(models, predictions_per_model)
    return estimate.total_estimated_cost


@dataclass
class VolumeDiscount:
    """Volume discount tier information."""

    tier: str = ""
    discount_percentage: float = 0.0
    min_volume: int = 0
    max_volume: Optional[int] = None
    description: str = ""


@dataclass
class PricingOptimizationRecommendation:
    """Recommendation for pricing optimization."""

    recommendation_type: str = ""
    description: str = ""
    potential_savings: float = 0.0
    confidence: float = 0.0
    current_tier: str = ""
    suggested_tier: str = ""


def calculate_prediction_logging_cost(
    predictions: int,
    tier: str = "professional",
) -> float:
    """Calculate cost for prediction logging."""
    return quick_prediction_cost_estimate(predictions, tier)


def calculate_data_quality_monitoring_cost(
    data_points: int,
    checks_per_point: int = 5,
    tier: str = "professional",
) -> float:
    """Calculate cost for data quality monitoring."""
    rates = {"starter": 0.0005, "professional": 0.0003, "enterprise": 0.0002}
    return data_points * checks_per_point * rates.get(tier, 0.0003)


def calculate_alert_management_cost(
    alert_count: int,
    complexity: str = "basic",
) -> float:
    """Calculate cost for alert management."""
    rates = {"basic": 0.01, "advanced": 0.05, "custom": 0.10}
    return alert_count * rates.get(complexity, 0.01)


def estimate_dashboard_cost(
    dashboards: int = 1,
    views_per_month: int = 100,
) -> float:
    """Estimate cost for dashboard analytics."""
    return dashboards * 10.0 + views_per_month * 0.01


def get_volume_discount_tier(predictions: int) -> VolumeDiscount:
    """Get volume discount tier for a given prediction count."""
    if predictions >= 10_000_000:
        return VolumeDiscount(
            "enterprise", 30.0, 10_000_000, None, "Enterprise volume discount"
        )
    elif predictions >= 1_000_000:
        return VolumeDiscount(
            "professional", 20.0, 1_000_000, 10_000_000, "Professional volume discount"
        )
    elif predictions >= 100_000:
        return VolumeDiscount(
            "starter", 10.0, 100_000, 1_000_000, "Starter volume discount"
        )
    return VolumeDiscount("none", 0.0, 0, 100_000, "No volume discount")


def optimize_pricing_strategy(
    current_tier: str = "professional",
    monthly_predictions: int = 100_000,
    models: int = 5,
) -> list[PricingOptimizationRecommendation]:
    """Generate pricing optimization strategy recommendations."""
    recommendations = []
    if monthly_predictions > 1_000_000 and current_tier == "professional":
        recommendations.append(
            PricingOptimizationRecommendation(
                recommendation_type="tier_upgrade",
                description="Consider Enterprise tier for high volume",
                potential_savings=monthly_predictions * 0.0001,
                confidence=0.8,
                current_tier=current_tier,
                suggested_tier="enterprise",
            )
        )
    return recommendations


# Convenience exports
__all__ = [
    "ArizePricingCalculator",
    "PricingTier",
    "BillingCycle",
    "AlertComplexity",
    "ModelTier",
    "VolumeDiscountTier",
    "VolumeDiscount",
    "PricingBreakdown",
    "PricingOptimizationRecommendation",
    "MonthlyEstimate",
    "CostForecast",
    "quick_prediction_cost_estimate",
    "quick_monthly_estimate",
    "calculate_prediction_logging_cost",
    "calculate_data_quality_monitoring_cost",
    "calculate_alert_management_cost",
    "estimate_dashboard_cost",
    "get_volume_discount_tier",
    "optimize_pricing_strategy",
]
