#!/usr/bin/env python3
"""
GenOps Arize AI Cost Aggregator

This module provides comprehensive cost aggregation and analysis for Arize AI
model monitoring operations, supporting multi-model and multi-environment
cost tracking with detailed breakdowns and optimization recommendations.

Features:
- Multi-model cost aggregation across monitoring operations
- Environment-specific cost tracking (dev/staging/production)
- Time-based cost analysis and trend detection
- Cost optimization recommendations based on usage patterns
- Budget tracking and forecasting for monitoring operations
- Team and project-level cost attribution
- Alert cost management and optimization

Cost Categories:
- Prediction Logging: Cost per prediction logged to Arize
- Data Quality Monitoring: Cost for drift detection and quality checks
- Alert Management: Cost for alert configuration and notifications
- Dashboard Analytics: Cost for dashboard views and analytics
- Model Performance Tracking: Cost for performance metric collection

Example usage:

    from genops.providers.arize_cost_aggregator import ArizeCostAggregator

    # Initialize cost aggregator
    cost_aggregator = ArizeCostAggregator(
        team="ml-platform",
        project="fraud-detection"
    )

    # Track monitoring session costs
    session_cost = cost_aggregator.calculate_monitoring_session_cost(
        model_id="fraud-model-v2",
        prediction_count=10000,
        data_quality_checks=5,
        active_alerts=3,
        session_duration_hours=24
    )

    # Get cost summary and optimization recommendations
    monthly_summary = cost_aggregator.get_monthly_cost_summary()
    optimization_tips = cost_aggregator.get_cost_optimization_recommendations()

    print(f"Session cost: ${session_cost.total_cost:.2f}")
    print(f"Monthly total: ${monthly_summary.total_cost:.2f}")
    print(f"Optimization potential: ${optimization_tips.potential_savings:.2f}")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories for Arize AI operations."""

    PREDICTION_LOGGING = "prediction_logging"
    DATA_QUALITY = "data_quality"
    ALERT_MANAGEMENT = "alert_management"
    DASHBOARD_ANALYTICS = "dashboard_analytics"
    MODEL_PERFORMANCE = "model_performance"
    STORAGE = "storage"
    API_CALLS = "api_calls"


class OptimizationRecommendationType(Enum):
    """Types of cost optimization recommendations."""

    REDUCE_LOGGING_FREQUENCY = "reduce_logging_frequency"
    OPTIMIZE_ALERT_CONFIGURATION = "optimize_alert_configuration"
    CONSOLIDATE_MODELS = "consolidate_models"
    REDUCE_RETENTION_PERIOD = "reduce_retention_period"
    BATCH_OPERATIONS = "batch_operations"
    ELIMINATE_REDUNDANT_MONITORING = "eliminate_redundant_monitoring"


@dataclass
class MonitoringSessionCost:
    """Cost breakdown for a single monitoring session."""

    session_id: str
    model_id: str
    model_version: str
    environment: str
    total_cost: float
    prediction_logging_cost: float
    data_quality_cost: float
    alert_management_cost: float
    dashboard_cost: float
    storage_cost: float
    duration_hours: float
    prediction_count: int
    efficiency_score: float
    cost_per_prediction: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MonthlyCostSummary:
    """Comprehensive monthly cost summary for Arize operations."""

    month: str
    total_cost: float
    cost_by_category: dict[CostCategory, float]
    cost_by_model: dict[str, float]
    cost_by_environment: dict[str, float]
    cost_by_team: dict[str, float]
    prediction_volume: int
    alert_count: int
    model_count: int
    average_cost_per_model: float
    cost_trend: float  # Percentage change from previous month
    budget_utilization: float
    top_cost_drivers: list[tuple[str, float]]


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation with actionable insights."""

    recommendation_type: OptimizationRecommendationType
    title: str
    description: str
    potential_savings: float
    effort_level: str  # "Low", "Medium", "High"
    implementation_steps: list[str]
    affected_models: list[str]
    risk_level: str  # "Low", "Medium", "High"
    priority_score: float  # 0-100


@dataclass
class CostForecast:
    """Cost forecasting for budget planning."""

    forecast_period: str
    forecasted_cost: float
    confidence_interval: tuple[float, float]
    key_assumptions: list[str]
    risk_factors: list[str]
    budget_recommendation: float


class ArizeCostAggregator:
    """
    Comprehensive cost aggregation and analysis for Arize AI monitoring operations.

    Provides detailed cost tracking, optimization recommendations, and budget
    management for model monitoring across multiple models and environments.
    """

    def __init__(
        self,
        team: str,
        project: str,
        cost_center: Optional[str] = None,
        budget_limit: float = 1000.0,
        retention_days: int = 90,
    ):
        """
        Initialize Arize cost aggregator.

        Args:
            team: Team name for cost attribution
            project: Project name for cost attribution
            cost_center: Cost center for financial reporting
            budget_limit: Monthly budget limit in USD
            retention_days: Days to retain cost data
        """
        self.team = team
        self.project = project
        self.cost_center = cost_center
        self.budget_limit = budget_limit
        self.retention_days = retention_days

        # Cost tracking storage
        self.session_costs: list[MonitoringSessionCost] = []
        self.monthly_summaries: dict[str, MonthlyCostSummary] = {}

        # Pricing configuration
        self.pricing = {
            CostCategory.PREDICTION_LOGGING: 0.001,  # $0.001 per prediction
            CostCategory.DATA_QUALITY: 0.01,  # $0.01 per quality check
            CostCategory.ALERT_MANAGEMENT: 0.05,  # $0.05 per active alert per day
            CostCategory.DASHBOARD_ANALYTICS: 0.10,  # $0.10 per dashboard per day
            CostCategory.MODEL_PERFORMANCE: 0.02,  # $0.02 per performance metric
            CostCategory.STORAGE: 0.001,  # $0.001 per MB per month
            CostCategory.API_CALLS: 0.0001,  # $0.0001 per API call
        }

        logger.info(f"Arize cost aggregator initialized for {team}/{project}")

    def calculate_monitoring_session_cost(
        self,
        model_id: str,
        model_version: str = "latest",
        environment: str = "production",
        prediction_count: int = 0,
        data_quality_checks: int = 0,
        active_alerts: int = 0,
        session_duration_hours: float = 24.0,
        dashboard_views: int = 1,
        api_calls: int = 0,
        storage_mb: float = 0.0,
        session_id: Optional[str] = None,
    ) -> MonitoringSessionCost:
        """
        Calculate comprehensive cost for a monitoring session.

        Args:
            model_id: Model identifier
            model_version: Model version
            environment: Environment (dev/staging/production)
            prediction_count: Number of predictions logged
            data_quality_checks: Number of data quality checks performed
            active_alerts: Number of active alerts
            session_duration_hours: Duration of monitoring session
            dashboard_views: Number of dashboard views
            api_calls: Number of API calls made
            storage_mb: Storage used in MB
            session_id: Optional session identifier

        Returns:
            MonitoringSessionCost with detailed cost breakdown
        """
        session_id = session_id or f"{model_id}_{int(datetime.utcnow().timestamp())}"

        # Calculate costs by category
        prediction_cost = (
            prediction_count * self.pricing[CostCategory.PREDICTION_LOGGING]
        )
        data_quality_cost = (
            data_quality_checks * self.pricing[CostCategory.DATA_QUALITY]
        )
        alert_cost = (
            active_alerts
            * self.pricing[CostCategory.ALERT_MANAGEMENT]
            * (session_duration_hours / 24)
        )
        dashboard_cost = (
            dashboard_views
            * self.pricing[CostCategory.DASHBOARD_ANALYTICS]
            * (session_duration_hours / 24)
        )
        storage_cost = storage_mb * self.pricing[CostCategory.STORAGE]
        api_cost = api_calls * self.pricing[CostCategory.API_CALLS]

        total_cost = (
            prediction_cost
            + data_quality_cost
            + alert_cost
            + dashboard_cost
            + storage_cost
            + api_cost
        )

        # Calculate efficiency metrics
        cost_per_prediction = total_cost / max(prediction_count, 1)
        efficiency_score = prediction_count / max(
            total_cost * 1000, 1
        )  # Predictions per $1

        session_cost = MonitoringSessionCost(
            session_id=session_id,
            model_id=model_id,
            model_version=model_version,
            environment=environment,
            total_cost=total_cost,
            prediction_logging_cost=prediction_cost,
            data_quality_cost=data_quality_cost,
            alert_management_cost=alert_cost,
            dashboard_cost=dashboard_cost,
            storage_cost=storage_cost,
            duration_hours=session_duration_hours,
            prediction_count=prediction_count,
            efficiency_score=efficiency_score,
            cost_per_prediction=cost_per_prediction,
        )

        # Store session cost
        self.session_costs.append(session_cost)
        self._cleanup_old_data()

        return session_cost

    def get_monthly_cost_summary(
        self, month: Optional[str] = None
    ) -> MonthlyCostSummary:
        """
        Get comprehensive monthly cost summary.

        Args:
            month: Month in YYYY-MM format (defaults to current month)

        Returns:
            MonthlyCostSummary with detailed breakdown
        """
        if not month:
            month = datetime.utcnow().strftime("%Y-%m")

        # Filter sessions for the specified month
        month_sessions = [
            session
            for session in self.session_costs
            if session.timestamp.strftime("%Y-%m") == month
        ]

        if not month_sessions:
            return self._create_empty_monthly_summary(month)

        # Calculate aggregated metrics
        total_cost = sum(session.total_cost for session in month_sessions)

        # Cost by category
        cost_by_category = {
            CostCategory.PREDICTION_LOGGING: sum(
                s.prediction_logging_cost for s in month_sessions
            ),
            CostCategory.DATA_QUALITY: sum(s.data_quality_cost for s in month_sessions),
            CostCategory.ALERT_MANAGEMENT: sum(
                s.alert_management_cost for s in month_sessions
            ),
            CostCategory.DASHBOARD_ANALYTICS: sum(
                s.dashboard_cost for s in month_sessions
            ),
            CostCategory.STORAGE: sum(s.storage_cost for s in month_sessions),
        }

        # Cost by model
        cost_by_model = {}
        for session in month_sessions:
            model_key = f"{session.model_id}-{session.model_version}"
            cost_by_model[model_key] = (
                cost_by_model.get(model_key, 0) + session.total_cost
            )

        # Cost by environment
        cost_by_environment = {}
        for session in month_sessions:
            cost_by_environment[session.environment] = (
                cost_by_environment.get(session.environment, 0) + session.total_cost
            )

        # Additional metrics
        prediction_volume = sum(session.prediction_count for session in month_sessions)
        model_count = len({f"{s.model_id}-{s.model_version}" for s in month_sessions})
        average_cost_per_model = total_cost / max(model_count, 1)
        budget_utilization = (total_cost / self.budget_limit) * 100

        # Top cost drivers
        top_cost_drivers = sorted(
            cost_by_model.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Calculate cost trend (placeholder - would need historical data)
        cost_trend = 0.0  # Would calculate based on previous month

        summary = MonthlyCostSummary(
            month=month,
            total_cost=total_cost,
            cost_by_category=cost_by_category,
            cost_by_model=cost_by_model,
            cost_by_environment=cost_by_environment,
            cost_by_team={self.team: total_cost},
            prediction_volume=prediction_volume,
            alert_count=0,  # Would aggregate from sessions
            model_count=model_count,
            average_cost_per_model=average_cost_per_model,
            cost_trend=cost_trend,
            budget_utilization=budget_utilization,
            top_cost_drivers=top_cost_drivers,
        )

        self.monthly_summaries[month] = summary
        return summary

    def get_cost_optimization_recommendations(
        self,
    ) -> list[CostOptimizationRecommendation]:
        """
        Generate cost optimization recommendations based on usage patterns.

        Returns:
            List of actionable cost optimization recommendations
        """
        recommendations = []

        if not self.session_costs:
            return recommendations

        # Analyze recent usage patterns
        recent_sessions = [
            s
            for s in self.session_costs
            if s.timestamp >= datetime.utcnow() - timedelta(days=30)
        ]

        if not recent_sessions:
            return recommendations

        # Recommendation 1: High-frequency logging optimization
        avg_predictions_per_session = sum(
            s.prediction_count for s in recent_sessions
        ) / len(recent_sessions)
        if avg_predictions_per_session > 50000:
            recommendations.append(
                CostOptimizationRecommendation(
                    recommendation_type=OptimizationRecommendationType.REDUCE_LOGGING_FREQUENCY,
                    title="Optimize High-Frequency Prediction Logging",
                    description="Consider sampling prediction logs or implementing batch logging to reduce per-prediction costs.",
                    potential_savings=sum(
                        s.prediction_logging_cost for s in recent_sessions
                    )
                    * 0.3,
                    effort_level="Medium",
                    implementation_steps=[
                        "Implement prediction sampling (e.g., log every 10th prediction)",
                        "Use batch logging API to reduce individual API calls",
                        "Configure different logging rates for different environments",
                    ],
                    affected_models=list({s.model_id for s in recent_sessions}),
                    risk_level="Low",
                    priority_score=75.0,
                )
            )

        # Recommendation 2: Alert optimization
        high_alert_sessions = [
            s for s in recent_sessions if s.alert_management_cost > 5.0
        ]
        if high_alert_sessions:
            recommendations.append(
                CostOptimizationRecommendation(
                    recommendation_type=OptimizationRecommendationType.OPTIMIZE_ALERT_CONFIGURATION,
                    title="Optimize Alert Configuration",
                    description="Review and consolidate alerts to reduce management costs while maintaining monitoring coverage.",
                    potential_savings=sum(
                        s.alert_management_cost for s in high_alert_sessions
                    )
                    * 0.25,
                    effort_level="Low",
                    implementation_steps=[
                        "Review alert thresholds and eliminate false positives",
                        "Consolidate similar alerts across models",
                        "Use alert suppression during maintenance windows",
                    ],
                    affected_models=list({s.model_id for s in high_alert_sessions}),
                    risk_level="Medium",
                    priority_score=60.0,
                )
            )

        # Recommendation 3: Environment consolidation
        env_costs = {}
        for session in recent_sessions:
            env_costs[session.environment] = (
                env_costs.get(session.environment, 0) + session.total_cost
            )

        if len(env_costs) > 2 and env_costs.get("development", 0) > 100:
            recommendations.append(
                CostOptimizationRecommendation(
                    recommendation_type=OptimizationRecommendationType.CONSOLIDATE_MODELS,
                    title="Consolidate Development Environment Monitoring",
                    description="Reduce monitoring scope in development environments to focus on production-critical metrics.",
                    potential_savings=env_costs.get("development", 0) * 0.5,
                    effort_level="Low",
                    implementation_steps=[
                        "Disable detailed monitoring in development environments",
                        "Use reduced sampling rates for non-production environments",
                        "Focus monitoring on critical production models only",
                    ],
                    affected_models=list(
                        {
                            s.model_id
                            for s in recent_sessions
                            if s.environment == "development"
                        }
                    ),
                    risk_level="Low",
                    priority_score=45.0,
                )
            )

        # Sort by priority score
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def generate_cost_forecast(self, forecast_months: int = 3) -> CostForecast:
        """
        Generate cost forecast for budget planning.

        Args:
            forecast_months: Number of months to forecast

        Returns:
            CostForecast with predictions and recommendations
        """
        if not self.session_costs:
            return CostForecast(
                forecast_period=f"{forecast_months} months",
                forecasted_cost=0.0,
                confidence_interval=(0.0, 0.0),
                key_assumptions=["No historical data available"],
                risk_factors=["Unable to predict without usage history"],
                budget_recommendation=self.budget_limit,
            )

        # Simple trend-based forecasting (in production, would use more sophisticated methods)
        recent_monthly_avg = self._calculate_recent_monthly_average()
        forecasted_monthly_cost = recent_monthly_avg * 1.1  # Assume 10% growth
        total_forecasted_cost = forecasted_monthly_cost * forecast_months

        # Confidence interval (±20%)
        confidence_interval = (total_forecasted_cost * 0.8, total_forecasted_cost * 1.2)

        return CostForecast(
            forecast_period=f"{forecast_months} months",
            forecasted_cost=total_forecasted_cost,
            confidence_interval=confidence_interval,
            key_assumptions=[
                "10% month-over-month growth in monitoring volume",
                "Current pricing structure remains stable",
                "No significant changes in monitoring scope",
            ],
            risk_factors=[
                "Increased model deployment could drive higher costs",
                "Changes in Arize pricing structure",
                "Expansion to additional environments",
            ],
            budget_recommendation=total_forecasted_cost * 1.2,  # 20% buffer
        )

    def export_cost_data(self, format: str = "json") -> str:
        """
        Export cost data for external analysis.

        Args:
            format: Export format ("json", "csv")

        Returns:
            Serialized cost data
        """
        if format.lower() == "json":
            data = {
                "team": self.team,
                "project": self.project,
                "session_costs": [
                    {
                        "session_id": s.session_id,
                        "model_id": s.model_id,
                        "model_version": s.model_version,
                        "environment": s.environment,
                        "total_cost": s.total_cost,
                        "timestamp": s.timestamp.isoformat(),
                        "prediction_count": s.prediction_count,
                        "efficiency_score": s.efficiency_score,
                    }
                    for s in self.session_costs
                ],
                "monthly_summaries": {
                    month: {
                        "total_cost": summary.total_cost,
                        "model_count": summary.model_count,
                        "budget_utilization": summary.budget_utilization,
                    }
                    for month, summary in self.monthly_summaries.items()
                },
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _create_empty_monthly_summary(self, month: str) -> MonthlyCostSummary:
        """Create an empty monthly summary for months with no data."""
        return MonthlyCostSummary(
            month=month,
            total_cost=0.0,
            cost_by_category=dict.fromkeys(CostCategory, 0.0),
            cost_by_model={},
            cost_by_environment={},
            cost_by_team={self.team: 0.0},
            prediction_volume=0,
            alert_count=0,
            model_count=0,
            average_cost_per_model=0.0,
            cost_trend=0.0,
            budget_utilization=0.0,
            top_cost_drivers=[],
        )

    def _calculate_recent_monthly_average(self) -> float:
        """Calculate average monthly cost based on recent data."""
        if not self.session_costs:
            return 0.0

        # Group by month and calculate monthly totals
        monthly_costs = {}
        for session in self.session_costs:
            month = session.timestamp.strftime("%Y-%m")
            monthly_costs[month] = monthly_costs.get(month, 0) + session.total_cost

        if not monthly_costs:
            return 0.0

        return sum(monthly_costs.values()) / len(monthly_costs)

    def _cleanup_old_data(self) -> None:
        """Remove cost data older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        self.session_costs = [
            session
            for session in self.session_costs
            if session.timestamp >= cutoff_date
        ]


# Convenience functions for common operations


def calculate_prediction_logging_cost(prediction_count: int) -> float:
    """Calculate cost for prediction logging operations."""
    return prediction_count * 0.001  # $0.001 per prediction


def calculate_data_quality_cost(quality_checks: int) -> float:
    """Calculate cost for data quality monitoring operations."""
    return quality_checks * 0.01  # $0.01 per quality check


def calculate_alert_management_cost(alerts: int, duration_days: float) -> float:
    """Calculate cost for alert management operations."""
    return alerts * 0.05 * duration_days  # $0.05 per alert per day


def estimate_monthly_monitoring_cost(
    models: int,
    predictions_per_model_per_day: int,
    alerts_per_model: int = 3,
    quality_checks_per_model_per_day: int = 10,
) -> float:
    """
    Estimate monthly monitoring cost for multiple models.

    Args:
        models: Number of models to monitor
        predictions_per_model_per_day: Average predictions per model per day
        alerts_per_model: Number of alerts per model
        quality_checks_per_model_per_day: Quality checks per model per day

    Returns:
        Estimated monthly cost in USD
    """
    daily_prediction_cost = models * predictions_per_model_per_day * 0.001
    daily_quality_cost = models * quality_checks_per_model_per_day * 0.01
    daily_alert_cost = models * alerts_per_model * 0.05
    daily_dashboard_cost = models * 0.10

    daily_total = (
        daily_prediction_cost
        + daily_quality_cost
        + daily_alert_cost
        + daily_dashboard_cost
    )
    return daily_total * 30  # Monthly estimate


# Convenience exports
__all__ = [
    "ArizeCostAggregator",
    "MonitoringSessionCost",
    "MonthlyCostSummary",
    "CostOptimizationRecommendation",
    "CostForecast",
    "CostCategory",
    "OptimizationRecommendationType",
    "calculate_prediction_logging_cost",
    "calculate_data_quality_cost",
    "calculate_alert_management_cost",
    "estimate_monthly_monitoring_cost",
]
