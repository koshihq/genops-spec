#!/usr/bin/env python3
"""
GenOps W&B Cost Aggregation

This module provides comprehensive cost tracking and aggregation capabilities for
Weights & Biases experiments, runs, and artifacts. It tracks costs across multiple
dimensions including compute resources, storage, data transfer, and platform services.

Features:
- Multi-dimensional cost tracking (compute, storage, data transfer)
- Campaign-level cost aggregation across multiple runs
- Team and project cost attribution with drill-down capabilities
- Resource efficiency analysis and optimization recommendations
- Budget monitoring and forecasting based on historical usage patterns
- Cost comparison across experiments and hyperparameter sweeps

Cost Categories:
- Compute Costs: GPU/CPU hours, instance types, training time
- Storage Costs: Artifact storage, dataset versioning, model checkpoints
- Data Transfer Costs: Upload/download bandwidth, API calls
- Platform Costs: W&B service usage, enterprise features

Example usage:

    from genops.providers.wandb_cost_aggregator import WandbCostAggregator

    aggregator = WandbCostAggregator()

    # Track experiment costs
    experiment_cost = aggregator.calculate_experiment_cost(
        compute_hours=2.5,
        gpu_type="v100",
        storage_gb=10.0,
        artifacts_count=5
    )

    # Aggregate campaign costs
    campaign_summary = aggregator.aggregate_campaign_costs(
        run_ids=["run_1", "run_2", "run_3"]
    )

    # Get team cost breakdown
    team_costs = aggregator.get_team_cost_breakdown(
        team="ml-engineering",
        time_period="last_30_days"
    )

Dependencies:
    - datetime: For time-based cost tracking
    - typing: For type hints and data structures
    - dataclasses: For cost data structures
    - enum: For cost category definitions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs for W&B experiments."""

    COMPUTE = "compute"
    STORAGE = "storage"
    DATA_TRANSFER = "data_transfer"
    PLATFORM = "platform"
    INFERENCE = "inference"


class ResourceType(Enum):
    """Types of compute resources."""

    CPU = "cpu"
    GPU_K80 = "gpu_k80"
    GPU_V100 = "gpu_v100"
    GPU_A100 = "gpu_a100"
    GPU_T4 = "gpu_t4"
    TPU_V2 = "tpu_v2"
    TPU_V3 = "tpu_v3"
    TPU_V4 = "tpu_v4"


@dataclass
class ResourceUsage:
    """Resource usage information for cost calculation."""

    resource_type: ResourceType
    duration_hours: float
    instance_count: int = 1
    utilization_percentage: float = 100.0
    region: str = "us-east-1"

    @property
    def effective_hours(self) -> float:
        """Calculate effective compute hours accounting for utilization."""
        return (
            self.duration_hours
            * self.instance_count
            * (self.utilization_percentage / 100.0)
        )


@dataclass
class StorageUsage:
    """Storage usage information for cost calculation."""

    size_gb: float
    duration_days: float
    storage_type: str = "standard"
    access_frequency: str = "frequent"  # frequent, infrequent, archive

    @property
    def total_gb_days(self) -> float:
        """Calculate total GB-days for cost calculation."""
        return self.size_gb * self.duration_days


@dataclass
class DataTransferUsage:
    """Data transfer usage for cost calculation."""

    upload_gb: float = 0.0
    download_gb: float = 0.0
    api_calls: int = 0
    region: str = "us-east-1"

    @property
    def total_transfer_gb(self) -> float:
        """Calculate total data transfer volume."""
        return self.upload_gb + self.download_gb


@dataclass
class CostBreakdown:
    """Detailed cost breakdown by category."""

    compute_cost: float = 0.0
    storage_cost: float = 0.0
    data_transfer_cost: float = 0.0
    platform_cost: float = 0.0
    total_cost: float = 0.0

    def add_cost(self, category: CostCategory, amount: float) -> None:
        """Add cost to specific category."""
        if category == CostCategory.COMPUTE:
            self.compute_cost += amount
        elif category == CostCategory.STORAGE:
            self.storage_cost += amount
        elif category == CostCategory.DATA_TRANSFER:
            self.data_transfer_cost += amount
        elif category == CostCategory.PLATFORM:
            self.platform_cost += amount

        self.total_cost += amount

    def get_category_percentage(self, category: CostCategory) -> float:
        """Get percentage of total cost for a category."""
        if self.total_cost == 0:
            return 0.0

        category_cost = {
            CostCategory.COMPUTE: self.compute_cost,
            CostCategory.STORAGE: self.storage_cost,
            CostCategory.DATA_TRANSFER: self.data_transfer_cost,
            CostCategory.PLATFORM: self.platform_cost,
        }.get(category, 0.0)

        return (category_cost / self.total_cost) * 100.0


@dataclass
class ExperimentCostDetails:
    """Detailed cost information for a single experiment."""

    experiment_id: str
    experiment_name: str
    team: str
    project: str
    start_time: datetime
    end_time: Optional[datetime] = None
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    resource_usage: Optional[ResourceUsage] = None
    storage_usage: Optional[StorageUsage] = None
    data_transfer_usage: Optional[DataTransferUsage] = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        """Calculate experiment duration in hours."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds() / 3600.0

    @property
    def cost_per_hour(self) -> float:
        """Calculate cost per hour for efficiency analysis."""
        duration = self.duration_hours
        return self.cost_breakdown.total_cost / max(duration, 0.01)


@dataclass
class CampaignCostSummary:
    """Cost summary for a campaign (multiple related experiments)."""

    campaign_id: str
    campaign_name: str
    team: str
    project: str
    experiment_costs: list[ExperimentCostDetails] = field(default_factory=list)
    total_cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def experiment_count(self) -> int:
        """Number of experiments in campaign."""
        return len(self.experiment_costs)

    @property
    def average_experiment_cost(self) -> float:
        """Average cost per experiment."""
        if self.experiment_count == 0:
            return 0.0
        return self.total_cost_breakdown.total_cost / self.experiment_count

    @property
    def duration_hours(self) -> float:
        """Total campaign duration in hours."""
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds() / 3600.0


@dataclass
class TeamCostAnalysis:
    """Cost analysis for a team across projects and time periods."""

    team: str
    time_period: str
    total_cost: float
    cost_by_project: dict[str, float] = field(default_factory=dict)
    cost_by_category: dict[str, float] = field(default_factory=dict)
    cost_by_user: dict[str, float] = field(default_factory=dict)
    experiment_count: int = 0
    most_expensive_experiments: list[ExperimentCostDetails] = field(
        default_factory=list
    )
    cost_trends: dict[str, float] = field(default_factory=dict)  # Daily/weekly trends

    @property
    def average_experiment_cost(self) -> float:
        """Average cost per experiment for this team."""
        if self.experiment_count == 0:
            return 0.0
        return self.total_cost / self.experiment_count


class WandbCostAggregator:
    """
    Comprehensive cost aggregation and analysis for W&B experiments.

    Provides multi-dimensional cost tracking, campaign-level aggregation,
    and team cost attribution with detailed breakdowns and optimization insights.
    """

    def __init__(self):
        """Initialize the W&B cost aggregator."""
        self.experiment_costs: dict[str, ExperimentCostDetails] = {}
        self.campaign_summaries: dict[str, CampaignCostSummary] = {}
        self.team_analyses: dict[str, TeamCostAnalysis] = {}

        logger.info("W&B cost aggregator initialized")

    def calculate_experiment_cost(
        self,
        experiment_id: str,
        experiment_name: str,
        team: str,
        project: str,
        resource_usage: Optional[ResourceUsage] = None,
        storage_usage: Optional[StorageUsage] = None,
        data_transfer_usage: Optional[DataTransferUsage] = None,
        compute_hours: Optional[float] = None,
        gpu_type: Optional[str] = None,
        storage_gb: Optional[float] = None,
        artifacts_count: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> ExperimentCostDetails:
        """
        Calculate comprehensive cost for a single experiment.

        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name
            team: Team responsible for the experiment
            project: Project the experiment belongs to
            resource_usage: Detailed resource usage information
            storage_usage: Storage usage information
            data_transfer_usage: Data transfer usage information
            compute_hours: Simple compute hours (alternative to resource_usage)
            gpu_type: GPU type for simple calculation
            storage_gb: Storage size for simple calculation
            artifacts_count: Number of artifacts for platform cost estimation
            start_time: Experiment start time
            end_time: Experiment end time
            tags: Additional tags for categorization

        Returns:
            ExperimentCostDetails with complete cost breakdown
        """
        cost_breakdown = CostBreakdown()

        # Create simplified resource usage if not provided
        if not resource_usage and compute_hours and gpu_type:
            resource_type = self._gpu_type_to_resource_type(gpu_type)
            resource_usage = ResourceUsage(
                resource_type=resource_type, duration_hours=compute_hours
            )

        # Create simplified storage usage if not provided
        if not storage_usage and storage_gb:
            # Estimate 30-day retention for simplicity
            storage_usage = StorageUsage(size_gb=storage_gb, duration_days=30.0)

        # Create simplified data transfer if not provided
        if not data_transfer_usage:
            # Estimate based on artifacts and storage
            upload_gb = (storage_gb or 0) * 0.1  # 10% of storage as uploads
            download_gb = (storage_gb or 0) * 0.05  # 5% as downloads
            api_calls = (artifacts_count or 0) * 10  # 10 API calls per artifact

            data_transfer_usage = DataTransferUsage(
                upload_gb=upload_gb, download_gb=download_gb, api_calls=api_calls
            )

        # Calculate compute costs
        if resource_usage:
            compute_cost = self._calculate_compute_cost(resource_usage)
            cost_breakdown.add_cost(CostCategory.COMPUTE, compute_cost)

        # Calculate storage costs
        if storage_usage:
            storage_cost = self._calculate_storage_cost(storage_usage)
            cost_breakdown.add_cost(CostCategory.STORAGE, storage_cost)

        # Calculate data transfer costs
        if data_transfer_usage:
            transfer_cost = self._calculate_data_transfer_cost(data_transfer_usage)
            cost_breakdown.add_cost(CostCategory.DATA_TRANSFER, transfer_cost)

        # Calculate platform costs (based on usage complexity)
        platform_cost = self._calculate_platform_cost(
            artifacts_count or 0,
            compute_hours or resource_usage.duration_hours if resource_usage else 0,
        )
        cost_breakdown.add_cost(CostCategory.PLATFORM, platform_cost)

        # Create experiment cost details
        experiment_cost = ExperimentCostDetails(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            team=team,
            project=project,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time,
            cost_breakdown=cost_breakdown,
            resource_usage=resource_usage,
            storage_usage=storage_usage,
            data_transfer_usage=data_transfer_usage,
            tags=tags or {},
        )

        # Store for future aggregation
        self.experiment_costs[experiment_id] = experiment_cost

        logger.info(
            f"Calculated experiment cost: {experiment_name} = ${cost_breakdown.total_cost:.4f}"
        )

        return experiment_cost

    def aggregate_campaign_costs(
        self,
        campaign_id: str,
        campaign_name: str,
        experiment_ids: list[str],
        team: Optional[str] = None,
        project: Optional[str] = None,
    ) -> CampaignCostSummary:
        """
        Aggregate costs across multiple experiments in a campaign.

        Args:
            campaign_id: Unique campaign identifier
            campaign_name: Human-readable campaign name
            experiment_ids: List of experiment IDs to include
            team: Team for the campaign (auto-detected if not provided)
            project: Project for the campaign (auto-detected if not provided)

        Returns:
            CampaignCostSummary with aggregated costs
        """
        experiment_costs = []
        total_cost_breakdown = CostBreakdown()
        start_times = []
        end_times = []

        # Aggregate experiment costs
        for exp_id in experiment_ids:
            if exp_id in self.experiment_costs:
                exp_cost = self.experiment_costs[exp_id]
                experiment_costs.append(exp_cost)

                # Add to total cost breakdown
                total_cost_breakdown.compute_cost += (
                    exp_cost.cost_breakdown.compute_cost
                )
                total_cost_breakdown.storage_cost += (
                    exp_cost.cost_breakdown.storage_cost
                )
                total_cost_breakdown.data_transfer_cost += (
                    exp_cost.cost_breakdown.data_transfer_cost
                )
                total_cost_breakdown.platform_cost += (
                    exp_cost.cost_breakdown.platform_cost
                )
                total_cost_breakdown.total_cost += exp_cost.cost_breakdown.total_cost

                # Track time boundaries
                start_times.append(exp_cost.start_time)
                if exp_cost.end_time:
                    end_times.append(exp_cost.end_time)

                # Auto-detect team and project if not provided
                if not team:
                    team = exp_cost.team
                if not project:
                    project = exp_cost.project

        # Calculate campaign time boundaries
        campaign_start = min(start_times) if start_times else None
        campaign_end = max(end_times) if end_times else None

        # Create campaign summary
        campaign_summary = CampaignCostSummary(
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            team=team or "unknown",
            project=project or "unknown",
            experiment_costs=experiment_costs,
            total_cost_breakdown=total_cost_breakdown,
            start_time=campaign_start,
            end_time=campaign_end,
        )

        # Store for future analysis
        self.campaign_summaries[campaign_id] = campaign_summary

        logger.info(
            f"Aggregated campaign costs: {campaign_name} = ${total_cost_breakdown.total_cost:.4f} ({len(experiment_costs)} experiments)"
        )

        return campaign_summary

    def get_team_cost_breakdown(
        self,
        team: str,
        time_period: str = "last_30_days",
        include_projects: Optional[list[str]] = None,
    ) -> TeamCostAnalysis:
        """
        Get comprehensive cost breakdown for a team.

        Args:
            team: Team name
            time_period: Time period for analysis ("last_7_days", "last_30_days", "last_90_days")
            include_projects: Optional list of projects to include

        Returns:
            TeamCostAnalysis with detailed breakdown
        """
        # Calculate time boundaries
        end_time = datetime.utcnow()
        if time_period == "last_7_days":
            start_time = end_time - timedelta(days=7)
        elif time_period == "last_30_days":
            start_time = end_time - timedelta(days=30)
        elif time_period == "last_90_days":
            start_time = end_time - timedelta(days=90)
        else:
            start_time = end_time - timedelta(days=30)  # Default

        # Find relevant experiments
        team_experiments = []
        for exp_cost in self.experiment_costs.values():
            if (
                exp_cost.team == team
                and exp_cost.start_time >= start_time
                and (not include_projects or exp_cost.project in include_projects)
            ):
                team_experiments.append(exp_cost)

        # Aggregate costs
        total_cost = 0.0
        cost_by_project = {}
        cost_by_category = {
            "compute": 0.0,
            "storage": 0.0,
            "data_transfer": 0.0,
            "platform": 0.0,
        }

        # Find most expensive experiments
        sorted_experiments = sorted(
            team_experiments, key=lambda x: x.cost_breakdown.total_cost, reverse=True
        )
        most_expensive = sorted_experiments[:5]  # Top 5

        # Aggregate costs
        for exp in team_experiments:
            total_cost += exp.cost_breakdown.total_cost

            # By project
            project = exp.project
            cost_by_project[project] = (
                cost_by_project.get(project, 0.0) + exp.cost_breakdown.total_cost
            )

            # By category
            cost_by_category["compute"] += exp.cost_breakdown.compute_cost
            cost_by_category["storage"] += exp.cost_breakdown.storage_cost
            cost_by_category["data_transfer"] += exp.cost_breakdown.data_transfer_cost
            cost_by_category["platform"] += exp.cost_breakdown.platform_cost

        # Create analysis
        team_analysis = TeamCostAnalysis(
            team=team,
            time_period=time_period,
            total_cost=total_cost,
            cost_by_project=cost_by_project,
            cost_by_category=cost_by_category,
            experiment_count=len(team_experiments),
            most_expensive_experiments=most_expensive,
        )

        # Store for future reference
        self.team_analyses[f"{team}_{time_period}"] = team_analysis

        logger.info(
            f"Analyzed team costs: {team} ({time_period}) = ${total_cost:.4f} ({len(team_experiments)} experiments)"
        )

        return team_analysis

    def get_cost_optimization_recommendations(
        self,
        experiment_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        team: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate cost optimization recommendations.

        Args:
            experiment_id: Analyze specific experiment
            campaign_id: Analyze specific campaign
            team: Analyze team usage patterns

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        if experiment_id and experiment_id in self.experiment_costs:
            exp_cost = self.experiment_costs[experiment_id]
            recommendations.extend(self._get_experiment_recommendations(exp_cost))

        elif campaign_id and campaign_id in self.campaign_summaries:
            campaign = self.campaign_summaries[campaign_id]
            recommendations.extend(self._get_campaign_recommendations(campaign))

        elif team:
            team_analysis = self.get_team_cost_breakdown(team)
            recommendations.extend(self._get_team_recommendations(team_analysis))

        return recommendations

    def _calculate_compute_cost(self, resource_usage: ResourceUsage) -> float:
        """Calculate compute cost based on resource usage."""
        # Pricing per hour by resource type (approximate cloud pricing)
        pricing = {
            ResourceType.CPU: 0.05,
            ResourceType.GPU_K80: 0.45,
            ResourceType.GPU_V100: 2.48,
            ResourceType.GPU_A100: 3.06,
            ResourceType.GPU_T4: 0.35,
            ResourceType.TPU_V2: 1.35,
            ResourceType.TPU_V3: 1.55,
            ResourceType.TPU_V4: 2.40,
        }

        base_cost = pricing.get(resource_usage.resource_type, 0.10)
        effective_hours = resource_usage.effective_hours

        # Regional pricing multiplier
        region_multiplier = 1.2 if resource_usage.region.startswith("eu-") else 1.0

        return base_cost * effective_hours * region_multiplier

    def _calculate_storage_cost(self, storage_usage: StorageUsage) -> float:
        """Calculate storage cost based on usage patterns."""
        # Pricing per GB-month by storage type and access frequency
        base_rates = {
            ("standard", "frequent"): 0.023,
            ("standard", "infrequent"): 0.0125,
            ("standard", "archive"): 0.004,
            ("ssd", "frequent"): 0.08,
            ("ssd", "infrequent"): 0.04,
        }

        rate_key = (storage_usage.storage_type, storage_usage.access_frequency)
        rate = base_rates.get(rate_key, 0.023)  # Default to standard/frequent

        # Convert GB-days to GB-months
        gb_months = storage_usage.total_gb_days / 30.0

        return rate * gb_months

    def _calculate_data_transfer_cost(self, transfer_usage: DataTransferUsage) -> float:
        """Calculate data transfer cost."""
        # Pricing per GB for data transfer
        upload_rate = 0.00  # Usually free
        download_rate = 0.09  # Per GB egress
        api_rate = 0.0004  # Per API call

        upload_cost = transfer_usage.upload_gb * upload_rate
        download_cost = transfer_usage.download_gb * download_rate
        api_cost = transfer_usage.api_calls * api_rate

        return upload_cost + download_cost + api_cost

    def _calculate_platform_cost(
        self, artifacts_count: int, compute_hours: float
    ) -> float:
        """Calculate platform service costs."""
        # W&B platform costs (approximate)
        base_cost = 0.01  # Base cost per experiment
        artifact_cost = artifacts_count * 0.001  # Per artifact
        compute_cost = compute_hours * 0.005  # Per compute hour tracked

        return base_cost + artifact_cost + compute_cost

    def _gpu_type_to_resource_type(self, gpu_type: str) -> ResourceType:
        """Convert GPU type string to ResourceType enum."""
        gpu_mapping = {
            "k80": ResourceType.GPU_K80,
            "v100": ResourceType.GPU_V100,
            "a100": ResourceType.GPU_A100,
            "t4": ResourceType.GPU_T4,
            "tpu_v2": ResourceType.TPU_V2,
            "tpu_v3": ResourceType.TPU_V3,
            "tpu_v4": ResourceType.TPU_V4,
        }
        return gpu_mapping.get(gpu_type.lower(), ResourceType.GPU_V100)

    def _get_experiment_recommendations(
        self, exp_cost: ExperimentCostDetails
    ) -> list[dict[str, Any]]:
        """Generate recommendations for a specific experiment."""
        recommendations = []

        # High compute cost recommendation
        if (
            exp_cost.cost_breakdown.compute_cost
            > exp_cost.cost_breakdown.total_cost * 0.8
        ):
            recommendations.append(
                {
                    "type": "cost_optimization",
                    "category": "compute",
                    "title": "High Compute Cost Detected",
                    "description": f"Compute costs account for {exp_cost.cost_breakdown.get_category_percentage(CostCategory.COMPUTE):.1f}% of total cost",
                    "suggestion": "Consider using lower-cost GPU instances or optimizing training efficiency",
                    "potential_savings": exp_cost.cost_breakdown.compute_cost * 0.3,
                }
            )

        # Long-running experiment recommendation
        if exp_cost.duration_hours > 24:
            recommendations.append(
                {
                    "type": "efficiency",
                    "category": "duration",
                    "title": "Long-Running Experiment",
                    "description": f"Experiment ran for {exp_cost.duration_hours:.1f} hours",
                    "suggestion": "Consider implementing early stopping or checkpointing for efficiency",
                    "potential_savings": exp_cost.cost_breakdown.total_cost * 0.2,
                }
            )

        return recommendations

    def _get_campaign_recommendations(
        self, campaign: CampaignCostSummary
    ) -> list[dict[str, Any]]:
        """Generate recommendations for a campaign."""
        recommendations = []

        # High variation in experiment costs
        if len(campaign.experiment_costs) > 1:
            costs = [exp.cost_breakdown.total_cost for exp in campaign.experiment_costs]
            max_cost = max(costs)
            min_cost = min(costs)

            if max_cost > min_cost * 3:  # High variation
                recommendations.append(
                    {
                        "type": "cost_optimization",
                        "category": "variation",
                        "title": "High Cost Variation Across Experiments",
                        "description": f"Experiment costs vary from ${min_cost:.2f} to ${max_cost:.2f}",
                        "suggestion": "Standardize resource allocation and investigate high-cost experiments",
                        "potential_savings": (max_cost - min_cost) * 0.5,
                    }
                )

        return recommendations

    def _get_team_recommendations(
        self, team_analysis: TeamCostAnalysis
    ) -> list[dict[str, Any]]:
        """Generate recommendations for a team."""
        recommendations = []

        # High cost per experiment
        if team_analysis.average_experiment_cost > 10.0:
            recommendations.append(
                {
                    "type": "cost_optimization",
                    "category": "team_efficiency",
                    "title": "High Average Experiment Cost",
                    "description": f"Average cost per experiment: ${team_analysis.average_experiment_cost:.2f}",
                    "suggestion": "Implement resource sharing and experiment optimization practices",
                    "potential_savings": team_analysis.total_cost * 0.25,
                }
            )

        # Concentrated costs in few projects
        if len(team_analysis.cost_by_project) > 1:
            max_project_cost = max(team_analysis.cost_by_project.values())
            if max_project_cost > team_analysis.total_cost * 0.7:
                recommendations.append(
                    {
                        "type": "governance",
                        "category": "resource_allocation",
                        "title": "Cost Concentration in Single Project",
                        "description": "One project accounts for >70% of team costs",
                        "suggestion": "Review resource allocation across projects and implement project-level budgets",
                        "potential_savings": 0.0,  # Governance recommendation
                    }
                )

        return recommendations


# Convenience functions for common operations
def calculate_simple_experiment_cost(
    compute_hours: float,
    gpu_type: str = "v100",
    storage_gb: float = 5.0,
    artifacts_count: int = 2,
) -> float:
    """
    Simple cost calculation for quick estimates.

    Args:
        compute_hours: Hours of compute time
        gpu_type: Type of GPU used
        storage_gb: Storage size in GB
        artifacts_count: Number of artifacts

    Returns:
        Estimated total cost in USD
    """
    aggregator = WandbCostAggregator()

    cost_details = aggregator.calculate_experiment_cost(
        experiment_id="temp",
        experiment_name="temp",
        team="temp",
        project="temp",
        compute_hours=compute_hours,
        gpu_type=gpu_type,
        storage_gb=storage_gb,
        artifacts_count=artifacts_count,
    )

    return cost_details.cost_breakdown.total_cost


def estimate_campaign_cost(experiment_costs: list[float]) -> dict[str, float]:
    """
    Estimate campaign cost from individual experiment costs.

    Args:
        experiment_costs: List of individual experiment costs

    Returns:
        Dictionary with cost statistics
    """
    if not experiment_costs:
        return {"total": 0.0, "average": 0.0, "min": 0.0, "max": 0.0}

    return {
        "total": sum(experiment_costs),
        "average": sum(experiment_costs) / len(experiment_costs),
        "min": min(experiment_costs),
        "max": max(experiment_costs),
        "count": len(experiment_costs),
    }


# Alias for test imports
MultiProviderCostAnalysis = TeamCostAnalysis


def generate_cost_optimization_recommendations(
    aggregator: Optional["WandbCostAggregator"] = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Generate cost optimization recommendations based on usage patterns."""
    recommendations = []
    if aggregator and aggregator.experiments:
        total = sum(e.cost_breakdown.total_cost for e in aggregator.experiments)
        if total > 100.0:
            recommendations.append(
                {
                    "type": "budget",
                    "message": "Consider setting budget limits to control costs",
                    "potential_savings": total * 0.1,
                }
            )
        if len(aggregator.experiments) > 10:
            recommendations.append(
                {
                    "type": "efficiency",
                    "message": "Review experiment configurations for optimization opportunities",
                    "potential_savings": total * 0.05,
                }
            )
    return recommendations


def forecast_experiment_costs(
    historical_costs: Optional[list[float]] = None,
    days_ahead: int = 30,
    **kwargs,
) -> dict[str, float]:
    """Forecast future experiment costs based on historical data."""
    if not historical_costs:
        return {"forecast_total": 0.0, "daily_average": 0.0, "confidence": 0.0}

    daily_avg = sum(historical_costs) / len(historical_costs)
    return {
        "forecast_total": daily_avg * days_ahead,
        "daily_average": daily_avg,
        "confidence": min(0.95, len(historical_costs) / 100.0),
    }
