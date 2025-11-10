#!/usr/bin/env python3
"""
GenOps W&B Pricing Models

This module provides comprehensive pricing models and cost calculation utilities for
Weights & Biases experiments and services. It includes cloud compute costs, storage
costs, data transfer costs, and W&B platform service costs.

The pricing models are based on current cloud provider rates (AWS, GCP, Azure) and
W&B service pricing, updated as of 2025. Costs are calculated across multiple
dimensions including compute resources, storage, data transfer, and platform services.

Features:
- Multi-cloud compute pricing (AWS, GCP, Azure)
- Storage cost calculation with tiering and access patterns
- Data transfer and bandwidth pricing
- W&B platform service costs
- Regional pricing variations
- Bulk and committed use discounts
- Cost forecasting and budgeting utilities

Example usage:

    from genops.providers.wandb_pricing import WandbPricingCalculator
    
    calculator = WandbPricingCalculator()
    
    # Calculate compute cost
    compute_cost = calculator.calculate_compute_cost(
        resource_type="gpu_v100",
        hours=2.5,
        region="us-east-1"
    )
    
    # Calculate storage cost
    storage_cost = calculator.calculate_storage_cost(
        size_gb=100.0,
        days=30,
        access_tier="frequent"
    )
    
    # Get total experiment cost
    total_cost = calculator.calculate_experiment_total_cost(
        compute_hours=2.5,
        gpu_type="v100",
        storage_gb=100.0,
        artifacts_count=10
    )

Dependencies:
    - datetime: For time-based calculations
    - typing: For type hints
    - enum: For pricing tiers and resource types
    - dataclasses: For pricing data structures
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Cloud providers for compute resources."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class ResourceType(Enum):
    """Types of compute resources with pricing implications."""
    CPU_SMALL = "cpu_small"      # 1-2 vCPUs
    CPU_MEDIUM = "cpu_medium"    # 4-8 vCPUs
    CPU_LARGE = "cpu_large"      # 16+ vCPUs
    GPU_K80 = "gpu_k80"          # NVIDIA K80
    GPU_V100 = "gpu_v100"        # NVIDIA V100
    GPU_A100 = "gpu_a100"        # NVIDIA A100
    GPU_T4 = "gpu_t4"            # NVIDIA T4
    TPU_V2 = "tpu_v2"            # Google TPU v2
    TPU_V3 = "tpu_v3"            # Google TPU v3
    TPU_V4 = "tpu_v4"            # Google TPU v4


class StorageTier(Enum):
    """Storage tiers with different pricing."""
    STANDARD = "standard"
    INFREQUENT = "infrequent"
    ARCHIVE = "archive"
    SSD = "ssd"
    NVME = "nvme"


class AccessPattern(Enum):
    """Data access patterns affecting storage costs."""
    FREQUENT = "frequent"        # Daily access
    INFREQUENT = "infrequent"    # Monthly access
    ARCHIVE = "archive"          # Yearly access
    COLD = "cold"                # Multi-year retention


@dataclass
class ComputePricing:
    """Compute resource pricing information."""
    resource_type: ResourceType
    provider: CloudProvider
    region: str
    price_per_hour: float
    currency: str = "USD"
    committed_discount: float = 0.0  # Percentage discount for committed use
    spot_discount: float = 0.0       # Percentage discount for spot instances

    def get_effective_price(self, use_committed: bool = False, use_spot: bool = False) -> float:
        """Calculate effective price with discounts."""
        price = self.price_per_hour

        if use_committed:
            price *= (1 - self.committed_discount / 100)
        elif use_spot:
            price *= (1 - self.spot_discount / 100)

        return price


@dataclass
class StoragePricing:
    """Storage pricing information."""
    tier: StorageTier
    access_pattern: AccessPattern
    price_per_gb_month: float
    retrieval_cost_per_gb: float = 0.0
    api_cost_per_request: float = 0.0
    currency: str = "USD"


@dataclass
class DataTransferPricing:
    """Data transfer pricing information."""
    ingress_cost_per_gb: float = 0.0    # Usually free
    egress_cost_per_gb: float = 0.09     # Standard egress rate
    api_cost_per_request: float = 0.0004  # API call cost
    currency: str = "USD"


@dataclass
class PlatformPricing:
    """W&B platform service pricing."""
    plan_type: str                          # "free", "team", "enterprise"
    monthly_cost_per_user: float = 0.0
    experiment_tracking_cost: float = 0.01   # Per experiment
    artifact_storage_cost: float = 0.001    # Per artifact
    additional_storage_cost: float = 0.02   # Per GB beyond free tier
    api_call_cost: float = 0.0001          # Per API call
    currency: str = "USD"


class WandbPricingCalculator:
    """
    Comprehensive pricing calculator for W&B experiments and services.
    
    Provides accurate cost calculations across compute, storage, data transfer,
    and platform services with support for multiple cloud providers and regions.
    """

    def __init__(self, default_provider: CloudProvider = CloudProvider.AWS, default_region: str = "us-east-1"):
        """
        Initialize the pricing calculator.
        
        Args:
            default_provider: Default cloud provider for pricing
            default_region: Default region for pricing calculations
        """
        self.default_provider = default_provider
        self.default_region = default_region
        self._initialize_pricing_data()

        logger.info(f"W&B pricing calculator initialized (provider={default_provider.value}, region={default_region})")

    def _initialize_pricing_data(self) -> None:
        """Initialize pricing data for compute, storage, and services."""

        # Compute pricing data (prices per hour in USD, as of 2025)
        self.compute_pricing = {
            # AWS pricing
            (CloudProvider.AWS, ResourceType.CPU_SMALL, "us-east-1"): ComputePricing(
                ResourceType.CPU_SMALL, CloudProvider.AWS, "us-east-1", 0.0464,
                committed_discount=20.0, spot_discount=70.0
            ),
            (CloudProvider.AWS, ResourceType.CPU_MEDIUM, "us-east-1"): ComputePricing(
                ResourceType.CPU_MEDIUM, CloudProvider.AWS, "us-east-1", 0.1856,
                committed_discount=20.0, spot_discount=70.0
            ),
            (CloudProvider.AWS, ResourceType.CPU_LARGE, "us-east-1"): ComputePricing(
                ResourceType.CPU_LARGE, CloudProvider.AWS, "us-east-1", 0.7424,
                committed_discount=20.0, spot_discount=70.0
            ),
            (CloudProvider.AWS, ResourceType.GPU_K80, "us-east-1"): ComputePricing(
                ResourceType.GPU_K80, CloudProvider.AWS, "us-east-1", 0.45,
                committed_discount=25.0, spot_discount=60.0
            ),
            (CloudProvider.AWS, ResourceType.GPU_V100, "us-east-1"): ComputePricing(
                ResourceType.GPU_V100, CloudProvider.AWS, "us-east-1", 2.48,
                committed_discount=25.0, spot_discount=50.0
            ),
            (CloudProvider.AWS, ResourceType.GPU_A100, "us-east-1"): ComputePricing(
                ResourceType.GPU_A100, CloudProvider.AWS, "us-east-1", 3.06,
                committed_discount=25.0, spot_discount=40.0
            ),
            (CloudProvider.AWS, ResourceType.GPU_T4, "us-east-1"): ComputePricing(
                ResourceType.GPU_T4, CloudProvider.AWS, "us-east-1", 0.35,
                committed_discount=20.0, spot_discount=60.0
            ),

            # GCP pricing
            (CloudProvider.GCP, ResourceType.CPU_SMALL, "us-central1"): ComputePricing(
                ResourceType.CPU_SMALL, CloudProvider.GCP, "us-central1", 0.0475,
                committed_discount=30.0, spot_discount=80.0
            ),
            (CloudProvider.GCP, ResourceType.CPU_MEDIUM, "us-central1"): ComputePricing(
                ResourceType.CPU_MEDIUM, CloudProvider.GCP, "us-central1", 0.1900,
                committed_discount=30.0, spot_discount=80.0
            ),
            (CloudProvider.GCP, ResourceType.GPU_V100, "us-central1"): ComputePricing(
                ResourceType.GPU_V100, CloudProvider.GCP, "us-central1", 2.35,
                committed_discount=30.0, spot_discount=60.0
            ),
            (CloudProvider.GCP, ResourceType.GPU_A100, "us-central1"): ComputePricing(
                ResourceType.GPU_A100, CloudProvider.GCP, "us-central1", 2.93,
                committed_discount=30.0, spot_discount=50.0
            ),
            (CloudProvider.GCP, ResourceType.TPU_V2, "us-central1"): ComputePricing(
                ResourceType.TPU_V2, CloudProvider.GCP, "us-central1", 1.35,
                committed_discount=30.0, spot_discount=70.0
            ),
            (CloudProvider.GCP, ResourceType.TPU_V3, "us-central1"): ComputePricing(
                ResourceType.TPU_V3, CloudProvider.GCP, "us-central1", 1.55,
                committed_discount=30.0, spot_discount=70.0
            ),
            (CloudProvider.GCP, ResourceType.TPU_V4, "us-central1"): ComputePricing(
                ResourceType.TPU_V4, CloudProvider.GCP, "us-central1", 2.40,
                committed_discount=30.0, spot_discount=60.0
            ),

            # Azure pricing
            (CloudProvider.AZURE, ResourceType.CPU_SMALL, "eastus"): ComputePricing(
                ResourceType.CPU_SMALL, CloudProvider.AZURE, "eastus", 0.0496,
                committed_discount=20.0, spot_discount=80.0
            ),
            (CloudProvider.AZURE, ResourceType.GPU_V100, "eastus"): ComputePricing(
                ResourceType.GPU_V100, CloudProvider.AZURE, "eastus", 2.52,
                committed_discount=25.0, spot_discount=50.0
            ),
            (CloudProvider.AZURE, ResourceType.GPU_A100, "eastus"): ComputePricing(
                ResourceType.GPU_A100, CloudProvider.AZURE, "eastus", 3.12,
                committed_discount=25.0, spot_discount=40.0
            ),

            # Local/on-premises (estimated costs)
            (CloudProvider.LOCAL, ResourceType.GPU_V100, "local"): ComputePricing(
                ResourceType.GPU_V100, CloudProvider.LOCAL, "local", 0.50,  # Amortized hardware cost
                committed_discount=0.0, spot_discount=0.0
            ),
        }

        # Storage pricing data (prices per GB-month in USD)
        self.storage_pricing = {
            StorageTier.STANDARD: StoragePricing(
                StorageTier.STANDARD, AccessPattern.FREQUENT, 0.023
            ),
            StorageTier.INFREQUENT: StoragePricing(
                StorageTier.INFREQUENT, AccessPattern.INFREQUENT, 0.0125,
                retrieval_cost_per_gb=0.01
            ),
            StorageTier.ARCHIVE: StoragePricing(
                StorageTier.ARCHIVE, AccessPattern.ARCHIVE, 0.004,
                retrieval_cost_per_gb=0.02, api_cost_per_request=0.0004
            ),
            StorageTier.SSD: StoragePricing(
                StorageTier.SSD, AccessPattern.FREQUENT, 0.08
            ),
            StorageTier.NVME: StoragePricing(
                StorageTier.NVME, AccessPattern.FREQUENT, 0.16
            ),
        }

        # Data transfer pricing
        self.data_transfer_pricing = DataTransferPricing(
            ingress_cost_per_gb=0.0,      # Free ingress
            egress_cost_per_gb=0.09,       # Standard egress rate
            api_cost_per_request=0.0004    # API call cost
        )

        # W&B platform pricing
        self.platform_pricing = {
            "free": PlatformPricing(
                "free", 0.0, 0.0, 0.0, 0.0, 0.0
            ),
            "team": PlatformPricing(
                "team", 20.0, 0.005, 0.0005, 0.02, 0.0001
            ),
            "enterprise": PlatformPricing(
                "enterprise", 50.0, 0.002, 0.0002, 0.015, 0.00005
            )
        }

    def calculate_compute_cost(
        self,
        resource_type: Union[ResourceType, str],
        hours: float,
        provider: Optional[CloudProvider] = None,
        region: Optional[str] = None,
        use_committed: bool = False,
        use_spot: bool = False,
        instance_count: int = 1
    ) -> float:
        """
        Calculate compute cost for specified resources.
        
        Args:
            resource_type: Type of compute resource
            hours: Number of hours to calculate cost for
            provider: Cloud provider (uses default if not specified)
            region: Region (uses default if not specified)
            use_committed: Whether to use committed use discounts
            use_spot: Whether to use spot instance discounts
            instance_count: Number of instances
            
        Returns:
            Total compute cost in USD
        """
        if isinstance(resource_type, str):
            resource_type = ResourceType(resource_type)

        provider = provider or self.default_provider
        region = region or self.default_region

        # Get pricing for the resource
        pricing_key = (provider, resource_type, region)
        pricing = self.compute_pricing.get(pricing_key)

        if not pricing:
            # Try to find pricing for similar region or fallback
            pricing = self._find_fallback_compute_pricing(provider, resource_type, region)

        if not pricing:
            # Use default pricing based on resource type
            pricing = self._get_default_compute_pricing(resource_type)

        effective_price = pricing.get_effective_price(use_committed, use_spot)
        total_cost = effective_price * hours * instance_count

        logger.debug(f"Compute cost: {resource_type.value} x{instance_count} for {hours}h = ${total_cost:.4f}")

        return total_cost

    def calculate_storage_cost(
        self,
        size_gb: float,
        days: int,
        tier: Union[StorageTier, str] = StorageTier.STANDARD,
        access_pattern: Union[AccessPattern, str] = AccessPattern.FREQUENT,
        retrieval_gb: float = 0.0,
        api_requests: int = 0
    ) -> float:
        """
        Calculate storage cost including retrieval and API costs.
        
        Args:
            size_gb: Storage size in GB
            days: Number of days to store data
            tier: Storage tier
            access_pattern: Access pattern
            retrieval_gb: Amount of data retrieved in GB
            api_requests: Number of API requests
            
        Returns:
            Total storage cost in USD
        """
        if isinstance(tier, str):
            tier = StorageTier(tier)
        if isinstance(access_pattern, str):
            access_pattern = AccessPattern(access_pattern)

        pricing = self.storage_pricing.get(tier)
        if not pricing:
            pricing = self.storage_pricing[StorageTier.STANDARD]  # Fallback

        # Calculate storage cost (GB-months)
        gb_months = (size_gb * days) / 30.0
        storage_cost = gb_months * pricing.price_per_gb_month

        # Calculate retrieval cost
        retrieval_cost = retrieval_gb * pricing.retrieval_cost_per_gb

        # Calculate API cost
        api_cost = api_requests * pricing.api_cost_per_request

        total_cost = storage_cost + retrieval_cost + api_cost

        logger.debug(f"Storage cost: {size_gb}GB for {days} days = ${total_cost:.4f}")

        return total_cost

    def calculate_data_transfer_cost(
        self,
        ingress_gb: float = 0.0,
        egress_gb: float = 0.0,
        api_requests: int = 0
    ) -> float:
        """
        Calculate data transfer costs.
        
        Args:
            ingress_gb: Data uploaded in GB
            egress_gb: Data downloaded in GB
            api_requests: Number of API requests
            
        Returns:
            Total data transfer cost in USD
        """
        pricing = self.data_transfer_pricing

        ingress_cost = ingress_gb * pricing.ingress_cost_per_gb
        egress_cost = egress_gb * pricing.egress_cost_per_gb
        api_cost = api_requests * pricing.api_cost_per_request

        total_cost = ingress_cost + egress_cost + api_cost

        logger.debug(f"Data transfer cost: {ingress_gb}GB in + {egress_gb}GB out + {api_requests} API = ${total_cost:.4f}")

        return total_cost

    def calculate_platform_cost(
        self,
        plan_type: str = "team",
        users: int = 1,
        experiments: int = 1,
        artifacts: int = 0,
        additional_storage_gb: float = 0.0,
        api_calls: int = 0,
        months: int = 1
    ) -> float:
        """
        Calculate W&B platform service costs.
        
        Args:
            plan_type: W&B plan type ("free", "team", "enterprise")
            users: Number of users
            experiments: Number of experiments
            artifacts: Number of artifacts
            additional_storage_gb: Additional storage beyond free tier
            api_calls: Number of API calls
            months: Number of months
            
        Returns:
            Total platform cost in USD
        """
        pricing = self.platform_pricing.get(plan_type, self.platform_pricing["team"])

        monthly_cost = users * pricing.monthly_cost_per_user
        experiment_cost = experiments * pricing.experiment_tracking_cost
        artifact_cost = artifacts * pricing.artifact_storage_cost
        storage_cost = additional_storage_gb * pricing.additional_storage_cost
        api_cost = api_calls * pricing.api_call_cost

        monthly_total = monthly_cost + experiment_cost + artifact_cost + storage_cost + api_cost
        total_cost = monthly_total * months

        logger.debug(f"Platform cost ({plan_type}): {users} users, {experiments} experiments for {months} months = ${total_cost:.4f}")

        return total_cost

    def calculate_experiment_total_cost(
        self,
        compute_hours: float,
        resource_type: Union[ResourceType, str] = ResourceType.GPU_V100,
        storage_gb: float = 10.0,
        artifacts_count: int = 5,
        experiment_duration_days: int = 1,
        data_upload_gb: float = 1.0,
        data_download_gb: float = 0.5,
        api_calls: int = 100,
        provider: Optional[CloudProvider] = None,
        region: Optional[str] = None,
        plan_type: str = "team"
    ) -> Dict[str, float]:
        """
        Calculate total cost for a complete experiment.
        
        Args:
            compute_hours: Compute hours required
            resource_type: Type of compute resource
            storage_gb: Storage size in GB
            artifacts_count: Number of artifacts
            experiment_duration_days: Duration in days
            data_upload_gb: Data uploaded in GB
            data_download_gb: Data downloaded in GB
            api_calls: Number of API calls
            provider: Cloud provider
            region: Region
            plan_type: W&B plan type
            
        Returns:
            Dictionary with cost breakdown
        """
        # Calculate individual cost components
        compute_cost = self.calculate_compute_cost(
            resource_type=resource_type,
            hours=compute_hours,
            provider=provider,
            region=region
        )

        storage_cost = self.calculate_storage_cost(
            size_gb=storage_gb,
            days=experiment_duration_days * 30,  # Convert to storage days
            retrieval_gb=data_download_gb,
            api_requests=api_calls // 2  # Assume half are storage API calls
        )

        transfer_cost = self.calculate_data_transfer_cost(
            ingress_gb=data_upload_gb,
            egress_gb=data_download_gb,
            api_requests=api_calls
        )

        platform_cost = self.calculate_platform_cost(
            plan_type=plan_type,
            users=1,
            experiments=1,
            artifacts=artifacts_count,
            api_calls=api_calls
        )

        total_cost = compute_cost + storage_cost + transfer_cost + platform_cost

        cost_breakdown = {
            "compute_cost": compute_cost,
            "storage_cost": storage_cost,
            "data_transfer_cost": transfer_cost,
            "platform_cost": platform_cost,
            "total_cost": total_cost
        }

        logger.info(f"Total experiment cost: ${total_cost:.4f} (compute: ${compute_cost:.4f}, storage: ${storage_cost:.4f}, transfer: ${transfer_cost:.4f}, platform: ${platform_cost:.4f})")

        return cost_breakdown

    def estimate_monthly_cost(
        self,
        experiments_per_month: int,
        avg_compute_hours_per_experiment: float,
        avg_storage_gb: float = 50.0,
        team_size: int = 5,
        plan_type: str = "team"
    ) -> Dict[str, float]:
        """
        Estimate monthly costs for a team.
        
        Args:
            experiments_per_month: Number of experiments per month
            avg_compute_hours_per_experiment: Average compute hours per experiment
            avg_storage_gb: Average storage per experiment
            team_size: Number of team members
            plan_type: W&B plan type
            
        Returns:
            Dictionary with monthly cost estimate
        """
        # Calculate per-experiment cost
        experiment_cost = self.calculate_experiment_total_cost(
            compute_hours=avg_compute_hours_per_experiment,
            storage_gb=avg_storage_gb,
            plan_type=plan_type
        )

        # Scale by number of experiments
        monthly_experiment_costs = {
            key: value * experiments_per_month
            for key, value in experiment_cost.items()
        }

        # Add team subscription cost
        team_subscription_cost = self.calculate_platform_cost(
            plan_type=plan_type,
            users=team_size,
            experiments=0,  # Already counted above
            months=1
        )

        monthly_experiment_costs["team_subscription"] = team_subscription_cost
        monthly_experiment_costs["total_cost"] += team_subscription_cost

        logger.info(f"Monthly cost estimate: ${monthly_experiment_costs['total_cost']:.2f} for {team_size} users, {experiments_per_month} experiments/month")

        return monthly_experiment_costs

    def get_cost_optimization_suggestions(
        self,
        current_cost_breakdown: Dict[str, float],
        usage_patterns: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate cost optimization suggestions based on usage patterns.
        
        Args:
            current_cost_breakdown: Current cost breakdown
            usage_patterns: Optional usage pattern data
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        total_cost = current_cost_breakdown.get("total_cost", 0.0)
        compute_cost = current_cost_breakdown.get("compute_cost", 0.0)

        # High compute cost suggestions
        if compute_cost > total_cost * 0.7:
            suggestions.append({
                "category": "compute",
                "title": "High Compute Cost Detected",
                "description": f"Compute accounts for {(compute_cost/total_cost)*100:.1f}% of total cost",
                "suggestions": [
                    "Consider using spot instances for non-critical workloads (up to 80% savings)",
                    "Implement auto-scaling to avoid idle compute time",
                    "Use committed use discounts for predictable workloads (up to 30% savings)",
                    "Optimize model architecture for faster training"
                ],
                "potential_savings_percentage": 40.0
            })

        # Storage optimization
        storage_cost = current_cost_breakdown.get("storage_cost", 0.0)
        if storage_cost > total_cost * 0.2:
            suggestions.append({
                "category": "storage",
                "title": "Storage Cost Optimization",
                "description": f"Storage accounts for {(storage_cost/total_cost)*100:.1f}% of total cost",
                "suggestions": [
                    "Move infrequently accessed data to archive tier (up to 80% savings)",
                    "Implement data lifecycle policies",
                    "Clean up old experiment artifacts",
                    "Use data deduplication for similar experiments"
                ],
                "potential_savings_percentage": 50.0
            })

        # Platform optimization
        platform_cost = current_cost_breakdown.get("platform_cost", 0.0)
        if platform_cost > total_cost * 0.3:
            suggestions.append({
                "category": "platform",
                "title": "Platform Cost Review",
                "description": "Platform costs are high relative to usage",
                "suggestions": [
                    "Review team plan requirements",
                    "Optimize API usage patterns",
                    "Consider enterprise plan for better rates at scale",
                    "Implement experiment batching to reduce overhead"
                ],
                "potential_savings_percentage": 25.0
            })

        return suggestions

    def _find_fallback_compute_pricing(
        self,
        provider: CloudProvider,
        resource_type: ResourceType,
        region: str
    ) -> Optional[ComputePricing]:
        """Find fallback pricing for similar regions or resources."""
        # Try to find pricing for the same provider and resource type in other regions
        for (p, rt, r), pricing in self.compute_pricing.items():
            if p == provider and rt == resource_type:
                # Apply regional multiplier
                multiplier = 1.2 if region.startswith(('eu-', 'ap-')) else 1.0
                fallback_pricing = ComputePricing(
                    resource_type=pricing.resource_type,
                    provider=pricing.provider,
                    region=region,
                    price_per_hour=pricing.price_per_hour * multiplier,
                    committed_discount=pricing.committed_discount,
                    spot_discount=pricing.spot_discount
                )
                return fallback_pricing

        return None

    def _get_default_compute_pricing(self, resource_type: ResourceType) -> ComputePricing:
        """Get default pricing for resource type."""
        default_prices = {
            ResourceType.CPU_SMALL: 0.05,
            ResourceType.CPU_MEDIUM: 0.20,
            ResourceType.CPU_LARGE: 0.80,
            ResourceType.GPU_K80: 0.45,
            ResourceType.GPU_V100: 2.50,
            ResourceType.GPU_A100: 3.00,
            ResourceType.GPU_T4: 0.35,
            ResourceType.TPU_V2: 1.35,
            ResourceType.TPU_V3: 1.55,
            ResourceType.TPU_V4: 2.40,
        }

        price = default_prices.get(resource_type, 1.00)

        return ComputePricing(
            resource_type=resource_type,
            provider=self.default_provider,
            region=self.default_region,
            price_per_hour=price,
            committed_discount=20.0,
            spot_discount=50.0
        )


# Convenience functions for quick calculations
def calculate_simple_experiment_cost(
    compute_hours: float,
    gpu_type: str = "v100",
    storage_gb: float = 10.0
) -> float:
    """
    Quick experiment cost calculation.
    
    Args:
        compute_hours: Hours of compute
        gpu_type: GPU type (k80, v100, a100, t4)
        storage_gb: Storage in GB
        
    Returns:
        Estimated total cost in USD
    """
    calculator = WandbPricingCalculator()

    resource_type = ResourceType(f"gpu_{gpu_type.lower()}")

    result = calculator.calculate_experiment_total_cost(
        compute_hours=compute_hours,
        resource_type=resource_type,
        storage_gb=storage_gb
    )

    return result["total_cost"]


def get_resource_hourly_cost(resource_type: str, provider: str = "aws") -> float:
    """
    Get hourly cost for a resource type.
    
    Args:
        resource_type: Resource type (cpu_small, gpu_v100, etc.)
        provider: Cloud provider (aws, gcp, azure)
        
    Returns:
        Hourly cost in USD
    """
    calculator = WandbPricingCalculator(CloudProvider(provider))

    return calculator.calculate_compute_cost(
        resource_type=ResourceType(resource_type),
        hours=1.0
    )
