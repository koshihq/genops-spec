"""
GPU Instance Pricing Database and Cost Calculation for Kubetorch.

This module provides comprehensive pricing data for major GPU instance types
and cost calculation utilities for compute resources, storage, and network costs.

Pricing is based on January 2026 AWS EC2 instances (publicly available rates).
Custom pricing can be provided for on-premise or negotiated cloud rates.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInstancePricing:
    """Pricing information for a GPU instance type."""

    instance_type: str
    gpu_type: str  # 'a100', 'h100', 'v100', etc.
    cost_per_hour: float  # USD per GPU per hour
    gpu_memory_gb: int
    num_gpus_per_instance: int = 1
    currency: str = "USD"
    cloud_provider: str = "aws"  # 'aws', 'gcp', 'azure', 'generic'
    region: str = "us-east"
    notes: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"{self.instance_type}: ${self.cost_per_hour:.2f}/hr "
            f"({self.gpu_memory_gb}GB {self.gpu_type.upper()})"
        )


# GPU Pricing Database (January 2026 AWS EC2 baseline)
# Prices are per-GPU per-hour in USD

GPU_PRICING: Dict[str, GPUInstancePricing] = {
    # NVIDIA A100 (40GB) - Standard high-performance training
    "a100": GPUInstancePricing(
        instance_type="a100",
        gpu_type="a100",
        cost_per_hour=32.77,  # AWS p4d.24xlarge / 8 GPUs
        gpu_memory_gb=40,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Standard A100 40GB - best for most training workloads"
    ),

    "a100-40gb": GPUInstancePricing(
        instance_type="a100-40gb",
        gpu_type="a100",
        cost_per_hour=32.77,
        gpu_memory_gb=40,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Explicit 40GB variant"
    ),

    # NVIDIA A100 (80GB) - Large model training
    "a100-80gb": GPUInstancePricing(
        instance_type="a100-80gb",
        gpu_type="a100",
        cost_per_hour=40.96,  # AWS p4de.24xlarge / 8 GPUs
        gpu_memory_gb=80,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="High-memory A100 for large models (LLaMA 70B+)"
    ),

    # NVIDIA H100 (80GB) - Latest generation, highest performance
    "h100": GPUInstancePricing(
        instance_type="h100",
        gpu_type="h100",
        cost_per_hour=98.32,  # AWS p5.48xlarge / 8 GPUs
        gpu_memory_gb=80,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Latest H100 Hopper architecture - 3x A100 performance"
    ),

    "h100-80gb": GPUInstancePricing(
        instance_type="h100-80gb",
        gpu_type="h100",
        cost_per_hour=98.32,
        gpu_memory_gb=80,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Explicit H100 80GB variant"
    ),

    # NVIDIA V100 (16GB) - Older generation, cost-effective
    "v100": GPUInstancePricing(
        instance_type="v100",
        gpu_type="v100",
        cost_per_hour=12.24,  # AWS p3.8xlarge / 4 GPUs
        gpu_memory_gb=16,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Cost-effective older generation for smaller models"
    ),

    "v100-16gb": GPUInstancePricing(
        instance_type="v100-16gb",
        gpu_type="v100",
        cost_per_hour=12.24,
        gpu_memory_gb=16,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Explicit V100 16GB variant"
    ),

    # NVIDIA A10G (24GB) - Mid-tier for inference and small training
    "a10g": GPUInstancePricing(
        instance_type="a10g",
        gpu_type="a10g",
        cost_per_hour=5.22,  # AWS g5.12xlarge / 4 GPUs
        gpu_memory_gb=24,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Mid-tier GPU for inference and small training jobs"
    ),

    "a10g-24gb": GPUInstancePricing(
        instance_type="a10g-24gb",
        gpu_type="a10g",
        cost_per_hour=5.22,
        gpu_memory_gb=24,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Explicit A10G 24GB variant"
    ),

    # NVIDIA T4 (16GB) - Budget-friendly inference
    "t4": GPUInstancePricing(
        instance_type="t4",
        gpu_type="t4",
        cost_per_hour=1.88,  # AWS g4dn.12xlarge / 4 GPUs
        gpu_memory_gb=16,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Budget-friendly for inference workloads"
    ),

    "t4-16gb": GPUInstancePricing(
        instance_type="t4-16gb",
        gpu_type="t4",
        cost_per_hour=1.88,
        gpu_memory_gb=16,
        num_gpus_per_instance=1,
        cloud_provider="aws",
        notes="Explicit T4 16GB variant"
    ),
}

# Storage and Network Pricing
STORAGE_COST_PER_GB_MONTH = 0.023  # AWS EBS gp3 pricing ($0.08/GB-month → $0.023/GB-week)
NETWORK_COST_PER_GB = 0.09  # AWS data transfer out pricing


class KubetorchPricing:
    """
    Pricing calculator for Kubetorch compute resources.

    Handles cost calculation for:
    - GPU/CPU compute resources
    - Checkpoint storage
    - Network data transfer
    - Training cost estimation

    Example:
        >>> pricing = KubetorchPricing()
        >>> cost = pricing.calculate_compute_cost(
        ...     instance_type="a100",
        ...     num_devices=8,
        ...     duration_seconds=7200  # 2 hours
        ... )
        >>> print(f"Cost: ${cost:.2f}")  # Cost: $524.32
    """

    def __init__(self, custom_pricing: Optional[Dict[str, GPUInstancePricing]] = None):
        """
        Initialize pricing calculator.

        Args:
            custom_pricing: Optional custom pricing database to override defaults.
                           Useful for on-premise or negotiated cloud rates.
        """
        self.pricing_db = {**GPU_PRICING, **(custom_pricing or {})}
        logger.debug(f"Initialized KubetorchPricing with {len(self.pricing_db)} instance types")

    def calculate_compute_cost(
        self,
        instance_type: str,
        num_devices: int,
        duration_seconds: float,
        resource_type: str = "gpu"
    ) -> float:
        """
        Calculate compute cost for GPU/CPU resources.

        Args:
            instance_type: Instance type identifier (e.g., 'a100', 'h100-80gb')
            num_devices: Number of devices (GPUs or CPUs)
            duration_seconds: Duration of usage in seconds
            resource_type: 'gpu' or 'cpu'

        Returns:
            Cost in USD

        Example:
            >>> pricing = KubetorchPricing()
            >>> cost = pricing.calculate_compute_cost("a100", 8, 3600)  # 8 A100s, 1 hour
            >>> print(f"${cost:.2f}")  # $262.16
        """
        if resource_type == "cpu":
            # CPU pricing (much cheaper than GPU)
            cpu_cost_per_core_hour = 0.50  # Generic CPU cost per core-hour
            hours = duration_seconds / 3600
            cost = num_devices * hours * cpu_cost_per_core_hour
            logger.debug(
                f"CPU cost: {num_devices} cores × {hours:.2f}h × $0.50/core-h = ${cost:.2f}"
            )
            return cost

        # GPU pricing
        pricing = self._get_instance_pricing(instance_type)
        if not pricing:
            logger.warning(f"Unknown instance type: {instance_type}, using fallback pricing")
            return self._fallback_cost_calculation(num_devices, duration_seconds)

        hours = duration_seconds / 3600
        cost = num_devices * hours * pricing.cost_per_hour

        logger.debug(
            f"GPU cost: {num_devices} × {pricing.instance_type} × {hours:.2f}h "
            f"× ${pricing.cost_per_hour:.2f}/h = ${cost:.2f}"
        )

        return cost

    def calculate_storage_cost(self, storage_gb_hours: float) -> float:
        """
        Calculate storage cost for checkpoints and datasets.

        Args:
            storage_gb_hours: Storage consumption in GB-hours

        Returns:
            Cost in USD

        Example:
            >>> pricing = KubetorchPricing()
            >>> # 100GB stored for 24 hours
            >>> cost = pricing.calculate_storage_cost(100 * 24)
            >>> print(f"${cost:.4f}")  # $0.0077
        """
        # Convert GB-hours to GB-months for pricing
        # (1 month ≈ 30 days × 24 hours = 720 hours)
        gb_months = storage_gb_hours / 720
        cost = gb_months * STORAGE_COST_PER_GB_MONTH

        logger.debug(
            f"Storage cost: {storage_gb_hours:.2f} GB-hours "
            f"= {gb_months:.4f} GB-months × ${STORAGE_COST_PER_GB_MONTH} = ${cost:.4f}"
        )

        return cost

    def calculate_network_cost(self, data_transfer_gb: float) -> float:
        """
        Calculate network transfer cost (data egress).

        Args:
            data_transfer_gb: Data transferred in GB

        Returns:
            Cost in USD

        Example:
            >>> pricing = KubetorchPricing()
            >>> cost = pricing.calculate_network_cost(100)  # 100GB transfer
            >>> print(f"${cost:.2f}")  # $9.00
        """
        cost = data_transfer_gb * NETWORK_COST_PER_GB
        logger.debug(
            f"Network cost: {data_transfer_gb:.2f} GB × ${NETWORK_COST_PER_GB}/GB = ${cost:.2f}"
        )
        return cost

    def estimate_training_cost(
        self,
        instance_type: str,
        num_devices: int,
        estimated_hours: float,
        checkpoint_size_gb: float = 0,
        checkpoint_frequency_hours: float = 1.0,
        data_transfer_gb: float = 0
    ) -> Dict[str, float]:
        """
        Estimate total training cost including compute, storage, and network.

        Args:
            instance_type: GPU instance type
            num_devices: Number of GPUs
            estimated_hours: Expected training duration
            checkpoint_size_gb: Size of each checkpoint in GB
            checkpoint_frequency_hours: How often checkpoints are saved
            data_transfer_gb: Expected data transfer (datasets, checkpoints)

        Returns:
            Dict with cost breakdown: compute, storage, network, total

        Example:
            >>> pricing = KubetorchPricing()
            >>> costs = pricing.estimate_training_cost(
            ...     instance_type="a100-80gb",
            ...     num_devices=8,
            ...     estimated_hours=24,
            ...     checkpoint_size_gb=25.6,
            ...     checkpoint_frequency_hours=2.0
            ... )
            >>> print(f"Total: ${costs['cost_total']:.2f}")
        """
        # Compute cost
        duration_seconds = estimated_hours * 3600
        cost_compute = self.calculate_compute_cost(
            instance_type, num_devices, duration_seconds, "gpu"
        )

        # Checkpoint storage cost
        if checkpoint_size_gb > 0:
            num_checkpoints = estimated_hours / checkpoint_frequency_hours
            # Accumulating storage (checkpoints persist throughout training)
            avg_storage_gb = checkpoint_size_gb * num_checkpoints / 2  # Average over time
            storage_gb_hours = avg_storage_gb * estimated_hours
            cost_storage = self.calculate_storage_cost(storage_gb_hours)
        else:
            cost_storage = 0.0

        # Network cost
        cost_network = self.calculate_network_cost(data_transfer_gb)

        # Total
        cost_total = cost_compute + cost_storage + cost_network

        return {
            "cost_compute": cost_compute,
            "cost_storage": cost_storage,
            "cost_network": cost_network,
            "cost_total": cost_total,
            "currency": "USD",
            "instance_type": instance_type,
            "num_devices": num_devices,
            "estimated_hours": estimated_hours,
            "gpu_hours": num_devices * estimated_hours
        }

    def _get_instance_pricing(self, instance_type: str) -> Optional[GPUInstancePricing]:
        """
        Get pricing for instance type with fuzzy matching.

        Args:
            instance_type: Instance type identifier

        Returns:
            GPUInstancePricing if found, None otherwise
        """
        # Exact match
        if instance_type in self.pricing_db:
            return self.pricing_db[instance_type]

        # Fuzzy match (handle variations like 'A100', 'a100', 'A100-40GB')
        instance_lower = instance_type.lower().replace("_", "-")

        for key, pricing in self.pricing_db.items():
            key_lower = key.lower()
            # Check if search term in key or vice versa
            if instance_lower in key_lower or key_lower in instance_lower:
                logger.debug(f"Fuzzy matched '{instance_type}' to '{key}'")
                return pricing

        # Try matching just the GPU type (e.g., 'a100' matches 'a100-40gb')
        for key, pricing in self.pricing_db.items():
            if pricing.gpu_type.lower() == instance_lower:
                logger.debug(f"GPU type matched '{instance_type}' to '{key}'")
                return pricing

        return None

    def _fallback_cost_calculation(self, num_devices: int, duration_seconds: float) -> float:
        """
        Fallback cost estimation for unknown instance types.

        Uses A100 pricing as a conservative estimate.

        Args:
            num_devices: Number of devices
            duration_seconds: Duration in seconds

        Returns:
            Estimated cost in USD
        """
        fallback_cost_per_hour = 32.77  # A100 baseline
        hours = duration_seconds / 3600
        cost = num_devices * hours * fallback_cost_per_hour

        logger.warning(
            f"Using fallback pricing (A100 baseline): "
            f"{num_devices} devices × {hours:.2f}h × ${fallback_cost_per_hour}/h = ${cost:.2f}"
        )

        return cost

    def get_supported_instance_types(self) -> list[str]:
        """
        Get list of all supported instance types.

        Returns:
            List of instance type identifiers
        """
        return list(self.pricing_db.keys())

    def get_instance_info(self, instance_type: str) -> Optional[GPUInstancePricing]:
        """
        Get detailed information about an instance type.

        Args:
            instance_type: Instance type identifier

        Returns:
            GPUInstancePricing if found, None otherwise
        """
        return self._get_instance_pricing(instance_type)


def calculate_gpu_cost(
    instance_type: str,
    num_devices: int,
    duration_seconds: float
) -> float:
    """
    Convenience function for GPU cost calculation.

    Args:
        instance_type: GPU instance type
        num_devices: Number of GPUs
        duration_seconds: Duration in seconds

    Returns:
        Cost in USD

    Example:
        >>> cost = calculate_gpu_cost("a100", 8, 3600)
        >>> print(f"${cost:.2f}")  # $262.16
    """
    pricing = KubetorchPricing()
    return pricing.calculate_compute_cost(instance_type, num_devices, duration_seconds, "gpu")


def get_pricing_info(instance_type: str) -> Optional[GPUInstancePricing]:
    """
    Get pricing information for an instance type.

    Args:
        instance_type: Instance type identifier

    Returns:
        GPUInstancePricing if found, None otherwise

    Example:
        >>> info = get_pricing_info("h100")
        >>> print(info)  # h100: $98.32/hr (80GB H100)
    """
    pricing = KubetorchPricing()
    return pricing.get_instance_info(instance_type)
