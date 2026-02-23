"""
Multi-resource cost aggregation for Kubetorch compute operations.

This module provides cost tracking and aggregation across multiple resource types:
- GPU/CPU compute resources
- Checkpoint storage
- Network data transfer
- Distributed training operations

Pattern follows LangChain cost_aggregator for consistency with GenOps patterns.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ComputeResourceCost:
    """Represents cost for a single compute resource usage."""

    resource_type: str  # 'gpu', 'cpu', 'storage', 'network'
    instance_type: str  # 'a100', 'h100', 'v100', 'cpu', 'storage', 'network'
    quantity: float  # GPU-hours, CPU-hours, GB-hours, GB transferred
    cost: float
    currency: str = "USD"
    operation_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.resource_type}:{self.instance_type} "
            f"{self.quantity:.2f} units = ${self.cost:.2f}"
        )


@dataclass
class ComputeCostSummary:
    """Aggregated cost summary for Kubetorch operations."""

    total_cost: float = 0.0
    currency: str = "USD"
    resource_costs: list[ComputeResourceCost] = field(default_factory=list)

    # Cost breakdowns
    cost_by_resource_type: dict[str, float] = field(default_factory=dict)
    cost_by_instance_type: dict[str, float] = field(default_factory=dict)
    cost_by_operation: dict[str, float] = field(default_factory=dict)

    # Resource consumption totals
    total_gpu_hours: float = 0.0
    total_cpu_hours: float = 0.0
    total_storage_gb_hours: float = 0.0
    total_network_gb: float = 0.0

    # Unique resources used
    unique_instance_types: set[str] = field(default_factory=set)
    total_operations: int = 0
    total_resources: int = 0

    # Time tracking
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def add_resource_cost(self, resource_cost: ComputeResourceCost) -> None:
        """
        Add a resource cost and recalculate aggregates.

        Args:
            resource_cost: ComputeResourceCost to add
        """
        self.resource_costs.append(resource_cost)
        self.total_resources += 1
        self._calculate_aggregates()

    def _calculate_aggregates(self) -> None:
        """Calculate aggregate cost and consumption values."""
        # Reset aggregates
        self.total_cost = 0.0
        self.cost_by_resource_type.clear()
        self.cost_by_instance_type.clear()
        self.cost_by_operation.clear()
        self.unique_instance_types.clear()
        self.total_gpu_hours = 0.0
        self.total_cpu_hours = 0.0
        self.total_storage_gb_hours = 0.0
        self.total_network_gb = 0.0

        # Aggregate across all resource costs
        for rc in self.resource_costs:
            # Total cost
            self.total_cost += rc.cost

            # Cost by resource type
            if rc.resource_type not in self.cost_by_resource_type:
                self.cost_by_resource_type[rc.resource_type] = 0.0
            self.cost_by_resource_type[rc.resource_type] += rc.cost

            # Cost by instance type
            if rc.instance_type not in self.cost_by_instance_type:
                self.cost_by_instance_type[rc.instance_type] = 0.0
            self.cost_by_instance_type[rc.instance_type] += rc.cost

            # Cost by operation (if specified)
            if rc.operation_name:
                if rc.operation_name not in self.cost_by_operation:
                    self.cost_by_operation[rc.operation_name] = 0.0
                self.cost_by_operation[rc.operation_name] += rc.cost

            # Track unique instance types
            self.unique_instance_types.add(rc.instance_type)

            # Resource consumption totals
            if rc.resource_type == "gpu":
                self.total_gpu_hours += rc.quantity
            elif rc.resource_type == "cpu":
                self.total_cpu_hours += rc.quantity
            elif rc.resource_type == "storage":
                self.total_storage_gb_hours += rc.quantity
            elif rc.resource_type == "network":
                self.total_network_gb += rc.quantity

    def get_summary_dict(self) -> dict[str, any]:
        """
        Get summary as dictionary for serialization.

        Returns:
            Dict with summary data
        """
        return {
            "total_cost": self.total_cost,
            "currency": self.currency,
            "cost_by_resource_type": self.cost_by_resource_type,
            "cost_by_instance_type": self.cost_by_instance_type,
            "cost_by_operation": self.cost_by_operation,
            "total_gpu_hours": self.total_gpu_hours,
            "total_cpu_hours": self.total_cpu_hours,
            "total_storage_gb_hours": self.total_storage_gb_hours,
            "total_network_gb": self.total_network_gb,
            "unique_instance_types": list(self.unique_instance_types),
            "total_operations": self.total_operations,
            "total_resources": self.total_resources,
            "duration_seconds": self.duration_seconds,
        }

    def __str__(self) -> str:
        """String representation of cost summary."""
        lines = [
            "Compute Cost Summary:",
            f"  Total Cost: ${self.total_cost:.2f} {self.currency}",
            f"  GPU-hours: {self.total_gpu_hours:.2f}",
            f"  CPU-hours: {self.total_cpu_hours:.2f}",
            f"  Storage: {self.total_storage_gb_hours:.2f} GB-hours",
            f"  Network: {self.total_network_gb:.2f} GB",
            f"  Operations: {self.total_operations}",
            f"  Resources: {self.total_resources}",
        ]

        if self.cost_by_resource_type:
            lines.append("  Cost by Resource Type:")
            for rtype, cost in sorted(self.cost_by_resource_type.items()):
                lines.append(f"    {rtype}: ${cost:.2f}")

        if self.cost_by_instance_type:
            lines.append("  Cost by Instance Type:")
            for itype, cost in sorted(self.cost_by_instance_type.items()):
                lines.append(f"    {itype}: ${cost:.2f}")

        return "\n".join(lines)


class KubetorchCostAggregator:
    """
    Aggregates costs across multiple compute resources.

    Tracks active operations and provides cost aggregation for:
    - GPU compute resources
    - CPU compute resources
    - Storage (checkpoints, datasets)
    - Network (data transfer)

    Example:
        >>> aggregator = KubetorchCostAggregator()
        >>> aggregator.start_operation_tracking("train-job-001")
        >>> aggregator.add_compute_cost(
        ...     "train-job-001",
        ...     resource_type="gpu",
        ...     instance_type="a100",
        ...     quantity=8.0,  # 8 GPU-hours
        ...     operation_name="training"
        ... )
        >>> summary = aggregator.finalize_operation_tracking("train-job-001")
        >>> print(f"Total: ${summary.total_cost:.2f}")
    """

    def __init__(self):
        """Initialize cost aggregator."""
        self.active_operations: dict[str, ComputeCostSummary] = {}
        self._setup_pricing_calculator()
        logger.debug("Initialized KubetorchCostAggregator")

    def _setup_pricing_calculator(self) -> None:
        """Setup pricing calculator for different resource types."""
        from .pricing import KubetorchPricing

        self.pricing = KubetorchPricing()

    def start_operation_tracking(self, operation_id: str) -> None:
        """
        Start tracking costs for a compute operation.

        Args:
            operation_id: Unique operation identifier
        """
        if operation_id in self.active_operations:
            logger.warning(f"Operation {operation_id} already being tracked")
            return

        summary = ComputeCostSummary(start_time=time.time())
        self.active_operations[operation_id] = summary
        logger.debug(f"Started tracking operation: {operation_id}")

    def add_compute_cost(
        self,
        operation_id: str,
        resource_type: str,
        instance_type: str,
        quantity: float,
        operation_name: Optional[str] = None,
        **metadata,
    ) -> Optional[ComputeResourceCost]:
        """
        Add compute resource cost to operation tracking.

        Args:
            operation_id: Unique operation identifier
            resource_type: Type of resource ('gpu', 'cpu', 'storage', 'network')
            instance_type: Instance type ('a100', 'h100', etc.)
            quantity: Resource quantity (GPU-hours, GB-hours, etc.)
            operation_name: Name of the operation (optional)
            **metadata: Additional metadata

        Returns:
            ComputeResourceCost object if successful, None if operation not found
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active tracking")
            return None

        # Calculate cost using pricing calculator
        cost = self._calculate_resource_cost(resource_type, instance_type, quantity)

        resource_cost = ComputeResourceCost(
            resource_type=resource_type,
            instance_type=instance_type,
            quantity=quantity,
            cost=cost,
            operation_name=operation_name,
            metadata=metadata,
        )

        self.active_operations[operation_id].add_resource_cost(resource_cost)

        logger.debug(
            f"Added {resource_type} cost to {operation_id}: "
            f"{quantity:.2f} {instance_type} = ${cost:.2f}"
        )

        return resource_cost

    def add_gpu_cost(
        self,
        operation_id: str,
        instance_type: str,
        gpu_hours: float,
        operation_name: Optional[str] = None,
        **metadata,
    ) -> Optional[ComputeResourceCost]:
        """
        Add GPU cost (convenience method).

        Args:
            operation_id: Unique operation identifier
            instance_type: GPU instance type
            gpu_hours: GPU-hours consumed
            operation_name: Name of the operation
            **metadata: Additional metadata

        Returns:
            ComputeResourceCost object if successful
        """
        return self.add_compute_cost(
            operation_id=operation_id,
            resource_type="gpu",
            instance_type=instance_type,
            quantity=gpu_hours,
            operation_name=operation_name,
            **metadata,
        )

    def add_storage_cost(
        self,
        operation_id: str,
        storage_gb_hours: float,
        operation_name: Optional[str] = None,
        **metadata,
    ) -> Optional[ComputeResourceCost]:
        """
        Add storage cost (convenience method).

        Args:
            operation_id: Unique operation identifier
            storage_gb_hours: Storage in GB-hours
            operation_name: Name of the operation
            **metadata: Additional metadata

        Returns:
            ComputeResourceCost object if successful
        """
        return self.add_compute_cost(
            operation_id=operation_id,
            resource_type="storage",
            instance_type="storage",
            quantity=storage_gb_hours,
            operation_name=operation_name,
            **metadata,
        )

    def add_network_cost(
        self,
        operation_id: str,
        data_transfer_gb: float,
        operation_name: Optional[str] = None,
        **metadata,
    ) -> Optional[ComputeResourceCost]:
        """
        Add network cost (convenience method).

        Args:
            operation_id: Unique operation identifier
            data_transfer_gb: Data transfer in GB
            operation_name: Name of the operation
            **metadata: Additional metadata

        Returns:
            ComputeResourceCost object if successful
        """
        return self.add_compute_cost(
            operation_id=operation_id,
            resource_type="network",
            instance_type="network",
            quantity=data_transfer_gb,
            operation_name=operation_name,
            **metadata,
        )

    def _calculate_resource_cost(
        self, resource_type: str, instance_type: str, quantity: float
    ) -> float:
        """
        Calculate cost for specific resource usage.

        Args:
            resource_type: Type of resource
            instance_type: Instance type
            quantity: Quantity of resource

        Returns:
            Cost in USD
        """
        try:
            if resource_type == "gpu":
                # Calculate GPU cost: quantity is GPU-hours
                # Convert to duration_seconds for pricing calculator
                num_devices = 1  # Already in GPU-hours
                duration_seconds = quantity * 3600
                return self.pricing.calculate_compute_cost(
                    instance_type=instance_type,
                    num_devices=num_devices,
                    duration_seconds=duration_seconds,
                    resource_type="gpu",
                )
            elif resource_type == "cpu":
                # Calculate CPU cost: quantity is CPU-hours
                num_devices = 1
                duration_seconds = quantity * 3600
                return self.pricing.calculate_compute_cost(
                    instance_type=instance_type,
                    num_devices=num_devices,
                    duration_seconds=duration_seconds,
                    resource_type="cpu",
                )
            elif resource_type == "storage":
                # Calculate storage cost: quantity is GB-hours
                return self.pricing.calculate_storage_cost(quantity)
            elif resource_type == "network":
                # Calculate network cost: quantity is GB transferred
                return self.pricing.calculate_network_cost(quantity)
            else:
                logger.warning(f"Unknown resource type: {resource_type}")
                return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate resource cost: {e}")
            return 0.0

    def finalize_operation_tracking(
        self, operation_id: str, increment_operation_count: bool = True
    ) -> Optional[ComputeCostSummary]:
        """
        Finalize and return cost summary for an operation.

        Args:
            operation_id: Unique operation identifier
            increment_operation_count: Whether to increment operation count

        Returns:
            ComputeCostSummary if operation found, None otherwise
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found")
            return None

        summary = self.active_operations.pop(operation_id)
        summary.end_time = time.time()

        if increment_operation_count:
            summary.total_operations = 1

        logger.debug(
            f"Finalized operation {operation_id}: "
            f"${summary.total_cost:.2f}, {summary.total_resources} resources"
        )

        return summary

    def get_active_operations(self) -> list[str]:
        """
        Get list of active operation IDs.

        Returns:
            List of operation IDs currently being tracked
        """
        return list(self.active_operations.keys())

    def clear_all_operations(self) -> None:
        """Clear all active operations (for cleanup)."""
        self.active_operations.clear()
        logger.debug("Cleared all active operations")


# ==========================================
# Context Manager
# ==========================================


def create_compute_cost_context(operation_id: str):
    """
    Create a context manager for compute cost tracking.

    Args:
        operation_id: Unique operation identifier

    Returns:
        ComputeCostContext instance

    Example:
        >>> with create_compute_cost_context("train-job-001") as ctx:
        ...     ctx.add_gpu_cost("a100", 8.0)
        ...     ctx.add_storage_cost(100 * 24)  # 100GB for 24 hours
        >>> print(ctx.summary)
    """
    return ComputeCostContext(operation_id)


class ComputeCostContext:
    """Context manager for compute cost tracking."""

    def __init__(self, operation_id: str):
        """
        Initialize cost tracking context.

        Args:
            operation_id: Unique operation identifier
        """
        self.operation_id = operation_id
        self.aggregator = _get_or_create_aggregator()
        self.summary: Optional[ComputeCostSummary] = None
        self.start_time: Optional[float] = None

    def __enter__(self) -> "ComputeCostContext":
        """Start cost tracking."""
        self.start_time = time.time()
        self.aggregator.start_operation_tracking(self.operation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize cost tracking."""
        self.summary = self.aggregator.finalize_operation_tracking(self.operation_id)

    def add_compute_cost(
        self,
        resource_type: str,
        instance_type: str,
        quantity: float,
        operation_name: Optional[str] = None,
        **metadata,
    ) -> Optional[ComputeResourceCost]:
        """Add compute resource cost within this context."""
        return self.aggregator.add_compute_cost(
            self.operation_id,
            resource_type,
            instance_type,
            quantity,
            operation_name,
            **metadata,
        )

    def add_gpu_cost(
        self,
        instance_type: str,
        gpu_hours: float,
        operation_name: Optional[str] = None,
        **metadata,
    ) -> Optional[ComputeResourceCost]:
        """Add GPU cost (convenience method)."""
        return self.aggregator.add_gpu_cost(
            self.operation_id, instance_type, gpu_hours, operation_name, **metadata
        )

    def add_storage_cost(
        self, storage_gb_hours: float, operation_name: Optional[str] = None, **metadata
    ) -> Optional[ComputeResourceCost]:
        """Add storage cost (convenience method)."""
        return self.aggregator.add_storage_cost(
            self.operation_id, storage_gb_hours, operation_name, **metadata
        )

    def add_network_cost(
        self, data_transfer_gb: float, operation_name: Optional[str] = None, **metadata
    ) -> Optional[ComputeResourceCost]:
        """Add network cost (convenience method)."""
        return self.aggregator.add_network_cost(
            self.operation_id, data_transfer_gb, operation_name, **metadata
        )


# ==========================================
# Global Aggregator (Singleton Pattern)
# ==========================================

_global_aggregator: Optional[KubetorchCostAggregator] = None


def _get_or_create_aggregator() -> KubetorchCostAggregator:
    """Get or create global cost aggregator instance."""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = KubetorchCostAggregator()
    return _global_aggregator


def get_cost_aggregator() -> KubetorchCostAggregator:
    """
    Get the global cost aggregator instance.

    Returns:
        KubetorchCostAggregator singleton instance
    """
    return _get_or_create_aggregator()


def reset_cost_aggregator() -> None:
    """Reset the global cost aggregator (mainly for testing)."""
    global _global_aggregator
    if _global_aggregator is not None:
        _global_aggregator.clear_all_operations()
    _global_aggregator = None
