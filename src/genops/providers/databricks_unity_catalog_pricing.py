"""Databricks Unity Catalog pricing models for cost calculation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatabricksPricingConfig:
    """Configuration for Databricks pricing calculations."""

    # DBU (Databricks Unit) pricing per hour in USD
    # These are example rates - actual pricing varies by cloud provider and region
    dbu_rate_usd: float = 0.10  # Default rate per DBU

    # SQL Warehouse pricing (DBU per hour by size)
    sql_warehouse_pricing: dict[str, float] = None  # type: ignore[assignment]

    # Compute cluster pricing (DBU per hour by type)
    compute_cluster_pricing: dict[str, float] = None  # type: ignore[assignment]

    # Storage pricing (USD per GB-month)
    storage_pricing_gb_month: float = 0.12

    # Data transfer pricing (USD per GB)
    data_transfer_pricing_gb: float = 0.09

    def __post_init__(self):
        """Initialize default pricing tables if not provided."""
        if self.sql_warehouse_pricing is None:
            self.sql_warehouse_pricing = {
                "2X-Small": 0.22,
                "X-Small": 0.44,
                "Small": 0.88,
                "Medium": 1.76,
                "Large": 3.52,
                "X-Large": 7.04,
                "2X-Large": 14.08,
                "3X-Large": 21.12,
                "4X-Large": 28.16,
            }

        if self.compute_cluster_pricing is None:
            self.compute_cluster_pricing = {
                "standard": 0.15,
                "memory_optimized": 0.20,
                "storage_optimized": 0.18,
                "compute_optimized": 0.22,
                "gpu_standard": 0.30,
                "gpu_ml": 0.35,
            }


class DatabricksUnityCatalogPricingCalculator:
    """
    Calculator for Databricks Unity Catalog operation costs.

    Handles multi-workspace cost calculation with different resource types.
    """

    def __init__(self, pricing_config: DatabricksPricingConfig | None = None):
        """
        Initialize pricing calculator.

        Args:
            pricing_config: Custom pricing configuration (optional)
        """
        self.config = pricing_config or DatabricksPricingConfig()

    def calculate_sql_warehouse_cost(
        self, warehouse_size: str, duration_ms: int, region: str = "us-west-2"
    ) -> float:
        """
        Calculate cost for SQL warehouse operation.

        Args:
            warehouse_size: SQL warehouse size (e.g., "X-Small")
            duration_ms: Operation duration in milliseconds
            region: Cloud region (affects pricing)

        Returns:
            Cost in USD
        """
        # Get DBU rate for warehouse size
        dbu_per_hour = self.config.sql_warehouse_pricing.get(warehouse_size, 0.44)

        # Convert duration to hours
        duration_hours = duration_ms / (1000 * 60 * 60)

        # Calculate DBU consumed
        dbu_consumed = dbu_per_hour * duration_hours

        # Calculate cost (DBU * rate)
        cost_usd = dbu_consumed * self.config.dbu_rate_usd

        # Apply regional pricing multiplier (simplified)
        regional_multiplier = self._get_regional_multiplier(region)
        cost_usd *= regional_multiplier

        logger.debug(
            f"SQL warehouse cost: {warehouse_size}, {duration_ms}ms, "
            f"{dbu_consumed:.4f} DBU, ${cost_usd:.6f}"
        )

        return cost_usd

    def calculate_compute_cluster_cost(
        self,
        cluster_type: str,
        node_count: int,
        duration_ms: int,
        region: str = "us-west-2",
    ) -> float:
        """
        Calculate cost for compute cluster operation.

        Args:
            cluster_type: Type of compute cluster
            node_count: Number of nodes in cluster
            duration_ms: Operation duration in milliseconds
            region: Cloud region

        Returns:
            Cost in USD
        """
        # Get DBU rate for cluster type
        dbu_per_node_hour = self.config.compute_cluster_pricing.get(cluster_type, 0.15)

        # Convert duration to hours
        duration_hours = duration_ms / (1000 * 60 * 60)

        # Calculate total DBU consumed
        dbu_consumed = dbu_per_node_hour * node_count * duration_hours

        # Calculate cost
        cost_usd = dbu_consumed * self.config.dbu_rate_usd

        # Apply regional pricing multiplier
        regional_multiplier = self._get_regional_multiplier(region)
        cost_usd *= regional_multiplier

        logger.debug(
            f"Compute cluster cost: {cluster_type}, {node_count} nodes, {duration_ms}ms, "
            f"{dbu_consumed:.4f} DBU, ${cost_usd:.6f}"
        )

        return cost_usd

    def calculate_storage_cost(
        self,
        data_size_gb: float,
        storage_duration_days: int = 30,
        region: str = "us-west-2",
    ) -> float:
        """
        Calculate storage cost for Unity Catalog data.

        Args:
            data_size_gb: Data size in GB
            storage_duration_days: Storage duration in days
            region: Cloud region

        Returns:
            Cost in USD
        """
        # Convert days to months (simplified)
        duration_months = storage_duration_days / 30.0

        # Calculate base storage cost
        cost_usd = data_size_gb * self.config.storage_pricing_gb_month * duration_months

        # Apply regional multiplier
        regional_multiplier = self._get_regional_multiplier(region)
        cost_usd *= regional_multiplier

        logger.debug(
            f"Storage cost: {data_size_gb} GB, {storage_duration_days} days, "
            f"${cost_usd:.6f}"
        )

        return cost_usd

    def calculate_data_transfer_cost(
        self,
        transfer_gb: float,
        transfer_type: str = "egress",
        region: str = "us-west-2",
    ) -> float:
        """
        Calculate data transfer cost.

        Args:
            transfer_gb: Data transferred in GB
            transfer_type: Type of transfer (egress, ingress, cross-region)
            region: Cloud region

        Returns:
            Cost in USD
        """
        # Different rates for different transfer types
        transfer_rates = {
            "ingress": 0.0,  # Usually free
            "egress": self.config.data_transfer_pricing_gb,
            "cross_region": self.config.data_transfer_pricing_gb * 1.5,
            "cross_cloud": self.config.data_transfer_pricing_gb * 2.0,
        }

        rate = transfer_rates.get(transfer_type, self.config.data_transfer_pricing_gb)
        cost_usd = transfer_gb * rate

        # Apply regional multiplier
        regional_multiplier = self._get_regional_multiplier(region)
        cost_usd *= regional_multiplier

        logger.debug(
            f"Data transfer cost: {transfer_gb} GB, {transfer_type}, ${cost_usd:.6f}"
        )

        return cost_usd

    def estimate_query_cost(
        self,
        query_complexity: str,
        data_scanned_gb: float,
        warehouse_size: str = "Small",
        region: str = "us-west-2",
    ) -> float:
        """
        Estimate cost for a SQL query based on complexity and data scanned.

        Args:
            query_complexity: Query complexity (simple, medium, complex)
            data_scanned_gb: Amount of data scanned in GB
            warehouse_size: SQL warehouse size
            region: Cloud region

        Returns:
            Estimated cost in USD
        """
        # Estimate query duration based on complexity and data size
        complexity_factors = {
            "simple": 1.0,  # Basic SELECT, simple WHERE
            "medium": 2.5,  # JOINs, aggregations
            "complex": 5.0,  # Complex analytics, window functions
        }

        factor = complexity_factors.get(query_complexity, 2.0)

        # Estimate duration: base time + data-dependent time
        base_time_ms = 1000  # 1 second base
        data_time_ms = data_scanned_gb * 100 * factor  # 100ms per GB * complexity

        estimated_duration_ms = base_time_ms + data_time_ms

        # Calculate warehouse cost
        warehouse_cost = self.calculate_sql_warehouse_cost(
            warehouse_size, int(estimated_duration_ms), region
        )

        # Add data scanning cost (simplified)
        scanning_cost = data_scanned_gb * 0.001  # $0.001 per GB scanned

        total_cost = warehouse_cost + scanning_cost

        logger.debug(
            f"Query cost estimate: {query_complexity}, {data_scanned_gb} GB, "
            f"{estimated_duration_ms:.0f}ms, ${total_cost:.6f}"
        )

        return total_cost

    def _get_regional_multiplier(self, region: str) -> float:
        """
        Get regional pricing multiplier.

        Args:
            region: Cloud region

        Returns:
            Pricing multiplier
        """
        # Simplified regional pricing multipliers
        regional_multipliers = {
            "us-east-1": 1.0,  # Base region
            "us-west-2": 1.0,
            "us-west-1": 1.05,
            "eu-west-1": 1.1,
            "eu-central-1": 1.15,
            "ap-southeast-1": 1.2,
            "ap-northeast-1": 1.25,
        }

        return regional_multipliers.get(region, 1.0)

    def get_pricing_summary(self) -> dict[str, any]:
        """
        Get summary of current pricing configuration.

        Returns:
            Dictionary with pricing information
        """
        return {
            "dbu_rate_usd": self.config.dbu_rate_usd,
            "sql_warehouse_sizes": list(self.config.sql_warehouse_pricing.keys()),
            "compute_cluster_types": list(self.config.compute_cluster_pricing.keys()),
            "storage_pricing_gb_month": self.config.storage_pricing_gb_month,
            "data_transfer_pricing_gb": self.config.data_transfer_pricing_gb,
        }


# Global pricing calculator instance
_pricing_calculator: DatabricksUnityCatalogPricingCalculator | None = None


def get_pricing_calculator(
    pricing_config: DatabricksPricingConfig | None = None,
) -> DatabricksUnityCatalogPricingCalculator:
    """
    Get or create global pricing calculator instance.

    Args:
        pricing_config: Optional custom pricing configuration

    Returns:
        DatabricksUnityCatalogPricingCalculator instance
    """
    global _pricing_calculator
    if _pricing_calculator is None or pricing_config is not None:
        _pricing_calculator = DatabricksUnityCatalogPricingCalculator(pricing_config)
    return _pricing_calculator
