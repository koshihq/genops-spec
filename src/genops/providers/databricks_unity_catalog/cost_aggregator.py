"""Cost aggregation for Databricks Unity Catalog operations."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceCost:
    """Represents cost information for a single workspace operation."""

    workspace_id: str
    operation_type: str
    resource_type: str  # sql_warehouse, compute_cluster, storage, etc.
    cost_usd: float
    compute_units: float | None = None
    duration_ms: int | None = None
    data_processed_gb: float | None = None

    # Governance attributes for cost attribution
    team: str | None = None
    project: str | None = None
    cost_center: str | None = None
    environment: str | None = None


@dataclass
class DatabricksCostSummary:
    """
    Aggregated cost summary for Databricks Unity Catalog operations.

    Supports multi-workspace cost tracking and attribution.
    """

    # Cost breakdowns by different dimensions
    cost_by_workspace: dict[str, float] = field(default_factory=dict)
    cost_by_resource_type: dict[str, float] = field(default_factory=dict)
    cost_by_operation: dict[str, float] = field(default_factory=dict)
    cost_by_team: dict[str, float] = field(default_factory=dict)
    cost_by_project: dict[str, float] = field(default_factory=dict)

    # Resource utilization metrics
    total_compute_units: float = 0.0
    total_duration_ms: int = 0
    total_data_processed_gb: float = 0.0

    # Metadata
    unique_workspaces: set[str] = field(default_factory=set)
    operation_count: int = 0
    total_cost_usd: float = 0.0

    def add_workspace_cost(self, workspace_cost: WorkspaceCost) -> None:
        """Add a workspace cost to the summary."""
        # Update cost breakdowns
        self.cost_by_workspace[workspace_cost.workspace_id] = (
            self.cost_by_workspace.get(workspace_cost.workspace_id, 0.0)
            + workspace_cost.cost_usd
        )

        self.cost_by_resource_type[workspace_cost.resource_type] = (
            self.cost_by_resource_type.get(workspace_cost.resource_type, 0.0)
            + workspace_cost.cost_usd
        )

        self.cost_by_operation[workspace_cost.operation_type] = (
            self.cost_by_operation.get(workspace_cost.operation_type, 0.0)
            + workspace_cost.cost_usd
        )

        # Team and project attribution
        if workspace_cost.team:
            self.cost_by_team[workspace_cost.team] = (
                self.cost_by_team.get(workspace_cost.team, 0.0)
                + workspace_cost.cost_usd
            )

        if workspace_cost.project:
            self.cost_by_project[workspace_cost.project] = (
                self.cost_by_project.get(workspace_cost.project, 0.0)
                + workspace_cost.cost_usd
            )

        # Update resource metrics
        if workspace_cost.compute_units:
            self.total_compute_units += workspace_cost.compute_units

        if workspace_cost.duration_ms:
            self.total_duration_ms += workspace_cost.duration_ms

        if workspace_cost.data_processed_gb:
            self.total_data_processed_gb += workspace_cost.data_processed_gb

        # Update totals
        self.unique_workspaces.add(workspace_cost.workspace_id)
        self.operation_count += 1
        self.total_cost_usd += workspace_cost.cost_usd

    def get_most_expensive_workspace(self) -> str | None:
        """Get the workspace with highest cost."""
        if not self.cost_by_workspace:
            return None
        return max(self.cost_by_workspace, key=self.cost_by_workspace.get)  # type: ignore

    def get_cost_per_gb_processed(self) -> float | None:
        """Calculate cost per GB of data processed."""
        if self.total_data_processed_gb > 0:
            return self.total_cost_usd / self.total_data_processed_gb
        return None

    def get_cost_efficiency_score(self) -> float:
        """
        Calculate cost efficiency score (0-100).
        Higher is better (more compute units per dollar).
        """
        if self.total_cost_usd > 0 and self.total_compute_units > 0:
            efficiency = self.total_compute_units / self.total_cost_usd
            return min(efficiency * 10, 100)  # Scale to 0-100
        return 0.0


class DatabricksUnityCatalogCostAggregator:
    """
    Aggregates and tracks costs for Databricks Unity Catalog operations.

    Handles multi-workspace environments with team-based cost attribution.
    """

    def __init__(self):
        """Initialize the cost aggregator."""
        self.workspace_costs: list[WorkspaceCost] = []
        self.active_contexts: dict[str, DatabricksCostSummary] = {}

        # Databricks pricing (simplified model - real pricing is complex)
        self.sql_warehouse_pricing = {
            "2X-Small": 0.22,  # DBU per hour
            "X-Small": 0.44,
            "Small": 0.88,
            "Medium": 1.76,
            "Large": 3.52,
            "X-Large": 7.04,
            "2X-Large": 14.08,
            "3X-Large": 21.12,
            "4X-Large": 28.16,
        }

        self.compute_pricing = {
            "standard": 0.15,  # DBU per hour
            "memory_optimized": 0.20,
            "storage_optimized": 0.18,
            "compute_optimized": 0.22,
        }

        # Storage pricing (per GB-month)
        self.storage_pricing = 0.12

    def add_sql_warehouse_cost(
        self,
        workspace_id: str,
        warehouse_size: str,
        query_duration_ms: int,
        operation_type: str,
        **governance_attrs,
    ) -> WorkspaceCost:
        """
        Calculate and add SQL warehouse operation cost.

        Args:
            workspace_id: Databricks workspace ID
            warehouse_size: SQL warehouse size (e.g., "X-Small")
            query_duration_ms: Query duration in milliseconds
            operation_type: Type of operation
            **governance_attrs: Governance attributes for cost attribution

        Returns:
            WorkspaceCost object with calculated cost
        """
        # Get DBU rate for warehouse size
        dbu_per_hour = self.sql_warehouse_pricing.get(
            warehouse_size, 0.44
        )  # Default to X-Small

        # Convert duration to hours
        duration_hours = query_duration_ms / (1000 * 60 * 60)

        # Calculate DBU consumed
        dbu_consumed = dbu_per_hour * duration_hours

        # Assume $0.10 per DBU (simplified)
        cost_usd = dbu_consumed * 0.10

        workspace_cost = WorkspaceCost(
            workspace_id=workspace_id,
            operation_type=operation_type,
            resource_type="sql_warehouse",
            cost_usd=cost_usd,
            compute_units=dbu_consumed,
            duration_ms=query_duration_ms,
            team=governance_attrs.get("team"),
            project=governance_attrs.get("project"),
            cost_center=governance_attrs.get("cost_center"),
            environment=governance_attrs.get("environment"),
        )

        self.workspace_costs.append(workspace_cost)

        logger.debug(
            f"Added SQL warehouse cost: {cost_usd:.4f} USD for {workspace_id}, "
            f"size: {warehouse_size}, duration: {query_duration_ms}ms"
        )

        return workspace_cost

    def add_compute_cluster_cost(
        self,
        workspace_id: str,
        cluster_type: str,
        node_count: int,
        duration_ms: int,
        operation_type: str,
        **governance_attrs,
    ) -> WorkspaceCost:
        """
        Calculate and add compute cluster operation cost.

        Args:
            workspace_id: Databricks workspace ID
            cluster_type: Type of cluster (standard, memory_optimized, etc.)
            node_count: Number of nodes in cluster
            duration_ms: Operation duration in milliseconds
            operation_type: Type of operation
            **governance_attrs: Governance attributes for cost attribution

        Returns:
            WorkspaceCost object with calculated cost
        """
        # Get DBU rate for cluster type
        dbu_per_node_hour = self.compute_pricing.get(
            cluster_type, 0.15
        )  # Default to standard

        # Convert duration to hours
        duration_hours = duration_ms / (1000 * 60 * 60)

        # Calculate total DBU consumed
        dbu_consumed = dbu_per_node_hour * node_count * duration_hours

        # Assume $0.10 per DBU (simplified)
        cost_usd = dbu_consumed * 0.10

        workspace_cost = WorkspaceCost(
            workspace_id=workspace_id,
            operation_type=operation_type,
            resource_type="compute_cluster",
            cost_usd=cost_usd,
            compute_units=dbu_consumed,
            duration_ms=duration_ms,
            team=governance_attrs.get("team"),
            project=governance_attrs.get("project"),
            cost_center=governance_attrs.get("cost_center"),
            environment=governance_attrs.get("environment"),
        )

        self.workspace_costs.append(workspace_cost)

        logger.debug(
            f"Added compute cluster cost: {cost_usd:.4f} USD for {workspace_id}, "
            f"type: {cluster_type}, nodes: {node_count}, duration: {duration_ms}ms"
        )

        return workspace_cost

    def add_storage_cost(
        self,
        workspace_id: str,
        data_size_gb: float,
        operation_type: str,
        **governance_attrs,
    ) -> WorkspaceCost:
        """
        Calculate and add storage operation cost.

        Args:
            workspace_id: Databricks workspace ID
            data_size_gb: Data size in GB
            operation_type: Type of operation
            **governance_attrs: Governance attributes for cost attribution

        Returns:
            WorkspaceCost object with calculated cost
        """
        # Storage cost per GB (simplified monthly rate)
        cost_per_gb = self.storage_pricing

        # For operations, use a fraction of monthly cost
        # Assume operation represents 1 day of storage
        cost_usd = cost_per_gb * data_size_gb * (1 / 30)  # Daily cost

        workspace_cost = WorkspaceCost(
            workspace_id=workspace_id,
            operation_type=operation_type,
            resource_type="storage",
            cost_usd=cost_usd,
            data_processed_gb=data_size_gb,
            team=governance_attrs.get("team"),
            project=governance_attrs.get("project"),
            cost_center=governance_attrs.get("cost_center"),
            environment=governance_attrs.get("environment"),
        )

        self.workspace_costs.append(workspace_cost)

        logger.debug(
            f"Added storage cost: {cost_usd:.4f} USD for {workspace_id}, "
            f"data: {data_size_gb} GB, operation: {operation_type}"
        )

        return workspace_cost

    def get_summary(self, context_id: str | None = None) -> DatabricksCostSummary:
        """
        Generate cost summary for all operations or specific context.

        Args:
            context_id: Optional context ID to filter operations

        Returns:
            DatabricksCostSummary with aggregated cost data
        """
        summary = DatabricksCostSummary()

        costs_to_summarize = self.workspace_costs
        if context_id and context_id in self.active_contexts:
            costs_to_summarize = self.active_contexts[context_id].workspace_costs

        for workspace_cost in costs_to_summarize:
            summary.add_workspace_cost(workspace_cost)

        return summary

    def get_team_costs(self) -> dict[str, float]:
        """Get costs grouped by team."""
        summary = self.get_summary()
        return summary.cost_by_team

    def get_workspace_costs(self) -> dict[str, float]:
        """Get costs grouped by workspace."""
        summary = self.get_summary()
        return summary.cost_by_workspace


# Global cost aggregator instance
_cost_aggregator: DatabricksUnityCatalogCostAggregator | None = None


def get_cost_aggregator() -> DatabricksUnityCatalogCostAggregator:
    """Get or create global cost aggregator instance."""
    global _cost_aggregator
    if _cost_aggregator is None:
        _cost_aggregator = DatabricksUnityCatalogCostAggregator()
    return _cost_aggregator


@contextmanager
def create_workspace_cost_context(workspace_id: str, context_name: str = "default"):
    """
    Context manager for tracking costs within a specific workspace context.

    Args:
        workspace_id: Databricks workspace ID
        context_name: Name for the cost tracking context

    Yields:
        DatabricksCostSummary for the context
    """
    aggregator = get_cost_aggregator()
    context_id = f"{workspace_id}:{context_name}"

    # Initialize context
    context_summary = DatabricksCostSummary()
    aggregator.active_contexts[context_id] = context_summary

    try:
        logger.debug(f"Starting workspace cost context: {context_id}")
        yield context_summary

    finally:
        # Finalize context and export telemetry
        final_summary = aggregator.get_summary(context_id)

        logger.info(
            f"Workspace cost context {context_id} completed: "
            f"${final_summary.total_cost_usd:.4f}, "
            f"{final_summary.operation_count} operations"
        )

        # Clean up context
        if context_id in aggregator.active_contexts:
            del aggregator.active_contexts[context_id]
