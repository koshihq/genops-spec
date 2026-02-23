"""Databricks Unity Catalog provider for GenOps AI governance."""

from .adapter import (
    GenOpsDatabricksUnityCatalogAdapter,
    instrument_databricks_unity_catalog,
)
from .cost_aggregator import (
    DatabricksCostSummary,
    DatabricksUnityCatalogCostAggregator,
    WorkspaceCost,
    create_workspace_cost_context,
    get_cost_aggregator,
)
from .governance_monitor import (
    DatabricksGovernanceMonitor,
    DataLineageMetrics,
    GovernanceOperationSummary,
    UnityMetastore,
    get_governance_monitor,
)
from .registration import auto_register, register_databricks_unity_catalog_provider
from .validation import (
    ValidationIssue,
    ValidationResult,
    print_validation_result,
    validate_setup,
)

# Auto-register with instrumentation system if available
auto_register()

__all__ = [
    "GenOpsDatabricksUnityCatalogAdapter",
    "instrument_databricks_unity_catalog",
    "register_databricks_unity_catalog_provider",
    "WorkspaceCost",
    "DatabricksCostSummary",
    "DatabricksUnityCatalogCostAggregator",
    "get_cost_aggregator",
    "create_workspace_cost_context",
    "DataLineageMetrics",
    "GovernanceOperationSummary",
    "UnityMetastore",
    "DatabricksGovernanceMonitor",
    "get_governance_monitor",
    "ValidationIssue",
    "ValidationResult",
    "validate_setup",
    "print_validation_result",
]
