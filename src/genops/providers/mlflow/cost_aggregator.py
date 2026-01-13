"""Cost aggregation and tracking for MLflow operations.

Handles hierarchical run costs, artifact storage costs, and model registry
operations across experiments.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class RunCost:
    """Cost information for a single MLflow run."""

    run_id: str
    run_name: str
    experiment_id: str
    experiment_name: str

    # Cost breakdown
    tracking_cost: float = 0.0      # API calls
    artifact_cost: float = 0.0      # Artifact storage
    model_cost: float = 0.0         # Model storage
    compute_cost: float = 0.0       # Training compute (if tracked)

    # Resource metrics
    artifact_count: int = 0
    artifact_size_mb: float = 0.0
    model_count: int = 0
    model_size_mb: float = 0.0
    metric_count: int = 0
    param_count: int = 0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Governance
    team: Optional[str] = None
    project: Optional[str] = None
    cost_center: Optional[str] = None

    @property
    def total_cost(self) -> float:
        """Calculate total cost for the run."""
        return (
            self.tracking_cost +
            self.artifact_cost +
            self.model_cost +
            self.compute_cost
        )


@dataclass
class ExperimentCost:
    """Aggregated cost for an MLflow experiment."""

    experiment_id: str
    experiment_name: str

    # Cost aggregations
    total_cost: float = 0.0
    cost_by_run: Dict[str, float] = field(default_factory=dict)
    cost_by_team: Dict[str, float] = field(default_factory=dict)

    # Run statistics
    run_count: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    # Resource totals
    total_artifacts: int = 0
    total_artifact_size_mb: float = 0.0
    total_models: int = 0
    total_model_size_mb: float = 0.0


@dataclass
class MLflowCostSummary:
    """Comprehensive cost summary for MLflow operations."""

    # Cost breakdowns
    cost_by_experiment: Dict[str, float] = field(default_factory=dict)
    cost_by_operation_type: Dict[str, float] = field(default_factory=dict)
    cost_by_team: Dict[str, float] = field(default_factory=dict)
    cost_by_storage_backend: Dict[str, float] = field(default_factory=dict)

    # Totals
    total_cost: float = 0.0
    operation_count: int = 0

    # Resource metrics
    total_storage_mb: float = 0.0
    total_api_calls: int = 0

    # Unique identifiers
    unique_experiments: Set[str] = field(default_factory=set)
    unique_runs: Set[str] = field(default_factory=set)

    def add_run_cost(self, run_cost: RunCost) -> None:
        """Add a run cost to the summary."""
        # Update cost breakdowns
        self.cost_by_experiment[run_cost.experiment_name] = (
            self.cost_by_experiment.get(run_cost.experiment_name, 0.0) +
            run_cost.total_cost
        )

        if run_cost.team:
            self.cost_by_team[run_cost.team] = (
                self.cost_by_team.get(run_cost.team, 0.0) +
                run_cost.total_cost
            )

        # Update totals
        self.total_cost += run_cost.total_cost
        self.operation_count += 1
        self.unique_experiments.add(run_cost.experiment_id)
        self.unique_runs.add(run_cost.run_id)

        # Update storage
        self.total_storage_mb += run_cost.artifact_size_mb + run_cost.model_size_mb


# ============================================================================
# Cost Calculator
# ============================================================================

class MLflowCostCalculator:
    """
    Cost calculator for MLflow operations.

    Provides methods to calculate costs for different MLflow operations
    based on configurable pricing models.
    """

    def __init__(self):
        """Initialize cost calculator with default pricing."""
        # Pricing configuration (USD)
        self.pricing = {
            'tracking_api_call': 0.0001,  # $0.0001 per API call
            'storage': {
                'local': 0.0,              # Free for local storage
                's3': 0.023,               # $0.023 per GB-month (S3 standard)
                'azure': 0.020,            # $0.020 per GB-month
                'gcs': 0.020,              # $0.020 per GB-month
            },
            'registry_operation': 0.0005,  # $0.0005 per registry operation
        }

    def calculate_tracking_cost(self, operation_count: int = 1) -> float:
        """Calculate cost for tracking API calls."""
        return self.pricing['tracking_api_call'] * operation_count

    def calculate_artifact_cost(
        self,
        artifact_size_mb: float,
        storage_backend: str = 'local'
    ) -> float:
        """Calculate cost for artifact storage."""
        if storage_backend == 'local':
            return 0.0

        # Convert MB to GB
        size_gb = artifact_size_mb / 1024

        # Get storage rate per GB-month
        storage_rate = self.pricing['storage'].get(storage_backend, 0.023)

        # Prorate to daily cost (assume 30 days per month)
        daily_cost = (size_gb * storage_rate) / 30

        return daily_cost

    def calculate_model_cost(
        self,
        model_size_mb: float,
        storage_backend: str = 'local'
    ) -> float:
        """Calculate cost for model storage (same as artifact cost)."""
        return self.calculate_artifact_cost(model_size_mb, storage_backend)

    def calculate_registry_cost(self, model_size_mb: float = 0.0) -> float:
        """Calculate cost for model registry operation."""
        return self.pricing['registry_operation']

    def calculate_run_cost(self) -> float:
        """Calculate cost for run creation (minimal)."""
        return self.pricing['tracking_api_call']


# Singleton instance for reuse
_cost_calculator: Optional[MLflowCostCalculator] = None


def get_cost_calculator() -> MLflowCostCalculator:
    """Get or create the singleton cost calculator instance."""
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = MLflowCostCalculator()
    return _cost_calculator


# ============================================================================
# Main Cost Aggregator
# ============================================================================

class MLflowCostAggregator:
    """
    Cost aggregation and tracking for MLflow operations.

    Handles hierarchical run costs, artifact storage costs, and
    model registry operations across experiments.
    """

    def __init__(self, context_name: str = "mlflow", **kwargs):
        """
        Initialize MLflow cost aggregator.

        Args:
            context_name: Descriptive name for this aggregation context
            **kwargs: Additional configuration options
        """
        self.context_name = context_name
        self.run_costs: List[RunCost] = []
        self.active_runs: Dict[str, RunCost] = {}

        # Cost calculator
        self.calculator = get_cost_calculator()

        logger.debug(f"Initialized MLflow cost aggregator: {context_name}")

    def start_run_tracking(
        self,
        run_id: str,
        run_name: str,
        experiment_id: str,
        experiment_name: str,
        **governance_attrs
    ) -> RunCost:
        """
        Start tracking costs for a new run.

        Args:
            run_id: MLflow run ID
            run_name: Run name
            experiment_id: Experiment ID
            experiment_name: Experiment name
            **governance_attrs: Governance attributes (team, project, etc.)

        Returns:
            RunCost instance for tracking
        """
        run_cost = RunCost(
            run_id=run_id,
            run_name=run_name,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            start_time=datetime.now(),
            team=governance_attrs.get('team'),
            project=governance_attrs.get('project'),
            cost_center=governance_attrs.get('cost_center')
        )

        self.active_runs[run_id] = run_cost
        logger.debug(f"Started tracking run: {run_id}")
        return run_cost

    def add_artifact_cost(
        self,
        run_id: str,
        artifact_size_mb: float,
        storage_backend: str = 'local'
    ) -> float:
        """
        Add artifact storage cost to a run.

        Args:
            run_id: MLflow run ID
            artifact_size_mb: Size of artifact in MB
            storage_backend: Storage backend (local, s3, azure, gcs)

        Returns:
            Cost added to the run
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return 0.0

        # Calculate storage cost
        cost = self.calculator.calculate_artifact_cost(artifact_size_mb, storage_backend)

        run_cost = self.active_runs[run_id]
        run_cost.artifact_cost += cost
        run_cost.artifact_count += 1
        run_cost.artifact_size_mb += artifact_size_mb

        logger.debug(
            f"Added artifact cost to run {run_id}: ${cost:.6f} "
            f"({artifact_size_mb:.2f} MB, {storage_backend})"
        )

        return cost

    def add_model_cost(
        self,
        run_id: str,
        model_size_mb: float,
        storage_backend: str = 'local'
    ) -> float:
        """
        Add model storage cost to a run.

        Args:
            run_id: MLflow run ID
            model_size_mb: Size of model in MB
            storage_backend: Storage backend (local, s3, azure, gcs)

        Returns:
            Cost added to the run
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return 0.0

        # Calculate storage cost
        cost = self.calculator.calculate_model_cost(model_size_mb, storage_backend)

        run_cost = self.active_runs[run_id]
        run_cost.model_cost += cost
        run_cost.model_count += 1
        run_cost.model_size_mb += model_size_mb

        logger.debug(
            f"Added model cost to run {run_id}: ${cost:.6f} "
            f"({model_size_mb:.2f} MB, {storage_backend})"
        )

        return cost

    def add_tracking_cost(self, run_id: str, operation_count: int = 1) -> float:
        """
        Add tracking API call cost to a run.

        Args:
            run_id: MLflow run ID
            operation_count: Number of API operations

        Returns:
            Cost added to the run
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return 0.0

        cost = self.calculator.calculate_tracking_cost(operation_count)

        run_cost = self.active_runs[run_id]
        run_cost.tracking_cost += cost

        # Increment appropriate counters (simplified - would be operation-specific)
        run_cost.metric_count += operation_count

        return cost

    def end_run_tracking(self, run_id: str) -> RunCost:
        """
        End tracking for a run and finalize costs.

        Args:
            run_id: MLflow run ID

        Returns:
            Finalized RunCost

        Raises:
            ValueError: If run_id not found in active runs
        """
        if run_id not in self.active_runs:
            raise ValueError(f"Run {run_id} not found in active runs")

        run_cost = self.active_runs.pop(run_id)
        run_cost.end_time = datetime.now()

        if run_cost.start_time:
            run_cost.duration_seconds = (
                run_cost.end_time - run_cost.start_time
            ).total_seconds()

        self.run_costs.append(run_cost)

        logger.info(
            f"Run {run_id} completed: ${run_cost.total_cost:.6f} "
            f"({run_cost.artifact_count} artifacts, {run_cost.model_count} models, "
            f"{run_cost.duration_seconds:.1f}s)"
        )

        return run_cost

    def get_current_summary(self) -> MLflowCostSummary:
        """
        Get current cost summary including active runs.

        Returns:
            MLflowCostSummary with current costs
        """
        summary = MLflowCostSummary()

        # Add completed runs
        for run_cost in self.run_costs:
            summary.add_run_cost(run_cost)

        # Add active runs (in-progress)
        for run_cost in self.active_runs.values():
            summary.add_run_cost(run_cost)

        return summary

    def get_summary(self) -> MLflowCostSummary:
        """
        Generate comprehensive cost summary.

        Returns:
            MLflowCostSummary with all tracked costs
        """
        return self.get_current_summary()


# ============================================================================
# Context Manager
# ============================================================================

@contextmanager
def create_mlflow_cost_context(
    context_name: str = "mlflow_operation",
    **kwargs
):
    """
    Create a cost tracking context for MLflow operations.

    Args:
        context_name: Descriptive name for the context
        **kwargs: Additional configuration

    Yields:
        MLflowCostAggregator instance

    Example:
        ```python
        with create_mlflow_cost_context("experiment-run") as aggregator:
            # Track costs
            aggregator.add_artifact_cost(run_id, 10.5, 's3')
            aggregator.add_model_cost(run_id, 50.0, 's3')

            # Get summary
            summary = aggregator.get_summary()
            print(f"Total cost: ${summary.total_cost:.6f}")
        ```
    """
    aggregator = MLflowCostAggregator(context_name=context_name, **kwargs)

    try:
        logger.debug(f"Starting MLflow cost context: {context_name}")
        yield aggregator

    finally:
        # Generate final summary
        summary = aggregator.get_summary()

        logger.info(
            f"MLflow cost context '{context_name}' completed: "
            f"${summary.total_cost:.6f} across {summary.operation_count} operations"
        )


# ============================================================================
# Singleton Aggregator
# ============================================================================

_global_aggregator: Optional[MLflowCostAggregator] = None


def get_cost_aggregator() -> MLflowCostAggregator:
    """
    Get or create the global cost aggregator instance.

    Returns:
        Global MLflowCostAggregator instance
    """
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = MLflowCostAggregator(context_name="global")
    return _global_aggregator
