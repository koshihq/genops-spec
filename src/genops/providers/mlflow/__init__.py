"""MLflow provider for GenOps AI governance.

This module provides comprehensive governance telemetry, cost tracking, and policy
enforcement for MLflow experiment tracking and model registry operations.

Example:
    ```python
    from genops.providers.mlflow import instrument_mlflow

    # Create adapter with governance attributes
    adapter = instrument_mlflow(
        tracking_uri="http://localhost:5000",
        team="ml-team",
        project="model-optimization"
    )

    # Track MLflow run with governance
    with adapter.track_mlflow_run(
        experiment_name="optimization-experiment",
        run_name="run-001"
    ) as run:
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_metric("accuracy", 0.92)
    ```

Zero-code auto-instrumentation:
    ```python
    from genops.providers.mlflow import auto_instrument_mlflow

    # Enable governance tracking with zero code changes
    auto_instrument_mlflow()

    # Your existing MLflow code works automatically
    import mlflow
    with mlflow.start_run():
        mlflow.log_metric("metric1", 0.95)
    ```
"""

from __future__ import annotations

# Import core components
from .adapter import (
    GenOpsMLflowAdapter,
    instrument_mlflow,
)
from .cost_aggregator import (
    RunCost,
    ExperimentCost,
    MLflowCostSummary,
    MLflowCostAggregator,
    create_mlflow_cost_context,
    get_cost_aggregator,
    get_cost_calculator,
    MLflowCostCalculator,
)
from .registration import (
    auto_register,
    register_mlflow_provider,
    auto_instrument_mlflow,
)
from .validation import (
    ValidationIssue,
    ValidationResult,
    validate_setup,
    print_validation_result,
)

# Auto-register with instrumentation system if available
auto_register()

__all__ = [
    # Adapter
    "GenOpsMLflowAdapter",
    "instrument_mlflow",

    # Cost tracking
    "RunCost",
    "ExperimentCost",
    "MLflowCostSummary",
    "MLflowCostAggregator",
    "MLflowCostCalculator",
    "create_mlflow_cost_context",
    "get_cost_aggregator",
    "get_cost_calculator",

    # Registration
    "auto_register",
    "register_mlflow_provider",
    "auto_instrument_mlflow",

    # Validation
    "ValidationIssue",
    "ValidationResult",
    "validate_setup",
    "print_validation_result",
]

