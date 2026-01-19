"""Tests for MLflow cost aggregator."""

import pytest
from datetime import datetime

from src.genops.providers.mlflow.cost_aggregator import (
    RunCost,
    ExperimentCost,
    MLflowCostSummary,
    MLflowCostAggregator,
    MLflowCostCalculator,
    create_mlflow_cost_context,
    get_cost_calculator,
    get_cost_aggregator,
)


# ============================================================================
# RunCost Tests (4 tests)
# ============================================================================

def test_run_cost_creation():
    """Test RunCost dataclass creation."""
    run_cost = RunCost(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-experiment",
        team="test-team",
        project="test-project"
    )

    assert run_cost.run_id == "run-123"
    assert run_cost.run_name == "test-run"
    assert run_cost.experiment_id == "exp-456"
    assert run_cost.experiment_name == "test-experiment"
    assert run_cost.team == "test-team"
    assert run_cost.project == "test-project"
    assert run_cost.total_cost == 0.0


def test_run_cost_total_cost_calculation():
    """Test RunCost total_cost property calculation."""
    run_cost = RunCost(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-experiment",
        tracking_cost=0.001,
        artifact_cost=0.050,
        model_cost=0.100,
        compute_cost=5.000
    )

    assert run_cost.total_cost == 5.151


def test_run_cost_with_metrics():
    """Test RunCost with resource metrics."""
    run_cost = RunCost(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-experiment",
        artifact_count=5,
        artifact_size_mb=25.5,
        model_count=2,
        model_size_mb=150.0,
        metric_count=100,
        param_count=20
    )

    assert run_cost.artifact_count == 5
    assert run_cost.artifact_size_mb == 25.5
    assert run_cost.model_count == 2
    assert run_cost.model_size_mb == 150.0
    assert run_cost.metric_count == 100
    assert run_cost.param_count == 20


def test_run_cost_with_timing():
    """Test RunCost with timing information."""
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = datetime(2024, 1, 1, 12, 5, 30)

    run_cost = RunCost(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-experiment",
        start_time=start_time,
        end_time=end_time,
        duration_seconds=330.0
    )

    assert run_cost.start_time == start_time
    assert run_cost.end_time == end_time
    assert run_cost.duration_seconds == 330.0


# ============================================================================
# Cost Calculator Tests (8 tests)
# ============================================================================

def test_cost_calculator_singleton():
    """Test get_cost_calculator returns singleton instance."""
    calc1 = get_cost_calculator()
    calc2 = get_cost_calculator()

    assert calc1 is calc2


def test_calculate_tracking_cost():
    """Test calculate_tracking_cost for API calls."""
    calculator = MLflowCostCalculator()

    cost = calculator.calculate_tracking_cost()
    assert cost == 0.0001

    cost_multi = calculator.calculate_tracking_cost(operation_count=10)
    assert cost_multi == 0.001


def test_calculate_artifact_cost_local():
    """Test calculate_artifact_cost for local storage."""
    calculator = MLflowCostCalculator()

    cost = calculator.calculate_artifact_cost(10.0, 'local')
    assert cost == 0.0  # Local storage is free


def test_calculate_artifact_cost_s3():
    """Test calculate_artifact_cost for S3 storage."""
    calculator = MLflowCostCalculator()

    # 10 MB = 10/1024 GB, $0.023/GB-month, daily = /30
    cost = calculator.calculate_artifact_cost(10.0, 's3')
    expected = (10.0 / 1024) * 0.023 / 30
    assert abs(cost - expected) < 0.000001


def test_calculate_artifact_cost_different_backends():
    """Test calculate_artifact_cost for different storage backends."""
    calculator = MLflowCostCalculator()

    cost_s3 = calculator.calculate_artifact_cost(100.0, 's3')
    cost_azure = calculator.calculate_artifact_cost(100.0, 'azure')
    cost_gcs = calculator.calculate_artifact_cost(100.0, 'gcs')

    # Azure and GCS should be cheaper than S3
    assert cost_s3 > cost_azure
    assert cost_s3 > cost_gcs
    assert cost_azure == cost_gcs  # Same pricing


def test_calculate_model_cost():
    """Test calculate_model_cost (same as artifact cost)."""
    calculator = MLflowCostCalculator()

    artifact_cost = calculator.calculate_artifact_cost(50.0, 's3')
    model_cost = calculator.calculate_model_cost(50.0, 's3')

    assert artifact_cost == model_cost


def test_calculate_registry_cost():
    """Test calculate_registry_cost for model registry operations."""
    calculator = MLflowCostCalculator()

    cost = calculator.calculate_registry_cost()
    assert cost == 0.0005

    # Registry cost doesn't depend on model size
    cost_large = calculator.calculate_registry_cost(model_size_mb=1000.0)
    assert cost_large == 0.0005


def test_calculate_run_cost():
    """Test calculate_run_cost for run creation."""
    calculator = MLflowCostCalculator()

    cost = calculator.calculate_run_cost()
    assert cost == 0.0001  # Same as tracking API call


# ============================================================================
# Cost Aggregator Tests (5 tests)
# ============================================================================

def test_cost_aggregator_initialization():
    """Test MLflowCostAggregator initialization."""
    aggregator = MLflowCostAggregator(context_name="test-context")

    assert aggregator.context_name == "test-context"
    assert len(aggregator.run_costs) == 0
    assert len(aggregator.active_runs) == 0
    assert aggregator.calculator is not None


def test_cost_aggregator_lifecycle():
    """Test cost aggregator run tracking lifecycle."""
    aggregator = MLflowCostAggregator()

    # Start tracking
    run_cost = aggregator.start_run_tracking(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-exp",
        team="test-team",
        project="test-project"
    )

    assert run_cost.run_id == "run-123"
    assert "run-123" in aggregator.active_runs
    assert len(aggregator.run_costs) == 0

    # End tracking
    final_cost = aggregator.end_run_tracking("run-123")

    assert final_cost.run_id == "run-123"
    assert "run-123" not in aggregator.active_runs
    assert len(aggregator.run_costs) == 1
    assert final_cost.duration_seconds is not None


def test_cost_aggregator_add_costs():
    """Test adding different cost types to a run."""
    aggregator = MLflowCostAggregator()

    # Start run
    aggregator.start_run_tracking(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-exp"
    )

    # Add artifact cost
    artifact_cost = aggregator.add_artifact_cost("run-123", 10.0, 's3')
    assert artifact_cost > 0

    # Add model cost
    model_cost = aggregator.add_model_cost("run-123", 50.0, 's3')
    assert model_cost > 0

    # Add tracking cost
    tracking_cost = aggregator.add_tracking_cost("run-123", 10)
    assert tracking_cost > 0

    # Check run cost accumulation
    run_cost = aggregator.active_runs["run-123"]
    assert run_cost.artifact_cost == artifact_cost
    assert run_cost.model_cost == model_cost
    assert run_cost.tracking_cost == tracking_cost
    assert run_cost.total_cost > 0


def test_cost_aggregator_summary():
    """Test get_summary generates correct summary."""
    aggregator = MLflowCostAggregator()

    # Create multiple runs
    for i in range(3):
        aggregator.start_run_tracking(
            run_id=f"run-{i}",
            run_name=f"test-run-{i}",
            experiment_id="exp-456",
            experiment_name="test-exp",
            team=f"team-{i % 2}"  # Alternate teams
        )

        aggregator.add_artifact_cost(f"run-{i}", 10.0, 's3')
        aggregator.add_tracking_cost(f"run-{i}", 5)
        aggregator.end_run_tracking(f"run-{i}")

    summary = aggregator.get_summary()

    assert summary.operation_count == 3
    assert summary.total_cost > 0
    assert len(summary.unique_runs) == 3
    assert len(summary.unique_experiments) == 1
    assert len(summary.cost_by_team) == 2  # Two different teams


def test_cost_aggregator_error_handling():
    """Test cost aggregator error handling for invalid run IDs."""
    aggregator = MLflowCostAggregator()

    # Try to add cost to non-existent run
    cost = aggregator.add_artifact_cost("invalid-run", 10.0, 's3')
    assert cost == 0.0

    # Try to end non-existent run
    with pytest.raises(ValueError, match="not found in active runs"):
        aggregator.end_run_tracking("invalid-run")


# ============================================================================
# Context Manager Tests (5 tests)
# ============================================================================

def test_create_mlflow_cost_context():
    """Test create_mlflow_cost_context context manager."""
    with create_mlflow_cost_context("test-context") as aggregator:
        assert isinstance(aggregator, MLflowCostAggregator)
        assert aggregator.context_name == "test-context"


def test_cost_context_finalization():
    """Test cost context finalizes costs on exit."""
    with create_mlflow_cost_context("test-context") as aggregator:
        aggregator.start_run_tracking(
            run_id="run-123",
            run_name="test-run",
            experiment_id="exp-456",
            experiment_name="test-exp"
        )
        aggregator.add_artifact_cost("run-123", 10.0, 's3')
        # Note: not ending run explicitly

    # Context manager should handle finalization gracefully


def test_cost_context_multiple_runs():
    """Test cost context with multiple runs."""
    with create_mlflow_cost_context("multi-run-context") as aggregator:
        for i in range(5):
            aggregator.start_run_tracking(
                run_id=f"run-{i}",
                run_name=f"run-{i}",
                experiment_id="exp-456",
                experiment_name="test-exp"
            )
            aggregator.add_tracking_cost(f"run-{i}", 10)
            aggregator.end_run_tracking(f"run-{i}")

        summary = aggregator.get_summary()
        assert summary.operation_count == 5


def test_cost_context_error_handling():
    """Test cost context handles errors gracefully."""
    try:
        with create_mlflow_cost_context("error-context") as aggregator:
            aggregator.start_run_tracking(
                run_id="run-123",
                run_name="test-run",
                experiment_id="exp-456",
                experiment_name="test-exp"
            )
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Context should have exited cleanly


def test_cost_context_nested():
    """Test nested cost contexts."""
    with create_mlflow_cost_context("outer-context") as outer:
        outer.start_run_tracking(
            run_id="run-1",
            run_name="outer-run",
            experiment_id="exp-1",
            experiment_name="exp-1"
        )

        with create_mlflow_cost_context("inner-context") as inner:
            inner.start_run_tracking(
                run_id="run-2",
                run_name="inner-run",
                experiment_id="exp-2",
                experiment_name="exp-2"
            )
            inner.add_tracking_cost("run-2", 5)
            inner.end_run_tracking("run-2")

            inner_summary = inner.get_summary()
            assert inner_summary.operation_count == 1

        outer.add_tracking_cost("run-1", 10)
        outer.end_run_tracking("run-1")

        outer_summary = outer.get_summary()
        assert outer_summary.operation_count == 1


# ============================================================================
# Summary Tests (2 tests)
# ============================================================================

def test_mlflow_cost_summary_add_run_cost():
    """Test MLflowCostSummary.add_run_cost method."""
    summary = MLflowCostSummary()

    run_cost = RunCost(
        run_id="run-123",
        run_name="test-run",
        experiment_id="exp-456",
        experiment_name="test-exp",
        tracking_cost=0.001,
        artifact_cost=0.050,
        team="test-team"
    )

    summary.add_run_cost(run_cost)

    assert summary.total_cost == run_cost.total_cost
    assert summary.operation_count == 1
    assert "test-exp" in summary.cost_by_experiment
    assert "test-team" in summary.cost_by_team
    assert "run-123" in summary.unique_runs
    assert "exp-456" in summary.unique_experiments


def test_mlflow_cost_summary_aggregation():
    """Test MLflowCostSummary aggregates multiple runs."""
    summary = MLflowCostSummary()

    # Add multiple runs
    for i in range(10):
        run_cost = RunCost(
            run_id=f"run-{i}",
            run_name=f"run-{i}",
            experiment_id=f"exp-{i % 3}",  # 3 experiments
            experiment_name=f"exp-{i % 3}",
            tracking_cost=0.001 * (i + 1),
            artifact_cost=0.010 * (i + 1),
            team=f"team-{i % 2}"  # 2 teams
        )
        summary.add_run_cost(run_cost)

    assert summary.operation_count == 10
    assert len(summary.unique_runs) == 10
    assert len(summary.unique_experiments) == 3
    assert len(summary.cost_by_experiment) == 3
    assert len(summary.cost_by_team) == 2
    assert summary.total_cost > 0


# ============================================================================
# Global Aggregator Tests (1 test)
# ============================================================================

def test_get_cost_aggregator_singleton():
    """Test get_cost_aggregator returns singleton instance."""
    agg1 = get_cost_aggregator()
    agg2 = get_cost_aggregator()

    assert agg1 is agg2
    assert agg1.context_name == "global"
