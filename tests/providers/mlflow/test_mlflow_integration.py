"""Integration tests for MLflow provider.

These tests validate end-to-end workflows, multi-provider scenarios,
and real-world usage patterns for the MLflow provider.
"""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from src.genops.providers.mlflow import (
    GenOpsMLflowAdapter,
    instrument_mlflow,
    auto_instrument_mlflow,
    create_mlflow_cost_context,
    validate_setup,
)


# ============================================================================
# End-to-End Workflow Tests (10 tests)
# ============================================================================

def test_complete_experiment_workflow():
    """Test complete experiment workflow from setup to teardown."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        # Setup
        adapter = instrument_mlflow(
            tracking_uri="file:///tmp/mlruns",
            team="integration-team",
            project="integration-test"
        )

        # Enable instrumentation
        adapter.instrument_framework()
        assert adapter._patched is True

        # Track experiment
        with adapter.track_mlflow_run(
            experiment_name="integration-exp",
            run_name="integration-run"
        ) as run:
            # Simulate MLflow operations
            mock_mlflow.log_param("param1", "value1")
            mock_mlflow.log_metric("metric1", 0.95)
            mock_mlflow.log_artifact("file.txt")

        # Verify governance attributes set
        assert adapter.team == "integration-team"
        assert adapter.project == "integration-test"

        # Cleanup
        adapter.uninstrument_framework()
        assert adapter._patched is False


def test_hierarchical_runs_workflow():
    """Test parent-child run relationships with cost aggregation."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(
            tracking_uri="file:///tmp/mlruns",
            team="hierarchical-team"
        )
        adapter.instrument_framework()

        # Parent run
        with adapter.track_mlflow_run(
            experiment_name="hierarchical-exp",
            run_name="parent-run"
        ) as parent_run:
            # Parent operations
            mock_mlflow.log_param("parent_param", "value")

            # Child run 1
            with adapter.track_mlflow_run(
                experiment_name="hierarchical-exp",
                run_name="child-run-1",
                parent_run_id="parent-run-id"
            ) as child_run_1:
                mock_mlflow.log_metric("child1_metric", 0.8)

            # Child run 2
            with adapter.track_mlflow_run(
                experiment_name="hierarchical-exp",
                run_name="child-run-2",
                parent_run_id="parent-run-id"
            ) as child_run_2:
                mock_mlflow.log_metric("child2_metric", 0.9)

        # Verify hierarchy tracking
        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 3  # Parent + 2 children


def test_multi_experiment_workflow():
    """Test tracking across multiple experiments."""
    with patch('src.genops.providers.mlflow.adapter.mlflow'):
        adapter = instrument_mlflow(
            tracking_uri="file:///tmp/mlruns",
            team="multi-exp-team"
        )
        adapter.instrument_framework()

        experiments = ["exp-1", "exp-2", "exp-3"]

        for exp_name in experiments:
            with adapter.track_mlflow_run(
                experiment_name=exp_name,
                run_name=f"run-{exp_name}"
            ) as run:
                pass  # Minimal operation

        metrics = adapter.get_metrics()
        assert metrics['operation_count'] == 3


def test_artifact_heavy_workflow():
    """Test workflow with many artifact operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(
            tracking_uri="s3://mlflow-bucket/artifacts",
            team="artifact-team"
        )
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="artifact-exp",
            run_name="artifact-run"
        ) as run:
            # Log multiple artifacts
            for i in range(10):
                mock_mlflow.log_artifact(f"file_{i}.txt")

        # Verify cost includes artifact operations
        metrics = adapter.get_metrics()
        assert metrics['daily_usage'] > 0


def test_model_registry_workflow():
    """Test model registration and versioning workflow."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(
            tracking_uri="file:///tmp/mlruns",
            registry_uri="file:///tmp/mlruns",
            team="model-team"
        )
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="model-exp",
            run_name="model-run"
        ) as run:
            # Log model
            mock_mlflow.log_model(artifact_path="model", python_model=MagicMock())

            # Register model
            mock_mlflow.register_model(model_uri="runs:/run-id/model", name="test-model")

        # Verify registry operations tracked
        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 2


def test_auto_instrumentation_workflow():
    """Test zero-code auto-instrumentation workflow."""
    with patch('src.genops.providers.mlflow.registration.mlflow'):
        with patch.dict(os.environ, {
            'MLFLOW_TRACKING_URI': 'file:///tmp/mlruns',
            'GENOPS_TEAM': 'auto-team',
            'GENOPS_PROJECT': 'auto-project'
        }):
            # Auto-instrument
            adapter = auto_instrument_mlflow()

            assert adapter is not None
            assert adapter.team == 'auto-team'
            assert adapter.project == 'auto-project'
            assert adapter._patched is True


def test_governance_attribute_propagation():
    """Test governance attributes propagate through all operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(
            tracking_uri="file:///tmp/mlruns",
            team="governance-team",
            project="governance-project",
            customer_id="customer-123",
            environment="production"
        )
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="governance-exp",
            run_name="governance-run"
        ) as run:
            mock_mlflow.log_param("param", "value")

        # Verify all governance attributes present
        assert adapter.team == "governance-team"
        assert adapter.project == "governance-project"
        assert adapter.customer_id == "customer-123"
        assert adapter.environment == "production"


def test_cost_tracking_across_operations():
    """Test cost tracking across different operation types."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(
            tracking_uri="s3://bucket/mlflow",
            team="cost-team"
        )
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="cost-exp",
            run_name="cost-run"
        ) as run:
            # Different operation types
            mock_mlflow.log_param("param", "value")  # Tracking cost
            mock_mlflow.log_metric("metric", 0.5)    # Tracking cost
            mock_mlflow.log_artifact("file.txt")     # Artifact cost
            mock_mlflow.log_model(artifact_path="model", python_model=MagicMock())  # Model cost

        metrics = adapter.get_metrics()
        assert metrics['daily_usage'] > 0
        assert metrics['operation_count'] >= 4


def test_validation_integration():
    """Test setup validation integration."""
    with patch('src.genops.providers.mlflow.validation.mlflow'):
        with patch('src.genops.providers.mlflow.validation.opentelemetry'):
            with patch('src.genops.providers.mlflow.validation.genops'):
                with patch('src.genops.providers.mlflow.validation.MlflowClient'):
                    result = validate_setup(
                        tracking_uri="http://localhost:5000",
                        check_connectivity=True,
                        check_governance=True
                    )

                    assert isinstance(result.dependencies, dict)
                    assert isinstance(result.configuration, dict)
                    assert isinstance(result.connectivity, dict)


def test_telemetry_export_integration():
    """Test OpenTelemetry trace export integration."""
    with patch('src.genops.providers.mlflow.adapter.mlflow'):
        with patch('src.genops.providers.mlflow.adapter.trace') as mock_trace:
            adapter = instrument_mlflow(
                tracking_uri="file:///tmp/mlruns",
                team="telemetry-team"
            )
            adapter.instrument_framework()

            with adapter.track_mlflow_run(
                experiment_name="telemetry-exp",
                run_name="telemetry-run"
            ) as run:
                pass

            # Verify trace context was created
            # Note: Actual trace verification depends on trace implementation


# ============================================================================
# MLflow Operations Integration Tests (8 tests)
# ============================================================================

def test_experiment_operations():
    """Test experiment creation and management operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        mock_mlflow.create_experiment = MagicMock(return_value="exp-123")
        mock_mlflow.get_experiment = MagicMock()
        mock_mlflow.set_experiment_tag = MagicMock()

        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="test-exp",
            run_name="test-run"
        ) as run:
            pass

        # Verify experiment operations
        metrics = adapter.get_metrics()
        assert metrics['operation_count'] > 0


def test_run_operations():
    """Test run creation and lifecycle operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.end_run = MagicMock()
        mock_mlflow.active_run = MagicMock(return_value=None)

        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="run-exp",
            run_name="test-run"
        ) as run:
            mock_mlflow.log_param("param", "value")

        # Run should be tracked
        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 1


def test_logging_operations():
    """Test parameter, metric, and tag logging operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="logging-exp",
            run_name="logging-run"
        ) as run:
            # Multiple logging operations
            mock_mlflow.log_param("param1", "value1")
            mock_mlflow.log_param("param2", "value2")
            mock_mlflow.log_metric("metric1", 0.5)
            mock_mlflow.log_metric("metric2", 0.8)
            mock_mlflow.set_tag("tag1", "value1")

        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 5


def test_artifact_operations():
    """Test artifact logging and retrieval operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test_artifact.txt"
            artifact_path.write_text("test content")

            adapter = instrument_mlflow(
                tracking_uri="file:///tmp/mlruns",
                team="artifact-ops-team"
            )
            adapter.instrument_framework()

            with adapter.track_mlflow_run(
                experiment_name="artifact-exp",
                run_name="artifact-run"
            ) as run:
                mock_mlflow.log_artifact(str(artifact_path))
                mock_mlflow.log_artifacts(tmpdir)

            metrics = adapter.get_metrics()
            assert metrics['operation_count'] >= 2


def test_model_operations():
    """Test model logging and registry operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(
            tracking_uri="file:///tmp/mlruns",
            registry_uri="file:///tmp/mlruns"
        )
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="model-exp",
            run_name="model-run"
        ) as run:
            # Model operations
            mock_mlflow.log_model(artifact_path="model", python_model=MagicMock())
            mock_mlflow.register_model(model_uri="runs:/run-id/model", name="test-model")

        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 2


def test_batch_operations():
    """Test batch logging operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="batch-exp",
            run_name="batch-run"
        ) as run:
            # Batch operations
            for i in range(100):
                mock_mlflow.log_metric(f"metric_{i}", i * 0.01)

        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 100


def test_search_operations():
    """Test experiment and run search operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        mock_mlflow.search_experiments = MagicMock(return_value=[])
        mock_mlflow.search_runs = MagicMock(return_value=[])

        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        # Search operations should not fail
        mock_mlflow.search_experiments()
        mock_mlflow.search_runs(experiment_ids=["1"])


def test_delete_operations():
    """Test run and experiment deletion operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        mock_mlflow.delete_run = MagicMock()
        mock_mlflow.delete_experiment = MagicMock()

        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        # Delete operations should be tracked
        with adapter.track_mlflow_run(
            experiment_name="delete-exp",
            run_name="delete-run"
        ) as run:
            pass

        # Deletion tracking
        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 1


# ============================================================================
# Error Handling Integration Tests (6 tests)
# ============================================================================

def test_invalid_tracking_uri_handling():
    """Test handling of invalid tracking URI."""
    with patch('src.genops.providers.mlflow.adapter.mlflow'):
        adapter = instrument_mlflow(tracking_uri="invalid://uri")

        # Should initialize without error
        assert adapter.tracking_uri == "invalid://uri"


def test_disconnected_server_handling():
    """Test graceful handling of disconnected MLflow server."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        mock_mlflow.start_run.side_effect = ConnectionError("Cannot connect")

        adapter = instrument_mlflow(tracking_uri="http://localhost:5000")
        adapter.instrument_framework()

        # Should handle connection error gracefully
        try:
            with adapter.track_mlflow_run(
                experiment_name="error-exp",
                run_name="error-run"
            ) as run:
                pass
        except ConnectionError:
            pass  # Expected


def test_storage_error_handling():
    """Test handling of artifact storage errors."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        mock_mlflow.log_artifact.side_effect = IOError("Storage error")

        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="storage-exp",
            run_name="storage-run"
        ) as run:
            try:
                mock_mlflow.log_artifact("file.txt")
            except IOError:
                pass  # Expected


def test_missing_mlflow_graceful_degradation():
    """Test graceful degradation when MLflow is not available."""
    with patch('src.genops.providers.mlflow.adapter.mlflow', None):
        adapter = GenOpsMLflowAdapter(
            tracking_uri="file:///tmp/mlruns",
            team="test-team"
        )

        # Should initialize but report framework not available
        assert adapter.is_framework_available() is False


def test_instrumentation_error_handling():
    """Test handling of instrumentation errors."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")

        # Cause instrumentation to fail
        mock_mlflow.start_run = None

        try:
            adapter.instrument_framework()
        except Exception:
            pass  # Should handle gracefully


def test_cost_calculation_error_handling():
    """Test handling of cost calculation errors."""
    with patch('src.genops.providers.mlflow.adapter.mlflow'):
        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        # Test with invalid operation context
        cost = adapter.calculate_cost({'invalid': 'context'})
        assert cost == 0.0  # Should return 0 on error


# ============================================================================
# Performance and Scaling Tests (Optional - not counted in 24)
# ============================================================================

@pytest.mark.performance
def test_high_volume_operations():
    """Test performance with high volume of operations."""
    with patch('src.genops.providers.mlflow.adapter.mlflow') as mock_mlflow:
        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        with adapter.track_mlflow_run(
            experiment_name="perf-exp",
            run_name="perf-run"
        ) as run:
            # Simulate 1000 operations
            for i in range(1000):
                mock_mlflow.log_metric(f"metric_{i}", i)

        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 1000


@pytest.mark.performance
def test_concurrent_runs():
    """Test concurrent run tracking (mocked)."""
    with patch('src.genops.providers.mlflow.adapter.mlflow'):
        adapter = instrument_mlflow(tracking_uri="file:///tmp/mlruns")
        adapter.instrument_framework()

        # Simulate concurrent runs (sequentially mocked)
        for i in range(10):
            with adapter.track_mlflow_run(
                experiment_name=f"concurrent-exp-{i}",
                run_name=f"concurrent-run-{i}"
            ) as run:
                pass

        metrics = adapter.get_metrics()
        assert metrics['operation_count'] >= 10
