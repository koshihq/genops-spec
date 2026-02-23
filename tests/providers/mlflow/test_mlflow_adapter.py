"""Tests for MLflow adapter."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import adapter
from src.genops.providers.mlflow.adapter import (
    GenOpsMLflowAdapter,
    instrument_mlflow,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mlflow():
    """Mock MLflow module."""
    with patch.dict("sys.modules", {"mlflow": MagicMock()}):
        yield


@pytest.fixture
def mock_mlflow_client():
    """Mock MlflowClient."""
    with patch("src.genops.providers.mlflow.adapter.MlflowClient") as mock:
        yield mock


@pytest.fixture
def adapter(mock_mlflow_client):
    """Create test adapter instance."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        adapter = GenOpsMLflowAdapter(
            tracking_uri="http://localhost:5000",
            team="test-team",
            project="test-project",
        )
        return adapter


@pytest.fixture
def mock_tracer():
    """Mock OpenTelemetry tracer."""
    mock = MagicMock()
    mock.start_as_current_span = MagicMock(
        return_value=MagicMock(__enter__=Mock(), __exit__=Mock())
    )
    return mock


# ============================================================================
# Initialization Tests (5 tests)
# ============================================================================


def test_adapter_initialization_basic(mock_mlflow_client):
    """Test basic adapter initialization."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        adapter = GenOpsMLflowAdapter(
            tracking_uri="http://localhost:5000",
            team="test-team",
            project="test-project",
        )

        assert adapter.tracking_uri == "http://localhost:5000"
        assert adapter.team == "test-team"
        assert adapter.project == "test-project"
        assert adapter._patched is False
        assert len(adapter._original_methods) == 0


def test_adapter_initialization_with_registry_uri(mock_mlflow_client):
    """Test adapter initialization with registry URI."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        adapter = GenOpsMLflowAdapter(
            tracking_uri="http://localhost:5000",
            registry_uri="http://localhost:5001",
            team="test-team",
            project="test-project",
        )

        assert adapter.tracking_uri == "http://localhost:5000"
        assert adapter.registry_uri == "http://localhost:5001"


def test_adapter_initialization_with_governance_attrs(mock_mlflow_client):
    """Test adapter initialization with governance attributes."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        adapter = GenOpsMLflowAdapter(
            tracking_uri="http://localhost:5000",
            team="test-team",
            project="test-project",
            customer_id="test-customer",
            environment="production",
            cost_center="ml-research",
        )

        assert adapter.team == "test-team"
        assert adapter.project == "test-project"
        assert adapter.customer_id == "test-customer"
        assert adapter.environment == "production"
        assert adapter.cost_center == "ml-research"


def test_adapter_initialization_from_env_vars(mock_mlflow_client):
    """Test adapter initialization from environment variables."""
    with patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "http://env-server:5000",
            "GENOPS_TEAM": "env-team",
            "GENOPS_PROJECT": "env-project",
        },
    ):
        with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
            adapter = GenOpsMLflowAdapter()

            assert adapter.tracking_uri == "http://env-server:5000"
            assert adapter.team == "env-team"
            assert adapter.project == "env-project"


def test_adapter_initialization_without_mlflow():
    """Test adapter initialization fails without MLflow."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", False):
        with pytest.raises(ImportError, match="MLflow package not found"):
            GenOpsMLflowAdapter(
                tracking_uri="http://localhost:5000",
                team="test-team",
                project="test-project",
            )


# ============================================================================
# Abstract Method Tests (10 tests)
# ============================================================================


def test_setup_governance_attributes(adapter):
    """Test setup_governance_attributes adds MLflow-specific attributes."""
    adapter.setup_governance_attributes()

    assert "ml_framework" in adapter.GOVERNANCE_ATTRIBUTES
    assert "algorithm_type" in adapter.GOVERNANCE_ATTRIBUTES
    assert "training_dataset" in adapter.GOVERNANCE_ATTRIBUTES
    assert "model_owner" in adapter.GOVERNANCE_ATTRIBUTES


def test_get_framework_name(adapter):
    """Test get_framework_name returns 'mlflow'."""
    assert adapter.get_framework_name() == "mlflow"


def test_get_framework_type(adapter):
    """Test get_framework_type returns DATA_PLATFORM."""
    assert adapter.get_framework_type() == adapter.FRAMEWORK_TYPE_DATA_PLATFORM


def test_get_framework_version(adapter):
    """Test get_framework_version returns version string."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.__version__ = "2.9.0"
        version = adapter.get_framework_version()
        assert version == "2.9.0"


def test_get_framework_version_unavailable(adapter):
    """Test get_framework_version returns None when unavailable."""
    with patch("src.genops.providers.mlflow.adapter.mlflow", side_effect=ImportError):
        version = adapter.get_framework_version()
        assert version is None


def test_is_framework_available_true():
    """Test is_framework_available returns True when MLflow available."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        with patch("src.genops.providers.mlflow.adapter.MlflowClient"):
            adapter = GenOpsMLflowAdapter(
                tracking_uri="http://localhost:5000",
                team="test-team",
                project="test-project",
            )
            assert adapter.is_framework_available() is True


def test_calculate_cost_log_metric(adapter):
    """Test calculate_cost for log_metric operation."""
    cost = adapter.calculate_cost({"operation_type": "log_metric"})
    assert cost > 0
    assert cost == 0.0001  # Tracking API call cost


def test_calculate_cost_log_artifact(adapter):
    """Test calculate_cost for log_artifact operation."""
    cost = adapter.calculate_cost(
        {
            "operation_type": "log_artifact",
            "artifact_size_mb": 10.0,
            "storage_backend": "s3",
        }
    )
    assert cost > 0


def test_get_operation_mappings(adapter):
    """Test get_operation_mappings returns correct mappings."""
    mappings = adapter.get_operation_mappings()

    assert "mlflow.start_run" in mappings
    assert "mlflow.log_metric" in mappings
    assert "mlflow.log_param" in mappings
    assert "mlflow.log_artifact" in mappings
    assert "mlflow.log_model" in mappings
    assert "mlflow.register_model" in mappings
    assert len(mappings) >= 8


def test_record_framework_metrics(adapter):
    """Test _record_framework_metrics sets correct attributes."""
    mock_span = MagicMock()
    context = {"experiment_id": "exp-123", "run_id": "run-456", "run_name": "test-run"}

    adapter._record_framework_metrics(mock_span, "log_metric", context)

    mock_span.set_attribute.assert_any_call("mlflow.experiment_id", "exp-123")
    mock_span.set_attribute.assert_any_call("mlflow.run_id", "run-456")
    mock_span.set_attribute.assert_any_call("mlflow.run_name", "test-run")


# ============================================================================
# Context Manager Tests (8 tests)
# ============================================================================


def test_track_mlflow_run_basic(adapter):
    """Test track_mlflow_run context manager basic usage."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_run.info.experiment_id = "exp-456"
        mock_mlflow.start_run.return_value = mock_run

        with patch(
            "src.genops.providers.mlflow.adapter.create_mlflow_cost_context"
        ) as mock_cost_ctx:
            mock_cost_context = MagicMock()
            mock_cost_summary = MagicMock()
            mock_cost_summary.total_cost = 0.001
            mock_cost_summary.operation_count = 5
            mock_cost_context.get_current_summary.return_value = mock_cost_summary
            mock_cost_ctx.return_value.__enter__.return_value = mock_cost_context

            with adapter.track_mlflow_run(
                experiment_name="test-exp", run_name="test-run"
            ) as run:
                assert run == mock_run

            mock_mlflow.set_experiment.assert_called_once_with("test-exp")
            mock_mlflow.start_run.assert_called_once()
            mock_mlflow.end_run.assert_called_once()


def test_track_mlflow_run_with_governance_attrs(adapter):
    """Test track_mlflow_run propagates governance attributes."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.genops.providers.mlflow.adapter.create_mlflow_cost_context"):
            with adapter.track_mlflow_run(
                experiment_name="test-exp", run_name="test-run", customer_id="cust-001"
            ):
                pass

            # Check MLflow tags were set
            mock_mlflow.set_tag.assert_called()


def test_track_mlflow_run_error_handling(adapter):
    """Test track_mlflow_run handles errors correctly."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.start_run.side_effect = Exception("Connection error")

        with patch("src.genops.providers.mlflow.adapter.create_mlflow_cost_context"):
            with pytest.raises(Exception, match="Connection error"):
                with adapter.track_mlflow_run(experiment_name="test-exp"):
                    pass

            # Ensure cleanup happened
            mock_mlflow.end_run.assert_called_once()


def test_track_mlflow_run_cost_tracking(adapter):
    """Test track_mlflow_run tracks costs correctly."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch(
            "src.genops.providers.mlflow.adapter.create_mlflow_cost_context"
        ) as mock_cost_ctx:
            mock_cost_context = MagicMock()
            mock_summary = MagicMock()
            mock_summary.total_cost = 0.005
            mock_cost_context.get_current_summary.return_value = mock_summary
            mock_cost_ctx.return_value.__enter__.return_value = mock_cost_context

            initial_usage = adapter.daily_usage

            with adapter.track_mlflow_run(experiment_name="test-exp"):
                pass

            assert adapter.daily_usage == initial_usage + 0.005
            assert adapter.operation_count >= 1


def test_track_mlflow_run_active_runs_cleanup(adapter):
    """Test track_mlflow_run cleans up active runs."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.genops.providers.mlflow.adapter.create_mlflow_cost_context"):
            assert len(adapter.active_runs) == 0

            with adapter.track_mlflow_run(experiment_name="test-exp"):
                assert "run-123" in adapter.active_runs

            assert "run-123" not in adapter.active_runs


def test_track_mlflow_run_without_experiment_name(adapter):
    """Test track_mlflow_run works without experiment name."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.genops.providers.mlflow.adapter.create_mlflow_cost_context"):
            with adapter.track_mlflow_run(run_name="test-run"):
                pass

            # Should not call set_experiment
            mock_mlflow.set_experiment.assert_not_called()


def test_track_mlflow_run_nested_contexts(adapter):
    """Test nested track_mlflow_run contexts."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run1 = MagicMock()
        mock_run1.info.run_id = "run-1"
        mock_run2 = MagicMock()
        mock_run2.info.run_id = "run-2"

        mock_mlflow.start_run.side_effect = [mock_run1, mock_run2]

        with patch("src.genops.providers.mlflow.adapter.create_mlflow_cost_context"):
            with adapter.track_mlflow_run(run_name="outer"):
                assert "run-1" in adapter.active_runs

                with adapter.track_mlflow_run(run_name="inner"):
                    assert "run-1" in adapter.active_runs
                    assert "run-2" in adapter.active_runs

                assert "run-2" not in adapter.active_runs

            assert "run-1" not in adapter.active_runs


def test_track_mlflow_run_telemetry_span(adapter):
    """Test track_mlflow_run creates telemetry span."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch.object(adapter, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
                mock_span
            )

            with patch(
                "src.genops.providers.mlflow.adapter.create_mlflow_cost_context"
            ):
                with adapter.track_mlflow_run(
                    experiment_name="test-exp", run_name="test-run"
                ):
                    pass

                mock_tracer.start_as_current_span.assert_called_once_with(
                    "genops.mlflow.run.test-run"
                )
                mock_span.set_attribute.assert_called()


# ============================================================================
# Patching Tests (12 tests)
# ============================================================================


def test_apply_instrumentation(adapter):
    """Test _apply_instrumentation patches MLflow methods."""
    with patch("src.genops.providers.mlflow.adapter.mlflow"):
        adapter._apply_instrumentation()

        assert adapter._patched is True
        assert len(adapter._original_methods) > 0


def test_apply_instrumentation_idempotent(adapter):
    """Test _apply_instrumentation is idempotent."""
    with patch("src.genops.providers.mlflow.adapter.mlflow"):
        adapter._apply_instrumentation()
        first_methods = dict(adapter._original_methods)

        adapter._apply_instrumentation()  # Second call

        assert adapter._original_methods == first_methods


def test_remove_instrumentation(adapter):
    """Test _remove_instrumentation restores originals."""
    with patch("src.genops.providers.mlflow.adapter.mlflow"):
        adapter._apply_instrumentation()
        assert adapter._patched is True

        adapter._remove_instrumentation()

        assert adapter._patched is False
        assert len(adapter._original_methods) == 0


def test_patch_start_run(adapter):
    """Test _patch_start_run wraps mlflow.start_run."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.start_run = MagicMock()
        original_start_run = mock_mlflow.start_run

        adapter._patch_start_run(mock_mlflow)

        assert mock_mlflow.start_run != original_start_run
        assert "start_run" in adapter._original_methods


def test_patch_log_metric(adapter):
    """Test _patch_log_metric wraps mlflow.log_metric."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.log_metric = MagicMock()
        original_log_metric = mock_mlflow.log_metric

        adapter._patch_log_metric(mock_mlflow)

        assert mock_mlflow.log_metric != original_log_metric
        assert "log_metric" in adapter._original_methods


def test_patch_log_param(adapter):
    """Test _patch_log_param wraps mlflow.log_param."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.log_param = MagicMock()

        adapter._patch_log_param(mock_mlflow)

        assert "log_param" in adapter._original_methods


def test_patch_set_tag(adapter):
    """Test _patch_set_tag wraps mlflow.set_tag."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.set_tag = MagicMock()

        adapter._patch_set_tag(mock_mlflow)

        assert "set_tag" in adapter._original_methods


def test_patch_log_artifact(adapter):
    """Test _patch_log_artifact wraps mlflow.log_artifact."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.log_artifact = MagicMock()

        adapter._patch_log_artifact(mock_mlflow)

        assert "log_artifact" in adapter._original_methods


def test_patch_log_model(adapter):
    """Test _patch_log_model wraps mlflow.log_model."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.log_model = MagicMock()

        adapter._patch_log_model(mock_mlflow)

        assert "log_model" in adapter._original_methods


def test_patch_register_model(adapter):
    """Test _patch_register_model wraps mlflow.register_model."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        mock_mlflow.register_model = MagicMock()

        adapter._patch_register_model(mock_mlflow)

        assert "register_model" in adapter._original_methods


def test_patched_method_calls_original(adapter):
    """Test patched methods call original and return result."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        # Setup original method
        original_log_metric = MagicMock(return_value="success")
        mock_mlflow.log_metric = original_log_metric

        # Patch it
        adapter._patch_log_metric(mock_mlflow)

        # Call patched version
        with patch.object(adapter, "tracer"):
            mock_mlflow.log_metric("accuracy", 0.95)

            # Should call original
            original_log_metric.assert_called_once_with("accuracy", 0.95)


def test_patch_cycle_preserves_functionality(adapter):
    """Test patch/unpatch cycle preserves functionality."""
    with patch("src.genops.providers.mlflow.adapter.mlflow") as mock_mlflow:
        original_method = MagicMock()
        mock_mlflow.log_metric = original_method

        # Patch
        adapter._patch_log_metric(mock_mlflow)

        # Unpatch
        adapter._remove_instrumentation()

        # Should restore original
        assert mock_mlflow.log_metric == original_method


# ============================================================================
# Factory Function Tests (2 tests)
# ============================================================================


def test_instrument_mlflow_factory():
    """Test instrument_mlflow factory function."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        with patch("src.genops.providers.mlflow.adapter.MlflowClient"):
            adapter = instrument_mlflow(
                tracking_uri="http://localhost:5000",
                team="factory-team",
                project="factory-project",
            )

            assert isinstance(adapter, GenOpsMLflowAdapter)
            assert adapter.tracking_uri == "http://localhost:5000"
            assert adapter.team == "factory-team"
            assert adapter.project == "factory-project"


def test_instrument_mlflow_with_all_params():
    """Test instrument_mlflow with all parameters."""
    with patch("src.genops.providers.mlflow.adapter.MLFLOW_AVAILABLE", True):
        with patch("src.genops.providers.mlflow.adapter.MlflowClient"):
            adapter = instrument_mlflow(
                tracking_uri="http://localhost:5000",
                registry_uri="http://localhost:5001",
                team="test-team",
                project="test-project",
                customer_id="cust-001",
                environment="production",
            )

            assert adapter.tracking_uri == "http://localhost:5000"
            assert adapter.registry_uri == "http://localhost:5001"
            assert adapter.team == "test-team"
            assert adapter.project == "test-project"
            assert adapter.customer_id == "cust-001"
            assert adapter.environment == "production"
