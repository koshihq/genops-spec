"""MLflow adapter for GenOps AI governance.

Provides comprehensive governance telemetry, cost tracking, and policy enforcement
for MLflow experiment tracking and model registry operations.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from genops.providers.base.provider import BaseFrameworkProvider

logger = logging.getLogger(__name__)

# Check MLflow availability
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class GenOpsMLflowAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for MLflow experiment tracking and model registry.

    Provides comprehensive governance telemetry, cost tracking, and policy enforcement
    for MLflow operations across experiment tracking, artifact logging, and model management.

    Example:
        ```python
        from genops.providers.mlflow import instrument_mlflow

        # Create adapter
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
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MLflow adapter.

        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI (optional)
            **kwargs: Additional configuration including governance attributes
                - team: Team identifier
                - project: Project identifier
                - customer_id: Customer identifier
                - environment: Environment (dev/staging/prod)
                - cost_center: Cost center for attribution
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow package not found. Install with: pip install mlflow"
            )

        super().__init__(**kwargs)

        # MLflow configuration
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI') or 'file:///mlruns'
        self.registry_uri = registry_uri or os.getenv('MLFLOW_REGISTRY_URI')

        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Governance attributes from kwargs or env
        self.team = kwargs.get('team') or os.getenv('GENOPS_TEAM')
        self.project = kwargs.get('project') or os.getenv('GENOPS_PROJECT')
        self.customer_id = kwargs.get('customer_id')
        self.environment = kwargs.get('environment', 'development')
        self.cost_center = kwargs.get('cost_center')

        # MLflow-specific attributes
        self.MLFLOW_ATTRIBUTES = {
            'experiment_id', 'experiment_name', 'run_id', 'run_name',
            'model_name', 'model_version', 'model_stage',
            'artifact_uri', 'parent_run_id', 'lifecycle_stage',
            'registered_model_name'
        }
        self.REQUEST_ATTRIBUTES.update(self.MLFLOW_ATTRIBUTES)

        # Patching state
        self._patched = False
        self._original_methods: Dict[str, Any] = {}

        # Runtime tracking
        self.active_runs: Dict[str, Any] = {}
        self.daily_usage = 0.0
        self.operation_count = 0

        # Telemetry
        self.tracer = trace.get_tracer(__name__)

        logger.debug(
            f"Initialized MLflow adapter: tracking_uri={self.tracking_uri}, "
            f"team={self.team}, project={self.project}"
        )

    # ============================================================================
    # Abstract Method Implementations (BaseFrameworkProvider)
    # ============================================================================

    def setup_governance_attributes(self) -> None:
        """Setup MLflow-specific governance attributes."""
        additional_attrs = {
            'ml_framework',        # sklearn, pytorch, tensorflow, etc.
            'algorithm_type',      # Model algorithm classification
            'training_dataset',    # Dataset used for training
            'model_owner',         # Model ownership
            'compliance_status',   # Compliance status
            'data_lineage_id',     # Data lineage tracking
        }
        self.GOVERNANCE_ATTRIBUTES.update(additional_attrs)
        logger.debug(f"MLflow governance attributes configured: {additional_attrs}")

    def get_framework_name(self) -> str:
        """Return the framework name."""
        return "mlflow"

    def get_framework_type(self) -> str:
        """Return the framework type."""
        return self.FRAMEWORK_TYPE_DATA_PLATFORM

    def get_framework_version(self) -> str | None:
        """Return the installed MLflow version."""
        try:
            import mlflow
            return mlflow.__version__
        except (ImportError, AttributeError):
            return None

    def is_framework_available(self) -> bool:
        """Check if MLflow is available."""
        return MLFLOW_AVAILABLE

    def calculate_cost(self, operation_context: dict) -> float:
        """
        Calculate cost for MLflow operations.

        Cost model:
        - Tracking API calls: $0.0001 per call
        - Artifact storage: Based on size and storage backend
        - Model registry operations: $0.0005 per operation
        - Remote tracking server: Based on compute time

        Args:
            operation_context: Contains operation_type, artifact_size_mb,
                              duration_ms, storage_backend, etc.

        Returns:
            Estimated cost in USD
        """
        operation_type = operation_context.get('operation_type', 'unknown')
        cost = 0.0

        # Import cost aggregator for calculations
        try:
            from .cost_aggregator import get_cost_calculator
            calculator = get_cost_calculator()

            if operation_type == 'log_artifact':
                artifact_size_mb = operation_context.get('artifact_size_mb', 0)
                storage_backend = operation_context.get('storage_backend', 'local')
                cost = calculator.calculate_artifact_cost(artifact_size_mb, storage_backend)

            elif operation_type == 'log_model':
                model_size_mb = operation_context.get('model_size_mb', 0)
                storage_backend = operation_context.get('storage_backend', 'local')
                cost = calculator.calculate_model_cost(model_size_mb, storage_backend)

            elif operation_type == 'register_model':
                model_size_mb = operation_context.get('model_size_mb', 0)
                cost = calculator.calculate_registry_cost(model_size_mb)

            elif operation_type in ['log_metric', 'log_param', 'set_tag']:
                # Tracking API calls
                cost = calculator.calculate_tracking_cost()

            elif operation_type == 'create_run':
                # Run creation cost (minimal)
                cost = calculator.calculate_run_cost()

            else:
                # Default minimal cost for other operations
                cost = 0.0001

        except (ImportError, Exception) as e:
            logger.debug(f"Cost calculator not available, using defaults: {e}")
            # Fallback to simple estimates
            if operation_type == 'log_artifact':
                artifact_size_mb = operation_context.get('artifact_size_mb', 0)
                cost = (artifact_size_mb / 1024) * 0.023 * (1/30)  # S3 pricing
            elif operation_type == 'register_model':
                cost = 0.0005
            else:
                cost = 0.0001

        return cost

    def get_operation_mappings(self) -> dict[str, str]:
        """
        Return mapping of MLflow operations to instrumentation methods.

        Returns:
            Dictionary mapping operation names to method names
        """
        return {
            'mlflow.start_run': 'instrument_start_run',
            'mlflow.log_metric': 'instrument_log_metric',
            'mlflow.log_param': 'instrument_log_param',
            'mlflow.set_tag': 'instrument_set_tag',
            'mlflow.log_artifact': 'instrument_log_artifact',
            'mlflow.log_artifacts': 'instrument_log_artifacts',
            'mlflow.log_model': 'instrument_log_model',
            'mlflow.register_model': 'instrument_register_model',
            'mlflow.sklearn.autolog': 'instrument_sklearn_autolog',
            'mlflow.pytorch.autolog': 'instrument_pytorch_autolog',
            'mlflow.tensorflow.autolog': 'instrument_tensorflow_autolog',
        }

    def _record_framework_metrics(
        self,
        span: Any,
        operation_type: str,
        context: dict
    ) -> None:
        """Record MLflow-specific metrics on span."""

        # Common MLflow attributes
        if 'experiment_id' in context:
            span.set_attribute('mlflow.experiment_id', context['experiment_id'])
        if 'run_id' in context:
            span.set_attribute('mlflow.run_id', context['run_id'])
        if 'run_name' in context:
            span.set_attribute('mlflow.run_name', context['run_name'])

        # Operation-specific metrics
        if operation_type == 'log_artifact':
            if 'artifact_size_mb' in context:
                span.set_attribute('mlflow.artifact_size_mb', context['artifact_size_mb'])
            if 'artifact_path' in context:
                span.set_attribute('mlflow.artifact_path', context['artifact_path'])

        elif operation_type == 'log_model':
            if 'model_size_mb' in context:
                span.set_attribute('mlflow.model_size_mb', context['model_size_mb'])
            if 'model_flavor' in context:
                span.set_attribute('mlflow.model_flavor', context['model_flavor'])

        elif operation_type == 'register_model':
            if 'model_name' in context:
                span.set_attribute('mlflow.model_name', context['model_name'])
            if 'model_version' in context:
                span.set_attribute('mlflow.model_version', context['model_version'])

        # Performance metrics
        if 'duration_ms' in context:
            span.set_attribute('mlflow.duration_ms', context['duration_ms'])

    def _apply_instrumentation(self, **config) -> None:
        """Apply MLflow instrumentation patches."""
        if self._patched:
            logger.warning("MLflow already instrumented")
            return

        try:
            import mlflow

            # Patch core tracking methods
            self._patch_start_run(mlflow)
            self._patch_log_metric(mlflow)
            self._patch_log_param(mlflow)
            self._patch_set_tag(mlflow)
            self._patch_log_artifact(mlflow)
            self._patch_log_model(mlflow)

            # Patch model registry methods
            self._patch_register_model(mlflow)

            # Patch auto-logging if enabled
            if config.get('instrument_autolog', True):
                self._patch_autolog_methods(mlflow)

            self._patched = True
            logger.info("MLflow instrumentation applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply MLflow instrumentation: {e}")
            raise

    def _remove_instrumentation(self) -> None:
        """Remove MLflow instrumentation patches."""
        if not self._patched:
            return

        try:
            import mlflow

            # Restore original methods
            for method_path, original_func in self._original_methods.items():
                parts = method_path.split('.')
                obj = mlflow
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], original_func)

            self._original_methods.clear()
            self._patched = False
            logger.info("MLflow instrumentation removed successfully")

        except Exception as e:
            logger.error(f"Failed to remove MLflow instrumentation: {e}")
            raise

    def instrument_framework(self, **config) -> None:
        """
        Enable MLflow instrumentation with governance tracking.

        This is the public method that should be called to enable
        instrumentation. It wraps the private _apply_instrumentation()
        method and provides a consistent public API.

        Args:
            **config: Configuration options for instrumentation

        Example:
            ```python
            adapter = instrument_mlflow(team="ml-team")
            adapter.instrument_framework()  # Enable instrumentation
            ```
        """
        if self._patched:
            logger.warning("MLflow already instrumented")
            return

        self._apply_instrumentation(**config)
        logger.info("MLflow instrumentation enabled")

    def uninstrument_framework(self) -> None:
        """
        Disable MLflow instrumentation and restore original methods.

        This is the public method for cleanup. It wraps the private
        _remove_instrumentation() method and provides a consistent
        public API for framework cleanup.

        Example:
            ```python
            adapter.uninstrument_framework()  # Restore MLflow
            ```
        """
        if not self._patched:
            logger.warning("MLflow not instrumented")
            return

        self._remove_instrumentation()
        logger.info("MLflow instrumentation disabled")

    # ============================================================================
    # Context Managers
    # ============================================================================

    @contextmanager
    def track_mlflow_run(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        **governance_attrs
    ):
        """
        Context manager for tracking MLflow runs with governance telemetry.

        Args:
            experiment_name: MLflow experiment name
            run_name: Name for the run
            **governance_attrs: Governance attributes (team, project, etc.)

        Yields:
            Run context with tracking information

        Example:
            ```python
            with adapter.track_mlflow_run(
                experiment_name="optimization",
                run_name="run-001",
                team="ml-team"
            ) as run:
                mlflow.log_param("lr", 0.01)
                mlflow.log_metric("accuracy", 0.95)
            ```
        """
        import mlflow
        from .cost_aggregator import create_mlflow_cost_context

        span_name = f"genops.mlflow.run.{run_name or 'unnamed'}"

        with self.tracer.start_as_current_span(span_name) as span:
            with create_mlflow_cost_context(run_name or 'unnamed') as cost_context:
                try:
                    # Set experiment if provided
                    if experiment_name:
                        mlflow.set_experiment(experiment_name)

                    # Start MLflow run
                    run = mlflow.start_run(run_name=run_name)

                    # Set span attributes
                    span.set_attribute("genops.provider", "mlflow")
                    span.set_attribute("genops.operation_type", "run")
                    span.set_attribute("mlflow.experiment_name", experiment_name or "default")
                    span.set_attribute("mlflow.run_id", run.info.run_id)
                    span.set_attribute("mlflow.run_name", run_name or "unnamed")

                    # Set governance attributes on span and as MLflow tags
                    merged_governance = {
                        'team': self.team,
                        'project': self.project,
                        'customer_id': self.customer_id,
                        'environment': self.environment,
                        **governance_attrs
                    }

                    for attr_name, attr_value in merged_governance.items():
                        if attr_value and attr_name in self.GOVERNANCE_ATTRIBUTES:
                            span.set_attribute(f"genops.{attr_name}", str(attr_value))
                            mlflow.set_tag(f"genops.{attr_name}", str(attr_value))

                    # Track active run
                    self.active_runs[run.info.run_id] = {
                        'run': run,
                        'cost_context': cost_context,
                        'start_time': datetime.now()
                    }

                    logger.debug(f"Started MLflow run tracking: {run.info.run_id}")

                    # Yield run context
                    yield run

                    # Success
                    span.set_status(Status(StatusCode.OK))

                    # Record final cost
                    final_summary = cost_context.get_current_summary()
                    span.set_attribute("genops.cost.total", final_summary.total_cost)
                    span.set_attribute("genops.cost.currency", "USD")

                    self.daily_usage += final_summary.total_cost
                    self.operation_count += 1

                    logger.debug(
                        f"Completed MLflow run: ${final_summary.total_cost:.6f} "
                        f"({final_summary.operation_count} operations)"
                    )

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(f"Failed MLflow run tracking: {e}")
                    raise
                finally:
                    # Ensure run is ended
                    mlflow.end_run()
                    # Remove from active runs
                    if 'run' in locals() and run.info.run_id in self.active_runs:
                        self.active_runs.pop(run.info.run_id)

    # ============================================================================
    # Patching Methods (Private)
    # ============================================================================

    def _patch_start_run(self, mlflow_module):
        """Patch mlflow.start_run to add governance tracking."""
        original_start_run = mlflow_module.start_run
        self._original_methods['start_run'] = original_start_run

        adapter = self

        def wrapped_start_run(*args, **kwargs):
            """Wrapped start_run with governance telemetry."""
            # Extract governance attrs
            governance_attrs, _, api_kwargs = adapter._extract_attributes(kwargs)

            with adapter.tracer.start_as_current_span("mlflow.start_run") as span:
                # Set attributes
                trace_attrs = adapter._build_trace_attributes(
                    "mlflow.start_run",
                    "ml.run.start",
                    governance_attrs,
                    tracking_uri=adapter.tracking_uri
                )

                for key, value in trace_attrs.items():
                    span.set_attribute(key, value)

                # Call original method
                result = original_start_run(*args, **api_kwargs)

                # Record run metadata
                span.set_attribute("mlflow.run_id", result.info.run_id)
                span.set_attribute("mlflow.experiment_id", result.info.experiment_id)

                # Calculate and record cost
                cost = adapter.calculate_cost({
                    'operation_type': 'create_run',
                    'tracking_uri': adapter.tracking_uri
                })
                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.start_run = wrapped_start_run

    def _patch_log_metric(self, mlflow_module):
        """Patch mlflow.log_metric to add cost tracking."""
        original_log_metric = mlflow_module.log_metric
        self._original_methods['log_metric'] = original_log_metric

        adapter = self

        def wrapped_log_metric(*args, **kwargs):
            """Wrapped log_metric with cost tracking."""
            with adapter.tracer.start_as_current_span("mlflow.log_metric") as span:
                span.set_attribute("genops.provider", "mlflow")
                span.set_attribute("genops.operation_type", "log_metric")

                result = original_log_metric(*args, **kwargs)

                # Estimate cost
                cost = adapter.calculate_cost({'operation_type': 'log_metric'})
                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.log_metric = wrapped_log_metric

    def _patch_log_param(self, mlflow_module):
        """Patch mlflow.log_param to add cost tracking."""
        original_log_param = mlflow_module.log_param
        self._original_methods['log_param'] = original_log_param

        adapter = self

        def wrapped_log_param(*args, **kwargs):
            """Wrapped log_param with cost tracking."""
            with adapter.tracer.start_as_current_span("mlflow.log_param") as span:
                span.set_attribute("genops.provider", "mlflow")
                span.set_attribute("genops.operation_type", "log_param")

                result = original_log_param(*args, **kwargs)

                cost = adapter.calculate_cost({'operation_type': 'log_param'})
                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.log_param = wrapped_log_param

    def _patch_set_tag(self, mlflow_module):
        """Patch mlflow.set_tag to add cost tracking."""
        original_set_tag = mlflow_module.set_tag
        self._original_methods['set_tag'] = original_set_tag

        adapter = self

        def wrapped_set_tag(*args, **kwargs):
            """Wrapped set_tag with cost tracking."""
            with adapter.tracer.start_as_current_span("mlflow.set_tag") as span:
                span.set_attribute("genops.provider", "mlflow")
                span.set_attribute("genops.operation_type", "set_tag")

                result = original_set_tag(*args, **kwargs)

                cost = adapter.calculate_cost({'operation_type': 'set_tag'})
                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.set_tag = wrapped_set_tag

    def _patch_log_artifact(self, mlflow_module):
        """Patch mlflow.log_artifact to add cost tracking."""
        original_log_artifact = mlflow_module.log_artifact
        self._original_methods['log_artifact'] = original_log_artifact

        adapter = self

        def wrapped_log_artifact(*args, **kwargs):
            """Wrapped log_artifact with cost tracking."""
            with adapter.tracer.start_as_current_span("mlflow.log_artifact") as span:
                span.set_attribute("genops.provider", "mlflow")
                span.set_attribute("genops.operation_type", "log_artifact")

                # Get artifact path
                local_path = args[0] if args else kwargs.get('local_path')
                if local_path:
                    span.set_attribute("mlflow.artifact_path", local_path)

                result = original_log_artifact(*args, **kwargs)

                # Estimate artifact size and cost
                artifact_size_mb = 0.0
                if local_path:
                    try:
                        import os
                        artifact_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    except:
                        pass

                storage_backend = 's3' if adapter.tracking_uri.startswith('s3://') else 'local'
                cost = adapter.calculate_cost({
                    'operation_type': 'log_artifact',
                    'artifact_size_mb': artifact_size_mb,
                    'storage_backend': storage_backend
                })

                span.set_attribute("mlflow.artifact_size_mb", artifact_size_mb)
                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.log_artifact = wrapped_log_artifact

    def _patch_log_model(self, mlflow_module):
        """Patch mlflow.log_model to add governance and cost tracking."""
        original_log_model = mlflow_module.log_model
        self._original_methods['log_model'] = original_log_model

        adapter = self

        def wrapped_log_model(*args, **kwargs):
            """Wrapped log_model with governance and cost tracking."""
            with adapter.tracer.start_as_current_span("mlflow.log_model") as span:
                span.set_attribute("genops.provider", "mlflow")
                span.set_attribute("genops.operation_type", "log_model")

                # Add governance tags if not present
                if 'registered_model_name' not in kwargs:
                    kwargs.setdefault('registered_model_name', None)

                result = original_log_model(*args, **kwargs)

                # Estimate model size and cost
                model_size_mb = 1.0  # Default estimate
                storage_backend = 's3' if adapter.tracking_uri.startswith('s3://') else 'local'

                cost = adapter.calculate_cost({
                    'operation_type': 'log_model',
                    'model_size_mb': model_size_mb,
                    'storage_backend': storage_backend
                })

                span.set_attribute("mlflow.model_size_mb", model_size_mb)
                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.log_model = wrapped_log_model

    def _patch_register_model(self, mlflow_module):
        """Patch mlflow.register_model to add governance tracking."""
        original_register_model = mlflow_module.register_model
        self._original_methods['register_model'] = original_register_model

        adapter = self

        def wrapped_register_model(*args, **kwargs):
            """Wrapped register_model with governance tracking."""
            with adapter.tracer.start_as_current_span("mlflow.register_model") as span:
                span.set_attribute("genops.provider", "mlflow")
                span.set_attribute("genops.operation_type", "register_model")

                # Get model name
                name = kwargs.get('name') or (args[1] if len(args) > 1 else "unknown")
                span.set_attribute("mlflow.model_name", name)

                result = original_register_model(*args, **kwargs)

                # Record registry operation cost
                cost = adapter.calculate_cost({
                    'operation_type': 'register_model',
                    'model_size_mb': 1.0
                })

                adapter.telemetry.record_cost(
                    span=span,
                    cost=cost,
                    currency="USD",
                    provider="mlflow"
                )

                return result

        mlflow_module.register_model = wrapped_register_model

    def _patch_autolog_methods(self, mlflow_module):
        """Patch auto-logging setup methods."""
        # TODO: Implement auto-logging patches for sklearn, pytorch, tensorflow
        # These are more complex and require patching framework-specific modules
        logger.debug("Auto-logging instrumentation not yet implemented")


# ============================================================================
# Factory Functions
# ============================================================================

def instrument_mlflow(
    tracking_uri: Optional[str] = None,
    registry_uri: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> GenOpsMLflowAdapter:
    """
    Create and return a GenOpsMLflowAdapter instance.

    Args:
        tracking_uri: MLflow tracking server URI
        registry_uri: MLflow model registry URI
        team: Team identifier for governance
        project: Project identifier for governance
        **kwargs: Additional configuration options

    Returns:
        Configured GenOpsMLflowAdapter instance

    Example:
        ```python
        from genops.providers.mlflow import instrument_mlflow

        adapter = instrument_mlflow(
            tracking_uri="http://localhost:5000",
            team="ml-team",
            project="model-optimization"
        )
        ```
    """
    return GenOpsMLflowAdapter(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        team=team,
        project=project,
        **kwargs
    )
