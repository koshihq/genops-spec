"""
Kubetorch provider adapter for GenOps AI governance.

This adapter extends GenOps governance to compute execution layer, providing:
- GPU resource allocation tracking
- Multi-resource cost aggregation (GPU, CPU, storage, network)
- Distributed training governance
- OpenTelemetry-based telemetry emission
- Integration with Kubernetes environment detection
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

from genops.providers.base import BaseFrameworkProvider
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Check for Kubetorch availability
try:
    import kubetorch
    HAS_KUBETORCH = True
except ImportError:
    HAS_KUBETORCH = False
    kubetorch = None  # type: ignore
    logger.info("Kubetorch not installed. Install with: pip install kubetorch")


@dataclass
class KubetorchOperation:
    """Represents a single Kubetorch compute operation for tracking."""

    operation_id: str
    operation_type: str  # 'compute.deploy', 'training.run', 'inference.run'
    workload_type: str  # 'training', 'fine-tuning', 'inference'

    # Resource allocation
    resource_type: str  # 'gpu', 'cpu'
    instance_type: str  # 'a100', 'h100', etc.
    num_devices: int
    device_memory_gb: Optional[int] = None

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Cost tracking
    cost_compute: Optional[float] = None
    cost_storage: Optional[float] = None
    cost_network: Optional[float] = None
    cost_total: Optional[float] = None
    currency: str = "USD"

    # Governance attributes
    governance_attributes: Dict[str, Any] = field(default_factory=dict)

    # Distributed training metadata
    distributed_strategy: Optional[str] = None  # 'ddp', 'fsdp', 'deepspeed'
    num_nodes: Optional[int] = None
    num_replicas: Optional[int] = None

    @property
    def gpu_hours(self) -> float:
        """Calculate GPU-hours consumed."""
        if self.duration_seconds is None or self.resource_type != 'gpu':
            return 0.0
        hours = self.duration_seconds / 3600
        return self.num_devices * hours

    @property
    def cpu_hours(self) -> float:
        """Calculate CPU-hours consumed."""
        if self.duration_seconds is None or self.resource_type != 'cpu':
            return 0.0
        hours = self.duration_seconds / 3600
        return self.num_devices * hours

    def finalize(self) -> None:
        """Finalize operation (calculate duration)."""
        if self.end_time is None:
            self.end_time = time.time()
        # Only calculate duration if not already set
        if self.duration_seconds is None:
            self.duration_seconds = self.end_time - self.start_time


class GenOpsKubetorchAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for Kubetorch with comprehensive compute governance.

    Provides cost tracking, telemetry, and policy enforcement for:
    - GPU/CPU resource allocation (.to(compute))
    - Dynamic scaling (.autoscale())
    - Distributed training (.distribute())
    - Fault recovery (retry/migrate/rescale)
    - Checkpoint management

    Example:
        >>> adapter = GenOpsKubetorchAdapter(
        ...     team="ml-research",
        ...     project="llm-training"
        ... )
        >>> result = adapter.track_compute_deployment(
        ...     instance_type="a100",
        ...     num_devices=8,
        ...     workload_type="training"
        ... )
    """

    def __init__(
        self,
        kubetorch_client: Optional[Any] = None,
        telemetry_enabled: bool = True,
        cost_tracking_enabled: bool = True,
        debug: bool = False,
        # Enterprise features (following Anyscale pattern)
        enable_retry: bool = True,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.0,
        enable_circuit_breaker: bool = False,
        circuit_breaker_threshold: int = 5,
        sampling_rate: float = 1.0,
        **governance_defaults
    ):
        """
        Initialize GenOps Kubetorch adapter.

        Args:
            kubetorch_client: Existing Kubetorch client (optional)
            telemetry_enabled: Enable OpenTelemetry export
            cost_tracking_enabled: Enable cost calculation and tracking
            debug: Enable debug logging

            Enterprise features:
            enable_retry: Enable automatic retry on transient failures
            max_retries: Maximum retry attempts
            retry_backoff_factor: Exponential backoff multiplier
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_threshold: Failures before opening circuit
            sampling_rate: Telemetry sampling rate 0.0-1.0 (default: 1.0)

            **governance_defaults: Default governance attributes (team, project, etc.)
        """
        # Initialize base provider
        super().__init__(client=kubetorch_client, **governance_defaults)

        # Configuration
        self.telemetry_enabled = telemetry_enabled
        self.cost_tracking_enabled = cost_tracking_enabled
        self.debug = debug
        self.governance_defaults = governance_defaults

        # Load pricing calculator
        from .pricing import KubetorchPricing
        self._pricing = KubetorchPricing()

        # Operation tracking
        self._current_operations: Dict[str, KubetorchOperation] = {}

        # Enterprise features
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))

        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._last_failure_time = 0.0

        # Detect Kubernetes environment
        self._detect_kubernetes_context()

        logger.info(
            f"GenOps Kubetorch adapter initialized "
            f"(telemetry={'enabled' if telemetry_enabled else 'disabled'}, "
            f"cost_tracking={'enabled' if cost_tracking_enabled else 'disabled'})"
        )

    def _detect_kubernetes_context(self) -> None:
        """Detect if running in Kubernetes and capture context."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector
            self.k8s_detector = KubernetesDetector()
            self.in_kubernetes = self.k8s_detector.is_kubernetes()

            if self.in_kubernetes:
                self.k8s_context = self.k8s_detector.context
                logger.debug(
                    f"✅ Kubernetes context detected: "
                    f"namespace={self.k8s_context.pod_namespace}, "
                    f"pod={self.k8s_context.pod_name}"
                )
            else:
                self.k8s_context = None
                logger.debug("Not running in Kubernetes environment")

        except Exception as e:
            logger.warning(f"Failed to detect Kubernetes context: {e}")
            self.in_kubernetes = False
            self.k8s_context = None
            self.k8s_detector = None

    # ==========================================
    # BaseFrameworkProvider Abstract Methods
    # ==========================================

    def setup_governance_attributes(self) -> None:
        """Setup Kubetorch-specific governance attributes."""
        self.REQUEST_ATTRIBUTES = {
            'instance_type', 'num_devices', 'device_memory_gb',
            'workload_type', 'distributed_strategy', 'num_nodes',
            'checkpoint_frequency_minutes', 'max_duration_hours',
            'priority', 'resource_type'
        }
        logger.debug(f"Kubetorch REQUEST_ATTRIBUTES: {self.REQUEST_ATTRIBUTES}")

    def get_framework_name(self) -> str:
        """Return framework name."""
        return "kubetorch"

    def get_framework_type(self) -> str:
        """Return framework type."""
        return self.FRAMEWORK_TYPE_DISTRIBUTED

    def get_framework_version(self) -> str | None:
        """Return Kubetorch version if available."""
        if not HAS_KUBETORCH or kubetorch is None:
            return None
        try:
            return getattr(kubetorch, '__version__', 'unknown')
        except (AttributeError, Exception):
            return None

    def is_framework_available(self) -> bool:
        """Check if Kubetorch is available."""
        return HAS_KUBETORCH

    def calculate_cost(self, operation_context: dict) -> float:
        """
        Calculate cost for Kubetorch compute operation.

        Args:
            operation_context: Dict with keys:
                - instance_type: GPU instance type (a100, h100, etc.)
                - num_devices: Number of GPU/CPU devices
                - duration_seconds: Operation duration
                - resource_type: 'gpu' or 'cpu'
                - storage_gb_hours: Optional storage consumption
                - network_cost: Optional network cost

        Returns:
            Total cost in USD
        """
        if not self.cost_tracking_enabled:
            return 0.0

        try:
            instance_type = operation_context.get('instance_type', '')
            num_devices = operation_context.get('num_devices', 1)
            duration_seconds = operation_context.get('duration_seconds', 0)
            resource_type = operation_context.get('resource_type', 'gpu')

            # Calculate compute cost
            cost_compute = self._pricing.calculate_compute_cost(
                instance_type=instance_type,
                num_devices=num_devices,
                duration_seconds=duration_seconds,
                resource_type=resource_type
            )

            # Add storage cost if provided
            cost_storage = 0.0
            storage_gb_hours = operation_context.get('storage_gb_hours', 0)
            if storage_gb_hours > 0:
                cost_storage = self._pricing.calculate_storage_cost(storage_gb_hours)

            # Add network cost if provided
            cost_network = operation_context.get('cost_network', 0.0)

            total_cost = cost_compute + cost_storage + cost_network

            if self.debug:
                logger.debug(
                    f"Cost calculation: compute=${cost_compute:.4f}, "
                    f"storage=${cost_storage:.4f}, network=${cost_network:.4f}, "
                    f"total=${total_cost:.4f}"
                )

            return total_cost

        except Exception as e:
            logger.warning(f"Failed to calculate cost: {e}")
            return 0.0

    def get_operation_mappings(self) -> dict[str, str]:
        """Return mapping of Kubetorch operations to instrumentation methods."""
        return {
            'compute.deploy': 'track_compute_deployment',
            'compute.scale': 'track_scaling_operation',
            'compute.checkpoint': 'track_checkpoint_operation',
            'compute.terminate': 'track_termination',
            'training.run': 'track_training_run',
            'inference.run': 'track_inference_run',
        }

    def _record_framework_metrics(
        self,
        span: Any,
        operation_type: str,
        context: dict
    ) -> None:
        """Record Kubetorch-specific metrics on span."""
        if not span:
            return

        try:
            # Compute resource metrics
            if 'instance_type' in context:
                span.set_attribute("genops.compute.instance_type", context['instance_type'])
            if 'num_devices' in context:
                span.set_attribute("genops.compute.num_devices", context['num_devices'])
            if 'resource_type' in context:
                span.set_attribute("genops.compute.resource_type", context['resource_type'])
            if 'device_memory_gb' in context:
                span.set_attribute("genops.compute.device_memory_gb", context['device_memory_gb'])

            # Workload classification
            if 'workload_type' in context:
                span.set_attribute("genops.workload.type", context['workload_type'])
            if 'workload_framework' in context:
                span.set_attribute("genops.workload.framework", context['workload_framework'])
            if 'workload_job_id' in context:
                span.set_attribute("genops.workload.job_id", context['workload_job_id'])

            # Cost metrics
            if 'cost_compute' in context:
                span.set_attribute("genops.cost.compute", context['cost_compute'])
            if 'cost_storage' in context:
                span.set_attribute("genops.cost.storage", context['cost_storage'])
            if 'cost_network' in context:
                span.set_attribute("genops.cost.network", context['cost_network'])
            if 'cost_total' in context:
                span.set_attribute("genops.cost.total", context['cost_total'])
                span.set_attribute("genops.cost.currency", "USD")

            # Resource consumption
            if 'gpu_hours' in context:
                span.set_attribute("genops.compute.gpu_hours", context['gpu_hours'])
            if 'cpu_hours' in context:
                span.set_attribute("genops.compute.cpu_hours", context['cpu_hours'])
            if 'duration_seconds' in context:
                span.set_attribute("genops.compute.duration_seconds", context['duration_seconds'])

            # Distributed training metrics
            if 'distributed_strategy' in context:
                span.set_attribute("genops.distributed.strategy", context['distributed_strategy'])
            if 'num_nodes' in context:
                span.set_attribute("genops.distributed.num_nodes", context['num_nodes'])
            if 'num_replicas' in context:
                span.set_attribute("genops.distributed.num_replicas", context['num_replicas'])

            # Kubernetes context if available
            if self.in_kubernetes and self.k8s_context:
                span.set_attribute("k8s.namespace", self.k8s_context.pod_namespace)
                span.set_attribute("k8s.pod.name", self.k8s_context.pod_name)
                if self.k8s_context.node_name:
                    span.set_attribute("k8s.node.name", self.k8s_context.node_name)

            # Operation type
            span.set_attribute("genops.compute.operation_type", operation_type)
            span.set_attribute("genops.compute.provider", "kubetorch")
            span.set_attribute("genops.compute.framework", "kubetorch")

        except Exception as e:
            logger.debug(f"Failed to record framework metrics: {e}")

    def _apply_instrumentation(self, **config) -> None:
        """Apply Kubetorch instrumentation."""
        # Kubetorch instrumentation will be handled by compute_monitor.py
        # For now, just log that the adapter is ready
        logger.info("Kubetorch adapter ready for instrumentation")

        if not HAS_KUBETORCH:
            logger.warning("Kubetorch not available - manual tracking only")

    def _remove_instrumentation(self) -> None:
        """Remove Kubetorch instrumentation."""
        logger.info("Kubetorch instrumentation removed")

    # ==========================================
    # Kubetorch-Specific Methods
    # ==========================================

    def track_compute_deployment(
        self,
        instance_type: str,
        num_devices: int,
        workload_type: str = "training",
        duration_seconds: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Track compute resource deployment with governance.

        Args:
            instance_type: GPU instance type (a100, h100, v100, etc.)
            num_devices: Number of devices to allocate
            workload_type: Type of workload (training, inference, fine-tuning)
            duration_seconds: Operation duration (if known)
            **kwargs: Additional parameters and governance attributes

        Returns:
            Dict with operation tracking info

        Example:
            >>> adapter = GenOpsKubetorchAdapter(team="ml-team")
            >>> result = adapter.track_compute_deployment(
            ...     instance_type="a100",
            ...     num_devices=8,
            ...     workload_type="training"
            ... )
        """
        # Extract governance attributes
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        effective_governance = {**self.governance_defaults, **governance_attrs}

        # Create operation tracking
        operation_id = str(uuid.uuid4())
        resource_type = kwargs.get('resource_type', 'gpu' if instance_type in ['a100', 'h100', 'v100', 'a10g', 't4'] else 'cpu')

        operation = KubetorchOperation(
            operation_id=operation_id,
            operation_type="compute.deploy",
            workload_type=workload_type,
            resource_type=resource_type,
            instance_type=instance_type,
            num_devices=num_devices,
            device_memory_gb=kwargs.get('device_memory_gb'),
            distributed_strategy=kwargs.get('distributed_strategy'),
            num_nodes=kwargs.get('num_nodes'),
            num_replicas=kwargs.get('num_replicas'),
            governance_attributes=effective_governance
        )

        # Store operation
        self._current_operations[operation_id] = operation

        # Build trace attributes
        trace_attrs = self._build_trace_attributes(
            operation_name="kubetorch.compute.deploy",
            operation_type="ai.compute",
            governance_attrs=effective_governance,
            instance_type=instance_type,
            num_devices=num_devices,
            workload_type=workload_type,
            resource_type=resource_type
        )

        # Start OpenTelemetry span
        with tracer.start_as_current_span(
            "kubetorch.compute.deploy",
            attributes=trace_attrs
        ) as span:
            try:
                # If duration provided, calculate cost immediately
                if duration_seconds is not None:
                    operation.duration_seconds = duration_seconds
                    operation.finalize()

                    if self.cost_tracking_enabled:
                        operation.cost_compute = self.calculate_cost({
                            'instance_type': instance_type,
                            'num_devices': num_devices,
                            'duration_seconds': duration_seconds,
                            'resource_type': resource_type
                        })
                        operation.cost_total = operation.cost_compute

                # Record metrics
                self._record_framework_metrics(span, "compute.deploy", {
                    'instance_type': instance_type,
                    'num_devices': num_devices,
                    'resource_type': resource_type,
                    'workload_type': workload_type,
                    'cost_compute': operation.cost_compute,
                    'cost_total': operation.cost_total,
                    'gpu_hours': operation.gpu_hours,
                    'cpu_hours': operation.cpu_hours,
                    'duration_seconds': operation.duration_seconds,
                    **kwargs
                })

                span.set_status(Status(StatusCode.OK))

                if self.debug:
                    logger.debug(
                        f"Compute deployed: {instance_type} x{num_devices}, "
                        f"cost=${operation.cost_total:.4f if operation.cost_total else 0:.4f}, "
                        f"gpu_hours={operation.gpu_hours:.2f}"
                    )

                return {
                    "operation_id": operation_id,
                    "cost_total": operation.cost_total,
                    "gpu_hours": operation.gpu_hours,
                    "cpu_hours": operation.cpu_hours,
                    "resource_type": resource_type,
                    "instance_type": instance_type,
                    "num_devices": num_devices
                }

            except Exception as e:
                operation.finalize()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Compute deployment failed: {e}")
                raise

    def finalize_operation(self, operation_id: str) -> Optional[KubetorchOperation]:
        """
        Finalize a tracked operation and calculate final costs.

        Args:
            operation_id: Operation ID returned from track_compute_deployment

        Returns:
            KubetorchOperation with final costs, or None if not found
        """
        operation = self._current_operations.get(operation_id)
        if not operation:
            logger.warning(f"Operation {operation_id} not found")
            return None

        operation.finalize()

        # Calculate final cost
        if self.cost_tracking_enabled:
            operation.cost_compute = self.calculate_cost({
                'instance_type': operation.instance_type,
                'num_devices': operation.num_devices,
                'duration_seconds': operation.duration_seconds,
                'resource_type': operation.resource_type
            })
            operation.cost_total = operation.cost_compute

        # Remove from tracking
        self._current_operations.pop(operation_id)

        return operation


# ==========================================
# Context Managers
# ==========================================

@contextmanager
def create_compute_context(
    workload_name: str,
    instance_type: str,
    num_devices: int,
    adapter: Optional[GenOpsKubetorchAdapter] = None,
    **governance_attrs
):
    """
    Context manager for Kubetorch compute operations.

    Args:
        workload_name: Name of the workload
        instance_type: GPU instance type
        num_devices: Number of devices
        adapter: Kubetorch adapter instance (auto-created if None)
        **governance_attrs: Governance attributes

    Yields:
        Dict with operation context

    Example:
        >>> with create_compute_context(
        ...     "train-bert",
        ...     "a100",
        ...     8,
        ...     team="ml-team"
        ... ) as ctx:
        ...     # Training code here
        ...     pass
    """
    if adapter is None:
        adapter = GenOpsKubetorchAdapter(**governance_attrs)

    start_time = time.time()

    logger.debug(f"🚀 Starting compute context: {workload_name}")

    # Track deployment
    result = adapter.track_compute_deployment(
        instance_type=instance_type,
        num_devices=num_devices,
        workload_type="training",
        **governance_attrs
    )

    operation_id = result["operation_id"]

    try:
        yield {
            "workload_name": workload_name,
            "instance_type": instance_type,
            "num_devices": num_devices,
            "start_time": start_time,
            "operation_id": operation_id
        }

        # Finalize operation on success
        duration = time.time() - start_time
        operation = adapter.finalize_operation(operation_id)

        if operation:
            logger.info(
                f"✅ Compute operation completed: {workload_name} "
                f"({operation.instance_type} x{operation.num_devices}, "
                f"{operation.gpu_hours:.2f} GPU-hours, ${operation.cost_total:.2f})"
            )

    except Exception as e:
        logger.error(f"❌ Compute operation failed: {workload_name} - {e}")
        # Still finalize to capture partial costs
        adapter.finalize_operation(operation_id)
        raise


def instrument_kubetorch(**governance_defaults) -> GenOpsKubetorchAdapter:
    """
    Create and initialize GenOps Kubetorch adapter.

    Args:
        **governance_defaults: Default governance attributes

    Returns:
        Initialized GenOpsKubetorchAdapter

    Example:
        >>> adapter = instrument_kubetorch(
        ...     team="ml-research",
        ...     project="llm-training",
        ...     environment="production"
        ... )
        >>> result = adapter.track_compute_deployment("a100", 8)
    """
    return GenOpsKubetorchAdapter(**governance_defaults)
