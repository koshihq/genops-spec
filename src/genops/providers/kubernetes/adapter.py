#!/usr/bin/env python3
"""
🚢 GenOps Kubernetes Adapter

Universal Kubernetes adapter for AI workload governance and observability.
Provides seamless integration between any AI provider and Kubernetes-native telemetry.

Features:
✅ Auto-detection and attribution of Kubernetes context
✅ Resource quota enforcement and monitoring
✅ Multi-tenant namespace isolation
✅ Integration with OpenTelemetry Kubernetes resource detection
✅ Support for any AI provider running in Kubernetes pods
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from ..base.provider import BaseFrameworkProvider
from .detector import KubernetesDetector
from .resource_monitor import KubernetesResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class KubernetesGovernanceContext:
    """Kubernetes-specific governance context for AI operations."""

    # Kubernetes identification
    namespace: Optional[str] = None
    pod_name: Optional[str] = None
    node_name: Optional[str] = None
    cluster_name: Optional[str] = None

    # Resource management
    cpu_request: Optional[str] = None
    cpu_limit: Optional[str] = None
    memory_request: Optional[str] = None
    memory_limit: Optional[str] = None

    # Multi-tenancy
    tenant_id: Optional[str] = None
    team: Optional[str] = None
    cost_center: Optional[str] = None

    # Policy context
    service_account: Optional[str] = None
    security_context: Optional[dict[str, Any]] = None

    # Runtime state
    operation_id: Optional[str] = None
    start_time: Optional[float] = None

    def to_telemetry_attributes(self) -> dict[str, Any]:
        """Convert context to telemetry attributes."""

        attributes = {}

        # Kubernetes attributes
        if self.namespace:
            attributes["k8s.namespace.name"] = self.namespace
            attributes["genops.tenant"] = (
                self.namespace
            )  # Use namespace as default tenant
        if self.pod_name:
            attributes["k8s.pod.name"] = self.pod_name
        if self.node_name:
            attributes["k8s.node.name"] = self.node_name
        if self.cluster_name:
            attributes["k8s.cluster.name"] = self.cluster_name

        # Resource context
        if self.cpu_request:
            attributes["k8s.container.cpu.request"] = self.cpu_request
        if self.cpu_limit:
            attributes["k8s.container.cpu.limit"] = self.cpu_limit
        if self.memory_request:
            attributes["k8s.container.memory.request"] = self.memory_request
        if self.memory_limit:
            attributes["k8s.container.memory.limit"] = self.memory_limit

        # Governance attributes
        if self.tenant_id:
            attributes["genops.tenant"] = self.tenant_id
        if self.team:
            attributes["genops.team"] = self.team
        if self.cost_center:
            attributes["genops.cost_center"] = self.cost_center

        # Runtime attributes
        attributes["genops.runtime"] = "kubernetes"
        if self.operation_id:
            attributes["genops.operation.id"] = self.operation_id

        return attributes


class KubernetesAdapter(BaseFrameworkProvider):
    """
    Universal Kubernetes adapter for AI workload governance.

    This adapter provides Kubernetes-native telemetry and governance capabilities
    for any AI provider or framework running in Kubernetes pods.
    """

    def __init__(
        self,
        auto_detect: bool = True,
        enable_resource_monitoring: bool = True,
        cluster_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Kubernetes adapter.

        Args:
            auto_detect: Automatically detect Kubernetes environment
            enable_resource_monitoring: Enable resource usage monitoring
            cluster_name: Override cluster name detection
            **kwargs: Additional configuration options
        """
        super().__init__()

        self.auto_detect = auto_detect
        self.enable_resource_monitoring = enable_resource_monitoring
        self.cluster_name_override = cluster_name

        # Initialize Kubernetes detection
        self.detector = KubernetesDetector()

        # Initialize resource monitoring if enabled
        self.resource_monitor = None
        if enable_resource_monitoring and self.detector.is_kubernetes():
            try:
                self.resource_monitor = KubernetesResourceMonitor()
                logger.debug("✅ Kubernetes resource monitoring enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize resource monitoring: {e}")

        # Override cluster name if provided
        if cluster_name:
            self.detector.context.cluster_name = cluster_name

        logger.info(
            f"🚢 Kubernetes adapter initialized (K8s detected: {self.detector.is_kubernetes()})"
        )

    def is_available(self) -> bool:
        """Check if running in Kubernetes environment."""
        return self.detector.is_kubernetes() if self.auto_detect else True

    def get_framework_name(self) -> str:
        """Get framework name for telemetry."""
        return "kubernetes"

    def get_version(self) -> str:
        """Get Kubernetes version if available."""
        # In production, this could query the Kubernetes API for server version
        return "auto-detected"

    def create_governance_context(
        self, operation_name: str, **governance_attrs
    ) -> KubernetesGovernanceContext:
        """
        Create Kubernetes governance context for an AI operation.

        Args:
            operation_name: Name of the AI operation
            **governance_attrs: Additional governance attributes

        Returns:
            Kubernetes governance context with full telemetry attribution
        """

        k8s_context = self.detector.context

        # Create governance context with Kubernetes metadata
        context = KubernetesGovernanceContext(
            # Kubernetes identification
            namespace=k8s_context.pod_namespace,
            pod_name=k8s_context.pod_name,
            node_name=k8s_context.node_name,
            cluster_name=k8s_context.cluster_name,
            # Resource information from monitoring
            **self._get_resource_context(),
            # Runtime state
            operation_id=f"{operation_name}-{int(time.time() * 1000)}",
            start_time=time.time(),
        )

        # Apply governance attributes
        for key, value in governance_attrs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            elif key == "tenant_id":
                context.tenant_id = value
            elif key == "team":
                context.team = value
            elif key == "cost_center":
                context.cost_center = value

        # Use namespace as default tenant if not specified
        if not context.tenant_id and context.namespace:
            context.tenant_id = context.namespace

        return context

    def _get_resource_context(self) -> dict[str, Any]:
        """Get current resource context from monitoring."""

        resource_context = {}

        if self.resource_monitor:
            try:
                resource_info = self.resource_monitor.get_current_resources()
                resource_context.update(
                    {
                        "cpu_request": resource_info.get("cpu_request"),
                        "cpu_limit": resource_info.get("cpu_limit"),
                        "memory_request": resource_info.get("memory_request"),
                        "memory_limit": resource_info.get("memory_limit"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get resource context: {e}")

        return resource_context

    def get_telemetry_attributes(self, **additional_attrs) -> dict[str, Any]:
        """
        Get comprehensive telemetry attributes for Kubernetes environment.

        Args:
            **additional_attrs: Additional attributes to include

        Returns:
            Dictionary of telemetry attributes with Kubernetes context
        """

        # Start with Kubernetes governance attributes
        attributes = self.detector.get_governance_attributes()

        # Add resource monitoring data
        if self.resource_monitor:
            try:
                resource_attrs = self.resource_monitor.get_telemetry_attributes()
                attributes.update(resource_attrs)
            except Exception as e:
                logger.warning(f"Failed to get resource telemetry: {e}")

        # Add any additional attributes
        attributes.update(additional_attrs)

        return attributes

    def instrument_operation(
        self, operation_name: str, operation_function, **governance_attrs
    ):
        """
        Instrument an AI operation with Kubernetes governance context.

        Args:
            operation_name: Name of the operation for telemetry
            operation_function: Function to instrument
            **governance_attrs: Governance attributes for the operation

        Returns:
            Instrumented operation result with full Kubernetes telemetry
        """

        # Create governance context
        context = self.create_governance_context(operation_name, **governance_attrs)

        # Get telemetry attributes
        telemetry_attrs = context.to_telemetry_attributes()

        logger.debug(f"🚢 Instrumenting K8s operation: {operation_name}")
        logger.debug(f"   Namespace: {context.namespace}")
        logger.debug(f"   Pod: {context.pod_name}")
        logger.debug(f"   Tenant: {context.tenant_id}")

        try:
            # Execute operation with telemetry context
            start_time = time.time()
            result = operation_function()
            duration = time.time() - start_time

            # Add performance metrics
            telemetry_attrs.update(
                {
                    "genops.operation.duration_ms": duration * 1000,
                    "genops.operation.success": True,
                    "genops.operation.name": operation_name,
                }
            )

            # TODO: Emit telemetry via OpenTelemetry
            logger.info(
                f"✅ K8s operation completed: {operation_name} ({duration * 1000:.2f}ms)"
            )

            return result

        except Exception as e:
            # Add error telemetry
            telemetry_attrs.update(
                {
                    "genops.operation.success": False,
                    "genops.operation.error": str(e),
                    "genops.operation.name": operation_name,
                }
            )

            logger.error(f"❌ K8s operation failed: {operation_name} - {e}")
            raise

    def get_resource_quotas(self) -> dict[str, Any]:
        """Get resource quotas and limits for current namespace."""

        if not self.resource_monitor:
            return {}

        try:
            return self.resource_monitor.get_namespace_quotas()
        except Exception as e:
            logger.warning(f"Failed to get resource quotas: {e}")
            return {}

    def check_resource_compliance(self, estimated_usage: dict[str, Any]) -> bool:
        """
        Check if estimated resource usage complies with quotas.

        Args:
            estimated_usage: Expected resource usage for an operation

        Returns:
            True if usage is within quotas, False otherwise
        """

        if not self.resource_monitor:
            return True  # Allow if monitoring unavailable

        try:
            return self.resource_monitor.check_quota_compliance(estimated_usage)
        except Exception as e:
            logger.warning(f"Resource compliance check failed: {e}")
            return True  # Allow on error to avoid blocking operations


@contextmanager
def create_kubernetes_context(
    operation_name: str, adapter: Optional[KubernetesAdapter] = None, **governance_attrs
):
    """
    Context manager for Kubernetes-aware AI operations.

    Args:
        operation_name: Name of the AI operation
        adapter: Kubernetes adapter instance (auto-created if None)
        **governance_attrs: Governance attributes for the operation

    Yields:
        Kubernetes governance context with full telemetry

    Example:
        ```python
        with create_kubernetes_context("chat-completion", team="ai-platform") as ctx:
            # Your AI operation here
            result = openai_client.chat.completions.create(...)
            # Telemetry automatically includes Kubernetes context
        ```
    """

    if adapter is None:
        adapter = KubernetesAdapter()

    # Create governance context
    context = adapter.create_governance_context(operation_name, **governance_attrs)

    logger.debug(f"🚢 Starting K8s governance context: {operation_name}")

    try:
        yield context

        # Calculate duration
        duration = time.time() - context.start_time if context.start_time else 0

        logger.info(
            f"✅ K8s operation completed: {operation_name} "
            f"(namespace: {context.namespace}, duration: {duration * 1000:.2f}ms)"
        )

    except Exception as e:
        logger.error(f"❌ K8s operation failed: {operation_name} - {e}")
        raise

    finally:
        logger.debug(f"🚢 K8s governance context closed: {operation_name}")
