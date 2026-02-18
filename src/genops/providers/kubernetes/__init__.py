#!/usr/bin/env python3
"""
🚢 GenOps Kubernetes Provider

Universal Kubernetes integration for AI workload governance and observability.
Provides cloud-native telemetry collection, resource attribution, and policy enforcement.

Features:
✅ Auto-discovery of Kubernetes context (namespace, pod, node)
✅ Integration with existing OpenTelemetry K8s resource detection
✅ Support for any AI provider running in Kubernetes pods
✅ Comprehensive governance telemetry with K8s metadata
✅ Resource quota and limit enforcement
✅ Multi-tenant namespace isolation
"""

from .adapter import KubernetesAdapter, create_kubernetes_context
from .detector import KubernetesDetector
from .resource_monitor import KubernetesResourceMonitor
from .validation import print_kubernetes_validation_result, validate_kubernetes_setup

__all__ = [
    "KubernetesAdapter",
    "create_kubernetes_context",
    "KubernetesDetector",
    "KubernetesResourceMonitor",
    "validate_kubernetes_setup",
    "print_kubernetes_validation_result",
]
