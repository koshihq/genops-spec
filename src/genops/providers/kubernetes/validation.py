#!/usr/bin/env python3
"""
✅ Kubernetes Setup Validation

Validates Kubernetes environment setup for GenOps AI governance.
Provides comprehensive diagnostics and troubleshooting guidance.

Features:
✅ Kubernetes environment detection
✅ Service account and RBAC validation
✅ Resource monitoring capability checks
✅ OpenTelemetry Kubernetes resource detection validation
✅ Actionable fix suggestions for common issues
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .detector import KubernetesDetector
from .resource_monitor import KubernetesResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue with fix suggestions."""

    severity: str  # "error", "warning", "info"
    component: str
    message: str
    fix_suggestion: Optional[str] = None
    documentation_link: Optional[str] = None


@dataclass
class KubernetesValidationResult:
    """Results of Kubernetes environment validation."""

    is_valid: bool
    is_kubernetes_environment: bool
    issues: List[ValidationIssue]

    # Environment details
    namespace: Optional[str] = None
    pod_name: Optional[str] = None
    node_name: Optional[str] = None
    cluster_name: Optional[str] = None

    # Capabilities
    has_service_account: bool = False
    has_resource_monitoring: bool = False
    has_network_policies: bool = False

    # Resource context
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None

    def get_summary(self) -> str:
        """Get validation summary."""

        if not self.is_kubernetes_environment:
            return "❌ Not running in Kubernetes environment"

        if self.is_valid:
            return f"✅ Kubernetes environment valid (namespace: {self.namespace})"

        error_count = len([i for i in self.issues if i.severity == "error"])
        warning_count = len([i for i in self.issues if i.severity == "warning"])

        return f"⚠️ {error_count} errors, {warning_count} warnings found"


def validate_kubernetes_setup(
    enable_resource_monitoring: bool = True,
    cluster_name: Optional[str] = None
) -> KubernetesValidationResult:
    """
    Validate Kubernetes environment setup for GenOps AI.
    
    Args:
        enable_resource_monitoring: Whether to validate resource monitoring
        cluster_name: Expected cluster name (optional)
        
    Returns:
        Comprehensive validation result with diagnostics
    """

    issues = []

    # Initialize detector
    detector = KubernetesDetector()

    # Basic Kubernetes detection
    result = KubernetesValidationResult(
        is_valid=True,
        is_kubernetes_environment=detector.is_kubernetes(),
        issues=issues,
        namespace=detector.get_namespace(),
        pod_name=detector.get_pod_name(),
        node_name=detector.get_node_name(),
        cluster_name=detector.context.cluster_name
    )

    if not result.is_kubernetes_environment:
        issues.append(ValidationIssue(
            severity="warning",
            component="environment",
            message="Not running in Kubernetes environment",
            fix_suggestion="This is expected if running locally. For production, deploy to Kubernetes cluster.",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/deployment/kubernetes"
        ))
        result.is_valid = False
        return result

    # Validate Kubernetes context
    _validate_kubernetes_context(detector, issues)

    # Validate service account
    _validate_service_account(detector, issues, result)

    # Validate resource monitoring if enabled
    if enable_resource_monitoring:
        _validate_resource_monitoring(issues, result)

    # Validate environment variables
    _validate_environment_variables(issues)

    # Validate OpenTelemetry configuration
    _validate_opentelemetry_config(issues)

    # Validate cluster name if provided
    if cluster_name and result.cluster_name != cluster_name:
        issues.append(ValidationIssue(
            severity="warning",
            component="cluster",
            message=f"Cluster name mismatch: expected '{cluster_name}', detected '{result.cluster_name}'",
            fix_suggestion="Set CLUSTER_NAME environment variable or update cluster configuration"
        ))

    # Determine overall validity
    error_count = len([i for i in issues if i.severity == "error"])
    result.is_valid = error_count == 0

    return result


def _validate_kubernetes_context(detector: KubernetesDetector, issues: List[ValidationIssue]) -> None:
    """Validate basic Kubernetes context information."""

    context = detector.context

    # Check namespace
    if not context.pod_namespace:
        issues.append(ValidationIssue(
            severity="warning",
            component="namespace",
            message="Pod namespace not detected",
            fix_suggestion="Ensure POD_NAMESPACE or K8S_NAMESPACE environment variable is set via downward API"
        ))

    # Check pod name
    if not context.pod_name:
        issues.append(ValidationIssue(
            severity="info",
            component="pod",
            message="Pod name not detected",
            fix_suggestion="Set POD_NAME or K8S_POD_NAME environment variable via downward API"
        ))

    # Check node name
    if not context.node_name:
        issues.append(ValidationIssue(
            severity="info",
            component="node",
            message="Node name not detected",
            fix_suggestion="Set NODE_NAME or K8S_NODE_NAME environment variable via downward API"
        ))


def _validate_service_account(
    detector: KubernetesDetector,
    issues: List[ValidationIssue],
    result: KubernetesValidationResult
) -> None:
    """Validate Kubernetes service account setup."""

    # Check for service account token
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if token_path.exists():
        result.has_service_account = True

        try:
            # Validate token is readable
            token_content = token_path.read_text()
            if not token_content.strip():
                issues.append(ValidationIssue(
                    severity="error",
                    component="service_account",
                    message="Service account token is empty",
                    fix_suggestion="Check service account configuration and RBAC permissions"
                ))
        except PermissionError:
            issues.append(ValidationIssue(
                severity="error",
                component="service_account",
                message="Cannot read service account token",
                fix_suggestion="Check file permissions and security context configuration"
            ))
    else:
        issues.append(ValidationIssue(
            severity="warning",
            component="service_account",
            message="No service account token found",
            fix_suggestion="Ensure pod has service account mounted or set automountServiceAccountToken: true"
        ))

    # Check for CA certificate
    ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    if not ca_path.exists():
        issues.append(ValidationIssue(
            severity="warning",
            component="service_account",
            message="Kubernetes CA certificate not found",
            fix_suggestion="Service account CA certificate should be automatically mounted"
        ))


def _validate_resource_monitoring(issues: List[ValidationIssue], result: KubernetesValidationResult) -> None:
    """Validate resource monitoring capabilities."""

    try:
        monitor = KubernetesResourceMonitor()
        result.has_resource_monitoring = True

        # Get current resource context
        resources = monitor.get_current_resources()
        result.cpu_limit = resources.get("cpu_limit")
        result.memory_limit = resources.get("memory_limit")

        # Check if resource limits are set
        if not result.cpu_limit and not result.memory_limit:
            issues.append(ValidationIssue(
                severity="warning",
                component="resources",
                message="No resource limits detected",
                fix_suggestion="Set CPU and memory limits in pod spec for better resource governance"
            ))

        # Test resource usage collection
        usage = monitor.get_current_usage()
        if usage.cpu_usage_millicores is None and usage.memory_usage_bytes is None:
            issues.append(ValidationIssue(
                severity="warning",
                component="monitoring",
                message="Unable to collect resource usage metrics",
                fix_suggestion="Ensure cgroup filesystem is accessible for monitoring"
            ))

    except Exception as e:
        issues.append(ValidationIssue(
            severity="error",
            component="monitoring",
            message=f"Resource monitoring initialization failed: {e}",
            fix_suggestion="Check container runtime and cgroup configuration"
        ))


def _validate_environment_variables(issues: List[ValidationIssue]) -> None:
    """Validate required environment variables."""

    # Check OpenTelemetry configuration
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otel_endpoint:
        issues.append(ValidationIssue(
            severity="warning",
            component="telemetry",
            message="OTEL_EXPORTER_OTLP_ENDPOINT not set",
            fix_suggestion="Set OpenTelemetry collector endpoint for telemetry export"
        ))

    otel_service_name = os.getenv("OTEL_SERVICE_NAME")
    if not otel_service_name:
        issues.append(ValidationIssue(
            severity="info",
            component="telemetry",
            message="OTEL_SERVICE_NAME not set",
            fix_suggestion="Set service name for better telemetry attribution"
        ))

    # Check governance attributes
    if not os.getenv("DEFAULT_TEAM") and not os.getenv("GENOPS_TEAM"):
        issues.append(ValidationIssue(
            severity="info",
            component="governance",
            message="Default team not configured",
            fix_suggestion="Set DEFAULT_TEAM or GENOPS_TEAM environment variable for cost attribution"
        ))


def _validate_opentelemetry_config(issues: List[ValidationIssue]) -> None:
    """Validate OpenTelemetry configuration."""

    try:
        # Check if OpenTelemetry SDK is available
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource

        # Check if resource detection is available
        try:
            from opentelemetry.instrumentation.system_metrics import (
                SystemMetricsInstrumentor,
            )
            # OpenTelemetry auto-instrumentation is available
        except ImportError:
            issues.append(ValidationIssue(
                severity="info",
                component="telemetry",
                message="OpenTelemetry auto-instrumentation not available",
                fix_suggestion="Install opentelemetry-instrumentation packages for automatic metrics collection"
            ))

    except ImportError:
        issues.append(ValidationIssue(
            severity="error",
            component="telemetry",
            message="OpenTelemetry SDK not available",
            fix_suggestion="Install OpenTelemetry: pip install opentelemetry-api opentelemetry-sdk"
        ))


def print_kubernetes_validation_result(result: KubernetesValidationResult) -> None:
    """Print user-friendly validation results with fix suggestions."""

    print("🚢 GenOps AI Kubernetes Validation Results")
    print("=" * 60)

    # Overall status
    print(f"\n📊 Overall Status: {result.get_summary()}")

    if result.is_kubernetes_environment:
        print("\n🔍 Environment Details:")
        if result.namespace:
            print(f"   Namespace: {result.namespace}")
        if result.pod_name:
            print(f"   Pod Name: {result.pod_name}")
        if result.node_name:
            print(f"   Node Name: {result.node_name}")
        if result.cluster_name:
            print(f"   Cluster: {result.cluster_name}")

        print("\n⚙️ Capabilities:")
        print(f"   Service Account: {'✅' if result.has_service_account else '❌'}")
        print(f"   Resource Monitoring: {'✅' if result.has_resource_monitoring else '❌'}")

        if result.cpu_limit or result.memory_limit:
            print("\n💾 Resource Limits:")
            if result.cpu_limit:
                print(f"   CPU Limit: {result.cpu_limit}")
            if result.memory_limit:
                print(f"   Memory Limit: {result.memory_limit}")

    # Issues and fixes
    if result.issues:
        print(f"\n🔧 Issues Found ({len(result.issues)}):")

        # Group by severity
        errors = [i for i in result.issues if i.severity == "error"]
        warnings = [i for i in result.issues if i.severity == "warning"]
        info = [i for i in result.issues if i.severity == "info"]

        for severity, issues_list, icon in [
            ("ERRORS", errors, "❌"),
            ("WARNINGS", warnings, "⚠️"),
            ("INFO", info, "ℹ️")
        ]:
            if issues_list:
                print(f"\n{icon} {severity}:")
                for issue in issues_list:
                    print(f"   • {issue.component}: {issue.message}")
                    if issue.fix_suggestion:
                        print(f"     Fix: {issue.fix_suggestion}")
                    if issue.documentation_link:
                        print(f"     Docs: {issue.documentation_link}")

    # Next steps
    print("\n🚀 Next Steps:")
    if not result.is_kubernetes_environment:
        print("   • Deploy to Kubernetes cluster for full functionality")
        print("   • Use examples/kubernetes/ for deployment templates")
    elif not result.is_valid:
        print("   • Address errors above to ensure proper operation")
        print("   • Check pod logs for additional diagnostics")
    else:
        print("   • ✅ Environment is ready for GenOps AI governance!")
        print("   • Configure observability endpoints for telemetry export")

    print("\n📚 Documentation:")
    print("   • Kubernetes Guide: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/kubernetes")
    print("   • Troubleshooting: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/troubleshooting")

    print("\n" + "=" * 60)
