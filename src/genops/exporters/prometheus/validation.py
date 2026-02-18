"""Validation system for Prometheus exporter setup and diagnostics."""

from __future__ import annotations

import logging
import os
import socket
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import requests  # noqa: F401

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from prometheus_client import REGISTRY  # noqa: F401

    HAS_PROMETHEUS_CLIENT = True
except ImportError:
    HAS_PROMETHEUS_CLIENT = False

try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader  # noqa: F401

    HAS_PROMETHEUS_EXPORTER = True
except ImportError:
    HAS_PROMETHEUS_EXPORTER = False


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""

    DEPENDENCIES = "dependencies"
    CONFIGURATION = "configuration"
    CONNECTIVITY = "connectivity"
    PROMETHEUS_SERVER = "prometheus_server"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    category: ValidationCategory
    level: ValidationLevel
    title: str
    description: str
    fix_suggestion: str = ""
    technical_details: str = ""

    def __str__(self) -> str:
        level_symbol = {
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.ERROR: "❌",
            ValidationLevel.CRITICAL: "🚨",
        }

        return f"{level_symbol[self.level]} {self.title}: {self.description}"


@dataclass
class ValidationResult:
    """Complete validation results."""

    success: bool
    total_checks: int = 0
    passed_checks: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    @property
    def score(self) -> float:
        """Calculate validation score (0-100)."""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)

        # Update success status
        if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
            self.success = False


class PrometheusValidator:
    """Comprehensive validator for Prometheus exporter setup."""

    def __init__(
        self,
        port: int = 8000,
        prometheus_url: str | None = None,
        namespace: str | None = "genops",
    ):
        """
        Initialize validator.

        Args:
            port: Port for metrics endpoint
            prometheus_url: Prometheus server URL (optional)
            namespace: Metrics namespace
        """
        self.port = port
        self.prometheus_url = prometheus_url or os.getenv(
            "PROMETHEUS_URL", "http://localhost:9090"
        )
        self.namespace = namespace

    def validate(self) -> ValidationResult:
        """
        Run comprehensive validation checks.

        Returns:
            ValidationResult with detailed diagnostics
        """
        result = ValidationResult(success=True)

        # Collect system information
        result.system_info = {
            "python_version": sys.version,
            "port": self.port,
            "prometheus_url": self.prometheus_url,
            "namespace": self.namespace,
        }

        # Run validation checks
        self._check_dependencies(result)
        self._check_configuration(result)
        self._check_connectivity(result)
        self._check_prometheus_server(result)

        # Generate recommendations
        self._generate_recommendations(result)

        return result

    def _check_dependencies(self, result: ValidationResult):
        """Check required and optional dependencies."""

        # Python version check
        result.total_checks += 1
        py_version = sys.version_info
        if py_version >= (3, 8):
            result.passed_checks += 1
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.INFO,
                    title="Python Version",
                    description=f"Python {py_version.major}.{py_version.minor}.{py_version.micro} detected",
                    fix_suggestion="Compatible Python version",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.CRITICAL,
                    title="Python Version",
                    description=f"Python {py_version.major}.{py_version.minor} is too old",
                    fix_suggestion="Upgrade to Python 3.8 or later",
                    technical_details="GenOps requires Python 3.8+ for type hints and async support",
                )
            )

        # prometheus_client check
        result.total_checks += 1
        if HAS_PROMETHEUS_CLIENT:
            result.passed_checks += 1
            import prometheus_client

            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.INFO,
                    title="Prometheus Client",
                    description=f"prometheus_client {prometheus_client.__version__} installed",
                    fix_suggestion="Prometheus metrics client available",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.ERROR,
                    title="Prometheus Client Missing",
                    description="prometheus_client library not found",
                    fix_suggestion="Install with: pip install prometheus-client",
                    technical_details="Required for /metrics endpoint",
                )
            )

        # OpenTelemetry Prometheus exporter check
        result.total_checks += 1
        if HAS_PROMETHEUS_EXPORTER:
            result.passed_checks += 1
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.INFO,
                    title="OpenTelemetry Prometheus Exporter",
                    description="OpenTelemetry Prometheus exporter installed",
                    fix_suggestion="OTLP to Prometheus conversion available",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.ERROR,
                    title="OpenTelemetry Prometheus Exporter Missing",
                    description="opentelemetry-exporter-prometheus not found",
                    fix_suggestion="Install with: pip install opentelemetry-exporter-prometheus",
                    technical_details="Required for OpenTelemetry metrics export to Prometheus format",
                )
            )

        # OpenTelemetry SDK check
        result.total_checks += 1
        try:
            import opentelemetry  # noqa: F401
            from opentelemetry.sdk.metrics import MeterProvider  # noqa: F401

            result.passed_checks += 1
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.INFO,
                    title="OpenTelemetry SDK",
                    description="OpenTelemetry SDK packages available",
                    fix_suggestion="Metrics instrumentation enabled",
                )
            )
        except ImportError:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.ERROR,
                    title="OpenTelemetry SDK Missing",
                    description="OpenTelemetry SDK packages not found",
                    fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk",
                    technical_details="Required for governance telemetry generation",
                )
            )

        # Requests library check (optional, for validation only)
        result.total_checks += 1
        if HAS_REQUESTS:
            result.passed_checks += 1
            import requests

            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.INFO,
                    title="Requests Library",
                    description=f"requests {requests.__version__} installed",
                    fix_suggestion="HTTP connectivity validation enabled",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.WARNING,
                    title="Requests Library Not Installed",
                    description="requests library enables Prometheus server validation",
                    fix_suggestion="Install with: pip install requests (optional)",
                    technical_details="Used only for validating Prometheus server connectivity",
                )
            )

    def _check_configuration(self, result: ValidationResult):
        """Check configuration validity."""

        # Port range check
        result.total_checks += 1
        if 1024 <= self.port <= 65535:
            result.passed_checks += 1
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title="Port Configuration",
                    description=f"Port {self.port} is within valid range",
                    fix_suggestion="Valid port configuration",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.ERROR,
                    title="Invalid Port",
                    description=f"Port {self.port} is outside valid range (1024-65535)",
                    fix_suggestion="Use a port between 1024 and 65535",
                    technical_details="Ports below 1024 require root privileges; ports above 65535 are invalid",
                )
            )

        # Namespace validation
        result.total_checks += 1
        if self.namespace and self.namespace.replace("_", "").isalnum():
            result.passed_checks += 1
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title="Namespace Configuration",
                    description=f"Namespace '{self.namespace}' is valid",
                    fix_suggestion="Valid Prometheus metric namespace",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.ERROR,
                    title="Invalid Namespace",
                    description=f"Namespace '{self.namespace}' contains invalid characters",
                    fix_suggestion="Use only alphanumeric characters and underscores",
                    technical_details="Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*",
                )
            )

    def _check_connectivity(self, result: ValidationResult):
        """Check port availability."""

        result.total_checks += 1
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result_code = sock.connect_ex(("localhost", self.port))

                if result_code == 0:
                    # Port is in use
                    result.add_issue(
                        ValidationIssue(
                            category=ValidationCategory.CONNECTIVITY,
                            level=ValidationLevel.ERROR,
                            title="Port Already in Use",
                            description=f"Port {self.port} is already occupied",
                            fix_suggestion=f"Stop the service on port {self.port} or use a different port (e.g., PROMETHEUS_EXPORTER_PORT=8001)",
                            technical_details=f"Cannot bind to port {self.port} - another process is using it",
                        )
                    )
                else:
                    # Port is available
                    result.passed_checks += 1
                    result.add_issue(
                        ValidationIssue(
                            category=ValidationCategory.CONNECTIVITY,
                            level=ValidationLevel.INFO,
                            title="Port Available",
                            description=f"Port {self.port} is available",
                            fix_suggestion="Ready to start metrics server",
                        )
                    )

        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.WARNING,
                    title="Port Check Failed",
                    description=f"Could not verify port availability: {e}",
                    fix_suggestion="Port check inconclusive - may still work",
                    technical_details=str(e),
                )
            )

    def _check_prometheus_server(self, result: ValidationResult):
        """Check Prometheus server connectivity (optional)."""

        if not HAS_REQUESTS:
            result.total_checks += 1
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.PROMETHEUS_SERVER,
                    level=ValidationLevel.INFO,
                    title="Prometheus Server Check Skipped",
                    description="Requests library not available for server validation",
                    fix_suggestion="Install requests to enable server connectivity validation",
                )
            )
            return

        result.total_checks += 1
        try:
            import requests

            # Try to reach Prometheus server
            response = requests.get(f"{self.prometheus_url}/-/healthy", timeout=2)

            if response.status_code == 200:
                result.passed_checks += 1
                result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.PROMETHEUS_SERVER,
                        level=ValidationLevel.INFO,
                        title="Prometheus Server Reachable",
                        description=f"Prometheus server at {self.prometheus_url} is healthy",
                        fix_suggestion="Prometheus server available for scraping",
                    )
                )
            else:
                result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.PROMETHEUS_SERVER,
                        level=ValidationLevel.WARNING,
                        title="Prometheus Server Unhealthy",
                        description=f"Prometheus server returned status {response.status_code}",
                        fix_suggestion="Check Prometheus server logs",
                        technical_details=f"GET {self.prometheus_url}/-/healthy returned {response.status_code}",
                    )
                )

        except requests.exceptions.ConnectionError:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.PROMETHEUS_SERVER,
                    level=ValidationLevel.WARNING,
                    title="Prometheus Server Not Reachable",
                    description=f"Cannot connect to Prometheus at {self.prometheus_url}",
                    fix_suggestion="Start Prometheus server or update PROMETHEUS_URL",
                    technical_details="This is optional - metrics endpoint will work without a running Prometheus server",
                )
            )

        except requests.exceptions.Timeout:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.PROMETHEUS_SERVER,
                    level=ValidationLevel.WARNING,
                    title="Prometheus Server Timeout",
                    description=f"Prometheus server at {self.prometheus_url} did not respond in time",
                    fix_suggestion="Check network connectivity or Prometheus server health",
                )
            )

        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.PROMETHEUS_SERVER,
                    level=ValidationLevel.INFO,
                    title="Prometheus Server Check Failed",
                    description=f"Could not validate Prometheus server: {e}",
                    fix_suggestion="Prometheus server validation is optional",
                    technical_details=str(e),
                )
            )

    def _generate_recommendations(self, result: ValidationResult):
        """Generate actionable recommendations based on validation results."""

        if result.has_critical_issues:
            result.recommendations.append(
                "🚨 Critical issues detected. GenOps Prometheus export will not work until these are resolved."
            )

        if result.has_errors:
            result.recommendations.append(
                "❌ Errors detected. Install missing dependencies before using the Prometheus exporter."
            )

        # Missing dependencies
        if not HAS_PROMETHEUS_CLIENT or not HAS_PROMETHEUS_EXPORTER:
            result.recommendations.append(
                "Install Prometheus dependencies: pip install genops-ai[prometheus]"
            )

        # Port conflicts
        if any(issue.title == "Port Already in Use" for issue in result.issues):
            result.recommendations.append(
                "Use a different port: export PROMETHEUS_EXPORTER_PORT=8001"
            )

        # No Prometheus server
        if any(
            "Prometheus Server Not Reachable" in issue.title for issue in result.issues
        ):
            result.recommendations.append(
                "Start Prometheus server (optional): docker run -p 9090:9090 prom/prometheus"
            )

        # All checks passed
        if result.score == 100:
            result.recommendations.append(
                "✅ All checks passed! Start the exporter with: from genops.exporters.prometheus import instrument_prometheus; instrument_prometheus()"
            )


def validate_setup(
    port: int | None = None,
    prometheus_url: str | None = None,
    namespace: str | None = None,
) -> ValidationResult:
    """Validate Prometheus exporter setup.

    Args:
        port: Port for metrics endpoint (default: 8000)
        prometheus_url: Prometheus server URL (optional)
        namespace: Metrics namespace (default: genops)

    Returns:
        ValidationResult with comprehensive diagnostics

    Example:
        from genops.exporters.prometheus import validate_setup, print_validation_result

        result = validate_setup()
        print_validation_result(result)
    """
    # Use environment defaults if not provided
    port = port or int(os.getenv("PROMETHEUS_EXPORTER_PORT", "8000"))
    prometheus_url = prometheus_url or os.getenv(
        "PROMETHEUS_URL", "http://localhost:9090"
    )
    namespace = namespace or os.getenv("PROMETHEUS_NAMESPACE", "genops")

    validator = PrometheusValidator(
        port=port, prometheus_url=prometheus_url, namespace=namespace
    )

    return validator.validate()


def print_validation_result(result: ValidationResult) -> None:
    """Print validation results in a user-friendly format.

    Args:
        result: ValidationResult to display

    Example:
        result = validate_setup()
        print_validation_result(result)
    """
    print("\n" + "=" * 80)
    print("GenOps Prometheus Exporter Validation")
    print("=" * 80)

    # Overall status
    if result.success:
        print("\n✅ Overall Status: PASSED")
    else:
        print("\n❌ Overall Status: FAILED")

    print(
        f"   Score: {result.score:.1f}% ({result.passed_checks}/{result.total_checks} checks passed)"
    )

    # System information
    print("\n📋 System Information:")
    for key, value in result.system_info.items():
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 80:
            value_str = value_str[:77] + "..."
        print(f"   {key}: {value_str}")

    # Issues by category
    print("\n📊 Validation Results:")

    for category in ValidationCategory:
        category_issues = [
            issue for issue in result.issues if issue.category == category
        ]
        if not category_issues:
            continue

        print(f"\n   {category.value.upper()}:")
        for issue in category_issues:
            print(f"   {issue}")
            if issue.fix_suggestion and issue.level in [
                ValidationLevel.ERROR,
                ValidationLevel.CRITICAL,
                ValidationLevel.WARNING,
            ]:
                print(f"      → Fix: {issue.fix_suggestion}")

    # Recommendations
    if result.recommendations:
        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   {rec}")

    print("\n" + "=" * 80 + "\n")
