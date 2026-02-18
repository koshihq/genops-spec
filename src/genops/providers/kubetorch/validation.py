"""
Setup validation and diagnostics for Kubetorch integration.

This module provides comprehensive validation of the Kubetorch setup,
checking dependencies, configuration, and environment to help developers
troubleshoot issues quickly.

Example:
    >>> from genops.providers.kubetorch import validate_kubetorch_setup, print_validation_result
    >>> result = validate_kubetorch_setup()
    >>> print_validation_result(result)
"""

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation issue severity level."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class ValidationIssue:
    """Represents a single validation issue or check result."""

    level: ValidationLevel
    component: str  # Component being validated
    message: str  # Description of the issue
    fix_suggestion: Optional[str] = None  # How to fix the issue
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of validation issue."""
        level_symbols = {
            ValidationLevel.ERROR: "❌",
            ValidationLevel.WARNING: "⚠️ ",
            ValidationLevel.INFO: "ℹ️ ",
            ValidationLevel.SUCCESS: "✅",
        }
        symbol = level_symbols.get(self.level, "•")

        result = f"{symbol} [{self.component}] {self.message}"

        if self.fix_suggestion:
            result += f"\n   Fix: {self.fix_suggestion}"

        return result


@dataclass
class ValidationResult:
    """Complete validation result with all checks."""

    issues: list[ValidationIssue] = field(default_factory=list)
    total_checks: int = 0
    successful_checks: int = 0
    warnings: int = 0
    errors: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the result."""
        self.issues.append(issue)
        self.total_checks += 1

        if issue.level == ValidationLevel.SUCCESS:
            self.successful_checks += 1
        elif issue.level == ValidationLevel.WARNING:
            self.warnings += 1
        elif issue.level == ValidationLevel.ERROR:
            self.errors += 1

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return self.errors == 0

    def summary(self) -> str:
        """Get validation summary."""
        if self.is_valid():
            return f"✅ Validation passed: {self.successful_checks}/{self.total_checks} checks successful"
        else:
            return (
                f"❌ Validation failed: {self.errors} errors, "
                f"{self.warnings} warnings, "
                f"{self.successful_checks} successful"
            )


def validate_kubetorch_setup(
    check_kubetorch: bool = True,
    check_kubernetes: bool = True,
    check_gpu: bool = True,
    check_opentelemetry: bool = True,
    check_genops: bool = True,
) -> ValidationResult:
    """
    Validate Kubetorch integration setup.

    Performs comprehensive checks on:
    - Kubetorch (runhouse) installation and version
    - Kubernetes environment detection
    - GPU availability (if applicable)
    - OpenTelemetry configuration
    - GenOps configuration

    Args:
        check_kubetorch: Check Kubetorch installation
        check_kubernetes: Check Kubernetes environment
        check_gpu: Check GPU availability
        check_opentelemetry: Check OpenTelemetry setup
        check_genops: Check GenOps configuration

    Returns:
        ValidationResult with all check results

    Example:
        >>> result = validate_kubetorch_setup()
        >>> if not result.is_valid():
        ...     print(result.summary())
        ...     for issue in result.issues:
        ...         if issue.level == ValidationLevel.ERROR:
        ...             print(issue)
    """
    result = ValidationResult()

    logger.info("Running Kubetorch setup validation")

    # Check Python version
    _check_python_version(result)

    # Check Kubetorch installation
    if check_kubetorch:
        _check_kubetorch_installation(result)

    # Check Kubernetes environment
    if check_kubernetes:
        _check_kubernetes_environment(result)

    # Check GPU availability
    if check_gpu:
        _check_gpu_availability(result)

    # Check OpenTelemetry
    if check_opentelemetry:
        _check_opentelemetry_setup(result)

    # Check GenOps configuration
    if check_genops:
        _check_genops_configuration(result)

    # Check GenOps Kubetorch modules
    _check_genops_kubetorch_modules(result)

    logger.info(f"Validation complete: {result.summary()}")

    return result


def _check_python_version(result: ValidationResult) -> None:
    """Check Python version compatibility."""
    py_version = sys.version_info

    if py_version >= (3, 8):
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Python",
                message=f"Python {py_version.major}.{py_version.minor}.{py_version.micro} (compatible)",
            )
        )
    else:
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="Python",
                message=f"Python {py_version.major}.{py_version.minor} is not supported",
                fix_suggestion="Upgrade to Python 3.8 or higher",
            )
        )


def _check_kubetorch_installation(result: ValidationResult) -> None:
    """Check Kubetorch (runhouse) installation and version."""
    try:
        import runhouse as rh

        version = getattr(rh, "__version__", "unknown")

        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Kubetorch",
                message=f"Runhouse {version} installed",
                details={"version": version},
            )
        )

    except ImportError:
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Kubetorch",
                message="Runhouse (Kubetorch) not installed",
                fix_suggestion="Install with: pip install runhouse",
                details={
                    "note": "GenOps will work without Kubetorch for cost estimation only"
                },
            )
        )


def _check_kubernetes_environment(result: ValidationResult) -> None:
    """Check Kubernetes environment detection."""
    try:
        # Check for Kubernetes environment variables
        k8s_indicators = {
            "KUBERNETES_SERVICE_HOST": os.getenv("KUBERNETES_SERVICE_HOST"),
            "KUBERNETES_PORT": os.getenv("KUBERNETES_PORT"),
        }

        if any(k8s_indicators.values()):
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="Kubernetes",
                    message="Running in Kubernetes environment",
                    details=k8s_indicators,
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    component="Kubernetes",
                    message="Not running in Kubernetes environment (local development)",
                    details={"note": "This is normal for local development"},
                )
            )

        # Check for kubectl
        import subprocess  # nosec B404 - subprocess required for CLI tool validation

        # Find absolute path to kubectl for security
        kubectl_path = shutil.which("kubectl")
        if not kubectl_path:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    component="kubectl",
                    message="kubectl not available",
                    fix_suggestion="Install kubectl for Kubernetes cluster management",
                )
            )
        else:
            try:
                subprocess.run(
                    [kubectl_path, "version", "--client"],  # nosec B607 - validated absolute path
                    capture_output=True,
                    check=True,
                    timeout=5,
                    shell=False,  # Explicit shell=False for security
                )
                result.add_issue(
                    ValidationIssue(
                        level=ValidationLevel.SUCCESS,
                        component="kubectl",
                        message="kubectl available",
                    )
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                result.add_issue(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        component="kubectl",
                        message=f"kubectl found but not working: {e}",
                        fix_suggestion="Ensure kubectl is properly configured",
                    )
                )

    except Exception as e:
        logger.debug(f"Kubernetes check failed: {e}")
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Kubernetes",
                message=f"Kubernetes check failed: {e}",
            )
        )


def _check_gpu_availability(result: ValidationResult) -> None:
    """Check GPU availability (PyTorch CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="GPU",
                    message=f"{gpu_count} GPU(s) available: {', '.join(gpu_names)}",
                    details={
                        "gpu_count": gpu_count,
                        "gpu_names": gpu_names,
                        "cuda_version": torch.version.cuda,
                    },
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    component="GPU",
                    message="No GPUs detected",
                    details={"note": "CPU-only mode - cost tracking still available"},
                )
            )

    except ImportError:
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.INFO,
                component="GPU",
                message="PyTorch not installed (GPU detection unavailable)",
                fix_suggestion="Install PyTorch to enable GPU detection: pip install torch",
            )
        )


def _check_opentelemetry_setup(result: ValidationResult) -> None:
    """Check OpenTelemetry configuration."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        # Check if tracer provider is set
        trace.get_tracer(__name__)

        if isinstance(trace.get_tracer_provider(), TracerProvider):
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="OpenTelemetry",
                    message="OpenTelemetry TracerProvider configured",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    component="OpenTelemetry",
                    message="OpenTelemetry TracerProvider not configured",
                    fix_suggestion="Configure OTLP exporter or use auto-instrumentation",
                )
            )

        # Check for OTLP endpoint configuration
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="OTLP Endpoint",
                    message=f"OTLP endpoint configured: {otlp_endpoint}",
                    details={"endpoint": otlp_endpoint},
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    component="OTLP Endpoint",
                    message="OTEL_EXPORTER_OTLP_ENDPOINT not set",
                    fix_suggestion="Set OTEL_EXPORTER_OTLP_ENDPOINT environment variable",
                )
            )

    except ImportError:
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="OpenTelemetry",
                message="OpenTelemetry not installed",
                fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk",
            )
        )


def _check_genops_configuration(result: ValidationResult) -> None:
    """Check GenOps configuration."""
    try:
        # Check GenOps environment variables
        genops_vars = {
            "GENOPS_TEAM": os.getenv("GENOPS_TEAM"),
            "GENOPS_PROJECT": os.getenv("GENOPS_PROJECT"),
            "GENOPS_ENVIRONMENT": os.getenv("GENOPS_ENVIRONMENT"),
        }

        configured_vars = {k: v for k, v in genops_vars.items() if v}

        if configured_vars:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="GenOps Config",
                    message=f"GenOps environment variables configured: {', '.join(configured_vars.keys())}",
                    details=configured_vars,
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    component="GenOps Config",
                    message="No GenOps environment variables set",
                    details={
                        "note": "You can pass governance attributes directly to instrumentation functions"
                    },
                )
            )

    except Exception as e:
        logger.debug(f"GenOps config check failed: {e}")


def _check_genops_kubetorch_modules(result: ValidationResult) -> None:
    """Check GenOps Kubetorch module availability."""
    try:
        from . import get_module_status

        status = get_module_status()

        for module, available in status.items():
            if available:
                result.add_issue(
                    ValidationIssue(
                        level=ValidationLevel.SUCCESS,
                        component=f"Module:{module}",
                        message=f"{module.capitalize()} module available",
                    )
                )
            else:
                result.add_issue(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        component=f"Module:{module}",
                        message=f"{module.capitalize()} module not available",
                    )
                )

    except Exception as e:
        logger.debug(f"Module status check failed: {e}")
        result.add_issue(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="GenOps Kubetorch",
                message=f"Failed to check module status: {e}",
            )
        )


def print_validation_result(
    result: ValidationResult, show_all: bool = False, show_details: bool = False
) -> None:
    """
    Print validation result in a user-friendly format.

    Args:
        result: ValidationResult to print
        show_all: Show all issues (default: only errors and warnings)
        show_details: Show detailed information for each issue

    Example:
        >>> result = validate_kubetorch_setup()
        >>> print_validation_result(result, show_all=True)
    """
    print("\n" + "=" * 60)
    print("GenOps Kubetorch Setup Validation")
    print("=" * 60)

    # Print summary
    print(f"\n{result.summary()}")
    print(f"  Total Checks: {result.total_checks}")
    print(f"  ✅ Successful: {result.successful_checks}")
    print(f"  ⚠️  Warnings: {result.warnings}")
    print(f"  ❌ Errors: {result.errors}")

    # Group issues by level
    errors = [i for i in result.issues if i.level == ValidationLevel.ERROR]
    warnings = [i for i in result.issues if i.level == ValidationLevel.WARNING]
    info = [i for i in result.issues if i.level == ValidationLevel.INFO]
    success = [i for i in result.issues if i.level == ValidationLevel.SUCCESS]

    # Print errors
    if errors:
        print("\n" + "-" * 60)
        print("ERRORS:")
        print("-" * 60)
        for issue in errors:
            print(f"\n{issue}")
            if show_details and issue.details:
                print(f"   Details: {issue.details}")

    # Print warnings
    if warnings:
        print("\n" + "-" * 60)
        print("WARNINGS:")
        print("-" * 60)
        for issue in warnings:
            print(f"\n{issue}")
            if show_details and issue.details:
                print(f"   Details: {issue.details}")

    # Print info and success if requested
    if show_all:
        if info:
            print("\n" + "-" * 60)
            print("INFO:")
            print("-" * 60)
            for issue in info:
                print(f"\n{issue}")
                if show_details and issue.details:
                    print(f"   Details: {issue.details}")

        if success:
            print("\n" + "-" * 60)
            print("SUCCESSFUL CHECKS:")
            print("-" * 60)
            for issue in success:
                print(f"\n{issue}")
                if show_details and issue.details:
                    print(f"   Details: {issue.details}")

    print("\n" + "=" * 60)

    # Final recommendation
    if result.is_valid():
        print("✅ Setup is ready! You can start using Kubetorch with GenOps.")
    else:
        print("❌ Please fix the errors above before using Kubetorch with GenOps.")

    print("=" * 60 + "\n")
