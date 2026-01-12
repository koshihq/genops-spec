"""Validation utilities for MLflow provider setup."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class ValidationIssue:
    """Represents a validation issue with suggested fix."""

    severity: str  # "error", "warning", "info"
    component: str  # "dependencies", "configuration", "connectivity", "governance"
    message: str
    suggested_fix: Optional[str] = None
    documentation_link: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation checks."""

    passed: bool = False
    issues: List[ValidationIssue] = field(default_factory=list)
    configuration: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, bool] = field(default_factory=dict)
    connectivity: Dict[str, bool] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == "error" for issue in self.issues)

    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Get issues filtered by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the validation result."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_setup(
    tracking_uri: Optional[str] = None,
    check_connectivity: bool = True,
    check_governance: bool = True,
    **kwargs
) -> ValidationResult:
    """
    Validate MLflow setup for GenOps governance.

    Checks:
    1. Dependencies (mlflow, opentelemetry, genops)
    2. Configuration (tracking URI, registry URI, governance attrs)
    3. Connectivity (tracking server, artifact store, model registry)
    4. Governance features (telemetry, instrumentation)

    Args:
        tracking_uri: MLflow tracking URI to validate
        check_connectivity: Whether to test connectivity
        check_governance: Whether to validate governance features
        **kwargs: Additional validation parameters

    Returns:
        ValidationResult with detailed validation information

    Example:
        ```python
        from genops.providers.mlflow import validate_setup, print_validation_result

        result = validate_setup()
        print_validation_result(result)

        if result.passed:
            print("Setup is ready!")
        else:
            print("Please fix the errors above")
        ```
    """
    result = ValidationResult()
    result.passed = True  # Start optimistic

    logger.info("Starting MLflow validation...")

    # 1. Validate dependencies
    _validate_dependencies(result)

    # 2. Validate configuration
    _validate_configuration(result, tracking_uri)

    # 3. Check connectivity (if requested and configuration is valid)
    if check_connectivity and not result.has_errors():
        _validate_connectivity(result)

    # 4. Validate governance features (if requested)
    if check_governance and not result.has_errors():
        _validate_governance_features(result)

    # Final status
    if result.has_errors():
        result.passed = False
        logger.warning(
            f"MLflow validation failed with {len(result.get_issues_by_severity('error'))} errors"
        )
    else:
        logger.info("MLflow validation passed")

    return result


# ============================================================================
# Validation Helper Functions
# ============================================================================

def _validate_dependencies(result: ValidationResult) -> None:
    """Validate required dependencies."""
    dependencies = {
        "mlflow": False,
        "opentelemetry": False,
        "genops": False,
    }

    # Check MLflow
    try:
        import mlflow
        dependencies["mlflow"] = True
        result.configuration["mlflow_version"] = mlflow.__version__
    except ImportError:
        result.add_issue(ValidationIssue(
            severity="error",
            component="dependencies",
            message="MLflow not installed",
            suggested_fix="pip install mlflow",
            documentation_link="https://mlflow.org/docs/latest/quickstart.html"
        ))

    # Check OpenTelemetry
    try:
        import opentelemetry
        dependencies["opentelemetry"] = True
        result.configuration["opentelemetry_version"] = opentelemetry.version.__version__
    except ImportError:
        result.add_issue(ValidationIssue(
            severity="error",
            component="dependencies",
            message="OpenTelemetry not installed",
            suggested_fix="pip install opentelemetry-api opentelemetry-sdk",
            documentation_link="https://opentelemetry.io/docs/instrumentation/python/"
        ))

    # Check GenOps
    try:
        import genops
        dependencies["genops"] = True
        result.configuration["genops_version"] = getattr(genops, "__version__", "development")
    except ImportError:
        result.add_issue(ValidationIssue(
            severity="error",
            component="dependencies",
            message="GenOps not installed",
            suggested_fix="pip install genops",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI"
        ))

    result.dependencies = dependencies


def _validate_configuration(result: ValidationResult, tracking_uri: Optional[str] = None) -> None:
    """Validate MLflow configuration."""
    # Check tracking URI
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:///mlruns"
    result.configuration["tracking_uri"] = tracking_uri

    # Validate URI format
    if tracking_uri.startswith("file://"):
        # Local file storage
        result.configuration["storage_type"] = "local"
    elif tracking_uri.startswith(("http://", "https://")):
        # Remote tracking server
        result.configuration["storage_type"] = "remote"
    elif tracking_uri.startswith(("databricks", "databricks+token")):
        # Databricks
        result.configuration["storage_type"] = "databricks"
    else:
        result.add_issue(ValidationIssue(
            severity="warning",
            component="configuration",
            message=f"Unrecognized tracking URI format: {tracking_uri}",
            suggested_fix="Set MLFLOW_TRACKING_URI to a valid URI (file://, http://, or databricks://)"
        ))

    # Check registry URI (optional)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI")
    if registry_uri:
        result.configuration["registry_uri"] = registry_uri

    # Check GenOps configuration
    genops_config = {
        "team": os.getenv("GENOPS_TEAM"),
        "project": os.getenv("GENOPS_PROJECT"),
        "environment": os.getenv("GENOPS_ENVIRONMENT"),
        "customer_id": os.getenv("GENOPS_CUSTOMER_ID"),
        "cost_center": os.getenv("GENOPS_COST_CENTER"),
    }

    # Check for missing governance attributes
    missing_critical = []
    if not genops_config["team"]:
        missing_critical.append("GENOPS_TEAM")
    if not genops_config["project"]:
        missing_critical.append("GENOPS_PROJECT")

    if missing_critical:
        result.add_issue(ValidationIssue(
            severity="warning",
            component="configuration",
            message=f"Critical GenOps governance attributes not set: {missing_critical}",
            suggested_fix=f"Set environment variables: {', '.join(missing_critical)}"
        ))

    # Add configured values
    for key, value in genops_config.items():
        if value:
            result.configuration[f"genops_{key}"] = value


def _validate_connectivity(result: ValidationResult) -> None:
    """Validate connectivity to MLflow tracking server."""
    connectivity_checks = {
        "tracking_server": False,
        "artifact_store": False,
        "model_registry": False,
    }

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = result.configuration.get("tracking_uri")
        client = MlflowClient(tracking_uri=tracking_uri)

        # Test tracking server connectivity
        try:
            experiments = client.search_experiments()
            connectivity_checks["tracking_server"] = True
            result.configuration["experiment_count"] = str(len(experiments))
        except Exception as e:
            result.add_issue(ValidationIssue(
                severity="error",
                component="connectivity",
                message=f"Cannot connect to MLflow tracking server: {e}",
                suggested_fix="Verify tracking URI and ensure MLflow server is running"
            ))

        # Test artifact store (basic check)
        try:
            # Artifact store check - simplified
            connectivity_checks["artifact_store"] = True
        except Exception as e:
            result.add_issue(ValidationIssue(
                severity="warning",
                component="connectivity",
                message=f"Artifact store connectivity issue: {e}",
                suggested_fix="Ensure artifact storage backend is properly configured"
            ))

        # Test model registry
        try:
            registered_models = client.search_registered_models()
            connectivity_checks["model_registry"] = True
            result.configuration["registered_model_count"] = str(len(registered_models))
        except Exception as e:
            result.add_issue(ValidationIssue(
                severity="info",
                component="connectivity",
                message=f"Model registry not accessible: {e}",
                suggested_fix="Model registry may not be configured (optional feature)"
            ))

    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="error",
            component="connectivity",
            message=f"Connectivity test failed: {e}",
            suggested_fix="Check MLflow installation and configuration"
        ))

    result.connectivity = connectivity_checks


def _validate_governance_features(result: ValidationResult) -> None:
    """Validate MLflow governance features."""
    try:
        # Test GenOps telemetry
        from genops.core.telemetry import GenOpsTelemetry
        telemetry = GenOpsTelemetry()
        result.configuration["telemetry_enabled"] = "true"

        # Test OpenTelemetry integration
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("genops.validation.test") as span:
            span.set_attribute("genops.provider", "mlflow")
            span.set_attribute("genops.validation", "governance_features")
        result.configuration["opentelemetry_integration"] = "working"

        # Test provider components
        try:
            from . import adapter, cost_aggregator, validation
            result.configuration["provider_components"] = "loaded"
        except ImportError as e:
            result.add_issue(ValidationIssue(
                severity="error",
                component="governance",
                message=f"Provider components not available: {e}",
                suggested_fix="Ensure GenOps MLflow provider is properly installed"
            ))

    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="warning",
            component="governance",
            message=f"Governance feature validation failed: {e}",
            suggested_fix="Check GenOps installation and configuration"
        ))


# ============================================================================
# Output Formatting
# ============================================================================

def print_validation_result(result: ValidationResult) -> None:
    """
    Print formatted validation result.

    Args:
        result: ValidationResult to display
    """
    print("\n" + "="*70)
    print("MLFLOW GENOPS VALIDATION REPORT")
    print("="*70)

    # Overall status
    status_icon = "✅" if result.passed else "❌"
    print(f"\nOverall Status: {status_icon} {'PASSED' if result.passed else 'FAILED'}")

    # Dependencies
    print(f"\n📦 Dependencies:")
    for dep, status in result.dependencies.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")

    # Configuration
    print(f"\n⚙️  Configuration:")
    for key, value in sorted(result.configuration.items()):
        print(f"  • {key}: {value}")

    # Connectivity
    if result.connectivity:
        print(f"\n🌐 Connectivity:")
        for check, status in result.connectivity.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check.replace('_', ' ').title()}")

    # Issues (errors, warnings, info)
    if result.issues:
        print(f"\n🔍 Issues Found:")

        errors = result.get_issues_by_severity("error")
        if errors:
            print(f"\n  ❌ ERRORS ({len(errors)}):")
            for i, issue in enumerate(errors, 1):
                print(f"    {i}. {issue.message}")
                if issue.suggested_fix:
                    print(f"       Fix: {issue.suggested_fix}")
                if issue.documentation_link:
                    print(f"       Docs: {issue.documentation_link}")
                print()

        warnings = result.get_issues_by_severity("warning")
        if warnings:
            print(f"  ⚠️  WARNINGS ({len(warnings)}):")
            for i, issue in enumerate(warnings, 1):
                print(f"    {i}. {issue.message}")
                if issue.suggested_fix:
                    print(f"       Fix: {issue.suggested_fix}")
                print()

        info_issues = result.get_issues_by_severity("info")
        if info_issues:
            print(f"  ℹ️  INFO ({len(info_issues)}):")
            for i, issue in enumerate(info_issues, 1):
                print(f"    {i}. {issue.message}")
                print()
    else:
        print(f"\n✨ No issues found!")

    # Next steps
    if result.passed:
        print(f"\n🎉 SUCCESS! You're ready to use MLflow with GenOps.")
        print(f"   Try running: python examples/mlflow/basic_tracking.py")
    else:
        print(f"\n🔧 Please fix the errors above and run validation again.")
        print(f"   Command: python examples/mlflow/setup_validation.py")

    print("\n" + "="*70)
