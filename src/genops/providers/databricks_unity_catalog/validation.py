"""Validation utilities for Databricks Unity Catalog provider."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue with suggested fix."""
    
    severity: str  # "error", "warning", "info"
    component: str  # "configuration", "dependencies", "connectivity"
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
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == "error" for issue in self.issues)
    
    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]


def validate_setup(
    workspace_url: Optional[str] = None,
    check_connectivity: bool = True,
    check_governance: bool = True,
    **kwargs
) -> ValidationResult:
    """
    Validate Databricks Unity Catalog setup for GenOps governance.

    Args:
        workspace_url: Databricks workspace URL to validate
        check_connectivity: Whether to check connectivity to Databricks
        check_governance: Whether to validate governance features
        **kwargs: Additional validation parameters

    Returns:
        ValidationResult with detailed validation information
    """
    result = ValidationResult()
    result.passed = True  # Start optimistic
    
    logger.info("Starting Databricks Unity Catalog validation...")
    
    # 1. Validate dependencies
    _validate_dependencies(result)
    
    # 2. Validate configuration
    _validate_configuration(result, workspace_url)
    
    # 3. Check connectivity (if requested and configuration is valid)
    if check_connectivity and not result.has_errors():
        _validate_connectivity(result)
    
    # 4. Validate governance features (if requested)
    if check_governance and not result.has_errors():
        _validate_governance_features(result)
    
    # Final status
    if result.has_errors():
        result.passed = False
        logger.warning(f"Databricks Unity Catalog validation failed with {len(result.get_issues_by_severity('error'))} errors")
    else:
        logger.info("Databricks Unity Catalog validation passed")
    
    return result


def _validate_dependencies(result: ValidationResult) -> None:
    """Validate required dependencies."""
    dependencies = {
        "databricks": False,
        "databricks.sdk": False,
        "pyspark": False,
        "opentelemetry": False,
        "genops": False,
    }
    
    # Check databricks-sdk
    try:
        import databricks.sdk
        dependencies["databricks"] = True
        dependencies["databricks.sdk"] = True
        result.configuration["databricks_sdk_version"] = getattr(databricks, "__version__", "unknown")
    except ImportError:
        result.add_issue(ValidationIssue(
            severity="error",
            component="dependencies",
            message="databricks-sdk not installed",
            suggested_fix="pip install databricks-sdk",
            documentation_link="https://databricks-sdk-py.readthedocs.io/"
        ))
    
    # Check PySpark (optional but recommended)
    try:
        import pyspark
        dependencies["pyspark"] = True
        result.configuration["pyspark_version"] = pyspark.__version__
    except ImportError:
        result.add_issue(ValidationIssue(
            severity="warning",
            component="dependencies",
            message="PySpark not installed - some features may be limited",
            suggested_fix="pip install pyspark",
            documentation_link="https://spark.apache.org/docs/latest/api/python/"
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


def _validate_configuration(result: ValidationResult, workspace_url: Optional[str] = None) -> None:
    """Validate Databricks configuration."""
    
    # Check workspace URL
    workspace_url = workspace_url or os.getenv("DATABRICKS_HOST")
    if not workspace_url:
        result.add_issue(ValidationIssue(
            severity="error",
            component="configuration",
            message="Databricks workspace URL not configured",
            suggested_fix="Set DATABRICKS_HOST environment variable or pass workspace_url parameter",
            documentation_link="https://docs.databricks.com/dev-tools/auth.html"
        ))
    else:
        result.configuration["workspace_url"] = workspace_url
        
        # Validate URL format
        if not workspace_url.startswith(("https://", "http://")):
            result.add_issue(ValidationIssue(
                severity="error",
                component="configuration",
                message=f"Invalid workspace URL format: {workspace_url}",
                suggested_fix="Workspace URL should start with https:// (e.g., https://your-workspace.cloud.databricks.com)",
            ))
    
    # Check access token
    access_token = os.getenv("DATABRICKS_TOKEN")
    if not access_token:
        result.add_issue(ValidationIssue(
            severity="error",
            component="configuration",
            message="Databricks access token not configured",
            suggested_fix="Set DATABRICKS_TOKEN environment variable",
            documentation_link="https://docs.databricks.com/dev-tools/auth.html#databricks-personal-access-tokens"
        ))
    else:
        result.configuration["access_token"] = "***configured***"
        
        # Basic token validation
        if len(access_token) < 10:
            result.add_issue(ValidationIssue(
                severity="warning",
                component="configuration",
                message="Access token appears to be invalid (too short)",
                suggested_fix="Verify your Databricks personal access token",
            ))
    
    # Check GenOps configuration
    genops_config = {
        "team": os.getenv("GENOPS_TEAM"),
        "project": os.getenv("GENOPS_PROJECT"),
        "environment": os.getenv("GENOPS_ENVIRONMENT"),
    }
    
    missing_config = [key for key, value in genops_config.items() if not value]
    if missing_config:
        result.add_issue(ValidationIssue(
            severity="warning",
            component="configuration",
            message=f"GenOps governance attributes not set: {missing_config}",
            suggested_fix=f"Set environment variables: {', '.join(f'GENOPS_{k.upper()}' for k in missing_config)}",
        ))
    
    # Add configured values
    for key, value in genops_config.items():
        if value:
            result.configuration[f"genops_{key}"] = value


def _validate_connectivity(result: ValidationResult) -> None:
    """Validate connectivity to Databricks workspace."""
    connectivity_checks = {
        "workspace_api": False,
        "unity_catalog_api": False,
        "sql_warehouses": False,
    }
    
    try:
        # Import Databricks SDK
        from databricks.sdk import WorkspaceClient
        
        # Create client
        workspace_url = result.configuration.get("workspace_url")
        if workspace_url:
            client = WorkspaceClient(host=workspace_url)
            
            # Test workspace API
            try:
                current_user = client.current_user.me()
                connectivity_checks["workspace_api"] = True
                result.configuration["authenticated_user"] = current_user.user_name or "unknown"
            except Exception as e:
                result.add_issue(ValidationIssue(
                    severity="error",
                    component="connectivity",
                    message=f"Cannot connect to Databricks workspace API: {e}",
                    suggested_fix="Verify workspace URL and access token are correct",
                ))
            
            # Test Unity Catalog API
            try:
                catalogs = list(client.catalogs.list())
                connectivity_checks["unity_catalog_api"] = True
                result.configuration["available_catalogs"] = len(catalogs)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    severity="warning",
                    component="connectivity", 
                    message=f"Unity Catalog API not accessible: {e}",
                    suggested_fix="Ensure Unity Catalog is enabled in your workspace",
                ))
            
            # Test SQL Warehouses
            try:
                warehouses = list(client.warehouses.list())
                connectivity_checks["sql_warehouses"] = True
                result.configuration["available_warehouses"] = len(warehouses)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    severity="info",
                    component="connectivity",
                    message=f"SQL Warehouses API not accessible: {e}",
                    suggested_fix="SQL Warehouses may not be available or configured",
                ))
    
    except ImportError:
        result.add_issue(ValidationIssue(
            severity="error",
            component="connectivity",
            message="Cannot test connectivity - databricks-sdk not available",
            suggested_fix="Install databricks-sdk to enable connectivity testing",
        ))
    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="error",
            component="connectivity",
            message=f"Connectivity test failed: {e}",
            suggested_fix="Check your network connection and credentials",
        ))
    
    result.connectivity = connectivity_checks


def _validate_governance_features(result: ValidationResult) -> None:
    """Validate Unity Catalog governance features."""
    
    try:
        # Test GenOps telemetry
        from genops.core.telemetry import GenOpsTelemetry
        telemetry = GenOpsTelemetry()
        result.configuration["telemetry_enabled"] = "true"
        
        # Test OpenTelemetry integration
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("genops.validation.test") as span:
            span.set_attribute("genops.provider", "databricks_unity_catalog")
            span.set_attribute("genops.validation", "governance_features")
        result.configuration["opentelemetry_integration"] = "working"
        
        # Test provider components
        try:
            from . import adapter, cost_aggregator, governance_monitor
            result.configuration["provider_components"] = "loaded"
        except ImportError as e:
            result.add_issue(ValidationIssue(
                severity="error",
                component="governance",
                message=f"Provider components not available: {e}",
                suggested_fix="Ensure GenOps Databricks provider is properly installed",
            ))
    
    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="warning",
            component="governance",
            message=f"Governance feature validation failed: {e}",
            suggested_fix="Check GenOps installation and configuration",
        ))


def print_validation_result(result: ValidationResult) -> None:
    """Print formatted validation result."""
    
    print("\n" + "="*60)
    print("DATABRICKS UNITY CATALOG GENOPS VALIDATION REPORT")
    print("="*60)
    
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
    for key, value in result.configuration.items():
        print(f"  • {key}: {value}")
    
    # Connectivity
    if result.connectivity:
        print(f"\n🌐 Connectivity:")
        for check, status in result.connectivity.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check}")
    
    # Issues
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
        print(f"\n🎉 SUCCESS! You're ready to use Databricks Unity Catalog with GenOps.")
        print(f"   Try running: python basic_tracking.py")
    else:
        print(f"\n🔧 Please fix the errors above and run validation again.")
        print(f"   Command: python setup_validation.py")
    
    print("\n" + "="*60)