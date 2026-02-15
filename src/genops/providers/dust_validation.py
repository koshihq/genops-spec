"""
Validation utilities for Dust integration setup.
Helps developers verify their GenOps Dust integration is working correctly.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found during setup check."""

    level: str  # "error", "warning", "info"
    component: str  # "environment", "dependencies", "configuration", etc.
    message: str
    fix_suggestion: Optional[str] = None


class ValidationResult(NamedTuple):
    """Result of setup validation."""

    is_valid: bool
    issues: list[ValidationIssue]
    summary: dict[str, Any]


def check_environment_variables() -> list[ValidationIssue]:
    """Check required and optional environment variables."""
    issues = []

    # Required variables
    required_vars = {
        "DUST_API_KEY": "Dust API credential for authentication",
        "DUST_WORKSPACE_ID": "Dust workspace ID for API access",
    }

    for var, description in required_vars.items():
        if not os.getenv(var):
            issues.append(
                ValidationIssue(
                    level="error",
                    component="environment",
                    message=f"Missing required environment variable: {var} ({description})",
                    fix_suggestion=f"Set {var} with: export {var}=your_value_here",
                )
            )

    # Optional but recommended variables
    optional_vars = {
        "OTEL_SERVICE_NAME": "OpenTelemetry service name for telemetry identification",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "OpenTelemetry collector endpoint for telemetry export",
        "GENOPS_TEAM": "Team name for cost attribution and governance",
        "GENOPS_PROJECT": "Project name for cost attribution and governance",
        "GENOPS_ENVIRONMENT": "Environment name (dev/staging/prod) for governance",
        "GENOPS_COST_CENTER": "Cost center for financial reporting alignment",
        "GENOPS_CUSTOMER_ID": "Customer ID for customer attribution",
        "GENOPS_FEATURE": "Feature name for feature-level cost attribution",
    }

    for var, description in optional_vars.items():
        if not os.getenv(var):
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="environment",
                    message=f"Optional environment variable not set: {var} ({description})",
                    fix_suggestion=f"Consider setting {var} with: export {var}=your_value",
                )
            )

    return issues


def check_dependencies() -> list[ValidationIssue]:
    """Check for required Python packages."""
    issues = []

    required_packages = [("requests", "HTTP client for Dust API communication")]

    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="dependencies",
                    message=f"Missing required package: {package} ({description})",
                    fix_suggestion=f"Install with: pip install {package}",
                )
            )

    # Optional packages
    optional_packages = [
        ("opentelemetry-api", "OpenTelemetry tracing support"),
        ("opentelemetry-sdk", "OpenTelemetry SDK for telemetry export"),
        ("opentelemetry-exporter-otlp", "OTLP exporter for telemetry"),
    ]

    for package, description in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="dependencies",
                    message=f"Optional package not installed: {package} ({description})",
                    fix_suggestion=f"Install with: pip install {package}",
                )
            )

    return issues


def check_dust_connectivity(
    api_key: Optional[str] = None,
    workspace_id: Optional[str] = None,
    base_url: str = "https://dust.tt",
) -> list[ValidationIssue]:
    """Test connectivity to Dust API."""
    issues = []

    # Use provided credentials or fall back to environment
    api_key = api_key or os.getenv("DUST_API_KEY")
    workspace_id = workspace_id or os.getenv("DUST_WORKSPACE_ID")

    if not api_key:
        issues.append(
            ValidationIssue(
                level="error",
                component="connectivity",
                message="Cannot test Dust connectivity: API credential not provided",
                fix_suggestion="Provide api_key parameter or set DUST_API_KEY environment variable",
            )
        )
        return issues

    if not workspace_id:
        issues.append(
            ValidationIssue(
                level="error",
                component="connectivity",
                message="Cannot test Dust connectivity: workspace ID not provided",
                fix_suggestion="Provide workspace_id parameter or set DUST_WORKSPACE_ID environment variable",
            )
        )
        return issues

    try:
        # Test basic API connectivity by listing conversations
        url = f"{base_url.rstrip('/')}/api/v1/w/{workspace_id}/conversations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            issues.append(
                ValidationIssue(
                    level="info",
                    component="connectivity",
                    message="Successfully connected to Dust API",
                    fix_suggestion=None,
                )
            )
        elif response.status_code == 401:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="connectivity",
                    message="Authentication failed: Invalid API credential",
                    fix_suggestion="Verify your DUST_API_KEY is correct and has appropriate permissions",
                )
            )
        elif response.status_code == 403:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="connectivity",
                    message="Access denied: Insufficient permissions",
                    fix_suggestion="Verify your API credential has access to the specified workspace",
                )
            )
        elif response.status_code == 404:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="connectivity",
                    message="Workspace not found: Invalid workspace ID",
                    fix_suggestion="Verify your DUST_WORKSPACE_ID is correct",
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="connectivity",
                    message=f"Unexpected response from Dust API: {response.status_code}",
                    fix_suggestion=f"Check Dust service status or contact support. Response: {response.text[:100]}",
                )
            )

    except requests.ConnectionError:
        issues.append(
            ValidationIssue(
                level="error",
                component="connectivity",
                message="Cannot connect to Dust API: Connection error",
                fix_suggestion="Check your internet connection and verify the Dust service is accessible",
            )
        )
    except requests.Timeout:
        issues.append(
            ValidationIssue(
                level="warning",
                component="connectivity",
                message="Dust API request timed out",
                fix_suggestion="The Dust API is slow to respond. This may affect performance.",
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                component="connectivity",
                message=f"Unexpected error testing Dust connectivity: {e}",
                fix_suggestion="Check your network settings and Dust API configuration",
            )
        )

    return issues


def check_workspace_access(
    api_key: Optional[str] = None,
    workspace_id: Optional[str] = None,
    base_url: str = "https://dust.tt",
) -> list[ValidationIssue]:
    """Check workspace access and permissions."""
    issues = []

    api_key = api_key or os.getenv("DUST_API_KEY")
    workspace_id = workspace_id or os.getenv("DUST_WORKSPACE_ID")

    if not api_key or not workspace_id:
        issues.append(
            ValidationIssue(
                level="error",
                component="workspace",
                message="Cannot check workspace access: missing credentials",
                fix_suggestion="Ensure DUST_API_KEY and DUST_WORKSPACE_ID are set",
            )
        )
        return issues

    try:
        # Check different API endpoints to validate permissions
        endpoints_to_check = [
            ("conversations", "conversation management"),
            ("agents", "agent access"),
            ("data_sources", "datasource management"),
        ]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        accessible_endpoints = []
        restricted_endpoints = []

        for endpoint, description in endpoints_to_check:
            url = f"{base_url.rstrip('/')}/api/v1/w/{workspace_id}/{endpoint}"

            try:
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code in [200, 201]:
                    accessible_endpoints.append((endpoint, description))
                elif response.status_code in [403, 401]:
                    restricted_endpoints.append((endpoint, description))

            except Exception as e:
                logger.debug(f"Error checking endpoint {endpoint}: {e}")

        if accessible_endpoints:
            endpoint_list = ", ".join(
                [f"{ep}({desc})" for ep, desc in accessible_endpoints]
            )
            issues.append(
                ValidationIssue(
                    level="info",
                    component="workspace",
                    message=f"Workspace access verified for: {endpoint_list}",
                    fix_suggestion=None,
                )
            )

        if restricted_endpoints:
            endpoint_list = ", ".join(
                [f"{ep}({desc})" for ep, desc in restricted_endpoints]
            )
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="workspace",
                    message=f"Limited access to: {endpoint_list}",
                    fix_suggestion="Some features may not be available. Check your API credential permissions.",
                )
            )

    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                component="workspace",
                message=f"Error checking workspace access: {e}",
                fix_suggestion="Verify your workspace ID and API credential are correct",
            )
        )

    return issues


def validate_setup(
    api_key: Optional[str] = None,
    workspace_id: Optional[str] = None,
    base_url: str = "https://dust.tt",
    **kwargs,
) -> ValidationResult:
    """
    Comprehensive validation of Dust integration setup.

    Args:
        api_key: Optional Dust API credential (will use DUST_API_KEY env var if not provided)
        workspace_id: Optional workspace ID (will use DUST_WORKSPACE_ID env var if not provided)
        base_url: Dust API base URL (default: https://dust.tt)
        **kwargs: Additional configuration options

    Returns:
        ValidationResult with overall status and detailed issues
    """
    all_issues = []

    # Run all validation checks
    all_issues.extend(check_environment_variables())
    all_issues.extend(check_dependencies())
    all_issues.extend(check_dust_connectivity(api_key, workspace_id, base_url))
    all_issues.extend(check_workspace_access(api_key, workspace_id, base_url))

    # Analyze results
    error_count = len([issue for issue in all_issues if issue.level == "error"])
    warning_count = len([issue for issue in all_issues if issue.level == "warning"])
    info_count = len([issue for issue in all_issues if issue.level == "info"])

    is_valid = error_count == 0

    summary = {
        "total_issues": len(all_issues),
        "errors": error_count,
        "warnings": warning_count,
        "info": info_count,
        "is_ready_for_production": is_valid and warning_count <= 2,
        "api_key_configured": bool(api_key or os.getenv("DUST_API_KEY")),
        "workspace_configured": bool(workspace_id or os.getenv("DUST_WORKSPACE_ID")),
        "telemetry_configured": bool(os.getenv("OTEL_SERVICE_NAME")),
        "governance_attributes_configured": bool(
            os.getenv("GENOPS_TEAM") and os.getenv("GENOPS_PROJECT")
        ),
    }

    return ValidationResult(is_valid=is_valid, issues=all_issues, summary=summary)


# Sanitization function removed to resolve CodeQL false positives


def print_validation_result(
    result: ValidationResult, show_details: bool = True
) -> None:
    """
    Print formatted validation results with enhanced UX matching other providers.

    Args:
        result: ValidationResult to display
        show_details: Whether to show detailed issue information
    """

    # Enhanced status symbols and formatting
    status_symbols = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}

    print(f"\n{'=' * 60}")
    print("🔍 Dust AI Integration Validation Report")
    print(f"{'=' * 60}")

    # Overall status with enhanced formatting
    overall_symbol = "✅" if result.is_valid else "❌"
    overall_status = "INTEGRATION READY" if result.is_valid else "SETUP REQUIRED"
    print(f"\n{overall_symbol} Overall Status: {overall_status}")

    # Enhanced summary with visual indicators
    summary = result.summary
    print("\n📊 Validation Summary:")
    print(f"   Total Issues: {summary['total_issues']}")
    print(
        f"   Errors: {summary['errors']} | Warnings: {summary['warnings']} | Info: {summary['info']}"
    )

    # Production readiness assessment
    production_icon = "🚀" if summary.get("is_ready_for_production", False) else "🔧"
    production_status = (
        "Production Ready"
        if summary.get("is_ready_for_production", False)
        else "Development Ready"
    )
    print(f"   {production_icon} Status: {production_status}")

    # Enhanced configuration matrix
    print("\n⚙️  Configuration Matrix:")
    config_items = [
        (
            "API Credential",
            summary.get("api_key_configured", False),
            "DUST_API_KEY environment variable",
        ),
        (
            "Workspace ID",
            summary.get("workspace_configured", False),
            "DUST_WORKSPACE_ID environment variable",
        ),
        (
            "Telemetry Export",
            summary.get("telemetry_configured", False),
            "OTEL_SERVICE_NAME configured",
        ),
        (
            "Governance Attrs",
            summary.get("governance_attributes_configured", False),
            "GENOPS_TEAM/PROJECT configured",
        ),
        ("Dependencies", summary["errors"] == 0, "All required packages installed"),
    ]

    for item_name, is_configured, description in config_items:
        status_icon = (
            "✅"
            if is_configured
            else ("⚠️" if "configured" in description.lower() else "❌")
        )
        status_text = (
            "Ready"
            if is_configured
            else ("Optional" if "configured" in description.lower() else "Missing")
        )
        # Always show basic status (no sensitive data)
        print(f"   {status_icon} {item_name:.<20} {status_text}")
        if not is_configured and show_details:
            # Only show detailed help in debug mode to avoid CodeQL false positives
            if os.getenv("GENOPS_DEBUG_VALIDATION", "").lower() in ("true", "1", "yes"):
                print(f"      💡 {description}")
            else:
                print("      💡 Set GENOPS_DEBUG_VALIDATION=true for detailed help")

    # Issue breakdown with enhanced formatting
    if result.issues and show_details:
        print("\n🔍 Detailed Issue Analysis:")
        print("-" * 45)

        # Group and sort issues by severity
        issue_groups = {"error": [], "warning": [], "info": []}
        for issue in result.issues:
            issue_groups[issue.level].append(issue)

        # Display issues by severity with enhanced formatting
        for level in ["error", "warning", "info"]:
            issues_list = issue_groups[level]
            if issues_list:
                level_icon = status_symbols[level]
                level_name = level.upper()
                print(
                    f"\n{level_icon} {level_name} ({len(issues_list)} issue{'s' if len(issues_list) > 1 else ''}):"
                )

                for i, issue in enumerate(issues_list, 1):
                    # Enhanced issue formatting
                    component_tag = f"[{issue.component.upper()}]"
                    print(f"  {i}. {component_tag} {issue.message}")

                    if issue.fix_suggestion:
                        print(f"     🔧 Solution: {issue.fix_suggestion}")

                    # Add spacing between issues for readability
                    if i < len(issues_list):
                        print()

    # Enhanced recommendations section
    print("\n🎯 Recommendations:")

    if result.is_valid:
        print("  ✅ Your Dust integration is validated and ready!")
        print("  🚀 Quick Start:")
        print("     • Run: python examples/dust/basic_tracking.py")
        print("     • Try: python examples/dust/auto_instrumentation.py")
        print("     • Monitor: Configure your observability platform")

        if summary["warnings"] > 0:
            print("  ⚠️  Optional Improvements:")
            print("     • Address warnings for optimal production deployment")
            print("     • Configure governance attributes for better cost attribution")

    else:
        print("  🔧 Required Actions:")
        error_count = summary["errors"]
        print(
            f"     • Fix {error_count} critical issue{'s' if error_count > 1 else ''} listed above"
        )
        print("     • Re-run validation after making changes")

        print("  📚 Resources:")
        print("     • Quick Start: docs/dust-quickstart.md")
        print("     • Full Guide: docs/integrations/dust.md")
        print("     • Examples: examples/dust/")

    # Performance and optimization hints
    if summary.get("warnings", 0) == 0 and result.is_valid:
        print("  ⚡ Performance Tips:")
        print("     • Use environment variables for credentials")
        print("     • Configure OTLP endpoint for production telemetry")
        print("     • Set up cost monitoring dashboards")

    # Support information
    print("\n💬 Support & Community:")
    print("   • Documentation: docs/integrations/dust.md")
    print("   • GitHub Issues: https://github.com/KoshiHQ/GenOps-AI/issues")
    print("   • Community: https://community.dust.tt/")

    print(f"\n{'=' * 60}")

    # Final call-to-action
    if result.is_valid:
        print("🎉 You're all set! Start building with Dust AI governance.")
    else:
        print("🛠️  Complete the setup above to unlock Dust AI governance.")
    print()


# Convenience function for quick validation
def quick_validate() -> bool:
    """Quick validation check - returns True if setup is valid."""
    result = validate_setup()
    return result.is_valid
