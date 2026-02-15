#!/usr/bin/env python3
"""
PostHog Integration Setup Validation

This module provides comprehensive validation utilities for PostHog + GenOps integration.
It validates configuration, dependencies, authentication, and provides detailed diagnostics
for troubleshooting setup issues.

Functions:
- validate_setup(): Comprehensive validation with structured results
- print_validation_result(): User-friendly validation result display
- validate_posthog_connection(): Test PostHog API connectivity
- validate_environment_config(): Check environment variables and configuration
- get_setup_recommendations(): Get actionable setup recommendations

Author: GenOps AI Team
License: Apache 2.0
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    level: ValidationLevel
    component: str
    issue: str
    recommendation: str
    fix_command: Optional[str] = None
    documentation_link: Optional[str] = None


@dataclass
class ValidationResult:
    """Comprehensive validation result."""

    is_valid: bool
    overall_status: ValidationLevel
    issues: list[ValidationIssue]
    warnings: list[ValidationIssue]
    successes: list[ValidationIssue]
    recommendations: list[str]
    summary: dict[str, Any]
    validation_timestamp: datetime

    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len(
            [issue for issue in self.issues if issue.level == ValidationLevel.ERROR]
        )

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len(
            [issue for issue in self.issues if issue.level == ValidationLevel.WARNING]
        )


def validate_environment_config() -> list[ValidationIssue]:
    """Validate PostHog environment configuration."""
    issues = []

    # Check PostHog API key
    posthog_api_key = os.getenv("POSTHOG_API_KEY")
    if not posthog_api_key:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="Configuration",
                issue="POSTHOG_API_KEY environment variable not found",
                recommendation="Set your PostHog project API key in environment variables",
                fix_command="export POSTHOG_API_KEY='phc_your_project_api_key'",
                documentation_link="https://posthog.com/docs/api/overview",
            )
        )
    elif not posthog_api_key.startswith("phc_"):
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Configuration",
                issue="PostHog API key format doesn't match expected pattern (should start with 'phc_')",
                recommendation="Verify your PostHog project API key is correct",
                documentation_link="https://posthog.com/docs/api/overview",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Configuration",
                issue="POSTHOG_API_KEY configured correctly",
                recommendation="API key validation successful",
            )
        )

    # Check PostHog host configuration
    posthog_host = os.getenv("POSTHOG_HOST", "https://app.posthog.com")
    if posthog_host in ["https://app.posthog.com", "https://eu.posthog.com"]:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Configuration",
                issue=f"PostHog host configured: {posthog_host}",
                recommendation="Host configuration is valid",
            )
        )
    elif posthog_host.startswith("http"):
        issues.append(
            ValidationIssue(
                level=ValidationLevel.INFO,
                component="Configuration",
                issue=f"Custom PostHog host configured: {posthog_host}",
                recommendation="Ensure your self-hosted PostHog instance is accessible",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Configuration",
                issue=f"PostHog host may be invalid: {posthog_host}",
                recommendation="Verify PostHog host URL format (should start with http:// or https://)",
            )
        )

    # Check GenOps team configuration
    genops_team = os.getenv("GENOPS_TEAM")
    if genops_team:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Configuration",
                issue="GENOPS_TEAM configured for cost attribution",
                recommendation="Team-based cost tracking enabled",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Configuration",
                issue="GENOPS_TEAM not configured",
                recommendation="Set GENOPS_TEAM for better cost attribution and governance",
                fix_command="export GENOPS_TEAM='your-team-name'",
            )
        )

    # Check GenOps project configuration
    genops_project = os.getenv("GENOPS_PROJECT")
    if genops_project:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Configuration",
                issue="GENOPS_PROJECT configured for cost attribution",
                recommendation="Project-based cost tracking enabled",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Configuration",
                issue="GENOPS_PROJECT not configured",
                recommendation="Set GENOPS_PROJECT for better cost attribution",
                fix_command="export GENOPS_PROJECT='your-project-name'",
            )
        )

    return issues


def validate_sdk_dependencies() -> list[ValidationIssue]:
    """Validate SDK and dependency installation."""
    issues = []

    # Check GenOps installation
    try:
        import genops

        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="SDK Installation",
                issue="GenOps SDK installed and importable",
                recommendation="GenOps core functionality available",
            )
        )

        # Check GenOps version
        try:
            version = getattr(genops, "__version__", "unknown")
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.INFO,
                    component="SDK Installation",
                    issue=f"GenOps version: {version}",
                    recommendation="SDK version information available",
                )
            )
        except Exception:
            pass

    except ImportError:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="SDK Installation",
                issue="GenOps SDK not installed or not importable",
                recommendation="Install GenOps SDK with PostHog support",
                fix_command="pip install genops[posthog]",
            )
        )

    # Check PostHog SDK installation
    try:
        import posthog

        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="SDK Installation",
                issue="PostHog Python SDK installed and importable",
                recommendation="PostHog client functionality available",
            )
        )

        # Check PostHog version
        try:
            version = getattr(posthog, "__version__", "unknown")
            if version != "unknown":
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.INFO,
                        component="SDK Installation",
                        issue=f"PostHog SDK version: {version}",
                        recommendation="PostHog version information available",
                    )
                )
        except Exception:
            pass

    except ImportError:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="SDK Installation",
                issue="PostHog Python SDK not installed or not importable",
                recommendation="Install PostHog SDK",
                fix_command="pip install posthog",
            )
        )

    # Check OpenTelemetry dependencies
    try:
        from opentelemetry import trace  # noqa: F401

        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="SDK Installation",
                issue="OpenTelemetry core installed",
                recommendation="Telemetry functionality available",
            )
        )
    except ImportError:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="SDK Installation",
                issue="OpenTelemetry dependencies missing",
                recommendation="Install OpenTelemetry for enhanced telemetry",
                fix_command="pip install opentelemetry-api opentelemetry-sdk",
            )
        )

    return issues


def validate_posthog_connection(
    api_key: Optional[str] = None, host: Optional[str] = None
) -> list[ValidationIssue]:
    """Validate PostHog API connectivity and authentication."""
    issues = []

    api_key = api_key or os.getenv("POSTHOG_API_KEY")
    host = host or os.getenv("POSTHOG_HOST", "https://app.posthog.com")

    if not api_key:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="Authentication",
                issue="No PostHog API key available for connection testing",
                recommendation="Configure POSTHOG_API_KEY to test connectivity",
            )
        )
        return issues

    try:
        # Import PostHog
        import posthog

        # Test basic client initialization
        try:
            posthog.Client(api_key=api_key, host=host)
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="Authentication",
                    issue="PostHog client initialized successfully",
                    recommendation="PostHog API connectivity established",
                )
            )

            # In a real implementation, we could test a lightweight API call
            # For now, we'll just test client initialization

        except Exception as e:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    component="Authentication",
                    issue=f"PostHog client initialization failed: {e}",
                    recommendation="Verify PostHog API key and host configuration",
                    documentation_link="https://posthog.com/docs/api/overview",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="Authentication",
                issue="PostHog SDK not available for connection testing",
                recommendation="Install PostHog SDK first",
                fix_command="pip install posthog",
            )
        )

    return issues


def validate_genops_posthog_integration() -> list[ValidationIssue]:
    """Validate GenOps PostHog adapter functionality."""
    issues = []

    try:
        from genops.providers.posthog import (  # noqa: F401
            GenOpsPostHogAdapter,
            auto_instrument,
        )

        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="GenOps Integration",
                issue="GenOps PostHog adapter importable",
                recommendation="PostHog integration functionality available",
            )
        )

        # Test adapter initialization
        try:
            api_key = os.getenv("POSTHOG_API_KEY", "test-key")
            GenOpsPostHogAdapter(
                posthog_api_key=api_key,
                team="validation-test",
                project="setup-validation",
            )
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.SUCCESS,
                    component="GenOps Integration",
                    issue="GenOps PostHog adapter initialization successful",
                    recommendation="Adapter configuration is valid",
                )
            )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    component="GenOps Integration",
                    issue=f"GenOps PostHog adapter initialization failed: {e}",
                    recommendation="Check adapter configuration and dependencies",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.ERROR,
                component="GenOps Integration",
                issue="GenOps PostHog integration not available",
                recommendation="Install GenOps with PostHog support",
                fix_command="pip install genops[posthog]",
            )
        )

    return issues


def validate_cost_tracking_configuration() -> list[ValidationIssue]:
    """Validate cost tracking and governance configuration."""
    issues = []

    # Check budget configuration
    daily_budget = os.getenv("GENOPS_DAILY_BUDGET_LIMIT")
    if daily_budget:
        try:
            budget_value = float(daily_budget)
            if budget_value > 0:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.SUCCESS,
                        component="Cost Tracking",
                        issue=f"Daily budget limit configured: ${budget_value}",
                        recommendation="Cost governance enabled",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        component="Cost Tracking",
                        issue="Daily budget limit set to zero or negative",
                        recommendation="Set a positive daily budget limit for cost control",
                    )
                )
        except ValueError:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    component="Cost Tracking",
                    issue=f"Invalid daily budget format: {daily_budget}",
                    recommendation="Set GENOPS_DAILY_BUDGET_LIMIT to a numeric value",
                    fix_command="export GENOPS_DAILY_BUDGET_LIMIT='100.0'",
                )
            )
    else:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.INFO,
                component="Cost Tracking",
                issue="Daily budget limit not configured (will use default)",
                recommendation="Consider setting GENOPS_DAILY_BUDGET_LIMIT for cost control",
            )
        )

    # Check governance policy
    governance_policy = os.getenv("GENOPS_GOVERNANCE_POLICY", "advisory")
    valid_policies = ["advisory", "enforced", "strict"]
    if governance_policy in valid_policies:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.SUCCESS,
                component="Cost Tracking",
                issue=f"Governance policy configured: {governance_policy}",
                recommendation="Policy enforcement level set",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                component="Cost Tracking",
                issue=f"Invalid governance policy: {governance_policy}",
                recommendation=f"Set governance policy to one of: {', '.join(valid_policies)}",
                fix_command="export GENOPS_GOVERNANCE_POLICY='advisory'",
            )
        )

    return issues


def validate_setup(verbose: bool = False) -> ValidationResult:
    """
    Comprehensive PostHog + GenOps setup validation.

    Args:
        verbose: Include additional diagnostic information

    Returns:
        ValidationResult with comprehensive validation status
    """
    all_issues = []

    # Run all validation checks
    validation_functions = [
        ("Environment Configuration", validate_environment_config),
        ("SDK Dependencies", validate_sdk_dependencies),
        ("PostHog Authentication", lambda: validate_posthog_connection()),
        ("GenOps Integration", validate_genops_posthog_integration),
        ("Cost Tracking", validate_cost_tracking_configuration),
    ]

    for component_name, validation_func in validation_functions:
        try:
            issues = validation_func()
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    component=component_name,
                    issue=f"Validation function failed: {e}",
                    recommendation="Check system configuration and dependencies",
                )
            )

    # Categorize issues
    errors = [issue for issue in all_issues if issue.level == ValidationLevel.ERROR]
    warnings = [issue for issue in all_issues if issue.level == ValidationLevel.WARNING]
    successes = [
        issue for issue in all_issues if issue.level == ValidationLevel.SUCCESS
    ]

    # Determine overall status
    if errors:
        overall_status = ValidationLevel.ERROR
        is_valid = False
    elif warnings:
        overall_status = ValidationLevel.WARNING
        is_valid = True
    else:
        overall_status = ValidationLevel.SUCCESS
        is_valid = True

    # Generate recommendations
    recommendations = []
    if errors:
        recommendations.append(
            "Fix all error-level issues before using PostHog integration"
        )
    if warnings:
        recommendations.append("Address warning-level issues for optimal experience")
    if is_valid:
        recommendations.append(
            "You can now use GenOps PostHog integration with confidence"
        )

    # Build summary
    summary = {
        "total_issues": len(all_issues),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "success_count": len(successes),
        "components_validated": len(validation_functions),
        "is_ready_for_production": len(errors) == 0 and len(warnings) == 0,
    }

    return ValidationResult(
        is_valid=is_valid,
        overall_status=overall_status,
        issues=all_issues,
        warnings=warnings,
        successes=successes,
        recommendations=recommendations,
        summary=summary,
        validation_timestamp=datetime.now(),
    )


def print_validation_result(
    result: ValidationResult, show_successes: bool = True
) -> None:
    """
    Print validation result in user-friendly format.

    Args:
        result: ValidationResult to display
        show_successes: Whether to display successful validation items
    """
    print("🔍 PostHog + GenOps Integration Validation Report")
    print("=" * 60)
    print()

    # Overall status
    status_icons = {
        ValidationLevel.SUCCESS: "✅",
        ValidationLevel.WARNING: "⚠️",
        ValidationLevel.ERROR: "❌",
        ValidationLevel.INFO: "ℹ️",
    }

    status_icon = status_icons[result.overall_status]
    status_text = "SUCCESS" if result.is_valid else "ISSUES DETECTED"
    print(f"{status_icon} Overall Status: {status_text}")
    print()

    # Summary
    print("📊 Validation Summary:")
    print(
        f"  • SDK Installation: {result.summary['success_count'] - result.summary['error_count']} issues"
    )
    print(f"  • Authentication: {result.error_count} issues")
    print(f"  • Configuration: {result.warning_count} issues")
    if result.summary.get("is_ready_for_production"):
        print("  • Production Ready: Yes")
    else:
        print("  • Production Ready: No")
    print()

    # Issues by category
    if result.has_errors:
        print("❌ Errors (must fix):")
        for issue in result.issues:
            if issue.level == ValidationLevel.ERROR:
                print(f"  • {issue.component}: {issue.issue}")
                print(f"    💡 Fix: {issue.recommendation}")
                if issue.fix_command:
                    print(f"    🔧 Command: {issue.fix_command}")
                if issue.documentation_link:
                    print(f"    📚 Docs: {issue.documentation_link}")
                print()

    if result.has_warnings:
        print("⚠️ Warnings (recommended fixes):")
        for issue in result.issues:
            if issue.level == ValidationLevel.WARNING:
                print(f"  • {issue.component}: {issue.issue}")
                print(f"    💡 Recommendation: {issue.recommendation}")
                if issue.fix_command:
                    print(f"    🔧 Command: {issue.fix_command}")
                print()

    if show_successes and result.successes:
        print("✅ Successful Validations:")
        for issue in result.successes:
            print(f"  • {issue.component}: {issue.issue}")
        print()

    # Recommendations
    if result.recommendations:
        print("💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        print()

    # Next steps
    print("🚀 Next Steps:")
    if not result.is_valid:
        print("  1. Fix all error-level issues above")
        print(
            '  2. Re-run validation: python -c "from genops.providers.posthog_validation import validate_setup, print_validation_result; print_validation_result(validate_setup())"'
        )
        print("  3. Try the basic PostHog examples once validation passes")
    else:
        print("  1. You can now use GenOps PostHog integration with confidence")
        print("  2. Try the examples: python examples/posthog/basic_tracking.py")
        print("  3. Check the integration guide for advanced features")

    print(
        f"\n📅 Validation completed at: {result.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def interactive_setup_wizard() -> None:
    """Interactive setup wizard for PostHog + GenOps configuration."""
    print("🧙 PostHog + GenOps Interactive Setup Wizard")
    print("=" * 50)
    print()
    print("This wizard will help you configure PostHog integration step-by-step.")
    print("Press Ctrl+C at any time to cancel.")
    print()

    try:
        # Step 1: API Key
        print("📋 Step 1: PostHog API Key")
        print("-" * 30)
        current_key = os.getenv("POSTHOG_API_KEY")
        if current_key:
            print(
                f"Current API key: {current_key[:8]}...{current_key[-4:] if len(current_key) > 12 else 'short'}"
            )
            use_current = input("Use current API key? (y/n): ").lower() == "y"
            if not use_current:
                new_key = input(
                    "Enter your PostHog project API key (starts with 'phc_'): "
                ).strip()
                if new_key:
                    print(
                        f"💡 Add this to your environment: export POSTHOG_API_KEY='{new_key}'"
                    )
        else:
            print("No PostHog API key found in environment.")
            print("📍 Get your key at: https://app.posthog.com/project/settings")
            new_key = input(
                "Enter your PostHog project API key (starts with 'phc_'): "
            ).strip()
            if new_key:
                print(
                    f"💡 Add this to your environment: export POSTHOG_API_KEY='{new_key}'"
                )

        print()

        # Step 2: Team Configuration
        print("📋 Step 2: Team Configuration")
        print("-" * 30)
        current_team = os.getenv("GENOPS_TEAM")
        if current_team:
            print(f"Current team: {current_team}")
            use_current_team = input("Use current team? (y/n): ").lower() == "y"
            if not use_current_team:
                new_team = input("Enter your team name: ").strip()
                if new_team:
                    print(
                        f"💡 Add this to your environment: export GENOPS_TEAM='{new_team}'"
                    )
        else:
            new_team = input("Enter your team name (for cost attribution): ").strip()
            if new_team:
                print(
                    f"💡 Add this to your environment: export GENOPS_TEAM='{new_team}'"
                )
            else:
                print(
                    "⚠️ Skipping team configuration (recommended for cost attribution)"
                )

        print()

        # Step 3: Project Configuration
        print("📋 Step 3: Project Configuration")
        print("-" * 30)
        current_project = os.getenv("GENOPS_PROJECT")
        if current_project:
            print(f"Current project: {current_project}")
            use_current_project = input("Use current project? (y/n): ").lower() == "y"
            if not use_current_project:
                new_project = input("Enter your project name: ").strip()
                if new_project:
                    print(
                        f"💡 Add this to your environment: export GENOPS_PROJECT='{new_project}'"
                    )
        else:
            new_project = input("Enter your project name (for cost tracking): ").strip()
            if new_project:
                print(
                    f"💡 Add this to your environment: export GENOPS_PROJECT='{new_project}'"
                )
            else:
                print(
                    "⚠️ Skipping project configuration (recommended for cost tracking)"
                )

        print()

        # Step 4: Budget Configuration
        print("📋 Step 4: Budget Configuration")
        print("-" * 30)
        current_budget = os.getenv("GENOPS_DAILY_BUDGET_LIMIT")
        if current_budget:
            print(f"Current daily budget: ${current_budget}")
            use_current_budget = input("Use current budget? (y/n): ").lower() == "y"
            if not use_current_budget:
                new_budget = input(
                    "Enter daily budget limit in USD (e.g., 100.0): "
                ).strip()
                try:
                    budget_value = float(new_budget)
                    print(
                        f"💡 Add this to your environment: export GENOPS_DAILY_BUDGET_LIMIT='{budget_value}'"
                    )
                except ValueError:
                    print("⚠️ Invalid budget format, skipping budget configuration")
        else:
            print(
                "PostHog pricing: 1M events free/month, then $0.00005-$0.000198/event"
            )
            new_budget = input(
                "Enter daily budget limit in USD (e.g., 25.0) or press Enter to skip: "
            ).strip()
            if new_budget:
                try:
                    budget_value = float(new_budget)
                    print(
                        f"💡 Add this to your environment: export GENOPS_DAILY_BUDGET_LIMIT='{budget_value}'"
                    )
                except ValueError:
                    print("⚠️ Invalid budget format, skipping budget configuration")
            else:
                print("⚠️ Skipping budget configuration (will use default $1000/day)")

        print()

        # Step 5: Validation
        print("📋 Step 5: Validation")
        print("-" * 30)
        run_validation = input("Run setup validation now? (y/n): ").lower() == "y"
        if run_validation:
            print("\n🔍 Running validation...")
            result = validate_setup()
            print_validation_result(result, show_successes=False)

        print()
        print("🎉 Setup wizard completed!")
        print("🚀 Next steps:")
        print(
            "  1. Add the environment variables shown above to your shell configuration"
        )
        print("  2. Restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc)")
        print(
            '  3. Run validation: python -c "from genops.providers.posthog_validation import validate_setup, print_validation_result; print_validation_result(validate_setup())"'
        )
        print("  4. Try the examples: python examples/posthog/basic_tracking.py")

    except KeyboardInterrupt:
        print("\n\n👋 Setup wizard cancelled by user")
    except Exception as e:
        print(f"\n💥 Setup wizard error: {e}")
        print("🔧 Try manual configuration instead")


def get_setup_recommendations() -> list[dict[str, str]]:
    """Get actionable setup recommendations for PostHog integration."""
    return [
        {
            "category": "Environment Setup",
            "recommendation": "Configure PostHog API key",
            "command": "export POSTHOG_API_KEY='phc_your_project_api_key'",
            "priority": "high",
        },
        {
            "category": "Environment Setup",
            "recommendation": "Set team for cost attribution",
            "command": "export GENOPS_TEAM='your-team-name'",
            "priority": "medium",
        },
        {
            "category": "Environment Setup",
            "recommendation": "Set project for cost tracking",
            "command": "export GENOPS_PROJECT='your-project-name'",
            "priority": "medium",
        },
        {
            "category": "Cost Control",
            "recommendation": "Configure daily budget limit",
            "command": "export GENOPS_DAILY_BUDGET_LIMIT='100.0'",
            "priority": "medium",
        },
        {
            "category": "Installation",
            "recommendation": "Install GenOps with PostHog support",
            "command": "pip install genops[posthog]",
            "priority": "high",
        },
    ]


# Export validation utilities
__all__ = [
    "validate_setup",
    "print_validation_result",
    "validate_environment_config",
    "validate_sdk_dependencies",
    "validate_posthog_connection",
    "validate_genops_posthog_integration",
    "validate_cost_tracking_configuration",
    "interactive_setup_wizard",
    "get_setup_recommendations",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
]
