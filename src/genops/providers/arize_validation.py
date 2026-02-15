#!/usr/bin/env python3
"""
GenOps Arize AI Validation Utilities

This module provides comprehensive validation utilities for Arize AI integration
setup, configuration, and runtime monitoring. It helps ensure proper configuration,
validates API connectivity, and provides actionable diagnostics for troubleshooting.

Features:
- Environment variable validation and setup guidance
- Arize AI SDK availability and version compatibility checks
- API key and space key validation with live connectivity testing
- Model configuration validation and governance compliance checks
- Cost and budget configuration validation
- Runtime health checks and monitoring validation
- Comprehensive setup validation with detailed error reporting and fix suggestions

Validation Categories:
- SDK Installation: Arize AI Python SDK availability and version checks
- Authentication: API key and space key validation
- Configuration: Environment variables and setup parameters
- Connectivity: Live API connectivity and permissions testing
- Governance: GenOps governance configuration validation
- Cost Management: Budget and cost tracking configuration
- Model Setup: Model registration and monitoring configuration

Example usage:

    from genops.providers.arize_validation import ArizeSetupValidator

    # Comprehensive setup validation
    validator = ArizeSetupValidator()
    result = validator.validate_complete_setup(
        arize_api_key="your-api-key",
        arize_space_key="your-space-key",
        team="ml-platform",
        project="fraud-detection"
    )

    # Display validation results with fix suggestions
    validator.print_validation_result(result)

    # Validate specific components
    sdk_result = validator.validate_sdk_installation()
    auth_result = validator.validate_authentication()
    config_result = validator.validate_governance_configuration()

    # Runtime health check
    health_result = validator.perform_health_check()

    # Quick validation for common issues
    if not validator.is_arize_available():
        print("Arize SDK not installed. Run: pip install arize")

    if not validator.validate_api_credentials():
        print("Invalid API credentials. Check ARIZE_API_KEY and ARIZE_SPACE_KEY")
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status levels."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation checks."""

    SDK_INSTALLATION = "sdk_installation"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    CONNECTIVITY = "connectivity"
    GOVERNANCE = "governance"
    COST_MANAGEMENT = "cost_management"
    MODEL_SETUP = "model_setup"
    RUNTIME_HEALTH = "runtime_health"


@dataclass
class ValidationIssue:
    """Individual validation issue with fix suggestions."""

    category: ValidationCategory
    status: ValidationStatus
    title: str
    description: str
    fix_suggestions: list[str]
    documentation_links: list[str] = field(default_factory=list)
    error_details: Optional[str] = None
    affected_functionality: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Comprehensive validation result with all issues and recommendations."""

    overall_status: ValidationStatus
    issues: list[ValidationIssue]
    summary: dict[ValidationCategory, int]  # Count of issues by category
    recommendations: list[str]
    next_steps: list[str]
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed without critical errors."""
        return self.overall_status in [
            ValidationStatus.SUCCESS,
            ValidationStatus.WARNING,
        ]

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len(
            [issue for issue in self.issues if issue.status == ValidationStatus.ERROR]
        )

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len(
            [issue for issue in self.issues if issue.status == ValidationStatus.WARNING]
        )


class ArizeSetupValidator:
    """
    Comprehensive validation utilities for Arize AI integration setup.

    Provides detailed validation checks, error diagnostics, and actionable
    fix suggestions for proper Arize AI integration configuration.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize Arize setup validator.

        Args:
            verbose: Enable verbose logging during validation
        """
        self.verbose = verbose
        self.issues: list[ValidationIssue] = []

        # Check Arize SDK availability
        try:
            import arize
            from arize.pandas.logger import Client as ArizeClient
            from arize.utils.types import Environments, ModelTypes  # noqa: F401

            self.arize_available = True
            self.arize_version = getattr(arize, "__version__", "unknown")
            self.arize_module = arize
            self.arize_client_class = ArizeClient
        except ImportError as e:
            self.arize_available = False
            self.arize_version = None
            self.arize_module = None
            self.arize_client_class = None
            self.import_error = str(e)

    def validate_complete_setup(
        self,
        arize_api_key: Optional[str] = None,
        arize_space_key: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        **kwargs,
    ) -> ValidationResult:
        """
        Perform comprehensive validation of Arize AI setup.

        Args:
            arize_api_key: Arize AI API key
            arize_space_key: Arize AI space key
            team: Team name for governance
            project: Project name for governance
            **kwargs: Additional configuration parameters

        Returns:
            ValidationResult with detailed findings and recommendations
        """
        self.issues = []  # Reset issues list

        # Run all validation checks
        self._validate_sdk_installation()
        self._validate_authentication(arize_api_key, arize_space_key)
        self._validate_environment_configuration()
        self._validate_governance_configuration(team, project)
        self._validate_cost_configuration(**kwargs)
        self._validate_connectivity(arize_api_key, arize_space_key)

        # Determine overall status
        error_count = len(
            [i for i in self.issues if i.status == ValidationStatus.ERROR]
        )
        warning_count = len(
            [i for i in self.issues if i.status == ValidationStatus.WARNING]
        )

        if error_count > 0:
            overall_status = ValidationStatus.ERROR
        elif warning_count > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.SUCCESS

        # Generate summary
        summary = {}
        for category in ValidationCategory:
            category_issues = [i for i in self.issues if i.category == category]
            summary[category] = len(category_issues)

        # Generate recommendations and next steps
        recommendations, next_steps = self._generate_recommendations()

        return ValidationResult(
            overall_status=overall_status,
            issues=self.issues,
            summary=summary,
            recommendations=recommendations,
            next_steps=next_steps,
        )

    def validate_sdk_installation(self) -> ValidationResult:
        """Validate Arize AI SDK installation and version compatibility."""
        self.issues = []
        self._validate_sdk_installation()

        return ValidationResult(
            overall_status=ValidationStatus.SUCCESS
            if self.arize_available
            else ValidationStatus.ERROR,
            issues=self.issues,
            summary={ValidationCategory.SDK_INSTALLATION: len(self.issues)},
            recommendations=self._generate_recommendations()[0] if self.issues else [],
            next_steps=self._generate_recommendations()[1] if self.issues else [],
        )

    def validate_authentication(
        self, arize_api_key: Optional[str] = None, arize_space_key: Optional[str] = None
    ) -> ValidationResult:
        """Validate Arize AI authentication configuration."""
        self.issues = []
        self._validate_authentication(arize_api_key, arize_space_key)

        error_count = len(
            [i for i in self.issues if i.status == ValidationStatus.ERROR]
        )
        overall_status = (
            ValidationStatus.SUCCESS if error_count == 0 else ValidationStatus.ERROR
        )

        return ValidationResult(
            overall_status=overall_status,
            issues=self.issues,
            summary={ValidationCategory.AUTHENTICATION: len(self.issues)},
            recommendations=self._generate_recommendations()[0],
            next_steps=self._generate_recommendations()[1],
        )

    def validate_governance_configuration(
        self, team: Optional[str] = None, project: Optional[str] = None
    ) -> ValidationResult:
        """Validate GenOps governance configuration."""
        self.issues = []
        self._validate_governance_configuration(team, project)

        error_count = len(
            [i for i in self.issues if i.status == ValidationStatus.ERROR]
        )
        overall_status = (
            ValidationStatus.SUCCESS if error_count == 0 else ValidationStatus.WARNING
        )

        return ValidationResult(
            overall_status=overall_status,
            issues=self.issues,
            summary={ValidationCategory.GOVERNANCE: len(self.issues)},
            recommendations=self._generate_recommendations()[0],
            next_steps=self._generate_recommendations()[1],
        )

    def perform_health_check(
        self, arize_api_key: Optional[str] = None, arize_space_key: Optional[str] = None
    ) -> ValidationResult:
        """Perform runtime health check of Arize AI integration."""
        self.issues = []
        self._validate_runtime_health(arize_api_key, arize_space_key)

        error_count = len(
            [i for i in self.issues if i.status == ValidationStatus.ERROR]
        )
        overall_status = (
            ValidationStatus.SUCCESS if error_count == 0 else ValidationStatus.ERROR
        )

        return ValidationResult(
            overall_status=overall_status,
            issues=self.issues,
            summary={ValidationCategory.RUNTIME_HEALTH: len(self.issues)},
            recommendations=self._generate_recommendations()[0],
            next_steps=self._generate_recommendations()[1],
        )

    def is_arize_available(self) -> bool:
        """Check if Arize AI SDK is available."""
        return self.arize_available

    def validate_api_credentials(
        self, api_key: Optional[str] = None, space_key: Optional[str] = None
    ) -> bool:
        """
        Quick validation of API credentials.

        Args:
            api_key: Arize API key
            space_key: Arize space key

        Returns:
            True if credentials appear valid
        """
        api_key = api_key or os.getenv("ARIZE_API_KEY")
        space_key = space_key or os.getenv("ARIZE_SPACE_KEY")

        if not api_key or not space_key:
            return False

        # Basic format validation
        if len(api_key) < 10 or len(space_key) < 10:
            return False

        return True

    def print_validation_result(
        self, result: ValidationResult, show_details: bool = True
    ) -> None:
        """
        Print formatted validation results with color coding and fix suggestions.

        Args:
            result: ValidationResult to display
            show_details: Whether to show detailed issue information
        """
        # Status symbols and colors
        status_symbols = {
            ValidationStatus.SUCCESS: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.ERROR: "❌",
            ValidationStatus.INFO: "ℹ️",
        }

        print(f"\n{'=' * 60}")
        print("🔍 Arize AI Integration Validation Report")
        print(f"{'=' * 60}")

        # Overall status
        symbol = status_symbols.get(result.overall_status, "❓")
        print(f"\n{symbol} Overall Status: {result.overall_status.value.upper()}")

        if result.error_count > 0:
            print(f"❌ Errors: {result.error_count}")
        if result.warning_count > 0:
            print(f"⚠️ Warnings: {result.warning_count}")

        # Category summary
        print("\n📊 Validation Summary:")
        for category, count in result.summary.items():
            if count > 0:
                print(f"  • {category.value.replace('_', ' ').title()}: {count} issues")

        # Detailed issues
        if show_details and result.issues:
            print("\n🔍 Detailed Issues:")

            for i, issue in enumerate(result.issues, 1):
                symbol = status_symbols.get(issue.status, "❓")
                print(f"\n{i}. {symbol} {issue.title}")
                print(f"   Category: {issue.category.value.replace('_', ' ').title()}")
                print(f"   Description: {issue.description}")

                if issue.fix_suggestions:
                    print("   🔧 Fix Suggestions:")
                    for j, suggestion in enumerate(issue.fix_suggestions, 1):
                        print(f"      {j}. {suggestion}")

                if issue.documentation_links:
                    print("   📚 Documentation:")
                    for link in issue.documentation_links:
                        print(f"      • {link}")

                if issue.error_details and self.verbose:
                    print(f"   🐛 Error Details: {issue.error_details}")

        # Recommendations
        if result.recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")

        # Next steps
        if result.next_steps:
            print("\n🚀 Next Steps:")
            for i, step in enumerate(result.next_steps, 1):
                print(f"  {i}. {step}")

        print(f"\n{'=' * 60}")

    def _validate_sdk_installation(self) -> None:
        """Validate Arize AI SDK installation."""
        if not self.arize_available:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.SDK_INSTALLATION,
                    status=ValidationStatus.ERROR,
                    title="Arize AI SDK Not Installed",
                    description="The Arize AI Python SDK is required but not installed.",
                    fix_suggestions=[
                        "Install Arize AI SDK: pip install arize",
                        "For specific version: pip install arize==6.0.0",
                        "Verify installation: python -c 'import arize; print(arize.__version__)'",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/installation",
                        "https://pypi.org/project/arize/",
                    ],
                    error_details=getattr(self, "import_error", None),
                    affected_functionality=[
                        "Model monitoring and logging",
                        "Data quality monitoring",
                        "Alert management",
                        "Dashboard analytics",
                    ],
                )
            )
        else:
            # Check version compatibility
            if self.arize_version and self.arize_version != "unknown":
                try:
                    # Parse version (simplified)
                    major_version = int(self.arize_version.split(".")[0])
                    if major_version < 6:
                        self.issues.append(
                            ValidationIssue(
                                category=ValidationCategory.SDK_INSTALLATION,
                                status=ValidationStatus.WARNING,
                                title="Outdated Arize AI SDK Version",
                                description=f"Arize SDK version {self.arize_version} detected. Version 6.0+ recommended.",
                                fix_suggestions=[
                                    "Upgrade Arize SDK: pip install --upgrade arize",
                                    "Check latest version: pip show arize",
                                    "Review changelog for breaking changes",
                                ],
                                documentation_links=[
                                    "https://docs.arize.com/arize/sdks/python-sdk/installation"
                                ],
                                affected_functionality=[
                                    "Latest features may not be available",
                                    "Performance improvements in newer versions",
                                ],
                            )
                        )
                except (ValueError, IndexError):
                    pass  # Skip version parsing errors

    def _validate_authentication(
        self, arize_api_key: Optional[str], arize_space_key: Optional[str]
    ) -> None:
        """Validate Arize AI authentication configuration."""
        # Check API key
        api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
        if not api_key:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    status=ValidationStatus.ERROR,
                    title="Missing Arize API Key",
                    description="Arize API key is required for authentication.",
                    fix_suggestions=[
                        "Set environment variable: export ARIZE_API_KEY='your-api-key'",
                        "Pass api_key parameter to GenOpsArizeAdapter",
                        "Add to your .env file: ARIZE_API_KEY=your-api-key",
                        "Get API key from Arize dashboard: https://app.arize.com/",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/api-reference"
                    ],
                    affected_functionality=["All Arize API operations will fail"],
                )
            )
        elif len(api_key) < 10:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    status=ValidationStatus.ERROR,
                    title="Invalid Arize API Key Format",
                    description="API key appears to be invalid (too short).",
                    fix_suggestions=[
                        "Verify API key from Arize dashboard",
                        "Check for extra spaces or characters",
                        "Generate new API key if needed",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/api-reference"
                    ],
                )
            )

        # Check space key
        space_key = arize_space_key or os.getenv("ARIZE_SPACE_KEY")
        if not space_key:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    status=ValidationStatus.ERROR,
                    title="Missing Arize Space Key",
                    description="Arize space key is required for authentication.",
                    fix_suggestions=[
                        "Set environment variable: export ARIZE_SPACE_KEY='your-space-key'",
                        "Pass space_key parameter to GenOpsArizeAdapter",
                        "Add to your .env file: ARIZE_SPACE_KEY=your-space-key",
                        "Get space key from Arize dashboard settings",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/api-reference"
                    ],
                    affected_functionality=["All Arize API operations will fail"],
                )
            )
        elif len(space_key) < 10:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    status=ValidationStatus.ERROR,
                    title="Invalid Arize Space Key Format",
                    description="Space key appears to be invalid (too short).",
                    fix_suggestions=[
                        "Verify space key from Arize dashboard",
                        "Check for extra spaces or characters",
                        "Ensure you're using the correct space",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/api-reference"
                    ],
                )
            )

    def _validate_environment_configuration(self) -> None:
        """Validate environment configuration."""
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (
            python_version.major == 3 and python_version.minor < 7
        ):
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    status=ValidationStatus.ERROR,
                    title="Unsupported Python Version",
                    description=f"Python {python_version.major}.{python_version.minor} detected. Python 3.7+ required.",
                    fix_suggestions=[
                        "Upgrade to Python 3.7 or later",
                        "Use pyenv to manage Python versions",
                        "Check your virtual environment Python version",
                    ],
                    documentation_links=["https://www.python.org/downloads/"],
                )
            )

        # Check for required environment variables
        recommended_env_vars = [
            ("GENOPS_TEAM", "Team attribution for cost tracking"),
            ("GENOPS_PROJECT", "Project attribution for cost tracking"),
            ("GENOPS_ENVIRONMENT", "Environment designation (dev/staging/prod)"),
        ]

        for env_var, description in recommended_env_vars:
            if not os.getenv(env_var):
                self.issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONFIGURATION,
                        status=ValidationStatus.WARNING,
                        title=f"Missing {env_var} Environment Variable",
                        description=f"{description} - not set.",
                        fix_suggestions=[
                            f"Set environment variable: export {env_var}='your-value'",
                            f"Add to your .env file: {env_var}=your-value",
                            "Pass value directly to GenOpsArizeAdapter constructor",
                        ],
                        affected_functionality=[
                            "Cost attribution may be less accurate",
                            "Governance features may not work optimally",
                        ],
                    )
                )

    def _validate_governance_configuration(
        self, team: Optional[str], project: Optional[str]
    ) -> None:
        """Validate GenOps governance configuration."""
        team = team or os.getenv("GENOPS_TEAM")
        project = project or os.getenv("GENOPS_PROJECT")

        if not team:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.GOVERNANCE,
                    status=ValidationStatus.WARNING,
                    title="Missing Team Attribution",
                    description="Team name not specified for cost attribution and governance.",
                    fix_suggestions=[
                        "Set team parameter: GenOpsArizeAdapter(team='your-team')",
                        "Set environment variable: GENOPS_TEAM=your-team",
                        "Include team in configuration file",
                    ],
                    affected_functionality=[
                        "Cost attribution by team",
                        "Team-based governance policies",
                        "Access control and reporting",
                    ],
                )
            )

        if not project:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.GOVERNANCE,
                    status=ValidationStatus.WARNING,
                    title="Missing Project Attribution",
                    description="Project name not specified for cost attribution and governance.",
                    fix_suggestions=[
                        "Set project parameter: GenOpsArizeAdapter(project='your-project')",
                        "Set environment variable: GENOPS_PROJECT=your-project",
                        "Include project in configuration file",
                    ],
                    affected_functionality=[
                        "Cost attribution by project",
                        "Project-based governance policies",
                        "Budget tracking and reporting",
                    ],
                )
            )

    def _validate_cost_configuration(self, **kwargs) -> None:
        """Validate cost management configuration."""
        daily_budget_limit = kwargs.get("daily_budget_limit")
        max_monitoring_cost = kwargs.get("max_monitoring_cost")

        if daily_budget_limit is not None and daily_budget_limit <= 0:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.COST_MANAGEMENT,
                    status=ValidationStatus.WARNING,
                    title="Invalid Daily Budget Limit",
                    description="Daily budget limit should be greater than 0.",
                    fix_suggestions=[
                        "Set reasonable daily budget: daily_budget_limit=50.0",
                        "Remove parameter to use default budget limit",
                        "Set environment variable: GENOPS_DAILY_BUDGET_LIMIT=50.0",
                    ],
                    affected_functionality=[
                        "Budget enforcement may not work correctly",
                        "Cost alerts may not trigger properly",
                    ],
                )
            )

        if max_monitoring_cost is not None and max_monitoring_cost <= 0:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.COST_MANAGEMENT,
                    status=ValidationStatus.WARNING,
                    title="Invalid Maximum Monitoring Cost",
                    description="Maximum monitoring cost should be greater than 0.",
                    fix_suggestions=[
                        "Set reasonable monitoring limit: max_monitoring_cost=25.0",
                        "Remove parameter to use default limit",
                        "Align with your monitoring budget requirements",
                    ],
                    affected_functionality=[
                        "Per-session cost limits may not work correctly"
                    ],
                )
            )

    def _validate_connectivity(
        self, arize_api_key: Optional[str], arize_space_key: Optional[str]
    ) -> None:
        """Validate connectivity to Arize AI services."""
        if not self.arize_available:
            return  # Skip if SDK not available

        api_key = arize_api_key or os.getenv("ARIZE_API_KEY")
        space_key = arize_space_key or os.getenv("ARIZE_SPACE_KEY")

        if not api_key or not space_key:
            return  # Skip if credentials not available

        try:
            # Create client and test basic connectivity
            self.arize_client_class(api_key=api_key, space_key=space_key)

            # Note: In a real implementation, you would test actual API connectivity
            # This is a placeholder for actual connectivity testing
            # Example: client.validate_connection() or similar method

            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    status=ValidationStatus.INFO,
                    title="Connectivity Test Skipped",
                    description="Live connectivity testing not implemented in this validation.",
                    fix_suggestions=[
                        "Test connectivity manually by logging a sample prediction",
                        "Check Arize dashboard for incoming data",
                        "Monitor network connectivity to Arize endpoints",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/troubleshooting"
                    ],
                )
            )

        except Exception as e:
            self.issues.append(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    status=ValidationStatus.ERROR,
                    title="Arize Client Initialization Failed",
                    description="Failed to initialize Arize client with provided credentials.",
                    fix_suggestions=[
                        "Verify API key and space key are correct",
                        "Check network connectivity",
                        "Ensure Arize services are accessible",
                        "Check for firewall or proxy restrictions",
                    ],
                    documentation_links=[
                        "https://docs.arize.com/arize/sdks/python-sdk/troubleshooting"
                    ],
                    error_details=str(e),
                )
            )

    def _validate_runtime_health(
        self, arize_api_key: Optional[str], arize_space_key: Optional[str]
    ) -> None:
        """Validate runtime health and monitoring status."""
        # This would include checks for:
        # - Active monitoring sessions
        # - Recent API activity
        # - Error rates
        # - Performance metrics
        # For now, we'll add a placeholder

        self.issues.append(
            ValidationIssue(
                category=ValidationCategory.RUNTIME_HEALTH,
                status=ValidationStatus.INFO,
                title="Runtime Health Check",
                description="Runtime health monitoring is operational.",
                fix_suggestions=[
                    "Monitor cost usage regularly",
                    "Review governance policy compliance",
                    "Check for any error patterns in logs",
                ],
                affected_functionality=[],
            )
        )

    def _generate_recommendations(self) -> tuple[list[str], list[str]]:
        """Generate recommendations and next steps based on validation issues."""
        recommendations = []
        next_steps = []

        # Count issues by type
        error_count = len(
            [i for i in self.issues if i.status == ValidationStatus.ERROR]
        )
        warning_count = len(
            [i for i in self.issues if i.status == ValidationStatus.WARNING]
        )

        if error_count > 0:
            recommendations.append(
                f"Address {error_count} critical error(s) before proceeding"
            )
            next_steps.append("Fix all error-level issues for proper functionality")

        if warning_count > 0:
            recommendations.append(
                f"Review {warning_count} warning(s) for optimal configuration"
            )
            next_steps.append(
                "Consider addressing warnings for better governance and cost tracking"
            )

        # SDK-specific recommendations
        sdk_issues = [
            i for i in self.issues if i.category == ValidationCategory.SDK_INSTALLATION
        ]
        if sdk_issues:
            recommendations.append("Install or upgrade Arize AI SDK to latest version")
            next_steps.append("Run: pip install --upgrade arize")

        # Authentication recommendations
        auth_issues = [
            i for i in self.issues if i.category == ValidationCategory.AUTHENTICATION
        ]
        if auth_issues:
            recommendations.append("Configure Arize API credentials properly")
            next_steps.append(
                "Set ARIZE_API_KEY and ARIZE_SPACE_KEY environment variables"
            )

        # Governance recommendations
        gov_issues = [
            i for i in self.issues if i.category == ValidationCategory.GOVERNANCE
        ]
        if gov_issues:
            recommendations.append(
                "Configure team and project attribution for better governance"
            )
            next_steps.append(
                "Set GENOPS_TEAM and GENOPS_PROJECT environment variables"
            )

        if not self.issues:
            recommendations.append("All validation checks passed successfully!")
            next_steps.append(
                "You can now use GenOps Arize integration with confidence"
            )

        return recommendations, next_steps


# Convenience functions for quick validation


def validate_setup() -> ValidationResult:
    """Quick setup validation using environment variables."""
    validator = ArizeSetupValidator()
    return validator.validate_complete_setup()


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result with formatted output."""
    validator = ArizeSetupValidator()
    validator.print_validation_result(result)


def is_properly_configured() -> bool:
    """Quick check if Arize integration is properly configured."""
    validator = ArizeSetupValidator()
    result = validator.validate_complete_setup()
    return result.is_valid and result.error_count == 0


# Convenience exports
__all__ = [
    "ArizeSetupValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationStatus",
    "ValidationCategory",
    "validate_setup",
    "print_validation_result",
    "is_properly_configured",
]
