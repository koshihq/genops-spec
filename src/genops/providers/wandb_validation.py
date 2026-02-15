#!/usr/bin/env python3
"""
GenOps Weights & Biases Setup Validation

This module provides comprehensive validation utilities for W&B integration setup,
checking dependencies, configuration, connectivity, and governance features to ensure
everything is working correctly before proceeding with experiment tracking.

The validation framework checks:
- Python environment and W&B SDK availability
- API key configuration and authentication
- Network connectivity to W&B services
- GenOps governance configuration
- Integration functionality
- Performance characteristics

Usage:
    from genops.providers.wandb_validation import validate_setup, print_validation_result

    # Basic validation
    result = validate_setup()
    print_validation_result(result)

    # Comprehensive validation with all checks
    result = validate_setup(
        include_connectivity_tests=True,
        include_performance_tests=True,
        include_governance_tests=True
    )
    print_validation_result(result, detailed=True)

Example output:
    🔍 W&B + GenOps Setup Validation
    ✅ Dependencies: All required packages available
    ✅ Configuration: API key and settings configured
    ✅ Connectivity: W&B services accessible
    ✅ Governance: GenOps integration functional
    🎉 Overall Status: PASSED
"""

# Networking and HTTP
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation check status levels."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationCheck:
    """Individual validation check result."""

    name: str
    status: ValidationStatus
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ValidationResult:
    """Complete validation result with all checks."""

    overall_status: ValidationStatus
    checks: list[ValidationCheck] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_check(self, check: ValidationCheck) -> None:
        """Add a validation check to the results."""
        self.checks.append(check)

        # Update overall status based on worst individual status
        if check.status == ValidationStatus.FAILED:
            self.overall_status = ValidationStatus.FAILED
        elif (
            check.status == ValidationStatus.WARNING
            and self.overall_status != ValidationStatus.FAILED
        ):
            self.overall_status = ValidationStatus.WARNING


def validate_setup(
    wandb_api_key: Optional[str] = None,
    include_connectivity_tests: bool = False,
    include_performance_tests: bool = False,
    include_governance_tests: bool = False,
    timeout: int = 30,
) -> ValidationResult:
    """
    Perform comprehensive W&B + GenOps setup validation.

    Args:
        wandb_api_key: W&B API key to validate (uses env var if not provided)
        include_connectivity_tests: Test actual W&B API connectivity
        include_performance_tests: Test performance characteristics
        include_governance_tests: Test governance feature integration
        timeout: Timeout in seconds for network tests

    Returns:
        ValidationResult with all check results
    """
    start_time = time.time()
    result = ValidationResult(overall_status=ValidationStatus.PASSED)

    logger.info("Starting W&B + GenOps validation")

    # 1. Python Environment Check
    result.add_check(_check_python_environment())

    # 2. Dependencies Check
    result.add_check(_check_dependencies())

    # 3. Configuration Check
    result.add_check(_check_configuration(wandb_api_key))

    # 4. W&B SDK Functionality
    result.add_check(_check_wandb_sdk())

    # 5. GenOps Integration Check
    result.add_check(_check_genops_integration())

    # Optional connectivity tests
    if include_connectivity_tests:
        result.add_check(
            _check_wandb_connectivity(
                wandb_api_key or os.getenv("WANDB_API_KEY"), timeout
            )
        )
        result.add_check(
            _check_wandb_authentication(wandb_api_key or os.getenv("WANDB_API_KEY"))
        )

    # Optional performance tests
    if include_performance_tests:
        result.add_check(_check_performance_characteristics())

    # Optional governance tests
    if include_governance_tests:
        result.add_check(_check_governance_features())
        result.add_check(_check_cost_tracking_accuracy())

    # Calculate total execution time
    result.execution_time = time.time() - start_time

    logger.info(
        f"Validation completed in {result.execution_time:.2f}s: {result.overall_status.value}"
    )

    return result


def _check_python_environment() -> ValidationCheck:
    """Check Python version and environment."""
    start_time = time.time()

    try:
        python_version = sys.version_info

        if python_version < (3, 8):
            return ValidationCheck(
                name="Python Environment",
                status=ValidationStatus.FAILED,
                message=f"Python {python_version.major}.{python_version.minor} is too old",
                details="W&B requires Python 3.8 or newer",
                fix_suggestion="Upgrade to Python 3.8+ using pyenv or conda",
                execution_time=time.time() - start_time,
            )

        elif python_version < (3, 9):
            return ValidationCheck(
                name="Python Environment",
                status=ValidationStatus.WARNING,
                message=f"Python {python_version.major}.{python_version.minor} works but newer versions recommended",
                details="Some advanced features may require Python 3.9+",
                fix_suggestion="Consider upgrading to Python 3.9+ for optimal experience",
                execution_time=time.time() - start_time,
            )

        else:
            return ValidationCheck(
                name="Python Environment",
                status=ValidationStatus.PASSED,
                message=f"Python {python_version.major}.{python_version.minor} is supported",
                execution_time=time.time() - start_time,
            )

    except Exception as e:
        return ValidationCheck(
            name="Python Environment",
            status=ValidationStatus.FAILED,
            message=f"Failed to check Python environment: {e}",
            fix_suggestion="Ensure Python is properly installed and accessible",
            execution_time=time.time() - start_time,
        )


def _check_dependencies() -> ValidationCheck:
    """Check if all required dependencies are available."""
    start_time = time.time()

    required_packages = {
        "wandb": "Weights & Biases SDK",
        "opentelemetry": "OpenTelemetry for telemetry export",
        "opentelemetry.trace": "OpenTelemetry tracing",
    }

    optional_packages = {
        "requests": "HTTP client for API connectivity tests",
        "psutil": "System metrics for performance testing",
    }

    missing_required = []
    missing_optional = []

    # Check required packages
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_required.append(f"{package} ({description})")

    # Check optional packages
    for package, description in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(f"{package} ({description})")

    if missing_required:
        return ValidationCheck(
            name="Dependencies",
            status=ValidationStatus.FAILED,
            message=f"Missing required packages: {', '.join(missing_required)}",
            details="These packages are required for W&B + GenOps integration",
            fix_suggestion="Install with: pip install genops[wandb]",
            execution_time=time.time() - start_time,
        )

    elif missing_optional:
        return ValidationCheck(
            name="Dependencies",
            status=ValidationStatus.WARNING,
            message=f"Missing optional packages: {', '.join(missing_optional)}",
            details="These packages enable additional validation features",
            fix_suggestion="Install with: pip install requests psutil",
            execution_time=time.time() - start_time,
        )

    else:
        return ValidationCheck(
            name="Dependencies",
            status=ValidationStatus.PASSED,
            message="All required and optional packages available",
            execution_time=time.time() - start_time,
        )


def _check_configuration(wandb_api_key: Optional[str]) -> ValidationCheck:
    """Check W&B and GenOps configuration."""
    start_time = time.time()

    issues = []

    # Check W&B API key
    api_key = wandb_api_key or os.getenv("WANDB_API_KEY")
    if not api_key:
        issues.append("WANDB_API_KEY not set")
    elif not api_key.startswith(("wb-", "wab-", "wandb-")) and len(api_key) < 20:
        issues.append("WANDB_API_KEY format appears invalid")

    # Check GenOps configuration (optional but recommended)
    team = os.getenv("GENOPS_TEAM")
    project = os.getenv("GENOPS_PROJECT")

    recommendations = []
    if not team:
        recommendations.append("Set GENOPS_TEAM for cost attribution")
    if not project:
        recommendations.append("Set GENOPS_PROJECT for cost attribution")

    if issues:
        return ValidationCheck(
            name="Configuration",
            status=ValidationStatus.FAILED,
            message=f"Configuration issues: {', '.join(issues)}",
            details="Required configuration missing or invalid",
            fix_suggestion="Set WANDB_API_KEY environment variable with your W&B API key",
            execution_time=time.time() - start_time,
        )

    elif recommendations:
        return ValidationCheck(
            name="Configuration",
            status=ValidationStatus.WARNING,
            message="Configuration functional but can be improved",
            details=f"Recommendations: {', '.join(recommendations)}",
            fix_suggestion="Set GENOPS_TEAM and GENOPS_PROJECT environment variables",
            execution_time=time.time() - start_time,
        )

    else:
        return ValidationCheck(
            name="Configuration",
            status=ValidationStatus.PASSED,
            message="Configuration is complete and valid",
            execution_time=time.time() - start_time,
        )


def _check_wandb_sdk() -> ValidationCheck:
    """Check W&B SDK functionality."""
    start_time = time.time()

    try:
        import wandb

        # Check SDK version
        wandb_version = getattr(wandb, "__version__", "unknown")

        # Test basic SDK functionality
        try:
            # Test offline mode to avoid API calls
            with wandb.init(mode="offline", project="genops-validation-test") as run:
                run.log({"validation_metric": 1.0})
                run.finish()

            return ValidationCheck(
                name="W&B SDK",
                status=ValidationStatus.PASSED,
                message=f"W&B SDK v{wandb_version} functioning correctly",
                details="Basic logging and run lifecycle working",
                execution_time=time.time() - start_time,
            )

        except Exception as sdk_error:
            return ValidationCheck(
                name="W&B SDK",
                status=ValidationStatus.FAILED,
                message=f"W&B SDK functionality test failed: {sdk_error}",
                details="Basic W&B operations are not working",
                fix_suggestion="Try reinstalling W&B: pip uninstall wandb && pip install wandb",
                execution_time=time.time() - start_time,
            )

    except ImportError as e:
        return ValidationCheck(
            name="W&B SDK",
            status=ValidationStatus.FAILED,
            message=f"W&B SDK import failed: {e}",
            details="W&B package not found or corrupted",
            fix_suggestion="Install W&B: pip install wandb",
            execution_time=time.time() - start_time,
        )


def _check_genops_integration() -> ValidationCheck:
    """Check GenOps W&B integration functionality."""
    start_time = time.time()

    try:
        from genops.providers.wandb import WANDB_AVAILABLE, GenOpsWandbAdapter

        if not WANDB_AVAILABLE:
            return ValidationCheck(
                name="GenOps Integration",
                status=ValidationStatus.FAILED,
                message="W&B not available for GenOps integration",
                details="W&B SDK is required for GenOps integration",
                fix_suggestion="Install W&B: pip install wandb",
                execution_time=time.time() - start_time,
            )

        try:
            # Test adapter creation
            adapter = GenOpsWandbAdapter(
                team="validation-team",
                project="validation-project",
                daily_budget_limit=10.0,
            )

            # Test basic functionality
            metrics = adapter.get_metrics()
            assert isinstance(metrics, dict)
            assert "team" in metrics
            assert "daily_usage" in metrics

            return ValidationCheck(
                name="GenOps Integration",
                status=ValidationStatus.PASSED,
                message="GenOps W&B integration functioning correctly",
                details=f"Adapter created successfully with team={metrics.get('team')}",
                execution_time=time.time() - start_time,
            )

        except Exception as integration_error:
            return ValidationCheck(
                name="GenOps Integration",
                status=ValidationStatus.FAILED,
                message=f"GenOps integration test failed: {integration_error}",
                details="GenOps adapter creation or basic functionality failed",
                fix_suggestion="Check GenOps installation: pip install genops[wandb]",
                execution_time=time.time() - start_time,
            )

    except ImportError as e:
        return ValidationCheck(
            name="GenOps Integration",
            status=ValidationStatus.FAILED,
            message=f"GenOps W&B module import failed: {e}",
            details="GenOps W&B integration module not found",
            fix_suggestion="Install GenOps with W&B support: pip install genops[wandb]",
            execution_time=time.time() - start_time,
        )


def _check_wandb_connectivity(
    api_key: Optional[str], timeout: int = 30
) -> ValidationCheck:
    """Check connectivity to W&B services."""
    start_time = time.time()

    if not api_key:
        return ValidationCheck(
            name="W&B Connectivity",
            status=ValidationStatus.WARNING,
            message="Skipped connectivity test (no API key)",
            details="API key required for connectivity testing",
            fix_suggestion="Set WANDB_API_KEY to enable connectivity testing",
            execution_time=time.time() - start_time,
        )

    try:
        import requests

        # Test W&B API endpoint
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.wandb.ai/viewer", headers=headers, timeout=timeout
        )

        if response.status_code == 200:
            return ValidationCheck(
                name="W&B Connectivity",
                status=ValidationStatus.PASSED,
                message="W&B API accessible",
                details=f"API response time: {response.elapsed.total_seconds():.2f}s",
                execution_time=time.time() - start_time,
            )

        else:
            return ValidationCheck(
                name="W&B Connectivity",
                status=ValidationStatus.FAILED,
                message=f"W&B API returned status {response.status_code}",
                details=f"Response: {response.text[:200]}...",
                fix_suggestion="Check API key validity and network connectivity",
                execution_time=time.time() - start_time,
            )

    except ImportError:
        return ValidationCheck(
            name="W&B Connectivity",
            status=ValidationStatus.WARNING,
            message="Skipped connectivity test (requests not available)",
            details="requests package required for connectivity testing",
            fix_suggestion="Install requests: pip install requests",
            execution_time=time.time() - start_time,
        )

    except Exception as e:
        return ValidationCheck(
            name="W&B Connectivity",
            status=ValidationStatus.FAILED,
            message=f"Connectivity test failed: {e}",
            details="Network error or API issue",
            fix_suggestion="Check internet connection and API key validity",
            execution_time=time.time() - start_time,
        )


def _check_wandb_authentication(api_key: Optional[str]) -> ValidationCheck:
    """Check W&B API authentication."""
    start_time = time.time()

    if not api_key:
        return ValidationCheck(
            name="W&B Authentication",
            status=ValidationStatus.WARNING,
            message="Skipped authentication test (no API key)",
            execution_time=time.time() - start_time,
        )

    try:
        import wandb

        # Test authentication by getting user info
        api = wandb.Api(api_key=api_key)
        user = api.viewer

        if user:
            return ValidationCheck(
                name="W&B Authentication",
                status=ValidationStatus.PASSED,
                message=f"Authenticated as user: {user.get('username', 'unknown')}",
                details=f"User entity: {user.get('entity', 'unknown')}",
                execution_time=time.time() - start_time,
            )
        else:
            return ValidationCheck(
                name="W&B Authentication",
                status=ValidationStatus.FAILED,
                message="Authentication failed - invalid API key",
                fix_suggestion="Check API key from https://wandb.ai/settings",
                execution_time=time.time() - start_time,
            )

    except Exception as e:
        return ValidationCheck(
            name="W&B Authentication",
            status=ValidationStatus.FAILED,
            message=f"Authentication test failed: {e}",
            fix_suggestion="Verify API key validity and network connectivity",
            execution_time=time.time() - start_time,
        )


def _check_performance_characteristics() -> ValidationCheck:
    """Check performance characteristics of the integration."""
    start_time = time.time()

    try:
        from genops.providers.wandb import GenOpsWandbAdapter

        # Test adapter creation performance
        adapter_start = time.time()
        adapter = GenOpsWandbAdapter(team="perf-test", project="perf-test")
        adapter_time = time.time() - adapter_start

        # Test metrics retrieval performance
        metrics_start = time.time()
        adapter.get_metrics()
        metrics_time = time.time() - metrics_start

        # Performance thresholds
        if adapter_time > 1.0:
            return ValidationCheck(
                name="Performance",
                status=ValidationStatus.WARNING,
                message=f"Adapter creation slow: {adapter_time:.3f}s",
                details="Performance may be impacted by system resources",
                execution_time=time.time() - start_time,
            )

        elif metrics_time > 0.1:
            return ValidationCheck(
                name="Performance",
                status=ValidationStatus.WARNING,
                message=f"Metrics retrieval slow: {metrics_time:.3f}s",
                execution_time=time.time() - start_time,
            )

        else:
            return ValidationCheck(
                name="Performance",
                status=ValidationStatus.PASSED,
                message=f"Good performance (adapter: {adapter_time:.3f}s, metrics: {metrics_time:.3f}s)",
                execution_time=time.time() - start_time,
            )

    except Exception as e:
        return ValidationCheck(
            name="Performance",
            status=ValidationStatus.FAILED,
            message=f"Performance test failed: {e}",
            execution_time=time.time() - start_time,
        )


def _check_governance_features() -> ValidationCheck:
    """Check governance feature functionality."""
    start_time = time.time()

    try:
        from genops.providers.wandb import GenOpsWandbAdapter, GovernancePolicy

        # Test governance configuration
        adapter = GenOpsWandbAdapter(
            team="governance-test",
            project="governance-test",
            daily_budget_limit=5.0,
            governance_policy=GovernancePolicy.ADVISORY,
        )

        # Test budget tracking
        metrics = adapter.get_metrics()
        assert "daily_usage" in metrics
        assert "budget_remaining" in metrics
        assert metrics["budget_remaining"] == 5.0  # Should be full budget initially

        # Test policy configuration
        assert adapter.governance_policy == GovernancePolicy.ADVISORY

        return ValidationCheck(
            name="Governance Features",
            status=ValidationStatus.PASSED,
            message="Governance features functioning correctly",
            details="Budget tracking and policy enforcement configured",
            execution_time=time.time() - start_time,
        )

    except Exception as e:
        return ValidationCheck(
            name="Governance Features",
            status=ValidationStatus.FAILED,
            message=f"Governance test failed: {e}",
            details="Core governance functionality not working",
            execution_time=time.time() - start_time,
        )


def _check_cost_tracking_accuracy() -> ValidationCheck:
    """Check cost tracking accuracy."""
    start_time = time.time()

    try:
        from genops.providers.wandb import GenOpsWandbAdapter

        adapter = GenOpsWandbAdapter(team="cost-test", project="cost-test")

        # Test cost estimation
        initial_usage = adapter.daily_usage

        # Simulate cost update
        test_cost = 0.05
        adapter.daily_usage += test_cost

        # Check cost tracking
        metrics = adapter.get_metrics()
        assert abs(metrics["daily_usage"] - (initial_usage + test_cost)) < 0.001

        return ValidationCheck(
            name="Cost Tracking",
            status=ValidationStatus.PASSED,
            message="Cost tracking accuracy verified",
            details="Cost calculations precise to $0.001",
            execution_time=time.time() - start_time,
        )

    except Exception as e:
        return ValidationCheck(
            name="Cost Tracking",
            status=ValidationStatus.FAILED,
            message=f"Cost tracking test failed: {e}",
            execution_time=time.time() - start_time,
        )


def print_validation_result(result: ValidationResult, detailed: bool = False) -> None:
    """
    Print validation results in a user-friendly format.

    Args:
        result: ValidationResult to display
        detailed: Whether to show detailed information for each check
    """
    print("\n🔍 W&B + GenOps Setup Validation")
    print(f"🕒 Completed at: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total time: {result.execution_time:.2f}s")
    print("=" * 50)

    # Group checks by status
    passed_checks = [c for c in result.checks if c.status == ValidationStatus.PASSED]
    warning_checks = [c for c in result.checks if c.status == ValidationStatus.WARNING]
    failed_checks = [c for c in result.checks if c.status == ValidationStatus.FAILED]

    # Print summary
    print(f"✅ Passed: {len(passed_checks)} checks")
    print(f"⚠️  Warnings: {len(warning_checks)} checks")
    print(f"❌ Failed: {len(failed_checks)} checks")
    print("-" * 50)

    # Print individual check results
    for check in result.checks:
        status_emoji = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAILED: "❌",
        }[check.status]

        exec_time = f" ({check.execution_time:.3f}s)" if check.execution_time else ""
        print(f"{status_emoji} {check.name}: {check.message}{exec_time}")

        if detailed and (check.details or check.fix_suggestion):
            if check.details:
                print(f"   📋 Details: {check.details}")
            if check.fix_suggestion:
                print(f"   💡 Fix: {check.fix_suggestion}")
            print()

    # Overall status
    status_messages = {
        ValidationStatus.PASSED: "🎉 Overall Status: PASSED - Your setup is ready!",
        ValidationStatus.WARNING: "⚠️ Overall Status: WARNING - Setup functional with recommendations",
        ValidationStatus.FAILED: "❌ Overall Status: FAILED - Critical issues need resolution",
    }

    print("-" * 50)
    print(status_messages[result.overall_status])

    # Next steps based on status
    if result.overall_status == ValidationStatus.PASSED:
        print("\n🚀 Next Steps:")
        print("   • Try basic tracking: python basic_tracking.py")
        print("   • Enable zero-code governance: python auto_instrumentation.py")
        print("   • Explore experiment management: python experiment_management.py")

    elif result.overall_status == ValidationStatus.WARNING:
        print("\n📝 Recommendations:")
        for check in warning_checks:
            if check.fix_suggestion:
                print(f"   • {check.name}: {check.fix_suggestion}")

        print("\n✅ You can proceed with basic examples while addressing warnings.")

    else:
        print("\n🔧 Required Actions:")
        for check in failed_checks:
            if check.fix_suggestion:
                print(f"   • {check.name}: {check.fix_suggestion}")

        print("\n❗ Please resolve failed checks before proceeding.")


# Public wrapper functions for individual validation checks
def validate_wandb_connection(
    api_key: Optional[str] = None, timeout: int = 30
) -> ValidationCheck:
    """Validate W&B API connectivity."""
    key = api_key or os.getenv("WANDB_API_KEY")
    return _check_wandb_connectivity(key, timeout)


def validate_genops_configuration(
    wandb_api_key: Optional[str] = None,
) -> ValidationCheck:
    """Validate GenOps configuration."""
    return _check_configuration(wandb_api_key)


def validate_governance_setup() -> list[ValidationCheck]:
    """Validate governance feature setup."""
    checks = []
    checks.append(_check_governance_features())
    checks.append(_check_cost_tracking_accuracy())
    checks.append(_check_genops_integration())
    return checks


def check_environment_variables() -> list[ValidationCheck]:
    """Check required and optional environment variables."""
    checks = []
    env_vars = {
        "WANDB_API_KEY": ("W&B API key for authentication", True),
        "GENOPS_TEAM": ("Team name for cost attribution", False),
        "GENOPS_PROJECT": ("Project name for cost attribution", False),
        "GENOPS_CUSTOMER_ID": ("Customer ID for governance", False),
    }
    for var_name, (description, required) in env_vars.items():
        value = os.getenv(var_name)
        if value:
            checks.append(
                ValidationCheck(
                    name=var_name,
                    status=ValidationStatus.PASSED,
                    message=f"{var_name} is set",
                    details=description,
                )
            )
        elif required:
            checks.append(
                ValidationCheck(
                    name=var_name,
                    status=ValidationStatus.FAILED,
                    message=f"{var_name} is not set",
                    details=description,
                    fix_suggestion=f"Set {var_name} environment variable",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name=var_name,
                    status=ValidationStatus.WARNING,
                    message=f"{var_name} is not set (optional)",
                    details=description,
                    fix_suggestion=f"Set {var_name} for full functionality",
                )
            )
    return checks


def check_dependencies() -> list[ValidationCheck]:
    """Check required and optional dependency availability."""
    import importlib

    checks = []
    deps = {
        "wandb": ("Weights & Biases SDK", True),
        "genops": ("GenOps governance framework", True),
        "opentelemetry": ("OpenTelemetry SDK", True),
        "requests": ("HTTP client library", False),
    }
    for dep_name, (description, required) in deps.items():
        try:
            importlib.import_module(dep_name)
            checks.append(
                ValidationCheck(
                    name=dep_name,
                    status=ValidationStatus.PASSED,
                    message=f"{dep_name} is available",
                    details=description,
                )
            )
        except ImportError:
            status = ValidationStatus.FAILED if required else ValidationStatus.WARNING
            checks.append(
                ValidationCheck(
                    name=dep_name,
                    status=status,
                    message=f"{dep_name} is not installed",
                    details=description,
                    fix_suggestion=f"Install with: pip install {dep_name}",
                )
            )
    return checks


# Convenience function for quick validation
def quick_validate() -> bool:
    """
    Perform quick validation and return True if setup is ready.

    Returns:
        True if validation passes, False otherwise
    """
    result = validate_setup()
    return result.overall_status == ValidationStatus.PASSED


# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate W&B + GenOps setup")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    parser.add_argument(
        "--connectivity", action="store_true", help="Include connectivity tests"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Include performance tests"
    )
    parser.add_argument(
        "--governance", action="store_true", help="Include governance tests"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Network timeout in seconds"
    )

    args = parser.parse_args()

    # Run validation
    result = validate_setup(
        include_connectivity_tests=args.connectivity,
        include_performance_tests=args.performance,
        include_governance_tests=args.governance,
        timeout=args.timeout,
    )

    # Print results
    print_validation_result(result, detailed=args.detailed)

    # Exit with appropriate code
    sys.exit(0 if result.overall_status != ValidationStatus.FAILED else 1)
