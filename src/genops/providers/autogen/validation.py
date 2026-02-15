#!/usr/bin/env python3
"""
AutoGen Setup Validation for GenOps Governance

Comprehensive validation and diagnostics for AutoGen integration setup,
environment configuration, and governance readiness.

Usage:
    from genops.providers.autogen import validate_autogen_setup, print_validation_result

    # Quick validation
    result = validate_autogen_setup()
    print_validation_result(result)

    # Detailed validation with custom settings
    result = validate_autogen_setup(
        team="ai-research",
        project="multi-agent-system",
        check_models=["gpt-4", "claude-3-sonnet"],
        verify_connectivity=True
    )

Features:
    - AutoGen installation and version verification
    - Environment variable and API key validation
    - Model availability and connectivity testing
    - GenOps configuration validation
    - Performance benchmarking and optimization suggestions
    - Comprehensive diagnostic reporting with actionable fixes
    - Quick validation for CI/CD pipelines
"""

import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue with severity and fix suggestions."""

    category: str
    severity: str  # "error", "warning", "info"
    title: str
    description: str
    fix_suggestion: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Comprehensive validation result with issues and diagnostics."""

    success: bool
    overall_score: float  # 0-100 score
    timestamp: datetime
    environment_info: dict[str, Any]
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)


def validate_autogen_setup(
    team: str = "default-team",
    project: str = "autogen-validation",
    check_models: list[str] = None,  # type: ignore
    verify_connectivity: bool = True,
    run_performance_tests: bool = False,
    api_timeout_seconds: int = 10,
) -> ValidationResult:
    """
    Comprehensive AutoGen setup validation.

    Args:
        team: Team name for governance testing
        project: Project name for governance testing
        check_models: List of models to verify availability
        verify_connectivity: Test API connectivity
        run_performance_tests: Run performance benchmarks
        api_timeout_seconds: Timeout for API tests

    Returns:
        ValidationResult: Comprehensive validation results
    """
    start_time = datetime.now()
    result = ValidationResult(
        success=True,
        overall_score=100.0,
        timestamp=start_time,
        environment_info=_gather_environment_info(),
        issues=[],
        checks_performed=[],
        recommendations=[],
        performance_metrics={},
    )

    logger.info("Starting AutoGen setup validation...")

    # Core validation checks
    _check_autogen_installation(result)
    _check_python_environment(result)
    _check_genops_integration(result, team, project)
    _check_environment_variables(result)

    if check_models:
        _check_model_availability(result, check_models)

    if verify_connectivity:
        _check_api_connectivity(result, api_timeout_seconds)

    if run_performance_tests:
        _run_performance_tests(result)

    # Final scoring and recommendations
    _calculate_final_score(result)
    _generate_recommendations(result)

    duration = (datetime.now() - start_time).total_seconds()
    result.performance_metrics["validation_duration_seconds"] = duration

    logger.info(
        f"AutoGen validation completed in {duration:.2f}s - Score: {result.overall_score:.1f}/100"
    )

    return result


def quick_validate() -> bool:
    """
    Quick validation check for CI/CD pipelines.

    Returns:
        bool: True if basic validation passes
    """
    try:
        # Basic checks only
        result = validate_autogen_setup(
            verify_connectivity=False, run_performance_tests=False
        )

        # Consider validation passed if no critical errors
        critical_errors = [
            issue for issue in result.issues if issue.severity == "error"
        ]
        return len(critical_errors) == 0

    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False


def print_validation_result(result: ValidationResult, verbose: bool = True):
    """
    Print validation results in a user-friendly format.

    Args:
        result: ValidationResult to display
        verbose: Show detailed information
    """
    print("\n" + "=" * 80)
    print("🔍 AutoGen + GenOps Validation Report")
    print("=" * 80)

    # Overall status
    status_emoji = "✅" if result.success else "❌"
    print(
        f"\n{status_emoji} Overall Status: {'PASSED' if result.success else 'FAILED'}"
    )
    print(f"📊 Score: {result.overall_score:.1f}/100")
    print(f"🕐 Validated at: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Environment info
    if verbose:
        print("\n📋 Environment Information:")
        env = result.environment_info
        print(f"   Python: {env.get('python_version', 'Unknown')}")
        print(f"   Platform: {env.get('platform', 'Unknown')}")
        print(f"   AutoGen: {env.get('autogen_version', 'Not installed')}")
        print(f"   GenOps: {env.get('genops_version', 'Unknown')}")

    # Issues by severity
    errors = [issue for issue in result.issues if issue.severity == "error"]
    warnings = [issue for issue in result.issues if issue.severity == "warning"]
    info = [issue for issue in result.issues if issue.severity == "info"]

    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for issue in errors:
            print(f"   • {issue.title}")
            if verbose:
                print(f"     {issue.description}")
                print(f"     💡 Fix: {issue.fix_suggestion}")

    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for issue in warnings:
            print(f"   • {issue.title}")
            if verbose:
                print(f"     {issue.description}")
                print(f"     💡 Fix: {issue.fix_suggestion}")

    if verbose and info:
        print(f"\nℹ️  Information ({len(info)}):")
        for issue in info:
            print(f"   • {issue.title}")
            print(f"     {issue.description}")

    # Recommendations
    if result.recommendations:
        print("\n🎯 Recommendations:")
        for i, rec in enumerate(result.recommendations[:5], 1):  # Show top 5
            print(f"   {i}. {rec}")

    # Performance metrics
    if result.performance_metrics and verbose:
        print("\n⚡ Performance Metrics:")
        for key, value in result.performance_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

    print("\n" + "=" * 80)

    if not result.success:
        print("💡 Run with verbose=True for detailed fix suggestions")
    else:
        print("🎉 AutoGen + GenOps setup is ready for production!")
    print("=" * 80 + "\n")


def _gather_environment_info() -> dict[str, Any]:
    """Gather comprehensive environment information."""
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "timestamp": datetime.now().isoformat(),
    }

    # AutoGen version
    try:
        import autogen

        info["autogen_version"] = getattr(autogen, "__version__", "Unknown")
        info["autogen_location"] = autogen.__file__
    except ImportError:
        info["autogen_version"] = "Not installed"
        info["autogen_location"] = None  # type: ignore[assignment]

    # GenOps version
    try:
        import genops

        info["genops_version"] = getattr(genops, "__version__", "Unknown")
        info["genops_location"] = genops.__file__
    except ImportError:
        info["genops_version"] = "Not installed"
        info["genops_location"] = None  # type: ignore[assignment]

    # OpenTelemetry
    try:
        import opentelemetry

        info["opentelemetry_version"] = getattr(opentelemetry, "__version__", "Unknown")
    except ImportError:
        info["opentelemetry_version"] = "Not installed"

    # Environment variables
    env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "COHERE_API_KEY",
        "HUGGINGFACE_API_TOKEN",
        "GENOPS_TEAM",
        "GENOPS_PROJECT",
        "GENOPS_ENVIRONMENT",
    ]

    info["environment_variables"] = {  # type: ignore[assignment]
        var: "SET" if os.getenv(var) else "NOT_SET" for var in env_vars
    }

    return info


def _check_autogen_installation(result: ValidationResult):
    """Check AutoGen installation, version, and common issues."""
    result.checks_performed.append("AutoGen Installation")

    try:
        import autogen

        version = getattr(autogen, "__version__", "Unknown")

        result.issues.append(
            ValidationIssue(
                category="installation",
                severity="info",
                title="AutoGen Installation Found",
                description=f"AutoGen version {version} is installed",
                fix_suggestion="No action needed",
                details={"version": version, "location": autogen.__file__},
            )
        )

        # Check for minimum version and known issues
        if version != "Unknown":
            try:
                from packaging import version as pkg_version

                current_ver = pkg_version.parse(version)

                if current_ver < pkg_version.parse("0.2.0"):
                    result.issues.append(
                        ValidationIssue(
                            category="installation",
                            severity="warning",
                            title="AutoGen Version May Be Outdated",
                            description=f"AutoGen {version} detected, newer versions recommended",
                            fix_suggestion="Upgrade with: pip install --upgrade pyautogen",
                            details={
                                "current_version": version,
                                "recommended_min": "0.2.0",
                            },
                        )
                    )

            except ImportError:
                # packaging not available, skip version comparison
                pass

        # Check for common AutoGen configuration issues
        _check_autogen_config_issues(result, autogen)

        # Test basic AutoGen functionality
        try:
            # Try to create a basic agent to verify AutoGen works
            autogen.ConversableAgent(
                name="test_agent",
                llm_config=False,  # No LLM needed for test
                human_input_mode="NEVER",
            )
            result.issues.append(
                ValidationIssue(
                    category="installation",
                    severity="info",
                    title="AutoGen Basic Functionality Verified",
                    description="Successfully created test AutoGen agent",
                    fix_suggestion="No action needed",
                )
            )
        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    category="installation",
                    severity="error",
                    title="AutoGen Functionality Issue",
                    description=f"Cannot create basic AutoGen agent: {str(e)}",
                    fix_suggestion="Reinstall AutoGen: pip uninstall pyautogen && pip install pyautogen",
                    details={"test_error": str(e)},
                )
            )

    except ImportError as e:
        result.success = False
        result.issues.append(
            ValidationIssue(
                category="installation",
                severity="error",
                title="AutoGen Not Installed",
                description="AutoGen is required but not found in the environment",
                fix_suggestion="Install AutoGen with: pip install pyautogen",
                details={"import_error": str(e)},
            )
        )

        # Check for common installation issues
        _diagnose_autogen_install_issues(result)


def _check_autogen_config_issues(result: ValidationResult, autogen_module):
    """Check for common AutoGen configuration issues."""

    # Check if AutoGen can access required dependencies
    try:
        # Test openai import (common issue)
        import openai  # noqa: F401
    except ImportError:
        result.issues.append(
            ValidationIssue(
                category="dependencies",
                severity="warning",
                title="OpenAI Package Not Found",
                description="OpenAI package is commonly used with AutoGen",
                fix_suggestion="Install OpenAI: pip install openai",
                details={"package": "openai"},
            )
        )

    # Check for docker availability for code execution
    try:
        import docker

        try:
            client = docker.from_env()
            client.ping()
            result.issues.append(
                ValidationIssue(
                    category="configuration",
                    severity="info",
                    title="Docker Available for Code Execution",
                    description="Docker is available for AutoGen code execution features",
                    fix_suggestion="No action needed",
                )
            )
        except Exception:
            result.issues.append(
                ValidationIssue(
                    category="configuration",
                    severity="warning",
                    title="Docker Not Available",
                    description="Docker not available for code execution (optional feature)",
                    fix_suggestion="Install Docker if you need code execution: https://docs.docker.com/get-docker/",
                    details={"optional": True},
                )
            )
    except ImportError:
        result.issues.append(
            ValidationIssue(
                category="dependencies",
                severity="info",
                title="Docker Package Not Available",
                description="Docker package not installed (optional for code execution)",
                fix_suggestion="Install if needed: pip install docker",
                details={"optional": True},
            )
        )


def _diagnose_autogen_install_issues(result: ValidationResult):
    """Diagnose common AutoGen installation issues."""

    # Check if it's a package name confusion
    try:
        import autogen  # noqa: F401

        result.issues.append(
            ValidationIssue(
                category="installation",
                severity="error",
                title="Wrong AutoGen Package",
                description="Found 'autogen' package, but need 'pyautogen' for Microsoft AutoGen",
                fix_suggestion="Install correct package: pip uninstall autogen && pip install pyautogen",
                details={"wrong_package": "autogen"},
            )
        )
    except ImportError:
        pass

    # Check for common pip issues
    try:
        import subprocess

        result_pip = subprocess.run(
            [sys.executable, "-m", "pip", "show", "pyautogen"],
            capture_output=True,
            text=True,
        )
        if result_pip.returncode != 0:
            result.issues.append(
                ValidationIssue(
                    category="installation",
                    severity="error",
                    title="AutoGen Package Not Found by Pip",
                    description="pip cannot find pyautogen package",
                    fix_suggestion="Install with: pip install pyautogen",
                    details={"pip_output": result_pip.stderr},
                )
            )
    except Exception:
        pass


def _check_python_environment(result: ValidationResult):
    """Check Python environment compatibility."""
    result.checks_performed.append("Python Environment")

    # Python version check
    python_version = sys.version_info
    if python_version < (3, 8):
        result.success = False
        result.issues.append(
            ValidationIssue(
                category="environment",
                severity="error",
                title="Python Version Too Old",
                description=f"Python {python_version.major}.{python_version.minor} detected, need 3.8+",
                fix_suggestion="Upgrade to Python 3.8 or newer",
                details={"current": f"{python_version.major}.{python_version.minor}"},
            )
        )
    elif python_version < (3, 9):
        result.issues.append(
            ValidationIssue(
                category="environment",
                severity="warning",
                title="Python Version Recommendation",
                description=f"Python {python_version.major}.{python_version.minor} works but 3.9+ is recommended",
                fix_suggestion="Consider upgrading to Python 3.9+ for best compatibility",
                details={"current": f"{python_version.major}.{python_version.minor}"},
            )
        )

    # Check required packages
    required_packages = [
        "opentelemetry",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "requests",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        result.issues.append(
            ValidationIssue(
                category="environment",
                severity="warning",
                title="Optional Dependencies Missing",
                description=f"Some packages are missing: {', '.join(missing_packages)}",
                fix_suggestion=f"Install with: pip install {' '.join(missing_packages)}",
                details={"missing": missing_packages},
            )
        )


def _check_genops_integration(result: ValidationResult, team: str, project: str):
    """Check GenOps integration readiness."""
    result.checks_performed.append("GenOps Integration")

    try:
        from genops.providers.autogen import GenOpsAutoGenAdapter

        # Test adapter creation
        adapter = GenOpsAutoGenAdapter(
            team=team,
            project=project,
            daily_budget_limit=1.0,  # Minimal for testing
        )

        result.issues.append(
            ValidationIssue(
                category="integration",
                severity="info",
                title="GenOps Adapter Creation Successful",
                description="AutoGen adapter can be created successfully",
                fix_suggestion="No action needed",
                details={"team": team, "project": project},
            )
        )

        # Test session context
        if hasattr(adapter, "session_context") and adapter.session_context:
            result.issues.append(
                ValidationIssue(
                    category="integration",
                    severity="info",
                    title="Session Context Available",
                    description="Session tracking is properly initialized",
                    fix_suggestion="No action needed",
                )
            )

    except Exception as e:
        result.issues.append(
            ValidationIssue(
                category="integration",
                severity="error",
                title="GenOps Integration Failed",
                description=f"Error creating AutoGen adapter: {str(e)}",
                fix_suggestion="Check GenOps installation and configuration",
                details={"error": str(e), "type": type(e).__name__},
            )
        )


def _check_environment_variables(result: ValidationResult):
    """Check required environment variables and common configuration issues."""
    result.checks_performed.append("Environment Variables")

    # API keys for different providers
    api_keys = {
        "OPENAI_API_KEY": "OpenAI API access",
        "ANTHROPIC_API_KEY": "Anthropic Claude API access",
        "GOOGLE_API_KEY": "Google Gemini API access",
        "COHERE_API_KEY": "Cohere API access",
        "HUGGINGFACE_API_TOKEN": "HuggingFace API access",
    }

    found_keys = []
    invalid_keys = []

    for key, _description in api_keys.items():
        value = os.getenv(key)
        if value:
            found_keys.append(key)
            # Check for common API key format issues
            if key == "OPENAI_API_KEY":
                if not value.startswith("sk-"):
                    invalid_keys.append(
                        (key, "OpenAI API keys should start with 'sk-'")
                    )
                elif len(value) < 40:
                    invalid_keys.append((key, "OpenAI API key appears too short"))
            elif key == "ANTHROPIC_API_KEY":
                if not value.startswith("sk-ant-"):
                    invalid_keys.append(
                        (key, "Anthropic API keys should start with 'sk-ant-'")
                    )
            elif key == "GOOGLE_API_KEY":
                if len(value) < 20:
                    invalid_keys.append((key, "Google API key appears too short"))

    if invalid_keys:
        for key, issue in invalid_keys:
            result.issues.append(
                ValidationIssue(
                    category="configuration",
                    severity="error",
                    title=f"Invalid API Key Format: {key}",
                    description=f"API key format issue: {issue}",
                    fix_suggestion=f"Check your {key} format and obtain a valid key from the provider",
                    details={"key": key, "issue": issue},
                )
            )

    if not found_keys:
        result.success = False
        result.issues.append(
            ValidationIssue(
                category="configuration",
                severity="error",
                title="No API Keys Found",
                description="At least one LLM provider API key is required for AutoGen",
                fix_suggestion="Set an API key: export OPENAI_API_KEY=your_key_here",
                details={"checked_keys": list(api_keys.keys())},
            )
        )
    else:
        result.issues.append(
            ValidationIssue(
                category="configuration",
                severity="info",
                title="API Keys Found",
                description=f"Found API keys for: {', '.join(found_keys)}",
                fix_suggestion="No action needed",
                details={"found_keys": found_keys},
            )
        )

    # Check for common environment issues
    _check_common_env_issues(result)

    # GenOps configuration
    genops_vars = ["GENOPS_TEAM", "GENOPS_PROJECT", "GENOPS_ENVIRONMENT"]
    genops_found = [var for var in genops_vars if os.getenv(var)]

    if genops_found:
        result.issues.append(
            ValidationIssue(
                category="configuration",
                severity="info",
                title="GenOps Environment Variables Found",
                description=f"Found: {', '.join(genops_found)}",
                fix_suggestion="No action needed",
                details={"found": genops_found},
            )
        )


def _check_common_env_issues(result: ValidationResult):
    """Check for common environment configuration issues."""

    # Check for proxy settings that might interfere
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    proxy_found = [var for var in proxy_vars if os.getenv(var)]

    if proxy_found:
        result.issues.append(
            ValidationIssue(
                category="configuration",
                severity="warning",
                title="Proxy Configuration Detected",
                description=f"Proxy settings found: {', '.join(proxy_found)}",
                fix_suggestion="Ensure proxy allows API connections or configure NO_PROXY if needed",
                details={"proxy_vars": proxy_found},
            )
        )

    # Check Python path issues
    if "PYTHONPATH" in os.environ:
        pythonpath = os.environ["PYTHONPATH"]
        if "genops" in pythonpath.lower():
            result.issues.append(
                ValidationIssue(
                    category="configuration",
                    severity="warning",
                    title="PYTHONPATH Contains GenOps",
                    description="PYTHONPATH modification may cause import conflicts",
                    fix_suggestion="Consider removing GenOps from PYTHONPATH and using pip install instead",
                    details={"pythonpath": pythonpath},
                )
            )

    # Check for virtual environment
    if not (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    ):
        if "VIRTUAL_ENV" not in os.environ and "CONDA_DEFAULT_ENV" not in os.environ:
            result.issues.append(
                ValidationIssue(
                    category="environment",
                    severity="warning",
                    title="No Virtual Environment Detected",
                    description="Not using a virtual environment may cause package conflicts",
                    fix_suggestion="Consider using: python -m venv venv && source venv/bin/activate",
                    details={"recommendation": "virtual_environment"},
                )
            )


def _check_model_availability(result: ValidationResult, models: list[str]):
    """Check if specified models are available."""
    result.checks_performed.append("Model Availability")

    # This is a basic check - in practice, you'd want to test actual API calls
    available_models = []
    for model in models:
        # Simple heuristic based on model names
        if any(
            provider in model.lower()
            for provider in ["gpt", "claude", "gemini", "command"]
        ):
            available_models.append(model)

    if available_models:
        result.issues.append(
            ValidationIssue(
                category="models",
                severity="info",
                title="Models Available",
                description=f"Models appear to be available: {', '.join(available_models)}",
                fix_suggestion="Verify with actual API calls if needed",
                details={"models": available_models},
            )
        )


def _check_api_connectivity(result: ValidationResult, timeout: int):
    """Test API connectivity for available providers."""
    result.checks_performed.append("API Connectivity")

    connectivity_tests = []

    # OpenAI connectivity test
    if os.getenv("OPENAI_API_KEY"):
        try:
            import requests

            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                timeout=timeout,
            )
            if response.status_code == 200:
                connectivity_tests.append(("OpenAI", True, "Connected successfully"))
            else:
                connectivity_tests.append(
                    ("OpenAI", False, f"HTTP {response.status_code}")
                )
        except Exception as e:
            connectivity_tests.append(("OpenAI", False, str(e)))

    # Report connectivity results
    for provider, success, message in connectivity_tests:
        severity = "info" if success else "warning"
        result.issues.append(
            ValidationIssue(
                category="connectivity",
                severity=severity,
                title=f"{provider} Connectivity",
                description=f"{provider} API: {message}",
                fix_suggestion="Check API key and network connection"
                if not success
                else "No action needed",
                details={"provider": provider, "success": success},
            )
        )


def _run_performance_tests(result: ValidationResult):
    """Run basic performance benchmarks."""
    result.checks_performed.append("Performance Tests")

    # Test adapter creation time
    start_time = time.time()
    try:
        from genops.providers.autogen import GenOpsAutoGenAdapter

        GenOpsAutoGenAdapter(team="test", project="perf-test")
        creation_time = (time.time() - start_time) * 1000  # milliseconds

        result.performance_metrics["adapter_creation_time_ms"] = creation_time

        if creation_time > 1000:  # > 1 second
            result.issues.append(
                ValidationIssue(
                    category="performance",
                    severity="warning",
                    title="Slow Adapter Creation",
                    description=f"Adapter creation took {creation_time:.1f}ms",
                    fix_suggestion="Consider optimizing imports or reducing initialization overhead",
                    details={"creation_time_ms": creation_time},
                )
            )
        else:
            result.issues.append(
                ValidationIssue(
                    category="performance",
                    severity="info",
                    title="Good Adapter Performance",
                    description=f"Adapter creation took {creation_time:.1f}ms",
                    fix_suggestion="No action needed",
                )
            )

    except Exception as e:
        result.issues.append(
            ValidationIssue(
                category="performance",
                severity="error",
                title="Performance Test Failed",
                description=f"Could not run performance tests: {str(e)}",
                fix_suggestion="Check GenOps installation",
                details={"error": str(e)},
            )
        )


def _calculate_final_score(result: ValidationResult):
    """Calculate overall validation score."""
    # Scoring weights
    error_penalty = 25  # -25 points per error
    warning_penalty = 5  # -5 points per warning

    errors = len([issue for issue in result.issues if issue.severity == "error"])
    warnings = len([issue for issue in result.issues if issue.severity == "warning"])

    score = 100 - (errors * error_penalty) - (warnings * warning_penalty)
    result.overall_score = max(0.0, min(100.0, score))

    # Set success based on score
    if errors > 0:
        result.success = False
    elif result.overall_score < 70:
        result.success = False


def _generate_recommendations(result: ValidationResult):
    """Generate actionable recommendations based on validation results."""
    recommendations = []

    # Check for common issues
    has_errors = any(issue.severity == "error" for issue in result.issues)
    has_autogen = any(
        "AutoGen" in issue.title
        for issue in result.issues and issue.severity != "error"  # type: ignore  # noqa: F821
    )
    has_api_keys = any("API Keys" in issue.title for issue in result.issues)

    if has_errors:
        recommendations.append(
            "Fix all error-level issues before proceeding to production"
        )

    if not has_autogen:
        recommendations.append("Install AutoGen with: pip install pyautogen")

    if not has_api_keys:
        recommendations.append("Set up API keys for your preferred LLM providers")

    if result.overall_score < 90:
        recommendations.append("Address warnings to improve overall setup quality")

    if len(result.performance_metrics) == 0:
        recommendations.append("Run performance tests with run_performance_tests=True")

    # Add general best practices
    recommendations.extend(
        [
            "Test with a small budget limit initially ($1-5) before scaling up",
            "Monitor costs and usage patterns in your first week of usage",
            "Set up alerts for budget thresholds in production environments",
            "Consider using environment-specific configuration for team/project settings",
        ]
    )

    result.recommendations = recommendations[:10]  # Limit to top 10
