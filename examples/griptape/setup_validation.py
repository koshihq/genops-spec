#!/usr/bin/env python3
"""
Griptape + GenOps Setup Validation

Comprehensive validation script to check Griptape integration setup,
diagnose common issues, and provide actionable fix recommendations.

Usage:
    python setup_validation.py

This script will:
- Check Griptape framework installation and version
- Validate GenOps provider availability
- Test API key configuration
- Verify instrumentation capabilities
- Provide detailed diagnostics and recommendations
"""

import importlib
import os
import sys
from typing import Any, Optional


def check_color_support() -> bool:
    """Check if terminal supports colors."""
    return (
        os.getenv("TERM") != "dumb"
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )


# Color codes if supported
if check_color_support():
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
else:
    GREEN = RED = YELLOW = BLUE = RESET = BOLD = ""


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{GREEN}✅ {text}{RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{RED}❌ {text}{RESET}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{BLUE}ℹ️  {text}{RESET}")


def check_python_version() -> dict[str, Any]:
    """Check Python version compatibility."""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    is_compatible = version_info >= (3, 9)

    return {
        "name": "Python Version",
        "passed": is_compatible,
        "version": version_str,
        "required": "3.9+",
        "message": f"Python {version_str}"
        + (" (compatible)" if is_compatible else " (too old)"),
        "recommendation": "Upgrade to Python 3.9 or higher"
        if not is_compatible
        else None,
    }


def check_package_installation(
    package_name: str, import_name: Optional[str] = None
) -> dict[str, Any]:
    """Check if a package is installed and importable."""
    import_name = import_name or package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")

        return {
            "name": f"{package_name.title()} Package",
            "passed": True,
            "version": version,
            "message": f"{package_name} {version} installed",
            "recommendation": None,
        }
    except ImportError as e:
        return {
            "name": f"{package_name.title()} Package",
            "passed": False,
            "version": None,
            "message": f"{package_name} not installed: {e}",
            "recommendation": f"Install with: pip install {package_name}",
        }


def check_griptape_structures() -> dict[str, Any]:
    """Check if core Griptape structures are available."""
    try:
        from griptape.structures import Agent, Pipeline, Workflow  # noqa: F401

        structures = []

        # Test Agent
        try:
            structures.append("Agent")
        except Exception:
            pass

        # Test Pipeline
        try:
            structures.append("Pipeline")
        except Exception:
            pass

        # Test Workflow
        try:
            structures.append("Workflow")
        except Exception:
            pass

        if structures:
            return {
                "name": "Griptape Structures",
                "passed": True,
                "structures": structures,
                "message": f"Available structures: {', '.join(structures)}",
                "recommendation": None,
            }
        else:
            return {
                "name": "Griptape Structures",
                "passed": False,
                "structures": [],
                "message": "No Griptape structures available",
                "recommendation": "Reinstall Griptape: pip install --upgrade griptape",
            }

    except ImportError as e:
        return {
            "name": "Griptape Structures",
            "passed": False,
            "structures": [],
            "message": f"Cannot import Griptape structures: {e}",
            "recommendation": "Install Griptape: pip install griptape",
        }


def check_genops_griptape_provider() -> dict[str, Any]:
    """Check if GenOps Griptape provider is available."""
    try:
        from genops.providers.griptape import (  # noqa: F401
            GenOpsGriptapeAdapter,
            auto_instrument,
        )

        # Test adapter creation (without actual instrumentation)
        try:
            GenOpsGriptapeAdapter(team="test", project="validation")
            adapter_available = True
        except Exception as e:
            adapter_available = False
            adapter_error = str(e)

        if adapter_available:
            return {
                "name": "GenOps Griptape Provider",
                "passed": True,
                "message": "GenOps Griptape provider available",
                "functions": ["auto_instrument", "GenOpsGriptapeAdapter"],
                "recommendation": None,
            }
        else:
            return {
                "name": "GenOps Griptape Provider",
                "passed": False,
                "message": f"GenOps Griptape provider has issues: {adapter_error}",
                "recommendation": "Reinstall GenOps: pip install --upgrade genops",
            }

    except ImportError as e:
        return {
            "name": "GenOps Griptape Provider",
            "passed": False,
            "message": f"Cannot import GenOps Griptape provider: {e}",
            "recommendation": "Install GenOps: pip install genops",
        }


def check_api_keys() -> dict[str, Any]:
    """Check LLM provider API keys."""
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "Cohere": os.getenv("COHERE_API_KEY"),
        "Mistral": os.getenv("MISTRAL_API_KEY"),
    }

    available_keys = {k: v for k, v in api_keys.items() if v}

    if available_keys:
        return {
            "name": "LLM Provider API Keys",
            "passed": True,
            "available": list(available_keys.keys()),
            "message": f"API keys found for: {', '.join(available_keys.keys())}",
            "recommendation": None,
        }
    else:
        return {
            "name": "LLM Provider API Keys",
            "passed": False,
            "available": [],
            "message": "No LLM provider API keys found",
            "recommendation": "Set at least one API key: export OPENAI_API_KEY='your-key'",
        }


def check_environment_variables() -> dict[str, Any]:
    """Check GenOps environment variables."""
    env_vars = {
        "GENOPS_TEAM": os.getenv("GENOPS_TEAM"),
        "GENOPS_PROJECT": os.getenv("GENOPS_PROJECT"),
        "GENOPS_ENVIRONMENT": os.getenv("GENOPS_ENVIRONMENT"),
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    }

    set_vars = {k: v for k, v in env_vars.items() if v}
    recommended_vars = ["GENOPS_TEAM", "GENOPS_PROJECT"]

    has_recommended = any(env_vars[var] for var in recommended_vars)

    return {
        "name": "GenOps Environment Variables",
        "passed": has_recommended,
        "set_vars": set_vars,
        "message": f"Set variables: {list(set_vars.keys())}"
        if set_vars
        else "No GenOps variables set",
        "recommendation": "Set recommended variables: export GENOPS_TEAM='your-team' GENOPS_PROJECT='your-project'"
        if not has_recommended
        else None,
    }


def test_basic_instrumentation() -> dict[str, Any]:
    """Test basic auto-instrumentation functionality."""
    try:
        from genops.providers.griptape import auto_instrument
        from genops.providers.griptape.registration import (
            disable_auto_instrument,
            is_instrumented,
        )

        # Test instrumentation enable/disable
        initial_state = is_instrumented()

        # Try to enable (with test config)
        try:
            auto_instrument(
                team="test-team",
                project="validation-test",
                enable_cost_tracking=False,  # Disable to avoid API calls
            )
            enabled_state = is_instrumented()

            # Try to disable
            disable_auto_instrument()
            disabled_state = is_instrumented()

            # Restore initial state
            if initial_state:
                auto_instrument(team="test-team", project="validation-test")

            if enabled_state and not disabled_state:
                return {
                    "name": "Auto-Instrumentation",
                    "passed": True,
                    "message": "Auto-instrumentation enable/disable works correctly",
                    "recommendation": None,
                }
            else:
                return {
                    "name": "Auto-Instrumentation",
                    "passed": False,
                    "message": f"Instrumentation state issues: enabled={enabled_state}, disabled={disabled_state}",
                    "recommendation": "Check GenOps Griptape provider installation",
                }

        except Exception as e:
            return {
                "name": "Auto-Instrumentation",
                "passed": False,
                "message": f"Instrumentation test failed: {e}",
                "recommendation": "Check all dependencies and API keys",
            }

    except ImportError as e:
        return {
            "name": "Auto-Instrumentation",
            "passed": False,
            "message": f"Cannot test instrumentation: {e}",
            "recommendation": "Install missing dependencies",
        }


def run_comprehensive_validation() -> list[dict[str, Any]]:
    """Run all validation checks."""
    checks = [
        check_python_version(),
        check_package_installation("genops"),
        check_package_installation("griptape"),
        check_griptape_structures(),
        check_genops_griptape_provider(),
        check_api_keys(),
        check_environment_variables(),
        test_basic_instrumentation(),
    ]

    return checks


def print_validation_results(checks: list[dict[str, Any]]) -> None:
    """Print detailed validation results."""

    print_header("🔍 Griptape + GenOps Setup Validation Results")

    passed_checks = 0
    total_checks = len(checks)

    for check in checks:
        if check["passed"]:
            print_success(f"{check['name']}: {check['message']}")
            passed_checks += 1
        else:
            print_error(f"{check['name']}: {check['message']}")
            if check.get("recommendation"):
                print(f"   💡 Fix: {check['recommendation']}")

    # Summary
    print_header("📊 Validation Summary")

    if passed_checks == total_checks:
        print_success(f"All {total_checks} checks passed! ✨")
        print_info("Your Griptape + GenOps setup is ready to use.")
    else:
        failed_checks = total_checks - passed_checks
        print_warning(
            f"{passed_checks}/{total_checks} checks passed, {failed_checks} failed"
        )
        print_info("Please address the failed checks above.")

    # Recommendations
    recommendations = [
        check.get("recommendation")
        for check in checks
        if not check["passed"] and check.get("recommendation")
    ]

    if recommendations:
        print_header("🚀 Next Steps")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    # Quick start if ready
    if passed_checks >= total_checks - 2:  # Allow minor issues
        print_header("🎯 Quick Start")
        print("Your setup looks good! Try these commands:")
        print()
        print("# Run basic example:")
        print("python examples/griptape/01_basic_agent.py")
        print()
        print("# Run auto-instrumentation example:")
        print("python examples/griptape/02_auto_instrumentation.py")
        print()
        print("# Read the full integration guide:")
        print("open docs/integrations/griptape.md")


def main():
    """Main validation function."""

    print(f"{BOLD}🤖 Griptape + GenOps Setup Validation{RESET}")
    print("Checking your installation and configuration...")

    try:
        checks = run_comprehensive_validation()
        print_validation_results(checks)

        # Exit code based on critical failures
        critical_failures = sum(
            1
            for check in checks
            if not check["passed"]
            and check["name"]
            in [
                "Python Version",
                "Genops Package",
                "Griptape Package",
                "GenOps Griptape Provider",
            ]
        )

        return critical_failures == 0

    except Exception as e:
        print_error(f"Validation script failed: {e}")
        print_info("This might indicate a serious installation issue.")
        return False


if __name__ == "__main__":
    success = main()

    print(f"\n{BOLD}Validation {'✅ PASSED' if success else '❌ FAILED'}{RESET}")

    if not success:
        print("\n🔧 For more help:")
        print(
            "  • Check the troubleshooting guide: docs/integrations/griptape.md#troubleshooting"
        )
        print("  • Open an issue: https://github.com/KoshiHQ/GenOps-AI/issues")
        print("  • Join discussions: https://github.com/KoshiHQ/GenOps-AI/discussions")

    exit(0 if success else 1)
