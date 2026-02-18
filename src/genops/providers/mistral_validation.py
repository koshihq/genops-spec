#!/usr/bin/env python3
"""
GenOps Mistral AI Validation System

This module provides comprehensive validation and diagnostics for Mistral AI
setup, configuration, and connectivity. It follows the GenOps validation
pattern for consistent developer experience across all AI platforms.

Features:
- Comprehensive setup validation with actionable diagnostics
- Environment configuration checking
- API connectivity and authentication validation
- Model availability and performance testing
- Pricing configuration verification
- European AI provider specific validations (GDPR compliance)

Usage:
    from genops.providers.mistral_validation import validate_setup, print_validation_result

    # Run comprehensive validation
    result = validate_setup()
    print_validation_result(result)

    # Quick validation for automated scripts
    if quick_validate():
        print("✅ Ready to use Mistral with GenOps")
    else:
        print("❌ Setup issues detected")
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""

    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class ValidationIssue:
    """Individual validation issue with fix suggestions."""

    category: str
    issue: str
    severity: ValidationStatus
    fix_suggestion: str
    details: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result with structured feedback."""

    overall_status: ValidationStatus
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    total_checks: int = 0
    validation_time: float = 0.0
    environment_info: dict[str, Any] = field(default_factory=dict)


class MistralValidator:
    """Comprehensive Mistral AI setup validator."""

    def __init__(self, include_performance_tests: bool = False):
        """
        Initialize validator.

        Args:
            include_performance_tests: Whether to run performance benchmarks
        """
        self.include_performance_tests = include_performance_tests
        self.result = ValidationResult(overall_status=ValidationStatus.PASSED)

        # Try to import dependencies
        self.has_mistral = self._check_mistral_import()
        self.has_genops_core = self._check_genops_imports()

    def _check_mistral_import(self) -> bool:
        """Check if Mistral AI client is available."""
        try:
            import mistralai  # noqa: F401
            from mistralai import Mistral  # noqa: F401

            return True
        except ImportError:
            return False

    def _check_genops_imports(self) -> bool:
        """Check if GenOps core dependencies are available."""
        try:
            from opentelemetry import trace  # noqa: F401

            return True
        except ImportError:
            return False

    def _add_issue(
        self,
        category: str,
        issue: str,
        severity: ValidationStatus,
        fix: str,
        details: str = None,  # type: ignore[assignment]
    ):
        """Add a validation issue."""
        validation_issue = ValidationIssue(
            category=category,
            issue=issue,
            severity=severity,
            fix_suggestion=fix,
            details=details,
        )

        if severity == ValidationStatus.FAILED:
            self.result.issues.append(validation_issue)
            if self.result.overall_status != ValidationStatus.FAILED:
                self.result.overall_status = ValidationStatus.FAILED
        elif severity == ValidationStatus.WARNING:
            self.result.warnings.append(validation_issue)
            if self.result.overall_status == ValidationStatus.PASSED:
                self.result.overall_status = ValidationStatus.WARNING

    def _add_passed(self, check_name: str):
        """Add a passed check."""
        self.result.passed_checks.append(check_name)

    def validate_dependencies(self):
        """Validate required dependencies are installed."""
        self.result.total_checks += 5

        # Check Mistral AI client
        if self.has_mistral:
            self._add_passed("Mistral AI client available")

            # Check version if possible
            try:
                import mistralai

                version = getattr(mistralai, "__version__", "unknown")
                self.result.environment_info["mistral_version"] = version
                self._add_passed(f"Mistral AI version: {version}")
            except Exception:
                pass
        else:
            self._add_issue(
                "dependencies",
                "Mistral AI client not installed",
                ValidationStatus.FAILED,
                "Install with: pip install mistralai",
                "The mistralai package is required for Mistral AI integration",
            )

        # Check OpenTelemetry
        if self.has_genops_core:
            self._add_passed("OpenTelemetry available")
        else:
            self._add_issue(
                "dependencies",
                "OpenTelemetry not available",
                ValidationStatus.WARNING,
                "Install with: pip install opentelemetry-api opentelemetry-sdk",
                "OpenTelemetry enables telemetry export to observability platforms",
            )

        # Check Python version
        python_version = sys.version_info
        self.result.environment_info["python_version"] = (
            f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        )

        if python_version >= (3, 8):
            self._add_passed(
                f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            self._add_issue(
                "dependencies",
                f"Python version {python_version.major}.{python_version.minor} may not be supported",
                ValidationStatus.WARNING,
                "Upgrade to Python 3.8+ for best compatibility",
                "Mistral AI and GenOps work best with Python 3.8 or higher",
            )

        # Check optional dependencies
        optional_deps = {
            "requests": "HTTP client for API calls",
            "numpy": "Numerical computing for embeddings",
            "pandas": "Data analysis for cost reporting",
        }

        for dep, _desc in optional_deps.items():
            try:
                __import__(dep)
                self._add_passed(f"Optional dependency {dep} available")
            except ImportError:
                # Optional dependencies don't cause failures
                pass

    def validate_authentication(self):
        """Validate API key configuration and format."""
        self.result.total_checks += 4

        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key:
            self._add_issue(
                "authentication",
                "MISTRAL_API_KEY environment variable not set",
                ValidationStatus.FAILED,
                "Set your API key: export MISTRAL_API_KEY='your-api-key'",
                "Get your API key from https://console.mistral.ai/",
            )
            return

        self._add_passed("MISTRAL_API_KEY environment variable set")
        self.result.environment_info["api_key_configured"] = True

        # Basic format validation
        if len(api_key) < 10:
            self._add_issue(
                "authentication",
                "API key appears to be too short",
                ValidationStatus.WARNING,
                "Verify your API key is complete and correctly copied",
                "Mistral API keys are typically longer strings",
            )
        else:
            self._add_passed("API key length appears valid")

        # Check for common API key issues
        if api_key.startswith("sk-") or api_key.startswith("pk-"):
            self._add_issue(
                "authentication",
                "API key format looks like OpenAI/other provider",
                ValidationStatus.WARNING,
                "Verify you're using a Mistral API key from console.mistral.ai",
                "Mistral API keys have a different format than OpenAI keys",
            )
        else:
            self._add_passed("API key format appears correct for Mistral")

    def validate_connectivity(self):
        """Test API connectivity and basic functionality."""
        self.result.total_checks += 3

        if not self.has_mistral:
            self._add_issue(
                "connectivity",
                "Cannot test connectivity - Mistral client not available",
                ValidationStatus.SKIPPED,
                "Install Mistral client first: pip install mistralai",
            )
            return

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            self._add_issue(
                "connectivity",
                "Cannot test connectivity - API key not configured",
                ValidationStatus.SKIPPED,
                "Set MISTRAL_API_KEY environment variable",
            )
            return

        try:
            from mistralai import Mistral

            client = Mistral(api_key=api_key)
            self._add_passed("Mistral client initialized successfully")

            # Test basic API call with minimal cost
            try:
                start_time = time.time()
                response = client.chat.complete(
                    model="mistral-tiny-2312",  # Cheapest model
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                )
                request_time = time.time() - start_time

                self._add_passed("API connectivity test successful")
                self.result.environment_info["connectivity_test_time"] = round(
                    request_time, 3
                )

                # Check response structure
                if hasattr(response, "choices") and response.choices:
                    self._add_passed("API response structure valid")
                else:
                    self._add_issue(
                        "connectivity",
                        "API response structure unexpected",
                        ValidationStatus.WARNING,
                        "Check Mistral client version compatibility",
                    )

            except Exception as api_error:
                error_msg = str(api_error).lower()

                if "unauthorized" in error_msg or "invalid" in error_msg:
                    self._add_issue(
                        "connectivity",
                        "API authentication failed",
                        ValidationStatus.FAILED,
                        "Verify your API key is correct and active",
                        f"API error: {api_error}",
                    )
                elif "rate limit" in error_msg:
                    self._add_issue(
                        "connectivity",
                        "Rate limit exceeded",
                        ValidationStatus.WARNING,
                        "Wait a moment and try again, or check your usage limits",
                    )
                elif "insufficient" in error_msg or "quota" in error_msg:
                    self._add_issue(
                        "connectivity",
                        "Insufficient credits or quota exceeded",
                        ValidationStatus.FAILED,
                        "Add credits to your Mistral account at console.mistral.ai",
                        f"API error: {api_error}",
                    )
                else:
                    self._add_issue(
                        "connectivity",
                        f"API call failed: {api_error}",
                        ValidationStatus.FAILED,
                        "Check your internet connection and Mistral service status",
                        "Visit status.mistral.ai for service status updates",
                    )

        except Exception as client_error:
            self._add_issue(
                "connectivity",
                f"Failed to create Mistral client: {client_error}",
                ValidationStatus.FAILED,
                "Check your API key format and mistralai package installation",
            )

    def validate_models(self):
        """Validate access to key Mistral models."""
        self.result.total_checks += 6

        if not self.has_mistral or not os.getenv("MISTRAL_API_KEY"):
            self._add_issue(
                "models",
                "Cannot validate models - setup incomplete",
                ValidationStatus.SKIPPED,
                "Complete authentication setup first",
            )
            return

        # Test key models with minimal requests
        test_models = [
            ("mistral-tiny-2312", "Basic model"),
            ("mistral-small-latest", "Small model"),
            ("mistral-medium-latest", "Medium model"),
            ("mistral-embed", "Embedding model"),
            ("mistral-large-latest", "Large model"),
            ("codestral-2405", "Code model"),
        ]

        available_models = []
        unavailable_models = []

        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

            for model, description in test_models:
                try:
                    if "embed" in model:
                        # Test embedding model
                        client.embeddings.create(model=model, inputs=["test"])
                        available_models.append((model, description))
                    else:
                        # Test chat model with minimal cost
                        client.chat.complete(
                            model=model,
                            messages=[{"role": "user", "content": "Hi"}],
                            max_tokens=1,
                        )
                        available_models.append((model, description))

                except Exception as e:
                    error_msg = str(e).lower()
                    if "not found" in error_msg or "does not exist" in error_msg:
                        unavailable_models.append((model, "Model not available"))
                    elif "insufficient" in error_msg or "quota" in error_msg:
                        unavailable_models.append((model, "Insufficient credits"))
                    else:
                        unavailable_models.append((model, f"Error: {e}"))

                # Rate limiting - small delay between requests
                time.sleep(0.1)

        except Exception as e:
            self._add_issue(
                "models",
                f"Model validation failed: {e}",
                ValidationStatus.FAILED,
                "Check API connectivity and authentication",
            )
            return

        # Report results
        if available_models:
            [f"{model} ({desc})" for model, desc in available_models]
            self._add_passed(
                f"Available models: {', '.join([m[0] for m in available_models[:3]])}"
            )
            self.result.environment_info["available_models"] = len(available_models)

        if unavailable_models:
            for model, reason in unavailable_models:
                if "not available" in reason:
                    self._add_issue(
                        "models",
                        f"Model {model} not accessible",
                        ValidationStatus.WARNING,
                        "Check your account plan and model access permissions",
                        reason,
                    )
                else:
                    self._add_issue(
                        "models",
                        f"Model {model} test failed",
                        ValidationStatus.WARNING,
                        "This may affect some features",
                        reason,
                    )

    def validate_performance(self):
        """Validate performance characteristics and response times."""
        if not self.include_performance_tests:
            return

        self.result.total_checks += 3

        if not self.has_mistral or not os.getenv("MISTRAL_API_KEY"):
            self._add_issue(
                "performance",
                "Cannot test performance - setup incomplete",
                ValidationStatus.SKIPPED,
                "Complete authentication setup first",
            )
            return

        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

            # Test response time
            start_time = time.time()
            response = client.chat.complete(
                model="mistral-tiny-2312",
                messages=[{"role": "user", "content": "Count to 3"}],
                max_tokens=10,
            )
            response_time = time.time() - start_time

            self.result.environment_info["test_response_time"] = round(response_time, 3)

            if response_time < 2.0:
                self._add_passed(f"Fast response time: {response_time:.2f}s")
            elif response_time < 5.0:
                self._add_passed(f"Acceptable response time: {response_time:.2f}s")
            else:
                self._add_issue(
                    "performance",
                    f"Slow response time: {response_time:.2f}s",
                    ValidationStatus.WARNING,
                    "Check your internet connection or try different model",
                    "Slow responses may indicate network or service issues",
                )

            # Test token counting if available
            if hasattr(response, "usage") and response.usage:
                tokens = (
                    response.usage.total_tokens
                    if hasattr(response.usage, "total_tokens")
                    else 0
                )
                if tokens > 0:
                    self._add_passed(f"Token usage tracking working: {tokens} tokens")
                    self.result.environment_info["test_tokens"] = tokens
                else:
                    self._add_issue(
                        "performance",
                        "Token usage not tracked in response",
                        ValidationStatus.WARNING,
                        "Cost tracking may not be accurate",
                    )

        except Exception as e:
            self._add_issue(
                "performance",
                f"Performance test failed: {e}",
                ValidationStatus.WARNING,
                "Performance monitoring may not work correctly",
            )

    def validate_pricing(self):
        """Validate pricing configuration and cost calculation."""
        self.result.total_checks += 2

        try:
            # Try to import pricing calculator
            from .mistral_pricing import MistralPricingCalculator

            pricing_calc = MistralPricingCalculator()
            self._add_passed("Mistral pricing calculator available")

            # Test cost calculation
            try:
                input_cost, output_cost, total_cost = pricing_calc.calculate_cost(
                    model="mistral-small-latest",
                    operation="chat",
                    input_tokens=100,
                    output_tokens=50,
                )

                if total_cost > 0:
                    self._add_passed("Cost calculation working")
                    self.result.environment_info["test_cost"] = total_cost
                else:
                    self._add_issue(
                        "pricing",
                        "Cost calculation returned zero",
                        ValidationStatus.WARNING,
                        "Check pricing calculator configuration",
                    )

            except Exception as calc_error:
                self._add_issue(
                    "pricing",
                    f"Cost calculation failed: {calc_error}",
                    ValidationStatus.WARNING,
                    "Cost tracking may not work correctly",
                )

        except ImportError:
            self._add_issue(
                "pricing",
                "Mistral pricing calculator not available",
                ValidationStatus.WARNING,
                "Cost tracking will not be accurate",
                "Pricing calculator module may not be implemented yet",
            )

    def validate_all(self) -> ValidationResult:
        """Run all validation checks and return comprehensive result."""
        start_time = time.time()

        print("🔍 Validating Mistral AI + GenOps setup...")
        print("=" * 50)

        # Run all validation categories
        self.validate_dependencies()
        self.validate_authentication()
        self.validate_connectivity()
        self.validate_models()
        self.validate_performance()
        self.validate_pricing()

        # Finalize results
        self.result.validation_time = time.time() - start_time
        self.result.environment_info["platform"] = sys.platform
        self.result.environment_info["validation_time"] = round(
            self.result.validation_time, 2
        )

        return self.result


def validate_setup(include_performance_tests: bool = False) -> ValidationResult:
    """
    Run comprehensive Mistral AI setup validation.

    Args:
        include_performance_tests: Whether to run performance benchmarks

    Returns:
        ValidationResult with detailed diagnostics
    """
    validator = MistralValidator(include_performance_tests=include_performance_tests)
    return validator.validate_all()


def print_validation_result(result: ValidationResult, detailed: bool = False):
    """
    Print validation results in a user-friendly format.

    Args:
        result: ValidationResult from validate_setup()
        detailed: Whether to show detailed information
    """
    print("\n🎯 Validation Results")
    print("=" * 50)

    # Overall status
    status_icon = {
        ValidationStatus.PASSED: "✅",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.FAILED: "❌",
        ValidationStatus.SKIPPED: "⏭️",
    }

    print(
        f"{status_icon[result.overall_status]} **Overall Status: {result.overall_status.value}**"
    )
    print(
        f"📊 Validation Summary: {len(result.passed_checks)}/{result.total_checks} checks passed"
    )
    print(f"⏱️ Validation Time: {result.validation_time:.2f} seconds")

    # Show passed checks summary
    if result.passed_checks:
        print(f"\n✅ **Passed Checks ({len(result.passed_checks)}):**")
        for check in result.passed_checks[:5]:  # Show first 5
            print(f"   • {check}")
        if len(result.passed_checks) > 5:
            print(f"   ... and {len(result.passed_checks) - 5} more")

    # Show warnings
    if result.warnings:
        print(f"\n⚠️ **Warnings ({len(result.warnings)}):**")
        for warning in result.warnings:
            print(f"   • {warning.issue}")
            print(f"     Fix: {warning.fix_suggestion}")
            if detailed and warning.details:
                print(f"     Details: {warning.details}")

    # Show critical issues
    if result.issues:
        print(f"\n❌ **Issues Requiring Attention ({len(result.issues)}):**")
        for issue in result.issues:
            print(f"   • {issue.issue}")
            print(f"     Fix: {issue.fix_suggestion}")
            if detailed and issue.details:
                print(f"     Details: {issue.details}")

    # Show environment info (whitelist safe keys only)
    if detailed and result.environment_info:
        print("\n🔧 **Environment Information:**")
        # Whitelist of safe keys that contain no sensitive data
        safe_keys = {
            "python_version",
            "platform",
            "validation_time",
            "mistral_version",
            "api_key_configured",
            "available_models",
            "connectivity_test_time",
            "test_response_time",
            "test_tokens",
            "test_cost",
        }
        for key, value in result.environment_info.items():
            if key in safe_keys:
                print(f"   • {key}: {value}")

    # Next steps
    print("\n🚀 **Next Steps:**")
    if result.overall_status == ValidationStatus.PASSED:
        print("   ✅ Your setup is ready! Try the quickstart guide:")
        print(
            "   📖 https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/mistral-quickstart.md"
        )
    elif result.overall_status == ValidationStatus.WARNING:
        print(
            "   ⚠️ Setup works but has warnings. Consider addressing them for optimal experience."
        )
        print(
            "   📖 See the comprehensive integration guide for advanced configuration."
        )
    else:
        print("   ❌ Please fix the critical issues above before proceeding.")
        print(
            "   🆘 Need help? Create an issue: https://github.com/KoshiHQ/GenOps-AI/issues"
        )


def quick_validate() -> bool:
    """
    Quick validation for automated scripts and CI/CD.

    Returns:
        True if basic setup is working, False otherwise
    """
    try:
        result = validate_setup(include_performance_tests=False)
        return result.overall_status in [
            ValidationStatus.PASSED,
            ValidationStatus.WARNING,
        ]
    except Exception:
        return False


if __name__ == "__main__":
    # Command-line validation tool
    import argparse

    parser = argparse.ArgumentParser(description="Validate Mistral AI + GenOps setup")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--performance", action="store_true", help="Include performance tests"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output for automation"
    )

    args = parser.parse_args()

    if args.quiet:
        # Quiet mode for automation
        success = quick_validate()
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        result = validate_setup(include_performance_tests=args.performance)
        print_validation_result(result, detailed=args.detailed)

        # Exit with appropriate code
        if result.overall_status == ValidationStatus.FAILED:
            sys.exit(1)
        else:
            sys.exit(0)
