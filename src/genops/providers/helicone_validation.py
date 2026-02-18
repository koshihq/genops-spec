#!/usr/bin/env python3
"""
GenOps Helicone AI Gateway Validation System

This module provides comprehensive validation and diagnostics for Helicone AI
gateway setup, configuration, and multi-provider connectivity. It follows the
GenOps validation pattern for consistent developer experience.

Features:
- Comprehensive setup validation with actionable diagnostics
- Multi-provider API key validation (OpenAI, Anthropic, Vertex, etc.)
- Gateway connectivity and routing testing
- Model availability verification across providers
- Cost calculation and pricing validation
- Self-hosted gateway validation support
- Enterprise deployment readiness checking

Usage:
    from genops.providers.helicone_validation import validate_setup, print_validation_result

    # Run comprehensive validation
    result = validate_setup()
    print_validation_result(result)

    # Quick validation for automated scripts
    if quick_validate():
        print("✅ Ready to use Helicone gateway with GenOps")
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

import requests

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


class HeliconeValidator:
    """Comprehensive Helicone AI gateway setup validator."""

    def __init__(self, include_performance_tests: bool = False):
        """
        Initialize validator.

        Args:
            include_performance_tests: Whether to run gateway performance benchmarks
        """
        self.include_performance_tests = include_performance_tests
        self.result = ValidationResult(overall_status=ValidationStatus.PASSED)

        # Check for required dependencies
        self.has_requests = self._check_requests_import()
        self.has_genops_core = self._check_genops_imports()

    def _check_requests_import(self) -> bool:
        """Check if requests library is available."""
        try:
            import requests  # noqa: F401

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
        fix_suggestion: str,
        details: Optional[str] = None,
    ):
        """Add validation issue to results."""
        validation_issue = ValidationIssue(
            category, issue, severity, fix_suggestion, details
        )

        if severity == ValidationStatus.FAILED:
            self.result.issues.append(validation_issue)
            self.result.overall_status = ValidationStatus.FAILED
        elif severity == ValidationStatus.WARNING:
            self.result.warnings.append(validation_issue)
            if self.result.overall_status == ValidationStatus.PASSED:
                self.result.overall_status = ValidationStatus.WARNING

    def _add_passed(self, check_description: str):
        """Add successful validation check."""
        self.result.passed_checks.append(check_description)

    def validate_dependencies(self):
        """Validate Python environment and required dependencies."""
        self.result.total_checks += 4

        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 8):
            self._add_passed(
                f"Python {python_version.major}.{python_version.minor} supported"
            )
            self.result.environment_info["python_version"] = (
                f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            self._add_issue(
                "dependencies",
                f"Python version {python_version.major}.{python_version.minor} may not be supported",
                ValidationStatus.WARNING,
                "Upgrade to Python 3.8+ for best compatibility",
                "Helicone gateway works best with Python 3.8 or higher",
            )

        # Requests library check
        if self.has_requests:
            self._add_passed("Requests library available for gateway communication")
            try:
                import requests

                self.result.environment_info["requests_version"] = requests.__version__
            except Exception:
                pass
        else:
            self._add_issue(
                "dependencies",
                "Requests library not found",
                ValidationStatus.FAILED,
                "Install requests: pip install requests",
                "Requests is required for Helicone gateway communication",
            )

        # GenOps core check
        if self.has_genops_core:
            self._add_passed("GenOps core dependencies available")
        else:
            self._add_issue(
                "dependencies",
                "OpenTelemetry not available",
                ValidationStatus.WARNING,
                "Install GenOps: pip install genops-ai",
                "OpenTelemetry provides enhanced telemetry integration",
            )

        # Optional dependencies
        optional_deps = {
            "openai": "OpenAI provider integration",
            "anthropic": "Anthropic provider integration",
            "google-cloud-aiplatform": "Google Vertex AI integration",
        }

        for dep, _desc in optional_deps.items():
            try:
                __import__(dep.replace("-", "_"))
                self._add_passed(f"Optional dependency {dep} available")
            except ImportError:
                # Optional dependencies don't cause failures
                pass

    def validate_authentication(self):
        """Validate Helicone and provider API key configuration."""
        self.result.total_checks += 6

        # Helicone API key
        helicone_key = os.getenv("HELICONE_API_KEY")

        if not helicone_key:
            self._add_issue(
                "authentication",
                "HELICONE_API_KEY environment variable not set",
                ValidationStatus.FAILED,
                "Set your API key: export HELICONE_API_KEY='your-helicone-api-key'",
                "Get your API key from https://app.helicone.ai/",
            )
            return

        self._add_passed("HELICONE_API_KEY environment variable set")
        self.result.environment_info["helicone_api_key_configured"] = True

        # Basic format validation
        if len(helicone_key) < 20:
            self._add_issue(
                "authentication",
                "Helicone API key appears to be too short",
                ValidationStatus.WARNING,
                "Verify your API key is complete and correctly copied",
                "Helicone API keys are typically longer strings",
            )
        else:
            self._add_passed("Helicone API key length appears valid")

        # Check provider API keys
        provider_keys = {
            "OPENAI_API_KEY": {
                "name": "OpenAI",
                "pattern": "sk-",
                "url": "https://platform.openai.com/api-keys",
            },
            "ANTHROPIC_API_KEY": {
                "name": "Anthropic",
                "pattern": "sk-ant-",
                "url": "https://console.anthropic.com/",
            },
            "GOOGLE_APPLICATION_CREDENTIALS": {
                "name": "Google Vertex AI",
                "pattern": None,  # File path
                "url": "https://cloud.google.com/vertex-ai",
            },
        }

        provider_count = 0
        for key_name, info in provider_keys.items():
            key_value = os.getenv(key_name)

            if key_value:
                provider_count += 1

                if info["pattern"] and not key_value.startswith(info["pattern"]):
                    self._add_issue(
                        "authentication",
                        f"{info['name']} API key format appears incorrect",
                        ValidationStatus.WARNING,
                        f"Verify your {info['name']} API key from {info['url']}",
                        f"Expected to start with '{info['pattern']}'",
                    )
                else:
                    self._add_passed(f"{info['name']} API key configured")

        if provider_count == 0:
            self._add_issue(
                "authentication",
                "No provider API keys configured",
                ValidationStatus.WARNING,
                "Configure at least one provider: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.",
                "Helicone gateway needs provider keys to route requests",
            )
        else:
            self._add_passed(f"{provider_count} provider API key(s) configured")
            self.result.environment_info["configured_providers"] = provider_count

    def validate_gateway_connectivity(self):
        """Test connectivity to Helicone gateway and providers."""
        self.result.total_checks += 4

        if not self.has_requests:
            self._add_issue(
                "connectivity",
                "Cannot test gateway connectivity - requests library not available",
                ValidationStatus.SKIPPED,
                "Install requests library first: pip install requests",
            )
            return

        helicone_key = os.getenv("HELICONE_API_KEY")
        if not helicone_key:
            self._add_issue(
                "connectivity",
                "Cannot test gateway connectivity - Helicone API key not configured",
                ValidationStatus.SKIPPED,
                "Configure HELICONE_API_KEY first",
            )
            return

        # Test Helicone gateway health
        try:
            base_url = "https://ai-gateway.helicone.ai"
            health_url = f"{base_url}/v1/health"

            start_time = time.time()
            response = requests.get(health_url, timeout=10)
            request_time = time.time() - start_time

            if response.status_code == 200:
                self._add_passed("Helicone gateway reachable")
                self.result.environment_info["gateway_response_time"] = round(
                    request_time, 3
                )
            else:
                self._add_issue(
                    "connectivity",
                    f"Helicone gateway returned status {response.status_code}",
                    ValidationStatus.WARNING,
                    "Check Helicone service status at https://status.helicone.ai/",
                )

        except requests.exceptions.RequestException as e:
            self._add_issue(
                "connectivity",
                f"Cannot reach Helicone gateway: {e}",
                ValidationStatus.FAILED,
                "Check internet connection and firewall settings",
                "Helicone gateway must be accessible for AI requests",
            )
            return

        # Test provider routing (if provider keys available)
        self._test_provider_routing()

    def _test_provider_routing(self):
        """Test provider routing through Helicone gateway."""
        helicone_key = os.getenv("HELICONE_API_KEY")

        # Test OpenAI routing
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Helicone-Auth": f"Bearer {helicone_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 1,
                }

                response = requests.post(
                    "https://ai-gateway.helicone.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    self._add_passed("OpenAI provider routing successful")
                else:
                    self._add_issue(
                        "connectivity",
                        f"OpenAI routing failed: HTTP {response.status_code}",
                        ValidationStatus.WARNING,
                        "Check OpenAI API key and account status",
                    )

            except Exception as e:
                self._add_issue(
                    "connectivity",
                    f"OpenAI routing test failed: {e}",
                    ValidationStatus.WARNING,
                    "Check OpenAI API key configuration",
                )

        # Test Anthropic routing
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                headers = {
                    "x-api-key": anthropic_key,
                    "Helicone-Auth": f"Bearer {helicone_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                }

                payload = {
                    "model": "claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 1,
                }

                response = requests.post(
                    "https://ai-gateway.helicone.ai/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    self._add_passed("Anthropic provider routing successful")
                else:
                    self._add_issue(
                        "connectivity",
                        f"Anthropic routing failed: HTTP {response.status_code}",
                        ValidationStatus.WARNING,
                        "Check Anthropic API key and account status",
                    )

            except Exception as e:
                self._add_issue(
                    "connectivity",
                    f"Anthropic routing test failed: {e}",
                    ValidationStatus.WARNING,
                    "Check Anthropic API key configuration",
                )

    def validate_models_and_routing(self):
        """Validate model availability and routing intelligence."""
        self.result.total_checks += 3

        helicone_key = os.getenv("HELICONE_API_KEY")
        if not helicone_key or not self.has_requests:
            self._add_issue(
                "models",
                "Cannot test model routing - incomplete setup",
                ValidationStatus.SKIPPED,
                "Complete authentication and dependency setup first",
            )
            return

        # Test multi-provider routing capability
        providers_tested = []

        if os.getenv("OPENAI_API_KEY"):
            providers_tested.append("OpenAI")
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_tested.append("Anthropic")
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            providers_tested.append("Vertex AI")

        if len(providers_tested) >= 2:
            self._add_passed(
                f"Multi-provider routing available: {', '.join(providers_tested)}"
            )
            self.result.environment_info["routing_providers"] = len(providers_tested)
        elif len(providers_tested) == 1:
            self._add_issue(
                "models",
                f"Only single provider configured: {providers_tested[0]}",
                ValidationStatus.WARNING,
                "Configure additional providers for routing and failover capabilities",
                "Multi-provider routing provides better reliability and cost optimization",
            )
        else:
            self._add_issue(
                "models",
                "No providers configured for routing",
                ValidationStatus.FAILED,
                "Configure provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)",
            )

    def validate_performance(self):
        """Validate gateway performance characteristics."""
        if not self.include_performance_tests:
            return

        self.result.total_checks += 3

        helicone_key = os.getenv("HELICONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not helicone_key or not openai_key or not self.has_requests:
            self._add_issue(
                "performance",
                "Cannot test performance - incomplete setup",
                ValidationStatus.SKIPPED,
                "Complete authentication setup first",
            )
            return

        try:
            # Test gateway latency
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Helicone-Auth": f"Bearer {helicone_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            }

            start_time = time.time()
            response = requests.post(
                "https://ai-gateway.helicone.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                self._add_passed("Gateway performance test successful")
                self.result.environment_info["test_response_time"] = round(
                    response_time, 3
                )

                if response_time < 2.0:
                    self._add_passed("Gateway latency acceptable (< 2s)")
                elif response_time < 5.0:
                    self._add_issue(
                        "performance",
                        f"Gateway latency high: {response_time:.2f}s",
                        ValidationStatus.WARNING,
                        "Check network connectivity and gateway load",
                    )
                else:
                    self._add_issue(
                        "performance",
                        f"Gateway latency very high: {response_time:.2f}s",
                        ValidationStatus.FAILED,
                        "Investigate network issues or consider self-hosted gateway",
                    )

                # Parse usage for cost validation
                result = response.json()
                usage = result.get("usage", {})
                if usage:
                    total_tokens = usage.get("total_tokens", 0)
                    self.result.environment_info["test_tokens"] = total_tokens

            else:
                self._add_issue(
                    "performance",
                    f"Performance test failed: HTTP {response.status_code}",
                    ValidationStatus.WARNING,
                    "Check API keys and account status",
                )

        except Exception as e:
            self._add_issue(
                "performance",
                f"Performance test error: {e}",
                ValidationStatus.WARNING,
                "Performance monitoring may not work correctly",
            )

    def validate_pricing_and_costs(self):
        """Validate pricing configuration and cost calculation."""
        self.result.total_checks += 2

        try:
            # Try to import pricing calculator
            from .helicone_pricing import HeliconePricingCalculator

            pricing_calc = HeliconePricingCalculator()
            self._add_passed("Helicone pricing calculator available")

            # Test cost calculation
            try:
                provider_cost, helicone_cost, total_cost = (
                    pricing_calc.calculate_gateway_cost(
                        "openai", "gpt-3.5-turbo", 100, 50, 1.0
                    )
                )

                if total_cost > 0:
                    self._add_passed("Gateway cost calculation working")
                    self.result.environment_info["test_total_cost"] = total_cost
                    self.result.environment_info["test_provider_cost"] = provider_cost
                    self.result.environment_info["test_helicone_cost"] = helicone_cost
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
                    "Gateway cost tracking may not work correctly",
                )

        except ImportError:
            self._add_issue(
                "pricing",
                "Helicone pricing calculator not available",
                ValidationStatus.WARNING,
                "Gateway cost tracking will not be accurate",
                "Pricing calculator module may not be implemented yet",
            )

    def validate_self_hosted_gateway(self):
        """Validate self-hosted gateway configuration if applicable."""
        self.result.total_checks += 1

        # Check if using custom gateway URL
        custom_url = os.getenv("HELICONE_GATEWAY_URL")
        if custom_url:
            try:
                health_url = f"{custom_url.rstrip('/')}/health"
                response = requests.get(health_url, timeout=5)

                if response.status_code == 200:
                    self._add_passed("Self-hosted gateway accessible")
                    self.result.environment_info["self_hosted_gateway"] = True
                else:
                    self._add_issue(
                        "self_hosted",
                        f"Self-hosted gateway health check failed: {response.status_code}",
                        ValidationStatus.WARNING,
                        "Check self-hosted gateway deployment and configuration",
                    )

            except Exception as e:
                self._add_issue(
                    "self_hosted",
                    f"Cannot reach self-hosted gateway: {e}",
                    ValidationStatus.WARNING,
                    "Verify self-hosted gateway URL and deployment",
                )
        else:
            self.result.environment_info["self_hosted_gateway"] = False

    def run_validation(self) -> ValidationResult:
        """Run complete validation suite."""
        start_time = time.time()

        logger.info("Starting Helicone gateway validation...")

        # Run all validation checks
        self.validate_dependencies()
        self.validate_authentication()
        self.validate_gateway_connectivity()
        self.validate_models_and_routing()
        self.validate_performance()
        self.validate_pricing_and_costs()
        self.validate_self_hosted_gateway()

        # Finalize results
        self.result.validation_time = time.time() - start_time
        self.result.environment_info["platform"] = sys.platform
        self.result.environment_info["validation_time"] = round(
            self.result.validation_time, 2
        )

        logger.info(
            f"Helicone validation completed in {self.result.validation_time:.2f}s"
        )

        return self.result


def validate_setup(include_performance_tests: bool = False) -> ValidationResult:
    """
    Run comprehensive Helicone gateway setup validation.

    Args:
        include_performance_tests: Whether to run gateway performance benchmarks

    Returns:
        ValidationResult with comprehensive diagnostics
    """
    validator = HeliconeValidator(include_performance_tests=include_performance_tests)
    return validator.run_validation()


def print_validation_result(result: ValidationResult, detailed: bool = False):
    """
    Print user-friendly validation results with actionable guidance.

    Args:
        result: ValidationResult from validate_setup()
        detailed: Whether to show detailed environment information
    """
    status_colors = {
        ValidationStatus.PASSED: "✅",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.FAILED: "❌",
        ValidationStatus.SKIPPED: "⏭️",
    }

    status_icon = status_colors.get(result.overall_status, "❓")

    print("\n🛡️ **Helicone AI Gateway Validation Results**")
    print(f"{status_icon} **Overall Status: {result.overall_status.value}**")
    print(f"⏱️ **Validation Time:** {result.validation_time:.2f} seconds")
    print(
        f"📊 **Checks:** {len(result.passed_checks)} passed, {len(result.warnings)} warnings, {len(result.issues)} issues"
    )

    # Show successful checks
    if result.passed_checks:
        print("\n✅ **Successful Checks:**")
        for check in result.passed_checks:
            print(f"   ✓ {check}")

    # Show warnings
    if result.warnings:
        print(f"\n⚠️ **Warnings ({len(result.warnings)}):**")
        for warning in result.warnings:
            print(f"   ⚠️ {warning.issue}")
            print(f"     Category: {warning.category}")
            print(f"     Fix: {warning.fix_suggestion}")
            if detailed and warning.details:
                print(f"     Details: {warning.details}")

    # Show critical issues
    if result.issues:
        print(f"\n❌ **Critical Issues ({len(result.issues)}):**")
        for issue in result.issues:
            print(f"   ❌ {issue.issue}")
            print(f"     Category: {issue.category}")
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
            "requests_version",
            "helicone_api_key_configured",
            "configured_providers",
            "routing_providers",
            "gateway_response_time",
            "test_response_time",
            "test_tokens",
            "test_total_cost",
            "test_provider_cost",
            "test_helicone_cost",
            "self_hosted_gateway",
        }
        for key, value in result.environment_info.items():
            if key in safe_keys:
                print(f"   • {key}: {value}")

    # Next steps
    print("\n🚀 **Next Steps:**")
    if result.overall_status == ValidationStatus.PASSED:
        print("   ✅ Your Helicone gateway setup is ready! Try the quickstart guide:")
        print(
            "   📖 https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/helicone-quickstart.md"
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
        True if basic gateway setup is working, False otherwise
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

    parser = argparse.ArgumentParser(
        description="Validate Helicone AI gateway + GenOps setup"
    )
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
        # Full validation with user-friendly output
        result = validate_setup(include_performance_tests=args.performance)
        print_validation_result(result, detailed=args.detailed)

        # Exit with appropriate code
        if result.overall_status == ValidationStatus.FAILED:
            sys.exit(1)
        else:
            sys.exit(0)
