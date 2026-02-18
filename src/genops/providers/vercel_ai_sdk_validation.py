"""Vercel AI SDK validation and setup verification module."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None
    fix_suggestion: str | None = None


@dataclass
class SetupValidationSummary:
    """Summary of all validation checks."""

    all_passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: list[ValidationResult]
    overall_message: str


class VercelAISDKValidator:
    """
    Comprehensive validation system for Vercel AI SDK integration with GenOps.

    Validates environment setup, dependencies, configuration, and connectivity
    to ensure proper GenOps governance integration.
    """

    def __init__(self):
        """Initialize the validator."""
        self.validation_results: list[ValidationResult] = []

    def validate_setup(
        self,
        check_nodejs: bool = True,
        check_npm_packages: bool = True,
        check_python_deps: bool = True,
        check_environment: bool = True,
        check_genops_config: bool = True,
        check_provider_access: bool = False,  # Optional as it requires API keys
        verbose: bool = True,
    ) -> SetupValidationSummary:
        """
        Comprehensive setup validation for Vercel AI SDK integration.

        Args:
            check_nodejs: Validate Node.js installation
            check_npm_packages: Check for Vercel AI SDK npm packages
            check_python_deps: Validate Python dependencies
            check_environment: Check environment variables
            check_genops_config: Validate GenOps configuration
            check_provider_access: Test API connectivity (optional)
            verbose: Print detailed validation results

        Returns:
            SetupValidationSummary: Complete validation results
        """
        self.validation_results.clear()

        if verbose:
            print("🔍 GenOps Vercel AI SDK Setup Validation")
            print("=" * 50)

        # Core system checks
        if check_nodejs:
            self._validate_nodejs()
        if check_npm_packages:
            self._validate_npm_packages()
        if check_python_deps:
            self._validate_python_dependencies()

        # Configuration checks
        if check_environment:
            self._validate_environment_variables()
        if check_genops_config:
            self._validate_genops_configuration()

        # Optional connectivity checks
        if check_provider_access:
            self._validate_provider_access()

        # Generate summary
        summary = self._generate_validation_summary()

        if verbose:
            self._print_validation_summary(summary)

        return summary

    def _validate_nodejs(self) -> None:
        """Validate Node.js installation and version."""
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                # Extract version number
                version_num = version.lstrip("v").split(".")[0]
                if int(version_num) >= 16:
                    self.validation_results.append(
                        ValidationResult(
                            check_name="Node.js Installation",
                            passed=True,
                            message=f"Node.js {version} installed (compatible)",
                            details={"version": version, "min_required": "v16.0.0"},
                        )
                    )
                else:
                    self.validation_results.append(
                        ValidationResult(
                            check_name="Node.js Installation",
                            passed=False,
                            message=f"Node.js {version} is too old (requires v16+)",
                            fix_suggestion="Update Node.js to v16 or later: https://nodejs.org/",
                            details={"version": version, "min_required": "v16.0.0"},
                        )
                    )
            else:
                self.validation_results.append(
                    ValidationResult(
                        check_name="Node.js Installation",
                        passed=False,
                        message="Node.js installed but not responding correctly",
                        fix_suggestion="Reinstall Node.js: https://nodejs.org/",
                        details={"error": result.stderr},
                    )
                )
        except FileNotFoundError:
            self.validation_results.append(
                ValidationResult(
                    check_name="Node.js Installation",
                    passed=False,
                    message="Node.js not found in PATH",
                    fix_suggestion="Install Node.js: https://nodejs.org/ or https://github.com/nvm-sh/nvm",
                    details={"error": "node command not found"},
                )
            )
        except subprocess.TimeoutExpired:
            self.validation_results.append(
                ValidationResult(
                    check_name="Node.js Installation",
                    passed=False,
                    message="Node.js command timed out",
                    fix_suggestion="Check Node.js installation: node --version",
                    details={"error": "timeout after 10 seconds"},
                )
            )
        except Exception as e:
            self.validation_results.append(
                ValidationResult(
                    check_name="Node.js Installation",
                    passed=False,
                    message=f"Unexpected error checking Node.js: {e}",
                    fix_suggestion="Verify Node.js installation manually: node --version",
                )
            )

    def _validate_npm_packages(self) -> None:
        """Validate Vercel AI SDK npm packages are available."""
        try:
            # Check if package.json exists in current directory
            package_json_path = Path("package.json")
            if package_json_path.exists():
                with open(package_json_path) as f:
                    package_data = json.load(f)

                dependencies = {
                    **package_data.get("dependencies", {}),
                    **package_data.get("devDependencies", {}),
                }

                # Check for Vercel AI SDK
                if "ai" in dependencies:
                    version = dependencies["ai"]
                    self.validation_results.append(
                        ValidationResult(
                            check_name="Vercel AI SDK Package",
                            passed=True,
                            message=f"Vercel AI SDK (ai) v{version} found in package.json",
                            details={"version": version, "package": "ai"},
                        )
                    )
                else:
                    self.validation_results.append(
                        ValidationResult(
                            check_name="Vercel AI SDK Package",
                            passed=False,
                            message="Vercel AI SDK not found in package.json",
                            fix_suggestion="Install with: npm install ai",
                            details={"available_deps": list(dependencies.keys())},
                        )
                    )

                # Check for common provider packages
                provider_packages = {
                    "@ai-sdk/openai": "OpenAI provider",
                    "@ai-sdk/anthropic": "Anthropic provider",
                    "@ai-sdk/google": "Google provider",
                    "@ai-sdk/cohere": "Cohere provider",
                    "@ai-sdk/mistral": "Mistral provider",
                }

                found_providers = []
                for pkg, desc in provider_packages.items():
                    if pkg in dependencies:
                        found_providers.append(f"{desc} ({dependencies[pkg]})")

                if found_providers:
                    self.validation_results.append(
                        ValidationResult(
                            check_name="AI Providers",
                            passed=True,
                            message=f"Found {len(found_providers)} AI provider(s)",
                            details={"providers": found_providers},
                        )
                    )
                else:
                    self.validation_results.append(
                        ValidationResult(
                            check_name="AI Providers",
                            passed=False,
                            message="No AI provider packages found",
                            fix_suggestion="Install provider: npm install @ai-sdk/openai (or other providers)",
                            details={
                                "available_providers": list(provider_packages.keys())
                            },
                        )
                    )
            else:
                self.validation_results.append(
                    ValidationResult(
                        check_name="Vercel AI SDK Package",
                        passed=False,
                        message="No package.json found in current directory",
                        fix_suggestion="Initialize npm project: npm init && npm install ai",
                        details={"cwd": str(Path.cwd())},
                    )
                )

        except json.JSONDecodeError as e:
            self.validation_results.append(
                ValidationResult(
                    check_name="Vercel AI SDK Package",
                    passed=False,
                    message=f"Invalid package.json format: {e}",
                    fix_suggestion="Fix package.json syntax or recreate with: npm init",
                )
            )
        except Exception as e:
            self.validation_results.append(
                ValidationResult(
                    check_name="Vercel AI SDK Package",
                    passed=False,
                    message=f"Error checking npm packages: {e}",
                    fix_suggestion="Verify package.json exists and is readable",
                )
            )

    def _validate_python_dependencies(self) -> None:
        """Validate Python dependencies for GenOps integration."""
        required_packages = [
            ("genops", "GenOps core package"),
            ("opentelemetry-api", "OpenTelemetry API"),
            ("requests", "HTTP requests (optional but recommended)"),
        ]

        optional_packages = [
            ("websockets", "WebSocket support for real-time telemetry"),
            ("aiohttp", "Async HTTP support"),
        ]

        # Check required packages
        missing_required = []
        for package, description in required_packages:
            try:
                __import__(package)
                self.validation_results.append(
                    ValidationResult(
                        check_name=f"Python Package: {package}",
                        passed=True,
                        message=f"{description} is available",
                        details={"package": package},
                    )
                )
            except ImportError:
                missing_required.append(package)
                self.validation_results.append(
                    ValidationResult(
                        check_name=f"Python Package: {package}",
                        passed=False,
                        message=f"{description} not found",
                        fix_suggestion=f"Install with: pip install {package}",
                        details={"package": package},
                    )
                )

        # Check optional packages
        missing_optional = []
        for package, description in optional_packages:
            try:
                __import__(package)
                self.validation_results.append(
                    ValidationResult(
                        check_name=f"Python Package: {package} (optional)",
                        passed=True,
                        message=f"{description} is available",
                        details={"package": package, "optional": True},
                    )
                )
            except ImportError:
                missing_optional.append(package)
                self.validation_results.append(
                    ValidationResult(
                        check_name=f"Python Package: {package} (optional)",
                        passed=True,  # Optional packages don't fail validation
                        message=f"{description} not found (optional)",
                        fix_suggestion=f"Install for enhanced features: pip install {package}",
                        details={"package": package, "optional": True},
                    )
                )

    def _validate_environment_variables(self) -> None:
        """Validate environment variables for GenOps configuration."""
        # GenOps governance variables
        governance_vars = {
            "GENOPS_TEAM": "Team name for cost attribution",
            "GENOPS_PROJECT": "Project name for tracking",
            "GENOPS_ENVIRONMENT": "Environment (dev/staging/prod)",
            "GENOPS_COST_CENTER": "Cost center for reporting",
            "GENOPS_CUSTOMER_ID": "Customer ID for attribution",
        }

        # Check governance variables
        found_governance = 0
        for var, description in governance_vars.items():
            value = os.getenv(var)
            if value:
                found_governance += 1
                self.validation_results.append(
                    ValidationResult(
                        check_name=f"Environment Variable: {var}",
                        passed=True,
                        message=f"{description} is set",
                        details={"variable": var, "value_length": len(value)},
                    )
                )
            else:
                self.validation_results.append(
                    ValidationResult(
                        check_name=f"Environment Variable: {var}",
                        passed=False,
                        message=f"{description} not set",
                        fix_suggestion=f"Set with: export {var}='your-value'",
                        details={"variable": var},
                    )
                )

        # Summary of governance configuration
        if found_governance > 0:
            self.validation_results.append(
                ValidationResult(
                    check_name="Governance Configuration",
                    passed=True,
                    message=f"Found {found_governance}/{len(governance_vars)} governance variables",
                    details={
                        "configured": found_governance,
                        "total": len(governance_vars),
                    },
                )
            )
        else:
            self.validation_results.append(
                ValidationResult(
                    check_name="Governance Configuration",
                    passed=False,
                    message="No governance variables configured",
                    fix_suggestion="Set at least GENOPS_TEAM and GENOPS_PROJECT for basic tracking",
                    details={"required_vars": ["GENOPS_TEAM", "GENOPS_PROJECT"]},
                )
            )

    def _validate_genops_configuration(self) -> None:
        """Validate GenOps configuration and telemetry setup."""
        try:
            from genops.core.telemetry import GenOpsTelemetry

            GenOpsTelemetry()

            self.validation_results.append(
                ValidationResult(
                    check_name="GenOps Telemetry",
                    passed=True,
                    message="GenOps telemetry system initialized successfully",
                    details={"component": "telemetry"},
                )
            )

        except ImportError as e:
            self.validation_results.append(
                ValidationResult(
                    check_name="GenOps Telemetry",
                    passed=False,
                    message=f"Cannot import GenOps telemetry: {e}",
                    fix_suggestion="Install GenOps package: pip install genops",
                )
            )
        except Exception as e:
            self.validation_results.append(
                ValidationResult(
                    check_name="GenOps Telemetry",
                    passed=False,
                    message=f"Error initializing GenOps telemetry: {e}",
                    fix_suggestion="Check GenOps configuration and dependencies",
                )
            )

        # Check for OTEL configuration
        otel_vars = [
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_SERVICE_NAME",
            "OTEL_RESOURCE_ATTRIBUTES",
        ]

        otel_configured = 0
        for var in otel_vars:
            if os.getenv(var):
                otel_configured += 1

        if otel_configured > 0:
            self.validation_results.append(
                ValidationResult(
                    check_name="OpenTelemetry Configuration",
                    passed=True,
                    message=f"OpenTelemetry partially configured ({otel_configured}/3 variables)",
                    details={"configured": otel_configured, "total": 3},
                )
            )
        else:
            self.validation_results.append(
                ValidationResult(
                    check_name="OpenTelemetry Configuration",
                    passed=False,
                    message="OpenTelemetry not configured",
                    fix_suggestion="Configure OTEL_EXPORTER_OTLP_ENDPOINT for telemetry export",
                    details={"missing_vars": otel_vars},
                )
            )

    def _validate_provider_access(self) -> None:
        """Validate access to AI providers (requires API keys)."""
        # This is optional and only runs if requested
        provider_vars = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Anthropic",
            "GOOGLE_API_KEY": "Google",
            "COHERE_API_KEY": "Cohere",
            "MISTRAL_API_KEY": "Mistral",
        }

        configured_providers = []
        for var, provider in provider_vars.items():
            if os.getenv(var):
                configured_providers.append(provider)

        if configured_providers:
            self.validation_results.append(
                ValidationResult(
                    check_name="AI Provider Access",
                    passed=True,
                    message=f"API keys found for: {', '.join(configured_providers)}",
                    details={"configured_providers": configured_providers},
                )
            )
        else:
            self.validation_results.append(
                ValidationResult(
                    check_name="AI Provider Access",
                    passed=False,
                    message="No AI provider API keys found",
                    fix_suggestion="Set API keys: export OPENAI_API_KEY='your-key'",
                    details={"available_vars": list(provider_vars.keys())},
                )
            )

    def _generate_validation_summary(self) -> SetupValidationSummary:
        """Generate a summary of all validation results."""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        all_passed = failed_checks == 0

        if all_passed:
            overall_message = (
                "✅ All validation checks passed! Vercel AI SDK integration is ready."
            )
        else:
            overall_message = f"❌ {failed_checks} validation check(s) failed. Review the issues above."

        return SetupValidationSummary(
            all_passed=all_passed,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            results=self.validation_results,
            overall_message=overall_message,
        )

    def _print_validation_summary(self, summary: SetupValidationSummary) -> None:
        """Print a formatted validation summary."""
        print("\n📊 Validation Summary")
        print("-" * 30)
        print(f"Total checks: {summary.total_checks}")
        print(f"✅ Passed: {summary.passed_checks}")
        print(f"❌ Failed: {summary.failed_checks}")
        print(f"\n{summary.overall_message}")

        if summary.failed_checks > 0:
            print("\n🔧 Issues to fix:")
            for result in summary.results:
                if not result.passed and result.fix_suggestion:
                    print(f"  • {result.check_name}: {result.fix_suggestion}")

        print("\n" + "=" * 50)


# Global validator instance
validator = VercelAISDKValidator()


def validate_setup(
    check_nodejs: bool = True,
    check_npm_packages: bool = True,
    check_python_deps: bool = True,
    check_environment: bool = True,
    check_genops_config: bool = True,
    check_provider_access: bool = False,
    verbose: bool = True,
) -> SetupValidationSummary:
    """
    Validate Vercel AI SDK integration setup.

    Args:
        check_nodejs: Validate Node.js installation
        check_npm_packages: Check for Vercel AI SDK npm packages
        check_python_deps: Validate Python dependencies
        check_environment: Check environment variables
        check_genops_config: Validate GenOps configuration
        check_provider_access: Test API connectivity (optional)
        verbose: Print detailed validation results

    Returns:
        SetupValidationSummary: Complete validation results
    """
    return validator.validate_setup(
        check_nodejs=check_nodejs,
        check_npm_packages=check_npm_packages,
        check_python_deps=check_python_deps,
        check_environment=check_environment,
        check_genops_config=check_genops_config,
        check_provider_access=check_provider_access,
        verbose=verbose,
    )


def print_validation_result(result: SetupValidationSummary) -> None:
    """Print validation results in a user-friendly format."""
    validator._print_validation_summary(result)


def quick_validation() -> bool:
    """
    Quick validation check - returns True if basic setup is working.

    Returns:
        bool: True if basic validation passes
    """
    result = validate_setup(
        check_nodejs=True,
        check_python_deps=True,
        check_genops_config=True,
        check_npm_packages=False,  # Skip for quick check
        check_environment=False,  # Skip for quick check
        check_provider_access=False,
        verbose=False,
    )
    return result.all_passed
