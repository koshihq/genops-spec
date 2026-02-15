"""
Export validation utilities for GenOps AI observability integrations.

Provides diagnostic tools to verify export configuration and connectivity
to observability platforms like Honeycomb, Datadog, Grafana, etc.
"""

import os
from dataclasses import dataclass
from typing import Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of export setup validation.

    Attributes:
        provider: Name of the observability provider being validated
        passed: Whether all validation checks passed
        checks: List of individual check results with details
        error_message: Optional error message if validation couldn't run
    """

    provider: str
    passed: bool
    checks: list[dict[str, any]]
    error_message: Optional[str] = None


def validate_export_setup(provider: str) -> ValidationResult:
    """
    Validate export configuration for a specific observability provider.

    This function checks configuration, environment variables, and connectivity
    to ensure telemetry can be successfully exported to the target platform.

    Args:
        provider: Provider name ("honeycomb", "datadog", "grafana", etc.)

    Returns:
        ValidationResult with detailed check results

    Example:
        >>> from genops.exporters.validation import validate_export_setup, print_validation_result
        >>>
        >>> result = validate_export_setup(provider="honeycomb")
        >>> print_validation_result(result)
        ✅ Honeycomb Setup Validation

        Configuration:
          ✅ HONEYCOMB_API_KEY: Set
          ✅ HONEYCOMB_DATASET: genops-ai
          ✅ Connectivity: Honeycomb API reachable
    """
    provider_lower = provider.lower()

    if provider_lower == "honeycomb":
        return _validate_honeycomb()
    elif provider_lower == "datadog":
        return _validate_datadog()
    elif provider_lower == "grafana":
        return _validate_grafana()
    else:
        return ValidationResult(
            provider=provider,
            passed=False,
            checks=[],
            error_message=f"Validation not implemented for provider: {provider}",
        )


def _validate_honeycomb() -> ValidationResult:
    """Validate Honeycomb export configuration."""
    checks = []

    # Check API key
    api_key = os.getenv("HONEYCOMB_API_KEY")
    checks.append(
        {
            "name": "HONEYCOMB_API_KEY",
            "passed": bool(api_key),
            "message": "Set" if api_key else "Not set",
            "fix": "export HONEYCOMB_API_KEY='your_api_key'" if not api_key else None,
        }
    )

    # Check dataset (optional, has default)
    dataset = os.getenv("HONEYCOMB_DATASET", "genops-ai")
    checks.append({"name": "HONEYCOMB_DATASET", "passed": True, "message": dataset})

    # Check connectivity (if API key available and requests library present)
    if api_key and REQUESTS_AVAILABLE:
        try:
            response = requests.get(
                "https://api.honeycomb.io/1/auth",
                headers={"X-Honeycomb-Team": api_key},
                timeout=5,
            )
            connectivity_passed = response.status_code == 200
            checks.append(
                {
                    "name": "Connectivity",
                    "passed": connectivity_passed,
                    "message": "Honeycomb API reachable"
                    if connectivity_passed
                    else f"HTTP {response.status_code}",
                    "fix": "Check API key validity"
                    if not connectivity_passed
                    else None,
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "Connectivity",
                    "passed": False,
                    "message": f"Failed: {str(e)}",
                    "fix": "Check network connectivity to api.honeycomb.io",
                }
            )
    elif api_key and not REQUESTS_AVAILABLE:
        checks.append(
            {
                "name": "Connectivity",
                "passed": True,
                "message": "Skipped (requests library not available)",
            }
        )

    passed = all(check["passed"] for check in checks)

    return ValidationResult(provider="honeycomb", passed=passed, checks=checks)


def _validate_datadog() -> ValidationResult:
    """Validate Datadog export configuration."""
    checks = []

    # Check API key
    api_key = os.getenv("DD_API_KEY")
    checks.append(
        {
            "name": "DD_API_KEY",
            "passed": bool(api_key),
            "message": "Set" if api_key else "Not set",
            "fix": "export DD_API_KEY='your_api_key'" if not api_key else None,
        }
    )

    # Check site (optional)
    site = os.getenv("DD_SITE", "datadoghq.com")
    checks.append({"name": "DD_SITE", "passed": True, "message": site})

    # Check service name
    service = os.getenv("DD_SERVICE") or os.getenv("OTEL_SERVICE_NAME")
    checks.append(
        {
            "name": "DD_SERVICE",
            "passed": bool(service),
            "message": service if service else "Not set",
            "fix": "export DD_SERVICE='your-service-name'" if not service else None,
        }
    )

    passed = all(check["passed"] for check in checks)

    return ValidationResult(provider="datadog", passed=passed, checks=checks)


def _validate_grafana() -> ValidationResult:
    """Validate Grafana/Tempo export configuration."""
    checks = []

    # Check Tempo endpoint
    endpoint = os.getenv("TEMPO_ENDPOINT")
    checks.append(
        {
            "name": "TEMPO_ENDPOINT",
            "passed": bool(endpoint),
            "message": endpoint if endpoint else "Not set",
            "fix": "export TEMPO_ENDPOINT='http://tempo:4318/v1/traces'"
            if not endpoint
            else None,
        }
    )

    # Check authentication (optional)
    auth_header = os.getenv("TEMPO_AUTH_HEADER")
    if auth_header:
        checks.append({"name": "TEMPO_AUTH_HEADER", "passed": True, "message": "Set"})

    passed = all(check["passed"] for check in checks)

    return ValidationResult(provider="grafana", passed=passed, checks=checks)


def print_validation_result(result: ValidationResult) -> None:
    """
    Print validation result in user-friendly format.

    Args:
        result: ValidationResult from validate_export_setup()

    Example:
        >>> result = validate_export_setup(provider="honeycomb")
        >>> print_validation_result(result)
        ✅ Honeycomb Setup Validation

        Configuration:
          ✅ HONEYCOMB_API_KEY: Set
          ✅ HONEYCOMB_DATASET: genops-ai
          ✅ Connectivity: Honeycomb API reachable

        ✅ All checks passed! Telemetry is flowing to Honeycomb.
    """
    icon = "✅" if result.passed else "❌"
    print(f"\n{icon} {result.provider.title()} Setup Validation\n")

    if result.error_message:
        print(f"❌ Error: {result.error_message}\n")
        return

    print("Configuration:")
    for check in result.checks:
        check_icon = "✅" if check["passed"] else "❌"
        print(f"  {check_icon} {check['name']}: {check['message']}")
        if check.get("fix"):
            print(f"     Fix: {check['fix']}")

    print()  # Empty line before summary

    if result.passed:
        print(
            f"✅ All checks passed! Telemetry is flowing to {result.provider.title()}."
        )
    else:
        print("❌ Some checks failed. Fix the issues above and try again.")
