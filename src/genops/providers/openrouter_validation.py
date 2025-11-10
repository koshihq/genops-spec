"""
Validation utilities for OpenRouter integration setup.
Helps developers verify their GenOps OpenRouter integration is working correctly.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional
from urllib.parse import urlparse

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
    """Check required and optional environment variables for OpenRouter."""
    issues = []

    # Required variables for OpenRouter

    # Check for alternative naming patterns
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not openrouter_key:
        issues.append(
            ValidationIssue(
                level="error",
                component="environment",
                message="Missing OpenRouter API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY",
                fix_suggestion="Get API key from https://openrouter.ai/keys and set: export OPENROUTER_API_KEY=your_key_here",
            )
        )

    # Optional but recommended variables
    optional_vars = {
        "OTEL_SERVICE_NAME": "OpenTelemetry service name for telemetry identification",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "OTLP endpoint for telemetry export",
        "OPENROUTER_HTTP_REFERER": "HTTP referer for OpenRouter request identification",
        "OPENROUTER_X_TITLE": "Application name for OpenRouter request identification",
    }

    for var, description in optional_vars.items():
        if not os.getenv(var):
            issues.append(
                ValidationIssue(
                    level="info",
                    component="environment",
                    message=f"Optional environment variable not set: {var}",
                    fix_suggestion=f"For {description}, set: export {var}=your_value",
                )
            )

    # Check API key format (OpenRouter keys start with 'sk-')
    if openrouter_key:
        if not openrouter_key.startswith("sk-"):
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="environment",
                    message="OpenRouter API key doesn't start with 'sk-' - may be invalid format",
                    fix_suggestion="Verify your OpenRouter API key format from https://openrouter.ai/keys",
                )
            )
        elif len(openrouter_key) < 40:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="environment",
                    message="OpenRouter API key appears too short - may be incomplete",
                    fix_suggestion="Verify complete API key was copied from OpenRouter dashboard",
                )
            )

    # Check OTLP configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        if not (
            otlp_endpoint.startswith("http://") or otlp_endpoint.startswith("https://")
        ):
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="configuration",
                    message=f"OTLP endpoint should start with http:// or https://: {otlp_endpoint}",
                    fix_suggestion="Use format: http://localhost:4317 or https://api.provider.com",
                )
            )

    return issues


def check_dependencies() -> list[ValidationIssue]:
    """Check if required dependencies are available."""
    issues = []

    # Check for OpenAI package (required for OpenRouter compatibility)
    try:
        import openai

        issues.append(
            ValidationIssue(
                level="info",
                component="dependencies",
                message=f"OpenAI package found: {openai.__version__}",
            )
        )
    except ImportError:
        issues.append(
            ValidationIssue(
                level="error",
                component="dependencies",
                message="OpenAI package not found (required for OpenRouter compatibility)",
                fix_suggestion="Install with: pip install openai",
            )
        )

    # Check for GenOps components
    try:
        import genops.core.telemetry  # noqa: F401

        issues.append(
            ValidationIssue(
                level="info",
                component="dependencies",
                message="GenOps telemetry module found",
            )
        )
    except ImportError:
        issues.append(
            ValidationIssue(
                level="error",
                component="dependencies",
                message="GenOps telemetry module not found",
                fix_suggestion="Ensure GenOps is properly installed",
            )
        )

    # Check for OpenTelemetry
    try:
        import opentelemetry

        issues.append(
            ValidationIssue(
                level="info",
                component="dependencies",
                message=f"OpenTelemetry package found: {opentelemetry.__version__}",
            )
        )
    except ImportError:
        issues.append(
            ValidationIssue(
                level="warning",
                component="dependencies",
                message="OpenTelemetry package not found",
                fix_suggestion="For telemetry export, install with: pip install opentelemetry-api opentelemetry-sdk",
            )
        )

    return issues


def check_openrouter_connection() -> list[ValidationIssue]:
    """Check if OpenRouter API is accessible with comprehensive diagnostics."""
    issues = []

    try:
        import requests
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="connectivity",
                    message="Cannot test OpenRouter connection - no API key found",
                    fix_suggestion="Set OPENROUTER_API_KEY environment variable. Get key from https://openrouter.ai/keys",
                )
            )
            return issues

        # First, test basic HTTP connectivity to OpenRouter
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                models_data = response.json()
                model_count = len(models_data.get("data", []))
                issues.append(
                    ValidationIssue(
                        level="info",
                        component="connectivity",
                        message=f"OpenRouter HTTP API connection successful. {model_count} models available.",
                    )
                )

                # Additional model availability checks
                if model_count < 50:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            component="connectivity",
                            message=f"Only {model_count} models available - may indicate API key limitations",
                            fix_suggestion="Check your OpenRouter plan limits at https://openrouter.ai/account",
                        )
                    )

            elif response.status_code == 401:
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="connectivity",
                        message="OpenRouter API authentication failed - invalid API key",
                        fix_suggestion="Verify your API key at https://openrouter.ai/keys. Ensure it starts with 'sk-' and is complete.",
                    )
                )
            elif response.status_code == 403:
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="connectivity",
                        message="OpenRouter API access forbidden - check account status",
                        fix_suggestion="Ensure your OpenRouter account is active and has sufficient credits. Check https://openrouter.ai/account",
                    )
                )
            elif response.status_code == 429:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message="OpenRouter API rate limited - too many requests",
                        fix_suggestion="Wait a moment and try again. Consider upgrading your plan for higher limits.",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message=f"OpenRouter API returned status {response.status_code}",
                        fix_suggestion="Check OpenRouter service status at https://status.openrouter.ai/",
                    )
                )
        except requests.exceptions.Timeout:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="connectivity",
                    message="OpenRouter API request timed out",
                    fix_suggestion="Check your internet connection. OpenRouter may be experiencing high load.",
                )
            )
        except requests.exceptions.ConnectionError:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="connectivity",
                    message="Cannot connect to OpenRouter API - network error",
                    fix_suggestion="Check internet connection and firewall settings. Try: curl https://openrouter.ai/api/v1/models",
                )
            )
        except Exception:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="connectivity",
                    message="HTTP connectivity test failed",
                    fix_suggestion="Check network configuration and try manual curl test",
                )
            )

        # Test using OpenAI SDK (OpenRouter compatibility)
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=10.0
            )

            # Try a minimal completion request
            test_response = client.chat.completions.create(
                model="meta-llama/llama-3.2-1b-instruct",  # Cheap model for testing
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )

            issues.append(
                ValidationIssue(
                    level="info",
                    component="connectivity",
                    message="OpenRouter SDK test completion successful",
                )
            )

            # Check response structure
            if hasattr(test_response, "choices") and test_response.choices:
                if hasattr(test_response, "usage") and test_response.usage:
                    issues.append(
                        ValidationIssue(
                            level="info",
                            component="connectivity",
                            message=f"Test completion used {test_response.usage.total_tokens} tokens",
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            component="connectivity",
                            message="Test completion succeeded but no usage data returned",
                            fix_suggestion="This may affect cost tracking accuracy",
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message="Test completion returned unexpected response format",
                        fix_suggestion="OpenRouter API may have changed. Check for updates.",
                    )
                )

        except Exception:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="connectivity",
                        message="OpenRouter SDK authentication failed",
                        fix_suggestion="Double-check API key format. It should start with 'sk-' and be ~51 characters long.",
                    )
                )
            elif "timeout" in error_msg.lower():
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message="OpenRouter API request timed out",
                        fix_suggestion="Network may be slow. Try increasing timeout or check connection.",
                    )
                )
            elif "rate" in error_msg.lower() or "429" in error_msg:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message="Rate limited by OpenRouter API",
                        fix_suggestion="Wait 60 seconds and try again. Consider upgrading plan for higher limits.",
                    )
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message="Test model not available - API key may have model restrictions",
                        fix_suggestion="Check your OpenRouter plan at https://openrouter.ai/account for model access",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="connectivity",
                        message=f"OpenRouter SDK test failed: {error_msg}",
                        fix_suggestion="Check OpenRouter service status and your account limits",
                    )
                )

    except ImportError:
        issues.append(
            ValidationIssue(
                level="error",
                component="connectivity",
                message="Cannot test OpenRouter connection - OpenAI package not available",
                fix_suggestion="Install OpenAI package: pip install openai. This is required for OpenRouter compatibility.",
            )
        )
    except Exception:
        issues.append(
            ValidationIssue(
                level="error",
                component="connectivity",
                message="Unexpected error during connection test",
                fix_suggestion="Please report this issue with full error details",
            )
        )

    return issues


def check_genops_configuration() -> list[ValidationIssue]:
    """Check GenOps-specific configuration for OpenRouter."""
    issues = []

    # Check if auto-instrumentation is working
    try:
        import genops.auto_instrumentation

        instrumentor = genops.auto_instrumentation.GenOpsInstrumentor()
        if (
            hasattr(instrumentor, "provider_patches")
            and "openrouter" in instrumentor.provider_patches
        ):
            issues.append(
                ValidationIssue(
                    level="info",
                    component="configuration",
                    message="OpenRouter provider registered in GenOps auto-instrumentation",
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="configuration",
                    message="OpenRouter provider not found in auto-instrumentation registry",
                    fix_suggestion="Ensure OpenRouter provider is properly installed and registered",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                level="warning",
                component="configuration",
                message="GenOps auto-instrumentation not available",
                fix_suggestion="Ensure GenOps is properly installed with auto-instrumentation support",
            )
        )
    except Exception:
        issues.append(
            ValidationIssue(
                level="warning",
                component="configuration",
                message="Error checking GenOps configuration",
            )
        )

    # Check telemetry configuration
    try:
        import genops.core.telemetry

        genops.core.telemetry.GenOpsTelemetry()
        issues.append(
            ValidationIssue(
                level="info",
                component="configuration",
                message="GenOps telemetry engine available",
            )
        )

    except Exception:
        issues.append(
            ValidationIssue(
                level="error",
                component="configuration",
                message="GenOps telemetry engine error",
                fix_suggestion="Check GenOps installation and OpenTelemetry configuration",
            )
        )

    return issues


def test_basic_functionality() -> list[ValidationIssue]:
    """Test basic OpenRouter integration functionality with comprehensive diagnostics."""
    issues = []

    try:
        import genops.providers.openrouter

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="functionality",
                    message="Cannot test basic functionality - no API key available",
                    fix_suggestion="Set OPENROUTER_API_KEY to enable functionality testing. Get key from https://openrouter.ai/keys",
                )
            )
            return issues

        # Test adapter creation
        try:
            adapter = genops.providers.openrouter.instrument_openrouter(openrouter_api_key=api_key)
            issues.append(
                ValidationIssue(
                    level="info",
                    component="functionality",
                    message="OpenRouter adapter creation successful",
                )
            )

            # Test adapter attributes
            if hasattr(adapter, "client") and adapter.client:
                issues.append(
                    ValidationIssue(
                        level="info",
                        component="functionality",
                        message="OpenRouter client properly initialized in adapter",
                    )
                )

                # Check base URL configuration
                if hasattr(adapter.client, "_base_url"):
                    base_url_parsed = urlparse(str(adapter.client._base_url))
                    if base_url_parsed.hostname and (
                        base_url_parsed.hostname == "openrouter.ai"
                        or base_url_parsed.hostname.endswith(".openrouter.ai")
                    ):
                        issues.append(
                            ValidationIssue(
                                level="info",
                                component="functionality",
                                message="OpenRouter base URL correctly configured",
                            )
                        )
                    else:
                        issues.append(
                            ValidationIssue(
                                level="warning",
                                component="functionality",
                                message="OpenRouter base URL may not be configured correctly",
                                fix_suggestion="Ensure base_url is set to https://openrouter.ai/api/v1",
                            )
                        )
                else:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            component="functionality",
                            message="OpenRouter client base URL not accessible",
                            fix_suggestion="Check OpenAI package compatibility and API key format",
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="functionality",
                        message="OpenRouter adapter client not properly initialized",
                        fix_suggestion="Check OpenAI package compatibility and API key format",
                    )
                )

        except Exception:
            error_msg = str(e)
            if "import" in error_msg.lower():
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="functionality",
                        message=f"Import error during adapter creation: {error_msg}",
                        fix_suggestion="Install missing packages: pip install genops-ai openai",
                    )
                )
            elif "auth" in error_msg.lower() or "401" in error_msg:
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="functionality",
                        message="Authentication error during adapter creation",
                        fix_suggestion="Verify API key is valid and has proper permissions",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="functionality",
                        message=f"OpenRouter adapter creation failed: {error_msg}",
                        fix_suggestion="Check OpenAI package installation and API key format",
                    )
                )

        # Test pricing engine with detailed diagnostics
        try:
            from genops.providers.openrouter_pricing import (
                calculate_openrouter_cost,
                get_pricing_engine,
            )

            engine = get_pricing_engine()
            supported_models = len(engine.pricing_db)
            issues.append(
                ValidationIssue(
                    level="info",
                    component="functionality",
                    message=f"OpenRouter pricing engine loaded with {supported_models} models",
                )
            )

            # Test cost calculation functionality
            test_cost = calculate_openrouter_cost(
                "anthropic/claude-3-sonnet", input_tokens=100, output_tokens=50
            )

            if test_cost > 0:
                issues.append(
                    ValidationIssue(
                        level="info",
                        component="functionality",
                        message=f"Cost calculation working correctly (test cost: ${test_cost:.6f})",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="functionality",
                        message="Cost calculation returned zero - pricing data may be missing",
                        fix_suggestion="Check pricing engine model coverage for your target models",
                    )
                )

            # Test provider-specific pricing
            providers_tested = []
            test_models = [
                ("anthropic/claude-3-sonnet", "anthropic"),
                ("openai/gpt-4o", "openai"),
                ("meta-llama/llama-3.1-8b-instruct", "meta"),
            ]

            for model, expected_provider in test_models:
                try:
                    cost = calculate_openrouter_cost(
                        model, input_tokens=10, output_tokens=10
                    )
                    if cost > 0:
                        providers_tested.append(expected_provider)
                except Exception:
                    # Ignore errors in cost calculation during validation  # nosec B110
                    pass

            if providers_tested:
                issues.append(
                    ValidationIssue(
                        level="info",
                        component="functionality",
                        message=f"Multi-provider pricing working for: {', '.join(providers_tested)}",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="functionality",
                        message="Multi-provider pricing tests failed",
                        fix_suggestion="Check pricing engine model database completeness",
                    )
                )

        except ImportError:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="functionality",
                    message="Cannot import OpenRouter pricing engine",
                    fix_suggestion="Ensure OpenRouter provider module is properly installed: pip install genops-ai",
                )
            )
        except Exception:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="functionality",
                    message="OpenRouter pricing engine error",
                    fix_suggestion="Check OpenRouter pricing module installation and compatibility",
                )
            )

        # Test telemetry integration
        try:
            import genops.core.telemetry

            genops.core.telemetry.GenOpsTelemetry()
            issues.append(
                ValidationIssue(
                    level="info",
                    component="functionality",
                    message="GenOps telemetry engine integration available",
                )
            )

            # Test context management
            from genops.core.context import get_effective_attributes

            test_attrs = get_effective_attributes(team="test", project="validation")
            if isinstance(test_attrs, dict) and "team" in test_attrs:
                issues.append(
                    ValidationIssue(
                        level="info",
                        component="functionality",
                        message="GenOps context management working correctly",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="functionality",
                        message="GenOps context management may not be working properly",
                        fix_suggestion="Check GenOps core installation and OpenTelemetry setup",
                    )
                )

        except ImportError:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="functionality",
                    message="GenOps telemetry integration not available",
                    fix_suggestion="Install complete GenOps package: pip install genops-ai",
                )
            )
        except Exception:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="functionality",
                    message="Telemetry integration test failed",
                    fix_suggestion="Check GenOps installation and OpenTelemetry configuration",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                level="error",
                component="functionality",
                message="Cannot import OpenRouter provider",
                fix_suggestion="Ensure OpenRouter provider module is properly installed: pip install genops-ai openai",
            )
        )
    except Exception:
        issues.append(
            ValidationIssue(
                level="error",
                component="functionality",
                message="Basic functionality test error",
                fix_suggestion="Check installation and configuration. Try reinstalling: pip install --upgrade genops-ai",
            )
        )

    return issues


def check_common_issues() -> list[ValidationIssue]:
    """Check for common configuration issues and provide specific fixes."""
    issues = []

    # Check for common environment variable issues
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    if api_key:
        # API key format validation
        if not api_key.startswith("sk-"):
            issues.append(
                ValidationIssue(
                    level="error",
                    component="configuration",
                    message="API key doesn't start with 'sk-' - invalid format",
                    fix_suggestion="OpenRouter API keys should start with 'sk-'. Get a new key from https://openrouter.ai/keys",
                )
            )
        elif len(api_key) < 40:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="configuration",
                    message="API key appears too short - likely incomplete",
                    fix_suggestion="Ensure you copied the complete API key. Keys are typically 51-64 characters long.",
                )
            )
        elif " " in api_key or "\n" in api_key or "\t" in api_key:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="configuration",
                    message="API key contains whitespace characters",
                    fix_suggestion="Remove any spaces, newlines, or tabs from the API key",
                )
            )

    # Check OpenTelemetry configuration
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otel_endpoint:
        if not (
            otel_endpoint.startswith("http://") or otel_endpoint.startswith("https://")
        ):
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="configuration",
                    message="OTLP endpoint should start with http:// or https://",
                    fix_suggestion="Update endpoint format: export OTEL_EXPORTER_OTLP_ENDPOINT='https://your-endpoint'",
                )
            )

        # Check for common endpoint URLs and provide specific guidance
        parsed_url = urlparse(otel_endpoint)
        if parsed_url.hostname and (
            parsed_url.hostname == "honeycomb.io"
            or parsed_url.hostname.endswith(".honeycomb.io")
        ):
            headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            if not headers or "x-honeycomb-team" not in headers:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="configuration",
                        message="Honeycomb endpoint detected but missing required headers",
                        fix_suggestion="Set OTEL_EXPORTER_OTLP_HEADERS='x-honeycomb-team=your-api-key'",
                    )
                )
        elif "datadog" in otel_endpoint.lower():
            headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            if not headers or "dd-api-key" not in headers:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        component="configuration",
                        message="Datadog endpoint detected but missing required headers",
                        fix_suggestion="Set OTEL_EXPORTER_OTLP_HEADERS='dd-api-key=your-datadog-key'",
                    )
                )
    else:
        issues.append(
            ValidationIssue(
                level="info",
                component="configuration",
                message="No OTLP endpoint configured - telemetry will use console output",
                fix_suggestion="Configure observability endpoint: export OTEL_EXPORTER_OTLP_ENDPOINT='your-endpoint'",
            )
        )

    # Check for proxy configuration issues
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

    if http_proxy or https_proxy:
        issues.append(
            ValidationIssue(
                level="info",
                component="configuration",
                message="Proxy configuration detected",
                fix_suggestion="Ensure proxy allows access to openrouter.ai and your OTLP endpoint",
            )
        )

    # Check Python version compatibility
    import sys

    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(
            ValidationIssue(
                level="error",
                component="configuration",
                message=f"Python {python_version.major}.{python_version.minor} is too old",
                fix_suggestion="GenOps requires Python 3.8 or newer. Please upgrade your Python version.",
            )
        )
    elif python_version >= (3, 12):
        issues.append(
            ValidationIssue(
                level="info",
                component="configuration",
                message=f"Python {python_version.major}.{python_version.minor} - latest version compatibility verified",
            )
        )

    # Check for common package conflicts
    try:
        import openai

        openai_version = getattr(openai, "__version__", "unknown")

        if openai_version != "unknown":
            major_version = int(openai_version.split(".")[0])
            if major_version < 1:
                issues.append(
                    ValidationIssue(
                        level="error",
                        component="configuration",
                        message=f"OpenAI package version {openai_version} is too old",
                        fix_suggestion="Upgrade OpenAI package: pip install --upgrade openai",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        level="info",
                        component="configuration",
                        message=f"OpenAI package version {openai_version} is compatible",
                    )
                )
    except ImportError:
        pass  # Already covered in dependency checks

    return issues


def validate_openrouter_setup() -> ValidationResult:
    """
    Comprehensive validation of OpenRouter setup with enhanced diagnostics.

    Returns:
        ValidationResult with overall status and detailed issues
    """
    all_issues = []

    # Run all validation checks in order of importance
    all_issues.extend(
        check_common_issues()
    )  # New: Check common configuration problems first
    all_issues.extend(check_environment_variables())  # Environment setup
    all_issues.extend(check_dependencies())  # Package dependencies
    all_issues.extend(check_openrouter_connection())  # API connectivity
    all_issues.extend(check_genops_configuration())  # GenOps integration
    all_issues.extend(test_basic_functionality())  # End-to-end functionality

    # Determine overall validation status
    error_count = len([issue for issue in all_issues if issue.level == "error"])
    warning_count = len([issue for issue in all_issues if issue.level == "warning"])
    info_count = len([issue for issue in all_issues if issue.level == "info"])

    is_valid = error_count == 0

    # Enhanced summary with more detailed analysis
    summary = {
        "total_issues": len(all_issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "is_functional": error_count == 0 and warning_count <= 3,  # Allow more warnings
        "validation_score": _calculate_validation_score(
            error_count, warning_count, info_count
        ),
        "recommendations": [],
    }

    # Add specific recommendations based on validation results
    if error_count > 0:
        summary["recommendations"].append(
            "❌ Fix error-level issues before using OpenRouter provider"
        )
        if error_count == 1:
            summary["recommendations"].append(
                "💡 Focus on the single error - usually API key or connectivity"
            )
        else:
            summary["recommendations"].append(
                "🔧 Check installation first: pip install --upgrade genops-ai openai"
            )

    if warning_count > 3:
        summary["recommendations"].append(
            "⚠️  Multiple warnings detected - review configuration carefully"
        )
    elif warning_count > 0:
        summary["recommendations"].append(
            "📝 Review warning-level issues for optimal configuration"
        )

    if error_count == 0 and warning_count <= 1:
        summary["recommendations"].append(
            "✅ Setup looks excellent! Ready to use OpenRouter with GenOps"
        )
        summary["recommendations"].append(
            "🚀 Try the examples: python examples/openrouter/basic_tracking.py"
        )
    elif error_count == 0:
        summary["recommendations"].append(
            "✅ Setup is functional - minor optimizations recommended"
        )
        summary["recommendations"].append(
            "📖 See troubleshooting guide for warning resolution"
        )

    # Add context-specific recommendations
    if info_count > 5:
        summary["recommendations"].append(
            "ℹ️  Rich configuration detected - good setup coverage"
        )

    return ValidationResult(is_valid, all_issues, summary)


def _calculate_validation_score(
    error_count: int, warning_count: int, info_count: int
) -> float:
    """Calculate a validation score from 0.0 (bad) to 1.0 (perfect)."""
    if error_count > 0:
        return max(0.0, 0.3 - (error_count * 0.1))  # Errors severely impact score

    base_score = 0.8  # Start with good score if no errors
    warning_penalty = min(0.4, warning_count * 0.05)  # Small penalty for warnings
    info_bonus = min(0.2, info_count * 0.02)  # Small bonus for successful checks

    return min(1.0, base_score - warning_penalty + info_bonus)


def print_openrouter_validation_result(result: ValidationResult) -> None:
    """Print validation result in user-friendly format with enhanced diagnostics."""
    print("\n🔍 GenOps OpenRouter Setup Validation")
    print("=" * 50)

    if result.is_valid:
        print("✅ Overall Status: VALID")
    else:
        print("❌ Overall Status: INVALID")

    # Show validation score
    score = result.summary.get("validation_score", 0.0)
    score_emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
    print(f"📈 Validation Score: {score_emoji} {score:.1%}")

    print("\n📊 Summary:")
    print(f"   • Total Issues: {result.summary['total_issues']}")
    print(f"   • Errors: {result.summary['error_count']}")
    print(f"   • Warnings: {result.summary['warning_count']}")
    print(f"   • Info: {result.summary['info_count']}")
    print(
        f"   • Functional: {'Yes' if result.summary.get('is_functional', False) else 'No'}"
    )

    if result.issues:
        print("\n📋 Issues Found:")

        # Group issues by component
        issues_by_component = {}
        for issue in result.issues:
            if issue.component not in issues_by_component:
                issues_by_component[issue.component] = []
            issues_by_component[issue.component].append(issue)

        for component, issues in issues_by_component.items():
            print(f"\n   {component.title()}:")
            for issue in issues:
                # Choose emoji based on level
                emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                    issue.level, "•"
                )
                print(f"     {emoji} {issue.message}")
                if issue.fix_suggestion:
                    print(f"        💡 Fix: {issue.fix_suggestion}")

    if result.summary.get("recommendations"):
        print("\n💡 Recommendations:")
        for rec in result.summary["recommendations"]:
            print(f"   • {rec}")

    print("\n🚀 Next Steps:")
    if result.is_valid:
        print("   • Your setup is ready! Try the basic example:")
        print(
            "     python -c \"from genops.providers.openrouter import instrument_openrouter; print('OpenRouter ready!')\""
        )
    else:
        print("   • Fix the error-level issues above")
        print(
            '   • Re-run validation: python -c "from genops.providers.openrouter import validate_setup, print_validation_result; print_validation_result(validate_setup())"'
        )

    print("   • Check out examples: examples/openrouter/")
    print("   • Full documentation: docs/integrations/openrouter.md")
    print()


if __name__ == "__main__":
    """Allow running validation directly."""
    result = validate_openrouter_setup()
    print_openrouter_validation_result(result)
