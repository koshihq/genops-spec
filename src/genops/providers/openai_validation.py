"""
Validation utilities for OpenAI integration setup.
Helps developers verify their GenOps OpenAI integration is working correctly.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

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
    """Check required and optional environment variables."""
    issues = []

    # Required variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM access and cost calculation"
    }

    for var, description in required_vars.items():
        if not os.getenv(var):
            issues.append(ValidationIssue(
                level="error",
                component="environment",
                message=f"Missing required environment variable: {var} ({description})",
                fix_suggestion=f"Set {var} with: export {var}=your_key_here"
            ))

    # Optional but recommended variables
    optional_vars = {
        "OTEL_SERVICE_NAME": "OpenTelemetry service name for telemetry identification",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "OTLP endpoint for telemetry export",
        "OPENAI_ORG_ID": "OpenAI organization ID for team billing",
        "OPENAI_PROJECT_ID": "OpenAI project ID for cost attribution"
    }

    for var, description in optional_vars.items():
        if not os.getenv(var):
            issues.append(ValidationIssue(
                level="warning",
                component="environment",
                message=f"Optional environment variable not set: {var}",
                fix_suggestion=f"For {description}, set: export {var}=your_value"
            ))

    # Check API key format
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        if not api_key.startswith("sk-"):
            issues.append(ValidationIssue(
                level="warning",
                component="environment",
                message="OPENAI_API_KEY doesn't start with 'sk-' - may be invalid format",
                fix_suggestion="Verify your OpenAI API key format from https://platform.openai.com/api-keys"
            ))
        elif len(api_key) < 50:
            issues.append(ValidationIssue(
                level="warning",
                component="environment",
                message="OPENAI_API_KEY appears too short - may be incomplete",
                fix_suggestion="Verify complete API key was copied from OpenAI dashboard"
            ))

    # Check OTLP configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        if not (otlp_endpoint.startswith("http://") or otlp_endpoint.startswith("https://")):
            issues.append(ValidationIssue(
                level="warning",
                component="configuration",
                message=f"OTLP endpoint should start with http:// or https://: {otlp_endpoint}",
                fix_suggestion="Use format: http://localhost:4317 or https://api.provider.com"
            ))

    return issues


def check_dependencies() -> list[ValidationIssue]:
    """Check if required dependencies are available."""
    issues = []

    # Core dependencies
    core_deps = {
        "opentelemetry": "OpenTelemetry SDK",
        "openai": "OpenAI Python client"
    }

    for module, description in core_deps.items():
        try:
            __import__(module)
        except ImportError:
            issues.append(ValidationIssue(
                level="error",
                component="dependencies",
                message=f"Required dependency not found: {module}",
                fix_suggestion=f"Install {description} with: pip install {module}"
            ))

    # Check OpenAI version compatibility
    try:
        import openai
        version = getattr(openai, '__version__', None)
        if version:
            major_version = int(version.split('.')[0])
            if major_version < 1:
                issues.append(ValidationIssue(
                    level="warning",
                    component="dependencies",
                    message=f"OpenAI client version {version} may have compatibility issues",
                    fix_suggestion="Update OpenAI client: pip install --upgrade openai>=1.0"
                ))
            else:
                issues.append(ValidationIssue(
                    level="info",
                    component="dependencies",
                    message=f"OpenAI client version {version} is compatible",
                    fix_suggestion=None
                ))
    except ImportError:
        pass  # Already handled above
    except Exception as e:
        issues.append(ValidationIssue(
            level="warning",
            component="dependencies",
            message=f"Could not verify OpenAI version: {e}",
            fix_suggestion="Ensure OpenAI client is properly installed"
        ))

    return issues


def check_genops_imports() -> list[ValidationIssue]:
    """Check if GenOps modules can be imported correctly."""
    issues = []

    genops_modules = {
        "genops.providers.openai": "GenOps OpenAI adapter",
        "genops.core.telemetry": "Core telemetry functionality",
        "genops.core.tracker": "Cost and evaluation tracking"
    }

    for module, _description in genops_modules.items():
        try:
            __import__(module)
        except ImportError:
            issues.append(ValidationIssue(
                level="error",
                component="genops",
                message=f"GenOps module not available: {module}",
                fix_suggestion="Ensure GenOps is installed: pip install genops-ai"
            ))

    return issues


def test_basic_functionality() -> list[ValidationIssue]:
    """Test basic GenOps OpenAI functionality."""
    issues = []

    try:
        # Test adapter creation
        from genops.providers.openai import GenOpsOpenAIAdapter

        # Try to create adapter (will fail without API key, but tests import)
        try:
            adapter = GenOpsOpenAIAdapter()

            # Test basic properties
            if hasattr(adapter, 'GOVERNANCE_ATTRIBUTES'):
                expected_attrs = {'team', 'project', 'customer_id', 'environment'}
                if not expected_attrs.issubset(adapter.GOVERNANCE_ATTRIBUTES):
                    issues.append(ValidationIssue(
                        level="warning",
                        component="functionality",
                        message="Missing some expected governance attributes",
                        fix_suggestion="Ensure all governance attributes are supported"
                    ))
            else:
                issues.append(ValidationIssue(
                    level="error",
                    component="functionality",
                    message="Governance attributes not found in adapter",
                    fix_suggestion="Check GenOps OpenAI adapter implementation"
                ))

        except Exception as e:
            if "API key" in str(e) or "OPENAI_API_KEY" in str(e):
                # Expected without API key - adapter structure is fine
                issues.append(ValidationIssue(
                    level="info",
                    component="functionality",
                    message="OpenAI adapter structure is valid (API key needed for full testing)",
                    fix_suggestion="Set OPENAI_API_KEY to test full functionality"
                ))
            else:
                issues.append(ValidationIssue(
                    level="error",
                    component="functionality",
                    message=f"Failed to create OpenAI adapter: {e}",
                    fix_suggestion="Check GenOps installation and dependencies"
                ))

    except Exception as e:
        issues.append(ValidationIssue(
            level="error",
            component="functionality",
            message=f"Failed to import OpenAI adapter: {e}",
            fix_suggestion="Check GenOps installation"
        ))

    return issues


def test_opentelemetry_setup() -> list[ValidationIssue]:
    """Test OpenTelemetry configuration."""
    issues = []

    try:
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)

        # Test span creation
        with tracer.start_as_current_span("validation_test") as span:
            span.set_attribute("genops.validation.test", "success")
            span.set_attribute("genops.provider", "openai")

    except Exception as e:
        issues.append(ValidationIssue(
            level="error",
            component="opentelemetry",
            message=f"OpenTelemetry not working: {e}",
            fix_suggestion="Check OpenTelemetry installation and configuration"
        ))

    # Check exporter configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = os.getenv("OTEL_SERVICE_NAME")

    if not service_name:
        issues.append(ValidationIssue(
            level="warning",
            component="opentelemetry",
            message="OTEL_SERVICE_NAME not set",
            fix_suggestion="Set service name: export OTEL_SERVICE_NAME=my-openai-app"
        ))

    if not otlp_endpoint:
        issues.append(ValidationIssue(
            level="info",
            component="opentelemetry",
            message="OTEL_EXPORTER_OTLP_ENDPOINT not set - telemetry will only be logged",
            fix_suggestion="For telemetry export, set: export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317"
        ))

    return issues


def test_live_openai_connection() -> list[ValidationIssue]:
    """Test actual OpenAI API connection (if API key available)."""
    issues = []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        issues.append(ValidationIssue(
            level="info",
            component="live_test",
            message="Skipping live test - no OpenAI API key",
            fix_suggestion="Set OPENAI_API_KEY to test live OpenAI connection"
        ))
        return issues

    try:
        from genops.providers.openai import GenOpsOpenAIAdapter

        # Create adapter and test simple completion
        adapter = GenOpsOpenAIAdapter()

        # Test simple completion with minimal cost
        result = adapter.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello from GenOps' in exactly those words."}
            ],
            max_tokens=10,
            temperature=0,

            # Governance attributes for test
            team="validation-test",
            project="setup-verification"
        )

        # Check if response contains expected text
        if result and hasattr(result, 'choices') and result.choices:
            response_text = result.choices[0].message.content
            if "Hello from GenOps" in response_text:
                issues.append(ValidationIssue(
                    level="info",
                    component="live_test",
                    message="Live OpenAI API test successful",
                    fix_suggestion=None
                ))
            else:
                issues.append(ValidationIssue(
                    level="warning",
                    component="live_test",
                    message=f"Unexpected OpenAI API response: {response_text}",
                    fix_suggestion="API works but response was unexpected"
                ))
        else:
            issues.append(ValidationIssue(
                level="warning",
                component="live_test",
                message="OpenAI API returned empty or invalid response",
                fix_suggestion="Check API key permissions and quota"
            ))

    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            issues.append(ValidationIssue(
                level="error",
                component="live_test",
                message="OpenAI API authentication failed",
                fix_suggestion="Check your OPENAI_API_KEY is valid and has sufficient permissions"
            ))
        elif "quota" in error_msg or "billing" in error_msg:
            issues.append(ValidationIssue(
                level="error",
                component="live_test",
                message="OpenAI API quota or billing issue",
                fix_suggestion="Check your OpenAI account has available credits"
            ))
        elif "rate limit" in error_msg:
            issues.append(ValidationIssue(
                level="warning",
                component="live_test",
                message="OpenAI API rate limit hit during testing",
                fix_suggestion="API key is valid but hit rate limits - this is normal"
            ))
        else:
            issues.append(ValidationIssue(
                level="error",
                component="live_test",
                message=f"Live OpenAI test failed: {e}",
                fix_suggestion="Check API key, network connectivity, and OpenAI service status"
            ))

    return issues


def validate_openai_setup() -> ValidationResult:
    """
    Comprehensive validation of GenOps OpenAI setup.

    Returns:
        ValidationResult with overall status and detailed issues
    """
    all_issues = []

    # Run all validation checks
    all_issues.extend(check_environment_variables())
    all_issues.extend(check_dependencies())
    all_issues.extend(check_genops_imports())
    all_issues.extend(test_basic_functionality())
    all_issues.extend(test_opentelemetry_setup())
    all_issues.extend(test_live_openai_connection())

    # Categorize issues
    errors = [issue for issue in all_issues if issue.level == "error"]
    warnings = [issue for issue in all_issues if issue.level == "warning"]
    info = [issue for issue in all_issues if issue.level == "info"]

    # Determine overall validity
    is_valid = len(errors) == 0

    # Create summary
    summary = {
        "total_checks": len(all_issues),
        "errors": len(errors),
        "warnings": len(warnings),
        "info": len(info),
        "components_checked": list({issue.component for issue in all_issues})
    }

    return ValidationResult(
        is_valid=is_valid,
        issues=all_issues,
        summary=summary
    )


def print_openai_validation_result(result: ValidationResult) -> None:
    """Print validation result in a user-friendly format."""

    if result.is_valid:
        print("✅ GenOps OpenAI setup is valid!")
    else:
        print("❌ GenOps OpenAI setup has issues that need attention")

    print("\n📊 Validation Summary:")
    print(f"   Total checks: {result.summary['total_checks']}")
    print(f"   Errors: {result.summary['errors']}")
    print(f"   Warnings: {result.summary['warnings']}")
    print(f"   Info: {result.summary['info']}")

    if result.issues:
        print("\n🔍 Issues Found:")

        # Group issues by component
        issues_by_component = {}
        for issue in result.issues:
            if issue.component not in issues_by_component:
                issues_by_component[issue.component] = []
            issues_by_component[issue.component].append(issue)

        for component, issues in issues_by_component.items():
            print(f"\n  📦 {component.title()}:")

            for issue in issues:
                if issue.level == "error":
                    icon = "❌"
                elif issue.level == "warning":
                    icon = "⚠️ "
                else:
                    icon = "ℹ️ "

                print(f"    {icon} {issue.message}")
                if issue.fix_suggestion:
                    print(f"       💡 {issue.fix_suggestion}")

    if not result.is_valid:
        print("\n🔧 Next Steps:")
        print("   1. Fix the errors listed above")
        print("   2. Run validation again: python -c \"from genops.providers.openai_validation import validate_openai_setup, print_openai_validation_result; print_openai_validation_result(validate_openai_setup())\"")
        print("   3. Check the troubleshooting guide in documentation")


if __name__ == "__main__":
    """Run validation when script is executed directly."""
    result = validate_openai_setup()
    print_openai_validation_result(result)
