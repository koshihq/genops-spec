"""
Validation utilities for LangChain integration setup.
Helps developers verify their GenOps LangChain integration is working correctly.
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
        "OPENAI_API_KEY": "OpenAI API key for cost calculation and LLM access"
    }

    for var, description in required_vars.items():
        if not os.getenv(var):
            issues.append(
                ValidationIssue(
                    level="error",
                    component="environment",
                    message=f"Missing required environment variable: {var} ({description})",
                    fix_suggestion=f"Set {var} with: export {var}=your_key_here",
                )
            )

    # Optional but recommended variables
    optional_vars = {
        "OTEL_SERVICE_NAME": "OpenTelemetry service name for telemetry identification",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "OTLP endpoint for telemetry export",
        "ANTHROPIC_API_KEY": "Anthropic API key for cost calculation",
        "COHERE_API_KEY": "Cohere API key for cost calculation",
    }

    for var, description in optional_vars.items():
        if not os.getenv(var):
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="environment",
                    message=f"Optional environment variable not set: {var}",
                    fix_suggestion=f"For {description}, set: export {var}=your_value",
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

    # Core dependencies
    core_deps = {
        "opentelemetry": "OpenTelemetry SDK",
        "langchain": "LangChain framework",
    }

    for module, description in core_deps.items():
        try:
            __import__(module)
        except ImportError:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="dependencies",
                    message=f"Required dependency not found: {module}",
                    fix_suggestion=f"Install {description} with: pip install {module}",
                )
            )

    # LangChain-specific imports
    langchain_modules = {
        "langchain.chains": "LangChain chains module",
        "langchain.llms": "LangChain LLMs module",
        "langchain.schema": "LangChain schema module",
    }

    for module, description in langchain_modules.items():
        try:
            __import__(module)
        except ImportError:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="dependencies",
                    message=f"LangChain module not available: {module} ({description})",
                    fix_suggestion="Ensure LangChain is properly installed: pip install langchain",
                )
            )

    # Optional provider dependencies
    optional_providers = {
        "anthropic": "Anthropic LLM provider",
        "cohere": "Cohere LLM provider",
        "chromadb": "ChromaDB vector store",
    }

    for module, description in optional_providers.items():
        try:
            __import__(module)
        except ImportError:
            issues.append(
                ValidationIssue(
                    level="info",
                    component="dependencies",
                    message=f"Optional dependency not available: {module}",
                    fix_suggestion=f"For {description} support, install: pip install {module}",
                )
            )

    return issues


def check_genops_imports() -> list[ValidationIssue]:
    """Check if GenOps modules can be imported correctly."""
    issues = []

    genops_modules = {
        "genops.providers.langchain": "GenOps LangChain adapter",
        "genops.providers.langchain.adapter": "LangChain adapter implementation",
        "genops.providers.langchain.cost_aggregator": "Cost aggregation functionality",
        "genops.providers.langchain.rag_monitor": "RAG monitoring capabilities",
        "genops.core.telemetry": "Core telemetry functionality",
    }

    for module, _description in genops_modules.items():
        try:
            __import__(module)
        except ImportError:
            issues.append(
                ValidationIssue(
                    level="error",
                    component="genops",
                    message=f"GenOps module not available: {module}",
                    fix_suggestion="Ensure GenOps is installed: pip install genops-ai[langchain]",
                )
            )

    return issues


def test_basic_functionality() -> list[ValidationIssue]:
    """Test basic GenOps LangChain functionality."""
    issues = []

    try:
        # Test adapter creation
        from genops.providers.langchain import instrument_langchain

        adapter = instrument_langchain()

        # Test framework properties
        framework_name = adapter.get_framework_name()
        if framework_name != "langchain":
            issues.append(
                ValidationIssue(
                    level="error",
                    component="functionality",
                    message=f"Unexpected framework name: {framework_name}",
                    fix_suggestion="Check GenOps LangChain adapter installation",
                )
            )

        framework_type = adapter.get_framework_type()
        if framework_type != "orchestration":
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="functionality",
                    message=f"Unexpected framework type: {framework_type}",
                    fix_suggestion="This may indicate a version mismatch",
                )
            )

    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                component="functionality",
                message=f"Failed to create LangChain adapter: {e}",
                fix_suggestion="Check GenOps installation and dependencies",
            )
        )

    try:
        # Test cost aggregator
        from genops.providers.langchain import get_cost_aggregator

        aggregator = get_cost_aggregator()

        if not hasattr(aggregator, "provider_cost_calculators"):
            issues.append(
                ValidationIssue(
                    level="error",
                    component="functionality",
                    message="Cost aggregator missing provider calculators",
                    fix_suggestion="Check GenOps cost calculation setup",
                )
            )

        # Check if any cost calculators are available
        if not aggregator.provider_cost_calculators:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="functionality",
                    message="No cost calculators available",
                    fix_suggestion="Install provider packages (openai, anthropic) for cost calculation",
                )
            )

    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                component="functionality",
                message=f"Failed to access cost aggregator: {e}",
                fix_suggestion="Check GenOps cost aggregation setup",
            )
        )

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

    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                component="opentelemetry",
                message=f"OpenTelemetry not working: {e}",
                fix_suggestion="Check OpenTelemetry installation and configuration",
            )
        )

    # Check exporter configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = os.getenv("OTEL_SERVICE_NAME")

    if not service_name:
        issues.append(
            ValidationIssue(
                level="warning",
                component="opentelemetry",
                message="OTEL_SERVICE_NAME not set",
                fix_suggestion="Set service name: export OTEL_SERVICE_NAME=my-langchain-app",
            )
        )

    if not otlp_endpoint:
        issues.append(
            ValidationIssue(
                level="info",
                component="opentelemetry",
                message="OTEL_EXPORTER_OTLP_ENDPOINT not set - telemetry will only be logged",
                fix_suggestion="For telemetry export, set: export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317",
            )
        )

    return issues


def test_live_chain_execution() -> list[ValidationIssue]:
    """Test actual chain execution with monitoring (if API key available)."""
    issues = []

    if not os.getenv("OPENAI_API_KEY"):
        issues.append(
            ValidationIssue(
                level="info",
                component="live_test",
                message="Skipping live test - no OpenAI API key",
                fix_suggestion="Set OPENAI_API_KEY to test live chain execution",
            )
        )
        return issues

    try:
        from langchain.chains import LLMChain
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate

        from genops.providers.langchain import instrument_langchain

        # Create simple test chain
        adapter = instrument_langchain()

        llm = OpenAI(temperature=0.1, max_tokens=50)
        prompt = PromptTemplate.from_template(
            "Say 'Hello from {name}' in exactly those words."
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        # Test instrumented execution
        result = adapter.instrument_chain_run(
            chain, name="GenOps", team="validation-test", project="setup-verification"
        )

        if "Hello from GenOps" in result:
            issues.append(
                ValidationIssue(
                    level="info",
                    component="live_test",
                    message="Live chain execution test successful",
                    fix_suggestion=None,
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    level="warning",
                    component="live_test",
                    message=f"Unexpected chain result: {result}",
                    fix_suggestion="Chain executed but result was unexpected",
                )
            )

    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                component="live_test",
                message=f"Live chain test failed: {e}",
                fix_suggestion="Check API key and network connectivity",
            )
        )

    return issues


def validate_setup() -> ValidationResult:
    """
    Comprehensive validation of GenOps LangChain setup.

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
    all_issues.extend(test_live_chain_execution())

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
        "components_checked": list({issue.component for issue in all_issues}),
    }

    return ValidationResult(is_valid=is_valid, issues=all_issues, summary=summary)


def print_validation_result(result: ValidationResult) -> None:
    """Print validation result in a user-friendly format."""

    if result.is_valid:
        print("✅ GenOps LangChain setup is valid!")
    else:
        print("❌ GenOps LangChain setup has issues that need attention")

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
        print(
            '   2. Run validation again: python -c "from genops.providers.langchain.validation import validate_setup, print_validation_result; print_validation_result(validate_setup())"'
        )
        print(
            "   3. Check the troubleshooting guide: docs/integrations/langchain.md#troubleshooting"
        )


if __name__ == "__main__":
    """Run validation when script is executed directly."""
    result = validate_setup()
    print_validation_result(result)
