"""
Validation utilities for Hugging Face integration setup.
Helps developers verify their GenOps Hugging Face integration is working correctly.
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

    # Optional but recommended variables for Hugging Face
    recommended_vars = {
        "HF_TOKEN": "Hugging Face token for accessing private models and higher rate limits",
        "HUGGINGFACE_HUB_TOKEN": "Alternative name for Hugging Face token",
    }

    # Check if at least one HF token is set
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        issues.append(ValidationIssue(
            level="warning",
            component="environment",
            message="No Hugging Face token found. Public models will work but with rate limits.",
            fix_suggestion="Set HF_TOKEN with: export HF_TOKEN=your_hf_token_here"
        ))

    # OpenTelemetry configuration
    otel_vars = {
        "OTEL_SERVICE_NAME": "OpenTelemetry service name for telemetry identification",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "OpenTelemetry collector endpoint for telemetry export",
        "OTEL_RESOURCE_ATTRIBUTES": "Additional OpenTelemetry resource attributes"
    }

    for var, description in otel_vars.items():
        if not os.getenv(var):
            issues.append(ValidationIssue(
                level="info",
                component="telemetry",
                message=f"Optional OpenTelemetry variable not set: {var} ({description})",
                fix_suggestion=f"Set {var} for enhanced telemetry: export {var}=your_value_here"
            ))

    return issues


def check_dependencies() -> list[ValidationIssue]:
    """Check if required and optional dependencies are installed."""
    issues = []

    # Core dependencies
    core_deps = {
        "huggingface_hub": "Required for Hugging Face API access",
    }

    for package, description in core_deps.items():
        try:
            __import__(package)
        except ImportError:
            issues.append(ValidationIssue(
                level="error",
                component="dependencies",
                message=f"Missing required dependency: {package} ({description})",
                fix_suggestion=f"Install with: pip install {package}"
            ))

    # OpenTelemetry dependencies
    otel_deps = {
        "opentelemetry": "Required for telemetry export",
        "opentelemetry.sdk": "Required for OpenTelemetry SDK",
        "opentelemetry.exporter.otlp": "Required for OTLP export"
    }

    for package, description in otel_deps.items():
        try:
            __import__(package)
        except ImportError:
            issues.append(ValidationIssue(
                level="warning",
                component="dependencies",
                message=f"Optional telemetry dependency missing: {package} ({description})",
                fix_suggestion=f"Install with: pip install {package.replace('.', '-')}"
            ))

    # Optional AI/ML dependencies
    optional_deps = {
        "torch": "Recommended for local model inference and advanced features",
        "transformers": "Recommended for local Transformers model support",
        "datasets": "Recommended for dataset integration features",
        "accelerate": "Recommended for optimized model loading"
    }

    missing_optional = []
    for package, description in optional_deps.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(f"{package} ({description})")

    if missing_optional:
        issues.append(ValidationIssue(
            level="info",
            component="dependencies",
            message=f"Optional AI/ML dependencies not installed: {', '.join(missing_optional[:2])}{'...' if len(missing_optional) > 2 else ''}",
            fix_suggestion="Install AI/ML extras with: pip install genops-ai[huggingface]"
        ))

    return issues


def check_huggingface_connectivity() -> list[ValidationIssue]:
    """Test basic connectivity to Hugging Face API."""
    issues = []

    try:
        from huggingface_hub import InferenceClient
        
        # Test basic connectivity with a simple model
        client = InferenceClient()
        
        # Try a very lightweight test - just checking if we can create the client
        # We avoid making actual API calls to prevent hitting rate limits during validation
        if hasattr(client, 'text_generation'):
            issues.append(ValidationIssue(
                level="info",
                component="connectivity",
                message="Hugging Face InferenceClient created successfully",
                fix_suggestion=None
            ))
        else:
            issues.append(ValidationIssue(
                level="warning",
                component="connectivity",
                message="Hugging Face client created but text_generation method not available",
                fix_suggestion="Update huggingface_hub to latest version: pip install --upgrade huggingface_hub"
            ))
            
    except ImportError:
        issues.append(ValidationIssue(
            level="error",
            component="connectivity",
            message="Cannot import huggingface_hub InferenceClient",
            fix_suggestion="Install huggingface_hub: pip install huggingface_hub"
        ))
    except Exception as e:
        issues.append(ValidationIssue(
            level="warning",
            component="connectivity",
            message=f"Issue creating Hugging Face client: {e}",
            fix_suggestion="Check your internet connection and Hugging Face token if using private models"
        ))

    return issues


def check_genops_integration() -> list[ValidationIssue]:
    """Verify GenOps Hugging Face adapter functionality."""
    issues = []

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Test adapter creation
        try:
            adapter = GenOpsHuggingFaceAdapter()
            
            # Test basic methods
            if hasattr(adapter, 'get_supported_tasks'):
                supported_tasks = adapter.get_supported_tasks()
                if supported_tasks:
                    issues.append(ValidationIssue(
                        level="info",
                        component="genops_integration",
                        message=f"GenOps Hugging Face adapter working. Supports {len(supported_tasks)} AI tasks.",
                        fix_suggestion=None
                    ))
                else:
                    issues.append(ValidationIssue(
                        level="warning",
                        component="genops_integration",
                        message="GenOps adapter created but no supported tasks found",
                        fix_suggestion="Check GenOps installation and Hugging Face integration"
                    ))
                    
            # Test provider detection
            if hasattr(adapter, 'detect_provider_for_model'):
                test_providers = {
                    "gpt-3.5-turbo": "openai",
                    "claude-3-sonnet": "anthropic",
                    "microsoft/DialoGPT-medium": "huggingface_hub"
                }
                
                correct_detections = 0
                for model, expected_provider in test_providers.items():
                    detected = adapter.detect_provider_for_model(model)
                    if detected == expected_provider:
                        correct_detections += 1
                
                if correct_detections == len(test_providers):
                    issues.append(ValidationIssue(
                        level="info",
                        component="genops_integration",
                        message="Provider detection working correctly for all test models",
                        fix_suggestion=None
                    ))
                else:
                    issues.append(ValidationIssue(
                        level="warning",
                        component="genops_integration",
                        message=f"Provider detection working for {correct_detections}/{len(test_providers)} test models",
                        fix_suggestion="Check model name patterns and provider detection logic"
                    ))
                    
        except Exception as e:
            issues.append(ValidationIssue(
                level="error",
                component="genops_integration",
                message=f"Failed to create GenOps Hugging Face adapter: {e}",
                fix_suggestion="Check GenOps installation: pip install --upgrade genops-ai"
            ))
            
    except ImportError:
        issues.append(ValidationIssue(
            level="error",
            component="genops_integration",
            message="Cannot import GenOps Hugging Face adapter",
            fix_suggestion="Install GenOps AI with Hugging Face support: pip install genops-ai[huggingface]"
        ))

    return issues


def check_cost_calculation() -> list[ValidationIssue]:
    """Test cost calculation functionality."""
    issues = []

    try:
        from genops.providers.huggingface_pricing import (
            detect_model_provider,
            calculate_huggingface_cost,
            get_provider_info
        )
        
        # Test provider detection
        test_cases = [
            ("gpt-4", "openai"),
            ("claude-3-sonnet", "anthropic"),
            ("microsoft/DialoGPT-medium", "huggingface_hub"),
            ("mistral-7b-instruct", "mistral")
        ]
        
        detection_success = 0
        for model, expected in test_cases:
            detected = detect_model_provider(model)
            if detected == expected:
                detection_success += 1
        
        if detection_success == len(test_cases):
            issues.append(ValidationIssue(
                level="info",
                component="cost_calculation",
                message="Provider detection working correctly for all test models",
                fix_suggestion=None
            ))
        else:
            issues.append(ValidationIssue(
                level="warning",
                component="cost_calculation",
                message=f"Provider detection working for {detection_success}/{len(test_cases)} models",
                fix_suggestion="Check provider detection patterns in pricing module"
            ))
        
        # Test cost calculation
        try:
            test_cost = calculate_huggingface_cost(
                provider="openai",
                model="gpt-3.5-turbo",
                input_tokens=100,
                output_tokens=50
            )
            
            if isinstance(test_cost, (int, float)) and test_cost >= 0:
                issues.append(ValidationIssue(
                    level="info",
                    component="cost_calculation",
                    message=f"Cost calculation working (test result: ${test_cost:.6f})",
                    fix_suggestion=None
                ))
            else:
                issues.append(ValidationIssue(
                    level="warning",
                    component="cost_calculation",
                    message="Cost calculation returned unexpected result",
                    fix_suggestion="Check pricing data and calculation logic"
                ))
                
        except Exception as e:
            issues.append(ValidationIssue(
                level="warning",
                component="cost_calculation",
                message=f"Cost calculation test failed: {e}",
                fix_suggestion="Check pricing module installation and data"
            ))
            
        # Test provider info
        try:
            provider_info = get_provider_info("gpt-3.5-turbo")
            if isinstance(provider_info, dict) and "provider" in provider_info:
                issues.append(ValidationIssue(
                    level="info",
                    component="cost_calculation",
                    message="Provider info lookup working correctly",
                    fix_suggestion=None
                ))
            else:
                issues.append(ValidationIssue(
                    level="warning",
                    component="cost_calculation",
                    message="Provider info lookup returned unexpected format",
                    fix_suggestion="Check provider info data structure"
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                level="warning",
                component="cost_calculation",
                message=f"Provider info test failed: {e}",
                fix_suggestion="Check pricing module provider info functionality"
            ))
            
    except ImportError:
        issues.append(ValidationIssue(
            level="error",
            component="cost_calculation",
            message="Cannot import Hugging Face pricing utilities",
            fix_suggestion="Check GenOps Hugging Face pricing module installation"
        ))

    return issues


def validate_huggingface_setup() -> ValidationResult:
    """
    Comprehensive validation of Hugging Face GenOps setup.
    
    Returns:
        ValidationResult with overall status and detailed issues
    """
    all_issues = []
    
    # Run all validation checks
    validation_functions = [
        check_environment_variables,
        check_dependencies,
        check_huggingface_connectivity,
        check_genops_integration,
        check_cost_calculation,
    ]
    
    for check_func in validation_functions:
        try:
            issues = check_func()
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(ValidationIssue(
                level="error",
                component="validation_framework",
                message=f"Validation check {check_func.__name__} failed: {e}",
                fix_suggestion="Contact support or check GenOps installation"
            ))
    
    # Determine overall validity
    has_errors = any(issue.level == "error" for issue in all_issues)
    is_valid = not has_errors
    
    # Create summary
    summary = {
        "total_issues": len(all_issues),
        "errors": len([i for i in all_issues if i.level == "error"]),
        "warnings": len([i for i in all_issues if i.level == "warning"]),
        "info": len([i for i in all_issues if i.level == "info"]),
        "components_checked": len(validation_functions),
    }
    
    return ValidationResult(
        is_valid=is_valid,
        issues=all_issues,
        summary=summary
    )


def print_huggingface_validation_result(result: ValidationResult) -> None:
    """Print validation result in user-friendly format."""
    print("\n" + "="*60)
    print("🤗 GenOps Hugging Face Setup Validation")
    print("="*60)
    
    # Overall status
    if result.is_valid:
        print("✅ Overall Status: VALID - Ready to use!")
    else:
        print("❌ Overall Status: ISSUES FOUND - See details below")
    
    # Summary
    summary = result.summary
    print(f"\n📊 Summary:")
    print(f"   • Components checked: {summary['components_checked']}")
    print(f"   • Total issues found: {summary['total_issues']}")
    if summary['errors'] > 0:
        print(f"   • ❌ Errors: {summary['errors']} (must fix)")
    if summary['warnings'] > 0:
        print(f"   • ⚠️  Warnings: {summary['warnings']} (recommended to fix)")  
    if summary['info'] > 0:
        print(f"   • ℹ️  Info: {summary['info']} (informational)")
    
    # Group issues by component
    if result.issues:
        print(f"\n🔍 Detailed Issues:")
        
        by_component = {}
        for issue in result.issues:
            if issue.component not in by_component:
                by_component[issue.component] = []
            by_component[issue.component].append(issue)
        
        for component, issues in by_component.items():
            print(f"\n   📂 {component.upper()}:")
            
            for issue in issues:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.level]
                print(f"      {icon} {issue.message}")
                if issue.fix_suggestion:
                    print(f"         💡 Fix: {issue.fix_suggestion}")
    
    # Next steps
    print(f"\n🚀 Next Steps:")
    
    if result.is_valid:
        print("   1. Your setup looks good! Try running the examples:")
        print("      python examples/huggingface/basic_usage.py")
        print("   2. Check out the documentation for advanced features")
        print("   3. Set up your OpenTelemetry exporter for production use")
    else:
        errors = [i for i in result.issues if i.level == "error"]
        if errors:
            print("   1. Fix the errors shown above (marked with ❌)")
            print("   2. Re-run validation: python -c 'from genops.providers.huggingface import validate_setup; validate_setup()'")
            print("   3. Check the Hugging Face quickstart guide for help")
        else:
            print("   1. Review warnings (⚠️) - they may affect functionality")
            print("   2. Try running basic examples to test your setup")
            print("   3. Consider fixing warnings for optimal experience")
    
    print(f"\n📖 Documentation:")
    print("   • Quickstart: docs/huggingface-quickstart.md")
    print("   • Integration Guide: docs/integrations/huggingface.md")
    print("   • Examples: examples/huggingface/")
    print("   • Support: https://github.com/KoshiHQ/GenOps-AI/issues")
    
    print("\n" + "="*60 + "\n")


# Convenience function for quick validation
def quick_validate() -> bool:
    """
    Quick validation check - returns True if setup is valid, False otherwise.
    Prints minimal output.
    """
    result = validate_huggingface_setup()
    if result.is_valid:
        print("✅ Hugging Face setup validation passed!")
        return True
    else:
        error_count = len([i for i in result.issues if i.level == "error"])
        print(f"❌ Hugging Face setup validation failed with {error_count} error(s)")
        print("Run full validation for details: from genops.providers.huggingface import print_validation_result, validate_setup; print_validation_result(validate_setup())")
        return False


if __name__ == "__main__":
    # When run directly, perform full validation
    result = validate_huggingface_setup()
    print_huggingface_validation_result(result)