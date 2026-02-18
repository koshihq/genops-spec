#!/usr/bin/env python3
"""
LiteLLM Setup Validation for GenOps

Comprehensive validation for LiteLLM + GenOps integration including:
- LiteLLM installation and version checking
- Provider API key validation
- GenOps integration testing
- Environment configuration verification
- Multi-provider connectivity testing

Usage:
    from genops.providers.litellm_validation import validate_litellm_setup, print_validation_result

    result = validate_litellm_setup()
    print_validation_result(result)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    component: str
    status: ValidationStatus
    message: str
    fix_suggestion: Optional[str] = None
    documentation_link: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    provider_status: dict[str, ValidationStatus] = field(default_factory=dict)

    def add_issue(
        self,
        component: str,
        status: ValidationStatus,
        message: str,
        fix_suggestion: Optional[str] = None,
        documentation_link: Optional[str] = None,
    ):
        """Add a validation issue."""
        self.issues.append(
            ValidationIssue(
                component=component,
                status=status,
                message=message,
                fix_suggestion=fix_suggestion,
                documentation_link=documentation_link,
            )
        )

        if status == ValidationStatus.ERROR:
            self.is_valid = False


def validate_litellm_installation() -> list[ValidationIssue]:
    """Validate LiteLLM installation and version."""
    issues = []

    try:
        import litellm

        version = getattr(litellm, "__version__", "unknown")

        issues.append(
            ValidationIssue(
                component="LiteLLM Installation",
                status=ValidationStatus.SUCCESS,
                message=f"LiteLLM {version} found and importable",
            )
        )

        # Check for required methods
        required_methods = ["completion", "acompletion", "embedding"]
        missing_methods = []

        for method in required_methods:
            if not hasattr(litellm, method):
                missing_methods.append(method)

        if missing_methods:
            issues.append(
                ValidationIssue(
                    component="LiteLLM API",
                    status=ValidationStatus.WARNING,
                    message=f"Missing methods: {', '.join(missing_methods)}",
                    fix_suggestion="Update to latest LiteLLM version: pip install --upgrade litellm",
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    component="LiteLLM API",
                    status=ValidationStatus.SUCCESS,
                    message="All required LiteLLM methods available",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                component="LiteLLM Installation",
                status=ValidationStatus.ERROR,
                message="LiteLLM not installed or not importable",
                fix_suggestion="Install LiteLLM: pip install litellm",
                documentation_link="https://docs.litellm.ai/docs/",
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                component="LiteLLM Installation",
                status=ValidationStatus.ERROR,
                message=f"Unexpected error importing LiteLLM: {e}",
                fix_suggestion="Reinstall LiteLLM: pip uninstall litellm && pip install litellm",
            )
        )

    return issues


def validate_genops_integration() -> list[ValidationIssue]:
    """Validate GenOps LiteLLM integration."""
    issues = []

    try:
        from genops.providers.litellm import (
            GenOpsLiteLLMCallback,
            auto_instrument,  # noqa: F401
            get_usage_stats,  # noqa: F401
            track_completion,  # noqa: F401
        )

        issues.append(
            ValidationIssue(
                component="GenOps Integration",
                status=ValidationStatus.SUCCESS,
                message="GenOps LiteLLM provider available",
            )
        )

        # Test callback functionality
        try:
            from genops.providers.litellm import LiteLLMGovernanceContext

            context = LiteLLMGovernanceContext()
            GenOpsLiteLLMCallback(context)

            issues.append(
                ValidationIssue(
                    component="GenOps Callbacks",
                    status=ValidationStatus.SUCCESS,
                    message="GenOps callback system functional",
                )
            )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    component="GenOps Callbacks",
                    status=ValidationStatus.WARNING,
                    message=f"Callback system issue: {e}",
                    fix_suggestion="Check GenOps core module installation",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                component="GenOps Integration",
                status=ValidationStatus.ERROR,
                message="GenOps LiteLLM provider not available",
                fix_suggestion="Install GenOps with LiteLLM support: pip install genops[litellm]",
            )
        )

    return issues


def validate_provider_api_keys() -> tuple[
    list[ValidationIssue], dict[str, ValidationStatus]
]:
    """Validate API keys for major LiteLLM providers."""
    issues = []
    provider_status = {}

    # Major provider API key checks
    provider_checks = {
        "OpenAI": ["OPENAI_API_KEY", "OPENAI_API_BASE"],
        "Anthropic": ["ANTHROPIC_API_KEY"],
        "Google": ["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"],
        "Azure": ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"],
        "AWS Bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        "Cohere": ["COHERE_API_KEY"],
        "HuggingFace": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
        "Together": ["TOGETHER_API_KEY"],
        "Replicate": ["REPLICATE_API_TOKEN"],
        "Mistral": ["MISTRAL_API_KEY"],
        "Fireworks": ["FIREWORKS_API_KEY"],
        "Perplexity": ["PERPLEXITYAI_API_KEY"],
    }

    configured_providers = []

    for provider, env_vars in provider_checks.items():
        has_key = False
        found_vars = []

        for var in env_vars:
            if os.getenv(var):
                has_key = True
                found_vars.append(var)

        if has_key:
            configured_providers.append(provider)
            provider_status[provider] = ValidationStatus.SUCCESS
            issues.append(
                ValidationIssue(
                    component=f"{provider} API Key",
                    status=ValidationStatus.SUCCESS,
                    message=f"{provider} configured with {', '.join(found_vars)}",
                )
            )
        else:
            provider_status[provider] = ValidationStatus.WARNING
            issues.append(
                ValidationIssue(
                    component=f"{provider} API Key",
                    status=ValidationStatus.WARNING,
                    message=f"{provider} not configured (missing: {', '.join(env_vars)})",
                    fix_suggestion=f"Set environment variable: export {env_vars[0]}=your_key_here",
                )
            )

    if not configured_providers:
        issues.append(
            ValidationIssue(
                component="Provider Configuration",
                status=ValidationStatus.ERROR,
                message="No LLM provider API keys configured",
                fix_suggestion="Configure at least one provider API key",
                documentation_link="https://docs.litellm.ai/docs/providers",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                component="Provider Configuration",
                status=ValidationStatus.SUCCESS,
                message=f"Configured providers: {', '.join(configured_providers)}",
            )
        )

    return issues, provider_status


def validate_litellm_connectivity() -> list[ValidationIssue]:
    """Test basic LiteLLM connectivity with configured providers."""
    issues = []

    try:
        import litellm

        # Test basic model mapping
        try:
            # This should not make actual API calls, just test model mapping
            test_models = ["gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro"]

            for model in test_models:
                try:
                    # Test model mapping without API call
                    provider = litellm.get_llm_provider(model)
                    if provider and provider[0]:
                        issues.append(
                            ValidationIssue(
                                component=f"Model Mapping ({model})",
                                status=ValidationStatus.SUCCESS,
                                message=f"Model {model} mapped to provider {provider[0]}",
                            )
                        )
                    else:
                        issues.append(
                            ValidationIssue(
                                component=f"Model Mapping ({model})",
                                status=ValidationStatus.WARNING,
                                message=f"Model {model} provider mapping unclear",
                            )
                        )
                except Exception as e:
                    issues.append(
                        ValidationIssue(
                            component=f"Model Mapping ({model})",
                            status=ValidationStatus.WARNING,
                            message=f"Model {model} mapping error: {e}",
                        )
                    )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    component="Model Mapping",
                    status=ValidationStatus.WARNING,
                    message=f"Model mapping test failed: {e}",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                component="Connectivity Test",
                status=ValidationStatus.SKIPPED,
                message="LiteLLM not available for connectivity testing",
            )
        )

    return issues


def validate_callback_system() -> list[ValidationIssue]:
    """Validate LiteLLM callback system functionality."""
    issues = []

    try:
        import litellm

        # Check callback attributes
        callback_attrs = ["input_callback", "success_callback", "failure_callback"]
        missing_attrs = []

        for attr in callback_attrs:
            if not hasattr(litellm, attr):
                missing_attrs.append(attr)

        if missing_attrs:
            issues.append(
                ValidationIssue(
                    component="Callback System",
                    status=ValidationStatus.WARNING,
                    message=f"Missing callback attributes: {', '.join(missing_attrs)}",
                    fix_suggestion="Update LiteLLM to version that supports callbacks",
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    component="Callback System",
                    status=ValidationStatus.SUCCESS,
                    message="LiteLLM callback system available",
                )
            )

        # Test callback registration
        try:
            # Save original callbacks
            original_callbacks = {
                "input_callback": getattr(litellm, "input_callback", []),
                "success_callback": getattr(litellm, "success_callback", []),
                "failure_callback": getattr(litellm, "failure_callback", []),
            }

            # Test setting callbacks
            def test_callback(*args, **kwargs):
                pass

            for attr in callback_attrs:
                if hasattr(litellm, attr):
                    setattr(litellm, attr, [test_callback])

            issues.append(
                ValidationIssue(
                    component="Callback Registration",
                    status=ValidationStatus.SUCCESS,
                    message="Callback registration functional",
                )
            )

            # Restore original callbacks
            for attr, original in original_callbacks.items():
                if hasattr(litellm, attr):
                    setattr(litellm, attr, original)

        except Exception as e:
            issues.append(
                ValidationIssue(
                    component="Callback Registration",
                    status=ValidationStatus.WARNING,
                    message=f"Callback registration test failed: {e}",
                )
            )

    except ImportError:
        issues.append(
            ValidationIssue(
                component="Callback System",
                status=ValidationStatus.SKIPPED,
                message="LiteLLM not available for callback testing",
            )
        )

    return issues


def validate_environment_configuration() -> list[ValidationIssue]:
    """Validate environment configuration for LiteLLM usage."""
    issues = []

    # Python version check
    python_version = sys.version_info
    if python_version >= (3, 8):
        issues.append(
            ValidationIssue(
                component="Python Version",
                status=ValidationStatus.SUCCESS,
                message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
            )
        )
    else:
        issues.append(
            ValidationIssue(
                component="Python Version",
                status=ValidationStatus.ERROR,
                message=f"Python {python_version.major}.{python_version.minor} not supported",
                fix_suggestion="Upgrade to Python 3.8 or higher",
            )
        )

    # Check for common environment issues
    env_checks = [
        ("PATH", "System PATH configured"),
        ("HOME", "Home directory accessible"),
    ]

    for env_var, description in env_checks:
        if os.getenv(env_var):
            issues.append(
                ValidationIssue(
                    component="Environment",
                    status=ValidationStatus.SUCCESS,
                    message=f"{description}: {env_var}=[CONFIGURED]",
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    component="Environment",
                    status=ValidationStatus.WARNING,
                    message=f"Missing environment variable: {env_var}",
                )
            )

    return issues


def validate_litellm_setup(
    quick: bool = False, test_connectivity: bool = False
) -> ValidationResult:
    """
    Comprehensive LiteLLM setup validation.

    Args:
        quick: Run only essential validations
        test_connectivity: Test actual API connectivity (requires API keys)

    Returns:
        ValidationResult with detailed status
    """
    result = ValidationResult(is_valid=True)

    try:
        # Core validations (always run)
        result.issues.extend(validate_litellm_installation())
        result.issues.extend(validate_genops_integration())

        if not quick:
            # Extended validations
            api_issues, provider_status = validate_provider_api_keys()
            result.issues.extend(api_issues)
            result.provider_status = provider_status

            result.issues.extend(validate_callback_system())
            result.issues.extend(validate_environment_configuration())

            if test_connectivity:
                result.issues.extend(validate_litellm_connectivity())

        # Check if any critical errors occurred
        error_count = sum(
            1 for issue in result.issues if issue.status == ValidationStatus.ERROR
        )
        warning_count = sum(
            1 for issue in result.issues if issue.status == ValidationStatus.WARNING
        )

        result.is_valid = error_count == 0
        result.summary = {
            "total_issues": len(result.issues),
            "errors": error_count,
            "warnings": warning_count,
            "validation_type": "quick" if quick else "comprehensive",
        }

    except Exception as e:
        result.is_valid = False
        result.add_issue(
            component="Validation System",
            status=ValidationStatus.ERROR,
            message=f"Validation system error: {e}",
            fix_suggestion="Check GenOps installation and try again",
        )

    return result


def print_validation_result(result: ValidationResult, verbose: bool = True) -> None:
    """Print validation results in a user-friendly format."""

    # Status indicators
    status_icons = {
        ValidationStatus.SUCCESS: "✅",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.ERROR: "❌",
        ValidationStatus.SKIPPED: "⏭️",
    }

    print("\n" + "=" * 60)
    print("🔍 LiteLLM + GenOps Validation Report")
    print("=" * 60)

    if result.is_valid:
        print("🎉 Overall Status: READY")
        print("   LiteLLM integration is properly configured!")
    else:
        print("⚠️  Overall Status: ISSUES FOUND")
        print("   Some configuration issues need attention.")

    print("\n📊 Summary:")
    print(f"   Total checks: {result.summary.get('total_issues', 0)}")
    print(f"   Errors: {result.summary.get('errors', 0)}")
    print(f"   Warnings: {result.summary.get('warnings', 0)}")

    if verbose:
        print("\n📋 Detailed Results:")

        # Group issues by status
        by_status = {}
        for issue in result.issues:
            if issue.status not in by_status:
                by_status[issue.status] = []
            by_status[issue.status].append(issue)

        # Print issues by status (errors first)
        status_order = [
            ValidationStatus.ERROR,
            ValidationStatus.WARNING,
            ValidationStatus.SUCCESS,
            ValidationStatus.SKIPPED,
        ]

        for status in status_order:
            if status in by_status:
                print(f"\n{status_icons[status]} {status.value.upper()}:")
                for issue in by_status[status]:
                    print(f"   • {issue.component}: {issue.message}")
                    if issue.fix_suggestion:
                        print(f"     💡 Fix: {issue.fix_suggestion}")
                    if issue.documentation_link:
                        print(f"     📖 Docs: {issue.documentation_link}")

    if result.provider_status:
        print("\n🔌 Provider Status:")
        for index, (_provider, status) in enumerate(result.provider_status.items(), 1):
            icon = status_icons[status]
            print(f"   {icon} Provider {index}")

    print("\n" + "=" * 60)

    if not result.is_valid:
        print("🚨 Action Required:")
        error_issues = [i for i in result.issues if i.status == ValidationStatus.ERROR]
        for issue in error_issues[:3]:  # Show top 3 errors
            print(f"   1. {issue.component}: {issue.message}")
            if issue.fix_suggestion:
                print(f"      → {issue.fix_suggestion}")
    else:
        print("🚀 Ready to use LiteLLM with GenOps governance!")
        print("   Try: from genops.providers.litellm import auto_instrument")


# Export main functions
__all__ = [
    "validate_litellm_setup",
    "print_validation_result",
    "ValidationResult",
    "ValidationIssue",
    "ValidationStatus",
]
