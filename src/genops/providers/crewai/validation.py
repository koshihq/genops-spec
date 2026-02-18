#!/usr/bin/env python3
"""
CrewAI Setup Validation

Comprehensive validation system for CrewAI integration with GenOps,
providing actionable diagnostics and setup verification.

Usage:
    from genops.providers.crewai import validate_crewai_setup, print_validation_result

    result = validate_crewai_setup()
    print_validation_result(result)

    if result.is_valid:
        print("✅ Ready to use CrewAI with GenOps!")
    else:
        print("❌ Setup issues found - check recommendations")

Features:
    - CrewAI framework detection and version validation
    - AI provider configuration verification
    - Environment variable and API key validation
    - GenOps component compatibility checks
    - Actionable error messages with fix suggestions
    - Integration testing with sample crew execution
"""

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class ValidationIssue:
    """Represents a validation issue with fix recommendations."""

    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None
    documentation_link: Optional[str] = None


@dataclass
class ValidationResult:
    """Comprehensive validation result."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)
    provider_status: dict[str, bool] = field(default_factory=dict)
    available_features: list[str] = field(default_factory=list)

    def add_issue(
        self,
        level: ValidationLevel,
        category: str,
        message: str,
        details: Optional[str] = None,
        fix_suggestion: Optional[str] = None,
        documentation_link: Optional[str] = None,
    ):
        """Add a validation issue."""
        issue = ValidationIssue(
            level=level,
            category=category,
            message=message,
            details=details,
            fix_suggestion=fix_suggestion,
            documentation_link=documentation_link,
        )
        self.issues.append(issue)

        # Update validation status
        if level == ValidationLevel.ERROR:
            self.is_valid = False


def check_crewai_installation() -> tuple[bool, str, Optional[str]]:
    """Check if CrewAI is properly installed."""
    try:
        import crewai

        version = getattr(crewai, "__version__", "unknown")
        return True, version, None
    except ImportError as e:
        return False, "not_installed", str(e)


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    # CrewAI requires Python 3.8+
    if version.major == 3 and version.minor >= 8:
        return True, version_str
    else:
        return False, version_str


def check_ai_provider_dependencies() -> dict[str, tuple[bool, str]]:
    """Check availability of AI provider dependencies."""
    providers = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google-generativeai": "Google Gemini",
        "cohere": "Cohere",
        "transformers": "Hugging Face Transformers",
    }

    results = {}
    for package, name in providers.items():
        try:
            importlib.import_module(package.replace("-", "_"))
            results[name] = (True, "available")
        except ImportError:
            results[name] = (False, "not_installed")

    return results


def check_environment_variables() -> dict[str, tuple[bool, str]]:
    """Check for required environment variables."""
    env_vars = {
        "OPENAI_API_KEY": "OpenAI API access",
        "ANTHROPIC_API_KEY": "Anthropic API access",
        "GOOGLE_API_KEY": "Google Gemini API access",
        "COHERE_API_KEY": "Cohere API access",
        "HF_TOKEN": "Hugging Face API access",
    }

    results = {}
    for var_name, _description in env_vars.items():
        value = os.getenv(var_name)
        if value:
            # Check if it looks like a valid API key
            if len(value) > 10 and not value.startswith("your_"):
                results[var_name] = (True, "configured")
            else:
                results[var_name] = (False, "invalid_format")
        else:
            results[var_name] = (False, "not_set")

    return results


def check_genops_components() -> dict[str, tuple[bool, str]]:
    """Check GenOps component availability."""
    components = {
        "adapter": "genops.providers.crewai.adapter",
        "cost_aggregator": "genops.providers.crewai.cost_aggregator",
        "agent_monitor": "genops.providers.crewai.agent_monitor",
        "registration": "genops.providers.crewai.registration",
    }

    results = {}
    for name, module_path in components.items():
        try:
            importlib.import_module(module_path)
            results[name] = (True, "available")
        except ImportError as e:
            results[name] = (False, f"error: {str(e)}")

    return results


def test_basic_crew_creation() -> tuple[bool, Optional[str]]:
    """Test basic CrewAI crew creation."""
    try:
        from crewai import Agent, Crew, Task

        # Create a simple agent
        agent = Agent(
            role="Test Agent",
            goal="Perform validation test",
            backstory="A test agent for GenOps validation",
        )

        # Create a simple task
        task = Task(description="Say hello for validation test", agent=agent)

        # Create a crew
        crew = Crew(agents=[agent], tasks=[task])

        # Check basic properties
        if hasattr(crew, "agents") and hasattr(crew, "tasks"):
            return True, None
        else:
            return False, "Crew missing expected attributes"

    except Exception as e:
        return False, str(e)


def validate_crewai_setup(quick: bool = False) -> ValidationResult:
    """
    Validate CrewAI setup for GenOps integration.

    Args:
        quick: If True, skip comprehensive tests (faster validation)

    Returns:
        ValidationResult: Comprehensive validation results
    """
    result = ValidationResult(is_valid=True)

    # System information
    result.system_info = {
        "platform": sys.platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "working_directory": os.getcwd(),
    }

    # 1. Check Python version
    py_valid, py_version = check_python_version()
    result.system_info["python_version"] = py_version

    if not py_valid:
        result.add_issue(
            ValidationLevel.ERROR,
            "python_version",
            f"Python {py_version} is not supported",
            "CrewAI requires Python 3.8 or higher",
            "Upgrade to Python 3.8+ using pyenv, conda, or your system package manager",
            "https://python.org/downloads",
        )
    else:
        result.add_issue(
            ValidationLevel.SUCCESS,
            "python_version",
            f"Python {py_version} is compatible",
        )

    # 2. Check CrewAI installation
    crewai_installed, crewai_version, crewai_error = check_crewai_installation()
    result.system_info["crewai_version"] = crewai_version

    if not crewai_installed:
        result.add_issue(
            ValidationLevel.ERROR,
            "crewai_installation",
            "CrewAI framework not installed",
            crewai_error,
            "Install CrewAI: pip install crewai",
            "https://docs.crewai.com/getting-started/installing-crewai",
        )
    else:
        result.add_issue(
            ValidationLevel.SUCCESS,
            "crewai_installation",
            f"CrewAI {crewai_version} is installed",
        )
        result.available_features.append("crewai_framework")

    # 3. Check AI provider dependencies
    provider_deps = check_ai_provider_dependencies()
    available_providers = 0

    for provider, (available, _status) in provider_deps.items():
        result.provider_status[provider] = available
        if available:
            available_providers += 1
            result.add_issue(
                ValidationLevel.SUCCESS,
                "provider_deps",
                f"{provider} client library is available",
            )
        else:
            result.add_issue(
                ValidationLevel.WARNING,
                "provider_deps",
                f"{provider} client library not installed",
                f"Install with: pip install {provider.lower().replace(' ', '-')}",
                f"Consider installing {provider} for expanded AI capabilities",
            )

    if available_providers == 0:
        result.add_issue(
            ValidationLevel.ERROR,
            "provider_deps",
            "No AI provider libraries found",
            "At least one AI provider library is required",
            "Install at least one: pip install openai anthropic google-generativeai cohere",
        )

    # 4. Check environment variables
    env_vars = check_environment_variables()
    configured_providers = 0

    for var_name, (configured, status) in env_vars.items():
        if configured:
            configured_providers += 1
            result.add_issue(
                ValidationLevel.SUCCESS, "environment", f"{var_name} is configured"
            )
        elif status == "invalid_format":
            result.add_issue(
                ValidationLevel.WARNING,
                "environment",
                f"{var_name} appears to be a placeholder",
                "API key format looks invalid",
                f"Set a valid API key: export {var_name}=your_actual_key_here",
            )
        else:
            result.add_issue(
                ValidationLevel.INFO,
                "environment",
                f"{var_name} not set",
                None,
                f"Set if using corresponding provider: export {var_name}=your_key",
            )

    if configured_providers == 0:
        result.add_issue(
            ValidationLevel.WARNING,
            "environment",
            "No AI provider API keys configured",
            "You'll need API keys to use AI providers with CrewAI",
            "Configure at least one provider API key for full functionality",
        )

    # 5. Check GenOps components
    genops_components = check_genops_components()
    working_components = 0

    for component, (available, status) in genops_components.items():
        if available:
            working_components += 1
            result.add_issue(
                ValidationLevel.SUCCESS,
                "genops_components",
                f"GenOps {component} component is available",
            )
            result.available_features.append(f"genops_{component}")
        else:
            result.add_issue(
                ValidationLevel.ERROR,
                "genops_components",
                f"GenOps {component} component not available",
                status,
                "Ensure GenOps is properly installed: pip install genops-ai[crewai]",
            )

    if working_components != len(genops_components):
        result.add_issue(
            ValidationLevel.ERROR,
            "genops_components",
            "Some GenOps components are missing",
            "GenOps CrewAI integration requires all components",
            "Reinstall GenOps: pip install --upgrade genops-ai[crewai]",
        )

    # 6. Test basic CrewAI functionality (if not quick validation)
    if not quick and crewai_installed:
        crew_test_ok, crew_error = test_basic_crew_creation()

        if crew_test_ok:
            result.add_issue(
                ValidationLevel.SUCCESS,
                "functionality",
                "Basic CrewAI crew creation works",
            )
            result.available_features.append("crew_creation")
        else:
            result.add_issue(
                ValidationLevel.ERROR,
                "functionality",
                "Basic CrewAI crew creation failed",
                crew_error,
                "Check CrewAI installation and dependencies",
            )

    # Final validation summary
    error_count = sum(
        1 for issue in result.issues if issue.level == ValidationLevel.ERROR
    )
    warning_count = sum(
        1 for issue in result.issues if issue.level == ValidationLevel.WARNING
    )

    result.system_info.update(
        {
            "error_count": error_count,
            "warning_count": warning_count,
            "available_providers": available_providers,
            "configured_providers": configured_providers,
            "available_features_count": len(result.available_features),
        }
    )

    return result


def print_validation_result(result: ValidationResult):
    """Print validation results in a user-friendly format."""
    print("\n🔍 CrewAI + GenOps Setup Validation")
    print("=" * 50)

    # Overall status
    if result.is_valid:
        print("✅ Setup Status: VALID - Ready to use!")
    else:
        print("❌ Setup Status: ISSUES FOUND - See details below")

    # System info
    print("\n📋 System Information:")
    print(f"   Platform: {result.system_info.get('platform', 'unknown')}")
    print(f"   Python: {result.system_info.get('python_version', 'unknown')}")
    if "crewai_version" in result.system_info:
        print(f"   CrewAI: {result.system_info['crewai_version']}")

    # Available features
    if result.available_features:
        print(f"\n✨ Available Features ({len(result.available_features)}):")
        for feature in result.available_features:
            print(f"   • {feature}")

    # Issues by category
    categories = {}
    for issue in result.issues:
        if issue.category not in categories:
            categories[issue.category] = []
        categories[issue.category].append(issue)

    # Group and display issues
    level_symbols = {
        ValidationLevel.ERROR: "❌",
        ValidationLevel.WARNING: "⚠️",
        ValidationLevel.INFO: "ℹ️",
        ValidationLevel.SUCCESS: "✅",
    }

    for category, issues in categories.items():
        print(f"\n📂 {category.replace('_', ' ').title()}:")

        for issue in issues:
            symbol = level_symbols.get(issue.level, "•")
            print(f"   {symbol} {issue.message}")

            if issue.details:
                print(f"      Details: {issue.details}")

            if issue.fix_suggestion:
                print(f"      🔧 Fix: {issue.fix_suggestion}")

            if issue.documentation_link:
                print(f"      📚 Docs: {issue.documentation_link}")

    # Summary
    error_count = result.system_info.get("error_count", 0)
    warning_count = result.system_info.get("warning_count", 0)

    print("\n📊 Summary:")
    print(f"   Errors: {error_count}")
    print(f"   Warnings: {warning_count}")
    print(f"   Features Available: {len(result.available_features)}")

    # Next steps
    if result.is_valid:
        print("\n🚀 Next Steps:")
        print("   • Try: from genops.providers.crewai import auto_instrument")
        print("   • Run: auto_instrument() before using CrewAI")
        print("   • Check examples/crewai/ for usage patterns")
    else:
        print("\n🔧 Required Actions:")
        print("   • Fix the errors listed above")
        print("   • Re-run validation: validate_crewai_setup()")
        print("   • Check the documentation links for detailed help")


def quick_validate() -> bool:
    """
    Quick validation check returning simple boolean result.

    Returns:
        bool: True if setup is valid, False otherwise
    """
    result = validate_crewai_setup(quick=True)
    return result.is_valid


# Export main functions
__all__ = [
    "validate_crewai_setup",
    "print_validation_result",
    "quick_validate",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
]
