#!/usr/bin/env python3
"""
Interactive Setup Validation Script for GenOps AI + Haystack

Provides comprehensive environment validation with interactive troubleshooting
and step-by-step setup guidance for new developers.

Usage:
    python scripts/validate_setup.py
    python scripts/validate_setup.py --provider openai
    python scripts/validate_setup.py --fix-issues
    python scripts/validate_setup.py --detailed
"""

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from genops.providers.haystack.validation import ValidationIssue, ValidationResult

    from genops.providers.haystack import (
        print_validation_result,
        validate_haystack_setup,
    )
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False

@dataclass
class InteractiveValidationResult:
    """Enhanced validation result with interactive guidance."""
    validation_result: Optional['ValidationResult'] = None
    environment_info: Dict[str, str] = field(default_factory=dict)
    missing_dependencies: List[str] = field(default_factory=list)
    configuration_issues: List[str] = field(default_factory=list)
    suggested_fixes: List[Dict[str, str]] = field(default_factory=list)
    interactive_prompts: List[str] = field(default_factory=list)


class InteractiveValidator:
    """Interactive setup validation with guided troubleshooting."""

    def __init__(self, provider_focus: Optional[str] = None, fix_mode: bool = False, detailed: bool = False):
        self.provider_focus = provider_focus
        self.fix_mode = fix_mode
        self.detailed = detailed
        self.issues_found = []
        self.fixes_applied = []

    def run_validation(self) -> InteractiveValidationResult:
        """Run comprehensive interactive validation."""
        print("🔍 GenOps AI + Haystack Interactive Setup Validation")
        print("=" * 60)

        result = InteractiveValidationResult()

        # Basic environment detection
        self._collect_environment_info(result)

        # Dependency validation
        self._validate_dependencies(result)

        # GenOps validation if available
        if GENOPS_AVAILABLE:
            result.validation_result = validate_haystack_setup()

        # Provider-specific validation
        if self.provider_focus:
            self._validate_specific_provider(result, self.provider_focus)

        # Generate interactive guidance
        self._generate_interactive_guidance(result)

        # Apply fixes if requested
        if self.fix_mode:
            self._apply_automated_fixes(result)

        return result

    def _collect_environment_info(self, result: InteractiveValidationResult):
        """Collect comprehensive environment information."""
        print("\n📊 Environment Information")
        print("-" * 30)

        # Python information
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_path = sys.executable

        # Platform information
        system_info = {
            "python_version": python_version,
            "python_executable": python_path,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "working_directory": str(Path.cwd())
        }

        result.environment_info = system_info

        print(f"🐍 Python: {python_version} ({platform.system().lower()})")
        print(f"📂 Working Directory: {Path.cwd()}")

        if self.detailed:
            print(f"🔧 Python Executable: {python_path}")
            print(f"🏗️ Architecture: {platform.machine()}")

    def _validate_dependencies(self, result: InteractiveValidationResult):
        """Validate required and optional dependencies."""
        print("\n🔍 Dependency Validation")
        print("-" * 30)

        required_packages = [
            ("genops", "genops-ai"),
            ("haystack", "haystack-ai"),
        ]

        optional_packages = [
            ("openai", "openai"),
            ("anthropic", "anthropic"),
            ("cohere", "cohere-ai"),
            ("transformers", "transformers"),
        ]

        provider_specific = {
            "openai": [("openai", "openai")],
            "anthropic": [("anthropic", "anthropic")],
            "cohere": [("cohere", "cohere-ai")],
            "huggingface": [("transformers", "transformers")]
        }

        # Check required packages
        print("📦 Required Packages:")
        for module, package in required_packages:
            try:
                importlib.import_module(module)
                if module == "genops":
                    try:
                        from genops import __version__
                        version_info = f" v{__version__}"
                    except:
                        version_info = " (unknown version)"
                elif module == "haystack":
                    try:
                        import haystack
                        version_info = f" v{haystack.__version__}"
                    except:
                        version_info = " (unknown version)"
                else:
                    version_info = ""

                print(f"   ✅ {module}{version_info}")
            except ImportError:
                print(f"   ❌ {module} (install: pip install {package})")
                result.missing_dependencies.append(package)
                result.suggested_fixes.append({
                    "issue": f"Missing required package: {module}",
                    "fix": f"pip install {package}",
                    "category": "required_dependency"
                })

        # Check optional packages
        print("\n🔧 AI Provider Packages:")
        available_providers = []

        for module, package in optional_packages:
            try:
                importlib.import_module(module)
                print(f"   ✅ {module} integration available")
                available_providers.append(module)
            except ImportError:
                print(f"   ⚠️  {module} not installed (optional: pip install {package})")
                if module == "openai":  # OpenAI is commonly used
                    result.suggested_fixes.append({
                        "issue": "OpenAI provider not available",
                        "fix": f"pip install {package}",
                        "category": "recommended_provider"
                    })

        # Provider-specific checks
        if self.provider_focus and self.provider_focus in provider_specific:
            print(f"\n🎯 Provider-Specific Check ({self.provider_focus}):")
            for module, package in provider_specific[self.provider_focus]:
                try:
                    importlib.import_module(module)
                    print(f"   ✅ {module} available for {self.provider_focus}")
                except ImportError:
                    print(f"   ❌ {module} required for {self.provider_focus} (pip install {package})")
                    result.missing_dependencies.append(package)

        if not available_providers:
            result.configuration_issues.append("No AI provider packages detected")
            result.suggested_fixes.append({
                "issue": "No AI providers available",
                "fix": "pip install openai anthropic  # Install preferred providers",
                "category": "provider_setup"
            })

        result.environment_info["available_providers"] = available_providers

    def _validate_specific_provider(self, result: InteractiveValidationResult, provider: str):
        """Validate specific AI provider configuration."""
        print(f"\n🔧 {provider.title()} Provider Configuration")
        print("-" * 30)

        provider_configs = {
            "openai": {
                "env_vars": ["OPENAI_API_KEY"],
                "test_import": "openai",
                "install_command": "pip install openai"
            },
            "anthropic": {
                "env_vars": ["ANTHROPIC_API_KEY"],
                "test_import": "anthropic",
                "install_command": "pip install anthropic"
            },
            "cohere": {
                "env_vars": ["COHERE_API_KEY"],
                "test_import": "cohere",
                "install_command": "pip install cohere-ai"
            }
        }

        if provider not in provider_configs:
            print(f"   ❌ Unknown provider: {provider}")
            return

        config = provider_configs[provider]

        # Check package installation
        try:
            importlib.import_module(config["test_import"])
            print(f"   ✅ {provider} package installed")
        except ImportError:
            print(f"   ❌ {provider} package not installed")
            result.suggested_fixes.append({
                "issue": f"{provider} package not installed",
                "fix": config["install_command"],
                "category": "provider_dependency"
            })
            return

        # Check environment variables
        for env_var in config["env_vars"]:
            if os.getenv(env_var):
                # Mask the key for security
                masked_value = f"{os.getenv(env_var)[:8]}..." if len(os.getenv(env_var)) > 8 else "***"
                print(f"   ✅ {env_var}: {masked_value}")
            else:
                print(f"   ❌ {env_var} not set")
                result.configuration_issues.append(f"Missing environment variable: {env_var}")
                result.suggested_fixes.append({
                    "issue": f"Missing API key: {env_var}",
                    "fix": f"export {env_var}=\"your-api-key-here\"",
                    "category": "api_key_setup"
                })

    def _generate_interactive_guidance(self, result: InteractiveValidationResult):
        """Generate interactive troubleshooting guidance."""
        if not result.suggested_fixes:
            return

        print("\n🚀 Interactive Setup Guidance")
        print("-" * 30)

        # Group fixes by category
        fix_categories = {}
        for fix in result.suggested_fixes:
            category = fix.get("category", "general")
            if category not in fix_categories:
                fix_categories[category] = []
            fix_categories[category].append(fix)

        # Present fixes by priority
        category_priority = [
            "required_dependency",
            "provider_dependency",
            "api_key_setup",
            "recommended_provider",
            "provider_setup"
        ]

        for category in category_priority:
            if category not in fix_categories:
                continue

            fixes = fix_categories[category]

            if category == "required_dependency":
                print("\n🔴 Critical Issues (must fix to continue):")
            elif category == "provider_dependency":
                print("\n🟡 Provider Setup Issues:")
            elif category == "api_key_setup":
                print("\n🔑 API Key Configuration:")
            else:
                print("\n🟢 Optional Improvements:")

            for fix in fixes:
                print(f"   Issue: {fix['issue']}")
                print(f"   Fix:   {fix['fix']}")
                print()

                if self.fix_mode and category in ["required_dependency", "provider_dependency"]:
                    result.interactive_prompts.append(f"Apply fix: {fix['fix']}")

    def _apply_automated_fixes(self, result: InteractiveValidationResult):
        """Apply automated fixes where safe to do so."""
        if not self.fix_mode or not result.suggested_fixes:
            return

        print("\n🔧 Automated Fix Application")
        print("-" * 30)

        for fix in result.suggested_fixes:
            if fix.get("category") in ["required_dependency", "provider_dependency"]:
                fix_command = fix["fix"]

                if fix_command.startswith("pip install"):
                    print(f"Applying: {fix_command}")

                    if self._confirm_action(f"Install {fix['issue'].split(':')[-1].strip()}?"):
                        try:
                            subprocess.run(fix_command.split(), check=True, capture_output=True)
                            print("   ✅ Successfully applied fix")
                            self.fixes_applied.append(fix["issue"])
                        except subprocess.CalledProcessError as e:
                            print(f"   ❌ Fix failed: {e}")
                    else:
                        print("   ⏸️ Skipped by user")

    def _confirm_action(self, prompt: str) -> bool:
        """Confirm user action in interactive mode."""
        try:
            response = input(f"{prompt} (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user")
            return False

    def display_results(self, result: InteractiveValidationResult):
        """Display comprehensive validation results."""
        print("\n" + "=" * 60)
        print("📋 Validation Summary")
        print("=" * 60)

        # Overall status
        if result.validation_result and GENOPS_AVAILABLE:
            print_validation_result(result.validation_result)

        # Environment summary
        print("\n🌍 Environment Summary:")
        print(f"   Python: {result.environment_info.get('python_version', 'Unknown')}")
        print(f"   Platform: {result.environment_info.get('platform', 'Unknown')}")

        if result.environment_info.get('available_providers'):
            providers = ", ".join(result.environment_info['available_providers'])
            print(f"   Available Providers: {providers}")

        # Issue summary
        total_issues = len(result.missing_dependencies) + len(result.configuration_issues)

        if total_issues == 0:
            print("\n🎉 Setup Validation Complete!")
            print("   Your environment is ready for GenOps + Haystack development")
            print("\n📚 Next Steps:")
            print("   • Try the quickstart: python examples/haystack/basic_pipeline_tracking.py")
            print("   • Read the docs: docs/integrations/haystack.md")
            print("   • Join our community: https://github.com/genops-ai/genops-ai")
        else:
            print(f"\n⚠️  Found {total_issues} setup issues")

            if self.fixes_applied:
                print(f"✅ Applied {len(self.fixes_applied)} automated fixes")

            if result.suggested_fixes:
                remaining_fixes = [f for f in result.suggested_fixes if f["issue"] not in self.fixes_applied]
                if remaining_fixes:
                    print(f"📝 {len(remaining_fixes)} fixes still needed (see guidance above)")

        # Provide quick commands
        print("\n💡 Quick Commands:")
        print("   Validate again:     python scripts/validate_setup.py")
        print("   Fix dependencies:   python scripts/validate_setup.py --fix-issues")
        print("   Provider-specific:  python scripts/validate_setup.py --provider openai")
        print("   Detailed info:      python scripts/validate_setup.py --detailed")


def main():
    """Main interactive validation entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Setup Validation for GenOps AI + Haystack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_setup.py                    # Basic validation
  python scripts/validate_setup.py --provider openai  # OpenAI-specific checks
  python scripts/validate_setup.py --fix-issues       # Auto-fix dependencies
  python scripts/validate_setup.py --detailed         # Verbose output
        """
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "cohere", "huggingface"],
        help="Focus validation on specific AI provider"
    )

    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Automatically apply safe fixes (e.g., install packages)"
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed environment information"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format for CI/automation"
    )

    args = parser.parse_args()

    try:
        # Run interactive validation
        validator = InteractiveValidator(
            provider_focus=args.provider,
            fix_mode=args.fix_issues,
            detailed=args.detailed
        )

        result = validator.run_validation()

        if args.json:
            # JSON output for automation
            json_result = {
                "validation_passed": result.validation_result.is_valid if result.validation_result else False,
                "environment": result.environment_info,
                "missing_dependencies": result.missing_dependencies,
                "configuration_issues": result.configuration_issues,
                "fixes_applied": validator.fixes_applied,
                "total_issues": len(result.missing_dependencies) + len(result.configuration_issues)
            }
            print(json.dumps(json_result, indent=2))
        else:
            # Interactive display
            validator.display_results(result)

        # Return appropriate exit code
        total_issues = len(result.missing_dependencies) + len(result.configuration_issues)
        if result.validation_result and not result.validation_result.is_valid:
            total_issues += len(result.validation_result.issues)

        return 0 if total_issues == 0 else 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Validation cancelled by user")
        return 130
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        if args.detailed:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
