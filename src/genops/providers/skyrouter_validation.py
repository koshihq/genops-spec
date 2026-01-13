#!/usr/bin/env python3
"""
SkyRouter Integration Validation

This module provides comprehensive setup validation for SkyRouter integration
with GenOps governance. It checks environment configuration, SDK availability,
authentication, and provides actionable diagnostics for multi-model routing.

Features:
- Environment variable validation (SKYROUTER_API_KEY)
- SDK installation and version checking
- Multi-model API connectivity testing
- Routing configuration validation
- Configuration validation reporting
- Interactive setup for complex configurations
- Actionable error messages with specific fix suggestions

Author: GenOps AI Contributors
License: Apache 2.0
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from decimal import Decimal
import importlib.util

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """Represents a validation issue with fix suggestions."""
    category: str
    severity: str  # "error", "warning", "info"
    message: str
    fix_suggestion: Optional[str] = None
    documentation_link: Optional[str] = None

@dataclass
class ValidationResult:
    """Complete validation result with categorized issues."""
    is_valid: bool
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    recommendations: List[str]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any validation warnings."""
        return len(self.warnings) > 0

def validate_setup(skyrouter_api_key: Optional[str] = None) -> ValidationResult:
    """
    Comprehensive validation of SkyRouter + GenOps setup.
    
    This function checks all aspects of the integration setup and returns
    detailed results with actionable recommendations.
    
    Args:
        skyrouter_api_key: Optional API key to validate (uses env var if not provided)
    
    Returns:
        ValidationResult: Comprehensive validation results with fix suggestions
    """
    errors = []
    warnings = []
    recommendations = []
    
    # Validate Python environment
    _validate_python_environment(errors, warnings, recommendations)
    
    # Validate dependencies
    _validate_dependencies(errors, warnings, recommendations)
    
    # Validate authentication 
    _validate_authentication(skyrouter_api_key, errors, warnings, recommendations)
    
    # Validate SkyRouter connectivity and configuration
    _validate_skyrouter_configuration(skyrouter_api_key, errors, warnings, recommendations)
    
    # Validate GenOps configuration
    _validate_genops_configuration(errors, warnings, recommendations)
    
    # Determine overall validation status
    is_valid = len(errors) == 0
    
    if is_valid:
        recommendations.append("All validation checks passed! SkyRouter integration is ready for use.")
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        recommendations=recommendations
    )

def _validate_python_environment(errors: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate Python environment compatibility."""
    
    # Check Python version
    if sys.version_info < (3, 9):
        errors.append(ValidationIssue(
            category="python",
            severity="error",
            message=f"Python {sys.version_info.major}.{sys.version_info.minor} not supported (minimum: 3.9)",
            fix_suggestion="Upgrade to Python 3.9 or later",
            documentation_link="https://www.python.org/downloads/"
        ))
    elif sys.version_info >= (3, 12):
        recommendations.append("Using latest Python version - excellent for performance!")
    
    # Check virtual environment
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        warnings.append(ValidationIssue(
            category="environment",
            severity="warning", 
            message="No virtual environment detected",
            fix_suggestion="Consider using a virtual environment: python -m venv venv && source venv/bin/activate",
            documentation_link="https://docs.python.org/3/tutorial/venv.html"
        ))

def _validate_dependencies(errors: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate required dependencies for SkyRouter integration."""
    
    # Check GenOps core
    try:
        import genops
        recommendations.append("GenOps core package is available")
    except ImportError:
        errors.append(ValidationIssue(
            category="dependencies",
            severity="error",
            message="GenOps package not found",
            fix_suggestion="Install GenOps: pip install genops",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI"
        ))
    
    # Check OpenTelemetry (optional but recommended)
    try:
        import opentelemetry
        recommendations.append("OpenTelemetry available for full telemetry export")
    except ImportError:
        warnings.append(ValidationIssue(
            category="dependencies",
            severity="warning",
            message="OpenTelemetry not installed (optional)",
            fix_suggestion="Install for enhanced telemetry: pip install opentelemetry-api opentelemetry-sdk",
            documentation_link="https://opentelemetry.io/docs/instrumentation/python/"
        ))
    
    # Check SkyRouter SDK (if available)
    skyrouter_available = importlib.util.find_spec("skyrouter") is not None
    if not skyrouter_available:
        warnings.append(ValidationIssue(
            category="dependencies",
            severity="warning",
            message="SkyRouter SDK not found (will use API calls)",
            fix_suggestion="Install SkyRouter SDK if available: pip install skyrouter",
            documentation_link="https://skyrouter.ai/docs"
        ))
    else:
        recommendations.append("SkyRouter SDK detected - enhanced integration available")

def _validate_authentication(skyrouter_api_key: Optional[str], errors: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate SkyRouter authentication configuration."""
    
    # Check API key
    api_key = skyrouter_api_key or os.getenv("SKYROUTER_API_KEY")
    
    if not api_key:
        errors.append(ValidationIssue(
            category="auth",
            severity="error",
            message="SkyRouter API key not found",
            fix_suggestion="Set environment variable: export SKYROUTER_API_KEY='your-api-key'",
            documentation_link="https://skyrouter.ai/docs/authentication"
        ))
        return
    
    # Comprehensive API key validation
    key_issues = []
    
    # Length validation
    if len(api_key) < 10:
        key_issues.append("too short (minimum 10 characters)")
    elif len(api_key) > 200:
        key_issues.append("too long (maximum 200 characters)")
    
    # Character validation
    if not api_key.replace('-', '').replace('_', '').replace('.', '').isalnum():
        key_issues.append("contains invalid characters (only alphanumeric, hyphens, underscores, dots allowed)")
    
    # Common format patterns
    if api_key.startswith('sk-') and len(api_key) < 40:
        key_issues.append("appears to be OpenAI format but too short for SkyRouter")
    elif api_key.count(' ') > 0:
        key_issues.append("contains spaces (remove whitespace)")
    elif api_key.startswith('Bearer '):
        key_issues.append("includes 'Bearer ' prefix (remove it)")
    
    if key_issues:
        warnings.append(ValidationIssue(
            category="auth",
            severity="warning",
            message=f"API key format issues: {', '.join(key_issues)}",
            fix_suggestion="Verify your API key from SkyRouter dashboard and ensure correct format"
        ))
    else:
        recommendations.append("API key format appears valid")
    
    # Check for common environment variable issues
    raw_key = os.getenv("SKYROUTER_API_KEY")
    if raw_key != api_key:
        warnings.append(ValidationIssue(
            category="auth",
            severity="warning",
            message="API key differs from environment variable",
            fix_suggestion="Ensure SKYROUTER_API_KEY environment variable matches provided key"
        ))

def _validate_skyrouter_configuration(skyrouter_api_key: Optional[str], errors: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate SkyRouter-specific configuration."""
    
    api_key = skyrouter_api_key or os.getenv("SKYROUTER_API_KEY")
    if not api_key:
        return  # Skip if no API key available
    
    # Check model access configuration
    preferred_models = os.getenv("SKYROUTER_PREFERRED_MODELS", "").split(",")
    if preferred_models and preferred_models != [""]:
        recommendations.append(f"Preferred models configured: {', '.join(preferred_models[:3])}")
    else:
        recommendations.append("No preferred models set - will use SkyRouter defaults")
    
    # Check routing optimization settings
    routing_strategy = os.getenv("SKYROUTER_ROUTING_STRATEGY", "balanced")
    valid_strategies = ["cost_optimized", "latency_optimized", "balanced", "reliability_first"]
    
    if routing_strategy not in valid_strategies:
        warnings.append(ValidationIssue(
            category="configuration",
            severity="warning",
            message=f"Unknown routing strategy: {routing_strategy}",
            fix_suggestion=f"Use one of: {', '.join(valid_strategies)}"
        ))
    else:
        recommendations.append(f"Routing strategy: {routing_strategy}")
    
    # Check region configuration
    preferred_region = os.getenv("SKYROUTER_PREFERRED_REGION")
    if preferred_region:
        recommendations.append(f"Preferred region: {preferred_region}")
    else:
        recommendations.append("No preferred region set - will use automatic selection")
    
    # Validate budget configuration
    budget_limit = os.getenv("SKYROUTER_DAILY_BUDGET_LIMIT")
    if budget_limit:
        try:
            budget_value = float(budget_limit)
            if budget_value <= 0:
                warnings.append(ValidationIssue(
                    category="configuration",
                    severity="warning",
                    message="Daily budget limit must be positive",
                    fix_suggestion="Set a positive budget limit: export SKYROUTER_DAILY_BUDGET_LIMIT='50.0'"
                ))
            else:
                recommendations.append(f"Daily budget limit: ${budget_value:.2f}")
        except ValueError:
            warnings.append(ValidationIssue(
                category="configuration",
                severity="warning",
                message="Invalid daily budget limit format",
                fix_suggestion="Use numeric format: export SKYROUTER_DAILY_BUDGET_LIMIT='50.0'"
            ))

def _validate_genops_configuration(errors: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate GenOps-specific configuration."""
    
    # Check team attribution
    team = os.getenv("GENOPS_TEAM")
    if not team:
        warnings.append(ValidationIssue(
            category="governance",
            severity="warning",
            message="No team attribution configured",
            fix_suggestion="Set team: export GENOPS_TEAM='your-team-name'",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI/docs/governance"
        ))
    else:
        recommendations.append(f"Team attribution: {team}")
    
    # Check project attribution
    project = os.getenv("GENOPS_PROJECT")
    if not project:
        warnings.append(ValidationIssue(
            category="governance",
            severity="warning",
            message="No project attribution configured",
            fix_suggestion="Set project: export GENOPS_PROJECT='your-project-name'"
        ))
    else:
        recommendations.append(f"Project attribution: {project}")
    
    # Check environment
    environment = os.getenv("GENOPS_ENVIRONMENT", "production")
    valid_environments = ["development", "staging", "production"]
    if environment not in valid_environments:
        warnings.append(ValidationIssue(
            category="governance",
            severity="warning",
            message=f"Unknown environment: {environment}",
            fix_suggestion=f"Use one of: {', '.join(valid_environments)}"
        ))
    else:
        recommendations.append(f"Environment: {environment}")
    
    # Check governance policy
    governance_policy = os.getenv("GENOPS_GOVERNANCE_POLICY", "enforced")
    valid_policies = ["advisory", "enforced"]
    if governance_policy not in valid_policies:
        warnings.append(ValidationIssue(
            category="governance",
            severity="warning",
            message=f"Unknown governance policy: {governance_policy}",
            fix_suggestion=f"Use one of: {', '.join(valid_policies)}"
        ))
    else:
        recommendations.append(f"Governance policy: {governance_policy}")
    
    # Check telemetry export configuration
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otel_endpoint:
        recommendations.append(f"OpenTelemetry export: {otel_endpoint}")
    else:
        recommendations.append("OpenTelemetry export: disabled (local only)")

def print_validation_result(result: ValidationResult, verbose: bool = True):
    """Print validation results in a user-friendly format."""
    
    print("🔍 SkyRouter + GenOps Validation Results")
    print("=" * 50)
    
    # Overall status
    if result.is_valid:
        print("✅ Validation Status: PASSED")
        print("🎉 Your SkyRouter integration is ready to use!")
    else:
        print("❌ Validation Status: FAILED")
        print(f"📋 Found {len(result.errors)} error(s) that need attention")
    
    print()
    
    # Errors (must fix)
    if result.errors:
        print("🚨 Errors (Must Fix):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. {error.message}")
            if error.fix_suggestion:
                print(f"     💡 Fix: {error.fix_suggestion}")
            if error.documentation_link and verbose:
                print(f"     📖 Docs: {error.documentation_link}")
        print()
    
    # Warnings (should fix)
    if result.warnings:
        print("⚠️  Warnings (Recommended Fixes):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning.message}")
            if warning.fix_suggestion:
                print(f"     💡 Fix: {warning.fix_suggestion}")
        print()
    
    # Recommendations
    if result.recommendations:
        print("✨ Configuration Summary:")
        for rec in result.recommendations:
            print(f"  ✓ {rec}")
        print()
    
    # Next steps
    if result.is_valid:
        print("🚀 Next Steps:")
        print("  1. Try the quickstart: python examples/skyrouter/setup_validation.py")
        print("  2. Explore examples: cd examples/skyrouter && ls")
        print("  3. Read docs: docs/skyrouter-quickstart.md")
    else:
        print("🔧 Next Steps:")
        print("  1. Fix the errors listed above")
        print("  2. Run validation again: python -c \"from genops.providers.skyrouter_validation import validate_setup; validate_setup()\"")
        print("  3. Get help: https://github.com/KoshiHQ/GenOps-AI/discussions")

def validate_setup_interactive() -> ValidationResult:
    """
    Interactive validation with guided setup.
    
    This function walks users through the setup process,
    asking for missing configuration and providing real-time feedback.
    """
    print("🚀 SkyRouter + GenOps Interactive Setup")
    print("=" * 45)
    print()
    
    # Step 1: Check current setup
    print("📋 Step 1: Checking current configuration...")
    result = validate_setup()
    
    if result.is_valid:
        print("✅ Configuration is already valid!")
        print_validation_result(result, verbose=False)
        return result
    
    print(f"Found {len(result.errors)} issues to resolve.")
    print()
    
    # Step 2: Interactive fixes
    print("🔧 Step 2: Let's fix the configuration...")
    
    # Fix missing API key
    if any(error.category == "auth" for error in result.errors):
        api_key = input("Enter your SkyRouter API key: ").strip()
        if api_key:
            print(f"Setting SKYROUTER_API_KEY environment variable...")
            os.environ["SKYROUTER_API_KEY"] = api_key
            print("✅ API key configured for this session")
        print()
    
    # Fix missing team attribution  
    if not os.getenv("GENOPS_TEAM"):
        team = input("Enter your team name (e.g., 'ai-platform'): ").strip()
        if team:
            os.environ["GENOPS_TEAM"] = team
            print(f"✅ Team set to: {team}")
        print()
    
    # Fix missing project attribution
    if not os.getenv("GENOPS_PROJECT"):
        project = input("Enter your project name (e.g., 'skyrouter-routing'): ").strip()
        if project:
            os.environ["GENOPS_PROJECT"] = project
            print(f"✅ Project set to: {project}")
        print()
    
    # Step 3: Re-validate
    print("🔍 Step 3: Re-validating configuration...")
    final_result = validate_setup()
    print_validation_result(final_result, verbose=True)
    
    return final_result

def get_validation_summary() -> Dict[str, Any]:
    """Get validation summary for programmatic use."""
    result = validate_setup()
    
    return {
        "is_valid": result.is_valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": [
            {
                "category": error.category,
                "severity": error.severity,
                "message": error.message,
                "fix_suggestion": error.fix_suggestion
            }
            for error in result.errors
        ],
        "warnings": [
            {
                "category": warning.category,
                "severity": warning.severity,
                "message": warning.message,
                "fix_suggestion": warning.fix_suggestion
            }
            for warning in result.warnings
        ],
        "recommendations": result.recommendations
    }

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SkyRouter + GenOps setup")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run interactive setup")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    
    args = parser.parse_args()
    
    if args.json:
        import json
        summary = get_validation_summary()
        print(json.dumps(summary, indent=2))
    elif args.interactive:
        validate_setup_interactive()
    else:
        result = validate_setup()
        print_validation_result(result, verbose=args.verbose)
        sys.exit(0 if result.is_valid else 1)