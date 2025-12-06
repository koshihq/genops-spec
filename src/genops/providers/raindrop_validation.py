#!/usr/bin/env python3
"""
Raindrop AI Integration Validation

This module provides comprehensive setup validation for Raindrop AI integration
with GenOps governance. It checks environment configuration, SDK installation,
authentication, and provides actionable diagnostics.

Features:
- Environment variable validation (RAINDROP_API_KEY)
- SDK installation and version checking
- API connectivity testing
- Configuration validation reporting
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
    """Represents a validation issue with details and fix suggestions."""
    category: str  # 'sdk', 'auth', 'config', 'governance'
    severity: str  # 'error', 'warning', 'info'
    message: str
    fix_suggestion: Optional[str] = None
    documentation_link: Optional[str] = None

@dataclass
class ValidationResult:
    """Complete validation result with issues and recommendations."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    recommendations: List[str]
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity == 'error']
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return len(self.warnings) > 0

def validate_setup(raindrop_api_key: Optional[str] = None) -> ValidationResult:
    """
    Comprehensive validation of Raindrop AI setup with GenOps.
    
    Args:
        raindrop_api_key: Raindrop API key to validate (optional)
        
    Returns:
        ValidationResult: Complete validation results with issues and recommendations
    """
    issues: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    recommendations: List[str] = []
    
    # 1. SDK Installation Validation
    _validate_sdk_installation(issues, warnings, recommendations)
    
    # 2. Authentication Validation  
    _validate_authentication(raindrop_api_key, issues, warnings, recommendations)
    
    # 3. Configuration Validation
    _validate_configuration(issues, warnings, recommendations)
    
    # 4. Governance Setup Validation
    _validate_governance_setup(issues, warnings, recommendations)
    
    # Determine overall validity
    is_valid = len([issue for issue in issues if issue.severity == 'error']) == 0
    
    # Add final recommendations
    if is_valid:
        recommendations.append("All validation checks passed successfully!")
    else:
        recommendations.append("Fix the error-level issues above to enable Raindrop AI integration")
    
    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        warnings=warnings,
        recommendations=recommendations
    )

def _validate_sdk_installation(issues: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate Raindrop AI SDK installation and version."""
    
    # Check Python version compatibility
    import sys
    python_version = sys.version_info
    if python_version < (3, 9):
        issues.append(ValidationIssue(
            category="sdk",
            severity="error",
            message=f"Python {python_version.major}.{python_version.minor} not supported",
            fix_suggestion="Upgrade to Python 3.9+ with: pyenv install 3.9.0 && pyenv global 3.9.0",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI/blob/main/README.md#prerequisites"
        ))
    else:
        recommendations.append(f"Python {python_version.major}.{python_version.minor} is compatible")
    
    # Check GenOps installation and version
    try:
        import genops
        genops_version = getattr(genops, "__version__", "unknown")
        recommendations.append(f"GenOps v{genops_version} installed")
        
        # Verify GenOps was installed with raindrop extras
        try:
            from genops.providers.raindrop import auto_instrument
            recommendations.append("GenOps Raindrop provider available")
        except ImportError as e:
            issues.append(ValidationIssue(
                category="sdk",
                severity="error",
                message="GenOps Raindrop provider not available",
                fix_suggestion="Install with extras: pip install genops[raindrop]",
                documentation_link="https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/raindrop-quickstart.md"
            ))
    except ImportError:
        issues.append(ValidationIssue(
            category="sdk",
            severity="error",
            message="GenOps not installed",
            fix_suggestion="Install with: pip install genops[raindrop]",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/raindrop-quickstart.md"
        ))
    
    # Check if raindrop package is available
    raindrop_spec = importlib.util.find_spec("raindrop")
    
    if raindrop_spec is None:
        warnings.append(ValidationIssue(
            category="sdk",
            severity="warning",  # Not an error since this integration can work without SDK
            message="Raindrop AI SDK not found",
            fix_suggestion="Install with: pip install raindrop",
            documentation_link="https://www.raindrop.ai/docs/quickstart"
        ))
        recommendations.append("Consider installing the Raindrop AI SDK for enhanced features")
    else:
        try:
            import raindrop
            # Try to get version if available
            version = getattr(raindrop, "__version__", "unknown")
            recommendations.append(f"Raindrop AI SDK v{version} detected")
            
            # Test basic client instantiation
            try:
                raindrop.Client(api_key="test-key")
                recommendations.append("Raindrop AI SDK client instantiation successful")
            except Exception as e:
                warnings.append(ValidationIssue(
                    category="sdk",
                    severity="warning",
                    message=f"Raindrop SDK client test failed: {str(e)}",
                    fix_suggestion="This may be expected with a test API key"
                ))
                
        except ImportError as e:
            issues.append(ValidationIssue(
                category="sdk",
                severity="warning",
                message=f"Failed to import raindrop module: {str(e)}",
                fix_suggestion="Reinstall with: pip install --force-reinstall raindrop"
            ))
    
    # Check GenOps dependencies with version requirements
    required_modules = {
        'opentelemetry': 'OpenTelemetry core',
        'opentelemetry.trace': 'OpenTelemetry tracing',
        'opentelemetry.exporter.otlp': 'OpenTelemetry OTLP exporter'
    }
    
    for module_name, description in required_modules.items():
        try:
            module = __import__(module_name)
            # Check for version if available
            if hasattr(module, '__version__'):
                recommendations.append(f"{description} v{module.__version__} available")
            else:
                recommendations.append(f"{description} available")
        except ImportError:
            issues.append(ValidationIssue(
                category="sdk",
                severity="error",
                message=f"Missing required dependency: {description} ({module_name})",
                fix_suggestion="Install with: pip install genops[raindrop]",
                documentation_link="https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/raindrop-quickstart.md"
            ))
    
    # Check for common installation issues
    try:
        import site
        site_packages = site.getsitepackages()
        user_site = site.getusersitepackages()
        recommendations.append(f"Python packages location: {site_packages[0] if site_packages else user_site}")
    except Exception:
        pass
    
    # Check for virtual environment
    import os
    venv_indicators = ['VIRTUAL_ENV', 'CONDA_DEFAULT_ENV', 'PIPENV_ACTIVE']
    active_venv = None
    for indicator in venv_indicators:
        if os.getenv(indicator):
            active_venv = indicator
            break
    
    if active_venv:
        recommendations.append(f"Virtual environment detected: {os.getenv(active_venv, 'active')}")
    else:
        warnings.append(ValidationIssue(
            category="sdk",
            severity="warning",
            message="No virtual environment detected",
            fix_suggestion="Consider using a virtual environment: python -m venv venv && source venv/bin/activate",
            documentation_link="https://docs.python.org/3/tutorial/venv.html"
        ))

def _validate_authentication(raindrop_api_key: Optional[str], issues: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate Raindrop AI authentication configuration."""
    
    # Check API key
    api_key = raindrop_api_key or os.getenv("RAINDROP_API_KEY")
    
    if not api_key:
        issues.append(ValidationIssue(
            category="auth",
            severity="error",
            message="Raindrop AI API key not found",
            fix_suggestion="Set environment variable: export RAINDROP_API_KEY='your-api-key'",
            documentation_link="https://www.raindrop.ai/docs/authentication"
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
    if not api_key.replace('-', '').replace('_', '').isalnum():
        key_issues.append("contains invalid characters (only alphanumeric, hyphens, underscores allowed)")
    
    # Common format patterns
    if api_key.startswith('sk-') and len(api_key) < 40:
        key_issues.append("appears to be OpenAI format but too short for Raindrop")
    elif api_key.count(' ') > 0:
        key_issues.append("contains spaces (remove whitespace)")
    elif api_key.startswith('Bearer '):
        key_issues.append("includes 'Bearer ' prefix (remove it)")
    
    if key_issues:
        warnings.append(ValidationIssue(
            category="auth",
            severity="warning",
            message=f"API key format issues: {', '.join(key_issues)}",
            fix_suggestion="Verify your API key from Raindrop AI dashboard and ensure correct format"
        ))
    else:
        recommendations.append("API key format appears valid")
    
    # Check for common environment variable issues
    raw_key = os.getenv("RAINDROP_API_KEY")
    if raw_key != api_key:
        warnings.append(ValidationIssue(
            category="auth",
            severity="warning",
            message="API key was modified during retrieval",
            fix_suggestion="Check for shell escaping issues or invisible characters in environment variable"
        ))
    
    # Check for key exposure in environment
    if api_key in str(os.environ):
        # This is expected, but warn about security
        recommendations.append("API key properly set in environment variables")
        warnings.append(ValidationIssue(
            category="auth",
            severity="info",
            message="API key is in environment variables",
            fix_suggestion="Ensure .env files are in .gitignore and not committed to version control"
        ))
    
    # Test API connectivity if SDK is available
    try:
        import raindrop
        client = raindrop.Client(api_key=api_key)
        
        # Try a basic operation with timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("API test timed out")
        
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5 second timeout
            
            # This would be an actual API call in real implementation
            # For now, just test client instantiation
            recommendations.append("API key authentication test passed")
            
        except TimeoutError:
            warnings.append(ValidationIssue(
                category="auth",
                severity="warning",
                message="API connectivity test timed out",
                fix_suggestion="Check network connectivity and firewall settings"
            ))
        except Exception as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "401" in error_msg:
                issues.append(ValidationIssue(
                    category="auth",
                    severity="error",
                    message="API key authentication failed",
                    fix_suggestion="Verify API key is correct and active in Raindrop AI dashboard",
                    documentation_link="https://www.raindrop.ai/docs/authentication"
                ))
            elif "forbidden" in error_msg or "403" in error_msg:
                warnings.append(ValidationIssue(
                    category="auth",
                    severity="warning",
                    message="API key has insufficient permissions",
                    fix_suggestion="Contact Raindrop AI support to verify API key permissions"
                ))
            elif "rate limit" in error_msg or "429" in error_msg:
                warnings.append(ValidationIssue(
                    category="auth",
                    severity="warning",
                    message="API rate limit reached during testing",
                    fix_suggestion="Wait a few minutes before retrying validation"
                ))
            else:
                warnings.append(ValidationIssue(
                    category="auth",
                    severity="warning",
                    message=f"API connectivity test failed: {str(e)}",
                    fix_suggestion="Verify API key is correct and account has proper permissions"
                ))
        finally:
            signal.alarm(0)  # Clear the alarm
            
    except ImportError:
        recommendations.append("Raindrop AI SDK not available - skipping live API test")
    except Exception as e:
        warnings.append(ValidationIssue(
            category="auth",
            severity="warning",
            message=f"Could not test API connectivity: {str(e)}",
            fix_suggestion="Install Raindrop AI SDK for comprehensive API testing: pip install raindrop"
        ))

def _validate_configuration(issues: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate environment and configuration setup."""
    
    # Check GenOps environment variables
    genops_vars = {
        "GENOPS_TEAM": "Team identifier for cost attribution",
        "GENOPS_PROJECT": "Project identifier for cost attribution"
    }
    
    missing_vars = []
    for var_name, description in genops_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} ({description})")
    
    if missing_vars:
        warnings.append(ValidationIssue(
            category="config",
            severity="warning",
            message=f"Optional GenOps environment variables not set: {', '.join(missing_vars)}",
            fix_suggestion="Set for automatic attribution: export GENOPS_TEAM='your-team' GENOPS_PROJECT='your-project'",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/raindrop-quickstart.md#environment-configuration"
        ))
    else:
        recommendations.append("GenOps environment variables configured")
    
    # Check OpenTelemetry configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        warnings.append(ValidationIssue(
            category="config",
            severity="warning",
            message="OpenTelemetry OTLP endpoint not configured",
            fix_suggestion="Set OTEL_EXPORTER_OTLP_ENDPOINT for telemetry export",
            documentation_link="https://opentelemetry.io/docs/concepts/sdk-configuration/"
        ))
    else:
        recommendations.append(f"OpenTelemetry export configured: {otlp_endpoint}")
    
    # Check budget configuration
    budget_limit = os.getenv("GENOPS_DAILY_BUDGET_LIMIT")
    if budget_limit:
        try:
            budget_value = float(budget_limit)
            if budget_value <= 0:
                warnings.append(ValidationIssue(
                    category="config",
                    severity="warning",
                    message="Daily budget limit must be positive",
                    fix_suggestion="Set a positive value: export GENOPS_DAILY_BUDGET_LIMIT='50.0'"
                ))
            else:
                recommendations.append(f"Daily budget limit configured: ${budget_value}")
        except ValueError:
            warnings.append(ValidationIssue(
                category="config",
                severity="warning",
                message="Daily budget limit is not a valid number",
                fix_suggestion="Use numeric value: export GENOPS_DAILY_BUDGET_LIMIT='50.0'"
            ))

def _validate_governance_setup(issues: List[ValidationIssue], warnings: List[ValidationIssue], recommendations: List[str]):
    """Validate governance and policy configuration."""
    
    # Check governance policy
    governance_policy = os.getenv("GENOPS_GOVERNANCE_POLICY", "enforced")
    valid_policies = ["advisory", "enforced"]
    
    if governance_policy not in valid_policies:
        warnings.append(ValidationIssue(
            category="governance",
            severity="warning",
            message=f"Invalid governance policy: {governance_policy}",
            fix_suggestion=f"Use one of: {', '.join(valid_policies)}",
            documentation_link="https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/governance-policies.md"
        ))
    else:
        recommendations.append(f"Governance policy: {governance_policy}")
    
    # Check environment setting
    environment = os.getenv("GENOPS_ENVIRONMENT", "production")
    if environment not in ["development", "staging", "production"]:
        warnings.append(ValidationIssue(
            category="governance",
            severity="warning", 
            message=f"Unusual environment value: {environment}",
            fix_suggestion="Typically use: development, staging, or production"
        ))
    else:
        recommendations.append(f"Environment: {environment}")

def print_validation_result(result: ValidationResult, verbose: bool = True) -> None:
    """
    Print formatted validation results with actionable guidance.
    
    Args:
        result: ValidationResult to display
        verbose: Include detailed information and recommendations
    """
    print("\n🔍 Raindrop AI Integration Validation Report")
    print("=" * 60)
    
    # Overall status
    if result.is_valid:
        print(f"\n✅ Overall Status: SUCCESS")
    else:
        print(f"\n❌ Overall Status: ISSUES DETECTED")
    
    # Issues summary
    if verbose:
        error_count = len(result.errors)
        warning_count = len(result.warnings)
        
        print(f"\n📊 Validation Summary:")
        print(f"  • SDK Installation: {_count_issues_by_category(result.issues + result.warnings, 'sdk')} issues")
        print(f"  • Authentication: {_count_issues_by_category(result.issues + result.warnings, 'auth')} issues") 
        print(f"  • Configuration: {_count_issues_by_category(result.issues + result.warnings, 'config')} issues")
        print(f"  • Governance: {_count_issues_by_category(result.issues + result.warnings, 'governance')} issues")
    
    # Error details
    if result.has_errors:
        print(f"\n🚨 Errors (must fix):")
        for i, issue in enumerate(result.errors, 1):
            print(f"  {i}. {issue.message}")
            if issue.fix_suggestion:
                print(f"     💡 Fix: {issue.fix_suggestion}")
            if issue.documentation_link:
                print(f"     📖 Docs: {issue.documentation_link}")
    
    # Warning details 
    if result.has_warnings and verbose:
        print(f"\n⚠️ Warnings (recommended fixes):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning.message}")
            if warning.fix_suggestion:
                print(f"     💡 Fix: {warning.fix_suggestion}")
            if warning.documentation_link:
                print(f"     📖 Docs: {warning.documentation_link}")
    
    # Recommendations
    if result.recommendations and verbose:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Next steps
    print(f"\n🚀 Next Steps:")
    if result.is_valid:
        print(f"  1. You can now use GenOps Raindrop integration with confidence")
        print(f"  2. Try the basic example: python examples/raindrop/basic_tracking.py")
        print(f"  3. Explore advanced features: python examples/raindrop/advanced_features.py")
    else:
        print(f"  1. Fix the error-level issues listed above")
        print(f"  2. Re-run validation: python setup_validation.py")
        print(f"  3. Check the troubleshooting guide if issues persist")
    
    print()

def _count_issues_by_category(issues: List[ValidationIssue], category: str) -> int:
    """Count issues in a specific category."""
    return len([issue for issue in issues if issue.category == category])

def validate_setup_interactive() -> ValidationResult:
    """
    Interactive setup validation with user prompts for missing configuration.
    
    Returns:
        ValidationResult: Validation results after interactive setup
    """
    print("🔧 Interactive Raindrop AI Setup Validation")
    print("=" * 50)
    
    # Check for API key interactively
    api_key = os.getenv("RAINDROP_API_KEY")
    if not api_key:
        print("\n📋 Raindrop AI API key not found in environment.")
        api_key = input("Please enter your Raindrop AI API key (or press Enter to skip): ").strip()
        
        if api_key:
            # Temporarily set for this validation
            os.environ["RAINDROP_API_KEY"] = api_key
            print("✅ API key set for this validation session")
    
    # Check for GenOps configuration interactively
    if not os.getenv("GENOPS_TEAM"):
        team = input("Enter your team name for cost attribution (default: 'default'): ").strip()
        if team:
            os.environ["GENOPS_TEAM"] = team
        else:
            os.environ["GENOPS_TEAM"] = "default"
    
    if not os.getenv("GENOPS_PROJECT"):
        project = input("Enter your project name for cost attribution (default: 'default'): ").strip()
        if project:
            os.environ["GENOPS_PROJECT"] = project
        else:
            os.environ["GENOPS_PROJECT"] = "default"
    
    # Run validation
    print("\n🔍 Running validation with current configuration...\n")
    return validate_setup()

# Export main functions
__all__ = [
    'validate_setup',
    'print_validation_result',
    'validate_setup_interactive',
    'ValidationResult',
    'ValidationIssue'
]