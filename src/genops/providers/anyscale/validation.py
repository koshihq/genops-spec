"""Validation system for Anyscale integration setup and diagnostics."""

import logging
import os
import re
import sys
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    DEPENDENCIES = "dependencies"
    CONFIGURATION = "configuration"
    CONNECTIVITY = "connectivity"
    MODELS = "models"
    PRICING = "pricing"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    category: ValidationCategory
    level: ValidationLevel
    title: str
    description: str
    fix_suggestion: str = ""
    technical_details: str = ""

    def __str__(self) -> str:
        level_symbol = {
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.ERROR: "❌",
            ValidationLevel.CRITICAL: "🚨"
        }

        return f"{level_symbol[self.level]} {self.title}: {self.description}"


@dataclass
class ValidationResult:
    """Complete validation results."""

    success: bool
    total_checks: int = 0
    passed_checks: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    @property
    def score(self) -> float:
        """Calculate validation score (0-100)."""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)

        # Update success status
        if issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
            self.success = False


class AnyscaleValidator:
    """Comprehensive validator for Anyscale setup."""

    def __init__(self, anyscale_api_key: str = None, anyscale_base_url: str = None):
        """
        Initialize validator.

        Args:
            anyscale_api_key: API key to validate (optional, will check env)
            anyscale_base_url: Base URL to validate (optional)
        """
        self.anyscale_api_key = anyscale_api_key or os.getenv("ANYSCALE_API_KEY")
        self.anyscale_base_url = anyscale_base_url or "https://api.endpoints.anyscale.com/v1"

    # Security methods for secret protection

    def _sanitize_response_text(self, text: str, max_length: int = 200) -> str:
        """Sanitize API response text before logging."""
        if not text:
            return "No response text"
        truncated = text[:max_length]
        sanitized = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', truncated)
        sanitized = re.sub(r'"token":\s*"\S+"', '"token": "[REDACTED]"', sanitized)
        return sanitized

    def _build_headers(self) -> dict:
        """Build HTTP headers with secret protection."""
        auth_value = "Bearer " + self.anyscale_api_key
        return {"Authorization": auth_value}

    def validate(self) -> ValidationResult:
        """
        Run comprehensive validation checks.

        Returns:
            ValidationResult with detailed diagnostics
        """
        result = ValidationResult(success=True)

        # Collect system information
        result.system_info = {
            "python_version": sys.version,
            "anyscale_base_url": self.anyscale_base_url,
            "has_api_key": bool(self.anyscale_api_key),
        }

        # Run validation checks
        self._check_dependencies(result)
        self._check_configuration(result)
        self._check_connectivity(result)
        self._check_models(result)
        self._check_pricing_database(result)

        # Generate recommendations
        self._generate_recommendations(result)

        return result

    def _check_dependencies(self, result: ValidationResult):
        """Check required and optional dependencies."""

        # Python version check
        result.total_checks += 1
        py_version = sys.version_info
        if py_version >= (3, 8):
            result.passed_checks += 1
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.INFO,
                title="Python Version",
                description=f"Python {py_version.major}.{py_version.minor}.{py_version.micro} detected",
                fix_suggestion="Compatible Python version"
            ))
        else:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.CRITICAL,
                title="Python Version",
                description=f"Python {py_version.major}.{py_version.minor} is too old",
                fix_suggestion="Upgrade to Python 3.8 or later",
                technical_details="GenOps requires Python 3.8+ for type hints and async support"
            ))

        # Requests library check
        result.total_checks += 1
        if HAS_REQUESTS:
            result.passed_checks += 1
            import requests
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.INFO,
                title="Requests Library",
                description=f"requests {requests.__version__} installed",
                fix_suggestion="HTTP client available"
            ))
        else:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR,
                title="Requests Library Missing",
                description="requests library not found",
                fix_suggestion="Install with: pip install requests",
                technical_details="Required for HTTP API calls to Anyscale"
            ))

        # OpenAI SDK check (optional but recommended)
        result.total_checks += 1
        if HAS_OPENAI_SDK:
            result.passed_checks += 1
            import openai
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.INFO,
                title="OpenAI SDK",
                description=f"openai {openai.__version__} installed",
                fix_suggestion="Enhanced compatibility available"
            ))
        else:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.WARNING,
                title="OpenAI SDK Not Installed",
                description="OpenAI SDK provides enhanced compatibility",
                fix_suggestion="Install with: pip install openai (optional but recommended)",
                technical_details="Anyscale is OpenAI-compatible; SDK provides better error handling"
            ))

        # OpenTelemetry check
        result.total_checks += 1
        try:
            import opentelemetry
            result.passed_checks += 1
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.INFO,
                title="OpenTelemetry",
                description="OpenTelemetry packages available",
                fix_suggestion="Telemetry export enabled"
            ))
        except ImportError:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR,
                title="OpenTelemetry Missing",
                description="OpenTelemetry packages not found",
                fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk",
                technical_details="Required for governance telemetry export"
            ))

    def _check_configuration(self, result: ValidationResult):
        """Check configuration and environment variables."""

        # API key check
        result.total_checks += 1
        if self.anyscale_api_key:
            # Check key format (basic validation)
            if len(self.anyscale_api_key) > 10:  # Reasonable minimum length
                result.passed_checks += 1
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title="API Key Configuration",
                    description="ANYSCALE_API_KEY is set",
                    fix_suggestion="API key configured correctly"
                ))
            else:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.WARNING,
                    title="API Key Format",
                    description="API key seems too short",
                    fix_suggestion="Verify your API key from Anyscale console Credentials page",
                    technical_details="API keys should be longer than 10 characters"
                ))
        else:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.CONFIGURATION,
                level=ValidationLevel.CRITICAL,
                title="API Key Missing",
                description="ANYSCALE_API_KEY environment variable not set",
                fix_suggestion="Set with: export ANYSCALE_API_KEY='your-key-here'",
                technical_details="API key required for authentication with Anyscale Endpoints"
            ))

        # Base URL check
        result.total_checks += 1
        if self.anyscale_base_url:
            if self.anyscale_base_url.startswith("https://"):
                result.passed_checks += 1
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title="Base URL Configuration",
                    description=f"Base URL: {self.anyscale_base_url}",
                    fix_suggestion="Using secure HTTPS connection"
                ))
            else:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.WARNING,
                    title="Insecure Base URL",
                    description="Base URL does not use HTTPS",
                    fix_suggestion="Use HTTPS for production: https://api.endpoints.anyscale.com/v1",
                    technical_details="HTTP connections are insecure and should only be used for testing"
                ))

    def _check_connectivity(self, result: ValidationResult):
        """Check network connectivity to Anyscale API."""

        if not self.anyscale_api_key:
            # Skip connectivity checks if no API key
            return

        if not HAS_REQUESTS and not HAS_OPENAI_SDK:
            # Skip if no HTTP client available
            return

        # Test API connectivity
        result.total_checks += 1
        try:
            if HAS_REQUESTS:
                response = requests.get(
                    f"{self.anyscale_base_url}/models",
                    headers=self._build_headers(),
                    timeout=10
                )

                if response.status_code == 200:
                    result.passed_checks += 1
                    models = response.json().get('data', [])
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.CONNECTIVITY,
                        level=ValidationLevel.INFO,
                        title="API Connectivity",
                        description=f"Successfully connected to Anyscale API ({len(models)} models available)",
                        fix_suggestion="API is reachable and responsive"
                    ))
                elif response.status_code == 401:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.CONNECTIVITY,
                        level=ValidationLevel.ERROR,
                        title="Authentication Failed",
                        description="API key rejected by Anyscale",
                        fix_suggestion="Verify your API key from Anyscale console Credentials page",
                        technical_details=f"HTTP {response.status_code}: {self._sanitize_response_text(response.text)}"
                    ))
                else:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.CONNECTIVITY,
                        level=ValidationLevel.WARNING,
                        title="API Response Error",
                        description=f"Unexpected response: HTTP {response.status_code}",
                        fix_suggestion="Check Anyscale service status",
                        technical_details=self._sanitize_response_text(response.text, 200)
                    ))

        except requests.exceptions.Timeout:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.WARNING,
                title="Connection Timeout",
                description="Request to Anyscale API timed out",
                fix_suggestion="Check network connectivity and firewall settings",
                technical_details="Connection timeout after 10 seconds"
            ))
        except requests.exceptions.ConnectionError as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.ERROR,
                title="Connection Failed",
                description="Could not connect to Anyscale API",
                fix_suggestion="Check internet connection and DNS resolution",
                technical_details=str(e)
            ))
        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.WARNING,
                title="Connectivity Check Failed",
                description=f"Unexpected error: {type(e).__name__}",
                fix_suggestion="Check error details for more information",
                technical_details=str(e)
            ))

    def _check_models(self, result: ValidationResult):
        """Check available models and their accessibility."""

        if not self.anyscale_api_key or not HAS_REQUESTS:
            return

        result.total_checks += 1
        try:
            response = requests.get(
                f"{self.anyscale_base_url}/models",
                headers=self._build_headers(),
                timeout=10
            )

            if response.status_code == 200:
                result.passed_checks += 1
                models_data = response.json().get('data', [])

                # Categorize models
                chat_models = [m for m in models_data if 'chat' in m.get('id', '').lower() or 'llama' in m.get('id', '').lower()]
                embedding_models = [m for m in models_data if 'embed' in m.get('id', '').lower() or 'gte' in m.get('id', '').lower()]

                result.add_issue(ValidationIssue(
                    category=ValidationCategory.MODELS,
                    level=ValidationLevel.INFO,
                    title="Model Availability",
                    description=f"Found {len(chat_models)} chat models and {len(embedding_models)} embedding models",
                    fix_suggestion=f"Models accessible: {', '.join([m['id'] for m in models_data[:3]])}..."
                ))

        except Exception as e:
            logger.debug(f"Model check failed: {e}")
            # Non-critical - don't fail validation

    def _check_pricing_database(self, result: ValidationResult):
        """Check pricing database completeness."""

        result.total_checks += 1
        try:
            from .pricing import ANYSCALE_PRICING, AnyscalePricing

            pricing = AnyscalePricing()
            num_models = len(ANYSCALE_PRICING)

            result.passed_checks += 1
            result.add_issue(ValidationIssue(
                category=ValidationCategory.PRICING,
                level=ValidationLevel.INFO,
                title="Pricing Database",
                description=f"Pricing data available for {num_models} models",
                fix_suggestion="Cost calculation ready"
            ))

        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.PRICING,
                level=ValidationLevel.WARNING,
                title="Pricing Database Error",
                description="Could not load pricing database",
                fix_suggestion="Verify pricing.py module is intact",
                technical_details=str(e)
            ))

    def _generate_recommendations(self, result: ValidationResult):
        """Generate setup recommendations based on validation results."""

        if not self.anyscale_api_key:
            result.recommendations.append(
                "🔑 Get your API key from: https://console.anyscale.com/credentials"
            )

        if not HAS_OPENAI_SDK:
            result.recommendations.append(
                "💡 Install OpenAI SDK for better compatibility: pip install openai"
            )

        if result.score == 100:
            result.recommendations.append(
                "✅ All checks passed! Your Anyscale setup is ready to use."
            )
        elif result.score >= 75:
            result.recommendations.append(
                "⚠️ Most checks passed. Review warnings above for optimal setup."
            )
        else:
            result.recommendations.append(
                "❌ Critical issues detected. Fix errors above before using Anyscale provider."
            )


def validate_setup(
    anyscale_api_key: str = None,
    anyscale_base_url: str = None
) -> ValidationResult:
    """
    Validate Anyscale setup and configuration.

    Args:
        anyscale_api_key: API key to validate (optional, checks env)
        anyscale_base_url: Base URL to validate (optional)

    Returns:
        ValidationResult with comprehensive diagnostics

    Example:
        from genops.providers.anyscale.validation import validate_setup, print_validation_result

        result = validate_setup()
        print_validation_result(result)
    """
    validator = AnyscaleValidator(anyscale_api_key, anyscale_base_url)
    return validator.validate()


def print_validation_result(result: ValidationResult):
    """
    Print validation results in user-friendly format.

    Args:
        result: ValidationResult to display
    """
    print("\n" + "="*70)
    print("🔍 GenOps Anyscale Setup Validation")
    print("="*70 + "\n")

    # Overall status
    if result.success:
        print(f"✅ Status: PASSED (Score: {result.score:.1f}/100)")
    else:
        print(f"❌ Status: FAILED (Score: {result.score:.1f}/100)")

    print(f"📊 Checks: {result.passed_checks}/{result.total_checks} passed\n")

    # Group issues by category
    categories = {}
    for issue in result.issues:
        if issue.category not in categories:
            categories[issue.category] = []
        categories[issue.category].append(issue)

    # Print issues by category
    for category, issues in sorted(categories.items(), key=lambda x: x[0].value):
        print(f"\n📋 {category.value.upper()}")
        print("-" * 70)

        for issue in issues:
            print(f"{issue}")
            if issue.fix_suggestion:
                print(f"   💡 Fix: {issue.fix_suggestion}")
            if issue.technical_details:
                print(f"   🔧 Details: {issue.technical_details}")
            print()

    # Print recommendations
    if result.recommendations:
        print("\n📝 RECOMMENDATIONS")
        print("-" * 70)
        for rec in result.recommendations:
            print(f"{rec}")

    print("\n" + "="*70)
    print(f"Validation completed: {result.total_checks} checks performed")
    print("="*70 + "\n")


# Export public API
__all__ = [
    'ValidationLevel',
    'ValidationCategory',
    'ValidationIssue',
    'ValidationResult',
    'AnyscaleValidator',
    'validate_setup',
    'print_validation_result',
]
