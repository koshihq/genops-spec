#!/usr/bin/env python3
"""
Gemini setup validation and diagnostics for GenOps.

This module provides comprehensive validation utilities for Google Gemini
integration, ensuring proper setup and configuration for optimal GenOps
governance and cost tracking.

Features:
- Comprehensive setup validation with actionable error messages
- API key and authentication verification
- Model availability testing
- GenOps integration validation
- Performance and connectivity testing
- Detailed diagnostic reporting

Usage:
    from genops.providers.gemini_validation import validate_gemini_setup, print_validation_result
    
    # Run full validation
    result = validate_gemini_setup()
    
    # Display user-friendly results
    print_validation_result(result, detailed=True)
    
    # Check specific aspects
    if result.success:
        print("✅ Gemini setup is ready for production")
    else:
        print("❌ Setup issues found - see recommendations")
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import dependencies with graceful fallback
try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

try:
    from genops.core.telemetry import GenOpsTelemetry
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False


class ValidationLevel(Enum):
    """Validation severity levels."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    level: ValidationLevel
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None
    documentation_link: Optional[str] = None


@dataclass
class GeminiValidationResult:
    """Comprehensive Gemini validation results."""
    success: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return len(self.errors) > 0 or any(check.level == ValidationLevel.ERROR for check in self.checks)

    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return len(self.warnings) > 0 or any(check.level == ValidationLevel.WARNING for check in self.checks)

    def get_error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors) + len([c for c in self.checks if c.level == ValidationLevel.ERROR])

    def get_warning_count(self) -> int:
        """Get total number of warnings."""
        return len(self.warnings) + len([c for c in self.checks if c.level == ValidationLevel.WARNING])


def validate_gemini_setup(
    api_key: Optional[str] = None,
    test_connectivity: bool = True,
    test_model_access: bool = True,
    performance_test: bool = False,
    detailed: bool = True
) -> GeminiValidationResult:
    """
    Perform comprehensive Gemini setup validation.
    
    Args:
        api_key: API key to validate (uses environment if not provided)
        test_connectivity: Test API connectivity
        test_model_access: Test access to specific models
        performance_test: Run performance benchmarks
        detailed: Include detailed diagnostic information
    
    Returns:
        GeminiValidationResult with comprehensive validation results
    """
    result = GeminiValidationResult(success=False)
    checks = []

    # Environment information
    result.environment_info = {
        "gemini_sdk_available": GEMINI_AVAILABLE,
        "genops_available": GENOPS_AVAILABLE,
        "api_key_env_set": bool(os.getenv("GEMINI_API_KEY")),
        "validation_timestamp": time.time()
    }

    # 1. Check Gemini SDK availability
    if GEMINI_AVAILABLE:
        checks.append(ValidationCheck(
            name="gemini_sdk_availability",
            level=ValidationLevel.SUCCESS,
            message="Google Gemini SDK is installed and available",
            details="Successfully imported google.genai package"
        ))
    else:
        checks.append(ValidationCheck(
            name="gemini_sdk_availability",
            level=ValidationLevel.ERROR,
            message="Google Gemini SDK not installed",
            details="The google-generativeai package is required for Gemini integration",
            fix_suggestion="Install with: pip install google-generativeai",
            documentation_link="https://ai.google.dev/gemini-api/docs/quickstart"
        ))
        result.errors.append("Google Gemini SDK not installed")

    # 2. Check GenOps core availability
    if GENOPS_AVAILABLE:
        checks.append(ValidationCheck(
            name="genops_core_availability",
            level=ValidationLevel.SUCCESS,
            message="GenOps core is available for telemetry",
            details="Full governance and cost tracking capabilities enabled"
        ))
    else:
        checks.append(ValidationCheck(
            name="genops_core_availability",
            level=ValidationLevel.WARNING,
            message="GenOps core not available",
            details="Running in basic mode without full telemetry integration",
            fix_suggestion="Ensure GenOps core modules are properly installed"
        ))
        result.warnings.append("GenOps core not available - limited functionality")

    # Early exit if SDK not available
    if not GEMINI_AVAILABLE:
        result.success = False
        result.checks = checks
        return result

    # 3. Check API key configuration
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY")

    if effective_api_key:
        # Validate API key format (basic check)
        if effective_api_key.startswith("AIza") and len(effective_api_key) > 20:
            checks.append(ValidationCheck(
                name="api_key_format",
                level=ValidationLevel.SUCCESS,
                message="API key appears to be in correct format",
                details="API key format validation passed"
            ))
        else:
            checks.append(ValidationCheck(
                name="api_key_format",
                level=ValidationLevel.WARNING,
                message="API key format appears unusual",
                details="API key doesn't match expected Google API key pattern",
                fix_suggestion="Verify API key is correct from Google AI Studio",
                documentation_link="https://ai.google.dev/"
            ))
            result.warnings.append("API key format validation failed")
    else:
        checks.append(ValidationCheck(
            name="api_key_configuration",
            level=ValidationLevel.ERROR,
            message="API key not configured",
            details="No API key found in environment variable GEMINI_API_KEY or parameter",
            fix_suggestion="Set GEMINI_API_KEY environment variable or pass api_key parameter",
            documentation_link="https://ai.google.dev/"
        ))
        result.errors.append("API key not configured")

    # 4. Test API connectivity
    if test_connectivity and effective_api_key:
        try:
            client = genai.Client(api_key=effective_api_key)

            # Test basic connectivity with minimal request
            start_time = time.time()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello"
            )
            connectivity_latency = (time.time() - start_time) * 1000

            if response and hasattr(response, 'text'):
                checks.append(ValidationCheck(
                    name="api_connectivity",
                    level=ValidationLevel.SUCCESS,
                    message="API connectivity test passed",
                    details=f"Successfully connected to Gemini API (latency: {connectivity_latency:.0f}ms)"
                ))
                result.performance_metrics["connectivity_latency_ms"] = connectivity_latency
            else:
                checks.append(ValidationCheck(
                    name="api_connectivity",
                    level=ValidationLevel.WARNING,
                    message="API connectivity test returned unexpected response",
                    details="API responded but response format was unexpected"
                ))
                result.warnings.append("API connectivity test returned unexpected response")

        except Exception as e:
            error_message = str(e).lower()

            if "api_key" in error_message or "authentication" in error_message:
                checks.append(ValidationCheck(
                    name="api_connectivity",
                    level=ValidationLevel.ERROR,
                    message="API key authentication failed",
                    details=f"Authentication error: {e}",
                    fix_suggestion="1) Verify API key is correct, 2) Check API key has proper permissions, 3) Ensure API key is not expired",
                    documentation_link="https://ai.google.dev/"
                ))
                result.errors.append("API key authentication failed")
            elif "quota" in error_message or "rate" in error_message:
                checks.append(ValidationCheck(
                    name="api_connectivity",
                    level=ValidationLevel.WARNING,
                    message="API quota or rate limit exceeded",
                    details=f"Rate limiting error: {e}",
                    fix_suggestion="Wait a few minutes and try again, or upgrade to paid tier for higher limits"
                ))
                result.warnings.append("API quota or rate limit exceeded")
            else:
                checks.append(ValidationCheck(
                    name="api_connectivity",
                    level=ValidationLevel.ERROR,
                    message="API connectivity test failed",
                    details=f"Connection error: {e}",
                    fix_suggestion="1) Check internet connection, 2) Verify Gemini API service status, 3) Try again in a few minutes"
                ))
                result.errors.append(f"API connectivity test failed: {e}")

    # 5. Test model access
    if test_model_access and effective_api_key and not result.has_errors():
        models_to_test = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash"
        ]

        accessible_models = []
        inaccessible_models = []

        try:
            client = genai.Client(api_key=effective_api_key)

            for model in models_to_test:
                try:
                    # Test with minimal request
                    response = client.models.generate_content(
                        model=model,
                        contents="Test"
                    )
                    if response:
                        accessible_models.append(model)
                except Exception as e:
                    inaccessible_models.append((model, str(e)))

            if accessible_models:
                checks.append(ValidationCheck(
                    name="model_access",
                    level=ValidationLevel.SUCCESS,
                    message=f"Successfully accessed {len(accessible_models)} model(s)",
                    details=f"Accessible models: {', '.join(accessible_models)}"
                ))
                result.performance_metrics["accessible_models"] = accessible_models

            if inaccessible_models:
                model_names = [model for model, _ in inaccessible_models]
                checks.append(ValidationCheck(
                    name="model_access_limited",
                    level=ValidationLevel.WARNING,
                    message=f"Some models are inaccessible: {', '.join(model_names)}",
                    details="This may be due to regional restrictions or API tier limitations",
                    fix_suggestion="Check model availability in your region or upgrade API tier"
                ))
                result.warnings.append(f"Some models inaccessible: {', '.join(model_names)}")

        except Exception as e:
            checks.append(ValidationCheck(
                name="model_access",
                level=ValidationLevel.ERROR,
                message="Model access testing failed",
                details=f"Unable to test model access: {e}",
                fix_suggestion="Check API key permissions and Gemini service availability"
            ))
            result.errors.append(f"Model access testing failed: {e}")

    # 6. Performance testing (optional)
    if performance_test and effective_api_key and not result.has_errors():
        try:
            client = genai.Client(api_key=effective_api_key)

            # Test different request sizes
            test_prompts = [
                ("small", "Hello"),
                ("medium", "Explain quantum computing in simple terms."),
                ("large", "Write a detailed analysis of the impact of artificial intelligence on modern society, covering economic, social, and technological aspects. Include examples and future predictions.")
            ]

            performance_results = {}

            for size, prompt in test_prompts:
                try:
                    start_time = time.time()
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    latency = (time.time() - start_time) * 1000

                    performance_results[f"{size}_request_latency_ms"] = latency

                    # Estimate tokens (rough)
                    input_tokens = len(prompt.split()) * 1.3
                    output_tokens = len(response.text.split()) * 1.3 if hasattr(response, 'text') else 0

                    performance_results[f"{size}_input_tokens"] = int(input_tokens)
                    performance_results[f"{size}_output_tokens"] = int(output_tokens)

                except Exception as e:
                    performance_results[f"{size}_request_error"] = str(e)

            result.performance_metrics.update(performance_results)

            # Analyze performance
            avg_latency = sum(
                v for k, v in performance_results.items()
                if k.endswith("_latency_ms") and isinstance(v, (int, float))
            ) / max(1, len([k for k in performance_results.keys() if k.endswith("_latency_ms")]))

            if avg_latency < 2000:
                checks.append(ValidationCheck(
                    name="performance_test",
                    level=ValidationLevel.SUCCESS,
                    message=f"Performance test passed (avg latency: {avg_latency:.0f}ms)",
                    details="API response times are within acceptable ranges"
                ))
            elif avg_latency < 5000:
                checks.append(ValidationCheck(
                    name="performance_test",
                    level=ValidationLevel.WARNING,
                    message=f"Performance test completed with higher latency (avg: {avg_latency:.0f}ms)",
                    details="API response times are acceptable but could be optimized",
                    fix_suggestion="Consider using Gemini Flash-Lite for faster responses"
                ))
                result.warnings.append("Higher than expected API latency")
            else:
                checks.append(ValidationCheck(
                    name="performance_test",
                    level=ValidationLevel.WARNING,
                    message=f"Performance test shows high latency (avg: {avg_latency:.0f}ms)",
                    details="API response times are higher than recommended",
                    fix_suggestion="Check network connectivity or consider different models/regions"
                ))
                result.warnings.append("High API latency detected")

        except Exception as e:
            checks.append(ValidationCheck(
                name="performance_test",
                level=ValidationLevel.WARNING,
                message="Performance testing failed",
                details=f"Unable to complete performance tests: {e}",
                fix_suggestion="Performance testing is optional - core functionality may still work"
            ))
            result.warnings.append("Performance testing failed")

    # 7. Generate recommendations
    recommendations = []

    if result.get_error_count() == 0:
        recommendations.append("✅ Gemini setup is ready for production use")

    if not GENOPS_AVAILABLE:
        recommendations.append("Consider installing full GenOps core for complete telemetry capabilities")

    if "connectivity_latency_ms" in result.performance_metrics:
        latency = result.performance_metrics["connectivity_latency_ms"]
        if latency > 2000:
            recommendations.append("High latency detected - consider optimizing network or using regional endpoints")

    if "accessible_models" in result.performance_metrics:
        accessible_count = len(result.performance_metrics["accessible_models"])
        if accessible_count < 2:
            recommendations.append("Limited model access - consider upgrading API tier for more model options")

    # Environment-specific recommendations
    if not os.getenv("GEMINI_API_KEY"):
        recommendations.append("Set GEMINI_API_KEY environment variable for easier configuration management")

    result.recommendations = recommendations
    result.checks = checks
    result.success = result.get_error_count() == 0

    return result


def validate_gemini_quick(api_key: Optional[str] = None) -> bool:
    """
    Quick validation check for Gemini setup.
    
    Args:
        api_key: API key to validate (optional)
    
    Returns:
        True if basic setup is valid, False otherwise
    """
    try:
        if not GEMINI_AVAILABLE:
            return False

        effective_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not effective_api_key:
            return False

        # Quick connectivity test
        client = genai.Client(api_key=effective_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello"
        )

        return bool(response and hasattr(response, 'text'))

    except Exception:
        return False


def print_validation_result(result: GeminiValidationResult, detailed: bool = False) -> None:
    """
    Print validation results in a user-friendly format.
    
    Args:
        result: GeminiValidationResult to display
        detailed: Whether to show detailed information
    """
    print("=" * 60)
    print("🔍 GenOps Gemini Validation Results")
    print("=" * 60)

    # Overall status
    if result.success:
        print("✅ OVERALL STATUS: PASSED")
        print("   Your Gemini integration is ready for production use!")
    else:
        print("❌ OVERALL STATUS: FAILED")
        print("   Setup issues found that need attention.")

    print()

    # Summary counts
    error_count = result.get_error_count()
    warning_count = result.get_warning_count()
    success_count = len([c for c in result.checks if c.level == ValidationLevel.SUCCESS])

    print("📊 SUMMARY:")
    print(f"   ✅ Passed: {success_count}")
    print(f"   ⚠️  Warnings: {warning_count}")
    print(f"   ❌ Errors: {error_count}")
    print()

    # Show individual checks if detailed or if there are issues
    if detailed or error_count > 0 or warning_count > 0:
        print("🔍 DETAILED RESULTS:")
        print()

        for check in result.checks:
            # Icon based on level
            if check.level == ValidationLevel.SUCCESS:
                icon = "✅"
            elif check.level == ValidationLevel.WARNING:
                icon = "⚠️ "
            else:
                icon = "❌"

            print(f"{icon} {check.message}")

            if detailed and check.details:
                print(f"   Details: {check.details}")

            if check.fix_suggestion:
                print(f"   💡 Fix: {check.fix_suggestion}")

            if detailed and check.documentation_link:
                print(f"   📖 Docs: {check.documentation_link}")

            print()

    # Performance metrics
    if detailed and result.performance_metrics:
        print("⚡ PERFORMANCE METRICS:")
        for key, value in result.performance_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value}")
            elif isinstance(value, list):
                print(f"   {key}: {', '.join(value)}")
        print()

    # Recommendations
    if result.recommendations:
        print("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        print()

    # Enhanced quick fixes for common issues (per CLAUDE.md standards)
    if error_count > 0:
        print("🚨 QUICK FIXES (Copy-paste these commands):")

        if not GEMINI_AVAILABLE:
            print("   📦 SDK Missing:")
            print("      pip install google-generativeai")
            print("      # Or with GenOps: pip install genops-ai[gemini]")

        if not os.getenv("GEMINI_API_KEY"):
            print("   🔑 API Key Missing:")
            print("      export GEMINI_API_KEY='your_api_key_here'")
            print("      # Get your FREE API key at: https://ai.google.dev/")
            print("      # Click 'Get API key' → 'Create API key in new project'")

        # Check for specific error patterns in results
        has_auth_error = any("authentication" in str(err).lower() for err in result.errors)
        has_quota_error = any("quota" in str(err).lower() for err in result.errors)
        has_network_error = any(any(net_term in str(err).lower() for net_term in ["network", "connection", "timeout"]) for err in result.errors)

        if has_auth_error:
            print("   🔐 Authentication Issue:")
            print("      # Your API key may be invalid or expired")
            print("      # 1. Generate new API key at https://ai.google.dev/")
            print("      # 2. export GEMINI_API_KEY='new_api_key_here'")
            print("      # 3. Test: python examples/gemini/hello_genops_minimal.py")

        if has_quota_error:
            print("   📊 Quota/Rate Limit:")
            print("      # Free tier has limits. Solutions:")
            print("      # 1. Wait 1-2 minutes and try again")
            print("      # 2. Upgrade to paid tier at https://ai.google.dev/")
            print("      # 3. Reduce request frequency")

        if has_network_error:
            print("   🌐 Network/Connectivity:")
            print("      # Check internet connection and firewall")
            print("      # Test: curl -I https://generativelanguage.googleapis.com/")
            print("      # Corporate firewall? Check with IT team")

        print("   🔧 Test Your Fix:")
        print("      python -c \"from genops.providers.gemini import validate_setup; validate_setup()\"")
        print()

    print("=" * 60)


def quick_validate() -> None:
    """
    Quick validation function for command-line use with actionable feedback.
    
    Per CLAUDE.md standards: provides specific fix suggestions for common issues.
    """
    print("🔍 Running quick Gemini validation...")

    if validate_gemini_quick():
        print("✅ Gemini setup appears to be working correctly!")
        print("🎯 Next steps:")
        print("   • Try: python examples/gemini/hello_genops_minimal.py")
        print("   • Learn: python examples/gemini/basic_tracking.py")
    else:
        print("❌ Gemini setup validation failed")
        print()

        # Provide specific guidance based on what's missing
        if not GEMINI_AVAILABLE:
            print("🔧 IMMEDIATE FIX NEEDED - SDK Missing:")
            print("   pip install google-generativeai")
            print()

        if not os.getenv("GEMINI_API_KEY"):
            print("🔧 IMMEDIATE FIX NEEDED - API Key Missing:")
            print("   1. Get FREE API key: https://ai.google.dev/")
            print("   2. export GEMINI_API_KEY='your_api_key_here'")
            print()

        print("📋 For comprehensive diagnostics, run:")
        print("   python -c \"from genops.providers.gemini_validation import validate_gemini_setup, print_validation_result; print_validation_result(validate_gemini_setup(), detailed=True)\"")
        print()
        print("💡 Or try the minimal example first:")
        print("   python examples/gemini/hello_genops_minimal.py")


# Export main functions and classes
__all__ = [
    'GeminiValidationResult',
    'ValidationCheck',
    'ValidationLevel',
    'validate_gemini_setup',
    'validate_gemini_quick',
    'print_validation_result',
    'quick_validate'
]
