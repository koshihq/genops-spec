"""
Together AI Setup Validation for GenOps Integration

Provides comprehensive validation for Together AI + GenOps configurations including:
- API key authentication and model access verification
- Environment setup validation and dependency checking
- Configuration testing with secure output formatting
- Model availability and pricing validation
- Security-compliant diagnostic output
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Import Together pricing and core utilities
from .together_pricing import TogetherPricingCalculator

logger = logging.getLogger(__name__)

# Optional dependencies with graceful handling
try:
    from together import Together
    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False
    Together = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual validation issue with details and remediation."""
    severity: ValidationSeverity
    component: str
    message: str
    remediation: str
    code: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation result with issues and recommendations."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_config: Dict[str, Any] = field(default_factory=dict)
    model_access: List[str] = field(default_factory=list)
    pricing_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]


class TogetherSetupValidator:
    """
    Comprehensive Together AI + GenOps setup validator.
    
    Validates API authentication, model access, configuration, and provides
    security-compliant diagnostic output with remediation guidance.
    """

    def __init__(self, together_api_key: Optional[str] = None):
        """
        Initialize validator with Together AI credentials.
        
        Args:
            together_api_key: Together API key (or uses TOGETHER_API_KEY env var)
        """
        self.together_api_key = together_api_key or os.getenv('TOGETHER_API_KEY')
        self.pricing_calculator = TogetherPricingCalculator()
        self.client = None

        # Initialize client if credentials available
        if self.together_api_key and HAS_TOGETHER:
            try:
                self.client = Together(api_key=self.together_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Together client: {e}")

    def validate_dependencies(self) -> List[ValidationIssue]:
        """Validate required dependencies are installed."""
        issues = []

        if not HAS_TOGETHER:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                component="dependencies",
                message="Together AI Python package not installed",
                remediation="Install with: pip install together",
                code="TOGETHER_MISSING"
            ))

        if not HAS_REQUESTS:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="dependencies",
                message="Requests library not available",
                remediation="Install with: pip install requests",
                code="REQUESTS_MISSING"
            ))

        # Check for optional but recommended packages
        try:
            import numpy
        except ImportError:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                component="dependencies",
                message="NumPy not available (optional but recommended for embeddings)",
                remediation="Install with: pip install numpy",
                code="NUMPY_MISSING"
            ))

        return issues

    def validate_api_key(self) -> List[ValidationIssue]:
        """Validate Together AI API key authentication."""
        issues = []

        if not self.together_api_key:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                component="authentication",
                message="Together API key not provided",
                remediation="Set TOGETHER_API_KEY environment variable or pass together_api_key parameter",
                code="API_KEY_MISSING"
            ))
            return issues

        # Check API key format (Together keys typically start with specific patterns)
        if not self.together_api_key.startswith(('sk-', 'pk-')):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                component="authentication",
                message="API key format may be incorrect",
                remediation="Verify API key from Together AI dashboard",
                code="API_KEY_FORMAT",
                details={"key_prefix": "***REDACTED***"}
            ))

        # Test API key by attempting to list models
        if self.client and HAS_TOGETHER:
            try:
                models = self.client.models.list()
                if models and hasattr(models, 'data') and len(models.data) > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        component="authentication",
                        message=f"API key authenticated successfully - access to {len(models.data)} models",
                        remediation="API authentication working correctly",
                        code="API_KEY_VALID",
                        details={"model_count": len(models.data)}
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        component="authentication",
                        message="API key valid but no models accessible",
                        remediation="Check account permissions and billing status",
                        code="NO_MODEL_ACCESS"
                    ))
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    component="authentication",
                    message="API key authentication failed",
                    remediation="Verify API key is correct and account is active",
                    code="API_KEY_INVALID",
                    details={"error_type": type(e).__name__}
                ))

        return issues

    def validate_model_access(self, test_models: Optional[List[str]] = None) -> Tuple[List[ValidationIssue], List[str]]:
        """Validate access to specific Together AI models."""
        issues = []
        accessible_models = []

        if not self.client:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                component="model_access",
                message="Cannot validate model access without valid client",
                remediation="Fix API key authentication first",
                code="CLIENT_UNAVAILABLE"
            ))
            return issues, accessible_models

        # Default test models if none provided
        if test_models is None:
            test_models = [
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "mistralai/Mixtral-8x7B-Instruct-v0.1"
            ]

        # Test each model with a minimal request
        for model in test_models:
            try:
                # Test with minimal chat completion
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1,
                    temperature=0.1
                )

                if response and hasattr(response, 'choices') and response.choices:
                    accessible_models.append(model)
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        component="model_access",
                        message=f"Model '{model}' accessible and responsive",
                        remediation="Model ready for use",
                        code="MODEL_ACCESSIBLE",
                        details={"model": model}
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        component="model_access",
                        message=f"Model '{model}' responded but with unexpected format",
                        remediation="Model may have API changes - test thoroughly",
                        code="MODEL_RESPONSE_UNEXPECTED",
                        details={"model": model}
                    ))

            except Exception as e:
                error_msg = str(e).lower()

                if "billing" in error_msg or "quota" in error_msg:
                    severity = ValidationSeverity.ERROR
                    remediation = "Check account billing and usage limits"
                    code = "BILLING_ISSUE"
                elif "permission" in error_msg or "access" in error_msg:
                    severity = ValidationSeverity.WARNING
                    remediation = "Model may require higher tier access or approval"
                    code = "ACCESS_RESTRICTED"
                else:
                    severity = ValidationSeverity.WARNING
                    remediation = "Check model availability and account permissions"
                    code = "MODEL_ERROR"

                issues.append(ValidationIssue(
                    severity=severity,
                    component="model_access",
                    message=f"Cannot access model '{model}'",
                    remediation=remediation,
                    code=code,
                    details={"model": model, "error_type": type(e).__name__}
                ))

        return issues, accessible_models

    def validate_configuration(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate GenOps configuration parameters."""
        issues = []

        # Required configuration
        required_fields = ['team', 'project', 'environment']
        for field in required_fields:
            if field not in config or not config[field]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    component="configuration",
                    message=f"Missing required field: {field}",
                    remediation=f"Set {field} for proper cost attribution",
                    code="MISSING_REQUIRED_FIELD",
                    details={"field": field}
                ))

        # Budget validation
        budget_fields = ['daily_budget_limit', 'monthly_budget_limit']
        for field in budget_fields:
            if field in config and config[field] is not None:
                try:
                    budget_value = float(config[field])
                    if budget_value <= 0:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            component="configuration",
                            message=f"Invalid {field}: must be positive",
                            remediation=f"Set {field} to a positive number",
                            code="INVALID_BUDGET"
                        ))
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        component="configuration",
                        message=f"Invalid {field}: must be a number",
                        remediation=f"Set {field} to a numeric value",
                        code="INVALID_BUDGET_TYPE"
                    ))

        # Governance policy validation
        if 'governance_policy' in config:
            valid_policies = ['advisory', 'enforced', 'strict']
            if config['governance_policy'] not in valid_policies:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    component="configuration",
                    message=f"Invalid governance_policy: {config['governance_policy']}",
                    remediation=f"Use one of: {', '.join(valid_policies)}",
                    code="INVALID_GOVERNANCE_POLICY"
                ))

        return issues

    def validate_environment(self) -> List[ValidationIssue]:
        """Validate environment setup and OpenTelemetry configuration."""
        issues = []

        # Check OTEL configuration
        otel_vars = [
            'OTEL_SERVICE_NAME',
            'OTEL_EXPORTER_OTLP_ENDPOINT',
            'OTEL_RESOURCE_ATTRIBUTES'
        ]

        otel_configured = False
        for var in otel_vars:
            if os.getenv(var):
                otel_configured = True
                break

        if not otel_configured:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                component="environment",
                message="OpenTelemetry not configured",
                remediation="Set OTEL_SERVICE_NAME and OTEL_EXPORTER_OTLP_ENDPOINT for observability",
                code="OTEL_NOT_CONFIGURED"
            ))

        # Check GenOps environment variables
        genops_vars = {
            'GENOPS_TEAM': 'Team name for cost attribution',
            'GENOPS_PROJECT': 'Project name for cost tracking',
            'GENOPS_ENVIRONMENT': 'Environment identifier (dev/staging/prod)'
        }

        for var, description in genops_vars.items():
            if not os.getenv(var):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    component="environment",
                    message=f"Optional environment variable {var} not set",
                    remediation=f"Set {var} for {description}",
                    code="GENOPS_VAR_MISSING"
                ))

        return issues

    def run_comprehensive_validation(
        self,
        config: Optional[Dict[str, Any]] = None,
        test_models: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Run complete validation suite for Together AI + GenOps setup.
        
        Args:
            config: Configuration dictionary to validate
            test_models: Specific models to test access for
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        config = config or {}
        all_issues = []
        accessible_models = []

        # Run all validation checks
        all_issues.extend(self.validate_dependencies())
        all_issues.extend(self.validate_api_key())

        model_issues, models = self.validate_model_access(test_models)
        all_issues.extend(model_issues)
        accessible_models = models

        all_issues.extend(self.validate_configuration(config))
        all_issues.extend(self.validate_environment())

        # Check for critical errors
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in all_issues)
        is_valid = not has_errors

        # Generate pricing info for accessible models
        pricing_info = {}
        if accessible_models:
            try:
                model_comparisons = self.pricing_calculator.compare_models(accessible_models[:5])
                pricing_info = {
                    'accessible_models': len(accessible_models),
                    'cost_comparison': model_comparisons,
                    'recommended_starter': model_comparisons[0] if model_comparisons else None
                }
            except Exception as e:
                logger.warning(f"Failed to generate pricing info: {e}")

        # Generate recommendations
        recommendations = []
        if not has_errors:
            recommendations.append("✅ Together AI integration ready for use")
        if accessible_models:
            recommendations.append(f"🎯 {len(accessible_models)} models available for use")

        # Add specific recommendations based on issues
        warning_count = len([i for i in all_issues if i.severity == ValidationSeverity.WARNING])
        if warning_count > 0:
            recommendations.append(f"⚠️ {warning_count} configuration improvements recommended")

        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            validated_config=config,
            model_access=accessible_models,
            pricing_info=pricing_info,
            recommendations=recommendations
        )


def validate_together_setup(
    together_api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    print_results: bool = True
) -> ValidationResult:
    """
    Convenience function for Together AI setup validation.
    
    Args:
        together_api_key: Together API key (optional)
        config: Configuration to validate (optional)
        print_results: Whether to print validation results
        
    Returns:
        ValidationResult: Comprehensive validation results
        
    Example:
        from genops.providers.together_validation import validate_together_setup
        
        result = validate_together_setup(
            config={
                'team': 'ai-research',
                'project': 'model-comparison',
                'environment': 'development',
                'daily_budget_limit': 100.0
            }
        )
    """
    validator = TogetherSetupValidator(together_api_key=together_api_key)
    result = validator.run_comprehensive_validation(config=config)

    if print_results:
        print_validation_result(result, config or {})

    return result


def print_validation_result(result: ValidationResult, config: Dict[str, Any]) -> None:
    """
    Print validation results in a user-friendly format with security compliance.
    
    Args:
        result: Validation results to display
        config: Configuration that was validated
    """
    print("\n" + "=" * 60)
    print("🔧 Together AI + GenOps Setup Validation")
    print("=" * 60)

    # Overall status
    if result.is_valid:
        print("✅ Setup validation PASSED - Ready for Together AI operations")
    else:
        print("❌ Setup validation FAILED - Issues require attention")

    # Print issues by severity
    errors = result.errors
    warnings = result.warnings

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}) - Must be resolved:")
        for error in errors:
            print(f"   • {error.message}")
            print(f"     → {error.remediation}")

    if warnings:
        print(f"\n⚠️ WARNINGS ({len(warnings)}) - Recommended fixes:")
        for warning in warnings:
            print(f"   • {warning.message}")
            print(f"     → {warning.remediation}")

    # Print model access info
    if result.model_access:
        print(f"\n🎯 Model Access ({len(result.model_access)} models available):")
        for model in result.model_access[:5]:  # Show first 5
            print(f"   ✅ {model}")
        if len(result.model_access) > 5:
            print(f"   ... and {len(result.model_access) - 5} more models")

    # Print pricing information
    if result.pricing_info and 'cost_comparison' in result.pricing_info:
        print("\n💰 Cost Overview (per 1000 tokens):")
        comparisons = result.pricing_info['cost_comparison'][:3]
        for comp in comparisons:
            print(f"   • {comp['model']}: ${comp['cost_per_1k_tokens']:.4f} ({comp['tier']} tier)")

    # Print recommendations
    if result.recommendations:
        print("\n📋 Recommendations:")
        for rec in result.recommendations:
            print(f"   {rec}")

    if result.is_valid:
        print("\n💻 Your Configuration:")
        print("```python")
        print("from genops.providers.together import GenOpsTogetherAdapter")
        print()
        print("adapter = GenOpsTogetherAdapter(")
        # Security: Use static configuration display to prevent sensitive data exposure
        print("    # Configuration values have been validated")
        print("    # Please check your environment variables or configuration file")
        print("    # All sensitive values like API keys are properly secured")
        print(")")
        print("```")

    else:
        print("❌ Configuration validation failed. Please check the issues above.")

    return config


def _sanitize_sensitive_field(field_name: str, value: Any) -> Any:
    """
    Comprehensive sanitization for sensitive fields.
    
    Ensures no sensitive data can be logged regardless of type or content.
    Uses allowlist approach - only explicitly safe fields pass through.
    """
    # Define comprehensive patterns for sensitive field detection
    sensitive_patterns = {
        'key', 'token', 'secret', 'password', 'credential', 'auth',
        'private', 'secure', 'sensitive', 'confidential', 'restricted'
    }

    # Check field name against all sensitive patterns
    field_lower = field_name.lower()
    if any(pattern in field_lower for pattern in sensitive_patterns):
        return "***REDACTED***"

    # Allowlist of explicitly safe configuration fields
    safe_fields = {
        'team', 'project', 'environment', 'daily_budget_limit',
        'monthly_budget_limit', 'governance_policy', 'enable_cost_alerts',
        'customer_id', 'cost_center', 'default_model', 'enable_governance',
        'enable_caching', 'retry_attempts', 'timeout_seconds', 'tags'
    }

    if field_name in safe_fields:
        return value
    else:
        # Any unknown field is treated as potentially sensitive
        return "***REDACTED***"


# Convenience exports
__all__ = [
    'TogetherSetupValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'validate_together_setup',
    'print_validation_result'
]
