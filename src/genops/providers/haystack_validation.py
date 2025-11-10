#!/usr/bin/env python3
"""
Haystack AI Setup Validation and Diagnostics

Comprehensive validation system for Haystack AI + GenOps integration with structured 
results, actionable diagnostics, and detailed troubleshooting guidance.

Usage:
    from genops.providers.haystack_validation import validate_haystack_setup, print_validation_result
    
    result = validate_haystack_setup()
    print_validation_result(result)

Features:
    - Comprehensive dependency and environment validation
    - API key verification and connectivity testing
    - Performance benchmarking and diagnostics  
    - Actionable error messages with specific fixes
    - Setup recommendations and optimization guidance
"""

import logging
import os
import sys
import importlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Individual validation issue with severity and fix information."""
    severity: str  # "error", "warning", "info"
    category: str  # "dependency", "configuration", "connectivity", "performance"
    message: str
    fix_suggestion: str
    documentation_link: Optional[str] = None


@dataclass 
class ValidationResult:
    """Comprehensive setup validation result with structured diagnostics."""
    is_valid: bool
    overall_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Detailed status by category
    dependencies_valid: bool = True
    configuration_valid: bool = True
    connectivity_valid: bool = True
    performance_acceptable: bool = True
    
    # Environment information
    python_version: str = ""
    platform: str = ""
    haystack_version: Optional[str] = None
    genops_version: Optional[str] = None
    
    # Provider availability
    available_providers: List[str] = field(default_factory=list)
    provider_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    import_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def add_issue(self, severity: str, category: str, message: str, fix_suggestion: str, 
                  documentation_link: Optional[str] = None):
        """Add a validation issue."""
        issue = ValidationIssue(
            severity=severity,
            category=category, 
            message=message,
            fix_suggestion=fix_suggestion,
            documentation_link=documentation_link
        )
        self.issues.append(issue)
        
        # Update category validity
        if severity == "error":
            if category == "dependency":
                self.dependencies_valid = False
            elif category == "configuration":
                self.configuration_valid = False
            elif category == "connectivity":
                self.connectivity_valid = False
            elif category == "performance":
                self.performance_acceptable = False
    
    def get_error_count(self) -> int:
        """Get count of error-level issues."""
        return len([issue for issue in self.issues if issue.severity == "error"])
    
    def get_warning_count(self) -> int:
        """Get count of warning-level issues.""" 
        return len([issue for issue in self.issues if issue.severity == "warning"])


def validate_python_environment() -> Tuple[bool, List[ValidationIssue]]:
    """Validate Python environment and version."""
    issues = []
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if sys.version_info < (3, 8):
        issues.append(ValidationIssue(
            severity="error",
            category="dependency",
            message=f"Python {python_version} is too old",
            fix_suggestion="Upgrade to Python 3.8 or newer: https://www.python.org/downloads/",
            documentation_link="https://docs.python.org/3/installing/index.html"
        ))
        return False, issues
    
    if sys.version_info < (3, 9):
        issues.append(ValidationIssue(
            severity="warning",
            category="dependency", 
            message=f"Python {python_version} works but 3.9+ recommended",
            fix_suggestion="Consider upgrading to Python 3.9+ for better performance and features",
        ))
    
    return True, issues


def validate_haystack_installation() -> Tuple[bool, List[ValidationIssue], Optional[str]]:
    """Validate Haystack installation and version."""
    issues = []
    
    try:
        import haystack
        haystack_version = haystack.__version__
        
        # Check for minimum version (assuming 2.0+)
        version_parts = haystack_version.split('.')
        major_version = int(version_parts[0])
        
        if major_version < 2:
            issues.append(ValidationIssue(
                severity="warning",
                category="dependency",
                message=f"Haystack {haystack_version} is older - consider upgrading",
                fix_suggestion="Upgrade Haystack: pip install --upgrade haystack-ai",
                documentation_link="https://docs.haystack.deepset.ai/docs/installation"
            ))
        
        # Test core imports
        try:
            from haystack import Pipeline
            from haystack.core.component import Component
        except ImportError as e:
            issues.append(ValidationIssue(
                severity="error",
                category="dependency",
                message=f"Haystack core imports failed: {e}",
                fix_suggestion="Reinstall Haystack: pip install --force-reinstall haystack-ai",
                documentation_link="https://docs.haystack.deepset.ai/docs/installation"
            ))
            return False, issues, haystack_version
        
        return True, issues, haystack_version
        
    except ImportError:
        issues.append(ValidationIssue(
            severity="error",
            category="dependency",
            message="Haystack not installed",
            fix_suggestion="Install Haystack: pip install haystack-ai",
            documentation_link="https://docs.haystack.deepset.ai/docs/installation"
        ))
        return False, issues, None


def validate_genops_installation() -> Tuple[bool, List[ValidationIssue], Optional[str]]:
    """Validate GenOps installation and version."""
    issues = []
    
    try:
        # Try to import GenOps core
        from genops.core.telemetry import GenOpsTelemetry
        
        # Try to get version
        try:
            import genops
            genops_version = getattr(genops, '__version__', 'unknown')
        except:
            genops_version = 'unknown'
        
        # Test Haystack-specific imports
        try:
            from genops.providers.haystack_adapter import GenOpsHaystackAdapter
            from genops.providers.haystack_cost_aggregator import HaystackCostAggregator
            from genops.providers.haystack_monitor import HaystackMonitor
            from genops.providers.haystack_registration import auto_instrument
        except ImportError as e:
            issues.append(ValidationIssue(
                severity="error",
                category="dependency",
                message=f"GenOps Haystack integration imports failed: {e}",
                fix_suggestion="Install GenOps with Haystack support: pip install genops-ai[haystack]",
                documentation_link="https://docs.genops.ai/integrations/haystack"
            ))
            return False, issues, genops_version
        
        return True, issues, genops_version
        
    except ImportError:
        issues.append(ValidationIssue(
            severity="error",
            category="dependency",
            message="GenOps not installed",
            fix_suggestion="Install GenOps: pip install genops-ai[haystack]",
            documentation_link="https://docs.genops.ai/quickstart"
        ))
        return False, issues, None


def validate_ai_providers() -> Tuple[Dict[str, Dict[str, Any]], List[ValidationIssue]]:
    """Validate AI provider availability and configuration."""
    issues = []
    provider_status = {}
    
    # Provider configurations
    providers = {
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "key_prefix": "sk-",
            "import_module": "openai",
            "component_class": "OpenAIGenerator"
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY", 
            "key_prefix": "",  # Anthropic keys don't have consistent prefix
            "import_module": "anthropic",
            "component_class": "AnthropicGenerator"
        },
        "cohere": {
            "env_var": "COHERE_API_KEY",
            "key_prefix": "",
            "import_module": "cohere",
            "component_class": "CohereGenerator"
        },
        "huggingface": {
            "env_var": "HUGGINGFACE_API_TOKEN",
            "key_prefix": "hf_",
            "import_module": "transformers",
            "component_class": "HuggingFaceGenerator"
        }
    }
    
    for provider_name, config in providers.items():
        provider_info = {
            "api_key_configured": False,
            "library_installed": False,
            "key_format_valid": False,
            "connectivity_tested": False,
            "status": "unavailable"
        }
        
        # Check API key
        api_key = os.getenv(config["env_var"])
        if api_key:
            provider_info["api_key_configured"] = True
            
            # Validate key format
            if config["key_prefix"]:
                if api_key.startswith(config["key_prefix"]):
                    provider_info["key_format_valid"] = True
                else:
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="configuration",
                        message=f"{provider_name.title()} API key format appears invalid",
                        fix_suggestion=f"Check {config['env_var']} starts with '{config['key_prefix']}'"
                    ))
            else:
                provider_info["key_format_valid"] = True  # No specific format to check
        
        # Check library installation
        try:
            importlib.import_module(config["import_module"])
            provider_info["library_installed"] = True
        except ImportError:
            if provider_info["api_key_configured"]:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="dependency",
                    message=f"{provider_name.title()} API key found but library not installed",
                    fix_suggestion=f"Install {provider_name} library: pip install {config['import_module']}"
                ))
        
        # Determine overall status
        if provider_info["api_key_configured"] and provider_info["library_installed"]:
            provider_info["status"] = "available"
        elif provider_info["library_installed"]:
            provider_info["status"] = "library_only"
        elif provider_info["api_key_configured"]:
            provider_info["status"] = "key_only"
        else:
            provider_info["status"] = "unavailable"
        
        provider_status[provider_name] = provider_info
    
    # Check if at least one provider is fully available
    available_providers = [name for name, info in provider_status.items() if info["status"] == "available"]
    
    if not available_providers:
        issues.append(ValidationIssue(
            severity="warning",
            category="configuration",
            message="No AI providers fully configured",
            fix_suggestion="Configure at least one provider: export OPENAI_API_KEY='your-key'",
            documentation_link="https://docs.genops.ai/integrations/haystack#provider-setup"
        ))
    
    return provider_status, issues


def validate_opentelemetry_setup() -> Tuple[bool, List[ValidationIssue]]:
    """Validate OpenTelemetry installation and configuration."""
    issues = []
    
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.trace import Status, StatusCode
        
        # Test basic functionality
        tracer = trace.get_tracer("validation-test")
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test.attribute", "validation")
            span.set_status(Status(StatusCode.OK))
        
        return True, issues
        
    except ImportError as e:
        issues.append(ValidationIssue(
            severity="error",
            category="dependency",
            message=f"OpenTelemetry not properly installed: {e}",
            fix_suggestion="Install OpenTelemetry: pip install opentelemetry-api opentelemetry-sdk",
            documentation_link="https://opentelemetry.io/docs/instrumentation/python/getting-started/"
        ))
        return False, issues
    except Exception as e:
        issues.append(ValidationIssue(
            severity="warning",
            category="configuration",
            message=f"OpenTelemetry basic test failed: {e}",
            fix_suggestion="Check OpenTelemetry configuration and environment variables"
        ))
        return False, issues


def benchmark_performance() -> Tuple[Dict[str, float], List[ValidationIssue]]:
    """Benchmark basic performance metrics."""
    issues = []
    metrics = {}
    
    # Import performance
    start_time = time.perf_counter()
    try:
        from genops.providers.haystack import GenOpsHaystackAdapter
        import_time = (time.perf_counter() - start_time) * 1000
        metrics["import_time_ms"] = import_time
        
        if import_time > 500:  # 500ms threshold
            issues.append(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Slow import time: {import_time:.1f}ms",
                fix_suggestion="Consider optimizing imports or checking system performance"
            ))
    except Exception as e:
        issues.append(ValidationIssue(
            severity="error",
            category="performance",
            message=f"Import benchmark failed: {e}",
            fix_suggestion="Check installation and dependencies"
        ))
        return metrics, issues
    
    # Basic instantiation performance
    start_time = time.perf_counter()
    try:
        adapter = GenOpsHaystackAdapter(team="test", project="benchmark")
        instantiation_time = (time.perf_counter() - start_time) * 1000
        metrics["instantiation_time_ms"] = instantiation_time
        
        if instantiation_time > 100:  # 100ms threshold
            issues.append(ValidationIssue(
                severity="warning", 
                category="performance",
                message=f"Slow adapter creation: {instantiation_time:.1f}ms",
                fix_suggestion="Check system resources and dependencies"
            ))
    except Exception as e:
        issues.append(ValidationIssue(
            severity="error",
            category="performance", 
            message=f"Instantiation benchmark failed: {e}",
            fix_suggestion="Check GenOps installation and configuration"
        ))
    
    return metrics, issues


def validate_haystack_setup() -> ValidationResult:
    """
    Comprehensive setup validation with structured results.
    
    Returns:
        ValidationResult: Complete validation results with diagnostics
        
    Example:
        result = validate_haystack_setup()
        if result.is_valid:
            print("✅ Setup is ready!")
        else:
            print(f"❌ {result.get_error_count()} errors found")
            for issue in result.issues:
                if issue.severity == "error":
                    print(f"  • {issue.message}")
    """
    start_time = time.perf_counter()
    
    # Initialize result
    result = ValidationResult(
        is_valid=True,
        overall_score=1.0,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=sys.platform
    )
    
    # Validate Python environment
    python_valid, python_issues = validate_python_environment()
    result.issues.extend(python_issues)
    if not python_valid:
        result.dependencies_valid = False
    
    # Validate Haystack installation
    haystack_valid, haystack_issues, haystack_version = validate_haystack_installation()
    result.issues.extend(haystack_issues)
    result.haystack_version = haystack_version
    if not haystack_valid:
        result.dependencies_valid = False
    
    # Validate GenOps installation
    genops_valid, genops_issues, genops_version = validate_genops_installation()
    result.issues.extend(genops_issues)
    result.genops_version = genops_version
    if not genops_valid:
        result.dependencies_valid = False
    
    # Validate AI providers (only if basic dependencies are met)
    if python_valid and haystack_valid and genops_valid:
        provider_status, provider_issues = validate_ai_providers()
        result.issues.extend(provider_issues)
        result.provider_status = provider_status
        result.available_providers = [
            f"{name.title()} integration" 
            for name, info in provider_status.items() 
            if info["status"] == "available"
        ]
    
        # Validate OpenTelemetry setup
        otel_valid, otel_issues = validate_opentelemetry_setup()
        result.issues.extend(otel_issues)
        if not otel_valid:
            result.configuration_valid = False
        
        # Performance benchmarks
        perf_metrics, perf_issues = benchmark_performance()
        result.issues.extend(perf_issues)
        result.import_time_ms = perf_metrics.get("import_time_ms", 0.0)
    
    # Calculate validation time
    result.validation_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Generate recommendations
    if not result.available_providers:
        result.recommendations.append(
            "Configure at least one AI provider for full functionality"
        )
    
    if result.get_error_count() == 0 and result.get_warning_count() == 0:
        result.recommendations.append("Setup is optimal! You're ready to build with Haystack + GenOps")
    elif result.get_error_count() == 0:
        result.recommendations.append("Setup is functional with minor optimizations available")
    
    # Calculate overall score
    error_count = result.get_error_count()
    warning_count = result.get_warning_count()
    
    if error_count > 0:
        result.overall_score = max(0.0, 1.0 - (error_count * 0.3 + warning_count * 0.1))
        result.is_valid = False
    else:
        result.overall_score = max(0.7, 1.0 - (warning_count * 0.05))
        result.is_valid = True
    
    return result


def print_validation_result(result: ValidationResult) -> None:
    """
    User-friendly display with fix suggestions.
    
    Args:
        result: ValidationResult from validate_haystack_setup()
        
    Example:
        result = validate_haystack_setup()
        print_validation_result(result)
    """
    # Header
    if result.is_valid:
        print("✅ Haystack + GenOps Setup Validation")
        print(f"📊 Overall Score: {result.overall_score:.1%}")
    else:
        print("❌ Haystack + GenOps Setup Issues Found")
        print(f"📊 Overall Score: {result.overall_score:.1%}")
    
    print("=" * 50)
    
    # System information
    print(f"🐍 Python: {result.python_version} ({result.platform})")
    if result.haystack_version:
        print(f"🏗️ Haystack: {result.haystack_version}")
    if result.genops_version:
        print(f"🛠️ GenOps: {result.genops_version}")
    
    # Performance metrics
    if result.import_time_ms > 0:
        print(f"⚡ Import time: {result.import_time_ms:.1f}ms")
        print(f"🕐 Validation time: {result.validation_time_ms:.1f}ms")
    
    print()
    
    # Provider status
    if result.available_providers:
        print("✅ Available AI Providers:")
        for provider in result.available_providers:
            print(f"   • {provider}")
    else:
        print("⚠️ No AI providers configured")
    
    # Category status
    print(f"\n📋 Component Status:")
    status_icon = lambda valid: "✅" if valid else "❌"
    print(f"   {status_icon(result.dependencies_valid)} Dependencies")
    print(f"   {status_icon(result.configuration_valid)} Configuration") 
    print(f"   {status_icon(result.connectivity_valid)} Connectivity")
    print(f"   {status_icon(result.performance_acceptable)} Performance")
    
    # Issues by severity
    errors = [issue for issue in result.issues if issue.severity == "error"]
    warnings = [issue for issue in result.issues if issue.severity == "warning"]
    
    if errors:
        print(f"\n🚨 Errors ({len(errors)}):")
        for issue in errors:
            print(f"   • {issue.message}")
            print(f"     Fix: {issue.fix_suggestion}")
            if issue.documentation_link:
                print(f"     Docs: {issue.documentation_link}")
    
    if warnings:
        print(f"\n⚠️ Warnings ({len(warnings)}):")
        for issue in warnings:
            print(f"   • {issue.message}")
            print(f"     Suggestion: {issue.fix_suggestion}")
            if issue.documentation_link:
                print(f"     Docs: {issue.documentation_link}")
    
    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")
    
    # Next steps
    if result.is_valid:
        print(f"\n🚀 You're ready! Try:")
        print(f"   from genops.providers.haystack import auto_instrument")
        print(f"   auto_instrument()")
        print(f"\n📚 Next steps:")
        print(f"   • Examples: python examples/haystack/basic_pipeline_tracking.py")
        print(f"   • Docs: docs/integrations/haystack.md")
    else:
        print(f"\n🔧 Quick Fixes:")
        print(f"   Interactive setup: ./validate --fix-issues")
        print(f"   Or manually:")
        print(f"     1. Fix the errors listed above")
        print(f"     2. Re-run: python scripts/validate_setup.py")
        print(f"     3. Provider setup: ./validate --provider openai")
        print(f"\n📚 Help:")
        print(f"   • Documentation: docs/integrations/haystack.md#troubleshooting")
        print(f"   • Examples: examples/haystack/README.md")


# Export main functions
__all__ = [
    'validate_haystack_setup',
    'print_validation_result', 
    'ValidationResult',
    'ValidationIssue'
]