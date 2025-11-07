#!/usr/bin/env python3
"""
GenOps Traceloop + OpenLLMetry Validation Utilities

This module provides comprehensive validation utilities for Traceloop + OpenLLMetry + GenOps
integration, ensuring proper setup, connectivity, and governance configuration.

The validation covers:
- OpenLLMetry framework availability and configuration
- Traceloop SDK availability (optional commercial platform)
- AI provider API keys and connectivity
- GenOps governance configuration
- Performance baseline testing
- Integration health checks

Usage:
    from genops.providers.traceloop_validation import validate_setup, print_validation_result
    
    result = validate_setup()
    print_validation_result(result, detailed=True)
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status levels."""
    PASSED = "PASSED"
    WARNING = "WARNING"  
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    DEPENDENCIES = "dependencies"
    CONFIGURATION = "configuration"
    CONNECTIVITY = "connectivity"
    GOVERNANCE = "governance"
    PERFORMANCE = "performance"


@dataclass
class ValidationResult:
    """Individual validation check result."""
    category: ValidationCategory
    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ValidationSummary:
    """Overall validation summary."""
    overall_status: ValidationStatus
    total_checks: int
    passed_checks: int
    warning_checks: int
    failed_checks: int
    skipped_checks: int
    results: List[ValidationResult] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the summary."""
        self.results.append(result)
        self.total_checks += 1
        self.total_execution_time_ms += result.execution_time_ms
        
        if result.status == ValidationStatus.PASSED:
            self.passed_checks += 1
        elif result.status == ValidationStatus.WARNING:
            self.warning_checks += 1
        elif result.status == ValidationStatus.FAILED:
            self.failed_checks += 1
        elif result.status == ValidationStatus.SKIPPED:
            self.skipped_checks += 1
            
        # Update overall status
        if self.failed_checks > 0:
            self.overall_status = ValidationStatus.FAILED
        elif self.warning_checks > 0 and self.overall_status != ValidationStatus.FAILED:
            self.overall_status = ValidationStatus.WARNING
        elif self.passed_checks > 0 and self.warning_checks == 0 and self.failed_checks == 0:
            self.overall_status = ValidationStatus.PASSED


def validate_dependencies() -> List[ValidationResult]:
    """Validate required dependencies are available."""
    results = []
    
    # Check Python version
    start_time = time.time()
    python_version = sys.version_info
    if python_version >= (3, 8):
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="python_version",
            status=ValidationStatus.PASSED,
            message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    else:
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="python_version", 
            status=ValidationStatus.FAILED,
            message=f"Python {python_version.major}.{python_version.minor} is too old",
            fix_suggestion="Upgrade to Python 3.8 or newer",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Check OpenLLMetry availability
    start_time = time.time()
    try:
        import openllmetry
        version = getattr(openllmetry, '__version__', 'unknown')
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="openllmetry_availability",
            status=ValidationStatus.PASSED,
            message=f"OpenLLMetry {version} available",
            details={"version": version},
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    except ImportError as e:
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="openllmetry_availability",
            status=ValidationStatus.FAILED,
            message="OpenLLMetry not available",
            details={"error": str(e)},
            fix_suggestion="Install with: pip install openllmetry",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Check Traceloop SDK availability (optional)
    start_time = time.time()
    try:
        from traceloop.sdk import Traceloop
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="traceloop_sdk_availability",
            status=ValidationStatus.PASSED,
            message="Traceloop SDK available for commercial platform features",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    except ImportError:
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="traceloop_sdk_availability",
            status=ValidationStatus.WARNING,
            message="Traceloop SDK not available (open-source mode only)",
            fix_suggestion="For commercial features: pip install traceloop-sdk",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Check OpenTelemetry availability
    start_time = time.time()
    try:
        from opentelemetry import trace
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="opentelemetry_availability",
            status=ValidationStatus.PASSED,
            message="OpenTelemetry available",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    except ImportError as e:
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="opentelemetry_availability",
            status=ValidationStatus.WARNING,
            message="OpenTelemetry not available",
            details={"error": str(e)},
            fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Check GenOps availability
    start_time = time.time()
    try:
        from genops.providers.traceloop import instrument_traceloop
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="genops_traceloop_integration",
            status=ValidationStatus.PASSED,
            message="GenOps Traceloop integration available",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    except ImportError as e:
        results.append(ValidationResult(
            category=ValidationCategory.DEPENDENCIES,
            check_name="genops_traceloop_integration",
            status=ValidationStatus.FAILED,
            message="GenOps Traceloop integration not available",
            details={"error": str(e)},
            fix_suggestion="Install with: pip install genops[traceloop]",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    return results


def validate_configuration() -> List[ValidationResult]:
    """Validate configuration and environment variables."""
    results = []
    
    # Check AI provider API keys
    providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "Groq": "GROQ_API_KEY"
    }
    
    provider_count = 0
    for provider_name, env_var in providers.items():
        start_time = time.time()
        api_key = os.getenv(env_var)
        
        if api_key:
            provider_count += 1
            results.append(ValidationResult(
                category=ValidationCategory.CONFIGURATION,
                check_name=f"{provider_name.lower()}_api_key",
                status=ValidationStatus.PASSED,
                message=f"{provider_name} API key configured",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        else:
            results.append(ValidationResult(
                category=ValidationCategory.CONFIGURATION,
                check_name=f"{provider_name.lower()}_api_key",
                status=ValidationStatus.SKIPPED,
                message=f"{provider_name} API key not configured",
                fix_suggestion=f"Set {env_var} environment variable",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
    
    # Check if at least one provider is configured
    start_time = time.time()
    if provider_count > 0:
        results.append(ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="ai_provider_available",
            status=ValidationStatus.PASSED,
            message=f"{provider_count} AI provider(s) configured",
            details={"provider_count": provider_count},
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    else:
        results.append(ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="ai_provider_available",
            status=ValidationStatus.FAILED,
            message="No AI providers configured",
            fix_suggestion="Set at least one provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Check Traceloop platform configuration (optional)
    start_time = time.time()
    traceloop_api_key = os.getenv('TRACELOOP_API_KEY')
    traceloop_base_url = os.getenv('TRACELOOP_BASE_URL', 'https://app.traceloop.com')
    
    if traceloop_api_key:
        results.append(ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="traceloop_platform_config",
            status=ValidationStatus.PASSED,
            message="Traceloop platform configured",
            details={"base_url": traceloop_base_url},
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    else:
        results.append(ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="traceloop_platform_config",
            status=ValidationStatus.SKIPPED,
            message="Traceloop platform not configured (open-source mode)",
            fix_suggestion="For commercial features, set TRACELOOP_API_KEY",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Check GenOps configuration
    start_time = time.time()
    genops_team = os.getenv('GENOPS_TEAM')
    genops_project = os.getenv('GENOPS_PROJECT')
    
    if genops_team and genops_project:
        results.append(ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="genops_governance_config",
            status=ValidationStatus.PASSED,
            message="GenOps governance configuration found",
            details={"team": genops_team, "project": genops_project},
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    else:
        results.append(ValidationResult(
            category=ValidationCategory.CONFIGURATION,
            check_name="genops_governance_config",
            status=ValidationStatus.WARNING,
            message="GenOps governance configuration incomplete",
            fix_suggestion="Set GENOPS_TEAM and GENOPS_PROJECT environment variables",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    return results


def validate_connectivity() -> List[ValidationResult]:
    """Validate connectivity to external services."""
    results = []
    
    # Test OpenAI connectivity (if configured)
    if os.getenv('OPENAI_API_KEY'):
        start_time = time.time()
        try:
            import openai
            client = openai.OpenAI()
            
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            
            results.append(ValidationResult(
                category=ValidationCategory.CONNECTIVITY,
                check_name="openai_connectivity",
                status=ValidationStatus.PASSED,
                message="OpenAI API connectivity verified",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                category=ValidationCategory.CONNECTIVITY,
                check_name="openai_connectivity",
                status=ValidationStatus.FAILED,
                message="OpenAI API connectivity failed",
                details={"error": str(e)},
                fix_suggestion="Check API key and network connectivity",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
    
    # Test Anthropic connectivity (if configured)
    if os.getenv('ANTHROPIC_API_KEY'):
        start_time = time.time()
        try:
            import anthropic
            client = anthropic.Anthropic()
            
            # Simple test call
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            
            results.append(ValidationResult(
                category=ValidationCategory.CONNECTIVITY,
                check_name="anthropic_connectivity",
                status=ValidationStatus.PASSED,
                message="Anthropic API connectivity verified",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                category=ValidationCategory.CONNECTIVITY,
                check_name="anthropic_connectivity",
                status=ValidationStatus.WARNING,
                message="Anthropic API connectivity failed",
                details={"error": str(e)},
                fix_suggestion="Check API key and network connectivity",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
    
    return results


def validate_governance() -> List[ValidationResult]:
    """Validate GenOps governance functionality."""
    results = []
    
    # Test GenOps adapter creation
    start_time = time.time()
    try:
        from genops.providers.traceloop import instrument_traceloop
        
        adapter = instrument_traceloop(
            team="validation-test",
            project="governance-check",
            environment="test"
        )
        
        results.append(ValidationResult(
            category=ValidationCategory.GOVERNANCE,
            check_name="genops_adapter_creation",
            status=ValidationStatus.PASSED,
            message="GenOps adapter created successfully",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
        
    except Exception as e:
        results.append(ValidationResult(
            category=ValidationCategory.GOVERNANCE,
            check_name="genops_adapter_creation",
            status=ValidationStatus.FAILED,
            message="GenOps adapter creation failed",
            details={"error": str(e)},
            fix_suggestion="Check GenOps installation and configuration",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    # Test auto-instrumentation
    start_time = time.time()
    try:
        from genops.providers.traceloop import auto_instrument
        
        # Test auto-instrumentation (non-destructive)
        results.append(ValidationResult(
            category=ValidationCategory.GOVERNANCE,
            check_name="auto_instrumentation_available",
            status=ValidationStatus.PASSED,
            message="Auto-instrumentation functionality available",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
        
    except Exception as e:
        results.append(ValidationResult(
            category=ValidationCategory.GOVERNANCE,
            check_name="auto_instrumentation_available",
            status=ValidationStatus.FAILED,
            message="Auto-instrumentation functionality failed",
            details={"error": str(e)},
            fix_suggestion="Check OpenLLMetry installation and compatibility",
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    return results


def validate_performance() -> List[ValidationResult]:
    """Validate performance baseline."""
    results = []
    
    # Test governance overhead
    start_time = time.time()
    try:
        from genops.providers.traceloop import instrument_traceloop
        
        adapter = instrument_traceloop(
            team="perf-test",
            project="baseline"
        )
        
        # Measure governance overhead
        governance_start = time.time()
        with adapter.track_operation(
            operation_type="performance_test",
            operation_name="baseline_test"
        ) as span:
            # Simulate minimal operation
            time.sleep(0.001)
            span.update_cost(0.001)
            
        governance_time = (time.time() - governance_start) * 1000
        
        if governance_time < 50:  # Less than 50ms overhead
            results.append(ValidationResult(
                category=ValidationCategory.PERFORMANCE,
                check_name="governance_overhead",
                status=ValidationStatus.PASSED,
                message=f"Governance overhead: {governance_time:.2f}ms",
                details={"overhead_ms": governance_time},
                execution_time_ms=(time.time() - start_time) * 1000
            ))
        else:
            results.append(ValidationResult(
                category=ValidationCategory.PERFORMANCE,
                check_name="governance_overhead",
                status=ValidationStatus.WARNING,
                message=f"High governance overhead: {governance_time:.2f}ms",
                details={"overhead_ms": governance_time},
                fix_suggestion="Check system performance and configuration",
                execution_time_ms=(time.time() - start_time) * 1000
            ))
            
    except Exception as e:
        results.append(ValidationResult(
            category=ValidationCategory.PERFORMANCE,
            check_name="governance_overhead",
            status=ValidationStatus.FAILED,
            message="Performance test failed",
            details={"error": str(e)},
            execution_time_ms=(time.time() - start_time) * 1000
        ))
    
    return results


def validate_setup(
    include_connectivity_tests: bool = True,
    include_performance_tests: bool = False
) -> ValidationSummary:
    """
    Run comprehensive validation of Traceloop + OpenLLMetry + GenOps setup.
    
    Args:
        include_connectivity_tests: Whether to test external API connectivity
        include_performance_tests: Whether to run performance baseline tests
        
    Returns:
        ValidationSummary with all check results
    """
    summary = ValidationSummary(
        overall_status=ValidationStatus.PASSED,
        total_checks=0,
        passed_checks=0,
        warning_checks=0,
        failed_checks=0,
        skipped_checks=0
    )
    
    # Run all validation categories
    all_results = []
    
    # Dependencies validation (always run)
    all_results.extend(validate_dependencies())
    
    # Configuration validation (always run)
    all_results.extend(validate_configuration())
    
    # Connectivity validation (optional)
    if include_connectivity_tests:
        all_results.extend(validate_connectivity())
    
    # Governance validation (always run)
    all_results.extend(validate_governance())
    
    # Performance validation (optional)
    if include_performance_tests:
        all_results.extend(validate_performance())
    
    # Add all results to summary
    for result in all_results:
        summary.add_result(result)
    
    return summary


def print_validation_result(summary: ValidationSummary, detailed: bool = False):
    """
    Print validation results in a user-friendly format.
    
    Args:
        summary: ValidationSummary to display
        detailed: Whether to show detailed information for each check
    """
    # Header
    print("\n🔍 Traceloop + OpenLLMetry + GenOps Validation Results")
    print("=" * 55)
    
    # Overall status
    status_symbols = {
        ValidationStatus.PASSED: "✅",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.FAILED: "❌",
        ValidationStatus.SKIPPED: "⏸️"
    }
    
    symbol = status_symbols.get(summary.overall_status, "❓")
    print(f"\n{symbol} Overall Status: {summary.overall_status.value}")
    
    # Summary stats
    print(f"\n📊 Check Summary:")
    print(f"   Total checks: {summary.total_checks}")
    print(f"   ✅ Passed: {summary.passed_checks}")
    print(f"   ⚠️  Warnings: {summary.warning_checks}")
    print(f"   ❌ Failed: {summary.failed_checks}")
    print(f"   ⏸️  Skipped: {summary.skipped_checks}")
    print(f"   ⏱️  Total time: {summary.total_execution_time_ms:.1f}ms")
    
    # Results by category
    if detailed:
        categories = {}
        for result in summary.results:
            category = result.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category_name, results in categories.items():
            print(f"\n📋 {category_name.title()} Checks:")
            print("-" * 30)
            
            for result in results:
                symbol = status_symbols.get(result.status, "❓")
                print(f"   {symbol} {result.check_name}: {result.message}")
                
                if result.status == ValidationStatus.FAILED and result.fix_suggestion:
                    print(f"      💡 Fix: {result.fix_suggestion}")
                
                if result.details and len(str(result.details)) < 100:
                    print(f"      ℹ️  Details: {result.details}")
                    
                if detailed and result.execution_time_ms > 10:
                    print(f"      ⏱️  Time: {result.execution_time_ms:.1f}ms")
    
    # Next steps
    if summary.overall_status == ValidationStatus.PASSED:
        print(f"\n🎉 Validation Complete - Ready to use!")
        print("   Next steps:")
        print("   • Run example scripts to see governance in action")
        print("   • Configure team and project settings")
        print("   • Explore advanced features and commercial platform")
    elif summary.overall_status == ValidationStatus.WARNING:
        print(f"\n⚠️  Validation Complete with Warnings")
        print("   You can proceed, but some features may not be available.")
        print("   Review warnings above and apply suggested fixes.")
    else:
        print(f"\n❌ Validation Failed")
        print("   Please fix the failed checks above before proceeding.")
    
    print()


# Example usage for testing
if __name__ == "__main__":
    result = validate_setup(include_performance_tests=True)
    print_validation_result(result, detailed=True)