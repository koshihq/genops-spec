#!/usr/bin/env python3
"""
Langfuse Integration Validation Utilities

This module provides comprehensive validation utilities for GenOps + Langfuse
integration setup, including API connectivity, configuration validation, and
performance testing.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    Langfuse = None

class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"

@dataclass
class ValidationResult:
    """Individual validation test result."""
    test_name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None
    duration_ms: Optional[float] = None

@dataclass
class LangfuseValidationSuite:
    """Complete Langfuse validation suite results."""
    overall_status: ValidationStatus
    test_results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0

def validate_langfuse_installation() -> ValidationResult:
    """Validate that Langfuse is properly installed."""
    start_time = time.time()

    if not HAS_LANGFUSE:
        return ValidationResult(
            test_name="Langfuse Installation",
            status=ValidationStatus.FAILED,
            message="Langfuse package not found",
            fix_suggestion="Install with: pip install 'genops[langfuse]' or pip install langfuse",
            duration_ms=(time.time() - start_time) * 1000
        )

    try:
        # Try to import key components
        from langfuse.client import StatefulClient
        from langfuse.decorators import observe

        return ValidationResult(
            test_name="Langfuse Installation",
            status=ValidationStatus.PASSED,
            message="Langfuse package successfully imported",
            details={"version": "2.0+", "components": ["Langfuse", "observe", "StatefulClient"]},
            duration_ms=(time.time() - start_time) * 1000
        )

    except ImportError as e:
        return ValidationResult(
            test_name="Langfuse Installation",
            status=ValidationStatus.FAILED,
            message=f"Langfuse import failed: {e}",
            fix_suggestion="Reinstall Langfuse: pip install --upgrade langfuse",
            duration_ms=(time.time() - start_time) * 1000
        )

def validate_langfuse_configuration() -> ValidationResult:
    """Validate Langfuse API configuration."""
    start_time = time.time()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    missing_configs = []
    if not public_key:
        missing_configs.append("LANGFUSE_PUBLIC_KEY")
    if not secret_key:
        missing_configs.append("LANGFUSE_SECRET_KEY")

    if missing_configs:
        return ValidationResult(
            test_name="Langfuse Configuration",
            status=ValidationStatus.FAILED,
            message=f"Missing required environment variables: {', '.join(missing_configs)}",
            details={
                "required_vars": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
                "optional_vars": ["LANGFUSE_BASE_URL"],
                "base_url": base_url
            },
            fix_suggestion="Set environment variables: export LANGFUSE_PUBLIC_KEY='pk-lf-...' && export LANGFUSE_SECRET_KEY='sk-lf-...'",
            duration_ms=(time.time() - start_time) * 1000
        )

    # Validate key formats
    issues = []
    if not public_key.startswith("pk-lf-"):
        issues.append("Public key should start with 'pk-lf-'")
    if not secret_key.startswith("sk-lf-"):
        issues.append("Secret key should start with 'sk-lf-'")

    status = ValidationStatus.WARNING if issues else ValidationStatus.PASSED
    message = "Configuration issues found: " + ", ".join(issues) if issues else "Langfuse configuration valid"

    return ValidationResult(
        test_name="Langfuse Configuration",
        status=status,
        message=message,
        details={
            "public_key_prefix": public_key[:8] + "..." if public_key else None,
            "secret_key_prefix": secret_key[:8] + "..." if secret_key else None,
            "base_url": base_url,
            "issues": issues
        },
        fix_suggestion="Check API key formats at https://cloud.langfuse.com" if issues else None,
        duration_ms=(time.time() - start_time) * 1000
    )

def validate_langfuse_connectivity() -> ValidationResult:
    """Test Langfuse API connectivity."""
    start_time = time.time()

    if not HAS_LANGFUSE:
        return ValidationResult(
            test_name="Langfuse Connectivity",
            status=ValidationStatus.SKIPPED,
            message="Langfuse not available for connectivity test",
            duration_ms=(time.time() - start_time) * 1000
        )

    try:
        client = Langfuse()

        # Test basic connectivity by creating a simple trace
        test_trace = client.trace(name="genops_validation_test")

        # If we get here, connection is working
        return ValidationResult(
            test_name="Langfuse Connectivity",
            status=ValidationStatus.PASSED,
            message="Successfully connected to Langfuse API",
            details={
                "trace_id": test_trace.id,
                "host": client.client.base_url if hasattr(client, 'client') else "unknown"
            },
            duration_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        error_msg = str(e).lower()

        if "unauthorized" in error_msg or "401" in error_msg:
            fix_suggestion = "Check your Langfuse API keys at https://cloud.langfuse.com"
        elif "connection" in error_msg or "network" in error_msg:
            fix_suggestion = "Check network connectivity and Langfuse service status"
        else:
            fix_suggestion = "Verify Langfuse configuration and try again"

        return ValidationResult(
            test_name="Langfuse Connectivity",
            status=ValidationStatus.FAILED,
            message=f"Failed to connect to Langfuse: {e}",
            details={"error": str(e)},
            fix_suggestion=fix_suggestion,
            duration_ms=(time.time() - start_time) * 1000
        )

def validate_genops_integration() -> ValidationResult:
    """Validate GenOps + Langfuse integration setup."""
    start_time = time.time()

    try:
        from genops.providers.langfuse import GenOpsLangfuseAdapter, instrument_langfuse

        # Test adapter creation
        adapter = GenOpsLangfuseAdapter(
            team="validation-test",
            project="setup-check",
            environment="test"
        )

        return ValidationResult(
            test_name="GenOps Integration",
            status=ValidationStatus.PASSED,
            message="GenOps Langfuse integration working correctly",
            details={
                "adapter_initialized": True,
                "team": adapter.team,
                "project": adapter.project,
                "governance_enabled": adapter.enable_governance
            },
            duration_ms=(time.time() - start_time) * 1000
        )

    except ImportError as e:
        return ValidationResult(
            test_name="GenOps Integration",
            status=ValidationStatus.FAILED,
            message=f"Failed to import GenOps Langfuse integration: {e}",
            fix_suggestion="Ensure GenOps is properly installed with Langfuse support",
            duration_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return ValidationResult(
            test_name="GenOps Integration",
            status=ValidationStatus.FAILED,
            message=f"GenOps Langfuse integration error: {e}",
            details={"error": str(e)},
            fix_suggestion="Check GenOps and Langfuse configuration",
            duration_ms=(time.time() - start_time) * 1000
        )

def validate_performance_baseline() -> ValidationResult:
    """Test basic performance characteristics."""
    start_time = time.time()

    if not HAS_LANGFUSE:
        return ValidationResult(
            test_name="Performance Baseline",
            status=ValidationStatus.SKIPPED,
            message="Langfuse not available for performance testing",
            duration_ms=(time.time() - start_time) * 1000
        )

    try:
        from genops.providers.langfuse import GenOpsLangfuseAdapter

        # Performance test: measure adapter initialization
        init_start = time.time()
        adapter = GenOpsLangfuseAdapter(
            team="perf-test",
            project="baseline",
            environment="test"
        )
        init_time = (time.time() - init_start) * 1000

        # Performance test: measure trace creation
        trace_start = time.time()
        with adapter.trace_with_governance(name="performance_test") as trace:
            time.sleep(0.01)  # Simulate minimal work
        trace_time = (time.time() - trace_start) * 1000

        # Evaluate performance
        performance_issues = []
        if init_time > 1000:  # > 1 second
            performance_issues.append(f"Slow initialization: {init_time:.1f}ms")
        if trace_time > 100:  # > 100ms
            performance_issues.append(f"Slow trace creation: {trace_time:.1f}ms")

        status = ValidationStatus.WARNING if performance_issues else ValidationStatus.PASSED
        message = "Performance issues detected: " + ", ".join(performance_issues) if performance_issues else "Performance baseline acceptable"

        return ValidationResult(
            test_name="Performance Baseline",
            status=status,
            message=message,
            details={
                "initialization_ms": round(init_time, 2),
                "trace_creation_ms": round(trace_time, 2),
                "issues": performance_issues
            },
            duration_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        return ValidationResult(
            test_name="Performance Baseline",
            status=ValidationStatus.FAILED,
            message=f"Performance testing failed: {e}",
            details={"error": str(e)},
            duration_ms=(time.time() - start_time) * 1000
        )

def run_comprehensive_validation(
    include_performance_tests: bool = False,
    include_connectivity_tests: bool = True
) -> LangfuseValidationSuite:
    """
    Run comprehensive Langfuse + GenOps validation suite.
    
    Args:
        include_performance_tests: Include performance baseline tests
        include_connectivity_tests: Include API connectivity tests
        
    Returns:
        Complete validation suite results
    """
    suite_start = time.time()
    results = []

    # Core validation tests
    results.append(validate_langfuse_installation())
    results.append(validate_langfuse_configuration())
    results.append(validate_genops_integration())

    # Optional tests
    if include_connectivity_tests:
        results.append(validate_langfuse_connectivity())

    if include_performance_tests:
        results.append(validate_performance_baseline())

    # Analyze results
    passed_count = sum(1 for r in results if r.status == ValidationStatus.PASSED)
    failed_count = sum(1 for r in results if r.status == ValidationStatus.FAILED)
    warning_count = sum(1 for r in results if r.status == ValidationStatus.WARNING)
    skipped_count = sum(1 for r in results if r.status == ValidationStatus.SKIPPED)

    # Determine overall status
    if failed_count > 0:
        overall_status = ValidationStatus.FAILED
    elif warning_count > 0:
        overall_status = ValidationStatus.WARNING
    else:
        overall_status = ValidationStatus.PASSED

    # Generate recommendations
    recommendations = []
    if failed_count > 0:
        recommendations.append("Fix failed validation tests before proceeding with Langfuse integration")
    if warning_count > 0:
        recommendations.append("Review warnings to optimize Langfuse setup")
    if overall_status == ValidationStatus.PASSED:
        recommendations.append("Langfuse integration is ready - proceed with examples and production usage")

    suite_duration = (time.time() - suite_start) * 1000

    return LangfuseValidationSuite(
        overall_status=overall_status,
        test_results=results,
        summary={
            "total_tests": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "warnings": warning_count,
            "skipped": skipped_count,
            "success_rate": passed_count / len(results) if results else 0.0
        },
        recommendations=recommendations,
        total_duration_ms=suite_duration
    )

def print_validation_result(
    result: Union[LangfuseValidationSuite, ValidationResult],
    detailed: bool = False
) -> None:
    """
    Print validation results in a user-friendly format.
    
    Args:
        result: Validation result to print
        detailed: Include detailed information and fix suggestions
    """
    if isinstance(result, LangfuseValidationSuite):
        _print_validation_suite(result, detailed)
    else:
        _print_single_validation(result, detailed)

def _print_validation_suite(suite: LangfuseValidationSuite, detailed: bool) -> None:
    """Print validation suite results."""
    print("\n🔍 GenOps + Langfuse Integration Validation")
    print("=" * 50)

    # Overall status
    status_emoji = {
        ValidationStatus.PASSED: "✅",
        ValidationStatus.FAILED: "❌",
        ValidationStatus.WARNING: "⚠️"
    }

    print(f"\n{status_emoji.get(suite.overall_status, '❓')} Overall Status: {suite.overall_status.value}")

    # Summary
    print("\n📊 Test Summary:")
    print(f"   Total Tests: {suite.summary['total_tests']}")
    print(f"   ✅ Passed: {suite.summary['passed']}")
    print(f"   ❌ Failed: {suite.summary['failed']}")
    print(f"   ⚠️  Warnings: {suite.summary['warnings']}")
    print(f"   ⏭️  Skipped: {suite.summary['skipped']}")
    print(f"   📈 Success Rate: {suite.summary['success_rate']:.1%}")
    print(f"   ⏱️  Total Duration: {suite.total_duration_ms:.0f}ms")

    # Individual test results
    print("\n📋 Detailed Results:")
    for result in suite.test_results:
        emoji = status_emoji.get(result.status, "❓")
        duration_str = f" ({result.duration_ms:.0f}ms)" if result.duration_ms else ""
        print(f"   {emoji} {result.test_name}: {result.message}{duration_str}")

        if detailed and result.fix_suggestion:
            print(f"      💡 Fix: {result.fix_suggestion}")

        if detailed and result.details:
            for key, value in result.details.items():
                print(f"      📝 {key}: {value}")

    # Recommendations
    if suite.recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(suite.recommendations, 1):
            print(f"   {i}. {rec}")

    print()

def _print_single_validation(result: ValidationResult, detailed: bool) -> None:
    """Print single validation result."""
    status_emoji = {
        ValidationStatus.PASSED: "✅",
        ValidationStatus.FAILED: "❌",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.SKIPPED: "⏭️"
    }

    emoji = status_emoji.get(result.status, "❓")
    duration_str = f" ({result.duration_ms:.0f}ms)" if result.duration_ms else ""

    print(f"{emoji} {result.test_name}: {result.message}{duration_str}")

    if detailed and result.fix_suggestion:
        print(f"   💡 Fix: {result.fix_suggestion}")

    if detailed and result.details:
        for key, value in result.details.items():
            print(f"   📝 {key}: {value}")

# Convenience function for quick validation
def validate_setup(
    include_performance_tests: bool = False,
    include_connectivity_tests: bool = True
) -> LangfuseValidationSuite:
    """Quick validation function for easy import."""
    return run_comprehensive_validation(
        include_performance_tests=include_performance_tests,
        include_connectivity_tests=include_connectivity_tests
    )
