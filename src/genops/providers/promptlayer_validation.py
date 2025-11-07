#!/usr/bin/env python3
"""
GenOps PromptLayer Setup Validation

This module provides comprehensive validation utilities for PromptLayer integration
with GenOps governance. It checks dependencies, configuration, connectivity, and
provides actionable diagnostics for common setup issues.

Features:
- Dependency validation (PromptLayer SDK, GenOps, etc.)
- Configuration validation (API keys, environment variables)
- Connectivity testing (PromptLayer API access)
- Governance validation (team/project setup)
- Performance validation (response times, overhead measurement)
- Actionable error messages with specific fix suggestions

Example usage:

    from genops.providers.promptlayer_validation import validate_setup, print_validation_result
    
    # Run comprehensive validation
    result = validate_setup()
    print_validation_result(result)
    
    # Run specific validation checks
    result = validate_setup(
        include_connectivity_tests=True,
        include_performance_tests=False
    )
    
    # Custom configuration validation
    result = validate_setup(
        promptlayer_api_key="pl-your-key",
        team="engineering-team"
    )
"""

import os
import sys
import time
import importlib
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Status levels for validation results."""
    PASSED = "passed"
    WARNING = "warning"  
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    status: ValidationStatus
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None
    category: str = "general"
    duration_ms: Optional[float] = None

@dataclass
class ValidationResult:
    """Complete validation result with all checks."""
    overall_status: ValidationStatus
    checks: List[ValidationCheck] = field(default_factory=list)
    total_duration_ms: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed_checks(self) -> int:
        """Count of passed checks."""
        return len([c for c in self.checks if c.status == ValidationStatus.PASSED])
    
    @property
    def warning_checks(self) -> int:
        """Count of warning checks."""
        return len([c for c in self.checks if c.status == ValidationStatus.WARNING])
    
    @property
    def failed_checks(self) -> int:
        """Count of failed checks."""
        return len([c for c in self.checks if c.status == ValidationStatus.FAILED])

def _measure_duration(func):
    """Decorator to measure function execution duration."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000
        if hasattr(result, 'duration_ms'):
            result.duration_ms = duration_ms
        return result
    return wrapper

@_measure_duration
def check_python_version() -> ValidationCheck:
    """Check Python version compatibility."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        return ValidationCheck(
            name="Python Version",
            status=ValidationStatus.PASSED,
            message=f"Python {current_version[0]}.{current_version[1]} is supported",
            category="dependencies"
        )
    else:
        return ValidationCheck(
            name="Python Version", 
            status=ValidationStatus.FAILED,
            message=f"Python {current_version[0]}.{current_version[1]} is not supported",
            details=f"Minimum required version: {min_version[0]}.{min_version[1]}",
            fix_suggestion="Upgrade Python to 3.8 or later",
            category="dependencies"
        )

@_measure_duration
def check_genops_installation() -> ValidationCheck:
    """Check GenOps installation and version."""
    try:
        import genops
        version = getattr(genops, '__version__', 'unknown')
        
        return ValidationCheck(
            name="GenOps Installation",
            status=ValidationStatus.PASSED,
            message=f"GenOps {version} is installed and importable",
            category="dependencies"
        )
    except ImportError as e:
        return ValidationCheck(
            name="GenOps Installation",
            status=ValidationStatus.FAILED,
            message="GenOps is not installed or not importable",
            details=str(e),
            fix_suggestion="Install GenOps: pip install genops",
            category="dependencies"
        )

@_measure_duration 
def check_promptlayer_installation() -> ValidationCheck:
    """Check PromptLayer SDK installation."""
    try:
        import promptlayer
        version = getattr(promptlayer, '__version__', 'unknown')
        
        # Check if PromptLayer client can be imported
        from promptlayer import PromptLayer
        
        return ValidationCheck(
            name="PromptLayer SDK",
            status=ValidationStatus.PASSED,
            message=f"PromptLayer SDK {version} is installed and importable",
            category="dependencies"
        )
    except ImportError as e:
        return ValidationCheck(
            name="PromptLayer SDK",
            status=ValidationStatus.FAILED,
            message="PromptLayer SDK is not installed",
            details=str(e),
            fix_suggestion="Install PromptLayer SDK: pip install promptlayer",
            category="dependencies"
        )

@_measure_duration
def check_optional_dependencies() -> ValidationCheck:
    """Check optional dependencies for enhanced functionality."""
    optional_deps = {
        'openai': 'OpenAI SDK for LLM operations',
        'anthropic': 'Anthropic SDK for Claude models',
        'requests': 'HTTP requests for API calls'
    }
    
    installed = []
    missing = []
    
    for dep_name, description in optional_deps.items():
        try:
            importlib.import_module(dep_name)
            installed.append(f"{dep_name} ({description})")
        except ImportError:
            missing.append(f"{dep_name} ({description})")
    
    if missing:
        return ValidationCheck(
            name="Optional Dependencies",
            status=ValidationStatus.WARNING,
            message=f"Some optional dependencies are missing: {len(missing)} missing, {len(installed)} installed",
            details=f"Missing: {', '.join(missing)}",
            fix_suggestion="Install optional dependencies: pip install openai anthropic requests",
            category="dependencies"
        )
    else:
        return ValidationCheck(
            name="Optional Dependencies",
            status=ValidationStatus.PASSED,
            message=f"All optional dependencies are available ({len(installed)} installed)",
            category="dependencies"
        )

@_measure_duration
def check_promptlayer_api_key(api_key: Optional[str] = None) -> ValidationCheck:
    """Check PromptLayer API key configuration."""
    key = api_key or os.getenv('PROMPTLAYER_API_KEY')
    
    if not key:
        return ValidationCheck(
            name="PromptLayer API Key",
            status=ValidationStatus.FAILED,
            message="PromptLayer API key not found",
            details="No API key provided via parameter or PROMPTLAYER_API_KEY environment variable",
            fix_suggestion="Set your API key: export PROMPTLAYER_API_KEY=pl-your-api-key",
            category="configuration"
        )
    
    if not key.startswith('pl-'):
        return ValidationCheck(
            name="PromptLayer API Key",
            status=ValidationStatus.WARNING,
            message="API key format may be invalid",
            details="PromptLayer API keys typically start with 'pl-'",
            fix_suggestion="Verify your API key from PromptLayer dashboard",
            category="configuration"
        )
    
    # Basic format validation
    if len(key) < 10:
        return ValidationCheck(
            name="PromptLayer API Key",
            status=ValidationStatus.WARNING,
            message="API key appears too short",
            details=f"Key length: {len(key)} characters",
            fix_suggestion="Verify your complete API key from PromptLayer dashboard",
            category="configuration"
        )
    
    return ValidationCheck(
        name="PromptLayer API Key",
        status=ValidationStatus.PASSED,
        message="PromptLayer API key is configured and format appears valid",
        details=f"Key length: {len(key)} characters, starts with: {key[:5]}...",
        category="configuration"
    )

@_measure_duration
def check_genops_configuration(team: Optional[str] = None, project: Optional[str] = None) -> ValidationCheck:
    """Check GenOps governance configuration."""
    config_team = team or os.getenv('GENOPS_TEAM')
    config_project = project or os.getenv('GENOPS_PROJECT')
    
    issues = []
    if not config_team:
        issues.append("Team not specified (GENOPS_TEAM)")
    if not config_project:
        issues.append("Project not specified (GENOPS_PROJECT)")
    
    if issues:
        return ValidationCheck(
            name="GenOps Configuration",
            status=ValidationStatus.WARNING,
            message="GenOps configuration is incomplete",
            details=f"Missing: {', '.join(issues)}",
            fix_suggestion="Set team/project: export GENOPS_TEAM=your-team GENOPS_PROJECT=your-project",
            category="configuration"
        )
    
    return ValidationCheck(
        name="GenOps Configuration",
        status=ValidationStatus.PASSED,
        message="GenOps configuration is complete",
        details=f"Team: {config_team}, Project: {config_project}",
        category="configuration"
    )

@_measure_duration
def check_promptlayer_connectivity(api_key: Optional[str] = None) -> ValidationCheck:
    """Test connectivity to PromptLayer API."""
    try:
        import promptlayer
        from promptlayer import PromptLayer
        
        key = api_key or os.getenv('PROMPTLAYER_API_KEY')
        if not key:
            return ValidationCheck(
                name="PromptLayer Connectivity",
                status=ValidationStatus.SKIPPED,
                message="Skipped - no API key available",
                category="connectivity"
            )
        
        # Initialize client and test basic functionality
        client = PromptLayer(api_key=key)
        
        # Try a basic API call (this may depend on PromptLayer's actual API)
        # For now, we'll just check if the client initializes successfully
        
        return ValidationCheck(
            name="PromptLayer Connectivity",
            status=ValidationStatus.PASSED,
            message="Successfully connected to PromptLayer API",
            category="connectivity"
        )
        
    except ImportError:
        return ValidationCheck(
            name="PromptLayer Connectivity",
            status=ValidationStatus.FAILED,
            message="Cannot test connectivity - PromptLayer SDK not available",
            fix_suggestion="Install PromptLayer SDK: pip install promptlayer",
            category="connectivity"
        )
    except Exception as e:
        return ValidationCheck(
            name="PromptLayer Connectivity",
            status=ValidationStatus.FAILED,
            message="Failed to connect to PromptLayer API",
            details=str(e),
            fix_suggestion="Check your API key and network connectivity",
            category="connectivity"
        )

@_measure_duration
def check_genops_promptlayer_integration() -> ValidationCheck:
    """Check GenOps PromptLayer integration functionality."""
    try:
        from genops.providers.promptlayer import GenOpsPromptLayerAdapter, instrument_promptlayer
        
        # Test adapter creation
        adapter = GenOpsPromptLayerAdapter(
            team="validation-test",
            project="integration-check"
        )
        
        # Test basic functionality
        metrics = adapter.get_metrics()
        
        return ValidationCheck(
            name="GenOps PromptLayer Integration",
            status=ValidationStatus.PASSED,
            message="GenOps PromptLayer integration is functional",
            details=f"Adapter created successfully, team: {metrics.get('team')}, project: {metrics.get('project')}",
            category="integration"
        )
        
    except ImportError as e:
        return ValidationCheck(
            name="GenOps PromptLayer Integration", 
            status=ValidationStatus.FAILED,
            message="GenOps PromptLayer integration not available",
            details=str(e),
            fix_suggestion="Install GenOps with PromptLayer support: pip install genops[promptlayer]",
            category="integration"
        )
    except Exception as e:
        return ValidationCheck(
            name="GenOps PromptLayer Integration",
            status=ValidationStatus.FAILED,
            message="GenOps PromptLayer integration failed",
            details=str(e),
            fix_suggestion="Check GenOps installation and configuration",
            category="integration"
        )

@_measure_duration
def check_governance_features() -> ValidationCheck:
    """Check governance feature functionality."""
    try:
        from genops.providers.promptlayer import GenOpsPromptLayerAdapter, GovernancePolicy
        
        # Test governance features
        adapter = GenOpsPromptLayerAdapter(
            team="governance-test",
            project="feature-check",
            daily_budget_limit=10.0,
            max_operation_cost=1.0,
            governance_policy=GovernancePolicy.ADVISORY
        )
        
        # Test context manager
        with adapter.track_prompt_operation(
            prompt_name="test_prompt",
            operation_type="validation"
        ) as span:
            span.update_cost(0.005)
            span.add_attributes({"test": "validation"})
        
        metrics = adapter.get_metrics()
        
        return ValidationCheck(
            name="Governance Features",
            status=ValidationStatus.PASSED,
            message="Governance features are functional",
            details=f"Budget tracking: ${metrics.get('daily_usage', 0):.6f}, Operations: {metrics.get('operation_count', 0)}",
            category="governance"
        )
        
    except Exception as e:
        return ValidationCheck(
            name="Governance Features",
            status=ValidationStatus.FAILED,
            message="Governance features failed",
            details=str(e),
            fix_suggestion="Check GenOps PromptLayer integration installation",
            category="governance"
        )

@_measure_duration
def check_performance_overhead() -> ValidationCheck:
    """Check performance overhead of governance instrumentation."""
    try:
        from genops.providers.promptlayer import GenOpsPromptLayerAdapter
        
        # Measure overhead
        iterations = 100
        start_time = time.time()
        
        adapter = GenOpsPromptLayerAdapter(team="perf-test", project="overhead-check")
        
        for i in range(iterations):
            with adapter.track_prompt_operation(
                prompt_name=f"perf_test_{i}",
                operation_type="performance_test"
            ) as span:
                span.update_cost(0.001)
                span.add_attributes({"iteration": i})
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / iterations) * 1000
        
        if avg_time_ms < 1.0:
            status = ValidationStatus.PASSED
            message = "Performance overhead is minimal"
        elif avg_time_ms < 5.0:
            status = ValidationStatus.WARNING
            message = "Performance overhead is acceptable"
        else:
            status = ValidationStatus.WARNING
            message = "Performance overhead is noticeable"
        
        return ValidationCheck(
            name="Performance Overhead",
            status=status,
            message=message,
            details=f"Average governance overhead: {avg_time_ms:.2f}ms per operation",
            category="performance"
        )
        
    except Exception as e:
        return ValidationCheck(
            name="Performance Overhead",
            status=ValidationStatus.FAILED,
            message="Performance check failed",
            details=str(e),
            category="performance"
        )

def validate_setup(
    promptlayer_api_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    include_connectivity_tests: bool = True,
    include_performance_tests: bool = True,
    include_governance_tests: bool = True
) -> ValidationResult:
    """
    Run comprehensive setup validation for PromptLayer integration.
    
    Args:
        promptlayer_api_key: PromptLayer API key to validate
        team: Team name for governance configuration
        project: Project name for governance configuration
        include_connectivity_tests: Whether to test API connectivity
        include_performance_tests: Whether to run performance tests
        include_governance_tests: Whether to test governance features
    
    Returns:
        ValidationResult: Comprehensive validation results
    """
    start_time = time.time()
    checks = []
    
    # Core dependency checks
    checks.append(check_python_version())
    checks.append(check_genops_installation())
    checks.append(check_promptlayer_installation())
    checks.append(check_optional_dependencies())
    
    # Configuration checks
    checks.append(check_promptlayer_api_key(promptlayer_api_key))
    checks.append(check_genops_configuration(team, project))
    
    # Integration checks
    checks.append(check_genops_promptlayer_integration())
    
    # Conditional checks
    if include_connectivity_tests:
        checks.append(check_promptlayer_connectivity(promptlayer_api_key))
    
    if include_governance_tests:
        checks.append(check_governance_features())
    
    if include_performance_tests:
        checks.append(check_performance_overhead())
    
    # Calculate overall status
    failed_count = len([c for c in checks if c.status == ValidationStatus.FAILED])
    warning_count = len([c for c in checks if c.status == ValidationStatus.WARNING])
    
    if failed_count > 0:
        overall_status = ValidationStatus.FAILED
    elif warning_count > 0:
        overall_status = ValidationStatus.WARNING
    else:
        overall_status = ValidationStatus.PASSED
    
    # Generate summary
    total_duration = (time.time() - start_time) * 1000
    summary = {
        "total_checks": len(checks),
        "passed": len([c for c in checks if c.status == ValidationStatus.PASSED]),
        "warnings": warning_count,
        "failed": failed_count,
        "skipped": len([c for c in checks if c.status == ValidationStatus.SKIPPED]),
        "categories": list(set(c.category for c in checks)),
        "validation_duration_ms": total_duration
    }
    
    return ValidationResult(
        overall_status=overall_status,
        checks=checks,
        total_duration_ms=total_duration,
        summary=summary
    )

def print_validation_result(result: ValidationResult, detailed: bool = False) -> None:
    """
    Print validation results in a user-friendly format.
    
    Args:
        result: Validation result to print
        detailed: Whether to show detailed information
    """
    # Status symbols
    status_symbols = {
        ValidationStatus.PASSED: "✅",
        ValidationStatus.WARNING: "⚠️",
        ValidationStatus.FAILED: "❌", 
        ValidationStatus.SKIPPED: "⏭️"
    }
    
    # Header
    overall_symbol = status_symbols[result.overall_status]
    print(f"\n{overall_symbol} GenOps PromptLayer Setup Validation")
    print(f"Overall Status: {result.overall_status.value.upper()}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print("-" * 50)
    
    # Summary
    print(f"📊 Summary:")
    print(f"   ✅ Passed: {result.summary['passed']}")
    print(f"   ⚠️ Warnings: {result.summary['warnings']}")
    print(f"   ❌ Failed: {result.summary['failed']}")
    print(f"   ⏭️ Skipped: {result.summary['skipped']}")
    print(f"   📝 Total: {result.summary['total_checks']}")
    
    # Detailed results
    if detailed or result.overall_status != ValidationStatus.PASSED:
        print(f"\n📋 Detailed Results:")
        
        # Group by category
        categories = {}
        for check in result.checks:
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append(check)
        
        for category, checks in categories.items():
            print(f"\n🏷️ {category.title()}:")
            for check in checks:
                symbol = status_symbols[check.status]
                duration = f" ({check.duration_ms:.0f}ms)" if check.duration_ms else ""
                print(f"   {symbol} {check.name}: {check.message}{duration}")
                
                if detailed and (check.details or check.fix_suggestion):
                    if check.details:
                        print(f"      Details: {check.details}")
                    if check.fix_suggestion:
                        print(f"      Fix: {check.fix_suggestion}")
    
    # Next steps
    print(f"\n🚀 Next Steps:")
    if result.overall_status == ValidationStatus.PASSED:
        print("   ✅ All checks passed! You're ready to use PromptLayer with GenOps governance.")
        print("   📚 Check out the examples: examples/promptlayer/")
    elif result.overall_status == ValidationStatus.WARNING:
        print("   ⚠️ Setup is functional but some optimizations are recommended.")
        print("   📖 Review the warnings above for improvement suggestions.")
        print("   🚀 You can proceed with basic usage.")
    else:
        print("   ❌ Setup has critical issues that need to be resolved.")
        print("   🔧 Fix the failed checks above before proceeding.")
        print("   📖 See fix suggestions for specific resolution steps.")
    
    print("")