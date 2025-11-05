#!/usr/bin/env python3
"""
GenOps Replicate Validation and Diagnostics

Comprehensive validation utilities for Replicate integration setup, configuration,
and operational health. Provides actionable diagnostics with specific fix guidance
following CLAUDE.md excellence standards.

Features:
- Complete setup validation with specific error messages
- API connectivity testing with detailed failure analysis
- Model availability verification across categories
- Performance benchmarking and optimization recommendations
- Environment configuration validation
- Network connectivity diagnostics

Usage:
    from genops.providers.replicate_validation import validate_setup, print_validation_result
    
    # Complete validation
    result = validate_setup()
    print_validation_result(result)
    
    # Quick validation
    from genops.providers.replicate_validation import quick_validate
    quick_validate()
"""

import json
import logging
import os
import time
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import requests

try:
    import replicate
except ImportError:
    replicate = None

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Comprehensive validation result with actionable diagnostics."""
    
    success: bool
    errors: List[str] = None
    warnings: List[str] = None  
    performance_metrics: Optional[Dict[str, Any]] = None
    environment_info: Optional[Dict[str, Any]] = None
    model_availability: Optional[Dict[str, bool]] = None
    optimization_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []

@dataclass  
class ModelTestResult:
    """Result of testing a specific Replicate model."""
    
    model_name: str
    available: bool
    latency_ms: Optional[float] = None
    cost_estimate: Optional[float] = None
    error: Optional[str] = None
    category: Optional[str] = None

class ReplicateValidator:
    """Comprehensive validator for Replicate integration setup."""
    
    def __init__(self):
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        
    def validate_complete_setup(self) -> ValidationResult:
        """Run complete validation of Replicate setup."""
        
        result = ValidationResult(success=True)
        
        # 1. Environment validation
        result.environment_info = self._validate_environment(result)
        
        # 2. Dependencies validation  
        self._validate_dependencies(result)
        
        # 3. Authentication validation
        self._validate_authentication(result)
        
        # 4. API connectivity validation
        self._validate_api_connectivity(result) 
        
        # 5. Model availability validation
        result.model_availability = self._validate_model_availability(result)
        
        # 6. Performance benchmarking
        result.performance_metrics = self._run_performance_benchmarks(result)
        
        # 7. Generate optimization recommendations
        result.optimization_recommendations = self._generate_recommendations(result)
        
        # Final success determination
        result.success = len(result.errors) == 0
        
        return result
    
    def _validate_environment(self, result: ValidationResult) -> Dict[str, Any]:
        """Validate environment configuration."""
        
        env_info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "replicate_token_set": bool(self.api_token),
            "replicate_token_valid_format": False,
            "environment_variables": {}
        }
        
        # Check Python version
        if sys.version_info < (3, 8):
            result.errors.append("Python 3.8+ required for Replicate integration")
            result.errors.append("🔧 PYTHON VERSION FIX:")
            result.errors.append("   1. Install Python 3.8+: https://python.org/downloads/")
            result.errors.append("   2. Update your environment: python --version")
        
        # Check API token format
        if self.api_token:
            if self.api_token.startswith('r8_') and len(self.api_token) > 10:
                env_info["replicate_token_valid_format"] = True
            else:
                result.errors.append("Invalid REPLICATE_API_TOKEN format")
                result.errors.append("🔧 API TOKEN FORMAT FIX:")
                result.errors.append("   1. Get token from: https://replicate.com/account/api-tokens")
                result.errors.append("   2. Token should start with 'r8_' and be ~40 characters")
                result.errors.append("   3. export REPLICATE_API_TOKEN='r8_your_actual_token_here'")
        else:
            result.errors.append("REPLICATE_API_TOKEN environment variable not set")
            result.errors.append("🔧 API TOKEN SETUP:")
            result.errors.append("   1. Visit: https://replicate.com/account/api-tokens")
            result.errors.append("   2. Click 'Create token' and copy the token")
            result.errors.append("   3. export REPLICATE_API_TOKEN='r8_your_token_here'")
            result.errors.append("   4. Restart your shell/IDE")
        
        # Check optional environment variables
        optional_vars = [
            "GENOPS_ENVIRONMENT",
            "GENOPS_PROJECT", 
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_SERVICE_NAME"
        ]
        
        for var in optional_vars:
            value = os.getenv(var)
            env_info["environment_variables"][var] = value
            if not value:
                result.warnings.append(f"Optional environment variable {var} not set")
        
        return env_info
    
    def _validate_dependencies(self, result: ValidationResult):
        """Validate required dependencies."""
        
        # Check replicate package
        if replicate is None:
            result.errors.append("Replicate Python SDK not installed")
            result.errors.append("🔧 DEPENDENCY FIX:")
            result.errors.append("   pip install replicate")
            result.errors.append("   # Or install with GenOps:")  
            result.errors.append("   pip install genops-ai[replicate]")
            return
        
        # Check replicate version
        try:
            version = replicate.__version__
            result.environment_info = result.environment_info or {}
            result.environment_info["replicate_version"] = version
            
            # Warn about old versions
            if version < "0.20.0":  # Adjust based on minimum supported version
                result.warnings.append(f"Replicate SDK version {version} may be outdated")
                result.warnings.append("🔧 UPDATE SUGGESTION:")
                result.warnings.append("   pip install --upgrade replicate")
        
        except AttributeError:
            result.warnings.append("Unable to determine Replicate SDK version")
        
        # Check OpenTelemetry dependencies
        try:
            from opentelemetry import trace
        except ImportError:
            result.warnings.append("OpenTelemetry not available - telemetry will be disabled")
            result.warnings.append("🔧 TELEMETRY SETUP (optional):")
            result.warnings.append("   pip install opentelemetry-api opentelemetry-sdk")
    
    def _validate_authentication(self, result: ValidationResult):
        """Validate Replicate API authentication."""
        
        if not self.api_token:
            return  # Already handled in environment validation
        
        # Test authentication with a simple API call
        try:
            # Try to list models (lightweight operation) 
            headers = {
                "Authorization": f"Token {self.api_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.replicate.com/v1/models",
                headers=headers,
                params={"limit": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                # Authentication successful
                pass
            elif response.status_code == 401:
                result.errors.append("Authentication failed - invalid API token")
                result.errors.append("🔧 AUTHENTICATION FIX:")
                result.errors.append("   1. Verify token: https://replicate.com/account/api-tokens")
                result.errors.append("   2. Copy the correct token (starts with 'r8_')")
                result.errors.append("   3. export REPLICATE_API_TOKEN='r8_your_correct_token'")
                result.errors.append("   4. Restart terminal and try again")
            else:
                result.warnings.append(f"Unexpected API response: {response.status_code}")
        
        except requests.exceptions.ConnectTimeout:
            result.errors.append("Connection timeout to Replicate API")
            result.errors.append("🔧 CONNECTIVITY FIX:")
            result.errors.append("   1. Check internet connection")
            result.errors.append("   2. Verify no firewall blocking api.replicate.com")
            result.errors.append("   3. Try again in a few minutes")
        
        except requests.exceptions.RequestException as e:
            result.errors.append(f"Network error connecting to Replicate: {e}")
            result.errors.append("🔧 NETWORK FIX:")
            result.errors.append("   1. Check internet connection")
            result.errors.append("   2. Verify DNS resolution: nslookup api.replicate.com")
            result.errors.append("   3. Try from different network if possible")
    
    def _validate_api_connectivity(self, result: ValidationResult):
        """Test API connectivity and response times."""
        
        if not self.api_token or result.errors:
            return  # Skip if authentication failed
        
        try:
            # Configure replicate client
            client = replicate.Client(api_token=self.api_token)
            
            # Test basic connectivity
            start_time = time.time()
            models = list(client.models.list(limit=1))
            api_latency = (time.time() - start_time) * 1000
            
            result.performance_metrics = result.performance_metrics or {}
            result.performance_metrics["api_latency_ms"] = api_latency
            
            if api_latency > 5000:  # 5 seconds
                result.warnings.append(f"High API latency: {api_latency:.0f}ms")
                result.warnings.append("🔧 PERFORMANCE OPTIMIZATION:")
                result.warnings.append("   1. Check network connection quality")
                result.warnings.append("   2. Consider using replicate.stream() for long operations")
        
        except Exception as e:
            result.errors.append(f"API connectivity test failed: {e}")
            result.errors.append("🔧 API CONNECTION FIX:")
            result.errors.append("   1. Verify REPLICATE_API_TOKEN is correct")
            result.errors.append("   2. Check https://replicate.com/status for service status")
            result.errors.append("   3. Try again in a few minutes")
    
    def _validate_model_availability(self, result: ValidationResult) -> Dict[str, bool]:
        """Test availability of key Replicate models."""
        
        if not self.api_token or result.errors:
            return {}
        
        # Test models across different categories
        test_models = [
            ("meta/llama-2-7b-chat", "text"),
            ("black-forest-labs/flux-schnell", "image"),
            ("openai/whisper", "audio"),
        ]
        
        availability = {}
        model_test_results = []
        
        for model_name, category in test_models:
            test_result = self._test_model_availability(model_name, category)
            availability[model_name] = test_result.available
            model_test_results.append(test_result)
            
            if not test_result.available:
                result.warnings.append(f"Model {model_name} not available: {test_result.error}")
        
        # Store detailed results in performance metrics
        result.performance_metrics = result.performance_metrics or {}
        result.performance_metrics["model_tests"] = [asdict(t) for t in model_test_results]
        
        return availability
    
    def _test_model_availability(self, model_name: str, category: str) -> ModelTestResult:
        """Test if a specific model is available and responsive."""
        
        test_result = ModelTestResult(
            model_name=model_name,
            available=False,
            category=category
        )
        
        try:
            client = replicate.Client(api_token=self.api_token)
            
            # Try to get model info (lightweight test)
            start_time = time.time()
            model = client.models.get(model_name)
            test_result.latency_ms = (time.time() - start_time) * 1000
            test_result.available = True
            
        except replicate.exceptions.ReplicateError as e:
            test_result.error = str(e)
        except Exception as e:
            test_result.error = f"Unexpected error: {e}"
        
        return test_result
    
    def _run_performance_benchmarks(self, result: ValidationResult) -> Dict[str, Any]:
        """Run performance benchmarks for optimization guidance."""
        
        metrics = result.performance_metrics or {}
        
        if result.errors:
            return metrics  # Skip benchmarks if setup is broken
        
        # Basic performance indicators already collected:
        # - api_latency_ms from connectivity test
        # - model_tests from availability test
        
        # Add system metrics
        metrics["system"] = {
            "python_version": result.environment_info.get("python_version"),
            "platform": result.environment_info.get("platform"),
            "timestamp": time.time()
        }
        
        return metrics
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate optimization and best practice recommendations."""
        
        recommendations = []
        
        # Performance recommendations
        if result.performance_metrics:
            api_latency = result.performance_metrics.get("api_latency_ms", 0)
            
            if api_latency > 2000:
                recommendations.append("High API latency detected - consider caching responses")
                recommendations.append("Use replicate.stream() for long-running operations")
            
            if api_latency < 500:
                recommendations.append("Good API performance - suitable for real-time applications")
        
        # Environment recommendations
        if result.environment_info:
            if not result.environment_info.get("environment_variables", {}).get("OTEL_EXPORTER_OTLP_ENDPOINT"):
                recommendations.append("Set OTEL_EXPORTER_OTLP_ENDPOINT to enable telemetry export")
            
            if not result.environment_info.get("environment_variables", {}).get("GENOPS_ENVIRONMENT"):
                recommendations.append("Set GENOPS_ENVIRONMENT (dev/staging/prod) for proper attribution")
        
        # Model availability recommendations
        if result.model_availability:
            available_models = sum(result.model_availability.values())
            total_models = len(result.model_availability)
            
            if available_models < total_models:
                recommendations.append("Some test models unavailable - check model names and access permissions")
            else:
                recommendations.append("All test models available - ready for production use")
        
        # Setup completion recommendations  
        if result.success:
            recommendations.append("✅ Setup validation passed - ready to use GenOps with Replicate!")
            recommendations.append("Next steps: Try the hello_genops_minimal.py example")
        else:
            recommendations.append("❌ Setup issues found - fix errors above before proceeding")
        
        return recommendations

def validate_setup() -> ValidationResult:
    """
    Run comprehensive Replicate setup validation.
    
    Returns:
        ValidationResult with detailed diagnostics and fix suggestions
    """
    validator = ReplicateValidator()
    return validator.validate_complete_setup()

def print_validation_result(result: ValidationResult, detailed: bool = False):
    """
    Print human-readable validation results with actionable guidance.
    
    Args:
        result: ValidationResult from validate_setup()
        detailed: Include detailed metrics and environment info
    """
    print("🔍 GenOps Replicate Validation Report")
    print("=" * 50)
    
    # Overall status
    if result.success:
        print("✅ SUCCESS: Replicate integration is ready!")
    else:
        print("❌ ISSUES FOUND: Setup needs attention")
    
    print()
    
    # Errors (blocking issues)
    if result.errors:
        print("🚨 ERRORS TO FIX:")
        for i, error in enumerate(result.errors, 1):
            if error.startswith("🔧"):
                print(f"   {error}")
            else:
                print(f"{i:2}. {error}")
        print()
    
    # Warnings (non-blocking issues)
    if result.warnings:
        print("⚠️  WARNINGS:")
        for i, warning in enumerate(result.warnings, 1):
            if warning.startswith("🔧"):
                print(f"   {warning}")
            else:
                print(f"{i:2}. {warning}")
        print()
    
    # Model availability
    if result.model_availability:
        print("🤖 MODEL AVAILABILITY:")
        for model, available in result.model_availability.items():
            status = "✅" if available else "❌"
            print(f"   {status} {model}")
        print()
    
    # Performance metrics
    if result.performance_metrics and detailed:
        print("📊 PERFORMANCE METRICS:")
        metrics = result.performance_metrics
        
        if "api_latency_ms" in metrics:
            print(f"   API Latency: {metrics['api_latency_ms']:.0f}ms")
        
        if "model_tests" in metrics:
            print("   Model Response Times:")
            for test in metrics["model_tests"]:
                if test["latency_ms"]:
                    print(f"     {test['model_name']}: {test['latency_ms']:.0f}ms")
        print()
    
    # Environment info
    if result.environment_info and detailed:
        print("🔧 ENVIRONMENT INFO:")
        env = result.environment_info
        print(f"   Python: {env.get('python_version')}")
        print(f"   Platform: {env.get('platform')}")
        print(f"   Replicate SDK: {env.get('replicate_version', 'Unknown')}")
        print(f"   API Token: {'✅ Set' if env.get('replicate_token_set') else '❌ Missing'}")
        
        if env.get("environment_variables"):
            print("   Environment Variables:")
            for var, value in env["environment_variables"].items():
                status = "✅" if value else "❌"
                print(f"     {status} {var}")
        print()
    
    # Optimization recommendations
    if result.optimization_recommendations:
        print("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result.optimization_recommendations, 1):
            print(f"{i:2}. {rec}")
        print()
    
    # Next steps
    if result.success:
        print("🎯 NEXT STEPS:")
        print("   1. Try the examples: python examples/replicate/hello_genops_minimal.py")
        print("   2. Explore the documentation: examples/replicate/README.md")
        print("   3. Start tracking your Replicate usage with GenOps!")
    else:
        print("🔧 FIX ERRORS ABOVE:")
        print("   1. Address all error messages with the provided fixes")
        print("   2. Run validation again: python -c \"from genops.providers.replicate_validation import validate_setup, print_validation_result; print_validation_result(validate_setup())\"")
    
    print("=" * 50)

def quick_validate() -> bool:
    """
    Quick validation with simple pass/fail result.
    
    Returns:
        True if validation passed, False if issues found
    """
    result = validate_setup()
    
    if result.success:
        print("✅ GenOps Replicate validation passed!")
        return True
    else:
        print("❌ GenOps Replicate validation failed")
        print("🔧 Run detailed validation for fix guidance:")
        print("   python -c \"from genops.providers.replicate_validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")
        return False

# Export main functions
__all__ = [
    'validate_setup',
    'print_validation_result', 
    'quick_validate',
    'ValidationResult',
    'ModelTestResult',
    'ReplicateValidator'
]