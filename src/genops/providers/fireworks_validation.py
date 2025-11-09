"""
Fireworks AI Setup Validation for GenOps Integration

Comprehensive validation utilities to ensure proper Fireworks AI setup with actionable
diagnostics and troubleshooting guidance for developers.
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from fireworks.client import Fireworks
    HAS_FIREWORKS = True
except ImportError:
    HAS_FIREWORKS = False
    Fireworks = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class ValidationResult:
    """Result of Fireworks AI setup validation."""
    is_valid: bool
    api_key_valid: bool
    dependencies_installed: bool
    connectivity_ok: bool
    model_access: List[str]
    error_message: Optional[str] = None
    warnings: List[str] = None
    recommendations: List[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class ModelAccessResult:
    """Result of model access validation."""
    model: str
    accessible: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    cost_estimate: Optional[Decimal] = None


class FireworksValidation:
    """
    Comprehensive validation utilities for Fireworks AI + GenOps setup.
    """
    
    def __init__(self):
        """Initialize validation utilities."""
        self.test_models = [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",      # Fast, cheap
            "accounts/fireworks/models/llama-v3p1-70b-instruct",     # Standard
            "accounts/fireworks/models/nomic-embed-text-v1p5",       # Embeddings
            "accounts/fireworks/models/whisper-v3",                  # Audio
            "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"  # Multimodal
        ]
    
    def validate_setup(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        test_model_access: bool = True,
        test_performance: bool = True,
        **kwargs
    ) -> ValidationResult:
        """
        Comprehensive setup validation for Fireworks AI integration.
        
        Args:
            api_key: Fireworks API key to validate
            config: Configuration to validate
            test_model_access: Whether to test model access
            test_performance: Whether to test performance
            **kwargs: Additional validation parameters
        
        Returns:
            ValidationResult with comprehensive diagnostic information
        """
        result = ValidationResult(
            is_valid=False,
            api_key_valid=False,
            dependencies_installed=False,
            connectivity_ok=False,
            model_access=[]
        )
        
        try:
            # Step 1: Validate dependencies
            result.dependencies_installed = self._validate_dependencies(result)
            
            if not result.dependencies_installed:
                result.error_message = "Required dependencies not installed"
                return result
            
            # Step 2: Validate API key
            api_key = api_key or os.getenv("FIREWORKS_API_KEY")
            result.api_key_valid = self._validate_api_key(api_key, result)
            
            if not result.api_key_valid:
                result.error_message = "Invalid or missing Fireworks API key"
                return result
            
            # Step 3: Test connectivity
            result.connectivity_ok = self._test_connectivity(api_key, result)
            
            if not result.connectivity_ok:
                result.error_message = "Cannot connect to Fireworks API"
                return result
            
            # Step 4: Test model access (if requested)
            if test_model_access:
                result.model_access = self._test_model_access(api_key, result)
            
            # Step 5: Performance testing (if requested)
            if test_performance:
                result.performance_metrics = self._test_performance(api_key, result)
            
            # Step 6: Validate configuration (if provided)
            if config:
                self._validate_config(config, result)
            
            # Step 7: Generate recommendations
            self._generate_recommendations(result)
            
            # Overall validation status
            result.is_valid = (
                result.dependencies_installed and 
                result.api_key_valid and 
                result.connectivity_ok and
                (not test_model_access or len(result.model_access) > 0)
            )
            
            if result.is_valid:
                logger.info("✅ Fireworks AI validation successful")
            else:
                logger.warning("⚠️ Fireworks AI validation completed with issues")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            result.error_message = f"Validation failed: {str(e)}"
            return result
    
    def _validate_dependencies(self, result: ValidationResult) -> bool:
        """Validate required Python dependencies."""
        missing_deps = []
        
        # Check Fireworks AI client
        if not HAS_FIREWORKS:
            missing_deps.append("fireworks-ai")
            result.warnings.append("Fireworks AI client not installed")
        
        # Check optional dependencies
        if not HAS_REQUESTS:
            missing_deps.append("requests")
            result.warnings.append("Requests library not installed (optional)")
        
        if not HAS_OPENAI:
            result.warnings.append("OpenAI library not installed (optional for OpenAI compatibility)")
        
        # Generate installation instructions
        if missing_deps:
            install_cmd = f"pip install {' '.join(missing_deps)}"
            result.recommendations.append(f"Install missing dependencies: {install_cmd}")
            
            if "fireworks-ai" in missing_deps:
                return False  # Critical dependency missing
        
        return True
    
    def _validate_api_key(self, api_key: Optional[str], result: ValidationResult) -> bool:
        """Validate Fireworks API key format and presence."""
        if not api_key:
            result.warnings.append("FIREWORKS_API_KEY environment variable not set")
            result.recommendations.extend([
                "Set your Fireworks API key: export FIREWORKS_API_KEY='your_key_here'",
                "Get your API key from: https://fireworks.ai/api-keys",
                "Ensure the key starts with 'fw-' or has the correct format"
            ])
            return False
        
        # Basic format validation
        if len(api_key) < 20:
            result.warnings.append("API key appears too short")
            result.recommendations.append("Verify your API key is complete and correct")
            return False
        
        # Check for common issues
        if api_key.startswith("sk-"):
            result.warnings.append("API key format looks like OpenAI key, not Fireworks")
            result.recommendations.append("Ensure you're using a Fireworks API key, not OpenAI")
        
        return True
    
    def _test_connectivity(self, api_key: str, result: ValidationResult) -> bool:
        """Test basic connectivity to Fireworks API."""
        if not HAS_FIREWORKS:
            return False
        
        try:
            # Initialize client
            client = Fireworks(api_key=api_key)
            
            # Test basic API call
            start_time = time.time()
            
            # Try to list models or make a minimal API call
            try:
                # Make a minimal completion request to test connectivity
                response = client.chat.completions.create(
                    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1,
                    temperature=0
                )
                
                connectivity_time = time.time() - start_time
                result.performance_metrics = result.performance_metrics or {}
                result.performance_metrics["connectivity_latency_ms"] = connectivity_time * 1000
                
                logger.info(f"Connectivity test successful ({connectivity_time:.2f}s)")
                return True
                
            except Exception as api_error:
                # Try to determine the specific issue
                error_str = str(api_error).lower()
                
                if "unauthorized" in error_str or "invalid api key" in error_str:
                    result.warnings.append("API key appears to be invalid")
                    result.recommendations.extend([
                        "Verify your API key is correct",
                        "Check that your API key is active in the Fireworks dashboard"
                    ])
                elif "quota" in error_str or "billing" in error_str:
                    result.warnings.append("Account may have billing or quota issues")
                    result.recommendations.append("Check your Fireworks account billing status")
                elif "model" in error_str:
                    result.warnings.append("Model access issue (but API key may be valid)")
                    # Still return True as basic connectivity works
                    return True
                else:
                    result.warnings.append(f"API connectivity issue: {api_error}")
                
                return False
                
        except Exception as e:
            result.warnings.append(f"Connectivity test failed: {e}")
            result.recommendations.append("Check internet connection and firewall settings")
            return False
    
    def _test_model_access(self, api_key: str, result: ValidationResult) -> List[str]:
        """Test access to various Fireworks models."""
        if not HAS_FIREWORKS:
            return []
        
        accessible_models = []
        client = Fireworks(api_key=api_key)
        
        for model in self.test_models:
            try:
                start_time = time.time()
                
                # Test different model types appropriately
                if "embed" in model.lower():
                    # Test embedding model
                    response = client.embeddings.create(
                        model=model,
                        input=["test"]
                    )
                elif "whisper" in model.lower():
                    # Skip audio model test for now (requires audio file)
                    result.warnings.append(f"Skipped audio model test: {model}")
                    continue
                else:
                    # Test chat model
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=1,
                        temperature=0
                    )
                
                latency = (time.time() - start_time) * 1000
                accessible_models.append(model)
                
                logger.info(f"✅ Model {model} accessible ({latency:.0f}ms)")
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "not found" in error_str or "does not exist" in error_str:
                    result.warnings.append(f"Model {model} not available")
                elif "quota" in error_str or "rate limit" in error_str:
                    result.warnings.append(f"Rate limited testing {model}")
                else:
                    result.warnings.append(f"Cannot access model {model}: {e}")
        
        if len(accessible_models) == 0:
            result.recommendations.append("No models accessible - check account status and permissions")
        elif len(accessible_models) < len(self.test_models) // 2:
            result.recommendations.append("Limited model access - some models may require special permissions")
        
        return accessible_models
    
    def _test_performance(self, api_key: str, result: ValidationResult) -> Dict[str, Any]:
        """Test basic performance characteristics."""
        if not HAS_FIREWORKS:
            return {}
        
        metrics = {}
        client = Fireworks(api_key=api_key)
        
        try:
            # Test simple completion performance
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=10,
                temperature=0
            )
            
            total_time = time.time() - start_time
            
            if hasattr(response, 'usage') and response.usage:
                tokens = response.usage.total_tokens
                metrics.update({
                    "test_completion_time_ms": total_time * 1000,
                    "test_tokens": tokens,
                    "tokens_per_second": tokens / total_time if total_time > 0 else 0
                })
            
            # Performance assessment
            if total_time < 2.0:
                result.recommendations.append("Excellent API performance detected")
            elif total_time < 5.0:
                result.recommendations.append("Good API performance")
            else:
                result.warnings.append("Slow API response times detected")
                result.recommendations.append("Check network connection for optimal performance")
            
        except Exception as e:
            result.warnings.append(f"Performance test failed: {e}")
        
        return metrics
    
    def _validate_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate GenOps configuration parameters."""
        required_config = ["team", "project"]
        recommended_config = ["environment", "daily_budget_limit"]
        
        # Check required configuration
        for key in required_config:
            if key not in config or not config[key]:
                result.warnings.append(f"Required config '{key}' missing")
                result.recommendations.append(f"Set {key} in your GenOpsFireworksAdapter configuration")
        
        # Check recommended configuration
        for key in recommended_config:
            if key not in config:
                result.recommendations.append(f"Consider setting '{key}' for better governance")
        
        # Validate budget limits
        if "daily_budget_limit" in config:
            try:
                daily_limit = float(config["daily_budget_limit"])
                if daily_limit <= 0:
                    result.warnings.append("Daily budget limit should be positive")
                elif daily_limit < 1.0:
                    result.recommendations.append("Very low daily budget limit may restrict usage")
            except (ValueError, TypeError):
                result.warnings.append("Invalid daily budget limit format")
        
        # Validate governance policy
        if "governance_policy" in config:
            valid_policies = ["advisory", "enforced", "strict"]
            if config["governance_policy"] not in valid_policies:
                result.warnings.append(f"Invalid governance policy. Use: {', '.join(valid_policies)}")
    
    def _generate_recommendations(self, result: ValidationResult):
        """Generate setup recommendations based on validation results."""
        if result.is_valid:
            result.recommendations.extend([
                "Setup validation successful - ready for Fireworks AI operations!",
                "Consider testing with basic_tracking.py example",
                "Review cost_optimization.py for intelligent model selection"
            ])
        else:
            result.recommendations.extend([
                "Complete the setup issues above before proceeding",
                "Run validation again after making changes",
                "Check the Fireworks AI documentation for additional help"
            ])
        
        # Performance recommendations
        if result.performance_metrics:
            latency = result.performance_metrics.get("test_completion_time_ms", 0)
            
            if latency > 5000:  # > 5 seconds
                result.recommendations.append("Consider using faster models for better performance")
            
            tokens_per_sec = result.performance_metrics.get("tokens_per_second", 0)
            if tokens_per_sec > 50:
                result.recommendations.append("Excellent throughput - suitable for high-volume applications")
        
        # Cost optimization recommendations
        if len(result.model_access) > 1:
            result.recommendations.extend([
                "Multiple models accessible - use cost_optimization.py to find best model for your use case",
                "Consider batch processing for 50% cost savings on large workloads"
            ])


def validate_fireworks_setup(
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    print_results: bool = False,
    **kwargs
) -> ValidationResult:
    """
    Convenience function for Fireworks AI setup validation.
    
    Args:
        api_key: Fireworks API key to validate  
        config: Configuration dictionary to validate
        print_results: Whether to print formatted results
        **kwargs: Additional validation parameters
    
    Returns:
        ValidationResult with comprehensive diagnostic information
    """
    validator = FireworksValidation()
    result = validator.validate_setup(api_key=api_key, config=config, **kwargs)
    
    if print_results:
        print_validation_result(result)
    
    return result


def print_validation_result(result: ValidationResult):
    """
    Print formatted validation results with actionable guidance.
    
    Args:
        result: ValidationResult to format and print
    """
    print("🔧 Fireworks AI + GenOps Setup Validation")
    print("=" * 50)
    
    # Overall status
    if result.is_valid:
        print("✅ VALIDATION SUCCESSFUL")
    else:
        print("❌ VALIDATION FAILED")
    
    print()
    
    # Detailed results
    print(f"✅ Dependencies: {'✅ Installed' if result.dependencies_installed else '❌ Missing'}")
    print(f"✅ API Key: {'✅ Valid' if result.api_key_valid else '❌ Invalid'}")
    print(f"✅ Connectivity: {'✅ Connected' if result.connectivity_ok else '❌ Failed'}")
    print(f"✅ Model Access: ✅ {len(result.model_access)} models accessible" if result.model_access else "❌ No models accessible")
    
    # Performance metrics
    if result.performance_metrics:
        print("\n📊 Performance Metrics:")
        for key, value in result.performance_metrics.items():
            if "time" in key.lower() or "latency" in key.lower():
                print(f"   {key}: {value:.0f}ms")
            elif "tokens_per_second" in key:
                print(f"   {key}: {value:.1f} tokens/s")
            else:
                print(f"   {key}: {value}")
    
    # Accessible models
    if result.model_access:
        print(f"\n🤖 Accessible Models ({len(result.model_access)}):")
        for model in result.model_access[:5]:  # Show first 5
            model_name = model.split("/")[-1] if "/" in model else model
            print(f"   ✅ {model_name}")
        
        if len(result.model_access) > 5:
            print(f"   ... and {len(result.model_access) - 5} more")
    
    # Warnings
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")
    
    # Error message
    if result.error_message:
        print(f"\n❌ Error: {result.error_message}")
    
    print()


# Convenience functions for specific validation scenarios
def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Quick API key validation."""
    result = validate_fireworks_setup(api_key=api_key, test_model_access=False, test_performance=False)
    return result.api_key_valid and result.connectivity_ok


def validate_model_access(api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """Validate access to Fireworks models."""
    result = validate_fireworks_setup(api_key=api_key, test_performance=False)
    error = result.error_message if not result.is_valid else None
    return result.model_access, error


def get_performance_metrics(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Get basic performance metrics for Fireworks API."""
    result = validate_fireworks_setup(api_key=api_key, test_model_access=False)
    return result.performance_metrics or {}


# Export key classes and functions
__all__ = [
    "ValidationResult",
    "ModelAccessResult", 
    "FireworksValidation",
    "validate_fireworks_setup",
    "print_validation_result",
    "validate_api_key",
    "validate_model_access",
    "get_performance_metrics"
]