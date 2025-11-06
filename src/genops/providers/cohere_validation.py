"""Validation system for Cohere integration setup and diagnostics."""

import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple
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
    import cohere
    from cohere import ClientV2
    HAS_COHERE_CLIENT = True
except ImportError:
    HAS_COHERE_CLIENT = False


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
    AUTHENTICATION = "authentication"
    MODELS = "models"
    PERFORMANCE = "performance"
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
    performance_metrics: Dict[str, float] = field(default_factory=dict)
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
    
    def add_passed_check(self, check_name: str = ""):
        """Record a passed validation check."""
        self.passed_checks += 1
        self.total_checks += 1
    
    def add_failed_check(self, issue: ValidationIssue):
        """Record a failed validation check."""
        self.total_checks += 1
        self.add_issue(issue)


class CohereValidator:
    """
    Comprehensive validation system for Cohere integration.
    
    Validates:
    - Dependency installation and versions
    - Cohere API key and authentication
    - Model availability and access
    - Pricing and cost calculation setup
    - Performance characteristics
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        include_performance_tests: bool = True
    ):
        """
        Initialize validator.
        
        Args:
            api_key: Cohere API key (defaults to CO_API_KEY env var)
            timeout: Request timeout in seconds
            include_performance_tests: Whether to run performance validation tests
        """
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.timeout = timeout
        self.include_performance_tests = include_performance_tests
        
        self.result = ValidationResult(success=True)
        
        # Initialize Cohere client for testing
        self.client = None
        if HAS_COHERE_CLIENT and self.api_key:
            try:
                self.client = ClientV2(api_key=self.api_key, timeout=timeout)
            except Exception as e:
                logger.debug(f"Could not initialize Cohere client for validation: {e}")
    
    def validate_all(self) -> ValidationResult:
        """
        Run complete validation suite.
        
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive Cohere validation")
        
        # Core validation checks
        self._validate_dependencies()
        self._validate_configuration()
        self._validate_authentication()
        self._validate_connectivity()
        self._validate_models()
        self._validate_pricing()
        
        # Optional performance validation
        if self.include_performance_tests:
            self._validate_performance()
        
        # Generate recommendations
        self._generate_recommendations()
        
        logger.info(f"Validation completed: {self.result.score:.1f}% ({self.result.passed_checks}/{self.result.total_checks} checks passed)")
        return self.result
    
    def _validate_dependencies(self):
        """Validate required dependencies."""
        logger.debug("Validating dependencies...")
        
        # Check Python version
        import sys
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.result.add_passed_check("Python version")
        else:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.CRITICAL,
                title="Python Version Too Old",
                description=f"Python {python_version.major}.{python_version.minor} detected, requires Python 3.8+",
                fix_suggestion="Upgrade to Python 3.8 or later"
            ))
        
        # Check requests library
        if HAS_REQUESTS:
            self.result.add_passed_check("requests library")
        else:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR,
                title="Missing requests library",
                description="requests library is required for HTTP communication",
                fix_suggestion="Install with: pip install requests"
            ))
        
        # Check Cohere client
        if HAS_COHERE_CLIENT:
            self.result.add_passed_check("cohere client")
            
            # Check cohere client version
            try:
                cohere_version = cohere.__version__
                self.result.system_info['cohere_client_version'] = cohere_version
                
                # Check if version is recent enough
                version_parts = cohere_version.split('.')
                if len(version_parts) >= 2:
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    if major >= 5:  # Cohere v5+ has ClientV2
                        self.result.add_passed_check("cohere client version")
                    else:
                        self.result.add_issue(ValidationIssue(
                            category=ValidationCategory.DEPENDENCIES,
                            level=ValidationLevel.WARNING,
                            title="Outdated Cohere client",
                            description=f"Cohere {cohere_version} detected, recommend 5.0+ for ClientV2 support",
                            fix_suggestion="Upgrade with: pip install --upgrade cohere"
                        ))
                        
            except Exception as e:
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.WARNING,
                    title="Cannot determine Cohere version",
                    description=f"Could not check Cohere client version: {e}",
                    technical_details=str(e)
                ))
        else:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.CRITICAL,
                title="Missing Cohere client",
                description="Cohere Python client is required for integration",
                fix_suggestion="Install with: pip install cohere"
            ))
        
        # Check GenOps core dependencies
        try:
            from opentelemetry import trace
            self.result.add_passed_check("OpenTelemetry")
        except ImportError:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.DEPENDENCIES,
                level=ValidationLevel.ERROR,
                title="Missing OpenTelemetry",
                description="OpenTelemetry is required for GenOps telemetry",
                fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk"
            ))
    
    def _validate_configuration(self):
        """Validate configuration and environment."""
        logger.debug("Validating configuration...")
        
        # Check API key configuration
        if self.api_key:
            self.result.add_passed_check("API key configured")
            
            # Basic API key format validation
            if len(self.api_key) < 10:
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.WARNING,
                    title="API key format suspicious",
                    description="API key appears too short for valid Cohere key",
                    fix_suggestion="Verify API key is complete and correct"
                ))
            
            # Check if key starts with expected prefix
            if not self.api_key.startswith(('co_', 'ck_')):
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title="Non-standard API key format",
                    description="API key doesn't match typical Cohere format",
                    technical_details="Expected format: co_* or ck_*"
                ))
        else:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.CONFIGURATION,
                level=ValidationLevel.ERROR,
                title="No API key configured",
                description="Cohere API key not found in CO_API_KEY environment variable",
                fix_suggestion="Set environment variable: export CO_API_KEY=your-api-key"
            ))
        
        # Check environment variables
        env_vars = {
            'CO_API_KEY': 'Cohere API key',
            'COHERE_API_URL': 'Custom Cohere API URL'
        }
        
        for var, description in env_vars.items():
            value = os.getenv(var)
            if value:
                self.result.system_info[f'env_{var.lower()}'] = f"Set ({len(value)} chars)" if 'key' in var.lower() else value
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.INFO,
                    title=f"Environment variable {var} set",
                    description=f"{description}: {'*' * min(8, len(value)) if 'key' in var.lower() else value}"
                ))
        
        # Check GenOps configuration
        genops_env_vars = {
            'GENOPS_TELEMETRY_ENABLED': 'true',
            'GENOPS_COST_TRACKING_ENABLED': 'true',
            'OTEL_EXPORTER_OTLP_ENDPOINT': None
        }
        
        for var, default in genops_env_vars.items():
            value = os.getenv(var, default)
            if value:
                self.result.system_info[f'genops_{var.lower()}'] = value
    
    def _validate_authentication(self):
        """Validate Cohere API authentication."""
        logger.debug("Validating authentication...")
        
        if not self.api_key:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.ERROR,
                title="Cannot test authentication",
                description="No API key available for authentication testing",
                fix_suggestion="Provide API key via CO_API_KEY environment variable"
            ))
            return
        
        if not HAS_COHERE_CLIENT:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.AUTHENTICATION,
                level=ValidationLevel.ERROR,
                title="Cannot test authentication",
                description="Cohere client not available for authentication testing",
                fix_suggestion="Install Cohere client: pip install cohere"
            ))
            return
        
        # Test authentication with a simple API call
        try:
            start_time = time.time()
            
            # Try to list available models as auth test
            if self.client:
                # Use a simple chat call with minimal tokens
                response = self.client.chat(
                    model="command-light",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                
                auth_time = (time.time() - start_time) * 1000
                
                self.result.add_passed_check("API authentication")
                self.result.performance_metrics['auth_response_time_ms'] = auth_time
                
            else:
                # Fallback: create client for this test
                test_client = ClientV2(api_key=self.api_key, timeout=self.timeout)
                response = test_client.chat(
                    model="command-light",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                
                auth_time = (time.time() - start_time) * 1000
                
                self.result.add_passed_check("API authentication")
                self.result.performance_metrics['auth_response_time_ms'] = auth_time
                
        except Exception as e:
            error_str = str(e).lower()
            
            if "unauthorized" in error_str or "invalid" in error_str or "api key" in error_str:
                self.result.add_failed_check(ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.CRITICAL,
                    title="Invalid API key",
                    description="API key authentication failed - key may be invalid or expired",
                    fix_suggestion="Verify API key is correct and active in Cohere dashboard",
                    technical_details=str(e)
                ))
            elif "rate limit" in error_str or "quota" in error_str:
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.WARNING,
                    title="Rate limit or quota exceeded",
                    description="Authentication test hit rate limit or quota",
                    fix_suggestion="Check usage limits in Cohere dashboard",
                    technical_details=str(e)
                ))
            else:
                self.result.add_failed_check(ValidationIssue(
                    category=ValidationCategory.AUTHENTICATION,
                    level=ValidationLevel.ERROR,
                    title="Authentication test failed",
                    description=f"Unexpected error during authentication: {str(e)}",
                    fix_suggestion="Check network connectivity and API key validity",
                    technical_details=str(e)
                ))
    
    def _validate_connectivity(self):
        """Validate Cohere API connectivity."""
        logger.debug("Validating API connectivity...")
        
        if not HAS_REQUESTS:
            self.result.add_issue(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.WARNING,
                title="Cannot test connectivity",
                description="requests library not available for connectivity testing",
                fix_suggestion="Install requests: pip install requests"
            ))
            return
        
        # Test basic connectivity to Cohere API
        try:
            start_time = time.time()
            
            # Test connectivity to Cohere API endpoint
            api_url = "https://api.cohere.ai"
            response = requests.get(f"{api_url}/check-api-key", timeout=self.timeout, headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
            })
            
            connectivity_time = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 401]:  # 401 is expected without valid auth, but shows connectivity
                self.result.add_passed_check("API connectivity")
                self.result.performance_metrics['connectivity_time_ms'] = connectivity_time
            else:
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.WARNING,
                    title="API connectivity issue",
                    description=f"Cohere API returned HTTP {response.status_code}",
                    technical_details=f"GET {api_url}/check-api-key -> {response.status_code}"
                ))
                
        except requests.exceptions.ConnectTimeout:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.ERROR,
                title="Connection timeout",
                description=f"Cannot connect to Cohere API (timeout after {self.timeout}s)",
                fix_suggestion="Check network connectivity and firewall settings",
                technical_details=f"Timeout after {self.timeout}s"
            ))
        
        except requests.exceptions.ConnectionError:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.ERROR,
                title="Connection failed",
                description="Cannot connect to Cohere API servers",
                fix_suggestion="Check internet connection and DNS resolution",
                technical_details="Connection refused or DNS failure"
            ))
        
        except Exception as e:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.CONNECTIVITY,
                level=ValidationLevel.ERROR,
                title="Connectivity test error",
                description=f"Unexpected error testing connectivity: {str(e)}",
                fix_suggestion="Check network configuration and API accessibility"
            ))
    
    def _validate_models(self):
        """Validate available models and access."""
        logger.debug("Validating model access...")
        
        if not self.client or not self.api_key:
            self.result.add_issue(ValidationIssue(
                category=ValidationCategory.MODELS,
                level=ValidationLevel.WARNING,
                title="Cannot validate models",
                description="No authenticated client available for model validation",
                fix_suggestion="Ensure valid API key is configured"
            ))
            return
        
        # Test access to different model types
        model_tests = [
            ("command-light", "generation", "Basic generation model"),
            ("embed-english-v3.0", "embedding", "Embedding model"),
            ("rerank-english-v3.0", "rerank", "Rerank model")
        ]
        
        available_models = []
        model_errors = []
        
        for model_name, model_type, description in model_tests:
            try:
                if model_type == "generation":
                    response = self.client.chat(
                        model=model_name,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    available_models.append((model_name, description))
                    
                elif model_type == "embedding":
                    response = self.client.embed(
                        model=model_name,
                        texts=["test"],
                        input_type="classification"
                    )
                    available_models.append((model_name, description))
                    
                elif model_type == "rerank":
                    response = self.client.rerank(
                        model=model_name,
                        query="test",
                        documents=["test document"],
                        top_n=1
                    )
                    available_models.append((model_name, description))
                    
            except Exception as e:
                error_str = str(e).lower()
                if "not found" in error_str or "unavailable" in error_str:
                    model_errors.append(f"{model_name}: Model not available")
                elif "permission" in error_str or "access" in error_str:
                    model_errors.append(f"{model_name}: Access denied")
                else:
                    model_errors.append(f"{model_name}: {str(e)[:100]}")
        
        if available_models:
            self.result.add_passed_check("Model access")
            self.result.system_info['available_models'] = [f"{name} ({desc})" for name, desc in available_models]
            
            self.result.add_issue(ValidationIssue(
                category=ValidationCategory.MODELS,
                level=ValidationLevel.INFO,
                title=f"Models available: {len(available_models)}",
                description=f"Successfully tested {len(available_models)} model types"
            ))
        
        if model_errors:
            self.result.add_issue(ValidationIssue(
                category=ValidationCategory.MODELS,
                level=ValidationLevel.WARNING,
                title=f"Model access issues: {len(model_errors)}",
                description=f"Some models unavailable: {', '.join(model_errors[:3])}{'...' if len(model_errors) > 3 else ''}",
                fix_suggestion="Check API key permissions and model availability",
                technical_details="; ".join(model_errors)
            ))
        
        if not available_models:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.MODELS,
                level=ValidationLevel.ERROR,
                title="No models accessible",
                description="Cannot access any Cohere models with current API key",
                fix_suggestion="Verify API key has model access permissions"
            ))
    
    def _validate_pricing(self):
        """Validate pricing calculation setup."""
        logger.debug("Validating pricing calculations...")
        
        try:
            from .cohere_pricing import CohereCalculator
            calculator = CohereCalculator()
            
            # Test basic cost calculations
            test_cases = [
                ("command-r-plus-08-2024", "CHAT", 100, 50, 0),
                ("embed-english-v4.0", "EMBED", 100, 0, 10),
                ("rerank-english-v3.0", "RERANK", 0, 0, 1)
            ]
            
            successful_calculations = 0
            
            for model, operation, input_tokens, output_tokens, operation_units in test_cases:
                try:
                    input_cost, output_cost, op_cost = calculator.calculate_cost(
                        model=model,
                        operation=operation,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        operation_units=operation_units
                    )
                    
                    if input_cost >= 0 and output_cost >= 0 and op_cost >= 0:
                        successful_calculations += 1
                        
                except Exception as e:
                    logger.debug(f"Cost calculation failed for {model}: {e}")
            
            if successful_calculations == len(test_cases):
                self.result.add_passed_check("Pricing calculations")
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.PRICING,
                    level=ValidationLevel.INFO,
                    title="Pricing calculator working",
                    description=f"Successfully calculated costs for {len(test_cases)} model types"
                ))
            elif successful_calculations > 0:
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.PRICING,
                    level=ValidationLevel.WARNING,
                    title="Partial pricing support",
                    description=f"Cost calculations work for {successful_calculations}/{len(test_cases)} model types",
                    fix_suggestion="Check pricing data for unsupported models"
                ))
            else:
                self.result.add_failed_check(ValidationIssue(
                    category=ValidationCategory.PRICING,
                    level=ValidationLevel.ERROR,
                    title="Pricing calculations failed",
                    description="Cannot calculate costs for any model types",
                    fix_suggestion="Check pricing calculator implementation"
                ))
                
        except ImportError:
            self.result.add_failed_check(ValidationIssue(
                category=ValidationCategory.PRICING,
                level=ValidationLevel.ERROR,
                title="Pricing calculator missing",
                description="Cannot import Cohere pricing calculator module",
                fix_suggestion="Ensure cohere_pricing.py module is available"
            ))
    
    def _validate_performance(self):
        """Validate system performance characteristics."""
        logger.debug("Validating performance...")
        
        if not self.client or not self.api_key:
            return
        
        # Test response times for different operations
        performance_tests = [
            ("chat", lambda: self.client.chat(
                model="command-light",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )),
        ]
        
        for test_name, test_func in performance_tests:
            try:
                start_time = time.time()
                response = test_func()
                response_time = (time.time() - start_time) * 1000
                
                self.result.performance_metrics[f'{test_name}_response_time_ms'] = response_time
                
                if response_time < 2000:  # Under 2 seconds
                    self.result.add_passed_check(f"{test_name} performance")
                elif response_time < 5000:  # Under 5 seconds
                    self.result.add_issue(ValidationIssue(
                        category=ValidationCategory.PERFORMANCE,
                        level=ValidationLevel.WARNING,
                        title=f"Slow {test_name} response",
                        description=f"{test_name} took {response_time:.0f}ms, consider optimization",
                        technical_details=f"Response time: {response_time:.0f}ms"
                    ))
                else:
                    self.result.add_issue(ValidationIssue(
                        category=ValidationCategory.PERFORMANCE,
                        level=ValidationLevel.WARNING,
                        title=f"Very slow {test_name} response",
                        description=f"{test_name} took {response_time:.0f}ms, performance issue likely",
                        fix_suggestion="Check network latency and API server status"
                    ))
                    
            except Exception as e:
                self.result.add_issue(ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    level=ValidationLevel.WARNING,
                    title=f"Performance test failed: {test_name}",
                    description=f"Cannot run performance test: {str(e)}"
                ))
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Based on critical issues
        if self.result.has_critical_issues:
            recommendations.append("🚨 Address critical issues before proceeding with GenOps integration")
        
        # Based on missing dependencies
        missing_deps = [issue for issue in self.result.issues 
                       if issue.category == ValidationCategory.DEPENDENCIES 
                       and issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
        
        if missing_deps:
            recommendations.append("📦 Install missing dependencies to enable full functionality")
        
        # Based on authentication issues
        auth_issues = [issue for issue in self.result.issues 
                      if issue.category == ValidationCategory.AUTHENTICATION
                      and issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
        
        if auth_issues:
            recommendations.append("🔑 Configure valid Cohere API key for full integration testing")
        
        # Based on model access
        if not self.result.system_info.get('available_models'):
            recommendations.append("🤖 Verify API key has access to required Cohere models")
        
        # Based on performance
        slow_operations = [metric for metric, value in self.result.performance_metrics.items() 
                          if 'response_time' in metric and value > 3000]
        
        if slow_operations:
            recommendations.append("⚡ Consider optimizing slow API operations or checking network latency")
        
        # Success recommendations
        if self.result.success and not self.result.has_errors:
            recommendations.append("✅ Your setup looks good! You can proceed with GenOps Cohere integration")
            recommendations.append("📚 Check out the quickstart guide for next steps")
        
        self.result.recommendations = recommendations


def validate_setup(api_key: Optional[str] = None, **kwargs) -> ValidationResult:
    """
    Quick validation of Cohere integration setup.
    
    Args:
        api_key: Cohere API key (defaults to CO_API_KEY env var)
        **kwargs: Additional validation options
        
    Returns:
        Validation results
    """
    validator = CohereValidator(api_key=api_key, **kwargs)
    return validator.validate_all()


def quick_validate(api_key: Optional[str] = None) -> bool:
    """
    Quick validation that returns simple success/failure.
    
    Args:
        api_key: Cohere API key (defaults to CO_API_KEY env var)
        
    Returns:
        True if basic validation passes, False otherwise
    """
    validator = CohereValidator(
        api_key=api_key, 
        include_performance_tests=False
    )
    result = validator.validate_all()
    return result.success and not result.has_critical_issues


def print_validation_result(result: ValidationResult, detailed: bool = False):
    """
    Print validation results in a user-friendly format.
    
    Args:
        result: Validation results to print
        detailed: Whether to include detailed technical information
    """
    print("\n" + "="*60)
    print("🔍 GenOps Cohere Validation Results")
    print("="*60)
    
    # Overall status
    if result.success and not result.has_errors:
        print("✅ Overall Status: PASSED")
    elif result.has_critical_issues:
        print("🚨 Overall Status: CRITICAL ISSUES")
    elif result.has_errors:
        print("❌ Overall Status: ERRORS FOUND")
    else:
        print("⚠️ Overall Status: WARNINGS")
    
    print(f"📊 Score: {result.score:.1f}% ({result.passed_checks}/{result.total_checks} checks passed)")
    
    # System information
    if result.system_info:
        print(f"\n📋 System Information:")
        for key, value in result.system_info.items():
            if isinstance(value, list):
                if value:
                    print(f"  • {key}: {len(value)} items")
                    if detailed:
                        for item in value[:5]:  # Show first 5
                            print(f"    - {item}")
                        if len(value) > 5:
                            print(f"    - ... and {len(value) - 5} more")
            else:
                print(f"  • {key}: {value}")
    
    # Performance metrics
    if result.performance_metrics:
        print(f"\n⚡ Performance Metrics:")
        for key, value in result.performance_metrics.items():
            if isinstance(value, float):
                if 'time' in key or 'latency' in key:
                    print(f"  • {key}: {value:.1f}ms")
                else:
                    print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")
    
    # Issues by category
    if result.issues:
        print(f"\n🔍 Validation Issues:")
        
        categories = {}
        for issue in result.issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append(issue)
        
        for category, issues in categories.items():
            print(f"\n  {category.value.title()}:")
            for issue in issues:
                print(f"    {issue}")
                if issue.fix_suggestion:
                    print(f"      💡 Fix: {issue.fix_suggestion}")
                if detailed and issue.technical_details:
                    print(f"      🔧 Technical: {issue.technical_details}")
    
    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")
    
    print("\n" + "="*60)


# Export main classes and functions
__all__ = [
    "CohereValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "ValidationCategory",
    "validate_setup",
    "quick_validate", 
    "print_validation_result"
]