"""Validation system for Ollama integration setup and diagnostics."""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import ollama  # noqa: F401

    HAS_OLLAMA_CLIENT = True
except ImportError:
    HAS_OLLAMA_CLIENT = False


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
    PERFORMANCE = "performance"
    SECURITY = "security"
    MODELS = "models"


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
            ValidationLevel.CRITICAL: "🚨",
        }

        return f"{level_symbol[self.level]} {self.title}: {self.description}"


@dataclass
class ValidationResult:
    """Complete validation results."""

    success: bool
    total_checks: int = 0
    passed_checks: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

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


class OllamaValidator:
    """
    Comprehensive validation system for Ollama integration.

    Validates:
    - Dependency installation and versions
    - Ollama server connectivity and health
    - Model availability and performance
    - GenOps integration configuration
    - System requirements and resources
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 10.0,
        include_performance_tests: bool = True,
    ):
        """
        Initialize validator.

        Args:
            ollama_base_url: Base URL for Ollama server
            timeout: Request timeout in seconds
            include_performance_tests: Whether to run performance validation tests
        """
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.timeout = timeout
        self.include_performance_tests = include_performance_tests

        self.result = ValidationResult(success=True)

    def validate_all(self) -> ValidationResult:
        """
        Run complete validation suite.

        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive Ollama validation")

        # Core validation checks
        self._validate_dependencies()
        self._validate_configuration()
        self._validate_connectivity()
        self._validate_models()

        # Optional performance validation
        if self.include_performance_tests:
            self._validate_performance()

        # Security and best practices
        self._validate_security()

        # Generate recommendations
        self._generate_recommendations()

        logger.info(
            f"Validation completed: {self.result.score:.1f}% ({self.result.passed_checks}/{self.result.total_checks} checks passed)"
        )
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
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.CRITICAL,
                    title="Python Version Too Old",
                    description=f"Python {python_version.major}.{python_version.minor} detected, requires Python 3.8+",
                    fix_suggestion="Upgrade to Python 3.8 or later",
                )
            )

        # Check requests library
        if HAS_REQUESTS:
            self.result.add_passed_check("requests library")
        else:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.ERROR,
                    title="Missing requests library",
                    description="requests library is required for HTTP communication",
                    fix_suggestion="Install with: pip install requests",
                )
            )

        # Check Ollama client (optional but recommended)
        if HAS_OLLAMA_CLIENT:
            self.result.add_passed_check("ollama client")

            # Check ollama client version
            try:
                import ollama

                if hasattr(ollama, "__version__"):
                    self.result.system_info["ollama_client_version"] = (
                        ollama.__version__
                    )
            except Exception:
                pass
        else:
            self.result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.WARNING,
                    title="Ollama client not installed",
                    description="Ollama Python client provides better integration",
                    fix_suggestion="Install with: pip install ollama",
                )
            )

        # Check GenOps core dependencies
        try:
            from opentelemetry import trace  # noqa: F401

            self.result.add_passed_check("OpenTelemetry")
        except ImportError:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.DEPENDENCIES,
                    level=ValidationLevel.ERROR,
                    title="Missing OpenTelemetry",
                    description="OpenTelemetry is required for GenOps telemetry",
                    fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk",
                )
            )

    def _validate_configuration(self):
        """Validate configuration and environment."""
        logger.debug("Validating configuration...")

        # Check Ollama URL format
        if self.ollama_base_url.startswith(("http://", "https://")):
            self.result.add_passed_check("Ollama URL format")
        else:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    level=ValidationLevel.ERROR,
                    title="Invalid Ollama URL",
                    description=f"URL must start with http:// or https://: {self.ollama_base_url}",
                    fix_suggestion="Use format: http://localhost:11434 or https://your-ollama-server",
                )
            )

        # Check environment variables (optional but useful)
        env_vars = {
            "OLLAMA_HOST": "Ollama server host override",
            "OLLAMA_MODELS": "Ollama models directory",
        }

        for var, description in env_vars.items():
            value = os.getenv(var)
            if value:
                self.result.system_info[f"env_{var.lower()}"] = value
                self.result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.CONFIGURATION,
                        level=ValidationLevel.INFO,
                        title=f"Environment variable {var} set",
                        description=f"{description}: {value}",
                    )
                )

        # Check GenOps configuration
        genops_env_vars = {
            "GENOPS_TELEMETRY_ENABLED": "true",
            "GENOPS_COST_TRACKING_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": None,
        }

        for var, default in genops_env_vars.items():
            value = os.getenv(var, default)
            if value:
                self.result.system_info[f"genops_{var.lower()}"] = value

    def _validate_connectivity(self):
        """Validate Ollama server connectivity."""
        logger.debug("Validating Ollama server connectivity...")

        if not HAS_REQUESTS:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.ERROR,
                    title="Cannot test connectivity",
                    description="requests library not available for connectivity testing",
                    fix_suggestion="Install requests: pip install requests",
                )
            )
            return

        # Test basic connectivity
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.ollama_base_url}/api/version", timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                self.result.add_passed_check("Ollama server connectivity")
                self.result.performance_metrics["server_response_time_ms"] = (
                    response_time
                )

                # Get server version
                try:
                    version_info = response.json()
                    self.result.system_info["ollama_version"] = version_info.get(
                        "version", "unknown"
                    )
                except Exception:
                    pass

                # Test additional endpoints
                self._test_ollama_endpoints()

            else:
                self.result.add_failed_check(
                    ValidationIssue(
                        category=ValidationCategory.CONNECTIVITY,
                        level=ValidationLevel.ERROR,
                        title="Ollama server error",
                        description=f"Server returned HTTP {response.status_code}",
                        fix_suggestion="Check if Ollama server is running and accessible",
                        technical_details=f"GET {self.ollama_base_url}/api/version -> {response.status_code}",
                    )
                )

        except requests.exceptions.ConnectTimeout:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.CRITICAL,
                    title="Connection timeout",
                    description=f"Cannot connect to Ollama server at {self.ollama_base_url}",
                    fix_suggestion="Ensure Ollama is running: ollama serve",
                    technical_details=f"Timeout after {self.timeout}s",
                )
            )

        except requests.exceptions.ConnectionError:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.CRITICAL,
                    title="Connection refused",
                    description=f"Cannot connect to Ollama server at {self.ollama_base_url}",
                    fix_suggestion="Start Ollama server: ollama serve",
                    technical_details="Connection refused - server not running",
                )
            )

        except Exception as e:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.CONNECTIVITY,
                    level=ValidationLevel.ERROR,
                    title="Connection error",
                    description=f"Unexpected error connecting to Ollama: {str(e)}",
                    fix_suggestion="Check Ollama server status and network configuration",
                )
            )

    def _test_ollama_endpoints(self):
        """Test additional Ollama API endpoints."""
        endpoints = [("/api/tags", "Model listing"), ("/api/ps", "Running models")]

        for endpoint, description in endpoints:
            try:
                response = requests.get(
                    f"{self.ollama_base_url}{endpoint}", timeout=self.timeout
                )
                if response.status_code == 200:
                    self.result.add_passed_check(f"Ollama {description.lower()}")
                else:
                    self.result.add_issue(
                        ValidationIssue(
                            category=ValidationCategory.CONNECTIVITY,
                            level=ValidationLevel.WARNING,
                            title=f"{description} endpoint issue",
                            description=f"Endpoint {endpoint} returned HTTP {response.status_code}",
                            technical_details=f"GET {endpoint} -> {response.status_code}",
                        )
                    )
            except Exception as e:
                self.result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.CONNECTIVITY,
                        level=ValidationLevel.WARNING,
                        title=f"{description} endpoint error",
                        description=f"Cannot access {endpoint}: {str(e)}",
                    )
                )

    def _validate_models(self):
        """Validate available models."""
        logger.debug("Validating Ollama models...")

        if not HAS_REQUESTS:
            return

        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags", timeout=self.timeout
            )

            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])

                if models:
                    self.result.add_passed_check("Model availability")
                    self.result.system_info["available_models_count"] = len(models)
                    self.result.system_info["available_models"] = [
                        m.get("name", "unknown") for m in models
                    ]

                    # Check for common models
                    model_names = [m.get("name", "").lower() for m in models]
                    common_models = ["llama", "mistral", "codellama", "gemma"]

                    found_common = any(
                        common in " ".join(model_names) for common in common_models
                    )
                    if found_common:
                        self.result.add_issue(
                            ValidationIssue(
                                category=ValidationCategory.MODELS,
                                level=ValidationLevel.INFO,
                                title="Common models available",
                                description=f"Found {len(models)} models including popular ones",
                            )
                        )

                    # Check model sizes
                    total_size_gb = sum(m.get("size", 0) for m in models) / (1024**3)
                    self.result.performance_metrics["total_models_size_gb"] = (
                        total_size_gb
                    )

                else:
                    self.result.add_failed_check(
                        ValidationIssue(
                            category=ValidationCategory.MODELS,
                            level=ValidationLevel.WARNING,
                            title="No models available",
                            description="No models found on Ollama server",
                            fix_suggestion="Pull a model: ollama pull llama3.2",
                        )
                    )
            else:
                self.result.add_failed_check(
                    ValidationIssue(
                        category=ValidationCategory.MODELS,
                        level=ValidationLevel.ERROR,
                        title="Cannot list models",
                        description=f"Model listing returned HTTP {response.status_code}",
                        fix_suggestion="Check Ollama server status",
                    )
                )

        except Exception as e:
            self.result.add_failed_check(
                ValidationIssue(
                    category=ValidationCategory.MODELS,
                    level=ValidationLevel.ERROR,
                    title="Model validation error",
                    description=f"Error checking models: {str(e)}",
                )
            )

    def _validate_performance(self):
        """Validate system performance characteristics."""
        logger.debug("Validating performance...")

        # Test a simple generation if models are available
        if self.result.system_info.get("available_models_count", 0) > 0:
            self._test_simple_generation()

        # Check system resources
        self._check_system_resources()

    def _test_simple_generation(self):
        """Test simple text generation performance."""
        models = self.result.system_info.get("available_models", [])
        if not models:
            return

        # Use first available model for test
        test_model = models[0]
        test_prompt = "Hello"

        try:
            start_time = time.time()

            # Try with ollama client first
            if HAS_OLLAMA_CLIENT:
                import ollama

                client = ollama.Client(host=self.ollama_base_url)
                response = client.generate(
                    model=test_model, prompt=test_prompt, stream=False
                )
                generation_time = (time.time() - start_time) * 1000

                if response and response.get("response"):
                    self.result.add_passed_check("Text generation")
                    self.result.performance_metrics["test_generation_time_ms"] = (
                        generation_time
                    )

                    # Extract token metrics if available
                    if "eval_count" in response:
                        eval_count = response["eval_count"]
                        eval_duration = (
                            response.get("eval_duration", 0) / 1_000_000
                        )  # ns to ms
                        if eval_duration > 0:
                            tokens_per_second = (eval_count / eval_duration) * 1000
                            self.result.performance_metrics["tokens_per_second"] = (
                                tokens_per_second
                            )

                else:
                    self.result.add_issue(
                        ValidationIssue(
                            category=ValidationCategory.PERFORMANCE,
                            level=ValidationLevel.WARNING,
                            title="Generation test failed",
                            description="Model generation returned empty response",
                        )
                    )

            elif HAS_REQUESTS:
                # Fallback to HTTP API
                payload = {"model": test_model, "prompt": test_prompt, "stream": False}

                response = requests.post(
                    f"{self.ollama_base_url}/api/generate", json=payload, timeout=30
                )

                generation_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    self.result.add_passed_check("Text generation")
                    self.result.performance_metrics["test_generation_time_ms"] = (
                        generation_time
                    )
                else:
                    self.result.add_issue(
                        ValidationIssue(
                            category=ValidationCategory.PERFORMANCE,
                            level=ValidationLevel.WARNING,
                            title="Generation test HTTP error",
                            description=f"Generation test returned HTTP {response.status_code}",
                        )
                    )

        except Exception as e:
            self.result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    level=ValidationLevel.WARNING,
                    title="Performance test error",
                    description=f"Cannot run performance test: {str(e)}",
                    technical_details=f"Model: {test_model}, Error: {str(e)}",
                )
            )

    def _check_system_resources(self):
        """Check system resource availability."""
        try:
            import psutil

            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            self.result.performance_metrics["system_memory_gb"] = memory_gb

            if memory_gb >= 8:
                self.result.add_passed_check("System memory")
            else:
                self.result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.PERFORMANCE,
                        level=ValidationLevel.WARNING,
                        title="Low system memory",
                        description=f"Only {memory_gb:.1f}GB RAM available, recommend 8GB+ for local models",
                        fix_suggestion="Consider upgrading system memory for better performance",
                    )
                )

            # Check CPU
            cpu_count = psutil.cpu_count()
            self.result.performance_metrics["cpu_cores"] = cpu_count

            if cpu_count >= 4:
                self.result.add_passed_check("CPU cores")
            else:
                self.result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.PERFORMANCE,
                        level=ValidationLevel.WARNING,
                        title="Low CPU core count",
                        description=f"Only {cpu_count} CPU cores, recommend 4+ for good performance",
                    )
                )

        except ImportError:
            self.result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    level=ValidationLevel.INFO,
                    title="psutil not available",
                    description="Cannot check system resources without psutil",
                    fix_suggestion="Install psutil for system resource monitoring: pip install psutil",
                )
            )

    def _validate_security(self):
        """Validate security and best practices."""
        logger.debug("Validating security...")

        # Check if using HTTP (security concern)
        if self.ollama_base_url.startswith("http://"):
            if (
                "localhost" in self.ollama_base_url
                or "127.0.0.1" in self.ollama_base_url
            ):
                self.result.add_passed_check("Local connection security")
            else:
                self.result.add_issue(
                    ValidationIssue(
                        category=ValidationCategory.SECURITY,
                        level=ValidationLevel.WARNING,
                        title="Unencrypted remote connection",
                        description="Using HTTP for remote Ollama connection",
                        fix_suggestion="Use HTTPS for remote connections or ensure network is secure",
                    )
                )
        else:
            self.result.add_passed_check("Encrypted connection")

        # Check for production considerations
        if (
            "localhost" not in self.ollama_base_url
            and "127.0.0.1" not in self.ollama_base_url
        ):
            self.result.add_issue(
                ValidationIssue(
                    category=ValidationCategory.SECURITY,
                    level=ValidationLevel.INFO,
                    title="Remote Ollama server",
                    description="Using remote Ollama server",
                    fix_suggestion="Ensure network security and access controls are properly configured",
                )
            )

    def _generate_recommendations(self):
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Based on critical issues
        if self.result.has_critical_issues:
            recommendations.append(
                "🚨 Address critical issues before proceeding with GenOps integration"
            )

        # Based on missing dependencies
        missing_deps = [
            issue
            for issue in self.result.issues
            if issue.category == ValidationCategory.DEPENDENCIES
            and issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
        ]

        if missing_deps:
            recommendations.append(
                "📦 Install missing dependencies to enable full functionality"
            )

        # Based on model availability
        if self.result.system_info.get("available_models_count", 0) == 0:
            recommendations.append(
                "🤖 Pull at least one model to test GenOps integration: ollama pull llama3.2"
            )

        # Based on performance metrics
        memory_gb = self.result.performance_metrics.get("system_memory_gb", 0)
        if memory_gb > 0 and memory_gb < 8:
            recommendations.append(
                "💾 Consider upgrading to 8GB+ RAM for better model performance"
            )

        # Based on security
        security_issues = [
            issue
            for issue in self.result.issues
            if issue.category == ValidationCategory.SECURITY
        ]
        if security_issues:
            recommendations.append(
                "🔒 Review security recommendations for production deployment"
            )

        # Success recommendations
        if self.result.success and not self.result.has_errors:
            recommendations.append(
                "✅ Your setup looks good! You can proceed with GenOps Ollama integration"
            )
            recommendations.append("📚 Check out the quickstart guide for next steps")

        self.result.recommendations = recommendations


def validate_setup(
    ollama_base_url: str = "http://localhost:11434", **kwargs
) -> ValidationResult:
    """
    Quick validation of Ollama setup.

    Args:
        ollama_base_url: Ollama server URL
        **kwargs: Additional validation options

    Returns:
        Validation results
    """
    validator = OllamaValidator(ollama_base_url=ollama_base_url, **kwargs)
    return validator.validate_all()


def quick_validate(ollama_base_url: str = "http://localhost:11434") -> bool:
    """
    Quick validation that returns simple success/failure.

    Args:
        ollama_base_url: Ollama server URL

    Returns:
        True if basic validation passes, False otherwise
    """
    validator = OllamaValidator(
        ollama_base_url=ollama_base_url, include_performance_tests=False
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
    print("\n" + "=" * 60)
    print("🔍 GenOps Ollama Validation Results")
    print("=" * 60)

    # Overall status
    if result.success and not result.has_errors:
        print("✅ Overall Status: PASSED")
    elif result.has_critical_issues:
        print("🚨 Overall Status: CRITICAL ISSUES")
    elif result.has_errors:
        print("❌ Overall Status: ERRORS FOUND")
    else:
        print("⚠️ Overall Status: WARNINGS")

    print(
        f"📊 Score: {result.score:.1f}% ({result.passed_checks}/{result.total_checks} checks passed)"
    )

    # System information
    if result.system_info:
        print("\n📋 System Information:")
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
        print("\n⚡ Performance Metrics:")
        for key, value in result.performance_metrics.items():
            if isinstance(value, float):
                if "time" in key or "latency" in key:
                    print(f"  • {key}: {value:.1f}ms")
                elif "gb" in key:
                    print(f"  • {key}: {value:.1f}GB")
                elif "second" in key:
                    print(f"  • {key}: {value:.1f}")
                else:
                    print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")

    # Issues by category
    if result.issues:
        print("\n🔍 Validation Issues:")

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
        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  {rec}")

    print("\n" + "=" * 60)


# Export main classes and functions
__all__ = [
    "OllamaValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "ValidationCategory",
    "validate_setup",
    "quick_validate",
    "print_validation_result",
]
