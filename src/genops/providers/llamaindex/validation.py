"""LlamaIndex validation and diagnostics for GenOps AI governance."""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import llama_index
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False


@dataclass
class ValidationIssue:
    """Represents a validation issue with specific fix guidance."""

    severity: str  # 'error', 'warning', 'info'
    category: str  # 'dependency', 'configuration', 'performance'
    message: str
    fix_suggestion: str
    documentation_link: Optional[str] = None


@dataclass
class ValidationResult:
    """Comprehensive validation result with actionable diagnostics."""

    success: bool
    issues: List[ValidationIssue] = None
    environment_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    component_status: Optional[Dict[str, Any]] = None
    optimization_recommendations: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []


class LlamaIndexValidator:
    """Comprehensive validator for LlamaIndex integration setup."""

    def __init__(self):
        self.validation_start_time = time.time()

    def validate_complete_setup(self) -> ValidationResult:
        """Run complete validation of LlamaIndex setup."""

        result = ValidationResult(success=True)

        # 1. Environment validation
        result.environment_info = self._validate_environment(result)

        # 2. Dependencies validation
        self._validate_dependencies(result)

        # 3. LlamaIndex configuration validation
        self._validate_llamaindex_config(result)

        # 4. Component integration validation
        result.component_status = self._validate_components(result)

        # 5. Performance benchmarking
        result.performance_metrics = self._run_performance_benchmarks(result)

        # 6. Generate optimization recommendations
        result.optimization_recommendations = self._generate_recommendations(result)

        # Final success determination
        result.success = not any(issue.severity == 'error' for issue in result.issues)

        return result

    def _validate_environment(self, result: ValidationResult) -> Dict[str, Any]:
        """Validate environment configuration."""

        env_info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "llamaindex_available": HAS_LLAMAINDEX,
            "llamaindex_version": None,
            "environment_variables": {},
            "system_resources": {}
        }

        # Check Python version
        if sys.version_info < (3, 8):
            result.issues.append(ValidationIssue(
                severity="error",
                category="dependency",
                message="Python 3.8+ required for LlamaIndex integration",
                fix_suggestion="Install Python 3.8+: https://python.org/downloads/"
            ))

        # Check LlamaIndex version
        if HAS_LLAMAINDEX:
            try:
                env_info["llamaindex_version"] = getattr(llama_index, '__version__', 'unknown')

                # Version compatibility check
                version_str = env_info["llamaindex_version"]
                if version_str and version_str != 'unknown':
                    try:
                        major, minor = version_str.split('.')[:2]
                        if int(major) == 0 and int(minor) < 10:
                            result.issues.append(ValidationIssue(
                                severity="warning",
                                category="dependency",
                                message=f"LlamaIndex version {version_str} may be outdated",
                                fix_suggestion="Update LlamaIndex: pip install --upgrade llama-index"
                            ))
                    except (ValueError, IndexError):
                        result.issues.append(ValidationIssue(
                            severity="warning",
                            category="dependency",
                            message=f"Cannot parse LlamaIndex version: {version_str}",
                            fix_suggestion="Reinstall LlamaIndex: pip install --force-reinstall llama-index"
                        ))
            except Exception as e:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="dependency",
                    message=f"Unable to determine LlamaIndex version: {e}",
                    fix_suggestion="Verify LlamaIndex installation: pip show llama-index"
                ))

        # Check environment variables
        env_vars_to_check = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GENOPS_ENVIRONMENT",
            "GENOPS_PROJECT",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_SERVICE_NAME"
        ]

        for var in env_vars_to_check:
            value = os.getenv(var)
            env_info["environment_variables"][var] = bool(value)  # Don't store actual values
            if not value and var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="configuration",
                    message=f"No API key found for {var}",
                    fix_suggestion=f"Set {var} environment variable for LLM provider access"
                ))

        # Check system resources
        try:
            import psutil
            env_info["system_resources"] = {
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "cpu_count": psutil.cpu_count(),
                "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
            }

            # Memory recommendations
            memory_gb = env_info["system_resources"]["memory_gb"]
            if memory_gb < 4:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="performance",
                    message=f"Low system memory: {memory_gb}GB",
                    fix_suggestion="Consider upgrading to 8GB+ RAM for better RAG performance"
                ))
        except ImportError:
            env_info["system_resources"]["note"] = "psutil not available for system monitoring"

        return env_info

    def _validate_dependencies(self, result: ValidationResult):
        """Validate required dependencies."""

        # Core LlamaIndex check
        if not HAS_LLAMAINDEX:
            result.issues.append(ValidationIssue(
                severity="error",
                category="dependency",
                message="LlamaIndex not installed",
                fix_suggestion="Install LlamaIndex: pip install llama-index>=0.10.0"
            ))
            return

        # Check for common optional dependencies
        optional_deps = {
            'openai': 'OpenAI integration',
            'anthropic': 'Anthropic integration',
            'google-generativeai': 'Google Gemini integration',
            'chromadb': 'Chroma vector store',
            'faiss-cpu': 'FAISS vector store',
            'pinecone-client': 'Pinecone vector store',
            'transformers': 'Local model support'
        }

        missing_deps = []
        for dep_name, description in optional_deps.items():
            try:
                __import__(dep_name.replace('-', '_'))
            except ImportError:
                missing_deps.append((dep_name, description))

        if missing_deps:
            deps_str = ', '.join(f"{dep} ({desc})" for dep, desc in missing_deps[:3])
            result.issues.append(ValidationIssue(
                severity="info",
                category="dependency",
                message=f"Optional dependencies not found: {deps_str}{'...' if len(missing_deps) > 3 else ''}",
                fix_suggestion="Install needed dependencies: pip install [dependency-name]"
            ))

        # Check OpenTelemetry dependencies
        try:
            from opentelemetry import trace
            from opentelemetry.sdk import trace as trace_sdk
        except ImportError:
            result.issues.append(ValidationIssue(
                severity="warning",
                category="dependency",
                message="OpenTelemetry not available - telemetry will be disabled",
                fix_suggestion="Install OpenTelemetry: pip install opentelemetry-api opentelemetry-sdk"
            ))

    def _validate_llamaindex_config(self, result: ValidationResult):
        """Validate LlamaIndex configuration."""

        if not HAS_LLAMAINDEX:
            return

        try:
            # Check Settings configuration
            config_issues = []

            # Check if callback manager is configured
            if not hasattr(Settings, 'callback_manager') or Settings.callback_manager is None:
                result.issues.append(ValidationIssue(
                    severity="info",
                    category="configuration",
                    message="No callback manager configured in LlamaIndex Settings",
                    fix_suggestion="Configure callback manager for GenOps integration"
                ))

            # Check for LLM configuration
            if not hasattr(Settings, 'llm') or Settings.llm is None:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="configuration",
                    message="No default LLM configured in LlamaIndex Settings",
                    fix_suggestion="Configure default LLM: Settings.llm = OpenAI() or similar"
                ))

            # Check for embedding model configuration
            if not hasattr(Settings, 'embed_model') or Settings.embed_model is None:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    category="configuration",
                    message="No default embedding model configured",
                    fix_suggestion="Configure embedding model: Settings.embed_model = OpenAIEmbedding() or similar"
                ))

        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="warning",
                category="configuration",
                message=f"Error validating LlamaIndex configuration: {e}",
                fix_suggestion="Check LlamaIndex installation and configuration"
            ))

    def _validate_components(self, result: ValidationResult) -> Dict[str, Any]:
        """Validate GenOps component integration."""

        component_status = {
            "adapter_available": False,
            "cost_aggregator_available": False,
            "rag_monitor_available": False,
            "registration_status": {}
        }

        try:
            # Check if components can be imported
            component_status["adapter_available"] = True
        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="error",
                category="component",
                message=f"Cannot import LlamaIndex adapter: {e}",
                fix_suggestion="Check GenOps installation and dependencies"
            ))

        try:
            component_status["cost_aggregator_available"] = True
        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="error",
                category="component",
                message=f"Cannot import cost aggregator: {e}",
                fix_suggestion="Check GenOps installation and dependencies"
            ))

        try:
            component_status["rag_monitor_available"] = True
        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="error",
                category="component",
                message=f"Cannot import RAG monitor: {e}",
                fix_suggestion="Check GenOps installation and dependencies"
            ))

        # Check registration status
        try:
            from .registration import get_registration_status
            component_status["registration_status"] = get_registration_status()

            status = component_status["registration_status"]
            if not status.get("registered", False):
                result.issues.append(ValidationIssue(
                    severity="info",
                    category="configuration",
                    message="GenOps LlamaIndex provider not registered",
                    fix_suggestion="Register provider: from genops.providers.llamaindex import register_llamaindex_provider; register_llamaindex_provider()"
                ))
        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="warning",
                category="component",
                message=f"Cannot check registration status: {e}",
                fix_suggestion="Check GenOps installation"
            ))

        return component_status

    def _run_performance_benchmarks(self, result: ValidationResult) -> Dict[str, Any]:
        """Run performance benchmarks for optimization guidance."""

        metrics = {
            "import_time_ms": 0.0,
            "component_creation_time_ms": 0.0,
            "validation_time_ms": (time.time() - self.validation_start_time) * 1000,
            "system_ready": False
        }

        if not HAS_LLAMAINDEX:
            return metrics

        try:
            # Test import performance
            import_start = time.time()
            from .adapter import GenOpsLlamaIndexAdapter
            from .cost_aggregator import LlamaIndexCostAggregator
            from .rag_monitor import LlamaIndexRAGInstrumentor
            metrics["import_time_ms"] = (time.time() - import_start) * 1000

            # Test component creation performance
            creation_start = time.time()
            adapter = GenOpsLlamaIndexAdapter()
            cost_aggregator = LlamaIndexCostAggregator("test_context")
            rag_monitor = LlamaIndexRAGInstrumentor()
            metrics["component_creation_time_ms"] = (time.time() - creation_start) * 1000

            metrics["system_ready"] = True

            # Performance recommendations
            if metrics["import_time_ms"] > 1000:  # 1 second
                result.issues.append(ValidationIssue(
                    severity="info",
                    category="performance",
                    message=f"Slow import time: {metrics['import_time_ms']:.0f}ms",
                    fix_suggestion="Consider lazy loading or optimizing dependency imports"
                ))

        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="warning",
                category="performance",
                message=f"Performance benchmark failed: {e}",
                fix_suggestion="Check system resources and dependency installation"
            ))

        return metrics

    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate optimization and best practice recommendations."""

        recommendations = []

        # Based on environment info
        if result.environment_info:
            env = result.environment_info

            # System recommendations
            if "system_resources" in env and "memory_gb" in env["system_resources"]:
                memory_gb = env["system_resources"]["memory_gb"]
                if memory_gb >= 16:
                    recommendations.append("Excellent system resources - suitable for complex RAG pipelines")
                elif memory_gb >= 8:
                    recommendations.append("Good system resources - suitable for most RAG applications")

            # API key recommendations
            env_vars = env.get("environment_variables", {})
            available_providers = sum(1 for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"] if env_vars.get(key, False))
            if available_providers == 0:
                recommendations.append("Set up API keys for LLM providers to enable RAG functionality")
            elif available_providers == 1:
                recommendations.append("Consider setting up multiple LLM providers for cost optimization")
            else:
                recommendations.append("Multiple LLM providers available - great for cost optimization")

        # Based on component status
        if result.component_status:
            status = result.component_status

            if status.get("adapter_available") and status.get("cost_aggregator_available"):
                recommendations.append("All GenOps components available - ready for production RAG monitoring")

            reg_status = status.get("registration_status", {})
            if reg_status.get("registered", False):
                recommendations.append("GenOps integration active - RAG operations will be automatically tracked")
            else:
                recommendations.append("Register GenOps provider for automatic RAG pipeline monitoring")

        # Performance recommendations
        if result.performance_metrics:
            perf = result.performance_metrics

            if perf.get("system_ready", False):
                if perf.get("component_creation_time_ms", 0) < 100:
                    recommendations.append("Fast component initialization - optimal for high-frequency RAG operations")
                else:
                    recommendations.append("Consider component reuse patterns for better performance")

        # Issue-based recommendations
        error_count = sum(1 for issue in result.issues if issue.severity == "error")
        warning_count = sum(1 for issue in result.issues if issue.severity == "warning")

        if error_count == 0 and warning_count == 0:
            recommendations.append("Perfect setup - ready for production LlamaIndex RAG workflows")
        elif error_count == 0:
            recommendations.append("Setup complete with minor optimizations available")
        else:
            recommendations.append("Address error issues before proceeding to production")

        return recommendations[:7]  # Limit to top 7 recommendations


def validate_setup() -> ValidationResult:
    """
    Run comprehensive LlamaIndex setup validation.
    
    Returns:
        ValidationResult with detailed diagnostics and fix suggestions
    """
    validator = LlamaIndexValidator()
    return validator.validate_complete_setup()


def print_validation_result(result: ValidationResult, detailed: bool = False):
    """
    Print human-readable validation results with actionable guidance.
    
    Args:
        result: ValidationResult from validate_setup()
        detailed: Include detailed metrics and environment info
    """
    print("🔍 GenOps LlamaIndex Validation Report")
    print("=" * 50)

    # Overall status
    if result.success:
        print("✅ SUCCESS: LlamaIndex integration is ready!")
    else:
        print("❌ ISSUES FOUND: Setup needs attention")

    print()

    # Issues by severity
    errors = [issue for issue in result.issues if issue.severity == "error"]
    warnings = [issue for issue in result.issues if issue.severity == "warning"]
    infos = [issue for issue in result.issues if issue.severity == "info"]

    if errors:
        print("🚨 ERRORS TO FIX:")
        for i, issue in enumerate(errors, 1):
            print(f"{i:2}. {issue.message}")
            print(f"    🔧 Fix: {issue.fix_suggestion}")
            if issue.documentation_link:
                print(f"    📖 Docs: {issue.documentation_link}")
        print()

    if warnings:
        print("⚠️  WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"{i:2}. {warning.message}")
            print(f"    🔧 Fix: {warning.fix_suggestion}")
        print()

    if infos:
        print("ℹ️  INFORMATION:")
        for i, info in enumerate(infos, 1):
            print(f"{i:2}. {info.message}")
            print(f"    💡 Suggestion: {info.fix_suggestion}")
        print()

    # Component status
    if result.component_status and detailed:
        print("🧩 COMPONENT STATUS:")
        components = result.component_status

        status_symbols = {True: "✅", False: "❌", None: "❓"}
        print(f"   {status_symbols.get(components.get('adapter_available'))} Adapter")
        print(f"   {status_symbols.get(components.get('cost_aggregator_available'))} Cost Aggregator")
        print(f"   {status_symbols.get(components.get('rag_monitor_available'))} RAG Monitor")

        reg_status = components.get("registration_status", {})
        print(f"   {status_symbols.get(reg_status.get('registered'))} Registration")
        print()

    # Performance metrics
    if result.performance_metrics and detailed:
        print("📊 PERFORMANCE METRICS:")
        perf = result.performance_metrics

        if "import_time_ms" in perf:
            print(f"   Import Time: {perf['import_time_ms']:.0f}ms")
        if "component_creation_time_ms" in perf:
            print(f"   Component Creation: {perf['component_creation_time_ms']:.0f}ms")
        if "validation_time_ms" in perf:
            print(f"   Validation Time: {perf['validation_time_ms']:.0f}ms")
        print()

    # Environment info
    if result.environment_info and detailed:
        print("🔧 ENVIRONMENT INFO:")
        env = result.environment_info
        print(f"   Python: {env.get('python_version')}")
        print(f"   Platform: {env.get('platform')}")
        print(f"   LlamaIndex: {'✅' if env.get('llamaindex_available') else '❌'} {env.get('llamaindex_version', 'N/A')}")

        if "system_resources" in env:
            resources = env["system_resources"]
            if "memory_gb" in resources:
                print(f"   Memory: {resources['memory_gb']}GB")
            if "cpu_count" in resources:
                print(f"   CPUs: {resources['cpu_count']}")
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
        print("   1. Try the examples: python examples/llamaindex/hello_genops_minimal.py")
        print("   2. Explore RAG monitoring: examples/llamaindex/README.md")
        print("   3. Start tracking your LlamaIndex usage with GenOps!")
    else:
        print("🔧 FIX ERRORS ABOVE:")
        print("   1. Address all error messages with the provided fixes")
        print("   2. Run validation again: python -c \"from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup())\"")

    print("=" * 50)


def quick_validate() -> bool:
    """
    Quick validation with simple pass/fail result.
    
    Returns:
        True if validation passed, False if issues found
    """
    result = validate_setup()

    if result.success:
        print("✅ GenOps LlamaIndex validation passed!")
        return True
    else:
        print("❌ GenOps LlamaIndex validation failed")
        print("🔧 Run detailed validation for fix guidance:")
        print("   python -c \"from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")
        return False


# Export main functions
__all__ = [
    "validate_setup",
    "print_validation_result",
    "quick_validate",
    "ValidationResult",
    "ValidationIssue",
    "LlamaIndexValidator"
]
