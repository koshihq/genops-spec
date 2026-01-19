"""Auto-instrumentation system for GenOps AI governance."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)


class GenOpsInstrumentor:
    """Auto-instrumentation system for GenOps AI governance."""

    _instance: "GenOpsInstrumentor" | None = None
    _initialized = False

    def __new__(cls) -> "GenOpsInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "patched_providers"):
            self.patched_providers: dict[str, Any] = {}
            self.available_providers: dict[str, bool] = {}
            self.provider_patches: dict[str, Callable] = {}
            self._setup_provider_registry()

    def _setup_provider_registry(self):
        """Set up the registry of available provider patches."""
        from genops.providers.anthropic import patch_anthropic, unpatch_anthropic
        from genops.providers.openai import patch_openai, unpatch_openai
        from genops.providers.openrouter import patch_openrouter, unpatch_openrouter
        
        # Import Bedrock provider with error handling
        try:
            from genops.providers.bedrock import instrument_bedrock
            _bedrock_patch_available = True
        except ImportError:
            _bedrock_patch_available = False
        
        # Import Arize AI provider with error handling
        try:
            from genops.providers.arize import auto_instrument as arize_auto_instrument
            _arize_patch_available = True
        except ImportError:
            _arize_patch_available = False

        self.provider_patches = {
            "openai": {
                "patch": patch_openai,
                "unpatch": unpatch_openai,
                "module": "openai",
                "provider_type": "llm_api",
                "framework_type": "inference",
            },
            "anthropic": {
                "patch": patch_anthropic,
                "unpatch": unpatch_anthropic,
                "module": "anthropic",
                "provider_type": "llm_api",
                "framework_type": "inference",
            },
            "openrouter": {
                "patch": patch_openrouter,
                "unpatch": unpatch_openrouter,
                "module": "openai",  # OpenRouter uses OpenAI-compatible SDK
                "provider_type": "llm_api_gateway",
                "framework_type": "inference",
            },
        }
        
        # Add Bedrock to registry if available
        if _bedrock_patch_available:
            self.provider_patches["bedrock"] = {
                "patch": instrument_bedrock,
                "unpatch": lambda: None,  # Bedrock uses different instrumentation pattern
                "module": "boto3",
                "provider_type": "llm_api",
                "framework_type": "inference",
            }
        
        # Add Arize AI to registry if available
        if _arize_patch_available:
            self.provider_patches["arize"] = {
                "patch": lambda **kwargs: arize_auto_instrument(**kwargs),
                "unpatch": lambda: None,  # Arize uses different instrumentation pattern
                "module": "arize",
                "provider_type": "ml_observability",
                "framework_type": "monitoring",
            }
        
        # Add Dust AI to registry
        try:
            from genops.providers.dust import auto_instrument as dust_auto_instrument
            self.provider_patches["dust"] = {
                "patch": lambda **kwargs: dust_auto_instrument(**kwargs),
                "unpatch": lambda: None,  # Dust uses different instrumentation pattern
                "module": "requests",  # Dust uses requests for HTTP client
                "provider_type": "ai_agent_platform",
                "framework_type": "conversation",
            }
        except ImportError:
            pass  # Dust provider not available

        # Add Anyscale to registry
        try:
            from genops.providers.anyscale.registration import auto_instrument as anyscale_auto_instrument
            from genops.providers.anyscale.registration import disable_auto_instrument as anyscale_disable
            self.provider_patches["anyscale"] = {
                "patch": lambda **kwargs: anyscale_auto_instrument(**kwargs),
                "unpatch": anyscale_disable,
                "module": "requests",  # Anyscale uses requests or OpenAI SDK
                "provider_type": "llm_api",
                "framework_type": "inference",
            }
        except ImportError:
            pass  # Anyscale provider not available

        # Framework providers will be added dynamically as they're implemented
        self.framework_registry = {}

    def _detect_available_providers(self) -> dict[str, bool]:
        """Detect which AI providers and frameworks are installed and available."""
        available = {}

        # Detect existing LLM API providers
        for provider_name, config in self.provider_patches.items():
            try:
                importlib.import_module(config["module"])
                available[provider_name] = True
                logger.debug(f"✓ {provider_name} available for instrumentation")
            except ImportError:
                available[provider_name] = False
                logger.debug(f"✗ {provider_name} not available")

        # Detect frameworks using the FrameworkDetector
        try:
            from genops.providers.base import detect_frameworks
            framework_info = detect_frameworks()

            for name, info in framework_info.items():
                if info.available:
                    # Check if we have a provider implementation for this framework
                    if name in self.framework_registry:
                        available[name] = True
                        logger.debug(f"✓ {name} framework available for instrumentation")
                    else:
                        # Framework detected but no provider implementation yet
                        logger.debug(f"? {name} framework available but no provider implementation")

        except Exception as e:
            logger.debug(f"Framework detection failed: {e}")

        return available

    def _setup_opentelemetry(
        self,
        service_name: str = "genops-ai-app",
        service_version: str = "0.1.0",
        environment: str | None = None,
        exporter_type: str = "console",
        otlp_endpoint: str | None = None,
        otlp_headers: dict[str, str] | None = None,
    ) -> TracerProvider:
        """Set up OpenTelemetry tracing if not already configured."""

        # Check if OpenTelemetry is already configured
        current_tracer_provider = trace.get_tracer_provider()
        if hasattr(current_tracer_provider, "add_span_processor"):
            logger.debug("OpenTelemetry already configured, using existing provider")
            return current_tracer_provider

        # Create resource with service information
        resource_attrs = {
            "service.name": service_name,
            "service.version": service_version,
        }
        if environment:
            resource_attrs["deployment.environment"] = environment

        resource = Resource.create(resource_attrs)

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Configure exporter based on type
        if exporter_type == "console":
            exporter = ConsoleSpanExporter()
        elif exporter_type == "otlp":
            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint or "http://localhost:4317",
                headers=otlp_headers or {},
            )
        else:
            logger.warning(f"Unknown exporter type: {exporter_type}, using console")
            exporter = ConsoleSpanExporter()

        # Add span processor
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)

        logger.info(f"✓ OpenTelemetry configured with {exporter_type} exporter")
        return tracer_provider

    def instrument(
        self,
        # OpenTelemetry configuration
        service_name: str = "genops-ai-app",
        service_version: str = "0.1.0",
        environment: str | None = None,
        exporter_type: str = "console",
        otlp_endpoint: str | None = None,
        otlp_headers: dict[str, str] | None = None,
        # Instrumentation configuration
        providers: list[str] | None = None,
        auto_detect: bool = True,
        patch_all: bool = True,
        # Governance configuration
        default_team: str | None = None,
        default_project: str | None = None,
        default_environment: str | None = None,
    ) -> "GenOpsInstrumentor":
        """
        Auto-instrument available AI providers with GenOps governance.

        Args:
            service_name: Service name for OpenTelemetry
            service_version: Service version for OpenTelemetry
            environment: Deployment environment
            exporter_type: Type of exporter ("console", "otlp")
            otlp_endpoint: OTLP endpoint URL
            otlp_headers: OTLP headers
            providers: Specific providers to instrument (None = all available)
            auto_detect: Whether to auto-detect available providers
            patch_all: Whether to patch all detected providers
            default_team: Default team attribute for spans
            default_project: Default project attribute for spans
            default_environment: Default environment attribute for spans

        Returns:
            GenOpsInstrumentor: The instrumentation instance

        Example:
            import genops

            # Simple usage - auto-detect and instrument everything
            genops.init()

            # Advanced usage with configuration
            genops.init(
                service_name="my-ai-service",
                environment="production",
                exporter_type="otlp",
                otlp_endpoint="https://api.honeycomb.io",
                default_team="ai-team",
                default_project="chatbot"
            )
        """

        if self._initialized:
            logger.warning("GenOps already initialized, skipping")
            return self

        logger.info("🚀 Initializing GenOps AI auto-instrumentation...")

        # Set up OpenTelemetry
        self._setup_opentelemetry(
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            exporter_type=exporter_type,
            otlp_endpoint=otlp_endpoint,
            otlp_headers=otlp_headers,
        )

        # Detect available providers
        if auto_detect:
            self.available_providers = self._detect_available_providers()

        # Determine which providers to instrument
        if providers is None and patch_all:
            providers_to_patch = [
                name
                for name, available in self.available_providers.items()
                if available
            ]
        elif providers:
            providers_to_patch = [
                name
                for name in providers
                if name in self.provider_patches
                and self.available_providers.get(name, False)
            ]
        else:
            providers_to_patch = []

        # Apply provider patches
        instrumented_count = 0
        for provider_name in providers_to_patch:
            try:
                config = self.provider_patches[provider_name]
                config["patch"](auto_track=True)
                self.patched_providers[provider_name] = config
                instrumented_count += 1
                logger.info(f"✓ {provider_name} instrumented with GenOps governance")
            except Exception as e:
                logger.error(f"✗ Failed to instrument {provider_name}: {e}")

        # Store default governance attributes
        self.default_attributes = {
            k: v
            for k, v in {
                "team": default_team,
                "project": default_project,
                "environment": default_environment or environment,
            }.items()
            if v is not None
        }

        self._initialized = True

        logger.info("🎉 GenOps AI initialized successfully!")
        logger.info(f"   Instrumented providers: {instrumented_count}")
        logger.info(f"   Available providers: {list(self.available_providers.keys())}")
        logger.info(f"   Service: {service_name}")

        return self

    def uninstrument(self) -> None:
        """Remove all GenOps instrumentation patches."""
        if not self._initialized:
            logger.warning("GenOps not initialized, nothing to uninstrument")
            return

        logger.info("Removing GenOps instrumentation...")

        for provider_name, config in self.patched_providers.items():
            try:
                config["unpatch"]()
                logger.debug(f"✓ {provider_name} uninstrumented")
            except Exception as e:
                logger.error(f"✗ Failed to uninstrument {provider_name}: {e}")

        self.patched_providers.clear()
        self._initialized = False

        logger.info("✓ GenOps instrumentation removed")

    def status(self) -> dict[str, Any]:
        """Get the current instrumentation status."""
        return {
            "initialized": self._initialized,
            "instrumented_providers": list(self.patched_providers.keys()),
            "available_providers": self.available_providers,
            "default_attributes": getattr(self, "default_attributes", {}),
        }

    def get_default_attributes(self) -> dict[str, str]:
        """Get default governance attributes for manual instrumentation."""
        return getattr(self, "default_attributes", {})

    def _check_provider_availability(self, provider_name: str) -> bool:
        """Check if a specific provider is available for instrumentation."""
        return self.available_providers.get(provider_name, False)

    def _instrument_provider(self, provider_name: str) -> bool:
        """Instrument a specific provider with GenOps governance."""
        # Check both provider patches and framework registry
        config = None
        if provider_name in self.provider_patches:
            config = self.provider_patches[provider_name]
        elif provider_name in self.framework_registry:
            config = self.framework_registry[provider_name]

        if not config:
            logger.warning(f"Unknown provider or framework: {provider_name}")
            return False

        if not self._check_provider_availability(provider_name):
            logger.warning(f"Provider not available: {provider_name}")
            return False

        try:
            config["patch"](auto_track=True)
            self.patched_providers[provider_name] = config
            logger.info(f"✓ {provider_name} instrumented with GenOps governance")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to instrument {provider_name}: {e}")
            return False

    def register_framework_provider(
        self,
        name: str,
        patch_func: Callable,
        unpatch_func: Callable,
        module: str,
        framework_type: str,
        provider_class: Any | None = None,
        **metadata
    ) -> None:
        """
        Register a framework provider for auto-instrumentation.

        Args:
            name: Framework name (e.g., 'langchain', 'pytorch')
            patch_func: Function to apply instrumentation
            unpatch_func: Function to remove instrumentation
            module: Python module name to check for availability
            framework_type: Type of framework (orchestration, training, etc.)
            provider_class: Optional provider class reference
            **metadata: Additional metadata about the framework
        """
        self.framework_registry[name] = {
            "patch": patch_func,
            "unpatch": unpatch_func,
            "module": module,
            "framework_type": framework_type,
            "provider_type": "framework",
            "provider_class": provider_class,
            **metadata
        }

        logger.debug(f"Registered framework provider: {name}")

    def get_available_frameworks(self, framework_type: str | None = None) -> dict[str, dict]:
        """
        Get available frameworks, optionally filtered by type.

        Args:
            framework_type: Filter by framework type (optional)

        Returns:
            Dictionary of available frameworks with their info
        """
        try:
            from genops.providers.base import detect_frameworks
            framework_info = detect_frameworks()

            available = {}
            for name, info in framework_info.items():
                if info.available:
                    if framework_type is None or info.framework_type == framework_type:
                        available[name] = {
                            "name": info.name,
                            "version": info.version,
                            "framework_type": info.framework_type,
                            "instrumented": name in self.framework_registry,
                            "patched": name in self.patched_providers
                        }

            return available

        except Exception as e:
            logger.error(f"Failed to get available frameworks: {e}")
            return {}

    def get_framework_status(self) -> dict[str, Any]:
        """
        Get comprehensive status of all providers and frameworks.

        Returns:
            Dictionary with status information
        """
        status = self.status()

        # Add framework-specific information
        status["frameworks"] = {
            "registered": list(self.framework_registry.keys()),
            "available": self.get_available_frameworks(),
            "registry_count": len(self.framework_registry)
        }

        # Categorize by type
        status["providers_by_type"] = {}
        all_providers = {**self.provider_patches, **self.framework_registry}

        for name, config in all_providers.items():
            provider_type = config.get("provider_type", "unknown")
            framework_type = config.get("framework_type", "unknown")

            if provider_type not in status["providers_by_type"]:
                status["providers_by_type"][provider_type] = {}

            if framework_type not in status["providers_by_type"][provider_type]:
                status["providers_by_type"][provider_type][framework_type] = []

            status["providers_by_type"][provider_type][framework_type].append({
                "name": name,
                "available": self.available_providers.get(name, False),
                "instrumented": name in self.patched_providers
            })

        return status


# Global instance for convenient access
_instrumentor = GenOpsInstrumentor()


def init(**kwargs) -> GenOpsInstrumentor:
    """
    Initialize GenOps AI auto-instrumentation.

    This is the main entry point for GenOps AI governance instrumentation.
    It automatically detects available AI providers and instruments them with
    governance telemetry.

    Args:
        **kwargs: Configuration options passed to GenOpsInstrumentor.instrument()

    Returns:
        GenOpsInstrumentor: The instrumentation instance

    Example:
        import genops

        # Simple initialization
        genops.init()

        # Your existing AI code now has governance telemetry
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(...)  # Automatically tracked!
    """
    return _instrumentor.instrument(**kwargs)


def uninstrument() -> None:
    """Remove GenOps AI instrumentation."""
    _instrumentor.uninstrument()


def status() -> dict[str, Any]:
    """Get GenOps AI instrumentation status."""
    return _instrumentor.status()


def get_default_attributes() -> dict[str, str]:
    """Get default governance attributes for manual instrumentation."""
    return _instrumentor.get_default_attributes()


def register_framework_provider(**kwargs) -> None:
    """Register a framework provider for auto-instrumentation."""
    return _instrumentor.register_framework_provider(**kwargs)


def get_available_frameworks(framework_type: str | None = None) -> dict[str, dict]:
    """Get available frameworks, optionally filtered by type."""
    return _instrumentor.get_available_frameworks(framework_type)


def get_framework_status() -> dict[str, Any]:
    """Get comprehensive status of all providers and frameworks."""
    return _instrumentor.get_framework_status()
