"""Auto-instrumentation system for GenOps AI governance."""

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)


class GenOpsInstrumentor:
    """Auto-instrumentation system for GenOps AI governance."""

    _instance: Optional["GenOpsInstrumentor"] = None
    _initialized = False

    def __new__(cls) -> "GenOpsInstrumentor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "patched_providers"):
            self.patched_providers: Dict[str, Any] = {}
            self.available_providers: Dict[str, bool] = {}
            self.provider_patches: Dict[str, Callable] = {}
            self._setup_provider_registry()

    def _setup_provider_registry(self):
        """Set up the registry of available provider patches."""
        from genops.providers.anthropic import patch_anthropic, unpatch_anthropic
        from genops.providers.openai import patch_openai, unpatch_openai

        self.provider_patches = {
            "openai": {
                "patch": patch_openai,
                "unpatch": unpatch_openai,
                "module": "openai",
            },
            "anthropic": {
                "patch": patch_anthropic,
                "unpatch": unpatch_anthropic,
                "module": "anthropic",
            },
        }

    def _detect_available_providers(self) -> Dict[str, bool]:
        """Detect which AI providers are installed and available."""
        available = {}

        for provider_name, config in self.provider_patches.items():
            try:
                importlib.import_module(config["module"])
                available[provider_name] = True
                logger.debug(f"✓ {provider_name} available for instrumentation")
            except ImportError:
                available[provider_name] = False
                logger.debug(f"✗ {provider_name} not available")

        return available

    def _setup_opentelemetry(
        self,
        service_name: str = "genops-ai-app",
        service_version: str = "0.1.0",
        environment: Optional[str] = None,
        exporter_type: str = "console",
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
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
        environment: Optional[str] = None,
        exporter_type: str = "console",
        otlp_endpoint: Optional[str] = None,
        otlp_headers: Optional[Dict[str, str]] = None,
        # Instrumentation configuration
        providers: Optional[List[str]] = None,
        auto_detect: bool = True,
        patch_all: bool = True,
        # Governance configuration
        default_team: Optional[str] = None,
        default_project: Optional[str] = None,
        default_environment: Optional[str] = None,
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

    def status(self) -> Dict[str, Any]:
        """Get the current instrumentation status."""
        return {
            "initialized": self._initialized,
            "instrumented_providers": list(self.patched_providers.keys()),
            "available_providers": self.available_providers,
            "default_attributes": getattr(self, "default_attributes", {}),
        }

    def get_default_attributes(self) -> Dict[str, str]:
        """Get default governance attributes for manual instrumentation."""
        return getattr(self, "default_attributes", {})

    def _check_provider_availability(self, provider_name: str) -> bool:
        """Check if a specific provider is available for instrumentation."""
        return self.available_providers.get(provider_name, False)

    def _instrument_provider(self, provider_name: str) -> bool:
        """Instrument a specific provider with GenOps governance."""
        if provider_name not in self.provider_patches:
            logger.warning(f"Unknown provider: {provider_name}")
            return False

        if not self._check_provider_availability(provider_name):
            logger.warning(f"Provider not available: {provider_name}")
            return False

        try:
            config = self.provider_patches[provider_name]
            config["patch"](auto_track=True)
            self.patched_providers[provider_name] = config
            logger.info(f"✓ {provider_name} instrumented with GenOps governance")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to instrument {provider_name}: {e}")
            return False


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


def status() -> Dict[str, Any]:
    """Get GenOps AI instrumentation status."""
    return _instrumentor.status()


def get_default_attributes() -> Dict[str, str]:
    """Get default governance attributes for manual instrumentation."""
    return _instrumentor.get_default_attributes()
