"""LlamaIndex registration and auto-instrumentation for GenOps AI governance."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    logger.warning("LlamaIndex not available for registration")

# Import our components
from .adapter import GenOpsLlamaIndexAdapter
from .cost_aggregator import LlamaIndexCostAggregator, set_cost_aggregator
from .rag_monitor import LlamaIndexRAGInstrumentor, set_rag_monitor


class LlamaIndexInstrumentationRegistry:
    """
    Registry for LlamaIndex instrumentation components.
    
    Manages automatic discovery and registration of GenOps components
    with LlamaIndex's callback system.
    """

    def __init__(self):
        self.is_registered = False
        self.adapter: Optional[GenOpsLlamaIndexAdapter] = None
        self.cost_aggregator: Optional[LlamaIndexCostAggregator] = None
        self.rag_monitor: Optional[LlamaIndexRAGInstrumentor] = None
        self._original_settings = {}

    def register(
        self,
        enable_cost_tracking: bool = True,
        enable_rag_monitoring: bool = True,
        enable_telemetry: bool = True,
        **governance_defaults
    ) -> bool:
        """
        Register GenOps instrumentation with LlamaIndex.
        
        Args:
            enable_cost_tracking: Enable cost aggregation
            enable_rag_monitoring: Enable RAG pipeline monitoring
            enable_telemetry: Enable OpenTelemetry export
            **governance_defaults: Default governance attributes
            
        Returns:
            True if registration successful, False otherwise
        """
        if not HAS_LLAMAINDEX:
            logger.warning("Cannot register LlamaIndex instrumentation - LlamaIndex not available")
            return False

        if self.is_registered:
            logger.debug("LlamaIndex instrumentation already registered")
            return True

        try:
            # Create adapter
            self.adapter = GenOpsLlamaIndexAdapter(
                telemetry_enabled=enable_telemetry,
                cost_tracking_enabled=enable_cost_tracking,
                **governance_defaults
            )

            # Create cost aggregator if enabled
            if enable_cost_tracking:
                self.cost_aggregator = LlamaIndexCostAggregator(
                    context_name="global_llamaindex",
                    **governance_defaults
                )
                set_cost_aggregator(self.cost_aggregator)

                # Connect cost aggregator to adapter
                self.adapter.cost_aggregator = self.cost_aggregator

            # Create RAG monitor if enabled
            if enable_rag_monitoring:
                self.rag_monitor = LlamaIndexRAGInstrumentor(
                    enable_cost_tracking=enable_cost_tracking,
                    enable_quality_metrics=True,
                    enable_performance_profiling=True
                )
                set_rag_monitor(self.rag_monitor)

            # Register with LlamaIndex Settings
            self._register_with_settings()

            self.is_registered = True
            logger.info("GenOps LlamaIndex instrumentation registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register LlamaIndex instrumentation: {e}")
            return False

    def _register_with_settings(self) -> None:
        """Register callback handler with LlamaIndex Settings."""
        if not self.adapter:
            return

        # Store original callback manager
        if hasattr(Settings, 'callback_manager'):
            self._original_settings['callback_manager'] = Settings.callback_manager

        # Add our callback handler to Settings
        if Settings.callback_manager is None:
            Settings.callback_manager = CallbackManager([self.adapter.callback_handler])
        else:
            # Add to existing callback manager if not already present
            existing_handlers = Settings.callback_manager.handlers
            if self.adapter.callback_handler not in existing_handlers:
                existing_handlers.append(self.adapter.callback_handler)

    def unregister(self) -> bool:
        """
        Unregister GenOps instrumentation from LlamaIndex.
        
        Returns:
            True if unregistration successful, False otherwise
        """
        if not self.is_registered:
            logger.debug("LlamaIndex instrumentation not registered")
            return True

        try:
            # Restore original Settings
            if 'callback_manager' in self._original_settings:
                Settings.callback_manager = self._original_settings['callback_manager']
            elif hasattr(Settings, 'callback_manager') and Settings.callback_manager:
                # Remove our callback handler
                if self.adapter and self.adapter.callback_handler in Settings.callback_manager.handlers:
                    Settings.callback_manager.handlers.remove(self.adapter.callback_handler)

            # Clear global references
            set_cost_aggregator(None)
            set_rag_monitor(None)

            # Reset instance state
            self.adapter = None
            self.cost_aggregator = None
            self.rag_monitor = None
            self.is_registered = False
            self._original_settings.clear()

            logger.info("GenOps LlamaIndex instrumentation unregistered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister LlamaIndex instrumentation: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current registration status and component health."""
        return {
            "registered": self.is_registered,
            "llamaindex_available": HAS_LLAMAINDEX,
            "components": {
                "adapter": self.adapter is not None,
                "cost_aggregator": self.cost_aggregator is not None,
                "rag_monitor": self.rag_monitor is not None
            },
            "settings_integration": {
                "callback_manager_configured": (
                    hasattr(Settings, 'callback_manager') and
                    Settings.callback_manager is not None and
                    self.adapter and
                    self.adapter.callback_handler in Settings.callback_manager.handlers
                ) if HAS_LLAMAINDEX else False
            }
        }


# Global registry instance
_registry = LlamaIndexInstrumentationRegistry()


def register_llamaindex_provider(
    enable_cost_tracking: bool = True,
    enable_rag_monitoring: bool = True,
    enable_telemetry: bool = True,
    **governance_defaults
) -> bool:
    """
    Register LlamaIndex provider with GenOps instrumentation.
    
    Args:
        enable_cost_tracking: Enable cost aggregation
        enable_rag_monitoring: Enable RAG pipeline monitoring
        enable_telemetry: Enable OpenTelemetry export
        **governance_defaults: Default governance attributes (team, project, etc.)
        
    Returns:
        True if registration successful, False otherwise
        
    Example:
        # Register with default settings
        register_llamaindex_provider()
        
        # Register with governance defaults
        register_llamaindex_provider(
            team="ai-research",
            project="rag-system",
            enable_cost_tracking=True
        )
    """
    return _registry.register(
        enable_cost_tracking=enable_cost_tracking,
        enable_rag_monitoring=enable_rag_monitoring,
        enable_telemetry=enable_telemetry,
        **governance_defaults
    )


def unregister_llamaindex_provider() -> bool:
    """
    Unregister LlamaIndex provider from GenOps instrumentation.
    
    Returns:
        True if unregistration successful, False otherwise
    """
    return _registry.unregister()


def get_registration_status() -> Dict[str, Any]:
    """
    Get current LlamaIndex provider registration status.
    
    Returns:
        Dictionary with registration status and component health
    """
    return _registry.get_status()


def auto_register() -> None:
    """
    Automatically register LlamaIndex provider if LlamaIndex is available.
    
    This function is called automatically when the llamaindex provider
    module is imported. It provides zero-configuration setup for basic
    cost tracking and telemetry.
    """
    if not HAS_LLAMAINDEX:
        logger.debug("LlamaIndex not available, skipping auto-registration")
        return

    try:
        # Check if we should auto-register (can be controlled by environment variable)
        import os
        if os.getenv("GENOPS_LLAMAINDEX_AUTO_REGISTER", "true").lower() in ("true", "1", "yes"):
            success = register_llamaindex_provider(
                enable_cost_tracking=True,
                enable_rag_monitoring=True,
                enable_telemetry=True
            )

            if success:
                logger.debug("LlamaIndex provider auto-registered with GenOps")
            else:
                logger.debug("LlamaIndex provider auto-registration failed")
        else:
            logger.debug("LlamaIndex provider auto-registration disabled")

    except Exception as e:
        logger.debug(f"LlamaIndex provider auto-registration error: {e}")


def patch_llamaindex() -> bool:
    """
    Apply patches to LlamaIndex for enhanced instrumentation.
    
    This function applies monkey patches to key LlamaIndex components
    to enable automatic cost tracking and governance without code changes.
    
    Returns:
        True if patching successful, False otherwise
    """
    if not HAS_LLAMAINDEX:
        logger.warning("Cannot patch LlamaIndex - not available")
        return False

    if _registry.is_registered:
        logger.debug("LlamaIndex already instrumented via registration")
        return True

    try:
        # Ensure we have an adapter
        if not _registry.adapter:
            success = register_llamaindex_provider()
            if not success:
                return False

        # The registration process already handles callback integration
        # Additional patching could be added here if needed for specific
        # LlamaIndex components that don't use the callback system

        logger.info("LlamaIndex patching completed")
        return True

    except Exception as e:
        logger.error(f"Failed to patch LlamaIndex: {e}")
        return False


def unpatch_llamaindex() -> bool:
    """
    Remove patches from LlamaIndex.
    
    Returns:
        True if unpatching successful, False otherwise
    """
    return unregister_llamaindex_provider()


def get_adapter() -> Optional[GenOpsLlamaIndexAdapter]:
    """Get the current LlamaIndex adapter instance."""
    return _registry.adapter


def get_cost_aggregator_instance() -> Optional[LlamaIndexCostAggregator]:
    """Get the current cost aggregator instance."""
    return _registry.cost_aggregator


def get_rag_monitor_instance() -> Optional[LlamaIndexRAGInstrumentor]:
    """Get the current RAG monitor instance."""
    return _registry.rag_monitor


# Compatibility with framework detection
def is_llamaindex_available() -> bool:
    """Check if LlamaIndex is available for instrumentation."""
    return HAS_LLAMAINDEX


def get_llamaindex_version() -> Optional[str]:
    """Get the installed LlamaIndex version."""
    if not HAS_LLAMAINDEX:
        return None

    try:
        import llama_index
        return getattr(llama_index, '__version__', 'unknown')
    except Exception:
        return 'unknown'


def validate_llamaindex_setup() -> Dict[str, Any]:
    """
    Validate LlamaIndex setup for GenOps integration.
    
    Returns:
        Dictionary with validation results and recommendations
    """
    results = {
        "llamaindex_installed": HAS_LLAMAINDEX,
        "version": get_llamaindex_version(),
        "registration_status": get_registration_status(),
        "issues": [],
        "recommendations": []
    }

    if not HAS_LLAMAINDEX:
        results["issues"].append("LlamaIndex not installed")
        results["recommendations"].append("Install LlamaIndex: pip install llama-index>=0.10.0")
    else:
        # Check version compatibility
        version = get_llamaindex_version()
        if version and version != 'unknown':
            # Add version-specific checks if needed
            pass

        # Check registration status
        status = get_registration_status()
        if not status["registered"]:
            results["recommendations"].append("Register LlamaIndex provider: register_llamaindex_provider()")

        if not status["components"]["cost_aggregator"]:
            results["recommendations"].append("Enable cost tracking for comprehensive governance")

        if not status["components"]["rag_monitor"]:
            results["recommendations"].append("Enable RAG monitoring for pipeline optimization")

    return results


# Export main functions
__all__ = [
    "register_llamaindex_provider",
    "unregister_llamaindex_provider",
    "get_registration_status",
    "auto_register",
    "patch_llamaindex",
    "unpatch_llamaindex",
    "get_adapter",
    "get_cost_aggregator_instance",
    "get_rag_monitor_instance",
    "is_llamaindex_available",
    "get_llamaindex_version",
    "validate_llamaindex_setup"
]
