"""Registration and auto-instrumentation system for Ollama integration."""

import logging
import functools
from typing import Any, Optional, Dict, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global registry state
_is_registered = False
_adapter_instance: Optional[Any] = None
_original_methods: Dict[str, Callable] = {}


def auto_register() -> bool:
    """
    Automatically register Ollama provider with GenOps instrumentation system.
    
    Returns:
        True if registration successful, False otherwise
    """
    global _is_registered
    
    if _is_registered:
        logger.debug("Ollama provider already registered")
        return True
    
    try:
        # Try to import and register with the instrumentation system
        from genops.core.instrumentation import register_provider
        from .adapter import GenOpsOllamaAdapter
        
        # Create default adapter instance
        global _adapter_instance
        _adapter_instance = GenOpsOllamaAdapter()
        
        # Register with instrumentation system
        provider_info = {
            'name': 'ollama',
            'adapter_class': GenOpsOllamaAdapter,
            'adapter_instance': _adapter_instance,
            'auto_instrument_function': auto_instrument,
            'supported_operations': ['generate', 'chat', 'list_models'],
            'provider_type': 'local_model',
            'cost_model': 'infrastructure_based'
        }
        
        register_provider('ollama', provider_info)
        _is_registered = True
        
        logger.info("Successfully registered Ollama provider with GenOps instrumentation")
        return True
        
    except ImportError as e:
        logger.debug(f"Core instrumentation system not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to register Ollama provider: {e}")
        return False


def auto_instrument(
    ollama_base_url: str = "http://localhost:11434",
    resource_monitoring: bool = True,
    model_management: bool = True,
    **governance_defaults
) -> bool:
    """
    Enable automatic instrumentation of Ollama operations.
    
    This patches Ollama client operations to automatically add GenOps tracking.
    
    Args:
        ollama_base_url: Base URL for Ollama server
        resource_monitoring: Enable resource monitoring
        model_management: Enable model management features
        **governance_defaults: Default governance attributes
        
    Returns:
        True if instrumentation successful, False otherwise
        
    Usage:
        from genops.providers.ollama import auto_instrument
        auto_instrument(team="ai-research", project="local-models")
        
        # Your existing Ollama code now has automatic tracking
        import ollama
        response = ollama.generate(model="llama2", prompt="Hello")
    """
    try:
        # Check if ollama client is available
        import ollama
        HAS_OLLAMA_CLIENT = True
    except ImportError:
        logger.warning("Ollama client not available for auto-instrumentation")
        return False
    
    global _adapter_instance, _original_methods
    
    # Create or update adapter instance
    if _adapter_instance is None:
        from .adapter import GenOpsOllamaAdapter
        _adapter_instance = GenOpsOllamaAdapter(
            ollama_base_url=ollama_base_url,
            **governance_defaults
        )
    else:
        # Update existing adapter configuration
        _adapter_instance.ollama_base_url = ollama_base_url
        _adapter_instance.governance_defaults.update(governance_defaults)
    
    try:
        # Store original methods if not already stored
        if 'generate' not in _original_methods:
            _original_methods['generate'] = getattr(ollama, 'generate', None)
            _original_methods['chat'] = getattr(ollama, 'chat', None)
            _original_methods['Client.generate'] = getattr(ollama.Client, 'generate', None)
            _original_methods['Client.chat'] = getattr(ollama.Client, 'chat', None)
        
        # Create instrumented methods
        def instrumented_generate(model, prompt, **kwargs):
            """Instrumented generate method."""
            try:
                return _adapter_instance.generate(model=model, prompt=prompt, **kwargs)
            except Exception as e:
                logger.error(f"Error in instrumented generate: {e}")
                # Fallback to original method if available
                if _original_methods['generate']:
                    return _original_methods['generate'](model, prompt, **kwargs)
                raise
        
        def instrumented_chat(model, messages, **kwargs):
            """Instrumented chat method."""
            try:
                return _adapter_instance.chat(model=model, messages=messages, **kwargs)
            except Exception as e:
                logger.error(f"Error in instrumented chat: {e}")
                # Fallback to original method if available
                if _original_methods['chat']:
                    return _original_methods['chat'](model, messages, **kwargs)
                raise
        
        def instrumented_client_generate(self, model, prompt, **kwargs):
            """Instrumented client generate method."""
            try:
                # Create temporary adapter for this client instance
                from .adapter import GenOpsOllamaAdapter
                temp_adapter = GenOpsOllamaAdapter(
                    ollama_base_url=self.host,
                    **_adapter_instance.governance_defaults
                )
                return temp_adapter.generate(model=model, prompt=prompt, **kwargs)
            except Exception as e:
                logger.error(f"Error in instrumented client generate: {e}")
                # Fallback to original method
                if _original_methods['Client.generate']:
                    return _original_methods['Client.generate'](self, model, prompt, **kwargs)
                raise
        
        def instrumented_client_chat(self, model, messages, **kwargs):
            """Instrumented client chat method."""
            try:
                from .adapter import GenOpsOllamaAdapter
                temp_adapter = GenOpsOllamaAdapter(
                    ollama_base_url=self.host,
                    **_adapter_instance.governance_defaults
                )
                return temp_adapter.chat(model=model, messages=messages, **kwargs)
            except Exception as e:
                logger.error(f"Error in instrumented client chat: {e}")
                if _original_methods['Client.chat']:
                    return _original_methods['Client.chat'](self, model, messages, **kwargs)
                raise
        
        # Apply patches
        if hasattr(ollama, 'generate') and _original_methods['generate']:
            ollama.generate = instrumented_generate
        
        if hasattr(ollama, 'chat') and _original_methods['chat']:
            ollama.chat = instrumented_chat
        
        if hasattr(ollama.Client, 'generate') and _original_methods['Client.generate']:
            ollama.Client.generate = instrumented_client_generate
        
        if hasattr(ollama.Client, 'chat') and _original_methods['Client.chat']:
            ollama.Client.chat = instrumented_client_chat
        
        # Initialize resource monitoring if enabled
        if resource_monitoring:
            try:
                from .resource_monitor import get_resource_monitor
                monitor = get_resource_monitor()
                monitor.start_monitoring()
                logger.debug("Started Ollama resource monitoring")
            except Exception as e:
                logger.warning(f"Failed to start resource monitoring: {e}")
        
        # Initialize model management if enabled
        if model_management:
            try:
                from .model_manager import get_model_manager
                manager = get_model_manager()
                manager.discover_models()
                logger.debug("Initialized Ollama model management")
            except Exception as e:
                logger.warning(f"Failed to initialize model management: {e}")
        
        logger.info("GenOps auto-instrumentation enabled for Ollama")
        return True
        
    except Exception as e:
        logger.error(f"Failed to enable Ollama auto-instrumentation: {e}")
        return False


def disable_auto_instrument() -> bool:
    """
    Disable automatic instrumentation and restore original Ollama methods.
    
    Returns:
        True if restoration successful, False otherwise
    """
    global _original_methods
    
    if not _original_methods:
        logger.debug("No auto-instrumentation to disable")
        return True
    
    try:
        import ollama
        
        # Restore original methods
        if 'generate' in _original_methods and _original_methods['generate']:
            ollama.generate = _original_methods['generate']
        
        if 'chat' in _original_methods and _original_methods['chat']:
            ollama.chat = _original_methods['chat']
        
        if 'Client.generate' in _original_methods and _original_methods['Client.generate']:
            ollama.Client.generate = _original_methods['Client.generate']
        
        if 'Client.chat' in _original_methods and _original_methods['Client.chat']:
            ollama.Client.chat = _original_methods['Client.chat']
        
        # Clear stored methods
        _original_methods.clear()
        
        # Stop resource monitoring
        try:
            from .resource_monitor import get_resource_monitor
            monitor = get_resource_monitor()
            monitor.stop_monitoring()
        except Exception:
            pass  # Ignore errors during cleanup
        
        logger.info("Disabled GenOps auto-instrumentation for Ollama")
        return True
        
    except ImportError:
        logger.debug("Ollama client not available for restoration")
        return True
    except Exception as e:
        logger.error(f"Failed to disable Ollama auto-instrumentation: {e}")
        return False


@contextmanager
def instrumentation_context(**kwargs):
    """
    Context manager for temporary instrumentation.
    
    Usage:
        with instrumentation_context(team="research"):
            response = ollama.generate("llama2", "Hello")
            # Instrumentation automatically enabled and disabled
    """
    instrumentation_enabled = auto_instrument(**kwargs)
    
    try:
        yield instrumentation_enabled
    finally:
        if instrumentation_enabled:
            disable_auto_instrument()


def get_instrumentation_status() -> Dict[str, Any]:
    """
    Get current instrumentation status and configuration.
    
    Returns:
        Dictionary with instrumentation status information
    """
    global _is_registered, _adapter_instance
    
    status = {
        'registered': _is_registered,
        'auto_instrumentation_active': bool(_original_methods),
        'adapter_configured': _adapter_instance is not None,
        'ollama_client_available': False,
        'governance_defaults': {}
    }
    
    # Check Ollama client availability
    try:
        import ollama
        status['ollama_client_available'] = True
    except ImportError:
        pass
    
    # Get adapter configuration
    if _adapter_instance:
        status['governance_defaults'] = _adapter_instance.governance_defaults.copy()
        status['ollama_base_url'] = getattr(_adapter_instance, 'ollama_base_url', None)
        status['telemetry_enabled'] = getattr(_adapter_instance, 'telemetry_enabled', None)
        status['cost_tracking_enabled'] = getattr(_adapter_instance, 'cost_tracking_enabled', None)
    
    # Get monitoring status
    try:
        from .resource_monitor import get_resource_monitor
        monitor = get_resource_monitor()
        status['resource_monitoring_active'] = monitor.is_monitoring
    except Exception:
        status['resource_monitoring_active'] = False
    
    # Get model management status
    try:
        from .model_manager import get_model_manager
        manager = get_model_manager()
        status['models_discovered'] = len(manager.models)
    except Exception:
        status['models_discovered'] = 0
    
    return status


def reset_instrumentation() -> None:
    """Reset all instrumentation state (useful for testing)."""
    global _is_registered, _adapter_instance, _original_methods
    
    # Disable auto-instrumentation first
    disable_auto_instrument()
    
    # Reset global state
    _is_registered = False
    _adapter_instance = None
    _original_methods.clear()
    
    # Reset component instances
    try:
        from .resource_monitor import set_resource_monitor
        set_resource_monitor(None)
    except Exception:
        pass
    
    try:
        from .model_manager import set_model_manager  
        set_model_manager(None)
    except Exception:
        pass
    
    logger.debug("Reset Ollama instrumentation state")


# Export main functions
__all__ = [
    "auto_register",
    "auto_instrument",
    "disable_auto_instrument",
    "instrumentation_context",
    "get_instrumentation_status",
    "reset_instrumentation"
]