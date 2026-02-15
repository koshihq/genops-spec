"""Auto-instrumentation registration for Anyscale provider."""

import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Global registry state
_is_registered = False
_adapter_instance: Optional[Any] = None
_original_methods: dict[str, Callable] = {}


def auto_instrument(**governance_defaults) -> bool:
    """
    Enable automatic instrumentation of Anyscale SDK.

    This function patches Anyscale SDK methods (if available) or OpenAI SDK
    methods to automatically track operations with GenOps governance.

    Args:
        **governance_defaults: Default governance attributes for all operations

    Returns:
        True if instrumentation successful, False otherwise

    Example:
        from genops.providers.anyscale.registration import auto_instrument

        auto_instrument(
            team="ml-research",
            project="chatbot",
            environment="production"
        )

        # Now all Anyscale API calls are automatically tracked
        import openai
        client = openai.OpenAI(
            api_key=os.getenv("ANYSCALE_API_KEY"),
            base_url="https://api.endpoints.anyscale.com/v1"
        )
        response = client.chat.completions.create(...)  # Tracked!
    """
    global _is_registered, _adapter_instance

    if _is_registered:
        logger.warning("Anyscale auto-instrumentation already enabled")
        return True

    try:
        # Import adapter
        from .adapter import GenOpsAnyscaleAdapter

        # Create adapter instance
        _adapter_instance = GenOpsAnyscaleAdapter(**governance_defaults)

        # Check if OpenAI SDK is available (Anyscale is OpenAI-compatible)
        try:
            from openai import OpenAI  # noqa: F401
            from openai.resources.chat import completions as chat_completions_module

            # Store original chat.completions.create method
            if "chat.completions.create" not in _original_methods:
                original_create = chat_completions_module.Completions.create
                _original_methods["chat.completions.create"] = original_create

                @functools.wraps(original_create)
                def _instrumented_create(self, *args, **kwargs):
                    """Instrumented completion with GenOps tracking."""
                    # Check if this is an Anyscale endpoint (by base_url)
                    base_url = (
                        getattr(self._client, "_base_url", None)
                        if hasattr(self, "_client")
                        else None
                    )
                    if base_url and "anyscale" in str(base_url).lower():
                        # Extract governance attributes
                        gov_attrs = {}
                        for key in [
                            "team",
                            "project",
                            "customer_id",
                            "environment",
                            "cost_center",
                            "feature",
                        ]:
                            if key in kwargs:
                                gov_attrs[key] = kwargs.pop(key)

                        # Merge with default governance attributes
                        if _adapter_instance:
                            final_gov_attrs = {
                                **_adapter_instance.governance_defaults,
                                **gov_attrs,
                            }

                            # Extract model and messages from args/kwargs
                            model = kwargs.get("model") or (
                                args[0] if len(args) > 0 else None
                            )
                            messages = kwargs.get("messages") or (
                                args[1] if len(args) > 1 else []
                            )

                            # Route through GenOps adapter for tracking
                            try:
                                return _adapter_instance.completion_create(
                                    model=model,
                                    messages=messages,
                                    **{
                                        k: v
                                        for k, v in kwargs.items()
                                        if k not in ["model", "messages"]
                                    },
                                    **final_gov_attrs,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"GenOps tracking failed, falling back to original method: {e}"
                                )
                                # Fall back to original method if tracking fails
                                return original_create(self, *args, **kwargs)

                    # Not an Anyscale endpoint, use original method
                    return original_create(self, *args, **kwargs)

                # Apply patch
                chat_completions_module.Completions.create = _instrumented_create
                logger.info(
                    "Anyscale auto-instrumentation enabled (OpenAI SDK patched)"
                )

            _is_registered = True
            return True

        except ImportError:
            logger.debug("OpenAI SDK not available, using manual instrumentation only")
            _is_registered = True
            return True

    except Exception as e:
        logger.error(f"Failed to enable Anyscale auto-instrumentation: {e}")
        return False


def disable_auto_instrument() -> bool:
    """
    Disable automatic instrumentation and restore original methods.

    Returns:
        True if uninstrumentation successful, False otherwise
    """
    global _is_registered, _adapter_instance, _original_methods

    if not _is_registered:
        logger.warning("Anyscale auto-instrumentation not enabled")
        return True

    try:
        # Restore original methods if any were patched
        if "chat.completions.create" in _original_methods:
            try:
                from openai.resources.chat import completions as chat_completions_module

                chat_completions_module.Completions.create = _original_methods[
                    "chat.completions.create"
                ]
                logger.debug("Restored original OpenAI chat.completions.create method")
            except ImportError:
                logger.debug("OpenAI SDK not available, nothing to unpatch")

        _is_registered = False
        _adapter_instance = None
        _original_methods.clear()

        logger.info("Anyscale auto-instrumentation disabled")
        return True

    except Exception as e:
        logger.error(f"Failed to disable Anyscale auto-instrumentation: {e}")
        return False


def register_anyscale_provider(instrumentor: "GenOpsInstrumentor") -> None:  # type: ignore  # noqa: F821
    """
    Register Anyscale provider with the auto-instrumentation system.

    Args:
        instrumentor: GenOpsInstrumentor instance

    Example:
        from genops.auto_instrumentation import _instrumentor
        from genops.providers.anyscale.registration import register_anyscale_provider

        register_anyscale_provider(_instrumentor)
    """
    from .adapter import GenOpsAnyscaleAdapter

    try:
        instrumentor.register_framework_provider(
            name="anyscale",
            patch_func=auto_instrument,
            unpatch_func=disable_auto_instrument,
            module="openai",  # Check for OpenAI SDK since Anyscale is compatible
            framework_type="inference",
            provider_class=GenOpsAnyscaleAdapter,
            description="Anyscale managed LLM endpoints",
            capabilities=[
                "openai_compatible_api",
                "cost_tracking",
                "multi_model_support",
                "chat_completions",
                "embeddings",
                "governance_attribution",
            ],
        )
        logger.debug("Anyscale provider registered with GenOps instrumentation system")

    except Exception as e:
        logger.warning(f"Failed to register Anyscale provider: {e}")


def get_adapter_instance() -> Optional["GenOpsAnyscaleAdapter"]:  # type: ignore  # noqa: F821
    """
    Get the current adapter instance (if auto-instrumentation is enabled).

    Returns:
        GenOpsAnyscaleAdapter instance or None
    """
    return _adapter_instance


# Export public API
__all__ = [
    "auto_instrument",
    "disable_auto_instrument",
    "register_anyscale_provider",
    "get_adapter_instance",
]
