"""OpenAI provider adapter for GenOps AI governance."""

import logging
from typing import Any, Optional, Tuple

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    logger.warning("OpenAI not installed. Install with: pip install openai")


class GenOpsOpenAIAdapter:
    """OpenAI adapter with automatic governance telemetry."""

    def __init__(self, client: Optional[Any] = None, **client_kwargs):
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package not found. Install with: pip install openai"
            )

        self.client = client or OpenAI(**client_kwargs)
        self.telemetry = GenOpsTelemetry()
        
        # Define governance and request attributes
        self.GOVERNANCE_ATTRIBUTES = {
            'team', 'project', 'feature', 'customer_id', 'customer', 
            'environment', 'cost_center', 'user_id'
        }
        self.REQUEST_ATTRIBUTES = {
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty',
            'presence_penalty', 'stop', 'seed', 'stream'
        }

    def _extract_attributes(self, kwargs: dict) -> Tuple[dict, dict, dict]:
        """Extract governance and request attributes from kwargs."""
        governance_attrs = {}
        request_attrs = {}
        api_kwargs = kwargs.copy()
        
        # Extract governance attributes
        for attr in self.GOVERNANCE_ATTRIBUTES:
            if attr in kwargs:
                governance_attrs[attr] = kwargs[attr]
                api_kwargs.pop(attr)
        
        # Extract request attributes
        for attr in self.REQUEST_ATTRIBUTES:
            if attr in kwargs:
                request_attrs[attr] = kwargs[attr]
        
        return governance_attrs, request_attrs, api_kwargs

    def chat_completions_create(self, **kwargs) -> Any:
        """Create chat completion with governance tracking."""
        # Extract attributes from kwargs
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        model = api_kwargs.get("model", "unknown")
        messages = api_kwargs.get("messages", [])

        # Estimate input tokens (rough approximation)
        input_text = " ".join(
            [msg.get("content", "") for msg in messages if isinstance(msg, dict)]
        )
        estimated_input_tokens = len(input_text.split()) * 1.3  # rough token estimate

        operation_name = "openai.chat.completions.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.inference",
            "provider": "openai",
            "model": model,
            "tokens_estimated_input": int(estimated_input_tokens),
        }
        
        # Add default attributes from instrumentation system
        try:
            from genops.auto_instrumentation import get_default_attributes
            default_attrs = get_default_attributes() or {}
            trace_attrs.update(default_attrs)
        except (ImportError, Exception):
            pass  # Fallback if not available
            
        trace_attrs.update(governance_attrs)

        with self.telemetry.trace_operation(**trace_attrs) as span:
            # Record request parameters in telemetry
            for param, value in request_attrs.items():
                span.set_attribute(f"genops.request.{param}", value)
            
            try:
                # Call OpenAI API with cleaned kwargs (no governance attributes)
                response = self.client.chat.completions.create(**api_kwargs)

                # Extract usage and cost information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Calculate cost based on model pricing (simplified)
                    cost = self._calculate_cost(model, input_tokens, output_tokens)

                    # Record telemetry
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider="openai",
                        model=model,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens,
                        tokens_total=total_tokens,
                    )

                return response

            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise

    def completions_create(self, **kwargs) -> Any:
        """Create completion with governance tracking."""
        # Extract attributes from kwargs
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        model = api_kwargs.get("model", "unknown")
        prompt = api_kwargs.get("prompt", "")

        # Estimate input tokens
        estimated_input_tokens = len(str(prompt).split()) * 1.3

        operation_name = "openai.completions.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.inference",
            "provider": "openai",
            "model": model,
            "tokens_estimated_input": int(estimated_input_tokens),
        }
        
        # Add default attributes from instrumentation system
        try:
            from genops.auto_instrumentation import get_default_attributes
            default_attrs = get_default_attributes() or {}
            trace_attrs.update(default_attrs)
        except (ImportError, Exception):
            pass  # Fallback if not available
            
        trace_attrs.update(governance_attrs)

        with self.telemetry.trace_operation(**trace_attrs) as span:
            # Record request parameters in telemetry
            for param, value in request_attrs.items():
                span.set_attribute(f"genops.request.{param}", value)
            
            try:
                # Call OpenAI API with cleaned kwargs (no governance attributes)
                response = self.client.completions.create(**api_kwargs)

                # Extract usage and cost information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Calculate cost
                    cost = self._calculate_cost(model, input_tokens, output_tokens)

                    # Record telemetry
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider="openai",
                        model=model,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens,
                        tokens_total=total_tokens,
                    )

                return response

            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate estimated cost based on OpenAI pricing."""
        # Simplified pricing - in production, use real pricing API or config
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
            "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
            "text-davinci-003": {"input": 0.02 / 1000, "output": 0.02 / 1000},
        }

        # Default pricing for unknown models
        default_pricing = {"input": 0.01 / 1000, "output": 0.02 / 1000}

        model_pricing = pricing.get(model, default_pricing)

        input_cost = input_tokens * model_pricing["input"]
        output_cost = output_tokens * model_pricing["output"]

        return input_cost + output_cost


def instrument_openai(
    client: Optional[Any] = None, **client_kwargs
) -> GenOpsOpenAIAdapter:
    """
    Instrument an OpenAI client with GenOps governance telemetry.

    Args:
        client: Existing OpenAI client (optional)
        **client_kwargs: Arguments to pass to OpenAI client if creating new one

    Returns:
        GenOpsOpenAIAdapter: Instrumented client with governance tracking

    Example:
        import genops

        # Method 1: Instrument existing client
        openai_client = OpenAI(api_key="your-key")
        genops_client = genops.providers.openai.instrument_openai(openai_client)

        # Method 2: Create instrumented client directly
        genops_client = genops.providers.openai.instrument_openai(api_key="your-key")

        # Use normally - telemetry is automatic
        response = genops_client.chat_completions_create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    return GenOpsOpenAIAdapter(client=client, **client_kwargs)


# Monkey patching support for transparent instrumentation
_original_openai_create = None
_original_completions_create = None


def patch_openai(auto_track: bool = True):
    """
    Monkey patch OpenAI to automatically add telemetry to all requests.

    Warning: This modifies the global OpenAI behavior. Use with caution.

    Args:
        auto_track: Whether to automatically track all OpenAI calls
    """
    if not HAS_OPENAI:
        logger.warning("OpenAI not available for patching")
        return

    global _original_openai_create, _original_completions_create

    if auto_track and _original_openai_create is None:
        try:
            # Store original methods
            _original_openai_create = openai.OpenAI.chat.completions.create
            _original_completions_create = openai.OpenAI.completions.create

            def patched_chat_create(self, **kwargs):
                adapter = GenOpsOpenAIAdapter(client=self)
                return adapter.chat_completions_create(**kwargs)

            def patched_completions_create(self, **kwargs):
                adapter = GenOpsOpenAIAdapter(client=self)
                return adapter.completions_create(**kwargs)

            # Apply patches
            openai.OpenAI.chat.completions.create = patched_chat_create
            openai.OpenAI.completions.create = patched_completions_create

            logger.info("OpenAI client patched with GenOps telemetry")
        except AttributeError as e:
            logger.warning(f"Failed to patch OpenAI: {e}")
            return


def unpatch_openai():
    """Remove OpenAI monkey patches and restore original behavior."""
    if not HAS_OPENAI:
        return

    global _original_openai_create, _original_completions_create

    if _original_openai_create is not None:
        openai.OpenAI.chat.completions.create = _original_openai_create
        openai.OpenAI.completions.create = _original_completions_create

        _original_openai_create = None
        _original_completions_create = None

        logger.info("OpenAI patches removed")
