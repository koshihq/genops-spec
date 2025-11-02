"""OpenRouter provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Set

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
    
    HAS_OPENROUTER_DEPS = True
except ImportError:
    HAS_OPENROUTER_DEPS = False
    OpenAI = None
    logger.warning("OpenAI package not installed (required for OpenRouter). Install with: pip install openai")


class GenOpsOpenRouterAdapter:
    """OpenRouter adapter with automatic governance telemetry and multi-provider routing awareness."""

    def __init__(self, client: Optional[Any] = None, **client_kwargs):
        if not HAS_OPENROUTER_DEPS:
            raise ImportError(
                "OpenAI package not found (required for OpenRouter compatibility). Install with: pip install openai"
            )

        # OpenRouter uses OpenAI-compatible API with custom base URL
        default_kwargs = {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key': client_kwargs.get('api_key') or client_kwargs.get('openrouter_api_key')
        }
        
        # Override defaults with any provided kwargs
        final_kwargs = {**default_kwargs, **client_kwargs}
        if 'openrouter_api_key' in final_kwargs:
            final_kwargs.pop('openrouter_api_key')  # Clean up custom key name
            
        self.client = client or OpenAI(**final_kwargs)
        self.telemetry = GenOpsTelemetry()

        # Define governance and request attributes
        self.GOVERNANCE_ATTRIBUTES = {
            'team', 'project', 'feature', 'customer_id', 'customer',
            'environment', 'cost_center', 'user_id'
        }
        self.REQUEST_ATTRIBUTES = {
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty',
            'presence_penalty', 'stop', 'seed', 'stream',
            'provider', 'route', 'models', 'fallbacks'  # OpenRouter-specific
        }
        
        # OpenRouter-specific routing attributes
        self.OPENROUTER_ATTRIBUTES = {
            'provider', 'route', 'models', 'fallbacks', 'transforms'
        }

    def _extract_attributes(self, kwargs: Dict) -> Tuple[Dict, Dict, Dict]:
        """Extract governance, request, and routing attributes from kwargs."""
        governance_attrs = {}
        request_attrs = {}
        api_kwargs = kwargs.copy()

        # Extract governance attributes
        for attr in self.GOVERNANCE_ATTRIBUTES:
            if attr in kwargs:
                governance_attrs[attr] = kwargs[attr]
                api_kwargs.pop(attr)

        # Extract request attributes (including OpenRouter-specific ones)
        for attr in self.REQUEST_ATTRIBUTES:
            if attr in kwargs:
                request_attrs[attr] = kwargs[attr]

        return governance_attrs, request_attrs, api_kwargs

    def _extract_routing_info(self, response: Any) -> Dict[str, Any]:
        """Extract OpenRouter-specific routing information from response."""
        routing_info = {}
        
        # Check for OpenRouter response headers or metadata
        if hasattr(response, 'response') and hasattr(response.response, 'headers'):
            headers = response.response.headers
            # OpenRouter typically includes routing info in headers
            routing_info['selected_provider'] = headers.get('x-openrouter-provider')
            routing_info['fallback_used'] = headers.get('x-openrouter-fallback') == 'true'
            routing_info['request_id'] = headers.get('x-request-id')
            
        # Alternative: check for routing info in response object
        elif hasattr(response, 'provider'):
            routing_info['selected_provider'] = response.provider
            
        return routing_info

    def _get_provider_from_model(self, model: str) -> str:
        """Extract likely provider from OpenRouter model name."""
        # OpenRouter model names often include provider info
        if 'openai' in model.lower() or 'gpt' in model.lower():
            return 'openai'
        elif 'anthropic' in model.lower() or 'claude' in model.lower():
            return 'anthropic'
        elif 'google' in model.lower() or 'gemini' in model.lower():
            return 'google'
        elif 'meta' in model.lower() or 'llama' in model.lower():
            return 'meta'
        elif 'mistral' in model.lower():
            return 'mistral'
        elif 'cohere' in model.lower():
            return 'cohere'
        else:
            return 'openrouter'  # Default fallback

    def chat_completions_create(self, **kwargs) -> Any:
        """Create chat completion with governance tracking and OpenRouter routing awareness."""
        # Extract attributes from kwargs
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model", "unknown")
        messages = api_kwargs.get("messages", [])
        
        # Extract OpenRouter-specific routing preferences
        preferred_provider = request_attrs.get('provider')
        routing_strategy = request_attrs.get('route', 'fallback')  # 'fallback', 'least-cost', 'fastest'

        # Estimate input tokens (rough approximation)
        input_text = " ".join(
            [msg.get("content", "") for msg in messages if isinstance(msg, dict)]
        )
        estimated_input_tokens = len(input_text.split()) * 1.3  # rough token estimate

        operation_name = "openrouter.chat.completions.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.inference",
            "provider": "openrouter",
            "model": model,
            "tokens_estimated_input": int(estimated_input_tokens),
            "openrouter.routing_strategy": routing_strategy,
        }
        
        # Add OpenRouter-specific attributes
        if preferred_provider:
            trace_attrs["openrouter.preferred_provider"] = preferred_provider
            
        # Predict likely backend provider for cost estimation
        predicted_provider = self._get_provider_from_model(model)
        trace_attrs["openrouter.predicted_provider"] = predicted_provider

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes
            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except (ImportError, Exception):
            # Fallback to just governance attributes
            trace_attrs.update(governance_attrs)

        with self.telemetry.trace_operation(**trace_attrs) as span:
            # Record request parameters in telemetry
            for param, value in request_attrs.items():
                span.set_attribute(f"genops.request.{param}", value)

            try:
                # Call OpenRouter API with cleaned kwargs (no governance attributes)
                response = self.client.chat.completions.create(**api_kwargs)

                # Extract routing information from response
                routing_info = self._extract_routing_info(response)
                actual_provider = routing_info.get('selected_provider', predicted_provider)
                
                # Record routing telemetry
                span.set_attribute("genops.openrouter.actual_provider", actual_provider)
                if routing_info.get('fallback_used'):
                    span.set_attribute("genops.openrouter.fallback_used", True)
                if routing_info.get('request_id'):
                    span.set_attribute("genops.openrouter.request_id", routing_info['request_id'])

                # Extract usage and cost information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Calculate cost using OpenRouter pricing and actual provider
                    cost = self._calculate_cost(model, actual_provider, input_tokens, output_tokens)

                    # Record telemetry with both OpenRouter and underlying provider info
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider="openrouter",  # Top-level provider
                        model=model,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens,
                        tokens_total=total_tokens,
                        underlying_provider=actual_provider,  # Actual LLM provider used
                    )

                return response

            except Exception as e:
                logger.error(f"OpenRouter API error: {e}")
                # Record error details for debugging
                span.set_attribute("genops.error.message", str(e))
                span.set_attribute("genops.error.type", type(e).__name__)
                raise

    def completions_create(self, **kwargs) -> Any:
        """Create completion with governance tracking (legacy API support)."""
        # Extract attributes from kwargs
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model", "unknown")
        prompt = api_kwargs.get("prompt", "")
        
        # Extract OpenRouter-specific routing preferences
        preferred_provider = request_attrs.get('provider')
        routing_strategy = request_attrs.get('route', 'fallback')

        # Estimate input tokens
        estimated_input_tokens = len(str(prompt).split()) * 1.3

        operation_name = "openrouter.completions.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.inference",
            "provider": "openrouter",
            "model": model,
            "tokens_estimated_input": int(estimated_input_tokens),
            "openrouter.routing_strategy": routing_strategy,
        }
        
        # Add OpenRouter-specific attributes
        if preferred_provider:
            trace_attrs["openrouter.preferred_provider"] = preferred_provider
            
        # Predict likely backend provider
        predicted_provider = self._get_provider_from_model(model)
        trace_attrs["openrouter.predicted_provider"] = predicted_provider

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes
            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except (ImportError, Exception):
            # Fallback to just governance attributes
            trace_attrs.update(governance_attrs)

        with self.telemetry.trace_operation(**trace_attrs) as span:
            # Record request parameters in telemetry
            for param, value in request_attrs.items():
                span.set_attribute(f"genops.request.{param}", value)

            try:
                # Call OpenRouter API with cleaned kwargs (no governance attributes)
                response = self.client.completions.create(**api_kwargs)

                # Extract routing information
                routing_info = self._extract_routing_info(response)
                actual_provider = routing_info.get('selected_provider', predicted_provider)
                
                # Record routing telemetry
                span.set_attribute("genops.openrouter.actual_provider", actual_provider)
                if routing_info.get('fallback_used'):
                    span.set_attribute("genops.openrouter.fallback_used", True)

                # Extract usage and cost information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Calculate cost
                    cost = self._calculate_cost(model, actual_provider, input_tokens, output_tokens)

                    # Record telemetry
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider="openrouter",
                        model=model,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens,
                        tokens_total=total_tokens,
                        underlying_provider=actual_provider,
                    )

                return response

            except Exception as e:
                logger.error(f"OpenRouter API error: {e}")
                span.set_attribute("genops.error.message", str(e))
                span.set_attribute("genops.error.type", type(e).__name__)
                raise

    def _calculate_cost(
        self, model: str, actual_provider: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate estimated cost based on OpenRouter pricing and routing."""
        # Import the pricing engine
        try:
            from .openrouter_pricing import calculate_openrouter_cost
            return calculate_openrouter_cost(model, actual_provider, input_tokens, output_tokens)
        except ImportError:
            # Fallback to simplified pricing estimation
            logger.warning("OpenRouter pricing engine not available, using simplified estimation")
            return self._fallback_cost_calculation(model, actual_provider, input_tokens, output_tokens)

    def _fallback_cost_calculation(
        self, model: str, actual_provider: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Fallback cost calculation when pricing engine is not available."""
        # Simplified pricing based on common patterns
        base_pricing = {
            'openai': {"input": 0.01 / 1000, "output": 0.02 / 1000},
            'anthropic': {"input": 3.00 / 1000000, "output": 15.00 / 1000000},
            'google': {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
            'meta': {"input": 0.0002 / 1000, "output": 0.0002 / 1000},
            'mistral': {"input": 0.0007 / 1000, "output": 0.0007 / 1000},
        }

        # Default to medium-cost provider pricing
        default_pricing = {"input": 0.005 / 1000, "output": 0.01 / 1000}
        
        provider_pricing = base_pricing.get(actual_provider, default_pricing)
        
        input_cost = input_tokens * provider_pricing["input"]
        output_cost = output_tokens * provider_pricing["output"]

        return input_cost + output_cost


def instrument_openrouter(
    client: Optional[Any] = None, **client_kwargs
) -> GenOpsOpenRouterAdapter:
    """
    Instrument an OpenRouter client with GenOps governance telemetry.

    Args:
        client: Existing OpenRouter/OpenAI client (optional)
        **client_kwargs: Arguments to pass to OpenRouter client if creating new one.
                        Use 'openrouter_api_key' or 'api_key' for authentication.

    Returns:
        GenOpsOpenRouterAdapter: Instrumented client with governance tracking

    Example:
        import genops

        # Method 1: Create instrumented client directly
        genops_client = genops.providers.openrouter.instrument_openrouter(
            openrouter_api_key="your-openrouter-key"
        )

        # Method 2: Use existing OpenAI client configured for OpenRouter
        from openai import OpenAI
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="your-openrouter-key"
        )
        genops_client = genops.providers.openrouter.instrument_openrouter(openrouter_client)

        # Use normally - telemetry and routing info is automatically captured
        response = genops_client.chat_completions_create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": "Hello!"}],
            team="ai-team",
            project="chatbot",
            customer_id="customer-123",
            provider="anthropic"  # Optional: prefer specific provider
        )
    """
    return GenOpsOpenRouterAdapter(client=client, **client_kwargs)


# Monkey patching support for transparent instrumentation
_original_openai_create = None
_original_completions_create = None


def patch_openrouter(auto_track: bool = True):
    """
    Monkey patch OpenAI client to automatically add telemetry when used with OpenRouter.

    Warning: This modifies the global OpenAI client behavior. Use with caution.
    Only patches clients that have OpenRouter base URL.

    Args:
        auto_track: Whether to automatically track all OpenRouter calls
    """
    if not HAS_OPENROUTER_DEPS:
        logger.warning("OpenAI package not available for OpenRouter patching")
        return

    global _original_openai_create, _original_completions_create

    if auto_track and _original_openai_create is None:
        try:
            # Store original methods
            _original_openai_create = openai.OpenAI.chat.completions.create
            _original_completions_create = openai.OpenAI.completions.create

            def patched_chat_create(self, **kwargs):
                # Only apply GenOps instrumentation for OpenRouter clients
                if hasattr(self, 'base_url') and 'openrouter.ai' in str(self.base_url):
                    adapter = GenOpsOpenRouterAdapter(client=self)
                    return adapter.chat_completions_create(**kwargs)
                else:
                    # Use original method for non-OpenRouter clients
                    return _original_openai_create(self, **kwargs)

            def patched_completions_create(self, **kwargs):
                # Only apply GenOps instrumentation for OpenRouter clients
                if hasattr(self, 'base_url') and 'openrouter.ai' in str(self.base_url):
                    adapter = GenOpsOpenRouterAdapter(client=self)
                    return adapter.completions_create(**kwargs)
                else:
                    # Use original method for non-OpenRouter clients
                    return _original_completions_create(self, **kwargs)

            # Apply patches
            openai.OpenAI.chat.completions.create = patched_chat_create
            openai.OpenAI.completions.create = patched_completions_create

            logger.info("OpenAI client patched with GenOps OpenRouter telemetry")
        except AttributeError as e:
            logger.warning(f"Failed to patch OpenAI for OpenRouter: {e}")
            return


def unpatch_openrouter():
    """Remove OpenRouter monkey patches and restore original OpenAI behavior."""
    if not HAS_OPENROUTER_DEPS:
        return

    global _original_openai_create, _original_completions_create

    if _original_openai_create is not None:
        openai.OpenAI.chat.completions.create = _original_openai_create
        openai.OpenAI.completions.create = _original_completions_create

        _original_openai_create = None
        _original_completions_create = None

        logger.info("OpenRouter patches removed")


# Import validation utilities
def validate_setup():
    """Validate OpenRouter provider setup."""
    try:
        from .openrouter_validation import validate_openrouter_setup
        return validate_openrouter_setup()
    except ImportError:
        logger.warning("OpenRouter validation utilities not available")
        return None


def print_validation_result(result):
    """Print validation result in user-friendly format."""
    try:
        from .openrouter_validation import print_openrouter_validation_result
        print_openrouter_validation_result(result)
    except ImportError:
        logger.warning("OpenRouter validation utilities not available")