"""Anthropic provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
from typing import Any

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

try:
    import anthropic
    from anthropic import Anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None
    logger.warning("Anthropic not installed. Install with: pip install anthropic")


class GenOpsAnthropicAdapter:
    """Anthropic adapter with automatic governance telemetry."""

    def __init__(self, client: Any | None = None, **client_kwargs):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic package not found. Install with: pip install anthropic"
            )

        self.client = client or Anthropic(**client_kwargs)
        self.telemetry = GenOpsTelemetry()

        # Define governance and request attributes
        self.GOVERNANCE_ATTRIBUTES = {
            'team', 'project', 'feature', 'customer_id', 'customer',
            'environment', 'cost_center', 'user_id'
        }
        self.REQUEST_ATTRIBUTES = {
            'temperature', 'max_tokens', 'top_p', 'top_k', 'stop_sequences'
        }

    def _extract_attributes(self, kwargs: dict) -> tuple[dict, dict, dict]:
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

    def messages_create(self, **kwargs) -> Any:
        """Create message with governance tracking."""
        # Extract attributes from kwargs
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model", "unknown")
        messages = api_kwargs.get("messages", [])
        system = api_kwargs.get("system", "")

        # Estimate input tokens (rough approximation)
        input_text = (
            system
            + " "
            + " ".join(
                [
                    msg.get("content", "")
                    for msg in messages
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str)
                ]
            )
        )
        estimated_input_tokens = len(input_text.split()) * 1.3  # rough token estimate

        operation_name = "anthropic.messages.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.inference",
            "provider": "anthropic",
            "model": model,
            "tokens_estimated_input": int(estimated_input_tokens),
        }

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
                # Call Anthropic API with cleaned kwargs (no governance attributes)
                response = self.client.messages.create(**api_kwargs)

                # Extract usage and cost information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens

                    # Calculate cost based on model pricing
                    cost = self._calculate_cost(model, input_tokens, output_tokens)

                    # Record telemetry
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider="anthropic",
                        model=model,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens,
                        tokens_total=input_tokens + output_tokens,
                    )

                return response

            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                raise

    def completions_create(self, **kwargs) -> Any:
        """Create completion with governance tracking (legacy API)."""
        # Extract attributes from kwargs
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model", "unknown")
        prompt = api_kwargs.get("prompt", "")

        # Estimate input tokens
        estimated_input_tokens = len(str(prompt).split()) * 1.3

        operation_name = "anthropic.completions.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.inference",
            "provider": "anthropic",
            "model": model,
            "tokens_estimated_input": int(estimated_input_tokens),
        }

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
                # Call Anthropic API (legacy)
                if hasattr(self.client, "completions"):
                    response = self.client.completions.create(**api_kwargs)
                else:
                    # Convert to messages format for newer API
                    messages_kwargs = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": api_kwargs.get("max_tokens_to_sample", 1024),
                    }
                    # Add any other request parameters from api_kwargs
                    for param in self.REQUEST_ATTRIBUTES:
                        if param in api_kwargs:
                            messages_kwargs[param] = api_kwargs[param]
                    response = self.client.messages.create(**messages_kwargs)

                # Extract usage and cost information
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)

                    # Calculate cost
                    cost = self._calculate_cost(model, input_tokens, output_tokens)

                    # Record telemetry
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider="anthropic",
                        model=model,
                        tokens_input=input_tokens,
                        tokens_output=output_tokens,
                        tokens_total=input_tokens + output_tokens,
                    )

                return response

            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                raise

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate estimated cost based on Anthropic pricing."""
        # Simplified pricing - in production, use real pricing API or config
        pricing = {
            "claude-3-5-sonnet-20241022": {
                "input": 3.00 / 1000000,
                "output": 15.00 / 1000000,
            },
            "claude-3-5-sonnet-20240620": {
                "input": 3.00 / 1000000,
                "output": 15.00 / 1000000,
            },
            "claude-3-5-haiku-20241022": {
                "input": 1.00 / 1000000,
                "output": 5.00 / 1000000,
            },
            "claude-3-opus-20240229": {
                "input": 15.00 / 1000000,
                "output": 75.00 / 1000000,
            },
            "claude-3-sonnet-20240229": {
                "input": 3.00 / 1000000,
                "output": 15.00 / 1000000,
            },
            "claude-3-haiku-20240307": {
                "input": 0.25 / 1000000,
                "output": 1.25 / 1000000,
            },
            # Simplified model name mappings
            "claude-3-5-sonnet": {"input": 3.00 / 1000000, "output": 15.00 / 1000000},
            "claude-3-5-haiku": {"input": 1.00 / 1000000, "output": 5.00 / 1000000},
            "claude-3-opus": {"input": 15.00 / 1000000, "output": 75.00 / 1000000},
            "claude-3-sonnet": {"input": 3.00 / 1000000, "output": 15.00 / 1000000},
            "claude-3-haiku": {"input": 0.25 / 1000000, "output": 1.25 / 1000000},
        }

        # Default pricing for unknown models (use Claude 3 Sonnet pricing)
        default_pricing = {"input": 3.00 / 1000000, "output": 15.00 / 1000000}

        model_pricing = pricing.get(model, default_pricing)

        input_cost = input_tokens * model_pricing["input"]
        output_cost = output_tokens * model_pricing["output"]

        return input_cost + output_cost


def instrument_anthropic(
    client: Any | None = None, **client_kwargs
) -> GenOpsAnthropicAdapter:
    """
    Instrument an Anthropic client with GenOps governance telemetry.

    Args:
        client: Existing Anthropic client (optional)
        **client_kwargs: Arguments to pass to Anthropic client if creating new one

    Returns:
        GenOpsAnthropicAdapter: Instrumented client with governance tracking

    Example:
        import genops

        # Method 1: Instrument existing client
        anthropic_client = Anthropic(api_key="your-key")
        genops_client = genops.providers.anthropic.instrument_anthropic(anthropic_client)

        # Method 2: Create instrumented client directly
        genops_client = genops.providers.anthropic.instrument_anthropic(api_key="your-key")

        # Use normally - telemetry is automatic
        response = genops_client.messages_create(
            model="claude-3-sonnet",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=100
        )
    """
    return GenOpsAnthropicAdapter(client=client, **client_kwargs)


# Monkey patching support for transparent instrumentation
_original_messages_create = None
_original_completions_create = None


def patch_anthropic(auto_track: bool = True):
    """
    Monkey patch Anthropic to automatically add telemetry to all requests.

    Warning: This modifies the global Anthropic behavior. Use with caution.

    Args:
        auto_track: Whether to automatically track all Anthropic calls
    """
    if not HAS_ANTHROPIC:
        logger.warning("Anthropic not available for patching")
        return

    global _original_messages_create, _original_completions_create

    if auto_track and _original_messages_create is None:
        try:
            # Store original methods
            _original_messages_create = anthropic.Anthropic.messages.create

            def patched_messages_create(self, **kwargs):
                adapter = GenOpsAnthropicAdapter(client=self)
                return adapter.messages_create(**kwargs)

            # Apply patches
            anthropic.Anthropic.messages.create = patched_messages_create

            # Patch completions if available (legacy API)
            if hasattr(anthropic.Anthropic, "completions"):
                _original_completions_create = anthropic.Anthropic.completions.create

                def patched_completions_create(self, **kwargs):
                    adapter = GenOpsAnthropicAdapter(client=self)
                    return adapter.completions_create(**kwargs)

                anthropic.Anthropic.completions.create = patched_completions_create

            logger.info("Anthropic client patched with GenOps telemetry")
        except AttributeError as e:
            logger.warning(f"Failed to patch Anthropic: {e}")
            return


def unpatch_anthropic():
    """Remove Anthropic monkey patches and restore original behavior."""
    if not HAS_ANTHROPIC:
        return

    global _original_messages_create, _original_completions_create

    if _original_messages_create is not None:
        anthropic.Anthropic.messages.create = _original_messages_create

        if _original_completions_create is not None and hasattr(
            anthropic.Anthropic, "completions"
        ):
            anthropic.Anthropic.completions.create = _original_completions_create

        _original_messages_create = None
        _original_completions_create = None

        logger.info("Anthropic patches removed")


# Import validation utilities
def validate_setup():
    """Validate Anthropic provider setup."""
    try:
        from .anthropic_validation import validate_anthropic_setup
        return validate_anthropic_setup()
    except ImportError:
        logger.warning("Anthropic validation utilities not available")
        return None


def print_validation_result(result):
    """Print validation result in user-friendly format."""
    try:
        from .anthropic_validation import print_anthropic_validation_result
        print_anthropic_validation_result(result)
    except ImportError:
        logger.warning("Anthropic validation utilities not available")
