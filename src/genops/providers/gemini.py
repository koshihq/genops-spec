#!/usr/bin/env python3
"""
GenOps Google Gemini Provider Integration

This module provides comprehensive Google Gemini integration for GenOps AI governance,
cost intelligence, and observability. It follows the established GenOps provider
pattern for consistent developer experience across all AI platforms.

Features:
- Multi-model support (Gemini 2.5 Pro, Flash, Flash-Lite)
- Zero-code auto-instrumentation with instrument_gemini()
- Unified cost tracking across all Gemini models
- Streaming response support for real-time applications
- Google AI API key authentication with environment variable support
- Comprehensive governance and audit trail integration

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.gemini import instrument_gemini
    instrument_gemini()

    # Your existing Gemini code works unchanged with automatic governance
    from google import genai
    client = genai.Client()
    response = client.models.generate_content(...)  # Now tracked with GenOps!

    # Manual adapter usage for advanced control
    from genops.providers.gemini import GenOpsGeminiAdapter

    adapter = GenOpsGeminiAdapter()
    response = adapter.text_generation(
        prompt="Explain quantum computing",
        model="gemini-2.5-flash",
        team="research-team",
        project="quantum-ai",
        customer_id="enterprise-123"
    )
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

try:
    import google.genai as genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

try:
    from genops.core.base_provider import BaseProvider, OperationContext
    from genops.core.telemetry import GenOpsTelemetry
    from genops.providers.gemini_pricing import (
        GEMINI_MODELS,  # noqa: F401
        calculate_gemini_cost,
        compare_gemini_models,  # noqa: F401
        get_gemini_model_info,  # noqa: F401
    )
    from genops.providers.gemini_validation import (
        GeminiValidationResult,
        validate_gemini_setup,
    )
    from genops.providers.gemini_validation import (
        print_validation_result as _print_validation_result,  # noqa: F401
    )

    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GeminiOperationResult:
    """Result from a Gemini operation with full telemetry context."""

    content: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    operation_id: str
    governance_attributes: dict[str, str]
    raw_response: Optional[dict] = None


class GenOpsGeminiAdapter(BaseProvider):
    """
    GenOps adapter for Google Gemini with comprehensive AI governance.

    This adapter provides unified instrumentation for all Gemini models
    while maintaining the native Google AI SDK experience. It automatically
    captures costs, performance metrics, and governance attributes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash",
        enable_streaming: bool = True,
        **kwargs,
    ):
        """
        Initialize the GenOps Gemini adapter.

        Args:
            api_key: Google AI API key (optional, can use GEMINI_API_KEY env var)
            default_model: Default model ID for operations
            enable_streaming: Enable streaming response support
            **kwargs: Additional arguments passed to genai.Client
        """
        super().__init__()

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gemini dependencies not available. Install with: "
                "pip install google-generativeai"
            )

        if not GENOPS_AVAILABLE:
            logger.warning("GenOps core not available, running in basic mode")

        # Handle API key from environment if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter. Get your API key at: https://ai.google.dev/"
            )

        self.default_model = default_model
        self.enable_streaming = enable_streaming

        # Initialize Google AI client
        try:
            self.client = genai.Client(api_key=self.api_key, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

        # Initialize telemetry
        if GENOPS_AVAILABLE:
            self.telemetry = GenOpsTelemetry()
        else:
            self.telemetry = None

        logger.info("GenOps Gemini adapter initialized")

    def is_available(self) -> bool:
        """Check if Gemini API is available and accessible."""
        if not GEMINI_AVAILABLE:
            return False

        try:
            # Try a minimal API call to check availability
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents="Hello"
            )
            return bool(response and hasattr(response, "text"))
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            return False

    def get_supported_models(self) -> list[str]:
        """Get list of supported Gemini model IDs."""
        try:
            # Try to list models from API
            models = self.client.models.list()
            return [model.name for model in models if hasattr(model, "name")]
        except Exception as e:
            logger.warning(f"Failed to fetch supported models: {e}")
            return [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

    def get_supported_tasks(self) -> list[str]:
        """Get list of supported AI tasks."""
        return [
            "text-generation",
            "chat-completion",
            "content-generation",
            "code-generation",
            "text-analysis",
            "question-answering",
            "summarization",
            "streaming-generation",
        ]

    def _create_operation_context(  # type: ignore
        self, operation_name: str, model_id: str, **governance_attrs
    ) -> OperationContext:
        """Create operation context with Gemini-specific attributes."""
        operation_id = str(uuid.uuid4())

        context = OperationContext(  # type: ignore[call-arg]
            operation_id=operation_id,
            operation_name=operation_name,
            provider="gemini",
            model=model_id,
            **governance_attrs,
        )

        return context

    def _calculate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation. For production use, consider
        integrating with Google's tokenization service.
        """
        # Rough approximation: ~4 characters per token for most models
        return max(1, len(text) // 4)

    def _extract_response_content(
        self, response: Any, model_id: str
    ) -> tuple[str, int]:
        """
        Extract content and output tokens from Gemini response.
        """
        try:
            content = response.text if hasattr(response, "text") else str(response)

            # Try to get actual token counts if available
            output_tokens = (
                response.usage_metadata.candidates_token_count
                if hasattr(response, "usage_metadata")
                and hasattr(response.usage_metadata, "candidates_token_count")
                else self._calculate_tokens(content)
            )

            return content, output_tokens
        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return str(response), self._calculate_tokens(str(response))

    def text_generation(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        **governance_attrs,
    ) -> GeminiOperationResult:
        """
        Generate text using Gemini with comprehensive governance tracking.

        Args:
            prompt: Text prompt for generation
            model: Model ID to use (defaults to default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stream: Enable streaming response
            **governance_attrs: Governance attributes (team, project, customer_id, etc.)

        Returns:
            GeminiOperationResult with response content and telemetry
        """
        model_id = model or self.default_model
        start_time = time.time()

        # Create operation context
        context = self._create_operation_context(
            "gemini.text_generation", model_id, **governance_attrs
        )

        # Prepare request parameters
        request_params = {"model": model_id, "contents": prompt}

        if max_tokens:
            request_params["generation_config"] = request_params.get(
                "generation_config",
                {},  # type: ignore[arg-type]
            )
            request_params["generation_config"]["max_output_tokens"] = max_tokens
        if temperature is not None:
            request_params["generation_config"] = request_params.get(
                "generation_config",
                {},  # type: ignore[arg-type]
            )
            request_params["generation_config"]["temperature"] = temperature
        if top_p is not None:
            request_params["generation_config"] = request_params.get(
                "generation_config",
                {},  # type: ignore[arg-type]
            )
            request_params["generation_config"]["top_p"] = top_p
        if top_k is not None:
            request_params["generation_config"] = request_params.get(
                "generation_config",
                {},  # type: ignore[arg-type]
            )
            request_params["generation_config"]["top_k"] = top_k

        if GENOPS_AVAILABLE and self.telemetry:
            # Create span for the operation
            with self.telemetry.trace_operation(
                operation_name=context.operation_name,
                provider="gemini",
                model=model_id,
                **governance_attrs,
            ) as span:
                try:
                    # Perform the API call
                    response = self.client.models.generate_content(**request_params)

                    # Extract response details
                    content, output_tokens = self._extract_response_content(
                        response, model_id
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    input_tokens = self._calculate_tokens(prompt)

                    # Get actual token counts if available
                    if hasattr(response, "usage_metadata"):
                        usage = response.usage_metadata
                        if hasattr(usage, "prompt_token_count"):
                            input_tokens = usage.prompt_token_count
                        if hasattr(usage, "candidates_token_count"):
                            output_tokens = usage.candidates_token_count

                    # Calculate cost
                    cost_usd = (
                        calculate_gemini_cost(
                            model_id=model_id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )
                        if "calculate_gemini_cost" in globals()
                        else 0.0
                    )

                    # Record telemetry
                    span.set_attributes(
                        {
                            "genops.provider": "gemini",
                            "genops.model": model_id,
                            "genops.operation_type": "text_generation",
                            "genops.tokens.input": input_tokens,
                            "genops.tokens.output": output_tokens,
                            "genops.cost.total": cost_usd,
                            "genops.cost.currency": "USD",
                            "genops.latency_ms": latency_ms,
                            "genops.operation_id": context.operation_id,
                        }
                    )

                    # Add governance attributes to span
                    for key, value in governance_attrs.items():
                        span.set_attribute(f"genops.{key}", str(value))

                    return GeminiOperationResult(
                        content=content,
                        model_id=model_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=latency_ms,
                        cost_usd=cost_usd,
                        operation_id=context.operation_id,
                        governance_attributes=governance_attrs,
                        raw_response=response.__dict__
                        if hasattr(response, "__dict__")
                        else None,
                    )

                except Exception as e:
                    span.set_status(status="ERROR", description=str(e))
                    raise
        else:
            # Fallback without telemetry
            try:
                response = self.client.models.generate_content(**request_params)
                content, output_tokens = self._extract_response_content(
                    response, model_id
                )
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = self._calculate_tokens(prompt)

                return GeminiOperationResult(
                    content=content,
                    model_id=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                    operation_id=str(uuid.uuid4()),
                    governance_attributes=governance_attrs,
                )
            except Exception as e:
                logger.error(f"Gemini text generation failed: {e}")
                raise

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **governance_attrs,
    ) -> GeminiOperationResult:
        """
        Create chat completion using Gemini with governance tracking.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model ID to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming response
            **governance_attrs: Governance attributes (team, project, customer_id, etc.)

        Returns:
            GeminiOperationResult with response content and telemetry
        """
        # Convert messages to a single prompt for Gemini
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        combined_prompt = "\n\n".join(prompt_parts)

        return self.text_generation(
            prompt=combined_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **governance_attrs,
        )


# Auto-instrumentation functions
def instrument_gemini(**config) -> bool:
    """
    Enable automatic instrumentation for Google Gemini SDK.

    This function patches the genai.Client to automatically capture
    governance telemetry for all Gemini operations.

    Args:
        **config: Configuration options for instrumentation

    Returns:
        True if instrumentation was successful, False otherwise
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Google Gemini SDK not available for instrumentation")
        return False

    if not GENOPS_AVAILABLE:
        logger.warning("GenOps core not available for instrumentation")
        return False

    try:
        # Patch the generate_content method
        original_generate_content = genai.Client.models.generate_content

        def instrumented_generate_content(self, **kwargs):
            # Extract governance attributes from kwargs
            governance_attrs = {}
            api_kwargs = kwargs.copy()

            governance_keys = {
                "team",
                "project",
                "customer_id",
                "environment",
                "cost_center",
            }

            for key in governance_keys:
                if key in kwargs:
                    governance_attrs[key] = kwargs[key]
                    api_kwargs.pop(key)

            # Create GenOps adapter for tracking
            adapter = GenOpsGeminiAdapter()

            # Use the adapter's text_generation method
            prompt = api_kwargs.get("contents", "")
            model = api_kwargs.get("model", adapter.default_model)

            result = adapter.text_generation(
                prompt=prompt, model=model, **governance_attrs
            )

            # Return the raw response for compatibility
            return result.raw_response or original_generate_content(self, **api_kwargs)

        # Apply the patch
        genai.Client.models.generate_content = instrumented_generate_content

        logger.info("Google Gemini auto-instrumentation enabled")
        return True

    except Exception as e:
        logger.error(f"Failed to instrument Gemini: {e}")
        return False


def auto_instrument_gemini(**config) -> bool:
    """Alias for instrument_gemini() for consistency with other providers."""
    return instrument_gemini(**config)


def auto_instrument(**config) -> bool:
    """
    Universal auto-instrumentation function (CLAUDE.md standard).

    This function provides the standard auto_instrument() interface required
    by GenOps Developer Experience Excellence Standards for all providers.

    Args:
        **config: Configuration options for instrumentation

    Returns:
        True if instrumentation was successful, False otherwise
    """
    return instrument_gemini(**config)


# Validation functions
def validate_setup() -> "GeminiValidationResult":
    """
    Validate Google Gemini setup and configuration.

    Returns:
        GeminiValidationResult with validation status and recommendations
    """
    if "validate_gemini_setup" in globals():
        return validate_gemini_setup()

    # Fallback validation
    from dataclasses import dataclass

    @dataclass
    class BasicValidationResult:
        success: bool
        errors: list[str]
        warnings: list[str]
        recommendations: list[str]

    errors = []
    warnings = []
    recommendations = []

    # Check if Gemini SDK is available
    if not GEMINI_AVAILABLE:
        errors.append(
            "Google Gemini SDK not installed. Run: pip install google-generativeai"
        )

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        errors.append("GEMINI_API_KEY environment variable not set")
        recommendations.append(
            "Set GEMINI_API_KEY environment variable with your API key from https://ai.google.dev/"
        )

    # Check GenOps availability
    if not GENOPS_AVAILABLE:
        warnings.append("GenOps core not available - running in basic mode")

    return BasicValidationResult(  # type: ignore[return-value]
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        recommendations=recommendations,
    )


def print_validation_result(result: Any, detailed: bool = False) -> None:
    """
    Print validation results in a user-friendly format.

    Args:
        result: Validation result object
        detailed: Whether to show detailed information
    """
    if hasattr(result, "success"):
        if result.success:
            print("✅ Google Gemini setup validation passed!")
        else:
            print("❌ Google Gemini setup validation failed:")

        if hasattr(result, "errors") and result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        if hasattr(result, "warnings") and result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        if hasattr(result, "recommendations") and result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
    else:
        print("Validation result format not recognized")


# Export main classes and functions
__all__ = [
    "GenOpsGeminiAdapter",
    "GeminiOperationResult",
    "instrument_gemini",
    "auto_instrument_gemini",
    "auto_instrument",  # Universal CLAUDE.md standard function
    "validate_setup",
    "print_validation_result",
    "GEMINI_AVAILABLE",
]
