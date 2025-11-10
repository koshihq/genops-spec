#!/usr/bin/env python3
"""
GenOps Replicate Provider Integration

This module provides comprehensive Replicate integration for GenOps AI governance,
cost intelligence, and observability. It follows the established GenOps provider
pattern for consistent developer experience across all AI platforms.

Features:
- Multi-modal model support (text, image, video, audio)
- Zero-code auto-instrumentation with auto_instrument()
- Unified cost tracking across all Replicate models
- Streaming response support for real-time applications  
- Replicate API token authentication with environment variable support
- Comprehensive governance and audit trail integration
- File input/output handling for multimedia models

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.replicate import auto_instrument
    auto_instrument()
    
    # Your existing Replicate code works unchanged with automatic governance
    import replicate
    output = replicate.run("model-name", input={"prompt": "test"})  # Now tracked with GenOps!
    
    # Manual adapter usage for advanced control
    from genops.providers.replicate import GenOpsReplicateAdapter
    
    adapter = GenOpsReplicateAdapter()
    response = adapter.text_generation(
        model="meta/llama-2-70b-chat",
        input={"prompt": "Explain quantum computing"},
        team="research-team",
        project="quantum-ai",
        customer_id="enterprise-123"
    )
"""

import logging
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import IO, Any, Dict, Optional, Union

try:
    import replicate
except ImportError:
    replicate = None

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# Get tracer for OpenTelemetry instrumentation
tracer = trace.get_tracer(__name__)

@dataclass
class ReplicateResponse:
    """Standardized response from Replicate operations with cost tracking."""

    content: Any
    model: str
    cost_usd: float
    latency_ms: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    prediction_id: Optional[str] = None
    hardware_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for telemetry."""
        return asdict(self)

@dataclass
class ReplicateModelInfo:
    """Information about a Replicate model for cost calculation."""

    name: str
    pricing_type: str  # 'time', 'token', 'output', 'hardware'
    base_cost: float
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    hardware_type: Optional[str] = None
    official: bool = False
    category: Optional[str] = None  # 'text', 'image', 'video', 'audio', 'multimodal'

class GenOpsReplicateAdapter:
    """
    GenOps adapter for Replicate with comprehensive cost tracking and governance.
    
    This adapter provides a unified interface for all Replicate models while
    maintaining accurate cost attribution and telemetry export.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        telemetry_enabled: bool = True,
        debug: bool = False
    ):
        """
        Initialize GenOps Replicate adapter.
        
        Args:
            api_token: Replicate API token (defaults to REPLICATE_API_TOKEN env var)
            telemetry_enabled: Enable OpenTelemetry export
            debug: Enable debug logging
        """
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        self.telemetry_enabled = telemetry_enabled
        self.debug = debug

        if not self.api_token:
            logger.warning("REPLICATE_API_TOKEN not found. Set environment variable for authentication.")

        if replicate is None:
            raise ImportError(
                "Replicate SDK not found. Install with: pip install replicate"
            )

        # Configure replicate client
        if self.api_token:
            replicate.Client(api_token=self.api_token)

        # Import pricing and validation modules
        try:
            from .replicate_pricing import ReplicatePricingCalculator
            from .replicate_validation import validate_setup
            self._pricing = ReplicatePricingCalculator()
            self._validator = validate_setup
        except ImportError:
            logger.warning("Replicate pricing/validation modules not available")
            self._pricing = None
            self._validator = None

    def run_model(
        self,
        model: str,
        input: Dict[str, Any],
        *,
        team: Optional[str] = None,
        project: Optional[str] = None,
        customer_id: Optional[str] = None,
        environment: Optional[str] = None,
        cost_center: Optional[str] = None,
        feature: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ReplicateResponse, Iterator[Any]]:
        """
        Run a Replicate model with comprehensive cost tracking and governance.
        
        Args:
            model: Replicate model identifier (e.g., "meta/llama-2-70b-chat")
            input: Input parameters for the model
            team: Team identifier for cost attribution
            project: Project identifier for cost tracking
            customer_id: Customer identifier for billing
            environment: Environment (dev/staging/prod)
            cost_center: Cost center for financial reporting
            feature: Feature identifier for attribution
            stream: Enable streaming response for compatible models
            **kwargs: Additional parameters for replicate.run()
            
        Returns:
            ReplicateResponse or streaming iterator with cost tracking
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        # Create governance attributes for telemetry
        governance_attrs = {
            "genops.operation_id": operation_id,
            "genops.provider": "replicate",
            "genops.model": model,
            "genops.team": team,
            "genops.project": project,
            "genops.customer_id": customer_id,
            "genops.environment": environment,
            "genops.cost_center": cost_center,
            "genops.feature": feature,
            "genops.stream": stream,
        }

        # Remove None values
        governance_attrs = {k: v for k, v in governance_attrs.items() if v is not None}

        with tracer.start_as_current_span(
            "replicate.run_model",
            attributes=governance_attrs
        ) as span:
            try:
                # Get model information for cost calculation
                model_info = self._get_model_info(model)

                # Record input details
                span.set_attribute("genops.input_size", len(str(input)))
                span.set_attribute("genops.model_category", model_info.category or "unknown")
                span.set_attribute("genops.pricing_type", model_info.pricing_type)

                if stream:
                    return self._run_streaming(model, input, model_info, governance_attrs, span, **kwargs)
                else:
                    return self._run_sync(model, input, model_info, governance_attrs, span, start_time, **kwargs)

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Error running Replicate model {model}: {e}")
                raise

    def _run_sync(
        self,
        model: str,
        input: Dict[str, Any],
        model_info: ReplicateModelInfo,
        governance_attrs: Dict[str, Any],
        span: trace.Span,
        start_time: float,
        **kwargs
    ) -> ReplicateResponse:
        """Run model synchronously with cost tracking."""

        # Execute the model
        output = replicate.run(model, input=input, **kwargs)

        # Calculate timing
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Calculate cost based on model type
        cost_usd = self._calculate_cost(model_info, input, output, latency_ms)

        # Create response object
        response = ReplicateResponse(
            content=output,
            model=model,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            hardware_used=model_info.hardware_type,
            processing_time_ms=latency_ms,
            metadata={
                "governance": governance_attrs,
                "model_info": asdict(model_info)
            }
        )

        # Record telemetry
        span.set_attribute("genops.cost_usd", cost_usd)
        span.set_attribute("genops.latency_ms", latency_ms)
        span.set_attribute("genops.success", True)

        if self.debug:
            logger.info(f"Replicate operation completed: {model} - ${cost_usd:.6f} ({latency_ms:.0f}ms)")

        return response

    def _run_streaming(
        self,
        model: str,
        input: Dict[str, Any],
        model_info: ReplicateModelInfo,
        governance_attrs: Dict[str, Any],
        span: trace.Span,
        **kwargs
    ) -> Iterator[Any]:
        """Run model with streaming response and cost tracking."""

        start_time = time.time()
        accumulated_cost = 0.0
        token_count = 0

        try:
            # Use replicate.stream for streaming models
            for chunk in replicate.stream(model, input=input, **kwargs):
                token_count += 1

                # Calculate incremental cost for streaming
                if model_info.pricing_type == 'token' and model_info.output_cost:
                    chunk_cost = model_info.output_cost / 1000  # Assume per-1K tokens
                    accumulated_cost += chunk_cost

                yield chunk

            # Final cost calculation and telemetry
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            if model_info.pricing_type != 'token':
                accumulated_cost = self._calculate_cost(model_info, input, None, latency_ms)

            # Record final telemetry
            span.set_attribute("genops.cost_usd", accumulated_cost)
            span.set_attribute("genops.latency_ms", latency_ms)
            span.set_attribute("genops.tokens_out", token_count)
            span.set_attribute("genops.success", True)

            if self.debug:
                logger.info(f"Replicate streaming completed: {model} - ${accumulated_cost:.6f} ({latency_ms:.0f}ms, {token_count} tokens)")

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def text_generation(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **governance_attrs
    ) -> Union[ReplicateResponse, Iterator[str]]:
        """
        Generate text using Replicate language models.
        
        Convenience method for text generation with common parameters.
        """
        input_params = {"prompt": prompt}

        if max_tokens is not None:
            input_params["max_length"] = max_tokens
        if temperature is not None:
            input_params["temperature"] = temperature

        return self.run_model(model, input_params, stream=stream, **governance_attrs)

    def image_generation(
        self,
        model: str,
        prompt: str,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_images: int = 1,
        **governance_attrs
    ) -> ReplicateResponse:
        """
        Generate images using Replicate image models.
        
        Convenience method for image generation with common parameters.
        """
        input_params = {"prompt": prompt}

        if width is not None:
            input_params["width"] = width
        if height is not None:
            input_params["height"] = height
        if num_images > 1:
            input_params["num_outputs"] = num_images

        return self.run_model(model, input_params, **governance_attrs)

    def video_generation(
        self,
        model: str,
        prompt: str,
        *,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        **governance_attrs
    ) -> ReplicateResponse:
        """
        Generate videos using Replicate video models.
        
        Convenience method for video generation with common parameters.
        """
        input_params = {"prompt": prompt}

        if duration is not None:
            input_params["duration"] = duration
        if fps is not None:
            input_params["fps"] = fps

        return self.run_model(model, input_params, **governance_attrs)

    def audio_processing(
        self,
        model: str,
        audio_input: Union[str, IO],
        *,
        task: Optional[str] = None,
        **governance_attrs
    ) -> ReplicateResponse:
        """
        Process audio using Replicate audio models.
        
        Convenience method for audio processing tasks.
        """
        input_params = {"audio": audio_input}

        if task is not None:
            input_params["task"] = task

        return self.run_model(model, input_params, **governance_attrs)

    def _get_model_info(self, model: str) -> ReplicateModelInfo:
        """Get model information for cost calculation."""
        if self._pricing:
            return self._pricing.get_model_info(model)

        # Fallback model info if pricing module not available
        return ReplicateModelInfo(
            name=model,
            pricing_type='time',
            base_cost=0.001,  # Default $0.001/second
            hardware_type='unknown',
            category='unknown'
        )

    def _calculate_cost(
        self,
        model_info: ReplicateModelInfo,
        input_data: Dict[str, Any],
        output: Any,
        latency_ms: float
    ) -> float:
        """Calculate cost based on model type and usage."""
        if self._pricing:
            return self._pricing.calculate_cost(model_info, input_data, output, latency_ms)

        # Fallback cost calculation
        time_seconds = latency_ms / 1000
        return model_info.base_cost * time_seconds

    def validate_setup(self):
        """Validate Replicate setup and configuration."""
        if self._validator:
            return self._validator()

        # Basic validation
        if not self.api_token:
            return {"success": False, "error": "REPLICATE_API_TOKEN not set"}

        return {"success": True}

# Auto-instrumentation function (CLAUDE.md standard)
def auto_instrument():
    """
    Enable automatic instrumentation of Replicate operations.
    
    This function patches the replicate.run function to automatically
    add GenOps cost tracking and governance to existing Replicate code.
    
    Usage:
        from genops.providers.replicate import auto_instrument
        auto_instrument()
        
        # Your existing code now has automatic GenOps tracking
        import replicate
        output = replicate.run("model-name", input={"prompt": "test"})
    """
    if replicate is None:
        logger.warning("Replicate SDK not available for auto-instrumentation")
        return

    # Store original function
    if not hasattr(replicate, '_original_run'):
        replicate._original_run = replicate.run

    # Create instrumented wrapper
    def instrumented_run(model, input, **kwargs):
        """Instrumented version of replicate.run with GenOps tracking."""
        adapter = GenOpsReplicateAdapter()

        # Extract governance attributes from kwargs if present
        governance_attrs = {}
        for attr in ['team', 'project', 'customer_id', 'environment', 'cost_center', 'feature']:
            if attr in kwargs:
                governance_attrs[attr] = kwargs.pop(attr)

        # Use adapter for tracking
        response = adapter.run_model(model, input, **governance_attrs, **kwargs)

        # Return raw content for compatibility
        return response.content

    # Patch the function
    replicate.run = instrumented_run

    logger.info("GenOps auto-instrumentation enabled for Replicate")

# Convenience function for creating adapter instances
def instrument_replicate(api_token: Optional[str] = None, **kwargs) -> GenOpsReplicateAdapter:
    """
    Create and configure a GenOps Replicate adapter.
    
    Args:
        api_token: Replicate API token (optional, uses env var if not provided)
        **kwargs: Additional configuration options
        
    Returns:
        Configured GenOpsReplicateAdapter instance
    """
    return GenOpsReplicateAdapter(api_token=api_token, **kwargs)

# Export main classes and functions
__all__ = [
    'GenOpsReplicateAdapter',
    'ReplicateResponse',
    'ReplicateModelInfo',
    'auto_instrument',
    'instrument_replicate'
]
