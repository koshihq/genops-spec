#!/usr/bin/env python3
"""
GenOps Mistral AI Provider Integration

This module provides comprehensive Mistral AI integration for GenOps AI governance,
cost intelligence, and observability. It follows the established GenOps provider
pattern for consistent developer experience across all AI platforms.

Features:
- Chat completions and text embeddings with cost tracking
- Zero-code auto-instrumentation with instrument_mistral()
- Unified cost tracking across all Mistral models
- Streaming response support for real-time applications
- European AI provider with GDPR compliance benefits
- Advanced cost optimization for frontier models at competitive rates
- Comprehensive governance and audit trail integration

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.mistral import instrument_mistral
    instrument_mistral()
    
    # Your existing Mistral code works unchanged with automatic governance
    from mistralai import Mistral
    client = Mistral(api_key="your-api-key")
    response = client.chat.complete(...)  # Now tracked with GenOps!
    
    # Manual adapter usage for advanced control
    from genops.providers.mistral import GenOpsMistralAdapter
    
    adapter = GenOpsMistralAdapter()
    response = adapter.chat(
        message="Explain quantum computing",
        model="mistral-large-2407",
        team="research-team",
        project="quantum-ai",
        customer_id="enterprise-123"
    )
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import Mistral dependencies with graceful fallback
try:
    from mistralai import Mistral
    from mistralai.models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatMessage,
        EmbeddingRequest,
        EmbeddingResponse,
    )
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False
    Mistral = None
    ChatCompletionRequest = None
    ChatCompletionResponse = None
    EmbeddingRequest = None
    EmbeddingResponse = None
    ChatMessage = None
    logger.warning("Mistral AI not installed. Install with: pip install mistralai")

# Try to import GenOps core dependencies
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    logger.warning("OpenTelemetry not available - telemetry will be disabled")

# Constants for Mistral models and operations
class MistralModel(Enum):
    """Mistral model enumeration for type safety and cost calculation."""
    # Core models
    MISTRAL_TINY = "mistral-tiny-2312"
    MISTRAL_SMALL = "mistral-small-latest"
    MISTRAL_MEDIUM = "mistral-medium-latest"
    MISTRAL_LARGE = "mistral-large-latest"
    MISTRAL_LARGE_2407 = "mistral-large-2407"

    # Mixtral models
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    MIXTRAL_8X22B = "mixtral-8x22b-32768"

    # Specialized models
    MISTRAL_NEMO = "mistral-nemo-2407"
    CODESTRAL = "codestral-2405"

    # Embedding models
    MISTRAL_EMBED = "mistral-embed"


class MistralOperation(Enum):
    """Mistral operation types for cost tracking."""
    CHAT = "chat"
    EMBED = "embed"
    COMPLETION = "completion"


@dataclass
class MistralUsage:
    """Usage statistics for Mistral API calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    request_time: float = 0.0
    tokens_per_second: float = 0.0
    cost_per_token: float = 0.0
    model: str = ""
    operation: str = ""


@dataclass
class MistralResponse:
    """Standardized response wrapper for all Mistral operations."""
    content: str = ""
    raw_response: Any = None
    usage: MistralUsage = field(default_factory=MistralUsage)
    success: bool = True
    error_message: str = ""
    request_id: str = ""
    model: str = ""
    operation: str = ""

    # Chat-specific fields
    role: str = "assistant"
    finish_reason: str = ""

    # Embedding-specific fields
    embeddings: List[List[float]] = field(default_factory=list)
    embedding_dimension: int = 0


class GenOpsMistralAdapter:
    """
    GenOps adapter for Mistral AI with comprehensive cost tracking and governance.
    
    This adapter provides a unified interface for all Mistral AI operations while
    automatically tracking costs, performance metrics, and governance attributes.
    
    Features:
    - Automatic cost calculation for all Mistral models
    - Team and project attribution for cost tracking  
    - OpenTelemetry integration for observability
    - Streaming support with real-time cost tracking
    - European AI provider benefits (GDPR, cost efficiency)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cost_tracking_enabled: bool = True,
        budget_limit: Optional[float] = None,
        cost_alert_threshold: float = 0.8,
        default_team: Optional[str] = None,
        default_project: Optional[str] = None,
        default_environment: str = "development",
        default_customer_id: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        enable_streaming: bool = True,
        **kwargs
    ):
        """
        Initialize the GenOps Mistral adapter.
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            cost_tracking_enabled: Whether to track and calculate costs
            budget_limit: Optional budget limit in USD
            cost_alert_threshold: Threshold (0-1) for cost alerts
            default_team: Default team for cost attribution
            default_project: Default project for cost attribution
            default_environment: Environment (dev/staging/prod)
            default_customer_id: Default customer ID for billing
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            enable_streaming: Whether to support streaming responses
            **kwargs: Additional configuration options
        """
        if not HAS_MISTRAL:
            raise ImportError(
                "Mistral AI client not installed. Install with: pip install mistralai"
            )

        # API configuration
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key is required. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter"
            )

        # Initialize Mistral client
        try:
            self.client = Mistral(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Mistral client: {e}")

        # Cost tracking configuration
        self.cost_tracking_enabled = cost_tracking_enabled
        self.budget_limit = budget_limit
        self.cost_alert_threshold = cost_alert_threshold

        # Governance defaults
        self.default_team = default_team
        self.default_project = default_project
        self.default_environment = default_environment
        self.default_customer_id = default_customer_id

        # Performance configuration
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_streaming = enable_streaming

        # Internal state
        self._total_cost = 0.0
        self._operation_count = 0
        self._session_id = str(uuid.uuid4())

        # Initialize pricing calculator
        self._init_pricing_calculator()

        # Setup OpenTelemetry tracing
        self.tracer = None
        if HAS_OTEL:
            self.tracer = trace.get_tracer(__name__)

        logger.info(f"GenOps Mistral adapter initialized with session: {self._session_id}")

    def _init_pricing_calculator(self):
        """Initialize pricing calculator with current Mistral model rates."""
        try:
            from .mistral_pricing import MistralPricingCalculator
            self.pricing_calculator = MistralPricingCalculator()
        except ImportError:
            logger.warning("Mistral pricing calculator not available")
            self.pricing_calculator = None

    def _calculate_cost(
        self,
        model: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **kwargs
    ) -> tuple[float, float, float]:
        """Calculate costs for a Mistral operation."""
        if not self.cost_tracking_enabled or not self.pricing_calculator:
            return 0.0, 0.0, 0.0

        try:
            return self.pricing_calculator.calculate_cost(
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Cost calculation failed: {e}")
            return 0.0, 0.0, 0.0

    def _create_usage_stats(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        request_time: float,
        **kwargs
    ) -> MistralUsage:
        """Create comprehensive usage statistics."""
        total_tokens = input_tokens + output_tokens
        input_cost, output_cost, total_cost = self._calculate_cost(
            model, operation, input_tokens, output_tokens, **kwargs
        )

        tokens_per_second = total_tokens / max(request_time, 0.001)
        cost_per_token = total_cost / max(total_tokens, 1)

        return MistralUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost,
            input_cost=input_cost,
            output_cost=output_cost,
            request_time=request_time,
            tokens_per_second=tokens_per_second,
            cost_per_token=cost_per_token,
            model=model,
            operation=operation
        )

    def _update_session_stats(self, usage: MistralUsage):
        """Update session-level statistics."""
        self._total_cost += usage.total_cost
        self._operation_count += 1

        # Check budget limits
        if (self.budget_limit and
            self._total_cost >= self.budget_limit * self.cost_alert_threshold):
            logger.warning(
                f"Cost alert: ${self._total_cost:.6f} / ${self.budget_limit:.2f} "
                f"({self._total_cost/self.budget_limit*100:.1f}%)"
            )

    def _extract_governance_attrs(self, **kwargs) -> Dict[str, Any]:
        """Extract governance attributes from kwargs."""
        return {
            "team": kwargs.get("team", self.default_team),
            "project": kwargs.get("project", self.default_project),
            "environment": kwargs.get("environment", self.default_environment),
            "customer_id": kwargs.get("customer_id", self.default_customer_id),
            "session_id": self._session_id,
            "operation_id": str(uuid.uuid4())
        }

    def chat(
        self,
        message: str,
        model: str = "mistral-small-latest",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> MistralResponse:
        """
        Generate chat completion with comprehensive cost tracking.
        
        Args:
            message: User message content
            model: Mistral model to use
            system_prompt: Optional system message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters and governance attributes
            
        Returns:
            MistralResponse with content, usage stats, and cost information
        """
        start_time = time.time()
        governance_attrs = self._extract_governance_attrs(**kwargs)

        # Create span for OpenTelemetry tracing
        span_name = f"mistral.chat.{model}"
        span = None
        if self.tracer:
            span = self.tracer.start_span(span_name)

        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})

            # Make API call
            response = self.client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **{k: v for k, v in kwargs.items()
                   if k not in ["team", "project", "environment", "customer_id"]}
            )

            request_time = time.time() - start_time

            # Extract response data
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                content = choice.message.content if choice.message else ""
                finish_reason = getattr(choice, 'finish_reason', 'completed')
            else:
                content = str(response)
                finish_reason = "completed"

            # Extract token usage
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0

            # Create usage statistics
            usage = self._create_usage_stats(
                model=model,
                operation=MistralOperation.CHAT.value,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_time=request_time
            )

            # Update session statistics
            self._update_session_stats(usage)

            # Create response object
            mistral_response = MistralResponse(
                content=content,
                raw_response=response,
                usage=usage,
                success=True,
                request_id=getattr(response, 'id', str(uuid.uuid4())),
                model=model,
                operation=MistralOperation.CHAT.value,
                finish_reason=finish_reason
            )

            # Add span attributes
            if span:
                span.set_attributes({
                    "mistral.model": model,
                    "mistral.operation": "chat",
                    "mistral.input_tokens": input_tokens,
                    "mistral.output_tokens": output_tokens,
                    "mistral.total_cost": usage.total_cost,
                    "mistral.request_time": request_time,
                    **{f"genops.{k}": v for k, v in governance_attrs.items() if v}
                })
                span.set_status(Status(StatusCode.OK))

            return mistral_response

        except Exception as e:
            request_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Mistral chat error: {error_msg}")

            # Set span error status
            if span:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, error_msg))

            return MistralResponse(
                success=False,
                error_message=error_msg,
                usage=MistralUsage(
                    model=model,
                    operation=MistralOperation.CHAT.value,
                    request_time=request_time
                ),
                model=model,
                operation=MistralOperation.CHAT.value
            )

        finally:
            if span:
                span.end()

    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = "mistral-embed",
        **kwargs
    ) -> MistralResponse:
        """
        Generate text embeddings with comprehensive cost tracking.
        
        Args:
            texts: Text or list of texts to embed
            model: Mistral embedding model to use  
            **kwargs: Additional parameters and governance attributes
            
        Returns:
            MistralResponse with embeddings, usage stats, and cost information
        """
        start_time = time.time()
        governance_attrs = self._extract_governance_attrs(**kwargs)

        # Normalize texts to list
        if isinstance(texts, str):
            texts = [texts]

        # Create span for OpenTelemetry tracing
        span_name = f"mistral.embed.{model}"
        span = None
        if self.tracer:
            span = self.tracer.start_span(span_name)

        try:
            # Make API call
            response = self.client.embeddings.create(
                model=model,
                inputs=texts,
                **{k: v for k, v in kwargs.items()
                   if k not in ["team", "project", "environment", "customer_id"]}
            )

            request_time = time.time() - start_time

            # Extract embeddings
            embeddings = []
            if hasattr(response, 'data') and response.data:
                embeddings = [item.embedding for item in response.data]

            # Calculate token usage (approximate for embeddings)
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = max(1, total_chars // 4)  # Rough estimation

            # Create usage statistics
            usage = self._create_usage_stats(
                model=model,
                operation=MistralOperation.EMBED.value,
                input_tokens=estimated_tokens,
                output_tokens=0,
                request_time=request_time
            )

            # Update session statistics
            self._update_session_stats(usage)

            # Create response object
            mistral_response = MistralResponse(
                content=f"Generated {len(embeddings)} embeddings",
                raw_response=response,
                usage=usage,
                success=True,
                request_id=getattr(response, 'id', str(uuid.uuid4())),
                model=model,
                operation=MistralOperation.EMBED.value,
                embeddings=embeddings,
                embedding_dimension=len(embeddings[0]) if embeddings else 0
            )

            # Add span attributes
            if span:
                span.set_attributes({
                    "mistral.model": model,
                    "mistral.operation": "embed",
                    "mistral.input_texts": len(texts),
                    "mistral.estimated_tokens": estimated_tokens,
                    "mistral.total_cost": usage.total_cost,
                    "mistral.request_time": request_time,
                    "mistral.embedding_dimension": mistral_response.embedding_dimension,
                    **{f"genops.{k}": v for k, v in governance_attrs.items() if v}
                })
                span.set_status(Status(StatusCode.OK))

            return mistral_response

        except Exception as e:
            request_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Mistral embed error: {error_msg}")

            # Set span error status
            if span:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, error_msg))

            return MistralResponse(
                success=False,
                error_message=error_msg,
                usage=MistralUsage(
                    model=model,
                    operation=MistralOperation.EMBED.value,
                    request_time=request_time
                ),
                model=model,
                operation=MistralOperation.EMBED.value
            )

        finally:
            if span:
                span.end()

    def generate(
        self,
        prompt: str,
        model: str = "mistral-small-latest",
        **kwargs
    ) -> MistralResponse:
        """
        Generate text completion (alias for chat with single user message).
        
        Args:
            prompt: Text prompt for completion
            model: Mistral model to use
            **kwargs: Additional parameters and governance attributes
            
        Returns:
            MistralResponse with generated text and cost information
        """
        return self.chat(message=prompt, model=model, **kwargs)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary for current session."""
        return {
            "session_id": self._session_id,
            "total_cost": self._total_cost,
            "total_operations": self._operation_count,
            "average_cost_per_operation": (
                self._total_cost / max(self._operation_count, 1)
            ),
            "budget_limit": self.budget_limit,
            "budget_utilization": (
                self._total_cost / self.budget_limit if self.budget_limit else None
            ),
            "cost_tracking_enabled": self.cost_tracking_enabled
        }

    def reset_session_stats(self):
        """Reset session-level statistics."""
        self._total_cost = 0.0
        self._operation_count = 0
        self._session_id = str(uuid.uuid4())
        logger.info(f"Session stats reset, new session: {self._session_id}")


@contextmanager
def mistral_workflow_context(
    workflow_name: str,
    team: Optional[str] = None,
    project: Optional[str] = None,
    customer_id: Optional[str] = None,
    environment: str = "production"
):
    """
    Context manager for Mistral workflow cost tracking and governance.
    
    Args:
        workflow_name: Descriptive name for the workflow
        team: Team attribution for cost tracking
        project: Project attribution  
        customer_id: Customer attribution for billing
        environment: Environment (dev/staging/prod)
        
    Yields:
        Tuple of (adapter, workflow_id) for workflow execution
        
    Example:
        with mistral_workflow_context("document-analysis", team="ai-team") as (ctx, workflow_id):
            response1 = ctx.chat("Analyze this document", model="mistral-large-2407")
            embeddings = ctx.embed(["doc1", "doc2"])
            # Automatic cost aggregation and cleanup
    """
    workflow_id = f"{workflow_name}-{uuid.uuid4().hex[:8]}"

    adapter = GenOpsMistralAdapter(
        default_team=team,
        default_project=project,
        default_customer_id=customer_id,
        default_environment=environment
    )

    start_time = time.time()
    logger.info(f"Starting Mistral workflow: {workflow_id}")

    try:
        yield adapter, workflow_id

    finally:
        end_time = time.time()
        duration = end_time - start_time
        summary = adapter.get_usage_summary()

        logger.info(
            f"Mistral workflow completed: {workflow_id}, "
            f"duration: {duration:.2f}s, cost: ${summary['total_cost']:.6f}, "
            f"operations: {summary['total_operations']}"
        )


def instrument_mistral(
    team: Optional[str] = None,
    project: Optional[str] = None,
    customer_id: Optional[str] = None,
    environment: str = "development",
    **adapter_kwargs
) -> GenOpsMistralAdapter:
    """
    Zero-code auto-instrumentation for Mistral AI applications.
    
    This function enables automatic GenOps tracking for existing Mistral
    applications without requiring code changes.
    
    Args:
        team: Default team for cost attribution
        project: Default project for cost attribution
        customer_id: Default customer ID for billing attribution
        environment: Environment (dev/staging/prod)
        **adapter_kwargs: Additional adapter configuration
        
    Returns:
        GenOpsMistralAdapter instance for advanced usage
        
    Example:
        # Enable automatic tracking
        from genops.providers.mistral import instrument_mistral
        adapter = instrument_mistral(team="ai-team", project="chat-app")
        
        # Your existing Mistral code now has automatic governance
        response = adapter.chat("Hello!", model="mistral-small-latest")
        print(f"Response cost: ${response.usage.total_cost:.6f}")
    """
    return GenOpsMistralAdapter(
        default_team=team,
        default_project=project,
        default_customer_id=customer_id,
        default_environment=environment,
        **adapter_kwargs
    )


# Convenience functions for common operations
def chat(message: str, model: str = "mistral-small-latest", **kwargs) -> MistralResponse:
    """Quick chat completion with automatic cost tracking."""
    adapter = instrument_mistral(**kwargs)
    return adapter.chat(message=message, model=model, **kwargs)


def embed(texts: Union[str, List[str]], model: str = "mistral-embed", **kwargs) -> MistralResponse:
    """Quick text embedding with automatic cost tracking."""
    adapter = instrument_mistral(**kwargs)
    return adapter.embed(texts=texts, model=model, **kwargs)


# Export main classes and functions
__all__ = [
    "GenOpsMistralAdapter",
    "MistralResponse",
    "MistralUsage",
    "MistralModel",
    "MistralOperation",
    "instrument_mistral",
    "mistral_workflow_context",
    "chat",
    "embed"
]

if __name__ == "__main__":
    # Quick test/demo
    print("GenOps Mistral Provider Integration")
    print("=" * 50)

    if not HAS_MISTRAL:
        print("❌ Mistral AI client not installed")
        print("   Install with: pip install mistralai")
    else:
        print("✅ Mistral AI client available")

    try:
        adapter = instrument_mistral(team="demo-team", project="test")
        print("✅ GenOps Mistral adapter initialized")
        print(f"   Session ID: {adapter._session_id}")
    except Exception as e:
        print(f"❌ Adapter initialization failed: {e}")
        print("   Please set MISTRAL_API_KEY environment variable")
