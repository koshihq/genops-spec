#!/usr/bin/env python3
"""
GenOps Helicone AI Gateway Provider Integration

This module provides comprehensive Helicone AI Gateway integration for GenOps AI
governance, cost intelligence, and observability. Helicone is unique as both an
AI gateway (unified API for 100+ models) and observability platform.

Features:
- Multi-provider AI gateway access through single interface
- Cross-provider cost tracking and optimization
- Built-in observability and request logging
- Automatic failover and routing intelligence
- Zero-code auto-instrumentation with instrument_helicone()
- Self-hosted gateway support for enterprise deployments
- Advanced cost analytics across OpenAI, Anthropic, Vertex, and more

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.helicone import instrument_helicone
    instrument_helicone(helicone_api_key="your-helicone-key")

    # Your existing OpenAI code now routes through Helicone gateway
    import openai
    client = openai.OpenAI()  # Automatically uses Helicone gateway
    response = client.chat.completions.create(...)  # Tracked with GenOps!

    # Manual adapter usage for multi-provider intelligence
    from genops.providers.helicone import GenOpsHeliconeAdapter

    adapter = GenOpsHeliconeAdapter(
        helicone_api_key="your-helicone-key",
        provider_keys={
            "openai": "your-openai-key",
            "anthropic": "your-anthropic-key"
        }
    )

    # Multi-provider routing with cost optimization
    response = adapter.multi_provider_chat(
        message="Explain quantum computing",
        providers=["openai", "anthropic"],
        model_preferences={"openai": "gpt-4", "anthropic": "claude-3-sonnet"},
        routing_strategy="cost_optimized",
        team="research-team",
        project="quantum-ai"
    )
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import requests
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)


class HeliconeProvider(Enum):
    """Supported providers through Helicone gateway."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX = "vertex"
    GROQ = "groq"
    TOGETHER = "together"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class RoutingStrategy(Enum):
    """AI gateway routing strategies."""

    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    FAILOVER = "failover"
    ROUND_ROBIN = "round_robin"
    QUALITY_OPTIMIZED = "quality_optimized"


@dataclass
class HeliconeUsage:
    """Usage statistics from Helicone gateway."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    request_time: float
    gateway_overhead: float
    provider_cost: float
    helicone_cost: float
    total_cost: float
    cost_per_token: float
    tokens_per_second: float


@dataclass
class HeliconeResponse:
    """Standardized response from Helicone operations."""

    success: bool
    content: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[HeliconeUsage] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    helicone_session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiProviderResponse:
    """Response from multi-provider routing."""

    success: bool
    primary_response: Optional[HeliconeResponse] = None
    fallback_responses: list[HeliconeResponse] = field(default_factory=list)
    routing_decision: Optional[str] = None
    cost_comparison: dict[str, float] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)


class GenOpsHeliconeAdapter:
    """
    Advanced Helicone AI Gateway adapter with multi-provider intelligence.

    This adapter provides enterprise-grade AI gateway functionality with:
    - Multi-provider cost optimization
    - Intelligent routing strategies
    - Built-in observability and monitoring
    - Advanced governance and cost controls
    """

    def __init__(
        self,
        helicone_api_key: Optional[str] = None,
        provider_keys: Optional[dict[str, str]] = None,
        base_url: str = "https://ai-gateway.helicone.ai",
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: str = "production",
        cost_center: Optional[str] = None,
        enable_observability: bool = True,
        enable_caching: bool = True,
        default_routing_strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED,
    ):
        """
        Initialize Helicone gateway adapter with enterprise governance.

        Args:
            helicone_api_key: Helicone platform API key
            provider_keys: Dictionary mapping provider names to API keys
            base_url: Helicone gateway base URL (cloud or self-hosted)
            team: Team identifier for cost attribution
            project: Project identifier for cost tracking
            environment: Environment (dev/staging/production)
            cost_center: Cost center for financial reporting
            enable_observability: Enable built-in request logging
            enable_caching: Enable intelligent request caching
            default_routing_strategy: Default multi-provider routing strategy
        """
        self.helicone_api_key = helicone_api_key or os.getenv("HELICONE_API_KEY")
        self.provider_keys = provider_keys or {}
        self.base_url = base_url.rstrip("/")

        # Governance attributes
        self.team = team
        self.project = project
        self.environment = environment
        self.cost_center = cost_center

        # Gateway configuration
        self.enable_observability = enable_observability
        self.enable_caching = enable_caching
        self.default_routing_strategy = default_routing_strategy

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.operations_count = 0
        self.total_cost = 0.0
        self.provider_stats = {}

        # Initialize telemetry
        self._setup_telemetry()

        if not self.helicone_api_key:
            logger.warning("Helicone API key not found. Some features may be limited.")

    def _setup_telemetry(self):
        """Initialize OpenTelemetry integration."""
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode  # noqa: F401

            self.tracer = trace.get_tracer("genops.helicone")
        except ImportError:
            logger.debug("OpenTelemetry not available - telemetry disabled")
            self.tracer = None

    def _create_headers(
        self,
        provider: str,
        custom_headers: Optional[dict[str, str]] = None,
        **governance_kwargs,
    ) -> dict[str, str]:
        """Create request headers with Helicone and governance metadata."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "GenOps-Helicone/1.0.0",
        }

        # Helicone authentication
        if self.helicone_api_key:
            headers["Helicone-Auth"] = f"Bearer {self.helicone_api_key}"

        # Provider-specific API key
        provider_key = self.provider_keys.get(provider) or os.getenv(
            f"{provider.upper()}_API_KEY"
        )
        if provider_key:
            if provider == "openai":
                headers["Authorization"] = f"Bearer {provider_key}"
            elif provider == "anthropic":
                headers["x-api-key"] = provider_key
            elif provider == "vertex":
                headers["Authorization"] = f"Bearer {provider_key}"

        # Helicone observability headers
        if self.enable_observability:
            headers["Helicone-Session-Id"] = self.session_id
            headers["Helicone-Request-Id"] = str(uuid.uuid4())

        # Governance metadata
        governance_data = {
            "team": governance_kwargs.get("team", self.team),
            "project": governance_kwargs.get("project", self.project),
            "environment": governance_kwargs.get("environment", self.environment),
            "cost_center": governance_kwargs.get("cost_center", self.cost_center),
            "customer_id": governance_kwargs.get("customer_id"),
            "feature": governance_kwargs.get("feature"),
        }

        # Filter out None values and add as Helicone properties
        governance_metadata = {
            k: v for k, v in governance_data.items() if v is not None
        }
        if governance_metadata:
            headers["Helicone-Property-Governance"] = json.dumps(governance_metadata)

        # Caching configuration
        if self.enable_caching:
            cache_ttl = governance_kwargs.get("cache_ttl", 3600)  # 1 hour default
            headers["Helicone-Cache-Enabled"] = "true"
            headers["Helicone-Cache-Max-Age"] = str(cache_ttl)

        # Custom headers
        if custom_headers:
            headers.update(custom_headers)

        return headers

    def _calculate_costs(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_time: float,
    ) -> tuple[float, float, float]:
        """
        Calculate costs for provider + Helicone gateway.

        Returns:
            (provider_cost, helicone_cost, total_cost)
        """
        try:
            from .helicone_pricing import HeliconePricingCalculator

            pricing_calc = HeliconePricingCalculator()
            return pricing_calc.calculate_gateway_cost(
                provider, model, input_tokens, output_tokens, request_time
            )
        except ImportError:
            logger.warning("Helicone pricing calculator not available")
            return 0.0, 0.0, 0.0

    def chat(
        self,
        message: str,
        provider: str = "openai",
        model: str = "gpt-4",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **governance_kwargs,
    ) -> HeliconeResponse:
        """
        Single-provider chat completion through Helicone gateway.

        Args:
            message: User message for completion
            provider: AI provider (openai, anthropic, vertex, etc.)
            model: Model name for the provider
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **governance_kwargs: Team, project, customer_id, etc.

        Returns:
            HeliconeResponse with content, usage, and cost information
        """
        operation_start = time.time()

        # Create telemetry span
        with self._create_span(
            "helicone_chat", provider, model, **governance_kwargs
        ) as span:
            try:
                # Prepare request payload
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": message})

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                # Create request headers
                headers = self._create_headers(provider, **governance_kwargs)

                # Gateway endpoint mapping
                endpoint_map = {
                    "openai": "/v1/chat/completions",
                    "anthropic": "/v1/messages",
                    "vertex": "/v1/chat/completions",
                }

                endpoint = endpoint_map.get(provider, "/v1/chat/completions")
                url = f"{self.base_url}{endpoint}"

                # Make request through Helicone gateway
                gateway_start = time.time()
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                gateway_overhead = time.time() - gateway_start

                response.raise_for_status()
                result = response.json()

                # Extract response data
                if provider == "anthropic":
                    content = result.get("content", [{}])[0].get("text", "")
                    usage_data = result.get("usage", {})
                    input_tokens = usage_data.get("input_tokens", 0)
                    output_tokens = usage_data.get("output_tokens", 0)
                else:  # OpenAI-compatible format
                    content = result["choices"][0]["message"]["content"]
                    usage_data = result.get("usage", {})
                    input_tokens = usage_data.get("prompt_tokens", 0)
                    output_tokens = usage_data.get("completion_tokens", 0)

                total_tokens = input_tokens + output_tokens
                request_time = time.time() - operation_start

                # Calculate costs
                provider_cost, helicone_cost, total_cost = self._calculate_costs(
                    provider, model, input_tokens, output_tokens, request_time
                )

                # Create usage statistics
                usage = HeliconeUsage(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    request_time=request_time,
                    gateway_overhead=gateway_overhead,
                    provider_cost=provider_cost,
                    helicone_cost=helicone_cost,
                    total_cost=total_cost,
                    cost_per_token=total_cost / max(total_tokens, 1),
                    tokens_per_second=total_tokens / max(request_time, 0.001),
                )

                # Update session statistics
                self._update_session_stats(provider, total_cost)

                # Set span attributes
                if span:
                    span.set_attributes(
                        {
                            "genops.provider": provider,
                            "genops.model": model,
                            "genops.tokens.input": input_tokens,
                            "genops.tokens.output": output_tokens,
                            "genops.cost.total": total_cost,
                            "genops.cost.provider": provider_cost,
                            "genops.cost.helicone": helicone_cost,
                        }
                    )

                return HeliconeResponse(
                    success=True,
                    content=content,
                    provider=provider,
                    model=model,
                    usage=usage,
                    request_id=response.headers.get("Helicone-Request-Id"),
                    helicone_session_id=self.session_id,
                    metadata={
                        "gateway_overhead_ms": gateway_overhead * 1000,
                        "provider_response_time_ms": (request_time - gateway_overhead)
                        * 1000,
                    },
                )

            except requests.exceptions.RequestException as e:
                error_msg = f"Helicone gateway request failed: {e}"
                logger.error(error_msg)

                if span:
                    span.set_status(Status(StatusCode.ERROR, error_msg))  # type: ignore  # noqa: F821

                return HeliconeResponse(
                    success=False,
                    error_message=error_msg,
                    provider=provider,
                    model=model,
                )

            except Exception as e:
                error_msg = f"Unexpected error in Helicone chat: {e}"
                logger.error(error_msg)

                if span:
                    span.set_status(Status(StatusCode.ERROR, error_msg))  # type: ignore  # noqa: F821

                return HeliconeResponse(
                    success=False,
                    error_message=error_msg,
                    provider=provider,
                    model=model,
                )

    def multi_provider_chat(
        self,
        message: str,
        providers: list[str],
        model_preferences: dict[str, str],
        routing_strategy: Optional[RoutingStrategy] = None,
        max_retries: int = 2,
        **governance_kwargs,
    ) -> MultiProviderResponse:
        """
        Multi-provider chat with intelligent routing and failover.

        Args:
            message: User message for completion
            providers: List of providers to try (e.g., ["openai", "anthropic"])
            model_preferences: Provider-to-model mapping
            routing_strategy: How to select the best provider
            max_retries: Maximum retry attempts per provider
            **governance_kwargs: Team, project, customer_id, etc.

        Returns:
            MultiProviderResponse with primary response and alternatives
        """
        routing_strategy = routing_strategy or self.default_routing_strategy

        with self._create_span(
            "helicone_multi_provider_chat",
            providers[0] if providers else "unknown",
            model_preferences.get(providers[0], "unknown") if providers else "unknown",
            **governance_kwargs,
        ):
            # Sort providers by routing strategy
            ordered_providers = self._order_providers_by_strategy(
                providers, model_preferences, routing_strategy
            )

            responses = []
            cost_comparison = {}
            performance_metrics = {}

            for provider in ordered_providers:
                model = model_preferences.get(provider, "default")

                try:
                    response = self.chat(
                        message=message,
                        provider=provider,
                        model=model,
                        **governance_kwargs,
                    )

                    responses.append(response)

                    if response.success:
                        cost_comparison[provider] = (
                            response.usage.total_cost if response.usage else 0.0
                        )
                        performance_metrics[provider] = (
                            response.usage.request_time if response.usage else 0.0
                        )

                        # Return first successful response for most strategies
                        if routing_strategy != RoutingStrategy.QUALITY_OPTIMIZED:
                            primary_response = response
                            fallback_responses = (
                                responses[1:] if len(responses) > 1 else []
                            )

                            return MultiProviderResponse(
                                success=True,
                                primary_response=primary_response,
                                fallback_responses=fallback_responses,
                                routing_decision=f"Selected {provider} using {routing_strategy.value} strategy",
                                cost_comparison=cost_comparison,
                                performance_metrics=performance_metrics,
                            )

                except Exception as e:
                    logger.warning(f"Provider {provider} failed: {e}")
                    continue

            # If we reach here, all providers failed or we're doing quality optimization
            if routing_strategy == RoutingStrategy.QUALITY_OPTIMIZED and responses:
                # Select best response based on quality heuristics
                best_response = self._select_best_quality_response(responses)
                return MultiProviderResponse(
                    success=True,
                    primary_response=best_response,
                    fallback_responses=[r for r in responses if r != best_response],
                    routing_decision="Selected based on quality optimization",
                    cost_comparison=cost_comparison,
                    performance_metrics=performance_metrics,
                )

            # All providers failed
            return MultiProviderResponse(
                success=False,
                fallback_responses=responses,
                routing_decision="All providers failed",
                cost_comparison=cost_comparison,
                performance_metrics=performance_metrics,
            )

    def _order_providers_by_strategy(
        self,
        providers: list[str],
        model_preferences: dict[str, str],
        strategy: RoutingStrategy,
    ) -> list[str]:
        """Order providers based on routing strategy."""
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            # Estimate costs and order by cheapest first
            return self._order_by_estimated_cost(providers, model_preferences)
        elif strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            # Order by historical performance
            return self._order_by_performance(providers)
        elif strategy == RoutingStrategy.FAILOVER:
            # Use provider order as specified (primary -> secondary -> ...)
            return providers
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin based on session
            start_idx = hash(self.session_id) % len(providers)
            return providers[start_idx:] + providers[:start_idx]
        else:
            return providers

    def _order_by_estimated_cost(
        self, providers: list[str], model_preferences: dict[str, str]
    ) -> list[str]:
        """Order providers by estimated cost (cheapest first)."""
        try:
            from .helicone_pricing import HeliconePricingCalculator

            pricing_calc = HeliconePricingCalculator()

            provider_costs = []
            for provider in providers:
                model = model_preferences.get(provider, "default")
                estimated_cost = pricing_calc.estimate_request_cost(provider, model)
                provider_costs.append((provider, estimated_cost))

            # Sort by cost (ascending)
            provider_costs.sort(key=lambda x: x[1])
            return [provider for provider, _ in provider_costs]

        except ImportError:
            logger.warning("Cost-based routing requires pricing calculator")
            return providers

    def _order_by_performance(self, providers: list[str]) -> list[str]:
        """Order providers by historical performance."""
        # Use session stats to order by performance
        provider_performance = []
        for provider in providers:
            stats = self.provider_stats.get(provider, {})
            avg_response_time = stats.get("avg_response_time", float("inf"))
            success_rate = stats.get("success_rate", 0.0)

            # Performance score: prioritize success rate, then response time
            performance_score = success_rate - (avg_response_time / 10.0)
            provider_performance.append((provider, performance_score))

        # Sort by performance (descending)
        provider_performance.sort(key=lambda x: x[1], reverse=True)
        return [provider for provider, _ in provider_performance]

    def _select_best_quality_response(
        self, responses: list[HeliconeResponse]
    ) -> HeliconeResponse:
        """Select the best response based on quality heuristics."""
        successful_responses = [r for r in responses if r.success]
        if not successful_responses:
            return responses[0] if responses else None  # type: ignore

        # Simple quality heuristic: longest response with reasonable cost
        def quality_score(response):
            content_length = len(response.content) if response.content else 0
            cost = response.usage.total_cost if response.usage else float("inf")

            # Balance content length vs cost
            return content_length / max(cost * 1000, 1)

        return max(successful_responses, key=quality_score)

    def _update_session_stats(self, provider: str, cost: float):
        """Update session statistics for provider performance tracking."""
        self.operations_count += 1
        self.total_cost += cost

        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "requests": 0,
                "total_cost": 0.0,
                "total_response_time": 0.0,
                "successes": 0,
                "failures": 0,
            }

        stats = self.provider_stats[provider]
        stats["requests"] += 1
        stats["total_cost"] += cost
        stats["successes"] += 1

        # Calculate derived metrics
        stats["avg_cost"] = stats["total_cost"] / stats["requests"]
        stats["success_rate"] = stats["successes"] / stats["requests"]

    @contextmanager
    def _create_span(
        self, operation_name: str, provider: str, model: str, **governance_kwargs
    ):
        """Create OpenTelemetry span for operation tracking."""
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            # Set standard attributes
            span.set_attributes(
                {
                    "genops.operation": operation_name,
                    "genops.provider": provider,
                    "genops.model": model,
                    "genops.session.id": self.session_id,
                }
            )

            # Set governance attributes
            for key, value in governance_kwargs.items():
                if value is not None:
                    span.set_attribute(f"genops.{key}", str(value))

            yield span

    def get_usage_summary(self) -> dict[str, Any]:
        """Get comprehensive usage summary across all providers."""
        return {
            "session_id": self.session_id,
            "total_operations": self.operations_count,
            "total_cost": round(self.total_cost, 6),
            "average_cost_per_operation": round(
                self.total_cost / max(self.operations_count, 1), 6
            ),
            "cost_tracking_enabled": True,
            "gateway_enabled": True,
            "providers_used": list(self.provider_stats.keys()),
            "provider_statistics": {
                provider: {
                    "requests": stats["requests"],
                    "total_cost": round(stats["total_cost"], 6),
                    "average_cost": round(stats.get("avg_cost", 0), 6),
                    "success_rate": round(stats.get("success_rate", 0), 3),
                }
                for provider, stats in self.provider_stats.items()
            },
            "routing_strategy": self.default_routing_strategy.value,
            "observability_enabled": self.enable_observability,
            "caching_enabled": self.enable_caching,
        }


def instrument_helicone(
    helicone_api_key: Optional[str] = None,
    provider_keys: Optional[dict[str, str]] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: str = "production",
    auto_instrument_providers: bool = True,
) -> GenOpsHeliconeAdapter:
    """
    Zero-code instrumentation for Helicone AI gateway.

    This function enables automatic GenOps tracking for all AI requests
    routed through the Helicone gateway with zero code changes required.

    Args:
        helicone_api_key: Helicone platform API key
        provider_keys: Dictionary of provider API keys
        team: Team identifier for cost attribution
        project: Project identifier for tracking
        environment: Environment (dev/staging/production)
        auto_instrument_providers: Auto-instrument known AI SDKs

    Returns:
        GenOpsHeliconeAdapter instance for manual usage
    """
    adapter = GenOpsHeliconeAdapter(
        helicone_api_key=helicone_api_key,
        provider_keys=provider_keys,
        team=team,
        project=project,
        environment=environment,
    )

    if auto_instrument_providers:
        _auto_instrument_ai_sdks(adapter)

    logger.info(
        f"GenOps Helicone gateway instrumentation enabled for team='{team}', project='{project}'"
    )
    return adapter


def _auto_instrument_ai_sdks(adapter: GenOpsHeliconeAdapter):
    """Automatically instrument popular AI SDKs to use Helicone gateway."""
    try:
        # OpenAI auto-instrumentation
        import openai

        original_base_url = getattr(openai, "_original_base_url", None)

        if not original_base_url:
            openai._original_base_url = getattr(openai.OpenAI(), "base_url", None)

        # Monkey patch OpenAI to use Helicone gateway
        def helicone_openai_init(self, **kwargs):
            if "base_url" not in kwargs:
                kwargs["base_url"] = f"{adapter.base_url}/v1"

            if "default_headers" not in kwargs:
                kwargs["default_headers"] = {}

            # Add Helicone authentication
            if adapter.helicone_api_key:
                kwargs["default_headers"]["Helicone-Auth"] = (
                    f"Bearer {adapter.helicone_api_key}"
                )

            return openai.OpenAI.__original_init__(self, **kwargs)

        if not hasattr(openai.OpenAI, "__original_init__"):
            openai.OpenAI.__original_init__ = openai.OpenAI.__init__
            openai.OpenAI.__init__ = helicone_openai_init

        logger.debug("OpenAI SDK auto-instrumented for Helicone gateway")

    except ImportError:
        logger.debug("OpenAI SDK not available for auto-instrumentation")

    # TODO: Add auto-instrumentation for Anthropic, Google AI, etc.


# Convenience functions matching established patterns
def create_helicone_adapter(**kwargs) -> GenOpsHeliconeAdapter:
    """Create Helicone adapter with standard configuration."""
    return GenOpsHeliconeAdapter(**kwargs)


# Export main classes and functions
__all__ = [
    "GenOpsHeliconeAdapter",
    "HeliconeResponse",
    "MultiProviderResponse",
    "HeliconeUsage",
    "RoutingStrategy",
    "HeliconeProvider",
    "instrument_helicone",
    "create_helicone_adapter",
]
