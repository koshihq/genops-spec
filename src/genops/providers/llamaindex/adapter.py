"""LlamaIndex provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from genops.providers.base import BaseFrameworkProvider

from .error_handling import (
    CircuitBreakerOpenError,
    GracefulDegradationError,
    RetryConfig,
    RetryExhaustedError,
    get_health_monitor,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

try:
    import llama_index  # noqa: F401
    from llama_index.core import Settings
    from llama_index.core.agent import BaseAgent
    from llama_index.core.callbacks import BaseCallbackHandler, CallbackManager
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.response import Response
    from llama_index.core.schema import NodeWithScore, QueryBundle

    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    BaseCallbackHandler = object  # Fallback for type hints
    NodeWithScore = None  # type: ignore[misc,assignment]
    QueryBundle = None  # type: ignore[misc,assignment]
    logger.warning("LlamaIndex not installed. Install with: pip install llama-index")


@dataclass
class LlamaIndexOperation:
    """Represents a single LlamaIndex operation for cost tracking."""

    operation_id: str
    operation_type: str  # 'query', 'embed', 'retrieve', 'synthesize', 'agent_step'
    start_time: float
    end_time: float | None = None
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    tokens_consumed: int | None = None
    cost_usd: float | None = None
    provider: str | None = None  # 'openai', 'anthropic', etc.
    model: str | None = None
    governance_attributes: dict[str, Any] | None = None

    def __post_init__(self):
        if self.governance_attributes is None:
            self.governance_attributes = {}

    @property
    def duration_ms(self) -> float:
        """Calculate operation duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


@dataclass
class RAGPipelineMetrics:
    """Metrics for a complete RAG pipeline operation."""

    query_id: str
    total_cost: float
    operations: list[LlamaIndexOperation]
    retrieval_count: int = 0
    embedding_tokens: int = 0
    synthesis_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: str | None = None


@dataclass
class LlamaIndexCostBreakdown:
    """Detailed cost breakdown for LlamaIndex operations."""

    embedding_cost: float = 0.0
    retrieval_cost: float = 0.0
    synthesis_cost: float = 0.0
    embedding_tokens: int = 0
    synthesis_tokens: int = 0
    retrieval_operations: int = 0
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    optimization_suggestions: list[str] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all categories."""
        return self.embedding_cost + self.retrieval_cost + self.synthesis_cost


@dataclass
class LlamaIndexOperationSummary:
    """Summary of LlamaIndex operations and costs."""

    total_cost: float = 0.0
    operation_count: int = 0
    rag_pipelines: int = 0
    avg_cost_per_operation: float = 0.0
    cost_breakdown: LlamaIndexCostBreakdown = field(
        default_factory=LlamaIndexCostBreakdown
    )
    budget_status: dict[str, Any] | None = None


class GenOpsLlamaIndexCallback(BaseCallbackHandler):
    """Custom callback handler for LlamaIndex to capture comprehensive telemetry."""

    def __init__(self, adapter: "GenOpsLlamaIndexAdapter"):
        super().__init__()
        self.adapter = adapter
        self.operations: dict[str, LlamaIndexOperation] = {}
        self.current_query_id: str | None = None
        self.pipeline_metrics: dict[str, RAGPipelineMetrics] = {}

    def on_event_start(
        self, event_type: str, payload: dict[str, Any] | None = None, **kwargs
    ) -> str:
        """Called when any LlamaIndex event starts."""
        event_id = str(uuid.uuid4())

        operation = LlamaIndexOperation(
            operation_id=event_id,
            operation_type=event_type,
            start_time=time.time(),
            input_data=payload or {},
            governance_attributes=self.adapter.get_current_governance_context(),
        )

        self.operations[event_id] = operation

        # Start telemetry span
        with tracer.start_as_current_span(f"llamaindex.{event_type}") as span:
            span.set_attributes(
                {
                    "genops.operation_id": event_id,
                    "genops.operation_type": event_type,
                    "genops.framework": "llamaindex",
                    **operation.governance_attributes,
                }
            )

        return event_id

    def on_event_end(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        event_id: str | None = None,
        **kwargs,
    ) -> None:
        """Called when any LlamaIndex event ends."""
        if event_id and event_id in self.operations:
            operation = self.operations[event_id]
            operation.end_time = time.time()
            operation.output_data = payload or {}

            # Extract cost information if available
            if payload:
                self._extract_cost_info(operation, payload)

            # Record in cost aggregator if available
            if (
                hasattr(self.adapter, "cost_aggregator")
                and self.adapter.cost_aggregator
            ):
                self.adapter.cost_aggregator.add_llamaindex_operation(operation)

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs
    ) -> None:
        """Called when LLM processing starts."""
        event_id = self.on_event_start(
            "llm_call", {"prompts": prompts, "model_info": serialized}
        )

        # Extract provider and model information
        if event_id in self.operations:
            operation = self.operations[event_id]
            self._extract_provider_info(operation, serialized)

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM processing ends."""
        # Find the most recent LLM operation
        llm_ops = [
            op
            for op in self.operations.values()
            if op.operation_type == "llm_call" and op.end_time is None
        ]

        if llm_ops:
            operation = llm_ops[-1]  # Most recent
            self.on_event_end(
                "llm_call", {"response": response}, operation.operation_id
            )

    def on_retrieve_start(self, query: QueryBundle, **kwargs) -> str:
        """Called when retrieval starts."""
        return self.on_event_start(
            "retrieve",
            {
                "query": query.query_str if query else None,
                "similarity_top_k": kwargs.get("similarity_top_k"),
            },
        )

    def on_retrieve_end(self, nodes: list[NodeWithScore], **kwargs) -> None:
        """Called when retrieval ends."""
        retrieve_ops = [
            op
            for op in self.operations.values()
            if op.operation_type == "retrieve" and op.end_time is None
        ]

        if retrieve_ops:
            operation = retrieve_ops[-1]
            self.on_event_end(
                "retrieve",
                {
                    "retrieved_nodes": len(nodes),
                    "scores": [node.score for node in nodes if node.score is not None],
                },
                operation.operation_id,
            )

    def _extract_cost_info(
        self, operation: LlamaIndexOperation, payload: dict[str, Any]
    ) -> None:
        """Extract cost information from operation payload."""
        # Try to extract token usage and cost information
        if "usage" in payload:
            usage = payload["usage"]
            operation.tokens_consumed = usage.get("total_tokens", 0)

        if "cost" in payload:
            operation.cost_usd = payload["cost"]

    def _extract_provider_info(
        self, operation: LlamaIndexOperation, model_info: dict[str, Any]
    ) -> None:
        """Extract provider and model information from LLM serialized data."""
        # Common provider detection patterns
        model_name = model_info.get("model_name", "").lower()
        class_name = model_info.get("class_name", "").lower()

        if "openai" in model_name or "openai" in class_name:
            operation.provider = "openai"
            operation.model = model_info.get("model_name", "gpt-3.5-turbo")
        elif "anthropic" in model_name or "anthropic" in class_name:
            operation.provider = "anthropic"
            operation.model = model_info.get("model_name", "claude-3-haiku")
        elif "gemini" in model_name or "google" in class_name:
            operation.provider = "google"
            operation.model = model_info.get("model_name", "gemini-pro")
        elif "llama" in model_name or "meta" in model_name:
            operation.provider = "meta"
            operation.model = model_info.get("model_name", "llama-2")
        else:
            operation.provider = "unknown"
            operation.model = model_info.get("model_name", "unknown")


class GenOpsLlamaIndexAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for LlamaIndex with comprehensive RAG pipeline governance.

    Provides cost tracking, team attribution, and observability for:
    - Query engines and RAG pipelines
    - Embedding operations and vector stores
    - Agent workflows and tool usage
    - Multi-provider LLM operations
    """

    def __init__(
        self,
        telemetry_enabled: bool = True,
        cost_tracking_enabled: bool = True,
        debug: bool = False,
        **governance_defaults,
    ):
        """
        Initialize GenOps LlamaIndex adapter.

        Args:
            telemetry_enabled: Enable OpenTelemetry export
            cost_tracking_enabled: Enable cost calculation and tracking
            debug: Enable debug logging
            **governance_defaults: Default governance attributes (team, project, etc.)
        """
        super().__init__()

        if not HAS_LLAMAINDEX:
            raise ImportError(
                "LlamaIndex not found. Install with: pip install llama-index>=0.10.0"
            )

        self.telemetry_enabled = telemetry_enabled
        self.cost_tracking_enabled = cost_tracking_enabled
        self.debug = debug
        self.governance_defaults = governance_defaults

        # Initialize callback handler
        self.callback_handler = GenOpsLlamaIndexCallback(self)

        # Cost aggregator (will be injected)
        self.cost_aggregator = None

        # Error handling and health monitoring
        self.health_monitor = get_health_monitor()
        self.retry_config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0)
        self.enable_graceful_degradation = governance_defaults.get(
            "enable_graceful_degradation", True
        )

        # Current operation context
        self._governance_context: dict[str, Any] = {}

    def get_current_governance_context(self) -> dict[str, Any]:
        """Get current governance context for operations."""
        return {**self.governance_defaults, **self._governance_context}

    @contextmanager
    def governance_context(self, **attributes):
        """Context manager to set governance attributes for operations."""
        old_context = self._governance_context.copy()
        self._governance_context.update(attributes)
        try:
            yield
        finally:
            self._governance_context = old_context

    def instrument_query_engine(
        self, query_engine: BaseQueryEngine, **governance_attrs
    ) -> BaseQueryEngine:
        """
        Instrument a LlamaIndex query engine with GenOps governance.

        Args:
            query_engine: LlamaIndex query engine to instrument
            **governance_attrs: Governance attributes (team, project, customer_id)

        Returns:
            Instrumented query engine with cost tracking
        """
        if not HAS_LLAMAINDEX:
            logger.warning("LlamaIndex not available, returning original query engine")
            return query_engine

        # Add our callback to the query engine's callback manager
        if hasattr(query_engine, "callback_manager"):
            if query_engine.callback_manager is None:
                query_engine.callback_manager = CallbackManager([self.callback_handler])
            else:
                query_engine.callback_manager.handlers.append(self.callback_handler)

        # Set governance context for this query engine
        with self.governance_context(**governance_attrs):
            return query_engine

    def instrument_agent(self, agent: BaseAgent, **governance_attrs) -> BaseAgent:
        """
        Instrument a LlamaIndex agent with GenOps governance.

        Args:
            agent: LlamaIndex agent to instrument
            **governance_attrs: Governance attributes (team, project, customer_id)

        Returns:
            Instrumented agent with cost tracking
        """
        if not HAS_LLAMAINDEX:
            logger.warning("LlamaIndex not available, returning original agent")
            return agent

        # Add our callback to the agent's callback manager
        if hasattr(agent, "callback_manager"):
            if agent.callback_manager is None:
                agent.callback_manager = CallbackManager([self.callback_handler])
            else:
                agent.callback_manager.handlers.append(self.callback_handler)

        return agent

    def track_query(
        self,
        query_engine: BaseQueryEngine,
        query: str,
        fallback_providers: list[str] | None = None,
        **governance_attrs,
    ) -> Response:
        """
        Execute and track a query with comprehensive governance and error handling.

        Args:
            query_engine: LlamaIndex query engine
            query: Query string
            fallback_providers: Optional list of fallback providers for graceful degradation
            **governance_attrs: Governance attributes for cost attribution

        Returns:
            Query response with cost tracking
        """
        with self.governance_context(**governance_attrs):
            # Instrument the query engine
            instrumented_engine = self.instrument_query_engine(
                query_engine, **governance_attrs
            )

            # Get primary provider for error handling
            primary_provider = governance_attrs.get("provider", "primary")

            # Execute query with telemetry and error handling
            with tracer.start_as_current_span("llamaindex.query") as span:
                span.set_attributes(
                    {
                        "genops.query": query[:100],  # Truncate long queries
                        "genops.framework": "llamaindex",
                        "genops.primary_provider": primary_provider,
                        **self.get_current_governance_context(),
                    }
                )

                def _execute_query():
                    """Internal query execution with error handling."""
                    return instrumented_engine.query(query)

                try:
                    # Use health monitor for error protection
                    response = self.health_monitor.call_with_protection(
                        primary_provider, _execute_query
                    )

                    span.set_attribute("genops.success", True)
                    span.set_attribute("genops.provider_used", primary_provider)

                    if hasattr(response, "response") and response.response:
                        span.set_attribute(
                            "genops.response_length", len(str(response.response))
                        )

                    return response

                except (CircuitBreakerOpenError, RetryExhaustedError) as e:
                    # Handle provider failures with graceful degradation
                    if self.enable_graceful_degradation and fallback_providers:
                        logger.warning(
                            f"Primary provider {primary_provider} failed: {e}. Attempting graceful degradation."
                        )

                        try:
                            fallback_response = self._handle_graceful_degradation(
                                query_engine,
                                query,
                                primary_provider,
                                fallback_providers,
                                span,
                            )
                            return fallback_response

                        except GracefulDegradationError as degradation_error:
                            span.record_exception(degradation_error)
                            span.set_status(
                                Status(
                                    StatusCode.ERROR,
                                    f"All providers failed: {degradation_error}",
                                )
                            )
                            logger.error(
                                f"Graceful degradation failed: {degradation_error}"
                            )
                            raise

                    # No fallback available or disabled
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("genops.error_type", "provider_failure")
                    logger.error(f"Provider failure in LlamaIndex query: {e}")
                    raise

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("genops.error_type", "unknown")
                    logger.error(f"Unexpected error in LlamaIndex query: {e}")
                    raise

    def _handle_graceful_degradation(
        self,
        query_engine: BaseQueryEngine,
        query: str,
        failed_provider: str,
        fallback_providers: list[str],
        span: trace.Span,
    ) -> Response:
        """Handle graceful degradation to fallback providers."""
        healthy_fallbacks = []

        for provider in fallback_providers:
            if (
                provider != failed_provider
                and provider in self.health_monitor.get_healthy_providers()
            ):
                healthy_fallbacks.append(provider)

        if not healthy_fallbacks:
            raise GracefulDegradationError("No healthy fallback providers available")

        # Try fallback providers in order
        last_error = None

        for fallback_provider in healthy_fallbacks:
            try:
                logger.info(f"Attempting fallback to provider: {fallback_provider}")
                span.add_event("fallback_attempt", {"provider": fallback_provider})

                def _fallback_query():
                    # In a full implementation, this might involve switching models/providers
                    # For now, we retry with the same engine but track the fallback
                    return query_engine.query(query)

                response = self.health_monitor.call_with_protection(
                    fallback_provider, _fallback_query
                )

                # Success with fallback provider
                span.set_attribute("genops.success", True)
                span.set_attribute("genops.provider_used", fallback_provider)
                span.set_attribute("genops.fallback_used", True)
                span.add_event("fallback_success", {"provider": fallback_provider})

                logger.info(
                    f"Successfully failed over to provider: {fallback_provider}"
                )
                return response

            except Exception as e:
                last_error = e
                span.add_event(
                    "fallback_failed", {"provider": fallback_provider, "error": str(e)}
                )
                logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                continue

        # All fallbacks failed
        raise GracefulDegradationError(
            f"All fallback providers failed. Last error: {last_error}"
        )

    def _handle_chat_graceful_degradation(
        self,
        agent: BaseAgent,
        message: str,
        failed_provider: str,
        fallback_providers: list[str],
        span: trace.Span,
    ) -> str:
        """Handle graceful degradation for agent chat interactions."""
        healthy_fallbacks = []

        for provider in fallback_providers:
            if (
                provider != failed_provider
                and provider in self.health_monitor.get_healthy_providers()
            ):
                healthy_fallbacks.append(provider)

        if not healthy_fallbacks:
            raise GracefulDegradationError("No healthy fallback providers available")

        # Try fallback providers in order
        last_error = None

        for fallback_provider in healthy_fallbacks:
            try:
                logger.info(
                    f"Attempting chat fallback to provider: {fallback_provider}"
                )
                span.add_event("fallback_attempt", {"provider": fallback_provider})

                def _fallback_chat():
                    # In a full implementation, this might involve switching models/providers
                    # For now, we retry with the same agent but track the fallback
                    return agent.chat(message)

                response = self.health_monitor.call_with_protection(
                    fallback_provider, _fallback_chat
                )

                # Success with fallback provider
                span.set_attribute("genops.success", True)
                span.set_attribute("genops.provider_used", fallback_provider)
                span.set_attribute("genops.fallback_used", True)
                span.add_event("fallback_success", {"provider": fallback_provider})

                logger.info(
                    f"Successfully failed over to provider: {fallback_provider}"
                )

                if hasattr(response, "response"):
                    return str(response.response)
                return str(response)

            except Exception as e:
                last_error = e
                span.add_event(
                    "fallback_failed", {"provider": fallback_provider, "error": str(e)}
                )
                logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                continue

        # All fallbacks failed
        raise GracefulDegradationError(
            f"All chat fallback providers failed. Last error: {last_error}"
        )

    def get_system_health(self) -> dict[str, Any]:
        """Get system health status for monitoring."""
        return self.health_monitor.get_system_health()

    def track_chat(
        self,
        agent: BaseAgent,
        message: str,
        fallback_providers: list[str] | None = None,
        **governance_attrs,
    ) -> str:
        """
        Execute and track an agent chat interaction with comprehensive error handling.

        Args:
            agent: LlamaIndex agent
            message: User message
            fallback_providers: Optional list of fallback providers for graceful degradation
            **governance_attrs: Governance attributes for cost attribution

        Returns:
            Agent response with cost tracking and error handling
        """
        with self.governance_context(**governance_attrs):
            # Instrument the agent
            instrumented_agent = self.instrument_agent(agent, **governance_attrs)

            # Get primary provider for error handling
            primary_provider = governance_attrs.get("provider", "primary")

            # Execute chat with telemetry and error handling
            with tracer.start_as_current_span("llamaindex.chat") as span:
                span.set_attributes(
                    {
                        "genops.message": message[:100],  # Truncate long messages
                        "genops.framework": "llamaindex",
                        "genops.primary_provider": primary_provider,
                        **self.get_current_governance_context(),
                    }
                )

                def _execute_chat():
                    """Internal chat execution with error handling."""
                    return instrumented_agent.chat(message)

                try:
                    # Use health monitor for error protection
                    response = self.health_monitor.call_with_protection(
                        primary_provider, _execute_chat
                    )

                    span.set_attribute("genops.success", True)
                    span.set_attribute("genops.provider_used", primary_provider)

                    if hasattr(response, "response"):
                        response_str = str(response.response)
                        span.set_attribute("genops.response_length", len(response_str))
                        return response_str

                    response_str = str(response)
                    span.set_attribute("genops.response_length", len(response_str))
                    return response_str

                except (CircuitBreakerOpenError, RetryExhaustedError) as e:
                    # Handle provider failures with graceful degradation
                    if self.enable_graceful_degradation and fallback_providers:
                        logger.warning(
                            f"Primary provider {primary_provider} failed: {e}. Attempting graceful degradation."
                        )

                        try:
                            fallback_response = self._handle_chat_graceful_degradation(
                                agent,
                                message,
                                primary_provider,
                                fallback_providers,
                                span,
                            )
                            return fallback_response

                        except GracefulDegradationError as degradation_error:
                            span.record_exception(degradation_error)
                            span.set_status(
                                Status(
                                    StatusCode.ERROR,
                                    f"All providers failed: {degradation_error}",
                                )
                            )
                            logger.error(
                                f"Graceful degradation failed: {degradation_error}"
                            )
                            raise

                    # No fallback available or disabled
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("genops.error_type", "provider_failure")
                    logger.error(f"Provider failure in LlamaIndex chat: {e}")
                    raise

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("genops.error_type", "unknown")
                    logger.error(f"Unexpected error in LlamaIndex chat: {e}")
                    raise

    def get_operation_summary(self) -> dict[str, Any]:
        """Get summary of all tracked operations."""
        operations = self.callback_handler.operations

        total_cost = sum(op.cost_usd for op in operations.values() if op.cost_usd)
        total_tokens = sum(
            op.tokens_consumed for op in operations.values() if op.tokens_consumed
        )

        providers = {op.provider for op in operations.values() if op.provider}
        operation_types = {op.operation_type for op in operations.values()}

        return {
            "total_operations": len(operations),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "unique_providers": list(providers),
            "operation_types": list(operation_types),
            "operations": [asdict(op) for op in operations.values()],
        }


def instrument_llamaindex(
    telemetry_enabled: bool = True,
    cost_tracking_enabled: bool = True,
    **governance_defaults,
) -> GenOpsLlamaIndexAdapter:
    """
    Create and configure a GenOps LlamaIndex adapter.

    Args:
        telemetry_enabled: Enable OpenTelemetry export
        cost_tracking_enabled: Enable cost tracking
        **governance_defaults: Default governance attributes

    Returns:
        Configured GenOpsLlamaIndexAdapter instance

    Example:
        adapter = instrument_llamaindex(team="ai-research", project="rag-system")
        response = adapter.track_query(query_engine, "What is RAG?")
    """
    return GenOpsLlamaIndexAdapter(
        telemetry_enabled=telemetry_enabled,
        cost_tracking_enabled=cost_tracking_enabled,
        **governance_defaults,
    )


# Auto-instrumentation function
def auto_instrument():
    """
    Enable automatic instrumentation of LlamaIndex operations.

    This patches LlamaIndex components to automatically add GenOps tracking.

    Usage:
        from genops.providers.llamaindex import auto_instrument
        auto_instrument()

        # Your existing LlamaIndex code now has automatic tracking
    """
    if not HAS_LLAMAINDEX:
        logger.warning("LlamaIndex not available for auto-instrumentation")
        return

    # Create global adapter instance
    global_adapter = GenOpsLlamaIndexAdapter()

    # Add global callback handler to Settings
    if not hasattr(Settings, "callback_manager") or Settings.callback_manager is None:
        Settings.callback_manager = CallbackManager([global_adapter.callback_handler])
    else:
        # Add to existing callback manager
        if global_adapter.callback_handler not in Settings.callback_manager.handlers:
            Settings.callback_manager.handlers.append(global_adapter.callback_handler)

    logger.info("GenOps auto-instrumentation enabled for LlamaIndex")


# Export main classes and functions
__all__ = [
    "GenOpsLlamaIndexAdapter",
    "GenOpsLlamaIndexCallback",
    "LlamaIndexOperation",
    "RAGPipelineMetrics",
    "instrument_llamaindex",
    "auto_instrument",
]
