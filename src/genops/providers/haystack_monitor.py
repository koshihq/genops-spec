#!/usr/bin/env python3
"""
Haystack Pipeline and Component Monitor

Advanced monitoring system for Haystack AI pipelines with component-level instrumentation,
performance tracking, and governance telemetry. Provides deep insights into pipeline
execution, component interactions, and resource utilization.

Usage:
    from genops.providers.haystack_monitor import HaystackMonitor

    monitor = HaystackMonitor(team="ai-team", project="rag-system")

    # Monitor entire pipeline execution
    with monitor.monitor_pipeline(pipeline, "document-qa") as execution:
        result = pipeline.run({"query": "What is RAG?"})

    # Get detailed execution metrics
    metrics = execution.get_metrics()
    print(f"Components executed: {metrics.total_components}")
    print(f"Total cost: ${metrics.total_cost:.6f}")

Features:
    - Real-time pipeline execution monitoring
    - Component-level performance and cost tracking
    - RAG workflow specialization (retrieval + generation tracking)
    - Agent workflow monitoring (tool usage and decision tracking)
    - Resource utilization and bottleneck detection
    - Error handling and failure analysis
    - Performance optimization recommendations
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from genops.providers.haystack_cost_aggregator import (
        HaystackCostAggregator,
    )
import statistics
import threading
from collections import defaultdict, deque
from datetime import datetime
from functools import wraps

# OpenTelemetry imports
from opentelemetry.trace import Status, StatusCode

# GenOps imports
from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

# Check for Haystack availability
try:
    import haystack  # noqa: F401
    from haystack import Document, Pipeline
    from haystack.components.embedders import (  # noqa: F401
        OpenAIDocumentEmbedder,
        OpenAITextEmbedder,
    )
    from haystack.components.generators import OpenAIGenerator  # noqa: F401
    from haystack.components.retrievers import InMemoryEmbeddingRetriever  # noqa: F401
    from haystack.core.component import Component

    HAS_HAYSTACK = True
except ImportError:
    HAS_HAYSTACK = False
    Pipeline = None
    Document = None
    Component = None
    logger.warning("Haystack not installed. Install with: pip install haystack-ai")


@dataclass
class ComponentExecutionMetrics:
    """Detailed metrics for a single component execution."""

    component_name: str
    component_type: str
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cost: Decimal = Decimal("0")
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    operations: int = 1
    status: str = "success"  # "success", "error", "timeout"
    error_message: Optional[str] = None
    input_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecutionMetrics:
    """Comprehensive metrics for pipeline execution."""

    pipeline_name: str
    pipeline_id: str
    start_time: datetime
    end_time: datetime
    total_execution_time_seconds: float
    total_cost: Decimal
    component_metrics: list[ComponentExecutionMetrics]

    # Aggregated metrics
    total_components: int = 0
    successful_components: int = 0
    failed_components: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0

    # Performance metrics
    slowest_component: Optional[str] = None
    slowest_component_time: float = 0.0
    most_expensive_component: Optional[str] = None
    highest_component_cost: Decimal = Decimal("0")

    # Resource metrics
    peak_memory_usage_mb: Optional[float] = None
    avg_cpu_usage_percent: Optional[float] = None

    # Cost breakdown
    cost_by_provider: dict[str, Decimal] = field(default_factory=dict)
    cost_by_component_type: dict[str, Decimal] = field(default_factory=dict)

    # Governance attributes
    governance_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGWorkflowMetrics:
    """Specialized metrics for RAG (Retrieval-Augmented Generation) workflows."""

    retrieval_metrics: Optional[ComponentExecutionMetrics] = None
    generation_metrics: Optional[ComponentExecutionMetrics] = None
    embedding_metrics: list[ComponentExecutionMetrics] = field(default_factory=list)

    # RAG-specific metrics
    documents_retrieved: int = 0
    avg_document_relevance: Optional[float] = None
    retrieval_latency_seconds: float = 0.0
    generation_latency_seconds: float = 0.0

    # Quality metrics
    retrieval_success_rate: float = 1.0
    generation_success_rate: float = 1.0
    end_to_end_latency_seconds: float = 0.0


@dataclass
class AgentWorkflowMetrics:
    """Specialized metrics for agent workflow monitoring."""

    decisions_made: int = 0
    tools_used: list[str] = field(default_factory=list)
    tool_usage_count: dict[str, int] = field(default_factory=dict)
    tool_success_rate: dict[str, float] = field(default_factory=dict)

    # Agent decision tracking
    decision_latency_seconds: float = 0.0
    avg_decision_confidence: Optional[float] = None

    # Loop and iteration tracking
    total_iterations: int = 0
    max_iterations_reached: bool = False
    early_termination: bool = False

    # Cost breakdown by tool/action
    cost_by_tool: dict[str, Decimal] = field(default_factory=dict)


class ComponentMonitor:
    """Monitor for individual Haystack components."""

    def __init__(
        self,
        telemetry: GenOpsTelemetry,
        cost_aggregator: Optional["HaystackCostAggregator"] = None,
    ):
        self.telemetry = telemetry
        self.cost_aggregator = cost_aggregator

        # Performance tracking
        self._start_times: dict[str, float] = {}
        self._memory_tracker = {}

        # Enhanced error handling and retry configuration
        self.error_tracker = {
            "component_failures": defaultdict(int),
            "error_types": defaultdict(int),
            "retry_attempts": defaultdict(int),
            "circuit_breaker_states": {},
            "error_patterns": deque(maxlen=100),  # Track recent error patterns
        }

        # Circuit breaker configuration
        self.circuit_breaker_config = {
            "failure_threshold": 5,  # Number of failures before opening circuit
            "recovery_timeout": 60,  # Seconds before trying to close circuit
            "half_open_max_calls": 3,  # Max calls in half-open state
        }

        # Retry configuration with backoff
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "backoff_factor": 2.0,
            "jitter": True,
        }

        # Component type registry for monitoring
        self.component_monitors = {
            "OpenAIGenerator": self._monitor_openai_generator,
            "AnthropicGenerator": self._monitor_anthropic_generator,
            "InMemoryEmbeddingRetriever": self._monitor_retriever,
            "OpenAIDocumentEmbedder": self._monitor_embedder,
            "OpenAITextEmbedder": self._monitor_embedder,
        }

        # Performance baseline tracking
        self.performance_baselines = {
            "response_times": deque(maxlen=50),
            "error_rates": deque(maxlen=20),
            "cost_patterns": deque(maxlen=30),
        }

    def monitor_component(
        self, component: Any, component_name: str, inputs: dict[str, Any]
    ) -> ComponentExecutionMetrics:
        """
        Monitor execution of a single Haystack component with enhanced error handling.

        Args:
            component: Haystack component instance
            component_name: Name of the component
            inputs: Input data for the component

        Returns:
            ComponentExecutionMetrics: Detailed execution metrics
        """
        component_type = component.__class__.__name__

        # Check circuit breaker status
        if self._is_circuit_open(component_name):
            logger.warning(f"Circuit breaker open for component {component_name}")
            return ComponentExecutionMetrics(
                component_name=component_name,
                component_type=component_type,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                execution_time_seconds=0.0,
                status="error",
                error_message="Circuit breaker open - too many recent failures",
            )

        return self._execute_component_with_retry(
            component, component_name, component_type, inputs
        )

    def _execute_component_with_retry(
        self,
        component: Any,
        component_name: str,
        component_type: str,
        inputs: dict[str, Any],
    ) -> ComponentExecutionMetrics:
        """
        Execute component with intelligent retry logic and circuit breaker.
        """
        last_exception = None
        retry_count = 0
        start_time = datetime.utcnow()

        for attempt in range(self.retry_config["max_retries"] + 1):  # type: ignore[call-overload]
            try:
                # Add delay for retries
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(
                        f"Retrying {component_name} attempt {attempt + 1} after {delay:.2f}s"
                    )
                    time.sleep(delay)
                    retry_count += 1
                    self.error_tracker["retry_attempts"][component_name] += 1

                # Attempt component execution
                start_perf = time.perf_counter()
                input_size_bytes = self._estimate_data_size(inputs)

                # Use specialized monitor if available
                if component_type in self.component_monitors:
                    result = self.component_monitors[component_type](
                        component, component_name, inputs
                    )
                else:
                    result = self._monitor_generic_component(
                        component, component_name, inputs
                    )

                end_time = datetime.utcnow()
                execution_time = time.perf_counter() - start_perf
                output_size_bytes = self._estimate_data_size(result)

                # Success - record metrics and close circuit breaker
                self._record_success(component_name)

                # Update performance baselines
                self.performance_baselines["response_times"].append(execution_time)

                if retry_count > 0:
                    logger.info(
                        f"Component {component_name} succeeded after {retry_count} retries"
                    )

                metrics = self._extract_component_metrics(
                    component_name,
                    component_type,
                    result,
                    start_time,
                    end_time,
                    execution_time,
                    input_size_bytes,
                    output_size_bytes,
                )

                # Add retry information to metrics
                metrics.custom_metrics["retry_count"] = retry_count
                metrics.custom_metrics["attempt_number"] = attempt + 1

                return metrics

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                # Track error patterns
                self._track_error(component_name, error_type, str(e))

                # Check if error is retryable
                if (
                    not self._is_retryable_error(e)
                    or attempt >= self.retry_config["max_retries"]
                ):
                    break

                logger.warning(
                    f"Component {component_name} failed with {error_type}: {str(e)}. "
                    f"Retrying... (attempt {attempt + 1}/{self.retry_config['max_retries']})"
                )

        # All retries exhausted - record failure and potentially open circuit breaker
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()

        self._record_failure(
            component_name, str(last_exception) if last_exception else "Unknown error"
        )

        logger.error(
            f"Component {component_name} failed permanently after {retry_count} retries. "
            f"Error: {str(last_exception) if last_exception else 'Unknown'}"
        )

        return ComponentExecutionMetrics(
            component_name=component_name,
            component_type=component_type,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            status="error",
            error_message=str(last_exception) if last_exception else "Unknown error",
            custom_metrics={
                "retry_count": retry_count,
                "final_attempt": self.retry_config["max_retries"] + 1,
                "error_type": type(last_exception).__name__
                if last_exception
                else "Unknown",
                "retryable": self._is_retryable_error(last_exception)
                if last_exception
                else False,
            },
        )

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.
        """
        import random

        delay = min(
            self.retry_config["base_delay"]
            * (self.retry_config["backoff_factor"] ** (attempt - 1)),
            self.retry_config["max_delay"],
        )

        # Add jitter to avoid thundering herd
        if self.retry_config["jitter"]:
            delay *= 0.5 + random.random() * 0.5

        return delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        """
        retryable_error_types = {
            "ConnectionError",
            "TimeoutError",
            "HTTPError",
            "ServiceUnavailableError",
            "RateLimitError",
            "APIError",
            "NetworkError",
            "TemporaryFailure",
        }

        error_type = type(error).__name__
        error_message = str(error).lower()

        # Check error type
        if error_type in retryable_error_types:
            return True

        # Check error message patterns
        retryable_patterns = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "service unavailable",
            "temporary",
            "busy",
            "overload",
        ]

        return any(pattern in error_message for pattern in retryable_patterns)

    def _is_circuit_open(self, component_name: str) -> bool:
        """
        Check if circuit breaker is open for a component.
        """
        if component_name not in self.error_tracker["circuit_breaker_states"]:
            return False

        cb_state = self.error_tracker["circuit_breaker_states"][component_name]

        if cb_state["state"] == "closed":
            return False
        elif cb_state["state"] == "open":
            # Check if recovery timeout has passed
            if (
                time.time() - cb_state["opened_at"]
                > self.circuit_breaker_config["recovery_timeout"]
            ):
                # Move to half-open state
                cb_state["state"] = "half-open"
                cb_state["half_open_calls"] = 0
                logger.info(f"Circuit breaker for {component_name} moved to half-open")
                return False
            return True
        elif cb_state["state"] == "half-open":
            # Allow limited calls in half-open state
            if (
                cb_state["half_open_calls"]
                < self.circuit_breaker_config["half_open_max_calls"]
            ):
                cb_state["half_open_calls"] += 1
                return False
            return True

        return False

    def _record_success(self, component_name: str):
        """
        Record successful component execution and potentially close circuit breaker.
        """
        if component_name in self.error_tracker["circuit_breaker_states"]:
            cb_state = self.error_tracker["circuit_breaker_states"][component_name]

            if cb_state["state"] == "half-open":
                # Successful call in half-open state - close the circuit
                cb_state["state"] = "closed"
                cb_state["failure_count"] = 0
                logger.info(
                    f"Circuit breaker for {component_name} closed after successful recovery"
                )

    def _record_failure(self, component_name: str, error_message: str):
        """
        Record component failure and potentially open circuit breaker.
        """
        # Initialize circuit breaker state if not exists
        if component_name not in self.error_tracker["circuit_breaker_states"]:
            self.error_tracker["circuit_breaker_states"][component_name] = {
                "state": "closed",
                "failure_count": 0,
                "opened_at": None,
                "half_open_calls": 0,
            }

        cb_state = self.error_tracker["circuit_breaker_states"][component_name]
        cb_state["failure_count"] += 1

        # Check if we should open the circuit breaker
        if (
            cb_state["state"] == "closed"
            and cb_state["failure_count"]
            >= self.circuit_breaker_config["failure_threshold"]
        ):
            cb_state["state"] = "open"
            cb_state["opened_at"] = time.time()
            logger.warning(
                f"Circuit breaker opened for {component_name} after {cb_state['failure_count']} failures"
            )

        elif cb_state["state"] == "half-open":
            # Failure in half-open state - go back to open
            cb_state["state"] = "open"
            cb_state["opened_at"] = time.time()
            cb_state["half_open_calls"] = 0
            logger.warning(
                f"Circuit breaker for {component_name} reopened after failure in half-open state"
            )

    def _track_error(self, component_name: str, error_type: str, error_message: str):
        """
        Track error for pattern analysis and diagnostics.
        """
        # Track component failures
        self.error_tracker["component_failures"][component_name] += 1

        # Track error types
        self.error_tracker["error_types"][error_type] += 1

        # Track error patterns for analysis
        error_pattern = {
            "timestamp": datetime.utcnow(),
            "component": component_name,
            "error_type": error_type,
            "error_message": error_message[:200],  # Truncate long messages
            "thread_id": threading.current_thread().ident,
        }

        self.error_tracker["error_patterns"].append(error_pattern)

    def get_error_diagnostics(self) -> dict[str, Any]:
        """
        Get comprehensive error diagnostics and insights.
        """
        recent_errors = list(self.error_tracker["error_patterns"])

        # Calculate error rates over time windows
        now = datetime.utcnow()
        last_hour_errors = [
            e for e in recent_errors if (now - e["timestamp"]).total_seconds() < 3600
        ]
        last_day_errors = [
            e for e in recent_errors if (now - e["timestamp"]).total_seconds() < 86400
        ]

        # Find error trends
        error_trends = self._analyze_error_trends(recent_errors)

        # Generate recommendations
        recommendations = self._generate_error_recommendations()

        return {
            "total_errors": len(recent_errors),
            "errors_last_hour": len(last_hour_errors),
            "errors_last_day": len(last_day_errors),
            "component_failures": dict(self.error_tracker["component_failures"]),
            "error_types": dict(self.error_tracker["error_types"]),
            "retry_attempts": dict(self.error_tracker["retry_attempts"]),
            "circuit_breaker_states": self._get_circuit_breaker_summary(),
            "error_trends": error_trends,
            "recommendations": recommendations,
            "performance_impact": self._assess_performance_impact(),
        }

    def _analyze_error_trends(self, errors: list[dict]) -> dict[str, Any]:
        """
        Analyze error trends and patterns.
        """
        if not errors:
            return {"trend": "stable", "pattern": "none"}

        # Group errors by hour
        hourly_counts = defaultdict(int)
        for error in errors:
            hour_key = error["timestamp"].replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        if len(hourly_counts) < 2:
            return {"trend": "insufficient_data", "pattern": "none"}

        # Calculate trend
        counts = list(hourly_counts.values())
        if len(counts) >= 3:
            recent_avg = statistics.mean(counts[-3:])
            earlier_avg = statistics.mean(counts[:-3]) if len(counts) > 3 else counts[0]

            if recent_avg > earlier_avg * 1.5:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.5:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_error_rate": statistics.mean(counts[-3:])
            if len(counts) >= 3
            else counts[-1],
            "peak_errors_hour": max(counts),
            "error_frequency_pattern": "high"
            if max(counts) > 10
            else "moderate"
            if max(counts) > 3
            else "low",
        }

    def _generate_error_recommendations(self) -> list[str]:
        """
        Generate actionable recommendations based on error patterns.
        """
        recommendations = []

        # Check for high error components
        max_component_errors = max(
            self.error_tracker["component_failures"].values(), default=0
        )
        if max_component_errors > 5:
            worst_component = max(
                self.error_tracker["component_failures"].items(), key=lambda x: x[1]
            )
            recommendations.append(
                f"Component '{worst_component[0]}' has {worst_component[1]} failures. "
                "Review configuration and increase timeout values."
            )

        # Check for circuit breakers
        open_circuits = [
            name
            for name, state in self.error_tracker["circuit_breaker_states"].items()
            if state["state"] == "open"
        ]

        if open_circuits:
            recommendations.append(
                f"Circuit breakers are open for: {', '.join(open_circuits)}. "
                "Investigate underlying issues before system recovery."
            )

        # Check retry patterns
        high_retry_components = [
            comp
            for comp, retries in self.error_tracker["retry_attempts"].items()
            if retries > 10
        ]

        if high_retry_components:
            recommendations.append(
                f"High retry counts for: {', '.join(high_retry_components)}. "
                "Consider increasing timeouts or checking network stability."
            )

        # Check error types
        common_errors = sorted(
            self.error_tracker["error_types"].items(), key=lambda x: x[1], reverse=True
        )[:3]

        for error_type, count in common_errors:
            if count > 3:
                if "timeout" in error_type.lower():
                    recommendations.append(
                        f"Frequent timeout errors ({count} occurrences). "
                        "Consider increasing timeout configurations."
                    )
                elif "connection" in error_type.lower():
                    recommendations.append(
                        f"Connection errors detected ({count} occurrences). "
                        "Check network connectivity and firewall settings."
                    )

        if not recommendations:
            recommendations.append(
                "System operating within normal error parameters. Continue monitoring."
            )

        return recommendations

    def _get_circuit_breaker_summary(self) -> dict[str, Any]:
        """
        Get summary of circuit breaker states.
        """
        summary = {"open": [], "half_open": [], "closed": []}

        for component, state in self.error_tracker["circuit_breaker_states"].items():
            summary[state["state"]].append(
                {
                    "component": component,
                    "failure_count": state["failure_count"],
                    "opened_at": state.get("opened_at"),
                    "half_open_calls": state.get("half_open_calls", 0),
                }
            )

        return summary

    def _assess_performance_impact(self) -> dict[str, Any]:
        """
        Assess the performance impact of errors and retries.
        """
        if not self.performance_baselines["response_times"]:
            return {"impact": "unknown", "reason": "insufficient_data"}

        recent_times = list(self.performance_baselines["response_times"])[-10:]
        all_times = list(self.performance_baselines["response_times"])

        if len(all_times) < 10:
            return {"impact": "minimal", "reason": "insufficient_baseline"}

        recent_avg = statistics.mean(recent_times)
        baseline_avg = statistics.mean(all_times[:-10])

        performance_degradation = (recent_avg - baseline_avg) / baseline_avg * 100

        if performance_degradation > 50:
            impact = "high"
        elif performance_degradation > 20:
            impact = "moderate"
        else:
            impact = "minimal"

        return {
            "impact": impact,
            "degradation_percent": performance_degradation,
            "recent_avg_response_time": recent_avg,
            "baseline_avg_response_time": baseline_avg,
        }

    def _monitor_openai_generator(
        self, component: Any, name: str, inputs: dict[str, Any]
    ) -> Any:
        """Monitor OpenAI generator component."""
        with self.telemetry.trace_operation(
            f"haystack.component.openai_generator.{name}"
        ) as span:
            try:
                result = component.run(**inputs)

                # Extract OpenAI-specific metrics
                if isinstance(result, dict) and "replies" in result:
                    replies = result["replies"]
                    if replies and hasattr(replies[0], "meta"):
                        meta = replies[0].meta
                        tokens_input = meta.get("prompt_tokens", 0)
                        tokens_output = meta.get("completion_tokens", 0)

                        # Record telemetry
                        span.set_attribute("genops.tokens.input", tokens_input)
                        span.set_attribute("genops.tokens.output", tokens_output)
                        span.set_attribute("genops.provider", "openai")

                        # Calculate and record cost
                        cost = self.cost_aggregator.calculate_accurate_cost(
                            provider="openai",
                            model=getattr(component, "model", "gpt-3.5-turbo"),
                            tokens_input=tokens_input,
                            tokens_output=tokens_output,
                        )

                        span.set_attribute("genops.cost.total", float(cost))

                        # Add to cost aggregator
                        self.cost_aggregator.add_component_cost(
                            component_name=name,
                            provider="openai",
                            cost=float(cost),
                            component_type="OpenAIGenerator",
                            tokens_input=tokens_input,
                            tokens_output=tokens_output,
                            model=getattr(component, "model", "gpt-3.5-turbo"),
                        )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def _monitor_anthropic_generator(
        self, component: Any, name: str, inputs: dict[str, Any]
    ) -> Any:
        """Monitor Anthropic generator component."""
        with self.telemetry.trace_operation(
            f"haystack.component.anthropic_generator.{name}"
        ) as span:
            try:
                result = component.run(**inputs)

                # Extract Anthropic-specific metrics
                if isinstance(result, dict) and "replies" in result:
                    replies = result["replies"]
                    if replies and hasattr(replies[0], "meta"):
                        meta = replies[0].meta
                        tokens_input = meta.get("input_tokens", 0)
                        tokens_output = meta.get("output_tokens", 0)

                        span.set_attribute("genops.tokens.input", tokens_input)
                        span.set_attribute("genops.tokens.output", tokens_output)
                        span.set_attribute("genops.provider", "anthropic")

                        # Calculate cost
                        cost = self.cost_aggregator.calculate_accurate_cost(
                            provider="anthropic",
                            model=getattr(component, "model", "claude-3-haiku"),
                            tokens_input=tokens_input,
                            tokens_output=tokens_output,
                        )

                        span.set_attribute("genops.cost.total", float(cost))

                        self.cost_aggregator.add_component_cost(
                            component_name=name,
                            provider="anthropic",
                            cost=float(cost),
                            component_type="AnthropicGenerator",
                            tokens_input=tokens_input,
                            tokens_output=tokens_output,
                            model=getattr(component, "model", "claude-3-haiku"),
                        )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def _monitor_retriever(
        self, component: Any, name: str, inputs: dict[str, Any]
    ) -> Any:
        """Monitor retrieval component."""
        with self.telemetry.trace_operation(
            f"haystack.component.retriever.{name}"
        ) as span:
            try:
                result = component.run(**inputs)

                # Extract retrieval metrics
                documents_retrieved = 0
                if isinstance(result, dict) and "documents" in result:
                    documents_retrieved = len(result["documents"])

                span.set_attribute(
                    "genops.retrieval.documents_count", documents_retrieved
                )
                span.set_attribute("genops.component.type", "retriever")

                # Estimate retrieval cost (minimal)
                cost = Decimal("0.0001") * documents_retrieved
                span.set_attribute("genops.cost.total", float(cost))

                self.cost_aggregator.add_component_cost(
                    component_name=name,
                    provider="local",
                    cost=float(cost),
                    component_type="Retriever",
                    operations=documents_retrieved,
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def _monitor_embedder(
        self, component: Any, name: str, inputs: dict[str, Any]
    ) -> Any:
        """Monitor embedding component."""
        with self.telemetry.trace_operation(
            f"haystack.component.embedder.{name}"
        ) as span:
            try:
                result = component.run(**inputs)

                # Extract embedding metrics
                embeddings_count = 0
                if isinstance(result, dict):
                    if "documents" in result:
                        embeddings_count = len(result["documents"])
                    elif "embedding" in result:
                        embeddings_count = 1

                span.set_attribute("genops.embedding.count", embeddings_count)
                span.set_attribute(
                    "genops.provider", "openai"
                )  # Assuming OpenAI embedder

                # Calculate embedding cost
                cost = self.cost_aggregator.calculate_accurate_cost(
                    provider="openai",
                    model="text-embedding-3-small",
                    operations=embeddings_count,
                )

                span.set_attribute("genops.cost.total", float(cost))

                self.cost_aggregator.add_component_cost(
                    component_name=name,
                    provider="openai",
                    cost=float(cost),
                    component_type="Embedder",
                    operations=embeddings_count,
                    model="text-embedding-3-small",
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def _monitor_generic_component(
        self, component: Any, name: str, inputs: dict[str, Any]
    ) -> Any:
        """Monitor generic component with basic tracking."""
        with self.telemetry.trace_operation(
            f"haystack.component.generic.{name}"
        ) as span:
            try:
                result = component.run(**inputs)

                span.set_attribute("genops.component.type", "generic")

                # Minimal cost for generic components
                cost = Decimal("0.001")
                span.set_attribute("genops.cost.total", float(cost))

                self.cost_aggregator.add_component_cost(
                    component_name=name,
                    provider="local",
                    cost=float(cost),
                    component_type="Generic",
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def _extract_component_metrics(
        self,
        component_name: str,
        component_type: str,
        result: Any,
        start_time: datetime,
        end_time: datetime,
        execution_time: float,
        input_size_bytes: Optional[int],
        output_size_bytes: Optional[int],
    ) -> ComponentExecutionMetrics:
        """Extract metrics from component execution result."""

        # Try to find cost entry for this component
        cost = Decimal("0")
        tokens_input = None
        tokens_output = None
        provider = None
        model = None

        # Look for recent cost entries for this component
        recent_entries = [
            entry
            for entry in self.cost_aggregator.cost_entries
            if entry.component_name == component_name
            and (datetime.utcnow() - entry.timestamp).total_seconds() < 5
        ]

        if recent_entries:
            latest_entry = recent_entries[-1]
            cost = latest_entry.cost
            tokens_input = latest_entry.tokens_input
            tokens_output = latest_entry.tokens_output
            provider = latest_entry.provider
            model = latest_entry.model

        return ComponentExecutionMetrics(
            component_name=component_name,
            component_type=component_type,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
            provider=provider,
            model=model,
        )

    def _estimate_data_size(self, data: Any) -> Optional[int]:
        """Estimate size of data in bytes."""
        try:
            if isinstance(data, str):
                return len(data.encode("utf-8"))
            elif isinstance(data, dict):
                return len(str(data).encode("utf-8"))
            elif isinstance(data, list):
                return sum(len(str(item).encode("utf-8")) for item in data)
            else:
                return len(str(data).encode("utf-8"))
        except Exception:
            return None


class PipelineExecutionContext:
    """Context for monitoring pipeline execution."""

    def __init__(
        self, monitor: "HaystackMonitor", pipeline_name: str, pipeline_id: str
    ):
        self.monitor = monitor
        self.pipeline_name = pipeline_name
        self.pipeline_id = pipeline_id
        self.start_time = None
        self.end_time = None
        self.component_metrics: list[ComponentExecutionMetrics] = []
        self.span = None

    def __enter__(self):
        """Start pipeline monitoring."""
        self.start_time = datetime.utcnow()

        # Create OpenTelemetry span
        self.span = self.monitor.telemetry.tracer.start_span(
            f"haystack.pipeline.execution.{self.pipeline_name}"
        )

        # Set initial attributes
        self.span.set_attribute("genops.pipeline.name", self.pipeline_name)
        self.span.set_attribute("genops.pipeline.id", self.pipeline_id)
        self.span.set_attribute("genops.framework", "haystack")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete pipeline monitoring."""
        self.end_time = datetime.utcnow()

        # Calculate total execution time
        total_execution_time = (self.end_time - self.start_time).total_seconds()

        # Aggregate metrics
        metrics = self._aggregate_metrics(total_execution_time)

        # Update span with final metrics
        self._update_span_metrics(metrics)

        # Set span status
        if exc_type is None:
            self.span.set_status(Status(StatusCode.OK))
        else:
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)

        # Finish span
        self.span.end()

        # Store metrics
        self.monitor._execution_results[self.pipeline_id] = metrics

        return metrics

    def add_component_metrics(self, component_metrics: ComponentExecutionMetrics):
        """Add component metrics to the pipeline execution."""
        self.component_metrics.append(component_metrics)

    def _aggregate_metrics(
        self, total_execution_time: float
    ) -> PipelineExecutionMetrics:
        """Aggregate all component metrics into pipeline metrics."""
        total_cost = sum(m.cost for m in self.component_metrics)

        # Component statistics
        total_components = len(self.component_metrics)
        successful_components = len(
            [m for m in self.component_metrics if m.status == "success"]
        )
        failed_components = total_components - successful_components

        # Token statistics
        total_tokens_input = sum(m.tokens_input or 0 for m in self.component_metrics)
        total_tokens_output = sum(m.tokens_output or 0 for m in self.component_metrics)

        # Performance analysis
        slowest_component = None
        slowest_component_time = 0.0
        most_expensive_component = None
        highest_component_cost = Decimal("0")

        for m in self.component_metrics:
            if m.execution_time_seconds > slowest_component_time:
                slowest_component = m.component_name
                slowest_component_time = m.execution_time_seconds

            if m.cost > highest_component_cost:
                most_expensive_component = m.component_name
                highest_component_cost = m.cost

        # Cost breakdowns
        cost_by_provider = defaultdict(Decimal)
        cost_by_component_type = defaultdict(Decimal)

        for m in self.component_metrics:
            if m.provider:
                cost_by_provider[m.provider] += m.cost
            cost_by_component_type[m.component_type] += m.cost

        return PipelineExecutionMetrics(
            pipeline_name=self.pipeline_name,
            pipeline_id=self.pipeline_id,
            start_time=self.start_time,  # type: ignore[arg-type]
            end_time=self.end_time,  # type: ignore[arg-type]
            total_execution_time_seconds=total_execution_time,
            total_cost=total_cost,  # type: ignore
            component_metrics=self.component_metrics,
            total_components=total_components,
            successful_components=successful_components,
            failed_components=failed_components,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            slowest_component=slowest_component,
            slowest_component_time=slowest_component_time,
            most_expensive_component=most_expensive_component,
            highest_component_cost=highest_component_cost,
            cost_by_provider=dict(cost_by_provider),
            cost_by_component_type=dict(cost_by_component_type),
        )

    def _update_span_metrics(self, metrics: PipelineExecutionMetrics):
        """Update OpenTelemetry span with aggregated metrics."""
        self.span.set_attribute("genops.cost.total", float(metrics.total_cost))
        self.span.set_attribute(
            "genops.pipeline.components.total", metrics.total_components
        )
        self.span.set_attribute(
            "genops.pipeline.components.successful", metrics.successful_components
        )
        self.span.set_attribute(
            "genops.pipeline.components.failed", metrics.failed_components
        )
        self.span.set_attribute(
            "genops.pipeline.execution_time_seconds",
            metrics.total_execution_time_seconds,
        )
        self.span.set_attribute("genops.tokens.input.total", metrics.total_tokens_input)
        self.span.set_attribute(
            "genops.tokens.output.total", metrics.total_tokens_output
        )

        if metrics.slowest_component:
            self.span.set_attribute(
                "genops.pipeline.slowest_component", metrics.slowest_component
            )
            self.span.set_attribute(
                "genops.pipeline.slowest_component_time", metrics.slowest_component_time
            )

        if metrics.most_expensive_component:
            self.span.set_attribute(
                "genops.pipeline.most_expensive_component",
                metrics.most_expensive_component,
            )
            self.span.set_attribute(
                "genops.pipeline.highest_component_cost",
                float(metrics.highest_component_cost),
            )

        # Set provider cost breakdown
        for provider, cost in metrics.cost_by_provider.items():
            self.span.set_attribute(f"genops.cost.provider.{provider}", float(cost))

    def get_metrics(self) -> PipelineExecutionMetrics:
        """Get current pipeline metrics (for in-progress monitoring)."""
        if self.end_time:
            # Execution completed
            return self.monitor._execution_results.get(self.pipeline_id)
        else:
            # Execution in progress
            current_time = datetime.utcnow()
            partial_execution_time = (current_time - self.start_time).total_seconds()
            return self._aggregate_metrics(partial_execution_time)


class HaystackMonitor:
    """
    Comprehensive monitoring system for Haystack AI pipelines.

    Provides deep insights into pipeline execution, component performance,
    cost tracking, and governance telemetry with specialized support for
    RAG workflows and agent systems.
    """

    def __init__(
        self,
        team: str = "default-team",
        project: str = "haystack-monitoring",
        environment: str = "development",
        enable_performance_monitoring: bool = True,
        enable_cost_tracking: bool = True,
        enable_rag_specialization: bool = True,
        enable_agent_tracking: bool = True,
        **kwargs,
    ):
        """
        Initialize Haystack monitor.

        Args:
            team: Team name for governance
            project: Project name for governance
            environment: Environment name
            enable_performance_monitoring: Enable detailed performance tracking
            enable_cost_tracking: Enable cost aggregation
            enable_rag_specialization: Enable RAG workflow specialization
            enable_agent_tracking: Enable agent workflow tracking
            **kwargs: Additional configuration
        """
        if not HAS_HAYSTACK:
            raise ImportError(
                "Haystack not installed. Install with: pip install haystack-ai"
            )

        self.team = team
        self.project = project
        self.environment = environment
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_rag_specialization = enable_rag_specialization
        self.enable_agent_tracking = enable_agent_tracking

        # Initialize telemetry
        self.telemetry = GenOpsTelemetry(tracer_name="haystack-monitor")

        # Initialize cost aggregator with lazy import
        self.cost_aggregator = None
        self._lazy_init_cost_aggregator()

        # Initialize component monitor
        self.component_monitor = ComponentMonitor(self.telemetry, self.cost_aggregator)

        # Execution results storage
        self._execution_results: dict[str, PipelineExecutionMetrics] = {}

        logger.info(
            f"Haystack monitor initialized for team '{team}', project '{project}'"
        )

    def _lazy_init_cost_aggregator(self):
        """Lazily initialize cost aggregator to avoid circular imports."""
        try:
            from genops.providers.haystack_cost_aggregator import HaystackCostAggregator

            self.cost_aggregator = HaystackCostAggregator()
        except ImportError as e:
            logger.warning(f"Could not initialize cost aggregator: {e}")
            self.cost_aggregator = None

    @contextmanager
    def monitor_pipeline(
        self, pipeline: Pipeline, pipeline_name: str, **governance_attrs
    ):
        """
        Monitor entire Haystack pipeline execution.

        Args:
            pipeline: Haystack pipeline to monitor
            pipeline_name: Name of the pipeline
            **governance_attrs: Additional governance attributes

        Yields:
            PipelineExecutionContext: Monitoring context

        Example:
            with monitor.monitor_pipeline(pipeline, "rag-qa") as execution:
                result = pipeline.run({"query": "What is RAG?"})
                metrics = execution.get_metrics()
        """
        import uuid

        pipeline_id = str(uuid.uuid4())

        with PipelineExecutionContext(self, pipeline_name, pipeline_id) as context:
            # TODO: Hook into pipeline execution to monitor components
            # This would require monkey-patching or Haystack hooks
            yield context

    def analyze_rag_workflow(
        self, metrics: PipelineExecutionMetrics
    ) -> RAGWorkflowMetrics:
        """
        Analyze metrics to extract RAG-specific insights.

        Args:
            metrics: Pipeline execution metrics

        Returns:
            RAGWorkflowMetrics: RAG-specific analysis
        """
        rag_metrics = RAGWorkflowMetrics()

        # Identify RAG components
        retrieval_components = [
            m
            for m in metrics.component_metrics
            if "retriever" in m.component_type.lower()
        ]
        generation_components = [
            m
            for m in metrics.component_metrics
            if "generator" in m.component_type.lower()
        ]
        embedding_components = [
            m
            for m in metrics.component_metrics
            if "embedder" in m.component_type.lower()
        ]

        # Extract retrieval metrics
        if retrieval_components:
            rag_metrics.retrieval_metrics = retrieval_components[0]
            rag_metrics.retrieval_latency_seconds = retrieval_components[
                0
            ].execution_time_seconds
            rag_metrics.documents_retrieved = retrieval_components[
                0
            ].custom_metrics.get("documents_count", 0)
            rag_metrics.retrieval_success_rate = (
                1.0 if retrieval_components[0].status == "success" else 0.0
            )

        # Extract generation metrics
        if generation_components:
            rag_metrics.generation_metrics = generation_components[0]
            rag_metrics.generation_latency_seconds = generation_components[
                0
            ].execution_time_seconds
            rag_metrics.generation_success_rate = (
                1.0 if generation_components[0].status == "success" else 0.0
            )

        # Extract embedding metrics
        rag_metrics.embedding_metrics = embedding_components

        # Calculate end-to-end latency
        rag_metrics.end_to_end_latency_seconds = metrics.total_execution_time_seconds

        return rag_metrics

    def analyze_agent_workflow(
        self, metrics: PipelineExecutionMetrics
    ) -> AgentWorkflowMetrics:
        """
        Analyze metrics to extract agent-specific insights.

        Args:
            metrics: Pipeline execution metrics

        Returns:
            AgentWorkflowMetrics: Agent-specific analysis
        """
        agent_metrics = AgentWorkflowMetrics()

        # Count decisions and tool usage
        for component_metric in metrics.component_metrics:
            if "agent" in component_metric.component_type.lower():
                agent_metrics.decisions_made += 1

            # Track tool usage based on component names
            tool_name = component_metric.component_name
            if tool_name not in agent_metrics.tool_usage_count:
                agent_metrics.tool_usage_count[tool_name] = 0
                agent_metrics.tool_success_rate[tool_name] = 0.0

            agent_metrics.tool_usage_count[tool_name] += 1
            agent_metrics.tools_used.append(tool_name)

            # Update success rate
            if component_metric.status == "success":
                current_success = agent_metrics.tool_success_rate[tool_name]
                current_count = agent_metrics.tool_usage_count[tool_name]
                agent_metrics.tool_success_rate[tool_name] = (
                    current_success * (current_count - 1) + 1.0
                ) / current_count

            # Track cost by tool
            agent_metrics.cost_by_tool[tool_name] = (
                agent_metrics.cost_by_tool.get(tool_name, Decimal("0"))
                + component_metric.cost
            )

        # Calculate average decision latency
        if agent_metrics.decisions_made > 0:
            agent_metrics.decision_latency_seconds = (
                metrics.total_execution_time_seconds / agent_metrics.decisions_made
            )

        return agent_metrics

    def get_execution_metrics(
        self, pipeline_id: str
    ) -> Optional[PipelineExecutionMetrics]:
        """Get metrics for a specific pipeline execution."""
        return self._execution_results.get(pipeline_id)

    def get_recent_executions(self, limit: int = 10) -> list[PipelineExecutionMetrics]:
        """Get recent pipeline execution metrics."""
        return sorted(
            self._execution_results.values(), key=lambda x: x.end_time, reverse=True
        )[:limit]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary with enhanced diagnostics."""
        if not self._execution_results:
            return {"message": "No pipeline executions recorded"}

        executions = list(self._execution_results.values())

        # Aggregate statistics
        total_executions = len(executions)
        total_cost = sum(e.total_cost for e in executions)
        avg_execution_time = (
            sum(e.total_execution_time_seconds for e in executions) / total_executions
        )
        avg_cost = total_cost / total_executions

        # Success rates
        successful_executions = len([e for e in executions if e.failed_components == 0])
        success_rate = successful_executions / total_executions

        # Most common components
        component_usage = defaultdict(int)
        component_errors = defaultdict(int)

        for execution in executions:
            for component in execution.component_metrics:
                component_usage[component.component_type] += 1
                if component.status == "error":
                    component_errors[component.component_type] += 1

        # Performance trends
        if len(executions) > 1:
            recent_executions = executions[-10:]  # Last 10 executions
            recent_avg_time = sum(
                e.total_execution_time_seconds for e in recent_executions
            ) / len(recent_executions)
            recent_avg_cost = sum(e.total_cost for e in recent_executions) / len(
                recent_executions
            )

            time_trend = (
                "improving"
                if recent_avg_time < avg_execution_time
                else "degrading"
                if recent_avg_time > avg_execution_time * 1.1
                else "stable"
            )
            cost_trend = (
                "improving"
                if recent_avg_cost < avg_cost
                else "degrading"
                if recent_avg_cost > avg_cost * 1.1
                else "stable"
            )
        else:
            time_trend = "insufficient_data"
            cost_trend = "insufficient_data"

        return {
            "total_executions": total_executions,
            "total_cost": float(total_cost),
            "average_execution_time_seconds": avg_execution_time,
            "average_cost_per_execution": float(avg_cost),
            "success_rate": success_rate,
            "most_used_components": dict(
                sorted(component_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "component_error_rates": {
                comp_type: (
                    component_errors[comp_type] / component_usage[comp_type] * 100
                )
                for comp_type in component_usage.keys()
            },
            "performance_trends": {"execution_time": time_trend, "cost": cost_trend},
            "error_diagnostics": self.get_error_diagnostics(),
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
        }

    def reset_error_tracking(self):
        """
        Reset error tracking statistics. Useful for testing or maintenance.
        """
        self.error_tracker = {
            "component_failures": defaultdict(int),
            "error_types": defaultdict(int),
            "retry_attempts": defaultdict(int),
            "circuit_breaker_states": {},
            "error_patterns": deque(maxlen=100),
        }

        self.performance_baselines = {
            "response_times": deque(maxlen=50),
            "error_rates": deque(maxlen=20),
            "cost_patterns": deque(maxlen=30),
        }

        logger.info("Error tracking and performance baselines reset")

    def get_health_status(self) -> dict[str, Any]:
        """
        Get overall health status of the monitoring system.
        """
        total_errors = sum(self.error_tracker["component_failures"].values())
        total_components_run = len(self.error_tracker["component_failures"])

        open_circuit_breakers = len(
            [
                state
                for state in self.error_tracker["circuit_breaker_states"].values()
                if state["state"] == "open"
            ]
        )

        if open_circuit_breakers > 0:
            health_status = "degraded"
            health_reason = f"{open_circuit_breakers} circuit breakers are open"
        elif total_components_run > 0 and (total_errors / total_components_run) > 0.1:
            health_status = "degraded"
            health_reason = (
                f"High error rate: {(total_errors / total_components_run * 100):.1f}%"
            )
        else:
            health_status = "healthy"
            health_reason = "All components operating normally"

        return {
            "status": health_status,
            "reason": health_reason,
            "total_errors": total_errors,
            "open_circuit_breakers": open_circuit_breakers,
            "monitoring_active": True,
            "components_tracked": total_components_run,
        }

    def with_monitoring_decorator(self):
        """
        Decorator factory for adding monitoring to external functions.

        Returns:
            Decorator function for adding monitoring

        Example:
            @monitor.with_monitoring_decorator()
            def my_component_function(inputs):
                return process_inputs(inputs)
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                component_name = func.__name__
                component_type = "DecoratedFunction"

                # Convert args/kwargs to inputs dict for monitoring

                # Monitor the execution
                start_time = datetime.utcnow()
                try:
                    result = func(*args, **kwargs)

                    # Create success metrics
                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds()

                    ComponentExecutionMetrics(
                        component_name=component_name,
                        component_type=component_type,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time_seconds=execution_time,
                        status="success",
                    )

                    # Record success
                    self._record_success(component_name)

                    return result

                except Exception as e:
                    # Create error metrics
                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds()

                    ComponentExecutionMetrics(
                        component_name=component_name,
                        component_type=component_type,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time_seconds=execution_time,
                        status="error",
                        error_message=str(e),
                    )

                    # Record failure
                    self._record_failure(component_name, str(e))

                    raise

            return wrapper

        return decorator


# Export main classes
__all__ = [
    "HaystackMonitor",
    "ComponentMonitor",
    "PipelineExecutionContext",
    "ComponentExecutionMetrics",
    "PipelineExecutionMetrics",
    "RAGWorkflowMetrics",
    "AgentWorkflowMetrics",
]
