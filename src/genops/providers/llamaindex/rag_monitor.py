"""LlamaIndex RAG monitor for GenOps AI governance."""

import logging
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

try:
    from llama_index.core.response import Response
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from llama_index.core.vector_stores import VectorStoreQuery

    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False
    NodeWithScore = None  # type: ignore[misc,assignment]
    QueryBundle = None  # type: ignore[misc,assignment]
    Response = None  # type: ignore[misc,assignment]
    VectorStoreQuery = None  # type: ignore[misc,assignment]
    logger.warning("LlamaIndex not available for RAG monitoring")


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations in RAG pipelines."""

    operation_id: str
    text_length: int
    embedding_model: str
    embedding_dimensions: int
    processing_time_ms: float
    cost_usd: float = 0.0
    provider: str = "unknown"
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval operations in RAG pipelines."""

    operation_id: str
    query: str
    similarity_top_k: int
    retrieved_count: int
    retrieval_time_ms: float
    vector_store_type: str = "unknown"

    # Quality metrics
    avg_similarity_score: Optional[float] = None
    min_similarity_score: Optional[float] = None
    max_similarity_score: Optional[float] = None

    # Performance metrics
    search_time_ms: Optional[float] = None
    postprocess_time_ms: Optional[float] = None

    # Cost tracking
    cost_usd: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    # Retrieved content analysis
    avg_content_length: Optional[float] = None
    content_diversity_score: Optional[float] = None


@dataclass
class SynthesisMetrics:
    """Metrics for synthesis (LLM generation) operations."""

    operation_id: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    synthesis_time_ms: float
    cost_usd: float = 0.0

    # Quality metrics
    response_length: int = 0
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None

    # Context utilization
    context_tokens: Optional[int] = None
    context_utilization_ratio: Optional[float] = None

    success: bool = True
    error_message: Optional[str] = None


@dataclass
class RAGOperationSummary:
    """Comprehensive summary of a RAG operation."""

    query_id: str
    query_text: str
    start_time: float
    end_time: Optional[float] = None

    # Component metrics
    embedding_metrics: Optional[EmbeddingMetrics] = None
    retrieval_metrics: Optional[RetrievalMetrics] = None
    synthesis_metrics: Optional[SynthesisMetrics] = None

    # Overall metrics
    total_cost_usd: float = 0.0
    total_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    # Governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    environment: Optional[str] = None

    def finalize(self) -> None:
        """Finalize the operation summary with calculated metrics."""
        if self.end_time is None:
            self.end_time = time.time()

        self.total_time_ms = (self.end_time - self.start_time) * 1000

        # Aggregate costs
        costs = []
        if self.embedding_metrics:
            costs.append(self.embedding_metrics.cost_usd)
        if self.retrieval_metrics:
            costs.append(self.retrieval_metrics.cost_usd)
        if self.synthesis_metrics:
            costs.append(self.synthesis_metrics.cost_usd)

        self.total_cost_usd = sum(costs)


@dataclass
class RAGPipelineAnalytics:
    """Analytics and insights for RAG pipeline performance."""

    total_operations: int
    avg_cost_per_query: float
    avg_response_time_ms: float

    # Component performance
    embedding_success_rate: float = 1.0
    retrieval_success_rate: float = 1.0
    synthesis_success_rate: float = 1.0

    # Cost breakdown
    cost_by_component: dict[str, float] = field(default_factory=dict)
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)

    # Quality insights
    avg_retrieval_relevance: Optional[float] = None
    avg_synthesis_quality: Optional[float] = None
    content_diversity_trends: list[float] = field(default_factory=list)

    # Performance trends
    response_time_trends: list[float] = field(default_factory=list)
    cost_trends: list[float] = field(default_factory=list)

    # Optimization recommendations
    recommendations: list[str] = field(default_factory=list)


class RAGOperationMonitor:
    """Monitor for individual RAG operations with detailed tracking."""

    def __init__(self, query_id: str, query_text: str, **governance_attrs):
        self.operation = RAGOperationSummary(
            query_id=query_id,
            query_text=query_text,
            start_time=time.time(),
            **governance_attrs,
        )
        self.span = None

    def __enter__(self) -> "RAGOperationMonitor":
        """Start monitoring context."""
        self.span = tracer.start_span("llamaindex.rag_operation")  # type: ignore[assignment]
        self.span.set_attributes(
            {
                "genops.query_id": self.operation.query_id,
                "genops.query_text": self.operation.query_text[:100],  # Truncate
                "genops.framework": "llamaindex",
                "genops.operation_type": "rag_pipeline",
            }
        )

        # Add governance attributes
        if self.operation.team:
            self.span.set_attribute("genops.team", self.operation.team)
        if self.operation.project:
            self.span.set_attribute("genops.project", self.operation.project)
        if self.operation.customer_id:
            self.span.set_attribute("genops.customer_id", self.operation.customer_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring context."""
        self.operation.finalize()

        if exc_type is not None:
            self.operation.success = False
            self.operation.error_message = str(exc_val)
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        else:
            self.span.set_status(Status(StatusCode.OK))

        # Record final metrics
        self.span.set_attributes(
            {
                "genops.total_cost_usd": self.operation.total_cost_usd,
                "genops.total_time_ms": self.operation.total_time_ms,
                "genops.success": self.operation.success,
            }
        )

        self.span.end()

    def record_embedding(self, embedding_metrics: EmbeddingMetrics) -> None:
        """Record embedding operation metrics."""
        self.operation.embedding_metrics = embedding_metrics

        if self.span:
            self.span.set_attributes(
                {
                    "genops.embedding.model": embedding_metrics.embedding_model,
                    "genops.embedding.provider": embedding_metrics.provider,
                    "genops.embedding.dimensions": embedding_metrics.embedding_dimensions,
                    "genops.embedding.cost_usd": embedding_metrics.cost_usd,
                    "genops.embedding.time_ms": embedding_metrics.processing_time_ms,
                }
            )

    def record_retrieval(self, retrieval_metrics: RetrievalMetrics) -> None:
        """Record retrieval operation metrics."""
        self.operation.retrieval_metrics = retrieval_metrics

        if self.span:
            self.span.set_attributes(
                {
                    "genops.retrieval.top_k": retrieval_metrics.similarity_top_k,
                    "genops.retrieval.retrieved_count": retrieval_metrics.retrieved_count,
                    "genops.retrieval.vector_store": retrieval_metrics.vector_store_type,
                    "genops.retrieval.cost_usd": retrieval_metrics.cost_usd,
                    "genops.retrieval.time_ms": retrieval_metrics.retrieval_time_ms,
                }
            )

            if retrieval_metrics.avg_similarity_score is not None:
                self.span.set_attribute(
                    "genops.retrieval.avg_similarity",
                    retrieval_metrics.avg_similarity_score,
                )

    def record_synthesis(self, synthesis_metrics: SynthesisMetrics) -> None:
        """Record synthesis operation metrics."""
        self.operation.synthesis_metrics = synthesis_metrics

        if self.span:
            self.span.set_attributes(
                {
                    "genops.synthesis.model": synthesis_metrics.model,
                    "genops.synthesis.provider": synthesis_metrics.provider,
                    "genops.synthesis.input_tokens": synthesis_metrics.input_tokens,
                    "genops.synthesis.output_tokens": synthesis_metrics.output_tokens,
                    "genops.synthesis.cost_usd": synthesis_metrics.cost_usd,
                    "genops.synthesis.time_ms": synthesis_metrics.synthesis_time_ms,
                }
            )


class LlamaIndexRAGInstrumentor:
    """
    Comprehensive RAG pipeline instrumentation for LlamaIndex.

    Provides detailed monitoring of:
    - Embedding operations and vector store interactions
    - Retrieval performance and relevance metrics
    - Synthesis quality and cost tracking
    - End-to-end pipeline analytics
    """

    def __init__(
        self,
        enable_quality_metrics: bool = True,
        enable_cost_tracking: bool = True,
        enable_performance_profiling: bool = True,
    ):
        """
        Initialize RAG instrumentation.

        Args:
            enable_quality_metrics: Track retrieval and synthesis quality
            enable_cost_tracking: Calculate operation costs
            enable_performance_profiling: Profile component performance
        """
        self.enable_quality_metrics = enable_quality_metrics
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_performance_profiling = enable_performance_profiling

        # Storage for completed operations
        self.completed_operations: list[RAGOperationSummary] = []
        self.active_monitors: dict[str, RAGOperationMonitor] = {}

        # Analytics aggregation
        self._cost_by_component = defaultdict(float)
        self._cost_by_provider = defaultdict(float)
        self._cost_by_model = defaultdict(float)
        self._response_times = []
        self._operation_costs = []

    @contextmanager
    def monitor_rag_operation(self, query: str, **governance_attrs):
        """
        Context manager for monitoring complete RAG operations.

        Args:
            query: The user query being processed
            **governance_attrs: Governance attributes (team, project, customer_id)

        Yields:
            RAGOperationMonitor for recording component metrics
        """
        query_id = str(uuid.uuid4())
        monitor = RAGOperationMonitor(query_id, query, **governance_attrs)

        self.active_monitors[query_id] = monitor

        try:
            with monitor:
                yield monitor
        finally:
            # Move to completed operations
            if query_id in self.active_monitors:
                completed_monitor = self.active_monitors.pop(query_id)
                self.completed_operations.append(completed_monitor.operation)
                self._update_analytics(completed_monitor.operation)

    def create_embedding_metrics(
        self,
        text: str,
        embedding_model: str,
        processing_time_ms: float,
        provider: str = "unknown",
        embedding_dimensions: int = 0,
        cost_usd: float = 0.0,
    ) -> EmbeddingMetrics:
        """Create embedding metrics from operation data."""
        return EmbeddingMetrics(
            operation_id=str(uuid.uuid4()),
            text_length=len(text),
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            processing_time_ms=processing_time_ms,
            cost_usd=cost_usd,
            provider=provider,
        )

    def create_retrieval_metrics(
        self,
        query: str,
        nodes: list[NodeWithScore],
        retrieval_time_ms: float,
        similarity_top_k: int,
        vector_store_type: str = "unknown",
    ) -> RetrievalMetrics:
        """Create retrieval metrics from LlamaIndex retrieval results."""

        # Calculate similarity statistics
        scores = [node.score for node in nodes if node.score is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None

        # Calculate content statistics
        content_lengths = [
            len(node.node.text) for node in nodes if hasattr(node.node, "text")
        ]
        avg_content_length = (
            sum(content_lengths) / len(content_lengths) if content_lengths else None
        )

        # Simple content diversity measure (unique words ratio)
        if content_lengths:
            all_text = " ".join(
                node.node.text for node in nodes if hasattr(node.node, "text")
            )
            words = all_text.split()
            unique_words = set(words)
            diversity_score = len(unique_words) / len(words) if words else None
        else:
            diversity_score = None

        return RetrievalMetrics(
            operation_id=str(uuid.uuid4()),
            query=query,
            similarity_top_k=similarity_top_k,
            retrieved_count=len(nodes),
            retrieval_time_ms=retrieval_time_ms,
            vector_store_type=vector_store_type,
            avg_similarity_score=avg_score,
            min_similarity_score=min_score,
            max_similarity_score=max_score,
            avg_content_length=avg_content_length,
            content_diversity_score=diversity_score,
        )

    def create_synthesis_metrics(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        provider: str,
        synthesis_time_ms: float,
        response_text: str = "",
        cost_usd: float = 0.0,
        context_tokens: Optional[int] = None,
    ) -> SynthesisMetrics:
        """Create synthesis metrics from LLM generation results."""

        # Calculate context utilization if available
        context_utilization = None
        if context_tokens and input_tokens:
            context_utilization = context_tokens / input_tokens

        return SynthesisMetrics(
            operation_id=str(uuid.uuid4()),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider=provider,
            synthesis_time_ms=synthesis_time_ms,
            cost_usd=cost_usd,
            response_length=len(response_text),
            context_tokens=context_tokens,
            context_utilization_ratio=context_utilization,
        )

    def _update_analytics(self, operation: RAGOperationSummary) -> None:
        """Update aggregated analytics with completed operation."""

        # Update cost tracking
        if operation.embedding_metrics:
            self._cost_by_component["embedding"] += operation.embedding_metrics.cost_usd
            self._cost_by_provider[operation.embedding_metrics.provider] += (
                operation.embedding_metrics.cost_usd
            )
            self._cost_by_model[operation.embedding_metrics.embedding_model] += (
                operation.embedding_metrics.cost_usd
            )

        if operation.retrieval_metrics:
            self._cost_by_component["retrieval"] += operation.retrieval_metrics.cost_usd

        if operation.synthesis_metrics:
            self._cost_by_component["synthesis"] += operation.synthesis_metrics.cost_usd
            self._cost_by_provider[operation.synthesis_metrics.provider] += (
                operation.synthesis_metrics.cost_usd
            )
            self._cost_by_model[operation.synthesis_metrics.model] += (
                operation.synthesis_metrics.cost_usd
            )

        # Update performance tracking
        self._response_times.append(operation.total_time_ms)
        self._operation_costs.append(operation.total_cost_usd)

    def get_analytics(self) -> RAGPipelineAnalytics:
        """Get comprehensive analytics for all monitored operations."""

        total_ops = len(self.completed_operations)
        if total_ops == 0:
            return RAGPipelineAnalytics(
                total_operations=0, avg_cost_per_query=0.0, avg_response_time_ms=0.0
            )

        # Calculate success rates
        successful_ops = [op for op in self.completed_operations if op.success]
        embedding_successes = [
            op
            for op in successful_ops
            if op.embedding_metrics and op.embedding_metrics.success
        ]
        retrieval_successes = [
            op
            for op in successful_ops
            if op.retrieval_metrics and op.retrieval_metrics.success
        ]
        synthesis_successes = [
            op
            for op in successful_ops
            if op.synthesis_metrics and op.synthesis_metrics.success
        ]

        embedding_success_rate = len(embedding_successes) / max(
            1, len([op for op in self.completed_operations if op.embedding_metrics])
        )
        retrieval_success_rate = len(retrieval_successes) / max(
            1, len([op for op in self.completed_operations if op.retrieval_metrics])
        )
        synthesis_success_rate = len(synthesis_successes) / max(
            1, len([op for op in self.completed_operations if op.synthesis_metrics])
        )

        # Calculate averages
        avg_cost = (
            sum(self._operation_costs) / len(self._operation_costs)
            if self._operation_costs
            else 0.0
        )
        avg_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times
            else 0.0
        )

        # Quality metrics
        retrieval_relevance_scores = []
        diversity_scores = []

        for op in self.completed_operations:
            if op.retrieval_metrics and op.retrieval_metrics.avg_similarity_score:
                retrieval_relevance_scores.append(
                    op.retrieval_metrics.avg_similarity_score
                )
            if op.retrieval_metrics and op.retrieval_metrics.content_diversity_score:
                diversity_scores.append(op.retrieval_metrics.content_diversity_score)

        avg_retrieval_relevance = (
            sum(retrieval_relevance_scores) / len(retrieval_relevance_scores)
            if retrieval_relevance_scores
            else None
        )

        # Generate recommendations
        recommendations = self._generate_recommendations()

        return RAGPipelineAnalytics(
            total_operations=total_ops,
            avg_cost_per_query=avg_cost,
            avg_response_time_ms=avg_time,
            embedding_success_rate=embedding_success_rate,
            retrieval_success_rate=retrieval_success_rate,
            synthesis_success_rate=synthesis_success_rate,
            cost_by_component=dict(self._cost_by_component),
            cost_by_provider=dict(self._cost_by_provider),
            cost_by_model=dict(self._cost_by_model),
            avg_retrieval_relevance=avg_retrieval_relevance,
            content_diversity_trends=diversity_scores[-10:],  # Last 10 operations
            response_time_trends=self._response_times[-10:],
            cost_trends=self._operation_costs[-10:],
            recommendations=recommendations,
        )

    def _generate_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on analytics."""
        recommendations = []

        if not self.completed_operations:
            return recommendations

        # Cost optimization recommendations
        total_cost = sum(self._operation_costs)
        if total_cost > 0:
            embedding_pct = (
                self._cost_by_component.get("embedding", 0) / total_cost
            ) * 100
            synthesis_pct = (
                self._cost_by_component.get("synthesis", 0) / total_cost
            ) * 100

            if embedding_pct > 40:
                recommendations.append(
                    f"Embedding costs are {embedding_pct:.1f}% of total - consider caching embeddings or using smaller models"
                )

            if synthesis_pct > 70:
                recommendations.append(
                    f"Synthesis costs are {synthesis_pct:.1f}% of total - consider using cheaper models for simple queries"
                )

        # Performance recommendations
        if self._response_times:
            avg_time = sum(self._response_times) / len(self._response_times)
            if avg_time > 5000:  # 5 seconds
                recommendations.append(
                    f"Average response time is {avg_time:.0f}ms - consider optimizing retrieval or using faster models"
                )

        # Quality recommendations
        analytics = self.get_analytics()
        if (
            analytics.avg_retrieval_relevance
            and analytics.avg_retrieval_relevance < 0.7
        ):
            recommendations.append(
                f"Average retrieval relevance is {analytics.avg_retrieval_relevance:.2f} - consider improving embedding quality or indexing strategy"
            )

        return recommendations[:5]  # Limit to top 5

    def export_operation_data(self) -> dict[str, Any]:
        """Export detailed operation data for analysis."""
        return {
            "completed_operations": [asdict(op) for op in self.completed_operations],
            "analytics": asdict(self.get_analytics()),
            "aggregated_metrics": {
                "cost_by_component": dict(self._cost_by_component),
                "cost_by_provider": dict(self._cost_by_provider),
                "cost_by_model": dict(self._cost_by_model),
                "response_times": self._response_times,
                "operation_costs": self._operation_costs,
            },
        }


# Global RAG monitor instance
_current_rag_monitor: Optional[LlamaIndexRAGInstrumentor] = None


def get_rag_monitor() -> Optional[LlamaIndexRAGInstrumentor]:
    """Get the current RAG monitor instance."""
    return _current_rag_monitor


def set_rag_monitor(monitor: LlamaIndexRAGInstrumentor) -> None:
    """Set the current RAG monitor instance."""
    global _current_rag_monitor
    _current_rag_monitor = monitor


def create_rag_monitor(**config) -> LlamaIndexRAGInstrumentor:
    """Create and configure a new RAG monitor."""
    monitor = LlamaIndexRAGInstrumentor(**config)
    set_rag_monitor(monitor)
    return monitor


@dataclass
class RAGQualityMetrics:
    """Quality metrics for RAG pipeline evaluations."""

    retrieval_relevance: float = 0.0
    response_faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    semantic_similarity: float = 0.0
    factual_consistency: float = 0.0


@dataclass
class RAGPerformanceMetrics:
    """Performance metrics for RAG pipeline operations."""

    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    synthesis_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class RAGOperationAnalytics:
    """Analytics summary for RAG pipeline operations."""

    total_operations: int = 0
    avg_cost_per_query: float = 0.0
    avg_response_time_ms: float = 0.0
    embedding_success_rate: float = 1.0
    retrieval_success_rate: float = 1.0
    synthesis_success_rate: float = 1.0
    avg_retrieval_relevance: Optional[float] = None
    recommendations: list[str] = field(default_factory=list)


class RAGMonitor:
    """High-level RAG pipeline monitor with quality, cost, and performance tracking."""

    def __init__(
        self,
        enable_quality_metrics: bool = True,
        enable_cost_tracking: bool = True,
        enable_performance_profiling: bool = True,
        **kwargs,
    ):
        self.enable_quality_metrics = enable_quality_metrics
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_performance_profiling = enable_performance_profiling
        self.operations: list[dict[str, Any]] = []
        self.quality_scores: list[RAGQualityMetrics] = []
        self.performance_metrics: list[RAGPerformanceMetrics] = []

        # Store governance attributes
        self.team = kwargs.get("team", "default")
        self.project = kwargs.get("project", "default")
        self.environment = kwargs.get("environment", "production")

    def record_operation(self, operation: dict[str, Any]) -> None:
        """Record a RAG pipeline operation."""
        self.operations.append(operation)

    def record_quality(self, metrics: RAGQualityMetrics) -> None:
        """Record quality metrics for a pipeline evaluation."""
        self.quality_scores.append(metrics)

    def record_performance(self, metrics: RAGPerformanceMetrics) -> None:
        """Record performance metrics for a pipeline operation."""
        self.performance_metrics.append(metrics)

    def get_analytics(self) -> RAGOperationAnalytics:
        """Get aggregated analytics for all recorded operations."""
        if not self.operations:
            return RAGOperationAnalytics()
        total = len(self.operations)
        return RAGOperationAnalytics(
            total_operations=total,
        )


# Export main classes and functions
__all__ = [
    "LlamaIndexRAGInstrumentor",
    "RAGOperationMonitor",
    "RAGOperationSummary",
    "RAGPipelineAnalytics",
    "EmbeddingMetrics",
    "RetrievalMetrics",
    "SynthesisMetrics",
    "RAGMonitor",
    "RAGQualityMetrics",
    "RAGPerformanceMetrics",
    "RAGOperationAnalytics",
    "get_rag_monitor",
    "set_rag_monitor",
    "create_rag_monitor",
]
