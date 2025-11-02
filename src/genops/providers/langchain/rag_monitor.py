"""RAG and vector operation monitoring for LangChain."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain.embeddings.base import Embeddings
    from langchain.retrievers.base import BaseRetriever
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStore
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    Document = Any
    VectorStore = Any
    BaseRetriever = Any
    Embeddings = Any


@dataclass
class RetrievalMetrics:
    """Metrics for a retrieval operation."""
    query: str
    documents_retrieved: int
    retrieval_time: float
    relevance_scores: list[float] = field(default_factory=list)
    vector_store_type: str | None = None
    embedding_model: str | None = None
    search_type: str = "similarity"
    search_params: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_relevance_score(self) -> float:
        """Calculate average relevance score."""
        return sum(self.relevance_scores) / len(self.relevance_scores) if self.relevance_scores else 0.0

    @property
    def min_relevance_score(self) -> float:
        """Get minimum relevance score."""
        return min(self.relevance_scores) if self.relevance_scores else 0.0

    @property
    def max_relevance_score(self) -> float:
        """Get maximum relevance score."""
        return max(self.relevance_scores) if self.relevance_scores else 0.0


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations."""
    texts_embedded: int
    embedding_time: float
    embedding_model: str
    total_tokens: int = 0
    cost: float = 0.0
    embedding_dimensions: int | None = None


@dataclass
class RAGOperationSummary:
    """Summary of a complete RAG operation."""
    operation_id: str
    query: str
    retrieval_metrics: RetrievalMetrics | None = None
    embedding_metrics: list[EmbeddingMetrics] = field(default_factory=list)
    generation_cost: float = 0.0
    total_cost: float = 0.0
    total_time: float = 0.0
    documents_processed: int = 0
    context_length: int = 0

    def calculate_total_cost(self) -> float:
        """Calculate total cost across all operations."""
        embedding_cost = sum(em.cost for em in self.embedding_metrics)
        return self.generation_cost + embedding_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "operation_id": self.operation_id,
            "query_length": len(self.query),
            "documents_retrieved": self.retrieval_metrics.documents_retrieved if self.retrieval_metrics else 0,
            "retrieval_time": self.retrieval_metrics.retrieval_time if self.retrieval_metrics else 0.0,
            "avg_relevance_score": self.retrieval_metrics.avg_relevance_score if self.retrieval_metrics else 0.0,
            "embedding_operations": len(self.embedding_metrics),
            "total_embeddings": sum(em.texts_embedded for em in self.embedding_metrics),
            "total_embedding_tokens": sum(em.total_tokens for em in self.embedding_metrics),
            "total_cost": self.calculate_total_cost(),
            "generation_cost": self.generation_cost,
            "embedding_cost": sum(em.cost for em in self.embedding_metrics),
            "total_time": self.total_time,
            "documents_processed": self.documents_processed,
            "context_length": self.context_length,
        }


class RAGOperationMonitor:
    """Monitor RAG operations in LangChain workflows."""

    def __init__(self):
        self.active_operations: dict[str, RAGOperationSummary] = {}
        self.embedding_cost_calculators = {
            "openai": self._calculate_openai_embedding_cost,
            "text-embedding-ada-002": self._calculate_openai_embedding_cost,
            "text-embedding-3-small": self._calculate_openai_embedding_cost,
            "text-embedding-3-large": self._calculate_openai_embedding_cost,
        }

    def start_rag_operation(self, query: str, operation_id: str | None = None) -> str:
        """Start tracking a RAG operation."""
        if not operation_id:
            operation_id = str(uuid.uuid4())

        self.active_operations[operation_id] = RAGOperationSummary(
            operation_id=operation_id,
            query=query
        )

        logger.debug(f"Started RAG operation tracking: {operation_id}")
        return operation_id

    def record_retrieval_operation(
        self,
        operation_id: str,
        query: str,
        documents: list[Document],
        retrieval_time: float,
        vector_store_type: str | None = None,
        embedding_model: str | None = None,
        search_type: str = "similarity",
        **search_params
    ) -> None:
        """Record a retrieval operation."""
        if operation_id not in self.active_operations:
            logger.warning(f"RAG operation {operation_id} not found")
            return

        # Extract relevance scores if available
        relevance_scores = []
        for doc in documents:
            if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                relevance_scores.append(doc.metadata['score'])

        retrieval_metrics = RetrievalMetrics(
            query=query,
            documents_retrieved=len(documents),
            retrieval_time=retrieval_time,
            relevance_scores=relevance_scores,
            vector_store_type=vector_store_type,
            embedding_model=embedding_model,
            search_type=search_type,
            search_params=search_params
        )

        self.active_operations[operation_id].retrieval_metrics = retrieval_metrics
        self.active_operations[operation_id].documents_processed = len(documents)

        # Calculate context length
        context_length = sum(len(doc.page_content) for doc in documents)
        self.active_operations[operation_id].context_length = context_length

        logger.debug(f"Recorded retrieval: {len(documents)} docs, {retrieval_time:.3f}s")

    def record_embedding_operation(
        self,
        operation_id: str,
        texts: list[str],
        embedding_time: float,
        embedding_model: str,
        embeddings: list[list[float]] | None = None
    ) -> None:
        """Record an embedding operation."""
        if operation_id not in self.active_operations:
            logger.warning(f"RAG operation {operation_id} not found")
            return

        # Calculate token count (rough estimate)
        total_tokens = sum(len(text.split()) * 1.3 for text in texts)

        # Calculate cost
        cost = self._calculate_embedding_cost(embedding_model, total_tokens)

        # Get embedding dimensions if available
        embedding_dimensions = None
        if embeddings and embeddings[0]:
            embedding_dimensions = len(embeddings[0])

        embedding_metrics = EmbeddingMetrics(
            texts_embedded=len(texts),
            embedding_time=embedding_time,
            embedding_model=embedding_model,
            total_tokens=int(total_tokens),
            cost=cost,
            embedding_dimensions=embedding_dimensions
        )

        self.active_operations[operation_id].embedding_metrics.append(embedding_metrics)
        logger.debug(f"Recorded embedding: {len(texts)} texts, ${cost:.4f}, {embedding_time:.3f}s")

    def record_generation_cost(self, operation_id: str, cost: float) -> None:
        """Record the cost of text generation in the RAG operation."""
        if operation_id not in self.active_operations:
            logger.warning(f"RAG operation {operation_id} not found")
            return

        self.active_operations[operation_id].generation_cost = cost

    def finalize_rag_operation(self, operation_id: str, total_time: float) -> RAGOperationSummary | None:
        """Finalize a RAG operation and return summary."""
        if operation_id not in self.active_operations:
            logger.warning(f"RAG operation {operation_id} not found")
            return None

        summary = self.active_operations.pop(operation_id)
        summary.total_time = total_time
        summary.total_cost = summary.calculate_total_cost()

        logger.debug(f"Finalized RAG operation {operation_id}: ${summary.total_cost:.4f}, {total_time:.3f}s")
        return summary

    def _calculate_embedding_cost(self, model: str, tokens: int) -> float:
        """Calculate embedding cost based on model and tokens."""
        model_key = model.lower()

        if model_key in self.embedding_cost_calculators:
            return self.embedding_cost_calculators[model_key](tokens)

        # Default OpenAI pricing if model not recognized
        return self._calculate_openai_embedding_cost(tokens)

    def _calculate_openai_embedding_cost(self, tokens: int) -> float:
        """Calculate OpenAI embedding cost."""
        # OpenAI text-embedding-ada-002 pricing: $0.0001 per 1K tokens
        # text-embedding-3-small: $0.00002 per 1K tokens
        # text-embedding-3-large: $0.00013 per 1K tokens
        # Use ada-002 as default
        cost_per_1k_tokens = 0.0001
        return (tokens / 1000) * cost_per_1k_tokens


class LangChainRAGInstrumentor:
    """Instrument LangChain RAG operations."""

    def __init__(self, telemetry_adapter):
        self.telemetry_adapter = telemetry_adapter
        self.monitor = RAGOperationMonitor()

    def instrument_retriever(self, retriever: BaseRetriever, operation_id: str) -> BaseRetriever:
        """Instrument a retriever with monitoring."""
        if not HAS_LANGCHAIN:
            return retriever

        original_get_relevant_documents = retriever.get_relevant_documents

        def instrumented_get_relevant_documents(query: str, **kwargs) -> list[Document]:
            start_time = time.time()

            try:
                documents = original_get_relevant_documents(query, **kwargs)

                retrieval_time = time.time() - start_time

                # Try to determine vector store type
                vector_store_type = None
                if hasattr(retriever, 'vectorstore'):
                    vector_store_type = type(retriever.vectorstore).__name__
                elif hasattr(retriever, 'vector_store'):
                    vector_store_type = type(retriever.vector_store).__name__

                # Try to determine embedding model
                embedding_model = None
                if hasattr(retriever, 'embedding') or hasattr(retriever, 'embeddings'):
                    embedding_obj = getattr(retriever, 'embedding', None) or getattr(retriever, 'embeddings', None)
                    if embedding_obj:
                        embedding_model = type(embedding_obj).__name__

                self.monitor.record_retrieval_operation(
                    operation_id=operation_id,
                    query=query,
                    documents=documents,
                    retrieval_time=retrieval_time,
                    vector_store_type=vector_store_type,
                    embedding_model=embedding_model,
                    **kwargs
                )

                return documents

            except Exception as e:
                logger.error(f"Error in retrieval instrumentation: {e}")
                raise

        retriever.get_relevant_documents = instrumented_get_relevant_documents
        return retriever

    def instrument_embeddings(self, embeddings: Embeddings, operation_id: str) -> Embeddings:
        """Instrument embeddings with monitoring."""
        if not HAS_LANGCHAIN:
            return embeddings

        original_embed_documents = embeddings.embed_documents
        original_embed_query = embeddings.embed_query

        def instrumented_embed_documents(texts: list[str]) -> list[list[float]]:
            start_time = time.time()

            try:
                result = original_embed_documents(texts)
                embedding_time = time.time() - start_time

                self.monitor.record_embedding_operation(
                    operation_id=operation_id,
                    texts=texts,
                    embedding_time=embedding_time,
                    embedding_model=type(embeddings).__name__,
                    embeddings=result
                )

                return result

            except Exception as e:
                logger.error(f"Error in embedding instrumentation: {e}")
                raise

        def instrumented_embed_query(text: str) -> list[float]:
            start_time = time.time()

            try:
                result = original_embed_query(text)
                embedding_time = time.time() - start_time

                self.monitor.record_embedding_operation(
                    operation_id=operation_id,
                    texts=[text],
                    embedding_time=embedding_time,
                    embedding_model=type(embeddings).__name__,
                    embeddings=[result] if result else None
                )

                return result

            except Exception as e:
                logger.error(f"Error in query embedding instrumentation: {e}")
                raise

        embeddings.embed_documents = instrumented_embed_documents
        embeddings.embed_query = instrumented_embed_query

        return embeddings

    def create_rag_context(self, query: str) -> 'RAGContext':
        """Create a context manager for RAG operation tracking."""
        return RAGContext(query, self.monitor, self.telemetry_adapter)


class RAGContext:
    """Context manager for RAG operation tracking."""

    def __init__(self, query: str, monitor: RAGOperationMonitor, telemetry_adapter):
        self.query = query
        self.monitor = monitor
        self.telemetry_adapter = telemetry_adapter
        self.operation_id = None
        self.start_time = None
        self.summary = None

    def __enter__(self) -> 'RAGContext':
        self.start_time = time.time()
        self.operation_id = self.monitor.start_rag_operation(self.query)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time and self.operation_id:
            total_time = time.time() - self.start_time
            self.summary = self.monitor.finalize_rag_operation(self.operation_id, total_time)

    def get_operation_id(self) -> str | None:
        """Get the operation ID for this RAG context."""
        return self.operation_id

    def record_generation_cost(self, cost: float) -> None:
        """Record generation cost within this context."""
        if self.operation_id:
            self.monitor.record_generation_cost(self.operation_id, cost)

    def get_summary(self) -> RAGOperationSummary | None:
        """Get the final summary (available after context exit)."""
        return self.summary


# Global RAG monitor instance
_rag_monitor: RAGOperationMonitor | None = None


def get_rag_monitor() -> RAGOperationMonitor:
    """Get the global RAG monitor instance."""
    global _rag_monitor
    if _rag_monitor is None:
        _rag_monitor = RAGOperationMonitor()
    return _rag_monitor
