"""LangChain provider for GenOps AI governance."""

from .adapter import (
    GenOpsLangChainAdapter,
    GenOpsLangChainCallbackHandler,
    instrument_langchain,
    patch_langchain,
    unpatch_langchain,
)
from .cost_aggregator import (
    ChainCostSummary,
    LangChainCostAggregator,
    LLMCallCost,
    create_chain_cost_context,
    get_cost_aggregator,
)
from .rag_monitor import (
    EmbeddingMetrics,
    LangChainRAGInstrumentor,
    RAGOperationMonitor,
    RAGOperationSummary,
    RetrievalMetrics,
    get_rag_monitor,
)
from .registration import auto_register, register_langchain_provider
from .validation import (
    ValidationIssue,
    ValidationResult,
    print_validation_result,
    validate_setup,
)

# Auto-register with instrumentation system if available
auto_register()

__all__ = [
    "GenOpsLangChainAdapter",
    "GenOpsLangChainCallbackHandler",
    "instrument_langchain",
    "patch_langchain",
    "unpatch_langchain",
    "register_langchain_provider",
    "LLMCallCost",
    "ChainCostSummary",
    "LangChainCostAggregator",
    "get_cost_aggregator",
    "create_chain_cost_context",
    "RetrievalMetrics",
    "EmbeddingMetrics",
    "RAGOperationSummary",
    "RAGOperationMonitor",
    "LangChainRAGInstrumentor",
    "get_rag_monitor",
    "ValidationIssue",
    "ValidationResult",
    "validate_setup",
    "print_validation_result",
]
