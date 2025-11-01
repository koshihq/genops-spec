"""LangChain provider registration for auto-instrumentation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genops.auto_instrumentation import GenOpsInstrumentor

logger = logging.getLogger(__name__)


def register_langchain_provider(instrumentor: 'GenOpsInstrumentor') -> None:
    """
    Register LangChain provider with the auto-instrumentation system.
    
    Args:
        instrumentor: GenOps auto-instrumentation instance
    """
    try:
        from .adapter import patch_langchain, unpatch_langchain, GenOpsLangChainAdapter
        
        instrumentor.register_framework_provider(
            name="langchain",
            patch_func=patch_langchain,
            unpatch_func=unpatch_langchain,
            module="langchain",
            framework_type="orchestration",
            provider_class=GenOpsLangChainAdapter,
            description="LangChain orchestration framework for LLM applications",
            capabilities=[
                "chain_execution_tracking",
                "multi_provider_cost_aggregation", 
                "agent_decision_telemetry",
                "rag_operation_monitoring"
            ]
        )
        
        logger.info("LangChain provider registered for auto-instrumentation")
        
    except ImportError as e:
        logger.debug(f"LangChain not available for registration: {e}")
    except Exception as e:
        logger.error(f"Failed to register LangChain provider: {e}")


def auto_register() -> None:
    """Automatically register LangChain provider if auto-instrumentation is available."""
    try:
        from genops.auto_instrumentation import _instrumentor
        register_langchain_provider(_instrumentor)
    except ImportError:
        logger.debug("Auto-instrumentation not available, skipping LangChain registration")