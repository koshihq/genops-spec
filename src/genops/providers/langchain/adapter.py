"""LangChain provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from genops.providers.base import BaseFrameworkProvider
from .cost_aggregator import get_cost_aggregator, create_chain_cost_context
from .rag_monitor import LangChainRAGInstrumentor, get_rag_monitor

logger = logging.getLogger(__name__)

try:
    import langchain
    from langchain.schema import BaseMessage
    from langchain.chains.base import Chain
    from langchain.agents.agent import AgentExecutor
    from langchain.callbacks.base import BaseCallbackHandler
    
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseCallbackHandler = object  # Fallback for type hints
    logger.warning("LangChain not installed. Install with: pip install langchain")


class GenOpsLangChainCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LangChain to capture telemetry."""
    
    def __init__(self, telemetry_adapter: 'GenOpsLangChainAdapter', chain_id: Optional[str] = None):
        self.telemetry_adapter = telemetry_adapter
        self.chain_id = chain_id or str(uuid.uuid4())
        self.chain_context = {}
        self.operation_stack = []
        self.cost_aggregator = get_cost_aggregator()
        
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Called when a chain starts running."""
        chain_name = serialized.get("name", "unknown_chain")
        self.operation_stack.append({
            "type": "chain",
            "name": chain_name,
            "inputs": inputs,
            "start_time": None  # Will be set by telemetry
        })
        
        logger.debug(f"Chain started: {chain_name}")
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes running."""
        if self.operation_stack:
            operation = self.operation_stack.pop()
            operation["outputs"] = outputs
            logger.debug(f"Chain ended: {operation['name']}")
            
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain encounters an error."""
        if self.operation_stack:
            operation = self.operation_stack.pop()
            operation["error"] = str(error)
            logger.debug(f"Chain error: {operation.get('name', 'unknown')} - {error}")
            
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when LLM starts generating."""
        model_name = serialized.get("name", "unknown_llm")
        self.operation_stack.append({
            "type": "llm",
            "name": model_name,
            "prompts": prompts,
            "prompt_tokens": sum(len(p.split()) for p in prompts) * 1.3,  # Rough estimate
        })
        
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM finishes generating."""
        if self.operation_stack:
            operation = self.operation_stack.pop()
            
            # Extract token usage and provider information if available
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                operation['token_usage'] = token_usage
                
                # Extract cost information and add to aggregator
                if token_usage:
                    tokens_input = token_usage.get('prompt_tokens', 0)
                    tokens_output = token_usage.get('completion_tokens', 0)
                    
                    # Try to determine provider from the model name or response
                    provider = self._detect_provider_from_response(response)
                    model = operation.get('name', 'unknown_model')
                    
                    if provider and tokens_input > 0:
                        try:
                            self.cost_aggregator.add_llm_call_cost(
                                chain_id=self.chain_id,
                                provider=provider,
                                model=model,
                                tokens_input=tokens_input,
                                tokens_output=tokens_output,
                                operation_name=f"llm.{model}"
                            )
                            logger.debug(f"Recorded LLM cost for {provider}/{model}: {tokens_input}+{tokens_output} tokens")
                        except Exception as e:
                            logger.warning(f"Failed to record LLM cost: {e}")
                            
    def _detect_provider_from_response(self, response: Any) -> Optional[str]:
        """Detect the provider from LLM response object."""
        # Try to detect provider based on response structure or model name
        if hasattr(response, 'llm_output') and response.llm_output:
            model_name = response.llm_output.get('model_name', '').lower()
            
            if 'gpt' in model_name or 'openai' in model_name:
                return 'openai'
            elif 'claude' in model_name or 'anthropic' in model_name:
                return 'anthropic'
            elif 'gemini' in model_name or 'google' in model_name:
                return 'google'
                
        # Fallback detection based on response type
        response_type = type(response).__name__.lower()
        if 'openai' in response_type:
            return 'openai'
        elif 'anthropic' in response_type:
            return 'anthropic'
            
        # Default fallback
        return 'unknown'
                
    def on_agent_action(self, action: Any, **kwargs) -> None:
        """Called when agent takes an action."""
        logger.debug(f"Agent action: {action.tool if hasattr(action, 'tool') else 'unknown'}")
        
    def on_agent_finish(self, finish: Any, **kwargs) -> None:
        """Called when agent finishes."""
        logger.debug(f"Agent finished: {getattr(finish, 'return_values', {})}")


class GenOpsLangChainAdapter(BaseFrameworkProvider):
    """LangChain adapter with automatic governance telemetry."""
    
    def __init__(self, **kwargs):
        if not HAS_LANGCHAIN:
            raise ImportError(
                "LangChain package not found. Install with: pip install langchain"
            )
            
        super().__init__(**kwargs)
        
        # LangChain-specific request attributes
        self.REQUEST_ATTRIBUTES = {
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty',
            'presence_penalty', 'stop', 'model', 'verbose', 'streaming'
        }
        
        # Chain cost tracking
        self._chain_costs = {}
        self._active_operations = {}
        
        # RAG instrumentation
        self.rag_instrumentor = LangChainRAGInstrumentor(self)
        
    def get_framework_name(self) -> str:
        """Return the framework name."""
        return "langchain"
        
    def get_framework_type(self) -> str:
        """Return the framework type."""
        return self.FRAMEWORK_TYPE_ORCHESTRATION
        
    def get_framework_version(self) -> Optional[str]:
        """Return the installed LangChain version."""
        try:
            return langchain.__version__
        except AttributeError:
            return None
            
    def is_framework_available(self) -> bool:
        """Check if LangChain is available."""
        return HAS_LANGCHAIN
        
    def calculate_cost(self, operation_context: Dict) -> float:
        """
        Calculate cost for LangChain operations.
        
        For LangChain, we need to aggregate costs from all underlying LLM calls.
        """
        total_cost = 0.0
        
        # If this is a chain operation, sum up all LLM costs
        if operation_context.get("operation_type") == "chain":
            llm_costs = operation_context.get("llm_costs", [])
            total_cost = sum(llm_costs)
            
        # For direct LLM calls, calculate based on token usage
        elif operation_context.get("operation_type") == "llm":
            tokens_input = operation_context.get("tokens_input", 0)
            tokens_output = operation_context.get("tokens_output", 0)
            model = operation_context.get("model", "gpt-3.5-turbo")
            
            # Use simplified pricing - in production, this should be more sophisticated
            cost_per_1k_input = 0.001  # Default pricing
            cost_per_1k_output = 0.002
            
            total_cost = (tokens_input * cost_per_1k_input / 1000) + \
                        (tokens_output * cost_per_1k_output / 1000)
                        
        return total_cost
        
    def get_operation_mappings(self) -> Dict[str, str]:
        """Return mapping of LangChain operations to instrumentation methods."""
        return {
            'chain.run': 'instrument_chain_run',
            'chain.invoke': 'instrument_chain_invoke',
            'chain.batch': 'instrument_chain_batch',
            'agent.run': 'instrument_agent_run',
            'llm.predict': 'instrument_llm_predict',
            'retriever.get_relevant_documents': 'instrument_retriever',
            'rag.query': 'instrument_rag_query',
            'embeddings.embed': 'instrument_embeddings',
            'vectorstore.similarity_search': 'instrument_vector_search'
        }
        
    def _record_framework_metrics(self, span: Any, operation_type: str, context: Dict) -> None:
        """Record LangChain-specific metrics."""
        # Record chain-specific metrics
        if operation_type == "chain":
            chain_name = context.get("chain_name", "unknown")
            chain_steps = context.get("chain_steps", 0)
            
            span.set_attribute("genops.langchain.chain.name", chain_name)
            span.set_attribute("genops.langchain.chain.steps", chain_steps)
            
        # Record LLM-specific metrics
        elif operation_type == "llm":
            model = context.get("model", "unknown")
            prompt_length = context.get("prompt_length", 0)
            
            span.set_attribute("genops.langchain.llm.model", model)
            span.set_attribute("genops.langchain.llm.prompt_length", prompt_length)
            
        # Record agent-specific metrics
        elif operation_type == "agent":
            agent_type = context.get("agent_type", "unknown")
            tool_calls = context.get("tool_calls", 0)
            
            span.set_attribute("genops.langchain.agent.type", agent_type)
            span.set_attribute("genops.langchain.agent.tool_calls", tool_calls)
            
    def instrument_chain_run(self, chain: Any, **kwargs) -> Any:
        """Instrument chain.run() with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        chain_name = getattr(chain, '_chain_type', chain.__class__.__name__)
        operation_name = f"langchain.chain.run.{chain_name}"
        chain_id = str(uuid.uuid4())
        
        # Build trace attributes
        trace_attrs = self._build_trace_attributes(
            operation_name=operation_name,
            operation_type="chain",
            governance_attrs=governance_attrs,
            chain_name=chain_name,
            chain_type=chain.__class__.__name__,
            chain_id=chain_id
        )
        
        with self.telemetry.trace_operation(**trace_attrs) as span:
            # Use cost aggregation context manager
            with create_chain_cost_context(chain_id) as cost_context:
                try:
                    # Record request parameters
                    for param, value in request_attrs.items():
                        span.set_attribute(f"genops.langchain.request.{param}", value)
                        
                    # Add our callback handler to capture nested operations
                    callback_handler = GenOpsLangChainCallbackHandler(self, chain_id)
                    
                    # Modify callbacks to include our handler
                    if 'callbacks' in api_kwargs:
                        if api_kwargs['callbacks'] is None:
                            api_kwargs['callbacks'] = []
                        api_kwargs['callbacks'].append(callback_handler)
                    else:
                        api_kwargs['callbacks'] = [callback_handler]
                    
                    # Execute the chain
                    result = chain.run(**api_kwargs)
                    
                    # Get the final cost summary
                    cost_summary = cost_context.get_final_summary()
                    
                    if cost_summary:
                        # Record aggregated cost telemetry
                        self.telemetry.record_cost(
                            span=span,
                            cost=cost_summary.total_cost,
                            currency=cost_summary.currency,
                            provider="langchain_aggregated",
                            model=chain_name,
                            tokens_input=cost_summary.total_tokens_input,
                            tokens_output=cost_summary.total_tokens_output
                        )
                        
                        # Record detailed cost breakdown
                        cost_breakdown = cost_summary.to_dict()
                        for key, value in cost_breakdown.items():
                            if isinstance(value, (int, float, str)):
                                span.set_attribute(f"genops.langchain.cost.{key}", value)
                            elif isinstance(value, list):
                                span.set_attribute(f"genops.langchain.cost.{key}_count", len(value))
                                
                        logger.info(f"Chain {chain_name} completed: ${cost_summary.total_cost:.4f} "
                                  f"({cost_summary.total_tokens_input}+{cost_summary.total_tokens_output} tokens, "
                                  f"{len(cost_summary.unique_providers)} providers)")
                    
                    # Calculate and record additional metrics
                    operation_context = {
                        "operation_type": "chain",
                        "chain_name": chain_name,
                        "chain_id": chain_id,
                        "cost_summary": cost_summary,
                        "provider_count": len(cost_summary.unique_providers) if cost_summary else 0,
                        "model_count": len(cost_summary.unique_models) if cost_summary else 0
                    }
                    
                    self.record_operation_telemetry(span, "chain", operation_context)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"LangChain chain error: {e}")
                    raise
                
    def instrument_chain_invoke(self, chain: Any, **kwargs) -> Any:
        """Instrument chain.invoke() with governance tracking."""
        # Similar implementation to run() but for the invoke interface
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        chain_name = getattr(chain, '_chain_type', chain.__class__.__name__)
        operation_name = f"langchain.chain.invoke.{chain_name}"
        
        trace_attrs = self._build_trace_attributes(
            operation_name=operation_name,
            operation_type="chain",
            governance_attrs=governance_attrs,
            chain_name=chain_name
        )
        
        with self.telemetry.trace_operation(**trace_attrs) as span:
            try:
                # Add callback handler
                callback_handler = GenOpsLangChainCallbackHandler(self)
                
                if 'config' in api_kwargs and api_kwargs['config']:
                    callbacks = api_kwargs['config'].get('callbacks', [])
                    callbacks.append(callback_handler)
                    api_kwargs['config']['callbacks'] = callbacks
                else:
                    api_kwargs['config'] = {'callbacks': [callback_handler]}
                
                result = chain.invoke(**api_kwargs)
                
                operation_context = {
                    "operation_type": "chain",
                    "chain_name": chain_name
                }
                
                self.record_operation_telemetry(span, "chain", operation_context)
                
                return result
                
            except Exception as e:
                logger.error(f"LangChain chain invoke error: {e}")
                raise
                
    def instrument_rag_query(self, query: str, retriever: Any = None, **kwargs) -> Any:
        """
        Instrument a complete RAG query operation.
        
        Args:
            query: The query string
            retriever: The retriever to use (optional)
            **kwargs: Additional arguments including governance attributes
        """
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        operation_name = "langchain.rag.query"
        
        trace_attrs = self._build_trace_attributes(
            operation_name=operation_name,
            operation_type="rag_query",
            governance_attrs=governance_attrs,
            query_length=len(query)
        )
        
        with self.telemetry.trace_operation(**trace_attrs) as span:
            with self.rag_instrumentor.create_rag_context(query) as rag_context:
                try:
                    # Record query parameters
                    span.set_attribute("genops.langchain.rag.query", query)
                    span.set_attribute("genops.langchain.rag.query_length", len(query))
                    
                    # If retriever provided, instrument it
                    if retriever:
                        instrumented_retriever = self.rag_instrumentor.instrument_retriever(
                            retriever, rag_context.get_operation_id()
                        )
                        
                        # Perform retrieval
                        documents = instrumented_retriever.get_relevant_documents(query)
                        
                        # Record RAG metrics
                        summary = rag_context.get_summary()
                        if summary:
                            rag_metrics = summary.to_dict()
                            for key, value in rag_metrics.items():
                                if isinstance(value, (int, float, str)):
                                    span.set_attribute(f"genops.langchain.rag.{key}", value)
                                    
                        return documents
                    else:
                        logger.warning("No retriever provided for RAG instrumentation")
                        return []
                        
                except Exception as e:
                    logger.error(f"RAG query error: {e}")
                    raise
                    
    def instrument_retriever(self, retriever: Any, **kwargs) -> Any:
        """
        Instrument a retriever with governance tracking.
        
        Args:
            retriever: LangChain retriever instance
            **kwargs: Additional arguments including governance attributes
        """
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        # Create a unique operation ID for this retriever session
        operation_id = str(uuid.uuid4())
        
        # Instrument the retriever
        instrumented_retriever = self.rag_instrumentor.instrument_retriever(retriever, operation_id)
        
        logger.info(f"Retriever instrumented with operation ID: {operation_id}")
        return instrumented_retriever
        
    def instrument_embeddings(self, embeddings: Any, **kwargs) -> Any:
        """
        Instrument embeddings with governance tracking.
        
        Args:
            embeddings: LangChain embeddings instance
            **kwargs: Additional arguments including governance attributes
        """
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        # Create a unique operation ID for this embeddings session
        operation_id = str(uuid.uuid4())
        
        # Instrument the embeddings
        instrumented_embeddings = self.rag_instrumentor.instrument_embeddings(embeddings, operation_id)
        
        logger.info(f"Embeddings instrumented with operation ID: {operation_id}")
        return instrumented_embeddings
        
    def instrument_vector_search(self, vector_store: Any, query: str, **kwargs) -> Any:
        """
        Instrument vector store similarity search.
        
        Args:
            vector_store: LangChain vector store instance
            query: Search query
            **kwargs: Search parameters and governance attributes
        """
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        
        operation_name = "langchain.vectorstore.similarity_search"
        
        trace_attrs = self._build_trace_attributes(
            operation_name=operation_name,
            operation_type="vector_search",
            governance_attrs=governance_attrs,
            query_length=len(query),
            vector_store_type=type(vector_store).__name__
        )
        
        with self.telemetry.trace_operation(**trace_attrs) as span:
            try:
                # Record search parameters
                span.set_attribute("genops.langchain.vector.query", query)
                span.set_attribute("genops.langchain.vector.store_type", type(vector_store).__name__)
                
                # Extract search parameters
                k = api_kwargs.get('k', 4)
                search_type = api_kwargs.get('search_type', 'similarity')
                
                span.set_attribute("genops.langchain.vector.k", k)
                span.set_attribute("genops.langchain.vector.search_type", search_type)
                
                # Perform the search
                start_time = time.time()
                if hasattr(vector_store, 'similarity_search'):
                    results = vector_store.similarity_search(query, **api_kwargs)
                else:
                    logger.warning("Vector store does not support similarity_search")
                    results = []
                    
                search_time = time.time() - start_time
                
                # Record search metrics
                span.set_attribute("genops.langchain.vector.results_count", len(results))
                span.set_attribute("genops.langchain.vector.search_time", search_time)
                
                if results:
                    # Calculate average document length
                    avg_doc_length = sum(len(doc.page_content) for doc in results) / len(results)
                    span.set_attribute("genops.langchain.vector.avg_doc_length", avg_doc_length)
                    
                logger.debug(f"Vector search completed: {len(results)} results in {search_time:.3f}s")
                return results
                
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                raise
    
    def _apply_instrumentation(self, **config) -> None:
        """Apply LangChain instrumentation."""
        # This will be implemented with monkey patching
        # For now, this is a placeholder
        logger.info("LangChain instrumentation applied (manual instrumentation required)")
        
    def _remove_instrumentation(self) -> None:
        """Remove LangChain instrumentation."""
        logger.info("LangChain instrumentation removed")


def instrument_langchain(**kwargs) -> GenOpsLangChainAdapter:
    """
    Instrument LangChain with GenOps governance telemetry.
    
    Returns:
        GenOpsLangChainAdapter: Instrumented LangChain adapter
        
    Example:
        import genops
        from langchain.chains import LLMChain
        
        # Create adapter
        adapter = genops.providers.langchain.instrument_langchain()
        
        # Wrap chain operations
        chain = LLMChain(...)
        result = adapter.instrument_chain_run(
            chain,
            input_variables={"query": "What is AI?"},
            team="ai-research",
            project="chatbot"
        )
    """
    return GenOpsLangChainAdapter(**kwargs)


# Monkey patching functions (placeholder for now)
def patch_langchain(auto_track: bool = True) -> None:
    """
    Apply monkey patches to LangChain for automatic instrumentation.
    
    Args:
        auto_track: Whether to automatically track all LangChain operations
    """
    if not HAS_LANGCHAIN:
        logger.warning("LangChain not available for patching")
        return
        
    # TODO: Implement monkey patching for automatic instrumentation
    logger.info("LangChain monkey patching not yet implemented")
    logger.info("Use manual instrumentation with GenOpsLangChainAdapter for now")
    

def unpatch_langchain() -> None:
    """Remove LangChain monkey patches."""
    logger.info("LangChain unpatching not yet implemented")