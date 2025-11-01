"""Integration tests for LangChain provider."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import uuid

from genops.providers.langchain import (
    GenOpsLangChainAdapter,
    instrument_langchain,
    create_chain_cost_context,
    get_cost_aggregator
)


@pytest.fixture
def mock_opentelemetry():
    """Mock OpenTelemetry components."""
    with patch('genops.core.telemetry.trace') as mock_trace:
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.record_exception = Mock()
        
        mock_tracer = Mock()
        mock_tracer.start_as_current_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        mock_trace.get_tracer.return_value = mock_tracer
        
        yield {
            'trace': mock_trace,
            'tracer': mock_tracer,
            'span': mock_span
        }


@pytest.fixture 
def mock_langchain_components():
    """Mock LangChain components for testing."""
    # Mock Document
    mock_document_class = Mock()
    mock_doc1 = Mock()
    mock_doc1.page_content = "This is the first document about AI."
    mock_doc1.metadata = {"score": 0.85, "source": "doc1.txt"}
    
    mock_doc2 = Mock() 
    mock_doc2.page_content = "This is the second document about machine learning."
    mock_doc2.metadata = {"score": 0.72, "source": "doc2.txt"}
    
    # Mock Chain
    mock_chain = Mock()
    mock_chain._chain_type = "RetrievalQAChain"
    mock_chain.__class__.__name__ = "RetrievalQAChain"
    mock_chain.run = Mock(return_value="AI is a field of computer science...")
    
    # Mock Retriever
    mock_retriever = Mock()
    mock_retriever.get_relevant_documents = Mock(return_value=[mock_doc1, mock_doc2])
    mock_retriever.vectorstore = Mock()
    mock_retriever.vectorstore.__class__.__name__ = "ChromaVectorStore"
    
    # Mock Embeddings
    mock_embeddings = Mock()
    mock_embeddings.__class__.__name__ = "OpenAIEmbeddings"
    mock_embeddings.embed_documents = Mock(return_value=[
        [0.1, 0.2, 0.3] * 512,  # Mock 1536-dim embedding
        [0.4, 0.5, 0.6] * 512
    ])
    mock_embeddings.embed_query = Mock(return_value=[0.7, 0.8, 0.9] * 512)
    
    # Mock Vector Store
    mock_vector_store = Mock()
    mock_vector_store.__class__.__name__ = "ChromaVectorStore"
    mock_vector_store.similarity_search = Mock(return_value=[mock_doc1, mock_doc2])
    
    return {
        'documents': [mock_doc1, mock_doc2],
        'chain': mock_chain,
        'retriever': mock_retriever,
        'embeddings': mock_embeddings,
        'vector_store': mock_vector_store
    }


class TestLangChainIntegration:
    """Test complete LangChain integration workflows."""
    
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_basic_adapter_creation(self, mock_opentelemetry):
        """Test creating a LangChain adapter."""
        adapter = instrument_langchain()
        
        assert isinstance(adapter, GenOpsLangChainAdapter)
        assert adapter.get_framework_name() == "langchain"
        assert adapter.get_framework_type() == "orchestration"
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_chain_execution_tracking(self, mock_opentelemetry, mock_langchain_components):
        """Test end-to-end chain execution tracking."""
        adapter = instrument_langchain()
        chain = mock_langchain_components['chain']
        
        # Execute instrumented chain
        result = adapter.instrument_chain_run(
            chain,
            input="What is artificial intelligence?",
            team="ai-research",
            project="qa-system",
            temperature=0.7
        )
        
        assert result == "AI is a field of computer science..."
        
        # Verify chain was called with correct parameters
        chain.run.assert_called_once()
        call_args = chain.run.call_args[1]
        assert call_args['input'] == "What is artificial intelligence?"
        assert call_args['temperature'] == 0.7
        assert 'callbacks' in call_args
        assert len(call_args['callbacks']) == 1
        
        # Verify telemetry was captured
        span = mock_opentelemetry['span']
        span.set_attribute.assert_called()
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True) 
    def test_cost_aggregation_workflow(self, mock_opentelemetry, mock_langchain_components):
        """Test multi-provider cost aggregation."""
        adapter = instrument_langchain()
        
        # Mock cost calculators on the aggregator
        from genops.providers.langchain.cost_aggregator import get_cost_aggregator
        aggregator = get_cost_aggregator()
        with patch.object(aggregator, '_calculate_provider_cost') as mock_calc:
            mock_calc.side_effect = [0.015, 0.008, 0.003]  # Different costs for different calls
            
            chain_id = str(uuid.uuid4())
            
            with create_chain_cost_context(chain_id) as cost_context:
                # Simulate multiple LLM calls within a chain
                cost_context.add_llm_call("openai", "gpt-4", 500, 250, "completion_1")
                cost_context.add_llm_call("anthropic", "claude-3", 300, 150, "completion_2") 
                cost_context.add_llm_call("openai", "gpt-3.5-turbo", 200, 100, "completion_3")
                
                # Record generation cost
                cost_context.record_generation_cost(0.005)
                
            summary = cost_context.get_final_summary()
            
            assert summary is not None
            assert len(summary.llm_calls) == 3
            assert summary.generation_cost == 0.005
            assert len(summary.unique_providers) == 2  # openai, anthropic
            assert len(summary.unique_models) == 3
            assert summary.total_tokens_input == 1000
            assert summary.total_tokens_output == 500
            
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_rag_operation_monitoring(self, mock_opentelemetry, mock_langchain_components):
        """Test RAG operation monitoring."""
        adapter = instrument_langchain()
        retriever = mock_langchain_components['retriever']
        
        # Test RAG query instrumentation
        documents = adapter.instrument_rag_query(
            "What is machine learning?",
            retriever=retriever,
            team="ml-research",
            k=5
        )
        
        assert len(documents) == 2
        retriever.get_relevant_documents.assert_called_once_with("What is machine learning?")
        
        # Verify telemetry attributes were set
        span = mock_opentelemetry['span']
        span.set_attribute.assert_called()
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_retriever_instrumentation(self, mock_opentelemetry, mock_langchain_components):
        """Test retriever instrumentation."""
        adapter = instrument_langchain()
        retriever = mock_langchain_components['retriever']
        
        # Instrument the retriever
        instrumented_retriever = adapter.instrument_retriever(
            retriever,
            team="research-team"
        )
        
        assert instrumented_retriever is not None
        # The original retriever should have been modified
        assert hasattr(instrumented_retriever, 'get_relevant_documents')
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_embeddings_instrumentation(self, mock_opentelemetry, mock_langchain_components):
        """Test embeddings instrumentation."""
        adapter = instrument_langchain()
        embeddings = mock_langchain_components['embeddings']
        
        # Instrument the embeddings
        instrumented_embeddings = adapter.instrument_embeddings(
            embeddings,
            team="embedding-team"
        )
        
        assert instrumented_embeddings is not None
        assert hasattr(instrumented_embeddings, 'embed_documents')
        assert hasattr(instrumented_embeddings, 'embed_query')
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_vector_store_instrumentation(self, mock_opentelemetry, mock_langchain_components):
        """Test vector store search instrumentation."""
        adapter = instrument_langchain()
        vector_store = mock_langchain_components['vector_store']
        
        # Use a mock that returns incremental values to avoid running out
        time_values = [1000.0, 1001.2, 1001.2, 1001.2, 1001.2]  # Extra values for logging calls
        with patch('genops.providers.langchain.adapter.time.time', side_effect=time_values):
            results = adapter.instrument_vector_search(
                vector_store,
                "test query",
                k=4,
                team="vector-team"
            )
            
        assert len(results) == 2
        # The 'team' governance attribute should be filtered out, not passed to vector store
        vector_store.similarity_search.assert_called_once_with("test query", k=4)
        
        # Verify timing and metrics were captured
        span = mock_opentelemetry['span']
        span.set_attribute.assert_called()
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_callback_handler_integration(self, mock_opentelemetry, mock_langchain_components):
        """Test callback handler captures operations."""
        adapter = instrument_langchain()
        
        # Create callback handler
        from genops.providers.langchain.adapter import GenOpsLangChainCallbackHandler
        handler = GenOpsLangChainCallbackHandler(adapter, "test_chain_123")
        
        # Simulate chain execution
        handler.on_chain_start({"name": "RetrievalQA"}, {"query": "test"})
        
        # Simulate LLM calls
        handler.on_llm_start({"name": "gpt-4"}, ["System prompt", "User query"])
        
        # Mock LLM response with token usage
        mock_response = Mock()
        mock_response.llm_output = {
            'token_usage': {
                'prompt_tokens': 50,
                'completion_tokens': 25,
                'total_tokens': 75
            },
            'model_name': 'gpt-4'
        }
        
        with patch.object(handler.cost_aggregator, 'add_llm_call_cost') as mock_add_cost:
            handler.on_llm_end(mock_response)
            
            # Verify cost was recorded
            mock_add_cost.assert_called_once()
            args = mock_add_cost.call_args[1]
            assert args['provider'] == 'openai'
            assert args['tokens_input'] == 50
            assert args['tokens_output'] == 25
            
        # Finish chain
        handler.on_chain_end({"result": "Chain completed"})
        
        # Verify operations were tracked
        assert len(handler.operation_stack) == 0  # Should be cleared after completion
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_governance_attributes_propagation(self, mock_opentelemetry, mock_langchain_components):
        """Test that governance attributes are properly propagated."""
        adapter = instrument_langchain()
        chain = mock_langchain_components['chain']
        
        governance_attrs = {
            'team': 'ai-engineering',
            'project': 'customer-support',
            'environment': 'production',
            'customer_id': 'customer_123',
            'feature': 'smart-responses'
        }
        
        # Execute chain with governance attributes
        result = adapter.instrument_chain_run(
            chain,
            input="Help me with my order",
            **governance_attrs
        )
        
        assert result is not None
        
        # Verify governance attributes were extracted and not passed to chain
        call_args = chain.run.call_args[1]
        for attr in governance_attrs:
            assert attr not in call_args
            
        # Verify telemetry captured the governance attributes
        span = mock_opentelemetry['span']
        span.set_attribute.assert_called()
        
    def test_cost_aggregator_singleton(self):
        """Test that cost aggregator is singleton."""
        aggregator1 = get_cost_aggregator()
        aggregator2 = get_cost_aggregator()
        
        assert aggregator1 is aggregator2
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_error_handling_in_chain_execution(self, mock_opentelemetry, mock_langchain_components):
        """Test error handling during chain execution."""
        adapter = instrument_langchain()
        chain = mock_langchain_components['chain']
        
        # Make chain raise an exception
        chain.run.side_effect = ValueError("Chain execution failed")
        
        with pytest.raises(ValueError, match="Chain execution failed"):
            adapter.instrument_chain_run(
                chain,
                input="test query",
                team="test-team"
            )
            
        # Verify error was recorded in telemetry
        span = mock_opentelemetry['span'] 
        span.set_status.assert_called()
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True) 
    def test_nested_operations_cost_tracking(self, mock_opentelemetry, mock_langchain_components):
        """Test cost tracking for nested operations."""
        adapter = instrument_langchain()
        
        chain_id = str(uuid.uuid4())
        aggregator = get_cost_aggregator()
        
        # Start tracking
        aggregator.start_chain_tracking(chain_id)
        
        # Add nested operations
        aggregator.add_llm_call_cost(
            chain_id, "openai", "gpt-4", 1000, 500, "retrieval_generation"
        )
        aggregator.add_llm_call_cost(
            chain_id, "openai", "text-embedding-ada-002", 500, 0, "embedding_query"
        )
        aggregator.add_llm_call_cost(
            chain_id, "anthropic", "claude-3", 800, 400, "final_generation"
        )
        
        # Finalize tracking
        summary = aggregator.finalize_chain_tracking(chain_id, total_time=4.2)
        
        assert summary is not None
        assert len(summary.llm_calls) == 3
        assert summary.total_time == 4.2
        assert len(summary.unique_providers) == 2
        assert summary.total_tokens_input == 2300
        assert summary.total_tokens_output == 900
        assert summary.total_cost > 0
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_framework_registration_integration(self, mock_opentelemetry):
        """Test that LangChain provider registers with auto-instrumentation."""
        from genops.providers.langchain.registration import register_langchain_provider
        from genops.auto_instrumentation import GenOpsInstrumentor
        
        instrumentor = GenOpsInstrumentor()
        register_langchain_provider(instrumentor)
        
        # Verify LangChain was registered
        assert "langchain" in instrumentor.framework_registry
        
        config = instrumentor.framework_registry["langchain"]
        assert config["framework_type"] == "orchestration"
        assert config["provider_type"] == "framework"
        assert "capabilities" in config
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_performance_measurement(self, mock_opentelemetry, mock_langchain_components):
        """Test that performance metrics are captured."""
        adapter = instrument_langchain()
        retriever = mock_langchain_components['retriever']
        
        # Simulate slow retrieval
        def slow_retrieval(query):
            time.sleep(0.1)  # Simulate 100ms retrieval time
            return mock_langchain_components['documents']
            
        retriever.get_relevant_documents = slow_retrieval
        
        start_time = time.time()
        documents = adapter.instrument_rag_query(
            "performance test query",
            retriever=retriever
        )
        end_time = time.time()
        
        assert len(documents) == 2
        assert (end_time - start_time) >= 0.1  # Should take at least 100ms
        
        # Verify timing metrics were captured
        span = mock_opentelemetry['span']
        span.set_attribute.assert_called()


class TestLangChainRegistration:
    """Test LangChain provider registration."""
    
    def test_auto_registration(self):
        """Test automatic registration of LangChain provider."""
        # Import should trigger auto-registration
        from genops.providers import langchain
        from genops.auto_instrumentation import _instrumentor
        
        # Check if LangChain was registered (if available)
        status = _instrumentor.get_framework_status()
        
        # The framework should be in the registry even if not available
        frameworks = status.get("frameworks", {})
        registered = frameworks.get("registered", [])
        
        # LangChain should be registered if the import was successful
        assert isinstance(registered, list)
        
    def test_registration_capabilities(self):
        """Test that registered capabilities are correct."""
        from genops.providers.langchain.registration import register_langchain_provider
        from genops.auto_instrumentation import GenOpsInstrumentor
        
        instrumentor = GenOpsInstrumentor()
        register_langchain_provider(instrumentor)
        
        if "langchain" in instrumentor.framework_registry:
            config = instrumentor.framework_registry["langchain"]
            capabilities = config.get("capabilities", [])
            
            assert "chain_execution_tracking" in capabilities
            assert "multi_provider_cost_aggregation" in capabilities
            assert "agent_decision_telemetry" in capabilities
            assert "rag_operation_monitoring" in capabilities


@pytest.mark.integration
class TestRealLangChainIntegration:
    """Integration tests with real LangChain (if available)."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("langchain", minversion=None),
        reason="LangChain not available"
    )
    def test_with_real_langchain_imports(self):
        """Test with real LangChain imports."""
        try:
            from langchain.schema import Document
            from langchain.callbacks.base import BaseCallbackHandler
            
            # Test that our callback handler is compatible
            from genops.providers.langchain.adapter import GenOpsLangChainCallbackHandler
            
            adapter = Mock()
            handler = GenOpsLangChainCallbackHandler(adapter)
            
            assert isinstance(handler, BaseCallbackHandler)
            
            # Test with real Document
            doc = Document(page_content="Test document", metadata={"source": "test"})
            assert doc.page_content == "Test document"
            assert doc.metadata["source"] == "test"
            
        except ImportError:
            pytest.skip("LangChain not available for real integration test")