"""Tests for LangChain adapter functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid
import time

from genops.providers.langchain.adapter import (
    GenOpsLangChainAdapter,
    GenOpsLangChainCallbackHandler,
    instrument_langchain
)


class TestGenOpsLangChainCallbackHandler:
    """Test GenOpsLangChainCallbackHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.handler = GenOpsLangChainCallbackHandler(self.mock_adapter)
        
    def test_callback_handler_initialization(self):
        """Test callback handler initializes correctly."""
        assert self.handler.telemetry_adapter is self.mock_adapter
        assert isinstance(self.handler.chain_id, str)
        assert len(self.handler.chain_context) == 0
        assert len(self.handler.operation_stack) == 0
        
    def test_callback_handler_with_chain_id(self):
        """Test callback handler with specific chain ID."""
        chain_id = "test_chain_123"
        handler = GenOpsLangChainCallbackHandler(self.mock_adapter, chain_id)
        
        assert handler.chain_id == chain_id
        
    def test_on_chain_start(self):
        """Test chain start callback."""
        serialized = {"name": "TestChain"}
        inputs = {"query": "test query"}
        
        self.handler.on_chain_start(serialized, inputs)
        
        assert len(self.handler.operation_stack) == 1
        operation = self.handler.operation_stack[0]
        assert operation["type"] == "chain"
        assert operation["name"] == "TestChain"
        assert operation["inputs"] == inputs
        
    def test_on_chain_end(self):
        """Test chain end callback."""
        # First start a chain
        self.handler.on_chain_start({"name": "TestChain"}, {})
        
        outputs = {"result": "test result"}
        self.handler.on_chain_end(outputs)
        
        assert len(self.handler.operation_stack) == 0  # Should be popped
        
    def test_on_chain_error(self):
        """Test chain error callback."""
        # First start a chain
        self.handler.on_chain_start({"name": "TestChain"}, {})
        
        error = Exception("Test error")
        self.handler.on_chain_error(error)
        
        assert len(self.handler.operation_stack) == 0  # Should be popped
        
    def test_on_llm_start(self):
        """Test LLM start callback."""
        serialized = {"name": "TestLLM"}
        prompts = ["Tell me about AI", "What is machine learning?"]
        
        self.handler.on_llm_start(serialized, prompts)
        
        assert len(self.handler.operation_stack) == 1
        operation = self.handler.operation_stack[0]
        assert operation["type"] == "llm"
        assert operation["name"] == "TestLLM"
        assert operation["prompts"] == prompts
        assert operation["prompt_tokens"] > 0  # Should estimate tokens
        
    def test_on_llm_end_with_token_usage(self):
        """Test LLM end callback with token usage."""
        # First start an LLM
        self.handler.on_llm_start({"name": "gpt-4"}, ["test prompt"])
        
        # Mock response with token usage
        mock_response = Mock()
        mock_response.llm_output = {
            'token_usage': {
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30
            },
            'model_name': 'gpt-4'
        }
        
        with patch.object(self.handler, '_detect_provider_from_response', return_value='openai'):
            self.handler.on_llm_end(mock_response)
            
        assert len(self.handler.operation_stack) == 0  # Should be popped
        
    def test_detect_provider_from_response_openai(self):
        """Test provider detection for OpenAI responses."""
        mock_response = Mock()
        mock_response.llm_output = {'model_name': 'gpt-4'}
        
        provider = self.handler._detect_provider_from_response(mock_response)
        assert provider == 'openai'
        
    def test_detect_provider_from_response_anthropic(self):
        """Test provider detection for Anthropic responses."""
        mock_response = Mock()
        mock_response.llm_output = {'model_name': 'claude-3-sonnet'}
        
        provider = self.handler._detect_provider_from_response(mock_response)
        assert provider == 'anthropic'
        
    def test_detect_provider_from_response_unknown(self):
        """Test provider detection for unknown responses."""
        mock_response = Mock()
        mock_response.llm_output = {'model_name': 'unknown-model'}
        
        provider = self.handler._detect_provider_from_response(mock_response)
        assert provider == 'unknown'
        
    def test_on_agent_action(self):
        """Test agent action callback."""
        mock_action = Mock()
        mock_action.tool = "calculator"
        
        # Should not raise exception
        self.handler.on_agent_action(mock_action)
        
    def test_on_agent_finish(self):
        """Test agent finish callback."""
        mock_finish = Mock()
        mock_finish.return_values = {"result": "42"}
        
        # Should not raise exception
        self.handler.on_agent_finish(mock_finish)


class TestGenOpsLangChainAdapter:
    """Test GenOpsLangChainAdapter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True):
            self.adapter = GenOpsLangChainAdapter()
            
    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        assert self.adapter.get_framework_name() == "langchain"
        assert self.adapter.get_framework_type() == "orchestration"
        assert isinstance(self.adapter.REQUEST_ATTRIBUTES, set)
        assert 'temperature' in self.adapter.REQUEST_ATTRIBUTES
        
    def test_framework_properties(self):
        """Test framework property methods."""
        assert self.adapter.get_framework_name() == "langchain"
        assert self.adapter.get_framework_type() == "orchestration"
        
        # Framework availability depends on import success
        with patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True):
            assert self.adapter.is_framework_available() is True
            
        with patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', False):
            adapter = GenOpsLangChainAdapter.__new__(GenOpsLangChainAdapter)  # Skip __init__
            assert adapter.is_framework_available() is False
            
    @patch('genops.providers.langchain.adapter.langchain')
    def test_get_framework_version(self, mock_langchain):
        """Test getting framework version."""
        mock_langchain.__version__ = "0.1.0"
        
        version = self.adapter.get_framework_version()
        assert version == "0.1.0"
        
    def test_get_operation_mappings(self):
        """Test getting operation mappings."""
        mappings = self.adapter.get_operation_mappings()
        
        assert isinstance(mappings, dict)
        assert 'chain.run' in mappings
        assert 'chain.invoke' in mappings
        assert 'rag.query' in mappings
        assert 'retriever.get_relevant_documents' in mappings
        
    def test_extract_attributes(self):
        """Test extracting governance and request attributes."""
        kwargs = {
            'team': 'ai-research',
            'project': 'chatbot',
            'temperature': 0.7,
            'max_tokens': 100,
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'Hello'}]
        }
        
        governance_attrs, request_attrs, api_kwargs = self.adapter._extract_attributes(kwargs)
        
        assert governance_attrs['team'] == 'ai-research'
        assert governance_attrs['project'] == 'chatbot'
        assert request_attrs['temperature'] == 0.7
        assert request_attrs['max_tokens'] == 100
        assert api_kwargs['model'] == 'gpt-4'
        assert api_kwargs['messages'] == [{'role': 'user', 'content': 'Hello'}]
        assert 'team' not in api_kwargs
        assert 'project' not in api_kwargs
        
    def test_calculate_cost_chain_operation(self):
        """Test calculating cost for chain operations."""
        context = {
            "operation_type": "chain",
            "llm_costs": [0.01, 0.02, 0.005]
        }
        
        cost = self.adapter.calculate_cost(context)
        assert abs(cost - 0.035) < 0.0001  # Use approximate comparison for float precision
        
    def test_calculate_cost_llm_operation(self):
        """Test calculating cost for LLM operations.""" 
        context = {
            "operation_type": "llm",
            "tokens_input": 1000,
            "tokens_output": 500,
            "model": "gpt-4"
        }
        
        cost = self.adapter.calculate_cost(context)
        assert cost > 0
        assert isinstance(cost, float)
        
    @patch('genops.providers.langchain.adapter.create_chain_cost_context')
    @patch('genops.providers.langchain.adapter.uuid.uuid4')
    def test_instrument_chain_run(self, mock_uuid, mock_create_context):
        """Test instrumenting chain.run() method."""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test_chain_id")
        
        # Mock the cost context
        mock_context = Mock()
        mock_summary = Mock()
        mock_summary.total_cost = 0.025
        mock_summary.currency = "USD"
        mock_summary.total_tokens_input = 100
        mock_summary.total_tokens_output = 50
        mock_summary.unique_providers = {"openai"}  # Use set, not list
        mock_summary.unique_models = {"gpt-4"}  # Use set, not list
        mock_summary.to_dict.return_value = {"total_cost": 0.025}
        mock_context.get_final_summary.return_value = mock_summary
        mock_context.__enter__ = Mock(return_value=mock_context)
        mock_context.__exit__ = Mock(return_value=None)
        mock_create_context.return_value = mock_context
        
        # Mock the chain
        mock_chain = Mock()
        mock_chain._chain_type = "TestChain"
        mock_chain.__class__.__name__ = "TestChain"
        mock_chain.run.return_value = "Test result"
        
        # Mock telemetry
        mock_span = Mock()
        self.adapter.telemetry.trace_operation = Mock()
        self.adapter.telemetry.trace_operation.return_value.__enter__ = Mock(return_value=mock_span)
        self.adapter.telemetry.trace_operation.return_value.__exit__ = Mock(return_value=None)
        self.adapter.telemetry.record_cost = Mock()
        
        kwargs = {
            'team': 'ai-team',
            'temperature': 0.7,
            'input': 'test query'
        }
        
        result = self.adapter.instrument_chain_run(mock_chain, **kwargs)
        
        assert result == "Test result"
        # Verify telemetry was recorded
        self.adapter.telemetry.record_cost.assert_called_once()
        
    def test_instrument_rag_query_no_retriever(self):
        """Test RAG query instrumentation without retriever."""
        mock_span = Mock()
        self.adapter.telemetry.trace_operation = Mock()
        self.adapter.telemetry.trace_operation.return_value.__enter__ = Mock(return_value=mock_span)
        self.adapter.telemetry.trace_operation.return_value.__exit__ = Mock(return_value=None)
        
        # Mock RAG context
        mock_rag_context = Mock()
        mock_rag_context.get_operation_id.return_value = "rag_op_id"
        mock_rag_context.__enter__ = Mock(return_value=mock_rag_context)
        mock_rag_context.__exit__ = Mock(return_value=None)
        self.adapter.rag_instrumentor.create_rag_context = Mock(return_value=mock_rag_context)
        
        result = self.adapter.instrument_rag_query("What is AI?")
        
        assert result == []  # Should return empty list when no retriever
        
    def test_instrument_retriever(self):
        """Test retriever instrumentation."""
        mock_retriever = Mock()
        
        result = self.adapter.instrument_retriever(mock_retriever, team="ai-team")
        
        assert result is not None
        # Should return the instrumented retriever
        
    def test_instrument_embeddings(self):
        """Test embeddings instrumentation."""
        mock_embeddings = Mock()
        
        result = self.adapter.instrument_embeddings(mock_embeddings, team="ai-team")
        
        assert result is not None
        # Should return the instrumented embeddings
        
    @patch('time.time')
    def test_instrument_vector_search(self, mock_time):
        """Test vector store search instrumentation."""
        mock_time.side_effect = [1000.0, 1001.5]  # 1.5 second search time
        
        mock_vector_store = Mock()
        mock_vector_store.__class__.__name__ = "MockVectorStore"
        mock_documents = [Mock(page_content="Document 1"), Mock(page_content="Document 2")]
        mock_vector_store.similarity_search.return_value = mock_documents
        
        mock_span = Mock()
        self.adapter.telemetry.trace_operation = Mock()
        self.adapter.telemetry.trace_operation.return_value.__enter__ = Mock(return_value=mock_span)
        self.adapter.telemetry.trace_operation.return_value.__exit__ = Mock(return_value=None)
        
        result = self.adapter.instrument_vector_search(
            mock_vector_store, 
            "test query",
            k=5,
            team="ai-team"
        )
        
        assert result == mock_documents
        # Verify span attributes were set
        mock_span.set_attribute.assert_called()
        
    def test_record_framework_metrics_chain(self):
        """Test recording chain-specific metrics."""
        mock_span = Mock()
        context = {
            "chain_name": "TestChain",
            "chain_steps": 3
        }
        
        self.adapter._record_framework_metrics(mock_span, "chain", context)
        
        # Verify attributes were set
        mock_span.set_attribute.assert_any_call("genops.langchain.chain.name", "TestChain")
        mock_span.set_attribute.assert_any_call("genops.langchain.chain.steps", 3)
        
    def test_record_framework_metrics_llm(self):
        """Test recording LLM-specific metrics."""
        mock_span = Mock()
        context = {
            "model": "gpt-4",
            "prompt_length": 100
        }
        
        self.adapter._record_framework_metrics(mock_span, "llm", context)
        
        mock_span.set_attribute.assert_any_call("genops.langchain.llm.model", "gpt-4")
        mock_span.set_attribute.assert_any_call("genops.langchain.llm.prompt_length", 100)
        
    def test_record_framework_metrics_agent(self):
        """Test recording agent-specific metrics."""
        mock_span = Mock()
        context = {
            "agent_type": "ReActAgent", 
            "tool_calls": 2
        }
        
        self.adapter._record_framework_metrics(mock_span, "agent", context)
        
        mock_span.set_attribute.assert_any_call("genops.langchain.agent.type", "ReActAgent")
        mock_span.set_attribute.assert_any_call("genops.langchain.agent.tool_calls", 2)


class TestInstrumentLangChainFunction:
    """Test instrument_langchain function."""
    
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_instrument_langchain_success(self):
        """Test successful instrumentation."""
        result = instrument_langchain()
        
        assert isinstance(result, GenOpsLangChainAdapter)
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', False)  
    def test_instrument_langchain_missing_dependency(self):
        """Test instrumentation with missing LangChain."""
        with pytest.raises(ImportError, match="LangChain package not found"):
            instrument_langchain()
            
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_instrument_langchain_with_kwargs(self):
        """Test instrumentation with additional kwargs."""
        result = instrument_langchain(custom_param="test_value")
        
        assert isinstance(result, GenOpsLangChainAdapter)


class TestMonkeyPatching:
    """Test monkey patching functions."""
    
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_patch_langchain_not_implemented(self):
        """Test that monkey patching logs not implemented message."""
        from genops.providers.langchain.adapter import patch_langchain
        
        with patch('genops.providers.langchain.adapter.logger') as mock_logger:
            patch_langchain()
            
            mock_logger.info.assert_called()
            
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', False)
    def test_patch_langchain_missing_dependency(self):
        """Test monkey patching with missing LangChain."""
        from genops.providers.langchain.adapter import patch_langchain
        
        with patch('genops.providers.langchain.adapter.logger') as mock_logger:
            patch_langchain()
            
            mock_logger.warning.assert_called_with("LangChain not available for patching")
            
    def test_unpatch_langchain(self):
        """Test unpatching LangChain."""
        from genops.providers.langchain.adapter import unpatch_langchain
        
        with patch('genops.providers.langchain.adapter.logger') as mock_logger:
            unpatch_langchain()
            
            mock_logger.info.assert_called()


@pytest.fixture
def mock_langchain_chain():
    """Mock LangChain chain for testing."""
    chain = Mock()
    chain._chain_type = "TestChain"
    chain.__class__.__name__ = "TestChain"
    chain.run = Mock(return_value="Test chain result")
    chain.invoke = Mock(return_value="Test chain result")
    return chain


@pytest.fixture  
def mock_langchain_retriever():
    """Mock LangChain retriever for testing."""
    retriever = Mock()
    retriever.get_relevant_documents = Mock(return_value=[
        Mock(page_content="Document 1", metadata={"score": 0.8}),
        Mock(page_content="Document 2", metadata={"score": 0.7})
    ])
    return retriever


class TestLangChainAdapterIntegration:
    """Integration tests for LangChain adapter."""
    
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_end_to_end_chain_instrumentation(self, mock_langchain_chain):
        """Test end-to-end chain instrumentation."""
        adapter = GenOpsLangChainAdapter()
        
        # Mock telemetry components
        with patch.object(adapter.telemetry, 'trace_operation') as mock_trace:
            mock_span = Mock()
            mock_trace.return_value.__enter__ = Mock(return_value=mock_span)
            mock_trace.return_value.__exit__ = Mock(return_value=None)
            
            with patch('genops.providers.langchain.adapter.create_chain_cost_context') as mock_context:
                mock_cost_context = Mock()
                mock_cost_context.__enter__ = Mock(return_value=mock_cost_context)
                mock_cost_context.__exit__ = Mock(return_value=None)
                mock_cost_context.get_final_summary.return_value = None
                mock_context.return_value = mock_cost_context
                
                result = adapter.instrument_chain_run(
                    mock_langchain_chain,
                    input="Test query",
                    team="ai-research",
                    project="chatbot",
                    temperature=0.7
                )
                
        assert result == "Test chain result"
        mock_langchain_chain.run.assert_called_once()
        
    @patch('genops.providers.langchain.adapter.HAS_LANGCHAIN', True)
    def test_rag_query_instrumentation(self, mock_langchain_retriever):
        """Test RAG query instrumentation."""
        adapter = GenOpsLangChainAdapter()
        
        with patch.object(adapter.telemetry, 'trace_operation') as mock_trace:
            mock_span = Mock()
            mock_trace.return_value.__enter__ = Mock(return_value=mock_span)
            mock_trace.return_value.__exit__ = Mock(return_value=None)
            
            with patch.object(adapter.rag_instrumentor, 'create_rag_context') as mock_rag_context:
                mock_context = Mock()
                mock_context.get_operation_id.return_value = "rag_op_123"
                mock_context.get_summary.return_value = None
                mock_context.__enter__ = Mock(return_value=mock_context)
                mock_context.__exit__ = Mock(return_value=None)
                mock_rag_context.return_value = mock_context
                
                with patch.object(adapter.rag_instrumentor, 'instrument_retriever') as mock_instrument:
                    mock_instrument.return_value = mock_langchain_retriever
                    
                    result = adapter.instrument_rag_query(
                        "What is artificial intelligence?",
                        retriever=mock_langchain_retriever,
                        team="research"
                    )
                    
        assert len(result) == 2  # Two mock documents
        mock_langchain_retriever.get_relevant_documents.assert_called_once_with("What is artificial intelligence?")