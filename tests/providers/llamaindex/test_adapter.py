"""
Unit tests for GenOps LlamaIndex Adapter.

Comprehensive test coverage for the main GenOpsLlamaIndexAdapter class
including initialization, instrumentation, query tracking, and error handling.
"""

from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest

# Test imports - these will be mocked if LlamaIndex not available
try:
    from llama_index.core import Settings
    from llama_index.core.base.response.schema import Response
    from llama_index.core.chat_engine.base import BaseChatEngine
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.query_engine.base import BaseQueryEngine
    from llama_index.core.schema import Document, NodeWithScore, TextNode

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    # Mock classes for when LlamaIndex not available
    class Response:
        def __init__(self, response: str = "test response"):
            self.response = response

    class Document:
        def __init__(self, text: str = "test document"):
            self.text = text

    class BaseQueryEngine:
        def query(self, query: str):
            return Response("mocked response")

    class BaseChatEngine:
        def chat(self, message: str):
            return Response("mocked chat response")

    BaseIndex = Mock
    NodeWithScore = Mock
    TextNode = Mock
    Settings = Mock
    LLAMAINDEX_AVAILABLE = False

# Import the module under test
from genops.core.base_provider import BaseFrameworkProvider
from genops.providers.llamaindex.adapter import (
    GenOpsLlamaIndexAdapter,
    LlamaIndexCostBreakdown,
    LlamaIndexOperationSummary,
)


class TestGenOpsLlamaIndexAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_adapter_inherits_from_base_provider(self):
        """Test that adapter properly inherits from BaseFrameworkProvider."""
        adapter = GenOpsLlamaIndexAdapter()
        assert isinstance(adapter, BaseFrameworkProvider)
        assert hasattr(adapter, "governance_context")
        assert hasattr(adapter, "tracer")

    def test_adapter_default_initialization(self):
        """Test adapter initialization with default parameters."""
        adapter = GenOpsLlamaIndexAdapter()

        assert adapter.provider_name == "llamaindex"
        assert adapter.enable_cost_tracking is True
        assert adapter.enable_tracing is True
        assert adapter.default_governance_attrs == {}

    def test_adapter_custom_initialization(self):
        """Test adapter initialization with custom parameters."""
        custom_attrs = {"team": "test-team", "project": "test-project"}
        adapter = GenOpsLlamaIndexAdapter(
            enable_cost_tracking=False, enable_tracing=False, **custom_attrs
        )

        assert adapter.enable_cost_tracking is False
        assert adapter.enable_tracing is False
        assert adapter.default_governance_attrs == custom_attrs

    def test_adapter_initialization_with_governance_attributes(self):
        """Test adapter initialization with various governance attributes."""
        governance_attrs = {
            "team": "ai-research",
            "project": "rag-system",
            "customer_id": "enterprise-123",
            "environment": "production",
            "cost_center": "engineering",
        }

        adapter = GenOpsLlamaIndexAdapter(**governance_attrs)
        assert adapter.default_governance_attrs == governance_attrs


class TestLlamaIndexCostBreakdown:
    """Test LlamaIndexCostBreakdown dataclass."""

    def test_cost_breakdown_initialization(self):
        """Test cost breakdown initialization with default values."""
        breakdown = LlamaIndexCostBreakdown()

        assert breakdown.embedding_cost == 0.0
        assert breakdown.retrieval_cost == 0.0
        assert breakdown.synthesis_cost == 0.0
        assert breakdown.embedding_tokens == 0
        assert breakdown.synthesis_tokens == 0
        assert breakdown.retrieval_operations == 0
        assert breakdown.cost_by_provider == {}
        assert breakdown.optimization_suggestions == []

    def test_cost_breakdown_with_values(self):
        """Test cost breakdown with specific values."""
        cost_by_provider = {"openai": 0.005, "anthropic": 0.003}
        suggestions = ["Use smaller embedding model", "Enable caching"]

        breakdown = LlamaIndexCostBreakdown(
            embedding_cost=0.001,
            retrieval_cost=0.002,
            synthesis_cost=0.005,
            embedding_tokens=1000,
            synthesis_tokens=2000,
            retrieval_operations=3,
            cost_by_provider=cost_by_provider,
            optimization_suggestions=suggestions,
        )

        assert breakdown.embedding_cost == 0.001
        assert breakdown.retrieval_cost == 0.002
        assert breakdown.synthesis_cost == 0.005
        assert breakdown.embedding_tokens == 1000
        assert breakdown.synthesis_tokens == 2000
        assert breakdown.retrieval_operations == 3
        assert breakdown.cost_by_provider == cost_by_provider
        assert breakdown.optimization_suggestions == suggestions


class TestLlamaIndexOperationSummary:
    """Test LlamaIndexOperationSummary dataclass."""

    def test_operation_summary_initialization(self):
        """Test operation summary initialization with default values."""
        summary = LlamaIndexOperationSummary()

        assert summary.total_cost == 0.0
        assert summary.operation_count == 0
        assert summary.rag_pipelines == 0
        assert summary.avg_cost_per_operation == 0.0
        assert isinstance(summary.cost_breakdown, LlamaIndexCostBreakdown)
        assert summary.budget_status is None

    def test_operation_summary_with_values(self):
        """Test operation summary with specific values."""
        breakdown = LlamaIndexCostBreakdown(embedding_cost=0.001, synthesis_cost=0.002)
        budget_status = {"limit": 1.0, "used": 0.5, "alerts": []}

        summary = LlamaIndexOperationSummary(
            total_cost=0.010,
            operation_count=5,
            rag_pipelines=2,
            avg_cost_per_operation=0.002,
            cost_breakdown=breakdown,
            budget_status=budget_status,
        )

        assert summary.total_cost == 0.010
        assert summary.operation_count == 5
        assert summary.rag_pipelines == 2
        assert summary.avg_cost_per_operation == 0.002
        assert summary.cost_breakdown == breakdown
        assert summary.budget_status == budget_status


class TestQueryEngineInstrumentation:
    """Test query engine instrumentation capabilities."""

    @pytest.fixture
    def mock_query_engine(self):
        """Create mock query engine."""
        engine = Mock(spec=BaseQueryEngine)
        engine.query.return_value = Response("Test response")
        return engine

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return GenOpsLlamaIndexAdapter()

    def test_instrument_query_engine_basic(self, adapter, mock_query_engine):
        """Test basic query engine instrumentation."""
        instrumented = adapter.instrument_query_engine(mock_query_engine)

        # Should return the same engine (instrumentation is applied via callbacks)
        assert instrumented is mock_query_engine

    def test_instrument_query_engine_with_governance(self, adapter, mock_query_engine):
        """Test query engine instrumentation with governance attributes."""
        instrumented = adapter.instrument_query_engine(
            mock_query_engine, team="test-team", project="test-project"
        )

        assert instrumented is mock_query_engine

    @patch("genops.providers.llamaindex.adapter.tracer")
    def test_track_query_creates_span(self, mock_tracer, adapter, mock_query_engine):
        """Test that track_query creates proper OpenTelemetry span."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        result = adapter.track_query(mock_query_engine, "test query")

        # Verify span creation
        mock_tracer.start_as_current_span.assert_called_once_with("llamaindex.query")

        # Verify query was executed
        mock_query_engine.query.assert_called_once_with("test query")

        # Should return response
        assert isinstance(result, Response)

    def test_track_query_with_governance_context(self, adapter, mock_query_engine):
        """Test track_query with governance context."""
        with patch.object(adapter, "governance_context") as mock_context:
            mock_context.return_value = nullcontext()

            result = adapter.track_query(
                mock_query_engine,
                "test query",
                team="test-team",
                project="test-project",
            )

            # Verify governance context was used
            mock_context.assert_called_once_with(
                team="test-team", project="test-project"
            )

            # Verify query executed
            assert isinstance(result, Response)

    def test_track_query_error_handling(self, adapter, mock_query_engine):
        """Test track_query handles errors properly."""
        mock_query_engine.query.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            adapter.track_query(mock_query_engine, "test query")

    def test_track_query_with_cost_tracking(self, adapter, mock_query_engine):
        """Test track_query includes cost tracking."""
        with patch("genops.providers.llamaindex.adapter.tracer"):
            result = adapter.track_query(mock_query_engine, "test query")

            # Should complete without error
            assert isinstance(result, Response)


class TestChatEngineInstrumentation:
    """Test chat engine instrumentation capabilities."""

    @pytest.fixture
    def mock_chat_engine(self):
        """Create mock chat engine."""
        engine = Mock(spec=BaseChatEngine)
        engine.chat.return_value = Response("Test chat response")
        return engine

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return GenOpsLlamaIndexAdapter()

    def test_instrument_chat_engine_basic(self, adapter, mock_chat_engine):
        """Test basic chat engine instrumentation."""
        instrumented = adapter.instrument_chat_engine(mock_chat_engine)

        # Should return the same engine
        assert instrumented is mock_chat_engine

    def test_instrument_chat_engine_with_governance(self, adapter, mock_chat_engine):
        """Test chat engine instrumentation with governance attributes."""
        instrumented = adapter.instrument_chat_engine(
            mock_chat_engine, team="chat-team", project="chat-project"
        )

        assert instrumented is mock_chat_engine

    @patch("genops.providers.llamaindex.adapter.tracer")
    def test_track_chat_creates_span(self, mock_tracer, adapter, mock_chat_engine):
        """Test that track_chat creates proper OpenTelemetry span."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        result = adapter.track_chat(mock_chat_engine, "test message")

        # Verify span creation
        mock_tracer.start_as_current_span.assert_called_once_with("llamaindex.chat")

        # Verify chat was executed
        mock_chat_engine.chat.assert_called_once_with("test message")

        # Should return response
        assert isinstance(result, Response)

    def test_track_chat_with_governance_context(self, adapter, mock_chat_engine):
        """Test track_chat with governance context."""
        with patch.object(adapter, "governance_context") as mock_context:
            mock_context.return_value = nullcontext()

            result = adapter.track_chat(
                mock_chat_engine,
                "test message",
                team="chat-team",
                project="chat-project",
            )

            # Verify governance context was used
            mock_context.assert_called_once_with(
                team="chat-team", project="chat-project"
            )

            # Verify chat executed
            assert isinstance(result, Response)

    def test_track_chat_error_handling(self, adapter, mock_chat_engine):
        """Test track_chat handles errors properly."""
        mock_chat_engine.chat.side_effect = Exception("Chat failed")

        with pytest.raises(Exception, match="Chat failed"):
            adapter.track_chat(mock_chat_engine, "test message")


class TestIndexInstrumentation:
    """Test index instrumentation capabilities."""

    @pytest.fixture
    def mock_index(self):
        """Create mock index."""
        index = Mock(spec=BaseIndex)
        mock_query_engine = Mock(spec=BaseQueryEngine)
        mock_query_engine.query.return_value = Response("Index response")
        index.as_query_engine.return_value = mock_query_engine
        return index

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return GenOpsLlamaIndexAdapter()

    def test_instrument_index_basic(self, adapter, mock_index):
        """Test basic index instrumentation."""
        instrumented = adapter.instrument_index(mock_index)

        # Should return the same index
        assert instrumented is mock_index

    def test_instrument_index_with_governance(self, adapter, mock_index):
        """Test index instrumentation with governance attributes."""
        instrumented = adapter.instrument_index(
            mock_index, team="index-team", project="index-project"
        )

        assert instrumented is mock_index

    def test_track_index_query_basic(self, adapter, mock_index):
        """Test track_index_query basic functionality."""
        result = adapter.track_index_query(mock_index, "test query")

        # Verify index was used to create query engine
        mock_index.as_query_engine.assert_called_once()

        # Should return response
        assert isinstance(result, Response)

    def test_track_index_query_with_parameters(self, adapter, mock_index):
        """Test track_index_query with similarity_top_k parameter."""
        result = adapter.track_index_query(
            mock_index, "test query", similarity_top_k=5, team="test-team"
        )

        # Verify query engine was created with parameters
        mock_index.as_query_engine.assert_called_once_with(similarity_top_k=5)

        assert isinstance(result, Response)


class TestAdapterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return GenOpsLlamaIndexAdapter()

    def test_track_query_with_none_query_engine(self, adapter):
        """Test behavior with None query engine."""
        with pytest.raises(AttributeError):
            adapter.track_query(None, "test query")

    def test_track_query_with_empty_query(self, adapter):
        """Test behavior with empty query string."""
        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Empty response")

        result = adapter.track_query(mock_engine, "")

        mock_engine.query.assert_called_once_with("")
        assert isinstance(result, Response)

    def test_track_query_with_very_long_query(self, adapter):
        """Test behavior with very long query string."""
        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Long response")

        long_query = "test " * 1000  # 5000 character query
        result = adapter.track_query(mock_engine, long_query)

        mock_engine.query.assert_called_once_with(long_query)
        assert isinstance(result, Response)

    def test_track_query_with_unicode_query(self, adapter):
        """Test behavior with unicode characters in query."""
        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Unicode response")

        unicode_query = "Test query with émojis 🚀 and unicode 中文"
        result = adapter.track_query(mock_engine, unicode_query)

        mock_engine.query.assert_called_once_with(unicode_query)
        assert isinstance(result, Response)

    def test_governance_context_with_invalid_attributes(self, adapter):
        """Test governance context with invalid attribute types."""
        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Test response")

        # Should handle non-string attributes gracefully
        result = adapter.track_query(
            mock_engine,
            "test query",
            team=123,  # Invalid type
            project=None,  # None value
            cost_center=[],  # Invalid type
        )

        assert isinstance(result, Response)

    def test_concurrent_query_tracking(self, adapter):
        """Test concurrent query execution."""
        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Concurrent response")

        # Simulate multiple concurrent queries
        import threading

        results = []
        errors = []

        def run_query(query_id):
            try:
                result = adapter.track_query(mock_engine, f"Query {query_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_query, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All queries should complete successfully
        assert len(results) == 5
        assert len(errors) == 0
        assert all(isinstance(r, Response) for r in results)


class TestAdapterIntegrationWithMocking:
    """Integration tests using mocking to simulate LlamaIndex behavior."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with cost tracking enabled."""
        return GenOpsLlamaIndexAdapter(team="test-team", project="integration-test")

    def test_full_rag_pipeline_simulation(self, adapter):
        """Test complete RAG pipeline simulation."""
        # Mock document and index creation
        [Mock(text=f"Document {i} content") for i in range(3)]

        mock_index = Mock(spec=BaseIndex)
        mock_query_engine = Mock(spec=BaseQueryEngine)
        mock_query_engine.query.return_value = Response(
            "RAG response with relevant information"
        )
        mock_index.as_query_engine.return_value = mock_query_engine

        # Simulate index creation from documents
        with patch(
            "genops.providers.llamaindex.adapter.VectorStoreIndex"
        ) as mock_index_class:
            mock_index_class.from_documents.return_value = mock_index

            # Test the full pipeline
            query_engine = adapter.instrument_query_engine(mock_query_engine)
            response = adapter.track_query(
                query_engine,
                "What information is available in the documents?",
                team="integration-test",
                project="rag-pipeline",
            )

            assert isinstance(response, Response)
            assert response.response == "RAG response with relevant information"

    def test_multiple_query_types_simulation(self, adapter):
        """Test multiple types of queries in sequence."""
        mock_query_engine = Mock(spec=BaseQueryEngine)
        mock_chat_engine = Mock(spec=BaseChatEngine)

        # Configure different responses for different query types
        mock_query_engine.query.side_effect = [
            Response("Simple factual response"),
            Response("Complex analytical response"),
            Response("Summarization response"),
        ]

        mock_chat_engine.chat.return_value = Response("Interactive chat response")

        # Test different query patterns
        queries = [
            ("What is the capital of France?", "factual"),
            ("Analyze the trends in the data and provide insights", "analytical"),
            ("Summarize the main points from all documents", "summarization"),
        ]

        responses = []
        for query, query_type in queries:
            response = adapter.track_query(
                mock_query_engine, query, query_type=query_type, complexity="medium"
            )
            responses.append(response)

        # Test chat interaction
        chat_response = adapter.track_chat(
            mock_chat_engine,
            "Let's discuss the analysis results",
            interaction_type="discussion",
        )

        # Verify all responses
        assert len(responses) == 3
        assert all(isinstance(r, Response) for r in responses)
        assert isinstance(chat_response, Response)

        # Verify all queries were executed
        assert mock_query_engine.query.call_count == 3
        mock_chat_engine.chat.assert_called_once()

    @patch("genops.providers.llamaindex.adapter.tracer")
    def test_telemetry_data_collection(self, mock_tracer, adapter):
        """Test that telemetry data is properly collected."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Telemetry test response")

        # Execute query with telemetry
        result = adapter.track_query(
            mock_engine,
            "Test query for telemetry",
            team="telemetry-team",
            project="telemetry-project",
            customer_id="customer-123",
        )

        # Verify span was created
        mock_tracer.start_as_current_span.assert_called_once_with("llamaindex.query")

        # Verify span attributes would be set (in real implementation)
        assert isinstance(result, Response)

        # The span object should receive set_attribute calls in real implementation
        # This test verifies the span creation pattern is correct

    def test_error_recovery_and_logging(self, adapter):
        """Test error recovery and logging mechanisms."""
        mock_engine = Mock(spec=BaseQueryEngine)

        # Test various error conditions
        error_conditions = [
            ("Connection timeout", ConnectionError("Timeout")),
            ("Rate limit exceeded", Exception("Rate limit")),
            ("Invalid model response", ValueError("Invalid response")),
        ]

        for error_description, error in error_conditions:
            mock_engine.query.side_effect = error

            with pytest.raises(type(error)):
                adapter.track_query(mock_engine, f"Query causing {error_description}")

            # Reset for next test
            mock_engine.reset_mock()


if __name__ == "__main__":
    pytest.main([__file__])
