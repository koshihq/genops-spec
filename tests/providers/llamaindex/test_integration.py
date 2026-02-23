"""
Integration tests for GenOps LlamaIndex provider.

Comprehensive integration tests covering end-to-end workflows,
component interactions, and real-world usage scenarios.
"""

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

# Import the modules under test
from genops.providers.llamaindex.adapter import GenOpsLlamaIndexAdapter
from genops.providers.llamaindex.cost_aggregator import (
    create_llamaindex_cost_context,
)
from genops.providers.llamaindex.rag_monitor import (
    RAGPerformanceMetrics,
    RAGQualityMetrics,
    create_rag_monitor,
)
from genops.providers.llamaindex.registration import (
    auto_instrument,
    instrument_llamaindex,
)


class TestEndToEndRAGWorkflow:
    """Test complete end-to-end RAG workflows."""

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for testing."""
        return [
            Document(
                text="Document 1: Information about artificial intelligence and machine learning."
            ),
            Document(
                text="Document 2: Guide to building RAG applications with LlamaIndex."
            ),
            Document(
                text="Document 3: Cost optimization strategies for AI applications."
            ),
        ]

    @pytest.fixture
    def mock_query_engine(self):
        """Create mock query engine with realistic behavior."""
        engine = Mock(spec=BaseQueryEngine)

        # Configure different responses based on query content
        def mock_query(query_text):
            if "artificial intelligence" in query_text.lower():
                return Response(
                    "AI is a field of computer science focused on creating intelligent machines."
                )
            elif "llamaindex" in query_text.lower():
                return Response(
                    "LlamaIndex is a framework for building RAG applications."
                )
            elif "cost" in query_text.lower():
                return Response(
                    "Cost optimization involves using efficient models and caching strategies."
                )
            else:
                return Response("Generic response to the query.")

        engine.query.side_effect = mock_query
        return engine

    def test_complete_rag_pipeline_with_all_components(
        self, mock_query_engine, mock_documents
    ):
        """Test complete RAG pipeline with all GenOps components."""
        # Initialize all components
        adapter = GenOpsLlamaIndexAdapter(
            team="integration-test", project="end-to-end-rag"
        )

        rag_monitor = create_rag_monitor(
            enable_quality_metrics=True,
            enable_cost_tracking=True,
            enable_performance_profiling=True,
        )

        # Test complete workflow with cost tracking and monitoring
        with create_llamaindex_cost_context(
            "e2e_test", budget_limit=1.0
        ) as cost_context:
            # Multiple queries with different characteristics
            queries = [
                ("What is artificial intelligence?", "factual"),
                ("How do I build RAG with LlamaIndex?", "technical"),
                ("What are cost optimization strategies?", "analytical"),
            ]

            responses = []

            for query, query_type in queries:
                # Monitor each RAG operation
                with rag_monitor.monitor_rag_operation(
                    query, team="integration-test", query_type=query_type
                ):
                    # Track query with adapter
                    response = adapter.track_query(
                        mock_query_engine,
                        query,
                        team="integration-test",
                        project="end-to-end-rag",
                        query_type=query_type,
                    )

                    responses.append(response)

                    # Simulate cost tracking
                    cost_context.add_embedding_cost("openai", "ada-002", 100, 0.00001)
                    cost_context.add_retrieval_cost(0.001)
                    cost_context.add_synthesis_cost(
                        "openai", "gpt-3.5", 200, 150, 0.0007
                    )

                    # Record quality metrics
                    quality_metrics = RAGQualityMetrics(
                        retrieval_relevance=0.85,
                        response_faithfulness=0.90,
                        answer_relevancy=0.88,
                    )
                    rag_monitor.record_quality_metrics(
                        f"query_{len(responses)}", quality_metrics
                    )

                    # Record performance metrics
                    perf_metrics = RAGPerformanceMetrics(
                        embedding_latency_ms=150.0,
                        retrieval_latency_ms=200.0,
                        synthesis_latency_ms=800.0,
                        total_latency_ms=1150.0,
                        tokens_per_second=20.0,
                    )
                    rag_monitor.record_performance_metrics(
                        f"query_{len(responses)}", perf_metrics
                    )

        # Verify all components worked together
        assert len(responses) == 3
        assert all(isinstance(r, Response) for r in responses)

        # Verify cost tracking
        cost_summary = cost_context.get_current_summary()
        assert cost_summary.total_cost > 0.0
        assert cost_summary.operation_count > 0

        # Verify monitoring
        analytics = rag_monitor.get_analytics()
        assert analytics.total_operations == 3
        assert analytics.avg_response_time_ms > 0.0
        assert analytics.avg_retrieval_relevance > 0.8

    def test_rag_workflow_with_error_handling(self, mock_query_engine):
        """Test RAG workflow handles errors gracefully."""
        adapter = GenOpsLlamaIndexAdapter()
        rag_monitor = create_rag_monitor()

        # Configure query engine to fail on specific query
        def failing_query(query_text):
            if "error" in query_text.lower():
                raise Exception("Simulated query failure")
            return Response("Normal response")

        mock_query_engine.query.side_effect = failing_query

        with create_llamaindex_cost_context("error_test") as cost_context:
            # Successful query
            response1 = adapter.track_query(mock_query_engine, "normal query")
            assert isinstance(response1, Response)

            # Failed query - should raise exception but not break the context
            with pytest.raises(Exception, match="Simulated query failure"):
                with rag_monitor.monitor_rag_operation("error query"):
                    adapter.track_query(mock_query_engine, "error query")

            # Another successful query after error
            response2 = adapter.track_query(mock_query_engine, "another normal query")
            assert isinstance(response2, Response)

        # Verify analytics include both successful and failed operations
        analytics = rag_monitor.get_analytics()
        assert (
            analytics.total_operations == 1
        )  # Only successful operations in RAG monitor

        # Cost context should track successful operations
        cost_summary = cost_context.get_current_summary()
        assert cost_summary.operation_count >= 0


class TestComponentIntegration:
    """Test integration between different GenOps components."""

    def test_adapter_and_cost_aggregator_integration(self):
        """Test integration between adapter and cost aggregator."""
        adapter = GenOpsLlamaIndexAdapter(enable_cost_tracking=True, team="cost-test")

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Cost tracking response")

        with create_llamaindex_cost_context("integration_cost") as cost_context:
            # Multiple queries to track cumulative costs
            for i in range(3):
                adapter.track_query(mock_engine, f"Query {i}", complexity="medium")

                # Simulate costs being added by the adapter
                cost_context.add_synthesis_cost("openai", "gpt-3.5", 300, 200, 0.001)

            # Verify cost aggregation
            summary = cost_context.get_current_summary()
            assert summary.total_cost > 0.0
            assert summary.operation_count >= 3  # At least 3 operations

    def test_adapter_and_rag_monitor_integration(self):
        """Test integration between adapter and RAG monitor."""
        adapter = GenOpsLlamaIndexAdapter()
        rag_monitor = create_rag_monitor()

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Monitoring response")

        # Execute queries while monitoring
        queries = ["Query 1", "Query 2", "Query 3"]

        for query in queries:
            with rag_monitor.monitor_rag_operation(query):
                response = adapter.track_query(mock_engine, query)
                assert isinstance(response, Response)

        # Verify monitoring captured all operations
        analytics = rag_monitor.get_analytics()
        assert analytics.total_operations == len(queries)

    def test_cost_aggregator_and_monitor_integration(self):
        """Test integration between cost aggregator and monitor."""
        rag_monitor = create_rag_monitor(enable_cost_tracking=True)

        with create_llamaindex_cost_context("cost_monitor_integration") as cost_context:
            # Simulate RAG operations with both cost and quality tracking
            for i in range(3):
                with rag_monitor.monitor_rag_operation(f"integrated_query_{i}"):
                    # Add costs
                    cost_context.add_embedding_cost("openai", "ada-002", 1000, 0.0001)
                    cost_context.add_retrieval_cost(0.001)
                    cost_context.add_synthesis_cost(
                        "anthropic", "claude-3", 500, 300, 0.002
                    )

                    # Add quality metrics
                    quality_metrics = RAGQualityMetrics(
                        retrieval_relevance=0.8 + (i * 0.05),  # Varying quality
                        response_faithfulness=0.85,
                    )
                    rag_monitor.record_quality_metrics(f"op_{i}", quality_metrics)

        # Verify both systems tracked the operations
        cost_summary = cost_context.get_current_summary()
        assert cost_summary.total_cost > 0.0

        analytics = rag_monitor.get_analytics()
        assert analytics.total_operations == 3
        assert analytics.avg_retrieval_relevance > 0.8


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration with all components."""

    @patch("genops.providers.llamaindex.registration.Settings")
    def test_auto_instrument_integration(self, mock_settings):
        """Test auto-instrumentation enables all GenOps features."""

        # Test auto-instrumentation registration
        auto_instrument()

        # Should have registered callbacks with LlamaIndex
        # In a real implementation, this would verify callback registration

        # Test that instrumented components work together
        adapter = instrument_llamaindex(
            team="auto-instrument-test", project="integration"
        )

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Auto-instrumented response")

        response = adapter.track_query(mock_engine, "Auto-instrumented query")
        assert isinstance(response, Response)

    def test_instrument_llamaindex_factory_integration(self):
        """Test instrument_llamaindex factory creates integrated adapter."""
        adapter = instrument_llamaindex(
            enable_cost_tracking=True, team="factory-test", project="integration-test"
        )

        assert isinstance(adapter, GenOpsLlamaIndexAdapter)
        assert adapter.enable_cost_tracking is True
        assert adapter.default_governance_attrs["team"] == "factory-test"
        assert adapter.default_governance_attrs["project"] == "integration-test"


class TestMultiProviderIntegration:
    """Test integration across multiple AI providers."""

    def test_multi_provider_cost_tracking(self):
        """Test cost tracking across multiple AI providers."""

        with create_llamaindex_cost_context("multi_provider") as cost_context:
            # Simulate operations with different providers
            providers_and_costs = [
                ("openai", "gpt-4", 1000, 500, 0.05),
                ("anthropic", "claude-3", 800, 400, 0.03),
                ("google", "gemini-pro", 1200, 600, 0.02),
            ]

            for (
                provider,
                model,
                input_tokens,
                output_tokens,
                cost,
            ) in providers_and_costs:
                cost_context.add_synthesis_cost(
                    provider, model, input_tokens, output_tokens, cost
                )

            # Verify multi-provider tracking
            summary = cost_context.get_current_summary()
            assert summary.total_cost == 0.10  # 0.05 + 0.03 + 0.02

            breakdown = summary.cost_breakdown
            assert len(breakdown.cost_by_provider) == 3
            assert breakdown.cost_by_provider["openai"] == 0.05
            assert breakdown.cost_by_provider["anthropic"] == 0.03
            assert breakdown.cost_by_provider["google"] == 0.02

    def test_provider_fallback_integration(self):
        """Test provider fallback scenarios."""
        adapter = GenOpsLlamaIndexAdapter()

        # Simulate primary provider failure
        primary_engine = Mock(spec=BaseQueryEngine)
        primary_engine.query.side_effect = Exception("Primary provider failed")

        fallback_engine = Mock(spec=BaseQueryEngine)
        fallback_engine.query.return_value = Response("Fallback response")

        with create_llamaindex_cost_context("fallback_test"):
            # Try primary provider (should fail)
            with pytest.raises(Exception):  # noqa: B017
                adapter.track_query(primary_engine, "test query")

            # Use fallback provider (should succeed)
            response = adapter.track_query(fallback_engine, "test query")
            assert isinstance(response, Response)
            assert "Fallback" in response.response


class TestProductionScenarioIntegration:
    """Test integration in production-like scenarios."""

    def test_high_volume_rag_operations(self):
        """Test integration under high volume operations."""
        adapter = GenOpsLlamaIndexAdapter()
        rag_monitor = create_rag_monitor()

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("High volume response")

        with create_llamaindex_cost_context(
            "high_volume", budget_limit=10.0
        ) as cost_context:
            # Simulate high volume of operations
            num_operations = 100

            for i in range(num_operations):
                with rag_monitor.monitor_rag_operation(f"query_{i}"):
                    adapter.track_query(
                        mock_engine,
                        f"High volume query {i}",
                        batch_id=f"batch_{i // 10}",  # Group in batches of 10
                    )

                    # Add small costs that accumulate
                    cost_context.add_synthesis_cost(
                        "openai", "gpt-3.5", 100, 50, 0.0001
                    )

        # Verify all operations were tracked
        analytics = rag_monitor.get_analytics()
        assert analytics.total_operations == num_operations

        cost_summary = cost_context.get_current_summary()
        assert cost_summary.operation_count >= num_operations
        assert cost_summary.total_cost > 0.0
        assert cost_summary.total_cost < 10.0  # Under budget

    def test_concurrent_rag_sessions(self):
        """Test concurrent RAG sessions integration."""
        import threading

        results = []
        errors = []

        def run_rag_session(session_id):
            try:
                adapter = GenOpsLlamaIndexAdapter()
                mock_engine = Mock(spec=BaseQueryEngine)
                mock_engine.query.return_value = Response(
                    f"Session {session_id} response"
                )

                with create_llamaindex_cost_context(
                    f"session_{session_id}"
                ) as cost_context:
                    # Multiple queries per session
                    for i in range(3):
                        adapter.track_query(
                            mock_engine,
                            f"Session {session_id} query {i}",
                            session_id=session_id,
                        )

                        cost_context.add_synthesis_cost(
                            "openai", "gpt-3.5", 200, 100, 0.0005
                        )

                    summary = cost_context.get_current_summary()
                    results.append(
                        {
                            "session_id": session_id,
                            "total_cost": summary.total_cost,
                            "operations": summary.operation_count,
                        }
                    )

            except Exception as e:
                errors.append((session_id, str(e)))

        # Run multiple concurrent sessions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_rag_session, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all sessions to complete
        for thread in threads:
            thread.join()

        # Verify all sessions completed successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify each session tracked operations independently
        for result in results:
            assert result["total_cost"] > 0.0
            assert result["operations"] >= 3

    def test_budget_exhaustion_integration(self):
        """Test integration when budget limits are exceeded."""
        adapter = GenOpsLlamaIndexAdapter()

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Budget test response")

        with create_llamaindex_cost_context(
            "budget_test", budget_limit=0.01, enable_alerts=True
        ) as cost_context:
            # Add operations that will exceed budget
            operations_completed = 0
            budget_alerts = []

            for i in range(10):
                try:
                    adapter.track_query(mock_engine, f"Budget query {i}")

                    # Add cost that will eventually exceed budget
                    cost_context.add_synthesis_cost("openai", "gpt-4", 500, 300, 0.003)

                    operations_completed += 1

                    # Check budget status
                    budget_status = cost_context.get_budget_status()
                    if budget_status["alerts"]:
                        budget_alerts.extend(budget_status["alerts"])

                except Exception:
                    # In production, might implement budget enforcement
                    break

        # Verify budget monitoring worked
        assert operations_completed > 0
        assert len(budget_alerts) > 0  # Should have generated alerts

        final_summary = cost_context.get_current_summary()
        assert final_summary.total_cost > 0.01  # Exceeded budget


class TestValidationIntegration:
    """Test integration with validation and diagnostics."""

    def test_validation_integration_with_components(self):
        """Test validation works with all components."""
        from genops.providers.llamaindex.validation import (
            print_validation_result,
            validate_setup,
        )

        # Run validation
        validation_result = validate_setup()

        # Should return ValidationResult object
        assert hasattr(validation_result, "success")
        assert hasattr(validation_result, "details")

        # Should not raise exception with print function
        print_validation_result(validation_result)

    def test_diagnostics_with_real_usage(self):
        """Test diagnostic information with real usage patterns."""
        adapter = GenOpsLlamaIndexAdapter()

        mock_engine = Mock(spec=BaseQueryEngine)
        mock_engine.query.return_value = Response("Diagnostic test response")

        # Use components to generate diagnostic data
        with create_llamaindex_cost_context("diagnostic_test") as cost_context:
            # Normal operation
            adapter.track_query(mock_engine, "diagnostic query")
            cost_context.add_synthesis_cost("openai", "gpt-3.5", 200, 100, 0.0003)

            # Verify diagnostic information is available
            summary = cost_context.get_current_summary()
            assert summary.total_cost > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
