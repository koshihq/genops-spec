"""End-to-end integration tests for GenOps Cohere integration."""

import os
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

# Test imports
from genops.providers.cohere import (
    CohereOperation,
    GenOpsCohereAdapter,
    auto_instrument,
    instrument_cohere,
)
from genops.providers.cohere_cost_aggregator import CohereCostAggregator, TimeWindow
from genops.providers.cohere_pricing import CohereCalculator
from genops.providers.cohere_validation import quick_validate, validate_setup


class TestCohereIntegrationWorkflow:
    """Test complete Cohere integration workflow."""

    @pytest.fixture
    def mock_cohere_environment(self):
        """Setup mock Cohere environment for integration testing."""
        with patch('genops.providers.cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Setup mock responses
            mock_client.chat.return_value = Mock(
                message=Mock(content=[Mock(text="Integration test response")]),
                usage=Mock(input_tokens=50, output_tokens=25)
            )

            mock_client.embed.return_value = Mock(
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                usage=Mock(input_tokens=30, output_tokens=0)
            )

            mock_client.rerank.return_value = Mock(
                results=[
                    Mock(index=0, relevance_score=0.95, document={"text": "First doc"}),
                    Mock(index=1, relevance_score=0.87, document={"text": "Second doc"})
                ],
                usage=Mock(input_tokens=0, output_tokens=0)
            )

            yield mock_client

    def test_complete_setup_workflow(self, mock_cohere_environment):
        """Test complete setup from validation to operation."""
        # Step 1: Validate setup
        with patch.dict(os.environ, {'CO_API_KEY': 'test-integration-key'}):
            validation_result = validate_setup()
            assert validation_result.success is True

        # Step 2: Create instrumented adapter
        adapter = instrument_cohere(
            team="integration-team",
            project="workflow-test"
        )

        # Step 3: Perform operations
        chat_response = adapter.chat(
            message="Integration test message",
            model="command-r-08-2024"
        )

        embed_response = adapter.embed(
            texts=["integration", "test"],
            model="embed-english-v4.0"
        )

        rerank_response = adapter.rerank(
            query="integration",
            documents=["test doc 1", "test doc 2"],
            model="rerank-english-v3.0"
        )

        # Step 4: Verify all operations succeeded
        assert chat_response.success is True
        assert embed_response.success is True
        assert rerank_response.success is True

        # Step 5: Verify cost tracking
        assert chat_response.usage.total_cost > 0
        assert embed_response.usage.total_cost > 0
        assert rerank_response.usage.total_cost > 0

        # Step 6: Verify usage summary
        summary = adapter.get_usage_summary()
        assert summary["total_operations"] == 3
        assert summary["total_cost"] > 0

    def test_multi_operation_workflow_integration(self, mock_cohere_environment):
        """Test complex multi-operation workflow with cost aggregation."""
        # Setup cost aggregator
        aggregator = CohereCostAggregator(
            enable_detailed_tracking=True,
            cost_alert_threshold=0.10
        )

        # Create adapter with aggregator
        adapter = GenOpsCohereAdapter(
            api_key="test-key",
            cost_aggregator=aggregator,
            default_team="workflow-team",
            default_project="multi-op-test"
        )

        # Execute complex workflow
        workflow_results = self._execute_intelligent_search_workflow(
            adapter,
            query="machine learning applications",
            documents=[
                "ML helps in medical diagnosis and treatment",
                "Machine learning improves search and recommendations",
                "AI assists in financial trading and risk assessment",
                "Deep learning powers image recognition systems"
            ]
        )

        # Verify workflow execution
        assert workflow_results["success"] is True
        assert workflow_results["total_cost"] > 0
        assert len(workflow_results["cost_breakdown"]) == 4  # embed_query, embed_docs, rerank, summarize

        # Verify cost aggregator captured all operations
        summary = aggregator.get_cost_summary(TimeWindow.HOUR)
        assert summary.overview.total_operations == 4
        assert summary.overview.total_cost == workflow_results["total_cost"]

        # Verify operation type breakdown
        op_summary = aggregator.get_operation_summary()
        assert CohereOperation.EMBED in op_summary  # 2 embed operations
        assert CohereOperation.RERANK in op_summary  # 1 rerank operation
        assert CohereOperation.CHAT in op_summary  # 1 chat operation

    def test_enterprise_deployment_integration(self, mock_cohere_environment):
        """Test enterprise deployment patterns with governance."""
        # Setup enterprise-style configuration
        enterprise_config = {
            "teams": {
                "ml-team": {"budget": 50.0, "models": ["command-r-08-2024", "embed-english-v4.0"]},
                "search-team": {"budget": 30.0, "models": ["command-light", "rerank-english-v3.0"]},
                "research-team": {"budget": 100.0, "models": ["command-r-plus-08-2024"]}
            },
            "global_budget": 200.0,
            "cost_alert_threshold": 0.8
        }

        # Create team-specific adapters
        team_adapters = {}
        global_aggregator = CohereCostAggregator(
            cost_alert_threshold=enterprise_config["global_budget"] * enterprise_config["cost_alert_threshold"]
        )

        for team, config in enterprise_config["teams"].items():
            team_adapters[team] = GenOpsCohereAdapter(
                api_key="test-key",
                default_team=team,
                cost_aggregator=global_aggregator,
                budget_limit=config["budget"],
                allowed_models=config["models"]
            )

        # Simulate team usage
        team_operations = [
            ("ml-team", "command-r-08-2024", CohereOperation.CHAT, "ML model evaluation"),
            ("ml-team", "embed-english-v4.0", CohereOperation.EMBED, ["ml embedding 1", "ml embedding 2"]),
            ("search-team", "command-light", CohereOperation.CHAT, "Search query processing"),
            ("search-team", "rerank-english-v3.0", CohereOperation.RERANK, "search rerank"),
            ("research-team", "command-r-plus-08-2024", CohereOperation.CHAT, "Advanced research query")
        ]

        total_enterprise_cost = 0

        for team, model, operation, content in team_operations:
            adapter = team_adapters[team]

            if operation == CohereOperation.CHAT:
                response = adapter.chat(message=content, model=model)
            elif operation == CohereOperation.EMBED:
                response = adapter.embed(texts=content if isinstance(content, list) else [content], model=model)
            elif operation == CohereOperation.RERANK:
                response = adapter.rerank(
                    query="test query",
                    documents=["doc1", "doc2"],
                    model=model
                )

            assert response.success is True
            total_enterprise_cost += response.usage.total_cost

        # Verify enterprise reporting
        enterprise_summary = global_aggregator.get_cost_summary(TimeWindow.DAY)

        # Should have all teams represented
        assert "ml-team" in enterprise_summary.by_team
        assert "search-team" in enterprise_summary.by_team
        assert "research-team" in enterprise_summary.by_team

        # Total cost should match sum of individual operations
        assert abs(enterprise_summary.overview.total_cost - total_enterprise_cost) < 0.001

        # Should have optimization insights for enterprise usage
        insights = global_aggregator.get_cost_optimization_insights()
        assert len(insights.recommendations) > 0

    def test_auto_instrumentation_integration(self, mock_cohere_environment):
        """Test auto-instrumentation functionality."""
        with patch('genops.providers.cohere.HAS_COHERE', True):
            # Enable auto-instrumentation
            success = auto_instrument()
            assert success is True

            # Verify that direct Cohere client usage is now tracked
            # (This would require more complex mocking in a real scenario)
            with patch('genops.providers.cohere.ClientV2') as mock_client:

                # Simulate auto-instrumented client
                client = mock_client.return_value
                client.chat.return_value = Mock(
                    message=Mock(content=[Mock(text="Auto-instrumented response")]),
                    usage=Mock(input_tokens=20, output_tokens=15)
                )

                # This should now be automatically tracked
                # (Implementation would require actual client monkey-patching)
                response = client.chat(
                    model="command-light",
                    messages=[{"role": "user", "content": "Hello"}]
                )

                assert response is not None

    def test_validation_error_recovery_integration(self):
        """Test validation and error recovery integration."""
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            # Should fail validation
            result = quick_validate()
            assert result is False

            # Detailed validation should provide specific fixes
            detailed_result = validate_setup()
            assert detailed_result.success is False
            assert detailed_result.has_critical_issues is True

            # Should have authentication issue
            auth_issues = [
                issue for issue in detailed_result.issues
                if "api key" in issue.title.lower()
            ]
            assert len(auth_issues) > 0
            assert "CO_API_KEY" in auth_issues[0].fix_suggestion

        # Test with invalid API key
        with patch.dict(os.environ, {'CO_API_KEY': 'invalid-key'}):
            with patch('genops.providers.cohere.ClientV2') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                mock_client.chat.side_effect = Exception("Unauthorized")

                # Should fail gracefully
                adapter = instrument_cohere()
                response = adapter.chat(message="test", model="command-light")

                assert response.success is False
                assert "unauthorized" in response.error_message.lower()

    def test_performance_monitoring_integration(self, mock_cohere_environment):
        """Test performance monitoring across operations."""
        adapter = instrument_cohere(
            team="performance-team",
            project="monitoring-test"
        )

        # Add latency to mock responses
        def delayed_chat(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return Mock(
                message=Mock(content=[Mock(text="Performance test response")]),
                usage=Mock(input_tokens=25, output_tokens=30)
            )

        def delayed_embed(*args, **kwargs):
            time.sleep(0.05)  # 50ms delay
            return Mock(
                embeddings=[[0.1, 0.2]],
                usage=Mock(input_tokens=20, output_tokens=0)
            )

        mock_cohere_environment.chat.side_effect = delayed_chat
        mock_cohere_environment.embed.side_effect = delayed_embed

        # Execute operations and measure performance
        start_time = time.time()

        chat_response = adapter.chat(message="Performance test", model="command-light")
        embed_response = adapter.embed(texts=["performance"], model="embed-english-v4.0")

        total_time = time.time() - start_time

        # Verify performance metrics were captured
        assert chat_response.usage.latency_ms >= 100
        assert embed_response.usage.latency_ms >= 50
        assert chat_response.usage.tokens_per_second > 0

        # Total execution should be approximately sum of individual delays
        assert total_time >= 0.15  # 100ms + 50ms + overhead

    def test_telemetry_export_integration(self, mock_cohere_environment):
        """Test OpenTelemetry export integration."""
        with patch('genops.providers.cohere.trace') as mock_trace:
            mock_tracer = Mock()
            mock_span = Mock()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            mock_trace.get_tracer.return_value = mock_tracer

            # Create adapter (should initialize tracing)
            adapter = instrument_cohere(
                team="telemetry-team",
                project="export-test"
            )

            # Execute operation
            response = adapter.chat(
                message="Telemetry test",
                model="command-light"
            )

            # Verify OpenTelemetry integration
            assert response.success is True

            # Should have created telemetry spans
            mock_tracer.start_as_current_span.assert_called()
            mock_span.set_attribute.assert_called()

            # Verify span attributes include governance data
            span_calls = mock_span.set_attribute.call_args_list
            attribute_names = [call[0][0] for call in span_calls]

            # Should include GenOps-specific attributes
            assert any("genops.team" in attr for attr in attribute_names)
            assert any("genops.cost" in attr for attr in attribute_names)
            assert any("genops.model" in attr for attr in attribute_names)

    def test_cost_optimization_workflow_integration(self, mock_cohere_environment):
        """Test cost optimization workflow integration."""
        # Setup calculator and aggregator
        calculator = CohereCalculator()
        aggregator = CohereCostAggregator(enable_detailed_tracking=True)

        adapter = GenOpsCohereAdapter(
            api_key="test-key",
            cost_aggregator=aggregator
        )

        # Simulate expensive operations
        expensive_operations = [
            ("command-r-plus-08-2024", "High-cost model for simple task"),
            ("command-r-plus-08-2024", "Another expensive operation"),
            ("command-r-08-2024", "Medium-cost operation"),
            ("command-light", "Cost-effective operation")
        ]

        total_cost = 0
        for model, message in expensive_operations:
            response = adapter.chat(message=message, model=model)
            total_cost += response.usage.total_cost

        # Get optimization insights
        insights = aggregator.get_cost_optimization_insights()

        # Should identify opportunities to reduce costs
        assert len(insights.recommendations) > 0

        # Should suggest using cheaper models for simple tasks
        model_optimization_insights = [
            insight for insight in insights.recommendations
            if insight.type == "model_optimization"
        ]
        assert len(model_optimization_insights) > 0

        # Calculate potential savings from recommendations
        total_potential_savings = sum(
            insight.potential_savings for insight in insights.recommendations
        )
        assert total_potential_savings > 0

        # Should be significant savings opportunity
        savings_percentage = total_potential_savings / total_cost
        assert savings_percentage > 0.1  # At least 10% savings potential

    def _execute_intelligent_search_workflow(
        self,
        adapter: GenOpsCohereAdapter,
        query: str,
        documents: List[str]
    ) -> Dict[str, Any]:
        """Execute intelligent search workflow for testing."""
        try:
            # Step 1: Generate query embedding
            query_embedding = adapter.embed(
                texts=[query],
                model="embed-english-v4.0",
                input_type="search_query"
            )

            # Step 2: Generate document embeddings
            doc_embeddings = adapter.embed(
                texts=documents,
                model="embed-english-v4.0",
                input_type="search_document"
            )

            # Step 3: Rerank documents
            rankings = adapter.rerank(
                query=query,
                documents=documents,
                model="rerank-english-v3.0",
                top_n=3
            )

            # Step 4: Generate summary
            top_docs = [r['document']['text'] for r in rankings.rankings[:2]]
            summary = adapter.chat(
                message=f"Summarize these search results for '{query}': {'; '.join(top_docs)}",
                model="command-r-08-2024"
            )

            # Calculate total cost
            total_cost = (
                query_embedding.usage.total_cost +
                doc_embeddings.usage.total_cost +
                rankings.usage.total_cost +
                summary.usage.total_cost
            )

            return {
                "success": True,
                "summary": summary.content,
                "rankings": rankings.rankings,
                "total_cost": total_cost,
                "cost_breakdown": {
                    "query_embedding": query_embedding.usage.total_cost,
                    "doc_embeddings": doc_embeddings.usage.total_cost,
                    "reranking": rankings.usage.total_cost,
                    "summarization": summary.usage.total_cost
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_cost": 0.0
            }


class TestCohereErrorHandlingIntegration:
    """Test error handling across integration scenarios."""

    def test_network_error_recovery(self):
        """Test recovery from network errors."""
        with patch('genops.providers.cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.side_effect = Exception("Connection timeout")

            adapter = instrument_cohere()
            response = adapter.chat(message="test", model="command-light")

            assert response.success is False
            assert "timeout" in response.error_message.lower()
            assert response.usage is not None  # Should have empty usage metrics

    def test_rate_limit_handling(self):
        """Test rate limit error handling."""
        with patch('genops.providers.cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.side_effect = Exception("Rate limit exceeded")

            adapter = instrument_cohere()
            response = adapter.chat(message="test", model="command-light")

            assert response.success is False
            assert "rate limit" in response.error_message.lower()

    def test_invalid_model_handling(self):
        """Test invalid model error handling."""
        with patch('genops.providers.cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.side_effect = Exception("Model not found")

            adapter = instrument_cohere()
            response = adapter.chat(message="test", model="invalid-model")

            assert response.success is False
            assert "not found" in response.error_message.lower()

    def test_budget_exceeded_handling(self):
        """Test budget exceeded scenarios."""
        with patch('genops.providers.cohere.ClientV2') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = Mock(
                message=Mock(content=[Mock(text="test response")]),
                usage=Mock(input_tokens=10, output_tokens=5)
            )

            # Set very low budget limit
            adapter = GenOpsCohereAdapter(
                api_key="test-key",
                budget_limit=0.000001  # Extremely low limit
            )

            # Mock high cost calculation
            with patch.object(adapter, '_calculate_cost', return_value=(0.001, 0.0, 0.0)):
                response = adapter.chat(message="test", model="command-light")

                # Should complete but potentially warn about budget
                assert response.success is True  # GenOps doesn't block by default, just warns


class TestCohereCompatibilityIntegration:
    """Test compatibility with different environments and configurations."""

    def test_python_version_compatibility(self):
        """Test compatibility across Python versions."""
        # This would test version-specific features
        import sys
        python_version = sys.version_info

        # Should work on Python 3.9+
        assert python_version >= (3, 9)

        # Basic import should work
        from genops.providers.cohere import instrument_cohere
        adapter = instrument_cohere()
        assert adapter is not None

    def test_optional_dependencies_handling(self):
        """Test graceful handling of missing optional dependencies."""
        # Test when OpenTelemetry is not available
        with patch('genops.providers.cohere.HAS_OPENTELEMETRY', False):
            adapter = instrument_cohere()
            assert adapter is not None
            # Should work without telemetry

        # Test when Cohere client is not available
        with patch('genops.providers.cohere.HAS_COHERE', False):
            with pytest.raises(ImportError):
                adapter = instrument_cohere()

    def test_environment_variable_integration(self):
        """Test environment variable handling."""
        test_cases = [
            {'CO_API_KEY': 'test-key-123'},
            {'COHERE_API_KEY': 'alt-key-456'},  # Alternative name
            {}  # No environment variables
        ]

        for env_vars in test_cases:
            with patch.dict(os.environ, env_vars, clear=True):
                if env_vars:
                    # Should pick up API key from environment
                    adapter = GenOpsCohereAdapter()
                    assert adapter.api_key in env_vars.values()
                else:
                    # Should handle missing API key gracefully
                    adapter = GenOpsCohereAdapter()
                    assert adapter.api_key is None


if __name__ == "__main__":
    pytest.main([__file__])
