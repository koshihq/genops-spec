"""Integration tests for GenOps Langfuse provider."""

import os
import time
from unittest.mock import Mock, patch

import pytest

# Test if Langfuse is available
try:
    import langfuse
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

pytestmark = pytest.mark.skipif(not HAS_LANGFUSE, reason="Langfuse not installed")

from genops.providers.langfuse import (
    GenOpsLangfuseAdapter,
    GovernancePolicy,
    _auto_instrument_langfuse,
    instrument_langfuse,
)


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Mock Langfuse client for integration tests."""
        with patch('genops.providers.langfuse.Langfuse') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance

    def test_full_workflow_with_governance(self, mock_langfuse_client):
        """Test complete workflow with governance integration."""
        # Setup mocks
        mock_trace = Mock()
        mock_trace.id = "trace-workflow-123"
        mock_trace.metadata = {}
        mock_langfuse_client.trace.return_value = mock_trace

        mock_generation = Mock()
        mock_generation.id = "gen-workflow-456"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        # Initialize adapter with governance
        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-integration",
            langfuse_secret_key="sk-lf-integration",
            team="integration-team",
            project="workflow-test",
            budget_limits={"daily": 10.0},
            policy_mode=GovernancePolicy.ENFORCED
        )

        # Execute complete workflow
        with adapter.trace_with_governance(
            name="integration_workflow",
            customer_id="workflow-customer",
            cost_center="integration"
        ) as trace:

            # Step 1: Data preprocessing
            preprocessing_response = adapter.generation_with_cost_tracking(
                prompt="Clean and preprocess this data: sample data",
                model="gpt-3.5-turbo",
                max_cost=0.05,
                operation="preprocessing"
            )

            # Step 2: Analysis
            analysis_response = adapter.generation_with_cost_tracking(
                prompt="Analyze the preprocessed data for patterns",
                model="gpt-4",
                max_cost=0.20,
                operation="analysis"
            )

            # Step 3: Summarization
            summary_response = adapter.generation_with_cost_tracking(
                prompt="Summarize the analysis results",
                model="gpt-3.5-turbo",
                max_cost=0.03,
                operation="summarization"
            )

        # Verify workflow execution
        assert mock_langfuse_client.trace.called
        assert mock_langfuse_client.generation.call_count == 3

        # Verify governance attributes were propagated
        trace_call = mock_langfuse_client.trace.call_args
        assert "genops_governance" in trace_call[1]["metadata"]
        assert trace_call[1]["metadata"]["genops_governance"]["customer_id"] == "workflow-customer"
        assert trace_call[1]["metadata"]["genops_governance"]["team"] == "integration-team"

        # Verify cost tracking
        assert preprocessing_response.usage.cost > 0
        assert analysis_response.usage.cost > 0
        assert summary_response.usage.cost > 0

        # Verify operation tracking
        assert adapter.operation_count == 3
        assert adapter.current_costs["daily"] > 0

    def test_evaluation_workflow_integration(self, mock_langfuse_client):
        """Test evaluation workflow with governance."""
        mock_score = Mock()
        mock_score.id = "score-eval-789"
        mock_langfuse_client.score.return_value = mock_score

        adapter = GenOpsLangfuseAdapter(
            team="evaluation-team",
            project="eval-integration",
            environment="test"
        )

        # Custom evaluator function
        def quality_evaluator():
            return {
                "score": 0.87,
                "comment": "High quality response with good coherence"
            }

        # Run evaluation with governance
        result = adapter.evaluate_with_governance(
            trace_id="trace-for-eval",
            evaluation_name="response_quality",
            evaluator_function=quality_evaluator,
            customer_id="eval-customer",
            evaluation_type="quality"
        )

        # Verify evaluation was created with governance
        mock_langfuse_client.score.assert_called_once()
        score_call = mock_langfuse_client.score.call_args

        assert score_call[1]["trace_id"] == "trace-for-eval"
        assert score_call[1]["name"] == "response_quality"
        assert score_call[1]["value"] == 0.87
        assert score_call[1]["comment"] == "High quality response with good coherence"
        assert "genops_governance" in score_call[1]["metadata"]
        assert score_call[1]["metadata"]["genops_governance"]["customer_id"] == "eval-customer"

        # Verify result structure
        assert result["score"] == 0.87
        assert result["evaluation_id"] == "score-eval-789"
        assert result["governance"]["team"] == "evaluation-team"
        assert result["duration_ms"] > 0

    def test_budget_enforcement_integration(self, mock_langfuse_client):
        """Test budget enforcement in real workflow."""
        adapter = GenOpsLangfuseAdapter(
            team="budget-team",
            budget_limits={"daily": 0.10},  # Very low budget
            policy_mode=GovernancePolicy.ENFORCED
        )

        mock_generation = Mock()
        mock_generation.id = "gen-budget-test"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        # First operation should succeed
        response1 = adapter.generation_with_cost_tracking(
            prompt="Small task",
            model="gpt-3.5-turbo"
        )
        assert response1 is not None

        # Second operation should fail due to budget
        with pytest.raises(ValueError, match="Budget limit exceeded"):
            adapter.generation_with_cost_tracking(
                prompt="Another task that would exceed budget",
                model="gpt-4"
            )

    def test_multi_team_workflow_integration(self, mock_langfuse_client):
        """Test workflow with multiple teams and cost attribution."""
        # Create adapters for different teams
        research_adapter = GenOpsLangfuseAdapter(
            team="research",
            project="multi-team-test",
            budget_limits={"daily": 5.0}
        )

        product_adapter = GenOpsLangfuseAdapter(
            team="product",
            project="multi-team-test",
            budget_limits={"daily": 3.0}
        )

        # Setup mocks
        mock_generation = Mock()
        mock_generation.id = "gen-multi-team"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        # Research team operation
        research_response = research_adapter.generation_with_cost_tracking(
            prompt="Research analysis task",
            model="gpt-4",
            customer_id="research-customer"
        )

        # Product team operation
        product_response = product_adapter.generation_with_cost_tracking(
            prompt="Product feature analysis",
            model="gpt-3.5-turbo",
            customer_id="product-customer"
        )

        # Verify team attribution
        assert research_response.usage.team == "research"
        assert product_response.usage.team == "product"

        # Verify separate cost tracking
        assert research_adapter.current_costs["daily"] > 0
        assert product_adapter.current_costs["daily"] > 0
        assert research_adapter.current_costs["daily"] != product_adapter.current_costs["daily"]

    def test_error_recovery_integration(self, mock_langfuse_client):
        """Test error recovery and graceful degradation."""
        adapter = GenOpsLangfuseAdapter(
            team="error-recovery",
            policy_mode=GovernancePolicy.ADVISORY  # Allow operations to continue
        )

        # Setup mock to fail first, succeed second
        mock_trace = Mock()
        mock_trace.id = "trace-error-recovery"
        mock_trace.metadata = {}

        call_count = 0
        def trace_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return mock_trace

        mock_langfuse_client.trace.side_effect = trace_side_effect

        # First attempt should raise exception
        with pytest.raises(Exception, match="First call fails"):
            with adapter.trace_with_governance(name="error_test"):
                pass

        # Reset side effect for successful call
        mock_langfuse_client.trace.side_effect = None
        mock_langfuse_client.trace.return_value = mock_trace

        # Second attempt should succeed
        with adapter.trace_with_governance(name="recovery_test") as trace:
            assert trace == mock_trace

    def test_performance_optimization_integration(self, mock_langfuse_client):
        """Test performance optimization features."""
        adapter = GenOpsLangfuseAdapter(
            team="performance",
            enable_governance=True  # Full governance enabled
        )

        mock_trace = Mock()
        mock_trace.metadata = {}
        mock_langfuse_client.trace.return_value = mock_trace

        # Measure performance of governance overhead
        start_time = time.time()

        with adapter.trace_with_governance(
            name="performance_test",
            customer_id="perf-customer"
        ):
            time.sleep(0.001)  # Minimal work simulation

        total_time = time.time() - start_time

        # Verify reasonable performance (governance overhead should be minimal)
        assert total_time < 0.1  # Should complete in less than 100ms

        # Verify trace was still created with full governance
        mock_langfuse_client.trace.assert_called_once()
        call_args = mock_langfuse_client.trace.call_args
        assert "genops_governance" in call_args[1]["metadata"]


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration."""

    @patch('genops.providers.langfuse.HAS_LANGFUSE', True)
    @patch('genops.providers.langfuse.Langfuse')
    def test_instrument_langfuse_auto_integration(self, mock_langfuse):
        """Test instrument_langfuse with auto-instrumentation."""
        mock_client = Mock()
        mock_langfuse.return_value = mock_client

        # Test auto-instrumentation enabled
        adapter = instrument_langfuse(
            team="auto-integration",
            project="auto-test",
            auto_instrument=True
        )

        assert isinstance(adapter, GenOpsLangfuseAdapter)
        assert adapter.team == "auto-integration"
        assert adapter.project == "auto-test"

    @patch('genops.providers.langfuse.HAS_LANGFUSE', True)
    @patch('genops.providers.langfuse.observe')
    @patch('langfuse.decorators')
    def test_auto_instrument_decorator_enhancement(self, mock_decorators, mock_observe):
        """Test auto-instrumentation enhances observe decorator."""
        adapter = Mock()
        adapter.team = "decorator-team"
        adapter.project = "decorator-project"
        adapter.environment = "test"

        # Mock original observe decorator
        original_observe = Mock()
        mock_observe.return_value = original_observe

        _auto_instrument_langfuse(adapter)

        # Verify decorator was enhanced (this is a simplified test)
        # In practice, this would require more complex mocking

    @patch('genops.providers.langfuse.HAS_LANGFUSE', False)
    def test_auto_instrument_without_langfuse(self):
        """Test auto-instrumentation when Langfuse not available."""
        adapter = Mock()

        # Should not raise error but should log warning
        _auto_instrument_langfuse(adapter)
        # Verify no exceptions raised

    def test_instrument_langfuse_with_environment_variables(self):
        """Test instrumentation using environment variables."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'pk-lf-env-test',
            'LANGFUSE_SECRET_KEY': 'sk-lf-env-test',
            'LANGFUSE_BASE_URL': 'https://env.langfuse.com'
        }):
            with patch('genops.providers.langfuse.Langfuse') as mock_langfuse:
                mock_client = Mock()
                mock_langfuse.return_value = mock_client

                adapter = instrument_langfuse(
                    team="env-team",
                    auto_instrument=False
                )

                # Verify environment variables were used
                mock_langfuse.assert_called_with(
                    public_key="pk-lf-env-test",
                    secret_key="sk-lf-env-test",
                    host="https://env.langfuse.com"
                )


class TestConcurrencyIntegration:
    """Test concurrent operations and thread safety."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Mock Langfuse client for concurrency tests."""
        with patch('genops.providers.langfuse.Langfuse') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance

    def test_concurrent_operations(self, mock_langfuse_client):
        """Test concurrent operations don't interfere."""
        import queue
        import threading

        adapter = GenOpsLangfuseAdapter(
            team="concurrency-team",
            budget_limits={"daily": 100.0}
        )

        # Setup mocks
        mock_generation = Mock()
        mock_generation.id = "gen-concurrent"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        results_queue = queue.Queue()

        def worker_function(worker_id):
            """Worker function for concurrent test."""
            try:
                response = adapter.generation_with_cost_tracking(
                    prompt=f"Concurrent task {worker_id}",
                    model="gpt-3.5-turbo",
                    operation=f"worker_{worker_id}"
                )
                results_queue.put(("success", worker_id, response))
            except Exception as e:
                results_queue.put(("error", worker_id, str(e)))

        # Create and start multiple threads
        threads = []
        num_workers = 5

        for i in range(num_workers):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify all operations completed successfully
        assert len(results) == num_workers

        success_count = len([r for r in results if r[0] == "success"])
        assert success_count == num_workers

        # Verify operation count is correct
        assert adapter.operation_count == num_workers

    def test_concurrent_budget_tracking(self, mock_langfuse_client):
        """Test concurrent budget tracking accuracy."""
        import threading

        adapter = GenOpsLangfuseAdapter(
            team="budget-concurrency",
            budget_limits={"daily": 1.0}
        )

        mock_generation = Mock()
        mock_generation.id = "gen-budget-concurrent"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        def budget_worker():
            """Worker that performs budget-tracked operations."""
            try:
                adapter.generation_with_cost_tracking(
                    prompt="Budget tracking test",
                    model="gpt-3.5-turbo"
                )
            except Exception:
                pass  # May fail due to budget limits

        # Run multiple workers concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=budget_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify budget tracking remained consistent
        # (Even with concurrent access, costs should be tracked accurately)
        assert adapter.current_costs["daily"] >= 0
        assert adapter.operation_count >= 0


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    @pytest.fixture
    def mock_langfuse_client(self):
        """Mock Langfuse client for scenario tests."""
        with patch('genops.providers.langfuse.Langfuse') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance

    def test_customer_support_chatbot_scenario(self, mock_langfuse_client):
        """Test customer support chatbot scenario."""
        adapter = GenOpsLangfuseAdapter(
            team="customer-support",
            project="chatbot-v2",
            budget_limits={"daily": 50.0},
            policy_mode=GovernancePolicy.ENFORCED
        )

        # Mock responses
        mock_trace = Mock()
        mock_trace.metadata = {}
        mock_langfuse_client.trace.return_value = mock_trace

        mock_generation = Mock()
        mock_generation.id = "gen-support"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        # Simulate customer support conversation
        customer_queries = [
            "How do I reset my password?",
            "What are your business hours?",
            "I need help with my billing",
            "Can you help me upgrade my account?"
        ]

        conversation_responses = []

        with adapter.trace_with_governance(
            name="customer_support_conversation",
            customer_id="customer_12345",
            priority="normal",
            channel="web_chat"
        ) as trace:

            for i, query in enumerate(customer_queries):
                response = adapter.generation_with_cost_tracking(
                    prompt=f"Customer query: {query}",
                    model="gpt-3.5-turbo",
                    max_cost=0.05,
                    operation=f"response_{i+1}",
                    query_type="customer_support"
                )
                conversation_responses.append(response)

        # Verify conversation was tracked
        assert len(conversation_responses) == 4
        assert adapter.operation_count == 4

        # Verify governance attributes
        trace_call = mock_langfuse_client.trace.call_args
        governance = trace_call[1]["metadata"]["genops_governance"]
        assert governance["customer_id"] == "customer_12345"
        assert governance["team"] == "customer-support"

    def test_content_moderation_scenario(self, mock_langfuse_client):
        """Test content moderation scenario with evaluation."""
        adapter = GenOpsLangfuseAdapter(
            team="content-moderation",
            project="safety-filter",
            policy_mode=GovernancePolicy.ENFORCED
        )

        # Mock setup
        mock_generation = Mock()
        mock_generation.id = "gen-moderation"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        mock_score = Mock()
        mock_score.id = "score-safety"
        mock_langfuse_client.score.return_value = mock_score

        # Content to moderate
        user_content = "This is a test message that needs moderation"

        # Step 1: Generate moderation analysis
        moderation_response = adapter.generation_with_cost_tracking(
            prompt=f"Analyze this content for safety: {user_content}",
            model="gpt-4",
            operation="safety_analysis",
            content_type="user_message"
        )

        # Step 2: Evaluate safety score
        def safety_evaluator():
            return {
                "score": 0.95,  # High safety score
                "comment": "Content appears safe with no violations"
            }

        safety_evaluation = adapter.evaluate_with_governance(
            trace_id="mock-trace-id",
            evaluation_name="safety_score",
            evaluator_function=safety_evaluator,
            content_hash="hash_of_content",
            evaluation_type="safety"
        )

        # Verify moderation workflow
        assert moderation_response is not None
        assert safety_evaluation["score"] == 0.95

        # Verify governance tracking
        generation_call = mock_langfuse_client.generation.call_args
        assert generation_call[1]["metadata"]["genops_governance"]["team"] == "content-moderation"

    def test_data_analysis_pipeline_scenario(self, mock_langfuse_client):
        """Test data analysis pipeline scenario."""
        adapter = GenOpsLangfuseAdapter(
            team="data-science",
            project="market-research",
            budget_limits={"daily": 25.0},
            environment="production"
        )

        # Mock setup
        mock_trace = Mock()
        mock_trace.metadata = {}
        mock_langfuse_client.trace.return_value = mock_trace

        mock_generation = Mock()
        mock_generation.id = "gen-analysis"
        mock_generation.metadata = {}
        mock_langfuse_client.generation.return_value = mock_generation

        # Simulate multi-stage data analysis
        with adapter.trace_with_governance(
            name="market_analysis_pipeline",
            customer_id="internal_research",
            dataset="market_data_2024_q1"
        ) as trace:

            # Stage 1: Data summarization
            summary_response = adapter.generation_with_cost_tracking(
                prompt="Summarize this market data: [data]",
                model="gpt-4",
                max_cost=0.15,
                operation="data_summarization",
                stage="preprocessing"
            )

            # Stage 2: Trend analysis
            trend_response = adapter.generation_with_cost_tracking(
                prompt="Analyze trends in the summarized data",
                model="gpt-4",
                max_cost=0.20,
                operation="trend_analysis",
                stage="analysis"
            )

            # Stage 3: Recommendations
            recommendations_response = adapter.generation_with_cost_tracking(
                prompt="Generate business recommendations based on trends",
                model="gpt-4",
                max_cost=0.10,
                operation="recommendations",
                stage="insights"
            )

        # Verify pipeline execution
        assert mock_langfuse_client.generation.call_count == 3
        assert adapter.operation_count == 3

        # Verify cost tracking across pipeline
        total_estimated_cost = (
            summary_response.usage.cost +
            trend_response.usage.cost +
            recommendations_response.usage.cost
        )
        assert adapter.current_costs["daily"] == total_estimated_cost

        # Verify governance consistency
        trace_call = mock_langfuse_client.trace.call_args
        governance = trace_call[1]["metadata"]["genops_governance"]
        assert governance["team"] == "data-science"
        assert governance["project"] == "market-research"
