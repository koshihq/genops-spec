"""
Integration tests for GenOps PromptLayer provider.

Tests end-to-end integration scenarios including:
- Complete workflow integration
- Real API interactions (when keys available)
- Multi-step operation tracking
- Cross-provider compatibility
- Performance benchmarking
- Error recovery patterns
"""

import os
import time
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.promptlayer import (
        GenOpsPromptLayerAdapter,
        GovernancePolicy,
        auto_instrument,
        get_current_adapter,
        instrument_promptlayer,  # noqa: F401
    )

    PROMPTLAYER_AVAILABLE = True
except ImportError:
    PROMPTLAYER_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(
    not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available"
)
class TestPromptLayerEndToEndIntegration:
    """End-to-end integration tests for PromptLayer with GenOps."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.test_api_key = os.getenv("PROMPTLAYER_API_KEY", "pl-test-key")
        self.has_real_api_key = os.getenv("PROMPTLAYER_API_KEY") is not None

    @pytest.mark.skipif(
        not os.getenv("PROMPTLAYER_API_KEY"), reason="Real API key required"
    )
    def test_real_promptlayer_workflow(self):
        """Test complete workflow with real PromptLayer API."""
        adapter = GenOpsPromptLayerAdapter(
            promptlayer_api_key=self.test_api_key,
            team="integration-test",
            project="e2e-testing",
            daily_budget_limit=0.50,
            max_operation_cost=0.10,
        )

        # Test complete workflow
        with adapter.track_prompt_operation(
            prompt_name="integration_test_prompt",
            operation_type="e2e_test",
            customer_id="test-customer-123",
        ) as span:
            # Execute prompt with governance
            result = adapter.run_prompt_with_governance(
                prompt_name="integration_test_prompt",
                input_variables={"query": "Integration test query"},
                tags=["integration", "e2e"],
            )

            # Verify governance context
            assert "governance" in result
            assert result["governance"]["team"] == "integration-test"
            assert result["governance"]["project"] == "e2e-testing"

            # Update span with costs
            span.update_cost(0.025)

        # Verify tracking was updated
        assert adapter.daily_usage >= 0.025
        assert adapter.operation_count >= 1

        # Test metrics retrieval
        metrics = adapter.get_metrics()
        assert metrics["team"] == "integration-test"
        assert metrics["daily_usage"] >= 0.025
        assert metrics["budget_remaining"] <= 0.475

    def test_mock_promptlayer_workflow(self):
        """Test complete workflow with mocked PromptLayer."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
            # Setup comprehensive mock
            mock_client = Mock()
            mock_pl_class.return_value = mock_client
            mock_client.run.return_value = {
                "response": "Mock response for integration test",
                "usage": {"input_tokens": 45, "output_tokens": 67, "total_tokens": 112},
            }

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-mock-key",
                team="mock-team",
                project="mock-project",
                enable_governance=True,
            )

            # Execute multi-step workflow
            results = []
            total_cost = 0.0

            # Step 1: Intent analysis
            with adapter.track_prompt_operation(
                prompt_name="intent_classifier",
                operation_type="classification",
                operation_name="analyze_intent",
            ) as intent_span:
                result1 = adapter.run_prompt_with_governance(
                    prompt_name="intent_classifier",
                    input_variables={"user_input": "I need help with billing"},
                    tags=["intent", "classification"],
                )

                intent_span.update_cost(0.015)
                intent_span.add_attributes({"intent_detected": "billing_inquiry"})
                results.append(result1)
                total_cost += 0.015

            # Step 2: Response generation
            with adapter.track_prompt_operation(
                prompt_name="response_generator",
                operation_type="generation",
                operation_name="generate_response",
            ) as response_span:
                result2 = adapter.run_prompt_with_governance(
                    prompt_name="response_generator",
                    input_variables={
                        "intent": "billing_inquiry",
                        "context": "customer support",
                    },
                    tags=["generation", "billing"],
                )

                response_span.update_cost(0.032)
                response_span.add_attributes({"response_quality": "high"})
                results.append(result2)
                total_cost += 0.032

            # Verify workflow execution
            assert len(results) == 2
            assert all("governance" in result for result in results)
            assert adapter.daily_usage == total_cost
            assert adapter.operation_count == 2

            # Verify mock calls
            assert mock_client.run.call_count == 2

            # Verify governance attributes in calls
            for call in mock_client.run.call_args_list:
                args, kwargs = call
                assert "team:mock-team" in kwargs["tags"]
                assert "project:mock-project" in kwargs["tags"]

    def test_concurrent_operations(self):
        """Test concurrent operation tracking."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
            mock_client = Mock()
            mock_pl_class.return_value = mock_client
            mock_client.run.return_value = {"response": "Concurrent test response"}

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-concurrent-test",
                team="concurrent-team",
                daily_budget_limit=1.0,
            )

            # Start multiple concurrent operations
            contexts = []
            spans = []

            for i in range(3):
                ctx = adapter.track_prompt_operation(f"concurrent_prompt_{i}")
                span = ctx.__enter__()
                contexts.append(ctx)
                spans.append(span)

            # All should be active
            assert len(adapter.active_spans) == 3
            assert all(span.operation_id in adapter.active_spans for span in spans)

            # Update costs for each
            costs = [0.15, 0.22, 0.08]
            for span, cost in zip(spans, costs):
                span.update_cost(cost)
                adapter.run_prompt_with_governance(
                    prompt_name=span.prompt_name,
                    input_variables={"test": f"concurrent_{span.operation_id}"},
                )

            # Close all contexts
            for ctx in contexts:
                ctx.__exit__(None, None, None)

            # Verify final state
            assert len(adapter.active_spans) == 0
            assert adapter.daily_usage == sum(costs)
            assert adapter.operation_count == 3

    def test_error_recovery_patterns(self):
        """Test error handling and recovery in integrated scenarios."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
            mock_client = Mock()
            mock_pl_class.return_value = mock_client

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-error-test", team="error-test-team"
            )

            # Test 1: API error recovery
            mock_client.run.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                with adapter.track_prompt_operation("error_test_1") as span:
                    adapter.run_prompt_with_governance(
                        prompt_name="error_test_1", input_variables={"test": "error"}
                    )

            # Span should still be properly finalized
            assert span.end_time is not None
            assert "error" in span.metadata

            # Test 2: Recovery after error
            mock_client.run.side_effect = None
            mock_client.run.return_value = {"response": "Recovery successful"}

            with adapter.track_prompt_operation("recovery_test") as span:
                result = adapter.run_prompt_with_governance(
                    prompt_name="recovery_test", input_variables={"test": "recovery"}
                )
                span.update_cost(0.01)

            assert "governance" in result
            assert adapter.operation_count == 2  # Both operations counted

    def test_governance_policy_integration(self):
        """Test governance policy enforcement in integrated scenarios."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
            mock_client = Mock()
            mock_pl_class.return_value = mock_client
            mock_client.run.return_value = {"response": "Policy test response"}

            # Test enforced policy with budget violation
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-policy-test",
                team="policy-team",
                governance_policy=GovernancePolicy.ENFORCED,
                daily_budget_limit=0.05,  # Very low limit
                max_operation_cost=0.02,
            )

            # First operation within limits
            with adapter.track_prompt_operation("policy_test_1") as span:
                adapter.run_prompt_with_governance(
                    prompt_name="policy_test_1",
                    input_variables={"test": "within_limits"},
                )
                span.update_cost(0.01)  # Within limits

            # Second operation should trigger budget enforcement
            with pytest.raises(ValueError, match="Daily budget limit"):
                with adapter.track_prompt_operation("policy_test_2") as span:
                    pass  # Should fail at context entry due to budget

    def test_performance_benchmarking(self):
        """Test performance characteristics of the integration."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
            mock_client = Mock()
            mock_pl_class.return_value = mock_client
            mock_client.run.return_value = {"response": "Perf test"}

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-perf-test", team="perf-team"
            )

            # Benchmark operation overhead
            num_operations = 50
            start_time = time.time()

            for i in range(num_operations):
                with adapter.track_prompt_operation(f"perf_test_{i}") as span:
                    adapter.run_prompt_with_governance(
                        prompt_name=f"perf_test_{i}", input_variables={"iteration": i}
                    )
                    span.update_cost(0.001)

            total_time = time.time() - start_time
            avg_time_per_op = (total_time / num_operations) * 1000  # ms

            # Performance assertions
            assert avg_time_per_op < 50  # Less than 50ms per operation
            assert adapter.operation_count == num_operations
            assert len(adapter.active_spans) == 0  # All cleaned up

    def test_metrics_aggregation_integration(self):
        """Test comprehensive metrics aggregation across operations."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
            mock_client = Mock()
            mock_pl_class.return_value = mock_client
            mock_client.run.return_value = {"response": "Metrics test"}

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-metrics-test",
                team="metrics-team",
                project="metrics-project",
                daily_budget_limit=5.0,
            )

            # Execute various operations with different characteristics
            operations = [
                {"name": "quick_op", "cost": 0.01, "tokens_in": 20, "tokens_out": 30},
                {
                    "name": "medium_op",
                    "cost": 0.05,
                    "tokens_in": 100,
                    "tokens_out": 150,
                },
                {
                    "name": "expensive_op",
                    "cost": 0.15,
                    "tokens_in": 300,
                    "tokens_out": 400,
                },
            ]

            for op in operations:
                with adapter.track_prompt_operation(
                    prompt_name=op["name"],
                    operation_type="metrics_test",
                    customer_id=f"customer_{op['name']}",
                ) as span:
                    adapter.run_prompt_with_governance(
                        prompt_name=op["name"],
                        input_variables={"operation": op["name"]},
                    )

                    span.update_cost(op["cost"])
                    span.update_token_usage(
                        op["tokens_in"], op["tokens_out"], "gpt-3.5-turbo"
                    )
                    span.add_attributes(
                        {
                            "operation_category": op["name"].split("_")[0],
                            "custom_metric": op["cost"] * 100,
                        }
                    )

            # Verify comprehensive metrics
            metrics = adapter.get_metrics()

            assert metrics["team"] == "metrics-team"
            assert metrics["project"] == "metrics-project"
            assert metrics["operation_count"] == 3
            assert abs(metrics["daily_usage"] - 0.21) < 0.001  # Sum of costs
            assert abs(metrics["budget_remaining"] - 4.79) < 0.001
            assert metrics["active_operations"] == 0

    def test_cross_environment_integration(self):
        """Test integration across different environments."""
        environments = ["development", "staging", "production"]

        for env in environments:
            with patch("genops.providers.promptlayer.PromptLayer") as mock_pl_class:
                mock_client = Mock()
                mock_pl_class.return_value = mock_client
                mock_client.run.return_value = {"response": f"{env} response"}

                adapter = GenOpsPromptLayerAdapter(
                    promptlayer_api_key=f"pl-{env}-key",
                    team=f"{env}-team",
                    environment=env,
                    daily_budget_limit=1.0 if env == "production" else 0.1,
                )

                with adapter.track_prompt_operation(
                    f"{env}_test_prompt", operation_type="environment_test"
                ) as span:
                    result = adapter.run_prompt_with_governance(
                        prompt_name=f"{env}_test_prompt",
                        input_variables={"environment": env},
                    )

                    span.update_cost(0.02)

                # Verify environment-specific behavior
                assert result["governance"]["team"] == f"{env}-team"
                metrics = adapter.get_metrics()
                assert metrics["environment"] == env

                # Production should have higher budget
                if env == "production":
                    assert metrics["budget_remaining"] == 0.98
                else:
                    assert metrics["budget_remaining"] == 0.08


@pytest.mark.integration
@pytest.mark.skipif(
    not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available"
)
class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration patterns."""

    def test_auto_instrument_global_setup(self):
        """Test global auto-instrumentation setup."""
        with patch(
            "genops.providers.promptlayer.GenOpsPromptLayerAdapter"
        ) as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter

            # Setup auto-instrumentation
            auto_instrument(
                promptlayer_api_key="pl-auto-key",
                team="auto-team",
                project="auto-project",
                environment="test",
            )

            # Verify adapter creation
            mock_adapter_class.assert_called_once_with(
                promptlayer_api_key="pl-auto-key",
                team="auto-team",
                project="auto-project",
                environment="test",
            )

            # Test global adapter access
            current_adapter = get_current_adapter()
            assert current_adapter == mock_adapter

    def test_auto_instrument_with_existing_code(self):
        """Test auto-instrumentation with simulated existing PromptLayer code."""
        with patch(
            "genops.providers.promptlayer.GenOpsPromptLayerAdapter"
        ) as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.get_metrics.return_value = {
                "team": "auto-team",
                "operation_count": 0,
                "daily_usage": 0.0,
            }
            mock_adapter_class.return_value = mock_adapter

            # Setup auto-instrumentation
            auto_instrument(
                team="auto-team", project="existing-code-test", daily_budget_limit=2.0
            )

            # Simulate existing PromptLayer code patterns
            # (In reality, this would intercept actual PromptLayer calls)

            current_adapter = get_current_adapter()
            assert current_adapter is not None

            metrics = current_adapter.get_metrics()
            assert metrics["team"] == "auto-team"


@pytest.mark.integration
@pytest.mark.skipif(
    not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available"
)
class TestCrossProviderCompatibility:
    """Test compatibility with other GenOps providers."""

    def test_promptlayer_with_openai_provider(self):
        """Test PromptLayer adapter alongside OpenAI provider."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl:
            mock_pl_client = Mock()
            mock_pl.return_value = mock_pl_client
            mock_pl_client.run.return_value = {"response": "Cross-provider test"}

            # Create PromptLayer adapter
            pl_adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-cross-test",
                team="cross-provider-team",
                project="compatibility-test",
            )

            # Simulate operations from both providers
            # PromptLayer operation
            with pl_adapter.track_prompt_operation("promptlayer_op") as pl_span:
                pl_result = pl_adapter.run_prompt_with_governance(
                    prompt_name="promptlayer_op",
                    input_variables={"provider": "promptlayer"},
                )
                pl_span.update_cost(0.025)

            # Verify cross-provider isolation
            assert pl_adapter.operation_count == 1
            assert pl_adapter.daily_usage == 0.025
            assert "governance" in pl_result

    def test_multiple_promptlayer_adapters(self):
        """Test multiple PromptLayer adapters for different teams/projects."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {"response": "Multi-adapter test"}

            # Create adapters for different teams
            team_a_adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-team-a",
                team="team-a",
                project="project-a",
                daily_budget_limit=1.0,
            )

            team_b_adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-team-b",
                team="team-b",
                project="project-b",
                daily_budget_limit=2.0,
            )

            # Execute operations on each adapter
            with team_a_adapter.track_prompt_operation("team_a_op") as span_a:
                team_a_adapter.run_prompt_with_governance(
                    prompt_name="team_a_op", input_variables={"team": "a"}
                )
                span_a.update_cost(0.15)

            with team_b_adapter.track_prompt_operation("team_b_op") as span_b:
                team_b_adapter.run_prompt_with_governance(
                    prompt_name="team_b_op", input_variables={"team": "b"}
                )
                span_b.update_cost(0.35)

            # Verify adapter isolation
            metrics_a = team_a_adapter.get_metrics()
            metrics_b = team_b_adapter.get_metrics()

            assert metrics_a["team"] == "team-a"
            assert metrics_a["daily_usage"] == 0.15
            assert metrics_a["budget_remaining"] == 0.85

            assert metrics_b["team"] == "team-b"
            assert metrics_b["daily_usage"] == 0.35
            assert metrics_b["budget_remaining"] == 1.65


@pytest.mark.integration
@pytest.mark.skipif(
    not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available"
)
class TestRealWorldScenarios:
    """Test realistic usage scenarios and patterns."""

    def test_customer_support_workflow(self):
        """Test realistic customer support workflow."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {"response": "Support workflow response"}

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-support-workflow",
                team="customer-support",
                project="ai-assistant",
                environment="production",
                daily_budget_limit=25.0,
                max_operation_cost=2.0,
            )

            # Simulate customer support ticket workflow
            customer_id = "customer_12345"
            ticket_id = "ticket_67890"

            total_workflow_cost = 0.0

            # Step 1: Classify customer inquiry
            with adapter.track_prompt_operation(
                prompt_name="inquiry_classifier",
                operation_type="classification",
                customer_id=customer_id,
                cost_center="support",
                tags={"ticket_id": ticket_id, "step": "classification"},
            ) as classify_span:
                adapter.run_prompt_with_governance(
                    prompt_name="inquiry_classifier",
                    input_variables={
                        "customer_message": "I cannot access my account after password reset",
                        "customer_tier": "premium",
                    },
                )

                classify_span.update_cost(0.08)
                classify_span.add_attributes({"classification": "account_access"})
                total_workflow_cost += 0.08

            # Step 2: Generate initial response
            with adapter.track_prompt_operation(
                prompt_name="response_generator",
                operation_type="generation",
                customer_id=customer_id,
                tags={"ticket_id": ticket_id, "step": "initial_response"},
            ) as response_span:
                initial_response = adapter.run_prompt_with_governance(
                    prompt_name="response_generator",
                    input_variables={
                        "classification": "account_access",
                        "customer_tier": "premium",
                        "urgency": "high",
                    },
                )

                response_span.update_cost(0.125)
                response_span.add_attributes({"response_type": "initial"})
                total_workflow_cost += 0.125

            # Step 3: Quality check
            with adapter.track_prompt_operation(
                prompt_name="quality_checker",
                operation_type="validation",
                customer_id=customer_id,
                tags={"ticket_id": ticket_id, "step": "quality_check"},
            ) as quality_span:
                adapter.run_prompt_with_governance(
                    prompt_name="quality_checker",
                    input_variables={
                        "response": initial_response.get("response", ""),
                        "classification": "account_access",
                    },
                )

                quality_span.update_cost(0.045)
                quality_span.add_attributes({"quality_score": 0.92})
                total_workflow_cost += 0.045

            # Verify complete workflow tracking
            assert adapter.operation_count == 3
            assert abs(adapter.daily_usage - total_workflow_cost) < 0.001

            metrics = adapter.get_metrics()
            assert metrics["team"] == "customer-support"
            assert metrics["project"] == "ai-assistant"
            assert metrics["environment"] == "production"

            # Verify all operations were properly attributed
            assert mock_client.run.call_count == 3

            # Check governance context in all calls
            for call in mock_client.run.call_args_list:
                _, kwargs = call
                tags = kwargs.get("tags", [])
                assert any("team:customer-support" in tag for tag in tags)
                assert any("customer:customer_12345" in tag for tag in tags)

    def test_content_generation_pipeline(self):
        """Test content generation pipeline with governance."""
        with patch("genops.providers.promptlayer.PromptLayer") as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {"response": "Content pipeline response"}

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key="pl-content-pipeline",
                team="content-team",
                project="content-generation",
                daily_budget_limit=15.0,
            )

            # Content generation pipeline
            content_request = {
                "topic": "AI governance best practices",
                "audience": "technical_leaders",
                "length": "medium",
                "tone": "professional",
            }

            pipeline_results = {}

            # Step 1: Research and outline
            with adapter.track_prompt_operation(
                prompt_name="content_researcher",
                operation_type="research",
                tags={"content_type": "article", "stage": "research"},
            ) as research_span:
                research = adapter.run_prompt_with_governance(
                    prompt_name="content_researcher", input_variables=content_request
                )

                research_span.update_cost(0.18)
                pipeline_results["research"] = research

            # Step 2: Draft generation
            with adapter.track_prompt_operation(
                prompt_name="content_writer",
                operation_type="generation",
                tags={"content_type": "article", "stage": "draft"},
            ) as draft_span:
                draft = adapter.run_prompt_with_governance(
                    prompt_name="content_writer",
                    input_variables={
                        **content_request,
                        "research_data": pipeline_results["research"],
                    },
                )

                draft_span.update_cost(0.285)
                pipeline_results["draft"] = draft

            # Step 3: Editorial review
            with adapter.track_prompt_operation(
                prompt_name="content_editor",
                operation_type="review",
                tags={"content_type": "article", "stage": "editing"},
            ) as edit_span:
                edited = adapter.run_prompt_with_governance(
                    prompt_name="content_editor",
                    input_variables={
                        "draft_content": pipeline_results["draft"],
                        "style_guide": "technical_blog",
                    },
                )

                edit_span.update_cost(0.142)
                pipeline_results["final"] = edited

            # Verify pipeline completion
            total_cost = 0.18 + 0.285 + 0.142
            assert abs(adapter.daily_usage - total_cost) < 0.001
            assert adapter.operation_count == 3
            assert len(pipeline_results) == 3

            # Verify governance tracking throughout pipeline
            metrics = adapter.get_metrics()
            assert metrics["team"] == "content-team"
            assert metrics["budget_remaining"] == 15.0 - total_cost
