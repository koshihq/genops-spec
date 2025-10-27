"""End-to-end integration tests for GenOps AI."""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genops
from genops.core.policy import PolicyResult, register_policy
from genops.providers.anthropic import GenOpsAnthropicAdapter
from genops.providers.openai import GenOpsOpenAIAdapter
from utils.mock_providers import MockAnthropicClient, MockOpenAIClient


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_governance_workflow(self, mock_otel_setup, cleanup_test_state):
        """Test complete governance workflow from init to policy enforcement."""

        # Step 1: Initialize GenOps AI
        with patch(
            "genops.auto_instrumentation.GenOpsInstrumentor._check_provider_availability",
            return_value=True,
        ):
            with patch(
                "genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider",
                return_value=True,
            ):
                instrumentor = genops.init(
                    service_name="e2e-test-service",
                    environment="testing",
                    default_team="integration-team",
                    default_project="e2e-testing",
                    exporter_type="console",
                )

        # Verify initialization
        status_info = genops.status()
        assert status_info["initialized"] is True
        assert status_info["default_attributes"]["team"] == "integration-team"
        assert status_info["default_attributes"]["project"] == "e2e-testing"

        # Step 2: Register governance policies
        register_policy(
            name="cost_control",
            description="Control AI operation costs",
            enforcement_level=PolicyResult.BLOCKED,
            max_cost=5.0,
        )

        register_policy(
            name="content_safety",
            description="Filter unsafe content",
            enforcement_level=PolicyResult.WARNING,
            blocked_patterns=["violence", "explicit"],
        )

        # Step 3: Use manual instrumentation with policies
        @genops.track_usage(
            operation_name="customer_support_query", feature="chat_support"
        )
        @genops.enforce_policy(["cost_control", "content_safety"])
        def process_customer_query(query: str) -> str:
            # Simulate AI processing
            return f"AI response to: {query}"

        # Test successful operation (under cost limit, safe content)
        result = process_customer_query("How can I reset my password?")
        assert "AI response to:" in result

        # Verify telemetry was recorded
        spans = mock_otel_setup.get_finished_spans()
        governance_spans = [s for s in spans if "customer_support_query" in s.name]
        assert len(governance_spans) > 0

        # Step 4: Test policy enforcement (this would raise exception in real scenario)
        # For test purposes, we'll verify the policy evaluation logic

        # Step 5: Uninstrument
        genops.uninstrument()

        # Verify uninstrumentation
        final_status = genops.status()
        assert final_status["initialized"] is False

    def test_provider_integration_openai(self, mock_openai_import, mock_otel_setup):
        """Test integration with OpenAI provider."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        # Execute OpenAI call with governance tracking
        response = adapter.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"},
            ],
            temperature=0.7,
            # Governance attributes
            team="ai-research",
            project="ml-education",
            feature="q_and_a",
            customer_id="student_123",
        )

        # Verify response
        assert response is not None
        assert hasattr(response, "choices")
        assert response.usage.total_tokens > 0

        # Verify governance telemetry
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attrs = span.attributes

        # Check all governance attributes were recorded
        assert attrs["genops.provider"] == "openai"
        assert attrs["genops.model"] == "gpt-3.5-turbo"
        assert attrs["genops.request.temperature"] == 0.7
        assert "genops.cost.total" in attrs
        assert attrs["genops.cost.currency"] == "USD"
        assert "genops.tokens.input" in attrs
        assert "genops.tokens.output" in attrs

    def test_provider_integration_anthropic(
        self, mock_anthropic_import, mock_otel_setup
    ):
        """Test integration with Anthropic provider."""
        mock_client = MockAnthropicClient()
        adapter = GenOpsAnthropicAdapter(client=mock_client)

        # Execute Claude call with governance tracking
        response = adapter.messages_create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system="You are an expert data analyst.",
            messages=[
                {
                    "role": "user",
                    "content": "Analyze this dataset and provide insights.",
                }
            ],
            temperature=0.3,
            # Governance attributes
            team="data-science",
            project="analytics-platform",
            feature="data_analysis",
            customer_id="enterprise_456",
        )

        # Verify response
        assert response is not None
        assert hasattr(response, "content")
        assert response.usage.input_tokens > 0

        # Verify governance telemetry
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attrs = span.attributes

        # Check governance attributes
        assert attrs["genops.provider"] == "anthropic"
        assert attrs["genops.model"] == "claude-3-sonnet-20240229"
        assert attrs["genops.request.temperature"] == 0.3
        assert attrs["genops.request.max_tokens"] == 1024
        assert "genops.cost.total" in attrs
        assert "genops.tokens.input" in attrs

    def test_multi_provider_governance(
        self,
        mock_openai_import,
        mock_anthropic_import,
        mock_otel_setup,
        cleanup_test_state,
    ):
        """Test governance across multiple providers."""

        # Initialize with both providers
        with patch(
            "genops.auto_instrumentation.GenOpsInstrumentor._check_provider_availability",
            return_value=True,
        ):
            with patch(
                "genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider",
                return_value=True,
            ):
                genops.init(
                    service_name="multi-provider-test",
                    default_team="ai-platform",
                    default_project="multi-modal-ai",
                )

        # Create provider adapters
        openai_client = MockOpenAIClient()
        openai_adapter = GenOpsOpenAIAdapter(client=openai_client)

        anthropic_client = MockAnthropicClient()
        anthropic_adapter = GenOpsAnthropicAdapter(client=anthropic_client)

        # Execute operations with both providers
        openai_response = openai_adapter.chat_completions_create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Generate creative content"}],
            feature="content_generation",
        )

        anthropic_response = anthropic_adapter.messages_create(
            model="claude-3-opus-20240229",
            max_tokens=2048,
            messages=[{"role": "user", "content": "Analyze complex reasoning"}],
            feature="reasoning_analysis",
        )

        # Verify both operations were tracked
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 2

        # Find spans by provider
        openai_span = next(
            s for s in spans if s.attributes.get("genops.provider") == "openai"
        )
        anthropic_span = next(
            s for s in spans if s.attributes.get("genops.provider") == "anthropic"
        )

        # Verify governance attributes are consistent
        assert openai_span.attributes["genops.team"] == "ai-platform"
        assert anthropic_span.attributes["genops.team"] == "ai-platform"
        assert openai_span.attributes["genops.project"] == "multi-modal-ai"
        assert anthropic_span.attributes["genops.project"] == "multi-modal-ai"

        # Verify provider-specific attributes
        assert openai_span.attributes["genops.model"] == "gpt-4"
        assert anthropic_span.attributes["genops.model"] == "claude-3-opus-20240229"

    def test_policy_enforcement_integration(self, mock_otel_setup, cleanup_test_state):
        """Test policy enforcement in realistic scenario."""

        # Register realistic policies
        register_policy(
            name="team_budget_control",
            description="Control per-team AI spending",
            enforcement_level=PolicyResult.WARNING,
            max_cost=10.0,
        )

        register_policy(
            name="production_safety",
            description="Safety controls for production",
            enforcement_level=PolicyResult.BLOCKED,
            blocked_patterns=["confidential", "internal"],
        )

        # Test function with policy enforcement
        @genops.track_usage(
            operation_name="document_processing",
            team="documents-team",
            project="doc-ai",
        )
        @genops.enforce_policy(["team_budget_control", "production_safety"])
        def process_document(content: str, cost: float) -> str:
            # Simulate document processing
            return f"Processed document: {len(content)} characters"

        # Test 1: Safe operation under budget
        result = process_document("Safe document content for processing", cost=2.0)
        assert "Processed document:" in result

        # Test 2: Operation that would trigger warning (over budget)
        # Note: In real implementation, this would log a warning but continue
        with patch(
            "genops.core.policy._global_policy_engine.evaluate_policy"
        ) as mock_evaluate:
            from genops.core.policy import PolicyEvaluationResult

            mock_evaluate.return_value = PolicyEvaluationResult(
                policy_name="team_budget_control",
                result=PolicyResult.WARNING,
                reason="Budget exceeded",
                metadata={"cost": 15.0, "limit": 10.0},
            )

            # This should execute with warning
            result = process_document("Another document", cost=15.0)
            assert "Processed document:" in result

        # Verify telemetry recorded policy evaluations
        spans = mock_otel_setup.get_finished_spans()
        doc_spans = [s for s in spans if "document_processing" in s.name]
        assert len(doc_spans) > 0

    def test_cost_attribution_workflow(self, mock_openai_import, mock_otel_setup):
        """Test complete cost attribution workflow."""
        mock_client = MockOpenAIClient()
        adapter = GenOpsOpenAIAdapter(client=mock_client)

        # Simulate multiple operations for different customers/features
        test_scenarios = [
            {
                "customer_id": "customer_a",
                "feature": "chat_support",
                "team": "support",
                "model": "gpt-3.5-turbo",
                "message": "Help with account issues",
            },
            {
                "customer_id": "customer_b",
                "feature": "content_generation",
                "team": "marketing",
                "model": "gpt-4",
                "message": "Generate marketing copy",
            },
            {
                "customer_id": "customer_a",
                "feature": "data_analysis",
                "team": "analytics",
                "model": "gpt-4",
                "message": "Analyze user behavior data",
            },
        ]

        for scenario in test_scenarios:
            adapter.chat_completions_create(
                model=scenario["model"],
                messages=[{"role": "user", "content": scenario["message"]}],
                # Governance attributes for cost attribution
                customer_id=scenario["customer_id"],
                feature=scenario["feature"],
                team=scenario["team"],
            )

        # Verify cost attribution telemetry
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 3

        # Group by customer
        customer_a_spans = [
            s for s in spans if s.attributes.get("genops.customer_id") == "customer_a"
        ]
        customer_b_spans = [
            s for s in spans if s.attributes.get("genops.customer_id") == "customer_b"
        ]

        assert len(customer_a_spans) == 2  # chat_support + data_analysis
        assert len(customer_b_spans) == 1  # content_generation

        # Verify each span has complete cost attribution data
        for span in spans:
            attrs = span.attributes
            assert "genops.cost.total" in attrs
            assert "genops.customer_id" in attrs
            assert "genops.feature" in attrs
            assert "genops.team" in attrs
            assert "genops.model" in attrs

        # Verify different models have different costs
        gpt35_spans = [
            s for s in spans if "gpt-3.5-turbo" in s.attributes.get("genops.model", "")
        ]
        gpt4_spans = [
            s for s in spans if "gpt-4" in s.attributes.get("genops.model", "")
        ]

        assert len(gpt35_spans) == 1
        assert len(gpt4_spans) == 2

        # GPT-4 should be more expensive than GPT-3.5-turbo
        gpt35_cost = gpt35_spans[0].attributes["genops.cost.total"]
        gpt4_cost = gpt4_spans[0].attributes["genops.cost.total"]
        assert gpt4_cost > gpt35_cost

    def test_error_handling_and_recovery(self, mock_openai_import, mock_otel_setup):
        """Test error handling and telemetry in failure scenarios."""

        # Create a client that will fail
        failing_client = MockOpenAIClient(fail_requests=True)
        adapter = GenOpsOpenAIAdapter(client=failing_client)

        # Test that failures are properly tracked
        with pytest.raises(Exception) as exc_info:
            adapter.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "This will fail"}],
                team="testing",
                feature="error_handling",
            )

        assert "Mock API error" in str(exc_info.value)

        # Verify error telemetry
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 1

        error_span = spans[0]
        assert error_span.status.status_code.name == "ERROR"

        # Verify governance attributes are still recorded even on failure
        attrs = error_span.attributes
        assert attrs["genops.provider"] == "openai"
        assert attrs["genops.team"] == "testing"
        assert attrs["genops.feature"] == "error_handling"

    def test_context_manager_integration(self, mock_otel_setup):
        """Test context manager integration with governance tracking."""

        # Test nested context managers with governance
        with genops.track(
            operation_name="batch_processing",
            team="data-platform",
            project="ml-pipeline",
        ) as outer_span:
            # Record batch-level governance data
            outer_span.record_budget(
                budget_name="monthly_ml_budget",
                allocated=1000.0,
                consumed=150.0,
                remaining=850.0,
            )

            # Nested operation
            with genops.track(
                operation_name="individual_inference", feature="prediction"
            ) as inner_span:
                # Record individual operation data
                inner_span.record_cost(cost=2.5, currency="USD")
                inner_span.record_evaluation(
                    metric_name="accuracy", score=0.92, evaluator="automated"
                )

        # Verify nested telemetry
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 2

        # Spans are finished in reverse order (inner first, then outer)
        inner_span, outer_span = spans

        # Verify outer span governance data
        outer_attrs = outer_span.attributes
        assert outer_attrs["genops.operation.name"] == "batch_processing"
        assert outer_attrs["genops.team"] == "data-platform"
        assert outer_attrs["genops.budget.name"] == "monthly_ml_budget"
        assert outer_attrs["genops.budget.allocated"] == 1000.0

        # Verify inner span governance data
        inner_attrs = inner_span.attributes
        assert inner_attrs["genops.operation.name"] == "individual_inference"
        assert inner_attrs["genops.feature"] == "prediction"
        assert inner_attrs["genops.cost.total"] == 2.5
        assert inner_attrs["genops.eval.metric"] == "accuracy"
        assert inner_attrs["genops.eval.score"] == 0.92
