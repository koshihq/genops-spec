#!/usr/bin/env python3
"""
Integration tests for Together AI provider.

Tests end-to-end workflows, real API interactions (when available),
context manager lifecycle, and complete governance scenarios.
"""

import os
import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.genops.providers.together import (
        GenOpsTogetherAdapter,
        TogetherModel,
        TogetherTaskType,
        auto_instrument,
    )
    from src.genops.providers.together_pricing import TogetherPricingCalculator
    from src.genops.providers.together_validation import validate_together_setup
except ImportError as e:
    pytest.skip(f"Together AI provider not available: {e}", allow_module_level=True)


@pytest.fixture
def mock_together_client():
    """Fixture providing mocked Together client."""
    with patch('src.genops.providers.together.Together') as mock:
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[{"message": {"content": "Test response"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        client.models.list.return_value = MagicMock(data=[
            {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"},
            {"id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"}
        ])
        mock.return_value = client
        yield client


@pytest.fixture
def test_adapter():
    """Fixture providing test adapter."""
    return GenOpsTogetherAdapter(
        team="integration-test",
        project="test-suite",
        environment="test",
        daily_budget_limit=5.0,
        governance_policy="advisory"
    )


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_chat_workflow(self, mock_together_client, test_adapter):
        """Test complete chat completion workflow."""
        # Test single chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]

        result = test_adapter.chat_with_governance(
            messages=messages,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=50,
            temperature=0.7
        )

        # Verify result structure
        assert hasattr(result, 'response')
        assert hasattr(result, 'tokens_used')
        assert hasattr(result, 'cost')
        assert hasattr(result, 'model_used')
        assert hasattr(result, 'governance_attributes')

        assert result.response == "Test response"
        assert result.tokens_used == 30
        assert isinstance(result.cost, Decimal)
        assert result.cost > 0
        assert result.model_used == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    def test_session_tracking_workflow(self, mock_together_client, test_adapter):
        """Test complete session tracking workflow."""
        session_id = "test-session-workflow"

        with test_adapter.track_session(session_id) as session:
            # First operation
            result1 = test_adapter.chat_with_governance(
                messages=[{"role": "user", "content": "First message"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                session_id=session.session_id
            )

            # Second operation in same session
            result2 = test_adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Second message"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                session_id=session.session_id
            )

            # Verify session tracking
            assert session.session_id == session_id
            assert session.total_operations >= 2
            assert session.total_cost > 0
            assert session.end_time is None  # Still in context

        # After context exit
        assert session.end_time is not None
        assert session.end_time > session.start_time

    def test_multi_model_workflow(self, mock_together_client, test_adapter):
        """Test workflow with multiple models."""
        models_to_test = [
            TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        ]

        results = []

        with test_adapter.track_session("multi-model-test") as session:
            for model in models_to_test:
                result = test_adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Test {model.value}"}],
                    model=model,
                    session_id=session.session_id,
                    feature=f"model-test-{model.name.lower()}"
                )
                results.append(result)

            # Verify all models were used
            assert len(results) == len(models_to_test)
            assert session.total_operations == len(models_to_test)

    def test_task_type_workflow(self, mock_together_client, test_adapter):
        """Test workflow with different task types."""
        task_scenarios = [
            {
                "task_type": TogetherTaskType.CHAT,
                "model": TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                "content": "Hello, how are you?"
            },
            {
                "task_type": TogetherTaskType.CODE_GENERATION,
                "model": TogetherModel.DEEPSEEK_CODER_V2,
                "content": "Write a Python function to reverse a string"
            },
            {
                "task_type": TogetherTaskType.REASONING,
                "model": TogetherModel.DEEPSEEK_R1,
                "content": "Solve this logic puzzle step by step"
            }
        ]

        results = []

        for scenario in task_scenarios:
            result = test_adapter.chat_with_governance(
                messages=[{"role": "user", "content": scenario["content"]}],
                model=scenario["model"],
                task_type=scenario["task_type"],
                max_tokens=100
            )
            results.append(result)

            # Verify task type is tracked
            assert result.task_type == scenario["task_type"]

        assert len(results) == len(task_scenarios)


class TestContextManagerLifecycle:
    """Test context manager lifecycle and resource management."""

    def test_session_context_normal_completion(self, test_adapter):
        """Test session context manager normal completion."""
        session_id = "lifecycle-test-normal"

        with test_adapter.track_session(session_id) as session:
            assert session.session_id == session_id
            assert session.start_time > 0
            assert session.end_time is None
            assert session.total_operations == 0
            assert session.total_cost == 0

        # After completion
        assert session.end_time is not None
        assert session.end_time >= session.start_time

    def test_session_context_with_exception(self, test_adapter):
        """Test session context manager with exception handling."""
        session_id = "lifecycle-test-exception"

        try:
            with test_adapter.track_session(session_id) as session:
                assert session.session_id == session_id
                assert session.start_time > 0
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # Should still clean up properly
        assert session.end_time is not None
        assert session.end_time >= session.start_time

    def test_nested_session_contexts(self, test_adapter):
        """Test nested session context managers."""
        outer_session_id = "outer-session"
        inner_session_id = "inner-session"

        with test_adapter.track_session(outer_session_id) as outer_session:
            assert outer_session.session_id == outer_session_id

            with test_adapter.track_session(inner_session_id) as inner_session:
                assert inner_session.session_id == inner_session_id
                assert inner_session.session_id != outer_session.session_id

            # Inner session should be completed
            assert inner_session.end_time is not None
            # Outer session should still be active
            assert outer_session.end_time is None

        # Both sessions should be completed
        assert outer_session.end_time is not None
        assert inner_session.end_time is not None

    def test_auto_generated_session_id(self, test_adapter):
        """Test automatic session ID generation."""
        with test_adapter.track_session() as session:
            assert session.session_id is not None
            assert len(session.session_id) > 0
            assert session.session_id.startswith("session-")


class TestGovernanceScenarios:
    """Test complete governance scenarios."""

    def test_budget_advisory_governance(self, mock_together_client):
        """Test advisory governance policy with budget tracking."""
        adapter = GenOpsTogetherAdapter(
            team="governance-test",
            project="advisory-policy",
            daily_budget_limit=0.005,  # Small budget
            governance_policy="advisory"
        )

        # Should allow operations even if they exceed budget
        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test advisory governance"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100
        )

        assert result is not None
        assert result.cost > 0

        # Check cost summary
        summary = adapter.get_cost_summary()
        assert summary["governance_policy"] == "advisory"
        assert summary["daily_costs"] >= 0

    def test_budget_enforced_governance(self, mock_together_client):
        """Test enforced governance policy with budget limits."""
        adapter = GenOpsTogetherAdapter(
            team="governance-test",
            project="enforced-policy",
            daily_budget_limit=1.0,
            governance_policy="enforced"
        )

        # First operation should work
        result1 = adapter.chat_with_governance(
            messages=[{"role": "user", "content": "First operation"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=50
        )

        assert result1 is not None

        # Check governance attributes
        attrs = result1.governance_attributes
        assert attrs["team"] == "governance-test"
        assert attrs["project"] == "enforced-policy"
        assert attrs["governance_policy"] == "enforced"

    def test_multi_tenant_governance(self, mock_together_client):
        """Test multi-tenant governance scenario."""
        # Create adapters for different tenants
        tenant_a = GenOpsTogetherAdapter(
            team="tenant-a-team",
            project="tenant-a-project",
            customer_id="customer-a",
            cost_center="division-1",
            daily_budget_limit=2.0,
            governance_policy="strict"
        )

        tenant_b = GenOpsTogetherAdapter(
            team="tenant-b-team",
            project="tenant-b-project",
            customer_id="customer-b",
            cost_center="division-2",
            daily_budget_limit=3.0,
            governance_policy="advisory"
        )

        # Test operations for each tenant
        result_a = tenant_a.chat_with_governance(
            messages=[{"role": "user", "content": "Tenant A request"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            feature="tenant-a-feature"
        )

        result_b = tenant_b.chat_with_governance(
            messages=[{"role": "user", "content": "Tenant B request"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            feature="tenant-b-feature"
        )

        # Verify tenant isolation
        assert result_a.governance_attributes["customer_id"] == "customer-a"
        assert result_b.governance_attributes["customer_id"] == "customer-b"
        assert result_a.governance_attributes["team"] != result_b.governance_attributes["team"]

        # Verify separate cost tracking
        summary_a = tenant_a.get_cost_summary()
        summary_b = tenant_b.get_cost_summary()

        assert summary_a["daily_budget_limit"] == 2.0
        assert summary_b["daily_budget_limit"] == 3.0
        assert summary_a["governance_policy"] == "strict"
        assert summary_b["governance_policy"] == "advisory"


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration scenarios."""

    @patch('src.genops.providers.together.Together')
    def test_auto_instrument_basic_integration(self, mock_together):
        """Test basic auto-instrumentation integration."""
        # Set up mock
        mock_client = MagicMock()
        mock_together.return_value = mock_client

        # Apply auto-instrumentation
        auto_instrument()

        # Should complete without errors
        assert True  # If we get here, auto-instrumentation worked

    @patch('src.genops.providers.together.Together')
    def test_auto_instrument_with_configuration(self, mock_together):
        """Test auto-instrumentation with custom configuration."""
        mock_client = MagicMock()
        mock_together.return_value = mock_client

        config = {
            "team": "auto-team",
            "project": "auto-project",
            "daily_budget_limit": 15.0
        }

        auto_instrument(**config)

        # Should complete without errors
        assert True


class TestValidationIntegration:
    """Test validation integration with main components."""

    @patch('src.genops.providers.together_validation.Together')
    def test_validation_integration_success(self, mock_together):
        """Test successful validation integration."""
        # Mock successful Together client
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[
            {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"}
        ])
        mock_together.return_value = mock_client

        # Run validation
        with patch.dict(os.environ, {'TOGETHER_API_KEY': 'sk-test-key'}):
            result = validate_together_setup()

            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'errors')
            assert isinstance(result.errors, list)

    def test_validation_with_adapter_creation(self, mock_together_client):
        """Test validation followed by adapter creation."""
        # This test verifies that validation and adapter creation work together

        with patch.dict(os.environ, {'TOGETHER_API_KEY': 'sk-test-key'}):
            # Run validation
            validation_result = validate_together_setup()

            # Create adapter (should work regardless of validation result)
            adapter = GenOpsTogetherAdapter(
                team="validation-integration",
                project="test-adapter-creation"
            )

            assert adapter is not None
            assert adapter.team == "validation-integration"


class TestPricingIntegration:
    """Test pricing calculator integration with adapter."""

    def test_pricing_adapter_integration(self, test_adapter):
        """Test pricing calculator integration with adapter."""
        calculator = TogetherPricingCalculator()

        # Test cost estimation
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        cost = calculator.estimate_chat_cost(model, tokens=100)

        assert isinstance(cost, Decimal)
        assert cost > 0

        # Verify adapter can use similar calculations
        adapter_cost = test_adapter._calculate_cost(
            model=model,
            input_tokens=50,
            output_tokens=50
        )

        assert isinstance(adapter_cost, Decimal)
        assert adapter_cost > 0

        # Costs should be in similar range
        assert abs(float(cost - adapter_cost)) < float(cost) * 0.5  # Within 50%

    def test_model_recommendation_integration(self, test_adapter):
        """Test model recommendation integration."""
        calculator = TogetherPricingCalculator()

        # Get model recommendation
        recommendation = calculator.recommend_model(
            task_complexity="simple",
            budget_per_operation=0.001
        )

        if recommendation["recommended_model"]:
            # Test that adapter can use recommended model
            model = recommendation["recommended_model"]

            # Verify model is valid for adapter
            assert isinstance(model, str)
            assert len(model) > 0

    def test_cost_analysis_integration(self, test_adapter):
        """Test cost analysis integration with real adapter usage."""
        calculator = TogetherPricingCalculator()

        # Analyze costs for typical usage
        analysis = calculator.analyze_costs(
            operations_per_day=10,
            avg_tokens_per_operation=100,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            days_to_analyze=7
        )

        assert "daily_cost" in analysis
        assert "weekly_cost" in analysis or "monthly_cost" in analysis
        assert isinstance(analysis["daily_cost"], (int, float, Decimal))

        # Verify adapter budget can accommodate this usage
        daily_cost = float(analysis["daily_cost"])
        if daily_cost < test_adapter.daily_budget_limit:
            # Should be able to create adapter with this budget
            budget_adapter = GenOpsTogetherAdapter(
                daily_budget_limit=daily_cost * 2,  # 2x buffer
                governance_policy="enforced"
            )
            assert budget_adapter.daily_budget_limit >= daily_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
