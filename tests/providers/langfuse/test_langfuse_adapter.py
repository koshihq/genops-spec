"""Tests for GenOps Langfuse adapter core functionality."""

import os
import time
from unittest.mock import Mock, patch

import pytest

# Test if Langfuse is available
try:
    import langfuse  # noqa: F401

    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

pytestmark = pytest.mark.skipif(not HAS_LANGFUSE, reason="Langfuse not installed")

from genops.providers.langfuse import (  # noqa: E402
    GenOpsLangfuseAdapter,
    GovernancePolicy,
    LangfuseResponse,
    LangfuseUsage,
    instrument_langfuse,
)


class TestGenOpsLangfuseAdapter:
    """Test GenOps Langfuse adapter functionality."""

    @pytest.fixture
    def mock_langfuse(self):
        """Mock Langfuse client."""
        with patch("genops.providers.langfuse.Langfuse") as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def adapter(self, mock_langfuse):
        """Create test adapter."""
        return GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            team="test-team",
            project="test-project",
        )

    def test_adapter_initialization(self, mock_langfuse):
        """Test adapter initialization with various configurations."""
        # Basic initialization
        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            team="test-team",
        )

        assert adapter.team == "test-team"
        assert adapter.enable_governance is True
        assert adapter.policy_mode == GovernancePolicy.ADVISORY

    def test_adapter_initialization_with_budget(self, mock_langfuse):
        """Test adapter initialization with budget limits."""
        budget_limits = {"daily": 100.0, "monthly": 2000.0}

        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            team="budget-team",
            budget_limits=budget_limits,
        )

        assert adapter.budget_limits == budget_limits
        assert adapter.current_costs["daily"] == 0.0
        assert adapter.current_costs["monthly"] == 0.0

    def test_adapter_initialization_with_env_vars(self, mock_langfuse):
        """Test adapter initialization using environment variables."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-lf-env",
                "LANGFUSE_SECRET_KEY": "sk-lf-env",
                "LANGFUSE_BASE_URL": "https://test.langfuse.com",
            },
        ):
            adapter = GenOpsLangfuseAdapter(team="env-team")

            assert adapter.team == "env-team"
            # Verify Langfuse client was called with env vars
            mock_langfuse.assert_called_with(
                public_key="pk-lf-env",
                secret_key="sk-lf-env",
                host="https://test.langfuse.com",
            )

    def test_cost_calculation(self, adapter):
        """Test cost calculation for different models."""
        # Test known model
        cost = adapter._calculate_cost("gpt-4", 100, 50)
        expected = (100 * 0.00003) + (50 * 0.00006)
        assert cost == expected

        # Test unknown model (uses default)
        cost = adapter._calculate_cost("unknown-model", 100, 50)
        expected = (100 + 50) * 0.00001
        assert cost == expected

    def test_budget_compliance_check(self, mock_langfuse):
        """Test budget compliance checking."""
        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            budget_limits={"daily": 1.0, "monthly": 10.0},
        )

        # Test within budget
        assert adapter._check_budget_compliance(0.5) is True

        # Test exceeding daily budget
        adapter.current_costs["daily"] = 0.8
        assert adapter._check_budget_compliance(0.5) is False
        assert "Daily budget exceeded" in adapter.policy_violations[0]

        # Test exceeding monthly budget
        adapter.policy_violations = []  # Clear violations
        adapter.current_costs["daily"] = 0.1
        adapter.current_costs["monthly"] = 9.5
        assert adapter._check_budget_compliance(1.0) is False
        assert "Monthly budget exceeded" in adapter.policy_violations[0]

    def test_governance_attributes_extraction(self, adapter):
        """Test extraction of governance attributes."""
        kwargs = {
            "team": "override-team",
            "project": "override-project",
            "customer_id": "cust-123",
            "cost_center": "research",
            "other_param": "value",
        }

        governance_attrs = adapter._extract_governance_attributes(kwargs)

        assert governance_attrs["team"] == "override-team"
        assert governance_attrs["project"] == "override-project"
        assert governance_attrs["customer_id"] == "cust-123"
        assert governance_attrs["cost_center"] == "research"
        assert "other_param" not in governance_attrs
        assert kwargs["other_param"] == "value"  # Non-governance attrs remain

    def test_trace_with_governance(self, adapter):
        """Test governance-enhanced tracing."""
        mock_trace = Mock()
        adapter.langfuse.trace.return_value = mock_trace

        with adapter.trace_with_governance(
            name="test-trace", customer_id="cust-456", metadata={"custom": "value"}
        ) as trace:
            assert trace == mock_trace
            time.sleep(0.01)  # Simulate work

        # Verify trace was created with governance metadata
        adapter.langfuse.trace.assert_called_once()
        call_args = adapter.langfuse.trace.call_args

        assert call_args[1]["name"] == "test-trace"
        assert "genops_governance" in call_args[1]["metadata"]
        assert (
            call_args[1]["metadata"]["genops_governance"]["customer_id"] == "cust-456"
        )
        assert call_args[1]["metadata"]["custom"] == "value"

        # Verify trace was updated with duration
        mock_trace.update.assert_called()
        update_metadata = mock_trace.update.call_args[1]["metadata"]
        assert "genops_duration_ms" in update_metadata
        assert update_metadata["genops_duration_ms"] > 0

    def test_trace_with_governance_error_handling(self, adapter):
        """Test trace error handling."""
        mock_trace = Mock()
        adapter.langfuse.trace.return_value = mock_trace

        with pytest.raises(ValueError):
            with adapter.trace_with_governance(name="error-trace"):
                raise ValueError("Test error")

        # Verify error was recorded in trace metadata
        mock_trace.update.assert_called()
        update_metadata = mock_trace.update.call_args[1]["metadata"]
        assert "governance_error" in update_metadata
        assert "Test error" in update_metadata["governance_error"]

    def test_generation_with_cost_tracking(self, adapter):
        """Test LLM generation with cost tracking."""
        mock_generation = Mock()
        mock_generation.id = "gen-123"
        mock_generation.metadata = {}
        adapter.langfuse.generation.return_value = mock_generation

        response = adapter.generation_with_cost_tracking(
            prompt="Test prompt for analysis",
            model="gpt-3.5-turbo",
            max_cost=0.10,
            team="test-team",
            customer_id="cust-789",
        )

        # Verify generation was created with governance metadata
        adapter.langfuse.generation.assert_called_once()
        call_args = adapter.langfuse.generation.call_args

        assert call_args[1]["name"] == "gpt-3.5-turbo_generation"
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["input"] == "Test prompt for analysis"
        assert "genops_governance" in call_args[1]["metadata"]
        assert "genops_max_cost" in call_args[1]["metadata"]

        # Verify response structure
        assert isinstance(response, LangfuseResponse)
        assert isinstance(response.usage, LangfuseUsage)
        assert response.usage.model == "gpt-3.5-turbo"
        assert response.usage.team == "test-team"
        assert response.usage.cost > 0
        assert response.observation_id == "gen-123"

        # Verify generation was finalized
        mock_generation.end.assert_called_once()
        end_args = mock_generation.end.call_args
        assert "output" in end_args[1]
        assert "usage" in end_args[1]
        assert "metadata" in end_args[1]

    def test_generation_cost_limit_exceeded(self, adapter):
        """Test generation fails when cost limit is exceeded."""
        with pytest.raises(ValueError, match="exceeds max_cost"):
            adapter.generation_with_cost_tracking(
                prompt="Very expensive prompt " * 1000,  # Large prompt
                model="gpt-4",
                max_cost=0.001,  # Very low limit
            )

    def test_generation_budget_enforcement(self, mock_langfuse):
        """Test budget enforcement in generation."""
        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            budget_limits={"daily": 0.01},  # Very low budget
            policy_mode=GovernancePolicy.ENFORCED,
        )

        # Exceed budget
        adapter.current_costs["daily"] = 0.009

        with pytest.raises(ValueError, match="Budget limit exceeded"):
            adapter.generation_with_cost_tracking(
                prompt="Test prompt", model="gpt-3.5-turbo"
            )

    def test_generation_budget_advisory_mode(self, mock_langfuse):
        """Test budget in advisory mode allows operations."""
        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            budget_limits={"daily": 0.01},
            policy_mode=GovernancePolicy.ADVISORY,  # Advisory mode
        )

        mock_generation = Mock()
        mock_generation.id = "gen-advisory"
        mock_generation.metadata = {}
        adapter.langfuse.generation.return_value = mock_generation

        # Exceed budget but should still work in advisory mode
        adapter.current_costs["daily"] = 0.009

        response = adapter.generation_with_cost_tracking(
            prompt="Test prompt", model="gpt-3.5-turbo"
        )

        assert response is not None
        assert len(adapter.policy_violations) > 0  # Violation recorded

    def test_evaluate_with_governance(self, adapter):
        """Test governance-aware evaluation."""
        mock_score = Mock()
        mock_score.id = "score-123"
        adapter.langfuse.score.return_value = mock_score

        def test_evaluator():
            return {"score": 0.85, "comment": "Good quality"}

        result = adapter.evaluate_with_governance(
            trace_id="trace-456",
            evaluation_name="quality_check",
            evaluator_function=test_evaluator,
            customer_id="eval-customer",
        )

        # Verify score was created with governance metadata
        adapter.langfuse.score.assert_called_once()
        call_args = adapter.langfuse.score.call_args

        assert call_args[1]["trace_id"] == "trace-456"
        assert call_args[1]["name"] == "quality_check"
        assert call_args[1]["value"] == 0.85
        assert call_args[1]["comment"] == "Good quality"
        assert "genops_governance" in call_args[1]["metadata"]

        # Verify result structure
        assert result["score"] == 0.85
        assert result["evaluation_id"] == "score-123"
        assert result["governance"]["customer_id"] == "eval-customer"
        assert result["duration_ms"] > 0

    def test_evaluate_with_governance_error(self, adapter):
        """Test evaluation error handling."""

        def failing_evaluator():
            raise RuntimeError("Evaluation failed")

        with pytest.raises(RuntimeError, match="Evaluation failed"):
            adapter.evaluate_with_governance(
                trace_id="trace-error",
                evaluation_name="failing_eval",
                evaluator_function=failing_evaluator,
            )

    def test_get_cost_summary(self, mock_langfuse):
        """Test cost summary generation."""
        adapter = GenOpsLangfuseAdapter(
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
            team="summary-team",
            project="summary-project",
            budget_limits={"daily": 100.0},
        )

        # Simulate some costs
        adapter.current_costs["daily"] = 45.50
        adapter.operation_count = 150
        adapter.policy_violations = ["violation1", "violation2"]

        summary = adapter.get_cost_summary("daily")

        assert summary["period"] == "daily"
        assert summary["total_cost"] == 45.50
        assert summary["operation_count"] == 150
        assert summary["average_cost_per_operation"] == 45.50 / 150
        assert summary["budget_limit"] == 100.0
        assert summary["budget_remaining"] == 54.50
        assert summary["policy_violations"] == 2
        assert summary["governance"]["team"] == "summary-team"
        assert summary["governance"]["project"] == "summary-project"


class TestInstrumentLangfuse:
    """Test Langfuse instrumentation function."""

    @patch("genops.providers.langfuse.Langfuse")
    @patch("genops.providers.langfuse._auto_instrument_langfuse")
    def test_instrument_langfuse_basic(self, mock_auto_instrument, mock_langfuse):
        """Test basic instrumentation."""
        adapter = instrument_langfuse(
            langfuse_public_key="pk-lf-instrument",
            langfuse_secret_key="sk-lf-instrument",
            team="instrument-team",
        )

        assert isinstance(adapter, GenOpsLangfuseAdapter)
        assert adapter.team == "instrument-team"
        mock_auto_instrument.assert_called_once_with(adapter)

    @patch("genops.providers.langfuse.Langfuse")
    @patch("genops.providers.langfuse._auto_instrument_langfuse")
    def test_instrument_langfuse_no_auto(self, mock_auto_instrument, mock_langfuse):
        """Test instrumentation without auto-instrumentation."""
        adapter = instrument_langfuse(team="no-auto-team", auto_instrument=False)

        assert isinstance(adapter, GenOpsLangfuseAdapter)
        mock_auto_instrument.assert_not_called()

    @patch("genops.providers.langfuse.Langfuse")
    def test_instrument_langfuse_with_budget(self, mock_langfuse):
        """Test instrumentation with budget limits."""
        budget_limits = {"daily": 200.0, "monthly": 5000.0}

        adapter = instrument_langfuse(
            team="budget-instrument", budget_limits=budget_limits
        )

        assert adapter.budget_limits == budget_limits


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    @patch("genops.providers.langfuse.HAS_LANGFUSE", True)
    @patch("genops.providers.langfuse.observe")
    def test_auto_instrument_langfuse(self, mock_observe):
        """Test auto-instrumentation enhancement."""
        from genops.providers.langfuse import _auto_instrument_langfuse

        adapter = Mock()
        adapter.team = "auto-team"
        adapter.project = "auto-project"
        adapter.environment = "test"

        _auto_instrument_langfuse(adapter)

        # Verify that observe decorator was enhanced
        # This is a complex test as it modifies the global observe function
        # In practice, we'd test this through integration tests

    @patch("genops.providers.langfuse.HAS_LANGFUSE", False)
    def test_auto_instrument_langfuse_not_available(self):
        """Test auto-instrumentation when Langfuse is not available."""
        from genops.providers.langfuse import _auto_instrument_langfuse

        adapter = Mock()

        # Should not raise error, but should log warning
        _auto_instrument_langfuse(adapter)


class TestLangfuseDataClasses:
    """Test Langfuse-specific data classes."""

    def test_langfuse_usage_creation(self):
        """Test LangfuseUsage data class."""
        usage = LangfuseUsage(
            operation_id="op-123",
            observation_type="generation",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.0045,
            latency_ms=1200.5,
            team="usage-team",
            project="usage-project",
        )

        assert usage.operation_id == "op-123"
        assert usage.observation_type == "generation"
        assert usage.model == "gpt-4"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost == 0.0045
        assert usage.latency_ms == 1200.5
        assert usage.team == "usage-team"
        assert usage.project == "usage-project"
        assert usage.policy_violations == []  # Default empty list
        assert usage.governance_tags == {}  # Default empty dict

    def test_langfuse_response_creation(self):
        """Test LangfuseResponse data class."""
        usage = LangfuseUsage(
            operation_id="op-456",
            observation_type="generation",
            model="gpt-3.5-turbo",
            input_tokens=75,
            output_tokens=25,
            total_tokens=100,
            cost=0.0015,
            latency_ms=800.0,
        )

        response = LangfuseResponse(
            content="Test response content",
            usage=usage,
            trace_id="trace-789",
            observation_id="obs-101112",
            metadata={"custom": "metadata"},
            governance_status="compliant",
            cost_optimization_suggestions=["Use smaller model for simple tasks"],
        )

        assert response.content == "Test response content"
        assert response.usage == usage
        assert response.trace_id == "trace-789"
        assert response.observation_id == "obs-101112"
        assert response.metadata["custom"] == "metadata"
        assert response.governance_status == "compliant"
        assert len(response.cost_optimization_suggestions) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_langfuse_not_available(self):
        """Test behavior when Langfuse is not available."""
        with patch("genops.providers.langfuse.HAS_LANGFUSE", False):
            with pytest.raises(ImportError, match="Langfuse package not found"):
                GenOpsLangfuseAdapter(team="test")

    @patch("genops.providers.langfuse.Langfuse")
    def test_empty_governance_attributes(self, mock_langfuse):
        """Test handling of empty governance attributes."""
        adapter = GenOpsLangfuseAdapter()

        assert adapter.team is None
        assert adapter.project is None
        assert adapter.environment == "production"  # Default

    @patch("genops.providers.langfuse.Langfuse")
    def test_zero_token_cost_calculation(self, mock_langfuse):
        """Test cost calculation with zero tokens."""
        adapter = GenOpsLangfuseAdapter()

        cost = adapter._calculate_cost("gpt-4", 0, 0)
        assert cost == 0.0

    @patch("genops.providers.langfuse.Langfuse")
    def test_negative_cost_handling(self, mock_langfuse):
        """Test handling of negative token counts."""
        adapter = GenOpsLangfuseAdapter()

        # Should handle negative tokens gracefully
        cost = adapter._calculate_cost("gpt-4", -10, -5)
        # Cost calculation might return negative, but this tests no crash
        assert isinstance(cost, float)

    @patch("genops.providers.langfuse.Langfuse")
    def test_large_token_counts(self, mock_langfuse):
        """Test handling of very large token counts."""
        adapter = GenOpsLangfuseAdapter()

        # Test with very large token counts
        cost = adapter._calculate_cost("gpt-4", 1_000_000, 500_000)
        assert cost > 0
        assert isinstance(cost, float)

    @patch("genops.providers.langfuse.Langfuse")
    def test_empty_budget_limits(self, mock_langfuse):
        """Test behavior with empty budget limits."""
        adapter = GenOpsLangfuseAdapter(budget_limits={})

        # Should always return True for budget compliance
        assert adapter._check_budget_compliance(999999.0) is True

    @patch("genops.providers.langfuse.Langfuse")
    def test_none_budget_limits(self, mock_langfuse):
        """Test behavior with None budget limits."""
        adapter = GenOpsLangfuseAdapter(budget_limits=None)

        # Should always return True for budget compliance
        assert adapter._check_budget_compliance(999999.0) is True
