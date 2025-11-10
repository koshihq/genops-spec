#!/usr/bin/env python3
"""
Unit tests for Together AI adapter.

Tests core functionality, context management, governance features,
and error handling for the GenOpsTogetherAdapter.
"""

import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict
from unittest.mock import patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.genops.core.exceptions import GenOpsBudgetExceededError
    from src.genops.providers.together import (
        GenOpsTogetherAdapter,
        TogetherModel,
        TogetherTaskType,
        auto_instrument,
    )
except ImportError as e:
    pytest.skip(f"Together AI provider not available: {e}", allow_module_level=True)


@dataclass
class MockTogetherResponse:
    """Mock Together AI API response."""
    choices: list
    usage: Dict[str, int]
    model: str


class TestGenOpsTogetherAdapter:
    """Unit tests for GenOpsTogetherAdapter class."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.adapter = GenOpsTogetherAdapter(
            team="test-team",
            project="test-project",
            environment="test",
            daily_budget_limit=10.0,
            governance_policy="advisory"
        )

    def test_adapter_initialization(self):
        """Test adapter initializes with correct parameters."""
        assert self.adapter.team == "test-team"
        assert self.adapter.project == "test-project"
        assert self.adapter.environment == "test"
        assert self.adapter.daily_budget_limit == 10.0
        assert self.adapter.governance_policy == "advisory"

    def test_adapter_initialization_with_defaults(self):
        """Test adapter initializes with default parameters."""
        adapter = GenOpsTogetherAdapter()
        assert adapter.team is not None
        assert adapter.project is not None
        assert adapter.daily_budget_limit > 0
        assert adapter.governance_policy in ["advisory", "enforced", "strict"]

    def test_adapter_initialization_with_customer_id(self):
        """Test adapter initializes with customer attribution."""
        adapter = GenOpsTogetherAdapter(
            customer_id="customer-123",
            cost_center="ai-research"
        )
        assert adapter.customer_id == "customer-123"
        assert adapter.cost_center == "ai-research"

    def test_adapter_initialization_with_tags(self):
        """Test adapter initializes with custom tags."""
        tags = {"service": "ai-assistant", "tier": "premium"}
        adapter = GenOpsTogetherAdapter(tags=tags)
        assert adapter.tags == tags

    def test_adapter_initialization_with_invalid_governance_policy(self):
        """Test adapter raises error with invalid governance policy."""
        with pytest.raises(ValueError, match="Invalid governance policy"):
            GenOpsTogetherAdapter(governance_policy="invalid")

    def test_adapter_initialization_with_negative_budget(self):
        """Test adapter raises error with negative budget."""
        with pytest.raises(ValueError, match="Budget limit must be positive"):
            GenOpsTogetherAdapter(daily_budget_limit=-1.0)

    def test_adapter_initialization_with_zero_budget(self):
        """Test adapter raises error with zero budget."""
        with pytest.raises(ValueError, match="Budget limit must be positive"):
            GenOpsTogetherAdapter(daily_budget_limit=0.0)

    @patch('src.genops.providers.together.Together')
    def test_chat_with_governance_basic(self, mock_together):
        """Test basic chat completion with governance."""
        # Mock Together client response
        mock_response = MockTogetherResponse(
            choices=[{"message": {"content": "Test response"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        mock_together.return_value.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = self.adapter.chat_with_governance(
            messages=messages,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=50
        )

        assert result.response == "Test response"
        assert result.tokens_used == 30
        assert result.model_used == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        assert result.cost > 0

    @patch('src.genops.providers.together.Together')
    def test_chat_with_governance_with_session_id(self, mock_together):
        """Test chat completion with session tracking."""
        mock_response = MockTogetherResponse(
            choices=[{"message": {"content": "Session response"}}],
            usage={"prompt_tokens": 15, "completion_tokens": 25},
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        mock_together.return_value.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        result = self.adapter.chat_with_governance(
            messages=messages,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            session_id="session-123",
            feature="test-feature"
        )

        assert result.response == "Session response"
        assert result.session_id == "session-123"
        assert "test-feature" in result.governance_attributes.get("feature", "")

    @patch('src.genops.providers.together.Together')
    def test_chat_with_governance_budget_exceeded(self, mock_together):
        """Test budget exceeded handling with strict governance."""
        adapter = GenOpsTogetherAdapter(
            daily_budget_limit=0.001,  # Very low budget
            governance_policy="strict"
        )

        with pytest.raises(GenOpsBudgetExceededError):
            adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Expensive request"}],
                model=TogetherModel.LLAMA_3_1_405B_INSTRUCT,
                max_tokens=1000
            )

    @patch('src.genops.providers.together.Together')
    def test_chat_with_governance_with_task_type(self, mock_together):
        """Test chat completion with specific task type."""
        mock_response = MockTogetherResponse(
            choices=[{"message": {"content": "Code response"}}],
            usage={"prompt_tokens": 20, "completion_tokens": 30},
            model="deepseek-ai/DeepSeek-Coder-V2-Instruct"
        )
        mock_together.return_value.chat.completions.create.return_value = mock_response

        result = self.adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Write a function"}],
            model=TogetherModel.DEEPSEEK_CODER_V2,
            task_type=TogetherTaskType.CODE_GENERATION,
            max_tokens=100
        )

        assert result.task_type == TogetherTaskType.CODE_GENERATION
        assert "code_generation" in result.governance_attributes.get("task_type", "")

    def test_track_session_context_manager(self):
        """Test session tracking context manager."""
        with self.adapter.track_session("test-session") as session:
            assert session.session_id == "test-session"
            assert session.adapter == self.adapter
            assert session.start_time > 0

        # Context manager should complete without errors
        assert session.end_time > session.start_time

    def test_track_session_with_auto_id(self):
        """Test session tracking with auto-generated ID."""
        with self.adapter.track_session() as session:
            assert session.session_id is not None
            assert len(session.session_id) > 0
            assert session.session_id.startswith("session-")

    def test_get_cost_summary(self):
        """Test cost summary generation."""
        summary = self.adapter.get_cost_summary()

        assert "daily_costs" in summary
        assert "daily_budget_limit" in summary
        assert "daily_budget_utilization" in summary
        assert "governance_policy" in summary
        assert "operations_count" in summary

        assert isinstance(summary["daily_costs"], (int, float, Decimal))
        assert summary["daily_budget_limit"] == 10.0
        assert summary["governance_policy"] == "advisory"

    def test_calculate_cost(self):
        """Test cost calculation for different models."""
        # Test lite tier model
        cost_8b = self.adapter._calculate_cost(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            input_tokens=100,
            output_tokens=50
        )

        # Test standard tier model
        cost_70b = self.adapter._calculate_cost(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            input_tokens=100,
            output_tokens=50
        )

        assert cost_8b > 0
        assert cost_70b > cost_8b  # 70B should cost more than 8B
        assert isinstance(cost_8b, Decimal)
        assert isinstance(cost_70b, Decimal)

    def test_calculate_cost_with_unknown_model(self):
        """Test cost calculation with unknown model defaults to generic pricing."""
        cost = self.adapter._calculate_cost(
            model="unknown/custom-model",
            input_tokens=100,
            output_tokens=50
        )

        assert cost > 0
        assert isinstance(cost, Decimal)

    def test_should_allow_operation_advisory_policy(self):
        """Test operation allowed with advisory governance policy."""
        adapter = GenOpsTogetherAdapter(
            daily_budget_limit=1.0,
            governance_policy="advisory"
        )

        # Should allow operation even if it would exceed budget
        allowed = adapter._should_allow_operation(estimated_cost=2.0)
        assert allowed is True

    def test_should_allow_operation_enforced_policy(self):
        """Test operation control with enforced governance policy."""
        adapter = GenOpsTogetherAdapter(
            daily_budget_limit=1.0,
            governance_policy="enforced"
        )

        # Should block operation that exceeds budget
        allowed = adapter._should_allow_operation(estimated_cost=2.0)
        assert allowed is False

        # Should allow operation within budget
        allowed = adapter._should_allow_operation(estimated_cost=0.5)
        assert allowed is True

    def test_should_allow_operation_strict_policy(self):
        """Test operation control with strict governance policy."""
        adapter = GenOpsTogetherAdapter(
            daily_budget_limit=1.0,
            governance_policy="strict"
        )

        # Should block operation that exceeds budget
        allowed = adapter._should_allow_operation(estimated_cost=1.5)
        assert allowed is False

    def test_create_governance_attributes(self):
        """Test governance attributes creation."""
        attrs = self.adapter._create_governance_attributes(
            session_id="test-session",
            feature="test-feature",
            custom_attr="custom-value"
        )

        assert attrs["team"] == "test-team"
        assert attrs["project"] == "test-project"
        assert attrs["environment"] == "test"
        assert attrs["session_id"] == "test-session"
        assert attrs["feature"] == "test-feature"
        assert attrs["custom_attr"] == "custom-value"

    def test_create_governance_attributes_with_customer_id(self):
        """Test governance attributes include customer attribution."""
        adapter = GenOpsTogetherAdapter(
            team="test",
            customer_id="customer-123",
            cost_center="research"
        )

        attrs = adapter._create_governance_attributes()

        assert attrs["customer_id"] == "customer-123"
        assert attrs["cost_center"] == "research"


class TestTogetherAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def test_auto_instrument_function_exists(self):
        """Test auto_instrument function is available."""
        assert callable(auto_instrument)

    @patch('src.genops.providers.together.Together')
    def test_auto_instrument_basic(self, mock_together):
        """Test basic auto-instrumentation setup."""
        # Should not raise any exceptions
        auto_instrument()

        # Verify it can be called multiple times safely
        auto_instrument()

    @patch('src.genops.providers.together.Together')
    def test_auto_instrument_with_config(self, mock_together):
        """Test auto-instrumentation with configuration."""
        config = {
            "team": "auto-team",
            "project": "auto-project",
            "daily_budget_limit": 25.0
        }

        auto_instrument(**config)
        # Should complete without errors


class TestTogetherModelEnum:
    """Test TogetherModel enum functionality."""

    def test_model_enum_values(self):
        """Test model enum contains expected values."""
        assert hasattr(TogetherModel, 'LLAMA_3_1_8B_INSTRUCT')
        assert hasattr(TogetherModel, 'LLAMA_3_1_70B_INSTRUCT')
        assert hasattr(TogetherModel, 'LLAMA_3_1_405B_INSTRUCT')
        assert hasattr(TogetherModel, 'DEEPSEEK_R1')
        assert hasattr(TogetherModel, 'DEEPSEEK_CODER_V2')
        assert hasattr(TogetherModel, 'QWEN_VL_72B')

    def test_model_enum_string_values(self):
        """Test model enum values are valid Together AI model names."""
        assert TogetherModel.LLAMA_3_1_8B_INSTRUCT.value.startswith("meta-llama/")
        assert TogetherModel.DEEPSEEK_R1.value.startswith("deepseek-ai/")
        assert "Instruct" in TogetherModel.LLAMA_3_1_70B_INSTRUCT.value


class TestTogetherTaskTypeEnum:
    """Test TogetherTaskType enum functionality."""

    def test_task_type_enum_values(self):
        """Test task type enum contains expected values."""
        assert hasattr(TogetherTaskType, 'CHAT')
        assert hasattr(TogetherTaskType, 'CODE_GENERATION')
        assert hasattr(TogetherTaskType, 'REASONING')
        assert hasattr(TogetherTaskType, 'MULTIMODAL')
        assert hasattr(TogetherTaskType, 'ANALYSIS')

    def test_task_type_enum_string_values(self):
        """Test task type enum values are strings."""
        assert isinstance(TogetherTaskType.CHAT.value, str)
        assert isinstance(TogetherTaskType.CODE_GENERATION.value, str)
        assert len(TogetherTaskType.REASONING.value) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
