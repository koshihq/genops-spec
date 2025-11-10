#!/usr/bin/env python3
"""
Cross-provider tests for Together AI integration.

Tests compatibility with other providers, migration scenarios,
unified governance across providers, and multi-provider operations.
"""

import os
import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.genops.providers.together import GenOpsTogetherAdapter, TogetherModel
    from src.genops.providers.together_pricing import TogetherPricingCalculator
except ImportError as e:
    pytest.skip(f"Together AI provider not available: {e}", allow_module_level=True)


# Mock other providers for cross-provider testing
class MockOpenAIAdapter:
    """Mock OpenAI adapter for cross-provider testing."""

    def __init__(self, **kwargs):
        self.team = kwargs.get('team', 'openai-team')
        self.project = kwargs.get('project', 'openai-project')
        self.daily_budget_limit = kwargs.get('daily_budget_limit', 10.0)
        self.governance_policy = kwargs.get('governance_policy', 'advisory')
        self.daily_costs = Decimal('0')

    def chat_with_governance(self, messages, model, **kwargs):
        """Mock OpenAI chat completion."""
        cost = Decimal('0.002')  # Higher cost than Together AI
        self.daily_costs += cost

        return MagicMock(
            response="OpenAI mock response",
            tokens_used=25,
            cost=cost,
            model_used=model,
            governance_attributes={
                "team": self.team,
                "project": self.project,
                "provider": "openai"
            }
        )

    def get_cost_summary(self):
        """Mock cost summary."""
        return {
            "daily_costs": float(self.daily_costs),
            "daily_budget_limit": self.daily_budget_limit,
            "daily_budget_utilization": (float(self.daily_costs) / self.daily_budget_limit) * 100,
            "governance_policy": self.governance_policy,
            "provider": "openai"
        }


class MockAnthropicAdapter:
    """Mock Anthropic adapter for cross-provider testing."""

    def __init__(self, **kwargs):
        self.team = kwargs.get('team', 'anthropic-team')
        self.project = kwargs.get('project', 'anthropic-project')
        self.daily_budget_limit = kwargs.get('daily_budget_limit', 15.0)
        self.governance_policy = kwargs.get('governance_policy', 'enforced')
        self.daily_costs = Decimal('0')

    def chat_with_governance(self, messages, model, **kwargs):
        """Mock Anthropic chat completion."""
        cost = Decimal('0.003')  # Higher cost than Together AI
        self.daily_costs += cost

        return MagicMock(
            response="Anthropic mock response",
            tokens_used=30,
            cost=cost,
            model_used=model,
            governance_attributes={
                "team": self.team,
                "project": self.project,
                "provider": "anthropic"
            }
        )

    def get_cost_summary(self):
        """Mock cost summary."""
        return {
            "daily_costs": float(self.daily_costs),
            "daily_budget_limit": self.daily_budget_limit,
            "daily_budget_utilization": (float(self.daily_costs) / self.daily_budget_limit) * 100,
            "governance_policy": self.governance_policy,
            "provider": "anthropic"
        }


@pytest.fixture
def mock_together_client():
    """Fixture providing mocked Together client."""
    with patch('src.genops.providers.together.Together') as mock:
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[{"message": {"content": "Together response"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 15},
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        mock.return_value = client
        yield client


@pytest.fixture
def together_adapter():
    """Fixture providing Together AI adapter."""
    return GenOpsTogetherAdapter(
        team="cross-provider-test",
        project="together-integration",
        daily_budget_limit=5.0,
        governance_policy="advisory"
    )


class TestCrossProviderGovernance:
    """Test governance consistency across providers."""

    def test_governance_attribute_consistency(self, mock_together_client, together_adapter):
        """Test governance attributes are consistent across providers."""
        # Together AI operation
        together_result = together_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test message"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            feature="cross-provider-test"
        )

        # Mock other providers
        openai_adapter = MockOpenAIAdapter(
            team="cross-provider-test",
            project="together-integration",
            daily_budget_limit=5.0
        )

        openai_result = openai_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test message"}],
            model="gpt-3.5-turbo",
            feature="cross-provider-test"
        )

        # Verify consistent governance attributes
        together_attrs = together_result.governance_attributes
        openai_attrs = openai_result.governance_attributes

        assert together_attrs["team"] == openai_attrs["team"]
        assert together_attrs["project"] == openai_attrs["project"]
        # Providers should be different
        assert together_attrs.get("provider") != openai_attrs.get("provider")

    def test_unified_cost_tracking(self, mock_together_client):
        """Test unified cost tracking across providers."""
        # Create adapters with same governance settings
        governance_config = {
            "team": "unified-team",
            "project": "multi-provider-project",
            "customer_id": "customer-123",
            "daily_budget_limit": 10.0
        }

        together_adapter = GenOpsTogetherAdapter(**governance_config)
        openai_adapter = MockOpenAIAdapter(**governance_config)
        anthropic_adapter = MockAnthropicAdapter(**governance_config)

        # Perform operations on each provider
        together_result = together_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Together test"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT
        )

        openai_result = openai_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "OpenAI test"}],
            model="gpt-3.5-turbo"
        )

        anthropic_result = anthropic_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Anthropic test"}],
            model="claude-3-sonnet"
        )

        # Verify cost tracking
        together_summary = together_adapter.get_cost_summary()
        openai_summary = openai_adapter.get_cost_summary()
        anthropic_summary = anthropic_adapter.get_cost_summary()

        assert together_summary["daily_costs"] > 0
        assert openai_summary["daily_costs"] > 0
        assert anthropic_summary["daily_costs"] > 0

        # Together AI should be most cost-effective
        assert together_summary["daily_costs"] < openai_summary["daily_costs"]
        assert together_summary["daily_costs"] < anthropic_summary["daily_costs"]

    def test_multi_provider_session_tracking(self, mock_together_client):
        """Test session tracking across multiple providers."""
        session_id = "multi-provider-session"

        together_adapter = GenOpsTogetherAdapter(
            team="session-test",
            project="multi-provider"
        )
        openai_adapter = MockOpenAIAdapter(
            team="session-test",
            project="multi-provider"
        )

        # Use same session ID across providers
        with together_adapter.track_session(session_id) as session:
            # Together AI operation
            together_result = together_adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Together in session"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                session_id=session.session_id
            )

            # OpenAI operation (mock doesn't have session tracking, but we can test the ID)
            openai_result = openai_adapter.chat_with_governance(
                messages=[{"role": "user", "content": "OpenAI in session"}],
                model="gpt-3.5-turbo",
                session_id=session_id  # Same session ID
            )

            assert session.session_id == session_id
            assert together_result.governance_attributes.get("session_id") == session_id


class TestProviderMigration:
    """Test migration scenarios between providers."""

    def test_migration_from_openai_to_together(self, mock_together_client):
        """Test migration from OpenAI to Together AI."""
        # Original OpenAI setup
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain machine learning"}
        ]

        # Migrate to Together AI with same interface
        together_adapter = GenOpsTogetherAdapter(
            team="migration-test",
            project="openai-to-together",
            migration_source="openai"
        )

        result = together_adapter.chat_with_governance(
            messages=openai_messages,  # Same message format
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=150,
            temperature=0.7,
            migration_context="from_openai"
        )

        assert result.response is not None
        assert result.tokens_used > 0
        assert result.cost > 0

        # Verify migration is tracked
        assert "migration" in result.governance_attributes.get("migration_context", "")

    def test_migration_cost_comparison(self, mock_together_client):
        """Test cost comparison for migration scenarios."""
        # Setup comparable scenarios
        test_message = [{"role": "user", "content": "Generate a product description for an AI chatbot"}]

        # Together AI
        together_adapter = GenOpsTogetherAdapter(
            team="cost-comparison",
            project="migration-analysis"
        )

        together_result = together_adapter.chat_with_governance(
            messages=test_message,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100
        )

        # Mock other providers
        openai_adapter = MockOpenAIAdapter(
            team="cost-comparison",
            project="migration-analysis"
        )

        openai_result = openai_adapter.chat_with_governance(
            messages=test_message,
            model="gpt-3.5-turbo",
            max_tokens=100
        )

        # Compare costs
        together_cost = float(together_result.cost)
        openai_cost = float(openai_result.cost)

        # Together AI should be more cost-effective
        assert together_cost < openai_cost

        # Calculate savings
        savings = openai_cost - together_cost
        savings_percentage = (savings / openai_cost) * 100

        assert savings > 0
        assert savings_percentage > 0

    def test_feature_parity_migration(self, mock_together_client):
        """Test feature parity during migration."""
        # Test that Together AI supports common features from other providers

        together_adapter = GenOpsTogetherAdapter(
            team="parity-test",
            project="feature-migration"
        )

        # Test common parameters that should work across providers
        common_params = {
            "messages": [{"role": "user", "content": "Test feature parity"}],
            "model": TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            "max_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }

        # Should handle common parameters without errors
        result = together_adapter.chat_with_governance(**common_params)

        assert result is not None
        assert result.tokens_used > 0


class TestMultiProviderOperations:
    """Test operations across multiple providers simultaneously."""

    def test_concurrent_provider_operations(self, mock_together_client):
        """Test concurrent operations across providers."""
        # Setup multiple providers
        together_adapter = GenOpsTogetherAdapter(
            team="concurrent-test",
            project="multi-provider-ops",
            customer_id="customer-concurrent"
        )

        openai_adapter = MockOpenAIAdapter(
            team="concurrent-test",
            project="multi-provider-ops",
            customer_id="customer-concurrent"
        )

        anthropic_adapter = MockAnthropicAdapter(
            team="concurrent-test",
            project="multi-provider-ops",
            customer_id="customer-concurrent"
        )

        # Simulate concurrent operations
        test_message = [{"role": "user", "content": "Concurrent test message"}]

        together_result = together_adapter.chat_with_governance(
            messages=test_message,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            operation_type="concurrent",
            provider_group="multi-provider"
        )

        openai_result = openai_adapter.chat_with_governance(
            messages=test_message,
            model="gpt-3.5-turbo",
            operation_type="concurrent",
            provider_group="multi-provider"
        )

        anthropic_result = anthropic_adapter.chat_with_governance(
            messages=test_message,
            model="claude-3-sonnet",
            operation_type="concurrent",
            provider_group="multi-provider"
        )

        # Verify all operations completed
        results = [together_result, openai_result, anthropic_result]
        assert all(result.response is not None for result in results)
        assert all(result.tokens_used > 0 for result in results)
        assert all(result.cost > 0 for result in results)

        # Verify Together AI is most cost-effective
        costs = [float(result.cost) for result in results]
        together_cost = costs[0]
        assert together_cost == min(costs)

    def test_provider_fallback_scenario(self, mock_together_client):
        """Test fallback from one provider to another."""
        # Primary provider (Together AI)
        primary_adapter = GenOpsTogetherAdapter(
            team="fallback-test",
            project="resilience-test"
        )

        # Backup provider
        backup_adapter = MockOpenAIAdapter(
            team="fallback-test",
            project="resilience-test"
        )

        test_message = [{"role": "user", "content": "Test fallback scenario"}]

        try:
            # Try primary provider first
            result = primary_adapter.chat_with_governance(
                messages=test_message,
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                fallback_available=True
            )

            # If primary succeeds, use it
            primary_used = True
            final_result = result

        except Exception:
            # If primary fails, fallback to backup
            primary_used = False
            final_result = backup_adapter.chat_with_governance(
                messages=test_message,
                model="gpt-3.5-turbo",
                fallback_operation=True,
                primary_provider="together"
            )

        # Verify operation completed successfully
        assert final_result is not None
        assert final_result.response is not None
        assert final_result.tokens_used > 0

    def test_cost_aggregation_across_providers(self, mock_together_client):
        """Test cost aggregation across multiple providers."""
        # Shared governance configuration
        shared_config = {
            "team": "aggregation-test",
            "project": "multi-provider-costs",
            "customer_id": "customer-aggregation"
        }

        together_adapter = GenOpsTogetherAdapter(**shared_config)
        openai_adapter = MockOpenAIAdapter(**shared_config)

        # Perform operations
        test_message = [{"role": "user", "content": "Cost aggregation test"}]

        together_result = together_adapter.chat_with_governance(
            messages=test_message,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT
        )

        openai_result = openai_adapter.chat_with_governance(
            messages=test_message,
            model="gpt-3.5-turbo"
        )

        # Get cost summaries
        together_summary = together_adapter.get_cost_summary()
        openai_summary = openai_adapter.get_cost_summary()

        # Calculate total costs across providers
        total_together_cost = together_summary["daily_costs"]
        total_openai_cost = openai_summary["daily_costs"]
        total_all_providers = total_together_cost + total_openai_cost

        assert total_together_cost > 0
        assert total_openai_cost > 0
        assert total_all_providers > max(total_together_cost, total_openai_cost)

        # Verify cost attribution is maintained
        assert together_result.governance_attributes["customer_id"] == "customer-aggregation"
        assert openai_result.governance_attributes["customer_id"] == "customer-aggregation"


class TestCrossProviderCompatibility:
    """Test compatibility with existing provider patterns."""

    def test_api_interface_compatibility(self, mock_together_client):
        """Test API interface compatibility with other providers."""
        together_adapter = GenOpsTogetherAdapter()

        # Test standard interface methods exist
        assert hasattr(together_adapter, 'chat_with_governance')
        assert hasattr(together_adapter, 'get_cost_summary')
        assert hasattr(together_adapter, 'track_session')
        assert hasattr(together_adapter, '_calculate_cost')
        assert hasattr(together_adapter, '_create_governance_attributes')

        # Test that methods are callable
        assert callable(together_adapter.chat_with_governance)
        assert callable(together_adapter.get_cost_summary)
        assert callable(together_adapter.track_session)

    def test_governance_attribute_compatibility(self, mock_together_client):
        """Test governance attribute compatibility across providers."""
        standard_attributes = [
            "team", "project", "customer_id", "environment",
            "cost_center", "feature", "session_id"
        ]

        together_adapter = GenOpsTogetherAdapter(
            team="compatibility-test",
            project="attr-test",
            customer_id="test-customer",
            environment="test-env",
            cost_center="test-center"
        )

        result = together_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Attribute test"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            feature="attr-test-feature",
            session_id="test-session"
        )

        governance_attrs = result.governance_attributes

        # Verify all standard attributes are present
        for attr in standard_attributes:
            assert attr in governance_attrs
            assert governance_attrs[attr] is not None

    def test_cost_calculation_compatibility(self, together_adapter):
        """Test cost calculation method compatibility."""
        # Test that cost calculation follows expected patterns
        cost = together_adapter._calculate_cost(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            input_tokens=100,
            output_tokens=50
        )

        assert isinstance(cost, Decimal)
        assert cost > 0

        # Test with different token amounts
        cost_small = together_adapter._calculate_cost(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            input_tokens=10,
            output_tokens=5
        )

        cost_large = together_adapter._calculate_cost(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            input_tokens=1000,
            output_tokens=500
        )

        # Should scale proportionally
        assert cost_small < cost < cost_large
        assert cost_large > cost_small * 5  # Rough proportionality check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
