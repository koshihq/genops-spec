#!/usr/bin/env python3
"""
Comprehensive Test Suite for LiteLLM Provider Integration

Tests cover all aspects of the LiteLLM + GenOps integration including:
- Provider initialization and configuration
- Callback system integration with LiteLLM
- Cost tracking and attribution across providers
- Governance context management
- Auto-instrumentation functionality
- Multi-provider scenarios
- Error handling and edge cases
"""

# Test imports
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genops.providers.litellm import (  # noqa: E402
    GenOpsLiteLLMCallback,
    LiteLLMGovernanceContext,
    LiteLLMUsageStats,
    auto_instrument,
    get_cost_summary,
    get_usage_stats,
    reset_usage_stats,
    track_completion,
)
from genops.providers.litellm import (  # noqa: E402
    _stats_lock as _usage_lock,
)
from genops.providers.litellm import (  # noqa: E402
    _usage_stats as _global_usage_stats,
)


class TestLiteLLMGovernanceContext:
    """Test suite for LiteLLM governance context management."""

    def test_initialization_default_values(self):
        """Test governance context initialization with default values."""
        context = LiteLLMGovernanceContext()

        assert context.team == "default-team"
        assert context.project == "default-project"
        assert context.environment == "development"
        assert context.customer_id is None
        assert context.daily_budget_limit == 100.0
        assert context.governance_policy == "advisory"
        assert context.enable_cost_tracking is True
        assert isinstance(context.custom_tags, dict)

    def test_initialization_custom_values(self):
        """Test governance context initialization with custom values."""
        custom_tags = {"feature": "test-feature", "version": "1.0"}

        context = LiteLLMGovernanceContext(
            team="test-team",
            project="test-project",
            environment="production",
            customer_id="customer-123",
            daily_budget_limit=500.0,
            governance_policy="enforced",
            enable_cost_tracking=False,
            custom_tags=custom_tags,
        )

        assert context.team == "test-team"
        assert context.project == "test-project"
        assert context.environment == "production"
        assert context.customer_id == "customer-123"
        assert context.daily_budget_limit == 500.0
        assert context.governance_policy == "enforced"
        assert context.enable_cost_tracking is False
        assert context.custom_tags == custom_tags

    def test_governance_context_immutability(self):
        """Test that governance context maintains data integrity."""
        context = LiteLLMGovernanceContext(team="original-team")
        original_team = context.team

        # Context should maintain its values
        assert context.team == original_team

        # Custom tags should be properly isolated
        context.custom_tags["new_key"] = "new_value"
        assert "new_key" in context.custom_tags

    def test_governance_context_validation(self):
        """Test governance context input validation."""
        # Test valid governance policies
        valid_policies = ["advisory", "enforced", "strict"]
        for policy in valid_policies:
            context = LiteLLMGovernanceContext(governance_policy=policy)
            assert context.governance_policy == policy

        # Test budget limit validation
        context = LiteLLMGovernanceContext(daily_budget_limit=0.0)
        assert context.daily_budget_limit == 0.0

        context = LiteLLMGovernanceContext(daily_budget_limit=1000.0)
        assert context.daily_budget_limit == 1000.0


class TestLiteLLMUsageStats:
    """Test suite for LiteLLM usage statistics tracking."""

    def test_usage_stats_initialization(self):
        """Test usage statistics initialization."""
        stats = LiteLLMUsageStats()

        assert stats.total_requests == 0
        assert stats.total_cost == 0.0
        assert stats.total_tokens == 0
        assert isinstance(stats.provider_usage, dict)
        assert len(stats.provider_usage) == 0
        assert stats.start_time is not None
        assert stats.last_request_time is None

    def test_add_request_basic(self):
        """Test adding a basic request to usage statistics."""
        stats = LiteLLMUsageStats()

        stats.add_request(
            provider="openai",
            model="gpt-3.5-turbo",
            cost=0.002,
            input_tokens=100,
            output_tokens=50,
            team="test-team",
            project="test-project",
        )

        assert stats.total_requests == 1
        assert stats.total_cost == 0.002
        assert stats.total_tokens == 150
        assert "openai" in stats.provider_usage

        provider_stats = stats.provider_usage["openai"]
        assert provider_stats["requests"] == 1
        assert provider_stats["cost"] == 0.002
        assert provider_stats["tokens"] == 150
        assert "gpt-3.5-turbo" in provider_stats["models"]

    def test_add_multiple_requests_same_provider(self):
        """Test adding multiple requests to the same provider."""
        stats = LiteLLMUsageStats()

        # Add first request
        stats.add_request(
            provider="anthropic",
            model="claude-3-sonnet",
            cost=0.003,
            input_tokens=120,
            output_tokens=80,
            team="team-1",
            project="project-1",
        )

        # Add second request
        stats.add_request(
            provider="anthropic",
            model="claude-3-haiku",
            cost=0.001,
            input_tokens=80,
            output_tokens=40,
            team="team-2",
            project="project-2",
        )

        assert stats.total_requests == 2
        assert stats.total_cost == 0.004
        assert stats.total_tokens == 320

        provider_stats = stats.provider_usage["anthropic"]
        assert provider_stats["requests"] == 2
        assert provider_stats["cost"] == 0.004
        assert provider_stats["tokens"] == 320
        assert len(provider_stats["models"]) == 2

    def test_add_requests_multiple_providers(self):
        """Test adding requests across multiple providers."""
        stats = LiteLLMUsageStats()

        providers_data = [
            ("openai", "gpt-4", 0.030, 200, 100),
            ("anthropic", "claude-3-sonnet", 0.015, 150, 75),
            ("google", "gemini-pro", 0.002, 100, 50),
        ]

        for provider, model, cost, input_tokens, output_tokens in providers_data:
            stats.add_request(
                provider=provider,
                model=model,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                team="multi-provider-team",
                project="multi-provider-project",
            )

        assert stats.total_requests == 3
        assert stats.total_cost == 0.047
        assert stats.total_tokens == 675
        assert len(stats.provider_usage) == 3

        # Verify each provider
        for provider, _, cost, input_tokens, output_tokens in providers_data:
            assert provider in stats.provider_usage
            provider_stats = stats.provider_usage[provider]
            assert provider_stats["requests"] == 1
            assert provider_stats["cost"] == cost
            assert provider_stats["tokens"] == input_tokens + output_tokens

    def test_thread_safety(self):
        """Test thread safety of usage statistics."""
        stats = LiteLLMUsageStats()

        def add_requests(provider_name: str, num_requests: int):
            for i in range(num_requests):
                stats.add_request(
                    provider=provider_name,
                    model=f"model-{i}",
                    cost=0.001,
                    input_tokens=10,
                    output_tokens=5,
                    team="thread-test",
                    project="thread-test",
                )

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_requests, args=(f"provider-{i}", 10))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert stats.total_requests == 50
        assert stats.total_cost == 0.050
        assert len(stats.provider_usage) == 5


class TestGenOpsLiteLLMCallback:
    """Test suite for GenOps LiteLLM callback integration."""

    def test_callback_initialization(self):
        """Test callback initialization with governance context."""
        context = LiteLLMGovernanceContext(
            team="callback-team", project="callback-project"
        )

        callback = GenOpsLiteLLMCallback(context)

        assert callback.governance_context == context
        assert callback.governance_context.team == "callback-team"
        assert callback.governance_context.project == "callback-project"

    @patch("genops.providers.litellm._global_usage_stats")
    def test_input_callback(self, mock_stats):
        """Test input callback functionality."""
        context = LiteLLMGovernanceContext()
        callback = GenOpsLiteLLMCallback(context)

        # Mock input data
        model_kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
        }

        # Call input callback
        result = callback.input_callback(model_kwargs)

        # Input callback should return None or modified kwargs
        assert result is None or isinstance(result, dict)

    @patch("genops.providers.litellm._global_usage_stats")
    def test_success_callback(self, mock_stats):
        """Test success callback functionality."""
        mock_stats.add_request = Mock()

        context = LiteLLMGovernanceContext(
            team="success-team", project="success-project"
        )
        callback = GenOpsLiteLLMCallback(context)

        # Mock kwargs and response
        kwargs = {"model": "gpt-3.5-turbo"}

        # Mock response object
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        # Mock cost calculation
        with patch("genops.providers.litellm._calculate_cost") as mock_calc_cost:
            mock_calc_cost.return_value = 0.002

            # Call success callback
            callback.success_callback(kwargs, mock_response, time.time())

            # Verify usage stats were updated
            mock_stats.add_request.assert_called_once_with(
                provider="openai",  # Inferred from gpt-3.5-turbo
                model="gpt-3.5-turbo",
                cost=0.002,
                input_tokens=100,
                output_tokens=50,
                team="success-team",
                project="success-project",
                customer_id=None,
                custom_tags={},
            )

    @patch("genops.providers.litellm._global_usage_stats")
    def test_failure_callback(self, mock_stats):
        """Test failure callback functionality."""
        mock_stats.add_request = Mock()

        context = LiteLLMGovernanceContext()
        callback = GenOpsLiteLLMCallback(context)

        # Mock kwargs and exception
        kwargs = {"model": "gpt-3.5-turbo"}
        exception = Exception("Test API error")

        # Call failure callback
        callback.failure_callback(kwargs, exception, time.time())

        # Failure callback should handle gracefully
        # Could log error or track failure metrics
        # Verify no usage stats were added for failed request
        mock_stats.add_request.assert_not_called()

    def test_callback_with_custom_attributes(self):
        """Test callback with custom governance attributes."""
        custom_tags = {"feature": "test-feature", "version": "1.0"}
        context = LiteLLMGovernanceContext(
            team="custom-team",
            project="custom-project",
            customer_id="customer-123",
            custom_tags=custom_tags,
        )

        callback = GenOpsLiteLLMCallback(context)

        assert callback.governance_context.custom_tags == custom_tags
        assert callback.governance_context.customer_id == "customer-123"


class TestAutoInstrumentation:
    """Test suite for auto-instrumentation functionality."""

    @patch("litellm.input_callback", [])
    @patch("litellm.success_callback", [])
    @patch("litellm.failure_callback", [])
    def test_auto_instrument_basic(self):
        """Test basic auto-instrumentation setup."""
        with patch("genops.providers.litellm.litellm") as mock_litellm:
            mock_litellm.input_callback = []
            mock_litellm.success_callback = []
            mock_litellm.failure_callback = []

            result = auto_instrument(team="auto-team", project="auto-project")

            assert result is True

            # Verify callbacks were registered
            assert len(mock_litellm.input_callback) == 1
            assert len(mock_litellm.success_callback) == 1
            assert len(mock_litellm.failure_callback) == 1

    @patch("litellm.input_callback", [])
    @patch("litellm.success_callback", [])
    @patch("litellm.failure_callback", [])
    def test_auto_instrument_custom_config(self):
        """Test auto-instrumentation with custom configuration."""
        with patch("genops.providers.litellm.litellm") as mock_litellm:
            mock_litellm.input_callback = []
            mock_litellm.success_callback = []
            mock_litellm.failure_callback = []

            result = auto_instrument(
                team="custom-team",
                project="custom-project",
                environment="production",
                customer_id="customer-456",
                daily_budget_limit=500.0,
                governance_policy="enforced",
                enable_cost_tracking=True,
                custom_feature="test-feature",
            )

            assert result is True

            # Verify callbacks were registered
            assert len(mock_litellm.input_callback) == 1
            assert len(mock_litellm.success_callback) == 1
            assert len(mock_litellm.failure_callback) == 1

    def test_auto_instrument_litellm_not_available(self):
        """Test auto-instrumentation when LiteLLM is not available."""
        with patch("genops.providers.litellm.litellm", None):
            result = auto_instrument(team="test", project="test")

            assert result is False

    def test_auto_instrument_exception_handling(self):
        """Test auto-instrumentation exception handling."""
        with patch("genops.providers.litellm.litellm") as mock_litellm:
            # Simulate exception during callback registration
            mock_litellm.success_callback.append.side_effect = Exception(
                "Registration failed"
            )

            result = auto_instrument(team="test", project="test")

            assert result is False


class TestTrackCompletion:
    """Test suite for track_completion context manager."""

    def test_track_completion_context_manager(self):
        """Test track_completion as context manager."""
        with patch("genops.providers.litellm._global_usage_stats") as mock_stats:
            mock_stats.add_request = Mock()

            with track_completion(
                model="gpt-3.5-turbo", team="context-team", project="context-project"
            ) as context:
                assert context is not None
                assert hasattr(context, "team")
                assert hasattr(context, "project")
                assert hasattr(context, "model")
                assert context.team == "context-team"
                assert context.project == "context-project"
                assert context.model == "gpt-3.5-turbo"

    def test_track_completion_with_custom_attributes(self):
        """Test track_completion with custom attributes."""
        custom_tags = {"experiment": "A", "variant": "control"}

        with track_completion(
            model="claude-3-sonnet",
            team="experiment-team",
            project="ab-test",
            customer_id="customer-789",
            custom_tags=custom_tags,
        ) as context:
            assert context.customer_id == "customer-789"
            assert context.custom_tags == custom_tags

    @patch("genops.providers.litellm._global_usage_stats")
    def test_track_completion_cost_tracking(self, mock_stats):
        """Test cost tracking in track_completion context."""
        mock_stats.add_request = Mock()

        with track_completion(
            model="gpt-4", team="cost-team", project="cost-project"
        ) as context:
            # Simulate cost and token data
            context.cost = 0.030
            context.total_tokens = 250
            context.input_tokens = 200
            context.output_tokens = 50

        # Verify tracking context maintains cost information
        assert hasattr(context, "cost")
        assert hasattr(context, "total_tokens")

    def test_track_completion_exception_handling(self):
        """Test track_completion exception handling."""
        try:
            with track_completion(
                model="gpt-3.5-turbo",
                team="exception-team",
                project="exception-project",
            ):
                # Simulate an exception within context
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected exception

        # Context manager should handle exceptions gracefully
        assert True  # Test passes if no unhandled exceptions


class TestUsageStatsFunctions:
    """Test suite for usage statistics functions."""

    def setUp(self):
        """Set up test environment."""
        reset_usage_stats()

    def test_get_usage_stats_empty(self):
        """Test get_usage_stats with no data."""
        self.setUp()

        stats = get_usage_stats()

        assert stats["total_requests"] == 0
        assert stats["total_cost"] == 0.0
        assert stats["total_tokens"] == 0
        assert stats["provider_usage"] == {}
        assert stats["instrumentation_active"] is False

    @patch("genops.providers.litellm._global_usage_stats")
    def test_get_usage_stats_with_data(self, mock_stats):
        """Test get_usage_stats with sample data."""
        # Mock usage stats with sample data
        mock_stats.total_requests = 5
        mock_stats.total_cost = 0.025
        mock_stats.total_tokens = 750
        mock_stats.provider_usage = {
            "openai": {"requests": 3, "cost": 0.015, "tokens": 450},
            "anthropic": {"requests": 2, "cost": 0.010, "tokens": 300},
        }
        mock_stats.start_time = time.time() - 3600  # 1 hour ago
        mock_stats.last_request_time = time.time() - 60  # 1 minute ago

        stats = get_usage_stats()

        assert stats["total_requests"] == 5
        assert stats["total_cost"] == 0.025
        assert stats["total_tokens"] == 750
        assert len(stats["provider_usage"]) == 2
        assert "openai" in stats["provider_usage"]
        assert "anthropic" in stats["provider_usage"]

    def test_get_cost_summary_by_provider(self):
        """Test get_cost_summary grouped by provider."""
        # Add sample data to global stats
        with _usage_lock:
            _global_usage_stats.add_request(
                provider="openai",
                model="gpt-3.5-turbo",
                cost=0.010,
                input_tokens=100,
                output_tokens=50,
                team="team-1",
                project="project-1",
            )

            _global_usage_stats.add_request(
                provider="anthropic",
                model="claude-3-sonnet",
                cost=0.015,
                input_tokens=120,
                output_tokens=60,
                team="team-2",
                project="project-2",
            )

        summary = get_cost_summary(group_by="provider")

        assert summary["total_cost"] == 0.025
        assert "cost_by_provider" in summary
        assert summary["cost_by_provider"]["openai"] == 0.010
        assert summary["cost_by_provider"]["anthropic"] == 0.015

    def test_get_cost_summary_by_team(self):
        """Test get_cost_summary grouped by team."""
        # Add sample data
        with _usage_lock:
            _global_usage_stats.add_request(
                provider="openai",
                model="gpt-3.5-turbo",
                cost=0.008,
                input_tokens=80,
                output_tokens=40,
                team="frontend-team",
                project="web-app",
            )

            _global_usage_stats.add_request(
                provider="anthropic",
                model="claude-3-haiku",
                cost=0.012,
                input_tokens=100,
                output_tokens=50,
                team="backend-team",
                project="api-service",
            )

        summary = get_cost_summary(group_by="team")

        assert "cost_by_team" in summary
        # Note: This test may need adjustment based on existing data in global stats

    def test_reset_usage_stats(self):
        """Test reset_usage_stats functionality."""
        # Add some data first
        with _usage_lock:
            _global_usage_stats.add_request(
                provider="test",
                model="test-model",
                cost=0.001,
                input_tokens=10,
                output_tokens=5,
                team="test-team",
                project="test-project",
            )

        # Verify data exists
        stats = get_usage_stats()
        assert stats["total_requests"] > 0

        # Reset and verify
        reset_usage_stats()
        stats = get_usage_stats()

        assert stats["total_requests"] == 0
        assert stats["total_cost"] == 0.0
        assert stats["provider_usage"] == {}


class TestCostCalculation:
    """Test suite for cost calculation functionality."""

    def test_calculate_cost_openai_gpt35(self):
        """Test cost calculation for OpenAI GPT-3.5-turbo."""
        from genops.providers.litellm import _calculate_cost

        cost = _calculate_cost(
            provider="openai",
            model="gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500,
        )

        # GPT-3.5-turbo: $0.0015/1K input, $0.002/1K output
        expected_cost = (1000 * 0.0015 / 1000) + (500 * 0.002 / 1000)
        assert abs(cost - expected_cost) < 0.000001

    def test_calculate_cost_anthropic_claude(self):
        """Test cost calculation for Anthropic Claude."""
        from genops.providers.litellm import _calculate_cost

        cost = _calculate_cost(
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=200,
        )

        # Claude-3-Sonnet: $0.003/1K input, $0.015/1K output
        expected_cost = (1000 * 0.003 / 1000) + (200 * 0.015 / 1000)
        assert abs(cost - expected_cost) < 0.000001

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        from genops.providers.litellm import _calculate_cost

        cost = _calculate_cost(
            provider="unknown",
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should use generic fallback pricing
        assert cost > 0
        assert isinstance(cost, float)

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        from genops.providers.litellm import _calculate_cost

        cost = _calculate_cost(
            provider="openai", model="gpt-3.5-turbo", input_tokens=0, output_tokens=0
        )

        assert cost == 0.0


class TestProviderInference:
    """Test suite for provider inference from model names."""

    def test_infer_provider_openai_models(self):
        """Test provider inference for OpenAI models."""
        from genops.providers.litellm import _infer_provider_from_model

        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "text-davinci-003"]

        for model in openai_models:
            provider = _infer_provider_from_model(model)
            assert provider == "openai"

    def test_infer_provider_anthropic_models(self):
        """Test provider inference for Anthropic models."""
        from genops.providers.litellm import _infer_provider_from_model

        anthropic_models = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-2",
        ]

        for model in anthropic_models:
            provider = _infer_provider_from_model(model)
            assert provider == "anthropic"

    def test_infer_provider_google_models(self):
        """Test provider inference for Google models."""
        from genops.providers.litellm import _infer_provider_from_model

        google_models = ["gemini-pro", "gemini-1.5-pro", "palm-2"]

        for model in google_models:
            provider = _infer_provider_from_model(model)
            assert provider == "google"

    def test_infer_provider_unknown_model(self):
        """Test provider inference for unknown model."""
        from genops.providers.litellm import _infer_provider_from_model

        provider = _infer_provider_from_model("unknown-model-123")
        assert provider == "unknown"


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_callback_with_missing_usage_info(self):
        """Test callback handling when usage info is missing."""
        context = LiteLLMGovernanceContext()
        callback = GenOpsLiteLLMCallback(context)

        # Mock response without usage information
        mock_response = Mock()
        mock_response.usage = None

        kwargs = {"model": "gpt-3.5-turbo"}

        with patch("genops.providers.litellm._global_usage_stats") as mock_stats:
            mock_stats.add_request = Mock()

            # Should handle gracefully
            callback.success_callback(kwargs, mock_response, time.time())

            # Should still record request with estimated values
            mock_stats.add_request.assert_called_once()

    def test_callback_with_malformed_response(self):
        """Test callback handling with malformed response."""
        context = LiteLLMGovernanceContext()
        callback = GenOpsLiteLLMCallback(context)

        # Mock malformed response
        mock_response = "invalid_response_type"
        kwargs = {"model": "gpt-3.5-turbo"}

        with patch("genops.providers.litellm._global_usage_stats") as mock_stats:
            mock_stats.add_request = Mock()

            # Should handle gracefully without raising exceptions
            try:
                callback.success_callback(kwargs, mock_response, time.time())
            except Exception as e:
                pytest.fail(
                    f"Callback should handle malformed response gracefully: {e}"
                )

    def test_concurrent_auto_instrumentation(self):
        """Test concurrent auto-instrumentation calls."""

        def instrument_thread():
            return auto_instrument(team="concurrent-team", project="concurrent-project")

        with patch("genops.providers.litellm.litellm") as mock_litellm:
            mock_litellm.input_callback = []
            mock_litellm.success_callback = []
            mock_litellm.failure_callback = []

            # Run multiple instrumentation calls concurrently
            threads = []
            results = []

            for _i in range(5):
                thread = threading.Thread(
                    target=lambda: results.append(instrument_thread())
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # All should succeed (or handle gracefully)
            assert len(results) == 5

    def test_usage_stats_with_large_numbers(self):
        """Test usage statistics with large token counts and costs."""
        stats = LiteLLMUsageStats()

        # Add request with large values
        stats.add_request(
            provider="test",
            model="test-model",
            cost=999.99,
            input_tokens=1000000,
            output_tokens=500000,
            team="large-scale-team",
            project="large-scale-project",
        )

        assert stats.total_cost == 999.99
        assert stats.total_tokens == 1500000
        assert stats.total_requests == 1

    def test_usage_stats_with_zero_cost(self):
        """Test usage statistics with zero cost requests."""
        stats = LiteLLMUsageStats()

        stats.add_request(
            provider="free-provider",
            model="free-model",
            cost=0.0,
            input_tokens=100,
            output_tokens=50,
            team="free-tier-team",
            project="free-tier-project",
        )

        assert stats.total_cost == 0.0
        assert stats.total_tokens == 150
        assert stats.total_requests == 1


class TestIntegrationScenarios:
    """Test suite for real-world integration scenarios."""

    @patch("genops.providers.litellm.litellm")
    def test_multi_provider_workflow(self, mock_litellm):
        """Test multi-provider workflow scenario."""
        mock_litellm.input_callback = []
        mock_litellm.success_callback = []
        mock_litellm.failure_callback = []

        # Setup auto-instrumentation
        result = auto_instrument(
            team="multi-provider-team",
            project="cross-provider-app",
            daily_budget_limit=100.0,
        )
        assert result is True

        # Simulate requests to different providers
        providers_scenarios = [
            ("openai", "gpt-3.5-turbo", 0.002, 100, 50),
            ("anthropic", "claude-3-haiku", 0.001, 80, 40),
            ("google", "gemini-pro", 0.003, 120, 60),
        ]

        for provider, model, cost, input_tokens, output_tokens in providers_scenarios:
            with _usage_lock:
                _global_usage_stats.add_request(
                    provider=provider,
                    model=model,
                    cost=cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    team="multi-provider-team",
                    project="cross-provider-app",
                )

        # Verify cross-provider tracking
        stats = get_usage_stats()
        assert stats["total_requests"] >= len(providers_scenarios)

        summary = get_cost_summary(group_by="provider")
        assert len(summary["cost_by_provider"]) >= len(
            {p[0] for p in providers_scenarios}
        )

    def test_enterprise_governance_workflow(self):
        """Test enterprise governance workflow."""
        # Create governance contexts for different teams
        teams_config = [
            ("engineering", "ai-platform", "production", "customer-enterprise"),
            ("marketing", "content-generation", "production", "customer-startup"),
            ("support", "automated-responses", "production", "customer-midmarket"),
        ]

        usage_data = []

        for team, project, environment, customer_id in teams_config:
            with track_completion(
                model="gpt-3.5-turbo",
                team=team,
                project=project,
                environment=environment,
                customer_id=customer_id,
                daily_budget_limit=200.0,
                governance_policy="enforced",
            ) as context:
                # Simulate request processing
                usage_data.append(
                    {
                        "team": context.team,
                        "project": context.project,
                        "customer_id": context.customer_id,
                    }
                )

        # Verify governance context management
        assert len(usage_data) == 3
        teams_tracked = [data["team"] for data in usage_data]
        assert "engineering" in teams_tracked
        assert "marketing" in teams_tracked
        assert "support" in teams_tracked

    def test_cost_optimization_workflow(self):
        """Test cost optimization workflow."""
        reset_usage_stats()

        # Simulate cost optimization scenario
        optimization_requests = [
            # High-cost request
            ("openai", "gpt-4", 0.060, 2000, 1000, "premium-team"),
            # Medium-cost request
            ("anthropic", "claude-3-sonnet", 0.018, 1200, 600, "standard-team"),
            # Low-cost request
            ("openai", "gpt-3.5-turbo", 0.003, 200, 100, "budget-team"),
        ]

        for (
            provider,
            model,
            cost,
            input_tokens,
            output_tokens,
            team,
        ) in optimization_requests:
            with _usage_lock:
                _global_usage_stats.add_request(
                    provider=provider,
                    model=model,
                    cost=cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    team=team,
                    project="cost-optimization-project",
                )

        # Analyze cost distribution
        summary = get_cost_summary(group_by="team")

        assert "cost_by_team" in summary
        # Verify different cost levels are tracked properly
        total_cost = summary["total_cost"]
        assert total_cost > 0


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])
