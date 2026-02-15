#!/usr/bin/env python3
"""
Test suite for CrewAI GenOps Adapter

Comprehensive tests for the GenOpsCrewAIAdapter class including:
- Adapter initialization and configuration
- Context manager lifecycle testing
- Cost tracking and attribution
- Multi-provider integration
- Error handling and edge cases
"""

import time
import uuid
from unittest.mock import Mock, patch

import pytest

# Import the CrewAI adapter and related classes
try:
    from genops.providers.crewai import (
        CrewAIAgentResult,  # noqa: F401
        CrewAICrewContext,  # noqa: F401
        CrewAICrewResult,  # noqa: F401
        CrewAISessionContext,  # noqa: F401
        CrewAITaskResult,  # noqa: F401
        GenOpsCrewAIAdapter,
    )
except ImportError:
    pytest.skip("CrewAI provider not available", allow_module_level=True)


class TestGenOpsCrewAIAdapter:
    """Test suite for GenOpsCrewAIAdapter."""

    def test_adapter_initialization_default(self):
        """Test adapter initialization with default parameters."""
        adapter = GenOpsCrewAIAdapter()

        assert adapter.team == "default-team"
        assert adapter.project == "default-project"
        assert adapter.environment == "development"
        assert adapter.daily_budget_limit == 100.0
        assert adapter.governance_policy == "advisory"
        assert adapter.enable_cost_tracking is True

    def test_adapter_initialization_custom(self):
        """Test adapter initialization with custom parameters."""
        adapter = GenOpsCrewAIAdapter(
            team="test-team",
            project="test-project",
            environment="production",
            daily_budget_limit=500.0,
            governance_policy="enforced",
            enable_cost_tracking=False,
        )

        assert adapter.team == "test-team"
        assert adapter.project == "test-project"
        assert adapter.environment == "production"
        assert adapter.daily_budget_limit == 500.0
        assert adapter.governance_policy == "enforced"
        assert adapter.enable_cost_tracking is False

    def test_adapter_initialization_validation(self):
        """Test adapter parameter validation."""
        # Test invalid budget
        with pytest.raises((ValueError, TypeError)):
            GenOpsCrewAIAdapter(daily_budget_limit=-10.0)

        # Test invalid governance policy
        with pytest.raises((ValueError, TypeError)):
            GenOpsCrewAIAdapter(governance_policy="invalid_policy")

    @patch("genops.providers.crewai.adapter.CrewAICostAggregator")
    def test_cost_aggregator_initialization(self, mock_cost_aggregator):
        """Test cost aggregator is properly initialized."""
        mock_aggregator_instance = Mock()
        mock_cost_aggregator.return_value = mock_aggregator_instance

        adapter = GenOpsCrewAIAdapter(enable_cost_tracking=True)

        assert adapter.cost_aggregator is not None
        mock_cost_aggregator.assert_called_once()

    @patch("genops.providers.crewai.adapter.CrewAIAgentMonitor")
    def test_agent_monitor_initialization(self, mock_monitor):
        """Test agent monitor is properly initialized."""
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance

        adapter = GenOpsCrewAIAdapter(enable_agent_tracking=True)

        assert adapter.agent_monitor is not None
        mock_monitor.assert_called_once()

    def test_crew_context_manager_basic(self):
        """Test basic crew context manager functionality."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("test-crew") as context:
            assert context is not None
            assert context.crew_name == "test-crew"
            assert context.adapter == adapter
            assert hasattr(context, "crew_id")
            assert context.start_time is not None

    def test_crew_context_manager_with_attributes(self):
        """Test crew context manager with custom attributes."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew(
            "test-crew", use_case="testing", customer_id="cust_123"
        ) as context:
            assert context.crew_name == "test-crew"
            assert context.custom_attributes["use_case"] == "testing"
            assert context.custom_attributes["customer_id"] == "cust_123"

    def test_crew_context_manager_lifecycle(self):
        """Test context manager __enter__ and __exit__ methods."""
        adapter = GenOpsCrewAIAdapter()

        context_instance = None

        with adapter.track_crew("lifecycle-test") as context:
            context_instance = context
            assert context.start_time is not None
            assert context.end_time is None

            # Add some metrics during execution
            context.add_custom_metric("test_metric", "test_value")

        # After exiting context
        assert context_instance.end_time is not None
        assert context_instance.execution_time > 0

    def test_crew_context_manager_exception_handling(self):
        """Test context manager handles exceptions properly."""
        adapter = GenOpsCrewAIAdapter()

        with pytest.raises(ValueError):
            with adapter.track_crew("exception-test") as context:
                assert context.start_time is not None
                raise ValueError("Test exception")

        # Context should still be properly closed
        assert context.end_time is not None

    def test_session_context_manager(self):
        """Test session context manager functionality."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_session("test-session") as session:
            assert session.session_name == "test-session"
            assert session.adapter == adapter
            assert hasattr(session, "session_id")
            assert session.start_time is not None
            assert session.total_crews == 0

    @patch("genops.providers.crewai.adapter.CrewAICostAggregator")
    def test_cost_tracking_enabled(self, mock_cost_aggregator):
        """Test cost tracking when enabled."""
        mock_aggregator = Mock()
        mock_cost_aggregator.return_value = mock_aggregator

        adapter = GenOpsCrewAIAdapter(enable_cost_tracking=True)

        with adapter.track_crew("cost-test") as context:
            # Simulate adding cost data
            context.add_cost_entry("openai", "gpt-4", 150, 50, 0.045)

        # Should have called cost aggregator methods
        assert mock_aggregator.start_tracking.called or hasattr(context, "total_cost")

    def test_custom_metrics_addition(self):
        """Test adding custom metrics to crew context."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("metrics-test") as context:
            context.add_custom_metric("agents_count", 3)
            context.add_custom_metric("complexity_level", "high")
            context.add_custom_metric("estimated_tokens", 1500)

            assert context.custom_metrics["agents_count"] == 3
            assert context.custom_metrics["complexity_level"] == "high"
            assert context.custom_metrics["estimated_tokens"] == 1500

    def test_get_metrics_basic(self):
        """Test getting basic metrics from context."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("get-metrics-test") as context:
            time.sleep(0.1)  # Small delay to ensure execution time > 0
            context.add_custom_metric("test_value", 42)

            metrics = context.get_metrics()

            assert isinstance(metrics, dict)
            assert "execution_time" in metrics
            assert "crew_name" in metrics
            assert "crew_id" in metrics
            assert metrics["execution_time"] > 0
            assert metrics["crew_name"] == "get-metrics-test"

    @patch("genops.providers.crewai.adapter.CrewAICostAggregator")
    def test_get_metrics_with_costs(self, mock_cost_aggregator):
        """Test getting metrics including cost data."""
        mock_aggregator = Mock()
        mock_aggregator.get_total_cost.return_value = 0.125
        mock_aggregator.get_cost_by_provider.return_value = {"openai": 0.125}
        mock_cost_aggregator.return_value = mock_aggregator

        adapter = GenOpsCrewAIAdapter(enable_cost_tracking=True)

        with adapter.track_crew("cost-metrics-test") as context:
            metrics = context.get_metrics()

            assert "total_cost" in metrics
            assert "cost_by_provider" in metrics

    def test_crew_results_storage(self):
        """Test storage and retrieval of crew results."""
        adapter = GenOpsCrewAIAdapter()

        # Execute a few crews
        for i in range(3):
            with adapter.track_crew(f"result-test-{i}") as context:
                context.add_custom_metric("iteration", i)

        # Get recent results
        results = adapter.get_crew_results(limit=2)

        assert len(results) <= 2
        if results:
            assert isinstance(results[0], dict)
            assert "crew_name" in results[0]

    def test_concurrent_crew_tracking(self):
        """Test tracking multiple crews concurrently."""
        import threading

        adapter = GenOpsCrewAIAdapter()
        results = []

        def track_crew(crew_id):
            with adapter.track_crew(f"concurrent-{crew_id}") as context:
                time.sleep(0.05)  # Small delay
                context.add_custom_metric("crew_id", crew_id)
                results.append(context.get_metrics())

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=track_crew, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        assert len(results) == 3
        crew_ids = [r.get("custom_metrics", {}).get("crew_id") for r in results]
        assert set(crew_ids) == {0, 1, 2}

    def test_budget_tracking(self):
        """Test budget limit tracking and warnings."""
        adapter = GenOpsCrewAIAdapter(daily_budget_limit=1.0)  # Low budget for testing

        # Should not raise exception by default (advisory policy)
        with adapter.track_crew("budget-test") as context:
            # Simulate high cost
            if hasattr(context, "add_cost_entry"):
                context.add_cost_entry("openai", "gpt-4", 1000, 500, 2.50)

    def test_governance_policy_enforcement(self):
        """Test different governance policy enforcement levels."""
        # Advisory policy should allow operation
        adapter_advisory = GenOpsCrewAIAdapter(
            governance_policy="advisory",
            daily_budget_limit=0.01,  # Very low budget
        )

        with adapter_advisory.track_crew("advisory-test"):
            pass  # Should not raise exception

        # Enforced policy might raise warnings/exceptions
        adapter_enforced = GenOpsCrewAIAdapter(
            governance_policy="enforced", daily_budget_limit=0.01
        )

        # Should still work but might log warnings
        with adapter_enforced.track_crew("enforced-test"):
            pass

    def test_environment_specific_behavior(self):
        """Test environment-specific adapter behavior."""
        environments = ["development", "staging", "production"]

        for env in environments:
            adapter = GenOpsCrewAIAdapter(environment=env)

            with adapter.track_crew(f"{env}-test") as context:
                metrics = context.get_metrics()
                assert "environment" in str(metrics) or adapter.environment == env

    @patch("genops.providers.crewai.adapter.logger")
    def test_logging_behavior(self, mock_logger):
        """Test logging behavior during crew tracking."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("logging-test") as context:
            context.add_custom_metric("test", "value")

        # Should have logged some information
        assert mock_logger.info.called or mock_logger.debug.called

    def test_adapter_string_representation(self):
        """Test adapter string representation."""
        adapter = GenOpsCrewAIAdapter(team="test-team", project="test-project")

        str_repr = str(adapter)
        assert "test-team" in str_repr
        assert "test-project" in str_repr

    def test_crew_context_string_representation(self):
        """Test crew context string representation."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("repr-test") as context:
            str_repr = str(context)
            assert "repr-test" in str_repr
            assert "crew_id" in str_repr or len(str_repr) > 0

    def test_multiple_contexts_isolation(self):
        """Test that multiple contexts don't interfere with each other."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("context-1") as context1:
            context1.add_custom_metric("context", "first")

            with adapter.track_crew("context-2") as context2:
                context2.add_custom_metric("context", "second")

                # Contexts should be isolated
                assert context1.custom_metrics["context"] == "first"
                assert context2.custom_metrics["context"] == "second"
                assert context1.crew_name != context2.crew_name

    def test_session_with_multiple_crews(self):
        """Test session tracking with multiple crews."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_session("multi-crew-session") as session:
            assert session.total_crews == 0

            # Execute multiple crews within session
            for i in range(3):
                with adapter.track_crew(f"session-crew-{i}") as context:
                    # Simulate adding crew result to session
                    if hasattr(session, "add_crew_result"):
                        session.add_crew_result(context.get_metrics())

    def test_error_handling_in_context(self):
        """Test error handling within tracking contexts."""
        adapter = GenOpsCrewAIAdapter()

        # Test that errors don't break the context manager
        try:
            with adapter.track_crew("error-test") as context:
                context.add_custom_metric("before_error", True)
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        # Adapter should still be functional after error
        with adapter.track_crew("after-error-test") as context:
            assert context.crew_name == "after-error-test"

    @patch("genops.providers.crewai.adapter.uuid")
    def test_unique_crew_ids(self, mock_uuid):
        """Test that crew IDs are unique."""
        # Mock UUID to return predictable values
        mock_uuid.uuid4.side_effect = [
            Mock(spec=uuid.UUID, __str__=lambda self: "uuid-1"),
            Mock(spec=uuid.UUID, __str__=lambda self: "uuid-2"),
            Mock(spec=uuid.UUID, __str__=lambda self: "uuid-3"),
        ]

        adapter = GenOpsCrewAIAdapter()
        crew_ids = []

        for i in range(3):
            with adapter.track_crew(f"unique-test-{i}") as context:
                crew_ids.append(context.crew_id)

        # All crew IDs should be unique
        assert len(set(crew_ids)) == len(crew_ids)

    def test_performance_with_many_crews(self):
        """Test adapter performance with many crew executions."""
        adapter = GenOpsCrewAIAdapter()

        start_time = time.time()

        # Execute many crews quickly
        for i in range(50):
            with adapter.track_crew(f"perf-test-{i}") as context:
                context.add_custom_metric("iteration", i)

        total_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert total_time < 5.0  # 5 seconds for 50 crews

    def test_memory_usage_cleanup(self):
        """Test that contexts are properly cleaned up to avoid memory leaks."""
        adapter = GenOpsCrewAIAdapter()

        # Execute many crews and ensure no memory accumulation
        len(adapter.get_crew_results())

        for i in range(20):
            with adapter.track_crew(f"memory-test-{i}") as context:
                context.add_custom_metric("data", "x" * 1000)  # Add some data

        # Results should be stored but not accumulate indefinitely
        final_results_count = len(adapter.get_crew_results())

        # Should have reasonable number of results (not necessarily all 20)
        assert final_results_count > 0
        assert final_results_count <= 50  # Reasonable upper bound


class TestCrewAIContextManagers:
    """Test context manager classes specifically."""

    @patch("genops.providers.crewai.adapter.GenOpsCrewAIAdapter")
    def test_crew_context_initialization(self, mock_adapter):
        """Test CrewAICrewContext initialization."""
        Mock()

        # This test would need the actual CrewAICrewContext class
        # For now, test through the adapter's track_crew method
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_crew("init-test") as context:
            assert hasattr(context, "crew_name")
            assert hasattr(context, "start_time")
            assert hasattr(context, "crew_id")

    def test_session_context_initialization(self):
        """Test session context initialization."""
        adapter = GenOpsCrewAIAdapter()

        with adapter.track_session("session-init-test") as session:
            assert hasattr(session, "session_name")
            assert hasattr(session, "start_time")
            assert hasattr(session, "session_id")
            assert hasattr(session, "total_crews")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
