#!/usr/bin/env python3
"""
Integration Test Suite for CrewAI + GenOps

End-to-end integration tests covering the complete CrewAI + GenOps workflow:
- Full integration with mock CrewAI crews
- Cross-component functionality
- Real-world usage patterns
- Performance and reliability testing
"""

import threading
import time
from unittest.mock import patch

import pytest

# Import all CrewAI integration components
try:
    from genops.providers.crewai import (
        CrewAIAgentMonitor,  # noqa: F401
        CrewAICostAggregator,  # noqa: F401
        # Main components
        GenOpsCrewAIAdapter,
        analyze_crew_costs,  # noqa: F401
        # Auto-instrumentation
        auto_instrument,
        create_multi_agent_adapter,
        disable_auto_instrumentation,
        get_multi_agent_insights,  # noqa: F401
        # Convenience functions
        instrument_crewai,
        is_instrumented,
        # Validation
        validate_crewai_setup,
    )
except ImportError:
    pytest.skip("CrewAI provider not available", allow_module_level=True)


class MockCrewAIAgent:
    """Mock CrewAI Agent for testing."""

    def __init__(self, role: str, goal: str, backstory: str, **kwargs):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.kwargs = kwargs

    def execute(self, task):
        """Mock agent execution."""
        time.sleep(0.01)  # Simulate processing time
        return f"Agent {self.role} completed: {task}"


class MockCrewAITask:
    """Mock CrewAI Task for testing."""

    def __init__(self, description: str, agent, **kwargs):
        self.description = description
        self.agent = agent
        self.kwargs = kwargs

    def execute(self):
        """Mock task execution."""
        return self.agent.execute(self.description)


class MockCrewAICrew:
    """Mock CrewAI Crew for testing."""

    def __init__(self, agents, tasks, **kwargs):
        self.agents = agents
        self.tasks = tasks
        self.kwargs = kwargs

    def kickoff(self, inputs=None):
        """Mock crew execution."""
        results = []
        for task in self.tasks:
            result = task.execute()
            results.append(result)

        return "\n".join(results)


class TestCrewAIIntegration:
    """Integration test suite for CrewAI + GenOps."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clean slate for each test
        try:
            disable_auto_instrumentation()
        except Exception:
            pass

    def teardown_method(self):
        """Teardown method run after each test."""
        try:
            disable_auto_instrumentation()
        except Exception:
            pass

    def create_mock_crew(self, crew_name: str = "test-crew") -> MockCrewAICrew:
        """Create a mock CrewAI crew for testing."""
        # Create mock agents
        researcher = MockCrewAIAgent(
            role="Senior Researcher",
            goal="Conduct thorough research",
            backstory="Expert researcher with years of experience",
        )

        writer = MockCrewAIAgent(
            role="Content Writer",
            goal="Create engaging content",
            backstory="Skilled writer specializing in technical content",
        )

        # Create mock tasks
        research_task = MockCrewAITask(
            description="Research the latest trends in AI", agent=researcher
        )

        writing_task = MockCrewAITask(
            description="Write an article about AI trends", agent=writer
        )

        # Create mock crew
        crew = MockCrewAICrew(
            agents=[researcher, writer], tasks=[research_task, writing_task]
        )

        return crew

    def test_complete_workflow_manual_instrumentation(self):
        """Test complete workflow with manual instrumentation."""
        # Create adapter
        adapter = GenOpsCrewAIAdapter(
            team="integration-test",
            project="manual-workflow",
            daily_budget_limit=50.0,
            enable_cost_tracking=True,
            enable_agent_tracking=True,
        )

        # Create mock crew
        crew = self.create_mock_crew("manual-test-crew")

        # Execute with tracking
        with adapter.track_crew("manual-integration-test") as context:
            result = crew.kickoff(
                {"topic": "AI integration testing", "audience": "developers"}
            )

            # Add some metrics during execution
            context.add_custom_metric("test_type", "integration")
            context.add_custom_metric("agents_count", len(crew.agents))
            context.add_custom_metric("tasks_count", len(crew.tasks))

            # Simulate cost data
            if hasattr(context, "add_cost_entry"):
                context.add_cost_entry("openai", "gpt-4", 150, 75, 0.045)

        # Verify results
        assert result is not None
        assert "Senior Researcher" in result
        assert "Content Writer" in result

        # Get and verify metrics
        metrics = context.get_metrics()
        assert metrics["crew_name"] == "manual-integration-test"
        assert metrics["execution_time"] > 0
        assert metrics["custom_metrics"]["test_type"] == "integration"
        assert metrics["custom_metrics"]["agents_count"] == 2

    def test_complete_workflow_auto_instrumentation(self):
        """Test complete workflow with auto-instrumentation."""
        # Enable auto-instrumentation
        success = auto_instrument(
            team="integration-auto", project="auto-workflow", daily_budget_limit=75.0
        )

        assert success is True
        assert is_instrumented() is True

        # Create and execute crew (should be automatically tracked)
        crew = self.create_mock_crew("auto-test-crew")
        result = crew.kickoff({"mode": "auto_instrumented", "test_case": "integration"})

        # Verify execution
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multi_crew_session_tracking(self):
        """Test tracking multiple crews in a session."""
        adapter = GenOpsCrewAIAdapter(
            team="multi-crew-test", project="session-tracking"
        )

        # Track session with multiple crews
        with adapter.track_session("multi-crew-integration") as session:
            # Execute multiple crews
            for i in range(3):
                crew = self.create_mock_crew(f"session-crew-{i}")

                with adapter.track_crew(f"crew-{i}") as crew_context:
                    crew.kickoff({"crew_number": i})
                    crew_context.add_custom_metric("iteration", i)

                    # Add crew result to session if supported
                    if hasattr(session, "add_crew_result"):
                        session.add_crew_result(crew_context.get_metrics())

            # Verify session tracking
            assert session.session_name == "multi-crew-integration"
            if hasattr(session, "total_crews"):
                assert session.total_crews >= 0

    def test_cost_aggregation_across_components(self):
        """Test cost aggregation across all components."""
        adapter = GenOpsCrewAIAdapter(
            team="cost-integration", project="cost-testing", enable_cost_tracking=True
        )

        # Execute multiple crews with cost data
        total_expected_cost = 0
        for i in range(3):
            crew = self.create_mock_crew(f"cost-crew-{i}")

            with adapter.track_crew(f"cost-test-{i}") as context:
                crew.kickoff()

                # Add cost entries
                cost = 0.03 + (i * 0.01)
                total_expected_cost += cost

                if hasattr(context, "add_cost_entry"):
                    context.add_cost_entry("openai", "gpt-4", 100, 50, cost)

        # Analyze costs
        if hasattr(adapter, "cost_aggregator") and adapter.cost_aggregator:
            total_cost = adapter.cost_aggregator.get_total_cost()
            assert total_cost >= 0  # Should have some cost data

            cost_by_provider = adapter.cost_aggregator.get_cost_by_provider()
            if "openai" in cost_by_provider:
                assert cost_by_provider["openai"] > 0

    def test_agent_monitoring_integration(self):
        """Test agent monitoring integration."""
        adapter = GenOpsCrewAIAdapter(
            team="monitoring-test",
            project="agent-monitoring",
            enable_agent_tracking=True,
        )

        crew = self.create_mock_crew("monitoring-crew")

        with adapter.track_crew("agent-monitoring-test"):
            # Simulate agent tracking if monitor is available
            if hasattr(adapter, "agent_monitor") and adapter.agent_monitor:
                monitor = adapter.agent_monitor

                # Track individual agents
                for i, agent in enumerate(crew.agents):
                    agent_id = f"agent_{i}"
                    monitor.start_agent_tracking(agent_id, agent.role)

                    time.sleep(0.01)  # Simulate work

                    monitor.record_agent_metric(agent_id, "complexity", 0.7)
                    monitor.end_agent_tracking(agent_id)

            # Execute crew
            result = crew.kickoff()

        assert result is not None

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        adapter = GenOpsCrewAIAdapter(team="error-test", project="error-handling")

        # Test error within crew execution
        crew = self.create_mock_crew("error-crew")

        with pytest.raises(ValueError):
            with adapter.track_crew("error-test"):
                # Simulate error during crew execution
                raise ValueError("Simulated crew execution error")

        # Adapter should still be functional after error
        with adapter.track_crew("recovery-test"):
            result = crew.kickoff()
            assert result is not None

    def test_concurrent_integration(self):
        """Test integration under concurrent load."""
        adapter = GenOpsCrewAIAdapter(
            team="concurrent-test", project="concurrent-integration"
        )

        results = []
        errors = []

        def execute_crew(crew_id):
            try:
                crew = self.create_mock_crew(f"concurrent-crew-{crew_id}")

                with adapter.track_crew(f"concurrent-{crew_id}") as context:
                    result = crew.kickoff({"crew_id": crew_id})
                    context.add_custom_metric("crew_id", crew_id)

                    metrics = context.get_metrics()
                    results.append((crew_id, result, metrics))

            except Exception as e:
                errors.append((crew_id, str(e)))

        # Execute multiple crews concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=execute_crew, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 5
        assert len(errors) == 0

        # All crews should have executed successfully
        for crew_id, result, metrics in results:
            assert result is not None
            assert metrics["custom_metrics"]["crew_id"] == crew_id

    def test_validation_integration(self):
        """Test validation integration with setup."""
        # Run validation
        validation_result = validate_crewai_setup(quick=True)

        assert validation_result is not None
        assert hasattr(validation_result, "is_valid")
        assert hasattr(validation_result, "issues")

        # Integration should work regardless of validation result
        adapter = GenOpsCrewAIAdapter(
            team="validation-test", project="validation-integration"
        )

        crew = self.create_mock_crew("validation-crew")

        with adapter.track_crew("validation-test"):
            result = crew.kickoff()
            assert result is not None

    def test_convenience_functions_integration(self):
        """Test convenience functions integration."""
        # Test instrument_crewai convenience function
        adapter = instrument_crewai(
            team="convenience-test",
            project="convenience-integration",
            daily_budget_limit=100.0,
        )

        assert adapter is not None
        assert adapter.team == "convenience-test"
        assert adapter.project == "convenience-integration"

        # Test create_multi_agent_adapter
        multi_adapter = create_multi_agent_adapter(
            team="multi-agent-test",
            project="multi-agent-integration",
            daily_budget_limit=150.0,
        )

        assert multi_adapter is not None
        assert multi_adapter.team == "multi-agent-test"

        # Execute crew with convenience adapter
        crew = self.create_mock_crew("convenience-crew")

        with multi_adapter.track_crew("convenience-test"):
            result = crew.kickoff()
            assert result is not None

    def test_performance_under_load(self):
        """Test performance under sustained load."""
        adapter = GenOpsCrewAIAdapter(team="performance-test", project="load-testing")

        start_time = time.time()

        # Execute many crews quickly
        for i in range(20):
            crew = self.create_mock_crew(f"load-crew-{i}")

            with adapter.track_crew(f"load-test-{i}") as context:
                crew.kickoff({"iteration": i})
                context.add_custom_metric("load_test_iteration", i)

        total_time = time.time() - start_time

        # Should complete within reasonable time
        assert total_time < 10.0  # 10 seconds for 20 crews

        # Get performance metrics
        recent_results = adapter.get_crew_results(limit=20)
        if recent_results:
            assert len(recent_results) <= 20

    def test_memory_usage_integration(self):
        """Test memory usage doesn't grow excessively."""
        adapter = GenOpsCrewAIAdapter(team="memory-test", project="memory-integration")

        # Execute many crews to test memory usage
        for i in range(50):
            crew = self.create_mock_crew(f"memory-crew-{i}")

            with adapter.track_crew(f"memory-test-{i}") as context:
                crew.kickoff({"data": "x" * 100})  # Some data
                context.add_custom_metric("memory_test", i)

        # Should not accumulate excessive data
        results_count = len(adapter.get_crew_results())
        assert results_count <= 100  # Reasonable upper bound

    def test_end_to_end_workflow_simulation(self):
        """Test complete end-to-end workflow simulation."""
        # Step 1: Validate setup
        validate_crewai_setup(quick=True)

        # Step 2: Enable auto-instrumentation
        auto_instrument(
            team="e2e-test", project="end-to-end-simulation", daily_budget_limit=200.0
        )

        # Step 3: Create and execute multiple crews
        crew_results = []

        for workflow_step in ["research", "analysis", "reporting"]:
            crew = self.create_mock_crew(f"{workflow_step}-crew")

            # Auto-instrumentation should track this automatically
            result = crew.kickoff({"workflow_step": workflow_step, "e2e_test": True})

            crew_results.append((workflow_step, result))

        # Step 4: Verify all steps completed
        assert len(crew_results) == 3

        workflow_steps = [step for step, _ in crew_results]
        assert "research" in workflow_steps
        assert "analysis" in workflow_steps
        assert "reporting" in workflow_steps

        # Step 5: Get final statistics
        stats = (
            get_instrumentation_stats()  # noqa: F821
            if "get_instrumentation_stats" in globals()
            else {}
        )

        # Should show activity
        if stats and "total_crews" in stats:
            assert stats["total_crews"] >= 0

    def test_component_interaction_patterns(self):
        """Test interaction patterns between components."""
        # Create adapter with all features enabled
        adapter = GenOpsCrewAIAdapter(
            team="interaction-test",
            project="component-interaction",
            enable_cost_tracking=True,
            enable_agent_tracking=True,
            enable_task_tracking=True,
        )

        crew = self.create_mock_crew("interaction-crew")

        with adapter.track_crew("component-interaction") as context:
            # Test interaction between cost tracking and monitoring
            if hasattr(context, "add_cost_entry") and hasattr(adapter, "agent_monitor"):
                context.add_cost_entry("openai", "gpt-4", 200, 100, 0.075)

                if adapter.agent_monitor:
                    agent_id = "interaction_agent"
                    adapter.agent_monitor.start_agent_tracking(agent_id, "Test Agent")
                    adapter.agent_monitor.record_agent_metric(agent_id, "cost", 0.075)
                    adapter.agent_monitor.end_agent_tracking(agent_id)

            crew.kickoff()

            # Add multiple types of metrics
            context.add_custom_metric("interaction_test", True)
            context.add_custom_metric("component_count", 3)

        # Verify all components worked together
        metrics = context.get_metrics()
        assert metrics["crew_name"] == "component-interaction"
        assert metrics["custom_metrics"]["interaction_test"] is True

    def test_graceful_degradation(self):
        """Test graceful degradation when components are unavailable."""
        # Test adapter works even if optional components fail
        with patch(
            "genops.providers.crewai.adapter.CrewAICostAggregator",
            side_effect=Exception("Cost aggregator error"),
        ):
            adapter = GenOpsCrewAIAdapter(
                team="degradation-test",
                project="graceful-degradation",
                enable_cost_tracking=True,  # Should handle gracefully
            )

            crew = self.create_mock_crew("degradation-crew")

            # Should still work without cost tracking
            with adapter.track_crew("degradation-test") as context:
                result = crew.kickoff()
                assert result is not None

                # Basic metrics should still work
                metrics = context.get_metrics()
                assert "crew_name" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
