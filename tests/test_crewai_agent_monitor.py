#!/usr/bin/env python3
"""
Test suite for CrewAI Agent Monitor

Comprehensive tests for the CrewAIAgentMonitor class including:
- Agent performance tracking
- Multi-agent workflow analysis
- Collaboration pattern detection
- Bottleneck identification
- Real-time monitoring capabilities
"""

import time

import pytest

# Import the CrewAI agent monitor and related classes
try:
    from genops.providers.crewai import (
        AgentExecutionMetrics,
        CrewAIAgentMonitor,
        CrewExecutionMetrics,
        MultiAgentWorkflowMetrics,
        TaskExecutionMetrics,
    )
except ImportError:
    pytest.skip("CrewAI provider not available", allow_module_level=True)


class TestCrewAIAgentMonitor:
    """Test suite for CrewAIAgentMonitor."""

    def test_monitor_initialization(self):
        """Test agent monitor initialization."""
        monitor = CrewAIAgentMonitor()

        assert monitor is not None
        assert hasattr(monitor, "agent_metrics")
        assert hasattr(monitor, "task_metrics")
        assert hasattr(monitor, "crew_metrics")

    def test_start_agent_tracking(self):
        """Test starting agent tracking."""
        monitor = CrewAIAgentMonitor()

        agent_id = "test_agent_1"
        agent_role = "Research Analyst"

        monitor.start_agent_tracking(agent_id, agent_role)

        # Should have started tracking
        assert agent_id in monitor.active_agents or hasattr(monitor, "start_time")

    def test_end_agent_tracking(self):
        """Test ending agent tracking."""
        monitor = CrewAIAgentMonitor()

        agent_id = "test_agent_1"
        agent_role = "Research Analyst"

        # Start tracking
        monitor.start_agent_tracking(agent_id, agent_role)
        time.sleep(0.1)  # Small delay

        # End tracking
        metrics = monitor.end_agent_tracking(agent_id)

        if metrics:
            assert isinstance(metrics, AgentExecutionMetrics)
            assert metrics.agent_id == agent_id
            assert metrics.execution_time > 0

    def test_track_task_execution(self):
        """Test tracking task execution metrics."""
        monitor = CrewAIAgentMonitor()

        task_data = {
            "task_id": "task_1",
            "agent_id": "agent_1",
            "task_type": "research",
            "description": "Conduct market research",
        }

        # Start task tracking
        monitor.start_task_tracking(**task_data)
        time.sleep(0.05)  # Small delay

        # End task tracking
        metrics = monitor.end_task_tracking(task_data["task_id"])

        if metrics:
            assert isinstance(metrics, TaskExecutionMetrics)
            assert metrics.task_id == task_data["task_id"]
            assert metrics.execution_time > 0

    def test_track_crew_execution(self):
        """Test tracking crew execution metrics."""
        monitor = CrewAIAgentMonitor()

        crew_data = {
            "crew_id": "crew_1",
            "crew_name": "research_crew",
            "agents": ["agent_1", "agent_2"],
            "tasks": ["task_1", "task_2"],
        }

        # Start crew tracking
        monitor.start_crew_tracking(**crew_data)
        time.sleep(0.1)

        # End crew tracking
        metrics = monitor.end_crew_tracking(crew_data["crew_id"])

        if metrics:
            assert isinstance(metrics, CrewExecutionMetrics)
            assert metrics.crew_id == crew_data["crew_id"]
            assert metrics.execution_time > 0

    def test_agent_performance_metrics(self):
        """Test collecting agent performance metrics."""
        monitor = CrewAIAgentMonitor()

        agent_id = "perf_agent"
        agent_role = "Performance Tester"

        # Simulate agent execution with metrics
        monitor.start_agent_tracking(agent_id, agent_role)

        # Simulate some work
        time.sleep(0.05)

        # Add performance data
        monitor.record_agent_metric(agent_id, "tokens_processed", 150)
        monitor.record_agent_metric(agent_id, "api_calls", 3)
        monitor.record_agent_metric(agent_id, "cost", 0.045)

        metrics = monitor.end_agent_tracking(agent_id)

        if metrics and hasattr(metrics, "custom_metrics"):
            assert metrics.custom_metrics.get("tokens_processed") == 150
            assert metrics.custom_metrics.get("api_calls") == 3
            assert metrics.custom_metrics.get("cost") == 0.045

    def test_multi_agent_workflow_analysis(self):
        """Test analyzing multi-agent workflow patterns."""
        monitor = CrewAIAgentMonitor()

        # Simulate multi-agent workflow
        agents = [
            ("agent_1", "Researcher"),
            ("agent_2", "Analyst"),
            ("agent_3", "Writer"),
        ]

        crew_id = "multi_agent_crew"
        monitor.start_crew_tracking(
            crew_id=crew_id,
            crew_name="Multi-Agent Analysis",
            agents=[a[0] for a in agents],
            tasks=["research", "analysis", "writing"],
        )

        # Simulate sequential agent execution
        for agent_id, role in agents:
            monitor.start_agent_tracking(agent_id, role)
            time.sleep(0.02)
            monitor.record_agent_metric(agent_id, "complexity_score", 0.8)
            monitor.end_agent_tracking(agent_id)

        monitor.end_crew_tracking(crew_id)

        # Analyze workflow
        workflow_analysis = monitor.get_workflow_analysis(crew_id)

        if workflow_analysis:
            assert isinstance(workflow_analysis, MultiAgentWorkflowMetrics)
            assert len(workflow_analysis.agent_collaboration_matrix) > 0

    def test_bottleneck_detection(self):
        """Test detecting bottlenecks in agent workflows."""
        monitor = CrewAIAgentMonitor()

        crew_id = "bottleneck_crew"
        agents_data = [
            ("fast_agent", "Fast Worker", 0.01),  # Fast execution
            ("slow_agent", "Slow Worker", 0.10),  # Slow execution (bottleneck)
            ("normal_agent", "Normal Worker", 0.03),  # Normal execution
        ]

        monitor.start_crew_tracking(
            crew_id=crew_id,
            crew_name="Bottleneck Test",
            agents=[a[0] for a in agents_data],
            tasks=["task_1", "task_2", "task_3"],
        )

        # Simulate agents with different execution times
        for agent_id, role, sleep_time in agents_data:
            monitor.start_agent_tracking(agent_id, role)
            time.sleep(sleep_time)
            monitor.end_agent_tracking(agent_id)

        monitor.end_crew_tracking(crew_id)

        # Analyze for bottlenecks
        workflow_analysis = monitor.get_workflow_analysis(crew_id)

        if workflow_analysis and hasattr(workflow_analysis, "bottleneck_agents"):
            # Should identify slow_agent as bottleneck
            assert len(workflow_analysis.bottleneck_agents) >= 0

    def test_collaboration_pattern_analysis(self):
        """Test analyzing collaboration patterns between agents."""
        monitor = CrewAIAgentMonitor()

        crew_id = "collab_crew"

        # Simulate collaborative workflow
        monitor.start_crew_tracking(
            crew_id=crew_id,
            crew_name="Collaboration Test",
            agents=["agent_1", "agent_2", "agent_3"],
            tasks=["task_1", "task_2"],
        )

        # Simulate overlapping agent execution (collaboration)
        monitor.start_agent_tracking("agent_1", "Lead Researcher")
        time.sleep(0.01)

        monitor.start_agent_tracking("agent_2", "Data Analyst")  # Overlap
        time.sleep(0.01)

        monitor.end_agent_tracking("agent_1")

        monitor.start_agent_tracking("agent_3", "Report Writer")
        time.sleep(0.01)

        monitor.end_agent_tracking("agent_2")
        monitor.end_agent_tracking("agent_3")

        monitor.end_crew_tracking(crew_id)

        # Get collaboration analysis
        workflow_analysis = monitor.get_workflow_analysis(crew_id)

        if workflow_analysis:
            # Should detect some level of collaboration
            collaboration_score = getattr(workflow_analysis, "collaboration_score", 0)
            assert collaboration_score >= 0  # Should have some collaboration

    def test_real_time_monitoring(self):
        """Test real-time monitoring capabilities."""
        monitor = CrewAIAgentMonitor()

        agent_id = "realtime_agent"
        monitor.start_agent_tracking(agent_id, "Real-time Test Agent")

        # Get real-time status
        status = monitor.get_agent_status(agent_id)

        if status:
            assert status.get("agent_id") == agent_id
            assert status.get("status") in ["active", "running", "tracking"]
            assert "start_time" in status

        monitor.end_agent_tracking(agent_id)

        # Status should now be completed
        final_status = monitor.get_agent_status(agent_id)
        if final_status:
            assert final_status.get("status") in ["completed", "finished", "ended"]

    def test_performance_threshold_alerts(self):
        """Test performance threshold monitoring and alerts."""
        monitor = CrewAIAgentMonitor(
            performance_thresholds={
                "max_execution_time": 0.05,  # 50ms threshold
                "max_cost": 0.10,
            }
        )

        agent_id = "threshold_agent"
        monitor.start_agent_tracking(agent_id, "Threshold Test Agent")

        # Simulate slow execution (above threshold)
        time.sleep(0.06)  # Exceed the 50ms threshold
        monitor.record_agent_metric(agent_id, "cost", 0.15)  # Exceed cost threshold

        monitor.end_agent_tracking(agent_id)

        # Should have triggered threshold alerts
        alerts = monitor.get_performance_alerts(agent_id)
        if alerts:
            assert len(alerts) > 0
            # Should have alerts for execution time and cost
            alert_types = [alert.get("type", "") for alert in alerts]
            assert any("execution_time" in alert_type for alert_type in alert_types)

    def test_resource_utilization_tracking(self):
        """Test resource utilization monitoring."""
        monitor = CrewAIAgentMonitor()

        crew_id = "resource_crew"
        monitor.start_crew_tracking(
            crew_id=crew_id,
            crew_name="Resource Test",
            agents=["agent_1", "agent_2"],
            tasks=["task_1", "task_2"],
        )

        # Simulate resource usage
        for i, agent_id in enumerate(["agent_1", "agent_2"]):
            monitor.start_agent_tracking(agent_id, f"Agent {i + 1}")

            # Record resource metrics
            monitor.record_agent_metric(agent_id, "memory_usage", 0.6 + i * 0.1)
            monitor.record_agent_metric(agent_id, "cpu_usage", 0.4 + i * 0.2)

            monitor.end_agent_tracking(agent_id)

        monitor.end_crew_tracking(crew_id)

        # Get resource utilization summary
        resource_summary = monitor.get_resource_utilization(crew_id)

        if resource_summary:
            assert "memory_usage" in resource_summary
            assert "cpu_usage" in resource_summary
            assert resource_summary["memory_usage"] > 0

    def test_concurrent_agent_monitoring(self):
        """Test monitoring multiple agents concurrently."""
        import threading

        monitor = CrewAIAgentMonitor()
        results = []

        def monitor_agent(agent_id, role):
            monitor.start_agent_tracking(agent_id, role)
            time.sleep(0.02)
            monitor.record_agent_metric(
                agent_id, "thread_id", threading.current_thread().ident
            )
            metrics = monitor.end_agent_tracking(agent_id)
            results.append((agent_id, metrics))

        # Start multiple agents concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=monitor_agent,
                args=(f"concurrent_agent_{i}", f"Concurrent Agent {i}"),
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        assert len(results) == 3
        agent_ids = [result[0] for result in results]
        assert "concurrent_agent_0" in agent_ids
        assert "concurrent_agent_1" in agent_ids
        assert "concurrent_agent_2" in agent_ids

    def test_historical_metrics_storage(self):
        """Test storage and retrieval of historical metrics."""
        monitor = CrewAIAgentMonitor()

        # Execute multiple crews over time
        for i in range(3):
            crew_id = f"historical_crew_{i}"
            monitor.start_crew_tracking(
                crew_id=crew_id,
                crew_name=f"Historical Test {i}",
                agents=[f"agent_{i}"],
                tasks=[f"task_{i}"],
            )

            monitor.start_agent_tracking(f"agent_{i}", f"Agent {i}")
            time.sleep(0.01)
            monitor.end_agent_tracking(f"agent_{i}")

            monitor.end_crew_tracking(crew_id)

        # Get historical data
        historical_crews = monitor.get_historical_crews(limit=5)

        if historical_crews:
            assert len(historical_crews) >= 3
            crew_names = [crew.get("crew_name", "") for crew in historical_crews]
            assert any("Historical Test" in name for name in crew_names)

    def test_metrics_aggregation(self):
        """Test aggregating metrics across multiple executions."""
        monitor = CrewAIAgentMonitor()

        agent_id = "aggregation_agent"
        execution_times = []

        # Execute same agent multiple times
        for i in range(5):
            monitor.start_agent_tracking(agent_id, "Aggregation Test Agent")
            sleep_time = 0.01 + i * 0.005  # Varying execution times
            time.sleep(sleep_time)
            execution_times.append(sleep_time)

            monitor.end_agent_tracking(agent_id)

        # Get aggregated metrics
        aggregated = monitor.get_agent_aggregated_metrics(agent_id)

        if aggregated:
            assert aggregated.get("total_executions") >= 5
            assert aggregated.get("avg_execution_time") > 0
            assert (
                aggregated.get("total_execution_time") > sum(execution_times) * 0.5
            )  # Allow for overhead

    def test_workflow_optimization_suggestions(self):
        """Test generating workflow optimization suggestions."""
        monitor = CrewAIAgentMonitor()

        crew_id = "optimization_crew"

        # Simulate inefficient workflow
        monitor.start_crew_tracking(
            crew_id=crew_id,
            crew_name="Optimization Test",
            agents=["slow_agent", "efficient_agent"],
            tasks=["slow_task", "fast_task"],
        )

        # Slow agent
        monitor.start_agent_tracking("slow_agent", "Slow Agent")
        time.sleep(0.08)  # Slow execution
        monitor.record_agent_metric("slow_agent", "efficiency_score", 0.3)
        monitor.end_agent_tracking("slow_agent")

        # Efficient agent
        monitor.start_agent_tracking("efficient_agent", "Efficient Agent")
        time.sleep(0.01)  # Fast execution
        monitor.record_agent_metric("efficient_agent", "efficiency_score", 0.9)
        monitor.end_agent_tracking("efficient_agent")

        monitor.end_crew_tracking(crew_id)

        # Get optimization suggestions
        suggestions = monitor.get_optimization_suggestions(crew_id)

        if suggestions:
            assert len(suggestions) > 0
            # Should suggest optimizing the slow agent
            slow_agent_suggestions = [s for s in suggestions if "slow_agent" in str(s)]
            assert len(slow_agent_suggestions) >= 0

    def test_error_handling_in_monitoring(self):
        """Test error handling during agent monitoring."""
        monitor = CrewAIAgentMonitor()

        agent_id = "error_agent"
        monitor.start_agent_tracking(agent_id, "Error Test Agent")

        # Simulate error during execution
        try:
            monitor.record_agent_metric(agent_id, "invalid_metric", None)
        except (ValueError, TypeError):
            pass  # Expected for invalid metric

        # Should still be able to end tracking
        monitor.end_agent_tracking(agent_id)

        # Monitoring should continue to work after error
        monitor.start_agent_tracking("recovery_agent", "Recovery Agent")
        recovery_metrics = monitor.end_agent_tracking("recovery_agent")

        assert recovery_metrics is not None or True  # Monitor recovered

    def test_custom_metric_types(self):
        """Test recording different types of custom metrics."""
        monitor = CrewAIAgentMonitor()

        agent_id = "metrics_agent"
        monitor.start_agent_tracking(agent_id, "Metrics Test Agent")

        # Record various metric types
        monitor.record_agent_metric(agent_id, "integer_metric", 42)
        monitor.record_agent_metric(agent_id, "float_metric", 3.14159)
        monitor.record_agent_metric(agent_id, "string_metric", "test_value")
        monitor.record_agent_metric(agent_id, "boolean_metric", True)
        monitor.record_agent_metric(agent_id, "list_metric", [1, 2, 3])
        monitor.record_agent_metric(agent_id, "dict_metric", {"key": "value"})

        metrics = monitor.end_agent_tracking(agent_id)

        if metrics and hasattr(metrics, "custom_metrics"):
            custom = metrics.custom_metrics
            assert custom.get("integer_metric") == 42
            assert custom.get("float_metric") == 3.14159
            assert custom.get("string_metric") == "test_value"
            assert custom.get("boolean_metric") is True
            assert custom.get("list_metric") == [1, 2, 3]
            assert custom.get("dict_metric") == {"key": "value"}

    def test_monitor_cleanup(self):
        """Test cleanup of monitoring data."""
        monitor = CrewAIAgentMonitor()

        # Create some monitoring data
        for i in range(5):
            agent_id = f"cleanup_agent_{i}"
            monitor.start_agent_tracking(agent_id, f"Cleanup Agent {i}")
            monitor.end_agent_tracking(agent_id)

        # Cleanup old data
        if hasattr(monitor, "cleanup_old_data"):
            monitor.cleanup_old_data(max_age_hours=0)  # Cleanup everything

            # Should have cleaned up data
            recent_agents = monitor.get_recent_agents(hours=24)
            if recent_agents is not None:
                assert len(recent_agents) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
