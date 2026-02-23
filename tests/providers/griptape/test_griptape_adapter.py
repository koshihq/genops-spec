#!/usr/bin/env python3
"""
Test suite for GenOps Griptape Adapter

Tests core adapter functionality including context managers, governance tracking,
cost attribution, and integration with Griptape structures.
"""

import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from genops.providers.griptape.adapter import (
    STRUCTURE_AGENT,
    STRUCTURE_ENGINE,
    STRUCTURE_MEMORY,
    STRUCTURE_PIPELINE,
    STRUCTURE_WORKFLOW,
    GenOpsGriptapeAdapter,
    GriptapeRequest,
)


class TestGriptapeRequest:
    """Test GriptapeRequest data class functionality."""

    def test_request_initialization(self):
        """Test basic request initialization."""
        request = GriptapeRequest(
            request_id="test-123",
            structure_type="agent",
            structure_id="test-agent",
            operation_type="run",
            start_time=time.time(),
        )

        assert request.request_id == "test-123"
        assert request.structure_type == "agent"
        assert request.structure_id == "test-agent"
        assert request.operation_type == "run"
        assert request.status == "running"
        assert request.total_cost == Decimal("0")
        assert len(request.providers_used) == 0
        assert len(request.models_used) == 0

    def test_request_finalization(self):
        """Test request finalization with metrics."""
        start_time = time.time()
        request = GriptapeRequest(
            request_id="test-123",
            structure_type="agent",
            structure_id="test-agent",
            operation_type="run",
            start_time=start_time,
        )

        # Simulate some activity
        request.task_count = 3
        request.completed_tasks = 2
        request.failed_tasks = 1

        request.finalize()

        assert request.end_time is not None
        assert request.duration is not None
        assert request.duration > 0
        assert request.status == "partial_failure"  # Some tasks failed

    def test_add_provider_cost(self):
        """Test adding provider costs and tracking."""
        request = GriptapeRequest(
            request_id="test-123",
            structure_type="agent",
            structure_id="test-agent",
            operation_type="run",
            start_time=time.time(),
        )

        # Add OpenAI cost
        request.add_provider_cost("openai", "gpt-4", 0.002)
        assert request.total_cost == Decimal("0.002")
        assert "openai" in request.providers_used
        assert "gpt-4" in request.models_used
        assert request.provider_costs["openai"] == Decimal("0.002")

        # Add Anthropic cost
        request.add_provider_cost("anthropic", "claude-3", 0.003)
        assert request.total_cost == Decimal("0.005")
        assert "anthropic" in request.providers_used
        assert "claude-3" in request.models_used

    def test_task_completion_tracking(self):
        """Test task completion status tracking."""
        request = GriptapeRequest(
            request_id="test-123",
            structure_type="pipeline",
            structure_id="test-pipeline",
            operation_type="run",
            start_time=time.time(),
        )

        # Add successful tasks
        request.add_task_completion(success=True)
        request.add_task_completion(success=True)
        assert request.completed_tasks == 2
        assert request.failed_tasks == 0

        # Add failed task
        request.add_task_completion(success=False)
        assert request.completed_tasks == 2
        assert request.failed_tasks == 1


class TestGenOpsGriptapeAdapter:
    """Test GenOpsGriptapeAdapter core functionality."""

    @pytest.fixture
    def adapter(self):
        """Create test adapter instance."""
        return GenOpsGriptapeAdapter(
            team="test-team",
            project="test-project",
            environment="test",
            enable_cost_tracking=True,
            enable_performance_monitoring=True,
        )

    def test_adapter_initialization(self, adapter):
        """Test adapter initialization with governance attributes."""
        assert adapter.governance_attrs.team == "test-team"
        assert adapter.governance_attrs.project == "test-project"
        assert adapter.governance_attrs.environment == "test"
        assert adapter.enable_cost_tracking is True
        assert adapter.enable_performance_monitoring is True

    def test_adapter_with_budget_limit(self):
        """Test adapter initialization with budget constraints."""
        adapter = GenOpsGriptapeAdapter(team="budget-team", daily_budget_limit=50.0)

        assert adapter.daily_budget_limit == 50.0

        # Test budget compliance check
        budget_status = adapter.check_budget_compliance()
        assert budget_status["status"] in ["within_budget", "over_budget"]
        assert "spending" in budget_status
        assert "limit" in budget_status

    @patch("genops.providers.griptape.adapter.TelemetryExporter")
    def test_track_agent_context_manager(self, mock_exporter, adapter):
        """Test agent tracking context manager."""
        mock_exporter_instance = Mock()
        mock_exporter.return_value = mock_exporter_instance

        with adapter.track_agent("test-agent") as request:
            assert isinstance(request, GriptapeRequest)
            assert request.structure_type == STRUCTURE_AGENT
            assert request.structure_id == "test-agent"
            assert request.status == "running"

            # Simulate adding cost
            request.add_provider_cost("openai", "gpt-4", 0.002)

        # Check request finalized
        assert request.status == "completed"
        assert request.end_time is not None
        assert request.duration is not None

        # Check telemetry was exported
        mock_exporter_instance.export_span.assert_called_once()

    def test_track_agent_with_error(self, adapter):
        """Test agent tracking with exception handling."""
        with pytest.raises(ValueError, match="test error"):
            with adapter.track_agent("failing-agent") as request:
                assert request.status == "running"
                raise ValueError("test error")

        # Request should be marked as failed
        assert request.status == "failed"
        assert request.error_message == "test error"

    @patch("genops.providers.griptape.adapter.TelemetryExporter")
    def test_track_pipeline_context_manager(self, mock_exporter, adapter):
        """Test pipeline tracking context manager."""
        mock_exporter_instance = Mock()
        mock_exporter.return_value = mock_exporter_instance

        with adapter.track_pipeline("test-pipeline") as request:
            assert request.structure_type == STRUCTURE_PIPELINE
            assert request.structure_id == "test-pipeline"

            # Simulate pipeline execution
            request.task_count = 3
            request.add_task_completion(success=True)
            request.add_task_completion(success=True)
            request.add_task_completion(success=False)

        assert request.status == "partial_failure"
        assert request.completed_tasks == 2
        assert request.failed_tasks == 1

    @patch("genops.providers.griptape.adapter.TelemetryExporter")
    def test_track_workflow_context_manager(self, mock_exporter, adapter):
        """Test workflow tracking context manager."""
        mock_exporter_instance = Mock()
        mock_exporter.return_value = mock_exporter_instance

        with adapter.track_workflow("test-workflow") as request:
            assert request.structure_type == STRUCTURE_WORKFLOW
            assert request.structure_id == "test-workflow"

            # Simulate parallel workflow
            request.task_count = 4
            for _ in range(4):
                request.add_task_completion(success=True)

        assert request.status == "completed"
        assert request.completed_tasks == 4
        assert request.failed_tasks == 0

    def test_track_engine_context_manager(self, adapter):
        """Test engine tracking context manager."""
        with adapter.track_engine("test-rag", "rag") as request:
            assert request.structure_type == STRUCTURE_ENGINE
            assert request.structure_id == "test-rag"
            assert request.operation_type == "rag"

            # Simulate engine operations
            request.reasoning_steps = 3
            request.memory_operations = 1

    def test_track_memory_context_manager(self, adapter):
        """Test memory operation tracking."""
        with adapter.track_memory("conversation-mem", "retrieve") as request:
            assert request.structure_type == STRUCTURE_MEMORY
            assert request.structure_id == "conversation-mem"
            assert request.operation_type == "retrieve"

    @patch.object(GenOpsGriptapeAdapter, "cost_aggregator")
    def test_daily_spending_calculation(self, mock_cost_aggregator, adapter):
        """Test daily spending calculation."""
        mock_cost_aggregator.get_daily_costs.return_value = Decimal("10.50")

        daily_spending = adapter.get_daily_spending()
        assert daily_spending == Decimal("10.50")
        mock_cost_aggregator.get_daily_costs.assert_called_once()

    def test_disabled_cost_tracking(self):
        """Test adapter behavior with cost tracking disabled."""
        adapter = GenOpsGriptapeAdapter(team="test-team", enable_cost_tracking=False)

        daily_spending = adapter.get_daily_spending()
        assert daily_spending == Decimal("0")

    def test_disabled_performance_monitoring(self):
        """Test adapter behavior with performance monitoring disabled."""
        adapter = GenOpsGriptapeAdapter(
            team="test-team", enable_performance_monitoring=False
        )

        # Context managers should still work but without monitoring
        with adapter.track_agent("test-agent") as request:
            assert request.structure_type == STRUCTURE_AGENT

    @patch("genops.providers.griptape.adapter.trace")
    def test_opentelemetry_integration(self, mock_trace, adapter):
        """Test OpenTelemetry tracer integration."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        with adapter.track_agent("otel-test"):
            pass

        # Verify tracer was used
        mock_trace.get_tracer.assert_called()
        mock_tracer.start_as_current_span.assert_called()

    def test_create_request_with_custom_attributes(self, adapter):
        """Test request creation with custom governance attributes."""
        request = adapter._create_request(
            structure_type="custom",
            structure_id="custom-structure",
            operation_type="custom_op",
        )

        assert request.structure_type == "custom"
        assert request.structure_id == "custom-structure"
        assert request.operation_type == "custom_op"
        assert request.governance_attrs["team"] == "test-team"
        assert request.governance_attrs["project"] == "test-project"

    def test_export_telemetry_attributes(self, adapter):
        """Test telemetry export with proper attributes."""
        with patch.object(adapter, "telemetry_exporter") as mock_exporter:
            request = GriptapeRequest(
                request_id="test-123",
                structure_type=STRUCTURE_AGENT,
                structure_id="test-agent",
                operation_type="run",
                start_time=time.time(),
            )
            request.add_provider_cost("openai", "gpt-4", 0.002)
            request.finalize()

            adapter._export_telemetry(request)

            # Verify export was called
            mock_exporter.export_span.assert_called_once()

            # Check attributes include governance data
            call_args = mock_exporter.export_span.call_args
            attributes = call_args[1]["attributes"]

            assert "genops.provider" in attributes
            assert attributes["genops.provider"] == "griptape"
            assert "genops.structure.type" in attributes
            assert "genops.cost.total" in attributes
            assert "team" in attributes  # Governance attribute


class TestGriptapeAdapterIntegration:
    """Integration tests for adapter with mocked Griptape components."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for integration tests."""
        return GenOpsGriptapeAdapter(
            team="integration-team", project="integration-test", daily_budget_limit=25.0
        )

    def test_multiple_structure_tracking(self, adapter):
        """Test tracking multiple structures simultaneously."""
        requests = []

        # Track multiple structures concurrently
        with adapter.track_agent("agent-1") as agent_req:
            with adapter.track_pipeline("pipeline-1") as pipeline_req:
                with adapter.track_workflow("workflow-1") as workflow_req:
                    # Simulate operations
                    agent_req.add_provider_cost("openai", "gpt-4", 0.001)
                    pipeline_req.add_provider_cost("anthropic", "claude-3", 0.002)
                    workflow_req.add_provider_cost("google", "gemini-pro", 0.001)

                    requests.extend([agent_req, pipeline_req, workflow_req])

        # All should be completed
        for req in requests:
            assert req.status == "completed"
            assert req.total_cost > 0

    def test_budget_enforcement(self):
        """Test budget limit enforcement."""
        adapter = GenOpsGriptapeAdapter(
            team="budget-test",
            daily_budget_limit=0.001,  # Very low limit
        )

        # Mock daily spending to exceed limit
        with patch.object(adapter, "get_daily_spending", return_value=Decimal("0.002")):
            budget_status = adapter.check_budget_compliance()
            assert budget_status["status"] == "over_budget"
            assert budget_status["utilization"] > 100

    def test_sampling_configuration(self):
        """Test sampling rate configuration."""
        adapter = GenOpsGriptapeAdapter(
            team="sampling-test",
            sampling_rate=0.5,  # 50% sampling
        )

        assert adapter.sampling_rate == 0.5

    @patch.object(GenOpsGriptapeAdapter, "workflow_monitor")
    def test_performance_monitoring_integration(self, mock_monitor, adapter):
        """Test integration with workflow monitor."""
        mock_metrics = Mock()
        mock_metrics.memory_operations = 5
        mock_metrics.tool_calls = 3
        mock_metrics.reasoning_steps = 7
        mock_monitor.stop_structure_monitoring.return_value = mock_metrics

        with adapter.track_agent("monitored-agent") as request:
            pass

        # Check monitoring was started and stopped
        mock_monitor.start_structure_monitoring.assert_called_once()
        mock_monitor.stop_structure_monitoring.assert_called_once()

        # Check metrics were applied to request
        assert request.memory_operations == 5
        assert request.tool_calls == 3
        assert request.reasoning_steps == 7

    def test_cost_aggregator_integration(self, adapter):
        """Test integration with cost aggregator."""
        # Mock cost aggregator
        with patch.object(adapter, "cost_aggregator") as mock_aggregator:
            mock_aggregator.get_daily_costs.return_value = Decimal("15.75")

            daily_spending = adapter.get_daily_spending()
            assert daily_spending == Decimal("15.75")

    def test_error_handling_and_recovery(self, adapter):
        """Test error handling and graceful recovery."""
        # Test telemetry export failure
        with patch.object(
            adapter, "_export_telemetry", side_effect=Exception("Export failed")
        ):
            with adapter.track_agent("error-test") as request:
                request.add_provider_cost("openai", "gpt-4", 0.001)

            # Request should still be finalized despite export failure
            assert request.status == "completed"
            assert request.total_cost > 0


if __name__ == "__main__":
    pytest.main([__file__])
