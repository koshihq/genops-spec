#!/usr/bin/env python3
"""
Comprehensive test suite for Haystack monitor functionality.

Tests cover component monitoring, pipeline execution tracking, RAG/agent workflow
analysis, and performance metrics as required by CLAUDE.md standards.
"""

import pytest
import time
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List

from genops.providers.haystack_monitor import (
    HaystackMonitor,
    ComponentExecutionMetrics,
    PipelineExecutionMetrics,
    RAGWorkflowMetrics,
    AgentWorkflowMetrics
)


class TestComponentExecutionMetrics:
    """Component execution metrics data structure tests."""

    def test_component_metrics_creation(self):
        """Test component execution metrics creation."""
        metrics = ComponentExecutionMetrics(
            component_name="test-generator",
            component_type="Generator",
            execution_time_seconds=2.5,
            memory_usage_mb=45.2,
            cpu_usage_percent=12.5,
            success=True
        )
        
        assert metrics.component_name == "test-generator"
        assert metrics.component_type == "Generator"
        assert metrics.execution_time_seconds == 2.5
        assert metrics.memory_usage_mb == 45.2
        assert metrics.cpu_usage_percent == 12.5
        assert metrics.success is True

    def test_component_metrics_with_optional_fields(self):
        """Test component metrics with optional fields."""
        metrics = ComponentExecutionMetrics(
            component_name="test-retriever",
            component_type="Retriever",
            execution_time_seconds=1.0,
            memory_usage_mb=20.0,
            cpu_usage_percent=8.0,
            success=True,
            tokens_processed=150,
            documents_processed=5,
            error_message=None
        )
        
        assert metrics.tokens_processed == 150
        assert metrics.documents_processed == 5
        assert metrics.error_message is None

    def test_component_metrics_with_error(self):
        """Test component metrics with error information."""
        metrics = ComponentExecutionMetrics(
            component_name="failing-component",
            component_type="Generator",
            execution_time_seconds=0.1,
            memory_usage_mb=10.0,
            cpu_usage_percent=5.0,
            success=False,
            error_message="Rate limit exceeded"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Rate limit exceeded"


class TestPipelineExecutionMetrics:
    """Pipeline execution metrics data structure tests."""

    def test_pipeline_metrics_creation(self):
        """Test pipeline execution metrics creation."""
        metrics = PipelineExecutionMetrics(
            pipeline_id="pipeline-123",
            pipeline_name="test-pipeline",
            total_execution_time_seconds=5.2,
            component_count=3,
            success=True,
            cost_breakdown={"OpenAI": Decimal("0.01"), "Anthropic": Decimal("0.005")}
        )
        
        assert metrics.pipeline_id == "pipeline-123"
        assert metrics.pipeline_name == "test-pipeline"
        assert metrics.total_execution_time_seconds == 5.2
        assert metrics.component_count == 3
        assert metrics.success is True
        assert len(metrics.cost_breakdown) == 2

    def test_pipeline_metrics_component_metrics(self):
        """Test pipeline metrics with component metrics."""
        component1 = ComponentExecutionMetrics(
            component_name="gen1",
            component_type="Generator",
            execution_time_seconds=2.0,
            memory_usage_mb=30.0,
            cpu_usage_percent=10.0,
            success=True
        )
        
        component2 = ComponentExecutionMetrics(
            component_name="ret1",
            component_type="Retriever",
            execution_time_seconds=1.0,
            memory_usage_mb=15.0,
            cpu_usage_percent=5.0,
            success=True
        )
        
        metrics = PipelineExecutionMetrics(
            pipeline_id="pipeline-123",
            pipeline_name="test-pipeline",
            total_execution_time_seconds=3.0,
            component_count=2,
            success=True,
            cost_breakdown={},
            component_metrics=[component1, component2]
        )
        
        assert len(metrics.component_metrics) == 2
        assert metrics.component_metrics[0].component_name == "gen1"
        assert metrics.component_metrics[1].component_name == "ret1"


class TestRAGWorkflowMetrics:
    """RAG workflow metrics data structure tests."""

    def test_rag_metrics_creation(self):
        """Test RAG workflow metrics creation."""
        metrics = RAGWorkflowMetrics(
            retrieval_latency_seconds=1.5,
            generation_latency_seconds=3.0,
            documents_retrieved=5,
            retrieval_success_rate=1.0,
            generation_success_rate=1.0,
            end_to_end_latency_seconds=4.5
        )
        
        assert metrics.retrieval_latency_seconds == 1.5
        assert metrics.generation_latency_seconds == 3.0
        assert metrics.documents_retrieved == 5
        assert metrics.retrieval_success_rate == 1.0
        assert metrics.generation_success_rate == 1.0
        assert metrics.end_to_end_latency_seconds == 4.5

    def test_rag_metrics_with_embedding_data(self):
        """Test RAG metrics with embedding information."""
        embedding_metrics = ComponentExecutionMetrics(
            component_name="embedder",
            component_type="Embedder", 
            execution_time_seconds=0.5,
            memory_usage_mb=25.0,
            cpu_usage_percent=15.0,
            success=True
        )
        
        metrics = RAGWorkflowMetrics(
            retrieval_latency_seconds=1.0,
            generation_latency_seconds=2.0,
            documents_retrieved=3,
            retrieval_success_rate=1.0,
            generation_success_rate=1.0,
            end_to_end_latency_seconds=3.0,
            embedding_metrics=[embedding_metrics]
        )
        
        assert len(metrics.embedding_metrics) == 1
        assert metrics.embedding_metrics[0].component_name == "embedder"


class TestAgentWorkflowMetrics:
    """Agent workflow metrics data structure tests."""

    def test_agent_metrics_creation(self):
        """Test agent workflow metrics creation."""
        metrics = AgentWorkflowMetrics(
            decisions_made=3,
            tools_used=["search", "calculator"],
            tool_usage_count=5,
            tool_success_rate=0.8,
            decision_latency_seconds=2.5,
            total_iterations=4
        )
        
        assert metrics.decisions_made == 3
        assert metrics.tools_used == ["search", "calculator"]
        assert metrics.tool_usage_count == 5
        assert metrics.tool_success_rate == 0.8
        assert metrics.decision_latency_seconds == 2.5
        assert metrics.total_iterations == 4

    def test_agent_metrics_with_cost_breakdown(self):
        """Test agent metrics with cost breakdown by tool."""
        metrics = AgentWorkflowMetrics(
            decisions_made=2,
            tools_used=["search", "summarize"],
            tool_usage_count=4,
            tool_success_rate=1.0,
            decision_latency_seconds=1.8,
            total_iterations=2,
            cost_by_tool={"search": Decimal("0.005"), "summarize": Decimal("0.010")}
        )
        
        assert len(metrics.cost_by_tool) == 2
        assert metrics.cost_by_tool["search"] == Decimal("0.005")
        assert metrics.cost_by_tool["summarize"] == Decimal("0.010")


class TestHaystackMonitor:
    """Core Haystack monitor functionality tests."""

    def test_monitor_initialization(self):
        """Test monitor initializes properly."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        assert monitor.team == "test-team"
        assert monitor.project == "test-project"
        assert monitor.pipeline_executions == {}
        assert monitor.component_metrics == {}

    def test_monitor_start_pipeline_execution(self):
        """Test monitor starts pipeline execution tracking."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("test-pipeline")
        
        assert pipeline_id is not None
        assert isinstance(pipeline_id, str)
        assert len(pipeline_id) > 0
        assert pipeline_id in monitor.pipeline_executions

    def test_monitor_track_component_execution(self):
        """Test monitor tracks component execution."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("test-pipeline")
        
        # Track component execution
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="test-generator",
            component_type="Generator",
            execution_time_seconds=2.0,
            memory_usage_mb=30.0,
            cpu_usage_percent=10.0,
            success=True
        )
        
        assert pipeline_id in monitor.component_metrics
        assert "test-generator" in monitor.component_metrics[pipeline_id]
        
        component_metrics = monitor.component_metrics[pipeline_id]["test-generator"]
        assert component_metrics.execution_time_seconds == 2.0
        assert component_metrics.success is True

    def test_monitor_finish_pipeline_execution(self):
        """Test monitor finishes pipeline execution."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("test-pipeline")
        
        # Add some component metrics
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="comp1",
            component_type="Generator",
            execution_time_seconds=1.0,
            memory_usage_mb=20.0,
            cpu_usage_percent=8.0,
            success=True
        )
        
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="comp2",
            component_type="Retriever",
            execution_time_seconds=0.5,
            memory_usage_mb=10.0,
            cpu_usage_percent=4.0,
            success=True
        )
        
        # Finish execution
        metrics = monitor.finish_pipeline_execution(
            pipeline_id=pipeline_id,
            success=True,
            cost_breakdown={"OpenAI": Decimal("0.01")}
        )
        
        assert isinstance(metrics, PipelineExecutionMetrics)
        assert metrics.pipeline_id == pipeline_id
        assert metrics.component_count == 2
        assert metrics.success is True
        assert len(metrics.component_metrics) == 2

    def test_monitor_get_execution_metrics(self):
        """Test monitor retrieves execution metrics."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("test-pipeline")
        
        # Add metrics and finish
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="test-comp",
            component_type="Generator",
            execution_time_seconds=1.5,
            memory_usage_mb=25.0,
            cpu_usage_percent=12.0,
            success=True
        )
        
        monitor.finish_pipeline_execution(pipeline_id, success=True)
        
        # Retrieve metrics
        metrics = monitor.get_execution_metrics(pipeline_id)
        
        assert metrics is not None
        assert metrics.pipeline_id == pipeline_id
        assert metrics.component_count == 1

    def test_monitor_get_execution_metrics_nonexistent(self):
        """Test monitor handles nonexistent pipeline gracefully."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        metrics = monitor.get_execution_metrics("nonexistent-pipeline")
        assert metrics is None


class TestRAGWorkflowAnalysis:
    """RAG workflow analysis tests."""

    def test_monitor_analyze_rag_workflow(self):
        """Test monitor analyzes RAG workflow."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        # Create pipeline with RAG components
        pipeline_id = monitor.start_pipeline_execution("rag-pipeline")
        
        # Track retriever
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="document-retriever",
            component_type="Retriever",
            execution_time_seconds=1.0,
            memory_usage_mb=20.0,
            cpu_usage_percent=8.0,
            success=True,
            documents_processed=5
        )
        
        # Track embedder
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="text-embedder",
            component_type="Embedder", 
            execution_time_seconds=0.5,
            memory_usage_mb=15.0,
            cpu_usage_percent=6.0,
            success=True
        )
        
        # Track generator
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="text-generator",
            component_type="Generator",
            execution_time_seconds=2.0,
            memory_usage_mb=35.0,
            cpu_usage_percent=15.0,
            success=True
        )
        
        pipeline_metrics = monitor.finish_pipeline_execution(pipeline_id, success=True)
        
        # Analyze RAG workflow
        rag_metrics = monitor.analyze_rag_workflow(pipeline_metrics)
        
        assert isinstance(rag_metrics, RAGWorkflowMetrics)
        assert rag_metrics.retrieval_latency_seconds == 1.0
        assert rag_metrics.generation_latency_seconds == 2.0
        assert rag_metrics.documents_retrieved == 5
        assert rag_metrics.retrieval_success_rate == 1.0
        assert rag_metrics.generation_success_rate == 1.0

    def test_monitor_rag_workflow_with_failures(self):
        """Test RAG workflow analysis with component failures."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("rag-pipeline")
        
        # Track failing retriever
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="failing-retriever",
            component_type="Retriever",
            execution_time_seconds=0.1,
            memory_usage_mb=10.0,
            cpu_usage_percent=2.0,
            success=False,
            error_message="Connection timeout"
        )
        
        # Track successful generator
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="text-generator",
            component_type="Generator",
            execution_time_seconds=1.5,
            memory_usage_mb=30.0,
            cpu_usage_percent=12.0,
            success=True
        )
        
        pipeline_metrics = monitor.finish_pipeline_execution(pipeline_id, success=False)
        rag_metrics = monitor.analyze_rag_workflow(pipeline_metrics)
        
        assert rag_metrics.retrieval_success_rate == 0.0
        assert rag_metrics.generation_success_rate == 1.0


class TestAgentWorkflowAnalysis:
    """Agent workflow analysis tests."""

    def test_monitor_analyze_agent_workflow(self):
        """Test monitor analyzes agent workflow."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        # Create pipeline with agent components
        pipeline_id = monitor.start_pipeline_execution("agent-pipeline")
        
        # Track decision component
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="agent-decision",
            component_type="Agent",
            execution_time_seconds=1.5,
            memory_usage_mb=25.0,
            cpu_usage_percent=10.0,
            success=True
        )
        
        # Track tool usage components
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="search-tool",
            component_type="Tool",
            execution_time_seconds=2.0,
            memory_usage_mb=30.0,
            cpu_usage_percent=12.0,
            success=True
        )
        
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="calculator-tool",
            component_type="Tool",
            execution_time_seconds=0.5,
            memory_usage_mb=10.0,
            cpu_usage_percent=5.0,
            success=True
        )
        
        pipeline_metrics = monitor.finish_pipeline_execution(
            pipeline_id, 
            success=True,
            cost_breakdown={"OpenAI": Decimal("0.015")}
        )
        
        # Analyze agent workflow
        agent_metrics = monitor.analyze_agent_workflow(pipeline_metrics)
        
        assert isinstance(agent_metrics, AgentWorkflowMetrics)
        assert agent_metrics.decisions_made >= 1
        assert len(agent_metrics.tools_used) >= 2
        assert agent_metrics.tool_success_rate == 1.0

    def test_monitor_agent_workflow_with_tool_failures(self):
        """Test agent workflow analysis with tool failures."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("agent-pipeline")
        
        # Track successful tool
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="successful-tool",
            component_type="Tool",
            execution_time_seconds=1.0,
            memory_usage_mb=20.0,
            cpu_usage_percent=8.0,
            success=True
        )
        
        # Track failing tool
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="failing-tool",
            component_type="Tool",
            execution_time_seconds=0.2,
            memory_usage_mb=10.0,
            cpu_usage_percent=3.0,
            success=False,
            error_message="Tool execution failed"
        )
        
        pipeline_metrics = monitor.finish_pipeline_execution(pipeline_id, success=False)
        agent_metrics = monitor.analyze_agent_workflow(pipeline_metrics)
        
        assert agent_metrics.tool_success_rate == 0.5  # 1 success out of 2 tools


class TestPerformanceBenchmarking:
    """Performance benchmarking tests."""

    def test_monitor_component_performance_tracking(self):
        """Test monitor tracks component performance properly."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("perf-test")
        
        # Track component with detailed performance metrics
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="performance-component",
            component_type="Generator",
            execution_time_seconds=3.5,
            memory_usage_mb=128.5,
            cpu_usage_percent=45.2,
            success=True,
            tokens_processed=2000
        )
        
        metrics = monitor.finish_pipeline_execution(pipeline_id, success=True)
        
        component = metrics.component_metrics[0]
        assert component.execution_time_seconds == 3.5
        assert component.memory_usage_mb == 128.5
        assert component.cpu_usage_percent == 45.2
        assert component.tokens_processed == 2000

    def test_monitor_pipeline_performance_aggregation(self):
        """Test monitor aggregates pipeline performance correctly."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("multi-component")
        
        # Add multiple components
        for i in range(3):
            monitor.track_component_execution(
                pipeline_id=pipeline_id,
                component_name=f"component-{i}",
                component_type="Generator",
                execution_time_seconds=1.0,
                memory_usage_mb=50.0,
                cpu_usage_percent=20.0,
                success=True
            )
        
        metrics = monitor.finish_pipeline_execution(pipeline_id, success=True)
        
        assert metrics.component_count == 3
        assert len(metrics.component_metrics) == 3
        # Total execution time should be tracked separately from component times


class TestErrorHandling:
    """Error handling tests."""

    def test_monitor_handles_invalid_pipeline_id(self):
        """Test monitor handles invalid pipeline IDs gracefully."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        # Try to track component for non-existent pipeline
        monitor.track_component_execution(
            pipeline_id="non-existent",
            component_name="test-component",
            component_type="Generator",
            execution_time_seconds=1.0,
            memory_usage_mb=20.0,
            cpu_usage_percent=10.0,
            success=True
        )
        
        # Should not crash, and pipeline should be created
        assert "non-existent" in monitor.component_metrics

    def test_monitor_handles_duplicate_pipeline_finish(self):
        """Test monitor handles duplicate pipeline finish calls."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("test-pipeline")
        
        # Finish pipeline twice
        metrics1 = monitor.finish_pipeline_execution(pipeline_id, success=True)
        metrics2 = monitor.finish_pipeline_execution(pipeline_id, success=True)
        
        # Second call should return same metrics or handle gracefully
        assert metrics1 is not None
        assert metrics2 is not None

    def test_monitor_component_tracking_with_missing_data(self):
        """Test monitor handles missing component data gracefully."""
        monitor = HaystackMonitor(team="test-team", project="test-project")
        
        pipeline_id = monitor.start_pipeline_execution("test-pipeline")
        
        # Track component with minimal data
        monitor.track_component_execution(
            pipeline_id=pipeline_id,
            component_name="minimal-component",
            component_type="Unknown",
            execution_time_seconds=1.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            success=True
        )
        
        metrics = monitor.finish_pipeline_execution(pipeline_id, success=True)
        
        assert metrics.component_count == 1
        component = metrics.component_metrics[0]
        assert component.component_name == "minimal-component"