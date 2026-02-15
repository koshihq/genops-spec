#!/usr/bin/env python3
"""
Griptape Workflow Monitor for GenOps Governance

Provides performance monitoring and analytics for Griptape AI framework structures,
including execution tracking, resource utilization, and optimization insights.

Usage:
    from genops.providers.griptape.workflow_monitor import GriptapeWorkflowMonitor

    monitor = GriptapeWorkflowMonitor(enable_performance_monitoring=True)

    # Start monitoring a structure
    monitor.start_structure_monitoring("agent-123", "agent")

    # Record operations during execution
    monitor.record_task_execution("task-1", duration=2.5, success=True)
    monitor.record_memory_access("conversation-memory", operation="read")
    monitor.record_tool_usage("web-search", duration=1.2)

    # Stop monitoring and get metrics
    metrics = monitor.stop_structure_monitoring("agent-123")
    print(f"Total duration: {metrics.total_duration:.3f}s")
    print(f"Tasks completed: {metrics.tasks_completed}")

Features:
    - Structure performance monitoring (Agent, Pipeline, Workflow)
    - Task execution tracking with success/failure rates
    - Memory operation analytics and optimization insights
    - Tool usage monitoring and performance profiling
    - Chain-of-thought reasoning step analysis
    - Resource utilization tracking and alerting
    - Performance bottleneck identification
    - Execution pattern analytics for optimization
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class GriptapeTaskMetrics:
    """Metrics for individual task execution."""

    task_id: str
    task_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    # Resource usage
    memory_usage: Optional[float] = None  # MB
    cpu_usage: Optional[float] = None  # Percentage

    # Task-specific metrics
    tokens_processed: int = 0
    tool_calls_made: int = 0
    memory_accesses: int = 0
    reasoning_steps: int = 0

    def finalize(self) -> None:
        """Finalize task metrics."""
        if self.end_time is None:
            self.end_time = time.time()

        if self.duration is None:
            self.duration = self.end_time - self.start_time


@dataclass
class GriptapeStructureMetrics:
    """Comprehensive metrics for Griptape structure execution."""

    # Structure identification
    structure_id: str
    structure_type: str  # agent, pipeline, workflow

    # Timing metrics
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None

    # Task metrics
    tasks_total: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    task_metrics: list[GriptapeTaskMetrics] = field(default_factory=list)

    # Performance metrics
    memory_operations: int = 0
    tool_calls: int = 0
    reasoning_steps: int = 0

    # Resource utilization
    peak_memory_usage: Optional[float] = None  # MB
    average_cpu_usage: Optional[float] = None  # Percentage

    # Execution patterns
    parallel_tasks: int = 0
    sequential_tasks: int = 0
    retry_count: int = 0

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def finalize(self) -> None:
        """Finalize structure metrics."""
        if self.end_time is None:
            self.end_time = time.time()

        if self.total_duration is None:
            self.total_duration = self.end_time - self.start_time

        # Calculate task completion rates
        self.tasks_total = len(self.task_metrics)
        self.tasks_completed = sum(1 for t in self.task_metrics if t.success)
        self.tasks_failed = sum(1 for t in self.task_metrics if not t.success)

    def get_task_performance(self) -> dict[str, Any]:
        """Get task performance analytics."""
        if not self.task_metrics:
            return {"average_duration": 0, "success_rate": 0, "total_tasks": 0}

        successful_tasks = [t for t in self.task_metrics if t.success and t.duration]
        durations = [t.duration for t in successful_tasks if t.duration]

        return {
            "average_duration": mean(durations) if durations else 0,
            "median_duration": median(durations) if durations else 0,
            "success_rate": len(successful_tasks) / len(self.task_metrics) * 100,
            "total_tasks": len(self.task_metrics),
            "completed_tasks": len(successful_tasks),
            "failed_tasks": len(self.task_metrics) - len(successful_tasks),
        }

    def get_efficiency_metrics(self) -> dict[str, Any]:
        """Get efficiency and optimization metrics."""
        performance = self.get_task_performance()

        # Calculate throughput
        throughput = 0
        if self.total_duration and self.total_duration > 0:
            throughput = self.tasks_completed / self.total_duration  # type: ignore[assignment]

        # Calculate resource efficiency
        resource_efficiency = 1.0
        if self.average_cpu_usage:
            resource_efficiency = min(1.0, 1.0 - (self.average_cpu_usage / 100))

        return {
            "tasks_per_second": throughput,
            "average_task_duration": performance["average_duration"],
            "resource_efficiency": resource_efficiency,
            "memory_efficiency": 1.0
            - min(1.0, (self.peak_memory_usage or 0) / 1000),  # Assume 1GB baseline
            "retry_rate": (self.retry_count / max(self.tasks_total, 1)) * 100,
        }


class GriptapeWorkflowMonitor:
    """
    Performance monitoring system for Griptape AI framework structures.

    Tracks execution metrics, resource utilization, and provides optimization
    insights for Agents, Pipelines, Workflows, and other Griptape components.
    """

    def __init__(
        self,
        enable_performance_monitoring: bool = True,
        enable_resource_tracking: bool = True,
        max_history_size: int = 1000,
    ):
        """Initialize workflow monitor."""

        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_resource_tracking = enable_resource_tracking
        self.max_history_size = max_history_size

        # Active monitoring sessions
        self.active_sessions: dict[str, GriptapeStructureMetrics] = {}
        self.active_tasks: dict[str, GriptapeTaskMetrics] = {}

        # Historical data
        self.completed_sessions: deque = deque(maxlen=max_history_size)

        # Thread safety
        self._lock = threading.Lock()

        # Performance baselines (updated based on historical data)
        self.performance_baselines = {
            "agent": {"average_duration": 5.0, "success_rate": 95.0},
            "pipeline": {"average_duration": 15.0, "success_rate": 92.0},
            "workflow": {"average_duration": 25.0, "success_rate": 90.0},
        }

        logger.info(
            f"Griptape workflow monitor initialized: "
            f"performance={enable_performance_monitoring}, "
            f"resources={enable_resource_tracking}"
        )

    def start_structure_monitoring(
        self, request_id: str, structure_type: str, structure_id: Optional[str] = None
    ) -> None:
        """Start monitoring a Griptape structure execution."""

        if not self.enable_performance_monitoring:
            return

        structure_id = structure_id or request_id

        metrics = GriptapeStructureMetrics(
            structure_id=structure_id,
            structure_type=structure_type,
            start_time=time.time(),
        )

        with self._lock:
            self.active_sessions[request_id] = metrics

        logger.debug(f"Started monitoring {structure_type}: {structure_id}")

    def stop_structure_monitoring(
        self, request_id: str
    ) -> Optional[GriptapeStructureMetrics]:
        """Stop monitoring and return final metrics."""

        if not self.enable_performance_monitoring:
            return None

        with self._lock:
            metrics = self.active_sessions.pop(request_id, None)

        if metrics:
            metrics.finalize()

            # Store in history
            with self._lock:
                self.completed_sessions.append(metrics)

            # Update performance baselines
            self._update_baselines(metrics)

            logger.debug(
                f"Stopped monitoring {metrics.structure_type}: {metrics.structure_id}, "
                f"duration={metrics.total_duration:.3f}s, "
                f"tasks={metrics.tasks_completed}/{metrics.tasks_total}"
            )

        return metrics

    def start_task_monitoring(
        self, request_id: str, task_id: str, task_type: str = "generic"
    ) -> None:
        """Start monitoring individual task execution."""

        if not self.enable_performance_monitoring:
            return

        task_metrics = GriptapeTaskMetrics(
            task_id=task_id, task_type=task_type, start_time=time.time()
        )

        with self._lock:
            self.active_tasks[f"{request_id}:{task_id}"] = task_metrics

            # Update structure metrics
            if request_id in self.active_sessions:
                self.active_sessions[request_id].tasks_total += 1

        logger.debug(f"Started task monitoring: {task_id} ({task_type})")

    def stop_task_monitoring(
        self,
        request_id: str,
        task_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Stop monitoring individual task."""

        if not self.enable_performance_monitoring:
            return

        task_key = f"{request_id}:{task_id}"

        with self._lock:
            task_metrics = self.active_tasks.pop(task_key, None)

            if task_metrics:
                task_metrics.success = success
                task_metrics.error_message = error_message
                task_metrics.finalize()

                # Update structure metrics
                if request_id in self.active_sessions:
                    structure_metrics = self.active_sessions[request_id]
                    structure_metrics.task_metrics.append(task_metrics)

                    if success:
                        structure_metrics.tasks_completed += 1
                    else:
                        structure_metrics.tasks_failed += 1
                        if error_message:
                            structure_metrics.errors.append(error_message)

        logger.debug(f"Stopped task monitoring: {task_id}, success={success}")

    def record_task_execution(
        self,
        task_id: str,
        duration: float,
        success: bool = True,
        tokens_processed: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Record task execution metrics directly."""

        if not self.enable_performance_monitoring:
            return

        # Find the task in active sessions
        task_key = None
        for key in self.active_tasks:
            if key.endswith(f":{task_id}"):
                task_key = key
                break

        if task_key:
            with self._lock:
                task_metrics = self.active_tasks.get(task_key)
                if task_metrics:
                    task_metrics.duration = duration
                    task_metrics.success = success
                    task_metrics.tokens_processed = tokens_processed
                    task_metrics.error_message = error_message

        logger.debug(
            f"Recorded task execution: {task_id}, "
            f"duration={duration:.3f}s, success={success}"
        )

    def record_memory_access(
        self,
        request_id: str,
        memory_type: str,
        operation: str = "access",
        duration: Optional[float] = None,
    ) -> None:
        """Record memory operation."""

        if not self.enable_performance_monitoring:
            return

        with self._lock:
            if request_id in self.active_sessions:
                self.active_sessions[request_id].memory_operations += 1

        logger.debug(f"Memory access: {memory_type} ({operation})")

    def record_tool_usage(
        self,
        request_id: str,
        tool_name: str,
        duration: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Record tool usage."""

        if not self.enable_performance_monitoring:
            return

        with self._lock:
            if request_id in self.active_sessions:
                self.active_sessions[request_id].tool_calls += 1

        logger.debug(f"Tool usage: {tool_name}, success={success}")

    def record_reasoning_step(
        self, request_id: str, step_type: str = "generic"
    ) -> None:
        """Record chain-of-thought reasoning step."""

        if not self.enable_performance_monitoring:
            return

        with self._lock:
            if request_id in self.active_sessions:
                self.active_sessions[request_id].reasoning_steps += 1

        logger.debug(f"Reasoning step: {step_type}")

    def get_performance_insights(
        self, structure_type: Optional[str] = None, days: int = 7
    ) -> dict[str, Any]:
        """Get performance insights and optimization recommendations."""

        cutoff_time = time.time() - (days * 24 * 60 * 60)

        # Filter recent sessions
        recent_sessions = []
        with self._lock:
            for session in self.completed_sessions:
                if session.start_time >= cutoff_time:
                    if not structure_type or session.structure_type == structure_type:
                        recent_sessions.append(session)

        if not recent_sessions:
            return {"sessions_analyzed": 0, "insights": [], "recommendations": []}

        # Calculate performance metrics
        durations = [s.total_duration for s in recent_sessions if s.total_duration]
        success_rates = [
            s.get_task_performance()["success_rate"] for s in recent_sessions
        ]

        insights = []
        recommendations = []

        # Performance analysis
        if durations:
            avg_duration = mean(durations)
            baseline = self.performance_baselines.get(
                structure_type or "agent", {}
            ).get("average_duration", 10.0)

            if avg_duration > baseline * 1.5:
                insights.append(
                    f"Average execution time ({avg_duration:.2f}s) is significantly above baseline"
                )
                recommendations.append(
                    "Consider optimizing task sequence or using faster models"
                )

        # Success rate analysis
        if success_rates:
            avg_success_rate = mean(success_rates)
            baseline_success = self.performance_baselines.get(
                structure_type or "agent", {}
            ).get("success_rate", 95.0)

            if avg_success_rate < baseline_success - 5:
                insights.append(
                    f"Success rate ({avg_success_rate:.1f}%) is below baseline"
                )
                recommendations.append(
                    "Review error patterns and improve error handling"
                )

        # Resource utilization analysis
        high_memory_sessions = [
            s
            for s in recent_sessions
            if s.peak_memory_usage and s.peak_memory_usage > 500
        ]
        if len(high_memory_sessions) > len(recent_sessions) * 0.3:
            insights.append("High memory usage detected in multiple sessions")
            recommendations.append(
                "Consider implementing memory optimization strategies"
            )

        # Task failure analysis
        failed_tasks = []
        for session in recent_sessions:
            failed_tasks.extend([t for t in session.task_metrics if not t.success])

        if len(failed_tasks) > 0:
            error_patterns = defaultdict(int)
            for task in failed_tasks:
                if task.error_message:
                    # Simple error categorization
                    if "timeout" in task.error_message.lower():
                        error_patterns["timeout"] += 1
                    elif "rate limit" in task.error_message.lower():
                        error_patterns["rate_limit"] += 1
                    elif "api" in task.error_message.lower():
                        error_patterns["api_error"] += 1
                    else:
                        error_patterns["other"] += 1

            if error_patterns:
                most_common_error = max(error_patterns.items(), key=lambda x: x[1])
                insights.append(
                    f"Most common error type: {most_common_error[0]} ({most_common_error[1]} occurrences)"
                )

                if most_common_error[0] == "timeout":
                    recommendations.append("Implement timeout handling and retry logic")
                elif most_common_error[0] == "rate_limit":
                    recommendations.append(
                        "Implement rate limiting and backoff strategies"
                    )

        return {
            "sessions_analyzed": len(recent_sessions),
            "time_period_days": days,
            "structure_type": structure_type,
            "insights": insights,
            "recommendations": recommendations,
            "performance_summary": {
                "average_duration": mean(durations) if durations else 0,
                "average_success_rate": mean(success_rates) if success_rates else 0,
                "total_sessions": len(recent_sessions),
                "total_tasks": sum(s.tasks_total for s in recent_sessions),
            },
        }

    def _update_baselines(self, metrics: GriptapeStructureMetrics) -> None:
        """Update performance baselines based on historical data."""

        if metrics.structure_type not in self.performance_baselines:
            self.performance_baselines[metrics.structure_type] = {
                "average_duration": 10.0,
                "success_rate": 95.0,
            }

        # Simple exponential moving average update
        current_baseline = self.performance_baselines[metrics.structure_type]

        if metrics.total_duration:
            current_baseline["average_duration"] = (
                current_baseline["average_duration"] * 0.9
                + metrics.total_duration * 0.1
            )

        task_performance = metrics.get_task_performance()
        if task_performance["total_tasks"] > 0:
            current_baseline["success_rate"] = (
                current_baseline["success_rate"] * 0.9
                + task_performance["success_rate"] * 0.1
            )

    def get_active_sessions(self) -> dict[str, GriptapeStructureMetrics]:
        """Get currently active monitoring sessions."""
        with self._lock:
            return dict(self.active_sessions)

    def get_session_history(
        self, limit: int = 100, structure_type: Optional[str] = None
    ) -> list[GriptapeStructureMetrics]:
        """Get historical session data."""

        sessions = []
        with self._lock:
            for session in list(self.completed_sessions)[-limit:]:
                if not structure_type or session.structure_type == structure_type:
                    sessions.append(session)

        return sessions
