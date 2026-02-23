#!/usr/bin/env python3
"""
CrewAI Agent and Workflow Monitor

Advanced monitoring system for CrewAI agents with workflow-level instrumentation,
performance tracking, and governance telemetry. Provides deep insights into
agent execution, crew interactions, and task completion patterns.

Usage:
    from genops.providers.crewai.agent_monitor import CrewAIAgentMonitor

    monitor = CrewAIAgentMonitor(team="ai-team", project="multi-agent-system")

    # Monitor entire crew execution
    with monitor.monitor_crew(crew, "research-analysis") as execution:
        result = crew.kickoff()

    # Get detailed execution metrics
    metrics = execution.get_metrics()
    print(f"Agents executed: {metrics.total_agents}")
    print(f"Total cost: ${metrics.total_cost:.6f}")

Features:
    - Real-time crew execution monitoring
    - Agent-level performance and cost tracking
    - Task workflow specialization (sequential, hierarchical, parallel)
    - Multi-agent collaboration analysis
    - Resource utilization and bottleneck detection
    - Error handling and failure analysis
    - Performance optimization recommendations
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Optional

# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

# GenOps imports
from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

# Check for CrewAI availability
try:
    import crewai  # noqa: F401
    from crewai import Agent, Crew, Task
    from crewai.agents.agent import Agent as CrewAIAgent
    from crewai.crew import Crew as CrewAICrew
    from crewai.task import Task as CrewAITask

    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    Agent = None
    Task = None
    Crew = None
    CrewAIAgent = None
    CrewAITask = None
    CrewAICrew = None
    logger.warning("CrewAI not installed. Install with: pip install crewai")


@dataclass
class AgentExecutionMetrics:
    """Detailed metrics for a single agent execution."""

    agent_name: str
    agent_role: str
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cost: Decimal = Decimal("0")
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    operations: int = 1
    status: str = "success"  # "success", "error", "timeout"
    error_message: Optional[str] = None
    input_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task_context: Optional[str] = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionMetrics:
    """Metrics for a CrewAI task execution."""

    task_description: str
    task_id: str
    agent_name: str
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float
    status: str = "success"
    error_message: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    custom_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrewExecutionMetrics:
    """Comprehensive metrics for a crew execution."""

    crew_name: str
    crew_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    total_agents: int = 0
    total_tasks: int = 0
    successful_agents: int = 0
    failed_agents: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_cost: Decimal = Decimal("0")
    agent_metrics: list[AgentExecutionMetrics] = field(default_factory=list)
    task_metrics: list[TaskExecutionMetrics] = field(default_factory=list)
    cost_by_provider: dict[str, Decimal] = field(default_factory=dict)
    cost_by_agent: dict[str, Decimal] = field(default_factory=dict)
    workflow_type: str = "sequential"  # "sequential", "hierarchical", "parallel"
    collaboration_score: Optional[float] = None
    efficiency_score: Optional[float] = None


@dataclass
class MultiAgentWorkflowMetrics:
    """Metrics for analyzing multi-agent workflow patterns."""

    crew_name: str
    agent_collaboration_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    task_dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    bottleneck_agents: list[str] = field(default_factory=list)
    optimal_agent_sequence: list[str] = field(default_factory=list)
    load_balancing_score: float = 0.0
    coordination_overhead_seconds: float = 0.0
    parallel_efficiency: float = 0.0


class CrewAIAgentMonitor:
    """Advanced monitoring system for CrewAI agents and workflows."""

    def __init__(
        self,
        team: str,
        project: str,
        environment: str = "development",
        enable_performance_monitoring: bool = True,
        enable_cost_tracking: bool = True,
        enable_task_tracking: bool = True,
        enable_workflow_analysis: bool = True,
    ):
        """
        Initialize the CrewAI agent monitor.

        Args:
            team: Team name for governance
            project: Project name for governance
            environment: Environment (development, staging, production)
            enable_performance_monitoring: Enable performance tracking
            enable_cost_tracking: Enable cost tracking
            enable_task_tracking: Enable task-level tracking
            enable_workflow_analysis: Enable workflow pattern analysis
        """
        self.team = team
        self.project = project
        self.environment = environment
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_task_tracking = enable_task_tracking
        self.enable_workflow_analysis = enable_workflow_analysis

        # Initialize telemetry
        self.telemetry = GenOpsTelemetry()

        # Thread-safe storage
        self._lock = threading.RLock()
        self._execution_results: dict[str, CrewExecutionMetrics] = {}
        self._agent_performance_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._workflow_patterns: dict[str, MultiAgentWorkflowMetrics] = {}

        # OpenTelemetry metrics
        self._setup_otel_metrics()

        logger.info(f"CrewAI agent monitor initialized for {team}/{project}")

    def _setup_otel_metrics(self):
        """Set up OpenTelemetry metrics."""
        meter = metrics.get_meter(__name__)

        self._agent_execution_counter = meter.create_counter(
            name="crewai_agent_executions_total",
            description="Total number of agent executions",
            unit="1",
        )

        self._crew_execution_duration = meter.create_histogram(
            name="crewai_crew_execution_duration_seconds",
            description="Crew execution duration in seconds",
            unit="s",
        )

        self._agent_cost_histogram = meter.create_histogram(
            name="crewai_agent_cost_usd",
            description="Agent execution cost in USD",
            unit="USD",
        )

    @contextmanager
    def monitor_crew(self, crew, crew_name: str, **attributes):
        """
        Monitor a complete CrewAI crew execution.

        Args:
            crew: CrewAI Crew instance
            crew_name: Name of the crew for tracking
            **attributes: Additional attributes for governance

        Example:
            with monitor.monitor_crew(research_crew, "market-analysis") as execution:
                result = research_crew.kickoff()
                metrics = execution.get_metrics()
        """
        crew_id = f"{crew_name}-{int(time.time())}"

        execution_metrics = CrewExecutionMetrics(
            crew_name=crew_name, crew_id=crew_id, start_time=datetime.now()
        )

        # Start OpenTelemetry span
        tracer = trace.get_tracer(__name__)
        span = tracer.start_span(f"crewai.crew.{crew_name}")

        try:
            # Set span attributes
            span.set_attributes(
                {
                    "genops.crew.name": crew_name,
                    "genops.crew.id": crew_id,
                    "genops.team": self.team,
                    "genops.project": self.project,
                    "genops.environment": self.environment,
                    **attributes,
                }
            )

            # Analyze crew structure if available
            if hasattr(crew, "agents") and hasattr(crew, "tasks"):
                execution_metrics.total_agents = len(crew.agents) if crew.agents else 0
                execution_metrics.total_tasks = len(crew.tasks) if crew.tasks else 0

                # Determine workflow type
                execution_metrics.workflow_type = self._detect_workflow_type(crew)

                # Set up agent monitoring if enabled
                if self.enable_performance_monitoring:
                    self._setup_agent_monitoring(crew, execution_metrics)

            with self._lock:
                self._execution_results[crew_id] = execution_metrics

            class CrewExecutionContext:
                def __init__(self, monitor, metrics, span):
                    self.monitor = monitor
                    self.metrics = metrics
                    self.span = span

                def add_agent_metrics(self, agent_metrics: AgentExecutionMetrics):
                    """Add agent execution metrics."""
                    self.metrics.agent_metrics.append(agent_metrics)
                    self.metrics.total_cost += agent_metrics.cost

                    if agent_metrics.status == "success":
                        self.metrics.successful_agents += 1
                    else:
                        self.metrics.failed_agents += 1

                    # Update cost by provider
                    if agent_metrics.provider:
                        if agent_metrics.provider not in self.metrics.cost_by_provider:
                            self.metrics.cost_by_provider[agent_metrics.provider] = (
                                Decimal("0")
                            )
                        self.metrics.cost_by_provider[agent_metrics.provider] += (
                            agent_metrics.cost
                        )

                    # Update cost by agent
                    if agent_metrics.agent_name not in self.metrics.cost_by_agent:
                        self.metrics.cost_by_agent[agent_metrics.agent_name] = Decimal(
                            "0"
                        )
                    self.metrics.cost_by_agent[agent_metrics.agent_name] += (
                        agent_metrics.cost
                    )

                def add_task_metrics(self, task_metrics: TaskExecutionMetrics):
                    """Add task execution metrics."""
                    self.metrics.task_metrics.append(task_metrics)

                    if task_metrics.status == "success":
                        self.metrics.successful_tasks += 1
                    else:
                        self.metrics.failed_tasks += 1

                def get_metrics(self) -> CrewExecutionMetrics:
                    """Get current execution metrics."""
                    return self.metrics

                def add_custom_metric(self, key: str, value: Any):
                    """Add custom metric to the execution."""
                    if hasattr(self.metrics, "custom_metrics"):
                        if not self.metrics.custom_metrics:
                            self.metrics.custom_metrics = {}
                        self.metrics.custom_metrics[key] = value

                    self.span.set_attribute(f"genops.custom.{key}", str(value))

            yield CrewExecutionContext(self, execution_metrics, span)

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            logger.error(f"Error monitoring crew {crew_name}: {e}")
            raise
        finally:
            # Finalize metrics
            execution_metrics.end_time = datetime.now()
            if execution_metrics.start_time and execution_metrics.end_time:
                execution_metrics.execution_time_seconds = (
                    execution_metrics.end_time - execution_metrics.start_time
                ).total_seconds()

            # Calculate performance scores
            if self.enable_workflow_analysis:
                execution_metrics.collaboration_score = (
                    self._calculate_collaboration_score(execution_metrics)
                )
                execution_metrics.efficiency_score = self._calculate_efficiency_score(
                    execution_metrics
                )

            # Set final span attributes
            span.set_attributes(
                {
                    "genops.crew.execution_time_seconds": execution_metrics.execution_time_seconds
                    or 0,
                    "genops.crew.total_cost": float(execution_metrics.total_cost),
                    "genops.crew.total_agents": execution_metrics.total_agents,
                    "genops.crew.total_tasks": execution_metrics.total_tasks,
                    "genops.crew.successful_agents": execution_metrics.successful_agents,
                    "genops.crew.failed_agents": execution_metrics.failed_agents,
                }
            )

            # Record OpenTelemetry metrics
            self._crew_execution_duration.record(
                execution_metrics.execution_time_seconds or 0,
                {"crew_name": crew_name, "team": self.team, "project": self.project},
            )

            span.end()

            # Store final results
            with self._lock:
                self._execution_results[crew_id] = execution_metrics

                # Update workflow patterns
                if self.enable_workflow_analysis:
                    self._analyze_workflow_patterns(execution_metrics)

    def _detect_workflow_type(self, crew) -> str:
        """Detect the workflow type of a crew."""
        # Simple heuristic - could be enhanced with actual CrewAI process detection
        try:
            if hasattr(crew, "process"):
                process_name = str(crew.process).lower()
                if "sequential" in process_name:
                    return "sequential"
                elif "hierarchical" in process_name:
                    return "hierarchical"
                elif "parallel" in process_name:
                    return "parallel"
        except Exception:
            pass

        return "sequential"  # Default assumption

    def _setup_agent_monitoring(self, crew, execution_metrics: CrewExecutionMetrics):
        """Set up monitoring for individual agents in the crew."""
        if not hasattr(crew, "agents") or not crew.agents:
            return

        # Store references for later monitoring
        # This would be enhanced with actual CrewAI hooks/callbacks
        for agent in crew.agents:
            if hasattr(agent, "role"):
                logger.debug(f"Monitoring setup for agent: {agent.role}")

    def track_agent_execution(
        self,
        agent_name: str,
        agent_role: str,
        execution_func: Callable,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Track execution of a single agent.

        Args:
            agent_name: Name of the agent
            agent_role: Role of the agent
            execution_func: Function to execute
            provider: AI provider used
            model: Model used
            **kwargs: Additional metrics

        Returns:
            Result of the execution function
        """
        start_time = datetime.now()

        try:
            result = execution_func()

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Create metrics
            agent_metrics = AgentExecutionMetrics(
                agent_name=agent_name,
                agent_role=agent_role,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                provider=provider,
                model=model,
                status="success",
            )

            # Add custom metrics
            for key, value in kwargs.items():
                agent_metrics.custom_metrics[key] = value

            # Store in performance history
            with self._lock:
                self._agent_performance_history[agent_name].append(agent_metrics)

            # Record OpenTelemetry metrics
            self._agent_execution_counter.add(
                1,
                {
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "status": "success",
                },
            )

            return result

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Create error metrics
            agent_metrics = AgentExecutionMetrics(
                agent_name=agent_name,
                agent_role=agent_role,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                provider=provider,
                model=model,
                status="error",
                error_message=str(e),
            )

            with self._lock:
                self._agent_performance_history[agent_name].append(agent_metrics)

            self._agent_execution_counter.add(
                1,
                {"agent_name": agent_name, "agent_role": agent_role, "status": "error"},
            )

            raise

    def _calculate_collaboration_score(self, metrics: CrewExecutionMetrics) -> float:
        """Calculate collaboration effectiveness score."""
        if metrics.total_agents <= 1:
            return 1.0

        # Simple heuristic based on success rates and timing
        success_rate = (metrics.successful_agents + metrics.successful_tasks) / max(
            1, metrics.total_agents + metrics.total_tasks
        )

        # Factor in execution time efficiency
        expected_time = metrics.total_agents * 30  # Assume 30s per agent baseline
        if metrics.execution_time_seconds and metrics.execution_time_seconds > 0:
            time_efficiency = min(1.0, expected_time / metrics.execution_time_seconds)
        else:
            time_efficiency = 1.0

        return (success_rate * 0.7) + (time_efficiency * 0.3)

    def _calculate_efficiency_score(self, metrics: CrewExecutionMetrics) -> float:
        """Calculate overall workflow efficiency score."""
        if not metrics.agent_metrics:
            return 0.0

        # Average agent efficiency
        agent_efficiencies = []
        for agent_metric in metrics.agent_metrics:
            if agent_metric.execution_time_seconds > 0:
                # Simple efficiency based on execution time and success
                base_efficiency = 1.0 if agent_metric.status == "success" else 0.0
                time_factor = min(
                    1.0, 60.0 / agent_metric.execution_time_seconds
                )  # 60s baseline
                agent_efficiencies.append(base_efficiency * time_factor)

        return statistics.mean(agent_efficiencies) if agent_efficiencies else 0.0

    def _analyze_workflow_patterns(self, metrics: CrewExecutionMetrics):
        """Analyze and store workflow patterns for optimization."""
        workflow_metrics = MultiAgentWorkflowMetrics(crew_name=metrics.crew_name)

        # Analyze agent collaboration patterns
        for i, agent1 in enumerate(metrics.agent_metrics):
            for _j, agent2 in enumerate(metrics.agent_metrics[i + 1 :], i + 1):
                # Simple collaboration detection based on timing overlap
                if (
                    agent1.start_time <= agent2.end_time
                    and agent2.start_time <= agent1.end_time
                ):
                    if (
                        agent1.agent_name
                        not in workflow_metrics.agent_collaboration_matrix
                    ):
                        workflow_metrics.agent_collaboration_matrix[
                            agent1.agent_name
                        ] = {}

                    workflow_metrics.agent_collaboration_matrix[agent1.agent_name][
                        agent2.agent_name
                    ] = (
                        workflow_metrics.agent_collaboration_matrix[
                            agent1.agent_name
                        ].get(agent2.agent_name, 0)
                        + 1
                    )

        # Identify bottlenecks (agents taking longest time)
        if metrics.agent_metrics:
            sorted_agents = sorted(
                metrics.agent_metrics,
                key=lambda x: x.execution_time_seconds,
                reverse=True,
            )
            avg_time = statistics.mean(
                a.execution_time_seconds for a in metrics.agent_metrics
            )
            workflow_metrics.bottleneck_agents = [
                a.agent_name
                for a in sorted_agents
                if a.execution_time_seconds > avg_time * 1.5
            ]

        with self._lock:
            self._workflow_patterns[metrics.crew_name] = workflow_metrics

    def get_execution_metrics(self, crew_id: str) -> Optional[CrewExecutionMetrics]:
        """Get execution metrics for a specific crew."""
        with self._lock:
            return self._execution_results.get(crew_id)

    def get_agent_performance_summary(self, agent_name: str) -> dict[str, Any]:
        """Get performance summary for a specific agent."""
        with self._lock:
            history = self._agent_performance_history.get(agent_name, deque())

            if not history:
                return {"error": "No performance data available"}

            recent_executions = list(history)[-10:]  # Last 10 executions

            avg_execution_time = statistics.mean(
                a.execution_time_seconds for a in recent_executions
            )
            success_rate = sum(
                1 for a in recent_executions if a.status == "success"
            ) / len(recent_executions)
            total_cost = sum(a.cost for a in recent_executions)

            return {
                "agent_name": agent_name,
                "total_executions": len(history),
                "recent_executions": len(recent_executions),
                "average_execution_time_seconds": avg_execution_time,
                "success_rate": success_rate,
                "total_cost_recent": float(total_cost),
                "performance_trend": "stable",  # Could be enhanced with trend analysis
            }

    def get_workflow_analysis(
        self, crew_name: str
    ) -> Optional[MultiAgentWorkflowMetrics]:
        """Get workflow pattern analysis for a crew."""
        with self._lock:
            return self._workflow_patterns.get(crew_name)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance summary."""
        with self._lock:
            total_executions = len(self._execution_results)

            if total_executions == 0:
                return {
                    "total_crews": 0,
                    "total_executions": 0,
                    "average_execution_time": 0,
                    "success_rate": 0,
                    "total_cost": 0,
                }

            successful_crews = sum(
                1
                for metrics in self._execution_results.values()
                if metrics.failed_agents == 0 and metrics.failed_tasks == 0
            )

            avg_execution_time = statistics.mean(
                metrics.execution_time_seconds or 0
                for metrics in self._execution_results.values()
            )

            total_cost = sum(
                metrics.total_cost for metrics in self._execution_results.values()
            )

            return {
                "total_crews": len(
                    {m.crew_name for m in self._execution_results.values()}
                ),
                "total_executions": total_executions,
                "successful_crews": successful_crews,
                "success_rate": successful_crews / total_executions,
                "average_execution_time_seconds": avg_execution_time,
                "total_cost": float(total_cost),
                "monitored_agents": len(self._agent_performance_history),
            }


# Export main classes and functions
__all__ = [
    "CrewAIAgentMonitor",
    "AgentExecutionMetrics",
    "TaskExecutionMetrics",
    "CrewExecutionMetrics",
    "MultiAgentWorkflowMetrics",
]
