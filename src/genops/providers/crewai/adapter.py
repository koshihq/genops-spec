#!/usr/bin/env python3
"""
CrewAI Framework Adapter for GenOps Governance

Provides comprehensive governance telemetry for CrewAI multi-agent systems,
including crew-level tracking, agent monitoring, and multi-provider cost aggregation.

Usage:
    from genops.providers.crewai import GenOpsCrewAIAdapter

    adapter = GenOpsCrewAIAdapter(
        team="ai-research",
        project="multi-agent-system",
        daily_budget_limit=100.0
    )

    # Track entire crew execution
    with adapter.track_crew("research-crew") as context:
        result = crew.kickoff()
        print(f"Total cost: ${context.total_cost:.6f}")

Features:
    - End-to-end crew governance and cost tracking
    - Agent-level instrumentation and performance monitoring
    - Multi-provider cost aggregation (OpenAI, Anthropic, etc.)
    - Task workflow specialization with execution tracking
    - Enterprise compliance patterns and multi-tenant governance
"""

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from genops.providers.crewai.agent_monitor import CrewAIAgentMonitor
    from genops.providers.crewai.cost_aggregator import CrewAICostAggregator
from datetime import datetime

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# GenOps core imports
from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

# Check for CrewAI availability
try:
    import crewai
    from crewai import Agent, Crew, Task

    HAS_CREWAI = True
    logger.info(f"CrewAI {crewai.__version__} detected")
except ImportError:
    HAS_CREWAI = False
    Agent = None
    Task = None
    Crew = None
    logger.warning("CrewAI not installed. Install with: pip install crewai")


@dataclass
class CrewAIAgentResult:
    """Result from a tracked CrewAI agent execution."""

    agent_name: str
    agent_role: str
    execution_time_seconds: float
    cost: Decimal
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    status: str = "success"
    error_message: Optional[str] = None
    custom_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrewAITaskResult:
    """Result from a tracked CrewAI task execution."""

    task_description: str
    task_id: str
    agent_name: str
    execution_time_seconds: float
    cost: Decimal
    status: str = "success"
    error_message: Optional[str] = None
    custom_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrewAICrewResult:
    """Result from a tracked CrewAI crew execution."""

    crew_name: str
    crew_id: str
    total_cost: Decimal
    total_execution_time_seconds: float
    agent_results: list[CrewAIAgentResult]
    task_results: list[CrewAITaskResult]
    cost_by_provider: dict[str, Decimal]
    cost_by_agent: dict[str, Decimal]
    total_agents: int
    successful_agents: int
    failed_agents: int
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    governance_attributes: dict[str, Any]


@dataclass
class CrewAISessionContext:
    """Context for tracking a CrewAI session with multiple crews."""

    session_name: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_cost: Decimal = Decimal("0")
    total_crews: int = 0
    crew_results: list[CrewAICrewResult] = field(default_factory=list)
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def add_crew_result(self, crew_result: CrewAICrewResult):
        """Add a crew result to the session."""
        self.crew_results.append(crew_result)
        self.total_cost += crew_result.total_cost
        self.total_crews += 1


class CrewAICrewContext:
    """Context manager for tracking a single CrewAI crew execution."""

    def __init__(self, adapter: "GenOpsCrewAIAdapter", crew_name: str, **attributes):
        self.adapter = adapter
        self.crew_name = crew_name
        self.crew_id = str(uuid.uuid4())
        self.start_time = None
        self.end_time = None
        self.agent_results: list[CrewAIAgentResult] = []
        self.task_results: list[CrewAITaskResult] = []
        self.cost_by_provider: dict[str, Decimal] = {}
        self.cost_by_agent: dict[str, Decimal] = {}
        self.custom_attributes = attributes
        self.span = None
        self.total_cost = Decimal("0")

    def __enter__(self):
        self.start_time = datetime.now()

        # Start OpenTelemetry span
        tracer = trace.get_tracer(__name__)
        self.span = tracer.start_span(f"crewai.crew.{self.crew_name}")

        # Set span attributes
        self.span.set_attributes(
            {
                "genops.crew.name": self.crew_name,
                "genops.crew.id": self.crew_id,
                "genops.team": self.adapter.team,
                "genops.project": self.adapter.project,
                "genops.environment": self.adapter.environment,
                **self.custom_attributes,
            }
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        execution_time = (self.end_time - self.start_time).total_seconds()

        if exc_type:
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        else:
            self.span.set_status(Status(StatusCode.OK))

        # Set final span attributes
        self.span.set_attributes(
            {
                "genops.crew.execution_time_seconds": execution_time,
                "genops.crew.total_cost": float(self.total_cost),
                "genops.crew.total_agents": len(self.agent_results),
                "genops.crew.total_tasks": len(self.task_results),
                "genops.crew.successful_agents": sum(
                    1 for r in self.agent_results if r.status == "success"
                ),
                "genops.crew.failed_agents": sum(
                    1 for r in self.agent_results if r.status != "success"
                ),
                "genops.crew.successful_tasks": sum(
                    1 for r in self.task_results if r.status == "success"
                ),
                "genops.crew.failed_tasks": sum(
                    1 for r in self.task_results if r.status != "success"
                ),
            }
        )

        self.span.end()

        # Create crew result
        crew_result = CrewAICrewResult(
            crew_name=self.crew_name,
            crew_id=self.crew_id,
            total_cost=self.total_cost,
            total_execution_time_seconds=execution_time,
            agent_results=self.agent_results,
            task_results=self.task_results,
            cost_by_provider=dict(self.cost_by_provider),
            cost_by_agent=dict(self.cost_by_agent),
            total_agents=len(self.agent_results),
            successful_agents=sum(
                1 for r in self.agent_results if r.status == "success"
            ),
            failed_agents=sum(1 for r in self.agent_results if r.status != "success"),
            total_tasks=len(self.task_results),
            successful_tasks=sum(1 for r in self.task_results if r.status == "success"),
            failed_tasks=sum(1 for r in self.task_results if r.status != "success"),
            governance_attributes={
                "team": self.adapter.team,
                "project": self.adapter.project,
                "environment": self.adapter.environment,
                **self.custom_attributes,
            },
        )

        # Add to adapter's cost aggregator if available
        if self.adapter.cost_aggregator:
            self.adapter.cost_aggregator.add_crew_execution(crew_result)

        # Store result in adapter
        self.adapter._crew_results.append(crew_result)

    def add_agent_result(self, agent_result: CrewAIAgentResult):
        """Add an agent execution result."""
        self.agent_results.append(agent_result)
        self.total_cost += agent_result.cost

        # Update cost by provider
        if agent_result.provider:
            if agent_result.provider not in self.cost_by_provider:
                self.cost_by_provider[agent_result.provider] = Decimal("0")
            self.cost_by_provider[agent_result.provider] += agent_result.cost

        # Update cost by agent
        if agent_result.agent_name not in self.cost_by_agent:
            self.cost_by_agent[agent_result.agent_name] = Decimal("0")
        self.cost_by_agent[agent_result.agent_name] += agent_result.cost

    def add_task_result(self, task_result: CrewAITaskResult):
        """Add a task execution result."""
        self.task_results.append(task_result)
        self.total_cost += task_result.cost

    def add_custom_metric(self, key: str, value: Any):
        """Add a custom metric to the crew context."""
        self.custom_attributes[key] = value
        if self.span:
            self.span.set_attribute(f"genops.custom.{key}", str(value))

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics for the crew execution."""
        return {
            "crew_name": self.crew_name,
            "crew_id": self.crew_id,
            "total_cost": float(self.total_cost),
            "total_agents": len(self.agent_results),
            "total_tasks": len(self.task_results),
            "cost_by_provider": {k: float(v) for k, v in self.cost_by_provider.items()},
            "cost_by_agent": {k: float(v) for k, v in self.cost_by_agent.items()},
            "custom_attributes": self.custom_attributes,
        }


class GenOpsCrewAIAdapter:
    """Main adapter for integrating CrewAI with GenOps governance telemetry."""

    def __init__(
        self,
        team: str,
        project: str,
        environment: str = "development",
        daily_budget_limit: float = 100.0,
        governance_policy: str = "advisory",
        enable_agent_tracking: bool = True,
        enable_cost_tracking: bool = True,
        enable_task_tracking: bool = True,
    ):
        """
        Initialize the GenOps CrewAI adapter.

        Args:
            team: Team name for governance
            project: Project name for governance
            environment: Environment (development, staging, production)
            daily_budget_limit: Daily spending limit in USD
            governance_policy: Policy enforcement level ("advisory", "enforced")
            enable_agent_tracking: Enable agent-level tracking
            enable_cost_tracking: Enable cost tracking
            enable_task_tracking: Enable task-level tracking
        """
        if not HAS_CREWAI:
            logger.warning(
                "CrewAI not installed - adapter available but limited functionality"
            )

        self.team = team
        self.project = project
        self.environment = environment
        self.daily_budget_limit = daily_budget_limit
        self.governance_policy = governance_policy
        self.enable_agent_tracking = enable_agent_tracking
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_task_tracking = enable_task_tracking

        # Initialize telemetry
        self.telemetry = GenOpsTelemetry()

        # Initialize components (lazy loading)
        self.cost_aggregator: Optional["CrewAICostAggregator"] = None
        self.agent_monitor: Optional["CrewAIAgentMonitor"] = None

        # Results storage
        self._crew_results: list[CrewAICrewResult] = []
        self._active_sessions: dict[str, CrewAISessionContext] = {}

        # Lazy initialization of components
        self._lazy_init_components()

    def _lazy_init_components(self):
        """Lazy initialization of components to avoid circular imports."""
        try:
            if self.enable_cost_tracking and not self.cost_aggregator:
                # Import at runtime to avoid circular imports
                from genops.providers.crewai.cost_aggregator import CrewAICostAggregator

                self.cost_aggregator = CrewAICostAggregator(
                    budget_limit=self.daily_budget_limit
                )

            if self.enable_agent_tracking and not self.agent_monitor:
                # Import at runtime to avoid circular imports
                from genops.providers.crewai.agent_monitor import CrewAIAgentMonitor

                self.agent_monitor = CrewAIAgentMonitor(
                    team=self.team,
                    project=self.project,
                    environment=self.environment,
                    enable_performance_monitoring=True,
                    enable_cost_tracking=self.enable_cost_tracking,
                    enable_task_tracking=self.enable_task_tracking,
                )
        except ImportError as e:
            logger.debug(f"Could not initialize components: {e}")

    def _ensure_components_initialized(self):
        """Ensure all components are initialized."""
        if (self.enable_cost_tracking and not self.cost_aggregator) or (
            self.enable_agent_tracking and not self.agent_monitor
        ):
            self._lazy_init_components()

    @contextmanager
    def track_crew(self, crew_name: str, **attributes):
        """
        Track a CrewAI crew execution.

        Args:
            crew_name: Name of the crew being tracked
            **attributes: Additional attributes for governance

        Returns:
            CrewAICrewContext: Context manager for the crew execution

        Example:
            with adapter.track_crew("research-crew", use_case="market-analysis") as context:
                result = crew.kickoff()
        """
        self._ensure_components_initialized()
        return CrewAICrewContext(self, crew_name, **attributes)

    @contextmanager
    def track_session(self, session_name: str, **attributes):
        """
        Track a session with multiple crew executions.

        Args:
            session_name: Name of the session
            **attributes: Additional attributes for governance

        Returns:
            CrewAISessionContext: Context manager for the session
        """
        session_id = str(uuid.uuid4())
        session = CrewAISessionContext(
            session_name=session_name, session_id=session_id, start_time=datetime.now()
        )

        # Add custom attributes
        for key, value in attributes.items():
            session.custom_metrics[key] = value

        self._active_sessions[session_id] = session

        try:
            yield session
        finally:
            session.end_time = datetime.now()
            # Session is kept in active sessions for retrieval

    def get_cost_summary(self, time_period_hours: int = 24) -> dict[str, Any]:
        """Get cost summary for the specified time period."""
        if not self.cost_aggregator:
            return {"error": "Cost tracking not enabled"}

        return self.cost_aggregator.get_cost_summary(time_period_hours)

    def get_crew_results(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent crew execution results."""
        results = []
        for crew_result in self._crew_results[-limit:]:
            results.append(
                {
                    "crew_name": crew_result.crew_name,
                    "crew_id": crew_result.crew_id,
                    "total_cost": float(crew_result.total_cost),
                    "execution_time_seconds": crew_result.total_execution_time_seconds,
                    "total_agents": crew_result.total_agents,
                    "total_tasks": crew_result.total_tasks,
                    "success_rate": (
                        crew_result.successful_agents + crew_result.successful_tasks
                    )
                    / max(1, crew_result.total_agents + crew_result.total_tasks),
                    "cost_by_provider": {
                        k: float(v) for k, v in crew_result.cost_by_provider.items()
                    },
                    "governance_attributes": crew_result.governance_attributes,
                }
            )
        return results


# Convenience functions for common patterns
def instrument_crewai(
    team: str = "default-team",
    project: str = "crewai-app",
    environment: str = "development",
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory",
) -> GenOpsCrewAIAdapter:
    """
    Convenience function to create and configure a CrewAI adapter.

    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        environment: Environment (development, staging, production)
        daily_budget_limit: Daily spending limit in USD
        governance_policy: Policy enforcement level ("advisory", "enforced")

    Returns:
        GenOpsCrewAIAdapter: Configured adapter

    Example:
        from genops.providers.crewai import instrument_crewai

        adapter = instrument_crewai(
            team="ml-team",
            project="research-agents",
            daily_budget_limit=50.0
        )

        with adapter.track_crew("market-research") as context:
            result = crew.kickoff()
    """
    return GenOpsCrewAIAdapter(
        team=team,
        project=project,
        environment=environment,
        daily_budget_limit=daily_budget_limit,
        governance_policy=governance_policy,
    )


def create_multi_agent_adapter(
    team: str,
    project: str,
    daily_budget_limit: float = 200.0,
    enable_advanced_monitoring: bool = True,
) -> GenOpsCrewAIAdapter:
    """
    Create a GenOps adapter optimized for multi-agent workflows.

    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        daily_budget_limit: Daily spending limit
        enable_advanced_monitoring: Enable advanced agent monitoring

    Returns:
        GenOpsCrewAIAdapter: Configured adapter for multi-agent workflows

    Example:
        from genops.providers.crewai import create_multi_agent_adapter

        adapter = create_multi_agent_adapter(
            team="ai-research",
            project="collaborative-agents",
            daily_budget_limit=300.0
        )

        with adapter.track_crew("research-analysis-crew") as context:
            result = multi_agent_crew.kickoff()
    """
    return GenOpsCrewAIAdapter(
        team=team,
        project=project,
        daily_budget_limit=daily_budget_limit,
        enable_agent_tracking=enable_advanced_monitoring,
        enable_task_tracking=enable_advanced_monitoring,
        enable_cost_tracking=True,
        governance_policy="advisory",
    )


# Export main classes and functions
__all__ = [
    "GenOpsCrewAIAdapter",
    "CrewAIAgentResult",
    "CrewAITaskResult",
    "CrewAICrewResult",
    "CrewAISessionContext",
    "CrewAICrewContext",
    "instrument_crewai",
    "create_multi_agent_adapter",
]
