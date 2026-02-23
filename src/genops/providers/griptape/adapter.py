#!/usr/bin/env python3
"""
Griptape AI Framework Adapter for GenOps Governance

Provides comprehensive governance telemetry for Griptape AI agent and workflow framework,
including structure-level tracking (Agent, Pipeline, Workflow), multi-provider cost aggregation,
and enterprise compliance patterns.

Usage:
    from genops.providers.griptape import GenOpsGriptapeAdapter

    adapter = GenOpsGriptapeAdapter(
        team="ai-research",
        project="multi-agent-system",
        daily_budget_limit=100.0
    )

    # Track agent execution
    with adapter.track_agent("research-agent") as context:
        result = agent.run("Research question")
        print(f"Total cost: ${context.total_cost:.6f}")

    # Track pipeline workflow
    with adapter.track_pipeline("analysis-pipeline") as context:
        result = pipeline.run({"data": input_data})
        print(f"Pipeline cost: ${context.total_cost:.6f}")

    # Track parallel workflow
    with adapter.track_workflow("parallel-workflow") as context:
        result = workflow.run({"tasks": task_list})
        print(f"Workflow cost: ${context.total_cost:.6f}")

Features:
    - Agent, Pipeline, and Workflow governance with unified cost tracking
    - Multi-provider cost aggregation across OpenAI, Anthropic, Google, etc.
    - Memory operation tracking (Conversation, Task, Meta Memory)
    - Engine operation governance (RAG, Extraction, Summary, Evaluation)
    - Tool usage monitoring and external API governance
    - Chain-of-thought reasoning analysis and optimization
    - Enterprise compliance patterns and multi-tenant support
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional, Union

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from .cost_aggregator import GriptapeCostAggregator
    from .workflow_monitor import GriptapeWorkflowMonitor

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# GenOps core imports
from genops.core.base_provider import BaseProvider
from genops.core.governance import GovernanceAttributes
from genops.core.telemetry import TelemetryExporter

logger = logging.getLogger(__name__)

# Structure type constants
STRUCTURE_AGENT = "agent"
STRUCTURE_PIPELINE = "pipeline"
STRUCTURE_WORKFLOW = "workflow"
STRUCTURE_ENGINE = "engine"
STRUCTURE_MEMORY = "memory"


@dataclass
class GriptapeRequest:
    """Represents a Griptape structure execution request with governance tracking."""

    # Core request identification
    request_id: str
    structure_type: str  # agent, pipeline, workflow, engine, memory
    structure_id: str
    operation_type: str  # run, execute, process, retrieve, store

    # Timing information
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Cost and usage tracking
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    provider_costs: dict[str, Decimal] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)

    # Structure-specific metrics
    task_count: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    memory_operations: int = 0
    tool_calls: int = 0
    reasoning_steps: int = 0

    # Governance and attribution
    governance_attrs: dict[str, Any] = field(default_factory=dict)

    # Provider and model tracking
    providers_used: set[str] = field(default_factory=set)
    models_used: set[str] = field(default_factory=set)

    # Error and status information
    status: str = "running"
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    # Structure execution details
    input_data: Optional[dict[str, Any]] = None
    output_data: Optional[dict[str, Any]] = None
    structure_config: Optional[dict[str, Any]] = None

    def finalize(self) -> None:
        """Finalize the request with completion metrics."""
        if self.end_time is None:
            self.end_time = time.time()

        self.duration = self.end_time - self.start_time

        # Update status based on task completion
        if self.failed_tasks > 0:
            self.status = "partial_failure" if self.completed_tasks > 0 else "failed"
        elif self.completed_tasks > 0:
            self.status = "completed"
        else:
            self.status = "no_execution"

        logger.debug(
            f"Griptape request {self.request_id} finalized: "
            f"{self.structure_type}={self.structure_id}, "
            f"duration={self.duration:.3f}s, "
            f"cost=${self.total_cost:.6f}, "
            f"tasks={self.completed_tasks}/{self.task_count}, "
            f"status={self.status}"
        )

    def add_provider_cost(
        self, provider: str, model: str, cost: Union[Decimal, float]
    ) -> None:
        """Add cost for a specific provider and model."""
        cost_decimal = Decimal(str(cost))

        if provider not in self.provider_costs:
            self.provider_costs[provider] = Decimal("0")

        self.provider_costs[provider] += cost_decimal
        self.total_cost += cost_decimal

        # Track providers and models used
        self.providers_used.add(provider)
        self.models_used.add(model)

        logger.debug(
            f"Added cost for {provider}/{model}: ${cost_decimal:.6f} "
            f"(total now: ${self.total_cost:.6f})"
        )

    def add_task_completion(self, success: bool = True) -> None:
        """Record task completion status."""
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1

        logger.debug(
            f"Task completed: success={success}, "
            f"completed={self.completed_tasks}, failed={self.failed_tasks}"
        )


class GenOpsGriptapeAdapter(BaseProvider):
    """
    GenOps adapter for Griptape AI framework providing comprehensive governance.

    Supports all Griptape structure types:
    - Agents: Single-task operations with LLM provider tracking
    - Pipelines: Sequential task execution with cost aggregation
    - Workflows: Parallel task monitoring and attribution
    - Engines: RAG, extraction, summary, evaluation tracking
    - Memory: Conversation and task memory governance
    """

    def __init__(
        self,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: Optional[str] = None,
        cost_center: Optional[str] = None,
        customer_id: Optional[str] = None,
        feature: Optional[str] = None,
        daily_budget_limit: Optional[float] = None,
        enable_cost_tracking: bool = True,
        enable_performance_monitoring: bool = True,
        sampling_rate: float = 1.0,
        **kwargs,
    ):
        """Initialize Griptape adapter with governance configuration."""
        super().__init__(**kwargs)

        # Governance attributes
        self.governance_attrs = GovernanceAttributes(
            team=team,  # type: ignore
            project=project,  # type: ignore
            environment=environment,  # type: ignore
            cost_center=cost_center,
            customer_id=customer_id,
            feature=feature,
        )

        # Cost and performance configuration
        self.daily_budget_limit = daily_budget_limit
        self.enable_cost_tracking = enable_cost_tracking
        self.enable_performance_monitoring = enable_performance_monitoring
        self.sampling_rate = sampling_rate

        # Initialize components (lazy loading to avoid import issues)
        self._cost_aggregator: Optional["GriptapeCostAggregator"] = None
        self._workflow_monitor: Optional["GriptapeWorkflowMonitor"] = None
        self._telemetry_exporter: Optional[TelemetryExporter] = None

        # OpenTelemetry tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info(
            f"GenOps Griptape adapter initialized: "
            f"team={team}, project={project}, "
            f"cost_tracking={enable_cost_tracking}, "
            f"performance_monitoring={enable_performance_monitoring}"
        )

    @property
    def cost_aggregator(self) -> "GriptapeCostAggregator":
        """Lazy load cost aggregator to avoid circular imports."""
        if self._cost_aggregator is None:
            from .cost_aggregator import GriptapeCostAggregator

            self._cost_aggregator = GriptapeCostAggregator()
        return self._cost_aggregator

    @property
    def workflow_monitor(self) -> "GriptapeWorkflowMonitor":
        """Lazy load workflow monitor to avoid circular imports."""
        if self._workflow_monitor is None:
            from .workflow_monitor import GriptapeWorkflowMonitor

            self._workflow_monitor = GriptapeWorkflowMonitor(
                enable_performance_monitoring=self.enable_performance_monitoring
            )
        return self._workflow_monitor

    @property
    def telemetry_exporter(self) -> TelemetryExporter:
        """Lazy load telemetry exporter."""
        if self._telemetry_exporter is None:
            self._telemetry_exporter = TelemetryExporter()
        return self._telemetry_exporter

    def _create_request(
        self,
        structure_type: str,
        structure_id: str,
        operation_type: str = "run",
        **kwargs,
    ) -> GriptapeRequest:
        """Create a new Griptape request with governance attributes."""
        request_id = f"griptape-{structure_type}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

        request = GriptapeRequest(
            request_id=request_id,
            structure_type=structure_type,
            structure_id=structure_id,
            operation_type=operation_type,
            start_time=time.time(),
            governance_attrs=self.governance_attrs.to_dict(),
            **kwargs,
        )

        logger.debug(
            f"Created Griptape request: {request_id} "
            f"({structure_type}={structure_id}, operation={operation_type})"
        )

        return request

    def _export_telemetry(self, request: GriptapeRequest) -> None:
        """Export telemetry data for a completed request."""
        try:
            # Structure telemetry attributes
            attributes = {
                # Core Griptape attributes
                "genops.provider": "griptape",
                "genops.structure.type": request.structure_type,
                "genops.structure.id": request.structure_id,
                "genops.operation.type": request.operation_type,
                # Request identification
                "genops.request.id": request.request_id,
                "genops.request.status": request.status,
                # Cost and usage metrics
                "genops.cost.total": float(request.total_cost),
                "genops.cost.currency": "USD",
                "genops.tasks.total": request.task_count,
                "genops.tasks.completed": request.completed_tasks,
                "genops.tasks.failed": request.failed_tasks,
                # Performance metrics
                "genops.duration.total": request.duration or 0,
                "genops.memory.operations": request.memory_operations,
                "genops.tools.calls": request.tool_calls,
                "genops.reasoning.steps": request.reasoning_steps,
                # Provider information
                "genops.providers.count": len(request.providers_used),
                "genops.providers.used": ",".join(sorted(request.providers_used)),
                "genops.models.count": len(request.models_used),
                "genops.models.used": ",".join(sorted(request.models_used)),
            }

            # Add governance attributes
            attributes.update(request.governance_attrs)

            # Add provider-specific costs
            for provider, cost in request.provider_costs.items():
                attributes[f"genops.cost.{provider}"] = float(cost)

            # Add token counts
            for provider, tokens in request.token_counts.items():
                attributes[f"genops.tokens.{provider}"] = tokens

            # Export telemetry
            self.telemetry_exporter.export_span(
                name=f"griptape.{request.structure_type}.{request.operation_type}",
                attributes=attributes,
                start_time=request.start_time,
                end_time=request.end_time or time.time(),
                status=Status(
                    StatusCode.OK if request.status == "completed" else StatusCode.ERROR
                ),
            )

            logger.debug(f"Exported telemetry for request {request.request_id}")

        except Exception as e:
            logger.error(
                f"Failed to export telemetry for request {request.request_id}: {e}"
            )

    @contextmanager
    def track_agent(self, agent_id: str, **kwargs):
        """Context manager for tracking Griptape Agent execution."""
        request = self._create_request(STRUCTURE_AGENT, agent_id, **kwargs)

        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(
            f"griptape.agent.{request.operation_type}",
            attributes={
                "griptape.structure.type": STRUCTURE_AGENT,
                "griptape.structure.id": agent_id,
                "genops.request.id": request.request_id,
            },
        ) as span:
            try:
                # Start performance monitoring
                if self.enable_performance_monitoring:
                    self.workflow_monitor.start_structure_monitoring(
                        request.request_id, STRUCTURE_AGENT
                    )

                logger.info(f"Starting Agent tracking: {agent_id}")
                yield request

                # Mark as completed
                request.status = "completed"
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    f"Agent {agent_id} completed: ${request.total_cost:.6f}, "
                    f"{request.completed_tasks} tasks, {request.duration:.3f}s"
                )

            except Exception as e:
                request.status = "failed"
                request.error_message = str(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                logger.error(f"Agent {agent_id} failed: {e}")
                raise

            finally:
                # Finalize request
                request.finalize()

                # Stop performance monitoring
                if self.enable_performance_monitoring:
                    metrics = self.workflow_monitor.stop_structure_monitoring(
                        request.request_id
                    )
                    if metrics:
                        request.memory_operations = metrics.memory_operations
                        request.tool_calls = metrics.tool_calls
                        request.reasoning_steps = metrics.reasoning_steps

                # Export telemetry
                self._export_telemetry(request)

    @contextmanager
    def track_pipeline(self, pipeline_id: str, **kwargs):
        """Context manager for tracking Griptape Pipeline execution."""
        request = self._create_request(STRUCTURE_PIPELINE, pipeline_id, **kwargs)

        with self.tracer.start_as_current_span(
            f"griptape.pipeline.{request.operation_type}",
            attributes={
                "griptape.structure.type": STRUCTURE_PIPELINE,
                "griptape.structure.id": pipeline_id,
                "genops.request.id": request.request_id,
            },
        ) as span:
            try:
                if self.enable_performance_monitoring:
                    self.workflow_monitor.start_structure_monitoring(
                        request.request_id, STRUCTURE_PIPELINE
                    )

                logger.info(f"Starting Pipeline tracking: {pipeline_id}")
                yield request

                request.status = "completed"
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    f"Pipeline {pipeline_id} completed: ${request.total_cost:.6f}, "
                    f"{request.completed_tasks} tasks, {request.duration:.3f}s"
                )

            except Exception as e:
                request.status = "failed"
                request.error_message = str(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                logger.error(f"Pipeline {pipeline_id} failed: {e}")
                raise

            finally:
                request.finalize()

                if self.enable_performance_monitoring:
                    metrics = self.workflow_monitor.stop_structure_monitoring(
                        request.request_id
                    )
                    if metrics:
                        request.memory_operations = metrics.memory_operations
                        request.tool_calls = metrics.tool_calls
                        request.reasoning_steps = metrics.reasoning_steps

                self._export_telemetry(request)

    @contextmanager
    def track_workflow(self, workflow_id: str, **kwargs):
        """Context manager for tracking Griptape Workflow execution."""
        request = self._create_request(STRUCTURE_WORKFLOW, workflow_id, **kwargs)

        with self.tracer.start_as_current_span(
            f"griptape.workflow.{request.operation_type}",
            attributes={
                "griptape.structure.type": STRUCTURE_WORKFLOW,
                "griptape.structure.id": workflow_id,
                "genops.request.id": request.request_id,
            },
        ) as span:
            try:
                if self.enable_performance_monitoring:
                    self.workflow_monitor.start_structure_monitoring(
                        request.request_id, STRUCTURE_WORKFLOW
                    )

                logger.info(f"Starting Workflow tracking: {workflow_id}")
                yield request

                request.status = "completed"
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    f"Workflow {workflow_id} completed: ${request.total_cost:.6f}, "
                    f"{request.completed_tasks} tasks, {request.duration:.3f}s"
                )

            except Exception as e:
                request.status = "failed"
                request.error_message = str(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                logger.error(f"Workflow {workflow_id} failed: {e}")
                raise

            finally:
                request.finalize()

                if self.enable_performance_monitoring:
                    metrics = self.workflow_monitor.stop_structure_monitoring(
                        request.request_id
                    )
                    if metrics:
                        request.memory_operations = metrics.memory_operations
                        request.tool_calls = metrics.tool_calls
                        request.reasoning_steps = metrics.reasoning_steps

                self._export_telemetry(request)

    @contextmanager
    def track_engine(self, engine_id: str, engine_type: str = "generic", **kwargs):
        """Context manager for tracking Griptape Engine operations (RAG, Extraction, Summary, etc.)."""
        request = self._create_request(
            STRUCTURE_ENGINE, engine_id, operation_type=engine_type, **kwargs
        )

        with self.tracer.start_as_current_span(
            f"griptape.engine.{engine_type}",
            attributes={
                "griptape.structure.type": STRUCTURE_ENGINE,
                "griptape.engine.type": engine_type,
                "griptape.structure.id": engine_id,
                "genops.request.id": request.request_id,
            },
        ) as span:
            try:
                logger.info(f"Starting Engine tracking: {engine_id} ({engine_type})")
                yield request

                request.status = "completed"
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    f"Engine {engine_id} ({engine_type}) completed: "
                    f"${request.total_cost:.6f}, {request.duration:.3f}s"
                )

            except Exception as e:
                request.status = "failed"
                request.error_message = str(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                logger.error(f"Engine {engine_id} ({engine_type}) failed: {e}")
                raise

            finally:
                request.finalize()
                self._export_telemetry(request)

    @contextmanager
    def track_memory(self, memory_id: str, operation_type: str = "access", **kwargs):
        """Context manager for tracking Griptape Memory operations."""
        request = self._create_request(
            STRUCTURE_MEMORY, memory_id, operation_type, **kwargs
        )

        with self.tracer.start_as_current_span(
            f"griptape.memory.{operation_type}",
            attributes={
                "griptape.structure.type": STRUCTURE_MEMORY,
                "griptape.structure.id": memory_id,
                "genops.request.id": request.request_id,
            },
        ) as span:
            try:
                logger.debug(
                    f"Starting Memory tracking: {memory_id} ({operation_type})"
                )
                yield request

                request.status = "completed"
                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                request.status = "failed"
                request.error_message = str(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                logger.error(f"Memory {memory_id} ({operation_type}) failed: {e}")
                raise

            finally:
                request.finalize()
                self._export_telemetry(request)

    def get_daily_spending(self) -> Decimal:
        """Get total daily spending across all Griptape operations."""
        if not self.enable_cost_tracking:
            return Decimal("0")

        return self.cost_aggregator.get_daily_costs()

    def check_budget_compliance(self) -> dict[str, Any]:
        """Check current spending against daily budget limits."""
        if not self.daily_budget_limit:
            return {"status": "no_limit", "spending": float(self.get_daily_spending())}

        current_spending = self.get_daily_spending()
        limit = Decimal(str(self.daily_budget_limit))

        return {
            "status": "over_budget" if current_spending > limit else "within_budget",
            "spending": float(current_spending),
            "limit": float(limit),
            "utilization": float((current_spending / limit) * 100) if limit > 0 else 0,
        }
