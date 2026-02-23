#!/usr/bin/env python3
"""
AutoGen Framework Adapter for GenOps Governance

Provides comprehensive governance telemetry for AutoGen multi-agent systems,
including conversation-level tracking, agent monitoring, and multi-provider cost aggregation.

Usage:
    from genops.providers.autogen import GenOpsAutoGenAdapter

    adapter = GenOpsAutoGenAdapter(
        team="ai-research",
        project="multi-agent-conversations",
        daily_budget_limit=100.0
    )

    # Track conversation between agents
    with adapter.track_conversation("user-assistant-chat") as context:
        response = assistant.generate_reply(messages=conversation_history)
        print(f"Total cost: ${context.total_cost:.6f}")

Features:
    - End-to-end conversation governance and cost tracking
    - Agent-level instrumentation and interaction monitoring
    - Multi-provider cost aggregation (OpenAI, Anthropic, etc.)
    - Group chat orchestration tracking with participant analysis
    - Code execution monitoring for AutoGen's code interpreter
    - Function calling telemetry for tool usage patterns
    - Enterprise compliance patterns and multi-tenant governance
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, Any

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from genops.providers.autogen.conversation_monitor import AutoGenConversationMonitor
    from genops.providers.autogen.cost_aggregator import AutoGenCostAggregator

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# GenOps core imports
from genops.providers.base.provider import BaseFrameworkProvider

logger = logging.getLogger(__name__)


# Data classes for AutoGen-specific results and metrics
@dataclass
class AutoGenConversationResult:
    """Result from an AutoGen conversation tracking operation."""

    conversation_id: str
    start_time: datetime
    end_time: datetime
    total_cost: Decimal
    turns_count: int
    participants: list[str]
    total_tokens: int
    code_executions: int = 0
    function_calls: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoGenAgentResult:
    """Result from tracking a specific agent's interactions."""

    agent_name: str
    role: str
    messages_sent: int
    messages_received: int
    total_cost: Decimal
    response_time_ms: float
    tokens_used: int
    function_calls_made: int = 0
    code_executions_initiated: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class AutoGenGroupChatResult:
    """Result from tracking a group chat session."""

    group_chat_id: str
    start_time: datetime
    end_time: datetime
    total_cost: Decimal
    participants: list[str]
    message_count: int
    speaker_transitions: int
    total_tokens: int
    coordination_overhead_ms: float
    parallel_efficiency: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoGenSessionContext:
    """Context for an AutoGen session with cost and governance tracking."""

    session_id: str
    team: str
    project: str
    environment: str
    governance_policy: str
    budget_limit: Decimal
    current_cost: Decimal = Decimal("0.0")
    start_time: datetime = field(default_factory=datetime.now)
    conversations: list[AutoGenConversationResult] = field(default_factory=list)
    active_agents: set[str] = field(default_factory=set)


class AutoGenConversationContext:
    """Context manager for tracking AutoGen conversation flows."""

    def __init__(
        self,
        adapter: "GenOpsAutoGenAdapter",
        conversation_id: str,
        participants: list[str],
        governance_attrs: dict[str, Any],
    ):
        self.adapter = adapter
        self.conversation_id = conversation_id
        self.participants = participants
        self.governance_attrs = governance_attrs
        self.start_time = datetime.now()
        self.span = None
        self.total_cost = Decimal("0.0")
        self.turns_count = 0
        self.total_tokens = 0
        self.code_executions = 0
        self.function_calls = 0
        self.errors = []
        self._active = False

    def __enter__(self):
        """Start conversation tracking with telemetry."""
        self._active = True

        # Create OpenTelemetry span for conversation
        tracer = trace.get_tracer(__name__)
        self.span = tracer.start_span(f"autogen.conversation.{self.conversation_id}")

        # Set span attributes
        if self.span:
            self.span.set_attributes(
                {
                    "genops.framework": "autogen",
                    "genops.operation": "conversation",
                    "genops.conversation.id": self.conversation_id,
                    "genops.conversation.participants": ",".join(self.participants),
                    "genops.conversation.start_time": self.start_time.isoformat(),
                    **{f"genops.{k}": str(v) for k, v in self.governance_attrs.items()},
                }
            )

        logger.info(f"Starting AutoGen conversation tracking: {self.conversation_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete conversation tracking and export telemetry."""
        if not self._active:
            return

        end_time = datetime.now()
        duration_seconds = (end_time - self.start_time).total_seconds()

        # Update span with final metrics
        if self.span:
            self.span.set_attributes(
                {
                    "genops.conversation.end_time": end_time.isoformat(),
                    "genops.conversation.duration_seconds": duration_seconds,
                    "genops.conversation.turns_count": self.turns_count,
                    "genops.conversation.total_cost": str(self.total_cost),
                    "genops.conversation.total_tokens": self.total_tokens,
                    "genops.conversation.code_executions": self.code_executions,
                    "genops.conversation.function_calls": self.function_calls,
                    "genops.conversation.errors_count": len(self.errors),
                }
            )

            if exc_type:
                self.span.record_exception(exc_val)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self.span.set_status(Status(StatusCode.OK))

            self.span.end()

        # Create conversation result
        result = AutoGenConversationResult(
            conversation_id=self.conversation_id,
            start_time=self.start_time,
            end_time=end_time,
            total_cost=self.total_cost,
            turns_count=self.turns_count,
            participants=self.participants,
            total_tokens=self.total_tokens,
            code_executions=self.code_executions,
            function_calls=self.function_calls,
            errors=self.errors,
        )

        # Update adapter with results
        if hasattr(self.adapter, "session_context") and self.adapter.session_context:
            self.adapter.session_context.conversations.append(result)
            self.adapter.session_context.current_cost += self.total_cost

        logger.info(
            f"Completed AutoGen conversation tracking: {self.conversation_id} "
            f"(${self.total_cost:.6f}, {self.turns_count} turns, {duration_seconds:.1f}s)"
        )

        self._active = False

    def add_turn(self, cost: Decimal, tokens: int, agent_name: str = None):  # type: ignore[assignment]
        """Add a conversation turn with associated costs."""
        if self._active:
            self.turns_count += 1
            self.total_cost += cost
            self.total_tokens += tokens

            if agent_name:
                if (
                    hasattr(self.adapter, "session_context")
                    and self.adapter.session_context
                ):
                    self.adapter.session_context.active_agents.add(agent_name)

    def add_code_execution(self, cost: Decimal = Decimal("0.0")):
        """Record a code execution event."""
        if self._active:
            self.code_executions += 1
            self.total_cost += cost

    def add_function_call(self, cost: Decimal = Decimal("0.0")):
        """Record a function call event."""
        if self._active:
            self.function_calls += 1
            self.total_cost += cost

    def add_error(self, error_msg: str):
        """Record an error during conversation."""
        if self._active:
            self.errors.append(error_msg)


class GenOpsAutoGenAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for AutoGen multi-agent conversation systems.

    Provides comprehensive governance telemetry including conversation tracking,
    agent interaction monitoring, cost aggregation, and compliance reporting.
    """

    def __init__(
        self,
        team: str = "default-team",
        project: str = "autogen-app",
        environment: str = "development",
        daily_budget_limit: float = 100.0,
        governance_policy: str = "advisory",
        enable_conversation_tracking: bool = True,
        enable_agent_tracking: bool = True,
        enable_cost_tracking: bool = True,
        **kwargs,
    ):
        """
        Initialize AutoGen adapter with governance configuration.

        Args:
            team: Team name for cost attribution
            project: Project name for cost attribution
            environment: Environment (development, staging, production)
            daily_budget_limit: Daily spending limit in USD
            governance_policy: Policy enforcement level ("advisory", "enforced")
            enable_conversation_tracking: Enable conversation flow tracking
            enable_agent_tracking: Enable individual agent monitoring
            enable_cost_tracking: Enable cost aggregation across providers
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.team = team
        self.project = project
        self.environment = environment
        self.daily_budget_limit = Decimal(str(daily_budget_limit))
        self.governance_policy = governance_policy

        # Feature flags
        self.enable_conversation_tracking = enable_conversation_tracking
        self.enable_agent_tracking = enable_agent_tracking
        self.enable_cost_tracking = enable_cost_tracking

        # Session context
        self.session_context = AutoGenSessionContext(
            session_id=str(uuid.uuid4()),
            team=team,
            project=project,
            environment=environment,
            governance_policy=governance_policy,
            budget_limit=self.daily_budget_limit,
        )

        # Lazy-loaded components (initialized on first use)
        self._cost_aggregator: AutoGenCostAggregator | None = None
        self._conversation_monitor: AutoGenConversationMonitor | None = None

        # AutoGen detection
        self._autogen_available = self._check_autogen_availability()

        logger.info(
            f"Initialized GenOps AutoGen adapter - "
            f"Team: {team}, Project: {project}, Budget: ${daily_budget_limit}"
        )

    def setup_governance_attributes(self) -> None:
        """Setup AutoGen-specific governance attributes."""
        self.REQUEST_ATTRIBUTES.update(
            {
                "conversation_id",
                "agent_name",
                "agent_role",
                "group_chat_id",
                "message_type",
                "code_execution",
                "function_call",
                "turn_number",
            }
        )

    def _check_autogen_availability(self) -> bool:
        """Check if AutoGen is available in the environment."""
        try:
            import autogen  # noqa: F401

            return True
        except ImportError:
            logger.warning("AutoGen not available - limited functionality")
            return False

    @property
    def cost_aggregator(self) -> "AutoGenCostAggregator":
        """Lazy-loaded cost aggregator."""
        if self._cost_aggregator is None and self.enable_cost_tracking:
            from genops.providers.autogen.cost_aggregator import AutoGenCostAggregator

            self._cost_aggregator = AutoGenCostAggregator(
                team=self.team,
                project=self.project,
                daily_budget_limit=float(self.daily_budget_limit),
            )
        return self._cost_aggregator  # type: ignore

    @property
    def conversation_monitor(self) -> "AutoGenConversationMonitor":
        """Lazy-loaded conversation monitor."""
        if self._conversation_monitor is None and self.enable_conversation_tracking:
            from genops.providers.autogen.conversation_monitor import (
                AutoGenConversationMonitor,
            )

            self._conversation_monitor = AutoGenConversationMonitor(
                team=self.team, project=self.project
            )
        return self._conversation_monitor  # type: ignore

    @contextmanager  # type: ignore
    def track_conversation(
        self,
        conversation_id: str,
        participants: list[str] | None = None,
        **governance_attrs,
    ) -> AutoGenConversationContext:
        """
        Context manager for tracking AutoGen conversation flows.

        Args:
            conversation_id: Unique identifier for the conversation
            participants: List of agent names participating
            **governance_attrs: Additional governance attributes

        Yields:
            AutoGenConversationContext: Context for tracking conversation

        Example:
            with adapter.track_conversation("user-assistant", ["user", "assistant"]) as context:
                response = assistant.generate_reply(messages=history)
                context.add_turn(Decimal('0.002'), 150, "assistant")
        """
        if not self.enable_conversation_tracking:
            # Return a minimal context if tracking is disabled
            from contextlib import nullcontext

            yield nullcontext()
            return

        # Merge governance attributes
        attrs = {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            **governance_attrs,
        }

        participants = participants or ["agent"]
        context = AutoGenConversationContext(
            adapter=self,
            conversation_id=conversation_id,
            participants=participants,
            governance_attrs=attrs,
        )

        try:
            yield context
        finally:
            # Context manager handles cleanup
            pass

    @contextmanager  # type: ignore
    def track_group_chat(
        self,
        group_chat_id: str,
        participants: list[str] | None = None,
        **governance_attrs,
    ) -> AutoGenConversationContext:
        """
        Context manager for tracking AutoGen group chat sessions.

        Args:
            group_chat_id: Unique identifier for the group chat
            participants: List of agent names in the group
            **governance_attrs: Additional governance attributes

        Yields:
            AutoGenConversationContext: Context for tracking group chat

        Example:
            with adapter.track_group_chat("research-team", ["analyst", "critic", "summarizer"]) as context:
                result = group_chat_manager.run_chat(messages)
                context.add_turn(Decimal('0.005'), 300, "analyst")
        """
        # Use the same context manager but with group chat semantics
        with self.track_conversation(
            conversation_id=f"group_chat_{group_chat_id}",
            participants=participants or ["group_member"],
            group_chat_id=group_chat_id,
            **governance_attrs,
        ) as context:
            yield context

    def instrument_agent(self, agent, agent_name: str = None) -> Any:  # type: ignore[assignment]
        """
        Instrument an AutoGen agent for governance tracking.

        Args:
            agent: AutoGen agent instance to instrument
            agent_name: Optional name for the agent

        Returns:
            Instrumented agent with governance telemetry

        Example:
            assistant = autogen.AssistantAgent(name="assistant")
            assistant = adapter.instrument_agent(assistant, "coding_assistant")
        """
        if not self._autogen_available or not self.enable_agent_tracking:
            return agent

        agent_name = agent_name or getattr(agent, "name", "unknown_agent")

        # Wrap agent's generate_reply method if it exists
        if hasattr(agent, "generate_reply"):
            original_generate_reply = agent.generate_reply

            @wraps(original_generate_reply)
            def instrumented_generate_reply(*args, **kwargs):
                start_time = time.time()

                # Create telemetry span
                tracer = trace.get_tracer(__name__)
                with tracer.start_span(
                    f"autogen.agent.{agent_name}.generate_reply"
                ) as span:
                    span.set_attributes(
                        {
                            "genops.framework": "autogen",
                            "genops.operation": "agent_reply",
                            "genops.agent.name": agent_name,
                            "genops.team": self.team,
                            "genops.project": self.project,
                            "genops.environment": self.environment,
                        }
                    )

                    try:
                        result = original_generate_reply(*args, **kwargs)

                        # Calculate response time
                        response_time_ms = (time.time() - start_time) * 1000
                        span.set_attribute(
                            "genops.agent.response_time_ms", response_time_ms
                        )

                        # Estimate tokens (simplified - could be enhanced with actual counting)
                        if isinstance(result, str):
                            estimated_tokens = (
                                len(result.split()) * 1.3
                            )  # Rough estimation
                            span.set_attribute(
                                "genops.agent.estimated_tokens", int(estimated_tokens)
                            )

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            agent.generate_reply = instrumented_generate_reply

        logger.info(f"Instrumented AutoGen agent: {agent_name}")
        return agent

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current session's activities and costs.

        Returns:
            Dictionary with session metrics and summaries
        """
        if not self.session_context:
            return {"error": "No active session"}

        total_conversations = len(self.session_context.conversations)
        total_cost = float(self.session_context.current_cost)
        total_turns = sum(
            conv.turns_count for conv in self.session_context.conversations
        )
        total_agents = len(self.session_context.active_agents)

        return {
            "session_id": self.session_context.session_id,
            "team": self.session_context.team,
            "project": self.session_context.project,
            "environment": self.session_context.environment,
            "total_conversations": total_conversations,
            "total_cost": total_cost,
            "budget_limit": float(self.session_context.budget_limit),
            "budget_utilization": (
                total_cost / float(self.session_context.budget_limit)
            )
            * 100,
            "total_turns": total_turns,
            "unique_agents": total_agents,
            "active_agents": list(self.session_context.active_agents),
            "session_duration": (
                datetime.now() - self.session_context.start_time
            ).total_seconds(),
            "avg_cost_per_conversation": total_cost / max(total_conversations, 1),
            "avg_cost_per_turn": total_cost / max(total_turns, 1),
        }

    def reset_session(self):
        """Reset the session context for a new tracking session."""
        self.session_context = AutoGenSessionContext(
            session_id=str(uuid.uuid4()),
            team=self.team,
            project=self.project,
            environment=self.environment,
            governance_policy=self.governance_policy,
            budget_limit=self.daily_budget_limit,
        )
        logger.info(f"Reset AutoGen session: {self.session_context.session_id}")

    def validate_budget(self, additional_cost: Decimal) -> bool:
        """
        Validate if an additional cost would exceed the budget limit.

        Args:
            additional_cost: Cost to validate against budget

        Returns:
            True if within budget, False if would exceed
        """
        if not self.session_context:
            return True

        projected_cost = self.session_context.current_cost + additional_cost
        return projected_cost <= self.session_context.budget_limit

    def __repr__(self):
        return (
            f"GenOpsAutoGenAdapter(team='{self.team}', project='{self.project}', "
            f"environment='{self.environment}', budget_limit=${self.daily_budget_limit})"
        )
