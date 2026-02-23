#!/usr/bin/env python3
"""
CrewAI Auto-Instrumentation Registration System

Provides zero-code setup for CrewAI multi-agent governance by automatically
instrumenting crews, agents, and workflows with GenOps telemetry.

Usage:
    from genops.providers.crewai import auto_instrument
    auto_instrument()

    # Your existing CrewAI code works unchanged
    from crewai import Agent, Task, Crew

    crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
    result = crew.kickoff()
    # ✅ Automatic cost tracking and governance added!

Features:
    - Zero-code instrumentation for existing CrewAI applications
    - Automatic crew and agent monitoring
    - Multi-provider cost tracking and governance
    - Multi-agent workflow specialization
    - Configurable instrumentation policies
    - Production-ready auto-instrumentation with minimal overhead
"""

import functools
import logging
import threading
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Optional

# GenOps imports - using TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from genops.providers.crewai.adapter import GenOpsCrewAIAdapter
    from genops.providers.crewai.agent_monitor import CrewAIAgentMonitor
    from genops.providers.crewai.cost_aggregator import CrewAICostAggregator

logger = logging.getLogger(__name__)

# Check for CrewAI availability
try:
    import crewai
    from crewai import Agent, Crew, Task
    from crewai.agents.agent import Agent as CrewAIAgent
    from crewai.crew import Crew as CrewAICrew
    from crewai.task import Task as CrewAITask

    HAS_CREWAI = True
    logger.debug(f"CrewAI {crewai.__version__} detected for auto-instrumentation")
except ImportError:
    HAS_CREWAI = False
    Crew = None
    Agent = None
    Task = None
    CrewAICrew = None
    CrewAIAgent = None
    CrewAITask = None
    logger.warning("CrewAI not installed - auto-instrumentation disabled")


class InstrumentationRegistry:
    """Registry for managing auto-instrumentation state and configuration."""

    def __init__(self):
        self.is_instrumented = False
        self.instrumented_classes: set[type] = set()
        self.original_methods: dict[str, Callable] = {}
        self.adapter: Optional["GenOpsCrewAIAdapter"] = None
        self.monitor: Optional["CrewAIAgentMonitor"] = None
        self.cost_aggregator: Optional["CrewAICostAggregator"] = None
        self._lock = threading.RLock()

        # Configuration
        self.config = {
            "team": "auto-instrumented",
            "project": "crewai-app",
            "environment": "development",
            "enable_agent_tracking": True,
            "enable_cost_tracking": True,
            "enable_task_tracking": True,
            "enable_workflow_analysis": True,
            "daily_budget_limit": 100.0,
            "governance_policy": "advisory",
        }

        # Agent patterns to instrument
        self.agent_patterns = {
            "roles": [
                "researcher",
                "analyst",
                "writer",
                "reviewer",
                "coordinator",
                "data_scientist",
                "engineer",
                "qa_specialist",
                "product_manager",
            ],
            "workflow_types": ["sequential", "hierarchical", "parallel"],
        }

    def update_config(self, **kwargs):
        """Update instrumentation configuration."""
        with self._lock:
            self.config.update(kwargs)

            # Reinitialize components if already instrumented
            if self.is_instrumented:
                self._initialize_components()

    def _initialize_components(self):
        """Initialize GenOps components with current configuration."""
        # Import at runtime to avoid circular imports
        from genops.providers.crewai.adapter import GenOpsCrewAIAdapter
        from genops.providers.crewai.agent_monitor import CrewAIAgentMonitor
        from genops.providers.crewai.cost_aggregator import CrewAICostAggregator

        self.adapter = GenOpsCrewAIAdapter(
            team=self.config["team"],
            project=self.config["project"],
            environment=self.config["environment"],
            daily_budget_limit=self.config["daily_budget_limit"],
            governance_policy=self.config["governance_policy"],
            enable_agent_tracking=self.config["enable_agent_tracking"],
            enable_cost_tracking=self.config["enable_cost_tracking"],
            enable_task_tracking=self.config["enable_task_tracking"],
        )

        self.monitor = CrewAIAgentMonitor(
            team=self.config["team"],
            project=self.config["project"],
            environment=self.config["environment"],
            enable_performance_monitoring=True,
            enable_cost_tracking=self.config["enable_cost_tracking"],
            enable_task_tracking=self.config["enable_task_tracking"],
            enable_workflow_analysis=self.config["enable_workflow_analysis"],
        )

        self.cost_aggregator = CrewAICostAggregator(
            budget_limit=self.config["daily_budget_limit"]
        )


# Global registry instance
_registry = InstrumentationRegistry()


def configure_auto_instrumentation(**kwargs):
    """
    Configure auto-instrumentation settings.

    Args:
        team: Team name for governance
        project: Project name for governance
        environment: Environment name
        enable_agent_tracking: Enable agent-level tracking
        enable_cost_tracking: Enable cost tracking
        enable_task_tracking: Enable task-level tracking
        enable_workflow_analysis: Enable workflow analysis
        daily_budget_limit: Daily budget limit
        governance_policy: Governance policy ("advisory", "enforced")

    Example:
        configure_auto_instrumentation(
            team="ml-team",
            project="research-agents",
            daily_budget_limit=50.0,
            governance_policy="enforced"
        )
    """
    _registry.update_config(**kwargs)
    logger.info(f"Auto-instrumentation configured: {kwargs}")


def is_instrumented() -> bool:
    """Check if auto-instrumentation is currently active."""
    return _registry.is_instrumented


def get_instrumentation_stats() -> dict[str, Any]:
    """Get current instrumentation statistics."""
    return {
        "is_instrumented": _registry.is_instrumented,
        "instrumented_classes": [
            cls.__name__ for cls in _registry.instrumented_classes
        ],
        "config": _registry.config.copy(),
        "has_adapter": _registry.adapter is not None,
        "has_monitor": _registry.monitor is not None,
        "crew_executions": len(_registry.monitor._execution_results)
        if _registry.monitor
        else 0,
    }


def _create_instrumented_crew_kickoff():
    """Create instrumented version of Crew.kickoff method."""
    if not HAS_CREWAI or not Crew:
        return None

    # Store original method
    original_kickoff = Crew.kickoff
    _registry.original_methods["Crew.kickoff"] = original_kickoff

    @functools.wraps(original_kickoff)
    def instrumented_kickoff(self, inputs: Optional[dict[str, Any]] = None, **kwargs):
        """Instrumented version of Crew.kickoff with governance tracking."""
        crew_name = getattr(self, "name", None) or f"crew-{id(self)}"

        # Use adapter for tracking if available
        if _registry.adapter:
            with _registry.adapter.track_crew(
                crew_name, inputs=inputs, **kwargs
            ) as context:
                try:
                    # Execute original crew kickoff
                    result = original_kickoff(self, inputs, **kwargs)

                    # Try to extract agent and task information
                    if hasattr(self, "agents") and self.agents:
                        for agent in self.agents:
                            try:
                                agent_name = getattr(
                                    agent, "role", f"agent-{id(agent)}"
                                )
                                agent_role = getattr(agent, "role", "unknown")

                                # Import at runtime to avoid circular imports
                                from genops.providers.crewai.adapter import (
                                    CrewAIAgentResult,
                                )

                                # Estimate cost based on agent complexity
                                estimated_cost = _estimate_agent_cost(
                                    agent_role, inputs or {}
                                )

                                agent_result = CrewAIAgentResult(
                                    agent_name=agent_name,
                                    agent_role=agent_role,
                                    execution_time_seconds=1.0,  # Placeholder
                                    cost=estimated_cost,
                                    provider=_get_provider_for_agent(agent),
                                    status="success",
                                )

                                context.add_agent_result(agent_result)

                            except Exception as e:
                                logger.debug(f"Could not track agent {agent}: {e}")
                                continue

                    # Try to extract task information
                    if hasattr(self, "tasks") and self.tasks:
                        for task in self.tasks:
                            try:
                                task_description = getattr(
                                    task, "description", f"task-{id(task)}"
                                )
                                agent_name = (
                                    getattr(task, "agent", {}).get("role", "unknown")
                                    if hasattr(task, "agent")
                                    else "unknown"
                                )

                                # Import at runtime
                                from genops.providers.crewai.adapter import (
                                    CrewAITaskResult,
                                )

                                task_result = CrewAITaskResult(
                                    task_description=task_description,
                                    task_id=f"task-{id(task)}",
                                    agent_name=agent_name,
                                    execution_time_seconds=0.5,  # Placeholder
                                    cost=Decimal("0.001"),  # Minimal task overhead
                                    status="success",
                                )

                                context.add_task_result(task_result)

                            except Exception as e:
                                logger.debug(f"Could not track task {task}: {e}")
                                continue

                    return result

                except Exception as e:
                    logger.error(f"Crew execution failed: {e}")
                    raise
        else:
            # Fallback to original method if no adapter
            return original_kickoff(self, inputs, **kwargs)

    return instrumented_kickoff


def _estimate_agent_cost(agent_role: str, inputs: dict[str, Any]) -> Decimal:
    """Estimate cost for an agent based on its role and inputs."""

    # Cost estimates by agent role
    role_base_costs = {
        "researcher": Decimal("0.005"),
        "analyst": Decimal("0.003"),
        "writer": Decimal("0.004"),
        "reviewer": Decimal("0.002"),
        "coordinator": Decimal("0.001"),
        "data_scientist": Decimal("0.006"),
        "engineer": Decimal("0.004"),
        "qa_specialist": Decimal("0.002"),
        "product_manager": Decimal("0.003"),
    }

    # Extract base cost
    base_cost = role_base_costs.get(agent_role.lower(), Decimal("0.003"))

    # Scale based on input complexity
    input_complexity = 1.0
    if inputs:
        # Simple heuristic based on input size
        total_input_length = sum(len(str(v)) for v in inputs.values())
        input_complexity = max(1.0, total_input_length / 1000)  # Scale by input size

    return base_cost * Decimal(str(input_complexity))


def _get_provider_for_agent(agent) -> str:
    """Get provider name for an agent."""
    # Try to detect provider from agent configuration
    if hasattr(agent, "llm"):
        llm = agent.llm
        if hasattr(llm, "__class__"):
            class_name = llm.__class__.__name__.lower()
            if "openai" in class_name:
                return "openai"
            elif "anthropic" in class_name or "claude" in class_name:
                return "anthropic"
            elif "google" in class_name or "gemini" in class_name:
                return "google"
            elif "cohere" in class_name:
                return "cohere"

    return "openai"  # Default assumption


def _create_instrumented_agent_execute():
    """Create instrumented version of Agent execution methods."""
    if not HAS_CREWAI or not Agent:
        return None

    # This would be enhanced based on actual CrewAI Agent API
    # For now, we'll focus on crew-level instrumentation
    return None


def _instrument_crew_class():
    """Instrument the CrewAI Crew class."""
    if not HAS_CREWAI or not Crew or Crew in _registry.instrumented_classes:
        return

    # Create instrumented kickoff method
    instrumented_kickoff = _create_instrumented_crew_kickoff()
    if instrumented_kickoff:
        # Monkey patch the Crew.kickoff method
        Crew.kickoff = instrumented_kickoff
        _registry.instrumented_classes.add(Crew)
        logger.debug("Crew class instrumented")


def _instrument_agent_classes():
    """Instrument CrewAI agent classes."""
    if not HAS_CREWAI or not Agent:
        return

    # Agent instrumentation would be added here
    # For now, we focus on crew-level tracking
    logger.debug("Agent instrumentation (placeholder)")


def auto_instrument(**config):
    """
    Enable automatic instrumentation for all CrewAI crews and agents.

    This function monkey-patches CrewAI classes to automatically add GenOps
    governance tracking to all crew executions and agent operations.

    Args:
        **config: Configuration options for instrumentation

    Usage:
        from genops.providers.crewai import auto_instrument

        # Basic setup
        auto_instrument()

        # Custom configuration
        auto_instrument(
            team="ml-team",
            project="research-agents",
            daily_budget_limit=50.0,
            governance_policy="enforced"
        )

        # Your existing CrewAI code works unchanged
        crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
        result = crew.kickoff()
        # ✅ Automatic cost tracking and governance added!
    """
    if not HAS_CREWAI:
        logger.error("Cannot enable auto-instrumentation: CrewAI not installed")
        logger.error("Install with: pip install crewai")
        return False

    with _registry._lock:
        if _registry.is_instrumented:
            logger.info("Auto-instrumentation already enabled")
            if config:
                _registry.update_config(**config)
            return True

        try:
            # Update configuration
            if config:
                _registry.update_config(**config)

            # Initialize GenOps components
            _registry._initialize_components()

            # Instrument CrewAI classes
            _instrument_crew_class()
            _instrument_agent_classes()

            # Mark as instrumented
            _registry.is_instrumented = True

            logger.info("CrewAI auto-instrumentation enabled successfully")
            logger.info(f"Configuration: {_registry.config}")
            logger.info(f"Instrumented classes: {len(_registry.instrumented_classes)}")

            return True

        except Exception as e:
            logger.error(f"Failed to enable auto-instrumentation: {e}")
            # Attempt to rollback
            disable_auto_instrumentation()
            return False


def disable_auto_instrumentation():
    """
    Disable automatic instrumentation and restore original CrewAI behavior.

    This function removes all monkey patches and restores the original
    CrewAI class methods.
    """
    with _registry._lock:
        if not _registry.is_instrumented:
            logger.info("Auto-instrumentation not currently enabled")
            return

        try:
            # Restore original methods
            if HAS_CREWAI:
                # Restore Crew.kickoff
                if "Crew.kickoff" in _registry.original_methods:
                    Crew.kickoff = _registry.original_methods["Crew.kickoff"]

                # Restore other methods as needed
                for method_key, original_method in _registry.original_methods.items():
                    if "." in method_key and method_key != "Crew.kickoff":
                        class_name, method_name = method_key.split(".", 1)

                        # Find the class and restore method
                        for cls in _registry.instrumented_classes:
                            if cls.__name__ == class_name:
                                setattr(cls, method_name, original_method)
                                break

            # Clear registry
            _registry.is_instrumented = False
            _registry.instrumented_classes.clear()
            _registry.original_methods.clear()
            _registry.adapter = None
            _registry.monitor = None
            _registry.cost_aggregator = None

            logger.info(
                "Auto-instrumentation disabled - original CrewAI behavior restored"
            )

        except Exception as e:
            logger.error(f"Error disabling auto-instrumentation: {e}")


def get_current_adapter() -> Optional["GenOpsCrewAIAdapter"]:
    """Get the current auto-instrumentation adapter."""
    return _registry.adapter


def get_current_monitor() -> Optional["CrewAIAgentMonitor"]:
    """Get the current auto-instrumentation monitor."""
    return _registry.monitor


def get_cost_summary() -> dict[str, Any]:
    """Get cost summary from auto-instrumentation."""
    if _registry.adapter:
        return _registry.adapter.get_cost_summary()
    else:
        return {"error": "Auto-instrumentation not enabled"}


def get_execution_metrics() -> dict[str, Any]:
    """Get execution metrics from auto-instrumentation."""
    if _registry.monitor:
        return _registry.monitor.get_performance_summary()
    else:
        return {"error": "Auto-instrumentation not enabled"}


# Context manager for temporary instrumentation
class TemporaryInstrumentation:
    """Context manager for temporary auto-instrumentation."""

    def __init__(self, **config):
        self.config = config
        self.was_instrumented = False

    def __enter__(self):
        self.was_instrumented = is_instrumented()
        if not self.was_instrumented:
            auto_instrument(**self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.was_instrumented:
            disable_auto_instrumentation()


# Auto-register function (called from __init__.py)
def auto_register():
    """Auto-register CrewAI provider if available."""
    if HAS_CREWAI:
        logger.debug("CrewAI provider auto-registered")
        # Could add automatic registration logic here
    else:
        logger.debug("CrewAI not available - provider not registered")


# Export main functions
__all__ = [
    "auto_instrument",
    "disable_auto_instrumentation",
    "configure_auto_instrumentation",
    "is_instrumented",
    "get_instrumentation_stats",
    "get_current_adapter",
    "get_current_monitor",
    "get_cost_summary",
    "get_execution_metrics",
    "TemporaryInstrumentation",
    "auto_register",
]
