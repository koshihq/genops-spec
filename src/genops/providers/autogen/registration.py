#!/usr/bin/env python3
"""
AutoGen Auto-Instrumentation Registration for GenOps Governance

Automatic detection and instrumentation of AutoGen components for zero-code
governance integration with comprehensive telemetry and cost tracking.

Usage:
    from genops.providers.autogen import auto_instrument

    # Zero-code setup - automatically detects and instruments AutoGen
    auto_instrument(team="ai-research", project="multi-agent-system")

    # Your existing AutoGen code works unchanged
    assistant = autogen.AssistantAgent(name="assistant")
    user_proxy = autogen.UserProxyAgent(name="user")
    # ↑ These are now automatically instrumented with governance telemetry

Features:
    - Zero-code auto-instrumentation for existing AutoGen applications
    - Automatic detection of AutoGen agents and group chats
    - Dynamic monkey-patching of core AutoGen methods
    - Conversation flow tracking with cost attribution
    - Agent interaction monitoring and performance analysis
    - Global instrumentation state management
    - Temporary instrumentation contexts for testing
"""

import logging
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

# GenOps imports
from genops.providers.autogen.adapter import GenOpsAutoGenAdapter
from genops.providers.autogen.conversation_monitor import AutoGenConversationMonitor

logger = logging.getLogger(__name__)

# Global instrumentation state
_instrumentation_state = {
    "enabled": False,
    "adapter": None,
    "monitor": None,
    "cost_aggregator": None,
    "instrumented_classes": set(),
    "instrumented_instances": weakref.WeakSet(),
    "config": {},
    "stats": {
        "agents_instrumented": 0,
        "conversations_tracked": 0,
        "total_cost_tracked": 0.0,
        "start_time": None,
    },
}
_state_lock = threading.RLock()


@dataclass
class InstrumentationConfig:
    """Configuration for AutoGen auto-instrumentation."""

    team: str = "default-team"
    project: str = "autogen-app"
    environment: str = "development"
    daily_budget_limit: float = 100.0
    governance_policy: str = "advisory"
    enable_conversation_tracking: bool = True
    enable_agent_tracking: bool = True
    enable_cost_tracking: bool = True
    enable_code_execution_tracking: bool = True
    enable_function_call_tracking: bool = True
    auto_detect_group_chats: bool = True
    conversation_timeout_seconds: int = 3600
    max_concurrent_conversations: int = 100


class TemporaryInstrumentation:
    """Context manager for temporary AutoGen instrumentation."""

    def __init__(self, **config):
        self.config = InstrumentationConfig(**config)
        self.was_enabled = False
        self.previous_adapter = None

    def __enter__(self):
        """Enable temporary instrumentation."""
        with _state_lock:
            self.was_enabled = _instrumentation_state["enabled"]
            self.previous_adapter = _instrumentation_state.get("adapter")

            if not self.was_enabled:
                auto_instrument(
                    team=self.config.team,
                    project=self.config.project,
                    environment=self.config.environment,
                    daily_budget_limit=self.config.daily_budget_limit,
                    governance_policy=self.config.governance_policy,
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous instrumentation state."""
        if not self.was_enabled:
            disable_auto_instrumentation()
        elif self.previous_adapter:
            # Restore previous adapter if one existed
            _instrumentation_state["adapter"] = self.previous_adapter


def auto_instrument(
    team: str = "default-team",
    project: str = "autogen-app",
    environment: str = "development",
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory",
    **kwargs,
) -> GenOpsAutoGenAdapter:
    """
    Automatically instrument AutoGen for governance tracking.

    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        environment: Environment (development, staging, production)
        daily_budget_limit: Daily spending limit in USD
        governance_policy: Policy enforcement level ("advisory", "enforced")
        **kwargs: Additional configuration options

    Returns:
        GenOpsAutoGenAdapter: Configured adapter instance

    Example:
        from genops.providers.autogen import auto_instrument

        # Zero-code setup
        adapter = auto_instrument(
            team="ai-research",
            project="customer-service",
            daily_budget_limit=50.0
        )

        # Existing AutoGen code now has governance telemetry
        assistant = autogen.AssistantAgent(name="assistant")
        user_proxy = autogen.UserProxyAgent(name="user")
        user_proxy.initiate_chat(assistant, message="Hello!")
    """
    with _state_lock:
        if _instrumentation_state["enabled"]:
            logger.warning("AutoGen auto-instrumentation already enabled")
            return _instrumentation_state["adapter"]  # type: ignore

        # Check AutoGen availability
        if not _check_autogen_availability():
            logger.error("AutoGen not available for instrumentation")
            return None  # type: ignore[return-value]

        # Create adapter and components
        adapter = GenOpsAutoGenAdapter(
            team=team,
            project=project,
            environment=environment,
            daily_budget_limit=daily_budget_limit,
            governance_policy=governance_policy,
            **kwargs,
        )

        # Store configuration
        config = InstrumentationConfig(
            team=team,
            project=project,
            environment=environment,
            daily_budget_limit=daily_budget_limit,
            governance_policy=governance_policy,
            **kwargs,
        )

        # Update global state
        _instrumentation_state.update(
            {
                "enabled": True,
                "adapter": adapter,
                "monitor": adapter.conversation_monitor,
                "cost_aggregator": adapter.cost_aggregator,
                "config": config,
                "stats": {
                    **_instrumentation_state["stats"],
                    "start_time": datetime.now(),
                },
            }
        )

        # Perform instrumentation
        _instrument_autogen_classes()

        logger.info(
            f"AutoGen auto-instrumentation enabled - "
            f"Team: {team}, Project: {project}, Budget: ${daily_budget_limit}"
        )

        return adapter


def disable_auto_instrumentation():
    """
    Disable AutoGen auto-instrumentation and restore original behavior.
    """
    with _state_lock:
        if not _instrumentation_state["enabled"]:
            logger.warning("AutoGen auto-instrumentation already disabled")
            return

        # Restore original methods
        _restore_autogen_classes()

        # Clear global state
        _instrumentation_state.update(
            {
                "enabled": False,
                "adapter": None,
                "monitor": None,
                "cost_aggregator": None,
                "instrumented_classes": set(),
                "config": {},
            }
        )
        _instrumentation_state["instrumented_instances"].clear()

        logger.info("AutoGen auto-instrumentation disabled")


def configure_auto_instrumentation(**config_updates):
    """
    Update auto-instrumentation configuration.

    Args:
        **config_updates: Configuration updates to apply
    """
    with _state_lock:
        if not _instrumentation_state["enabled"]:
            logger.warning("AutoGen auto-instrumentation not enabled")
            return

        # Update configuration
        current_config = _instrumentation_state["config"]
        for key, value in config_updates.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
                logger.info(f"Updated instrumentation config: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")


def is_instrumented() -> bool:
    """
    Check if AutoGen auto-instrumentation is currently enabled.

    Returns:
        bool: True if instrumentation is enabled
    """
    with _state_lock:
        return _instrumentation_state["enabled"]  # type: ignore


def get_current_adapter() -> Optional[GenOpsAutoGenAdapter]:
    """
    Get the current AutoGen adapter instance.

    Returns:
        GenOpsAutoGenAdapter or None: Current adapter if instrumentation enabled
    """
    with _state_lock:
        return _instrumentation_state.get("adapter")  # type: ignore


def get_current_monitor() -> Optional[AutoGenConversationMonitor]:
    """
    Get the current conversation monitor instance.

    Returns:
        AutoGenConversationMonitor or None: Current monitor if enabled
    """
    with _state_lock:
        return _instrumentation_state.get("monitor")  # type: ignore


def get_cost_summary() -> dict[str, Any]:
    """
    Get cost summary from the current cost aggregator.

    Returns:
        Dictionary with cost summary or error message
    """
    with _state_lock:
        cost_aggregator = _instrumentation_state.get("cost_aggregator")
        if not cost_aggregator:
            return {"error": "Cost aggregator not available"}

        # Get cost analysis for the last 24 hours
        try:
            analysis = cost_aggregator.get_cost_analysis(time_period_hours=24)
            return {
                "total_cost": str(analysis.total_cost),
                "cost_by_provider": {
                    k.value: str(v) for k, v in analysis.cost_by_provider.items()
                },
                "cost_by_agent": {k: str(v) for k, v in analysis.cost_by_agent.items()},
                "optimization_recommendations": len(
                    analysis.optimization_recommendations
                ),
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {"error": str(e)}


def get_conversation_metrics() -> dict[str, Any]:
    """
    Get conversation metrics from the current monitor.

    Returns:
        Dictionary with conversation metrics or error message
    """
    with _state_lock:
        monitor = _instrumentation_state.get("monitor")
        if not monitor:
            return {"error": "Conversation monitor not available"}

        try:
            return monitor.get_conversation_summary(time_period_hours=24)
        except Exception as e:
            logger.error(f"Error getting conversation metrics: {e}")
            return {"error": str(e)}


def get_instrumentation_stats() -> dict[str, Any]:
    """
    Get statistics about the current instrumentation state.

    Returns:
        Dictionary with instrumentation statistics
    """
    with _state_lock:
        stats = _instrumentation_state["stats"].copy()
        if stats["start_time"]:
            uptime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["uptime_seconds"] = uptime

        return {
            "enabled": _instrumentation_state["enabled"],
            "instrumented_classes": len(_instrumentation_state["instrumented_classes"]),  # type: ignore
            "instrumented_instances": len(
                _instrumentation_state["instrumented_instances"]  # type: ignore
            ),
            "stats": stats,
            "config": _instrumentation_state.get("config", {}).__dict__
            if _instrumentation_state.get("config")
            else {},
        }


def _check_autogen_availability() -> bool:
    """Check if AutoGen is available for instrumentation."""
    try:
        import autogen  # noqa: F401

        return True
    except ImportError:
        logger.warning("AutoGen not available for instrumentation")
        return False


def _instrument_autogen_classes():
    """Instrument core AutoGen classes with governance telemetry."""
    try:
        import autogen

        # Get adapter from global state
        adapter = _instrumentation_state["adapter"]
        if not adapter:
            logger.error("No adapter available for instrumentation")
            return

        # Instrument ConversableAgent (base class for all agents)
        if hasattr(autogen, "ConversableAgent"):
            _instrument_conversable_agent(autogen.ConversableAgent, adapter)
            _instrumentation_state["instrumented_classes"].add("ConversableAgent")

        # Instrument GroupChatManager
        if hasattr(autogen, "GroupChatManager"):
            _instrument_group_chat_manager(autogen.GroupChatManager, adapter)
            _instrumentation_state["instrumented_classes"].add("GroupChatManager")

        # Instrument GroupChat
        if hasattr(autogen, "GroupChat"):
            _instrument_group_chat(autogen.GroupChat, adapter)
            _instrumentation_state["instrumented_classes"].add("GroupChat")

        logger.info(
            f"Instrumented {len(_instrumentation_state['instrumented_classes'])} AutoGen classes"
        )

    except Exception as e:
        logger.error(f"Error instrumenting AutoGen classes: {e}")


def _instrument_conversable_agent(agent_class, adapter: GenOpsAutoGenAdapter):
    """Instrument ConversableAgent class methods."""
    # Store original methods
    if not hasattr(agent_class, "_genops_original_generate_reply"):
        agent_class._genops_original_generate_reply = agent_class.generate_reply
        agent_class._genops_original_send = agent_class.send
        agent_class._genops_original_receive = agent_class.receive

    def instrumented_generate_reply(self, messages=None, sender=None, **kwargs):
        """Instrumented generate_reply with telemetry."""
        agent_name = getattr(self, "name", "unknown_agent")
        start_time = datetime.now()

        # Create conversation context if needed
        conversation_id = f"{agent_name}_{sender.name if sender else 'unknown'}_{int(start_time.timestamp())}"

        try:
            # Call original method
            result = agent_class._genops_original_generate_reply(
                self, messages, sender, **kwargs
            )

            # Track the interaction
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Estimate tokens (simplified)
            if isinstance(result, str):
                estimated_tokens = len(result.split()) * 1.3

                # Add to cost aggregator if available
                if adapter.cost_aggregator:
                    adapter.cost_aggregator.add_agent_interaction(
                        agent_name=agent_name,
                        provider="openai",  # Default assumption
                        model="gpt-3.5-turbo",  # Default assumption
                        input_tokens=int(estimated_tokens * 0.3),
                        output_tokens=int(estimated_tokens * 0.7),
                        conversation_id=conversation_id,
                        metadata={"response_time_ms": response_time_ms},
                    )

                # Update stats
                _instrumentation_state["stats"]["agents_instrumented"] += 1

            return result

        except Exception as e:
            logger.error(f"Error in instrumented generate_reply: {e}")
            # Fall back to original method
            return agent_class._genops_original_generate_reply(
                self, messages, sender, **kwargs
            )

    def instrumented_send(self, message, recipient, **kwargs):
        """Instrumented send with telemetry."""
        agent_name = getattr(self, "name", "unknown_agent")
        recipient_name = getattr(recipient, "name", "unknown_recipient")

        # Track message sending
        logger.debug(f"Agent {agent_name} sending message to {recipient_name}")

        # Call original method
        return agent_class._genops_original_send(self, message, recipient, **kwargs)

    def instrumented_receive(self, message, sender, **kwargs):
        """Instrumented receive with telemetry."""
        agent_name = getattr(self, "name", "unknown_agent")
        sender_name = getattr(sender, "name", "unknown_sender")

        # Track message receiving
        logger.debug(f"Agent {agent_name} receiving message from {sender_name}")

        # Call original method
        return agent_class._genops_original_receive(self, message, sender, **kwargs)

    # Apply instrumentation
    agent_class.generate_reply = instrumented_generate_reply
    agent_class.send = instrumented_send
    agent_class.receive = instrumented_receive


def _instrument_group_chat_manager(manager_class, adapter: GenOpsAutoGenAdapter):
    """Instrument GroupChatManager class methods."""
    if not hasattr(manager_class, "_genops_original_run_chat"):
        manager_class._genops_original_run_chat = manager_class.run_chat

    def instrumented_run_chat(self, messages=None, **kwargs):
        """Instrumented run_chat with group conversation tracking."""
        group_chat_id = f"group_chat_{int(datetime.now().timestamp())}"

        # Track group chat session
        if adapter.conversation_monitor:
            participants = (
                [agent.name for agent in self.groupchat.agents]
                if hasattr(self, "groupchat")
                else []
            )

            with adapter.track_group_chat(group_chat_id, participants) as context:
                try:
                    result = manager_class._genops_original_run_chat(
                        self, messages, **kwargs
                    )

                    # Update conversation stats
                    _instrumentation_state["stats"]["conversations_tracked"] += 1

                    return result

                except Exception as e:
                    context.add_error(str(e))
                    raise
        else:
            # Fall back to original method
            return manager_class._genops_original_run_chat(self, messages, **kwargs)

    # Apply instrumentation
    manager_class.run_chat = instrumented_run_chat


def _instrument_group_chat(group_chat_class, adapter: GenOpsAutoGenAdapter):
    """Instrument GroupChat class methods."""
    if not hasattr(group_chat_class, "_genops_original_init"):
        group_chat_class._genops_original_init = group_chat_class.__init__

    def instrumented_init(self, agents, **kwargs):
        """Instrumented __init__ to track group chat creation."""
        result = group_chat_class._genops_original_init(self, agents, **kwargs)

        # Track group chat creation
        agent_names = [getattr(agent, "name", "unknown") for agent in agents]
        logger.info(f"Created AutoGen group chat with agents: {agent_names}")

        return result

    # Apply instrumentation
    group_chat_class.__init__ = instrumented_init


def _restore_autogen_classes():
    """Restore original AutoGen class methods."""
    try:
        import autogen

        # Restore ConversableAgent
        if hasattr(autogen, "ConversableAgent") and hasattr(
            autogen.ConversableAgent, "_genops_original_generate_reply"
        ):
            autogen.ConversableAgent.generate_reply = (
                autogen.ConversableAgent._genops_original_generate_reply
            )
            autogen.ConversableAgent.send = (
                autogen.ConversableAgent._genops_original_send
            )
            autogen.ConversableAgent.receive = (
                autogen.ConversableAgent._genops_original_receive
            )

            delattr(autogen.ConversableAgent, "_genops_original_generate_reply")
            delattr(autogen.ConversableAgent, "_genops_original_send")
            delattr(autogen.ConversableAgent, "_genops_original_receive")

        # Restore GroupChatManager
        if hasattr(autogen, "GroupChatManager") and hasattr(
            autogen.GroupChatManager, "_genops_original_run_chat"
        ):
            autogen.GroupChatManager.run_chat = (
                autogen.GroupChatManager._genops_original_run_chat
            )
            delattr(autogen.GroupChatManager, "_genops_original_run_chat")

        # Restore GroupChat
        if hasattr(autogen, "GroupChat") and hasattr(
            autogen.GroupChat, "_genops_original_init"
        ):
            autogen.GroupChat.__init__ = autogen.GroupChat._genops_original_init
            delattr(autogen.GroupChat, "_genops_original_init")

        logger.info("Restored original AutoGen class methods")

    except Exception as e:
        logger.error(f"Error restoring AutoGen classes: {e}")


# Context manager for temporary instrumentation
@contextmanager
def temporary_instrumentation(**config):
    """
    Context manager for temporary AutoGen instrumentation.

    Args:
        **config: Configuration for temporary instrumentation

    Example:
        with temporary_instrumentation(team="test-team", project="test"):
            assistant = autogen.AssistantAgent(name="assistant")
            # ↑ This agent is now instrumented
        # ↑ Instrumentation is removed here
    """
    temp = TemporaryInstrumentation(**config)
    try:
        yield temp.__enter__()
    finally:
        temp.__exit__(None, None, None)
