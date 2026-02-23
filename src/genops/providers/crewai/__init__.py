#!/usr/bin/env python3
"""
CrewAI Integration for GenOps Governance

Comprehensive integration for CrewAI multi-agent systems with GenOps governance,
providing end-to-end tracking for agent workflows, crew orchestration, and multi-provider cost management.

Usage:
    # Quick setup with auto-instrumentation
    from genops.providers.crewai import auto_instrument
    auto_instrument()

    # Manual setup with full control
    from genops.providers.crewai import GenOpsCrewAIAdapter
    adapter = GenOpsCrewAIAdapter(
        team="ai-research",
        project="multi-agent-system",
        daily_budget_limit=100.0
    )

    with adapter.track_crew("research-crew") as context:
        result = crew.kickoff()
        print(f"Total cost: ${context.total_cost:.6f}")

Features:
    - Zero-code auto-instrumentation for existing CrewAI applications
    - End-to-end crew governance and cost tracking
    - Multi-provider cost aggregation (OpenAI, Anthropic, Google, etc.)
    - Multi-agent workflow specialization with collaboration tracking
    - Task execution monitoring and performance analysis
    - Enterprise compliance patterns and multi-tenant governance
"""

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Define create_chain_cost_context at module level for CodeQL compliance
try:
    from genops.providers.crewai.cost_aggregator import create_chain_cost_context
except ImportError:

    def create_chain_cost_context(chain_id: str):
        """Fallback implementation if cost_aggregator is not available."""
        from genops.providers.crewai.cost_aggregator import (
            create_chain_cost_context as _real_func,
        )

        return _real_func(chain_id)


# Lazy import registry to avoid circular dependencies
_import_cache = {}


# Custom module type to handle lazy loading (applying Haystack lessons)
class LazyModule(type(sys.modules[__name__])):
    """Custom module type that handles lazy loading sentinels."""

    def __getattribute__(self, name):
        """Override attribute access to handle lazy loading sentinels."""
        # Get the attribute using the default behavior
        value = super().__getattribute__(name)

        # If it's a sentinel, perform the lazy loading
        if isinstance(value, _LazyImportSentinel):
            # Use the module's __getattr__ to get the actual value
            actual_value = self.__getattr__(name)
            # Update the module's dict to avoid repeated lazy loading
            setattr(self, name, actual_value)
            return actual_value

        return value


# Apply the custom module type to this module
sys.modules[__name__].__class__ = LazyModule


# Sentinel class for lazy-loaded symbols (satisfies static analysis while enabling lazy loading)
class _LazyImportSentinel:
    """Sentinel class indicating a symbol should be lazy-loaded."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<LazyImport: {self.name}>"


# Check for CrewAI availability
try:
    import crewai

    HAS_CREWAI = True
    logger.info(
        f"GenOps CrewAI integration loaded - CrewAI {crewai.__version__} detected"
    )
except ImportError:
    HAS_CREWAI = False
    logger.warning(
        "CrewAI not installed - integration available but limited functionality"
    )

# Version info
__version__ = "1.0.0"
__author__ = "GenOps AI"


# Callable class placeholders for instantiable classes
def GenOpsCrewAIAdapter(*args, **kwargs):
    """Lazy-loaded GenOpsCrewAIAdapter class."""
    real_class = __getattr__("GenOpsCrewAIAdapter")
    globals()["GenOpsCrewAIAdapter"] = real_class  # Replace placeholder
    return real_class(*args, **kwargs)


def CrewAIAgentMonitor(*args, **kwargs):
    """Lazy-loaded CrewAIAgentMonitor class."""
    real_class = __getattr__("CrewAIAgentMonitor")
    globals()["CrewAIAgentMonitor"] = real_class
    return real_class(*args, **kwargs)


def CrewAICostAggregator(*args, **kwargs):
    """Lazy-loaded CrewAICostAggregator class."""
    real_class = __getattr__("CrewAICostAggregator")
    globals()["CrewAICostAggregator"] = real_class
    return real_class(*args, **kwargs)


def TemporaryInstrumentation(*args, **kwargs):
    """Lazy-loaded TemporaryInstrumentation class."""
    real_class = __getattr__("TemporaryInstrumentation")
    globals()["TemporaryInstrumentation"] = real_class
    return real_class(*args, **kwargs)


# Data classes (sentinels - not instantiated directly)
CrewAIAgentResult = _LazyImportSentinel("CrewAIAgentResult")
CrewAITaskResult = _LazyImportSentinel("CrewAITaskResult")
CrewAICrewResult = _LazyImportSentinel("CrewAICrewResult")
CrewAISessionContext = _LazyImportSentinel("CrewAISessionContext")
AgentExecutionMetrics = _LazyImportSentinel("AgentExecutionMetrics")
TaskExecutionMetrics = _LazyImportSentinel("TaskExecutionMetrics")
CrewExecutionMetrics = _LazyImportSentinel("CrewExecutionMetrics")
MultiAgentWorkflowMetrics = _LazyImportSentinel("MultiAgentWorkflowMetrics")
AgentCostEntry = _LazyImportSentinel("AgentCostEntry")
CrewCostSummary = _LazyImportSentinel("CrewCostSummary")
ProviderCostSummary = _LazyImportSentinel("ProviderCostSummary")
CostOptimizationRecommendation = _LazyImportSentinel("CostOptimizationRecommendation")
CostAnalysisResult = _LazyImportSentinel("CostAnalysisResult")
ValidationResult = _LazyImportSentinel("ValidationResult")
ValidationIssue = _LazyImportSentinel("ValidationIssue")
ProviderType = _LazyImportSentinel("ProviderType")


# Callable placeholder functions that trigger lazy loading
def auto_instrument(*args, **kwargs):
    """Lazy-loaded auto_instrument function."""
    real_func = __getattr__("auto_instrument")
    globals()["auto_instrument"] = real_func  # Replace placeholder
    return real_func(*args, **kwargs)


def disable_auto_instrumentation(*args, **kwargs):
    """Lazy-loaded disable_auto_instrumentation function."""
    real_func = __getattr__("disable_auto_instrumentation")
    globals()["disable_auto_instrumentation"] = real_func
    return real_func(*args, **kwargs)


def configure_auto_instrumentation(*args, **kwargs):
    """Lazy-loaded configure_auto_instrumentation function."""
    real_func = __getattr__("configure_auto_instrumentation")
    globals()["configure_auto_instrumentation"] = real_func
    return real_func(*args, **kwargs)


def is_instrumented(*args, **kwargs):
    """Lazy-loaded is_instrumented function."""
    real_func = __getattr__("is_instrumented")
    globals()["is_instrumented"] = real_func
    return real_func(*args, **kwargs)


def validate_crewai_setup(*args, **kwargs):
    """Lazy-loaded validate_crewai_setup function."""
    real_func = __getattr__("validate_crewai_setup")
    globals()["validate_crewai_setup"] = real_func
    return real_func(*args, **kwargs)


def print_validation_result(*args, **kwargs):
    """Lazy-loaded print_validation_result function."""
    real_func = __getattr__("print_validation_result")
    globals()["print_validation_result"] = real_func
    return real_func(*args, **kwargs)


def quick_validate(*args, **kwargs):
    """Lazy-loaded quick_validate function."""
    real_func = __getattr__("quick_validate")
    globals()["quick_validate"] = real_func
    return real_func(*args, **kwargs)


def get_current_adapter(*args, **kwargs):
    """Lazy-loaded get_current_adapter function."""
    real_func = __getattr__("get_current_adapter")
    globals()["get_current_adapter"] = real_func
    return real_func(*args, **kwargs)


def get_current_monitor(*args, **kwargs):
    """Lazy-loaded get_current_monitor function."""
    real_func = __getattr__("get_current_monitor")
    globals()["get_current_monitor"] = real_func
    return real_func(*args, **kwargs)


def get_cost_summary(*args, **kwargs):
    """Lazy-loaded get_cost_summary function."""
    real_func = __getattr__("get_cost_summary")
    globals()["get_cost_summary"] = real_func
    return real_func(*args, **kwargs)


def get_execution_metrics(*args, **kwargs):
    """Lazy-loaded get_execution_metrics function."""
    real_func = __getattr__("get_execution_metrics")
    globals()["get_execution_metrics"] = real_func
    return real_func(*args, **kwargs)


def get_instrumentation_stats(*args, **kwargs):
    """Lazy-loaded get_instrumentation_stats function."""
    real_func = __getattr__("get_instrumentation_stats")
    globals()["get_instrumentation_stats"] = real_func
    return real_func(*args, **kwargs)


def create_crewai_cost_context(*args, **kwargs):
    """Lazy-loaded create_crewai_cost_context function."""
    real_func = __getattr__("create_crewai_cost_context")
    globals()["create_crewai_cost_context"] = real_func
    return real_func(*args, **kwargs)


def multi_provider_cost_tracking(*args, **kwargs):
    """Lazy-loaded multi_provider_cost_tracking function."""
    real_func = __getattr__("multi_provider_cost_tracking")
    globals()["multi_provider_cost_tracking"] = real_func
    return real_func(*args, **kwargs)


# Convenience functions for common patterns (defined in this module)
def instrument_crewai(
    team: str = "default-team",
    project: str = "crewai-app",
    environment: str = "development",
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory",
) -> "GenOpsCrewAIAdapter":
    """
    Convenience function to instrument CrewAI with common settings.

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

        # Basic setup
        adapter = instrument_crewai(
            team="ml-team",
            project="research-agents",
            daily_budget_limit=50.0
        )

        with adapter.track_crew("market-research") as context:
            result = crew.kickoff()
    """
    # Lazy import to avoid circular dependency
    GenOpsCrewAIAdapter = __getattr__("GenOpsCrewAIAdapter")
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
) -> "GenOpsCrewAIAdapter":
    """
    Create a GenOps adapter optimized for multi-agent workflows.

    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        daily_budget_limit: Daily spending limit
        enable_advanced_monitoring: Enable advanced monitoring features

    Returns:
        GenOpsCrewAIAdapter: Configured adapter for multi-agent workflows

    Example:
        from genops.providers.crewai import create_multi_agent_adapter

        adapter = create_multi_agent_adapter(
            team="ai-research",
            project="collaborative-agents",
            daily_budget_limit=300.0
        )

        with adapter.track_crew("research-analysis") as context:
            result = multi_agent_crew.kickoff()
    """
    # Lazy import to avoid circular dependency
    GenOpsCrewAIAdapter = __getattr__("GenOpsCrewAIAdapter")
    return GenOpsCrewAIAdapter(
        team=team,
        project=project,
        daily_budget_limit=daily_budget_limit,
        enable_agent_tracking=enable_advanced_monitoring,
        enable_task_tracking=enable_advanced_monitoring,
        enable_cost_tracking=True,
        governance_policy="advisory",
    )


def analyze_crew_costs(
    adapter: "GenOpsCrewAIAdapter", time_period_hours: int = 24
) -> dict:
    """
    Analyze crew costs and provide optimization recommendations.

    Args:
        adapter: GenOps CrewAI adapter
        time_period_hours: Time period for analysis in hours

    Returns:
        dict: Cost analysis with recommendations

    Example:
        from genops.providers.crewai import analyze_crew_costs

        analysis = analyze_crew_costs(adapter, time_period_hours=24)

        print(f"Total cost: ${analysis['total_cost']:.2f}")
        print(f"Most expensive agent: {analysis['most_expensive_agent']}")

        for rec in analysis['recommendations']:
            print(f"💡 {rec['reasoning']}")
    """
    if not hasattr(adapter, "cost_aggregator") or not adapter.cost_aggregator:
        return {"error": "Cost aggregator not available"}

    # Get cost analysis from aggregator
    analysis = adapter.cost_aggregator.get_cost_analysis(
        time_period_hours=time_period_hours
    )

    # Convert to more friendly format
    return {
        "total_cost": float(analysis.total_cost),
        "cost_by_provider": {k: float(v) for k, v in analysis.cost_by_provider.items()},
        "cost_by_agent": {k: float(v) for k, v in analysis.cost_by_agent.items()},
        "most_expensive_agent": max(
            analysis.cost_by_agent.items(), key=lambda x: x[1], default=(None, 0)
        )[0],
        "recommendations": [
            {
                "agent": rec.agent_name,
                "current_provider": rec.current_provider,
                "recommended_provider": rec.recommended_provider,
                "potential_savings": float(rec.potential_savings),
                "reasoning": rec.reasoning,
            }
            for rec in analysis.optimization_recommendations
        ],
        "provider_summaries": {
            provider: {
                "total_cost": float(summary.total_cost),
                "total_operations": summary.total_operations,
                "agents_used": list(summary.agents_used),
                "models_used": list(summary.models_used),
            }
            for provider, summary in analysis.provider_summaries.items()
        },
    }


def get_multi_agent_insights(monitor: "CrewAIAgentMonitor", crew_name: str) -> dict:
    """
    Get specialized insights for multi-agent workflows.

    Args:
        monitor: CrewAI monitor instance
        crew_name: Crew name for analysis

    Returns:
        dict: Multi-agent specific insights and metrics

    Example:
        insights = get_multi_agent_insights(monitor, "research-crew")

        print(f"Collaboration score: {insights['collaboration_score']:.2f}")
        print(f"Agent efficiency: {insights['efficiency_score']:.2f}")
        print(f"Bottleneck agents: {insights['bottleneck_agents']}")
    """
    workflow_metrics = monitor.get_workflow_analysis(crew_name)
    if not workflow_metrics:
        return {"error": "Workflow analysis not found"}

    return {
        "collaboration_matrix": workflow_metrics.agent_collaboration_matrix,
        "bottleneck_agents": workflow_metrics.bottleneck_agents,
        "load_balancing_score": workflow_metrics.load_balancing_score,
        "coordination_overhead": workflow_metrics.coordination_overhead_seconds,
        "parallel_efficiency": workflow_metrics.parallel_efficiency,
        "optimal_sequence": workflow_metrics.optimal_agent_sequence,
    }


# Lazy loading implementation to avoid circular imports
def __getattr__(name: str) -> Any:
    """Dynamically import requested attributes to avoid circular dependencies."""
    if name in _import_cache:
        return _import_cache[name]

    # Adapter imports
    if name in (
        "GenOpsCrewAIAdapter",
        "CrewAIAgentResult",
        "CrewAITaskResult",
        "CrewAICrewResult",
        "CrewAISessionContext",
        "CrewAICrewContext",
    ):
        from genops.providers.crewai.adapter import (
            CrewAIAgentResult,
            CrewAICrewContext,
            CrewAICrewResult,
            CrewAISessionContext,
            CrewAITaskResult,
            GenOpsCrewAIAdapter,
        )

        _import_cache.update(
            {
                "GenOpsCrewAIAdapter": GenOpsCrewAIAdapter,
                "CrewAIAgentResult": CrewAIAgentResult,
                "CrewAITaskResult": CrewAITaskResult,
                "CrewAICrewResult": CrewAICrewResult,
                "CrewAISessionContext": CrewAISessionContext,
                "CrewAICrewContext": CrewAICrewContext,
            }
        )
        return _import_cache[name]

    # Cost aggregator imports
    elif name in (
        "CrewAICostAggregator",
        "AgentCostEntry",
        "CrewCostSummary",
        "ProviderCostSummary",
        "CostOptimizationRecommendation",
        "CostAnalysisResult",
        "ProviderType",
        "create_crewai_cost_context",
        "multi_provider_cost_tracking",
    ):
        from genops.providers.crewai.cost_aggregator import (
            AgentCostEntry,
            CostAnalysisResult,
            CostOptimizationRecommendation,
            CrewAICostAggregator,
            CrewCostSummary,
            ProviderCostSummary,
            ProviderType,
            create_crewai_cost_context,
            multi_provider_cost_tracking,
        )

        _import_cache.update(
            {
                "CrewAICostAggregator": CrewAICostAggregator,
                "AgentCostEntry": AgentCostEntry,
                "CrewCostSummary": CrewCostSummary,
                "ProviderCostSummary": ProviderCostSummary,
                "CostOptimizationRecommendation": CostOptimizationRecommendation,
                "CostAnalysisResult": CostAnalysisResult,
                "ProviderType": ProviderType,
                "create_crewai_cost_context": create_crewai_cost_context,
                "multi_provider_cost_tracking": multi_provider_cost_tracking,
            }
        )
        return _import_cache[name]

    # Monitor imports
    elif name in (
        "CrewAIAgentMonitor",
        "AgentExecutionMetrics",
        "TaskExecutionMetrics",
        "CrewExecutionMetrics",
        "MultiAgentWorkflowMetrics",
    ):
        from genops.providers.crewai.agent_monitor import (
            AgentExecutionMetrics,
            CrewAIAgentMonitor,
            CrewExecutionMetrics,
            MultiAgentWorkflowMetrics,
            TaskExecutionMetrics,
        )

        _import_cache.update(
            {
                "CrewAIAgentMonitor": CrewAIAgentMonitor,
                "AgentExecutionMetrics": AgentExecutionMetrics,
                "TaskExecutionMetrics": TaskExecutionMetrics,
                "CrewExecutionMetrics": CrewExecutionMetrics,
                "MultiAgentWorkflowMetrics": MultiAgentWorkflowMetrics,
            }
        )
        return _import_cache[name]

    # Registration imports
    elif name in (
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
    ):
        from genops.providers.crewai.registration import (
            TemporaryInstrumentation,
            auto_instrument,
            configure_auto_instrumentation,
            disable_auto_instrumentation,
            get_cost_summary,
            get_current_adapter,
            get_current_monitor,
            get_execution_metrics,
            get_instrumentation_stats,
            is_instrumented,
        )

        _import_cache.update(
            {
                "auto_instrument": auto_instrument,
                "disable_auto_instrumentation": disable_auto_instrumentation,
                "configure_auto_instrumentation": configure_auto_instrumentation,
                "is_instrumented": is_instrumented,
                "get_instrumentation_stats": get_instrumentation_stats,
                "get_current_adapter": get_current_adapter,
                "get_current_monitor": get_current_monitor,
                "get_cost_summary": get_cost_summary,
                "get_execution_metrics": get_execution_metrics,
                "TemporaryInstrumentation": TemporaryInstrumentation,
            }
        )
        return _import_cache[name]

    # Validation imports
    elif name in (
        "validate_crewai_setup",
        "print_validation_result",
        "quick_validate",
        "ValidationResult",
        "ValidationIssue",
    ):
        from genops.providers.crewai.validation import (
            ValidationIssue,
            ValidationResult,
            print_validation_result,
            quick_validate,
            validate_crewai_setup,
        )

        _import_cache.update(
            {
                "validate_crewai_setup": validate_crewai_setup,
                "print_validation_result": print_validation_result,
                "quick_validate": quick_validate,
                "ValidationResult": ValidationResult,
                "ValidationIssue": ValidationIssue,
            }
        )
        return _import_cache[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all main classes and functions (maintains API compatibility with lazy loading)
__all__ = [
    # Core classes
    "GenOpsCrewAIAdapter",
    "CrewAIAgentMonitor",
    "CrewAICostAggregator",
    # Data classes
    "CrewAIAgentResult",
    "CrewAITaskResult",
    "CrewAICrewResult",
    "CrewAISessionContext",
    "AgentExecutionMetrics",
    "TaskExecutionMetrics",
    "CrewExecutionMetrics",
    "MultiAgentWorkflowMetrics",
    "AgentCostEntry",
    "CrewCostSummary",
    "ProviderCostSummary",
    "CostOptimizationRecommendation",
    "CostAnalysisResult",
    # Auto-instrumentation
    "auto_instrument",
    "disable_auto_instrumentation",
    "configure_auto_instrumentation",
    "is_instrumented",
    "TemporaryInstrumentation",
    # Convenience functions
    "instrument_crewai",
    "create_multi_agent_adapter",
    "analyze_crew_costs",
    "get_multi_agent_insights",
    # Validation functions
    "validate_crewai_setup",
    "print_validation_result",
    "quick_validate",
    "ValidationResult",
    "ValidationIssue",
    # Monitoring functions
    "get_current_adapter",
    "get_current_monitor",
    "get_cost_summary",
    "get_execution_metrics",
    "get_instrumentation_stats",
    # Cost tracking
    "create_crewai_cost_context",
    "multi_provider_cost_tracking",
    "create_chain_cost_context",  # CLAUDE.md standard alias
    # Utilities
    "ProviderType",
]
