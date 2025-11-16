#!/usr/bin/env python3
"""
AutoGen Integration for GenOps Governance

Comprehensive integration for AutoGen multi-agent systems with GenOps governance,
providing end-to-end tracking for conversation flows, agent interactions, and multi-provider cost management.

Usage:
    # Quick setup with auto-instrumentation
    from genops.providers.autogen import auto_instrument
    auto_instrument()
    
    # Manual setup with full control
    from genops.providers.autogen import GenOpsAutoGenAdapter
    adapter = GenOpsAutoGenAdapter(
        team="ai-research",
        project="multi-agent-conversations", 
        daily_budget_limit=100.0
    )
    
    with adapter.track_conversation("assistant-user-chat") as context:
        response = assistant.generate_reply(messages=conversation_history)
        print(f"Total cost: ${context.total_cost:.6f}")

Features:
    - Zero-code auto-instrumentation for existing AutoGen applications
    - End-to-end conversation governance and cost tracking
    - Multi-provider cost aggregation (OpenAI, Anthropic, Google, etc.)
    - Group chat orchestration monitoring with participant tracking
    - Code execution tracking for AutoGen's code interpreter capabilities
    - Function calling telemetry for tool usage patterns
    - Enterprise compliance patterns and multi-tenant governance
"""

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Define create_chain_cost_context at module level for CodeQL compliance
try:
    from genops.providers.autogen.cost_aggregator import create_chain_cost_context
except ImportError:
    def create_chain_cost_context(chain_id: str):
        """Fallback implementation if cost_aggregator is not available."""
        from genops.providers.autogen.cost_aggregator import create_chain_cost_context as _real_func
        return _real_func(chain_id)

# Lazy import registry to avoid circular dependencies
_import_cache = {}

# Custom module type to handle lazy loading (applying CrewAI lessons)
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

# Sentinel class for lazy-loaded symbols
class _LazyImportSentinel:
    """Sentinel class indicating a symbol should be lazy-loaded."""
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"<LazyImport: {self.name}>"

# Check for AutoGen availability
try:
    import autogen
    HAS_AUTOGEN = True
    logger.info(f"GenOps AutoGen integration loaded - AutoGen {autogen.__version__} detected")
except ImportError:
    HAS_AUTOGEN = False
    logger.warning("AutoGen not installed - integration available but limited functionality")

# Version info
__version__ = "1.0.0"
__author__ = "GenOps AI"

# Callable class placeholders for instantiable classes
def GenOpsAutoGenAdapter(*args, **kwargs):
    """Lazy-loaded GenOpsAutoGenAdapter class."""
    real_class = __getattr__('GenOpsAutoGenAdapter')
    globals()['GenOpsAutoGenAdapter'] = real_class  # Replace placeholder
    return real_class(*args, **kwargs)

def AutoGenConversationMonitor(*args, **kwargs):
    """Lazy-loaded AutoGenConversationMonitor class."""
    real_class = __getattr__('AutoGenConversationMonitor')
    globals()['AutoGenConversationMonitor'] = real_class
    return real_class(*args, **kwargs)

def AutoGenCostAggregator(*args, **kwargs):
    """Lazy-loaded AutoGenCostAggregator class."""
    real_class = __getattr__('AutoGenCostAggregator')
    globals()['AutoGenCostAggregator'] = real_class
    return real_class(*args, **kwargs)

def TemporaryInstrumentation(*args, **kwargs):
    """Lazy-loaded TemporaryInstrumentation class."""
    real_class = __getattr__('TemporaryInstrumentation')
    globals()['TemporaryInstrumentation'] = real_class
    return real_class(*args, **kwargs)

# Data classes (sentinels - not instantiated directly)
AutoGenConversationResult = _LazyImportSentinel("AutoGenConversationResult")
AutoGenAgentResult = _LazyImportSentinel("AutoGenAgentResult")
AutoGenGroupChatResult = _LazyImportSentinel("AutoGenGroupChatResult")
AutoGenSessionContext = _LazyImportSentinel("AutoGenSessionContext")
ConversationMetrics = _LazyImportSentinel("ConversationMetrics")
AgentInteractionMetrics = _LazyImportSentinel("AgentInteractionMetrics")
GroupChatMetrics = _LazyImportSentinel("GroupChatMetrics")
CodeExecutionMetrics = _LazyImportSentinel("CodeExecutionMetrics")
AgentCostEntry = _LazyImportSentinel("AgentCostEntry")
ConversationCostSummary = _LazyImportSentinel("ConversationCostSummary")
ProviderCostSummary = _LazyImportSentinel("ProviderCostSummary")
CostOptimizationRecommendation = _LazyImportSentinel("CostOptimizationRecommendation")
CostAnalysisResult = _LazyImportSentinel("CostAnalysisResult")
ValidationResult = _LazyImportSentinel("ValidationResult")
ValidationIssue = _LazyImportSentinel("ValidationIssue")
ProviderType = _LazyImportSentinel("ProviderType")

# Callable placeholder functions that trigger lazy loading
def auto_instrument(*args, **kwargs):
    """Lazy-loaded auto_instrument function."""
    real_func = __getattr__('auto_instrument')
    globals()['auto_instrument'] = real_func  # Replace placeholder
    return real_func(*args, **kwargs)

def disable_auto_instrumentation(*args, **kwargs):
    """Lazy-loaded disable_auto_instrumentation function."""
    real_func = __getattr__('disable_auto_instrumentation')
    globals()['disable_auto_instrumentation'] = real_func
    return real_func(*args, **kwargs)

def configure_auto_instrumentation(*args, **kwargs):
    """Lazy-loaded configure_auto_instrumentation function."""
    real_func = __getattr__('configure_auto_instrumentation')
    globals()['configure_auto_instrumentation'] = real_func
    return real_func(*args, **kwargs)

def is_instrumented(*args, **kwargs):
    """Lazy-loaded is_instrumented function."""
    real_func = __getattr__('is_instrumented')
    globals()['is_instrumented'] = real_func
    return real_func(*args, **kwargs)

def validate_autogen_setup(*args, **kwargs):
    """Lazy-loaded validate_autogen_setup function."""
    real_func = __getattr__('validate_autogen_setup')
    globals()['validate_autogen_setup'] = real_func
    return real_func(*args, **kwargs)

def print_validation_result(*args, **kwargs):
    """Lazy-loaded print_validation_result function."""
    real_func = __getattr__('print_validation_result')
    globals()['print_validation_result'] = real_func
    return real_func(*args, **kwargs)

def quick_validate(*args, **kwargs):
    """Lazy-loaded quick_validate function."""
    real_func = __getattr__('quick_validate')
    globals()['quick_validate'] = real_func
    return real_func(*args, **kwargs)

def get_current_adapter(*args, **kwargs):
    """Lazy-loaded get_current_adapter function."""
    real_func = __getattr__('get_current_adapter')
    globals()['get_current_adapter'] = real_func
    return real_func(*args, **kwargs)

def get_current_monitor(*args, **kwargs):
    """Lazy-loaded get_current_monitor function."""
    real_func = __getattr__('get_current_monitor')
    globals()['get_current_monitor'] = real_func
    return real_func(*args, **kwargs)

def get_cost_summary(*args, **kwargs):
    """Lazy-loaded get_cost_summary function."""
    real_func = __getattr__('get_cost_summary')
    globals()['get_cost_summary'] = real_func
    return real_func(*args, **kwargs)

def get_conversation_metrics(*args, **kwargs):
    """Lazy-loaded get_conversation_metrics function."""
    real_func = __getattr__('get_conversation_metrics')
    globals()['get_conversation_metrics'] = real_func
    return real_func(*args, **kwargs)

def get_instrumentation_stats(*args, **kwargs):
    """Lazy-loaded get_instrumentation_stats function."""
    real_func = __getattr__('get_instrumentation_stats')
    globals()['get_instrumentation_stats'] = real_func
    return real_func(*args, **kwargs)

def create_autogen_cost_context(*args, **kwargs):
    """Lazy-loaded create_autogen_cost_context function."""
    real_func = __getattr__('create_autogen_cost_context')
    globals()['create_autogen_cost_context'] = real_func
    return real_func(*args, **kwargs)

def multi_provider_cost_tracking(*args, **kwargs):
    """Lazy-loaded multi_provider_cost_tracking function."""
    real_func = __getattr__('multi_provider_cost_tracking')
    globals()['multi_provider_cost_tracking'] = real_func
    return real_func(*args, **kwargs)

def quick_validate(*args, **kwargs):
    """Lazy-loaded quick_validate function."""
    real_func = __getattr__('quick_validate')
    globals()['quick_validate'] = real_func
    return real_func(*args, **kwargs)

# Convenience functions for common patterns

def enable_governance(**kwargs):
    """
    Ultra-simple one-line setup for AutoGen governance.
    
    This is the simplest way to add GenOps governance to existing AutoGen code.
    Just call this once and your existing AutoGen code gets automatic governance tracking.
    
    Args:
        **kwargs: Optional configuration (team, project, budget_limit, etc.)
        
    Returns:
        GenOpsAutoGenAdapter: Configured adapter
        
    Example:
        from genops.providers.autogen import enable_governance
        enable_governance()  # That's it! One line.
        
        # Your existing AutoGen code works unchanged with governance
        import autogen
        assistant = autogen.AssistantAgent(name="assistant")
        # ↑ Now automatically tracked with cost and governance telemetry
    """
    # Use environment variables or sensible defaults
    import os
    
    team = kwargs.get('team', os.getenv('GENOPS_TEAM', 'my-team'))
    project = kwargs.get('project', os.getenv('GENOPS_PROJECT', 'autogen-project'))
    budget = kwargs.get('daily_budget_limit', float(os.getenv('GENOPS_BUDGET_LIMIT', '50.0')))
    
    return auto_instrument(
        team=team,
        project=project, 
        daily_budget_limit=budget,
        **{k: v for k, v in kwargs.items() if k not in ['team', 'project', 'daily_budget_limit']}
    )


def instrument_autogen(
    team: str = "default-team",
    project: str = "autogen-app",
    environment: str = "development",
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory"
) -> 'GenOpsAutoGenAdapter':
    """
    Convenience function to instrument AutoGen with common settings.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        environment: Environment (development, staging, production)
        daily_budget_limit: Daily spending limit in USD
        governance_policy: Policy enforcement level ("advisory", "enforced")
        
    Returns:
        GenOpsAutoGenAdapter: Configured adapter
        
    Example:
        from genops.providers.autogen import instrument_autogen
        
        # Basic setup
        adapter = instrument_autogen(
            team="ai-team",
            project="customer-service-bot",
            daily_budget_limit=50.0
        )
        
        with adapter.track_conversation("user-assistant") as context:
            response = assistant.generate_reply(messages=history)
    """
    # Lazy import to avoid circular dependency
    GenOpsAutoGenAdapter = __getattr__('GenOpsAutoGenAdapter')
    return GenOpsAutoGenAdapter(
        team=team,
        project=project,
        environment=environment,
        daily_budget_limit=daily_budget_limit,
        governance_policy=governance_policy
    )


def create_multi_agent_adapter(
    team: str,
    project: str,
    daily_budget_limit: float = 200.0,
    enable_advanced_monitoring: bool = True
) -> 'GenOpsAutoGenAdapter':
    """
    Create a GenOps adapter optimized for multi-agent AutoGen workflows.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        daily_budget_limit: Daily spending limit
        enable_advanced_monitoring: Enable advanced monitoring features
        
    Returns:
        GenOpsAutoGenAdapter: Configured adapter for multi-agent workflows
        
    Example:
        from genops.providers.autogen import create_multi_agent_adapter
        
        adapter = create_multi_agent_adapter(
            team="ai-research",
            project="collaborative-agents",
            daily_budget_limit=300.0
        )
        
        with adapter.track_group_chat("research-discussion") as context:
            result = group_chat_manager.run_chat(messages)
    """
    # Lazy import to avoid circular dependency
    GenOpsAutoGenAdapter = __getattr__('GenOpsAutoGenAdapter')
    return GenOpsAutoGenAdapter(
        team=team,
        project=project,
        daily_budget_limit=daily_budget_limit,
        enable_conversation_tracking=enable_advanced_monitoring,
        enable_agent_tracking=enable_advanced_monitoring,
        enable_cost_tracking=True,
        governance_policy="advisory"
    )


def analyze_conversation_costs(adapter: 'GenOpsAutoGenAdapter', time_period_hours: int = 24) -> dict:
    """
    Analyze conversation costs and provide optimization recommendations.
    
    Args:
        adapter: GenOps AutoGen adapter
        time_period_hours: Time period for analysis in hours
        
    Returns:
        dict: Cost analysis with recommendations
        
    Example:
        from genops.providers.autogen import analyze_conversation_costs
        
        analysis = analyze_conversation_costs(adapter, time_period_hours=24)
        
        print(f"Total cost: ${analysis['total_cost']:.2f}")
        print(f"Most expensive agent: {analysis['most_expensive_agent']}")
        
        for rec in analysis['recommendations']:
            print(f"💡 {rec['reasoning']}")
    """
    if not hasattr(adapter, 'cost_aggregator') or not adapter.cost_aggregator:
        return {"error": "Cost aggregator not available"}
    
    # Get cost analysis from aggregator
    analysis = adapter.cost_aggregator.get_cost_analysis(time_period_hours=time_period_hours)
    
    # Convert to more friendly format
    return {
        "total_cost": float(analysis.total_cost),
        "cost_by_provider": {k: float(v) for k, v in analysis.cost_by_provider.items()},
        "cost_by_agent": {k: float(v) for k, v in analysis.cost_by_agent.items()},
        "most_expensive_agent": max(analysis.cost_by_agent.items(), 
                                   key=lambda x: x[1], default=(None, 0))[0],
        "recommendations": [
            {
                "agent": rec.agent_name,
                "current_provider": rec.current_provider,
                "recommended_provider": rec.recommended_provider,
                "potential_savings": float(rec.potential_savings),
                "reasoning": rec.reasoning
            }
            for rec in analysis.optimization_recommendations
        ],
        "provider_summaries": {
            provider: {
                "total_cost": float(summary.total_cost),
                "total_operations": summary.total_operations,
                "agents_used": list(summary.agents_used),
                "models_used": list(summary.models_used)
            }
            for provider, summary in analysis.provider_summaries.items()
        }
    }


def get_conversation_insights(monitor: 'AutoGenConversationMonitor', conversation_id: str) -> dict:
    """
    Get specialized insights for AutoGen conversation flows.
    
    Args:
        monitor: AutoGen conversation monitor instance
        conversation_id: Conversation ID for analysis
        
    Returns:
        dict: Conversation-specific insights and metrics
        
    Example:
        insights = get_conversation_insights(monitor, "user-assistant-chat")
        
        print(f"Turns count: {insights['turns_count']}")
        print(f"Avg response time: {insights['avg_response_time_ms']:.1f}ms")
        print(f"Code executions: {insights['code_executions_count']}")
    """
    conversation_metrics = monitor.get_conversation_analysis(conversation_id)
    if not conversation_metrics:
        return {"error": "Conversation analysis not found"}
    
    return {
        "turns_count": conversation_metrics.turns_count,
        "avg_response_time_ms": conversation_metrics.avg_response_time_ms,
        "total_tokens": conversation_metrics.total_tokens,
        "cost_per_turn": conversation_metrics.cost_per_turn,
        "code_executions_count": conversation_metrics.code_executions_count,
        "function_calls_count": conversation_metrics.function_calls_count,
        "agent_participation": conversation_metrics.agent_participation,
        "conversation_quality_score": conversation_metrics.quality_score
    }


# Lazy loading implementation to avoid circular imports
def __getattr__(name: str) -> Any:
    """Dynamically import requested attributes to avoid circular dependencies."""
    if name in _import_cache:
        return _import_cache[name]
    
    # Adapter imports
    if name in ('GenOpsAutoGenAdapter', 'AutoGenConversationResult', 'AutoGenAgentResult',
                'AutoGenGroupChatResult', 'AutoGenSessionContext', 'AutoGenConversationContext'):
        from genops.providers.autogen.adapter import (
            GenOpsAutoGenAdapter, AutoGenConversationResult, AutoGenAgentResult,
            AutoGenGroupChatResult, AutoGenSessionContext, AutoGenConversationContext
        )
        _import_cache.update({
            'GenOpsAutoGenAdapter': GenOpsAutoGenAdapter,
            'AutoGenConversationResult': AutoGenConversationResult,
            'AutoGenAgentResult': AutoGenAgentResult,
            'AutoGenGroupChatResult': AutoGenGroupChatResult,
            'AutoGenSessionContext': AutoGenSessionContext,
            'AutoGenConversationContext': AutoGenConversationContext
        })
        return _import_cache[name]
    
    # Cost aggregator imports
    elif name in ('AutoGenCostAggregator', 'AgentCostEntry', 'ConversationCostSummary',
                  'ProviderCostSummary', 'CostOptimizationRecommendation', 
                  'CostAnalysisResult', 'ProviderType', 'create_autogen_cost_context',
                  'multi_provider_cost_tracking'):
        from genops.providers.autogen.cost_aggregator import (
            AutoGenCostAggregator, AgentCostEntry, ConversationCostSummary,
            ProviderCostSummary, CostOptimizationRecommendation,
            CostAnalysisResult, ProviderType, create_autogen_cost_context,
            multi_provider_cost_tracking
        )
        _import_cache.update({
            'AutoGenCostAggregator': AutoGenCostAggregator,
            'AgentCostEntry': AgentCostEntry,
            'ConversationCostSummary': ConversationCostSummary,
            'ProviderCostSummary': ProviderCostSummary,
            'CostOptimizationRecommendation': CostOptimizationRecommendation,
            'CostAnalysisResult': CostAnalysisResult,
            'ProviderType': ProviderType,
            'create_autogen_cost_context': create_autogen_cost_context,
            'multi_provider_cost_tracking': multi_provider_cost_tracking
        })
        return _import_cache[name]
    
    # Monitor imports
    elif name in ('AutoGenConversationMonitor', 'ConversationMetrics', 'AgentInteractionMetrics',
                  'GroupChatMetrics', 'CodeExecutionMetrics'):
        from genops.providers.autogen.conversation_monitor import (
            AutoGenConversationMonitor, ConversationMetrics, AgentInteractionMetrics,
            GroupChatMetrics, CodeExecutionMetrics
        )
        _import_cache.update({
            'AutoGenConversationMonitor': AutoGenConversationMonitor,
            'ConversationMetrics': ConversationMetrics,
            'AgentInteractionMetrics': AgentInteractionMetrics,
            'GroupChatMetrics': GroupChatMetrics,
            'CodeExecutionMetrics': CodeExecutionMetrics
        })
        return _import_cache[name]
    
    # Registration imports
    elif name in ('auto_instrument', 'disable_auto_instrumentation', 'configure_auto_instrumentation',
                  'is_instrumented', 'get_instrumentation_stats', 'get_current_adapter',
                  'get_current_monitor', 'get_cost_summary', 'get_conversation_metrics',
                  'TemporaryInstrumentation'):
        from genops.providers.autogen.registration import (
            auto_instrument, disable_auto_instrumentation, configure_auto_instrumentation,
            is_instrumented, get_instrumentation_stats, get_current_adapter,
            get_current_monitor, get_cost_summary, get_conversation_metrics,
            TemporaryInstrumentation
        )
        _import_cache.update({
            'auto_instrument': auto_instrument,
            'disable_auto_instrumentation': disable_auto_instrumentation,
            'configure_auto_instrumentation': configure_auto_instrumentation,
            'is_instrumented': is_instrumented,
            'get_instrumentation_stats': get_instrumentation_stats,
            'get_current_adapter': get_current_adapter,
            'get_current_monitor': get_current_monitor,
            'get_cost_summary': get_cost_summary,
            'get_conversation_metrics': get_conversation_metrics,
            'TemporaryInstrumentation': TemporaryInstrumentation
        })
        return _import_cache[name]
    
    # Validation imports
    elif name in ('validate_autogen_setup', 'print_validation_result', 'quick_validate',
                  'ValidationResult', 'ValidationIssue'):
        from genops.providers.autogen.validation import (
            validate_autogen_setup, print_validation_result, quick_validate,
            ValidationResult, ValidationIssue
        )
        _import_cache.update({
            'validate_autogen_setup': validate_autogen_setup,
            'print_validation_result': print_validation_result,
            'quick_validate': quick_validate,
            'ValidationResult': ValidationResult,
            'ValidationIssue': ValidationIssue
        })
        return _import_cache[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all main classes and functions (maintains API compatibility with lazy loading)
__all__ = [
    # Core classes
    'GenOpsAutoGenAdapter',
    'AutoGenConversationMonitor',
    'AutoGenCostAggregator',
    
    # Data classes
    'AutoGenConversationResult',
    'AutoGenAgentResult',
    'AutoGenGroupChatResult',
    'AutoGenSessionContext',
    'ConversationMetrics',
    'AgentInteractionMetrics',
    'GroupChatMetrics',
    'CodeExecutionMetrics',
    'AgentCostEntry',
    'ConversationCostSummary',
    'ProviderCostSummary',
    'CostOptimizationRecommendation',
    'CostAnalysisResult',
    
    # Auto-instrumentation
    'auto_instrument',
    'disable_auto_instrumentation',
    'configure_auto_instrumentation',
    'is_instrumented',
    'TemporaryInstrumentation',
    
    # Convenience functions
    'enable_governance',
    'instrument_autogen', 
    'create_multi_agent_adapter',
    'analyze_conversation_costs',
    'get_conversation_insights',
    
    # Validation functions
    'validate_autogen_setup',
    'print_validation_result',
    'quick_validate',
    'ValidationResult',
    'ValidationIssue',
    
    # Monitoring functions
    'get_current_adapter',
    'get_current_monitor',
    'get_cost_summary',
    'get_conversation_metrics',
    'get_instrumentation_stats',
    
    # Cost tracking
    'create_autogen_cost_context',
    'multi_provider_cost_tracking',
    'create_chain_cost_context',  # CLAUDE.md standard alias
    
    # Utilities
    'ProviderType'
]