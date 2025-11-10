#!/usr/bin/env python3
"""
Haystack AI Integration for GenOps Governance

Comprehensive integration for Haystack AI orchestration framework with GenOps governance,
providing end-to-end tracking for RAG workflows, agent systems, and multi-provider pipelines.

Usage:
    # Quick setup with auto-instrumentation
    from genops.providers.haystack import auto_instrument
    auto_instrument()
    
    # Manual setup with full control
    from genops.providers.haystack import GenOpsHaystackAdapter
    adapter = GenOpsHaystackAdapter(
        team="ai-research",
        project="rag-system", 
        daily_budget_limit=100.0
    )
    
    with adapter.track_pipeline("document-qa") as context:
        result = pipeline.run({"query": "What is RAG?"})
        print(f"Total cost: ${context.total_cost:.6f}")

Features:
    - Zero-code auto-instrumentation for existing Haystack applications
    - End-to-end pipeline governance and cost tracking
    - Multi-provider cost aggregation (OpenAI, Anthropic, Hugging Face, etc.)
    - RAG workflow specialization with retrieval and generation tracking
    - Agent workflow governance with decision and tool usage monitoring
    - Enterprise compliance patterns and multi-tenant governance
"""

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import registry to avoid circular dependencies
_import_cache = {}

# Sentinel class for lazy-loaded symbols (satisfies static analysis while enabling lazy loading)
class _LazyImportSentinel:
    """Sentinel class indicating a symbol should be lazy-loaded."""
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"<LazyImport: {self.name}>"

# Placeholder definitions for exported symbols (satisfies static analysis while maintaining lazy loading)
# These sentinels will be replaced by actual imports when accessed

# Core classes
GenOpsHaystackAdapter = _LazyImportSentinel("GenOpsHaystackAdapter")
HaystackMonitor = _LazyImportSentinel("HaystackMonitor")
HaystackCostAggregator = _LazyImportSentinel("HaystackCostAggregator")

# Data classes
HaystackComponentResult = _LazyImportSentinel("HaystackComponentResult")
HaystackPipelineResult = _LazyImportSentinel("HaystackPipelineResult")
HaystackSessionContext = _LazyImportSentinel("HaystackSessionContext")
ComponentExecutionMetrics = _LazyImportSentinel("ComponentExecutionMetrics")
PipelineExecutionMetrics = _LazyImportSentinel("PipelineExecutionMetrics")
RAGWorkflowMetrics = _LazyImportSentinel("RAGWorkflowMetrics")
AgentWorkflowMetrics = _LazyImportSentinel("AgentWorkflowMetrics")
ComponentCostEntry = _LazyImportSentinel("ComponentCostEntry")
CostAnalysisResult = _LazyImportSentinel("CostAnalysisResult")
ProviderCostSummary = _LazyImportSentinel("ProviderCostSummary")
CostOptimizationRecommendation = _LazyImportSentinel("CostOptimizationRecommendation")

# Auto-instrumentation
auto_instrument = _LazyImportSentinel("auto_instrument")
disable_auto_instrumentation = _LazyImportSentinel("disable_auto_instrumentation")
configure_auto_instrumentation = _LazyImportSentinel("configure_auto_instrumentation")
is_instrumented = _LazyImportSentinel("is_instrumented")
TemporaryInstrumentation = _LazyImportSentinel("TemporaryInstrumentation")

# Validation functions
validate_haystack_setup = _LazyImportSentinel("validate_haystack_setup")
print_validation_result = _LazyImportSentinel("print_validation_result")
ValidationResult = _LazyImportSentinel("ValidationResult")
ValidationIssue = _LazyImportSentinel("ValidationIssue")

# Monitoring functions
get_current_adapter = _LazyImportSentinel("get_current_adapter")
get_current_monitor = _LazyImportSentinel("get_current_monitor")
get_cost_summary = _LazyImportSentinel("get_cost_summary")
get_execution_metrics = _LazyImportSentinel("get_execution_metrics")
get_instrumentation_stats = _LazyImportSentinel("get_instrumentation_stats")

# Mixins and utilities
GenOpsComponentMixin = _LazyImportSentinel("GenOpsComponentMixin")
ProviderType = _LazyImportSentinel("ProviderType")

# Check for Haystack availability
try:
    import haystack
    HAS_HAYSTACK = True
    logger.info(f"GenOps Haystack integration loaded - Haystack {haystack.__version__} detected")
except ImportError:
    HAS_HAYSTACK = False
    logger.warning("Haystack not installed - integration available but limited functionality")

# Version info
__version__ = "1.0.0"
__author__ = "GenOps AI"

# Convenience functions for common patterns
def instrument_haystack(
    team: str = "default-team",
    project: str = "haystack-app", 
    environment: str = "development",
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory"
) -> bool:
    """
    Convenience function to instrument Haystack with common settings.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        environment: Environment (development, staging, production)
        daily_budget_limit: Daily spending limit in USD
        governance_policy: Policy enforcement level ("advisory", "enforced")
        
    Returns:
        bool: True if instrumentation successful
        
    Example:
        from genops.providers.haystack import instrument_haystack
        
        # Basic setup
        instrument_haystack(
            team="ml-team",
            project="rag-chatbot",
            daily_budget_limit=50.0
        )
        
        # Your existing Haystack code works unchanged
        pipeline = Pipeline()
        result = pipeline.run({"query": "What is RAG?"})
    """
    # Lazy import to avoid circular dependency
    auto_instrument = __getattr__('auto_instrument')
    return auto_instrument(
        team=team,
        project=project,
        environment=environment,
        daily_budget_limit=daily_budget_limit,
        governance_policy=governance_policy
    )


def create_rag_adapter(
    team: str,
    project: str,
    daily_budget_limit: float = 100.0,
    enable_retrieval_tracking: bool = True,
    enable_generation_tracking: bool = True
) -> 'GenOpsHaystackAdapter':
    """
    Create a GenOps adapter optimized for RAG (Retrieval-Augmented Generation) workflows.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        daily_budget_limit: Daily spending limit
        enable_retrieval_tracking: Enable detailed retrieval tracking
        enable_generation_tracking: Enable detailed generation tracking
        
    Returns:
        GenOpsHaystackAdapter: Configured adapter for RAG workflows
        
    Example:
        from genops.providers.haystack import create_rag_adapter
        
        adapter = create_rag_adapter(
            team="research-team",
            project="document-qa",
            daily_budget_limit=200.0
        )
        
        with adapter.track_pipeline("rag-qa") as context:
            # Retrieval phase
            retriever_result = retriever.run(query="What is RAG?")
            
            # Generation phase  
            generator_result = generator.run(
                prompt=build_prompt(query, retriever_result["documents"])
            )
    """
    # Lazy import to avoid circular dependency
    GenOpsHaystackAdapter = __getattr__('GenOpsHaystackAdapter')
    return GenOpsHaystackAdapter(
        team=team,
        project=project,
        daily_budget_limit=daily_budget_limit,
        enable_component_tracking=True,
        # RAG-specific optimizations would go here
        governance_policy="advisory"
    )


def create_agent_adapter(
    team: str,
    project: str,
    daily_budget_limit: float = 100.0,
    enable_decision_tracking: bool = True,
    enable_tool_tracking: bool = True
) -> 'GenOpsHaystackAdapter':
    """
    Create a GenOps adapter optimized for agent workflows.
    
    Args:
        team: Team name for cost attribution
        project: Project name for cost attribution
        daily_budget_limit: Daily spending limit
        enable_decision_tracking: Enable agent decision tracking
        enable_tool_tracking: Enable tool usage tracking
        
    Returns:
        GenOpsHaystackAdapter: Configured adapter for agent workflows
        
    Example:
        from genops.providers.haystack import create_agent_adapter
        
        adapter = create_agent_adapter(
            team="ai-agents",
            project="research-assistant",
            daily_budget_limit=300.0
        )
        
        with adapter.track_session("research-task") as session:
            for step in agent_steps:
                with adapter.track_pipeline(f"agent-step-{step}") as context:
                    result = agent_pipeline.run(step_input)
    """
    # Lazy import to avoid circular dependency
    GenOpsHaystackAdapter = __getattr__('GenOpsHaystackAdapter')
    return GenOpsHaystackAdapter(
        team=team,
        project=project,
        daily_budget_limit=daily_budget_limit,
        enable_component_tracking=True,
        # Agent-specific optimizations would go here
        governance_policy="advisory"
    )


def analyze_pipeline_costs(adapter: 'GenOpsHaystackAdapter', time_period_hours: int = 24) -> dict:
    """
    Analyze pipeline costs and provide optimization recommendations.
    
    Args:
        adapter: GenOps Haystack adapter
        time_period_hours: Time period for analysis in hours
        
    Returns:
        dict: Cost analysis with recommendations
        
    Example:
        from genops.providers.haystack import analyze_pipeline_costs
        
        analysis = analyze_pipeline_costs(adapter, time_period_hours=24)
        
        print(f"Total cost: ${analysis['total_cost']:.2f}")
        print(f"Most expensive component: {analysis['most_expensive_component']}")
        
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
        "cost_by_component": {k: float(v) for k, v in analysis.cost_by_component.items()},
        "most_expensive_component": max(analysis.cost_by_component.items(), 
                                       key=lambda x: x[1], default=(None, 0))[0],
        "recommendations": [
            {
                "component": rec.component_name,
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
                "components_used": list(summary.components_used),
                "models_used": list(summary.models_used)
            }
            for provider, summary in analysis.provider_summaries.items()
        }
    }


def get_rag_insights(monitor: 'HaystackMonitor', pipeline_id: str) -> dict:
    """
    Get specialized insights for RAG workflows.
    
    Args:
        monitor: Haystack monitor instance
        pipeline_id: Pipeline execution ID
        
    Returns:
        dict: RAG-specific insights and metrics
        
    Example:
        insights = get_rag_insights(monitor, pipeline_id)
        
        print(f"Retrieval latency: {insights['retrieval_latency']:.2f}s")
        print(f"Generation latency: {insights['generation_latency']:.2f}s") 
        print(f"Documents retrieved: {insights['documents_retrieved']}")
    """
    metrics = monitor.get_execution_metrics(pipeline_id)
    if not metrics:
        return {"error": "Pipeline execution not found"}
    
    rag_metrics = monitor.analyze_rag_workflow(metrics)
    
    return {
        "retrieval_latency": rag_metrics.retrieval_latency_seconds,
        "generation_latency": rag_metrics.generation_latency_seconds,
        "documents_retrieved": rag_metrics.documents_retrieved,
        "retrieval_success_rate": rag_metrics.retrieval_success_rate,
        "generation_success_rate": rag_metrics.generation_success_rate,
        "end_to_end_latency": rag_metrics.end_to_end_latency_seconds,
        "embedding_components": len(rag_metrics.embedding_metrics)
    }


def get_agent_insights(monitor: 'HaystackMonitor', pipeline_id: str) -> dict:
    """
    Get specialized insights for agent workflows.
    
    Args:
        monitor: Haystack monitor instance
        pipeline_id: Pipeline execution ID
        
    Returns:
        dict: Agent-specific insights and metrics
        
    Example:
        insights = get_agent_insights(monitor, pipeline_id)
        
        print(f"Decisions made: {insights['decisions_made']}")
        print(f"Tools used: {insights['tools_used']}")
        print(f"Decision latency: {insights['decision_latency']:.2f}s")
    """
    metrics = monitor.get_execution_metrics(pipeline_id)
    if not metrics:
        return {"error": "Pipeline execution not found"}
    
    agent_metrics = monitor.analyze_agent_workflow(metrics)
    
    return {
        "decisions_made": agent_metrics.decisions_made,
        "tools_used": agent_metrics.tools_used,
        "tool_usage_count": agent_metrics.tool_usage_count,
        "tool_success_rate": agent_metrics.tool_success_rate,
        "decision_latency": agent_metrics.decision_latency_seconds,
        "total_iterations": agent_metrics.total_iterations,
        "cost_by_tool": {k: float(v) for k, v in agent_metrics.cost_by_tool.items()}
    }


# Custom module type to handle lazy loading
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

# Lazy loading implementation to avoid circular imports
def __getattr__(name: str) -> Any:
    """Dynamically import requested attributes to avoid circular dependencies."""
    if name in _import_cache:
        return _import_cache[name]
    
    # Haystack adapter imports
    if name in ('GenOpsHaystackAdapter', 'HaystackComponentResult', 'HaystackPipelineResult', 
                'HaystackSessionContext', 'GenOpsComponentMixin'):
        from genops.providers.haystack_adapter import (
            GenOpsHaystackAdapter, HaystackComponentResult, HaystackPipelineResult,
            HaystackSessionContext, GenOpsComponentMixin
        )
        _import_cache.update({
            'GenOpsHaystackAdapter': GenOpsHaystackAdapter,
            'HaystackComponentResult': HaystackComponentResult,
            'HaystackPipelineResult': HaystackPipelineResult,
            'HaystackSessionContext': HaystackSessionContext,
            'GenOpsComponentMixin': GenOpsComponentMixin
        })
        return _import_cache[name]
    
    # Cost aggregator imports
    elif name in ('HaystackCostAggregator', 'ComponentCostEntry', 'ProviderCostSummary', 
                  'CostAnalysisResult', 'CostOptimizationRecommendation', 'ProviderType'):
        from genops.providers.haystack_cost_aggregator import (
            HaystackCostAggregator, ComponentCostEntry, ProviderCostSummary,
            CostAnalysisResult, CostOptimizationRecommendation, ProviderType
        )
        _import_cache.update({
            'HaystackCostAggregator': HaystackCostAggregator,
            'ComponentCostEntry': ComponentCostEntry,
            'ProviderCostSummary': ProviderCostSummary,
            'CostAnalysisResult': CostAnalysisResult,
            'CostOptimizationRecommendation': CostOptimizationRecommendation,
            'ProviderType': ProviderType
        })
        return _import_cache[name]
    
    # Monitor imports
    elif name in ('HaystackMonitor', 'ComponentExecutionMetrics', 'PipelineExecutionMetrics', 
                  'RAGWorkflowMetrics', 'AgentWorkflowMetrics'):
        from genops.providers.haystack_monitor import (
            HaystackMonitor, ComponentExecutionMetrics, PipelineExecutionMetrics,
            RAGWorkflowMetrics, AgentWorkflowMetrics
        )
        _import_cache.update({
            'HaystackMonitor': HaystackMonitor,
            'ComponentExecutionMetrics': ComponentExecutionMetrics,
            'PipelineExecutionMetrics': PipelineExecutionMetrics,
            'RAGWorkflowMetrics': RAGWorkflowMetrics,
            'AgentWorkflowMetrics': AgentWorkflowMetrics
        })
        return _import_cache[name]
    
    # Registration imports
    elif name in ('auto_instrument', 'disable_auto_instrumentation', 'configure_auto_instrumentation',
                  'is_instrumented', 'get_instrumentation_stats', 'get_current_adapter',
                  'get_current_monitor', 'get_cost_summary', 'get_execution_metrics', 'TemporaryInstrumentation'):
        from genops.providers.haystack_registration import (
            auto_instrument, disable_auto_instrumentation, configure_auto_instrumentation,
            is_instrumented, get_instrumentation_stats, get_current_adapter,
            get_current_monitor, get_cost_summary, get_execution_metrics, TemporaryInstrumentation
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
            'get_execution_metrics': get_execution_metrics,
            'TemporaryInstrumentation': TemporaryInstrumentation
        })
        return _import_cache[name]
    
    # Validation imports
    elif name in ('validate_haystack_setup', 'print_validation_result', 'ValidationResult', 'ValidationIssue'):
        from genops.providers.haystack_validation import (
            validate_haystack_setup, print_validation_result, ValidationResult, ValidationIssue
        )
        _import_cache.update({
            'validate_haystack_setup': validate_haystack_setup,
            'print_validation_result': print_validation_result,
            'ValidationResult': ValidationResult,
            'ValidationIssue': ValidationIssue
        })
        return _import_cache[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all main classes and functions (maintains API compatibility with lazy loading)
__all__ = [
    # Core classes
    'GenOpsHaystackAdapter',
    'HaystackMonitor', 
    'HaystackCostAggregator',
    
    # Data classes
    'HaystackComponentResult',
    'HaystackPipelineResult', 
    'HaystackSessionContext',
    'ComponentExecutionMetrics',
    'PipelineExecutionMetrics',
    'RAGWorkflowMetrics',
    'AgentWorkflowMetrics',
    'ComponentCostEntry',
    'CostAnalysisResult',
    'ProviderCostSummary',
    'CostOptimizationRecommendation',
    
    # Auto-instrumentation
    'auto_instrument',
    'disable_auto_instrumentation',
    'configure_auto_instrumentation',
    'is_instrumented',
    'TemporaryInstrumentation',
    
    # Convenience functions
    'instrument_haystack',
    'create_rag_adapter',
    'create_agent_adapter',
    'analyze_pipeline_costs',
    'get_rag_insights',
    'get_agent_insights',
    
    # Validation functions
    'validate_haystack_setup',
    'print_validation_result',
    'ValidationResult',
    'ValidationIssue',
    
    # Monitoring functions
    'get_current_adapter',
    'get_current_monitor',
    'get_cost_summary',
    'get_execution_metrics',
    'get_instrumentation_stats',
    
    # Mixins and utilities
    'GenOpsComponentMixin',
    'ProviderType'
]