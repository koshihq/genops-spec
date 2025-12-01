#!/usr/bin/env python3
"""
Griptape AI Framework Integration for GenOps Governance

This module provides comprehensive governance telemetry for Griptape AI agent and workflow framework,
including structure-level tracking, multi-provider cost aggregation, and enterprise compliance patterns.

Quick Start:
    from genops.providers.griptape import auto_instrument
    
    # Enable governance for all Griptape operations
    auto_instrument(team="ai-team", project="agent-workflows")
    
    # Your existing Griptape code works unchanged
    from griptape.structures import Agent
    from griptape.tasks import PromptTask
    
    agent = Agent(tasks=[PromptTask("Summarize this text")])
    result = agent.run("Long text to summarize...")
    # ✅ Now includes full GenOps governance tracking

Usage Patterns:
    # Manual adapter approach
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
    - Real-time performance monitoring and alerting
    - Production deployment patterns with scaling strategies
"""

from .adapter import GenOpsGriptapeAdapter, GriptapeRequest
from .cost_aggregator import GriptapeCostAggregator, GriptapeCostSummary
from .workflow_monitor import GriptapeWorkflowMonitor, GriptapeStructureMetrics
from .registration import auto_instrument, instrument_griptape

# Convenience functions for common patterns
def track_agent(agent_id: str, **kwargs):
    """Convenience function for tracking Agent execution."""
    from .adapter import GenOpsGriptapeAdapter
    adapter = GenOpsGriptapeAdapter(**kwargs)
    return adapter.track_agent(agent_id)

def track_pipeline(pipeline_id: str, **kwargs):
    """Convenience function for tracking Pipeline execution."""
    from .adapter import GenOpsGriptapeAdapter
    adapter = GenOpsGriptapeAdapter(**kwargs)
    return adapter.track_pipeline(pipeline_id)

def track_workflow(workflow_id: str, **kwargs):
    """Convenience function for tracking Workflow execution."""
    from .adapter import GenOpsGriptapeAdapter
    adapter = GenOpsGriptapeAdapter(**kwargs)
    return adapter.track_workflow(workflow_id)

__all__ = [
    # Main adapter and request classes
    "GenOpsGriptapeAdapter",
    "GriptapeRequest",
    
    # Cost aggregation
    "GriptapeCostAggregator", 
    "GriptapeCostSummary",
    
    # Performance monitoring
    "GriptapeWorkflowMonitor",
    "GriptapeStructureMetrics",
    
    # Auto-instrumentation
    "auto_instrument",
    "instrument_griptape",
    
    # Convenience functions
    "track_agent",
    "track_pipeline", 
    "track_workflow",
]

# Version info
__version__ = "0.1.0"
__author__ = "GenOps AI Contributors"
__description__ = "GenOps governance integration for Griptape AI framework"