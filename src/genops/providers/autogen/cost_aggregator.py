#!/usr/bin/env python3
"""
AutoGen Multi-Provider Cost Aggregation for GenOps Governance

Comprehensive cost tracking and optimization for AutoGen multi-agent systems
across multiple LLM providers (OpenAI, Anthropic, Google, etc.).

Usage:
    from genops.providers.autogen.cost_aggregator import AutoGenCostAggregator, create_autogen_cost_context
    
    aggregator = AutoGenCostAggregator(
        team="ai-research",
        project="multi-agent-conversations",
        daily_budget_limit=100.0
    )
    
    # Context manager for cost tracking
    with create_autogen_cost_context("user-assistant-chat") as context:
        context.add_agent_interaction("assistant", "openai", "gpt-4", 150, 50)
        context.add_agent_interaction("user", "anthropic", "claude-3", 100, 75)
        print(f"Conversation cost: ${context.get_total_cost():.6f}")

Features:
    - Multi-provider cost aggregation (OpenAI, Anthropic, Google, Bedrock, etc.)
    - Conversation-level cost tracking with agent attribution
    - Real-time budget monitoring and alerting
    - Cost optimization recommendations based on usage patterns
    - Provider-specific cost calculations with accurate pricing
    - Enterprise cost reporting with team/project attribution
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

# Provider type enumeration
class ProviderType(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    COHERE = "cohere"
    MISTRAL = "mistral"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    GROQ = "groq"
    UNKNOWN = "unknown"


@dataclass
class AgentCostEntry:
    """Cost entry for a single agent interaction."""
    agent_name: str
    provider: ProviderType
    model: str
    input_tokens: int
    output_tokens: int
    cost: Decimal
    timestamp: datetime
    conversation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationCostSummary:
    """Cost summary for an AutoGen conversation."""
    conversation_id: str
    total_cost: Decimal
    start_time: datetime
    end_time: Optional[datetime]
    agent_costs: Dict[str, Decimal]
    provider_costs: Dict[ProviderType, Decimal]
    model_costs: Dict[str, Decimal]
    total_tokens: int
    cost_entries: List[AgentCostEntry] = field(default_factory=list)


@dataclass
class ProviderCostSummary:
    """Cost summary for a specific provider."""
    provider: ProviderType
    total_cost: Decimal
    total_operations: int
    agents_used: Set[str]
    models_used: Set[str]
    total_input_tokens: int
    total_output_tokens: int
    avg_cost_per_operation: Decimal
    cost_by_model: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation for an agent."""
    agent_name: str
    current_provider: ProviderType
    recommended_provider: ProviderType
    potential_savings: Decimal
    confidence_score: float
    reasoning: str
    estimated_impact: str


@dataclass  
class CostAnalysisResult:
    """Comprehensive cost analysis result."""
    total_cost: Decimal
    time_period_hours: int
    analysis_timestamp: datetime
    cost_by_provider: Dict[ProviderType, Decimal]
    cost_by_agent: Dict[str, Decimal]
    cost_by_model: Dict[str, Decimal]
    provider_summaries: Dict[ProviderType, ProviderCostSummary]
    optimization_recommendations: List[CostOptimizationRecommendation]
    trends: Dict[str, Any] = field(default_factory=dict)


class AutoGenCostAggregator:
    """
    Multi-provider cost aggregation for AutoGen conversations.
    
    Tracks costs across all supported LLM providers with conversation-level
    attribution, real-time budget monitoring, and optimization recommendations.
    """
    
    # Provider pricing (USD per 1K tokens) - approximate rates
    PROVIDER_PRICING = {
        ProviderType.OPENAI: {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        },
        ProviderType.ANTHROPIC: {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        },
        ProviderType.GOOGLE: {
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
        },
        ProviderType.COHERE: {
            "command": {"input": 0.0015, "output": 0.002},
            "command-light": {"input": 0.0003, "output": 0.0006},
        }
    }
    
    def __init__(
        self,
        team: str,
        project: str,
        daily_budget_limit: float = 100.0,
        alert_threshold_percentage: float = 80.0
    ):
        """
        Initialize cost aggregator with team and budget configuration.
        
        Args:
            team: Team name for cost attribution
            project: Project name for cost attribution
            daily_budget_limit: Daily spending limit in USD
            alert_threshold_percentage: Budget alert threshold (0-100)
        """
        self.team = team
        self.project = project
        self.daily_budget_limit = Decimal(str(daily_budget_limit))
        self.alert_threshold = self.daily_budget_limit * Decimal(str(alert_threshold_percentage / 100))
        
        # Cost tracking
        self.conversation_summaries: Dict[str, ConversationCostSummary] = {}
        self.cost_entries: List[AgentCostEntry] = []
        self.daily_cost_tracker: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache for optimization recommendations
        self._recommendation_cache: Dict[str, List[CostOptimizationRecommendation]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)
        
        logger.info(
            f"Initialized AutoGen cost aggregator - Team: {team}, "
            f"Project: {project}, Daily budget: ${daily_budget_limit}"
        )

    def add_agent_interaction(
        self,
        agent_name: str,
        provider: Union[str, ProviderType],
        model: str,
        input_tokens: int,
        output_tokens: int,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentCostEntry:
        """
        Add cost entry for an agent interaction.
        
        Args:
            agent_name: Name of the agent
            provider: LLM provider used
            model: Model name used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            conversation_id: Conversation identifier
            metadata: Additional metadata
            
        Returns:
            AgentCostEntry: Created cost entry
        """
        with self._lock:
            # Normalize provider
            if isinstance(provider, str):
                try:
                    provider = ProviderType(provider.lower())
                except ValueError:
                    provider = ProviderType.UNKNOWN
            
            # Calculate cost
            cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
            
            # Create cost entry
            entry = AgentCostEntry(
                agent_name=agent_name,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                timestamp=datetime.now(),
                conversation_id=conversation_id,
                metadata=metadata or {}
            )
            
            self.cost_entries.append(entry)
            
            # Update conversation summary
            self._update_conversation_summary(entry)
            
            # Update daily cost tracking
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_cost_tracker[today] += cost
            
            # Check budget alerts
            self._check_budget_alert()
            
            logger.debug(
                f"Added cost entry: {agent_name} - ${cost:.6f} "
                f"({input_tokens + output_tokens} tokens via {provider.value}/{model})"
            )
            
            return entry

    def _calculate_cost(
        self,
        provider: ProviderType,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Decimal:
        """Calculate cost for a provider/model interaction."""
        if provider not in self.PROVIDER_PRICING:
            # Generic estimation for unknown providers
            return Decimal(str((input_tokens + output_tokens) * 0.001 / 1000))
        
        model_pricing = self.PROVIDER_PRICING[provider].get(model, {})
        if not model_pricing:
            # Use average pricing for provider if model not found
            all_models = self.PROVIDER_PRICING[provider].values()
            avg_input = sum(m.get("input", 0) for m in all_models) / len(all_models)
            avg_output = sum(m.get("output", 0) for m in all_models) / len(all_models)
            model_pricing = {"input": avg_input, "output": avg_output}
        
        input_cost = Decimal(str(model_pricing.get("input", 0))) * Decimal(str(input_tokens)) / 1000
        output_cost = Decimal(str(model_pricing.get("output", 0))) * Decimal(str(output_tokens)) / 1000
        
        total_cost = input_cost + output_cost
        return total_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def _update_conversation_summary(self, entry: AgentCostEntry):
        """Update conversation summary with new cost entry."""
        conversation_id = entry.conversation_id
        
        if conversation_id not in self.conversation_summaries:
            self.conversation_summaries[conversation_id] = ConversationCostSummary(
                conversation_id=conversation_id,
                total_cost=Decimal('0'),
                start_time=entry.timestamp,
                end_time=None,
                agent_costs={},
                provider_costs={},
                model_costs={},
                total_tokens=0,
                cost_entries=[]
            )
        
        summary = self.conversation_summaries[conversation_id]
        summary.total_cost += entry.cost
        summary.end_time = entry.timestamp
        summary.total_tokens += entry.input_tokens + entry.output_tokens
        summary.cost_entries.append(entry)
        
        # Update agent costs
        if entry.agent_name not in summary.agent_costs:
            summary.agent_costs[entry.agent_name] = Decimal('0')
        summary.agent_costs[entry.agent_name] += entry.cost
        
        # Update provider costs
        if entry.provider not in summary.provider_costs:
            summary.provider_costs[entry.provider] = Decimal('0')
        summary.provider_costs[entry.provider] += entry.cost
        
        # Update model costs
        if entry.model not in summary.model_costs:
            summary.model_costs[entry.model] = Decimal('0')
        summary.model_costs[entry.model] += entry.cost

    def _check_budget_alert(self):
        """Check if current spending approaches budget limits."""
        today = datetime.now().strftime("%Y-%m-%d")
        current_daily_cost = self.daily_cost_tracker[today]
        
        if current_daily_cost >= self.alert_threshold:
            logger.warning(
                f"Budget alert: Daily cost ${current_daily_cost:.2f} "
                f"exceeds {(current_daily_cost/self.daily_budget_limit*100):.1f}% "
                f"of ${self.daily_budget_limit} budget"
            )

    def get_conversation_summary(self, conversation_id: str) -> Optional[ConversationCostSummary]:
        """Get cost summary for a specific conversation."""
        with self._lock:
            return self.conversation_summaries.get(conversation_id)

    def get_daily_cost(self, date: str = None) -> Decimal:
        """Get total cost for a specific date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return self.daily_cost_tracker.get(date, Decimal('0'))

    def get_cost_analysis(self, time_period_hours: int = 24) -> CostAnalysisResult:
        """
        Get comprehensive cost analysis for a time period.
        
        Args:
            time_period_hours: Time period for analysis
            
        Returns:
            CostAnalysisResult: Comprehensive cost analysis
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            relevant_entries = [
                entry for entry in self.cost_entries
                if entry.timestamp >= cutoff_time
            ]
            
            # Aggregate costs
            total_cost = sum(entry.cost for entry in relevant_entries)
            cost_by_provider = defaultdict(lambda: Decimal('0'))
            cost_by_agent = defaultdict(lambda: Decimal('0'))
            cost_by_model = defaultdict(lambda: Decimal('0'))
            
            for entry in relevant_entries:
                cost_by_provider[entry.provider] += entry.cost
                cost_by_agent[entry.agent_name] += entry.cost
                cost_by_model[entry.model] += entry.cost
            
            # Generate provider summaries
            provider_summaries = {}
            for provider_type in cost_by_provider.keys():
                provider_entries = [e for e in relevant_entries if e.provider == provider_type]
                
                provider_summaries[provider_type] = ProviderCostSummary(
                    provider=provider_type,
                    total_cost=cost_by_provider[provider_type],
                    total_operations=len(provider_entries),
                    agents_used=set(e.agent_name for e in provider_entries),
                    models_used=set(e.model for e in provider_entries),
                    total_input_tokens=sum(e.input_tokens for e in provider_entries),
                    total_output_tokens=sum(e.output_tokens for e in provider_entries),
                    avg_cost_per_operation=cost_by_provider[provider_type] / max(len(provider_entries), 1),
                    cost_by_model={
                        model: sum(e.cost for e in provider_entries if e.model == model)
                        for model in set(e.model for e in provider_entries)
                    }
                )
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(relevant_entries)
            
            return CostAnalysisResult(
                total_cost=total_cost,
                time_period_hours=time_period_hours,
                analysis_timestamp=datetime.now(),
                cost_by_provider=dict(cost_by_provider),
                cost_by_agent=dict(cost_by_agent),
                cost_by_model=dict(cost_by_model),
                provider_summaries=provider_summaries,
                optimization_recommendations=recommendations
            )

    def _generate_optimization_recommendations(
        self,
        entries: List[AgentCostEntry]
    ) -> List[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations based on usage patterns."""
        # Check cache validity
        if (self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl and
            self._recommendation_cache):
            return list(self._recommendation_cache.values())[0]
        
        recommendations = []
        agent_usage = defaultdict(list)
        
        # Group entries by agent
        for entry in entries:
            agent_usage[entry.agent_name].append(entry)
        
        # Analyze each agent's usage patterns
        for agent_name, agent_entries in agent_usage.items():
            if len(agent_entries) < 5:  # Need sufficient data
                continue
                
            # Calculate current cost and usage patterns
            current_cost = sum(e.cost for e in agent_entries)
            avg_input_tokens = sum(e.input_tokens for e in agent_entries) / len(agent_entries)
            avg_output_tokens = sum(e.output_tokens for e in agent_entries) / len(agent_entries)
            
            # Find current most-used provider
            provider_usage = defaultdict(int)
            for entry in agent_entries:
                provider_usage[entry.provider] += 1
            current_provider = max(provider_usage, key=provider_usage.get)
            
            # Simulate costs with other providers
            best_alternative = None
            max_savings = Decimal('0')
            
            for alt_provider, pricing in self.PROVIDER_PRICING.items():
                if alt_provider == current_provider:
                    continue
                    
                # Use the most cost-effective model for the provider
                cheapest_model = min(pricing.keys(), 
                                   key=lambda m: pricing[m]["input"] + pricing[m]["output"])
                
                # Calculate potential cost with alternative
                alt_cost = self._calculate_cost(
                    alt_provider, cheapest_model, 
                    int(avg_input_tokens), int(avg_output_tokens)
                ) * len(agent_entries)
                
                savings = current_cost - alt_cost
                if savings > max_savings:
                    max_savings = savings
                    best_alternative = (alt_provider, cheapest_model)
            
            # Generate recommendation if significant savings possible
            if max_savings > current_cost * Decimal('0.1'):  # 10% savings threshold
                recommendations.append(CostOptimizationRecommendation(
                    agent_name=agent_name,
                    current_provider=current_provider,
                    recommended_provider=best_alternative[0],
                    potential_savings=max_savings,
                    confidence_score=0.8,  # Static confidence for now
                    reasoning=f"Could save ${max_savings:.4f} using {best_alternative[0].value}/{best_alternative[1]}",
                    estimated_impact=f"{(max_savings/current_cost*100):.1f}% cost reduction"
                ))
        
        # Update cache
        self._recommendation_cache[datetime.now().isoformat()] = recommendations
        self._cache_timestamp = datetime.now()
        
        return recommendations

    def reset_daily_costs(self):
        """Reset daily cost tracking (useful for testing)."""
        with self._lock:
            self.daily_cost_tracker.clear()
            logger.info("Reset daily cost tracking")

    def export_cost_data(self, format_type: str = "dict") -> Union[Dict, str]:
        """
        Export cost data in various formats.
        
        Args:
            format_type: Export format ("dict", "csv", "json")
            
        Returns:
            Cost data in requested format
        """
        with self._lock:
            data = {
                "team": self.team,
                "project": self.project,
                "total_conversations": len(self.conversation_summaries),
                "total_cost_entries": len(self.cost_entries),
                "daily_costs": {k: str(v) for k, v in self.daily_cost_tracker.items()},
                "conversations": {
                    conv_id: {
                        "total_cost": str(summary.total_cost),
                        "agent_costs": {k: str(v) for k, v in summary.agent_costs.items()},
                        "provider_costs": {k.value: str(v) for k, v in summary.provider_costs.items()},
                        "start_time": summary.start_time.isoformat(),
                        "end_time": summary.end_time.isoformat() if summary.end_time else None,
                        "total_tokens": summary.total_tokens
                    }
                    for conv_id, summary in self.conversation_summaries.items()
                }
            }
            
            if format_type == "dict":
                return data
            elif format_type == "json":
                import json
                return json.dumps(data, indent=2)
            elif format_type == "csv":
                # Simple CSV export of cost entries
                lines = ["agent_name,provider,model,input_tokens,output_tokens,cost,timestamp,conversation_id"]
                for entry in self.cost_entries:
                    lines.append(
                        f"{entry.agent_name},{entry.provider.value},{entry.model},"
                        f"{entry.input_tokens},{entry.output_tokens},{entry.cost},"
                        f"{entry.timestamp.isoformat()},{entry.conversation_id}"
                    )
                return "\n".join(lines)
            else:
                raise ValueError(f"Unsupported format: {format_type}")


# Context manager for conversation-level cost tracking
class AutoGenCostContext:
    """Context manager for tracking costs within an AutoGen conversation."""
    
    def __init__(self, conversation_id: str, aggregator: AutoGenCostAggregator):
        self.conversation_id = conversation_id
        self.aggregator = aggregator
        self.start_time = datetime.now()
        self.cost_entries = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Error in conversation {self.conversation_id}: {exc_val}")
            
    def add_agent_interaction(
        self,
        agent_name: str,
        provider: Union[str, ProviderType],
        model: str,
        input_tokens: int,
        output_tokens: int,
        **metadata
    ) -> AgentCostEntry:
        """Add cost entry for an agent interaction within this conversation."""
        entry = self.aggregator.add_agent_interaction(
            agent_name=agent_name,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            conversation_id=self.conversation_id,
            metadata=metadata
        )
        self.cost_entries.append(entry)
        return entry
        
    def get_total_cost(self) -> Decimal:
        """Get total cost for this conversation."""
        summary = self.aggregator.get_conversation_summary(self.conversation_id)
        return summary.total_cost if summary else Decimal('0')
        
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown for this conversation."""
        summary = self.aggregator.get_conversation_summary(self.conversation_id)
        if not summary:
            return {}
            
        return {
            "total_cost": str(summary.total_cost),
            "agent_costs": {k: str(v) for k, v in summary.agent_costs.items()},
            "provider_costs": {k.value: str(v) for k, v in summary.provider_costs.items()},
            "model_costs": {k: str(v) for k, v in summary.model_costs.items()},
            "total_tokens": summary.total_tokens,
            "duration_seconds": (summary.end_time - summary.start_time).total_seconds() if summary.end_time else None
        }


@contextmanager
def create_autogen_cost_context(conversation_id: str, **kwargs) -> AutoGenCostContext:
    """
    Create a cost tracking context for AutoGen conversations.
    
    Args:
        conversation_id: Unique identifier for the conversation
        **kwargs: Additional parameters for cost aggregator
        
    Yields:
        AutoGenCostContext: Context for tracking conversation costs
        
    Example:
        with create_autogen_cost_context("user-assistant") as context:
            context.add_agent_interaction("assistant", "openai", "gpt-4", 150, 50)
            print(f"Cost: ${context.get_total_cost():.6f}")
    """
    # Create a default aggregator if not provided
    team = kwargs.get('team', 'default-team')
    project = kwargs.get('project', 'autogen-conversation')
    budget_limit = kwargs.get('daily_budget_limit', 100.0)
    
    aggregator = AutoGenCostAggregator(
        team=team,
        project=project,
        daily_budget_limit=budget_limit
    )
    
    context = AutoGenCostContext(conversation_id, aggregator)
    
    try:
        yield context
    finally:
        # Context cleanup is handled in __exit__
        pass


# Convenience function for multi-provider cost tracking
def multi_provider_cost_tracking(
    conversation_id: str,
    interactions: List[Tuple[str, str, str, int, int]],  # (agent, provider, model, input_tokens, output_tokens)
    **kwargs
) -> Dict[str, Any]:
    """
    Track costs for multiple provider interactions in a single call.
    
    Args:
        conversation_id: Conversation identifier
        interactions: List of (agent_name, provider, model, input_tokens, output_tokens) tuples
        **kwargs: Additional parameters for cost aggregator
        
    Returns:
        Dict with cost breakdown and summary
        
    Example:
        interactions = [
            ("assistant", "openai", "gpt-4", 150, 50),
            ("critic", "anthropic", "claude-3-sonnet", 100, 75),
            ("summarizer", "google", "gemini-pro", 200, 25)
        ]
        
        result = multi_provider_cost_tracking("research-session", interactions)
        print(f"Total cost: ${result['total_cost']}")
    """
    with create_autogen_cost_context(conversation_id, **kwargs) as context:
        for agent_name, provider, model, input_tokens, output_tokens in interactions:
            context.add_agent_interaction(agent_name, provider, model, input_tokens, output_tokens)
        
        return context.get_cost_breakdown()


# Alias for CLAUDE.md compatibility
create_chain_cost_context = create_autogen_cost_context