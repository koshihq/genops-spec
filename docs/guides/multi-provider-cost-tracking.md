# Multi-Provider Cost Tracking Guide

**Unified cost attribution and optimization across AI providers**

This guide demonstrates how to track, attribute, and optimize costs when using multiple AI providers (OpenAI, Anthropic, AWS Bedrock, Google Gemini, etc.) in your applications.

---

## Table of Contents

- [Overview](#overview)
- [Why Multi-Provider Cost Tracking](#why-multi-provider-cost-tracking)
- [Architecture Patterns](#architecture-patterns)
- [Provider Orchestration](#provider-orchestration)
- [Cost Aggregation Strategies](#cost-aggregation-strategies)
- [Budget-Constrained Operations](#budget-constrained-operations)
- [Provider Selection & Optimization](#provider-selection--optimization)
- [Migration Cost Analysis](#migration-cost-analysis)
- [Production Best Practices](#production-best-practices)
- [Real-World Examples](#real-world-examples)

---

## Overview

### The Multi-Provider Challenge

Modern AI applications often use multiple providers for:
- **Reliability**: Fallback when primary provider fails
- **Cost optimization**: Route to cheapest provider for task
- **Performance**: Choose fastest provider for latency-sensitive operations
- **Compliance**: Use specific providers for regulatory requirements
- **Feature parity**: Leverage unique capabilities of each provider

**The problem:** How do you track costs, attribute usage, and optimize spend across all providers?

### GenOps Solution

GenOps AI provides **unified cost tracking** across providers:
- **Single governance layer**: Consistent cost attribution regardless of provider
- **Cross-provider aggregation**: Total costs across OpenAI, Anthropic, Bedrock, etc.
- **Budget enforcement**: Constrain costs across all providers
- **Optimization insights**: Compare costs and recommend cheaper alternatives

---

## Why Multi-Provider Cost Tracking

### Business Drivers

**1. Cost Transparency**
```
Question: "How much did our chatbot cost last month?"
Problem: OpenAI bills separately, Anthropic separately, Bedrock buried in AWS bill
Solution: GenOps aggregates all costs with unified governance attributes
```

**2. Budget Control**
```
Scenario: Team has $1000/month budget for AI operations
Problem: No way to enforce budget across multiple providers
Solution: GenOps tracks total spend and blocks operations when budget exceeded
```

**3. Cost Attribution**
```
Question: "What's the AI cost per customer?"
Problem: Customer requests span multiple providers
Solution: GenOps attributes costs to customer_id regardless of provider
```

**4. Optimization Opportunities**
```
Scenario: Using gpt-4 for all operations at $0.03/1K tokens
Alternative: Use Claude 3 Sonnet at $0.003/1K tokens for 90% of tasks (10x cheaper)
Solution: GenOps tracks costs by task type and recommends provider switches
```

### Technical Benefits

- **Unified telemetry**: Single stream of cost data to observability backend
- **Consistent attribution**: Same governance attributes (team, project, customer_id) across providers
- **Real-time visibility**: Immediate cost tracking as requests happen
- **Historical analysis**: Query and analyze costs across time and providers
- **Automated optimization**: Programmatic provider selection based on cost/performance

---

## Architecture Patterns

### Pattern 1: Unified Cost Adapter

**Single adapter tracks all providers:**

```python
from genops.providers.elastic import instrument_elastic

# Single adapter for all providers
adapter = instrument_elastic(
    elastic_url="http://localhost:9200",
    team="ml-platform",
    project="ai-chatbot",
    environment="production"
)

# Track OpenAI
from genops.providers.openai import instrument_openai
instrument_openai(
    team="ml-platform",
    project="ai-chatbot",
    elastic_adapter=adapter  # Reuse same adapter
)

# Track Anthropic
from genops.providers.anthropic import instrument_anthropic
instrument_anthropic(
    team="ml-platform",
    project="ai-chatbot",
    elastic_adapter=adapter  # Reuse same adapter
)

# Now all costs flow to single Elasticsearch index with unified governance
```

**Benefits:**
- Single source of truth for all AI costs
- Consistent governance attributes across providers
- Simplified querying and analysis

### Pattern 2: Provider-Specific Adapters with Aggregation

**Separate adapters per provider, aggregated in observability backend:**

```python
# OpenAI adapter
openai_adapter = instrument_elastic(
    elastic_url="http://localhost:9200",
    index_prefix="genops-openai",
    team="ml-platform"
)

# Anthropic adapter
anthropic_adapter = instrument_elastic(
    elastic_url="http://localhost:9200",
    index_prefix="genops-anthropic",
    team="ml-platform"
)

# Query both indexes in Elasticsearch
# Query: _index: genops-* | stats sum(genops.cost.total) by genops.team
```

**Benefits:**
- Provider-specific indexing and retention policies
- Easier to debug provider-specific issues
- Can route providers to different backends

### Pattern 3: Collector-Based Aggregation

**Send all provider telemetry to OpenTelemetry Collector for aggregation:**

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# All providers export to collector
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")

# Collector aggregates costs and exports to multiple backends
# See: OpenTelemetry Integration guide
```

**Benefits:**
- Centralized processing and filtering
- Can enrich data with additional attributes
- Route to multiple backends (Datadog, Elastic, Prometheus)

---

## Provider Orchestration

### Basic Multi-Provider Fallback

**Try providers in priority order until success:**

```python
from genops.providers.elastic import instrument_elastic
import logging

logger = logging.getLogger(__name__)

adapter = instrument_elastic(
    elastic_url="http://localhost:9200",
    team="customer-support",
    project="chatbot-v2"
)

class AIOrchestrator:
    """Coordinate AI requests across multiple providers."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.providers = ["openai", "anthropic", "bedrock"]

    def complete(self, prompt: str, customer_id: str) -> tuple[str, dict]:
        """
        Try providers in order until success, tracking all costs.

        Args:
            prompt: User prompt
            customer_id: Customer ID for cost attribution

        Returns:
            (response_text, cost_summary)
        """
        total_cost = 0.0
        attempts = []

        # Main operation span
        with self.adapter.track_ai_operation(
            "multi-provider-completion",
            operation_type="llm.completion",
            customer_id=customer_id
        ) as main_span:

            for provider_name in self.providers:
                try:
                    # Track each provider attempt
                    with self.adapter.track_ai_operation(
                        f"{provider_name}-attempt",
                        operation_type="llm.completion",
                        provider=provider_name
                    ) as provider_span:

                        # Call provider
                        response, cost = self._call_provider(provider_name, prompt)

                        # Record cost
                        self.adapter.record_cost(
                            span=provider_span,
                            cost=cost,
                            provider=provider_name,
                            model=self._get_model(provider_name)
                        )

                        total_cost += cost
                        attempts.append({
                            "provider": provider_name,
                            "status": "success",
                            "cost": cost
                        })

                        # Update main span
                        main_span.set_attribute("genops.cost.total", total_cost)
                        main_span.set_attribute("genops.successful_provider", provider_name)
                        main_span.set_attribute("genops.attempts", len(attempts))

                        logger.info(f"✓ {provider_name} succeeded - ${cost:.4f}")

                        return response, {
                            "total_cost": total_cost,
                            "successful_provider": provider_name,
                            "attempts": attempts,
                            "customer_id": customer_id
                        }

                except Exception as e:
                    logger.warning(f"✗ {provider_name} failed: {e}")
                    attempts.append({
                        "provider": provider_name,
                        "status": "failed",
                        "error": str(e)
                    })
                    continue

            # All providers failed
            main_span.set_attribute("genops.all_providers_failed", True)
            raise Exception(f"All providers failed after {len(attempts)} attempts")

    def _call_provider(self, provider: str, prompt: str) -> tuple[str, float]:
        """Call specific provider (implementation varies)."""
        if provider == "openai":
            return self._call_openai(prompt)
        elif provider == "anthropic":
            return self._call_anthropic(prompt)
        elif provider == "bedrock":
            return self._call_bedrock(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _call_openai(self, prompt: str) -> tuple[str, float]:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        # Calculate cost (example rates)
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = (tokens_in * 0.03 + tokens_out * 0.06) / 1000

        return response.choices[0].message.content, cost

    def _call_anthropic(self, prompt: str) -> tuple[str, float]:
        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Calculate cost
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = (tokens_in * 0.003 + tokens_out * 0.015) / 1000

        return response.content[0].text, cost

    def _call_bedrock(self, prompt: str) -> tuple[str, float]:
        import boto3
        import json

        bedrock = boto3.client('bedrock-runtime')

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        })

        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )

        result = json.loads(response['body'].read())

        # Calculate cost (Bedrock pricing)
        tokens_in = result['usage']['input_tokens']
        tokens_out = result['usage']['output_tokens']
        cost = (tokens_in * 0.003 + tokens_out * 0.015) / 1000

        return result['content'][0]['text'], cost

    def _get_model(self, provider: str) -> str:
        """Get model name for provider."""
        models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-sonnet-20240229",
            "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0"
        }
        return models.get(provider, "unknown")

# Usage
orchestrator = AIOrchestrator(adapter)

response, summary = orchestrator.complete(
    prompt="Summarize this customer inquiry: ...",
    customer_id="customer-123"
)

print(f"Response: {response}")
print(f"Total cost: ${summary['total_cost']:.4f}")
print(f"Provider: {summary['successful_provider']}")
print(f"Attempts: {len(summary['attempts'])}")
```

**Query costs in Elasticsearch:**
```kql
# Total cost per customer across all providers
genops.cost.total > 0
| stats sum(genops.cost.total) by genops.customer_id

# Success rate by provider
genops.successful_provider: *
| stats count() by genops.successful_provider

# Average attempts per request
genops.attempts: *
| stats avg(genops.attempts)
```

---

## Cost Aggregation Strategies

### Strategy 1: Real-Time Cost Tracking

**Track costs as operations happen:**

```python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class CostEvent:
    """Single cost event."""
    timestamp: float
    provider: str
    model: str
    cost: float
    customer_id: str
    operation_id: str

class RealTimeCostTracker:
    """Track and aggregate costs in real-time."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.events: List[CostEvent] = []

    def record_operation(self, provider: str, model: str, cost: float,
                        customer_id: str, operation_id: str):
        """Record a single cost event."""
        event = CostEvent(
            timestamp=time.time(),
            provider=provider,
            model=model,
            cost=cost,
            customer_id=customer_id,
            operation_id=operation_id
        )
        self.events.append(event)

        # Send to observability backend
        self.adapter.record_cost(
            cost=cost,
            provider=provider,
            model=model,
            customer_id=customer_id
        )

    def get_summary(self, window_seconds: int = 3600) -> Dict:
        """Get cost summary for recent time window."""
        now = time.time()
        recent_events = [
            e for e in self.events
            if now - e.timestamp <= window_seconds
        ]

        # Aggregate by provider
        by_provider = {}
        for event in recent_events:
            by_provider[event.provider] = by_provider.get(event.provider, 0) + event.cost

        # Aggregate by customer
        by_customer = {}
        for event in recent_events:
            by_customer[event.customer_id] = by_customer.get(event.customer_id, 0) + event.cost

        return {
            "total_cost": sum(e.cost for e in recent_events),
            "event_count": len(recent_events),
            "by_provider": by_provider,
            "by_customer": by_customer,
            "window_seconds": window_seconds
        }

# Usage
tracker = RealTimeCostTracker(adapter)

# Record operations as they happen
tracker.record_operation("openai", "gpt-4", 0.05, "customer-123", "op-1")
tracker.record_operation("anthropic", "claude-3-sonnet", 0.03, "customer-123", "op-2")
tracker.record_operation("bedrock", "claude-3-sonnet", 0.02, "customer-456", "op-3")

# Get recent summary
summary = tracker.get_summary(window_seconds=3600)  # Last hour
print(f"Total cost (last hour): ${summary['total_cost']:.4f}")
print(f"By provider: {summary['by_provider']}")
print(f"By customer: {summary['by_customer']}")
```

### Strategy 2: Periodic Batch Aggregation

**Aggregate costs periodically for reporting:**

```python
from datetime import datetime, timedelta
from typing import Dict, List
from elasticsearch import Elasticsearch

class BatchCostAggregator:
    """Aggregate costs from Elasticsearch periodically."""

    def __init__(self, es_url: str):
        self.es = Elasticsearch([es_url])

    def aggregate_daily_costs(self, date: datetime) -> Dict:
        """Aggregate all costs for a specific day."""
        start = date.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)

        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": start.isoformat(), "lt": end.isoformat()}}},
                        {"exists": {"field": "genops.cost.total"}}
                    ]
                }
            },
            "aggs": {
                "by_provider": {
                    "terms": {"field": "genops.provider.keyword"},
                    "aggs": {"total_cost": {"sum": {"field": "genops.cost.total"}}}
                },
                "by_team": {
                    "terms": {"field": "genops.team.keyword"},
                    "aggs": {"total_cost": {"sum": {"field": "genops.cost.total"}}}
                },
                "by_customer": {
                    "terms": {"field": "genops.customer_id.keyword"},
                    "aggs": {"total_cost": {"sum": {"field": "genops.cost.total"}}}
                }
            },
            "size": 0
        }

        result = self.es.search(index="genops-ai-*", body=query)

        return {
            "date": date.isoformat(),
            "total_cost": sum(
                bucket["total_cost"]["value"]
                for bucket in result["aggregations"]["by_provider"]["buckets"]
            ),
            "by_provider": {
                bucket["key"]: bucket["total_cost"]["value"]
                for bucket in result["aggregations"]["by_provider"]["buckets"]
            },
            "by_team": {
                bucket["key"]: bucket["total_cost"]["value"]
                for bucket in result["aggregations"]["by_team"]["buckets"]
            },
            "by_customer": {
                bucket["key"]: bucket["total_cost"]["value"]
                for bucket in result["aggregations"]["by_customer"]["buckets"]
            }
        }

    def generate_monthly_report(self, year: int, month: int) -> Dict:
        """Generate monthly cost report."""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        daily_costs = []
        current_date = start_date
        while current_date < end_date:
            daily_costs.append(self.aggregate_daily_costs(current_date))
            current_date += timedelta(days=1)

        # Aggregate monthly totals
        total_by_provider = {}
        total_by_team = {}
        total_by_customer = {}

        for day in daily_costs:
            for provider, cost in day["by_provider"].items():
                total_by_provider[provider] = total_by_provider.get(provider, 0) + cost

            for team, cost in day["by_team"].items():
                total_by_team[team] = total_by_team.get(team, 0) + cost

            for customer, cost in day["by_customer"].items():
                total_by_customer[customer] = total_by_customer.get(customer, 0) + cost

        return {
            "year": year,
            "month": month,
            "total_cost": sum(day["total_cost"] for day in daily_costs),
            "daily_costs": daily_costs,
            "by_provider": total_by_provider,
            "by_team": total_by_team,
            "by_customer": total_by_customer
        }

# Usage
aggregator = BatchCostAggregator("http://localhost:9200")

# Daily report
today = datetime.now()
daily_report = aggregator.aggregate_daily_costs(today)
print(f"Total cost today: ${daily_report['total_cost']:.2f}")

# Monthly report
monthly_report = aggregator.generate_monthly_report(2024, 1)
print(f"Total cost in January: ${monthly_report['total_cost']:.2f}")
print(f"By provider: {monthly_report['by_provider']}")
```

---

## Budget-Constrained Operations

### Budget Enforcement with Multi-Provider Fallback

**Enforce budget limits and fallback to cheaper providers:**

```python
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""
    name: str
    model: str
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    max_tokens: int
    call_function: callable

class BudgetConstrainedOrchestrator:
    """Orchestrate AI operations with strict budget enforcement."""

    def __init__(self, adapter, daily_budget: float):
        self.adapter = adapter
        self.daily_budget = daily_budget
        self.providers = [
            ProviderConfig("openai", "gpt-4", 0.03, 0.06, 8000, self._call_openai),
            ProviderConfig("anthropic", "claude-3-sonnet", 0.003, 0.015, 200000, self._call_anthropic),
            ProviderConfig("bedrock", "claude-3-haiku", 0.00025, 0.00125, 200000, self._call_bedrock)
        ]

    def estimate_cost(self, provider: ProviderConfig, prompt: str,
                     max_output_tokens: int = 1000) -> float:
        """Estimate cost for operation."""
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = max_output_tokens

        cost = (
            (input_tokens * provider.cost_per_1k_input / 1000) +
            (output_tokens * provider.cost_per_1k_output / 1000)
        )
        return cost

    def get_today_spend(self) -> float:
        """Query today's total spend from Elasticsearch."""
        # Implementation: Query ES for sum(genops.cost.total) WHERE date = today
        # For demo, return mock value
        return 45.67

    def get_remaining_budget(self) -> float:
        """Calculate remaining budget for today."""
        spent = self.get_today_spend()
        remaining = self.daily_budget - spent
        return max(0, remaining)

    def complete_with_budget(self, prompt: str, customer_id: str,
                            max_output_tokens: int = 1000) -> tuple[str, dict]:
        """
        Complete request while respecting budget constraints.

        Strategy:
        1. Check remaining budget
        2. Estimate cost for each provider
        3. Try cheapest provider that fits budget
        4. Fallback to cheaper options if needed
        """
        remaining_budget = self.get_remaining_budget()

        if remaining_budget <= 0:
            raise Exception(f"Daily budget ${self.daily_budget} exceeded")

        logger.info(f"Remaining budget: ${remaining_budget:.2f}")

        # Sort providers by cost (cheapest first)
        providers_with_estimates = [
            (provider, self.estimate_cost(provider, prompt, max_output_tokens))
            for provider in self.providers
        ]
        providers_with_estimates.sort(key=lambda x: x[1])

        # Try providers in order of cost
        for provider, estimated_cost in providers_with_estimates:
            if estimated_cost > remaining_budget:
                logger.warning(
                    f"Skipping {provider.name} - "
                    f"estimated ${estimated_cost:.4f} exceeds "
                    f"remaining ${remaining_budget:.2f}"
                )
                continue

            try:
                logger.info(
                    f"Trying {provider.name} - "
                    f"estimated cost: ${estimated_cost:.4f}"
                )

                # Track operation
                with self.adapter.track_ai_operation(
                    f"budget-constrained-{provider.name}",
                    operation_type="llm.completion",
                    customer_id=customer_id,
                    provider=provider.name
                ) as span:

                    # Call provider
                    response, actual_cost = provider.call_function(prompt, max_output_tokens)

                    # Record cost
                    self.adapter.record_cost(
                        span=span,
                        cost=actual_cost,
                        provider=provider.name,
                        model=provider.model
                    )

                    # Update span with budget info
                    span.set_attribute("genops.budget.remaining", remaining_budget)
                    span.set_attribute("genops.budget.estimated_cost", estimated_cost)
                    span.set_attribute("genops.budget.actual_cost", actual_cost)

                    return response, {
                        "provider": provider.name,
                        "model": provider.model,
                        "cost": actual_cost,
                        "estimated_cost": estimated_cost,
                        "remaining_budget": remaining_budget - actual_cost
                    }

            except Exception as e:
                logger.error(f"{provider.name} failed: {e}")
                continue

        raise Exception(
            f"No provider available within remaining budget ${remaining_budget:.2f}"
        )

    def _call_openai(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        # Implementation similar to earlier examples
        return "OpenAI response", 0.05

    def _call_anthropic(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        return "Claude response", 0.03

    def _call_bedrock(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        return "Bedrock response", 0.02

# Usage
orchestrator = BudgetConstrainedOrchestrator(
    adapter=adapter,
    daily_budget=100.0  # $100/day
)

response, summary = orchestrator.complete_with_budget(
    prompt="Analyze this customer feedback: ...",
    customer_id="customer-789"
)

print(f"Provider: {summary['provider']}")
print(f"Cost: ${summary['cost']:.4f} (estimated: ${summary['estimated_cost']:.4f})")
print(f"Remaining budget: ${summary['remaining_budget']:.2f}")
```

---

## Provider Selection & Optimization

### Intelligent Provider Selection

**Choose provider based on task complexity and cost:**

```python
from enum import Enum
from typing import Dict

class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = 1      # Classification, simple Q&A
    MODERATE = 2    # Summarization, extraction
    COMPLEX = 3     # Reasoning, analysis, generation

class IntelligentProviderSelector:
    """Select optimal provider based on task requirements."""

    def __init__(self, adapter):
        self.adapter = adapter

        # Provider capabilities and costs
        self.provider_matrix = {
            TaskComplexity.SIMPLE: [
                {"provider": "bedrock", "model": "claude-3-haiku", "cost_score": 1, "quality_score": 8},
                {"provider": "openai", "model": "gpt-3.5-turbo", "cost_score": 2, "quality_score": 7},
            ],
            TaskComplexity.MODERATE: [
                {"provider": "anthropic", "model": "claude-3-sonnet", "cost_score": 5, "quality_score": 9},
                {"provider": "openai", "model": "gpt-4", "cost_score": 8, "quality_score": 9},
            ],
            TaskComplexity.COMPLEX: [
                {"provider": "openai", "model": "gpt-4", "cost_score": 10, "quality_score": 10},
                {"provider": "anthropic", "model": "claude-3-opus", "cost_score": 12, "quality_score": 10},
            ]
        }

    def select_provider(self, complexity: TaskComplexity,
                       optimize_for: str = "cost") -> Dict:
        """
        Select optimal provider for task complexity.

        Args:
            complexity: Task complexity level
            optimize_for: "cost" or "quality"

        Returns:
            Provider configuration
        """
        candidates = self.provider_matrix[complexity]

        if optimize_for == "cost":
            # Choose cheapest option
            return min(candidates, key=lambda p: p["cost_score"])
        elif optimize_for == "quality":
            # Choose highest quality
            return max(candidates, key=lambda p: p["quality_score"])
        else:
            # Balance cost and quality
            return min(candidates, key=lambda p: p["cost_score"] / p["quality_score"])

    def complete_with_optimization(self, prompt: str, complexity: TaskComplexity,
                                   optimize_for: str = "cost") -> tuple[str, dict]:
        """Complete request with optimized provider selection."""

        # Select optimal provider
        provider_config = self.select_provider(complexity, optimize_for)

        logger.info(
            f"Selected {provider_config['provider']} "
            f"({provider_config['model']}) for {complexity.name} task "
            f"(optimizing for {optimize_for})"
        )

        # Track operation
        with self.adapter.track_ai_operation(
            "intelligent-completion",
            operation_type="llm.completion",
            provider=provider_config["provider"]
        ) as span:

            # Add optimization metadata
            span.set_attribute("genops.task_complexity", complexity.name)
            span.set_attribute("genops.optimization_goal", optimize_for)
            span.set_attribute("genops.selected_provider", provider_config["provider"])

            # Call provider (implementation varies)
            response, cost = self._call_provider(provider_config, prompt)

            # Record cost
            self.adapter.record_cost(
                span=span,
                cost=cost,
                provider=provider_config["provider"],
                model=provider_config["model"]
            )

            return response, {
                "provider": provider_config["provider"],
                "model": provider_config["model"],
                "cost": cost,
                "complexity": complexity.name,
                "optimization": optimize_for
            }

    def _call_provider(self, config: Dict, prompt: str) -> tuple[str, float]:
        # Implementation varies by provider
        return f"Response from {config['provider']}", 0.02

# Usage
selector = IntelligentProviderSelector(adapter)

# Simple task - optimize for cost
response1, summary1 = selector.complete_with_optimization(
    prompt="Classify this email as spam or not spam",
    complexity=TaskComplexity.SIMPLE,
    optimize_for="cost"
)
print(f"Simple task: {summary1['provider']} - ${summary1['cost']:.4f}")

# Complex task - optimize for quality
response2, summary2 = selector.complete_with_optimization(
    prompt="Analyze this legal contract and identify risks",
    complexity=TaskComplexity.COMPLEX,
    optimize_for="quality"
)
print(f"Complex task: {summary2['provider']} - ${summary2['cost']:.4f}")
```

---

## Migration Cost Analysis

### Compare Costs Across Providers

**Analyze costs for migrating between providers:**

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class MigrationScenario:
    """Provider migration scenario."""
    from_provider: str
    to_provider: str
    current_monthly_cost: float
    projected_monthly_cost: float
    cost_savings: float
    savings_percentage: float
    migration_effort: str  # "Low", "Medium", "High"

class MigrationCostAnalyzer:
    """Analyze costs for provider migration."""

    def __init__(self, es_url: str):
        self.es = Elasticsearch([es_url])

    def analyze_current_usage(self, days: int = 30) -> Dict:
        """
        Analyze current provider usage and costs.

        Returns:
            Usage breakdown by provider, model, and operation type
        """
        query = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": f"now-{days}d",
                        "lte": "now"
                    }
                }
            },
            "aggs": {
                "by_provider": {
                    "terms": {"field": "genops.provider.keyword"},
                    "aggs": {
                        "total_cost": {"sum": {"field": "genops.cost.total"}},
                        "total_requests": {"value_count": {"field": "genops.cost.total"}},
                        "by_model": {
                            "terms": {"field": "ai.model.name.keyword"},
                            "aggs": {"cost": {"sum": {"field": "genops.cost.total"}}}
                        }
                    }
                }
            },
            "size": 0
        }

        result = self.es.search(index="genops-ai-*", body=query)

        return {
            "period_days": days,
            "providers": [
                {
                    "name": bucket["key"],
                    "total_cost": bucket["total_cost"]["value"],
                    "requests": bucket["total_requests"]["value"],
                    "avg_cost_per_request": bucket["total_cost"]["value"] / bucket["total_requests"]["value"],
                    "models": {
                        model["key"]: model["cost"]["value"]
                        for model in bucket["by_model"]["buckets"]
                    }
                }
                for bucket in result["aggregations"]["by_provider"]["buckets"]
            ]
        }

    def simulate_migration(self, from_provider: str, to_provider: str,
                          cost_multiplier: float) -> MigrationScenario:
        """
        Simulate cost impact of migrating from one provider to another.

        Args:
            from_provider: Current provider
            to_provider: Target provider
            cost_multiplier: Cost ratio (e.g., 0.5 = 50% of current cost)
        """
        # Get current usage
        usage = self.analyze_current_usage(days=30)

        # Find current provider stats
        current_stats = next(
            (p for p in usage["providers"] if p["name"] == from_provider),
            None
        )

        if not current_stats:
            raise ValueError(f"No usage found for provider: {from_provider}")

        # Calculate projections
        current_monthly = current_stats["total_cost"]
        projected_monthly = current_monthly * cost_multiplier
        savings = current_monthly - projected_monthly
        savings_pct = (savings / current_monthly) * 100

        # Estimate migration effort
        request_count = current_stats["requests"]
        if request_count < 1000:
            effort = "Low"
        elif request_count < 10000:
            effort = "Medium"
        else:
            effort = "High"

        return MigrationScenario(
            from_provider=from_provider,
            to_provider=to_provider,
            current_monthly_cost=current_monthly,
            projected_monthly_cost=projected_monthly,
            cost_savings=savings,
            savings_percentage=savings_pct,
            migration_effort=effort
        )

    def recommend_migrations(self) -> List[MigrationScenario]:
        """Generate migration recommendations based on cost savings."""

        # Cost multipliers (example ratios)
        migration_scenarios = [
            ("openai", "anthropic", 0.6),  # Claude 40% cheaper for similar quality
            ("openai", "bedrock", 0.3),    # Bedrock 70% cheaper
            ("anthropic", "bedrock", 0.5),  # Bedrock 50% cheaper
        ]

        recommendations = []
        for from_prov, to_prov, multiplier in migration_scenarios:
            try:
                scenario = self.simulate_migration(from_prov, to_prov, multiplier)
                if scenario.cost_savings > 100:  # Only recommend if saves $100+
                    recommendations.append(scenario)
            except ValueError:
                continue

        # Sort by savings
        recommendations.sort(key=lambda s: s.cost_savings, reverse=True)

        return recommendations

# Usage
analyzer = MigrationCostAnalyzer("http://localhost:9200")

# Analyze current usage
usage = analyzer.analyze_current_usage(days=30)
print("Current Usage (last 30 days):")
for provider in usage["providers"]:
    print(f"  {provider['name']}: ${provider['total_cost']:.2f} ({provider['requests']} requests)")

# Get migration recommendations
recommendations = analyzer.recommend_migrations()
print("\nMigration Recommendations:")
for scenario in recommendations:
    print(f"\n{scenario.from_provider} → {scenario.to_provider}")
    print(f"  Current: ${scenario.current_monthly_cost:.2f}/month")
    print(f"  Projected: ${scenario.projected_monthly_cost:.2f}/month")
    print(f"  Savings: ${scenario.cost_savings:.2f}/month ({scenario.savings_percentage:.1f}%)")
    print(f"  Effort: {scenario.migration_effort}")
```

---

## Production Best Practices

### 1. Always Set Governance Attributes

```python
# Good: Consistent attribution across all providers
adapter = instrument_elastic(
    elastic_url="http://localhost:9200",
    team="ml-platform",              # Required
    project="ai-chatbot",            # Required
    customer_id="customer-123",      # If applicable
    environment="production",        # Required
    cost_center="engineering",       # For financial reporting
    feature="chat-support"          # Feature-level tracking
)
```

### 2. Implement Cost Alerting

```python
def check_budget_alerts(adapter, budget: float, alert_threshold: float = 0.8):
    """Alert when approaching budget limit."""
    spent = adapter.get_total_cost_today()
    remaining = budget - spent
    utilization = spent / budget

    if utilization >= alert_threshold:
        logger.warning(
            f"Budget alert: ${spent:.2f} / ${budget:.2f} "
            f"({utilization * 100:.1f}% utilized)"
        )

        # Send alert (email, Slack, PagerDuty, etc.)
        send_alert(f"AI budget {utilization * 100:.1f}% consumed")
```

### 3. Cache Responses to Reduce Costs

```python
from functools import lru_cache
import hashlib

class CachedOrchestrator:
    """Orchestrator with response caching."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.cache = {}

    def complete_with_cache(self, prompt: str, customer_id: str) -> tuple[str, dict]:
        """Complete with caching to avoid redundant API calls."""

        # Generate cache key
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()

        # Check cache
        if cache_key in self.cache:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return self.cache[cache_key], {"cost": 0.0, "source": "cache"}

        # Cache miss - call provider
        response, cost_summary = self.complete(prompt, customer_id)

        # Store in cache
        self.cache[cache_key] = response

        return response, {**cost_summary, "source": "provider"}
```

### 4. Monitor Cost Trends

```python
def analyze_cost_trends(es_url: str, days: int = 7):
    """Analyze cost trends over time."""
    es = Elasticsearch([es_url])

    query = {
        "query": {"range": {"@timestamp": {"gte": f"now-{days}d"}}},
        "aggs": {
            "daily_costs": {
                "date_histogram": {
                    "field": "@timestamp",
                    "calendar_interval": "day"
                },
                "aggs": {
                    "total_cost": {"sum": {"field": "genops.cost.total"}}
                }
            }
        },
        "size": 0
    }

    result = es.search(index="genops-ai-*", body=query)

    daily_costs = [
        {
            "date": bucket["key_as_string"],
            "cost": bucket["total_cost"]["value"]
        }
        for bucket in result["aggregations"]["daily_costs"]["buckets"]
    ]

    # Calculate trend
    if len(daily_costs) >= 2:
        first_day = daily_costs[0]["cost"]
        last_day = daily_costs[-1]["cost"]
        change_pct = ((last_day - first_day) / first_day) * 100 if first_day > 0 else 0

        logger.info(f"Cost trend ({days} days): {change_pct:+.1f}%")

        if abs(change_pct) > 20:
            logger.warning(f"Significant cost change detected: {change_pct:+.1f}%")

    return daily_costs
```

---

## Real-World Examples

### Example 1: Customer Support Chatbot

**Requirements:**
- Multiple customers with independent budgets
- Per-customer cost attribution
- Budget enforcement per customer
- Provider fallback for reliability

```python
class SupportChatbot:
    """Customer support chatbot with multi-provider support."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.customer_budgets = {
            "customer-123": 500.0,  # $500/month
            "customer-456": 1000.0,  # $1000/month
        }

    def handle_request(self, customer_id: str, message: str) -> str:
        """Handle customer support request."""

        # Check customer budget
        budget = self.customer_budgets.get(customer_id, 0)
        spent = self.get_customer_spend_this_month(customer_id)
        remaining = budget - spent

        if remaining <= 0:
            return "Budget exceeded. Please contact your account manager."

        # Create budget-constrained orchestrator
        orchestrator = BudgetConstrainedOrchestrator(
            adapter=self.adapter,
            daily_budget=remaining
        )

        try:
            # Complete with budget enforcement
            response, summary = orchestrator.complete_with_budget(
                prompt=message,
                customer_id=customer_id
            )

            logger.info(
                f"Customer {customer_id}: ${summary['cost']:.4f} "
                f"(${remaining - summary['cost']:.2f} remaining)"
            )

            return response

        except Exception as e:
            logger.error(f"Request failed for {customer_id}: {e}")
            return "Sorry, I'm having trouble processing your request right now."

    def get_customer_spend_this_month(self, customer_id: str) -> float:
        """Query customer spend from Elasticsearch."""
        # Implementation: Query ES for sum(genops.cost.total)
        # WHERE customer_id = X AND date >= start_of_month
        return 0.0  # Placeholder
```

### Example 2: Multi-Model Research Assistant

**Requirements:**
- Use best model for each task type
- Optimize costs while maintaining quality
- Track costs by research project

```python
class ResearchAssistant:
    """Research assistant with intelligent provider selection."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.selector = IntelligentProviderSelector(adapter)

    def research_query(self, query: str, project: str) -> Dict:
        """Handle research query with optimized provider selection."""

        # Determine task complexity
        complexity = self.classify_query_complexity(query)

        # Select optimal provider (optimize for quality for research)
        response, summary = self.selector.complete_with_optimization(
            prompt=query,
            complexity=complexity,
            optimize_for="quality"
        )

        # Track by project
        with self.adapter.track_ai_operation(
            "research-query",
            project=project
        ) as span:
            span.set_attribute("research.query_type", complexity.name)
            span.set_attribute("research.project", project)

        return {
            "response": response,
            "provider": summary["provider"],
            "cost": summary["cost"],
            "complexity": complexity.name
        }

    def classify_query_complexity(self, query: str) -> TaskComplexity:
        """Classify query complexity based on content."""
        # Simple heuristic (can be replaced with ML model)
        if len(query) < 100:
            return TaskComplexity.SIMPLE
        elif "analyze" in query.lower() or "compare" in query.lower():
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.MODERATE
```

---

## Next Steps

- **[Elastic Integration](../integrations/elastic.md)** - Complete Elasticsearch setup
- **[OpenTelemetry Integration](../integrations/opentelemetry.md)** - Cross-platform telemetry
- **[Example Code](../../examples/)** - Working implementations
- **[Production Readiness Checklist](../integrations/elastic.md#production-readiness-checklist)** - Production deployment guide

---

**Questions or issues?** Open an issue on [GitHub](https://github.com/KoshiHQ/GenOps-AI/issues).
