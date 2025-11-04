# Hugging Face Integration Guide

Comprehensive integration guide for GenOps AI governance with Hugging Face. This guide covers all features, advanced use cases, and production deployment patterns.

## 🧭 Quick Navigation

**First Time Setup?** → [Installation & Setup](#installation--setup) → [Validation](#validation) → [Basic Examples](#automatic-provider-detection)  
**Advanced Features?** → [Context Managers](#advanced-context-manager-patterns-new) → [Performance Features](#performance-features-new) → [Enterprise Workflows](#production-workflow-context)  
**Production Deployment?** → [Environment Config](#environment-configuration) → [Performance Tuning](#performance-features-new) → [Troubleshooting](#troubleshooting)

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Core Features](#core-features)
- [Integration Patterns](#integration-patterns)
- [Multi-Provider Support](#multi-provider-support)
- [Cost Intelligence](#cost-intelligence)
- [Governance & Attribution](#governance--attribution)
- [AI Task Coverage](#ai-task-coverage)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

The GenOps Hugging Face integration provides comprehensive AI governance, cost intelligence, and observability for applications using Hugging Face Hub, including models from OpenAI, Anthropic, Cohere, Meta, Mistral, and Google accessed through Hugging Face.

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Your App     │ -> │   GenOps Layer   │ -> │  Hugging Face API   │
│                │    │                  │    │                     │
│ - Zero changes │    │ - Cost tracking  │    │ - OpenAI models     │
│ - Same API     │    │ - Governance     │    │ - Anthropic models  │
│ - Full features│    │ - Performance    │    │ - Hub models        │
└─────────────────┘    │ - Observability  │    │ - Image generation  │
                       └──────────────────┘    └─────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  OpenTelemetry       │
                    │                      │
                    │ -> Datadog           │
                    │ -> Grafana           │
                    │ -> Custom Backend    │
                    └──────────────────────┘
```

### GenOps Integration Flow

```
1. instrument_huggingface()     ┌─→ Automatic cost calculation
   │                           │
   ▼                           │
2. Your HF API call ──────────┼─→ Provider detection
   │                           │
   ▼                           │
3. GenOps intercepts ─────────┼─→ Governance attribution  
   │                           │
   ▼                           │
4. Enhanced with telemetry ───┼─→ Performance monitoring
   │                           │
   ▼                           │
5. Response + observability ──┘─→ OpenTelemetry export
```

### Key Benefits

- **🤗 Universal Hugging Face Support**: Works with Hub models and third-party providers
- **💰 Multi-Provider Cost Intelligence**: Unified cost tracking across all providers
- **🏛️ Enterprise Governance**: Team, project, and customer cost attribution
- **📊 OpenTelemetry Native**: Seamless integration with existing observability stacks
- **🚀 Zero-Code Setup**: Auto-instrumentation with no application changes
- **🎯 Progressive Complexity**: 5-minute quickstart to 2-hour enterprise deployment

## Installation & Setup

### Installation Options

```bash
# Recommended: Install with Hugging Face support
pip install genops-ai[huggingface]

# Or install components separately
pip install genops-ai huggingface_hub

# Development installation
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI
pip install -e ".[huggingface]"
```

### Optional Dependencies

```bash
# AI/ML enhancements
pip install torch transformers datasets accelerate

# Observability integrations
pip install opentelemetry-exporter-datadog
pip install opentelemetry-exporter-jaeger  
pip install opentelemetry-exporter-prometheus
```

### Environment Configuration

```bash
# Optional but recommended
export HF_TOKEN="your-hugging-face-token"

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="my-ai-application"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_RESOURCE_ATTRIBUTES="environment=production,version=1.0.0"

# GenOps configuration
export GENOPS_ENVIRONMENT="production"
export GENOPS_PROJECT="my-ai-project"

# NEW! Performance and Production Configuration
export GENOPS_SAMPLING_RATE="1.0"              # Full sampling (0.0-1.0)
export GENOPS_ASYNC_EXPORT="true"              # Async telemetry export
export GENOPS_BATCH_SIZE="100"                 # Telemetry batch size
export GENOPS_EXPORT_TIMEOUT="5"               # Export timeout (seconds)

# NEW! Circuit Breaker Configuration
export GENOPS_CIRCUIT_BREAKER="true"           # Enable circuit breaker
export GENOPS_CB_THRESHOLD="5"                 # Failure threshold
export GENOPS_CB_WINDOW="60"                   # Reset window (seconds)
```

### Validation

```python
# Quick validation
from genops.providers.huggingface import quick_validate
quick_validate()

# Comprehensive validation with detailed diagnostics
from genops.providers.huggingface_validation import validate_huggingface_setup, print_huggingface_validation_result

result = validate_huggingface_setup()
print_huggingface_validation_result(result)
```

## Core Features

### Advanced Context Manager Patterns (NEW!)

GenOps now provides powerful context managers for advanced cost tracking and enterprise workflow orchestration:

#### Cost Context Manager

Track costs across multiple operations with automatic aggregation:

```python
from genops.providers.huggingface import create_huggingface_cost_context

# Multi-operation cost tracking
with create_huggingface_cost_context("cross_provider_analysis") as context:
    adapter = GenOpsHuggingFaceAdapter()
    
    # Multiple providers - costs automatically unified
    openai_result = adapter.text_generation(
        "Analyze market trends",
        model="gpt-3.5-turbo",
        team="analytics"
    )
    
    anthropic_result = adapter.text_generation(
        "Cross-validate the analysis",
        model="claude-3-haiku", 
        team="analytics"
    )
    
    hub_embeddings = adapter.feature_extraction(
        ["market trends", "validation"],
        model="sentence-transformers/all-MiniLM-L6-v2",
        team="analytics"
    )
    
    # Get comprehensive cost summary
    summary = context.get_current_summary()
    print(f"Total cost: ${summary.total_cost:.4f}")
    print(f"Providers: {list(summary.unique_providers)}")
    print(f"Models: {list(summary.unique_models)}")
    
    # Get detailed provider breakdown
    breakdown = summary.get_provider_breakdown()
    for provider, details in breakdown.items():
        print(f"{provider}: ${details['cost']:.4f} ({details['calls']} calls)")
```

#### Production Workflow Context

Enterprise-grade workflow orchestration with full governance:

```python
from genops.providers.huggingface import production_workflow_context

# Enterprise workflow with comprehensive tracking
with production_workflow_context(
    workflow_name="document_processing_pipeline",
    customer_id="enterprise-client-001",
    team="document-ai",
    project="intelligent-document-processing", 
    environment="production",
    
    # Enterprise governance attributes
    cost_center="R&D",
    compliance_level="SOC2",
    data_classification="confidential",
    budget_limit=10.00  # $10 budget limit
) as (workflow, workflow_id):
    
    adapter = GenOpsHuggingFaceAdapter()
    
    # Step 1: Document classification
    workflow.record_step("document_classification")
    classification = adapter.text_generation(
        f"Classify this document: {document_content}",
        model="microsoft/DialoGPT-medium",
        max_new_tokens=100
    )
    
    # Step 2: Information extraction
    workflow.record_step("information_extraction")
    extraction = adapter.text_generation(
        f"Extract key information: {document_content}",
        model="gpt-3.5-turbo",
        max_new_tokens=300
    )
    
    # Step 3: Quality validation
    workflow.record_step("quality_validation")
    validation = adapter.text_generation(
        f"Validate extraction: {extraction}",
        model="claude-3-haiku",
        max_new_tokens=150
    )
    
    # Record comprehensive metrics
    final_cost = workflow.get_current_cost_summary()
    workflow.record_performance_metric("total_workflow_cost", final_cost.total_cost, "USD")
    workflow.record_performance_metric("documents_processed", 1, "count")
    
    # Budget alerting
    if final_cost.total_cost > 5.00:
        workflow.record_alert(
            "high_cost_workflow",
            f"Workflow cost ${final_cost.total_cost:.4f} exceeds threshold",
            "warning"
        )
    
    print(f"✅ Workflow {workflow_id} completed")
    print(f"💰 Total cost: ${final_cost.total_cost:.4f}")
```

### Performance Features (NEW!)

Production-ready performance optimization:

#### Sampling Configuration
```python
import os
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

# Configure sampling for high-volume applications
os.environ['GENOPS_SAMPLING_RATE'] = '0.2'  # Sample 20% of operations

adapter = GenOpsHuggingFaceAdapter()

# Operations automatically respect sampling
for i in range(100):
    result = adapter.text_generation(f"Test {i}", model="microsoft/DialoGPT-medium")
    # Only ~20 operations generate telemetry, but all complete
```

#### Circuit Breaker Protection
```python
import os
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

# Configure circuit breaker
os.environ['GENOPS_CIRCUIT_BREAKER'] = 'true'
os.environ['GENOPS_CB_THRESHOLD'] = '3'
os.environ['GENOPS_CB_WINDOW'] = '60'

adapter = GenOpsHuggingFaceAdapter()

try:
    result = adapter.text_generation("Test", model="microsoft/DialoGPT-medium")
except Exception as e:
    if "Circuit breaker is open" in str(e):
        print("Service protected by circuit breaker")
        
        # Check circuit breaker status
        config = adapter.get_performance_config()
        print(f"Failures: {config['circuit_breaker_failures']}")
```

#### Async Telemetry Export
```python
import os
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

# Enable non-blocking telemetry export
os.environ['GENOPS_ASYNC_EXPORT'] = 'true'
os.environ['GENOPS_BATCH_SIZE'] = '50'

adapter = GenOpsHuggingFaceAdapter()

# Operation completes immediately, telemetry exported asynchronously
result = adapter.text_generation("Fast operation", model="microsoft/DialoGPT-medium")
# ✅ Operation returns immediately
# 📊 Telemetry exported in background
```

### Automatic Provider Detection

GenOps automatically detects the underlying provider for accurate cost calculation:

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

adapter = GenOpsHuggingFaceAdapter()

# Test provider detection
models = [
    "gpt-3.5-turbo",                    # → openai
    "claude-3-sonnet",                  # → anthropic  
    "command-r",                        # → cohere
    "mistral-7b-instruct",             # → mistral
    "microsoft/DialoGPT-medium",        # → huggingface_hub
]

for model in models:
    provider = adapter.detect_provider_for_model(model)
    print(f"{model} → {provider}")
```

### Cost Calculation Engine

Advanced cost calculation with provider-specific pricing:

```python
from genops.providers.huggingface_pricing import (
    calculate_huggingface_cost,
    compare_model_costs,
    get_cost_optimization_suggestions
)

# Calculate cost for specific operation
cost = calculate_huggingface_cost(
    provider="openai",
    model="gpt-3.5-turbo",
    input_tokens=1000,
    output_tokens=500,
    task="text-generation"
)

# Compare costs across models
comparison = compare_model_costs(
    ["gpt-3.5-turbo", "claude-3-haiku", "microsoft/DialoGPT-medium"],
    input_tokens=1000,
    output_tokens=500
)

# Get optimization suggestions
suggestions = get_cost_optimization_suggestions("gpt-4")
```

## Integration Patterns

### 1. Auto-Instrumentation Pattern (Zero-Code)

Perfect for adding governance to existing applications:

```python
from genops.providers.huggingface import instrument_huggingface

# Enable automatic instrumentation
instrument_huggingface()

# All existing Hugging Face code automatically tracked
from huggingface_hub import InferenceClient

client = InferenceClient()
response = client.text_generation(
    "Generate creative content",
    model="microsoft/DialoGPT-medium",
    
    # Just add governance attributes - everything else unchanged
    team="content-team",
    project="blog-automation", 
    customer_id="content-client-123"
)
```

### 2. Manual Adapter Pattern

For full control and advanced features:

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

# Create adapter with custom configuration
adapter = GenOpsHuggingFaceAdapter()

# Text generation with comprehensive governance
response = adapter.text_generation(
    prompt="Write a technical blog post intro about AI governance",
    model="microsoft/DialoGPT-medium",
    max_new_tokens=200,
    temperature=0.7,
    
    # Governance attributes
    team="engineering-team",
    project="technical-blog",
    customer_id="internal-content",
    environment="production",
    cost_center="engineering-ops",
    feature="blog-generation"
)
```

### 3. Context Manager Pattern

For tracking complex multi-step operations:

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
from genops import track

adapter = GenOpsHuggingFaceAdapter()

with track(
    operation_name="content_pipeline",
    team="content-team",
    project="automated-content",
    customer_id="content-client"
) as span:
    # Step 1: Generate outline
    outline = adapter.text_generation(
        "Create a blog post outline about sustainable technology",
        model="microsoft/DialoGPT-medium",
        max_new_tokens=150
    )
    span.set_attribute("content.outline_length", len(outline))
    
    # Step 2: Generate full content
    full_content = adapter.text_generation(
        f"Write a full blog post based on this outline: {outline}",
        model="gpt-3.5-turbo",  # Switch to different provider
        max_new_tokens=800
    )
    span.set_attribute("content.final_length", len(full_content))
    
    # Step 3: Generate metadata
    metadata = adapter.text_generation(
        f"Generate SEO metadata for: {full_content[:200]}",
        model="claude-3-haiku",  # Another provider
        max_new_tokens=100
    )
    
    # All costs automatically aggregated across providers
```

### 4. Decorator Pattern

For function-level instrumentation:

```python
from genops import track_usage

@track_usage(
    operation_name="customer_response_generation",
    team="support-team",
    project="customer-service-ai",
    customer_id="support-automation"
)
def generate_customer_response(inquiry: str, customer_context: str) -> str:
    from genops.providers.huggingface import create_instrumented_client
    
    client = create_instrumented_client()
    
    # Multi-step response generation
    sentiment = client.text_generation(
        f"Analyze sentiment: {inquiry}",
        model="microsoft/DialoGPT-medium",
        max_new_tokens=50
    )
    
    response = client.text_generation(
        f"Generate helpful response for {sentiment} inquiry: {inquiry}. Context: {customer_context}",
        model="gpt-3.5-turbo",
        max_new_tokens=200
    )
    
    return response

# Usage with automatic governance
response = generate_customer_response(
    "I'm having trouble with my subscription billing",
    "Premium customer since 2020"
)
```

## Multi-Provider Support

### Provider Coverage

GenOps supports all major AI providers accessible through Hugging Face:

| Provider | Models Supported | Cost Tracking | Special Features |
|----------|------------------|---------------|------------------|
| **OpenAI** | GPT-3.5, GPT-4, Embeddings, DALL-E | ✅ Accurate | Function calling support |
| **Anthropic** | Claude-3 Family | ✅ Accurate | Long context handling |
| **Cohere** | Command, Embed models | ✅ Accurate | Multilingual support |
| **Meta** | Llama-2, Llama-3 Family | ✅ Accurate | Open source models |
| **Mistral** | Mistral, Mixtral models | ✅ Accurate | European provider |
| **Google** | Gemma, Flan-T5 | ✅ Accurate | Research models |
| **Hugging Face Hub** | 200,000+ models | ✅ Optimized | Community models |

### Multi-Provider Operations

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

adapter = GenOpsHuggingFaceAdapter()

# Define multi-provider workflow
providers_workflow = [
    {
        "step": "initial_draft",
        "model": "microsoft/DialoGPT-medium",  # Hub model (cost-effective)
        "prompt": "Generate initial product description",
        "governance": {"feature": "draft-generation"}
    },
    {
        "step": "quality_enhancement", 
        "model": "gpt-3.5-turbo",            # OpenAI (higher quality)
        "prompt": "Enhance and polish this product description",
        "governance": {"feature": "content-enhancement"}
    },
    {
        "step": "final_review",
        "model": "claude-3-haiku",            # Anthropic (good reasoning)
        "prompt": "Review and suggest final improvements",
        "governance": {"feature": "content-review"}
    }
]

results = {}
total_cost = 0

for step_config in providers_workflow:
    if step_config["step"] == "initial_draft":
        result = adapter.text_generation(
            prompt=step_config["prompt"],
            model=step_config["model"],
            max_new_tokens=150,
            team="content-team",
            project="multi-provider-content",
            **step_config["governance"]
        )
    else:
        # Use previous result as input
        enhanced_prompt = f"{step_config['prompt']}: {results[prev_step]}"
        result = adapter.text_generation(
            prompt=enhanced_prompt,
            model=step_config["model"],
            max_new_tokens=200,
            team="content-team", 
            project="multi-provider-content",
            **step_config["governance"]
        )
    
    results[step_config["step"]] = result
    prev_step = step_config["step"]

# All costs automatically tracked and attributed across providers
```

## Cost Intelligence

### Real-Time Cost Optimization

```python
from genops.providers.huggingface_pricing import (
    compare_model_costs,
    get_cost_optimization_suggestions,
    calculate_huggingface_cost
)

class IntelligentModelSelector:
    """Intelligent model selection based on cost and quality requirements."""
    
    def __init__(self):
        self.model_tiers = {
            'basic': ['microsoft/DialoGPT-medium', 'mistral-7b-instruct'],
            'standard': ['gpt-3.5-turbo', 'claude-3-haiku'], 
            'premium': ['gpt-4', 'claude-3-sonnet'],
            'ultra': ['gpt-4-turbo', 'claude-3-opus']
        }
    
    def select_optimal_model(self, 
                           quality_requirement: str,
                           budget_per_operation: float,
                           input_tokens: int,
                           output_tokens: int) -> dict:
        
        candidates = self.model_tiers.get(quality_requirement, self.model_tiers['standard'])
        
        best_option = None
        best_cost = float('inf')
        
        for model in candidates:
            from genops.providers.huggingface_pricing import detect_model_provider
            provider = detect_model_provider(model)
            
            cost = calculate_huggingface_cost(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            if cost <= budget_per_operation and cost < best_cost:
                best_cost = cost
                best_option = {
                    'model': model,
                    'provider': provider,
                    'cost': cost,
                    'within_budget': True
                }
        
        return best_option or {
            'error': f'No models found within ${budget_per_operation:.6f} budget',
            'cheapest_option': candidates[-1] if candidates else None
        }

# Usage example
selector = IntelligentModelSelector()

# High-volume, cost-sensitive operation
budget_option = selector.select_optimal_model(
    quality_requirement='standard',
    budget_per_operation=0.005,
    input_tokens=200,
    output_tokens=100
)

print(f"Recommended: {budget_option.get('model')} at ${budget_option.get('cost', 0):.6f}")
```

### Cost Attribution and Billing

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
from dataclasses import dataclass
from typing import Dict

@dataclass
class CostSummary:
    """Cost summary for billing and reporting."""
    team_costs: Dict[str, float]
    customer_costs: Dict[str, float]  
    project_costs: Dict[str, float]
    provider_costs: Dict[str, float]
    total_cost: float

class CostTracker:
    """Track and aggregate costs across operations."""
    
    def __init__(self):
        self.operations = []
    
    def record_operation(self, governance_attrs: dict, cost: float, provider: str):
        """Record an operation with governance and cost data."""
        self.operations.append({
            'governance': governance_attrs,
            'cost': cost,
            'provider': provider,
            'timestamp': datetime.now()
        })
    
    def get_cost_summary(self) -> CostSummary:
        """Get aggregated cost summary for billing."""
        team_costs = {}
        customer_costs = {}
        project_costs = {}
        provider_costs = {}
        
        for op in self.operations:
            # Team attribution
            team = op['governance'].get('team', 'unknown')
            team_costs[team] = team_costs.get(team, 0) + op['cost']
            
            # Customer billing
            customer = op['governance'].get('customer_id', 'internal')
            customer_costs[customer] = customer_costs.get(customer, 0) + op['cost']
            
            # Project tracking
            project = op['governance'].get('project', 'unassigned') 
            project_costs[project] = project_costs.get(project, 0) + op['cost']
            
            # Provider costs
            provider = op['provider']
            provider_costs[provider] = provider_costs.get(provider, 0) + op['cost']
        
        return CostSummary(
            team_costs=team_costs,
            customer_costs=customer_costs,
            project_costs=project_costs,
            provider_costs=provider_costs,
            total_cost=sum(op['cost'] for op in self.operations)
        )

# Usage in production application
cost_tracker = CostTracker()
adapter = GenOpsHuggingFaceAdapter()

# Simulate operations with different governance contexts
operations = [
    {
        "prompt": "Generate marketing copy",
        "governance": {"team": "marketing", "customer_id": "client-A", "project": "campaign-q4"}
    },
    {
        "prompt": "Technical documentation", 
        "governance": {"team": "engineering", "customer_id": "internal", "project": "api-docs"}
    },
    {
        "prompt": "Customer support response",
        "governance": {"team": "support", "customer_id": "client-B", "project": "customer-service"}
    }
]

for op in operations:
    # Execute operation (simplified for example)
    cost = 0.002  # Would be calculated from actual operation
    provider = "openai"  # Would be detected
    
    cost_tracker.record_operation(op["governance"], cost, provider)

# Generate billing report
summary = cost_tracker.get_cost_summary()
print(f"Total costs: ${summary.total_cost:.4f}")
print(f"Customer billing: {summary.customer_costs}")
print(f"Team attribution: {summary.team_costs}")
```

## Governance & Attribution

### Governance Attribute Reference

GenOps supports comprehensive governance attributes for cost attribution and access control:

| Attribute | Purpose | Example | Required |
|-----------|---------|---------|----------|
| `team` | Team cost attribution | `"marketing-team"` | Recommended |
| `project` | Project-level tracking | `"product-launch-q4"` | Recommended |
| `customer_id` | Customer billing | `"enterprise-client-123"` | Recommended |
| `environment` | Environment segregation | `"production"` | Optional |
| `cost_center` | Financial reporting | `"marketing-ops"` | Optional |
| `feature` | Feature-level attribution | `"content-generation"` | Optional |
| `user_id` | User-level tracking | `"user-456"` | Optional |
| `experiment_id` | A/B testing | `"exp-content-v2"` | Optional |

### Advanced Governance Patterns

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
from genops.core.context import set_default_attributes

# Set organization-wide defaults
set_default_attributes(
    environment="production",
    cost_center="ai-operations", 
    model_version="v2.1"
)

adapter = GenOpsHuggingFaceAdapter()

# Hierarchical governance context
class GovernedAIService:
    """Service with built-in governance."""
    
    def __init__(self, service_team: str, service_project: str):
        self.service_team = service_team
        self.service_project = service_project
        self.adapter = GenOpsHuggingFaceAdapter()
    
    def generate_content(self, 
                        prompt: str,
                        customer_id: str,
                        feature: str,
                        **ai_params) -> str:
        """Generate content with automatic governance."""
        
        return self.adapter.text_generation(
            prompt=prompt,
            
            # Service-level governance (automatic)
            team=self.service_team,
            project=self.service_project,
            
            # Operation-level governance (provided)
            customer_id=customer_id,
            feature=feature,
            
            # AI parameters
            **ai_params
        )

# Usage with automatic governance inheritance
marketing_service = GovernedAIService(
    service_team="marketing-team",
    service_project="content-automation"
)

# All operations automatically inherit service governance
blog_post = marketing_service.generate_content(
    prompt="Write about sustainable technology trends",
    customer_id="tech-blog-client",
    feature="blog-generation",
    model="gpt-3.5-turbo",
    max_new_tokens=500
)

social_post = marketing_service.generate_content(
    prompt="Create social media post about our new feature",
    customer_id="social-media-client", 
    feature="social-content",
    model="microsoft/DialoGPT-medium",
    max_new_tokens=100
)

# Both operations attributed to marketing-team/content-automation automatically
```

## AI Task Coverage

### Text Generation

```python
adapter = GenOpsHuggingFaceAdapter()

# Creative content generation
creative_content = adapter.text_generation(
    prompt="Write a creative story about AI in the year 2030",
    model="microsoft/DialoGPT-medium",
    max_new_tokens=300,
    temperature=0.9,  # Higher creativity
    team="content-team",
    feature="creative-writing"
)

# Technical documentation
tech_docs = adapter.text_generation(
    prompt="Explain how to implement OAuth 2.0 authentication",
    model="gpt-3.5-turbo", 
    max_new_tokens=400,
    temperature=0.3,  # Lower temperature for factual content
    team="engineering-team",
    feature="documentation"
)

# Customer support responses
support_response = adapter.text_generation(
    prompt="Generate empathetic response for billing inquiry",
    model="claude-3-haiku",
    max_new_tokens=150,
    temperature=0.6,
    team="support-team",
    feature="customer-service"
)
```

### Chat Completions

```python
# Multi-turn conversation with context
messages = [
    {"role": "system", "content": "You are a helpful AI assistant for financial planning."},
    {"role": "user", "content": "I want to start saving for retirement. I'm 25 years old."},
    {"role": "assistant", "content": "Great time to start! At 25, you have time for compound growth. What's your current income?"},
    {"role": "user", "content": "I make about $60,000 per year. How much should I save?"}
]

chat_response = adapter.chat_completion(
    messages=messages,
    model="gpt-3.5-turbo",
    max_new_tokens=200,
    temperature=0.7,
    team="financial-services",
    customer_id="advisory-client-789",
    feature="retirement-planning"
)
```

### Embeddings and Feature Extraction

```python
# Document embeddings for search
documents = [
    "GenOps provides AI governance and cost tracking",
    "Hugging Face offers access to thousands of AI models", 
    "OpenTelemetry enables observability for distributed systems",
    "Cost optimization helps manage AI infrastructure expenses"
]

embeddings = adapter.feature_extraction(
    inputs=documents,
    model="sentence-transformers/all-MiniLM-L6-v2",
    team="search-team",
    project="document-search",
    feature="semantic-search"
)

print(f"Generated embeddings for {len(documents)} documents")
print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 'N/A'}")
```

### Text-to-Image Generation

```python
# Marketing visual generation
marketing_image = adapter.text_to_image(
    prompt="Professional office space with diverse team collaborating on AI projects",
    model="runwayml/stable-diffusion-v1-5",
    team="creative-team",
    project="marketing-visuals",
    customer_id="brand-assets",
    feature="campaign-imagery"
)

# Product mockup creation
product_mockup = adapter.text_to_image(
    prompt="Modern smartphone displaying productivity app with clean interface",
    model="runwayml/stable-diffusion-v1-5", 
    team="product-design",
    project="app-mockups",
    feature="concept-visualization"
)
```

## Production Deployment

### High-Volume Configuration

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
from concurrent.futures import ThreadPoolExecutor
import asyncio

class ProductionHuggingFaceService:
    """Production-ready Hugging Face service."""
    
    def __init__(self, 
                 max_workers: int = 10,
                 timeout: float = 30.0,
                 retry_attempts: int = 3):
        self.adapter = GenOpsHuggingFaceAdapter()
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate_batch(self, requests: list) -> list:
        """Process multiple requests concurrently."""
        
        async def process_request(request):
            for attempt in range(self.retry_attempts):
                try:
                    return await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(
                                self.adapter.text_generation,
                                **request
                            )
                        ),
                        timeout=self.timeout
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    if attempt == self.retry_attempts - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Process all requests concurrently
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

# Usage
service = ProductionHuggingFaceService(max_workers=5)

# Batch processing
requests = [
    {
        "prompt": f"Process customer feedback #{i}",
        "model": "microsoft/DialoGPT-medium",
        "max_new_tokens": 100,
        "team": "support-team",
        "customer_id": f"batch-{i//10}"
    }
    for i in range(50)
]

# results = await service.generate_batch(requests)
```

### Circuit Breaker Pattern

```python
import time
from datetime import datetime, timedelta
from typing import Optional

class CircuitBreakerHuggingFaceAdapter:
    """Hugging Face adapter with circuit breaker for resilience."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_duration: int = 60):
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        self.adapter = GenOpsHuggingFaceAdapter()
        
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.circuit_state = "closed"  # closed, open, half-open
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_state == "closed":
            return False
        
        if self.circuit_state == "open":
            # Check if timeout period has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_duration)):
                self.circuit_state = "half-open"
                return False
            return True
        
        return False  # half-open allows one test request
    
    def generate_with_circuit_breaker(self, **kwargs) -> Optional[str]:
        """Generate text with circuit breaker protection."""
        
        if self._is_circuit_open():
            raise Exception(f"Circuit breaker is open. Too many failures (threshold: {self.failure_threshold})")
        
        try:
            result = self.adapter.text_generation(**kwargs)
            
            # Success - reset failure count
            if self.circuit_state == "half-open":
                self.circuit_state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.circuit_state = "open"
                print(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

# Usage
resilient_adapter = CircuitBreakerHuggingFaceAdapter(
    failure_threshold=3,
    timeout_duration=60
)

try:
    response = resilient_adapter.generate_with_circuit_breaker(
        prompt="Generate response",
        model="microsoft/DialoGPT-medium",
        team="production-team"
    )
except Exception as e:
    print(f"Service unavailable: {e}")
```

### Monitoring and Alerting

```python
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class ServiceMetrics:
    """Production service metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0

class MonitoredHuggingFaceService:
    """Hugging Face service with comprehensive monitoring."""
    
    def __init__(self):
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        self.adapter = GenOpsHuggingFaceAdapter()
        self.metrics = ServiceMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Alert thresholds
        self.error_rate_threshold = 0.1  # 10%
        self.cost_per_hour_threshold = 50.0  # $50/hour
        self.latency_threshold = 5.0  # 5 seconds
    
    def generate_with_monitoring(self, **kwargs) -> str:
        """Generate with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            result = self.adapter.text_generation(**kwargs)
            
            # Record success metrics
            self.metrics.successful_requests += 1
            latency = time.time() - start_time
            self._update_metrics(latency, success=True)
            
            return result
            
        except Exception as e:
            # Record failure metrics
            self.metrics.failed_requests += 1
            latency = time.time() - start_time
            self._update_metrics(latency, success=False)
            
            self.logger.error(f"Generation failed: {e}")
            raise e
    
    def _update_metrics(self, latency: float, success: bool):
        """Update service metrics and check alerts."""
        self.metrics.total_requests += 1
        
        # Update average latency
        total_latency = self.metrics.avg_latency * (self.metrics.total_requests - 1) + latency
        self.metrics.avg_latency = total_latency / self.metrics.total_requests
        
        # Check alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """Check metrics against thresholds and generate alerts."""
        if self.metrics.total_requests > 0:
            error_rate = self.metrics.failed_requests / self.metrics.total_requests
            
            if error_rate > self.error_rate_threshold:
                self.logger.warning(f"High error rate: {error_rate:.2%} (threshold: {self.error_rate_threshold:.2%})")
        
        if self.metrics.avg_latency > self.latency_threshold:
            self.logger.warning(f"High latency: {self.metrics.avg_latency:.2f}s (threshold: {self.latency_threshold}s)")
    
    def get_health_status(self) -> dict:
        """Get service health status."""
        if self.metrics.total_requests == 0:
            return {"status": "unknown", "reason": "no requests processed"}
        
        error_rate = self.metrics.failed_requests / self.metrics.total_requests
        
        if error_rate > self.error_rate_threshold:
            return {"status": "unhealthy", "reason": f"error_rate: {error_rate:.2%}"}
        
        if self.metrics.avg_latency > self.latency_threshold:
            return {"status": "degraded", "reason": f"latency: {self.metrics.avg_latency:.2f}s"}
        
        return {"status": "healthy", "metrics": self.metrics}

# Usage
service = MonitoredHuggingFaceService()

# Process requests with monitoring
for i in range(10):
    try:
        response = service.generate_with_monitoring(
            prompt=f"Process request {i}",
            model="microsoft/DialoGPT-medium",
            team="production-team"
        )
    except Exception:
        pass  # Continue processing other requests

# Check health
health = service.get_health_status()
print(f"Service status: {health['status']}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Auto-Instrumentation Not Working

```python
# Check if instrumentation was successful
from genops.providers.huggingface import instrument_huggingface

result = instrument_huggingface()
if not result:
    print("Instrumentation failed - check huggingface_hub installation")

# Verify instrumentation is active
from huggingface_hub import InferenceClient
client = InferenceClient()

# Check if methods have GenOps wrappers
if hasattr(client, '_genops_original_text_generation'):
    print("✅ Auto-instrumentation is active")
else:
    print("❌ Auto-instrumentation not detected")
```

#### 2. Provider Detection Issues

```python
# Test provider detection manually
from genops.providers.huggingface_pricing import detect_model_provider

test_models = [
    "gpt-3.5-turbo",              # Should detect: openai
    "claude-3-sonnet",            # Should detect: anthropic
    "microsoft/DialoGPT-medium"   # Should detect: huggingface_hub
]

for model in test_models:
    provider = detect_model_provider(model)
    print(f"{model} → {provider}")
```

#### 3. Cost Calculation Problems

```python
# Test cost calculation directly
from genops.providers.huggingface_pricing import calculate_huggingface_cost

try:
    cost = calculate_huggingface_cost(
        provider="openai",
        model="gpt-3.5-turbo", 
        input_tokens=100,
        output_tokens=50
    )
    print(f"Cost calculation working: ${cost:.6f}")
except Exception as e:
    print(f"Cost calculation failed: {e}")
```

#### 4. Telemetry Not Appearing

```bash
# Check OpenTelemetry configuration
echo "Service Name: $OTEL_SERVICE_NAME"
echo "OTLP Endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT"

# Test with console exporter
export OTEL_EXPORTER_TYPE=console

# Run your application - telemetry should appear in console
```

#### 5. Rate Limiting and Timeouts

```python
# Configure timeouts and retries
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

# Create adapter with timeout configuration
adapter = GenOpsHuggingFaceAdapter(
    timeout=30,  # 30 second timeout
    max_retries=3  # Retry failed requests 3 times
)

# Use models with higher rate limits
response = adapter.text_generation(
    prompt="Test prompt",
    model="microsoft/DialoGPT-medium",  # Usually more stable than API models
    max_new_tokens=50
)
```

### Debugging Tools

#### Comprehensive Diagnostics

```python
def run_comprehensive_diagnostics():
    """Run full diagnostic suite."""
    
    print("🔍 Running GenOps Hugging Face Diagnostics")
    print("=" * 50)
    
    # 1. Import test
    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # 2. Validation test
    try:
        from genops.providers.huggingface_validation import validate_huggingface_setup
        result = validate_huggingface_setup()
        
        if result.is_valid:
            print("✅ Validation passed")
        else:
            print(f"⚠️ Validation issues: {result.summary['errors']} errors, {result.summary['warnings']} warnings")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # 3. Adapter creation test
    try:
        adapter = GenOpsHuggingFaceAdapter()
        if adapter.is_available():
            print("✅ Adapter creation successful")
        else:
            print("⚠️ Adapter created but dependencies missing")
    except Exception as e:
        print(f"❌ Adapter creation failed: {e}")
    
    # 4. Provider detection test
    try:
        test_models = ["gpt-3.5-turbo", "microsoft/DialoGPT-medium"]
        for model in test_models:
            provider = adapter.detect_provider_for_model(model)
            print(f"✅ Provider detection: {model} → {provider}")
    except Exception as e:
        print(f"❌ Provider detection failed: {e}")
    
    # 5. Cost calculation test
    try:
        from genops.providers.huggingface_pricing import calculate_huggingface_cost
        cost = calculate_huggingface_cost("openai", "gpt-3.5-turbo", 100, 50)
        print(f"✅ Cost calculation: ${cost:.6f}")
    except Exception as e:
        print(f"❌ Cost calculation failed: {e}")
    
    print("=" * 50)
    print("🎯 Diagnostics Complete")

# Run diagnostics
run_comprehensive_diagnostics()
```

### Performance Optimization

#### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_requests_efficiently(requests: list, max_workers: int = 5):
    """Process requests efficiently with concurrency control."""
    
    from genops.providers.huggingface import GenOpsHuggingFaceAdapter
    
    def process_single_request(request):
        adapter = GenOpsHuggingFaceAdapter()
        start_time = time.time()
        
        try:
            result = adapter.text_generation(**request)
            return {
                'success': True,
                'result': result,
                'duration': time.time() - start_time,
                'request_id': request.get('operation_id', 'unknown')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time,
                'request_id': request.get('operation_id', 'unknown')
            }
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_request = {
            executor.submit(process_single_request, req): req 
            for req in requests
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_request):
            result = future.result()
            results.append(result)
    
    # Calculate summary statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        avg_duration = sum(r['duration'] for r in successful) / len(successful)
    else:
        avg_duration = 0
    
    print(f"Batch Results: {len(successful)} successful, {len(failed)} failed")
    print(f"Average duration: {avg_duration:.2f}s")
    
    return results

# Usage
batch_requests = [
    {
        "prompt": f"Process batch item {i}",
        "model": "microsoft/DialoGPT-medium",
        "max_new_tokens": 50,
        "team": "batch-processing",
        "operation_id": f"batch-{i:03d}"
    }
    for i in range(20)
]

# results = process_requests_efficiently(batch_requests, max_workers=3)
```

## API Reference

### Core Classes

#### GenOpsHuggingFaceAdapter

Main adapter class for Hugging Face integration.

```python
class GenOpsHuggingFaceAdapter:
    def __init__(self, client: Any | None = None, **client_kwargs: Any)
    
    def text_generation(self, prompt: str, **kwargs) -> Any
    def chat_completion(self, messages: list, **kwargs) -> Any  
    def feature_extraction(self, inputs: Union[str, list], **kwargs) -> Any
    def text_to_image(self, prompt: str, **kwargs) -> Any
    
    def get_supported_tasks(self) -> list[str]
    def detect_provider_for_model(self, model: str) -> str
    def is_available(self) -> bool
```

#### Cost Calculation Functions

```python
def calculate_huggingface_cost(
    provider: str,
    model: str, 
    input_tokens: int = 0,
    output_tokens: int = 0,
    task: str = "text-generation"
) -> float

def compare_model_costs(
    models: list[str],
    input_tokens: int = 1000,
    output_tokens: int = 500,
    task: str = "text-generation" 
) -> Dict[str, Dict[str, any]]

def get_cost_optimization_suggestions(
    model: str,
    task: str = "text-generation"
) -> Dict[str, any]
```

#### Validation Functions

```python
def validate_huggingface_setup() -> ValidationResult
def print_huggingface_validation_result(result: ValidationResult) -> None
def quick_validate() -> bool
```

#### Auto-Instrumentation Functions

```python
def instrument_huggingface(**config) -> bool
def uninstrument_huggingface() -> bool
def create_instrumented_client(**client_kwargs) -> GenOpsHuggingFaceAdapter
```

### Governance Attributes

All GenOps Hugging Face methods support these governance attributes:

- `team: str` - Team cost attribution
- `project: str` - Project-level tracking  
- `customer_id: str` - Customer billing attribution
- `environment: str` - Environment segregation
- `cost_center: str` - Financial reporting alignment
- `feature: str` - Feature-level attribution
- `user_id: str` - User-level tracking
- `experiment_id: str` - A/B testing identification

### Supported AI Tasks

- `text-generation` - General text generation
- `chat-completion` - Chat and conversational AI
- `feature-extraction` - Embeddings and feature extraction
- `text-to-image` - Image generation from text
- `speech-to-text` - Audio transcription
- `text-to-speech` - Speech synthesis
- `image-classification` - Image classification tasks
- `sentiment-analysis` - Sentiment analysis
- `summarization` - Text summarization
- `translation` - Language translation
- `question-answering` - Q&A systems

---

This comprehensive integration guide covers all aspects of using GenOps with Hugging Face. For additional support, examples, and community resources, visit our [GitHub repository](https://github.com/KoshiHQ/GenOps-AI) and [documentation site](https://docs.genops.ai).