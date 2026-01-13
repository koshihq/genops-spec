# Anyscale Endpoints Integration Guide

Comprehensive guide for integrating Anyscale Endpoints with GenOps AI governance and telemetry.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Integration Patterns](#integration-patterns)
- [Multi-Model Support](#multi-model-support)
- [Cost Intelligence](#cost-intelligence)
- [Enterprise Governance](#enterprise-governance)
- [Production Deployment](#production-deployment)
- [Performance Optimization](#performance-optimization)
- [Observability Integration](#observability-integration)
- [Advanced Use Cases](#advanced-use-cases)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

GenOps provides comprehensive Anyscale Endpoints integration with:

- **Multi-model support**: Llama-2, Llama-3, Mistral, CodeLlama, and embedding models
- **Real-time cost tracking**: Token-level precision with client-side cost calculation
- **Enterprise governance**: Team, project, and customer-level cost attribution
- **Zero-code instrumentation**: Works with existing OpenAI SDK applications unchanged
- **OpenTelemetry native**: Exports to any OTLP-compatible observability platform
- **Cost optimization**: Model recommendations and alternative suggestions

### Architecture Overview

```
Application Code
      ↓
GenOps Anyscale Adapter
      ↓
Anyscale Endpoints API ← OpenAI-compatible interface
      ↓
OpenTelemetry Pipeline ← Rich governance telemetry
      ↓
Your Observability Platform ← Datadog, Grafana, Honeycomb, etc.
```

### Why Anyscale + GenOps?

**Anyscale Endpoints** provides managed LLM inference with:
- Production-scale infrastructure
- OpenAI-compatible API for easy migration
- Competitive pricing (often 50%+ cheaper than alternatives)
- High availability and reliability

**GenOps adds governance layer**:
- Per-customer cost attribution for billing
- Team and project-level budget tracking
- Real-time cost optimization recommendations
- Compliance and audit trails via OpenTelemetry

## Installation & Setup

### Quick Installation

```bash
# Core installation
pip install genops-ai

# Verify installation
python -c "from genops.providers.anyscale import instrument_anyscale; print('✅ GenOps Anyscale provider installed')"
```

### Anyscale API Key Setup

GenOps requires an Anyscale API key to access Endpoints:

```bash
# Get your API key from: https://console.anyscale.com/credentials

# Set environment variable
export ANYSCALE_API_KEY='your-api-key-here'

# Verify it's set
echo $ANYSCALE_API_KEY
```

### Environment Configuration

```bash
# Required
export ANYSCALE_API_KEY="your-api-key-here"
export ANYSCALE_BASE_URL="https://api.endpoints.anyscale.com/v1"  # Optional, this is the default

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="anyscale-ai-application"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps configuration
export GENOPS_ENVIRONMENT="production"
export GENOPS_PROJECT="anyscale-ai-project"
export GENOPS_TEAM="ml-engineering"

# Performance tuning (optional)
export GENOPS_SAMPLING_RATE="1.0"        # Full sampling (0.0-1.0)
export GENOPS_ASYNC_EXPORT="true"        # Non-blocking telemetry
export GENOPS_DEBUG="false"              # Debug logging
```

### Setup Validation

```python
from genops.providers.anyscale import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

if result.success:
    print("✅ Ready to start using GenOps with Anyscale!")
else:
    print("❌ Please resolve the issues above before continuing")
```

**Expected validation output:**

```
┌─────────────────────────────────────────────────────────┐
│ Anyscale Setup Validation                               │
├─────────────────────────────────────────────────────────┤
│ ✅ Dependencies: All required packages installed        │
│ ✅ Configuration: ANYSCALE_API_KEY set                  │
│ ✅ Connectivity: Anyscale API reachable                 │
│ ✅ Models: 12+ models available                         │
│ ✅ Pricing: Complete pricing database loaded            │
├─────────────────────────────────────────────────────────┤
│ Status: PASSED (Score: 100/100)                         │
└─────────────────────────────────────────────────────────┘
```

## Integration Patterns

### 1. Zero-Code Auto-Instrumentation

**Automatically instrument existing OpenAI SDK applications with zero code changes:**

```python
from genops.providers.anyscale import auto_instrument

# Enable automatic instrumentation with default governance attributes
auto_instrument(
    team="ml-research",
    project="chatbot",
    environment="production"
)

# Your existing OpenAI SDK code now automatically tracked!
import openai

client = openai.OpenAI(
    api_key=os.getenv("ANYSCALE_API_KEY"),
    base_url="https://api.endpoints.anyscale.com/v1"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    # Governance attributes automatically added
    customer_id="acme-corp"  # Per-request governance override
)

# Cost, tokens, and governance automatically tracked and exported via OpenTelemetry
```

**Benefits:**
- Zero refactoring required
- Existing applications work unchanged
- Governance attributes propagate automatically
- Full OpenTelemetry tracing with cost attribution

### 2. Manual Adapter Integration

**Full control over instrumentation with governance attributes:**

```python
from genops.providers.anyscale import instrument_anyscale

# Create adapter with default governance attributes
adapter = instrument_anyscale(
    team="ml-engineering",
    project="customer-support-bot",
    environment="production",
    cost_center="Engineering"
)

# Make a completion request with per-request governance
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this customer feedback..."}
    ],
    temperature=0.7,
    max_tokens=500,

    # Per-request governance attributes (override defaults)
    customer_id="customer-789",
    feature="feedback-analysis"
)

# Response includes usage and governance metadata
print(f"Response: {response['choices'][0]['message']['content']}")
print(f"Tokens used: {response['usage']['total_tokens']}")

# Calculate cost
from genops.providers.anyscale import calculate_completion_cost
cost = calculate_completion_cost(
    model="meta-llama/Llama-2-70b-chat-hf",
    input_tokens=response['usage']['prompt_tokens'],
    output_tokens=response['usage']['completion_tokens']
)
print(f"💰 Cost: ${cost:.6f}")
```

### 3. Context Manager Pattern

**Multi-operation workflows with unified governance:**

```python
from genops.providers.anyscale import instrument_anyscale

adapter = instrument_anyscale(
    team="data-science",
    project="analytics-pipeline"
)

# Context manager for workflow-level governance
with adapter.governance_context(
    customer_id="enterprise-client",
    feature="document-processing",
    workflow_id="doc-proc-12345"
) as context:

    # Step 1: Classify document
    classification = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",  # Cheaper model for classification
        messages=[{"role": "user", "content": f"Classify: {document_text[:100]}"}],
        max_tokens=50
    )

    # Step 2: Extract entities (if needed)
    if needs_extraction(classification):
        entities = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",  # More powerful model
            messages=[{"role": "user", "content": f"Extract entities: {document_text}"}],
            max_tokens=300
        )

    # Step 3: Summarize
    summary = adapter.completion_create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"role": "user", "content": f"Summarize: {document_text}"}],
        max_tokens=200
    )

# All operations automatically attributed to customer, feature, and workflow
# Total cost aggregated and exported to observability platform
```

## Multi-Model Support

### Supported Models

GenOps Anyscale integration supports 12+ models across multiple categories:

#### Chat Completion Models

**Llama-2 Series:**
```python
models = [
    "meta-llama/Llama-2-70b-chat-hf",    # $1.00/M tokens
    "meta-llama/Llama-2-13b-chat-hf",    # $0.25/M tokens
    "meta-llama/Llama-2-7b-chat-hf",     # $0.15/M tokens
]
```

**Llama-3 Series:**
```python
models = [
    "meta-llama/Meta-Llama-3-70B-Instruct",  # $1.00/M tokens
    "meta-llama/Meta-Llama-3-8B-Instruct",   # $0.15/M tokens
]
```

**Mistral Series:**
```python
models = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",      # $0.50/M tokens
    "mistralai/Mistral-7B-Instruct-v0.1",        # $0.15/M tokens
    "mistralai/Mistral-7B-Instruct-v0.2",        # $0.15/M tokens
]
```

**CodeLlama Series:**
```python
models = [
    "codellama/CodeLlama-70b-Instruct-hf",   # $1.00/M tokens
    "codellama/CodeLlama-34b-Instruct-hf",   # $0.80/M tokens
]
```

#### Embedding Models

```python
embedding_models = [
    "thenlper/gte-large",           # $0.05/M tokens
    "BAAI/bge-large-en-v1.5",       # $0.05/M tokens
]
```

### Model Comparison and Selection

```python
from genops.providers.anyscale import AnyscalePricing

pricing = AnyscalePricing()

# Get pricing for specific model
model_info = pricing.get_model_info("meta-llama/Llama-2-70b-chat-hf")
print(f"Model: {model_info.model_name}")
print(f"Input cost: ${model_info.input_cost_per_million}/M tokens")
print(f"Output cost: ${model_info.output_cost_per_million}/M tokens")
print(f"Context window: {model_info.context_window} tokens")

# Get cost-effective alternatives
alternatives = pricing.get_model_alternatives("meta-llama/Llama-2-70b-chat-hf")
print("\n💡 Cost-effective alternatives:")
for model, cost_ratio, description in alternatives:
    print(f"   {model}: {description}")

# Output:
# meta-llama/Llama-2-13b-chat-hf: 75% cheaper, good for most tasks
# meta-llama/Llama-2-7b-chat-hf: 85% cheaper, best for simple tasks
# mistralai/Mistral-7B-Instruct-v0.1: 85% cheaper, alternative architecture
```

### Multi-Model Workflows

```python
# Route by task complexity
def select_model(task_complexity: str) -> str:
    """Cost-optimized model selection."""
    if task_complexity == "simple":
        return "meta-llama/Llama-2-7b-chat-hf"      # $0.15/M
    elif task_complexity == "medium":
        return "meta-llama/Llama-2-13b-chat-hf"     # $0.25/M
    elif task_complexity == "complex":
        return "meta-llama/Llama-2-70b-chat-hf"     # $1.00/M
    else:
        return "mistralai/Mistral-7B-Instruct-v0.1"  # Default

# Example: Adaptive model selection
adapter = instrument_anyscale(team="optimization-team")

for query in user_queries:
    complexity = estimate_complexity(query)
    model = select_model(complexity)

    response = adapter.completion_create(
        model=model,
        messages=[{"role": "user", "content": query}],
        customer_id=query.customer_id
    )

    # Cost automatically tracked per customer and model
```

## Cost Intelligence

### Real-Time Cost Tracking

```python
from genops.providers.anyscale import instrument_anyscale, calculate_completion_cost

adapter = instrument_anyscale(
    team="finance-ai",
    project="cost-monitoring"
)

# Make request
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "Analyze quarterly revenue..."}],
    max_tokens=500
)

# Calculate cost
cost = calculate_completion_cost(
    model="meta-llama/Llama-2-70b-chat-hf",
    input_tokens=response['usage']['prompt_tokens'],
    output_tokens=response['usage']['completion_tokens']
)

print(f"📊 Token Usage:")
print(f"   Input: {response['usage']['prompt_tokens']} tokens")
print(f"   Output: {response['usage']['completion_tokens']} tokens")
print(f"   Total: {response['usage']['total_tokens']} tokens")
print(f"💰 Cost: ${cost:.6f}")
```

### Cost Attribution

**Team-Level Attribution:**
```python
# All costs automatically attributed to team
adapter = instrument_anyscale(team="data-science-team")

response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "..."}]
)

# OpenTelemetry span includes: genops.team="data-science-team"
```

**Project-Level Attribution:**
```python
adapter = instrument_anyscale(
    team="ml-engineering",
    project="customer-support-bot"
)

# Costs attributed to project
response = adapter.completion_create(...)

# OpenTelemetry span includes:
#   genops.team="ml-engineering"
#   genops.project="customer-support-bot"
```

**Customer-Level Attribution:**
```python
adapter = instrument_anyscale(team="saas-platform")

# Per-customer cost tracking for billing
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[...],
    customer_id="enterprise-client-123"
)

# Query your observability platform to aggregate costs per customer:
# SUM(genops.anyscale.cost.total) WHERE genops.customer_id="enterprise-client-123"
```

### Cost Optimization Strategies

**1. Model Selection by Task:**
```python
# Use cheaper models for simple tasks
simple_tasks = ["classification", "routing", "validation"]
complex_tasks = ["analysis", "generation", "reasoning"]

model = (
    "meta-llama/Llama-2-7b-chat-hf" if task in simple_tasks
    else "meta-llama/Llama-2-70b-chat-hf"
)

# Potential savings: 85% for simple tasks
```

**2. Max Tokens Optimization:**
```python
# Set appropriate max_tokens to avoid waste
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "Yes or no: ..."}],
    max_tokens=10  # Don't pay for unused tokens
)
```

**3. Batch Processing:**
```python
# Process multiple items in single request
batch_prompt = "Classify each of the following:\n" + "\n".join(items)

response = adapter.completion_create(
    model="meta-llama/Llama-2-13b-chat-hf",
    messages=[{"role": "user", "content": batch_prompt}],
    max_tokens=len(items) * 50
)

# Cost per item reduced by sharing prompt overhead
```

**4. Caching Strategy:**
```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_completion(prompt_hash: str, model: str):
    """Cache identical prompts to avoid redundant API calls."""
    response = adapter.completion_create(
        model=model,
        messages=[{"role": "user", "content": prompt_hash}]
    )
    return response

# Use cache
prompt = "What is the capital of France?"
prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
result = cached_completion(prompt_hash, "meta-llama/Llama-2-7b-chat-hf")
```

## Enterprise Governance

### Multi-Tenant Cost Attribution

```python
from genops.providers.anyscale import instrument_anyscale

# SaaS application with multiple customers
adapter = instrument_anyscale(
    team="saas-platform",
    project="ai-features",
    environment="production"
)

def process_customer_request(customer_id: str, request_data: dict):
    """Process customer request with cost attribution."""

    response = adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=request_data['messages'],

        # Governance attributes for billing
        customer_id=customer_id,
        feature=request_data.get('feature', 'chat'),
        cost_center="Product-AI"
    )

    # Cost automatically attributed to customer
    # Query observability platform for monthly billing:
    # SUM(cost) WHERE customer_id="..." AND month="2026-01"

    return response

# Process requests from different customers
process_customer_request("customer-A", {...})
process_customer_request("customer-B", {...})
process_customer_request("customer-C", {...})

# Each customer's costs tracked separately in OpenTelemetry
```

### Budget Enforcement

```python
# Track spending against budget
from datetime import datetime
import os

class BudgetEnforcer:
    def __init__(self, monthly_budget_usd: float):
        self.monthly_budget = monthly_budget_usd
        self.current_month = datetime.now().strftime("%Y-%m")

    def check_budget(self, customer_id: str) -> bool:
        """Check if customer has budget remaining."""
        # Query your observability platform for current month spend
        current_spend = self.get_customer_spend(customer_id, self.current_month)
        return current_spend < self.monthly_budget

    def get_customer_spend(self, customer_id: str, month: str) -> float:
        """Query observability platform for customer spend."""
        # Example: Query Datadog, Grafana, or Honeycomb
        # This is pseudo-code - implement based on your observability platform
        pass

# Usage
adapter = instrument_anyscale(team="saas-platform")
budget_enforcer = BudgetEnforcer(monthly_budget_usd=100.0)

def process_with_budget_check(customer_id: str, messages: list):
    """Process request with budget enforcement."""

    if not budget_enforcer.check_budget(customer_id):
        raise BudgetExceededError(
            f"Customer {customer_id} has exceeded monthly budget"
        )

    return adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=messages,
        customer_id=customer_id
    )
```

### Compliance and Audit Trails

```python
# All operations automatically generate audit trails via OpenTelemetry

adapter = instrument_anyscale(
    team="healthcare-ai",
    project="patient-analysis",
    environment="production",
    cost_center="Healthcare-IT"
)

# HIPAA-compliant request tracking
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "user", "content": "Analyze patient symptoms..."}
    ],

    # Audit trail attributes
    customer_id="hospital-123",
    feature="symptom-analysis",
    request_id="req-abc-123",
    user_id="doctor-456"
)

# OpenTelemetry span includes complete audit trail:
# - timestamp
# - team, project, environment
# - customer_id, user_id, request_id
# - model, tokens, cost
# - latency, success/failure
# - All governance attributes

# Query your observability platform for compliance reports:
# - All operations by customer
# - All operations by user
# - Cost attribution by cost center
# - Performance SLAs by environment
```

### Access Control Integration

```python
# Integrate with existing access control systems

from typing import Set

class AccessControlAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.permissions = {}  # Load from your access control system

    def check_model_access(self, user_id: str, model: str) -> bool:
        """Check if user has permission to use model."""
        allowed_models = self.permissions.get(user_id, set())
        return model in allowed_models

    def completion_create(self, user_id: str, model: str, **kwargs):
        """Completion with access control check."""

        if not self.check_model_access(user_id, model):
            raise PermissionError(
                f"User {user_id} not authorized to use {model}"
            )

        return self.adapter.completion_create(
            model=model,
            user_id=user_id,  # Include in governance attributes
            **kwargs
        )

# Usage
adapter = instrument_anyscale(team="enterprise")
access_controlled_adapter = AccessControlAdapter(adapter)

try:
    response = access_controlled_adapter.completion_create(
        user_id="employee-789",
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[...]
    )
except PermissionError as e:
    print(f"Access denied: {e}")
```

## Production Deployment

### High-Availability Configuration

```python
from genops.providers.anyscale import instrument_anyscale
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Production adapter with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def resilient_completion(adapter, **kwargs):
    """Completion with automatic retry on transient failures."""
    return adapter.completion_create(**kwargs)

# Initialize adapter
adapter = instrument_anyscale(
    team="production-team",
    project="customer-facing-app",
    environment="production"
)

# Use in production
try:
    response = resilient_completion(
        adapter,
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[{"role": "user", "content": "..."}],
        customer_id="customer-123"
    )
except Exception as e:
    # Log error and fallback
    print(f"Failed after 3 retries: {e}")
    # Implement fallback logic
```

### Load Balancing and Rate Limiting

```python
import asyncio
from asyncio import Semaphore

class RateLimitedAdapter:
    def __init__(self, adapter, max_concurrent: int = 10):
        self.adapter = adapter
        self.semaphore = Semaphore(max_concurrent)

    async def completion_create(self, **kwargs):
        """Rate-limited completion."""
        async with self.semaphore:
            # Implement your async completion here
            # This ensures max 10 concurrent requests
            return self.adapter.completion_create(**kwargs)

# Usage
adapter = instrument_anyscale(team="high-volume-app")
rate_limited = RateLimitedAdapter(adapter, max_concurrent=10)

# Process high-volume requests
async def process_batch(requests):
    tasks = [
        rate_limited.completion_create(**req)
        for req in requests
    ]
    return await asyncio.gather(*tasks)
```

### Monitoring and Alerting

```python
# Configure OpenTelemetry metrics for alerting

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Setup metrics pipeline
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://localhost:4317")
)
provider = MeterProvider(metric_readers=[metric_reader])
metrics.set_meter_provider(provider)

meter = metrics.get_meter("anyscale.monitoring")

# Create custom metrics
request_counter = meter.create_counter(
    "anyscale.requests.total",
    description="Total Anyscale API requests"
)

error_counter = meter.create_counter(
    "anyscale.errors.total",
    description="Total Anyscale API errors"
)

cost_gauge = meter.create_observable_gauge(
    "anyscale.cost.current",
    description="Current Anyscale cost"
)

# Use in application
adapter = instrument_anyscale(team="monitored-app")

def monitored_completion(**kwargs):
    """Completion with custom metrics."""
    request_counter.add(1, {"model": kwargs.get("model")})

    try:
        response = adapter.completion_create(**kwargs)

        # Record cost metric
        cost = calculate_completion_cost(
            model=kwargs.get("model"),
            input_tokens=response['usage']['prompt_tokens'],
            output_tokens=response['usage']['completion_tokens']
        )
        # cost_gauge.set(cost)  # Update gauge

        return response

    except Exception as e:
        error_counter.add(1, {"error_type": type(e).__name__})
        raise

# Configure alerts in your observability platform:
# - Alert when anyscale.errors.total > 10 in 5 minutes
# - Alert when anyscale.cost.current > budget_threshold
# - Alert when p99 latency > 5 seconds
```

### Disaster Recovery

```python
# Implement fallback to alternative providers

class MultiProviderAdapter:
    def __init__(self):
        self.anyscale_adapter = instrument_anyscale(team="multi-provider")
        self.fallback_available = self._check_fallback()

    def _check_fallback(self) -> bool:
        """Check if fallback provider is available."""
        try:
            # Check OpenAI, Replicate, or other fallback
            return True
        except:
            return False

    def completion_create(self, **kwargs):
        """Completion with automatic fallback."""
        try:
            return self.anyscale_adapter.completion_create(**kwargs)
        except Exception as e:
            print(f"Anyscale failed: {e}")

            if self.fallback_available:
                print("Falling back to alternative provider...")
                # Implement fallback to OpenAI or others
                return self._fallback_completion(**kwargs)
            else:
                raise

    def _fallback_completion(self, **kwargs):
        """Fallback completion implementation."""
        # Implement OpenAI or other provider fallback
        pass

# Usage
adapter = MultiProviderAdapter()
response = adapter.completion_create(...)  # Automatic fallback on failure
```

## Performance Optimization

### Telemetry Sampling

```python
# Reduce overhead in high-volume scenarios

adapter = instrument_anyscale(
    team="high-volume-app",
    project="production-api",

    # Sample 10% of requests for telemetry
    sampling_rate=0.1
)

# 90% of requests skip detailed telemetry, reducing overhead
# 10% of requests include full governance tracking
```

### Async Operations

```python
import asyncio
from typing import List

async def async_batch_processing(prompts: List[str]):
    """Process multiple prompts concurrently."""

    adapter = instrument_anyscale(team="async-team")

    async def process_single(prompt: str):
        # Implement async completion
        # Note: Current adapter is synchronous, but shows pattern
        return adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": prompt}]
        )

    # Process all prompts concurrently
    tasks = [process_single(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    return results

# Usage
prompts = ["Prompt 1", "Prompt 2", "Prompt 3", ...]
results = asyncio.run(async_batch_processing(prompts))
```

### Caching and Memoization

```python
from functools import lru_cache
import hashlib
import json

class CachedAnyscaleAdapter:
    def __init__(self, adapter, cache_size: int = 1000):
        self.adapter = adapter
        self.cache_size = cache_size

    def _hash_request(self, model: str, messages: list, **kwargs) -> str:
        """Create hash of request parameters."""
        request_dict = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        request_str = json.dumps(request_dict, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def _cached_completion(self, request_hash: str, model: str, messages_str: str, **kwargs):
        """Cached completion to avoid redundant API calls."""
        messages = json.loads(messages_str)
        return self.adapter.completion_create(
            model=model,
            messages=messages,
            **kwargs
        )

    def completion_create(self, model: str, messages: list, **kwargs):
        """Completion with caching."""
        request_hash = self._hash_request(model, messages, **kwargs)
        messages_str = json.dumps(messages)

        return self._cached_completion(
            request_hash,
            model,
            messages_str,
            **kwargs
        )

# Usage
adapter = instrument_anyscale(team="cached-app")
cached_adapter = CachedAnyscaleAdapter(adapter, cache_size=1000)

# Identical requests return cached results
response1 = cached_adapter.completion_create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

response2 = cached_adapter.completion_create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

# response2 returned from cache, no API call made
```

### Connection Pooling

```python
# Reuse HTTP connections for better performance

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_resilient_session():
    """Create HTTP session with connection pooling and retries."""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

# Use custom session in production
# (Note: Adapter would need to be modified to accept custom session)
```

## Observability Integration

### Datadog Integration

```python
# Export Anyscale telemetry to Datadog

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Datadog OTLP endpoint
provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint="http://localhost:4317",  # Datadog Agent OTLP endpoint
        headers={
            "DD-API-KEY": os.getenv("DD_API_KEY")
        }
    )
)
provider.add_span_processor(processor)

# Use adapter - telemetry automatically exported to Datadog
adapter = instrument_anyscale(
    team="datadog-integration",
    project="production-app"
)

response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[...]
)

# Query in Datadog:
# - Trace search: service:anyscale-ai-application
# - Metrics: genops.anyscale.cost.total
# - Logs: genops.team:datadog-integration
```

### Grafana / Prometheus Integration

```python
# Export metrics to Prometheus

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import start_http_server

# Define metrics
anyscale_requests = Counter(
    'anyscale_requests_total',
    'Total Anyscale API requests',
    ['model', 'team', 'customer_id']
)

anyscale_cost = Gauge(
    'anyscale_cost_usd',
    'Anyscale operation cost in USD',
    ['model', 'customer_id']
)

anyscale_latency = Histogram(
    'anyscale_latency_seconds',
    'Anyscale request latency',
    ['model']
)

# Start Prometheus metrics server
start_http_server(8000)

# Instrument adapter
adapter = instrument_anyscale(team="prometheus-integration")

def monitored_completion(**kwargs):
    """Completion with Prometheus metrics."""
    model = kwargs.get("model")
    customer_id = kwargs.get("customer_id", "unknown")

    anyscale_requests.labels(
        model=model,
        team="prometheus-integration",
        customer_id=customer_id
    ).inc()

    import time
    start_time = time.time()

    response = adapter.completion_create(**kwargs)

    latency = time.time() - start_time
    anyscale_latency.labels(model=model).observe(latency)

    cost = calculate_completion_cost(
        model=model,
        input_tokens=response['usage']['prompt_tokens'],
        output_tokens=response['usage']['completion_tokens']
    )
    anyscale_cost.labels(model=model, customer_id=customer_id).set(cost)

    return response

# Metrics available at: http://localhost:8000/metrics
# Import into Grafana for visualization
```

### Honeycomb Integration

```python
# Export to Honeycomb for observability

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Honeycomb
provider = TracerProvider()
processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint="https://api.honeycomb.io/v1/traces",
        headers={
            "x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY"),
            "x-honeycomb-dataset": "anyscale-telemetry"
        }
    )
)
provider.add_span_processor(processor)

# Use adapter
adapter = instrument_anyscale(team="honeycomb-team")

response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[...],
    customer_id="customer-123"
)

# Query in Honeycomb:
# - Traces with genops.anyscale.* attributes
# - Cost analysis by customer: SUM(genops.anyscale.cost.total) GROUP BY genops.customer_id
# - Latency p99 by model: P99(duration_ms) GROUP BY genops.anyscale.model
```

## Advanced Use Cases

### Multi-Model Router

```python
# Intelligent routing based on task complexity and cost

from genops.providers.anyscale import instrument_anyscale, get_model_pricing

class IntelligentRouter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.model_tiers = {
            "simple": "meta-llama/Llama-2-7b-chat-hf",
            "medium": "meta-llama/Llama-2-13b-chat-hf",
            "complex": "meta-llama/Llama-2-70b-chat-hf",
        }

    def estimate_complexity(self, prompt: str) -> str:
        """Estimate task complexity."""
        # Simple heuristic - replace with ML model in production
        if len(prompt) < 100:
            return "simple"
        elif len(prompt) < 500:
            return "medium"
        else:
            return "complex"

    def route_completion(self, messages: list, **kwargs):
        """Route to appropriate model based on complexity."""
        prompt = messages[0]['content'] if messages else ""
        complexity = self.estimate_complexity(prompt)
        model = self.model_tiers[complexity]

        print(f"📍 Routing to {complexity} tier: {model}")

        return self.adapter.completion_create(
            model=model,
            messages=messages,
            **kwargs
        )

# Usage
adapter = instrument_anyscale(team="intelligent-routing")
router = IntelligentRouter(adapter)

# Automatically routed to optimal model
response = router.route_completion(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    customer_id="customer-123"
)
```

### A/B Testing Framework

```python
# A/B test different models for performance and cost

import random
from typing import Dict, List

class ABTestingAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.experiments = {}
        self.results = []

    def create_experiment(
        self,
        name: str,
        variants: Dict[str, str],
        traffic_split: Dict[str, float]
    ):
        """Create A/B test experiment."""
        self.experiments[name] = {
            "variants": variants,
            "traffic_split": traffic_split
        }

    def select_variant(self, experiment_name: str) -> str:
        """Select variant based on traffic split."""
        experiment = self.experiments[experiment_name]
        rand = random.random()

        cumulative = 0
        for variant, percentage in experiment["traffic_split"].items():
            cumulative += percentage
            if rand <= cumulative:
                return variant

        return list(experiment["variants"].keys())[0]

    def experimental_completion(self, experiment_name: str, messages: list, **kwargs):
        """Run completion as part of A/B test."""
        variant = self.select_variant(experiment_name)
        model = self.experiments[experiment_name]["variants"][variant]

        import time
        start_time = time.time()

        response = self.adapter.completion_create(
            model=model,
            messages=messages,
            experiment_name=experiment_name,
            variant=variant,
            **kwargs
        )

        latency = time.time() - start_time

        # Record results
        self.results.append({
            "experiment": experiment_name,
            "variant": variant,
            "model": model,
            "latency": latency,
            "tokens": response['usage']['total_tokens'],
            "cost": calculate_completion_cost(
                model=model,
                input_tokens=response['usage']['prompt_tokens'],
                output_tokens=response['usage']['completion_tokens']
            )
        })

        return response

    def analyze_results(self, experiment_name: str):
        """Analyze A/B test results."""
        exp_results = [r for r in self.results if r["experiment"] == experiment_name]

        by_variant = {}
        for result in exp_results:
            variant = result["variant"]
            if variant not in by_variant:
                by_variant[variant] = {"latency": [], "cost": []}

            by_variant[variant]["latency"].append(result["latency"])
            by_variant[variant]["cost"].append(result["cost"])

        # Calculate averages
        for variant, data in by_variant.items():
            avg_latency = sum(data["latency"]) / len(data["latency"])
            avg_cost = sum(data["cost"]) / len(data["cost"])

            print(f"\n{variant}:")
            print(f"  Average latency: {avg_latency:.3f}s")
            print(f"  Average cost: ${avg_cost:.6f}")

# Usage
adapter = instrument_anyscale(team="ab-testing")
ab_adapter = ABTestingAdapter(adapter)

# Create experiment: Llama-2-70B vs Llama-2-13B
ab_adapter.create_experiment(
    name="model_comparison",
    variants={
        "control": "meta-llama/Llama-2-70b-chat-hf",
        "variant_a": "meta-llama/Llama-2-13b-chat-hf"
    },
    traffic_split={
        "control": 0.5,
        "variant_a": 0.5
    }
)

# Run experiment
for i in range(100):
    response = ab_adapter.experimental_completion(
        experiment_name="model_comparison",
        messages=[{"role": "user", "content": f"Query {i}"}]
    )

# Analyze results
ab_adapter.analyze_results("model_comparison")
```

### Cost Budgeting and Alerts

```python
# Implement cost budgets with real-time alerts

from datetime import datetime
from typing import Optional

class BudgetManager:
    def __init__(
        self,
        adapter,
        daily_budget_usd: float,
        monthly_budget_usd: float
    ):
        self.adapter = adapter
        self.daily_budget = daily_budget_usd
        self.monthly_budget = monthly_budget_usd
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.last_reset_date = datetime.now().date()
        self.alert_thresholds = [0.5, 0.75, 0.9, 1.0]  # 50%, 75%, 90%, 100%
        self.alerts_sent = set()

    def check_and_reset_daily(self):
        """Reset daily spend if new day."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_spend = 0.0
            self.last_reset_date = current_date
            self.alerts_sent.clear()

    def check_budget(self, estimated_cost: float) -> tuple[bool, Optional[str]]:
        """Check if request would exceed budget."""
        self.check_and_reset_daily()

        new_daily = self.daily_spend + estimated_cost
        new_monthly = self.monthly_spend + estimated_cost

        if new_daily > self.daily_budget:
            return False, f"Would exceed daily budget: ${new_daily:.2f} > ${self.daily_budget:.2f}"

        if new_monthly > self.monthly_budget:
            return False, f"Would exceed monthly budget: ${new_monthly:.2f} > ${self.monthly_budget:.2f}"

        return True, None

    def send_alert(self, threshold: float, budget_type: str):
        """Send budget alert."""
        alert_key = f"{budget_type}_{threshold}"
        if alert_key not in self.alerts_sent:
            percentage = int(threshold * 100)
            print(f"🚨 ALERT: {percentage}% of {budget_type} budget consumed")
            self.alerts_sent.add(alert_key)
            # Implement actual alerting: email, Slack, PagerDuty, etc.

    def check_alert_thresholds(self):
        """Check if alert thresholds reached."""
        daily_pct = self.daily_spend / self.daily_budget
        monthly_pct = self.monthly_spend / self.monthly_budget

        for threshold in self.alert_thresholds:
            if daily_pct >= threshold:
                self.send_alert(threshold, "daily")
            if monthly_pct >= threshold:
                self.send_alert(threshold, "monthly")

    def completion_create(self, model: str, messages: list, **kwargs):
        """Completion with budget enforcement."""
        # Estimate cost before making request
        prompt_tokens = sum(len(m['content'].split()) for m in messages) * 1.3  # Rough estimate
        estimated_output_tokens = kwargs.get('max_tokens', 500)

        estimated_cost = calculate_completion_cost(
            model=model,
            input_tokens=int(prompt_tokens),
            output_tokens=estimated_output_tokens
        )

        # Check budget
        allowed, reason = self.check_budget(estimated_cost)
        if not allowed:
            raise BudgetExceededError(reason)

        # Make request
        response = self.adapter.completion_create(
            model=model,
            messages=messages,
            **kwargs
        )

        # Record actual cost
        actual_cost = calculate_completion_cost(
            model=model,
            input_tokens=response['usage']['prompt_tokens'],
            output_tokens=response['usage']['completion_tokens']
        )

        self.daily_spend += actual_cost
        self.monthly_spend += actual_cost

        # Check alert thresholds
        self.check_alert_thresholds()

        return response

# Usage
adapter = instrument_anyscale(team="budget-controlled")
budget_manager = BudgetManager(
    adapter,
    daily_budget_usd=10.0,
    monthly_budget_usd=200.0
)

# Use with budget enforcement
try:
    response = budget_manager.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[{"role": "user", "content": "..."}]
    )
except BudgetExceededError as e:
    print(f"Budget limit reached: {e}")
```

## Troubleshooting

### Common Issues

#### Issue: "ANYSCALE_API_KEY not set"

**Symptom:**
```
ValidationError: ANYSCALE_API_KEY environment variable not set
```

**Fix:**
```bash
# Set API key
export ANYSCALE_API_KEY='your-api-key-here'

# Verify
echo $ANYSCALE_API_KEY

# Permanent fix (add to ~/.bashrc or ~/.zshrc)
echo 'export ANYSCALE_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### Issue: "Authentication Failed"

**Symptom:**
```
AuthenticationError: Invalid API key
```

**Fix:**
1. Verify API key at: https://console.anyscale.com/credentials
2. Check for extra spaces when copying
3. Ensure key hasn't expired
4. Create new API key if needed

```bash
# Test API key manually
curl -H "Authorization: Bearer $ANYSCALE_API_KEY" \
     https://api.endpoints.anyscale.com/v1/models
```

#### Issue: "Model not found"

**Symptom:**
```
ModelNotFoundError: Model 'meta-llama/Llama-2-70b' not available
```

**Fix:**
```python
# List available models
from genops.providers.anyscale import ANYSCALE_PRICING

print("Available models:")
for model in ANYSCALE_PRICING.keys():
    print(f"  - {model}")

# Use exact model name including suffix
model = "meta-llama/Llama-2-70b-chat-hf"  # Correct
# model = "meta-llama/Llama-2-70b"        # Wrong - missing suffix
```

#### Issue: "Connection timeout"

**Symptom:**
```
ConnectionError: Request timeout after 60s
```

**Fix:**
1. Check network connectivity
2. Verify firewall settings
3. Check DNS resolution
4. Try different network

```bash
# Test connectivity
curl https://api.endpoints.anyscale.com/v1/models

# Check DNS
nslookup api.endpoints.anyscale.com

# Test with timeout
curl --max-time 30 https://api.endpoints.anyscale.com/v1/models
```

#### Issue: "Rate limit exceeded"

**Symptom:**
```
RateLimitError: Too many requests (429)
```

**Fix:**
```python
# Implement rate limiting
import time
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
def rate_limited_completion(adapter, **kwargs):
    """Completion with automatic retry on rate limits."""
    try:
        return adapter.completion_create(**kwargs)
    except RateLimitError:
        print("Rate limit hit, retrying...")
        raise  # Retry will handle this

# Usage
adapter = instrument_anyscale(team="rate-limited-app")
response = rate_limited_completion(adapter, model="...", messages=[...])
```

#### Issue: "Telemetry not appearing in observability platform"

**Symptom:**
OpenTelemetry spans not visible in Datadog/Grafana/Honeycomb

**Fix:**
1. Verify OTLP exporter configuration
2. Check endpoint URL and port
3. Verify authentication headers
4. Test OTLP endpoint connectivity

```python
# Debug telemetry export
import os
os.environ['OTEL_LOG_LEVEL'] = 'debug'

# Check exporter configuration
from opentelemetry import trace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Add console exporter for debugging
provider = trace.get_tracer_provider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

# Use adapter - spans will print to console
adapter = instrument_anyscale(team="debug-team")
response = adapter.completion_create(...)
```

### Validation Troubleshooting

```python
# Run comprehensive validation
from genops.providers.anyscale import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

# Check specific validation categories
if not result.success:
    for issue in result.issues:
        print(f"\n❌ {issue.title}")
        print(f"   Category: {issue.category}")
        print(f"   Level: {issue.level}")
        print(f"   Description: {issue.description}")
        print(f"   Fix: {issue.fix_suggestion}")
```

### Debug Logging

```python
# Enable debug logging
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('genops.providers.anyscale').setLevel(logging.DEBUG)

# Use adapter - detailed logs will show all operations
adapter = instrument_anyscale(
    team="debug-team",
    debug=True  # Enable debug mode
)

response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "Test"}]
)

# Output includes:
# - API request details
# - Token usage calculations
# - Cost calculations
# - OpenTelemetry span creation
# - Governance attribute propagation
```

## API Reference

### Core Functions

#### `instrument_anyscale(**governance_defaults)`

Create GenOps Anyscale adapter with governance defaults.

**Parameters:**
- `anyscale_api_key` (str, optional): Anyscale API key (defaults to `ANYSCALE_API_KEY` env var)
- `anyscale_base_url` (str, optional): Base URL (default: "https://api.endpoints.anyscale.com/v1")
- `telemetry_enabled` (bool): Enable OpenTelemetry tracing (default: True)
- `cost_tracking_enabled` (bool): Enable cost tracking (default: True)
- `debug` (bool): Enable debug logging (default: False)
- `**governance_defaults`: Default governance attributes (team, project, environment, etc.)

**Returns:** `GenOpsAnyscaleAdapter`

**Example:**
```python
adapter = instrument_anyscale(
    team="ml-team",
    project="chatbot",
    environment="production"
)
```

#### `auto_instrument(**governance_defaults)`

Enable zero-code auto-instrumentation of OpenAI SDK.

**Parameters:**
- `**governance_defaults`: Default governance attributes for all operations

**Returns:** `bool` - True if successful

**Example:**
```python
from genops.providers.anyscale import auto_instrument

auto_instrument(team="auto-team", project="auto-project")

# Existing OpenAI SDK code now automatically tracked
import openai
client = openai.OpenAI(base_url="https://api.endpoints.anyscale.com/v1")
response = client.chat.completions.create(...)  # Tracked!
```

#### `validate_setup(anyscale_api_key=None, anyscale_base_url=None)`

Validate Anyscale setup and configuration.

**Parameters:**
- `anyscale_api_key` (str, optional): API key to validate
- `anyscale_base_url` (str, optional): Base URL to validate

**Returns:** `ValidationResult`

**Example:**
```python
from genops.providers.anyscale import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

if result.success:
    print("✅ Setup validated")
```

### Adapter Methods

#### `adapter.completion_create(model, messages, **kwargs)`

Create chat completion with governance tracking.

**Parameters:**
- `model` (str): Model ID (e.g., "meta-llama/Llama-2-70b-chat-hf")
- `messages` (list): Chat messages in OpenAI format
- `temperature` (float, optional): Sampling temperature (0.0-2.0)
- `max_tokens` (int, optional): Maximum tokens to generate
- `top_p` (float, optional): Nucleus sampling parameter
- `frequency_penalty` (float, optional): Frequency penalty (-2.0-2.0)
- `presence_penalty` (float, optional): Presence penalty (-2.0-2.0)
- `**governance_attrs`: Per-request governance attributes

**Returns:** dict with OpenAI-compatible response

**Example:**
```python
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=500,
    customer_id="customer-123"
)
```

#### `adapter.embeddings_create(model, input, **kwargs)`

Create embeddings with governance tracking.

**Parameters:**
- `model` (str): Embedding model ID (e.g., "thenlper/gte-large")
- `input` (str or list): Text to embed
- `**governance_attrs`: Per-request governance attributes

**Returns:** dict with OpenAI-compatible response

**Example:**
```python
response = adapter.embeddings_create(
    model="thenlper/gte-large",
    input="Text to embed",
    customer_id="customer-123"
)

embeddings = response['data'][0]['embedding']
```

### Pricing Functions

#### `calculate_completion_cost(model, input_tokens, output_tokens)`

Calculate cost for chat completion.

**Parameters:**
- `model` (str): Model ID
- `input_tokens` (int): Number of input tokens
- `output_tokens` (int): Number of output tokens

**Returns:** float (cost in USD)

**Example:**
```python
from genops.providers.anyscale import calculate_completion_cost

cost = calculate_completion_cost(
    model="meta-llama/Llama-2-70b-chat-hf",
    input_tokens=100,
    output_tokens=50
)
print(f"Cost: ${cost:.6f}")
```

#### `calculate_embedding_cost(model, tokens)`

Calculate cost for embeddings.

**Parameters:**
- `model` (str): Embedding model ID
- `tokens` (int): Number of tokens

**Returns:** float (cost in USD)

#### `get_model_pricing(model)`

Get pricing information for model.

**Parameters:**
- `model` (str): Model ID

**Returns:** `ModelPricing` dataclass

**Example:**
```python
from genops.providers.anyscale import get_model_pricing

pricing = get_model_pricing("meta-llama/Llama-2-70b-chat-hf")
print(f"Input: ${pricing.input_cost_per_million}/M tokens")
print(f"Output: ${pricing.output_cost_per_million}/M tokens")
print(f"Context window: {pricing.context_window} tokens")
```

### Data Classes

#### `AnyscaleCostSummary`

Cost summary for operations.

**Attributes:**
- `total_cost` (float): Total cost in USD
- `cost_by_model` (dict): Costs grouped by model
- `cost_by_customer` (dict): Costs grouped by customer_id
- `total_tokens` (int): Total tokens used
- `operation_count` (int): Number of operations

#### `ModelPricing`

Pricing information for a model.

**Attributes:**
- `model_name` (str): Model identifier
- `input_cost_per_million` (float): Input cost per million tokens
- `output_cost_per_million` (float): Output cost per million tokens
- `currency` (str): Currency (USD)
- `category` (str): Model category (chat, embedding)
- `context_window` (int): Maximum context length
- `notes` (str): Additional notes

---

## Next Steps

**Congratulations!** You now have comprehensive knowledge of GenOps Anyscale integration.

### Recommended Actions

1. **Start Simple**: Use the [Quickstart Guide](../anyscale-quickstart.md) for 5-minute setup
2. **Explore Examples**: Try `examples/anyscale/basic_completion.py`
3. **Enable Auto-Instrumentation**: Zero-code setup for existing applications
4. **Configure Observability**: Export to your platform (Datadog, Grafana, etc.)
5. **Optimize Costs**: Use model selection and caching strategies
6. **Scale to Production**: Implement budgets, monitoring, and high-availability patterns

### Additional Resources

- **Quickstart Guide**: [docs/anyscale-quickstart.md](../anyscale-quickstart.md)
- **Example Scripts**: `examples/anyscale/`
- **Anyscale Documentation**: https://docs.anyscale.com
- **GenOps GitHub**: https://github.com/KoshiHQ/GenOps-AI

### Community

- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Contributing**: See [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

**Built with GenOps AI** - Governance for AI, Built on OpenTelemetry
