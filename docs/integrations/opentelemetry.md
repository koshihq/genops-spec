# OpenTelemetry Integration with GenOps AI

**GenOps AI extends OpenTelemetry with governance semantics for AI systems — interoperable by design, independent by governance.**

This guide explains how GenOps AI integrates with the OpenTelemetry ecosystem to provide AI governance telemetry using standard OTLP (OpenTelemetry Protocol) signals.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Semantic Conventions](#semantic-conventions)
- [Integration Patterns](#integration-patterns)
- [Using OTel SDK Directly](#using-otel-sdk-directly)
- [Collector Integration](#collector-integration)
- [Backend Compatibility](#backend-compatibility)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is OpenTelemetry?

OpenTelemetry (OTel) is an open-source observability framework that provides:
- **Standard telemetry signals**: Traces, metrics, and logs
- **Vendor-neutral protocol**: OTLP for data export
- **Cross-platform SDKs**: Consistent instrumentation across languages
- **Ecosystem integration**: Works with 15+ observability platforms

### How GenOps AI Extends OpenTelemetry

GenOps AI builds on OpenTelemetry by adding **governance semantics** for AI systems:

```
OpenTelemetry (foundation)
   └── GenOps-OTel (AI governance: cost, policy, compliance, evaluation)
```

**Key extensions:**
- `genops.cost.*` - Cost attribution and budget tracking
- `genops.policy.*` - Policy evaluation and enforcement
- `genops.budget.*` - Budget management and constraints
- `genops.eval.*` - Quality assessment and compliance

**Benefits:**
- **Interoperable by design**: Standard OTLP signals work with any OTel-compatible backend
- **Vendor neutrality**: No lock-in with any observability or AI provider
- **Reuse existing stack**: Integrate with Datadog, Honeycomb, Grafana, Prometheus, etc.
- **Cross-stack tracking**: Unified governance across LLM providers and frameworks

---

## Architecture

### Signal Flow

```
┌─────────────────┐
│  GenOps AI SDK  │
│  (Python)       │
└────────┬────────┘
         │ Extends OpenTelemetry with governance semantics
         ▼
┌─────────────────┐
│ OpenTelemetry   │
│ SDK (Python)    │
└────────┬────────┘
         │ Exports OTLP signals
         ▼
┌─────────────────┐
│ OTLP Exporter   │
│ (gRPC/HTTP)     │
└────────┬────────┘
         │ Sends to observability backend
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │Datadog  │   │Honeycomb│   │ Grafana │   │Elastic  │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### Trace Structure

GenOps AI creates OpenTelemetry spans for AI operations:

```
Root Span: "ai.completion"
├── Attributes:
│   ├── genops.cost.total: 0.0032
│   ├── genops.cost.input_cost: 0.0015
│   ├── genops.cost.output_cost: 0.0017
│   ├── genops.team: "ml-platform"
│   ├── genops.project: "chatbot"
│   ├── genops.customer_id: "customer-123"
│   ├── genops.environment: "production"
│   ├── genops.cost_center: "engineering"
│   ├── genops.feature: "chat-support"
│   ├── ai.model.provider: "openai"
│   ├── ai.model.name: "gpt-4"
│   ├── ai.request.input_tokens: 500
│   └── ai.request.output_tokens: 300
└── Events:
    ├── "policy.evaluated" (t=10ms)
    │   └── genops.policy.result: "allowed"
    └── "cost.calculated" (t=2500ms)
        └── genops.cost.total: 0.0032
```

**Key concepts:**
- **Spans** represent AI operations (LLM calls, evaluations, policy checks)
- **Attributes** contain governance metadata (cost, team, policy results)
- **Events** capture state changes (policy evaluations, budget alerts)
- **Context propagation** maintains governance attributes across distributed operations

---

## Semantic Conventions

### Governance Attributes

GenOps AI defines standard attributes in the `genops.*` namespace:

#### Cost Tracking (`genops.cost.*`)

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `genops.cost.total` | float | Total cost in USD | `0.0032` |
| `genops.cost.input_cost` | float | Input token cost in USD | `0.0015` |
| `genops.cost.output_cost` | float | Output token cost in USD | `0.0017` |
| `genops.cost.currency` | string | Currency code | `USD` |
| `genops.cost.model` | string | Model name for cost calculation | `gpt-4` |
| `genops.cost.provider` | string | Provider for cost calculation | `openai` |

#### Policy Enforcement (`genops.policy.*`)

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `genops.policy.result` | string | Policy evaluation result | `allowed`, `blocked`, `error` |
| `genops.policy.name` | string | Policy name evaluated | `pii-filter` |
| `genops.policy.violation_type` | string | Type of violation | `pii-detected` |
| `genops.policy.action` | string | Action taken | `block`, `log`, `alert` |

#### Budget Management (`genops.budget.*`)

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `genops.budget.limit` | float | Budget limit in USD | `1000.0` |
| `genops.budget.consumed` | float | Budget consumed in USD | `750.0` |
| `genops.budget.remaining` | float | Budget remaining in USD | `250.0` |
| `genops.budget.period` | string | Budget period | `monthly`, `daily` |
| `genops.budget.alert_threshold` | float | Alert threshold (0-1) | `0.8` |

#### Attribution (`genops.*`)

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `genops.team` | string | Team identifier | `ml-platform` |
| `genops.project` | string | Project identifier | `chatbot` |
| `genops.customer_id` | string | Customer identifier | `customer-123` |
| `genops.environment` | string | Environment | `production`, `staging`, `dev` |
| `genops.cost_center` | string | Cost center for financial reporting | `engineering` |
| `genops.feature` | string | Feature identifier | `chat-support` |

#### Evaluation (`genops.eval.*`)

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `genops.eval.score` | float | Evaluation score (0-1) | `0.92` |
| `genops.eval.metric` | string | Evaluation metric | `accuracy`, `relevance` |
| `genops.eval.threshold` | float | Threshold for pass/fail | `0.8` |
| `genops.eval.result` | string | Evaluation result | `pass`, `fail` |

### Standard AI Attributes

GenOps AI also uses standard AI semantic conventions where applicable:

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `ai.model.provider` | string | AI provider | `openai`, `anthropic`, `bedrock` |
| `ai.model.name` | string | Model name | `gpt-4`, `claude-3-opus` |
| `ai.request.input_tokens` | int | Input tokens consumed | `500` |
| `ai.request.output_tokens` | int | Output tokens generated | `300` |
| `ai.request.temperature` | float | Sampling temperature | `0.7` |
| `ai.request.max_tokens` | int | Maximum tokens | `1000` |

---

## Integration Patterns

### Pattern 1: GenOps Auto-Instrumentation

**Easiest approach** - GenOps handles all OpenTelemetry integration:

```python
from genops.providers.openai import instrument_openai
from openai import OpenAI

# GenOps automatically creates OTel spans with governance attributes
instrument_openai(team="ml-platform", project="chatbot")

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
# Span created automatically with cost, team, and policy attributes
```

**What happens:**
1. GenOps creates OpenTelemetry span for each LLM call
2. Adds governance attributes (cost, team, project)
3. Exports span via configured OTLP exporter
4. Works with any OTel-compatible backend

### Pattern 2: Manual Span Creation with GenOps Context

**More control** - Create custom spans with governance context:

```python
from opentelemetry import trace
from genops.core.context import create_genops_context

tracer = trace.get_tracer(__name__)

# Create governance context
with create_genops_context(
    team="ml-platform",
    project="chatbot",
    customer_id="customer-123"
) as context:
    # Create custom span
    with tracer.start_as_current_span("ai.workflow") as span:
        # Governance attributes automatically added from context

        # Your AI operations
        result = perform_ai_operations()

        # Add custom attributes
        span.set_attribute("workflow.steps", 3)
        span.set_attribute("genops.cost.total", calculate_cost(result))
```

**Benefits:**
- Full control over span structure
- Custom attributes alongside governance data
- Context propagation across services
- Works with existing OTel instrumentation

### Pattern 3: Direct OTel SDK with GenOps Exporters

**Maximum flexibility** - Use OpenTelemetry SDK directly with GenOps exporters:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from genops.exporters.elastic import GenOpsElasticSpanExporter

# Configure OpenTelemetry with GenOps exporter
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()

# Use GenOps exporter (already has governance awareness)
elastic_exporter = GenOpsElasticSpanExporter(
    elasticsearch_url="http://localhost:9200"
)
tracer_provider.add_span_processor(
    BatchSpanProcessor(elastic_exporter)
)

# Now use standard OTel SDK
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("ai.operation") as span:
    # Add GenOps governance attributes manually
    span.set_attribute("genops.team", "ml-platform")
    span.set_attribute("genops.cost.total", 0.0032)
    span.set_attribute("ai.model.provider", "openai")

    # Your logic here
    result = call_llm()
```

**Use when:**
- You need complete control over OpenTelemetry configuration
- Integrating with existing OTel instrumentation
- Building custom exporters or processors

---

## Using OTel SDK Directly

### Installation

```bash
pip install opentelemetry-api opentelemetry-sdk genops-ai
```

### Basic Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter  # For testing
)

# 1. Create tracer provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# 2. Add span processor with exporter
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(BatchSpanProcessor(console_exporter))

# 3. Get tracer
tracer = trace.get_tracer(__name__)
```

### Adding Governance Attributes

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("ai.completion") as span:
    # Standard AI attributes
    span.set_attribute("ai.model.provider", "openai")
    span.set_attribute("ai.model.name", "gpt-4")
    span.set_attribute("ai.request.input_tokens", 500)
    span.set_attribute("ai.request.output_tokens", 300)

    # GenOps governance attributes
    span.set_attribute("genops.team", "ml-platform")
    span.set_attribute("genops.project", "chatbot")
    span.set_attribute("genops.customer_id", "customer-123")
    span.set_attribute("genops.environment", "production")

    # Cost tracking
    span.set_attribute("genops.cost.total", 0.0032)
    span.set_attribute("genops.cost.input_cost", 0.0015)
    span.set_attribute("genops.cost.output_cost", 0.0017)

    # Policy tracking
    span.set_attribute("genops.policy.result", "allowed")

    # Your AI operation
    response = call_llm()
```

### Context Propagation

OpenTelemetry automatically propagates context across services:

```python
from opentelemetry import trace, context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

tracer = trace.get_tracer(__name__)

# Service A: Create span and inject context into HTTP headers
with tracer.start_as_current_span("service-a") as span:
    span.set_attribute("genops.team", "ml-platform")

    # Inject context into headers
    headers = {}
    TraceContextTextMapPropagator().inject(headers)

    # Make HTTP request with headers
    response = requests.post("http://service-b", headers=headers)

# Service B: Extract context from headers
propagator = TraceContextTextMapPropagator()
ctx = propagator.extract(carrier=request.headers)

# Create child span with extracted context
with tracer.start_as_current_span("service-b", context=ctx) as span:
    # Governance attributes from parent span are preserved
    # Add additional attributes
    span.set_attribute("genops.cost.total", 0.005)
```

**Benefits:**
- Unified governance across distributed AI systems
- Automatic cost attribution across service boundaries
- Policy enforcement throughout request lifecycle

---

## Collector Integration

### OpenTelemetry Collector

The OpenTelemetry Collector can receive, process, and export GenOps telemetry:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

  # Filter spans with governance attributes
  filter:
    spans:
      include:
        match_type: strict
        attributes:
          - key: genops.team
            value: "ml-platform"

  # Add additional attributes
  attributes:
    actions:
      - key: deployment.environment
        value: "production"
        action: insert

exporters:
  # Elasticsearch exporter
  elasticsearch:
    endpoints: ["http://elasticsearch:9200"]
    index: "genops-ai-traces"

  # Prometheus exporter for metrics
  prometheus:
    endpoint: "0.0.0.0:8889"

  # Datadog exporter
  datadog:
    api:
      key: "${DATADOG_API_KEY}"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, filter, attributes]
      exporters: [elasticsearch, datadog]

    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus, datadog]
```

### GenOps Configuration for Collector

```python
from genops.providers.openai import instrument_openai
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OTLP exporter to send to collector
otlp_exporter = OTLPSpanExporter(
    endpoint="http://otel-collector:4317",
    insecure=True  # Use TLS in production
)

# Set up tracer provider
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# GenOps auto-instrumentation with custom provider
instrument_openai(
    team="ml-platform",
    project="chatbot",
    tracer_provider=provider
)
```

### Custom Collector Processors

GenOps provides custom collector processors for advanced governance:

**1. Cost Aggregation Processor**
```yaml
processors:
  genops/cost_aggregator:
    # Aggregate costs by team and project
    dimensions:
      - genops.team
      - genops.project

    # Export aggregated metrics
    export_interval: 60s
```

**2. Budget Enforcement Processor**
```yaml
processors:
  genops/budget_enforcer:
    # Enforce budget limits
    budgets:
      - team: "ml-platform"
        limit: 1000.0
        period: "daily"
        action: "block"  # or "alert"
```

**3. Policy Evaluation Processor**
```yaml
processors:
  genops/policy_evaluator:
    # Evaluate policies on collected telemetry
    policies:
      - name: "pii-detection"
        pattern: ".*SSN.*|.*credit_card.*"
        action: "redact"
```

---

## Backend Compatibility

GenOps AI telemetry works with any OpenTelemetry-compatible backend:

### Datadog

```python
from opentelemetry.exporter.datadog import DatadogExporter

exporter = DatadogExporter(
    agent_url="http://localhost:8126",
    service="genops-ai"
)
```

**View in Datadog:**
- APM → Traces → Filter by `genops.team:ml-platform`
- Metrics → Custom metrics → `genops.cost.total`
- Dashboards → Create custom dashboard with governance metrics

### Honeycomb

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="api.honeycomb.io:443",
    headers=(("x-honeycomb-team", "YOUR_API_KEY"),)
)
```

**Query in Honeycomb:**
```
BREAKDOWN(genops.cost.total) BY genops.team
WHERE genops.environment = "production"
```

### Grafana Tempo

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="http://tempo:4318/v1/traces"
)
```

**Query in Grafana:**
```promql
sum(rate(traces{genops_team="ml-platform"}[5m])) by (genops_project)
```

### Prometheus (Metrics)

Export GenOps metrics to Prometheus:

```python
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider

reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
```

**PromQL Queries:**
```promql
# Total cost by team
sum(genops_cost_total) by (genops_team)

# P95 latency by model
histogram_quantile(0.95,
  rate(ai_request_duration_bucket{ai_model_name="gpt-4"}[5m])
)

# Budget consumption rate
rate(genops_budget_consumed[1h])
```

### Elasticsearch

See **[Elastic Integration Guide](./elastic.md)** for complete Elasticsearch setup.

---

## Best Practices

### 1. Use Semantic Conventions Consistently

**Good:**
```python
span.set_attribute("genops.team", "ml-platform")
span.set_attribute("genops.cost.total", 0.0032)
```

**Bad:**
```python
span.set_attribute("team_name", "ml-platform")  # Non-standard
span.set_attribute("total_cost", 0.0032)  # Non-standard
```

### 2. Always Set Governance Attributes

Ensure all AI operations include governance context:

```python
required_attributes = {
    "genops.team": "ml-platform",
    "genops.project": "chatbot",
    "genops.environment": "production"
}

with tracer.start_as_current_span("ai.operation") as span:
    for key, value in required_attributes.items():
        span.set_attribute(key, value)
```

### 3. Use Context Propagation

Don't lose governance context across service boundaries:

```python
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Inject context into outgoing requests
carrier = {}
TraceContextTextMapPropagator().inject(carrier)
requests.post("http://service", headers=carrier)
```

### 4. Batch Span Exports

Use `BatchSpanProcessor` for better performance:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Good: Batch exports (lower overhead)
provider.add_span_processor(BatchSpanProcessor(exporter))

# Avoid: SimpleSpanProcessor (exports immediately, higher overhead)
# provider.add_span_processor(SimpleSpanProcessor(exporter))
```

### 5. Sample High-Volume Operations

For high-volume systems, use sampling:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
provider = TracerProvider(sampler=sampler)
```

**Note:** Always capture governance attributes even for sampled-out spans.

### 6. Handle Errors Gracefully

```python
from opentelemetry.trace import Status, StatusCode

with tracer.start_as_current_span("ai.operation") as span:
    try:
        result = call_llm()
        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
```

### 7. Use Span Events for State Changes

```python
with tracer.start_as_current_span("ai.workflow") as span:
    # Record policy evaluation
    span.add_event("policy.evaluated", {
        "genops.policy.result": "allowed",
        "genops.policy.name": "pii-filter"
    })

    # Record cost calculation
    span.add_event("cost.calculated", {
        "genops.cost.total": 0.0032
    })
```

---

## Troubleshooting

### No Spans Exported

**Symptoms:**
- No traces appearing in observability backend
- GenOps operations not tracked

**Diagnosis:**
```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Temporarily export to console
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(BatchSpanProcessor(console_exporter))
```

**Common causes:**
1. **Exporter not configured**: Verify exporter setup
2. **Provider not set**: Call `trace.set_tracer_provider(provider)`
3. **Batch timeout**: Wait for batch flush (default 5s) or call `provider.force_flush()`

**Fix:**
```python
import time
from opentelemetry import trace

# Ensure provider is set
trace.set_tracer_provider(provider)

# Force flush before exit
provider.force_flush()
time.sleep(1)  # Allow time for export
```

### Governance Attributes Missing

**Symptoms:**
- Spans exported but missing `genops.*` attributes
- Cost attribution not working

**Diagnosis:**
```python
with tracer.start_as_current_span("test") as span:
    print(f"Span context: {span.get_span_context()}")
    print(f"Attributes: {span.attributes}")
```

**Common causes:**
1. **Context not set**: GenOps context not created
2. **Manual span creation**: Attributes not added explicitly

**Fix:**
```python
from genops.core.context import create_genops_context

# Always use GenOps context
with create_genops_context(team="ml-platform") as context:
    with tracer.start_as_current_span("ai.operation") as span:
        # Governance attributes automatically added
        pass
```

### Context Not Propagating

**Symptoms:**
- Child spans missing parent's governance attributes
- Distributed traces disconnected

**Diagnosis:**
```python
from opentelemetry import context

# Check current context
current_context = context.get_current()
print(f"Context: {current_context}")
```

**Common causes:**
1. **Context not injected**: Forgot to inject into headers
2. **Context not extracted**: Forgot to extract from headers

**Fix:**
```python
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

propagator = TraceContextTextMapPropagator()

# Inject into outgoing requests
headers = {}
propagator.inject(headers)
requests.post("http://service", headers=headers)

# Extract from incoming requests
ctx = propagator.extract(carrier=request.headers)
with tracer.start_as_current_span("operation", context=ctx):
    pass
```

### High Overhead

**Symptoms:**
- Application slowdown after adding instrumentation
- High memory usage

**Diagnosis:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Check for excessive span creation
```

**Common causes:**
1. **Synchronous export**: Using `SimpleSpanProcessor`
2. **No sampling**: Capturing 100% of traces
3. **Large batch sizes**: Memory pressure

**Fix:**
```python
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

# Use async batch processor
provider.add_span_processor(BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    schedule_delay_millis=5000,
    max_export_batch_size=512
))

# Apply sampling for high-volume operations
sampler = ParentBasedTraceIdRatio(0.1)  # 10% sampling
provider = TracerProvider(sampler=sampler)
```

### Collector Connection Issues

**Symptoms:**
- "Connection refused" errors
- Spans not reaching backend

**Diagnosis:**
```bash
# Test collector endpoint
curl -v http://otel-collector:4318/v1/traces

# Check collector logs
docker logs otel-collector
```

**Common causes:**
1. **Wrong endpoint**: Incorrect collector URL
2. **Network issues**: Firewall or DNS problems
3. **Collector not running**: Service down

**Fix:**
```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="http://otel-collector:4317",
    insecure=True,  # For non-TLS development
    timeout=30  # Increase timeout
)

# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def export_with_retry():
    provider.force_flush()
```

---

## Next Steps

- **[Elastic Integration](./elastic.md)** - Complete Elasticsearch setup with GenOps AI
- **[Multi-Provider Cost Tracking](../guides/multi-provider-cost-tracking.md)** - Unified cost attribution across providers
- **[Example Integrations](../../examples/)** - Working code examples
- **[OpenTelemetry Documentation](https://opentelemetry.io/docs/)** - Official OTel docs

---

## Additional Resources

- **[OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/)** - Standard attributes
- **[OTLP Specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/otlp.md)** - Protocol details
- **[OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)** - Collector documentation
- **[GenOps AI GitHub](https://github.com/KoshiHQ/GenOps-AI)** - Source code and examples

---

**Questions or issues?** Open an issue on [GitHub](https://github.com/KoshiHQ/GenOps-AI/issues).
