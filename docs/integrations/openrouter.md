# OpenRouter Integration Guide

Complete integration guide for using GenOps with OpenRouter to achieve unified AI governance across 400+ models from 60+ providers.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Integration Patterns](#integration-patterns)
- [Multi-Provider Cost Attribution](#multi-provider-cost-attribution)
- [Advanced Features](#advanced-features)
- [Production Deployment](#production-deployment)
- [Observability Integration](#observability-integration)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

### What is OpenRouter?

OpenRouter is the world's largest AI model marketplace, providing unified access to 400+ models from 60+ providers including OpenAI, Anthropic, Google, Meta, Mistral, Cohere, and many more through a single API endpoint.

### GenOps + OpenRouter Value Proposition

| Feature | Without GenOps | With GenOps |
|---------|----------------|-------------|
| **Cost Tracking** | Manual, provider-specific | Automatic across all 400+ models |
| **Governance** | No attribution | Multi-dimensional (team, project, customer) |
| **Observability** | Basic logs | Rich OpenTelemetry traces |
| **Routing Intelligence** | Limited visibility | Full routing decision capture |
| **Budget Control** | None | Real-time limits and alerts |
| **Compliance** | Manual tracking | Automated audit trails |

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Your App     │───▶│   GenOps        │───▶│   OpenRouter    │
│                │    │   Governance    │    │   400+ Models   │
│                │    │   Layer         │    │   60+ Providers │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  OpenTelemetry  │
                    │  Observability  │
                    └─────────────────┘
```

## Installation

### Basic Installation

```bash
pip install genops-ai openai
```

The `openai` package is required for OpenRouter compatibility (OpenRouter uses OpenAI SDK interface).

### Development Installation

```bash
pip install genops-ai[dev] openai
```

### Verification

```bash
python -c "
from genops.providers.openrouter import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result)
"
```

## Integration Patterns

### Pattern 1: Zero-Code Auto-Instrumentation

**Best for**: Existing OpenRouter applications, minimal changes required.

```python
# Add these 2 lines at the top of your application
import genops
genops.init(
    service_name="my-openrouter-app",
    default_team="ai-platform",
    default_project="production-chatbot"
)

# Your existing OpenRouter code works unchanged
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"
)

# This request now has automatic governance telemetry
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**What GenOps captures automatically:**
- Request/response data and timing
- Token usage and cost calculations
- Provider routing decisions
- Default governance attributes
- OpenTelemetry traces

### Pattern 2: Manual Instrumentation

**Best for**: Fine-grained control, custom governance attributes.

```python
from genops.providers.openrouter import instrument_openrouter

# Create instrumented client with custom configuration
client = instrument_openrouter(
    openrouter_api_key="your-key",
    # Optional: custom headers for OpenRouter
    default_headers={
        "HTTP-Referer": "https://my-app.com",
        "X-Title": "My AI Application"
    }
)

# Add governance attributes per request
response = client.chat_completions_create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Generate product description"}],
    # Governance attributes
    team="marketing",
    project="product-launch", 
    customer_id="enterprise-001",
    cost_center="marketing-ops",
    environment="production",
    # OpenRouter-specific controls
    provider="openai",        # Prefer specific provider
    route="least-cost",       # Cost optimization
    max_tokens=200
)
```

### Pattern 3: Context Manager Pattern

**Best for**: Request grouping, batch operations, cost aggregation.

```python
from genops.providers.openrouter import instrument_openrouter
from genops.core.context import set_governance_context

client = instrument_openrouter(openrouter_api_key="your-key")

# Set context for batch of requests
with set_governance_context(
    team="data-science",
    project="model-evaluation",
    experiment_id="exp-2024-001"
):
    # All requests in this context inherit governance attributes
    models_to_test = [
        "openai/gpt-4o",
        "anthropic/claude-3-sonnet", 
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemini-1.5-pro"
    ]
    
    results = []
    for model in models_to_test:
        response = client.chat_completions_create(
            model=model,
            messages=[{"role": "user", "content": "Explain quantum computing"}],
            max_tokens=150
        )
        results.append({
            "model": model,
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        })
    
    # Unified cost analysis available in telemetry
```

## Multi-Provider Cost Attribution

### Understanding OpenRouter's Multi-Provider Routing

OpenRouter intelligently routes requests across 60+ providers. GenOps captures the full routing journey:

```python
response = client.chat_completions_create(
    model="anthropic/claude-3-sonnet",  # Requested model
    messages=[{"role": "user", "content": "Hello"}],
    provider="anthropic",                # Preferred provider
    route="fallback"                    # Allow fallback routing
)

# GenOps automatically captures:
# - genops.openrouter.predicted_provider: "anthropic" (initial prediction)
# - genops.openrouter.actual_provider: "anthropic" (actual provider used)
# - genops.openrouter.fallback_used: false (no fallback occurred)
# - genops.cost.total: 0.000045 (actual cost from anthropic pricing)
```

### Cost Attribution Dimensions

GenOps tracks costs across multiple dimensions simultaneously:

```python
# Multi-dimensional cost attribution
response = client.chat_completions_create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Customer support query"}],
    # Business dimensions
    team="customer-success",
    project="ai-support-bot",
    customer_id="enterprise-customer-123",
    feature="ticket-classification",
    # Financial dimensions  
    cost_center="customer-ops",
    billing_tier="premium",
    # Operational dimensions
    environment="production",
    region="us-west-2",
    deployment="prod-cluster-01"
)

# Results in rich telemetry attributes:
# - genops.team: "customer-success"
# - genops.project: "ai-support-bot" 
# - genops.customer_id: "enterprise-customer-123"
# - genops.cost.total: calculated cost
# - genops.openrouter.actual_provider: "meta"
# - ... all other attributes preserved
```

### Cost Aggregation Patterns

```python
from genops.providers.openrouter_pricing import get_cost_breakdown

# Detailed cost analysis
cost_breakdown = get_cost_breakdown(
    "anthropic/claude-3-sonnet",
    actual_provider="anthropic", 
    input_tokens=150,
    output_tokens=75
)

print(f"Total cost: ${cost_breakdown['total_cost']:.6f}")
print(f"Input cost: ${cost_breakdown['input_cost']:.6f}")  
print(f"Output cost: ${cost_breakdown['output_cost']:.6f}")
print(f"Provider: {cost_breakdown['provider']}")
print(f"Model family: {cost_breakdown['model_family']}")
```

## Advanced Features

### Provider Selection and Routing

```python
# Explicit provider preference
response = client.chat_completions_create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    provider="anthropic",           # Prefer Anthropic
    route="fallback",              # Allow fallback if unavailable
    team="research"
)

# Cost-optimized routing
response = client.chat_completions_create(
    model="openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Simple FAQ response"}], 
    route="least-cost",            # Optimize for cost
    team="customer-support"
)

# Performance-optimized routing  
response = client.chat_completions_create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Real-time analysis"}],
    route="fastest",               # Optimize for speed
    team="real-time-analytics"
)
```

### Budget-Constrained Operations

```python
from genops.providers.openrouter_pricing import estimate_cost_for_text

# Pre-flight cost estimation
estimated_cost = estimate_cost_for_text(
    "anthropic/claude-3-sonnet",
    "Long text to process...",
    completion_ratio=0.4
)

if estimated_cost[0] < 0.01:  # Budget check
    response = client.chat_completions_create(
        model="anthropic/claude-3-sonnet",
        messages=[{"role": "user", "content": "Long text to process..."}],
        team="content-team",
        budget_limit=0.01
    )
else:
    # Use more cost-effective model
    response = client.chat_completions_create(
        model="meta-llama/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": "Long text to process..."}],
        team="content-team"
    )
```

### Intelligent Model Selection

```python
def select_optimal_model(task_complexity: str, budget: float, latency_req: str):
    """Intelligent model selection based on requirements."""
    
    if task_complexity == "simple" and budget < 0.001:
        return "meta-llama/llama-3.2-1b-instruct"
    elif task_complexity == "medium" and latency_req == "fast":
        return "anthropic/claude-3-haiku"
    elif task_complexity == "complex":
        return "anthropic/claude-3-5-sonnet"
    else:
        return "openai/gpt-4o"  # Balanced choice

# Usage
model = select_optimal_model("complex", 0.01, "medium")
response = client.chat_completions_create(
    model=model,
    messages=[{"role": "user", "content": "Complex analysis task"}],
    team="analysis-team",
    complexity=task_complexity,
    budget_allocated=budget
)
```

## Production Deployment

### Environment Configuration

```bash
# Required
export OPENROUTER_API_KEY="your-production-key"

# Recommended  
export OTEL_SERVICE_NAME="openrouter-production-service"
export OTEL_SERVICE_VERSION="1.0.0"
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-key"

# Optional - Enhanced OpenRouter integration
export OPENROUTER_HTTP_REFERER="https://your-production-app.com"
export OPENROUTER_X_TITLE="Production AI Service"
export ENVIRONMENT="production"
```

### Production Client Pattern

```python
import logging
from genops.providers.openrouter import instrument_openrouter
from genops.core.context import set_governance_context

# Production logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ProductionOpenRouterService:
    def __init__(self):
        self.client = instrument_openrouter(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30.0,
            max_retries=3
        )
        
        # Default governance for all requests
        self.default_governance = {
            "service": os.getenv("OTEL_SERVICE_NAME", "openrouter-service"),
            "version": os.getenv("OTEL_SERVICE_VERSION", "unknown"),
            "environment": os.getenv("ENVIRONMENT", "production")
        }
    
    def safe_completion(self, model: str, messages: list, **governance_attrs):
        """Production-safe completion with error handling."""
        
        # Merge with default governance
        final_attrs = {**self.default_governance, **governance_attrs}
        
        try:
            with set_governance_context(**final_attrs):
                response = self.client.chat_completions_create(
                    model=model,
                    messages=messages,
                    max_retries=3,
                    timeout=30
                )
                return {
                    "success": True,
                    "response": response.choices[0].message.content,
                    "usage": response.usage,
                    "cost": "calculated_automatically"
                }
        except Exception as e:
            logging.error(f"OpenRouter request failed: {e}")
            return {
                "success": False, 
                "error": str(e)
            }

# Usage
service = ProductionOpenRouterService()
result = service.safe_completion(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Production query"}],
    team="production-team",
    customer_id="customer-456"
)
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Production environment
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from genops.providers.openrouter import validate_setup; validate_setup()"

CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openrouter-service
  labels:
    app: openrouter-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openrouter-service
  template:
    metadata:
      labels:
        app: openrouter-service
    spec:
      containers:
      - name: app
        image: your-registry/openrouter-service:latest
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi" 
            cpu: "250m"
        env:
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: openrouter-secrets
              key: api-key
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "https://api.honeycomb.io"
        - name: OTEL_SERVICE_NAME
          value: "openrouter-k8s-service"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Secret
metadata:
  name: openrouter-secrets
type: Opaque
data:
  api-key: <base64-encoded-openrouter-key>

---
apiVersion: v1  
kind: Service
metadata:
  name: openrouter-service
spec:
  selector:
    app: openrouter-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Observability Integration

### Supported Platforms

GenOps OpenRouter integration works with any OpenTelemetry-compatible observability platform:

- **Honeycomb** - Recommended for AI workload analysis
- **Datadog** - Enterprise APM with AI cost dashboards  
- **New Relic** - Full-stack observability with AI insights
- **Grafana Tempo** - Open-source distributed tracing
- **Dynatrace** - AI-powered application monitoring
- **Splunk** - Enterprise search and analytics

### Honeycomb Integration

```python
import os
import genops

# Configure Honeycomb export
genops.init(
    service_name="openrouter-honeycomb-demo",
    exporter_type="otlp",
    otlp_endpoint="https://api.honeycomb.io",
    otlp_headers={"x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY")},
    default_team="platform-team",
    default_environment="production"
)

# Your OpenRouter requests now appear in Honeycomb with rich context
```

**Honeycomb Queries:**
```
# Cost analysis by team
COUNT | WHERE genops.provider = "openrouter" | GROUP BY genops.team

# High-cost requests  
AVG(genops.cost.total) | WHERE genops.cost.total > 0.01 | GROUP BY genops.model

# Provider routing analysis
COUNT | WHERE genops.openrouter.fallback_used = true | GROUP BY genops.openrouter.actual_provider
```

### Datadog Integration

```python
import genops

genops.init(
    service_name="openrouter-datadog-demo", 
    exporter_type="otlp",
    otlp_endpoint="https://otlp.datadoghq.com",
    otlp_headers={"dd-api-key": os.getenv("DATADOG_API_KEY")},
    resource_attributes={
        "env": "production",
        "service.version": "1.2.0"
    }
)
```

### Custom Metrics

```python
from genops.providers.openrouter import instrument_openrouter
from opentelemetry import metrics

# Custom metrics for business KPIs
meter = metrics.get_meter("openrouter.business")
request_counter = meter.create_counter(
    "openrouter_requests_total",
    description="Total OpenRouter requests"
)
cost_histogram = meter.create_histogram(
    "openrouter_cost_per_request", 
    description="Cost per OpenRouter request"
)

client = instrument_openrouter(openrouter_api_key="your-key")

def instrumented_request(model, messages, **governance_attrs):
    response = client.chat_completions_create(
        model=model,
        messages=messages,
        **governance_attrs
    )
    
    # Custom business metrics
    request_counter.add(1, {
        "team": governance_attrs.get("team", "unknown"),
        "model_family": model.split("/")[0] if "/" in model else "unknown"
    })
    
    if hasattr(response, 'usage'):
        estimated_cost = calculate_cost(model, response.usage)
        cost_histogram.record(estimated_cost, {
            "team": governance_attrs.get("team", "unknown")
        })
    
    return response
```

## Troubleshooting

### Common Issues

#### 1. API Key Problems

**Problem**: `401 Unauthorized` errors

**Diagnosis**:
```bash
# Test your API key directly
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models

# Validate with GenOps
python -c "
from genops.providers.openrouter import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result)
"
```

**Solutions**:
- Verify API key from [openrouter.ai/keys](https://openrouter.ai/keys)
- Check environment variable: `echo $OPENROUTER_API_KEY`
- Ensure sufficient credits in OpenRouter account

#### 2. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'genops'`

**Solutions**:
```bash
# Install GenOps
pip install genops-ai

# Install OpenRouter dependencies  
pip install openai

# Verify installation
python -c "import genops; import openai; print('All imports successful')"
```

#### 3. No Telemetry Data

**Problem**: Requests work but no telemetry appears

**Diagnosis**:
```python
# Check if auto-instrumentation is active
from genops.auto_instrumentation import GenOpsInstrumentor
instrumentor = GenOpsInstrumentor()
print("OpenRouter registered:", "openrouter" in instrumentor.provider_patches)

# Verify OTLP configuration
import os
print("OTLP endpoint:", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
```

**Solutions**:
- Ensure `genops.init()` is called before OpenRouter usage
- Configure OTLP endpoint: `export OTEL_EXPORTER_OTLP_ENDPOINT="your-endpoint"`
- Check observability platform connectivity

#### 4. High Costs / Unexpected Billing

**Problem**: Higher than expected OpenRouter costs

**Analysis**:
```python
from genops.providers.openrouter_pricing import get_cost_breakdown

# Analyze specific request cost
breakdown = get_cost_breakdown(
    "anthropic/claude-3-opus",  # Expensive model
    input_tokens=1000,
    output_tokens=500
)

print(f"Model: {breakdown['model_name']}")
print(f"Provider: {breakdown['provider']}")  
print(f"Total cost: ${breakdown['total_cost']:.6f}")
print(f"Cost per token: ${breakdown['input_cost_per_token']:.8f}")
```

**Cost Optimization**:
- Use cost-effective models for simple tasks (`meta-llama/llama-3.2-3b-instruct`)
- Set budget constraints with `max_budget` parameter
- Monitor costs in real-time through telemetry dashboards
- Use routing strategies: `route="least-cost"`

### Debug Mode

```python
import logging
import genops

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("genops").setLevel(logging.DEBUG)

# Initialize with debug telemetry
genops.init(
    service_name="debug-session",
    exporter_type="console",  # Print traces to console
    debug=True
)

# Your OpenRouter requests will show detailed debug output
```

### Support Channels

- **Documentation**: This guide and [examples/openrouter/](../../examples/openrouter/)
- **Validation**: Run `python examples/openrouter/setup_validation.py`
- **Community**: GitHub Issues and Discussions
- **Enterprise**: Professional support available

## API Reference

### Core Functions

#### `instrument_openrouter()`

Create an instrumented OpenRouter client.

```python
from genops.providers.openrouter import instrument_openrouter

client = instrument_openrouter(
    client=None,                    # Optional existing client
    openrouter_api_key="your-key",  # API key (or use OPENROUTER_API_KEY env)
    api_key="your-key",             # Alternative parameter name
    base_url="https://openrouter.ai/api/v1",  # Default OpenRouter URL
    timeout=30.0,                   # Request timeout
    max_retries=3,                  # Retry attempts
    default_headers={               # Custom headers
        "HTTP-Referer": "https://your-app.com",
        "X-Title": "Your Application Name"
    }
)
```

#### `chat_completions_create()`

Create chat completion with governance tracking.

```python
response = client.chat_completions_create(
    # OpenAI-compatible parameters
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=150,
    stream=False,
    
    # OpenRouter-specific parameters
    provider="anthropic",           # Preferred provider
    route="least-cost",            # Routing strategy: "least-cost", "fastest", "fallback"
    fallbacks=["openai/gpt-4o"],   # Fallback models
    
    # GenOps governance attributes
    team="your-team",              # Team attribution
    project="your-project",        # Project attribution  
    customer_id="customer-123",    # Customer attribution
    environment="production",      # Environment tag
    cost_center="engineering",     # Cost center
    feature="ai-assistant",        # Feature attribution
    user_id="user-456",           # User attribution
    
    # Custom attributes (any additional key-value pairs)
    experiment_id="exp-001",
    model_version="v1.2",
    region="us-west-2"
)
```

#### `completions_create()`

Create completion with governance tracking (legacy API).

```python  
response = client.completions_create(
    model="openai/gpt-3.5-turbo-instruct",
    prompt="Complete this text: The future of AI is",
    max_tokens=100,
    # Same governance attributes as chat_completions_create
    team="content-team",
    project="text-generation"
)
```

### Pricing Functions

#### `calculate_openrouter_cost()`

Calculate cost for specific model and token usage.

```python
from genops.providers.openrouter_pricing import calculate_openrouter_cost

cost = calculate_openrouter_cost(
    model_name="anthropic/claude-3-sonnet",
    actual_provider="anthropic",       # Optional: actual provider used
    input_tokens=150,
    output_tokens=75
)
# Returns: 0.000675 (cost in USD)
```

#### `get_cost_breakdown()`

Get detailed cost breakdown.

```python
from genops.providers.openrouter_pricing import get_cost_breakdown

breakdown = get_cost_breakdown(
    model_name="openai/gpt-4o",
    input_tokens=200,
    output_tokens=100
)

# Returns detailed dictionary:
{
    "total_cost": 0.002,
    "input_cost": 0.001,
    "output_cost": 0.001,
    "input_tokens": 200,
    "output_tokens": 100, 
    "input_cost_per_token": 0.000005,
    "output_cost_per_token": 0.00001,
    "provider": "openai",
    "model_family": "gpt-4o",
    "currency": "USD",
    "model_name": "openai/gpt-4o"
}
```

#### `get_supported_models()`

Get all supported models and their pricing.

```python
from genops.providers.openrouter_pricing import get_supported_models

models = get_supported_models()
# Returns: Dict[str, OpenRouterPricing] with 400+ entries

# Example usage
for model_name, pricing in models.items():
    if pricing.provider == "anthropic":
        print(f"{model_name}: ${pricing.input_cost_per_token:.8f}/token input")
```

### Validation Functions

#### `validate_setup()`

Comprehensive setup validation.

```python
from genops.providers.openrouter import validate_setup

result = validate_setup()
# Returns ValidationResult with:
# - is_valid: bool
# - issues: List[ValidationIssue]  
# - summary: Dict[str, Any]
```

#### `print_validation_result()`

User-friendly validation output.

```python
from genops.providers.openrouter import print_validation_result

print_validation_result(result)
# Prints formatted validation report with actionable fixes
```

### Auto-Instrumentation Functions

#### `patch_openrouter()`

Apply global monkey patches for zero-code instrumentation.

```python
from genops.providers.openrouter import patch_openrouter

patch_openrouter(auto_track=True)
# Now all OpenAI clients with OpenRouter base URL are automatically instrumented
```

#### `unpatch_openrouter()`

Remove monkey patches.

```python
from genops.providers.openrouter import unpatch_openrouter

unpatch_openrouter()
# Restores original OpenAI client behavior
```

### Governance Attributes Reference

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `team` | str | Team responsible for request | "ml-platform" |
| `project` | str | Project or application name | "chatbot-v2" |
| `customer_id` | str | Customer identifier for billing | "enterprise-001" |  
| `customer` | str | Customer name (alternative to customer_id) | "Acme Corp" |
| `environment` | str | Environment (dev/staging/prod) | "production" |
| `cost_center` | str | Financial cost center | "engineering" |
| `feature` | str | Specific feature or capability | "document-qa" |
| `user_id` | str | End-user identifier | "user-123" |
| `experiment_id` | str | A/B test or experiment ID | "exp-2024-001" |
| `region` | str | Geographic region | "us-west-2" |
| `model_version` | str | Model version or variant | "v1.2.0" |
| `priority` | str | Request priority level | "high" |
| `compliance_level` | str | Data sensitivity level | "confidential" |

All attributes are optional and can be combined in any way. Custom attributes are also supported.

## Model Support

### Supported Providers

GenOps OpenRouter integration supports cost tracking for all major providers:

- **OpenAI**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, GPT-4o Mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku  
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash, Gemma 2
- **Meta**: Llama 3.2 (90B, 11B, 3B, 1B), Llama 3.1 (405B, 70B, 8B)
- **Mistral**: Mistral Large, Medium, Small, Mixtral 8x7B, Mixtral 8x22B
- **Cohere**: Command R+, Command R, Command
- **Perplexity**: Sonar models for online search
- **Together AI**: Various open-source models
- **And 300+ more models from 60+ providers**

### Model Categories

Models are categorized for intelligent selection:

**Economy Tier** (< $0.001/1K tokens):
- `meta-llama/llama-3.2-1b-instruct`
- `meta-llama/llama-3.2-3b-instruct`  
- `google/gemma-2-9b-it`

**Balanced Tier** ($0.001-$0.01/1K tokens):
- `openai/gpt-3.5-turbo`
- `anthropic/claude-3-haiku`
- `mistralai/mistral-small`

**Premium Tier** ($0.01-$0.05/1K tokens):  
- `openai/gpt-4o`
- `anthropic/claude-3-5-sonnet`
- `google/gemini-1.5-pro`

**Flagship Tier** (> $0.05/1K tokens):
- `anthropic/claude-3-opus` 
- `meta-llama/llama-3.1-405b-instruct`

---

## Next Steps

1. **Start with Quickstart**: [../openrouter-quickstart.md](../openrouter-quickstart.md)
2. **Try Examples**: [../../examples/openrouter/](../../examples/openrouter/)
3. **Production Deploy**: Use Kubernetes manifests above
4. **Set Up Monitoring**: Configure your observability platform
5. **Optimize Costs**: Implement intelligent model selection

**Questions?** Check our [examples](../../examples/openrouter/) or open a GitHub issue.

---

*This integration brings together OpenRouter's 400+ models with GenOps' enterprise governance - giving you the best of both worlds: maximum model choice with complete operational control.*