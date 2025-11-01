# GenOps AI Examples

This directory contains examples demonstrating how to use GenOps AI for AI governance and telemetry.

## Examples Overview

### Core Examples

#### 1. `basic_usage.py`
Comprehensive examples showing all major GenOps AI features:
- **Function decorators** for automatic tracking
- **Context managers** for block-level tracking
- **Policy enforcement** for governance
- **Provider instrumentation** for OpenAI and Anthropic
- **Manual telemetry recording** for cost and evaluation metrics

#### 2. `otel_setup.py` 
OpenTelemetry integration examples:
- Console exporter for development/testing
- OTLP exporter for production environments
- Jaeger exporter for distributed tracing
- Datadog exporter for monitoring platforms

### Framework Integrations

#### 3. `langchain/` Directory 📚
**Comprehensive LangChain integration examples** with governance telemetry:

**Getting Started:**
- **[setup_validation.py](langchain/setup_validation.py)** - Verify your setup is working
- **[basic_chain_tracking.py](langchain/basic_chain_tracking.py)** - Simple chain execution tracking
- **[auto_instrumentation.py](langchain/auto_instrumentation.py)** - Zero-code setup

**Advanced Use Cases:**
- **[multi_provider_costs.py](langchain/multi_provider_costs.py)** - Track costs across OpenAI, Anthropic, Cohere
- **[rag_pipeline_monitoring.py](langchain/rag_pipeline_monitoring.py)** - RAG workflow telemetry
- **Cost attribution** and **customer billing** scenarios

**Key Features:**
- ✅ **Chain execution tracking** with detailed performance metrics
- ✅ **Multi-provider cost aggregation** across different LLM providers  
- ✅ **RAG operation monitoring** for retrieval and generation costs
- ✅ **Governance attribute propagation** for team/project/customer attribution
- ✅ **Auto-instrumentation** for zero-code setup

**Quick Start:**
```bash
# Install with LangChain support
pip install genops-ai[langchain]

# Verify setup
python examples/langchain/setup_validation.py

# Try basic example
python examples/langchain/basic_chain_tracking.py
```

See the **[LangChain Quickstart Guide](../docs/langchain-quickstart.md)** for detailed setup instructions.

## Quick Start

### 1. Install Dependencies

```bash
# Core package
pip install -e .

# For OpenAI examples
pip install openai

# For Anthropic examples  
pip install anthropic

# For additional exporters
pip install opentelemetry-exporter-jaeger
pip install opentelemetry-exporter-datadog
```

### 2. Run Basic Examples

```bash
# Run all basic usage examples
python examples/basic_usage.py

# Run with OpenTelemetry console output
python examples/otel_setup.py
```

### 3. Set Environment Variables

```bash
# For OpenAI examples
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic examples
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For OpenTelemetry configuration
export OTEL_EXPORTER_TYPE="console"  # or "otlp", "jaeger", "datadog"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

## Usage Patterns

### Function Decorator Pattern

```python
from genops import track_usage

@track_usage(
    operation_name="sentiment_analysis",
    team="nlp-team",
    project="customer-feedback", 
    feature="sentiment"
)
def analyze_sentiment(text: str) -> dict:
    # Your AI logic here
    return {"sentiment": "positive", "confidence": 0.85}
```

### Context Manager Pattern

```python
from genops import track

with track(
    operation_name="document_processing",
    team="content-team",
    customer="enterprise-123"
) as span:
    # Process documents
    span.set_attribute("doc_count", 10)
    # Telemetry is automatically captured
```

### Provider Instrumentation

```python
from genops.providers import instrument_openai

# Automatic telemetry for all OpenAI calls
client = instrument_openai(api_key="your-key")

response = client.chat_completions_create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Cost, token usage, and performance metrics automatically recorded
```

### Policy Enforcement

```python
from genops import enforce_policy
from genops.core.policy import register_policy, PolicyResult

# Register policies
register_policy(
    name="cost_limit",
    enforcement_level=PolicyResult.BLOCKED,
    max_cost=1.00
)

# Enforce on functions
@enforce_policy(["cost_limit"])
def expensive_ai_operation():
    # Will be blocked if estimated cost > $1.00
    pass
```

## OpenTelemetry Integration

### Console Output (Development)

```python
from examples.otel_setup import setup_console_exporter
setup_console_exporter()

# Now all GenOps telemetry will print to console
```

### OTLP Exporter (Production)

```python
from examples.otel_setup import setup_otlp_exporter
setup_otlp_exporter("http://your-collector:4317")

# Telemetry will be sent to your OpenTelemetry collector
```

### Integration with Existing Observability

GenOps AI telemetry integrates seamlessly with:

- **Jaeger** - Distributed tracing
- **Datadog** - APM and monitoring
- **Honeycomb** - Observability platform
- **New Relic** - Application monitoring
- **Grafana Tempo** - Tracing backend
- **Any OTLP-compatible backend**

## Telemetry Data Structure

GenOps AI adds standardized attributes to OpenTelemetry spans:

### Core Attributes
```
genops.operation.type = "ai.inference"
genops.operation.name = "sentiment_analysis"
genops.team = "nlp-team"
genops.project = "customer-feedback"
genops.customer = "enterprise-123"
```

### Cost Attributes
```
genops.cost.amount = 0.05
genops.cost.currency = "USD"
genops.cost.provider = "openai"
genops.cost.model = "gpt-4"
genops.cost.tokens.input = 150
genops.cost.tokens.output = 50
```

### Policy Attributes
```
genops.policy.name = "cost_limit"
genops.policy.result = "allowed"
genops.policy.reason = "Under budget limit"
```

### Evaluation Attributes
```
genops.eval.name = "quality_score"
genops.eval.score = 0.85
genops.eval.threshold = 0.8
genops.eval.passed = true
```

## Next Steps

1. **Review the examples** to understand different usage patterns
2. **Set up OpenTelemetry** with your preferred backend
3. **Configure policies** for your governance requirements
4. **Instrument your AI applications** with GenOps decorators and context managers
5. **Monitor your telemetry data** in your observability platform

For more advanced usage, see the main documentation in [GitHub Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs).