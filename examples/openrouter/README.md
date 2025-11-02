# OpenRouter Examples

This directory contains comprehensive examples demonstrating how to use GenOps with OpenRouter for AI governance across 400+ models and 60+ providers.

## Overview

OpenRouter provides unified access to the world's largest collection of AI models through a single API. GenOps seamlessly integrates with OpenRouter to provide:

- **Multi-provider cost attribution** - Track costs across all underlying providers (OpenAI, Anthropic, Google, Meta, etc.)
- **Unified governance** - Apply consistent policies across 400+ models
- **Routing telemetry** - Monitor provider selection, failover, and performance
- **Budget controls** - Set limits across all models and providers
- **Zero-config setup** - Auto-instrumentation with no code changes required

## Quick Start

```python
# Zero-code auto-instrumentation
import genops
genops.init()

# Your existing OpenRouter code works unchanged
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"
)

response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Governance telemetry automatically captured!
```

## Examples

### 1. Basic Integration
- **[setup_validation.py](setup_validation.py)** - Validate your OpenRouter + GenOps setup
- **[basic_tracking.py](basic_tracking.py)** - Simple cost and usage tracking
- **[auto_instrumentation.py](auto_instrumentation.py)** - Zero-code auto-instrumentation demo

### 2. Advanced Features  
- **[multi_provider_costs.py](multi_provider_costs.py)** - Cost aggregation across multiple backend providers
- **[advanced_features.py](advanced_features.py)** - Provider selection, routing strategies, fallback handling
- **[cost_optimization.py](cost_optimization.py)** - Intelligent model/provider selection for cost optimization

### 3. Production Patterns
- **[production_patterns.py](production_patterns.py)** - Enterprise deployment, monitoring, and error handling
- **[routing_intelligence.py](routing_intelligence.py)** - Advanced routing strategies and provider health monitoring

## Key Features Demonstrated

### Multi-Provider Cost Attribution
OpenRouter routes your requests to different underlying providers. GenOps tracks:
- Total cost across all providers
- Per-provider cost breakdown
- Model-specific costs and usage
- Routing decisions and fallback events

### Unified Governance Attributes
Consistent governance across all 400+ models:
```python
response = client.chat.completions.create(
    model="meta-llama/llama-3.1-405b-instruct",  # Any OpenRouter model
    messages=[{"role": "user", "content": "Hello!"}],
    # Governance attributes work across ALL models
    team="ai-team",
    project="chatbot", 
    customer_id="customer-123",
    environment="production"
)
```

### OpenRouter-Specific Features
- **Provider preferences**: `provider="anthropic"` to prefer specific providers
- **Routing strategies**: `route="least-cost"` or `route="fastest"`
- **Fallback monitoring**: Track when failover occurs
- **Model selection intelligence**: Cost-aware model recommendations

## Environment Setup

```bash
# Required
export OPENROUTER_API_KEY="your-openrouter-key"

# Optional - Enhanced functionality
export OTEL_SERVICE_NAME="my-ai-service"
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export OPENROUTER_HTTP_REFERER="https://myapp.com"  # App identification
export OPENROUTER_X_TITLE="My AI Application"       # Request identification
```

## Installation

```bash
# Install GenOps with OpenRouter support
pip install genops-ai openai  # OpenAI SDK for OpenRouter compatibility

# Validate setup
python examples/openrouter/setup_validation.py
```

## Model Coverage

GenOps supports cost tracking for 400+ models across major providers:

- **OpenAI**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku  
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro, Gemma 2
- **Meta**: Llama 3.2 (90B, 11B, 3B, 1B), Llama 3.1 (405B, 70B, 8B)
- **Mistral**: Mistral Large, Medium, Small, Mixtral 8x7B/8x22B
- **Cohere**: Command R+, Command R
- **And 300+ more models from 60+ providers**

## Benefits for OpenRouter Users

### Immediate Value (5 minutes)
- Zero-code setup with automatic cost tracking
- Real-time visibility into multi-provider spending
- Built-in budget alerts and policy enforcement

### Progressive Enhancement (30 minutes)
- Custom governance attributes for team/project attribution
- Advanced routing strategies with cost optimization
- Integration with existing observability stacks

### Enterprise Features (2 hours)
- Multi-tenant cost attribution and billing
- Compliance automation and audit trails
- Custom policy enforcement and content filtering
- Production deployment patterns and monitoring

## Community

OpenRouter serves thousands of developers with 400+ models. GenOps provides the governance layer that scales with your usage:

- **Developer-first**: Minimal setup, maximum value
- **Enterprise-ready**: Security, compliance, and cost control
- **Community-driven**: Open source with extensible architecture

## Next Steps

1. **Quick validation**: Run `python setup_validation.py`
2. **Basic tracking**: Try `python basic_tracking.py` 
3. **Auto-instrumentation**: Test `python auto_instrumentation.py`
4. **Production setup**: Review `python production_patterns.py`

For comprehensive integration guidance, see [../../docs/integrations/openrouter.md](../../docs/integrations/openrouter.md).