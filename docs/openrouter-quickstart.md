# OpenRouter Quickstart - 5 Minutes to AI Governance

**Get started with GenOps + OpenRouter in under 5 minutes.** This guide shows you how to add comprehensive AI governance to your OpenRouter applications with zero code changes.

## What You'll Get

✅ **Automatic cost tracking** across 400+ models from 60+ providers  
✅ **Multi-provider governance** with unified attribution  
✅ **Zero code changes** to your existing OpenRouter applications  
✅ **Real-time observability** with OpenTelemetry integration  

## Prerequisites (30 seconds)

1. **OpenRouter API Key**: Get yours from [openrouter.ai/keys](https://openrouter.ai/keys)
2. **Python 3.8+**: Any recent Python version
3. **5 minutes**: That's all you need!

## Step 1: Install (1 minute)

```bash
pip install genops-ai openai
```

The `openai` package is required for OpenRouter compatibility (OpenRouter uses the OpenAI SDK).

## Step 2: Setup API Key (30 seconds)

```bash
export OPENROUTER_API_KEY="your-openrouter-key-here"
```

> **Tip**: Add this to your `.bashrc` or `.zshrc` to make it permanent.

## Step 3: Validate Setup (30 seconds)

```bash
python -c "
from genops.providers.openrouter import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result)
"
```

**Expected output:**
```
✅ Overall Status: VALID
📊 Summary: 0 errors, 2 warnings, 6 info
💡 Recommendations: Setup looks good! Ready to use OpenRouter with GenOps
```

## Step 4: Add Governance (2 minutes)

### Option A: Zero-Code Auto-Instrumentation (Recommended)

Add **just 2 lines** to your existing OpenRouter code:

```python
# Add these 2 lines at the top
import genops
genops.init()

# Your existing OpenRouter code works unchanged!
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-openrouter-key"
)

response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "What is machine learning?"}]
)

print(response.choices[0].message.content)
```

**That's it!** Your OpenRouter requests now have automatic:
- Cost tracking and attribution
- Multi-provider governance telemetry  
- Budget monitoring and alerting
- OpenTelemetry traces for observability

### Option B: Manual Instrumentation (More Control)

For additional governance attributes:

```python
from genops.providers.openrouter import instrument_openrouter

# Create instrumented client
client = instrument_openrouter(openrouter_api_key="your-key")

# Add governance attributes to any request
response = client.chat_completions_create(
    model="meta-llama/llama-3.2-3b-instruct",
    messages=[{"role": "user", "content": "Explain renewable energy"}],
    # Governance attributes
    team="sustainability-team",
    project="green-chatbot",
    customer_id="demo-customer-001",
    environment="development"
)
```

## Step 5: See It Working (1 minute)

Run this quick test to see governance in action:

```python
import genops
genops.init(service_name="openrouter-quickstart-test")

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key="your-openrouter-key"
)

# Test different providers for cost comparison
test_models = [
    "meta-llama/llama-3.2-3b-instruct",  # Very cost-effective
    "openai/gpt-3.5-turbo",              # Balanced
    "anthropic/claude-3-sonnet"          # Premium reasoning
]

for model in test_models:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is AI?"}],
        max_tokens=50
    )
    print(f"✅ {model}: {response.choices[0].message.content[:50]}...")

print("🎉 All requests automatically tracked with governance telemetry!")
```

## What Just Happened?

**GenOps automatically captured:**
- ✅ **Cost data** for each model and underlying provider
- ✅ **Token usage** and performance metrics
- ✅ **Provider routing** decisions (OpenAI vs Anthropic vs Meta)
- ✅ **Governance attributes** for cost attribution
- ✅ **OpenTelemetry traces** for your observability dashboard

## View Your Telemetry

If you have an observability platform configured:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-key"
```

Your traces will appear with rich governance attributes:
- `genops.cost.total` - Request cost in USD
- `genops.provider` - "openrouter" 
- `genops.openrouter.actual_provider` - Backend provider used
- `genops.team`, `genops.project` - Attribution dimensions

## Common Use Cases

### Multi-Team Cost Attribution
```python
# Marketing team request  
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Write a product description"}],
    team="marketing",
    project="product-launch",
    cost_center="marketing-ops"
)

# Engineering team request
response = client.chat.completions.create(
    model="meta-llama/llama-3.2-3b-instruct", 
    messages=[{"role": "user", "content": "Review this code"}],
    team="engineering", 
    project="code-review-ai",
    cost_center="engineering-tools"
)
```

### Customer-Specific Billing
```python
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Customer support query"}],
    customer_id="enterprise-customer-001",
    billing_tier="premium",
    feature="ai-support"
)
```

### Cost Optimization
```python
# Prefer cost-effective providers
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Simple FAQ response"}],
    provider="anthropic",  # Provider preference
    route="least-cost",    # Cost-optimized routing
    max_budget=0.01       # Budget constraint
)
```

## Troubleshooting

**API Key Issues:**
```bash
# Test your OpenRouter key
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
```

**Import Errors:**
```bash
pip install --upgrade genops-ai openai
```

**No Telemetry Visible:**
```python
# Verify instrumentation is active
from genops.auto_instrumentation import GenOpsInstrumentor
instrumentor = GenOpsInstrumentor()
print("OpenRouter registered:", "openrouter" in instrumentor.provider_patches)
```

## Next Steps (5 more minutes)

🚀 **Production Ready**: Try [production_patterns.py](../examples/openrouter/production_patterns.py) for enterprise deployment  
💰 **Cost Intelligence**: Explore [cost_optimization.py](../examples/openrouter/cost_optimization.py) for smart routing  
🔍 **Advanced Features**: Check [advanced_features.py](../examples/openrouter/advanced_features.py) for provider control  
📊 **Multi-Provider**: See [multi_provider_costs.py](../examples/openrouter/multi_provider_costs.py) for unified reporting  

## Learn More

- **📖 Complete Guide**: [integrations/openrouter.md](integrations/openrouter.md)
- **🎯 All Examples**: [examples/openrouter/](../examples/openrouter/)  
- **🔧 Validation**: [examples/openrouter/setup_validation.py](../examples/openrouter/setup_validation.py)
- **🏭 Production**: [examples/openrouter/production_patterns.py](../examples/openrouter/production_patterns.py)

---

## Why GenOps + OpenRouter?

**OpenRouter** provides access to 400+ AI models from 60+ providers.  
**GenOps** provides unified governance across all of them.

**Together**, you get:
- ✅ **Zero vendor lock-in** with OpenTelemetry standards
- ✅ **Unified cost control** across all providers and models  
- ✅ **Enterprise governance** with team/project/customer attribution
- ✅ **Drop-in integration** with existing OpenRouter applications
- ✅ **Automatic optimization** through intelligent routing and cost awareness

**Start building governed AI applications today!** 🚀