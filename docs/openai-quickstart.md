# OpenAI Quickstart

Get GenOps governance telemetry running with your OpenAI application in under 5 minutes.

## 🚀 Quick Setup

### 1. Install GenOps with OpenAI Support

```bash
pip install genops-ai[openai]
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key_here"
export OTEL_SERVICE_NAME="my-openai-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # Optional
```

### 3. Enable Auto-Instrumentation (Zero Code Changes)

```python
from genops import auto_instrument

# This one line enables telemetry for all OpenAI operations
auto_instrument()

# Your existing OpenAI code works unchanged!
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
# Automatically tracked with cost, tokens, and performance metrics!
```

**That's it!** Your OpenAI application now captures:
- ✅ Chat completion costs and performance
- ✅ Token usage tracking by model
- ✅ Error tracking and success rates
- ✅ Complete request/response telemetry

## 💰 Add Cost Attribution

For cost attribution and billing, add governance attributes:

```python
from genops.core.context import set_governance_context

# Set once - applies to all operations
set_governance_context({
    "team": "ai-engineering",
    "project": "customer-chatbot", 
    "customer_id": "enterprise_customer_123",
    "environment": "production"
})

# All OpenAI operations now include governance attributes
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "How can I help you today?"}]
)
```

## 🔧 Manual Instrumentation (Fine-Grained Control)

For more control, use manual instrumentation:

```python
from genops.providers.openai import instrument_openai

# Create instrumented client
client = instrument_openai(api_key="your_key_here")

# Use with governance attributes
response = client.chat_completions_create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze this text"}],
    
    # Governance attributes for cost attribution
    team="nlp-team",
    project="text-analysis",
    customer_id="customer_456"
)

print(f"Response: {response.choices[0].message.content}")
```

## 📊 Cost Tracking

Track costs across multiple OpenAI operations:

```python
from genops.core.tracker import track_cost
from genops import track

# Method 1: Manual cost tracking
with track("batch_processing", team="data-team") as span:
    response1 = client.chat.completions.create(model="gpt-3.5-turbo", messages=[...])
    response2 = client.chat.completions.create(model="gpt-4", messages=[...])
    
    # Costs automatically aggregated and attributed to "data-team"

# Method 2: Function-level tracking
from genops import track_usage

@track_usage(
    operation_name="sentiment_analysis",
    team="ml-team",
    project="customer-feedback"
)
def analyze_sentiment(text: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze sentiment: positive, negative, or neutral"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# Usage - automatically tracked
sentiment = analyze_sentiment("I love this product!")
```

## 📈 View Your Telemetry

### Option 1: Local Observability Stack

```bash
# From your project root
curl -O https://raw.githubusercontent.com/genops-ai/genops-ai/main/docker-compose.observability.yml
docker-compose -f docker-compose.observability.yml up -d

# View dashboards
open http://localhost:3000  # Grafana
open http://localhost:16686 # Jaeger
```

### Option 2: Your Existing Platform

GenOps works with any OpenTelemetry-compatible platform:

**Datadog:** See the **[Datadog Quickstart Guide](datadog-quickstart.md)** for complete 5-minute setup.

```bash
# Quick Datadog setup
export DATADOG_API_KEY="your_datadog_api_key"
export DATADOG_SITE="datadoghq.com"
```

**Other Platforms:**

```bash
# Honeycomb
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export HONEYCOMB_API_KEY="your_honeycomb_key"

# New Relic
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.nr-data.net"
export NEW_RELIC_API_KEY="your_newrelic_key"
```

## ✅ Verify Setup

Run this verification script:

```python
from genops.providers.openai import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

# Expected output:
# ✅ GenOps OpenAI setup is valid!
# 📊 Validation Summary:
#    Total checks: 12
#    Errors: 0
#    Warnings: 2
#    Info: 3
```

## 🎯 Common Use Cases

### Web Application Integration

```python
# FastAPI example
from fastapi import FastAPI
from genops import auto_instrument
from genops.core.context import set_governance_context

app = FastAPI()
auto_instrument()  # Enable for all routes

@app.post("/chat")
async def chat_endpoint(message: str, user_id: str):
    # Set governance context for this request
    set_governance_context({"customer_id": user_id, "team": "api-team"})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    
    return {"response": response.choices[0].message.content}
```

### Batch Processing

```python
def process_customer_texts(texts: list, customer_id: str):
    total_cost = 0
    results = []
    
    with track(f"batch_processing_{customer_id}", 
               customer_id=customer_id, team="batch-team") as span:
        
        for text in texts:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Summarize: {text}"}]
            )
            results.append(response.choices[0].message.content)
        
        # All costs automatically tracked for billing
        span.set_attribute("texts_processed", len(texts))
    
    return results
```

### Multi-Model Cost Optimization

```python
def smart_completion(prompt: str, complexity: str = "simple"):
    """Choose model based on complexity for cost optimization."""
    
    model_map = {
        "simple": "gpt-3.5-turbo",      # $0.001/1K tokens
        "complex": "gpt-4",             # $0.03/1K tokens  
        "advanced": "gpt-4-turbo"       # $0.01/1K tokens
    }
    
    model = model_map.get(complexity, "gpt-3.5-turbo")
    
    response = client.chat_completions_create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        
        # Cost attribution
        team="optimization-team",
        project="smart-routing",
        complexity_level=complexity  # Custom attribute
    )
    
    return response.choices[0].message.content
```

## 🔧 Troubleshooting

### Issue: No telemetry appearing

```bash
# Check OpenTelemetry configuration
python -c "import os; print('OTLP endpoint:', os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'))"

# Enable debug logging
export OTEL_LOG_LEVEL=debug
export GENOPS_LOG_LEVEL=debug
```

### Issue: OpenAI authentication errors

```bash
# Verify API key
python -c "import os; print('API key set:', bool(os.getenv('OPENAI_API_KEY')))"

# Test API key
python -c "
from openai import OpenAI
client = OpenAI()
print('API key valid:', bool(client.models.list()))
"
```

### Issue: Cost tracking not working

```python
# Verify instrumentation
from genops.providers.openai import validate_setup
result = validate_setup()
if not result.is_valid:
    print("Setup issues found - check validation output")
```

## 📚 Next Steps

Once you have basic telemetry working:

1. **[Complete Integration Guide](integrations/openai.md)** - Comprehensive documentation
2. **[Examples](examples/openai/)** - Practical implementation patterns  
3. **[Multi-Provider Costs](examples/multi_provider_costs.py)** - Advanced cost tracking
4. **[Policy Enforcement](examples/governance_scenarios/)** - Governance and compliance

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)
- **Documentation**: [Complete Docs](https://docs.genops.ai)

---

**🎉 You now have complete governance telemetry for your OpenAI application!**

Your telemetry includes cost tracking, performance metrics, error monitoring, and governance attribution - all with minimal code changes.