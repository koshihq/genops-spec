# Anthropic Quickstart

Get GenOps governance telemetry running with your Anthropic Claude application in under 5 minutes.

## 🚀 Quick Setup

### 1. Install GenOps with Anthropic Support

```bash
pip install genops-ai[anthropic]
```

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export OTEL_SERVICE_NAME="my-claude-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # Optional
```

### 3. Enable Auto-Instrumentation (Zero Code Changes)

```python
from genops import auto_instrument

# This one line enables telemetry for all Anthropic operations
auto_instrument()

# Your existing Anthropic code works unchanged!
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello, Claude!"}]
)
# Automatically tracked with cost, tokens, and performance metrics!
```

**That's it!** Your Anthropic application now captures:
- ✅ Message completion costs and performance
- ✅ Token usage tracking by Claude model
- ✅ Error tracking and success rates
- ✅ Complete request/response telemetry

## 💰 Add Cost Attribution

For cost attribution and billing, add governance attributes:

```python
from genops.core.context import set_governance_context

# Set once - applies to all operations
set_governance_context({
    "team": "ai-research",
    "project": "claude-assistant", 
    "customer_id": "research_customer_123",
    "environment": "production"
})

# All Anthropic operations now include governance attributes
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=200,
    messages=[{"role": "user", "content": "Help me understand quantum computing"}]
)
```

## 🔧 Manual Instrumentation (Fine-Grained Control)

For more control, use manual instrumentation:

```python
from genops.providers.anthropic import instrument_anthropic

# Create instrumented client
client = instrument_anthropic(api_key="your_key_here")

# Use with governance attributes
response = client.messages_create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "Analyze this business strategy document"}
    ],
    
    # Governance attributes for cost attribution
    team="strategy-team",
    project="business-analysis",
    customer_id="enterprise_789"
)

print(f"Response: {response.content[0].text}")
```

## 📊 Cost Tracking

Track costs across multiple Claude operations:

```python
from genops.core.tracker import track_cost
from genops import track

# Method 1: Manual cost tracking
with track("document_analysis", team="content-team") as span:
    response1 = client.messages.create(
        model="claude-3-haiku-20240307",  # Fast and cheap for simple tasks
        max_tokens=100,
        messages=[{"role": "user", "content": "Summarize this text"}]
    )
    
    response2 = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # More capable for complex tasks
        max_tokens=500,
        messages=[{"role": "user", "content": "Provide detailed analysis"}]
    )
    
    # Costs automatically aggregated and attributed to "content-team"

# Method 2: Function-level tracking
from genops import track_usage

@track_usage(
    operation_name="document_review",
    team="legal-team",
    project="contract-analysis"
)
def review_contract(contract_text: str):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a legal contract reviewer. Identify key terms and potential issues."},
            {"role": "user", "content": f"Review this contract: {contract_text}"}
        ]
    )
    return response.content[0].text

# Usage - automatically tracked
review = review_contract("Contract content here...")
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
from genops.providers.anthropic import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

# Expected output:
# ✅ GenOps Anthropic setup is valid!
# 📊 Validation Summary:
#    Total checks: 11
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

@app.post("/analyze")
async def analyze_endpoint(text: str, user_id: str):
    # Set governance context for this request
    set_governance_context({"customer_id": user_id, "team": "analysis-api"})
    
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=200,
        messages=[
            {"role": "system", "content": "Provide concise analysis of the given text"},
            {"role": "user", "content": text}
        ]
    )
    
    return {"analysis": response.content[0].text}
```

### Batch Processing

```python
def process_customer_documents(documents: list, customer_id: str):
    results = []
    
    with track(f"batch_analysis_{customer_id}", 
               customer_id=customer_id, team="document-processing") as span:
        
        for doc in documents:
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Cost-effective for batch
                max_tokens=150,
                messages=[
                    {"role": "system", "content": "Extract key points from document"},
                    {"role": "user", "content": doc}
                ]
            )
            results.append(response.content[0].text)
        
        # All costs automatically tracked for billing
        span.set_attribute("documents_processed", len(documents))
    
    return results
```

### Model Selection for Cost Optimization

```python
def smart_claude_completion(prompt: str, complexity: str = "simple"):
    """Choose Claude model based on complexity for cost optimization."""
    
    model_map = {
        "simple": "claude-3-haiku-20240307",      # $0.25/$1.25 per 1M tokens
        "balanced": "claude-3-5-haiku-20241022",  # $1/$5 per 1M tokens
        "complex": "claude-3-5-sonnet-20241022",  # $3/$15 per 1M tokens
        "advanced": "claude-3-opus-20240229"      # $15/$75 per 1M tokens
    }
    
    model = model_map.get(complexity, "claude-3-haiku-20240307")
    
    response = client.messages_create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        
        # Cost attribution
        team="optimization-team",
        project="smart-routing",
        complexity_level=complexity  # Custom attribute
    )
    
    return response.content[0].text
```

### Multi-Turn Conversations

```python
def conversational_assistant(conversation_history: list, customer_id: str):
    """Handle multi-turn conversations with cost tracking."""
    
    with track("conversation_session", 
               customer_id=customer_id, team="chat-team") as span:
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=conversation_history,
            
            # Governance attributes
            team="customer-support",
            customer_id=customer_id,
            conversation_type="support"
        )
        
        # Track conversation metrics
        span.set_attribute("turn_count", len(conversation_history))
        span.set_attribute("total_chars", sum(len(msg.get("content", "")) for msg in conversation_history))
        
        return response.content[0].text
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

### Issue: Anthropic authentication errors

```bash
# Verify API key format
python -c "
import os
key = os.getenv('ANTHROPIC_API_KEY')
print('API key set:', bool(key))
print('Correct format:', key.startswith('sk-ant-') if key else False)
"
```

### Issue: Cost tracking not working

```python
# Verify instrumentation
from genops.providers.anthropic import validate_setup
result = validate_setup()
if not result.is_valid:
    print("Setup issues found - check validation output")
```

### Issue: Model not found errors

```python
# Check available models
from anthropic import Anthropic
client = Anthropic()

# Use current model names
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",  # Latest Sonnet
    # model="claude-3-5-haiku-20241022", # Latest Haiku  
    # model="claude-3-opus-20240229",    # Opus
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 📚 Next Steps

Once you have basic telemetry working:

1. **[Complete Integration Guide](integrations/anthropic.md)** - Comprehensive documentation
2. **[Examples](examples/anthropic/)** - Practical implementation patterns  
3. **[Multi-Provider Costs](examples/multi_provider_costs.py)** - Compare Claude with other providers
4. **[Policy Enforcement](examples/governance_scenarios/)** - Governance and compliance

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)
- **Documentation**: [Complete Docs](https://docs.genops.ai)
- **Anthropic Docs**: [Claude API Documentation](https://docs.anthropic.com/claude/reference/)

---

**🎉 You now have complete governance telemetry for your Anthropic Claude application!**

Your telemetry includes cost tracking, performance metrics, error monitoring, and governance attribution - all with minimal code changes.