# Anyscale Integration - 5-Minute Quickstart

**🎯 Get GenOps tracking for Anyscale Endpoints in 5 minutes**

This guide gets you from zero to tracking Anyscale LLM costs and governance in under 5 minutes.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Anyscale API key**
   - Get it from: [https://console.anyscale.com/credentials](https://console.anyscale.com/credentials)
   - Create a new API key if you don't have one

2. **Set your API key as environment variable**
   ```bash
   export ANYSCALE_API_KEY='your-api-key-here'
   ```

3. **Verify API key is set**
   ```bash
   echo $ANYSCALE_API_KEY  # Should show your key
   ```

---

## ⚡ Quick Setup (2 minutes)

### Step 1: Install GenOps (30 seconds)
```bash
pip install genops-ai
```

### Step 2: Verify Setup (30 seconds)
Run this validation script to check everything is working:

```python
from genops.providers.anyscale.validation import validate_setup, print_validation_result

# Check your Anyscale setup
result = validate_setup()
print_validation_result(result)
```

You should see: ✅ **Status: PASSED**

**What if validation fails?**

If you see ❌ **Status: FAILED**, don't worry! Here's how to fix common issues:

#### Issue: ANYSCALE_API_KEY not set

**Error message:**
```
❌ Configuration: ANYSCALE_API_KEY environment variable not set
```

**Fix:**
```bash
# Set your API key
export ANYSCALE_API_KEY='your-key-from-console'

# Verify it's set
echo $ANYSCALE_API_KEY
```

Get your API key from: [https://console.anyscale.com/credentials](https://console.anyscale.com/credentials)

#### Issue: Authentication Failed

**Error message:**
```
❌ Connectivity: Authentication failed - invalid API key
```

**Fixes:**
1. Verify your API key is correct at [https://console.anyscale.com/credentials](https://console.anyscale.com/credentials)
2. Check for extra spaces when copying the key:
   ```bash
   # Wrong (has trailing space)
   export ANYSCALE_API_KEY='abc123 '

   # Correct (no trailing space)
   export ANYSCALE_API_KEY='abc123'
   ```
3. Ensure the key hasn't expired - create a new one if needed

#### Issue: Connection Timeout

**Error message:**
```
❌ Connectivity: Cannot reach Anyscale API endpoints
```

**Fixes:**
1. Check your internet connection
2. Verify firewall settings allow HTTPS to `api.endpoints.anyscale.com`
3. If behind corporate proxy, configure proxy settings:
   ```bash
   export HTTPS_PROXY='http://proxy.company.com:8080'
   ```

#### Re-run Validation After Fixes

After fixing any issues, run validation again:

```python
from genops.providers.anyscale import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)
```

**Continue to Step 3 when you see: ✅ Status: PASSED**

### Step 3: Test Basic Tracking (60 seconds)
Create this minimal test file:

```python
# test_anyscale_genops.py
from genops.providers.anyscale import instrument_anyscale

# Create GenOps adapter with governance tracking
adapter = instrument_anyscale(
    team="ml-research",
    project="quickstart-test"
)

print("🚀 Testing Anyscale with GenOps tracking...")

# Make a completion request (costs and governance automatically tracked)
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    customer_id="demo-customer"  # Governance attribute
)

print(f"📝 Response: {response['choices'][0]['message']['content']}")
print("✅ SUCCESS! GenOps is now tracking your Anyscale usage")
```

**Run it:**
```bash
python test_anyscale_genops.py
```

**Expected output:**
```
🚀 Testing Anyscale with GenOps tracking...
📝 Response: The capital of France is Paris...
✅ SUCCESS! GenOps is now tracking your Anyscale usage
```

---

## 🎯 What Just Happened?

**GenOps automatically tracked:**
- ✅ **Token usage** (input and output tokens)
- ✅ **Cost attribution** ($0.00015 for ~150 tokens at $1/M token rate)
- ✅ **Team attribution** (costs attributed to "ml-research" team)
- ✅ **Customer tracking** (usage tied to "demo-customer")
- ✅ **OpenTelemetry traces** (complete distributed tracing with governance attributes)

**All with built-in governance from the start!**

---

## 📊 See Your Data (1 minute)

### Option 1: Check Token Usage and Cost
```python
from genops.providers.anyscale import instrument_anyscale

adapter = instrument_anyscale(
    team="ml-research",
    project="cost-analysis"
)

response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=200
)

# Token usage is in the response
usage = response['usage']
print(f"📊 Token Usage:")
print(f"   Input tokens: {usage['prompt_tokens']}")
print(f"   Output tokens: {usage['completion_tokens']}")
print(f"   Total tokens: {usage['total_tokens']}")

# Calculate cost
from genops.providers.anyscale import calculate_completion_cost
cost = calculate_completion_cost(
    model="meta-llama/Llama-2-70b-chat-hf",
    input_tokens=usage['prompt_tokens'],
    output_tokens=usage['completion_tokens']
)
print(f"💰 Total Cost: ${cost:.6f}")
```

### Option 2: Get Model Pricing Information
```python
from genops.providers.anyscale import get_model_pricing

# Check pricing for any model
pricing = get_model_pricing("meta-llama/Llama-2-70b-chat-hf")
print(f"Model: {pricing.model_name}")
print(f"Input cost: ${pricing.input_cost_per_million}/M tokens")
print(f"Output cost: ${pricing.output_cost_per_million}/M tokens")
print(f"Context window: {pricing.context_window} tokens")
```

### Option 3: Compare Model Costs
```python
from genops.providers.anyscale.pricing import AnyscalePricing

pricing = AnyscalePricing()

# Get cost alternatives
alternatives = pricing.get_model_alternatives("meta-llama/Llama-2-70b-chat-hf")
print("💡 Cost-effective alternatives:")
for model, cost_ratio, description in alternatives:
    print(f"   {model}: {description}")
```

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps tracking your Anyscale usage!**

### Option A: Add More Governance Attributes
```python
adapter = instrument_anyscale(
    team="ml-platform",
    project="production-chatbot",
    environment="production",
    cost_center="engineering"
)

# Per-request attributes
response = adapter.completion_create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    customer_id="enterprise-client-123",
    feature="chat-completion"
)
```

### Option B: Use Context Manager for Multi-Step Workflows
```python
with adapter.governance_context(customer_id="acme-corp", feature="batch-processing"):
    # All operations in this block inherit governance attributes
    response1 = adapter.completion_create(...)
    response2 = adapter.completion_create(...)
    embeddings = adapter.embeddings_create(...)
```

### Option C: Try Different Models
```python
# Smaller, faster model for simple tasks
response = adapter.completion_create(
    model="meta-llama/Llama-2-7b-chat-hf",  # 10x cheaper!
    messages=[{"role": "user", "content": "Hello!"}]
)

# Embedding model for vector search
embeddings = adapter.embeddings_create(
    model="thenlper/gte-large",
    input="Document text to embed"
)
```

### Option D: Integrate with Your Observability Stack
GenOps exports OpenTelemetry traces automatically. Configure your preferred backend:

**Datadog:**
```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(
    endpoint="http://localhost:4317"
))
provider.add_span_processor(processor)
```

**Honeycomb / Grafana Tempo / etc:** Similar OTLP configuration

---

## 🔍 Troubleshooting

### Issue: "ANYSCALE_API_KEY not set"
**Fix:**
```bash
export ANYSCALE_API_KEY='your-api-key-here'
# Verify it's set:
echo $ANYSCALE_API_KEY
```

### Issue: "Authentication Failed"
**Fix:** Verify your API key is valid:
- Go to [https://console.anyscale.com/credentials](https://console.anyscale.com/credentials)
- Create a new API key if needed
- Make sure there are no extra spaces when setting the environment variable

### Issue: "Model not found"
**Fix:** Check available models:
```python
# List Anyscale Endpoints models
from genops.providers.anyscale.pricing import ANYSCALE_PRICING
print("Available models:")
for model in ANYSCALE_PRICING.keys():
    print(f"  - {model}")
```

### Issue: "Connection timeout"
**Fix:** Check network connectivity and firewall settings. Anyscale Endpoints requires outbound HTTPS access to `api.endpoints.anyscale.com`.

---

## 📚 Learn More

- **Full Integration Guide:** [docs/integrations/anyscale.md](integrations/anyscale.md)
- **Anyscale Endpoints Docs:** [https://docs.anyscale.com](https://docs.anyscale.com)
- **GenOps Documentation:** [README.md](../README.md)
- **GitHub Repository:** [https://github.com/KoshiHQ/GenOps-AI](https://github.com/KoshiHQ/GenOps-AI)

---

## 💬 Get Help

- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**🎉 Congratulations! You're now tracking Anyscale costs with governance.**

Total time: **~5 minutes** ✅
