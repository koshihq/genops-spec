# Cohere Integration - 5-Minute Quickstart

**🎯 Get GenOps tracking for Cohere AI models in 5 minutes**

This guide gets you from zero to tracking Cohere costs and performance with GenOps in under 5 minutes.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Cohere API key**
   ```bash
   # Get your API key from https://dashboard.cohere.ai/
   export CO_API_KEY="your-cohere-api-key-here"
   ```

2. **Install Cohere client** (if not already installed)
   ```bash
   pip install cohere
   ```

3. **Verify Cohere access**
   ```bash
   python -c "import cohere; print('Cohere client ready')"
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
from genops.providers.cohere_validation import validate_setup, print_validation_result

# Check your Cohere setup
result = validate_setup()
print_validation_result(result)
```

You should see: ✅ **Overall Status: PASSED**

### Step 3: Test Basic Tracking (60 seconds)
Create this minimal test file:

```python
# test_cohere_genops.py
from genops.providers.cohere import instrument_cohere

# Enable GenOps tracking for Cohere (zero code changes needed!)
adapter = instrument_cohere(team="ai-team", project="quickstart-test")

print("🚀 Testing Cohere with GenOps tracking...")

# Generate text (costs and performance automatically tracked)
response = adapter.chat(
    message="What is the capital of France?",
    model="command-light"
)

print(f"📝 Response: {response.content[:100]}...")
print(f"💰 Cost: ${response.usage.total_cost:.6f}")
print("✅ SUCCESS! GenOps is now tracking your Cohere usage")
```

**Run it:**
```bash
python test_cohere_genops.py
```

**Expected output:**
```
🚀 Testing Cohere with GenOps tracking...
📝 Response: The capital of France is Paris. Paris is located in the north-central part of France...
💰 Cost: $0.000015
✅ SUCCESS! GenOps is now tracking your Cohere usage
```

---

## 🎯 What Just Happened?

**GenOps automatically tracked:**
- ✅ **Token-based costs** (input/output tokens with precise pricing)
- ✅ **Operation performance** (latency, tokens per second)
- ✅ **Team attribution** (costs attributed to "ai-team" and "quickstart-test")
- ✅ **Model efficiency** (cost per operation, tokens per dollar)

**All with zero changes to your Cohere workflow!**

---

## 📊 See Your Data (1 minute)

### Option 1: Get Usage Summary
```python
from genops.providers.cohere import instrument_cohere

adapter = instrument_cohere(team="analytics-team")

# Run some operations first...
response1 = adapter.chat(message="Hello", model="command-light")
response2 = adapter.embed(texts=["test document"], model="embed-english-v4.0")

# Get comprehensive usage summary
summary = adapter.get_usage_summary()
print(f"💰 Total Cost: ${summary['total_cost']:.6f}")
print(f"🔢 Operations: {summary['total_operations']}")
print(f"⚡ Avg Cost/Op: ${summary['average_cost_per_operation']:.6f}")
```

### Option 2: Multi-Operation Tracking
```python
from genops.providers.cohere import instrument_cohere

adapter = instrument_cohere(team="research-team", project="multi-modal-ai")

# Text generation
chat_response = adapter.chat(
    message="Explain machine learning",
    model="command-r-plus-08-2024"
)

# Text embedding
embed_response = adapter.embed(
    texts=["machine learning", "artificial intelligence", "neural networks"],
    model="embed-english-v4.0"
)

# Document reranking
rerank_response = adapter.rerank(
    query="machine learning",
    documents=["ML is about algorithms", "AI includes ML", "Neural networks are ML models"],
    model="rerank-english-v3.0"
)

print(f"💬 Chat cost: ${chat_response.usage.total_cost:.6f}")
print(f"📊 Embedding cost: ${embed_response.usage.total_cost:.6f}")
print(f"🔍 Rerank cost: ${rerank_response.usage.total_cost:.6f}")
```

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps tracking all your Cohere operations!**

**Choose your next adventure:**

### 🎯 **30-Second Next Step: Try Different Models**
```python
# Compare costs across Cohere models
from genops.providers.cohere import instrument_cohere

adapter = instrument_cohere(team="research", project="model-comparison")

models = ["command-light", "command-r-08-2024", "command-r-plus-08-2024"]
prompt = "Explain quantum computing in one paragraph"

for model in models:
    response = adapter.chat(message=prompt, model=model)
    print(f"✅ {model}: ${response.usage.total_cost:.6f} ({response.usage.total_tokens} tokens)")
```

### 🚀 **5-Minute Next Step: Advanced Features**
- **[Auto-Instrumentation Guide](../examples/cohere/auto_instrumentation.py)** - Zero-code integration
- **[Cost Optimization Examples](../examples/cohere/cost_optimization.py)** - Model comparison and efficiency
- **[Multi-Operation Workflows](../examples/cohere/embedding_optimization.py)** - Embedding and rerank optimization

### 📚 **15-Minute Next Step: Complete Integration**
- **[Complete Cohere Integration Guide](../docs/integrations/cohere.md)** - Full reference documentation
- **[All Cohere Examples](../examples/cohere/)** - Progressive complexity tutorials

---

## 🆘 Troubleshooting

**Getting errors? Here are quick fixes:**

### ❌ "Invalid API key" or "Unauthorized"
```bash
# Make sure your API key is set correctly
echo $CO_API_KEY
# Should show your key (co_* or ck_* format)

# Or set it in Python
import os
os.environ["CO_API_KEY"] = "your-api-key-here"
```

### ❌ "No module named 'cohere'"
```bash
# Install Cohere Python client
pip install cohere

# Verify installation
python -c "import cohere; print('✅ Cohere installed')"
```

### ❌ "Model not found" or "Model not available"
```python
# Check available models for your API key
import cohere
client = cohere.ClientV2()

# Try a basic model that should be available
try:
    response = client.chat(
        model="command-light",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=1
    )
    print("✅ Cohere API working")
except Exception as e:
    print(f"❌ API Error: {e}")
```

### ❌ "Import error for genops"
```bash
# Reinstall GenOps
pip install --upgrade genops-ai
```

**Still stuck?** Run the diagnostic:
```python
from genops.providers.cohere_validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, detailed=True)
```

---

## 💡 Key Differences from Other AI Providers

**Cohere tracking is optimized for multi-operation workflows:**

| Aspect | OpenAI/Anthropic | Cohere |
|--------|------------------|--------|
| **Operations** | Mainly text generation | Generation + Embedding + Reranking |
| **Cost Model** | Simple token pricing | Multi-operation pricing (tokens + searches + embeddings) |
| **Optimization** | Model selection | Operation type optimization + model selection |
| **Specialization** | General chat/completion | Enterprise search, classification, embeddings |

**That's why GenOps Cohere integration focuses on:**
- 🔄 **Multi-operation cost tracking** (generation, embedding, rerank in one workflow)
- 💰 **Operation-specific optimization** (when to embed vs generate vs rerank)
- 🎯 **Enterprise use case support** (search, classification, document analysis)
- 📊 **Comprehensive cost attribution** across all operation types

---

## 🎉 Success!

**🎯 In 5 minutes, you've accomplished:**
- ✅ Set up GenOps tracking for Cohere AI operations
- ✅ Automatically tracked costs across generation, embedding, and rerank
- ✅ Attributed costs to teams and projects
- ✅ Got insights into multi-operation performance and efficiency

**Your Cohere AI operations now have enterprise-grade governance and cost tracking!**

**🚀 Ready for more advanced features?** Check out:
- **[Multi-Operation Examples](../examples/cohere/)**
- **[Cost Optimization Strategies](../examples/cohere/cost_optimization.py)**
- **[Complete Integration Guide](../docs/integrations/cohere.md)**

---

**Questions? Issues?** 
- 📝 [Create an issue](https://github.com/KoshiHQ/GenOps-AI/issues)
- 💬 [Join discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)