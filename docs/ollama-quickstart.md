# Ollama Integration - 5-Minute Quickstart

**🎯 Get GenOps tracking for local Ollama models in 5 minutes**

This guide gets you from zero to tracking local model costs and performance with GenOps + Ollama in under 5 minutes.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Ollama installed and running**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   ```

2. **At least one model pulled**
   ```bash
   # Pull a small, fast model for testing
   ollama pull llama3.2:1b
   
   # Or pull a more capable model if you have resources
   ollama pull llama3.2:3b
   ```

3. **Verify Ollama is working**
   ```bash
   ollama list  # Should show your downloaded models
   ```

---

## ⚡ Quick Setup (2 minutes)

### Step 1: Install GenOps (30 seconds)
```bash
pip install genops-ai[ollama]
```

### Step 2: Verify Setup (30 seconds)
Run this validation script to check everything is working:

```python
from genops.providers.ollama.validation import validate_setup, print_validation_result

# Check your Ollama setup
result = validate_setup()
print_validation_result(result)
```

You should see: ✅ **Overall Status: PASSED**

### Step 3: Test Basic Tracking (60 seconds)
Create this minimal test file:

```python
# test_ollama_genops.py
from genops.providers.ollama import auto_instrument

# Enable GenOps tracking for Ollama (zero code changes needed!)
auto_instrument(team="ai-team", project="local-testing")

# Your existing Ollama code now includes GenOps tracking
import ollama

print("🚀 Testing Ollama with GenOps tracking...")

# Generate text (costs and performance automatically tracked)
response = ollama.generate(
    model="llama3.2:1b",  # or your preferred model
    prompt="What is the capital of France?"
)

print(f"📝 Response: {response['response'][:100]}...")
print("✅ SUCCESS! GenOps is now tracking your local Ollama usage")
```

**Run it:**
```bash
python test_ollama_genops.py
```

**Expected output:**
```
🚀 Testing Ollama with GenOps tracking...
📝 Response: The capital of France is Paris. Paris is located in the north-central part of France...
✅ SUCCESS! GenOps is now tracking your local Ollama usage
```

---

## 🎯 What Just Happened?

**GenOps automatically tracked:**
- ✅ **Infrastructure costs** (GPU/CPU hours, electricity usage)
- ✅ **Resource utilization** (memory usage, inference time)  
- ✅ **Team attribution** (costs attributed to "ai-team" and "local-testing")
- ✅ **Model performance** (tokens/second, latency, success rates)

**All with zero changes to your existing Ollama code!**

---

## 📊 See Your Data (1 minute)

### Option 1: Get a Quick Summary
```python
from genops.providers.ollama import get_resource_monitor

# Get current resource monitor
monitor = get_resource_monitor()

# See current metrics
current = monitor.get_current_metrics()
print(f"🖥️ Current CPU: {current.cpu_usage_percent:.1f}%")
print(f"🎮 Current GPU: {current.gpu_usage_percent:.1f}%")
print(f"💾 Memory Usage: {current.memory_usage_mb:.0f}MB")

# Get model performance
from genops.providers.ollama import get_model_manager
manager = get_model_manager()
performance = manager.get_model_performance_summary()
print(f"📈 Model Performance: {performance}")
```

### Option 2: Get Cost Analysis
```python
from genops.providers.ollama import instrument_ollama

# Create adapter with cost tracking
adapter = instrument_ollama(team="ai-team", project="cost-analysis")

# Get operation summary
summary = adapter.get_operation_summary()
print(f"💰 Total Infrastructure Cost: ${summary['total_infrastructure_cost']:.6f}")
print(f"🔢 Total Operations: {summary['total_operations']}")
print(f"⚡ Avg Inference Time: {summary['avg_inference_time_ms']:.0f}ms")
print(f"🤖 Models Used: {', '.join(summary['models_used'])}")
```

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps tracking your local Ollama models!**

**Choose your next adventure:**

### 🎯 **30-Second Next Step: Try Different Models**
```python
# Compare performance across models
import ollama
from genops.providers.ollama import auto_instrument

auto_instrument(team="research", project="model-comparison")

# Test different models (if you have them)
models = ["llama3.2:1b", "llama3.2:3b"]
for model in models:
    try:
        response = ollama.generate(model=model, prompt="Hello!")
        print(f"✅ {model}: {response['response'][:50]}...")
    except Exception as e:
        print(f"❌ {model}: {e}")
```

### 🚀 **5-Minute Next Step: Advanced Features**
- **[Local Model Optimization Guide](../examples/ollama/local_model_optimization.py)** - Optimize costs and performance
- **[Resource Monitoring Deep Dive](../examples/ollama/)** - Comprehensive resource tracking
- **[Production Deployment](../examples/ollama/ollama_production_deployment.py)** - Enterprise patterns

### 📚 **15-Minute Next Step: Complete Integration**
- **[Complete Ollama Integration Guide](../docs/integrations/ollama.md)** - Full reference documentation
- **[All Ollama Examples](../examples/ollama/)** - Step-by-step tutorials

---

## 🆘 Troubleshooting

**Getting errors? Here are quick fixes:**

### ❌ "Cannot connect to Ollama server"
```bash
# Make sure Ollama is running
ollama serve

# Check if models are available
ollama list

# Test basic connection
curl http://localhost:11434/api/version
```

### ❌ "No module named 'ollama'"
```bash
# Install Ollama Python client
pip install ollama

# Or install without client (uses HTTP)
pip install requests
```

### ❌ "Model not found"
```bash
# List available models
ollama list

# Pull a model if none available
ollama pull llama3.2:1b
```

### ❌ "Import error for genops"
```bash
# Reinstall with Ollama support
pip install --upgrade genops-ai[ollama]
```

**Still stuck?** Run the diagnostic:
```python
from genops.providers.ollama.validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, detailed=True)
```

---

## 💡 Key Differences from Cloud Providers

**Ollama tracking is optimized for local models:**

| Aspect | Cloud Providers (OpenAI, etc.) | Ollama (Local Models) |
|--------|-------------------------------|------------------------|
| **Cost Model** | Token-based pricing | Infrastructure costs (GPU/CPU hours) |
| **Resource Tracking** | API calls only | Full system monitoring (memory, GPU) |
| **Optimization** | Model selection | Hardware utilization + model efficiency |
| **Privacy** | Data sent to cloud | Everything stays local |

**That's why GenOps Ollama integration focuses on:**
- 🖥️ **Infrastructure cost attribution** instead of token pricing
- 📊 **Resource utilization monitoring** (GPU, CPU, memory)  
- ⚡ **Local performance optimization** recommendations
- 🔒 **Complete privacy** - no data leaves your system

---

## 🎉 Success!

**🎯 In 5 minutes, you've accomplished:**
- ✅ Set up GenOps tracking for local Ollama models
- ✅ Automatically tracked infrastructure costs and performance
- ✅ Attributed costs to teams and projects
- ✅ Got insights into resource utilization

**Your local AI models now have enterprise-grade governance and cost tracking!**

**🚀 Ready for more advanced features?** Check out:
- **[Local Model Optimization Examples](../examples/ollama/)**
- **[Production Deployment Patterns](../examples/ollama/ollama_production_deployment.py)**
- **[Complete Integration Guide](../docs/integrations/ollama.md)**

---

**Questions? Issues?** 
- 📝 [Create an issue](https://github.com/KoshiHQ/GenOps-AI/issues)
- 💬 [Join discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)