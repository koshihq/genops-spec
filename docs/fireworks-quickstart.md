# Fireworks AI Quickstart Guide

**🎯 What you'll learn:** Get Fireworks AI's 4x faster inference + complete cost governance working in exactly 5 minutes, with zero code changes to your existing applications.

## What is GenOps?

**GenOps AI** is a governance telemetry layer built on OpenTelemetry that provides cost tracking, budget enforcement, and compliance monitoring for AI systems. It extends your existing observability stack with AI-specific governance capabilities without replacing your current tools.

**Why this matters for Fireworks AI:**
- **4x Speed + Cost Tracking**: Get Fireworks' speed advantage with automatic cost attribution
- **100+ Model Governance**: Track costs across Fireworks' entire model ecosystem
- **50% Batch Savings**: Automatic optimization for high-volume workloads
- **Zero Migration Pain**: Add governance to existing Fireworks AI code with one line

**Key Benefits:**
- **Cost Transparency**: Real-time cost tracking across all AI operations
- **Budget Controls**: Configurable spending limits with enforcement policies
- **Multi-tenant Governance**: Per-team, per-project, per-customer attribution
- **Vendor Independence**: Works with 15+ observability platforms via OpenTelemetry
- **Zero Code Changes**: Auto-instrumentation for existing applications

Get started with Fireworks AI + GenOps governance in under 5 minutes. This guide provides the essential patterns for immediate productivity with Fireworks AI's 100+ models and 4x faster inference.

**⏱️ Time commitment:** 5 minutes | **✅ Result:** Full Fireworks AI governance with cost tracking

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies (30 seconds)

```bash
# Install GenOps with Fireworks AI support
pip install genops-ai[fireworks] fireworks-ai

# Or install separately
pip install genops-ai fireworks-ai
```

### 2. Set Your API Key (30 seconds)

**🔑 API Key Setup:**
1. Get your free API key: [fireworks.ai/api-keys](https://fireworks.ai/api-keys) (includes $1 free credit)
2. Set the environment variable:

```bash
export FIREWORKS_API_KEY="fw-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**💡 API Key Format:** Fireworks AI keys start with `fw-` followed by 32+ characters

**✅ Verification:** Your key is working if the validation step below passes

### 3. Validate Setup (60 seconds)

```python
# Verify everything is working
from genops.providers.fireworks_validation import validate_fireworks_setup, print_validation_result

result = validate_fireworks_setup()
print_validation_result(result)
```

Expected output:
```
✅ Fireworks AI + GenOps Setup Validation
✅ API Key: Valid format and authenticated
✅ Dependencies: All required packages installed
✅ Connectivity: Successfully connected to Fireworks AI
✅ Model Access: 5+ models available across all modalities
```

### 4. Zero-Code Auto-Instrumentation (60 seconds)

**🎯 The Magic:** Add ONE line to existing Fireworks AI code for complete governance

```python
# Add this single line for automatic governance
from genops.providers.fireworks import auto_instrument
auto_instrument()  # ✨ This enables automatic cost tracking and governance

# Your existing Fireworks AI code works unchanged
from fireworks.client import Fireworks
client = Fireworks()

response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    messages=[{"role": "user", "content": "Hello! Explain Fireworks AI in one sentence."}],
    max_tokens=50
)

print(response.choices[0].message.content)
# ✅ Automatic cost tracking, governance, and observability added!
```

**🔥 What just happened:**
- Your existing Fireworks AI code got automatic cost tracking
- 4x faster inference with Fireattention optimization
- Zero code changes required to your application logic
- Complete observability integration with OpenTelemetry

### 5. Manual Governance Control (120 seconds)

**🎛️ Full Control Mode:** Explicit governance configuration with model enums

```python
# Full control with explicit governance
from genops.providers.fireworks import GenOpsFireworksAdapter, FireworksModel

# Create adapter with governance settings
adapter = GenOpsFireworksAdapter(
    team="your-team",
    project="quickstart-demo", 
    daily_budget_limit=5.0,
    governance_policy="advisory"  # "advisory" warns, "enforcing" blocks
)

# Chat with automatic governance tracking
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "What are the benefits of Fireworks AI's fast inference?"}],
    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,  # Pre-defined model enum
    max_tokens=100
)

print(f"Response: {result.response}")
print(f"Cost: ${result.cost:.6f}")
print(f"Model: {result.model_used}")
print(f"Speed: {result.execution_time_seconds:.2f}s (🔥 4x faster!)")
```

**📚 About FireworksModel Enums:**
- `FireworksModel.LLAMA_3_1_8B_INSTRUCT` → Fast, cost-effective model ($0.20/1M tokens)
- `FireworksModel.LLAMA_3_1_70B_INSTRUCT` → High-quality model ($0.90/1M tokens)
- `FireworksModel.LLAMA_3_2_1B_INSTRUCT` → Ultra-fast, cheapest ($0.10/1M tokens)
- See 100+ available models in the [full integration guide](integrations/fireworks.md)

## 🎯 **You're Ready!** 

In 5 minutes you now have:
- ✅ Fireworks AI + GenOps governance working
- ✅ Automatic cost tracking and attribution  
- ✅ Access to 100+ models with 4x faster inference
- ✅ Production-ready governance controls
- ✅ Up to 10x cost savings vs proprietary models

## 🚀 Next Steps (Optional)

### Explore Cost Optimization

```python
# Smart model selection based on task and budget
from genops.providers.fireworks_pricing import FireworksPricingCalculator

calc = FireworksPricingCalculator()
recommendation = calc.recommend_model(
    task_complexity="simple",
    budget_per_operation=0.001
)

print(f"Recommended: {recommendation.recommended_model}")
print(f"Estimated cost: ${recommendation.estimated_cost:.6f}")
```

### Session Tracking

```python
# Track multiple operations in a session
with adapter.track_session("quickstart-session") as session:
    for i in range(3):
        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": f"Quick question {i+1}"}],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            session_id=session.session_id,
            max_tokens=30
        )
    
    print(f"Session cost: ${session.total_cost:.6f}")
    print(f"Operations: {session.total_operations}")
```

### Budget Enforcement

```python
# Create adapter with strict budget controls
budget_adapter = GenOpsFireworksAdapter(
    team="budget-demo",
    project="cost-control",
    daily_budget_limit=1.0,
    governance_policy="enforced"  # Blocks operations that exceed budget
)

# Operations automatically respect budget limits
result = budget_adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Budget-controlled operation"}],
    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
    max_tokens=50
)
```

### Multi-Modal Operations

```python
# Vision-language analysis with cost tracking
result = adapter.chat_with_governance(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    model=FireworksModel.LLAMA_VISION_11B,
    feature="image-analysis"
)

print(f"Vision analysis: {result.response}")
print(f"Multimodal cost: ${result.cost:.6f}")
```

### Embeddings with Governance

```python
# Generate embeddings with cost tracking
result = adapter.embeddings_with_governance(
    input_texts=["Document to embed", "Another document"],
    model=FireworksModel.NOMIC_EMBED_TEXT,
    feature="semantic-search"
)

print(f"Generated embeddings with cost: ${result.cost:.6f}")
```

## 🛠️ Troubleshooting

### API Key Issues
```bash
# Check API key format (should start with valid Fireworks format)
echo $FIREWORKS_API_KEY

# Test API access directly
python -c "from fireworks.client import Fireworks; print('Connected!' if Fireworks().chat else 'Failed')"
```

### Import Errors
```bash
# Verify installations
pip show genops-ai fireworks-ai

# Reinstall if needed
pip install --upgrade genops-ai[fireworks] fireworks-ai
```

### No Models Available
```python
# Check model access
from genops.providers.fireworks_validation import validate_model_access

models, error = validate_model_access("your_api_key")
if models:
    print(f"✅ {len(models)} models available")
else:
    print(f"❌ {error}")
```

### Budget Issues
```python
# Check current usage
cost_summary = adapter.get_cost_summary()
print(f"Daily usage: ${cost_summary['daily_costs']:.6f}")
print(f"Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
```

### Performance Issues
```python
# Test performance
from genops.providers.fireworks_validation import get_performance_metrics

metrics = get_performance_metrics()
print(f"API latency: {metrics.get('connectivity_latency_ms', 0):.0f}ms")
print(f"Throughput: {metrics.get('tokens_per_second', 0):.1f} tokens/s")
```

## 📚 Learn More

**🎯 Next Learning Paths:**
- **[Complete Examples](../../examples/fireworks/)** - 7 comprehensive examples from basic to enterprise
- **[Full Integration Guide](integrations/fireworks.md)** - Complete documentation and advanced patterns
- **[Cost Optimization Examples](../../examples/fireworks/cost_optimization.py)** - Multi-model cost analysis
- **[Production Patterns](../../examples/fireworks/production_patterns.py)** - Enterprise deployment examples

**🔍 Interactive Tools:**
- **[Setup Wizard](../../examples/fireworks/interactive_setup_wizard.py)** - Guided team onboarding
- **[Setup Validation](../../examples/fireworks/setup_validation.py)** - Test your configuration
- **[Auto-Instrumentation](../../examples/fireworks/auto_instrumentation.py)** - Zero-code integration

## 🔗 Key Resources

**🔥 Fireworks AI:**
- **Platform Dashboard**: https://fireworks.ai
- **100+ Model Catalog**: https://fireworks.ai/models  
- **API Documentation**: https://docs.fireworks.ai
- **Performance Benchmarks**: https://fireworks.ai/blog/fireattention-4x-faster-inference

**🛠️ GenOps Platform:**
- **Documentation Hub**: https://docs.genops.ai
- **GitHub Repository**: https://github.com/KoshiHQ/GenOps-AI
- **Community Discussions**: https://github.com/KoshiHQ/GenOps-AI/discussions

---

**🏆 Success Metrics**: After this quickstart, developers achieve immediate productivity with Fireworks AI's 100+ models under full GenOps governance, with 4x faster inference and complete observability.