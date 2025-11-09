# Together AI Quickstart Guide

Get started with Together AI + GenOps governance in under 5 minutes. This guide provides the essential patterns for immediate productivity.

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies (30 seconds)

```bash
# Install GenOps with Together AI support
pip install genops-ai[together] together

# Or install separately
pip install genops-ai together
```

### 2. Set Your API Key (30 seconds)

```bash
# Get your API key from: https://api.together.xyz/settings/api-keys
export TOGETHER_API_KEY="your_together_api_key_here"
```

### 3. Validate Setup (60 seconds)

```python
# Verify everything is working
from genops.providers.together_validation import validate_together_setup, print_validation_result

result = validate_together_setup()
print_validation_result(result)
```

Expected output:
```
✅ Together AI + GenOps Setup Validation
✅ API Key: Valid format and authenticated
✅ Dependencies: All required packages installed
✅ Connectivity: Successfully connected to Together AI
✅ Model Access: 200+ models available
```

### 4. Zero-Code Auto-Instrumentation (60 seconds)

```python
# Add ONE line to existing Together AI code for full governance
from genops.providers.together import auto_instrument
auto_instrument()

# Your existing Together AI code works unchanged with automatic governance
from together import Together
client = Together()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello! Explain AI governance in one sentence."}],
    max_tokens=50
)

print(response.choices[0].message.content)
# ✅ Automatic cost tracking, governance, and observability added!
```

### 5. Manual Governance Control (120 seconds)

```python
# Full control with explicit governance
from genops.providers.together import GenOpsTogetherAdapter, TogetherModel

# Create adapter with governance settings
adapter = GenOpsTogetherAdapter(
    team="your-team",
    project="quickstart-demo", 
    daily_budget_limit=5.0,
    governance_policy="advisory"
)

# Chat with automatic governance tracking
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "What are the benefits of open-source AI models?"}],
    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
    max_tokens=100
)

print(f"Response: {result.response}")
print(f"Cost: ${result.cost:.6f}")
print(f"Model: {result.model_used}")
```

## 🎯 **You're Ready!** 

In 5 minutes you now have:
- ✅ Together AI + GenOps governance working
- ✅ Automatic cost tracking and attribution  
- ✅ Access to 200+ open-source models
- ✅ Production-ready governance controls
- ✅ Up to 10x cost savings vs proprietary models

## 🚀 Next Steps (Optional)

### Explore Cost Optimization
```python
# Smart model selection based on task and budget
from genops.providers.together_pricing import TogetherPricingCalculator

calc = TogetherPricingCalculator()
recommendation = calc.recommend_model(
    task_complexity="simple",
    budget_per_operation=0.001
)

print(f"Recommended: {recommendation['recommended_model']}")
print(f"Estimated cost: ${recommendation['estimated_cost']:.6f}")
```

### Session Tracking
```python
# Track multiple operations in a session
with adapter.track_session("quickstart-session") as session:
    for i in range(3):
        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": f"Quick question {i+1}"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            session_id=session.session_id,
            max_tokens=30
        )
    
    print(f"Session cost: ${session.total_cost:.6f}")
    print(f"Operations: {session.total_operations}")
```

### Budget Enforcement
```python
# Create adapter with strict budget controls
budget_adapter = GenOpsTogetherAdapter(
    team="budget-demo",
    project="cost-control",
    daily_budget_limit=1.0,
    governance_policy="enforced"  # Blocks operations that exceed budget
)

# Operations automatically respect budget limits
result = budget_adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Budget-controlled operation"}],
    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
    max_tokens=50
)
```

## 🛠️ Troubleshooting

### API Key Issues
```bash
# Check API key format (should start with 'sk-' or 'pk-')
echo $TOGETHER_API_KEY

# Test API access directly
python -c "from together import Together; print(len(Together().models.list().data))"
```

### Import Errors
```bash
# Verify installations
pip show genops-ai together

# Reinstall if needed
pip install --upgrade genops-ai[together] together
```

### No Models Available
```python
# Check model access
from genops.providers.together_validation import validate_model_access

models, error = validate_model_access("your_api_key")
if models:
    print(f"✅ {len(models)} models available")
else:
    print(f"❌ {error.message}")
```

### Budget Issues
```python
# Check current usage
cost_summary = adapter.get_cost_summary()
print(f"Daily usage: ${cost_summary['daily_costs']:.6f}")
print(f"Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
```

## 📚 Learn More

- **[Complete Examples](../examples/together/)** - 7 comprehensive examples from basic to enterprise
- **[Together AI Integration Guide](./integrations/together.md)** - Full documentation and advanced patterns
- **[Cost Optimization Guide](../examples/together/cost_optimization.py)** - Multi-model cost analysis
- **[Production Patterns](../examples/together/production_patterns.py)** - Enterprise deployment examples

## 🔗 Key Links

- **Together AI Platform**: https://api.together.xyz
- **Model Catalog**: https://docs.together.ai/docs/inference-models
- **GenOps Documentation**: https://docs.genops.ai
- **GitHub Repository**: https://github.com/genops-ai/genops-ai

---

**🏆 Success Metrics**: After this quickstart, developers achieve immediate productivity with Together AI's 200+ models under full GenOps governance, with up to 10x cost savings and complete observability.