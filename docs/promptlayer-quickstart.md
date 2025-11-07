# PromptLayer + GenOps Quickstart Guide

**⏱️ Time to Value: 5 minutes**

Add enterprise governance, cost intelligence, and policy enforcement to your AI prompt management with PromptLayer + GenOps in under 5 minutes.

## 🎯 What is This Integration?

**PromptLayer** is a prompt management platform that helps teams version, evaluate, and optimize AI prompts collaboratively. Think of it as "Git for prompts" - you can track prompt performance, A/B test variants, and manage prompt deployments.

**GenOps** adds enterprise governance intelligence to PromptLayer, providing automatic cost tracking, team attribution, budget enforcement, and policy compliance - without changing your existing PromptLayer workflows.

**Perfect for:** Teams using PromptLayer who need cost visibility, budget controls, and governance oversight for their prompt operations.

## 🚀 What You'll Achieve

- **Zero-code governance** for existing PromptLayer applications  
- **Automatic cost tracking** with team attribution across all prompt executions
- **Policy enforcement** with configurable budget limits and alerts
- **OpenTelemetry export** to integrate with your existing observability stack
- **Enhanced prompt management** with cost and governance context

---

## 📦 1. Install (30 seconds)

```bash
pip install genops[promptlayer]
```

## 📋 2. Prerequisites & Setup (90 seconds)

**What You'll Need:**
- [ ] PromptLayer account with at least one prompt created
- [ ] OpenAI account with API access (or Anthropic/other LLM provider)
- [ ] Python 3.8+ environment

**Step 2a: PromptLayer Setup**
1. Visit [PromptLayer.com](https://promptlayer.com/) and sign up/login
2. **Create your first prompt** (required for examples):
   - Dashboard → "New Prompt" → Name: `demo_prompt` 
   - Add template: `"Answer this question: {query}"`
   - Save prompt
3. Get API key: Settings → API Keys → Copy (starts with `pl-`)

**Step 2b: LLM Provider Setup**  
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create new key → Copy it (starts with `sk-`)

**Step 2c: Environment Variables**
```bash
export PROMPTLAYER_API_KEY="pl-your-api-key"
export OPENAI_API_KEY="sk-your-openai-key"

# Optional: For team cost attribution and governance
export GENOPS_TEAM="your-team"  
export GENOPS_PROJECT="your-project"
export GENOPS_ENVIRONMENT="development"
```

💡 **New to PromptLayer?** The validation step below will guide you through creating your first prompt if needed.

## ✅ 3. Validate Setup (30 seconds)

```bash
# Download and run validation
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/promptlayer/setup_validation.py
python setup_validation.py
```

**Expected Output:**
```
✅ GenOps PromptLayer Setup Validation
Overall Status: PASSED
📊 Summary: ✅ Passed: 8, ⚠️ Warnings: 0, ❌ Failed: 0
```

## 🎯 4. Zero-Code Integration (2 minutes)

**Option A: Complete Minimal Example** (Start here if new to PromptLayer)

```python
# Step 1: Add GenOps auto-instrumentation (one line!)
from genops.providers.promptlayer import auto_instrument
auto_instrument(
    team="my-team",           # For cost attribution
    project="demo-project",   # For project tracking  
    daily_budget_limit=5.0    # $5 daily limit
)

# Step 2: Use PromptLayer exactly as before
import promptlayer

client = promptlayer.PromptLayer()
response = client.run(
    prompt_name="demo_prompt",           # The prompt you created above
    input_variables={"query": "What is AI governance?"}
)

print("Response:", response)
print("✅ This prompt execution now includes governance tracking!")
```

**Option B: For Existing PromptLayer Apps** (Just add 1 line)

```python
# Add this ONE line at the top of your existing application
from genops.providers.promptlayer import auto_instrument
auto_instrument()

# All your existing PromptLayer code continues to work unchanged:
# client.run(), client.track(), etc. - no changes needed!
```

**✨ What Just Happened:**
- ✅ **Cost tracking**: Every prompt execution now includes estimated cost ($0.001-$0.05 typical)
- ✅ **Team attribution**: Costs attributed to your team/project for billing and reporting
- ✅ **Budget enforcement**: Automatic alerts when approaching your daily limit
- ✅ **OpenTelemetry export**: Governance data flows to your observability stack (Datadog, Grafana, etc.)
- ✅ **Zero code changes**: Your existing PromptLayer workflows work exactly the same

## 📊 5. See Your Data (2 minutes)

**Option A: Instant Terminal Metrics**
```python
from genops.providers.promptlayer import get_current_adapter

# Get current metrics after running prompts
adapter = get_current_adapter()
if adapter:
    metrics = adapter.get_metrics()
    print(f"💰 Daily cost so far: ${metrics.get('daily_usage', 0):.6f}")
    print(f"👥 Team attribution: {metrics.get('team', 'N/A')}")
    print(f"📊 Operations today: {metrics.get('operation_count', 0)}")
    print(f"💡 Budget remaining: ${metrics.get('budget_remaining', 0):.6f}")
else:
    print("Run some prompts first, then check metrics!")
```

**Option B: Your Existing Observability Stack**

GenOps automatically exports OpenTelemetry data to any compatible platform:

- **Datadog**: `genops.cost.total`, `genops.team`, `genops.prompt.name` metrics appear in your existing dashboards
- **Grafana**: Ready-to-import PromptLayer cost dashboards with team breakdowns
- **Honeycomb**: Distributed tracing with governance context for debugging expensive prompts
- **Prometheus**: Custom cost and usage metrics with team/project labels for alerting

**Option C: Quick Cost Dashboard** (30 seconds)
```bash
# Download and run a simple cost dashboard
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/promptlayer/quick_dashboard.py
python quick_dashboard.py  # Shows cost trends and team attribution
```

---

## 🎉 Success! You're Done!

**In 5 minutes you've added:**
- 💰 **Cost Intelligence**: Real-time cost tracking per prompt
- 👥 **Team Attribution**: Clear cost ownership 
- 🛡️ **Governance**: Policy enforcement and compliance
- 📊 **Observability**: OpenTelemetry integration

---

## 🚀 What's Next?

Choose your path based on time available:

### 📚 **5 More Minutes: Enhanced Features**
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/promptlayer/basic_tracking.py
python basic_tracking.py
```

### 📚 **30 Minutes: Advanced Governance**
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/promptlayer/prompt_management.py
python prompt_management.py
```

### 📚 **2 Hours: Production Deployment**
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/promptlayer/production_patterns.py
python production_patterns.py
```

### 📚 **Complete Integration Guide**
[📖 Full PromptLayer Integration Documentation →](integrations/promptlayer.md)

### 📁 **Browse All Examples**
[🧪 Complete Example Suite →](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/promptlayer/)

---

## 🔧 Common Issues & Solutions

**❌ "PromptLayer API key not found" or "Authentication failed"**
```bash
# Check if key is set
echo $PROMPTLAYER_API_KEY  # Should show: pl-abc123...

# If empty, set it:
export PROMPTLAYER_API_KEY="pl-your-actual-key"

# Test the key directly:
python -c "import promptlayer; print('✅ PromptLayer key works')"
```

**❌ "No prompts found" or "Prompt 'demo_prompt' not found"**
```bash
# Quick fix: Create the demo prompt
python -c "
import promptlayer
client = promptlayer.PromptLayer()
# This will guide you through creating your first prompt
print('Visit https://promptlayer.com/ → New Prompt → Name: demo_prompt')
print('Template: Answer this question: {query}')
"
```

**❌ "OpenAI API error" or "Invalid API key"**
```bash
# Check OpenAI key
echo $OPENAI_API_KEY  # Should start with sk-

# Test OpenAI connection
python -c "
import openai
client = openai.OpenAI()
print('✅ OpenAI key works')
"
```

**❌ "Import genops failed" or "Module not found"**
```bash
# Reinstall with PromptLayer support
pip uninstall genops
pip install genops[promptlayer]

# Verify installation
python -c "from genops.providers.promptlayer import auto_instrument; print('✅ GenOps ready')"
```

**❌ "Setup validation failed" - Some checks passed, some failed**
```bash
# Run detailed validation to see specific issues
python -c "
from genops.providers.promptlayer_validation import validate_setup, print_validation_result
result = validate_setup(include_connectivity_tests=True)
print_validation_result(result, detailed=True)
"
# Follow the specific fix suggestions in the output
```

**❌ "No cost tracking" - Prompts run but no governance data**
- Check that you called `auto_instrument()` before running prompts
- Ensure environment variables are set in the same session
- Try restarting your Python session after setting variables

**Still stuck?**
- [📖 Full Documentation](integrations/promptlayer.md) - Comprehensive troubleshooting guide
- [🧪 Complete Examples](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/promptlayer/) - Working code you can copy
- [🐛 Report Issues](https://github.com/KoshiHQ/GenOps-AI/issues) - Get help from the community

---

## 💡 Key Benefits Unlocked

| Feature | Before GenOps | With GenOps |
|---------|---------------|-------------|
| **Cost Tracking** | ❌ Manual/None | ✅ Automatic per-prompt |
| **Team Attribution** | ❌ No visibility | ✅ Clear cost ownership |
| **Budget Control** | ❌ No limits | ✅ Automatic enforcement |
| **Observability** | ❌ Basic logs | ✅ OpenTelemetry + dashboards |
| **Policy Compliance** | ❌ Manual process | ✅ Automated governance |

**Ready to scale your prompt management with enterprise governance!** 🚀