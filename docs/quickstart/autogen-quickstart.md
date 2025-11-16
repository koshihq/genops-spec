# AutoGen + GenOps: 3-Step Quickstart ⚡

Add comprehensive AutoGen governance and cost tracking in **under 3 minutes** with **zero code changes** to your existing AutoGen applications.

## 3-Step Setup (Under 3 Minutes)

### Step 1: Install (30 seconds)

```bash
pip install genops[autogen]
```

### Step 2: Validate Setup (30 seconds)

```python
# Quick validation - copy/paste and run
python -c "
from genops.providers.autogen import quick_validate
result = quick_validate()
print('✅ Ready!' if result else '❌ Issues found - run setup_validation.py')
"
```

### Step 3: Enable Governance (1 line of code)

**Add one import line to your existing AutoGen code:**

```python
# Add this ONE line to any AutoGen script
from genops.providers.autogen import enable_governance; enable_governance()

# Your existing AutoGen code works unchanged (zero modifications needed!)
import autogen

config_list = [{"model": "gpt-3.5-turbo", "api_key": "your-key"}]
assistant = autogen.AssistantAgent(name="assistant", llm_config={"config_list": config_list})
user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="NEVER")

user_proxy.initiate_chat(assistant, message="Hello, AutoGen!")
# ↑ This conversation now has comprehensive governance tracking!
```

**That's it!** Your AutoGen conversations now have enterprise-grade governance.

## What You Get Automatically

✅ **Cost Tracking**: Real-time cost calculation across all LLM providers  
✅ **Budget Monitoring**: Automatic alerts when approaching spending limits  
✅ **Conversation Analytics**: Turn-by-turn analysis and performance metrics  
✅ **Agent Performance**: Individual agent monitoring and optimization insights  
✅ **OpenTelemetry Export**: Standard telemetry compatible with 15+ observability platforms  
✅ **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, and 20+ providers  

## Quick Validation

Validate your setup works correctly:

```python
from genops.providers.autogen import validate_autogen_setup, print_validation_result

result = validate_autogen_setup()
print_validation_result(result)
```

## View Your Data

### Option 1: Built-in Summary
```python
from genops.providers.autogen import get_current_adapter

adapter = get_current_adapter()
summary = adapter.get_session_summary()
print(f"Total conversations: {summary['total_conversations']}")
print(f"Total cost: ${summary['total_cost']:.4f}")
print(f"Budget utilization: {summary['budget_utilization']:.1f}%")
```

### Option 2: Cost Analysis
```python
from genops.providers.autogen import analyze_conversation_costs

analysis = analyze_conversation_costs(adapter, time_period_hours=24)
print(f"Total cost: ${analysis['total_cost']}")

for rec in analysis['recommendations']:
    print(f"💡 {rec['reasoning']}")
```

### Option 3: Observability Platform
GenOps exports standard OpenTelemetry data that works with:
- Datadog
- Grafana + Tempo
- Honeycomb  
- New Relic
- Any OTLP-compatible platform

## Real Example

Here's a complete working example:

```python
import os
import autogen
from genops.providers.autogen import auto_instrument

# Enable governance (one line!)
adapter = auto_instrument(
    team="ai-research", 
    project="customer-service", 
    daily_budget_limit=25.0
)

# Standard AutoGen setup
config_list = [{
    "model": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_API_KEY")
}]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful customer service assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2
)

# Your conversation (now tracked!)
user_proxy.initiate_chat(
    assistant, 
    message="I need help with my order status. Order #12345."
)

# Check results
summary = adapter.get_session_summary()
print(f"💰 Conversation cost: ${summary['total_cost']:.4f}")
print(f"📊 Budget used: {summary['budget_utilization']:.1f}%")
```

## Next Steps

🎯 **Ready for more?** Check out these advanced patterns:

- **[Group Chat Monitoring](../examples/autogen/multi_agent_group_chat_monitoring.py)** - Track multi-agent group conversations
- **[Code Execution Tracking](../examples/autogen/code_execution_governance.py)** - Monitor AutoGen's code interpreter
- **[Production Deployment](../examples/autogen/production_deployment_patterns.py)** - Enterprise governance patterns
- **[Cost Optimization](../examples/autogen/performance_optimization.py)** - Advanced cost reduction strategies

🔍 **Want comprehensive docs?** See the [complete AutoGen integration guide](../integrations/autogen.md)

## Troubleshooting

### Common Issues

**❌ "AutoGen not installed"**
```bash
pip install pyautogen
```

**❌ "No API key found"**
```bash
export OPENAI_API_KEY=your_key_here
# or set in your code: os.environ["OPENAI_API_KEY"] = "your_key"
```

**❌ "GenOps import error"**
```bash
pip install --upgrade genops
```

### Quick Diagnostics

Run the validation to identify issues:

```python
from genops.providers.autogen import validate_autogen_setup, print_validation_result

result = validate_autogen_setup(verify_connectivity=True)
print_validation_result(result, verbose=True)
```

### Get Help

- 📖 [Complete AutoGen Documentation](../integrations/autogen.md)
- 🐛 [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)  
- 💬 [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**You're now ready to use AutoGen with comprehensive governance tracking!** 🚀

The zero-code instrumentation means your existing AutoGen applications work unchanged while gaining enterprise-grade cost tracking, budget monitoring, and observability integration.