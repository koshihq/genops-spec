# Griptape QuickStart Guide

**Get GenOps governance for your Griptape AI applications in under 5 minutes.**

## 🧠 Key Concepts (2 minutes)

Before we start, understand these core concepts:

- **Auto-instrumentation**: Automatically adds cost/usage tracking to your existing Griptape code without any changes
- **Governance attributes**: Team/project tags that enable cost attribution and budget tracking across your organization  
- **OpenTelemetry**: Industry standard for exporting tracking data to monitoring tools like Datadog, Grafana, or Honeycomb

**What you get**: Your existing Griptape Agents, Pipelines, and Workflows automatically include cost tracking, team attribution, and governance telemetry.

## 🚀 5-Minute Setup

### 1. Install (30 seconds)
```bash
# Install GenOps
pip install genops

# Install Griptape (if not already installed)
pip install griptape
```

### 2. Set Environment Variables (30 seconds)
```bash
export OPENAI_API_KEY="your-openai-key"
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="griptape-demo"
```

### 3. Validate Setup (30 seconds)
```bash
python -c "from genops.providers.griptape.registration import validate_griptape_setup; print(validate_griptape_setup())"
```

**✅ Expected Output**: You should see `'griptape_available': True` and `'instrumentation_enabled': False` (we'll enable it next).

### 4. Copy-Paste Working Example (3 minutes)

Create `quickstart_demo.py`:

```python
#!/usr/bin/env python3
"""5-Minute Griptape + GenOps Demo"""

import os
from griptape.structures import Agent
from griptape.tasks import PromptTask
from griptape.rules import Rule

# Enable GenOps governance with one import
from genops.providers.griptape import auto_instrument

def main():
    print("🤖 GenOps + Griptape - 5 Minute Demo")
    print("=" * 50)
    
    # Enable GenOps governance (1 line!)
    adapter = auto_instrument(
        team=os.getenv('GENOPS_TEAM', 'your-team'),
        project=os.getenv('GENOPS_PROJECT', 'griptape-demo')
    )
    
    print("📊 Starting Griptape Agent with GenOps governance...")
    
    # Create Griptape Agent (your existing code unchanged!)
    agent = Agent(
        tasks=[
            PromptTask(
                prompt="Explain AI governance in one clear sentence."
            )
        ],
        rules=[
            Rule("Keep response concise and professional"),
            Rule("Focus on practical benefits")
        ]
    )
    
    # Run agent - now includes automatic governance tracking
    print("🚀 Executing Griptape Agent...")
    result = agent.run()
    
    print("\n📝 Agent Response:")
    print(result.output.value)
    
    # Check governance metrics
    daily_spending = adapter.get_daily_spending()
    budget_status = adapter.check_budget_compliance()
    
    print(f"\n🎯 GenOps Tracking Details:")
    print(f"  Daily Spending: ${daily_spending:.6f}")
    print(f"  Budget Status: {budget_status['status']}")
    print(f"  Team: {adapter.governance_attrs.team}")
    print(f"  Project: {adapter.governance_attrs.project}")
    
    print("\n🎉 Demo Complete!")
    print("\nWhat just happened:")
    print("1. ✅ GenOps auto-instrumentation enabled")
    print("2. ✅ Griptape Agent executed with governance")  
    print("3. ✅ Cost and usage automatically tracked")
    print("4. ✅ Team and project attribution added")
    print("5. ✅ OpenTelemetry telemetry generated")

if __name__ == "__main__":
    main()
```

### 5. Run the Demo (30 seconds)
```bash
python quickstart_demo.py
```

**✅ Verify Success**: After running, you should see cost tracking data and governance attributes in the output.

## ✅ Expected Output

```
🤖 GenOps + Griptape - 5 Minute Demo
==================================================
📊 Starting Griptape Agent with GenOps governance...
🚀 Executing Griptape Agent...

📝 Agent Response:
AI governance ensures responsible AI development through policies, monitoring, and ethical guidelines for safe deployment.

🎯 GenOps Tracking Details:
  Daily Spending: $0.000523
  Budget Status: within_budget
  Team: your-team
  Project: griptape-demo

🎉 Demo Complete!

What just happened:
1. ✅ GenOps auto-instrumentation enabled
2. ✅ Griptape Agent executed with governance
3. ✅ Cost and usage automatically tracked
4. ✅ Team and project attribution added
5. ✅ OpenTelemetry telemetry generated
```

## 🔧 Troubleshooting

### "Griptape not found"
```bash
# Install Griptape
pip install griptape
```

### "GenOps not installed"
```bash
pip install genops
```

### "OpenAI API key not found"
```bash
export OPENAI_API_KEY="your-actual-api-key"
# Get one from: https://platform.openai.com/api-keys
```

### "Validation failed"
```bash
# Run detailed validation
python -c "from genops.providers.griptape.registration import validate_griptape_setup; import pprint; pprint.pprint(validate_griptape_setup())"
```

### "Auto-instrumentation not working?"
```bash
# Check if instrumentation is enabled
python -c "from genops.providers.griptape.registration import is_instrumented; print(f'Instrumented: {is_instrumented()}')"

# If False, restart your Python application after calling auto_instrument()
```

### "Cost tracking showing $0.00?"
This usually means:
- API calls aren't completing successfully (check API key validity)  
- Using a local model (cost tracking works, but costs are $0)
- Network connectivity issues preventing API calls from completing

## 🚀 What's Next?

### Immediate Next Steps (5 minutes each):
1. **Try Auto-Instrumentation**: `python ../examples/griptape/02_auto_instrumentation.py`
2. **Explore Multi-Provider**: Add Anthropic or Google models with unified governance
3. **Set Up Observability**: Connect to your monitoring dashboard

### Learn More (30 minutes):
- **[Complete Integration Guide](integrations/griptape.md)** - All features and patterns
- **[Examples Suite](../examples/griptape/)** - Progressive examples with working code
- **[Production Deployment](integrations/griptape.md#production-deployment)** - Docker, Kubernetes patterns

### Production Ready (2 hours):
- **Multi-Provider Setup**: Add Anthropic, Google, Cohere providers
- **Enterprise Governance**: Budget controls, compliance monitoring
- **Dashboard Integration**: Grafana, Datadog, Honeycomb setup

## 💡 Key Benefits You Just Enabled

- ✅ **Zero Code Changes**: Existing Griptape code works unchanged
- ✅ **Automatic Cost Tracking**: Real-time cost attribution across providers
- ✅ **Team Attribution**: Per-team, per-project cost breakdown
- ✅ **OpenTelemetry Native**: Works with any observability platform
- ✅ **Multi-Structure Support**: Agents, Pipelines, Workflows unified governance
- ✅ **Production Ready**: Enterprise patterns and scaling support

## 🤝 Need Help?

- **Quick Questions**: Check the [troubleshooting section](#-troubleshooting) above
- **Documentation**: [Complete integration guide](integrations/griptape.md)  
- **Examples**: [Progressive examples suite](../examples/griptape/)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community**: [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**⏱️ Total Time**: Under 5 minutes | **Result**: Full GenOps governance for Griptape | **Next**: [Integration Guide](integrations/griptape.md)