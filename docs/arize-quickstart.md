# Arize AI + GenOps Quick Start (5 minutes)

Get Arize AI model monitoring with GenOps governance running in under 5 minutes with zero code changes to your existing Arize workflows.

> 📖 **Navigation:** **Start Here** → [Complete Guide](integrations/arize.md) → [Examples](../examples/arize/)

⏱️ **Total time: 4-5 minutes** | 🎯 **Success rate: 95%+** | 🔧 **Zero code changes required**

## 🎯 You Are Here: 5-Minute Quickstart

**Perfect for:** First-time users who want immediate results with minimal setup

**What you'll get:** Working governance for your existing Arize AI monitoring with zero code changes

**Next steps:** After completing this guide, you'll be ready to explore [interactive examples](../examples/arize/) or dive into [advanced features](integrations/arize.md)

## Prerequisites ⏱️ 30 seconds

```bash
# Install dependencies
pip install genops[arize]

# ✅ Verify installation
python -c "import genops; print('✅ GenOps installed successfully!')"
```

**✅ Success check:** You should see "✅ GenOps installed successfully!" 

## Step 1: Get Your Arize Credentials ⏱️ 60 seconds

1. Open [Arize AI Dashboard](https://app.arize.com) in a new tab
2. Navigate to **Settings** → **API Keys** (top right menu)
3. Copy your **API Key** and **Space Key**

💡 **Pro tip:** Keep these tabs open - you'll paste the keys in the next step.

## Step 2: Set Environment Variables ⏱️ 45 seconds

```bash
# Required: Arize credentials
export ARIZE_API_KEY="your-arize-api-key-here"
export ARIZE_SPACE_KEY="your-arize-space-key-here"

# Recommended: Team attribution
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name"
```

**✅ Success check:**
```bash
echo "API Key: ${ARIZE_API_KEY:0:8}..." && echo "Space Key: ${ARIZE_SPACE_KEY:0:8}..."
```
You should see truncated versions of your keys.

## Step 3: Enable Auto-Instrumentation ⏱️ 30 seconds

Add **just 2 lines** to the top of your Python file:

```python
from genops.providers.arize import auto_instrument
auto_instrument()  # ✨ This enables governance for ALL Arize operations
```

**✅ Success check:**
```python
# Run this to confirm auto-instrumentation is active
from genops.providers.arize import get_current_adapter
print("✅ Auto-instrumentation active!" if get_current_adapter() else "❌ Setup issue detected")
```

## Step 4: Use Arize Normally ⏱️ 90 seconds

Your existing Arize code now automatically includes cost tracking and governance:

```python
from arize.pandas.logger import Client
import pandas as pd

# Your existing Arize code - no changes needed!
arize_client = Client(
    api_key=os.getenv("ARIZE_API_KEY"),
    space_key=os.getenv("ARIZE_SPACE_KEY")
)

# This prediction logging is now automatically tracked with GenOps governance
response = arize_client.log(
    prediction_id="pred-001",
    prediction_label="fraud",
    actual_label="fraud", 
    model_id="fraud-detection-model",
    model_version="v1.0",
    features={"amount": 150.50, "merchant": "online"},
    tags={"environment": "production"}
)

print(f"✅ Prediction logged! Status: {response.get('status', 'success')}")
```

**✅ Success check:** You should see "✅ Prediction logged! Status: success"

## Step 5: Verify Governance is Active ⏱️ 60 seconds

Run this validation script:

```python
from genops.providers.arize_validation import validate_setup, print_validation_result

# One-liner validation check
result = validate_setup()
print_validation_result(result)

# Quick cost check
from genops.providers.arize import get_current_adapter
adapter = get_current_adapter()
if adapter:
    metrics = adapter.get_metrics()
    print(f"\n💰 Cost tracking active: ${metrics['daily_usage']:.2f} used today")
```

**✅ Expected output:**
```
🔍 Arize AI Integration Validation Report
============================================================

✅ Overall Status: SUCCESS

📊 Validation Summary:
  • SDK Installation: 0 issues
  • Authentication: 0 issues  
  • Configuration: 0 issues

💡 Recommendations:
  1. All validation checks passed successfully!

🚀 Next Steps:
  1. You can now use GenOps Arize integration with confidence

💰 Cost tracking active: $0.00 used today
```

🎉 **Congratulations!** If you see this output, your integration is working perfectly.

## What You Get Automatically

- 📊 **Cost Tracking**: Every Arize operation is tracked with costs
- 🏷️ **Team Attribution**: All operations tagged with your team/project
- 💰 **Budget Monitoring**: Automatic budget alerts and limits
- 📈 **Usage Analytics**: Detailed breakdowns of monitoring costs
- 🔍 **Governance Telemetry**: OpenTelemetry spans for all operations

## Quick Cost Check

```python
from genops.providers.arize import get_current_adapter

# Get cost metrics anytime
adapter = get_current_adapter()
if adapter:
    metrics = adapter.get_metrics()
    print(f"Today's usage: ${metrics['daily_usage']:.2f}")
    print(f"Budget remaining: ${metrics['budget_remaining']:.2f}")
    print(f"Operations tracked: {metrics['operation_count']}")
```

## Instant Troubleshooting 🔧

### ❌ "ModuleNotFoundError: No module named 'arize'"
```bash
pip install arize>=6.0.0
# ✅ Test: python -c "import arize; print('Arize installed!')"
```

### ❌ "Missing Arize API Key" or Authentication Failed
```bash
# Get keys from: https://app.arize.com → Settings → API Keys
export ARIZE_API_KEY="your-actual-api-key-here"
export ARIZE_SPACE_KEY="your-actual-space-key-here"
# ✅ Test: echo "Keys set: ${ARIZE_API_KEY:0:8}...${ARIZE_SPACE_KEY:0:8}"
```

### ❌ "Budget limit exceeded" warnings
```python
# Quick fix: Increase budget temporarily
auto_instrument(daily_budget_limit=100.0)
# Or disable cost alerts: auto_instrument(enable_cost_alerts=False)
```

### ❌ Import errors or auto-instrumentation not working
```python
# Clear and restart:
from genops.providers.arize import set_global_adapter
set_global_adapter(None)
from genops.providers.arize import auto_instrument
auto_instrument()  # Fresh start
```

### 🆘 Still stuck?
```bash
# Run comprehensive diagnostics:
python -c "
from genops.providers.arize_validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, show_details=True)
"
```

## One-Liner Commands 🚀

```bash
# Quick validation check
python -c "from genops.providers.arize_validation import validate_setup, print_validation_result; print_validation_result(validate_setup())"

# Cost usage summary
python -c "from genops.providers.arize import get_current_adapter; a=get_current_adapter(); print(f'Daily usage: ${a.get_metrics()[\"daily_usage\"]:.2f}') if a else print('Auto-instrument not active')"

# Reset and restart auto-instrumentation
python -c "from genops.providers.arize import set_global_adapter, auto_instrument; set_global_adapter(None); auto_instrument(); print('✅ Auto-instrumentation restarted')"

# Run example (after downloading)
python -c "import subprocess; subprocess.run(['python', 'examples/arize/basic_tracking.py'])"
```

## What's Next? Choose Your Learning Path 🗺️

### 🏃‍♂️ **I want to keep exploring (5-10 min)**
*Perfect if you learn by doing and want hands-on examples*
- 📋 **Start here:** [Interactive Examples](../examples/arize/) - Copy-paste ready code that works immediately
- 🎯 **Try first:** `basic_tracking.py` - See governance in action with sample data
- 💰 **Then explore:** `cost_optimization.py` - Learn cost intelligence features
- 🔧 **Advanced:** `advanced_features.py` - Multi-model monitoring patterns

### 📚 **I need to understand the details (15-30 min)**
*Perfect if you're planning integration or need comprehensive docs*
- 📖 **Start here:** [Complete Integration Guide](integrations/arize.md) - Comprehensive documentation
- 🏗️ **Focus on:** [Manual Adapter Usage](integrations/arize.md#manual-adapter-usage) - Full control and configuration
- 📊 **Then:** [Cost Intelligence](cost-intelligence-guide.md) - ROI analysis and budget planning
- 🔧 **Advanced:** [Production Monitoring](integrations/arize.md#production-monitoring--alerting) - Dashboards and alerting

### 🚀 **I'm ready for production deployment (30+ min)**
*Perfect if you're implementing enterprise-grade monitoring*
- 🎯 **Start here:** [Enterprise Governance Templates](enterprise-governance-templates.md) - SOX, GDPR, HIPAA compliance
- 🏗️ **Then:** [Production Deployment Patterns](integrations/arize.md#enterprise-deployment-patterns) - HA, scaling, security
- 📈 **Set up:** [Cost Monitoring Dashboards](integrations/arize.md#dashboard-integration-patterns) - Grafana, DataDog integration
- 🔐 **Secure:** [Multi-Environment Governance](integrations/arize.md#environment-specific-governance) - Dev, staging, prod policies

### 💼 **I need to justify the business case**
*Perfect for presenting to stakeholders or budget planning*
- 💰 **Start here:** [Cost Intelligence & ROI Guide](cost-intelligence-guide.md) - Calculate ROI and savings
- 📊 **Use:** ROI calculator templates for your specific use case
- 📈 **Show:** Cost optimization opportunities and budget forecasting
- 📋 **Present:** Business value and compliance benefits

---

🎉 **Success!** You now have enterprise-grade governance for your Arize AI model monitoring with **zero changes** to your existing code. Every ML operation is automatically tracked, attributed, and governed.

**Need help?** 
- 🔍 Check our [troubleshooting guide](integrations/arize.md#validation-and-troubleshooting)
- 💬 Join [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions) 
- 🐛 [Report issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- 📧 Enterprise support: [contact us](mailto:support@genops.ai)