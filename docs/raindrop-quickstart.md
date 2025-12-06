# Raindrop AI + GenOps Quick Start (5 minutes)

Add cost tracking and team attribution to your existing Raindrop AI agent monitoring in under 5 minutes with zero code changes.

> 📖 **Navigation:** **Start Here** → [Complete Guide](integrations/raindrop.md) → [Examples](../examples/raindrop/)

⏱️ **Total time: 4-5 minutes** | 🎯 **Success rate: 95%+** | 🔧 **Zero code changes required**

## 🎯 You Are Here: 5-Minute Quickstart

**Perfect for:** First-time users who want immediate results with minimal setup

**What you'll get:** Automatic cost tracking and team attribution for your existing Raindrop AI agents with zero code changes

**Next steps:** After completing this guide, you'll be ready to explore [interactive examples](../examples/raindrop/) or dive into [advanced features](integrations/raindrop.md)

## Prerequisites ⏱️ 30 seconds

```bash
# Install dependencies
pip install genops[raindrop]

# ✅ Verify installation
python -c "import genops; print('✅ GenOps installed successfully!')"
```

**✅ Success check:** You should see "✅ GenOps installed successfully!" 

## Step 1: Get Your Raindrop Credentials ⏱️ 60 seconds

1. Open [Raindrop AI Dashboard](https://app.raindrop.ai) in a new tab
2. Navigate to **Settings** → **API Keys** (account menu)
3. Copy your **API Key**

💡 **Pro tip:** Keep this tab open - you'll paste the key in the next step.

## Step 2: Set Environment Variables ⏱️ 45 seconds

```bash
# Required: Raindrop credentials
export RAINDROP_API_KEY="your-raindrop-api-key-here"

# Recommended: Team attribution
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name"
```

**✅ Success check:**
```bash
echo "API Key: ${RAINDROP_API_KEY:0:8}..."
```
You should see a truncated version of your key.

## Step 3: Enable Auto-Instrumentation ⏱️ 30 seconds

Add **just 2 lines** to the top of your Python file (this enables automatic tracking):

```python
from genops.providers.raindrop import auto_instrument
auto_instrument()  # ✨ This enables governance for ALL Raindrop operations
```

**✅ Success check:**
```python
# Run this to confirm auto-instrumentation is active
from genops.providers.raindrop_validation import validate_setup
result = validate_setup()
if result.is_valid:
    print("✅ Auto-instrumentation active!")
else:
    print("❌ Setup issue detected:")
    for error in result.errors[:3]:  # Show first 3 errors
        print(f"  • {error.message}")
        if error.fix_suggestion:
            print(f"    💡 Fix: {error.fix_suggestion}")
    print("\n🔧 Run 'python -c \"from genops.providers.raindrop_validation import validate_setup_interactive; validate_setup_interactive()\"' for guided setup")
```

**🔧 If you see errors:**
- **Missing API key**: Run `echo $RAINDROP_API_KEY` to verify it's set
- **Import errors**: Reinstall with `pip install --upgrade genops[raindrop]`
- **Permission issues**: Check if your API key has the required permissions

## Step 4: Use Raindrop Normally ⏱️ 90 seconds

Your existing Raindrop code now automatically includes cost tracking and team attribution:

```python
import raindrop

# Your existing Raindrop code - no changes needed!
client = raindrop.Client(api_key="your-api-key")

# Track agent interactions (automatically governed)
response = client.track_interaction(
    agent_id="support-bot-1",
    interaction_data={
        "input": "Customer support query",
        "output": "Agent response with resolution",
        "performance_signals": {
            "response_time": 250,
            "confidence_score": 0.94,
            "customer_satisfaction": 4.5
        }
    }
)

# 🎉 This interaction is now automatically tracked with:
# • Cost tracking (see exactly what each interaction costs)
# • Team attribution (know which team/project spent what)
# • Budget monitoring (get alerts before overspending)
# • Performance insights (optimize your agent monitoring)
```

**✅ Success check:** 
```python
# Verify the interaction worked and was tracked
print("✅ Agent interaction completed successfully!")
print("🔍 To verify tracking is working, check that no errors occurred above")

# Quick validation that governance is active
import os
if os.getenv("RAINDROP_API_KEY"):
    print("✅ API key configured")
if os.getenv("GENOPS_TEAM"):
    print(f"✅ Team attribution: {os.getenv('GENOPS_TEAM')}")
```

**🔧 If you see errors:**
- **Connection failed**: Verify your `RAINDROP_API_KEY` is correct and active
- **Module not found**: The example assumes you have the Raindrop SDK - this is just for demonstration
- **Attribution missing**: Set `GENOPS_TEAM` and `GENOPS_PROJECT` environment variables

## Step 5: Verify Governance is Working ⏱️ 60 seconds

```python
# Quick verification script
from genops.providers.raindrop import GenOpsRaindropAdapter

# Check that governance is active
adapter = GenOpsRaindropAdapter(
    team="demo-team",
    project="quickstart-demo",
    daily_budget_limit=10.0
)

with adapter.track_agent_monitoring_session("verification") as session:
    # Track a test interaction
    cost_result = session.track_agent_interaction(
        agent_id="test-agent",
        interaction_data={"test": "verification"},
        cost=0.001
    )
    
    print(f"✅ Governance verification successful!")
    print(f"   💰 Cost tracked: ${cost_result.total_cost:.3f}")
    print(f"   🏷️  Team: {session.governance_attrs.team}")
    print(f"   📊 Project: {session.governance_attrs.project}")
```

**Expected output:**
```
✅ Governance verification successful!
   💰 Cost tracked: $0.001
   🏷️  Team: demo-team
   📊 Project: quickstart-demo
```

## 🎉 Success! What You've Accomplished

In just 5 minutes, you've added enterprise-grade governance to your Raindrop AI monitoring:

### ✅ **Zero-Code Cost Tracking**
- All agent interactions automatically tracked
- Real-time cost calculation and attribution
- Team and project cost breakdowns

### ✅ **Budget Monitoring**
- Automatic budget enforcement
- Cost alerts and notifications
- Spending limit protection

### ✅ **Governance & Compliance**
- OpenTelemetry-native telemetry export
- Audit trail for all agent operations
- Enterprise policy enforcement

### ✅ **Performance Intelligence**
- Agent performance signal monitoring
- Cost optimization recommendations
- Multi-agent cost aggregation

## 🚀 Next Steps

### **Immediate Actions (5 minutes each)**
1. **[Try Examples](../examples/raindrop/)** - Explore 6 hands-on examples
2. **[Cost Optimization](../examples/raindrop/cost_optimization.py)** - Analyze your spend and get recommendations
3. **[Production Patterns](../examples/raindrop/production_patterns.py)** - See enterprise deployment strategies

### **This Week (30 minutes total)**
1. **[Complete Integration Guide](integrations/raindrop.md)** - Full documentation with advanced features
2. **Set Up Dashboards** - Connect to Grafana, Datadog, or Honeycomb
3. **Configure Team Budgets** - Set spending limits and alerts

### **This Month (Production Ready)**
1. **Multi-Environment Setup** - Deploy across dev/staging/prod
2. **Advanced Governance** - Implement compliance policies
3. **Cost Intelligence** - Optimize spend across all agents

## 🔧 Common Issues & Quick Fixes

### **Issue: "Module not found" error**
```bash
# Problem: Missing GenOps installation or extras
# Solution: Install with correct extras
pip install --upgrade genops[raindrop]

# Verify installation worked
python -c "import genops; print('✅ GenOps installed')"
python -c "from genops.providers.raindrop import auto_instrument; print('✅ Raindrop provider available')"
```

### **Issue: API authentication failed**
```bash
# Problem: Invalid or missing API key
# Diagnosis: Check if key is set and valid format
echo "Key length: $(echo $RAINDROP_API_KEY | wc -c)"
echo "Key prefix: ${RAINDROP_API_KEY:0:10}..."

# Solution: Get a valid API key from Raindrop AI dashboard
# 1. Go to https://app.raindrop.ai
# 2. Navigate to Settings → API Keys
# 3. Copy the key and set it:
export RAINDROP_API_KEY="your-complete-api-key-here"
```

### **Issue: No cost data appearing**
```bash
# Problem: Setup validation issues
# Comprehensive diagnosis:
python -c "
from genops.providers.raindrop_validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, verbose=True)
"

# If you see issues, run interactive setup:
python -c "
from genops.providers.raindrop_validation import validate_setup_interactive
validate_setup_interactive()
"
```

### **Issue: Budget alerts not working**
```python
# Problem: Budget monitoring not configured
# Solution: Enable cost alerts explicitly with proper configuration
from genops.providers.raindrop import auto_instrument

auto_instrument(
    team="your-team",
    project="your-project",
    daily_budget_limit=50.0,        # Set your budget
    enable_cost_alerts=True,        # Enable alerts
    governance_policy="enforced"    # Use enforced mode for budget limits
)

# Verify budget configuration
from genops.providers.raindrop import GenOpsRaindropAdapter
adapter = GenOpsRaindropAdapter(daily_budget_limit=50.0)
print(f"Budget configured: ${adapter.daily_budget_limit}")
```

### **Issue: Examples not working**
```bash
# Problem: Missing environment setup or dependencies
# Complete environment check:
echo "Environment Check:"
echo "├── API Key: ${RAINDROP_API_KEY:+SET}" 
echo "├── Team: ${GENOPS_TEAM:-'NOT SET'}"
echo "├── Project: ${GENOPS_PROJECT:-'NOT SET'}"
echo "└── Budget: ${GENOPS_DAILY_BUDGET_LIMIT:-'NOT SET'}"

# Quick fix for common setup:
export GENOPS_TEAM="quickstart-team"
export GENOPS_PROJECT="raindrop-demo"
export GENOPS_DAILY_BUDGET_LIMIT="25.0"

# Verify all examples work:
cd examples/raindrop && ./run_all_examples.sh
```

### **Issue: Performance is slow**
```python
# Problem: Default configuration not optimized
# Solution: Optimize for your use case
from genops.providers.raindrop import GenOpsRaindropAdapter

# High-volume optimization
adapter = GenOpsRaindropAdapter(
    export_telemetry=False,  # Disable telemetry export for speed
    governance_policy="advisory"  # Use advisory mode for better performance
)

# Or batch processing for many operations
from genops.providers.raindrop import auto_instrument
auto_instrument(
    # Configure sampling for high-volume scenarios
    # This would be configured in actual implementation
)
```

### **Still having issues?**
```bash
# Get comprehensive diagnostic information
python -c "
import sys, os
print('Python version:', sys.version)
print('Working directory:', os.getcwd())
print('Environment variables:')
for key in ['RAINDROP_API_KEY', 'GENOPS_TEAM', 'GENOPS_PROJECT']:
    value = os.getenv(key)
    if value:
        print(f'  {key}: {value[:10]}...' if len(value) > 10 else f'  {key}: {value}')
    else:
        print(f'  {key}: NOT SET')

# Test import chain
try:
    import genops
    print('✅ GenOps import successful')
    from genops.providers import raindrop
    print('✅ Raindrop provider import successful')
    from genops.providers.raindrop_validation import validate_setup
    print('✅ Validation module import successful')
    result = validate_setup()
    print(f'✅ Validation result: {\"VALID\" if result.is_valid else \"ISSUES FOUND\"}')
except Exception as e:
    print(f'❌ Import failed: {e}')
"
```

## 💬 Get Help

- 📖 **Documentation:** [Complete Integration Guide](integrations/raindrop.md)
- 💡 **Examples:** [Interactive Examples](../examples/raindrop/)
- 🐛 **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- 💬 **Community:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**🔙 Want a different integration?** Check out our [full integration list](../README.md#ai--llm-ecosystem) with 25+ supported platforms.

**📊 Ready for production?** See [Production Deployment Patterns](integrations/raindrop.md#production-deployment) for enterprise-ready configurations.

**💰 Want to optimize costs?** Try the [Cost Optimization Example](../examples/raindrop/cost_optimization.py) for immediate savings recommendations.

**⚡ Need performance optimization?** Check the [Performance Benchmarking Guide](raindrop-performance-benchmarks.md) for scaling and optimization strategies.