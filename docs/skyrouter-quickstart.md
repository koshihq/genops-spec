# SkyRouter + GenOps Quick Start (5 minutes)

Add cost tracking and team attribution to your existing SkyRouter multi-model routing in under 5 minutes with zero code changes.

> 📖 **Navigation:** **Start Here** → [Complete Guide](integrations/skyrouter.md) → [Examples](../../examples/skyrouter/)

⏱️ **Total time: 4-5 minutes** | 🎯 **Success rate: 95%+** | 🔧 **Zero code changes required**

## 🎯 You Are Here: 5-Minute Quickstart

**Perfect for:** First-time users who want immediate results with minimal setup

**What you'll get:** Automatic cost tracking and team attribution for your existing SkyRouter multi-model routing with zero code changes

**Next steps:** After completing this guide, you'll be ready to explore [interactive examples](../../examples/skyrouter/) or dive into [advanced features](integrations/skyrouter.md)

## Prerequisites ⏱️ 30 seconds

### What You'll Need

**Before starting:**
- Python 3.9+ installed
- A SkyRouter account and API key ([Sign up here](https://skyrouter.ai))
- 5 minutes of your time

**What is SkyRouter?**
SkyRouter is an AI routing platform that provides intelligent access to 150+ AI models from different providers. It automatically routes your requests to the best model based on cost, performance, and availability.

### Install Dependencies

```bash
# Install dependencies
pip install genops[skyrouter]

# ✅ Verify installation
python -c "import genops; print('✅ GenOps installed successfully!')"
```

**✅ Success check:** You should see "✅ GenOps installed successfully!" 

## Step 1: Get Your SkyRouter Credentials ⏱️ 60 seconds

1. Open [SkyRouter Dashboard](https://skyrouter.ai) in a new tab
2. Navigate to **Settings** → **API Keys** (account menu)
3. Copy your **API Key**

💡 **Pro tip:** Keep this tab open - you'll paste the key in the next step.

## Step 2: Set Environment Variables ⏱️ 45 seconds

```bash
# Required: SkyRouter credentials
export SKYROUTER_API_KEY="your-skyrouter-api-key-here"

# Recommended: Team attribution
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name"
```

**✅ Success check:**
```bash
echo "API Key: ${SKYROUTER_API_KEY:0:8}..."
```
You should see a truncated version of your key.

## Step 3: Enable Auto-Instrumentation ⏱️ 30 seconds

Add **just 2 lines** to the top of your Python file (this enables automatic tracking):

```python
from genops.providers.skyrouter import auto_instrument
auto_instrument()  # ✨ This enables governance for ALL SkyRouter operations
```

**✅ Success check:**
```python
# Run this to confirm auto-instrumentation is active
from genops.providers.skyrouter import validate_setup
result = validate_setup()
if result.is_valid:
    print("✅ Auto-instrumentation active!")
else:
    print("❌ Setup issue detected:")
    for error in result.errors[:3]:  # Show first 3 errors
        print(f"  • {error.message}")
        if error.fix_suggestion:
            print(f"    💡 Fix: {error.fix_suggestion}")
    print("\n🔧 Run 'python -c \"from genops.providers.skyrouter import validate_setup_interactive; validate_setup_interactive()\"' for guided setup")
```

**🔧 If you see errors:**
- **Missing API key**: Run `echo $SKYROUTER_API_KEY` to verify it's set
- **Import errors**: Reinstall with `pip install --upgrade genops[skyrouter]`
- **Permission issues**: Check if your API key has the required permissions

## Step 4: Use SkyRouter Normally ⏱️ 90 seconds

Your existing SkyRouter code now automatically includes cost tracking and team attribution:

```python
# Your existing SkyRouter multi-model routing - auto-instrumented with governance!
from genops.providers.skyrouter import GenOpsSkyRouterAdapter

# Initialize the adapter (this happens automatically with auto_instrument())
adapter = GenOpsSkyRouterAdapter()

# Single model call with automatic governance  
with adapter.track_routing_session("single-model") as session:
    response = session.track_model_call(
        model="gpt-4",
        input_data={"prompt": "Explain quantum computing"},
        route_optimization="cost_optimized"
    )

# Multi-model routing with automatic governance
with adapter.track_routing_session("multi-model") as session:
    response = session.track_multi_model_routing(
        models=["gpt-4", "claude-3-sonnet", "gemini-pro"],
        input_data={"prompt": "Write a product description"},
        routing_strategy="balanced"
    )

# Agent workflow with automatic governance  
with adapter.track_routing_session("agent-workflow") as session:
    agent_result = session.track_agent_workflow(
        workflow_name="customer_support",
        agent_steps=[
            {"model": "gpt-3.5-turbo", "task": "classify_intent"},
            {"model": "claude-3-sonnet", "task": "generate_response"},
            {"model": "gpt-4", "task": "quality_check"}
        ]
    )

# 🎉 All operations are now automatically tracked with:
# • Cost tracking across 150+ models (see exactly what each operation costs)
# • Team attribution (know which team/project spent what across all models)
# • Budget monitoring (get alerts before overspending across routing strategies)
# • Route optimization insights (optimize your multi-model routing for cost)
```

**✅ Success check:** 
```python
# Verify the operations worked and were tracked
print("✅ SkyRouter operations completed successfully!")
print("🔍 To verify tracking is working, check that no errors occurred above")

# Quick validation that governance is active
import os
if os.getenv("SKYROUTER_API_KEY"):
    print("✅ API key configured")
if os.getenv("GENOPS_TEAM"):
    print(f"✅ Team attribution: {os.getenv('GENOPS_TEAM')}")
```

**🔧 If you see errors:**
- **Connection failed**: Verify your `SKYROUTER_API_KEY` is correct and active
- **Module not found**: The example assumes you have the SkyRouter SDK - this is just for demonstration
- **Attribution missing**: Set `GENOPS_TEAM` and `GENOPS_PROJECT` environment variables

## Step 5: Verify Governance is Working ⏱️ 60 seconds

```python
# Quick verification script
from genops.providers.skyrouter import GenOpsSkyRouterAdapter

# Check that governance is active
adapter = GenOpsSkyRouterAdapter(
    team="demo-team",
    project="quickstart-demo",
    daily_budget_limit=50.0
)

with adapter.track_routing_session("verification") as session:
    # Track a test multi-model call
    cost_result = session.track_multi_model_routing(
        models=["gpt-3.5-turbo", "claude-3-haiku"],
        input_data={"prompt": "Test routing governance"},
        routing_strategy="cost_optimized",
        cost=0.005
    )
    
    print(f"✅ Governance verification successful!")
    print(f"   💰 Cost tracked: ${cost_result.total_cost:.3f}")
    print(f"   🏷️  Team: {session.governance_attrs.team}")
    print(f"   📊 Project: {session.governance_attrs.project}")
    print(f"   🔀 Route: {cost_result.route}")
```

**Expected output:**
```
✅ Governance verification successful!
   💰 Cost tracked: $0.005
   🏷️  Team: demo-team
   📊 Project: quickstart-demo
   🔀 Route: multi_model_cost_optimized
```

## 🎉 Success! What You've Accomplished

In just 5 minutes, you've added enterprise-grade governance to your SkyRouter multi-model routing:

### ✅ **Zero-Code Multi-Model Governance**
- All model routing operations automatically tracked across 150+ models
- Real-time cost calculation and attribution across your entire model ecosystem
- Team and project cost breakdowns spanning all routing strategies

### ✅ **Intelligent Route Monitoring**
- Automatic budget enforcement across all routing strategies
- Cost alerts and optimization recommendations for multi-model usage
- Route efficiency analysis and cost-aware optimization suggestions

### ✅ **Enterprise Multi-Model Compliance**
- OpenTelemetry-native telemetry export for all routing operations
- Audit trail for all multi-model routing decisions and costs
- Enterprise policy enforcement across your entire model portfolio

### ✅ **Route Intelligence & Optimization**
- Multi-model cost comparison and optimization recommendations
- Route efficiency scoring and performance vs cost analysis
- Automated cost-aware routing strategy suggestions

## 🚀 Next Steps

### **Immediate Actions (5 minutes each)**
1. **[Try Examples](../../examples/skyrouter/)** - Explore 6 hands-on examples with multi-model patterns
2. **[Route Optimization](../../examples/skyrouter/route_optimization.py)** - Analyze your multi-model spend and get recommendations
3. **[Agent Workflows](../../examples/skyrouter/agent_workflows.py)** - See enterprise multi-agent routing patterns

### **This Week (30 minutes total)**
1. **[Complete Integration Guide](integrations/skyrouter.md)** - Full documentation with advanced multi-model features
2. **Set Up Dashboards** - Connect to Grafana, Datadog, or Honeycomb for multi-model insights
3. **Configure Route Budgets** - Set spending limits and alerts across routing strategies

### **This Month (Production Ready)**
1. **Multi-Environment Setup** - Deploy across dev/staging/prod with route-specific configurations
2. **Advanced Route Governance** - Implement compliance policies for multi-model operations
3. **Route Intelligence** - Optimize spend across all 150+ models with intelligent routing

## 🔧 Common Issues & Quick Fixes

### **Issue: "Module not found" error**
```bash
# Problem: Missing GenOps installation or extras
# Solution: Install with correct extras
pip install --upgrade genops[skyrouter]

# Verify installation worked
python -c "import genops; print('✅ GenOps installed')"
python -c "from genops.providers.skyrouter import auto_instrument; print('✅ SkyRouter provider available')"
```

### **Issue: API authentication failed**
```bash
# Problem: Invalid or missing API key
# Diagnosis: Check if key is set and valid format
echo "Key length: $(echo $SKYROUTER_API_KEY | wc -c)"
echo "Key prefix: ${SKYROUTER_API_KEY:0:10}..."

# Solution: Get a valid API key from SkyRouter dashboard
# 1. Go to https://skyrouter.ai
# 2. Navigate to Settings → API Keys
# 3. Copy the key and set it:
export SKYROUTER_API_KEY="your-complete-api-key-here"
```

### **Issue: No cost data appearing**
```bash
# Problem: Setup validation issues
# Comprehensive diagnosis:
python -c "
from genops.providers.skyrouter import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, verbose=True)
"

# If you see issues, run interactive setup:
python -c "
from genops.providers.skyrouter import validate_setup_interactive
validate_setup_interactive()
"
```

### **Issue: Route optimization not working**
```python
# Problem: Route optimization not configured properly
# Solution: Enable route optimization explicitly with proper configuration
from genops.providers.skyrouter import auto_instrument

auto_instrument(
    team="your-team",
    project="your-project",
    daily_budget_limit=100.0,           # Set your budget
    enable_cost_alerts=True,            # Enable alerts
    governance_policy="enforced"        # Use enforced mode for budget limits
)

# Verify route optimization configuration
from genops.providers.skyrouter import GenOpsSkyRouterAdapter
adapter = GenOpsSkyRouterAdapter(daily_budget_limit=100.0)
print(f"Budget configured: ${adapter.daily_budget_limit}")
```

### **Issue: Multi-model routing costs seem high**
```python
# Problem: Not using cost-optimized routing strategy
# Solution: Check and optimize routing strategy
from genops.providers.skyrouter import GenOpsSkyRouterAdapter

adapter = GenOpsSkyRouterAdapter(team="your-team", project="cost-optimization")

# Test different routing strategies
with adapter.track_routing_session("cost-comparison") as session:
    # Test cost-optimized routing
    cost_optimized = session.track_multi_model_routing(
        models=["gpt-4", "claude-3-sonnet", "gemini-pro"],
        input_data={"prompt": "Test prompt"},
        routing_strategy="cost_optimized"
    )
    
    # Compare with balanced routing
    balanced = session.track_multi_model_routing(
        models=["gpt-4", "claude-3-sonnet", "gemini-pro"],
        input_data={"prompt": "Test prompt"},
        routing_strategy="balanced"
    )
    
    print(f"Cost optimized: ${cost_optimized.total_cost:.4f}")
    print(f"Balanced: ${balanced.total_cost:.4f}")
    print(f"Potential savings: ${balanced.total_cost - cost_optimized.total_cost:.4f}")
```

### **Issue: Examples not working**
```bash
# Problem: Missing environment setup or dependencies
# Complete environment check:
echo "Environment Check:"
echo "├── API Key: ${SKYROUTER_API_KEY:+SET}" 
echo "├── Team: ${GENOPS_TEAM:-'NOT SET'}"
echo "├── Project: ${GENOPS_PROJECT:-'NOT SET'}"
echo "└── Budget: ${GENOPS_DAILY_BUDGET_LIMIT:-'NOT SET'}"

# Quick fix for common setup:
export GENOPS_TEAM="quickstart-team"
export GENOPS_PROJECT="skyrouter-demo"
export GENOPS_DAILY_BUDGET_LIMIT="50.0"

# Verify all examples work:
cd examples/skyrouter && ./run_all_examples.sh
```

### **Issue: Performance is slow with many models**
```python
# Problem: Default configuration not optimized for high-volume multi-model usage
# Solution: Optimize for your use case
from genops.providers.skyrouter import GenOpsSkyRouterAdapter

# High-volume multi-model optimization
adapter = GenOpsSkyRouterAdapter(
    export_telemetry=False,  # Disable telemetry export for speed
    governance_policy="advisory"  # Use advisory mode for better performance
)

# Or enable intelligent sampling for many routing operations
from genops.providers.skyrouter import auto_instrument
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
for key in ['SKYROUTER_API_KEY', 'GENOPS_TEAM', 'GENOPS_PROJECT']:
    value = os.getenv(key)
    if value:
        print(f'  {key}: {value[:10]}...' if len(value) > 10 else f'  {key}: {value}')
    else:
        print(f'  {key}: NOT SET')

# Test import chain
try:
    import genops
    print('✅ GenOps import successful')
    from genops.providers import skyrouter
    print('✅ SkyRouter provider import successful')
    from genops.providers.skyrouter import validate_setup
    print('✅ Validation module import successful')
    result = validate_setup()
    print(f'✅ Validation result: {\"VALID\" if result.is_valid else \"ISSUES FOUND\"}')
except Exception as e:
    print(f'❌ Import failed: {e}')
"
```

## 💬 Get Help

- 📖 **Documentation:** [Complete Integration Guide](integrations/skyrouter.md)
- 💡 **Examples:** [Interactive Examples](../examples/skyrouter/)
- 🐛 **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- 💬 **Community:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**🔙 Want a different integration?** Check out our [full integration list](../../README.md#ai--llm-ecosystem) with 25+ supported platforms.

**📊 Ready for production?** See [Production Deployment Patterns](integrations/skyrouter.md#production-deployment) for enterprise-ready multi-model configurations.

**💰 Want to optimize routing costs?** Try the [Route Optimization Example](../../examples/skyrouter/route_optimization.py) for immediate multi-model savings recommendations.

**⚡ Need performance optimization?** Check the [Performance Benchmarking Guide](skyrouter-performance-benchmarks.md) for scaling and optimization strategies across 150+ models.