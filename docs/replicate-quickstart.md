# Replicate Quickstart Guide

**⚡ 5-Minute Time-to-Value Guarantee**

Get GenOps cost tracking and governance working with Replicate in exactly 5 minutes or less. **This follows the GenOps Progressive Complexity Architecture**: immediate value first, then progressive mastery.

## 🔧 Prerequisites (2 minutes)

**Before starting, you need:**

1. **Replicate API Token**: Get your free token from [Replicate](https://replicate.com/account/api-tokens)
2. **Python Environment**: Python 3.8+ with pip installed

**⚠️ Cost Notice**: Replicate pricing varies by model - text models start at ~$0.50/1K tokens, images ~$0.003-0.04/image. Most examples cost under $0.01.

## ⚡ Zero-Code Setup (30 seconds)

```bash
# Install GenOps with Replicate support
pip install genops-ai[replicate]

# Set your API token
export REPLICATE_API_TOKEN="r8_your_token_here"
```

## 🎯 Immediate Value Demo (2 minutes)

**Copy-paste this working example:**

```python
from genops.providers.replicate import auto_instrument
import replicate

# Enable automatic instrumentation (zero code changes needed!)
auto_instrument()

# Your existing Replicate code works unchanged and is now tracked
output = replicate.run(
    "meta/llama-2-7b-chat",
    input={
        "prompt": "Hello from GenOps! Explain AI cost tracking in one sentence.",
        "max_length": 50
    }
)

print("✅ Success! Your Replicate calls now include GenOps cost tracking!")
print(f"🤖 Response: {output}")
```

## 🚀 Add Team Attribution (1 minute)

**Track costs by team, project, and customer:**

```python
from genops.providers.replicate import GenOpsReplicateAdapter

adapter = GenOpsReplicateAdapter()

# Text generation with governance
text_result = adapter.text_generation(
    model="meta/llama-2-7b-chat",
    prompt="Create a marketing tagline for AI cost management",
    max_tokens=30,
    # Governance attributes - automatic cost attribution!
    team="marketing-team",
    project="cost-campaign", 
    customer_id="internal-demo"
)

print(f"💬 Text: {text_result.content}")
print(f"💰 Cost: ${text_result.cost_usd:.6f}")

# Image generation with governance  
image_result = adapter.image_generation(
    model="black-forest-labs/flux-schnell",
    prompt="Simple icon representing cost optimization",
    num_images=1,
    team="design-team",
    project="cost-campaign",
    customer_id="internal-demo"
)

print(f"🎨 Images: Generated successfully")
print(f"💰 Cost: ${image_result.cost_usd:.6f}")
print(f"🏷️ Team: design-team, Project: cost-campaign")
```

## ✅ Validation (1 minute)

**Verify everything is working:**

```python
from genops.providers.replicate_validation import validate_setup, print_validation_result

# Comprehensive setup check with actionable fixes
result = validate_setup()

if result.success:
    print("🎉 GenOps Replicate setup is ready!")
    print("➡️ Your Replicate calls will now include cost tracking and governance")
else:
    print("❌ Setup issues found:")
    for error in result.errors:
        print(f"   - {error}")
    print("\n💡 For detailed diagnostics, run:")
    print("   python -c \"from genops.providers.replicate_validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")
```

## 🎯 What Just Happened?

- **✅ Zero-code auto-instrumentation** - Your existing Replicate calls are now automatically tracked
- **💰 Real-time cost tracking** - Every operation shows accurate cost across all model types
- **🏷️ Team attribution** - Costs automatically attributed to teams, projects, and customers
- **📊 OpenTelemetry export** - Data flows to your existing observability platform
- **🎯 Multi-modal optimization** - Built-in cost optimization across text, image, video, audio models

## 🚨 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| `ImportError: replicate` | Run `pip install replicate` |
| `API token` error | Set `export REPLICATE_API_TOKEN="r8_your_token_here"` and get token from https://replicate.com/account/api-tokens |
| `Model not found` error | Try `meta/llama-2-7b-chat` or browse https://replicate.com/explore |
| `Rate limit` exceeded | Wait a few minutes (free tier has rate limits) or try cheaper models |
| No telemetry data | **Optional**: Set `export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"` to send to local collector |

## 🚀 Progressive Learning Path (GenOps Developer Experience Standard)

**🎯 Phase 1: Immediate Value (≤ 5 minutes) - COMPLETE! ✅**
You've just completed the 5-minute quickstart. You now have working GenOps tracking.

**🎯 Phase 2: Multi-Modal Control & Attribution (≤ 30 minutes)**
Ready to add team cost tracking and multi-modal optimization? Continue here:
```bash
python examples/replicate/basic_tracking.py          # Team attribution across model types
python examples/replicate/auto_instrumentation.py   # Zero-code setup patterns
```
*Time estimate: 15-30 minutes*

**🎯 Phase 3: Production Mastery (≤ 2 hours)**
Ready for advanced cost optimization and production deployment?
```bash
python examples/replicate/cost_optimization.py      # Advanced cost intelligence
# More production examples in examples/replicate/README.md
```
*Time estimate: 1-2 hours*

**📚 Documentation by Experience Level:**
- **Phase 2 (30-min)**: [`examples/replicate/README.md`](../examples/replicate/) - Complete practical guide
- **Phase 3 (2-hr)**: [`docs/integrations/replicate.md`](../integrations/replicate.md) *(Coming Soon)* - Full reference

---

## 🎉 Success! You're Now Tracking AI Costs

**Your GenOps Replicate integration is complete.** Every AI operation is now:
- ✅ Automatically tracked with accurate costs across all model types
- ✅ Attributed to teams and projects for governance
- ✅ Exported to your observability platform
- ✅ Optimized with intelligent model recommendations

**Questions?** Join our [community discussions](https://github.com/KoshiHQ/GenOps-AI/discussions) or check the [examples directory](../examples/replicate/).