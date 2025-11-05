# AWS Bedrock Quickstart Guide

**Time to Value: 5 minutes** ⚡

Get GenOps cost tracking and governance working with AWS Bedrock in under 5 minutes.

## 🔧 Prerequisites (2 minutes)

**Before starting, you need:**

1. **AWS Account**: With Bedrock access enabled ([AWS Console](https://console.aws.amazon.com/bedrock))
2. **AWS Credentials**: Configured via `aws configure` or environment variables
3. **Model Access**: Enable Claude 3 Haiku in [Bedrock Model Access](https://console.aws.amazon.com/bedrock/home#/model-access)

**⚠️ Cost Notice**: Bedrock charges per API call (~$0.00025 per 1k tokens for Claude Haiku)

## ⚡ Zero-Code Setup (30 seconds)

```bash
# Install GenOps with Bedrock support
pip install genops-ai[bedrock]

# Verify your AWS credentials are configured
aws sts get-caller-identity
```

## 🎯 Immediate Value Demo (2 minutes)

**Copy-paste this working example:**

```python
from genops.providers.bedrock import auto_instrument_bedrock
import boto3
import json

# Enable automatic instrumentation (zero code changes needed!)
auto_instrument_bedrock()

# Your existing Bedrock code works unchanged and is now tracked
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Properly formatted request for Claude
body = json.dumps({
    "messages": [{"role": "user", "content": "Hello from GenOps!"}],
    "max_tokens": 50,
    "anthropic_version": "bedrock-2023-05-31"
})

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    body=body,
    contentType='application/json'
)

print("✅ Success! Your Bedrock calls now include GenOps cost tracking!")
```

## 🚀 Add Team Attribution (1 minute)

**Track costs by team, project, and customer:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter(region_name='us-east-1')

result = adapter.text_generation(
    prompt="Analyze this quarterly report...",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    # Governance attributes - automatic cost attribution!
    team="ai-platform",
    project="document-analysis", 
    customer_id="enterprise-client-123"
)

print(f"💰 Cost: ${result.cost_usd:.6f}")
print(f"⚡ Latency: {result.latency_ms}ms")
print(f"🏷️ Team: ai-platform, Project: document-analysis")
```

## ✅ Validation (1 minute)

**Verify everything is working:**

```python
from genops.providers.bedrock import validate_setup, print_validation_result

# Comprehensive setup check with actionable fixes
result = validate_setup()

if result.success:
    print("🎉 GenOps Bedrock setup is ready!")
    print("➡️ Your Bedrock calls will now include cost tracking and governance")
else:
    print("❌ Setup issues found:")
    for error in result.errors:
        print(f"   - {error}")
    print("\n💡 For detailed diagnostics, run:")
    print("   python -c \"from genops.providers.bedrock import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")
```

## 🎯 What Just Happened?

- **✅ Zero-code auto-instrumentation** - Your existing Bedrock calls are now automatically tracked
- **💰 Real-time cost tracking** - Every operation shows accurate cost with token-level precision
- **🏷️ Team attribution** - Costs automatically attributed to teams, projects, and customers
- **📊 OpenTelemetry export** - Data flows to your existing observability platform
- **🛡️ AWS compliance** - Built-in CloudTrail integration and SOC2 compliance support

## 🚨 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| `NoCredentialsError` | Run `aws configure` or set AWS environment variables |
| `AccessDeniedException` | Enable Bedrock model access in AWS Console → Bedrock → Model access |
| `EndpointConnectionError` | Try `region_name='us-east-1'` (Bedrock availability) |
| No telemetry data | **Optional**: Set `export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"` to send to local collector |

## ⚡ Next Steps (Choose Your Path)

**🎯 Just want to see it work?**
```bash
python examples/bedrock/hello_genops_minimal.py  # Ultra-simple test
```

**🎯 Want team cost tracking?**
```bash
python examples/bedrock/basic_tracking.py  # Team attribution & cost comparison
```

**🚀 Ready for advanced features?**
```bash
python examples/bedrock/cost_optimization.py     # Multi-model optimization
python examples/bedrock/production_patterns.py   # Enterprise governance
```

**📚 Want comprehensive documentation?**
- **Integration Guide**: [`docs/integrations/bedrock.md`](../integrations/bedrock.md)
- **Examples Directory**: [`examples/bedrock/`](../../examples/bedrock/)
- **API Reference**: [`docs/api/providers/bedrock.md`](../api/providers/bedrock.md)

---

## 🎉 Success! You're Now Tracking AI Costs

**Your GenOps Bedrock integration is complete.** Every AI operation is now:
- ✅ Automatically tracked with accurate costs
- ✅ Attributed to teams and projects
- ✅ Exported to your observability platform
- ✅ Compliant with enterprise governance requirements

**Questions?** Join our [community discussions](https://github.com/KoshiHQ/GenOps-AI/discussions) or check the [comprehensive integration guide](../integrations/bedrock.md).