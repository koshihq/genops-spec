# Hugging Face GenOps Quickstart Guide

Get started with GenOps AI governance for Hugging Face in under 5 minutes! This guide demonstrates immediate value with zero-code auto-instrumentation.

## 🧭 Quick Navigation

**Just Getting Started?** → [Install](#1-install-genops-with-hugging-face-support) → [Zero-Code Setup](#3-zero-code-auto-instrumentation-🚀-instant-value) → [First Success](#5-minute-success-checklist)  
**Want Cost Tracking?** → [Multi-Provider Costs](#multi-provider-cost-tracking-2-minutes) → [Advanced Context Managers](#advanced-cost-context-manager-new-2-minutes)  
**Going to Production?** → [Performance Features](#performance-features-new-1-minute) → [Production Deployment](#production-deployment)  
**Having Issues?** → [Troubleshooting](#troubleshooting) → [Get Help](#get-help)

## Quick Setup

### 1. Install GenOps with Hugging Face Support

```bash
# Core installation with Hugging Face support
pip install genops-ai[huggingface]

# Or install components separately
pip install genops-ai huggingface_hub
```

### 2. Verify Your Setup

```bash
# Quick validation check
python -c "from genops.providers.huggingface import quick_validate; quick_validate()"

# Or run comprehensive validation
python -m genops.providers.huggingface_validation
```

### 2.5. Ultra-Simple Hello World (30 seconds)

Before diving deeper, let's confirm everything works with the simplest possible example:

```python
# Save this as hello_genops.py and run it
from genops.providers.huggingface import instrument_huggingface
instrument_huggingface()

from huggingface_hub import InferenceClient
client = InferenceClient()

# This single line now has comprehensive AI governance!
result = client.text_generation("Hello GenOps!", model="microsoft/DialoGPT-medium")
print(f"✅ Success! Generated: {result}")
print("🎉 You now have cost tracking, governance, and observability!")
```

Run it:
```bash
python hello_genops.py
```

**That's it!** If you see output, GenOps is working perfectly. You've just added enterprise-grade AI governance with one line of code.

### 3. Zero-Code Auto-Instrumentation (🚀 **Instant Value!**)

```python
# This is ALL the code you need to add comprehensive AI governance!
from genops.providers.huggingface import instrument_huggingface

# Enable automatic telemetry for ALL Hugging Face API calls
instrument_huggingface()

# Your existing code works unchanged with automatic GenOps tracking
from huggingface_hub import InferenceClient

client = InferenceClient()

# This call now automatically captures:
# ✅ Cost calculation and tracking
# ✅ Provider detection (OpenAI, Anthropic, Hub models)
# ✅ Performance metrics
# ✅ Error tracking and debugging info
# ✅ OpenTelemetry export to your observability platform
response = client.text_generation(
    "Write a creative story opening",
    model="microsoft/DialoGPT-medium"
)
print(response)
```

**🎉 Congratulations!** You now have comprehensive AI governance with zero changes to your existing Hugging Face code!

## Add Governance Attributes (30 seconds)

For team cost attribution and customer billing, just add governance attributes:

```python
# Your existing calls work exactly the same, just add governance attributes
response = client.text_generation(
    "Generate a product description",
    model="microsoft/DialoGPT-medium",
    
    # Add these governance attributes for cost attribution
    team="marketing-team",           # Team cost tracking
    project="product-launch-q4",     # Project attribution  
    customer_id="enterprise-123",    # Customer billing
    environment="production"         # Environment segregation
)

# All costs automatically attributed to marketing-team for enterprise-123
```

## Multi-Provider Cost Tracking (2 minutes)

GenOps automatically detects and optimizes costs across different providers accessed through Hugging Face:

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

# Create adapter for manual control
adapter = GenOpsHuggingFaceAdapter()

# OpenAI models via Hugging Face - costs tracked accurately
openai_response = adapter.text_generation(
    "Explain quantum computing briefly",
    model="gpt-3.5-turbo",           # Detected as OpenAI provider
    team="research-team",
    customer_id="science-client"
)

# Anthropic models via Hugging Face - unified cost tracking  
anthropic_response = adapter.text_generation(
    "Explain quantum computing briefly", 
    model="claude-3-haiku",          # Detected as Anthropic provider
    team="research-team",
    customer_id="science-client" 
)

# Native Hub models - optimized for cost
hub_response = adapter.text_generation(
    "Explain quantum computing briefly",
    model="microsoft/DialoGPT-medium", # Detected as Hugging Face Hub
    team="research-team", 
    customer_id="science-client"
)

# All costs automatically aggregated by team and customer across providers!
```

## Advanced Cost Context Manager (NEW! 2 minutes)

Track costs across multiple operations with automatic aggregation:

```python
from genops.providers.huggingface import create_huggingface_cost_context

# Advanced cost tracking with automatic aggregation
with create_huggingface_cost_context("multi_provider_demo") as context:
    adapter = GenOpsHuggingFaceAdapter()
    
    # Multiple operations - costs automatically unified
    openai_result = adapter.text_generation(
        "Analyze market trends",
        model="gpt-3.5-turbo",
        team="research"
    )
    
    anthropic_result = adapter.text_generation(
        "Cross-validate analysis",
        model="claude-3-haiku",
        team="research" 
    )
    
    hub_result = adapter.feature_extraction(
        ["market", "analysis", "validation"],
        model="sentence-transformers/all-MiniLM-L6-v2",
        team="research"
    )
    
    # Get comprehensive cost summary
    summary = context.get_current_summary()
    print(f"💰 Total cost: ${summary.total_cost:.4f}")
    print(f"🏗️  Providers used: {list(summary.unique_providers)}")
    print(f"🔧 Models used: {list(summary.unique_models)}")
    
    # Get detailed cost breakdown
    breakdown = summary.get_provider_breakdown()
    for provider, details in breakdown.items():
        print(f"   {provider}: ${details['cost']:.4f} ({details['calls']} calls)")
```

## Production Workflow Context (NEW! 2 minutes)

Enterprise-grade workflow orchestration with full governance:

```python
from genops.providers.huggingface import production_workflow_context

# Enterprise workflow with comprehensive governance
with production_workflow_context(
    workflow_name="content_analysis_pipeline",
    customer_id="demo-enterprise",
    team="content-team", 
    project="ai-content-analysis",
    environment="production",
    cost_center="R&D"
) as (workflow, workflow_id):
    
    adapter = GenOpsHuggingFaceAdapter()
    
    # Step 1: Content analysis
    workflow.record_step("content_analysis")
    analysis = adapter.text_generation(
        "Analyze the sentiment and key themes in this content...",
        model="gpt-3.5-turbo",
        max_new_tokens=150
    )
    
    # Step 2: Generate embeddings
    workflow.record_step("embedding_generation")
    embeddings = adapter.feature_extraction(
        [analysis],
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Record performance metrics
    workflow.record_performance_metric("documents_processed", 1, "count")
    
    # Get workflow cost summary
    final_cost = workflow.get_current_cost_summary()
    workflow.record_performance_metric("workflow_cost", final_cost.total_cost, "USD")
    
    print(f"✅ Workflow {workflow_id} completed")
    print(f"💰 Total cost: ${final_cost.total_cost:.4f}")
    print(f"📊 Full governance telemetry exported")
```

## Performance Features (NEW! 1 minute)

Configure performance optimization for production workloads:

```bash
# Performance and production configuration
export GENOPS_SAMPLING_RATE="0.5"          # Sample 50% of operations (reduces overhead)
export GENOPS_ASYNC_EXPORT="true"          # Non-blocking telemetry export
export GENOPS_CIRCUIT_BREAKER="true"       # Automatic failure protection
export GENOPS_CB_THRESHOLD="3"             # Circuit breaker failure threshold
export GENOPS_CB_WINDOW="60"               # Reset window (seconds)
```

Test performance configuration:

```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

adapter = GenOpsHuggingFaceAdapter()

# Check current performance settings
config = adapter.get_performance_config()
print("Performance Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Operations automatically respect sampling and circuit breaker settings
result = adapter.text_generation(
    "Performance test prompt",
    model="microsoft/DialoGPT-medium", 
    team="performance-team"
)
# ✅ Sampling, circuit breaker, and async export working automatically
```

## Observability Integration (1 minute)

Export telemetry to your existing observability platform:

```bash
# Set up OpenTelemetry export (choose your platform)
export OTEL_SERVICE_NAME="my-ai-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # Local collector
# export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-datadog-endpoint"  # Datadog
# export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-honeycomb-endpoint"  # Honeycomb

# Your GenOps telemetry automatically flows to your dashboards!
```

## What You Get Automatically

### 💰 **Cost Intelligence**
- Real-time cost calculation across OpenAI, Anthropic, Cohere, Meta, Google, Hub models
- Provider cost comparison and optimization recommendations  
- Team and customer cost attribution for accurate billing
- Budget alerts and cost optimization suggestions

### 🏛️ **Enterprise Governance**
- Team, project, and customer cost attribution
- Environment segregation (dev/staging/production)
- Cost center tracking for financial reporting
- Audit trails for compliance and debugging

### 📊 **Observability Integration**
- OpenTelemetry-native telemetry export
- Works with Datadog, Honeycomb, Grafana, Jaeger, New Relic, etc.
- Rich performance and cost metrics
- Error tracking and debugging information

### 🤗 **Comprehensive Hugging Face Support**
- Text generation, chat completions, embeddings, image generation
- Automatic provider detection and cost optimization
- Works with Hub models and third-party providers via HF
- Zero-code instrumentation for existing applications

### ⚡ **Production-Ready Performance** (NEW!)
- Configurable sampling rates to control telemetry overhead (0.0-1.0)
- Async telemetry export for non-blocking operations
- Circuit breaker protection for API failure resilience
- Batch processing optimization for high-volume applications

### 🏗️  **Advanced Context Managers** (NEW!)
- `create_huggingface_cost_context()` for multi-operation cost aggregation
- `production_workflow_context()` for enterprise workflow orchestration
- Automatic step recording and performance metric tracking
- Comprehensive governance attribute propagation

## 5-Minute Success Checklist

✅ **Install**: `pip install genops-ai[huggingface]`  
✅ **Validate**: Run validation to check setup  
✅ **Instrument**: Add `instrument_huggingface()` to your app  
✅ **Test**: Run existing Hugging Face code - costs automatically tracked!  
✅ **Govern**: Add team/project/customer attributes for attribution  
✅ **NEW! Context Managers**: Try `create_huggingface_cost_context()` for advanced cost tracking  
✅ **NEW! Performance**: Configure sampling and circuit breaker for production  

## Troubleshooting

### Common Issues

**❓ "Auto-instrumentation not working"**
```python
# Check if validation passes
from genops.providers.huggingface import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result)
```

**❓ "Models seem slow or timing out"** 
```python
# Some models may have rate limits - try Hub models for testing
response = client.text_generation(
    "Test prompt",
    model="microsoft/DialoGPT-medium"  # Usually fast and reliable
)
```

**❓ "Not seeing telemetry data"**
```bash
# Check OpenTelemetry configuration
echo $OTEL_SERVICE_NAME
echo $OTEL_EXPORTER_OTLP_ENDPOINT

# Test with console output
export OTEL_EXPORTER_TYPE=console
```

**❓ "Circuit breaker is blocking operations"** (NEW!)
```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
adapter = GenOpsHuggingFaceAdapter()
config = adapter.get_performance_config()
if config['circuit_breaker_open']:
    print("Circuit breaker is open - wait for reset or disable it")
    print(f"Failures: {config['circuit_breaker_failures']}")
```

**❓ "Operations not being sampled correctly"** (NEW!)
```bash
# Check sampling configuration
echo "Sampling rate: $GENOPS_SAMPLING_RATE"
echo "Async export: $GENOPS_ASYNC_EXPORT"

# Reset to full sampling for testing
export GENOPS_SAMPLING_RATE="1.0"
```

### Get Help

- 📖 **Complete Guide**: [Integration Documentation](integrations/huggingface.md)
- 🧪 **Examples**: Run `python examples/huggingface/setup_validation.py`
- 🐛 **Issues**: Report at [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- 💬 **Community**: Join our [Discord](https://discord.gg/genops-ai)

## Next Steps

### Explore More Features (Optional)

1. **🔍 Multi-Provider Costs**: `python examples/huggingface/cost_tracking.py`
2. **🚀 Advanced Features**: `python examples/huggingface/huggingface_specific_advanced.py`  
3. **🏭 Production Patterns**: `python examples/huggingface/production_patterns.py`
4. **🐳 Container Deployment**: `python examples/huggingface/docker_integration.py`
5. **☸️  Kubernetes Deployment**: `python examples/huggingface/kubernetes_integration.py`
6. **🔄 CI/CD Integration**: `python examples/huggingface/cicd_integration.py`

### Production Deployment

1. **Configure OpenTelemetry export** to your observability platform
2. **Set up monitoring dashboards** for cost and performance metrics
3. **Implement budget alerts** and cost optimization policies
4. **Create team governance policies** for cost attribution and access control

---

**🚀 You're now ready to build AI applications with comprehensive governance, cost intelligence, and observability!**

The power of GenOps is that it works invisibly alongside your existing code, providing enterprise-grade AI governance without changing how you develop.