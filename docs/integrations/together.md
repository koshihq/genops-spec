# Together AI Integration Guide

Complete integration documentation for Together AI with GenOps governance telemetry. Access 200+ open-source models with full cost tracking, budget controls, and enterprise governance.

## What is GenOps?

**GenOps AI** is a governance telemetry layer built on OpenTelemetry that provides cost tracking, budget enforcement, and compliance monitoring for AI systems. It extends your existing observability stack with AI-specific governance capabilities without replacing your current tools.

**Key Benefits:**
- **Cost Transparency**: Real-time cost tracking across all AI operations
- **Budget Controls**: Configurable spending limits with enforcement policies
- **Multi-tenant Governance**: Per-team, per-project, per-customer attribution
- **Vendor Independence**: Works with 15+ observability platforms via OpenTelemetry
- **Zero Code Changes**: Auto-instrumentation for existing applications

## 🚀 Quick Start

### 1. Installation

```bash
# Install GenOps with Together AI support
pip install genops-ai[together] together

# Or install separately  
pip install genops-ai together
```

### 2. Environment Setup

```bash
# Get your API key from: https://api.together.xyz/settings/api-keys
export TOGETHER_API_KEY="your_together_api_key_here"

# Optional: Configure observability endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="your-service-name"
```

### 3. Validate Setup

```python
from genops.providers.together_validation import validate_together_setup

result = validate_together_setup()
if result.is_valid:
    print("✅ Ready for Together AI + GenOps integration!")
else:
    print(f"❌ Setup issues: {result.error_message}")
```

## 🏗️ Integration Patterns

### Pattern 1: Zero-Code Auto-Instrumentation

Add **one line** to existing Together AI code for complete governance:

```python
# Add this single line for automatic governance
from genops.providers.together import auto_instrument
auto_instrument()

# Your existing Together AI code works unchanged
from together import Together
client = Together()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
# ✅ Automatic cost tracking, governance, and observability added!
```

**Benefits:**
- Zero code changes to existing applications
- Automatic cost calculation and attribution
- Seamless OpenTelemetry integration
- Compatible with all Together AI features

### Pattern 2: Manual Adapter Control

Full control with explicit governance configuration:

```python
from genops.providers.together import GenOpsTogetherAdapter, TogetherModel

# Create adapter with governance settings
adapter = GenOpsTogetherAdapter(
    team="ai-research",
    project="model-analysis",
    environment="production",
    daily_budget_limit=100.0,
    governance_policy="enforced",  # Strict budget enforcement
    enable_cost_alerts=True
)

# Chat with comprehensive governance
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Analyze market trends"}],
    model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
    max_tokens=200,
    # Governance attributes for attribution
    customer_id="enterprise-client",
    feature="market-analysis"
)

print(f"Response: {result.response}")
print(f"Cost: ${result.cost:.6f}")
print(f"Model: {result.model_used}")
print(f"Tokens: {result.tokens_used}")
```

### Pattern 3: Session-Based Tracking

Group related operations for unified governance:

```python
# Track multiple operations in a session
with adapter.track_session("analysis-workflow") as session:
    # Step 1: Initial analysis
    result1 = adapter.chat_with_governance(
        messages=[{"role": "user", "content": "Analyze the dataset"}],
        model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        session_id=session.session_id,
        operation="initial-analysis"
    )
    
    # Step 2: Follow-up questions
    result2 = adapter.chat_with_governance(
        messages=[
            {"role": "user", "content": "Analyze the dataset"},
            {"role": "assistant", "content": result1.response},
            {"role": "user", "content": "What are the key insights?"}
        ],
        model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        session_id=session.session_id,
        operation="insight-extraction"
    )
    
    print(f"Session cost: ${session.total_cost:.6f}")
    print(f"Operations: {session.total_operations}")
```

### Pattern 4: Context Manager Integration

Use context managers for operation lifecycle management:

```python
from genops.providers.together import create_together_context

# Context manager for comprehensive tracking
with create_together_context(
    operation_name="competitive-analysis",
    team="product-team",
    customer_id="client-123"
) as (span, context):
    
    # Operations within context get automatic attribution
    result = adapter.chat_with_governance(
        messages=[{"role": "user", "content": "Compare competitors"}],
        model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        context=context
    )
    
    # Context automatically captures governance attributes
    span.set_attribute("genops.result.satisfaction", "high")
    span.set_attribute("genops.business.impact", "strategic")
```

## 🤖 Available Models & Pricing

### Chat & Reasoning Models

| Model | Cost/1M Tokens | Context Length | Best Use Case |
|-------|---------------|----------------|---------------|
| **Llama 3.1 8B Instruct** | $0.10 | 128K | High-volume, cost-sensitive |
| **Llama 3.1 70B Instruct** | $0.88 | 128K | Balanced performance |
| **Llama 3.1 405B Instruct** | $5.00 | 128K | Highest quality responses |
| **DeepSeek R1** | $0.14 | 32K | Advanced reasoning tasks |
| **DeepSeek R1 Distilled** | $0.14 | 32K | Reasoning with faster inference |
| **Mixtral 8x7B** | $0.60 | 32K | Balanced multilingual |
| **Mixtral 8x22B** | $1.20 | 64K | Advanced multilingual |

### Multimodal & Specialized Models

| Model | Cost/1M Tokens | Context Length | Capabilities |
|-------|---------------|----------------|--------------|
| **Qwen2.5-VL-72B** | $4.00 | 32K | Vision-language understanding |
| **Llama-Vision-Free** | $0.10 | 128K | Lightweight multimodal |
| **DeepSeek-Coder-V2** | $0.14 | 64K | Code generation & analysis |
| **Qwen2.5-Coder-32B** | $0.60 | 32K | Advanced programming tasks |

### Model Selection Examples

```python
from genops.providers.together_pricing import TogetherPricingCalculator

calc = TogetherPricingCalculator()

# Get cost-optimized model recommendation
recommendation = calc.recommend_model(
    task_complexity="moderate",      # simple, moderate, complex
    budget_per_operation=0.01,      # $0.01 budget
    min_context_length=8192
)

print(f"Recommended: {recommendation['recommended_model']}")
print(f"Estimated cost: ${recommendation['estimated_cost']:.6f}")
print(f"Reasoning: {recommendation['selection_reasoning']}")

# Compare costs across models
comparisons = calc.compare_models([
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
], estimated_tokens=1000)

for comp in comparisons:
    print(f"{comp['model']}: ${comp['estimated_cost']:.4f}")
```

## 💰 Cost Intelligence & Optimization

### Smart Model Selection

GenOps automatically selects optimal models based on task complexity and budget:

```python
# Budget-constrained operations
adapter = GenOpsTogetherAdapter(
    team="budget-team",
    project="cost-optimization",
    daily_budget_limit=10.0,
    governance_policy="enforced",
    auto_optimize_costs=True  # Enable intelligent model selection
)

# Adapter automatically selects cost-effective models
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Simple question"}],
    task_complexity="simple",  # Triggers 8B model selection
    budget_per_operation=0.001,
    fallback_models=[
        TogetherModel.LLAMA_3_1_8B_INSTRUCT,
        TogetherModel.DEEPSEEK_R1_DISTILL
    ]
)
```

### Cost Analysis & Projections

```python
from genops.providers.together_pricing import TogetherPricingCalculator

calc = TogetherPricingCalculator()

# Analyze costs for projected usage
analysis = calc.analyze_costs(
    operations_per_day=1000,
    avg_tokens_per_operation=500,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    days_to_analyze=30
)

print(f"Daily cost: ${analysis['daily_cost']:.2f}")
print(f"Monthly cost: ${analysis['monthly_cost']:.2f}")
print(f"Cost per operation: ${analysis['cost_per_operation']:.6f}")

# Get cost optimization suggestions
if analysis['potential_savings']['best_alternative']:
    alt = analysis['potential_savings']['best_alternative']
    print(f"Alternative: {alt['model']}")
    print(f"Potential monthly savings: ${analysis['potential_savings']['potential_monthly_savings']:.2f}")
```

### Budget Management

```python
# Real-time budget tracking
cost_summary = adapter.get_cost_summary()

print(f"Daily spending: ${cost_summary['daily_costs']:.6f}")
print(f"Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
print(f"Remaining budget: ${cost_summary['daily_budget_limit'] - cost_summary['daily_costs']:.6f}")

# Budget enforcement policies
if cost_summary['daily_budget_utilization'] > 80:
    print("⚠️ Approaching budget limit")
    # Switch to cheaper models automatically
    
elif cost_summary['daily_budget_utilization'] > 95:
    print("🚨 Budget limit reached")
    # Operations blocked if governance_policy="enforced"
```

## 🔧 Advanced Features

### Multimodal Operations

```python
# Vision-language analysis with cost tracking
result = adapter.chat_with_governance(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in this image"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }],
    model=TogetherModel.QWEN_VL_72B,
    multimodal_operation=True,
    feature="image-analysis"
)

print(f"Vision analysis: {result.response}")
print(f"Multimodal cost: ${result.cost:.6f}")
```

### Streaming Responses

```python
# Streaming with real-time cost tracking
def handle_stream_chunk(chunk, accumulated_cost):
    print(f"Chunk: {chunk.choices[0].delta.content}")
    print(f"Accumulated cost: ${accumulated_cost:.6f}")

# Stream with governance
stream_result = adapter.stream_with_governance(
    messages=[{"role": "user", "content": "Write a long story"}],
    model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
    max_tokens=500,
    on_chunk=handle_stream_chunk,
    stream_session="story-generation"
)

print(f"Final cost: ${stream_result.total_cost:.6f}")
```

### Code Generation Workflows

```python
# Specialized code generation with governance
code_result = adapter.chat_with_governance(
    messages=[{
        "role": "user", 
        "content": "Write a Python function to calculate Fibonacci numbers"
    }],
    model=TogetherModel.DEEPSEEK_CODER_V2,
    task_type="code_generation",
    programming_language="python",
    feature="code-assistant"
)

print(f"Generated code:\n{code_result.response}")
print(f"Code generation cost: ${code_result.cost:.6f}")
```

### Async Batch Processing

```python
import asyncio
from genops.providers.together import AsyncGenOpsTogetherAdapter

async def batch_process_with_governance():
    async_adapter = AsyncGenOpsTogetherAdapter(
        team="async-team",
        project="batch-processing",
        daily_budget_limit=50.0
    )
    
    # Process multiple operations concurrently
    tasks = []
    for i in range(10):
        task = async_adapter.achat_with_governance(
            messages=[{"role": "user", "content": f"Process item {i}"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            batch_id=f"batch-{i//5}",  # Group into batches
            operation_index=i
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    total_cost = sum(float(r.cost) for r in results)
    print(f"Batch processing cost: ${total_cost:.6f}")
    
    return results

# Run async batch processing
results = asyncio.run(batch_process_with_governance())
```

## 🏢 Enterprise Patterns

### Multi-Tenant Architecture

```python
# Enterprise multi-tenant setup
class EnterpriseTogetherAdapter:
    def __init__(self):
        self.tenant_adapters = {}
    
    def get_tenant_adapter(self, tenant_id: str, customer_config: dict):
        if tenant_id not in self.tenant_adapters:
            self.tenant_adapters[tenant_id] = GenOpsTogetherAdapter(
                team=customer_config["team"],
                project=customer_config["project"],
                customer_id=tenant_id,
                daily_budget_limit=customer_config["budget_limit"],
                governance_policy=customer_config.get("policy", "enforced"),
                cost_center=customer_config.get("cost_center"),
                tenant_id=tenant_id
            )
        return self.tenant_adapters[tenant_id]
    
    async def process_tenant_request(self, tenant_id: str, messages: list, **kwargs):
        adapter = self.get_tenant_adapter(tenant_id, kwargs["customer_config"])
        
        return adapter.chat_with_governance(
            messages=messages,
            model=kwargs.get("model", TogetherModel.LLAMA_3_1_8B_INSTRUCT),
            customer_id=tenant_id,
            feature=kwargs.get("feature", "multi-tenant-chat")
        )

# Usage
enterprise = EnterpriseTogetherAdapter()
result = await enterprise.process_tenant_request(
    tenant_id="client-123",
    messages=[{"role": "user", "content": "Customer query"}],
    customer_config={
        "team": "client-123-team",
        "project": "customer-ai",
        "budget_limit": 100.0,
        "policy": "enforced"
    }
)
```

### Circuit Breaker Pattern

```python
from genops.providers.together import TogetherCircuitBreaker

# Circuit breaker for resilient operations
circuit_breaker = TogetherCircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=30,      # Try recovery after 30s
    expected_recovery_time=10 # Expected recovery time
)

@circuit_breaker.protected_operation
def resilient_chat(adapter, messages, **kwargs):
    return adapter.chat_with_governance(
        messages=messages,
        **kwargs
    )

# Automatic fallback handling
try:
    result = resilient_chat(
        adapter,
        messages=[{"role": "user", "content": "Protected operation"}],
        model=TogetherModel.LLAMA_3_1_70B_INSTRUCT
    )
except circuit_breaker.CircuitOpenException:
    # Circuit is open, use fallback
    result = fallback_response_generator(messages)
```

### Production Monitoring

```python
# Production monitoring setup
adapter = GenOpsTogetherAdapter(
    team="production-team",
    project="customer-service",
    environment="production",
    daily_budget_limit=1000.0,
    governance_policy="enforced",
    enable_performance_monitoring=True,
    alert_thresholds={
        "high_cost_operation": 0.10,    # Alert if operation > $0.10
        "budget_utilization": 0.80,     # Alert at 80% budget
        "error_rate": 0.05,             # Alert at 5% error rate
        "latency_p95": 2.0              # Alert if P95 > 2s
    }
)

# Operations automatically monitored
with adapter.monitor_production_workload("customer-chat") as monitor:
    result = adapter.chat_with_governance(
        messages=messages,
        model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        production_workload="customer-service",
        sla_target_seconds=1.5,
        quality_threshold=0.8
    )
    
    # Automatic performance tracking
    monitor.record_success_metrics(result)
    
    if result.execution_time_seconds > 2.0:
        monitor.trigger_latency_alert(result)
```

## 📊 Observability Integration

### OpenTelemetry Configuration

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry for GenOps
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to your observability platform
otlp_exporter = OTLPSpanExporter(
    endpoint="http://your-otlp-endpoint:4317",
    headers={
        "api-key": "your-observability-api-key"
    }
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# GenOps automatically uses configured tracer
adapter = GenOpsTogetherAdapter(
    team="observability-team",
    project="ai-monitoring",
    use_opentelemetry=True,  # Enable OTel integration
    custom_tracer=tracer     # Use custom tracer
)
```

### Custom Metrics Export

```python
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Configure metrics export
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://your-otlp-endpoint:4317"),
    export_interval_millis=5000
)

metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# GenOps automatically exports custom metrics
adapter = GenOpsTogetherAdapter(
    team="metrics-team",
    project="ai-analytics",
    enable_custom_metrics=True,
    metric_labels={
        "service": "ai-service",
        "version": "1.0.0",
        "region": "us-west-2"
    }
)
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### API Key Problems

```bash
# Check API key format
echo $TOGETHER_API_KEY  # Should start with valid prefix

# Test API access
python -c "from together import Together; print(len(Together().models.list().data))"

# Validate with GenOps
python -c "from genops.providers.together_validation import validate_together_setup; print(validate_together_setup().is_valid)"
```

#### Import Errors

```bash
# Check installation
pip show genops-ai together

# Reinstall if needed
pip install --upgrade genops-ai[together] together

# Verify imports
python -c "from genops.providers.together import GenOpsTogetherAdapter; print('✅ Import successful')"
```

#### Model Access Issues

```python
# Test specific model access
from genops.providers.together import GenOpsTogetherAdapter
from genops.providers.together_models import TogetherModel

adapter = GenOpsTogetherAdapter()

try:
    result = adapter.chat_with_governance(
        messages=[{"role": "user", "content": "test"}],
        model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
        max_tokens=5,
        test_mode=True
    )
    print(f"✅ Model access successful: {result.model_used}")
except Exception as e:
    print(f"❌ Model access failed: {e}")
```

#### Budget and Cost Issues

```python
# Diagnose budget problems
cost_summary = adapter.get_cost_summary()
print(f"Current utilization: {cost_summary['daily_budget_utilization']:.1f}%")
print(f"Daily costs: ${cost_summary['daily_costs']:.6f}")
print(f"Budget limit: ${cost_summary['daily_budget_limit']:.2f}")

if cost_summary['daily_budget_utilization'] > 95:
    print("🚨 Budget exhausted - increase limit or wait for reset")
elif cost_summary['daily_budget_utilization'] > 80:
    print("⚠️ High budget utilization - consider cost optimization")
```

#### Performance Issues

```python
# Performance diagnostics
import time
start_time = time.time()

result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Performance test"}],
    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
    max_tokens=10,
    diagnostic_mode=True
)

local_overhead = time.time() - start_time - result.execution_time_seconds
print(f"Local overhead: {local_overhead:.3f}s")
print(f"API latency: {result.execution_time_seconds:.3f}s")

if local_overhead > 0.1:
    print("⚠️ High local overhead detected")
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
adapter = GenOpsTogetherAdapter(
    team="debug-team",
    project="troubleshooting",
    debug_mode=True,
    log_level="DEBUG"
)

# Operations will show detailed logs
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Debug test"}],
    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
    verbose=True
)
```

## 🔗 External Resources

### Documentation Links
- **[5-Minute Quickstart Guide](../together-quickstart.md)** - Get started immediately
- **[Complete Examples](../../examples/together/)** - 7+ working examples
- **[Performance Benchmarks](../together-performance-benchmarks.md)** - Optimization guide

### Platform Resources
- **[Together AI Platform](https://api.together.xyz)** - API dashboard and keys
- **[Together AI Model Catalog](https://docs.together.ai/docs/inference-models)** - Complete model list
- **[GenOps Documentation](https://docs.genops.ai)** - Full platform documentation
- **[OpenTelemetry Documentation](https://opentelemetry.io/docs/)** - Observability standards

### Community & Support
- **[GitHub Repository](https://github.com/genops-ai/genops-ai)** - Source code and issues
- **[GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)** - Community Q&A
- **[Contribution Guide](https://github.com/genops-ai/genops-ai/blob/main/CONTRIBUTING.md)** - How to contribute

## 📈 Success Metrics

After implementing Together AI + GenOps integration, teams typically achieve:

- **💰 Cost Reduction**: Up to 10x lower costs vs proprietary models
- **📊 Cost Visibility**: 100% cost attribution and budget compliance
- **🎯 Model Optimization**: Intelligent model selection for each use case
- **🔍 Observability**: Complete telemetry integration with existing tools
- **⚡ Performance**: Sub-100ms local overhead with full governance
- **🏢 Enterprise Ready**: Multi-tenant governance and audit compliance

---

*This integration guide provides comprehensive documentation for Together AI + GenOps. For quick setup, see the [5-minute quickstart guide](../together-quickstart.md). For working examples, explore the [examples directory](../../examples/together/).*