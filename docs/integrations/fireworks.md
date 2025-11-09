# Fireworks AI Integration Guide

Complete integration documentation for Fireworks AI with GenOps governance telemetry. Access 100+ models across all modalities with 4x faster inference, comprehensive cost tracking, and enterprise-grade governance controls.

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
# Install GenOps with Fireworks AI support
pip install genops-ai[fireworks] fireworks-ai

# Or install separately  
pip install genops-ai fireworks-ai
```

### 2. Environment Setup

```bash
# Get your API key from: https://fireworks.ai/api-keys
export FIREWORKS_API_KEY="your_fireworks_api_key_here"

# Optional: Configure observability endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_SERVICE_NAME="your-service-name"
```

### 3. Validate Setup

```python
from genops.providers.fireworks_validation import validate_fireworks_setup

result = validate_fireworks_setup()
if result.is_valid:
    print("✅ Ready for Fireworks AI + GenOps integration!")
else:
    print(f"❌ Setup issues: {result.error_message}")
```

## 🏗️ Integration Patterns

### Pattern 1: Zero-Code Auto-Instrumentation

Add **one line** to existing Fireworks AI code for complete governance:

```python
# Add this single line for automatic governance
from genops.providers.fireworks import auto_instrument
auto_instrument()

# Your existing Fireworks AI code works unchanged
from fireworks.client import Fireworks
client = Fireworks()

response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
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
- Compatible with all Fireworks AI features

### Pattern 2: Manual Adapter Control

Full control with explicit governance configuration:

```python
from genops.providers.fireworks import GenOpsFireworksAdapter, FireworksModel

# Create adapter with governance settings
adapter = GenOpsFireworksAdapter(
    team="ai-research",
    project="model-analysis",
    environment="production",
    daily_budget_limit=100.0,
    governance_policy="enforced",  # Strict budget enforcement
    enable_cost_alerts=True
)

# Chat with comprehensive governance
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Analyze market trends with fast inference"}],
    model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
    max_tokens=200,
    # Governance attributes for attribution
    customer_id="enterprise-client",
    feature="market-analysis"
)

print(f"Response: {result.response}")
print(f"Cost: ${result.cost:.6f}")
print(f"Model: {result.model_used}")
print(f"Speed: {result.execution_time_seconds:.2f}s")
```

### Pattern 3: Session-Based Tracking

Group related operations for unified governance:

```python
# Track multiple operations in a session
with adapter.track_session("analysis-workflow") as session:
    # Step 1: Initial analysis with fast model
    result1 = adapter.chat_with_governance(
        messages=[{"role": "user", "content": "Analyze the dataset quickly"}],
        model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,  # Fast model
        session_id=session.session_id,
        operation="initial-analysis"
    )
    
    # Step 2: Deep analysis with larger model
    result2 = adapter.chat_with_governance(
        messages=[
            {"role": "user", "content": "Analyze the dataset quickly"},
            {"role": "assistant", "content": result1.response},
            {"role": "user", "content": "Provide detailed insights"}
        ],
        model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,  # Detailed model
        session_id=session.session_id,
        operation="deep-analysis"
    )
    
    print(f"Session cost: ${session.total_cost:.6f}")
    print(f"Operations: {session.total_operations}")
    print(f"Average speed: {(result1.execution_time_seconds + result2.execution_time_seconds) / 2:.2f}s")
```

### Pattern 4: Multi-Modal Operations

Leverage Fireworks AI's multimodal capabilities with governance:

```python
# Vision-language analysis with cost tracking
result = adapter.chat_with_governance(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image for business insights"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }],
    model=FireworksModel.LLAMA_VISION_11B,
    multimodal_operation=True,
    feature="visual-analysis"
)

print(f"Vision analysis: {result.response}")
print(f"Multimodal cost: ${result.cost:.6f}")

# Embedding operations with governance
embedding_result = adapter.embeddings_with_governance(
    input_texts=["Document 1 content", "Document 2 content"],
    model=FireworksModel.NOMIC_EMBED_TEXT,
    feature="semantic-search",
    use_case="document-similarity"
)

print(f"Embeddings cost: ${embedding_result.cost:.6f}")
```

## 🤖 Available Models & Pricing

### Chat & Reasoning Models

| Model | Parameters | Cost/1M Tokens | Context Length | Best Use Case |
|-------|-----------|---------------|----------------|---------------|
| **Llama 3.1 8B Instruct** | 8B | $0.20 | 128K | High-throughput, fast responses |
| **Llama 3.1 70B Instruct** | 70B | $0.90 | 128K | Balanced quality and performance |
| **Llama 3.1 405B Instruct** | 405B | $3.00 | 128K | Highest quality responses |
| **DeepSeek R1** | 70B | $1.35 input, $5.40 output | 32K | Advanced reasoning tasks |
| **DeepSeek R1 Distilled** | 70B | $0.14 input, $0.56 output | 32K | Cost-effective reasoning |
| **Mixtral 8x7B** | 8x7B MoE | $0.50 | 32K | Efficient multilingual |
| **Mixtral 8x22B** | 8x22B MoE | $1.20 | 65K | Advanced multilingual |

### Multimodal & Specialized Models

| Model | Cost/1M Tokens | Context Length | Capabilities |
|-------|---------------|----------------|--------------|
| **Llama Vision 11B** | $0.20 | 32K | Vision-language understanding |
| **Qwen2-VL-72B** | $0.90 | 32K | Advanced vision-language |
| **Pixtral 12B** | $0.15 | 128K | Lightweight multimodal |
| **DeepSeek Coder V2 Lite** | $0.20 | 65K | Code generation & analysis |
| **Qwen2.5 Coder 32B** | $0.20 | 32K | Advanced programming tasks |
| **Nomic Embed Text** | $0.02 | 8K | Text embeddings |
| **Whisper V3** | $0.006/min | - | Audio transcription |

### Model Selection Examples

```python
from genops.providers.fireworks_pricing import FireworksPricingCalculator

calc = FireworksPricingCalculator()

# Get cost-optimized model recommendation
recommendation = calc.recommend_model(
    task_complexity="moderate",      # simple, moderate, complex
    budget_per_operation=0.01,      # $0.01 budget
    min_context_length=8192
)

print(f"Recommended: {recommendation.recommended_model}")
print(f"Estimated cost: ${recommendation.estimated_cost:.6f}")
print(f"Reasoning: {recommendation.reasoning}")

# Compare costs across models
comparisons = calc.compare_models([
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "accounts/deepseek-ai/models/deepseek-r1-distill-llama-70b"
], estimated_tokens=1000)

for comp in comparisons:
    print(f"{comp['model']}: ${comp['estimated_cost']:.4f}")
    if comp.get('batch_cost'):
        print(f"  Batch: ${comp['batch_cost']:.4f} (saves ${comp['batch_savings']:.4f})")
```

## 💰 Cost Intelligence & Optimization

### Smart Model Selection

GenOps automatically selects optimal models based on task complexity and budget:

```python
# Budget-constrained operations with intelligent selection
adapter = GenOpsFireworksAdapter(
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
        FireworksModel.LLAMA_3_1_8B_INSTRUCT,
        FireworksModel.LLAMA_3_2_1B_INSTRUCT
    ]
)
```

### Batch Processing Optimization

Fireworks AI offers 50% cost savings for batch processing:

```python
# Batch processing with 50% discount
batch_messages = [
    [{"role": "user", "content": f"Process item {i}"}] for i in range(100)
]

total_cost = Decimal("0.00")

for messages in batch_messages:
    result = adapter.chat_with_governance(
        messages=messages,
        model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
        is_batch=True,  # Applies 50% discount
        batch_operation="bulk-processing"
    )
    total_cost += result.cost

print(f"Batch processing saved: ${(total_cost * 2 - total_cost):.2f}")
```

### Cost Analysis & Projections

```python
from genops.providers.fireworks_pricing import FireworksPricingCalculator

calc = FireworksPricingCalculator()

# Analyze costs for projected usage
analysis = calc.analyze_costs(
    operations_per_day=1000,
    avg_tokens_per_operation=500,
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    days_to_analyze=30,
    batch_percentage=0.3  # 30% of operations use batch pricing
)

print(f"Daily cost: ${analysis['cost_analysis']['daily_cost']:.2f}")
print(f"Monthly cost: ${analysis['cost_analysis']['monthly_cost']:.2f}")
print(f"Cost per operation: ${analysis['cost_analysis']['cost_per_operation']:.6f}")

# Get cost optimization suggestions
if analysis['optimization']['best_alternative']:
    alt = analysis['optimization']['best_alternative']
    print(f"Alternative: {alt['model']}")
    print(f"Potential monthly savings: ${analysis['optimization']['potential_monthly_savings']:.2f}")
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

### Function Calling with Governance

```python
# Define functions for the model to call
functions = [
    {
        "name": "get_weather",
        "description": "Get weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            }
        }
    }
]

result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
    functions=functions,
    function_call="auto",
    feature="weather-assistant"
)

print(f"Function calling result: {result.response}")
print(f"Cost: ${result.cost:.6f}")
```

### Structured Output Generation

```python
# Generate structured JSON output
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "analysis_result",
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_themes": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["sentiment", "confidence", "key_themes"]
        }
    }
}

result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Analyze the sentiment of this text: 'I love the fast performance!'"}],
    model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
    response_format=response_format,
    feature="sentiment-analysis"
)

print(f"Structured output: {result.response}")
```

### Streaming with Real-Time Cost Tracking

```python
# Streaming responses with governance
def handle_stream_chunk(chunk, accumulated_cost):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
    print(f"\nAccumulated cost: ${accumulated_cost:.6f}")

result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Write a long story about AI"}],
    model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
    max_tokens=500,
    stream=True,
    on_chunk=handle_stream_chunk,
    feature="creative-writing"
)

print(f"\nFinal streaming cost: ${result.cost:.6f}")
```

### Audio Processing with Governance

```python
# Audio transcription with cost tracking
import requests

# Download sample audio (you would use your own audio file)
audio_url = "https://example.com/sample-audio.wav"
audio_response = requests.get(audio_url)

# Note: This is a conceptual example - actual implementation would handle audio files
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Transcribe this audio"}],
    model=FireworksModel.WHISPER_V3,
    # In practice, you'd pass audio data differently
    audio_duration_minutes=2.5,  # For cost calculation
    feature="audio-transcription"
)

print(f"Transcription cost: ${result.cost:.6f}")
```

## 🏢 Enterprise Patterns

### Multi-Tenant Architecture with SOC 2 Compliance

```python
# Enterprise multi-tenant setup with compliance features
class EnterpriseFireworksAdapter:
    def __init__(self):
        self.tenant_adapters = {}
        self.compliance_logger = self._init_compliance_logging()
    
    def get_tenant_adapter(self, tenant_id: str, customer_config: dict):
        if tenant_id not in self.tenant_adapters:
            self.tenant_adapters[tenant_id] = GenOpsFireworksAdapter(
                team=customer_config["team"],
                project=customer_config["project"],
                customer_id=tenant_id,
                daily_budget_limit=customer_config["budget_limit"],
                governance_policy=customer_config.get("policy", "enforced"),
                cost_center=customer_config.get("cost_center"),
                tenant_id=tenant_id,
                # Enterprise compliance features
                enable_audit_trail=True,
                compliance_level="SOC2",
                enable_data_residency=True
            )
        return self.tenant_adapters[tenant_id]
    
    async def process_tenant_request(self, tenant_id: str, messages: list, **kwargs):
        adapter = self.get_tenant_adapter(tenant_id, kwargs["customer_config"])
        
        # Log for compliance audit
        self.compliance_logger.info(f"Processing request for tenant {tenant_id}")
        
        return adapter.chat_with_governance(
            messages=messages,
            model=kwargs.get("model", FireworksModel.LLAMA_3_1_8B_INSTRUCT),
            customer_id=tenant_id,
            feature=kwargs.get("feature", "multi-tenant-chat")
        )
    
    def _init_compliance_logging(self):
        # Initialize compliance-specific logging
        import logging
        compliance_logger = logging.getLogger("fireworks.compliance")
        # Configure for SOC 2 compliance requirements
        return compliance_logger

# Usage
enterprise = EnterpriseFireworksAdapter()
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

### Circuit Breaker Pattern for Resilience

```python
from genops.providers.fireworks import create_circuit_breaker

# Circuit breaker for resilient operations
circuit_breaker = create_circuit_breaker(
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
        model=FireworksModel.LLAMA_3_1_70B_INSTRUCT
    )
except circuit_breaker.CircuitOpenException:
    # Circuit is open, use fallback
    result = fallback_response_generator(messages)
```

### Production Monitoring & Alerting

```python
# Production monitoring setup with performance optimization
adapter = GenOpsFireworksAdapter(
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
        "latency_p95": 2.0,             # Alert if P95 > 2s
        "slow_inference": 5.0           # Alert if inference > 5s (Fireworks should be faster)
    }
)

# Operations automatically monitored with Fireworks performance expectations
with adapter.monitor_production_workload("customer-chat") as monitor:
    result = adapter.chat_with_governance(
        messages=messages,
        model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
        production_workload="customer-service",
        sla_target_seconds=1.0,  # Expect fast Fireworks inference
        quality_threshold=0.8
    )
    
    # Automatic performance tracking
    monitor.record_success_metrics(result)
    
    # Alert on unexpected slow performance (Fireworks should be fast)
    if result.execution_time_seconds > 3.0:
        monitor.trigger_performance_alert(result, "unexpectedly_slow_inference")
```

## 📊 Performance Optimization

### Fireattention Performance Benefits

Fireworks AI's custom Fireattention CUDA kernels provide 4x faster inference:

```python
import time

# Measure Fireworks performance advantage
def benchmark_fireworks_speed():
    adapter = GenOpsFireworksAdapter(
        team="performance-team",
        project="speed-test"
    )
    
    test_messages = [{"role": "user", "content": "Explain quantum computing in detail"}]
    
    # Test with different model sizes
    models_to_test = [
        (FireworksModel.LLAMA_3_1_8B_INSTRUCT, "8B"),
        (FireworksModel.LLAMA_3_1_70B_INSTRUCT, "70B"),
    ]
    
    results = {}
    
    for model, size in models_to_test:
        start_time = time.time()
        
        result = adapter.chat_with_governance(
            messages=test_messages,
            model=model,
            max_tokens=200,
            temperature=0.7
        )
        
        end_time = time.time()
        
        results[size] = {
            "total_time": end_time - start_time,
            "tokens": result.tokens_used,
            "tokens_per_second": result.tokens_used / (end_time - start_time),
            "cost": float(result.cost),
            "cost_per_token": float(result.cost) / result.tokens_used
        }
        
        print(f"{size} Model Performance:")
        print(f"  Speed: {results[size]['tokens_per_second']:.1f} tokens/s")
        print(f"  Cost efficiency: ${results[size]['cost_per_token']:.6f}/token")
        print()
    
    return results

# Run benchmark
performance_results = benchmark_fireworks_speed()
```

### Batch Processing Optimization

```python
# Optimize for high-throughput batch processing
async def optimized_batch_processing(adapter, batch_data, batch_size=50):
    import asyncio
    
    # Process in optimized batches
    results = []
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i + batch_size]
        
        # Process batch concurrently
        batch_tasks = []
        
        for item in batch:
            task = adapter.chat_with_governance(
                messages=item["messages"],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,  # Fast model for throughput
                max_tokens=item.get("max_tokens", 100),
                is_batch=True,  # 50% cost savings
                batch_id=f"batch_{i//batch_size}",
                operation_id=item["id"]
            )
            batch_tasks.append(task)
        
        # Wait for batch completion
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Calculate batch metrics
        batch_cost = sum(float(r.cost) for r in batch_results)
        batch_time = max(r.execution_time_seconds for r in batch_results)
        
        print(f"Batch {i//batch_size + 1}: {len(batch)} items, ${batch_cost:.4f}, {batch_time:.2f}s")
    
    return results

# Usage
batch_data = [
    {"id": i, "messages": [{"role": "user", "content": f"Process item {i}"}]}
    for i in range(1000)
]

# results = await optimized_batch_processing(adapter, batch_data)
```

## 📊 Observability Integration

### OpenTelemetry Configuration for Fireworks

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry for GenOps + Fireworks
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
adapter = GenOpsFireworksAdapter(
    team="observability-team",
    project="ai-monitoring",
    use_opentelemetry=True,  # Enable OTel integration
    custom_tracer=tracer     # Use custom tracer
)

# Operations automatically create rich telemetry spans
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Test with observability"}],
    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
    feature="observability-test"
)

# Span includes:
# - Fireworks-specific attributes (model, speed, cost)
# - GenOps governance attributes (team, project, customer)
# - Performance metrics (latency, throughput)
# - Cost attribution data
```

### Custom Metrics Export

```python
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Configure metrics export for Fireworks performance tracking
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://your-otlp-endpoint:4317"),
    export_interval_millis=5000
)

metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# GenOps automatically exports Fireworks-specific metrics
adapter = GenOpsFireworksAdapter(
    team="metrics-team",
    project="fireworks-analytics",
    enable_custom_metrics=True,
    metric_labels={
        "service": "fireworks-ai-service",
        "version": "1.0.0",
        "region": "us-west-2",
        "provider": "fireworks"
    }
)

# Metrics automatically exported:
# - fireworks.inference.latency (with 4x speed advantage)
# - fireworks.cost.per_token (with cost efficiency data)
# - fireworks.throughput.tokens_per_second
# - fireworks.model.utilization
# - fireworks.batch.savings (50% discount tracking)
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### API Key Problems

```bash
# Check API key format
echo $FIREWORKS_API_KEY  # Should have valid Fireworks format

# Test API access
python -c "from fireworks.client import Fireworks; print('✅ Connected' if Fireworks().chat else '❌ Failed')"

# Validate with GenOps
python -c "from genops.providers.fireworks_validation import validate_fireworks_setup; print('✅' if validate_fireworks_setup().is_valid else '❌')"
```

#### Import Errors

```bash
# Check installation
pip show genops-ai fireworks-ai

# Reinstall if needed
pip install --upgrade genops-ai[fireworks] fireworks-ai

# Verify imports
python -c "from genops.providers.fireworks import GenOpsFireworksAdapter; print('✅ Import successful')"
```

#### Model Access Issues

```python
# Test specific model access
from genops.providers.fireworks import GenOpsFireworksAdapter
from genops.providers.fireworks import FireworksModel

adapter = GenOpsFireworksAdapter()

try:
    result = adapter.chat_with_governance(
        messages=[{"role": "user", "content": "test"}],
        model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
        max_tokens=5,
        test_mode=True
    )
    print(f"✅ Model access successful: {result.model_used}")
except Exception as e:
    print(f"❌ Model access failed: {e}")
```

#### Performance Issues

```python
# Performance diagnostics for Fireworks
import time

start_time = time.time()
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Performance test"}],
    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
    max_tokens=10,
    diagnostic_mode=True
)

total_time = time.time() - start_time
local_overhead = total_time - result.execution_time_seconds

print(f"Total time: {total_time:.3f}s")
print(f"Fireworks inference: {result.execution_time_seconds:.3f}s")
print(f"Local overhead: {local_overhead:.3f}s")

# Fireworks should be very fast - alert if slow
if result.execution_time_seconds > 2.0:
    print("⚠️ Unexpectedly slow Fireworks inference - check network or model")
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
    print("⚠️ High budget utilization - consider:")
    print("  • Switch to smaller models (8B instead of 70B)")
    print("  • Use batch processing for 50% savings")
    print("  • Optimize token usage with shorter max_tokens")
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
adapter = GenOpsFireworksAdapter(
    team="debug-team",
    project="troubleshooting",
    debug_mode=True,
    log_level="DEBUG"
)

# Operations will show detailed logs including:
# - Fireworks API calls and responses
# - Cost calculations with model-specific pricing
# - Performance metrics and timing
# - Governance attribute tracking
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Debug test"}],
    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
    verbose=True
)
```

## 🔗 External Resources

### Documentation Hub
- **[🚀 5-Minute Quickstart Guide](../fireworks-quickstart.md)** - Get started immediately with zero-code setup
- **[📚 Complete Examples Suite](../../examples/fireworks/)** - 7+ working examples from basic to enterprise
- **[🧪 Interactive Setup Wizard](../../examples/fireworks/interactive_setup_wizard.py)** - Guided team onboarding
- **[✅ Setup Validation Tool](../../examples/fireworks/setup_validation.py)** - Comprehensive diagnostics
- **[⚡ Performance Optimization](../../examples/fireworks/cost_optimization.py)** - Speed and cost optimization

### Platform Resources
- **[🔥 Fireworks AI Platform](https://fireworks.ai)** - API dashboard, keys, and $1 free credit
- **[🧠 100+ Model Catalog](https://fireworks.ai/models)** - Complete model library with pricing
- **[📖 Fireworks AI Documentation](https://docs.fireworks.ai)** - Official API reference
- **[🛠️ GenOps Documentation](https://docs.genops.ai)** - Full platform documentation
- **[📊 OpenTelemetry Standards](https://opentelemetry.io/docs/)** - Observability specifications

### Community & Support
- **[🏗️ GitHub Repository](https://github.com/KoshiHQ/GenOps-AI)** - Source code, issues, and contributions
- **[💬 GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Community Q&A and feature requests
- **[🤝 Contribution Guide](https://github.com/KoshiHQ/GenOps-AI/blob/main/CONTRIBUTING.md)** - How to contribute and improve integration

## 📈 Success Metrics

After implementing Fireworks AI + GenOps integration, teams typically achieve:

- **⚡ 4x Faster Inference**: Fireattention CUDA kernels provide significant speed advantages
- **💰 Cost Efficiency**: Up to 50% savings with batch processing, competitive per-token pricing
- **📊 Complete Observability**: 100% cost attribution and performance tracking
- **🎯 Intelligent Optimization**: Smart model selection based on task complexity and budget
- **🔍 Enterprise Governance**: Multi-tenant controls, SOC 2/GDPR/HIPAA compliance support
- **🏢 Production Ready**: Circuit breakers, resilience patterns, and comprehensive monitoring

---

*This integration guide provides comprehensive documentation for Fireworks AI + GenOps. For quick setup, see the [5-minute quickstart guide](../fireworks-quickstart.md). For working examples, explore the [examples directory](../../examples/fireworks/).*