# Kubetorch Integration - 5-Minute Quickstart

**Goal:** Get GPU cost tracking and governance telemetry working in 5 minutes.

---

## Prerequisites (30 seconds)

- **Python 3.8+** - Required
- **Kubetorch/Runhouse** - Optional (cost estimation works without it)
- **OpenTelemetry endpoint** - Optional (telemetry export needs it)
- **GPU hardware or Kubernetes cluster** - Optional for cost estimation; required for tracking actual operations

**Note:** You can use GenOps for cost estimation even without GPU hardware installed. Actual operation tracking requires GPU compute or Kubernetes environment.

---

## Quick Setup (2 minutes)

### Step 1: Install GenOps

```bash
pip install genops-ai
```

### Step 2: Verify Setup

```python
from genops.providers.kubetorch import validate_kubetorch_setup, print_validation_result

result = validate_kubetorch_setup()
print_validation_result(result)
```

**Expected Output:**
```
✅ Validation passed: 7/14 checks successful
  Total Checks: 14
  ✅ Successful: 7
  ⚠️  Warnings: 3
  ❌ Errors: 0
```

### Step 3: Understanding Governance Attributes

GenOps uses these attributes to track and attribute costs:

- **`team`**: Your ML team name - all costs are tagged with this for team-level reporting
- **`project`**: Specific project within your team for project-level cost tracking
- **`customer_id`**: For multi-tenant platforms, tag costs per customer for accurate billing
- **`environment`**: Segregate costs by environment (dev/staging/production)
- **`cost_center`**: Align with your financial reporting structure

**Example:** If you're on the "ml-research" team working on "llm-training" project:

```python
from genops.providers.kubetorch import auto_instrument_kubetorch

auto_instrument_kubetorch(
    team="ml-research",         # Team-level attribution
    project="llm-training"      # Project-level tracking
)
```

### Step 4: Enable Auto-Instrumentation (Zero-Code Setup)

```python
from genops.providers.kubetorch import auto_instrument_kubetorch

# Enable governance tracking globally
auto_instrument_kubetorch(
    team="ml-research",
    project="llm-training",
    environment="production"
)

# Your Kubetorch code now tracked automatically!
```

---

## What Just Happened? ✅

After running `auto_instrument_kubetorch()`:

- ✅ **GPU Hour Tracking** - All compute operations automatically tracked
- ✅ **Cost Attribution** - Costs tagged with team/project/customer
- ✅ **OpenTelemetry Traces** - Governance telemetry exported to your OTLP endpoint
- ✅ **Multi-Resource Tracking** - GPU, CPU, storage, network costs aggregated
- ✅ **Zero Code Changes** - Works with existing Kubetorch applications

---

## Basic Usage Examples (2 minutes)

### Example 1: Cost Estimation (No Kubetorch Required)

```python
from genops.providers.kubetorch import calculate_gpu_cost, get_pricing_info

# Calculate training cost
# A100 cost: $32.77/hour × 8 GPUs × 1 hour = $262.16
# (Based on AWS on-demand pricing, January 2026)
cost = calculate_gpu_cost(
    instance_type="a100",
    num_devices=8,
    duration_seconds=3600  # 1 hour
)
print(f"Training cost: ${cost:.2f}")  # $262.16
print(f"Cost per GPU: ${cost / 8:.2f}")  # $32.77

# Get pricing information
info = get_pricing_info("h100")
print(f"H100: ${info.cost_per_hour:.2f}/hr, {info.gpu_memory_gb}GB")
```

### Example 2: Manual Cost Tracking

```python
from genops.providers.kubetorch import create_compute_cost_context

# Track a training job
with create_compute_cost_context("train-bert-001") as ctx:
    # Add GPU costs
    ctx.add_gpu_cost("a100", gpu_hours=8.0, operation_name="training")

    # Add storage costs (checkpoints)
    ctx.add_storage_cost(storage_gb_hours=100 * 24, operation_name="checkpoints")

    # Add network costs (data transfer)
    ctx.add_network_cost(data_transfer_gb=50, operation_name="data_sync")

# Automatic cost summary
# Available summary attributes: total_cost, total_gpu_hours,
# cost_by_resource_type, cost_by_gpu_type, cost_by_operation
print(f"Total Cost: ${ctx.summary.total_cost:.2f}")
print(f"GPU Hours: {ctx.summary.total_gpu_hours}")
print(f"Cost Breakdown: {ctx.summary.cost_by_resource_type}")  # Optional: show breakdown
```

### Example 3: Adapter-Based Tracking

```python
from genops.providers.kubetorch import instrument_kubetorch

# Create adapter with governance attributes
adapter = instrument_kubetorch(
    team="ml-research",
    project="llm-training",
    customer_id="customer-123"
)

# Track compute deployment
result = adapter.track_compute_deployment(
    instance_type="a100",
    num_devices=8,
    workload_type="training",
    duration_seconds=3600,
    metadata={"model": "bert-large", "batch_size": 64}
)

print(f"Operation ID: {result['operation_id']}")
print(f"Total Cost: ${result['cost_total']:.2f}")
print(f"GPU Hours: {result['gpu_hours']}")
```

---

## Supported GPU Types

GenOps includes pricing for:

| GPU Type | Memory | Cost/Hour | Use Case |
|----------|--------|-----------|----------|
| **H100** | 80GB | $98.32 | Large-scale training, inference |
| **A100** | 40GB/80GB | $32.77/$40.96 | Training, fine-tuning |
| **V100** | 16GB | $12.24 | Training, general compute |
| **A10G** | 24GB | $5.22 | Inference, light training |
| **T4** | 16GB | $1.88 | Inference, development |

*Pricing based on AWS EC2 instances (January 2026)*

---

## Configuration Options

### Environment Variables

```bash
# OpenTelemetry endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Governance defaults (optional)
export GENOPS_TEAM="ml-research"
export GENOPS_PROJECT="llm-training"
export GENOPS_ENVIRONMENT="production"
```

### Programmatic Configuration

```python
from genops.providers.kubetorch import auto_instrument_kubetorch

auto_instrument_kubetorch(
    # Governance attribution
    team="ml-research",
    project="llm-training",
    customer_id="customer-123",
    environment="production",
    cost_center="ml-infrastructure",

    # Feature toggles
    enable_monitoring=True,      # Enable operation monitoring
    enable_cost_tracking=True,   # Enable cost aggregation
)
```

---

## Telemetry Output

### Semantic Conventions

All Kubetorch operations emit OpenTelemetry spans with these attributes:

```python
{
    # Compute identification
    "genops.compute.provider": "kubetorch",
    "genops.compute.instance_type": "a100",
    "genops.compute.num_devices": 8,
    "genops.compute.gpu_hours": 8.0,

    # Cost attribution
    "genops.cost.compute": 262.16,
    "genops.cost.storage": 12.50,
    "genops.cost.network": 2.34,
    "genops.cost.total": 277.00,

    # Governance attributes
    "genops.team": "ml-research",
    "genops.project": "llm-training",
    "genops.customer_id": "customer-123",

    # Workload classification
    "genops.workload.type": "training",
    "genops.workload.framework": "pytorch"
}
```

---

## Troubleshooting

### Common Issues

**Issue:** "Runhouse (Kubetorch) not installed"

```bash
# Kubetorch is optional for cost estimation
# Install only if you need framework monitoring
pip install runhouse
```

**Issue:** "OpenTelemetry TracerProvider not configured"

```bash
# Configure OTLP exporter
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Or use auto-instrumentation
pip install opentelemetry-instrumentation
```

**Issue:** "No GenOps environment variables set"

This is informational - you can pass governance attributes directly:

```python
auto_instrument_kubetorch(team="your-team", project="your-project")
```

### Validation

Run comprehensive validation anytime:

```python
from genops.providers.kubetorch import validate_kubetorch_setup, print_validation_result

result = validate_kubetorch_setup()
print_validation_result(result, show_all=True, show_details=True)
```

### Understanding Validation Results

**Validation shows warnings (⚠️) or partial success?** This is normal!

GenOps is designed to work with or without optional dependencies:

**Expected Warning Scenarios:**
- ⚠️ "Kubetorch/Runhouse not installed" → Cost estimation still works! You can calculate costs without framework installed.
- ⚠️ "No OTEL_EXPORTER_OTLP_ENDPOINT configured" → Spans are created locally; telemetry export is optional for development.
- ⚠️ "No GenOps environment variables set" → Pass governance attributes as function arguments instead.

**What's the difference?**
- ✅ **Success**: Feature fully functional
- ⚠️ **Warning**: Feature works with reduced capabilities (e.g., cost estimation only, no live tracking)
- ❌ **Error**: Feature blocked (requires fix)

**Only actual errors (❌) prevent functionality. Warnings mean graceful degradation, not failure.**

**Example validation output:**
```
✅ Validation passed: 7/14 checks successful
  Total Checks: 14
  ✅ Successful: 7
  ⚠️  Warnings: 3    ← This is OK! Not a failure.
  ❌ Errors: 0       ← No errors means you're ready to go.
```

---

## Next Steps

- **[Comprehensive Guide](integrations/kubetorch.md)** - Complete documentation with advanced patterns
- **[Examples](../examples/kubetorch/)** - Working examples for all use cases
- **[API Reference](integrations/kubetorch.md#api-reference)** - Complete API documentation

---

## Quick Reference

### Import Paths

```python
# Auto-instrumentation
from genops.providers.kubetorch import auto_instrument_kubetorch, uninstrument_kubetorch

# Manual instrumentation
from genops.providers.kubetorch import instrument_kubetorch

# Cost tracking
from genops.providers.kubetorch import create_compute_cost_context, get_cost_aggregator

# Pricing
from genops.providers.kubetorch import calculate_gpu_cost, get_pricing_info

# Validation
from genops.providers.kubetorch import validate_kubetorch_setup, print_validation_result
```

### Minimal Working Example

```python
from genops.providers.kubetorch import auto_instrument_kubetorch, create_compute_cost_context

# Enable tracking
auto_instrument_kubetorch(team="ml-team")

# Track operation
with create_compute_cost_context("train-001") as ctx:
    ctx.add_gpu_cost("a100", gpu_hours=8.0)

print(f"Cost: ${ctx.summary.total_cost:.2f}")
```

---

**Time to Value: < 5 minutes** ⏱️

You're now tracking GPU costs and emitting governance telemetry! 🚀
