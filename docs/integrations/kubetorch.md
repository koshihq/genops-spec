# Kubetorch Integration - Comprehensive Guide

**Complete reference for integrating GenOps governance with Kubetorch compute infrastructure.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Usage Patterns](#usage-patterns)
- [Cost Tracking](#cost-tracking)
- [Distributed Training](#distributed-training)
- [Production Deployment](#production-deployment)
- [Observability Integration](#observability-integration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Overview

### What is Kubetorch?

Kubetorch (via [run.house](https://www.run.house/)) transforms Kubernetes into a dynamic compute substrate for ML workloads, providing:

- **Dynamic Resource Allocation** - `.to(compute)` for flexible GPU placement
- **Auto-Scaling** - `.autoscale()` for dynamic worker management
- **Fault Recovery** - Automatic retry and migration for failed operations
- **Distributed Training** - `.distribute()` for multi-GPU/multi-node training

### GenOps Governance for Kubetorch

GenOps extends Kubetorch with governance capabilities:

- **GPU Hour Tracking** - Automatic tracking of all compute resource usage
- **Multi-Resource Cost Attribution** - GPU, CPU, storage, network cost aggregation
- **Team/Project/Customer Attribution** - Fine-grained cost allocation
- **OpenTelemetry Telemetry** - Standards-based observability integration
- **Budget Enforcement** - (Phase 2) Real-time cost constraints

### Integration Approach

GenOps provides **three instrumentation patterns**:

1. **Zero-Code Auto-Instrumentation** - Global hooks with no code changes
2. **Manual Adapter Instrumentation** - Explicit tracking with full control
3. **Cost Estimation Only** - Offline cost analysis without runtime tracking

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Your Kubetorch/PyTorch Training Code)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│              GenOps Kubetorch Provider                      │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐                │
│  │  Registration   │  │  Compute Monitor │                │
│  │  (Auto-Instr.)  │  │  (Hooks)         │                │
│  └────────┬────────┘  └────────┬─────────┘                │
│           │                     │                           │
│  ┌────────┴─────────────────────┴─────────┐                │
│  │     GenOpsKubetorchAdapter            │                │
│  │  (BaseFrameworkProvider)              │                │
│  └────────┬──────────────────────────────┘                │
│           │                                                 │
│  ┌────────┴─────────────┐  ┌──────────────────┐          │
│  │  Cost Aggregator     │  │  GPU Pricing DB  │          │
│  │  (Multi-Resource)    │  │  (A100/H100/etc) │          │
│  └──────────────────────┘  └──────────────────┘          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│              OpenTelemetry SDK                              │
│  (Spans, Traces, Semantic Conventions)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│         Observability Backend                               │
│  (Jaeger, Tempo, Datadog, Honeycomb, etc.)                 │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/genops/providers/kubetorch/
├── __init__.py              # Public API exports
├── pricing.py               # GPU pricing database (A100, H100, V100, A10G, T4)
├── adapter.py               # Main adapter (BaseFrameworkProvider)
├── cost_aggregator.py       # Multi-resource cost tracking
├── compute_monitor.py       # Framework-specific instrumentation
├── registration.py          # Auto-instrumentation hooks
└── validation.py            # Setup validation utilities
```

### Semantic Conventions

These attributes are automatically added to OpenTelemetry spans by GenOps. They appear as span attributes, not configuration files.

**In practice, you'll see them as:**
```python
# GenOps automatically adds these to your spans:
span.set_attribute("genops.compute.provider", "kubetorch")
span.set_attribute("genops.compute.gpu_type", "a100")
span.set_attribute("genops.cost.total", 262.16)
```

**Reference specification (YAML format for documentation):**

GenOps extends OpenTelemetry with Kubetorch-specific attributes:

```yaml
# Compute Provider Identification
genops.compute.provider: "kubetorch"
genops.compute.framework: "kubetorch"
genops.compute.resource_type: "gpu|cpu|tpu"
genops.compute.instance_type: "a100|h100|v100|a10g|t4"
genops.compute.num_devices: 8
genops.compute.duration_seconds: 3600
genops.compute.gpu_hours: 8.0

# Workload Classification
genops.workload.type: "training|fine-tuning|inference"
genops.workload.framework: "pytorch|tensorflow|jax"
genops.workload.job_id: "train-bert-001"

# Cost Attribution
genops.cost.compute: 262.16
genops.cost.storage: 12.50
genops.cost.network: 2.34
genops.cost.total: 277.00
genops.cost.rate_per_gpu_hour: 32.77
genops.cost.currency: "USD"

# Governance Attributes
genops.team: "ml-research"
genops.project: "llm-training"
genops.customer_id: "customer-123"
genops.environment: "production"
genops.cost_center: "ml-infrastructure"
```

### Understanding Governance Attributes

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

---

## Installation & Setup

### Basic Installation

```bash
pip install genops-ai
```

### Optional Dependencies

```bash
# Kubetorch/Runhouse (for framework monitoring)
pip install runhouse

# PyTorch (for GPU detection)
pip install torch

# OpenTelemetry exporters
pip install opentelemetry-exporter-otlp
```

### Environment Configuration

```bash
# OpenTelemetry endpoint (required for telemetry export)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Optional: Governance defaults
export GENOPS_TEAM="ml-research"
export GENOPS_PROJECT="llm-training"
export GENOPS_ENVIRONMENT="production"
export GENOPS_COST_CENTER="ml-infrastructure"
```

### Validation

Run comprehensive validation to verify setup:

```python
from genops.providers.kubetorch import validate_kubetorch_setup, print_validation_result

result = validate_kubetorch_setup()
print_validation_result(result, show_all=True)
```

---

## Usage Patterns

### Which Pattern Should I Use?

Choose your instrumentation pattern based on your use case:

```
┌─────────────────────────────────────────────────────────────┐
│  Need to track costs in existing Kubetorch code?           │
│  → Pattern 1: Zero-Code Auto-Instrumentation               │
│     Just add: auto_instrument_kubetorch(team="...")         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Need fine-grained control over specific operations?       │
│  → Pattern 2: Manual Instrumentation with Adapters         │
│     Use: instrument_kubetorch() for granular tracking      │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Want to track multi-step workflows with cleanup?          │
│  → Pattern 3: Context Managers                             │
│     Use: with create_compute_cost_context() as ctx:        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Just estimating costs before running code?                │
│  → Pattern 4: Cost Estimation                              │
│     Use: calculate_gpu_cost()                              │
└─────────────────────────────────────────────────────────────┘
```

**Quick Examples:**
- **Pattern 1:** "I have existing Kubetorch training code" → Auto-instrumentation
- **Pattern 2:** "I need per-operation cost tracking" → Manual adapters
- **Pattern 3:** "I have multi-phase training pipelines" → Context managers
- **Pattern 4:** "I'm budgeting before implementation" → Cost estimation

---

### Pattern 1: Zero-Code Auto-Instrumentation

**Best for:** Quick setup, existing applications, minimal code changes

```python
from genops.providers.kubetorch import auto_instrument_kubetorch

# Enable global instrumentation
auto_instrument_kubetorch(
    team="ml-research",
    project="llm-training",
    customer_id="customer-123",
    environment="production",
)

# Your existing Kubetorch code now tracked automatically
# No further changes required!
```

**Cleanup:**

```python
from genops.providers.kubetorch import uninstrument_kubetorch

uninstrument_kubetorch()  # Remove instrumentation
```

### Pattern 2: Manual Adapter Instrumentation

**Best for:** Fine-grained control, per-operation attribution, custom metadata

```python
from genops.providers.kubetorch import instrument_kubetorch

# Create adapter
adapter = instrument_kubetorch(
    team="ml-research",
    project="llm-training",
    cost_tracking_enabled=True,
    debug=False
)

# Track specific operations
result = adapter.track_compute_deployment(
    instance_type="a100",
    num_devices=8,
    workload_type="training",
    duration_seconds=3600,
    metadata={
        "model": "bert-large",
        "dataset": "wikipedia",
        "batch_size": 64,
        "learning_rate": 1e-4
    }
)

print(f"Operation ID: {result['operation_id']}")
print(f"Total Cost: ${result['cost_total']:.2f}")
print(f"GPU Hours: {result['gpu_hours']}")
```

### Pattern 3: Context Manager for Operations

**Best for:** Scoped tracking, automatic cleanup, multi-resource operations

```python
from genops.providers.kubetorch import create_compute_cost_context

with create_compute_cost_context("train-bert-001") as ctx:
    # Track GPU usage
    ctx.add_gpu_cost(
        instance_type="a100",
        gpu_hours=8.0,
        operation_name="training"
    )

    # Track checkpoint storage
    ctx.add_storage_cost(
        storage_gb_hours=100 * 24,  # 100GB for 24 hours
        operation_name="checkpoints"
    )

    # Track data transfer
    ctx.add_network_cost(
        data_transfer_gb=50,
        operation_name="data_sync"
    )

# Automatic finalization and cost summary
print(f"Total Cost: ${ctx.summary.total_cost:.2f}")
print(f"GPU Hours: {ctx.summary.total_gpu_hours}")
print(f"Cost Breakdown: {ctx.summary.cost_by_resource_type}")
```

### Pattern 4: Cost Estimation Only

**Best for:** Offline analysis, budget planning, cost forecasting

```python
from genops.providers.kubetorch import calculate_gpu_cost, get_pricing_info
from genops.providers.kubetorch.pricing import KubetorchPricing

# Quick cost calculation
cost = calculate_gpu_cost("a100", num_devices=8, duration_seconds=3600)
print(f"Training cost: ${cost:.2f}")

# Get pricing information
info = get_pricing_info("h100")
print(f"H100: ${info.cost_per_hour:.2f}/hr")

# Estimate complete training cost
pricing = KubetorchPricing()
estimate = pricing.estimate_training_cost(
    instance_type="a100",
    num_devices=8,
    estimated_hours=24,
    checkpoint_size_gb=25.6,
    checkpoint_frequency_hours=2.0,
    data_transfer_gb=100
)

print(f"Total Training Cost: ${estimate['cost_total']:.2f}")
print(f"  Compute: ${estimate['cost_compute']:.2f}")
print(f"  Storage: ${estimate['cost_storage']:.2f}")
print(f"  Network: ${estimate['cost_network']:.2f}")
```

---

## Cost Tracking

### GPU Pricing Database

GenOps includes pricing for major GPU types (AWS EC2 baseline, January 2026):

| GPU Type | Instance | Memory | Cost/Hour | Best For |
|----------|----------|--------|-----------|----------|
| H100 | p5.48xlarge/8 | 80GB | $98.32 | Large-scale training, LLM inference |
| A100 (80GB) | p4de.24xlarge/8 | 80GB | $40.96 | Training large models, fine-tuning |
| A100 (40GB) | p4d.24xlarge/8 | 40GB | $32.77 | General training, fine-tuning |
| V100 | p3.16xlarge/8 | 16GB | $12.24 | Training, general compute |
| A10G | g5.48xlarge/8 | 24GB | $5.22 | Inference, light training |
| T4 | g4dn.12xlarge/4 | 16GB | $1.88 | Inference, development |

### Multi-Resource Cost Aggregation

Track costs across all resource types:

```python
from genops.providers.kubetorch import get_cost_aggregator, reset_cost_aggregator

# Get global aggregator
aggregator = get_cost_aggregator()

# Start tracking operation
aggregator.start_operation_tracking("train-job-001")

# Add GPU costs
aggregator.add_gpu_cost("train-job-001", "a100", gpu_hours=8.0)

# Add CPU costs (for data preprocessing)
aggregator.add_compute_cost(
    "train-job-001",
    resource_type="cpu",
    instance_type="cpu",
    quantity=32.0,  # 32 CPU-hours
)

# Add storage costs
aggregator.add_storage_cost("train-job-001", storage_gb_hours=2400.0)

# Add network costs
aggregator.add_network_cost("train-job-001", data_transfer_gb=50.0)

# Finalize and get summary
summary = aggregator.finalize_operation_tracking("train-job-001")

print(f"Total Cost: ${summary.total_cost:.2f}")
print(f"\nCost Breakdown:")
for resource, cost in summary.cost_by_resource_type.items():
    print(f"  {resource}: ${cost:.2f}")
```

### Cost Attribution Strategies

**Team-Level Attribution:**

```python
auto_instrument_kubetorch(team="ml-research")
# All costs tagged with team="ml-research"
```

**Project-Level Attribution:**

```python
auto_instrument_kubetorch(
    team="ml-research",
    project="llm-training"
)
# Costs tracked per project
```

**Customer-Level Attribution (Multi-Tenant):**

```python
# Per-customer tracking
adapter = instrument_kubetorch(
    team="platform-team",
    customer_id=customer_id,  # Dynamic per request
)
```

**Per-Operation Attribution:**

```python
with create_compute_cost_context(f"train-{user_id}") as ctx:
    ctx.add_gpu_cost("a100", 8.0, operation_name=f"user-{user_id}-training")
```

---

## Distributed Training

### Multi-GPU Training Tracking

```python
from genops.providers.kubetorch import instrument_kubetorch

adapter = instrument_kubetorch(team="ml-research", project="bert-training")

# Track distributed training job
result = adapter.track_compute_deployment(
    instance_type="a100",
    num_devices=64,  # 8 nodes × 8 GPUs
    workload_type="training",
    duration_seconds=7200,  # 2 hours
    metadata={
        "distributed_strategy": "ddp",
        "num_nodes": 8,
        "gpus_per_node": 8,
        "model": "bert-large",
        "global_batch_size": 512,
    }
)

print(f"Total GPU Hours: {result['gpu_hours']}")  # 128 GPU-hours
print(f"Total Cost: ${result['cost_total']:.2f}")  # $4,194.56
```

### Multi-Node Cost Aggregation

```python
from genops.providers.kubetorch import create_compute_cost_context

# Track multi-node distributed job
with create_compute_cost_context("distributed-training-001") as ctx:
    # Node 1-8: Primary training
    for node_id in range(8):
        ctx.add_gpu_cost(
            instance_type="a100",
            gpu_hours=8.0,  # 8 GPUs × 1 hour
            operation_name=f"node-{node_id}-training"
        )

    # Checkpoint storage across all nodes
    ctx.add_storage_cost(
        storage_gb_hours=200 * 24,  # 200GB × 24 hours
        operation_name="distributed-checkpoints"
    )

    # Inter-node communication
    ctx.add_network_cost(
        data_transfer_gb=500,  # 500GB gradient sync
        operation_name="allreduce-communication"
    )

print(f"Total Cost: ${ctx.summary.total_cost:.2f}")
print(f"Cost per GPU: ${ctx.summary.total_cost / 64:.2f}")
```

---

## Production Deployment

### Kubernetes Integration

**Deployment Configuration:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-config
data:
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4317"
  GENOPS_TEAM: "ml-research"
  GENOPS_PROJECT: "llm-training"
  GENOPS_ENVIRONMENT: "production"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: your-training-image:latest
        envFrom:
        - configMapRef:
            name: genops-config
        resources:
          limits:
            nvidia.com/gpu: 8
```

**Application Code:**

```python
from genops.providers.kubetorch import auto_instrument_kubetorch
import os

# Auto-configure from environment
auto_instrument_kubetorch(
    team=os.getenv("GENOPS_TEAM"),
    project=os.getenv("GENOPS_PROJECT"),
    environment=os.getenv("GENOPS_ENVIRONMENT"),
)

# Your training code here
```

### High-Availability Setup

```python
from genops.providers.kubetorch import instrument_kubetorch

# Configure with retry and circuit breaker
adapter = instrument_kubetorch(
    team="ml-research",
    enable_retry=True,
    max_retries=3,
    telemetry_enabled=True,
    cost_tracking_enabled=True,
)

# Adapter handles transient failures automatically
```

### Sampling for High-Volume Workloads

```python
from genops.providers.kubetorch import auto_instrument_kubetorch

# Enable sampling for high-throughput scenarios
auto_instrument_kubetorch(
    team="ml-inference",
    project="serving",
    # Only track 10% of operations (sampling handled by OTel SDK)
)
```

---

## Observability Integration

### Datadog Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.datadog import DatadogSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from genops.providers.kubetorch import auto_instrument_kubetorch

# Configure Datadog exporter
trace.set_tracer_provider(TracerProvider())
exporter = DatadogSpanExporter(
    agent_url="http://datadog-agent:8126",
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(exporter)
)

# Enable GenOps tracking
auto_instrument_kubetorch(team="ml-research")
```

### Grafana/Tempo Integration

```bash
# Configure OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://tempo:4317"
```

```python
from genops.providers.kubetorch import auto_instrument_kubetorch

# Telemetry automatically exported to Tempo
auto_instrument_kubetorch(team="ml-research")
```

### Prometheus Metrics

```python
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider

from genops.providers.kubetorch import instrument_kubetorch

# Configure Prometheus exporter
reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))

# Track costs as metrics
adapter = instrument_kubetorch(team="ml-research")
```

---

## Performance Optimization

### Telemetry Overhead

GenOps is designed for minimal overhead:

- **Instrumentation:** < 1% of operation time
- **Memory:** < 50MB for typical workloads
- **Telemetry Export:** Asynchronous, non-blocking

### Optimization Techniques

**1. Batch Operations:**

```python
from genops.providers.kubetorch import get_cost_aggregator

aggregator = get_cost_aggregator()

# Track multiple operations efficiently
for i in range(100):
    aggregator.start_operation_tracking(f"job-{i}")
    aggregator.add_gpu_cost(f"job-{i}", "a100", 1.0)
    aggregator.finalize_operation_tracking(f"job-{i}")
```

**2. Disable Telemetry for Development:**

```python
from genops.providers.kubetorch import instrument_kubetorch

# Disable telemetry for local development
adapter = instrument_kubetorch(
    team="ml-research",
    telemetry_enabled=False,  # No telemetry overhead
    cost_tracking_enabled=True,  # Still calculate costs
)
```

**3. Sampling Configuration:**

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

# Configure 10% sampling
sampler = TraceIdRatioBased(0.1)
```

---

## Troubleshooting

### Common Issues

#### Issue: "Kubetorch (runhouse) not installed"

**Cause:** Kubetorch is not installed, but framework monitoring attempted.

**Solution:**

```bash
# Option 1: Install Kubetorch
pip install runhouse

# Option 2: Disable monitoring (cost estimation still works)
auto_instrument_kubetorch(enable_monitoring=False)
```

#### Issue: "OpenTelemetry TracerProvider not configured"

**Cause:** OpenTelemetry SDK not properly initialized.

**Solution:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)
```

#### Issue: "Operation not found in active tracking"

**Cause:** Operation was finalized or never started.

**Solution:**

```python
from genops.providers.kubetorch import get_cost_aggregator

aggregator = get_cost_aggregator()

# Always start tracking before adding costs
aggregator.start_operation_tracking("job-001")
aggregator.add_gpu_cost("job-001", "a100", 8.0)
aggregator.finalize_operation_tracking("job-001")
```

### Debug Mode

Enable debug logging for detailed diagnostics:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

from genops.providers.kubetorch import instrument_kubetorch

adapter = instrument_kubetorch(team="ml-research", debug=True)
```

### Validation Diagnostics

Run comprehensive validation for troubleshooting:

```python
from genops.providers.kubetorch import validate_kubetorch_setup, print_validation_result

result = validate_kubetorch_setup()
print_validation_result(result, show_all=True, show_details=True)
```

---

## Quick Import Reference

All Kubetorch functions are available from a single import path:

```python
from genops.providers.kubetorch import (
    # Auto-instrumentation (zero-code)
    auto_instrument_kubetorch,
    uninstrument_kubetorch,

    # Manual instrumentation
    instrument_kubetorch,

    # Context managers
    create_compute_cost_context,

    # Cost estimation
    calculate_gpu_cost,

    # Validation
    validate_kubetorch_setup,
    print_validation_result,

    # Cost aggregation
    get_cost_aggregator,
    reset_cost_aggregator,
)
```

**Most Common Imports:**
```python
# For quickstart (zero-code):
from genops.providers.kubetorch import auto_instrument_kubetorch

# For manual tracking:
from genops.providers.kubetorch import instrument_kubetorch

# For context managers:
from genops.providers.kubetorch import create_compute_cost_context
```

---

## API Reference

### Auto-Instrumentation

#### `auto_instrument_kubetorch()`

Enable zero-code global instrumentation.

**Parameters:**
- `team` (str, optional): Team name for governance attribution
- `project` (str, optional): Project name for governance attribution
- `customer_id` (str, optional): Customer ID for billing attribution
- `environment` (str, optional): Environment (dev/staging/prod)
- `cost_center` (str, optional): Cost center for financial reporting
- `enable_monitoring` (bool, default=True): Enable operation monitoring
- `enable_cost_tracking` (bool, default=True): Enable cost aggregation

**Returns:** `bool` - True if instrumentation enabled, False if already enabled

**Example:**
```python
auto_instrument_kubetorch(
    team="ml-research",
    project="llm-training",
    customer_id="customer-123"
)
```

#### `uninstrument_kubetorch()`

Disable and remove auto-instrumentation.

**Returns:** `bool` - True if instrumentation disabled, False if not enabled

#### `is_kubetorch_instrumented()`

Check if auto-instrumentation is active.

**Returns:** `bool` - True if instrumentation active

---

### Manual Instrumentation

#### `instrument_kubetorch()`

Create adapter instance for manual instrumentation.

**Parameters:**
- `kubetorch_client` (Any, optional): Kubetorch client instance
- `telemetry_enabled` (bool, default=True): Enable telemetry emission
- `cost_tracking_enabled` (bool, default=True): Enable cost tracking
- `debug` (bool, default=False): Enable debug logging
- `enable_retry` (bool, default=True): Enable retry logic
- `max_retries` (int, default=3): Maximum retry attempts
- `**governance_defaults`: Governance attributes (team, project, etc.)

**Returns:** `GenOpsKubetorchAdapter`

**Example:**
```python
adapter = instrument_kubetorch(
    team="ml-research",
    cost_tracking_enabled=True,
    debug=False
)
```

#### `GenOpsKubetorchAdapter.track_compute_deployment()`

Track compute deployment operation.

**Parameters:**
- `instance_type` (str): GPU instance type (e.g., "a100")
- `num_devices` (int): Number of devices
- `workload_type` (str): Workload type (training/inference)
- `duration_seconds` (float, optional): Operation duration
- `**kwargs`: Additional metadata

**Returns:** `Dict[str, Any]` - Operation details including cost and operation_id

---

### Cost Tracking

#### `create_compute_cost_context()`

Create context manager for cost tracking.

**Parameters:**
- `operation_id` (str): Unique operation identifier

**Returns:** `ComputeCostContext`

**Example:**
```python
with create_compute_cost_context("job-001") as ctx:
    ctx.add_gpu_cost("a100", 8.0)
```

#### `get_cost_aggregator()`

Get global cost aggregator singleton.

**Returns:** `KubetorchCostAggregator`

#### `reset_cost_aggregator()`

Reset global cost aggregator (mainly for testing).

---

### Pricing

#### `calculate_gpu_cost()`

Calculate GPU cost.

**Parameters:**
- `instance_type` (str): GPU instance type
- `num_devices` (int): Number of devices
- `duration_seconds` (float): Duration in seconds

**Returns:** `float` - Cost in USD

**Example:**
```python
cost = calculate_gpu_cost("a100", num_devices=8, duration_seconds=3600)
# Returns: 262.16
```

#### `get_pricing_info()`

Get pricing information for instance type.

**Parameters:**
- `instance_type` (str): GPU instance type

**Returns:** `GPUInstancePricing` or `None`

**Example:**
```python
info = get_pricing_info("h100")
print(f"H100: ${info.cost_per_hour:.2f}/hr, {info.gpu_memory_gb}GB")
```

---

### Validation

#### `validate_kubetorch_setup()`

Validate Kubetorch integration setup.

**Parameters:**
- `check_kubetorch` (bool, default=True): Check Kubetorch installation
- `check_kubernetes` (bool, default=True): Check Kubernetes environment
- `check_gpu` (bool, default=True): Check GPU availability
- `check_opentelemetry` (bool, default=True): Check OpenTelemetry setup
- `check_genops` (bool, default=True): Check GenOps configuration

**Returns:** `ValidationResult`

**Example:**
```python
result = validate_kubetorch_setup()
if not result.is_valid():
    print(result.summary())
```

#### `print_validation_result()`

Print validation result in user-friendly format.

**Parameters:**
- `result` (ValidationResult): Validation result to print
- `show_all` (bool, default=False): Show all issues (not just errors/warnings)
- `show_details` (bool, default=False): Show detailed information

---

## Additional Resources

- **[Quickstart Guide](../kubetorch-quickstart.md)** - 5-minute setup guide
- **[Examples Directory](../../examples/kubetorch/)** - Working code examples
- **[OpenTelemetry Documentation](https://opentelemetry.io/)** - OTel reference
- **[Kubetorch/Runhouse Docs](https://www.run.house/docs)** - Kubetorch reference

---

**Last Updated:** 2026-01-16
