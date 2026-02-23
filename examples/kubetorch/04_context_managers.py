"""
Context Manager Patterns - Manual Instrumentation

This example demonstrates manual instrumentation patterns using
adapters and context managers for fine-grained control.

Time to run: < 1 minute
"""

import time

from genops.providers.kubetorch import (
    create_compute_cost_context,
    instrument_kubetorch,
    reset_cost_aggregator,
)

print("=" * 60)
print("GenOps Kubetorch - Context Manager Patterns")
print("=" * 60)

# =============================================
# Example 1: Basic Context Manager
# =============================================
print("\n1. Basic Context Manager Usage")
print("-" * 60)

reset_cost_aggregator()

with create_compute_cost_context("simple-job") as ctx:
    # Add GPU costs
    ctx.add_gpu_cost("a100", gpu_hours=4.0)

print(f"Job Cost: ${ctx.summary.total_cost:.2f}")
print(f"GPU Hours: {ctx.summary.total_gpu_hours}")

# =============================================
# Example 2: Multi-Step Operation Tracking
# =============================================
print("\n2. Multi-Step Operation with Context Manager")
print("-" * 60)

reset_cost_aggregator()

with create_compute_cost_context("multi-step-training") as ctx:
    # Step 1: Data preprocessing on CPU
    print("  Step 1: Data preprocessing...")
    ctx.add_compute_cost(
        resource_type="cpu",
        instance_type="cpu",
        quantity=16.0,  # 16 CPU-hours
        operation_name="preprocessing",
    )

    # Step 2: Model training on GPU
    print("  Step 2: Model training...")
    ctx.add_gpu_cost(instance_type="a100", gpu_hours=8.0, operation_name="training")

    # Step 3: Checkpoint storage
    print("  Step 3: Saving checkpoints...")
    ctx.add_storage_cost(
        storage_gb_hours=50 * 24,  # 50GB for 24 hours
        operation_name="checkpoints",
    )

    # Step 4: Model export and upload
    print("  Step 4: Exporting model...")
    ctx.add_network_cost(data_transfer_gb=25, operation_name="model_export")

print(f"\nTotal Training Pipeline Cost: ${ctx.summary.total_cost:.2f}")
print("\nCost Breakdown by Step:")
for operation, cost in ctx.summary.cost_by_operation.items():
    print(f"  {operation:20s}: ${cost:7.2f}")

# =============================================
# Example 3: Adapter-Based Tracking
# =============================================
print("\n3. Adapter-Based Manual Tracking")
print("-" * 60)

# Create adapter with governance attributes
adapter = instrument_kubetorch(
    team="ml-research",
    project="bert-training",
    customer_id="customer-001",
    cost_tracking_enabled=True,
)

# Track compute deployment
result = adapter.track_compute_deployment(
    instance_type="a100",
    num_devices=8,
    workload_type="training",
    duration_seconds=7200,  # 2 hours
    metadata={
        "model": "bert-large",
        "dataset": "wikipedia",
        "batch_size": 64,
        "epochs": 3,
    },
)

print(f"Operation ID: {result['operation_id']}")
print(f"Total Cost: ${result['cost_total']:.2f}")
print(f"GPU Hours: {result['gpu_hours']}")
print(f"Instance Type: {result['instance_type']}")

# =============================================
# Example 4: Nested Context Managers
# =============================================
print("\n4. Nested Context Managers (Phased Training)")
print("-" * 60)

reset_cost_aggregator()

# Outer context: Full training run
with create_compute_cost_context("full-training-run") as outer_ctx:
    print("  Starting full training run...")

    # Phase 1: Warmup
    with create_compute_cost_context("phase-1-warmup") as phase1_ctx:
        print("    Phase 1: Warmup (2 GPUs)")
        phase1_ctx.add_gpu_cost("a100", gpu_hours=2.0, operation_name="warmup")

    print(f"    Phase 1 Cost: ${phase1_ctx.summary.total_cost:.2f}")

    # Phase 2: Full training
    with create_compute_cost_context("phase-2-training") as phase2_ctx:
        print("    Phase 2: Full Training (8 GPUs)")
        phase2_ctx.add_gpu_cost("a100", gpu_hours=16.0, operation_name="full_training")

    print(f"    Phase 2 Cost: ${phase2_ctx.summary.total_cost:.2f}")

    # Phase 3: Fine-tuning
    with create_compute_cost_context("phase-3-finetune") as phase3_ctx:
        print("    Phase 3: Fine-tuning (4 GPUs)")
        phase3_ctx.add_gpu_cost("a100", gpu_hours=4.0, operation_name="fine_tuning")

    print(f"    Phase 3 Cost: ${phase3_ctx.summary.total_cost:.2f}")

    # Aggregate all phases in outer context
    total_gpu_hours = (
        phase1_ctx.summary.total_gpu_hours
        + phase2_ctx.summary.total_gpu_hours
        + phase3_ctx.summary.total_gpu_hours
    )
    outer_ctx.add_gpu_cost(
        "a100", gpu_hours=total_gpu_hours, operation_name="aggregate"
    )

print(f"\n  Total Training Cost (All Phases): ${outer_ctx.summary.total_cost:.2f}")
print(f"  Total GPU Hours: {outer_ctx.summary.total_gpu_hours}")

# =============================================
# Example 5: Exception Handling in Context
# =============================================
print("\n5. Context Manager with Exception Handling")
print("-" * 60)

reset_cost_aggregator()

try:
    with create_compute_cost_context("job-with-error") as ctx:
        # Add some costs
        ctx.add_gpu_cost("a100", gpu_hours=2.0)

        # Simulate an error
        print("  Simulating error during training...")
        raise ValueError("Training failed - out of memory")

except ValueError as e:
    print(f"  ✗ Error caught: {e}")

# Context manager still finalized costs
print(f"  Cost tracked even with error: ${ctx.summary.total_cost:.2f}")

# =============================================
# Example 6: Real-Time Cost Monitoring
# =============================================
print("\n6. Real-Time Cost Monitoring During Operation")
print("-" * 60)

reset_cost_aggregator()

with create_compute_cost_context("monitored-job") as ctx:
    # Simulate training loop with periodic cost checks
    for step in range(5):
        # Simulate training step
        print(f"  Step {step + 1}/5: Training...")
        time.sleep(0.1)  # Simulate work

        # Add incremental costs
        ctx.add_gpu_cost("a100", gpu_hours=1.0, operation_name=f"step-{step + 1}")

        # Check current cost
        current_cost = ctx.summary.total_cost if ctx.summary else 0
        print(f"    Current Total: ${current_cost:.2f}")

print(f"\n  Final Total Cost: ${ctx.summary.total_cost:.2f}")

print("\n" + "=" * 60)
print("✅ All context manager examples completed!")
print("=" * 60)
