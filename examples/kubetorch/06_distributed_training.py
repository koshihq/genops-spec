"""
Distributed Training Patterns - Multi-GPU & Multi-Node

This example demonstrates cost tracking for distributed training scenarios:
- Single-node multi-GPU training
- Multi-node distributed training
- Data-parallel training
- Model-parallel training
- Gradient accumulation cost optimization

Time to run: < 1 minute
"""

from genops.providers.kubetorch import (
    instrument_kubetorch,
    create_compute_cost_context,
    reset_cost_aggregator,
    get_cost_aggregator,
)

print("=" * 60)
print("GenOps Kubetorch - Distributed Training Patterns")
print("=" * 60)

# =============================================
# Example 1: Single-Node Multi-GPU Training
# =============================================
print("\n1. Single-Node Multi-GPU Training (8x A100)")
print("-" * 60)

adapter = instrument_kubetorch(
    team="ml-research",
    project="llm-training",
)

# Track single-node 8-GPU training
result = adapter.track_compute_deployment(
    instance_type="a100",
    num_devices=8,
    workload_type="training",
    duration_seconds=7200,  # 2 hours
    metadata={
        "distributed_strategy": "ddp",  # Data Parallel
        "num_nodes": 1,
        "gpus_per_node": 8,
        "model": "bert-large",
        "global_batch_size": 256,
    }
)

print(f"Configuration:")
print(f"  Nodes: 1 × 8 GPUs")
print(f"  Strategy: Data Parallel (DDP)")
print(f"  Duration: 2 hours")
print(f"\nCosts:")
print(f"  Total GPU Hours: {result['gpu_hours']}")
print(f"  Total Cost: ${result['cost_total']:.2f}")
print(f"  Cost per GPU: ${result['cost_total'] / 8:.2f}")

# =============================================
# Example 2: Multi-Node Distributed Training
# =============================================
print("\n2. Multi-Node Distributed Training (4 nodes × 8 GPUs)")
print("-" * 60)

reset_cost_aggregator()

# Track 4-node distributed training
num_nodes = 4
gpus_per_node = 8
total_gpus = num_nodes * gpus_per_node

with create_compute_cost_context("multi-node-training") as ctx:
    # Track GPU costs for all nodes
    ctx.add_gpu_cost(
        instance_type="a100",
        gpu_hours=total_gpus * 2.0,  # 32 GPUs for 2 hours = 64 GPU-hours
        operation_name="distributed_training"
    )

    # Track inter-node network communication
    # Estimate: 1GB per GPU per epoch, 10 epochs, 4-way allreduce
    network_gb = total_gpus * 1 * 10 * 4
    ctx.add_network_cost(
        data_transfer_gb=network_gb,
        operation_name="gradient_sync"
    )

    # Track distributed checkpoint storage
    # Checkpoint every 2 hours, 50GB per checkpoint
    ctx.add_storage_cost(
        storage_gb_hours=50 * 24,  # 50GB for 24 hours
        operation_name="distributed_checkpoints"
    )

print(f"Configuration:")
print(f"  Nodes: {num_nodes} × {gpus_per_node} GPUs = {total_gpus} total GPUs")
print(f"  Strategy: Distributed Data Parallel")
print(f"  Duration: 2 hours")
print(f"\nCosts:")
print(f"  Compute: ${ctx.summary.cost_by_resource_type.get('gpu', 0):.2f}")
print(f"  Network: ${ctx.summary.cost_by_resource_type.get('network', 0):.2f}")
print(f"  Storage: ${ctx.summary.cost_by_resource_type.get('storage', 0):.2f}")
print(f"  Total: ${ctx.summary.total_cost:.2f}")
print(f"  Cost per GPU-hour: ${ctx.summary.total_cost / (total_gpus * 2):.2f}")

# =============================================
# Example 3: Model-Parallel Training (Large Models)
# =============================================
print("\n3. Model-Parallel Training (Large LLM)")
print("-" * 60)

reset_cost_aggregator()

# Track model-parallel training for very large model
num_nodes = 8
gpus_per_node = 8
total_gpus = 64
training_hours = 10

with create_compute_cost_context("model-parallel-llm") as ctx:
    # GPU compute for model-parallel training
    ctx.add_gpu_cost(
        instance_type="h100",  # H100 for large models
        gpu_hours=total_gpus * training_hours,
        operation_name="model_parallel_training"
    )

    # High network overhead for model parallelism
    # ~10GB per GPU per hour for pipeline and tensor parallelism
    network_gb = total_gpus * 10 * training_hours
    ctx.add_network_cost(
        data_transfer_gb=network_gb,
        operation_name="model_parallel_communication"
    )

    # Large checkpoint storage for 175B parameter model
    # ~350GB per checkpoint, checkpoint every 2 hours
    num_checkpoints = training_hours // 2
    storage_gb_hours = 350 * 24 * num_checkpoints
    ctx.add_storage_cost(
        storage_gb_hours=storage_gb_hours,
        operation_name="large_model_checkpoints"
    )

print(f"Configuration:")
print(f"  Model: 175B parameters")
print(f"  Nodes: {num_nodes} × {gpus_per_node} H100 GPUs")
print(f"  Strategy: Tensor + Pipeline Parallel")
print(f"  Duration: {training_hours} hours")
print(f"\nCosts:")
print(f"  Compute: ${ctx.summary.cost_by_resource_type.get('gpu', 0):.2f}")
print(f"  Network: ${ctx.summary.cost_by_resource_type.get('network', 0):.2f}")
print(f"  Storage: ${ctx.summary.cost_by_resource_type.get('storage', 0):.2f}")
print(f"  Total: ${ctx.summary.total_cost:.2f}")

# =============================================
# Example 4: Gradient Accumulation Cost Optimization
# =============================================
print("\n4. Cost Optimization: Gradient Accumulation")
print("-" * 60)

reset_cost_aggregator()

# Compare two strategies:
# Strategy A: 8 GPUs without gradient accumulation
# Strategy B: 4 GPUs with gradient accumulation (2x)

strategies = [
    {
        "name": "8 GPUs (No Accumulation)",
        "num_gpus": 8,
        "gpu_type": "a100",
        "hours": 4.0,
        "batch_per_gpu": 32,
    },
    {
        "name": "4 GPUs (2x Accumulation)",
        "num_gpus": 4,
        "gpu_type": "a100",
        "hours": 4.5,  # Slightly longer due to accumulation overhead
        "batch_per_gpu": 32,
    },
]

print("Comparing Training Strategies:")
print("-" * 60)

for strategy in strategies:
    with create_compute_cost_context(f"strategy-{strategy['name']}") as ctx:
        ctx.add_gpu_cost(
            instance_type=strategy["gpu_type"],
            gpu_hours=strategy["num_gpus"] * strategy["hours"],
            operation_name="training"
        )

    effective_batch = strategy["num_gpus"] * strategy["batch_per_gpu"]
    if "Accumulation" in strategy["name"]:
        effective_batch *= 2  # 2x accumulation

    print(f"\n  {strategy['name']}:")
    print(f"    GPUs: {strategy['num_gpus']}")
    print(f"    Duration: {strategy['hours']} hours")
    print(f"    Effective Batch: {effective_batch}")
    print(f"    Cost: ${ctx.summary.total_cost:.2f}")
    print(f"    Cost per Sample: ${ctx.summary.total_cost / effective_batch:.4f}")

# =============================================
# Example 5: Heterogeneous Cluster Training
# =============================================
print("\n5. Heterogeneous Cluster (Mixed GPU Types)")
print("-" * 60)

reset_cost_aggregator()
aggregator = get_cost_aggregator()

# Simulate training on a heterogeneous cluster
# Primary training: 4x A100
# Secondary training: 8x V100
# Inference testing: 4x T4

aggregator.start_operation_tracking("heterogeneous-training")

# Primary training nodes (A100)
aggregator.add_gpu_cost(
    "heterogeneous-training",
    "a100",
    gpu_hours=4 * 5.0,  # 4 GPUs for 5 hours
    operation_name="primary_training"
)

# Secondary training nodes (V100)
aggregator.add_gpu_cost(
    "heterogeneous-training",
    "v100",
    gpu_hours=8 * 5.0,  # 8 GPUs for 5 hours
    operation_name="secondary_training"
)

# Inference testing (T4)
aggregator.add_gpu_cost(
    "heterogeneous-training",
    "t4",
    gpu_hours=4 * 2.0,  # 4 GPUs for 2 hours
    operation_name="inference_testing"
)

summary = aggregator.finalize_operation_tracking("heterogeneous-training")

print("Heterogeneous Cluster Configuration:")
print("  Primary:   4 × A100 (5 hours)")
print("  Secondary: 8 × V100 (5 hours)")
print("  Testing:   4 × T4 (2 hours)")
print(f"\nTotal Cost: ${summary.total_cost:.2f}")
print(f"\nCost by Operation:")
for operation, cost in summary.cost_by_operation.items():
    print(f"  {operation:20s}: ${cost:.2f}")

# =============================================
# Example 6: Fault Recovery Cost Tracking
# =============================================
print("\n6. Fault Recovery and Retry Costs")
print("-" * 60)

reset_cost_aggregator()

# Simulate training with retries due to failures
with create_compute_cost_context("training-with-retries") as ctx:
    # Attempt 1: Failed after 1 hour (node failure)
    print("  Attempt 1: Node failure after 1 hour")
    ctx.add_gpu_cost(
        instance_type="a100",
        gpu_hours=8 * 1.0,
        operation_name="attempt_1_failed"
    )

    # Attempt 2: Failed after 0.5 hours (OOM error)
    print("  Attempt 2: OOM error after 0.5 hours")
    ctx.add_gpu_cost(
        instance_type="a100",
        gpu_hours=8 * 0.5,
        operation_name="attempt_2_failed"
    )

    # Attempt 3: Success after 4 hours
    print("  Attempt 3: Success after 4 hours")
    ctx.add_gpu_cost(
        instance_type="a100",
        gpu_hours=8 * 4.0,
        operation_name="attempt_3_success"
    )

print(f"\nTotal Cost (including retries): ${ctx.summary.total_cost:.2f}")
print(f"Wasted Cost (failed attempts): ${ctx.summary.cost_by_operation.get('attempt_1_failed', 0) + ctx.summary.cost_by_operation.get('attempt_2_failed', 0):.2f}")
print(f"Effective Cost (successful): ${ctx.summary.cost_by_operation.get('attempt_3_success', 0):.2f}")
print(f"Overhead from Failures: {((ctx.summary.total_cost / ctx.summary.cost_by_operation.get('attempt_3_success', 1)) - 1) * 100:.1f}%")

# =============================================
# Example 7: Multi-Region Distributed Training
# =============================================
print("\n7. Multi-Region Distributed Training")
print("-" * 60)

reset_cost_aggregator()

regions = [
    ("us-west-2", 4, "a100", 4.0),
    ("us-east-1", 4, "a100", 4.0),
    ("eu-west-1", 2, "a100", 4.0),
]

total_cost = 0

print("Multi-Region Training Configuration:")
for region, num_gpus, gpu_type, hours in regions:
    with create_compute_cost_context(f"region-{region}") as ctx:
        ctx.add_gpu_cost(gpu_type, gpu_hours=num_gpus * hours)

        # Cross-region network costs (significantly higher)
        ctx.add_network_cost(
            data_transfer_gb=num_gpus * 50,  # 50GB per GPU
            operation_name=f"cross_region_{region}"
        )

    print(f"  {region:12s}: {num_gpus} × {gpu_type.upper()} = ${ctx.summary.total_cost:.2f}")
    total_cost += ctx.summary.total_cost

print(f"\n  Total Multi-Region Cost: ${total_cost:.2f}")

print("\n" + "=" * 60)
print("✅ All distributed training examples completed!")
print("=" * 60)
