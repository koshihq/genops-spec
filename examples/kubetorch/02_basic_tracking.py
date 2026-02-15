"""
Basic Tracking Example - Cost Tracking Patterns

This example demonstrates the core cost tracking patterns:
- GPU cost calculation
- Multi-resource tracking (GPU, storage, network)
- Context manager usage
- Cost aggregation

Time to run: < 1 minute
"""

from genops.providers.kubetorch import (
    calculate_gpu_cost,
    create_compute_cost_context,
    get_cost_aggregator,
    get_pricing_info,
    reset_cost_aggregator,
)

print("=" * 60)
print("GenOps Kubetorch - Basic Tracking Patterns")
print("=" * 60)

# =============================================
# Example 1: Simple GPU Cost Calculation
# =============================================
print("\n1. Simple GPU Cost Calculation")
print("-" * 60)

cost_a100 = calculate_gpu_cost("a100", num_devices=8, duration_seconds=3600)
cost_h100 = calculate_gpu_cost("h100", num_devices=8, duration_seconds=3600)

print(f"8x A100 for 1 hour: ${cost_a100:.2f}")
print(f"8x H100 for 1 hour: ${cost_h100:.2f}")

# =============================================
# Example 2: Pricing Information
# =============================================
print("\n2. GPU Pricing Information")
print("-" * 60)

for gpu_type in ["h100", "a100", "v100", "t4"]:
    info = get_pricing_info(gpu_type)
    print(
        f"{gpu_type.upper():6s}: ${info.cost_per_hour:7.2f}/hr | {info.gpu_memory_gb:3d}GB"
    )

# =============================================
# Example 3: Context Manager for Multi-Resource Tracking
# =============================================
print("\n3. Multi-Resource Cost Tracking")
print("-" * 60)

reset_cost_aggregator()  # Clean slate

# Track a complete training job with GPU, storage, and network costs
with create_compute_cost_context("train-bert-001") as ctx:
    # GPU compute costs
    ctx.add_gpu_cost(
        instance_type="a100",
        gpu_hours=8.0,  # 8 GPUs for 1 hour
        operation_name="training",
    )

    # Checkpoint storage (100GB stored for 24 hours)
    ctx.add_storage_cost(storage_gb_hours=100 * 24, operation_name="checkpoints")

    # Data transfer (50GB)
    ctx.add_network_cost(data_transfer_gb=50, operation_name="data_sync")

# Print cost summary
print(f"Total Cost: ${ctx.summary.total_cost:.2f}")
print("\nCost Breakdown:")
for resource_type, cost in ctx.summary.cost_by_resource_type.items():
    print(f"  {resource_type:8s}: ${cost:7.2f}")

print("\nResource Usage:")
print(f"  GPU Hours: {ctx.summary.total_gpu_hours:.1f}")
print(f"  Storage:   {ctx.summary.total_storage_gb_hours:.0f} GB-hours")
print(f"  Network:   {ctx.summary.total_network_gb:.0f} GB")

# =============================================
# Example 4: Manual Aggregator Usage
# =============================================
print("\n4. Manual Cost Aggregator")
print("-" * 60)

reset_cost_aggregator()
aggregator = get_cost_aggregator()

# Start tracking an operation
aggregator.start_operation_tracking("inference-job-001")

# Add costs
aggregator.add_gpu_cost("inference-job-001", "t4", gpu_hours=1.0)
aggregator.add_network_cost("inference-job-001", data_transfer_gb=10)

# Finalize
summary = aggregator.finalize_operation_tracking("inference-job-001")

print(f"Inference Job Cost: ${summary.total_cost:.2f}")
print(f"GPU Hours: {summary.total_gpu_hours:.1f}")
print(f"Network GB: {summary.total_network_gb:.0f}")

# =============================================
# Example 5: Multiple Concurrent Jobs
# =============================================
print("\n5. Tracking Multiple Jobs Concurrently")
print("-" * 60)

reset_cost_aggregator()
aggregator = get_cost_aggregator()

# Track 3 jobs concurrently
jobs = [
    ("job-1", "a100", 8.0),
    ("job-2", "h100", 4.0),
    ("job-3", "v100", 16.0),
]

for job_id, gpu_type, gpu_hours in jobs:
    aggregator.start_operation_tracking(job_id)
    aggregator.add_gpu_cost(job_id, gpu_type, gpu_hours)

# Finalize all jobs
total_cost = 0
for job_id, gpu_type, gpu_hours in jobs:
    summary = aggregator.finalize_operation_tracking(job_id)
    print(
        f"{job_id}: {gpu_hours:.1f} {gpu_type.upper()} GPU-hours = ${summary.total_cost:.2f}"
    )
    total_cost += summary.total_cost

print(f"\nTotal Cost (All Jobs): ${total_cost:.2f}")

print("\n" + "=" * 60)
print("✅ All examples completed successfully!")
print("=" * 60)
