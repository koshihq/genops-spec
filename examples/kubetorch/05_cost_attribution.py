"""
Cost Attribution Patterns - Team, Project, Customer Tracking

This example demonstrates cost attribution strategies for:
- Team-level tracking
- Project-level tracking
- Customer/tenant tracking
- Per-user attribution

Time to run: < 1 minute
"""

from genops.providers.kubetorch import (
    auto_instrument_kubetorch,
    uninstrument_kubetorch,
    instrument_kubetorch,
    create_compute_cost_context,
    reset_cost_aggregator,
)

print("=" * 60)
print("GenOps Kubetorch - Cost Attribution Patterns")
print("=" * 60)

# =============================================
# Example 1: Team-Level Attribution
# =============================================
print("\n1. Team-Level Cost Attribution")
print("-" * 60)

# Enable tracking for ml-research team
auto_instrument_kubetorch(team="ml-research")

# All operations now tagged with team="ml-research"
reset_cost_aggregator()

with create_compute_cost_context("team-training-job") as ctx:
    ctx.add_gpu_cost("a100", gpu_hours=8.0)

print(f"ML Research Team Cost: ${ctx.summary.total_cost:.2f}")

# Clean up
uninstrument_kubetorch()

# =============================================
# Example 2: Project-Level Attribution
# =============================================
print("\n2. Project-Level Cost Attribution")
print("-" * 60)

# Track costs per project
projects = [
    ("llm-training", "a100", 16.0),
    ("computer-vision", "v100", 32.0),
    ("reinforcement-learning", "a10g", 8.0),
]

reset_cost_aggregator()

for project_name, gpu_type, gpu_hours in projects:
    adapter = instrument_kubetorch(
        team="ml-research",
        project=project_name,
    )

    result = adapter.track_compute_deployment(
        instance_type=gpu_type,
        num_devices=int(gpu_hours),
        workload_type="training",
        duration_seconds=3600,
    )

    print(f"  {project_name:25s}: ${result['cost_total']:7.2f} "
          f"({result['gpu_hours']:.0f} {gpu_type.upper()} GPU-hours)")

# =============================================
# Example 3: Customer/Tenant Attribution (Multi-Tenant)
# =============================================
print("\n3. Customer-Level Attribution (Multi-Tenant)")
print("-" * 60)

# Simulate multiple customers using the platform
customers = [
    ("customer-001", "Acme Corp", "a100", 8.0),
    ("customer-002", "TechStart Inc", "h100", 4.0),
    ("customer-003", "ML Labs", "v100", 16.0),
]

reset_cost_aggregator()
total_platform_cost = 0

print("Customer Usage Report:")
print("-" * 60)

for customer_id, customer_name, gpu_type, gpu_hours in customers:
    # Create adapter with customer attribution
    adapter = instrument_kubetorch(
        team="platform-team",
        customer_id=customer_id,
        metadata={"customer_name": customer_name}
    )

    result = adapter.track_compute_deployment(
        instance_type=gpu_type,
        num_devices=int(gpu_hours),
        workload_type="training",
        duration_seconds=3600,
    )

    cost = result['cost_total']
    total_platform_cost += cost

    print(f"  {customer_name:20s} ({customer_id}): ${cost:8.2f}")

print(f"  {'Total Platform Revenue':20s}:             ${total_platform_cost:8.2f}")

# =============================================
# Example 4: Per-User Attribution
# =============================================
print("\n4. Per-User Cost Attribution")
print("-" * 60)

# Simulate multiple users in the same team
users = [
    ("user-alice", "ml-research", "a100", 4.0),
    ("user-bob", "ml-research", "v100", 8.0),
    ("user-charlie", "ml-research", "t4", 16.0),
]

reset_cost_aggregator()
user_costs = {}

for user_id, team, gpu_type, gpu_hours in users:
    with create_compute_cost_context(f"{user_id}-job") as ctx:
        ctx.add_gpu_cost(
            instance_type=gpu_type,
            gpu_hours=gpu_hours,
            operation_name=f"{user_id}-training"
        )

    user_costs[user_id] = ctx.summary.total_cost

print("Team Cost Breakdown by User:")
for user_id, cost in user_costs.items():
    print(f"  {user_id:15s}: ${cost:7.2f}")

print(f"  {'Team Total':15s}: ${sum(user_costs.values()):7.2f}")

# =============================================
# Example 5: Multi-Dimensional Attribution
# =============================================
print("\n5. Multi-Dimensional Attribution")
print("-" * 60)

# Track with team, project, customer, and environment
adapter = instrument_kubetorch(
    team="platform-engineering",
    project="recommendation-system",
    customer_id="customer-enterprise-001",
    environment="production",
    cost_center="ml-infrastructure",
)

result = adapter.track_compute_deployment(
    instance_type="a100",
    num_devices=8,
    workload_type="inference",
    duration_seconds=3600,
    metadata={
        "service": "recommendation-api",
        "version": "v2.3.0",
        "region": "us-west-2",
    }
)

print("Multi-Dimensional Attribution:")
print(f"  Team:        {adapter.team}")
print(f"  Project:     {adapter.project}")
print(f"  Customer:    {adapter.customer_id}")
print(f"  Environment: {adapter.environment}")
print(f"  Cost Center: {adapter.cost_center}")
print(f"  Total Cost:  ${result['cost_total']:.2f}")

# =============================================
# Example 6: Dynamic Attribution (Request-Based)
# =============================================
print("\n6. Dynamic Attribution (Request-Based)")
print("-" * 60)

def process_training_request(request_data):
    """Process training request with dynamic attribution."""
    # Extract attribution from request
    team = request_data.get("team", "default-team")
    project = request_data.get("project", "default-project")
    user_id = request_data.get("user_id")

    # Create adapter with request-specific attribution
    adapter = instrument_kubetorch(
        team=team,
        project=project,
        metadata={"user_id": user_id}
    )

    # Track training operation
    result = adapter.track_compute_deployment(
        instance_type=request_data["gpu_type"],
        num_devices=request_data["num_gpus"],
        workload_type="training",
        duration_seconds=request_data["duration_seconds"],
    )

    return result

# Simulate multiple requests
requests = [
    {"team": "ml-research", "project": "nlp", "user_id": "alice", "gpu_type": "a100", "num_gpus": 8, "duration_seconds": 3600},
    {"team": "ml-vision", "project": "image-classification", "user_id": "bob", "gpu_type": "v100", "num_gpus": 4, "duration_seconds": 1800},
    {"team": "ml-research", "project": "rl", "user_id": "charlie", "gpu_type": "a10g", "num_gpus": 2, "duration_seconds": 7200},
]

print("Processing Training Requests:")
for i, request in enumerate(requests, 1):
    result = process_training_request(request)
    print(f"  Request {i}: {request['team']}/{request['project']}/{request['user_id']} = ${result['cost_total']:.2f}")

# =============================================
# Example 7: Cost Center Reporting
# =============================================
print("\n7. Cost Center Reporting")
print("-" * 60)

# Simulate different cost centers
cost_centers = {
    "ml-infrastructure": [("a100", 32.0), ("h100", 8.0)],
    "research-compute": [("v100", 64.0), ("a10g", 16.0)],
    "production-serving": [("t4", 128.0)],
}

reset_cost_aggregator()

print("Cost Center Report:")
print("-" * 60)

for cost_center, operations in cost_centers.items():
    cost_center_total = 0

    for gpu_type, gpu_hours in operations:
        adapter = instrument_kubetorch(
            team="finance-reporting",
            cost_center=cost_center,
        )

        result = adapter.track_compute_deployment(
            instance_type=gpu_type,
            num_devices=int(gpu_hours),
            workload_type="training",
            duration_seconds=3600,
        )

        cost_center_total += result['cost_total']

    print(f"  {cost_center:25s}: ${cost_center_total:10.2f}")

print("\n" + "=" * 60)
print("✅ All cost attribution examples completed!")
print("=" * 60)
