"""
Production Deployment Patterns - Enterprise Best Practices

This example demonstrates production-ready patterns for:
- Environment-based configuration
- Kubernetes integration
- High-availability setup
- Monitoring and alerting
- Cost budgets and optimization
- Multi-tenant isolation

Time to run: < 1 minute
"""

import os

from genops.providers.kubetorch import (
    create_compute_cost_context,
    instrument_kubetorch,
    reset_cost_aggregator,
    validate_kubetorch_setup,
)

print("=" * 60)
print("GenOps Kubetorch - Production Deployment Patterns")
print("=" * 60)

# =============================================
# Example 1: Environment-Based Configuration
# =============================================
print("\n1. Environment-Based Configuration")
print("-" * 60)


def setup_genops_for_environment(env: str):
    """Configure GenOps based on deployment environment."""
    config = {
        "development": {
            "telemetry_enabled": False,  # No telemetry overhead in dev
            "cost_tracking_enabled": True,
            "debug": True,
        },
        "staging": {
            "telemetry_enabled": True,
            "cost_tracking_enabled": True,
            "debug": True,
        },
        "production": {
            "telemetry_enabled": True,
            "cost_tracking_enabled": True,
            "debug": False,
            "enable_retry": True,
            "max_retries": 3,
        },
    }

    env_config = config.get(env, config["production"])

    return instrument_kubetorch(
        team=os.getenv("GENOPS_TEAM", "default-team"),
        project=os.getenv("GENOPS_PROJECT", "default-project"),
        environment=env,
        **env_config,
    )


# Configure for production
adapter = setup_genops_for_environment("production")
print("✓ Configured for production environment")
print(f"  Team: {adapter.team}")
print(f"  Environment: {adapter.environment}")
print(f"  Telemetry: {'Enabled' if adapter.telemetry_enabled else 'Disabled'}")
print(f"  Debug: {'Enabled' if adapter.debug else 'Disabled'}")

# =============================================
# Example 2: Kubernetes ConfigMap Integration
# =============================================
print("\n2. Kubernetes ConfigMap Configuration")
print("-" * 60)

# Simulating environment variables from Kubernetes ConfigMap
os.environ.update(
    {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4317",
        "GENOPS_TEAM": "ml-platform",
        "GENOPS_PROJECT": "recommendation-engine",
        "GENOPS_ENVIRONMENT": "production",
        "GENOPS_COST_CENTER": "ml-infrastructure",
    }
)

# Auto-configure from environment
adapter_k8s = instrument_kubetorch(
    team=os.getenv("GENOPS_TEAM"),
    project=os.getenv("GENOPS_PROJECT"),
    environment=os.getenv("GENOPS_ENVIRONMENT"),
    cost_center=os.getenv("GENOPS_COST_CENTER"),
)

print("✓ Configured from Kubernetes ConfigMap")
print(f"  Team: {adapter_k8s.team}")
print(f"  Project: {adapter_k8s.project}")
print(f"  Environment: {adapter_k8s.environment}")
print(f"  Cost Center: {adapter_k8s.cost_center}")

# =============================================
# Example 3: Production Validation Workflow
# =============================================
print("\n3. Production Startup Validation")
print("-" * 60)


def production_startup_validation():
    """Run validation checks at startup."""
    result = validate_kubetorch_setup(
        check_kubetorch=True,
        check_kubernetes=True,
        check_opentelemetry=True,
        check_genops=True,
    )

    if not result.is_valid():
        print("❌ CRITICAL: Validation failed!")
        for issue in result.issues:
            if issue.level.value == "error":
                print(f"  ERROR: {issue.message}")
        return False

    if result.warnings > 0:
        print(f"⚠️  WARNING: {result.warnings} warnings found")
        for issue in result.issues:
            if issue.level.value == "warning":
                print(f"  {issue.message}")

    print(
        f"✓ Validation passed: {result.successful_checks}/{result.total_checks} checks successful"
    )
    return True


# Run validation
validation_passed = production_startup_validation()

# =============================================
# Example 4: Cost Budget Monitoring
# =============================================
print("\n4. Cost Budget Monitoring")
print("-" * 60)


class CostBudgetMonitor:
    """Monitor costs against budget limits."""

    def __init__(self, daily_budget: float, warning_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.warning_threshold = warning_threshold
        self.current_cost = 0.0

    def track_operation(self, cost: float, operation_id: str):
        """Track operation cost and check budget."""
        self.current_cost += cost

        utilization = self.current_cost / self.daily_budget

        if utilization >= 1.0:
            print(
                f"  🚨 BUDGET EXCEEDED: ${self.current_cost:.2f} / ${self.daily_budget:.2f}"
            )
            print(f"     Operation: {operation_id}")
            return "budget_exceeded"
        elif utilization >= self.warning_threshold:
            print(
                f"  ⚠️  BUDGET WARNING: {utilization * 100:.1f}% used (${self.current_cost:.2f} / ${self.daily_budget:.2f})"
            )
            print(f"     Operation: {operation_id}")
            return "budget_warning"
        else:
            print(f"  ✓ Budget OK: {utilization * 100:.1f}% used")
            return "budget_ok"

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.daily_budget - self.current_cost)


# Create budget monitor
budget_monitor = CostBudgetMonitor(daily_budget=1000.0, warning_threshold=0.8)

# Simulate operations throughout the day
operations = [
    ("morning-training", "a100", 16.0),
    ("afternoon-training", "a100", 24.0),
    ("evening-training", "a100", 16.0),
]

reset_cost_aggregator()

for op_id, gpu_type, gpu_hours in operations:
    with create_compute_cost_context(op_id) as ctx:
        ctx.add_gpu_cost(gpu_type, gpu_hours=gpu_hours)

    status = budget_monitor.track_operation(ctx.summary.total_cost, op_id)

print("\nDaily Summary:")
print(f"  Total Spent: ${budget_monitor.current_cost:.2f}")
print(f"  Remaining: ${budget_monitor.get_remaining_budget():.2f}")

# =============================================
# Example 5: Multi-Tenant Isolation
# =============================================
print("\n5. Multi-Tenant Cost Isolation")
print("-" * 60)


class TenantCostTracker:
    """Track costs per tenant with isolation."""

    def __init__(self):
        self.tenant_costs = {}

    def track_tenant_operation(self, tenant_id: str, operation_cost: float):
        """Track cost for specific tenant."""
        if tenant_id not in self.tenant_costs:
            self.tenant_costs[tenant_id] = {
                "total_cost": 0.0,
                "operation_count": 0,
            }

        self.tenant_costs[tenant_id]["total_cost"] += operation_cost
        self.tenant_costs[tenant_id]["operation_count"] += 1

    def get_tenant_report(self):
        """Generate tenant cost report."""
        return self.tenant_costs


# Create tenant tracker
tenant_tracker = TenantCostTracker()

# Simulate multi-tenant operations
tenants = [
    ("tenant-acme", "a100", 8.0),
    ("tenant-techstart", "h100", 4.0),
    ("tenant-mlabs", "v100", 16.0),
    ("tenant-acme", "a100", 4.0),  # Second operation for acme
]

reset_cost_aggregator()

for tenant_id, gpu_type, gpu_hours in tenants:
    # Create isolated adapter for tenant
    adapter = instrument_kubetorch(
        team="platform-team",
        customer_id=tenant_id,
    )

    result = adapter.track_compute_deployment(
        instance_type=gpu_type,
        num_devices=int(gpu_hours),
        workload_type="training",
        duration_seconds=3600,
    )

    tenant_tracker.track_tenant_operation(tenant_id, result["cost_total"])

# Generate report
print("Tenant Cost Report:")
for tenant_id, data in tenant_tracker.get_tenant_report().items():
    print(
        f"  {tenant_id:20s}: ${data['total_cost']:8.2f} ({data['operation_count']} ops)"
    )

# =============================================
# Example 6: High-Availability Configuration
# =============================================
print("\n6. High-Availability Setup")
print("-" * 60)


def create_ha_adapter():
    """Create adapter with HA configuration."""
    return instrument_kubetorch(
        team="production-ml",
        project="critical-service",
        environment="production",
        # Retry configuration
        enable_retry=True,
        max_retries=3,
        # Telemetry configuration
        telemetry_enabled=True,
        cost_tracking_enabled=True,
        # Debug disabled for performance
        debug=False,
    )


ha_adapter = create_ha_adapter()

print("✓ High-Availability Adapter Created")
print("  Features:")
print("    - Automatic retry on transient failures")
print("    - Telemetry export with error handling")
print("    - Cost tracking with graceful degradation")

# =============================================
# Example 7: Operational Metrics and Monitoring
# =============================================
print("\n7. Operational Metrics Dashboard")
print("-" * 60)


class OperationalMetrics:
    """Track operational metrics for monitoring."""

    def __init__(self):
        self.metrics = {
            "total_operations": 0,
            "total_cost": 0.0,
            "total_gpu_hours": 0.0,
            "operations_by_type": {},
            "cost_by_team": {},
        }

    def record_operation(
        self, team: str, workload_type: str, cost: float, gpu_hours: float
    ):
        """Record operation metrics."""
        self.metrics["total_operations"] += 1
        self.metrics["total_cost"] += cost
        self.metrics["total_gpu_hours"] += gpu_hours

        # By type
        if workload_type not in self.metrics["operations_by_type"]:
            self.metrics["operations_by_type"][workload_type] = {
                "count": 0,
                "cost": 0.0,
            }
        self.metrics["operations_by_type"][workload_type]["count"] += 1
        self.metrics["operations_by_type"][workload_type]["cost"] += cost

        # By team
        if team not in self.metrics["cost_by_team"]:
            self.metrics["cost_by_team"][team] = 0.0
        self.metrics["cost_by_team"][team] += cost

    def print_dashboard(self):
        """Print operational dashboard."""
        print("Operational Dashboard:")
        print(f"  Total Operations: {self.metrics['total_operations']}")
        print(f"  Total Cost: ${self.metrics['total_cost']:.2f}")
        print(f"  Total GPU Hours: {self.metrics['total_gpu_hours']:.1f}")
        print(
            f"  Avg Cost/Operation: ${self.metrics['total_cost'] / max(1, self.metrics['total_operations']):.2f}"
        )

        print("\n  By Workload Type:")
        for wtype, data in self.metrics["operations_by_type"].items():
            print(f"    {wtype:15s}: {data['count']:3d} ops, ${data['cost']:8.2f}")

        print("\n  By Team:")
        for team, cost in self.metrics["cost_by_team"].items():
            pct = (
                (cost / self.metrics["total_cost"]) * 100
                if self.metrics["total_cost"] > 0
                else 0
            )
            print(f"    {team:20s}: ${cost:8.2f} ({pct:5.1f}%)")


# Create metrics tracker
metrics = OperationalMetrics()

# Simulate production workload
workloads = [
    ("ml-research", "training", "a100", 16.0),
    ("ml-vision", "training", "v100", 32.0),
    ("ml-serving", "inference", "t4", 64.0),
    ("ml-research", "fine-tuning", "a100", 8.0),
    ("ml-nlp", "training", "h100", 8.0),
]

reset_cost_aggregator()

for team, workload_type, gpu_type, gpu_hours in workloads:
    adapter = instrument_kubetorch(team=team)

    result = adapter.track_compute_deployment(
        instance_type=gpu_type,
        num_devices=int(gpu_hours),
        workload_type=workload_type,
        duration_seconds=3600,
    )

    metrics.record_operation(
        team, workload_type, result["cost_total"], result["gpu_hours"]
    )

# Print dashboard
metrics.print_dashboard()

# =============================================
# Example 8: Graceful Shutdown
# =============================================
print("\n8. Graceful Shutdown Procedure")
print("-" * 60)


def graceful_shutdown():
    """Graceful shutdown procedure for production."""
    print("  1. Finalizing active operations...")
    # In production, finalize any active tracking

    print("  2. Flushing telemetry buffers...")
    # Ensure all telemetry is exported

    print("  3. Generating final cost report...")
    # Generate final cost summary

    print("  ✓ Graceful shutdown complete")


graceful_shutdown()

print("\n" + "=" * 60)
print("✅ All production deployment examples completed!")
print("=" * 60)
print("\nProduction Checklist:")
print("  ✓ Environment-based configuration")
print("  ✓ Kubernetes integration")
print("  ✓ Startup validation")
print("  ✓ Budget monitoring")
print("  ✓ Multi-tenant isolation")
print("  ✓ High-availability setup")
print("  ✓ Operational metrics")
print("  ✓ Graceful shutdown")
print("=" * 60)
