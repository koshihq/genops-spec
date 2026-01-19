"""
Multi-tenant trace isolation in Grafana Tempo.

This example demonstrates:
- Multi-tenant Tempo configuration
- Trace isolation by tenant
- Cross-tenant cost analysis
- Tenant-specific governance policies

Prerequisites:
    - Tempo configured with multi-tenancy support
    - X-Scope-OrgID header support enabled in Tempo

Tempo Multi-Tenancy Config:
    multitenancy_enabled: true
    multitenancy_tenant_header: X-Scope-OrgID
"""

import time
import random
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from genops import track_usage


class TenantTracking:
    """
    Multi-tenant trace tracking with isolated telemetry.

    Each tenant gets isolated traces in Tempo using X-Scope-OrgID header.
    """

    def __init__(self, tempo_endpoint: str = "http://localhost:3200"):
        """
        Initialize multi-tenant tracking.

        Args:
            tempo_endpoint: Tempo base endpoint
        """
        self.tempo_endpoint = tempo_endpoint
        self.tenant_providers: Dict[str, TracerProvider] = {}

    def configure_tenant(self, tenant_id: str) -> TracerProvider:
        """
        Configure dedicated tracer provider for a tenant.

        Args:
            tenant_id: Unique tenant identifier

        Returns:
            Configured TracerProvider for the tenant
        """
        if tenant_id in self.tenant_providers:
            return self.tenant_providers[tenant_id]

        # Create tenant-specific OTLP exporter with X-Scope-OrgID header
        otlp_endpoint = f"{self.tempo_endpoint}:4318/v1/traces"

        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers={"X-Scope-OrgID": tenant_id}
        )

        # Create tenant-specific tracer provider
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))

        self.tenant_providers[tenant_id] = provider

        print(f"✅ Configured tenant: {tenant_id}")
        return provider

    def get_tracer(self, tenant_id: str, name: str):
        """
        Get tracer for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            name: Tracer name

        Returns:
            Tracer instance for the tenant
        """
        provider = self.configure_tenant(tenant_id)
        return provider.get_tracer(name)


def simulate_tenant_operation(
    tenant_tracking: TenantTracking,
    tenant_id: str,
    operation_name: str,
    cost: float
) -> Dict[str, Any]:
    """
    Simulate an operation for a specific tenant.

    Args:
        tenant_tracking: TenantTracking instance
        tenant_id: Tenant identifier
        operation_name: Name of the operation
        cost: Operation cost in USD

    Returns:
        Operation result
    """
    tracer = tenant_tracking.get_tracer(tenant_id, __name__)

    with tracer.start_as_current_span(operation_name) as span:
        # Set tenant attributes
        span.set_attribute("tenant_id", tenant_id)
        span.set_attribute("genops.cost.total_cost", cost)
        span.set_attribute("genops.cost.currency", "USD")

        # Simulate operation
        time.sleep(random.uniform(0.05, 0.15))

        return {
            "tenant_id": tenant_id,
            "operation": operation_name,
            "cost": cost,
            "status": "success"
        }


def main():
    """
    Demonstrate multi-tenant trace isolation in Tempo.
    """
    print("=" * 70)
    print("Grafana Tempo Multi-Tenant Example")
    print("=" * 70)
    print()

    print("⚠️  Note: This example requires Tempo with multi-tenancy enabled")
    print("   Configure Tempo with: multitenancy_enabled: true")
    print()

    # Initialize multi-tenant tracking
    tenant_tracking = TenantTracking(tempo_endpoint="http://localhost")

    # ========================================================================
    # Scenario 1: Isolated Tenant Operations
    # ========================================================================

    print("=" * 70)
    print("Scenario 1: Isolated Tenant Operations")
    print("=" * 70)
    print()

    tenants = ["acme-corp", "globex-inc", "initech-ltd"]

    print("Configuring tenants...")
    for tenant_id in tenants:
        tenant_tracking.configure_tenant(tenant_id)

    print()
    print("Executing isolated tenant operations...")

    for tenant_id in tenants:
        result = simulate_tenant_operation(
            tenant_tracking,
            tenant_id,
            "ai_query",
            cost=random.uniform(0.01, 0.10)
        )
        print(f"  {tenant_id}: ${result['cost']:.4f}")

    print()

    # ========================================================================
    # Scenario 2: Tenant Cost Tracking
    # ========================================================================

    print("=" * 70)
    print("Scenario 2: Tenant Cost Tracking")
    print("=" * 70)
    print()

    tenant_costs = {}

    print("Simulating usage for each tenant...")
    for tenant_id in tenants:
        # Simulate multiple operations per tenant
        tenant_total = 0.0

        for i in range(random.randint(3, 8)):
            result = simulate_tenant_operation(
                tenant_tracking,
                tenant_id,
                f"operation_{i}",
                cost=random.uniform(0.01, 0.05)
            )
            tenant_total += result["cost"]

        tenant_costs[tenant_id] = tenant_total
        print(f"  {tenant_id}: {len(range(random.randint(3, 8)))} operations, ${tenant_total:.4f} total")

    print()

    # ========================================================================
    # Scenario 3: Tenant-Specific Governance
    # ========================================================================

    print("=" * 70)
    print("Scenario 3: Tenant-Specific Governance Policies")
    print("=" * 70)
    print()

    # Different tiers with different policies
    tenant_tiers = {
        "acme-corp": {"tier": "enterprise", "monthly_limit": 5000.0},
        "globex-inc": {"tier": "professional", "monthly_limit": 1000.0},
        "initech-ltd": {"tier": "starter", "monthly_limit": 100.0}
    }

    print("Applying tenant-specific policies...")
    for tenant_id, policy in tenant_tiers.items():
        tracer = tenant_tracking.get_tracer(tenant_id, __name__)

        with tracer.start_as_current_span("policy_check") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("genops.policy.tier", policy["tier"])
            span.set_attribute("genops.budget.monthly_limit", policy["monthly_limit"])

            # Simulate policy enforcement
            current_spend = tenant_costs.get(tenant_id, 0.0)
            remaining = policy["monthly_limit"] - current_spend

            span.set_attribute("genops.budget.remaining", remaining)
            span.set_attribute("genops.budget.utilization_pct", (current_spend / policy["monthly_limit"]) * 100)

            print(f"  {tenant_id} ({policy['tier']}): ${remaining:.2f} remaining")

    print()

    # ========================================================================
    # Scenario 4: Cross-Tenant Analysis
    # ========================================================================

    print("=" * 70)
    print("Scenario 4: Cross-Tenant Analysis")
    print("=" * 70)
    print()

    print("Tenant Cost Summary:")
    print("-" * 70)

    total_cost = 0.0
    for tenant_id in sorted(tenant_costs.keys()):
        cost = tenant_costs[tenant_id]
        tier = tenant_tiers[tenant_id]["tier"]
        limit = tenant_tiers[tenant_id]["monthly_limit"]
        utilization = (cost / limit) * 100

        print(f"  {tenant_id:15} ({tier:12}): ${cost:7.4f} ({utilization:5.2f}% of limit)")
        total_cost += cost

    print("-" * 70)
    print(f"  {'Total':15} {'':12}  ${total_cost:7.4f}")
    print()

    # ========================================================================
    # Scenario 5: Tenant Isolation Verification
    # ========================================================================

    print("=" * 70)
    print("Scenario 5: Tenant Isolation Verification")
    print("=" * 70)
    print()

    print("Creating high-volume operations for one tenant...")
    print("(Other tenants should remain unaffected)")
    print()

    # Simulate high load for one tenant
    high_volume_tenant = "acme-corp"

    for i in range(20):
        simulate_tenant_operation(
            tenant_tracking,
            high_volume_tenant,
            f"bulk_operation_{i}",
            cost=0.001
        )

    print(f"✅ Created 20 operations for {high_volume_tenant}")
    print(f"   Other tenants' traces remain isolated in Tempo")
    print()

    # Wait for export
    print("⏳ Waiting for spans to export...")
    time.sleep(2)

    # ========================================================================
    # Query Examples for Multi-Tenant Analysis
    # ========================================================================

    print("=" * 70)
    print("Multi-Tenant Query Examples")
    print("=" * 70)
    print("""
Query tenant-specific traces in Tempo:

1. **Query Specific Tenant** (set X-Scope-OrgID header)
   curl -H "X-Scope-OrgID: acme-corp" \\
     "http://localhost:3200/api/search?q={}&limit=10"

2. **Tenant Cost Summary**
   {.tenant_id = "acme-corp"} | sum(.genops.cost.total_cost)

3. **Cross-Tenant Cost Comparison**
   {} | sum(.genops.cost.total_cost) by (.tenant_id)

4. **Tenant Budget Utilization**
   {.genops.budget.utilization_pct > 80} | count() by (.tenant_id)

5. **High-Spending Tenants**
   {} | sum(.genops.cost.total_cost) by (.tenant_id) > 1.0

6. **Tenant Tier Analysis**
   {} | avg(.genops.cost.total_cost) by (.genops.policy.tier)

7. **Tenant Operations Count**
   {.tenant_id = "globex-inc"} | rate()

8. **Cross-Tenant Performance**
   {} | avg(duration) by (.tenant_id)
    """)

    # ========================================================================
    # Multi-Tenancy Benefits Summary
    # ========================================================================

    print("=" * 70)
    print("Multi-Tenancy Benefits")
    print("=" * 70)
    print("""
Grafana Tempo Multi-Tenancy provides:

1. **Trace Isolation**
   - Each tenant's traces are stored separately
   - No cross-tenant data leakage
   - Independent data retention policies

2. **Cost Attribution**
   - Per-tenant cost tracking
   - Tenant-specific billing/chargeback
   - Cost center allocation

3. **Governance Policies**
   - Tenant-specific budget limits
   - Tier-based policies (enterprise/professional/starter)
   - Custom retention per tenant

4. **Security & Compliance**
   - Data isolation for regulatory compliance
   - Tenant-specific access controls
   - Audit trails per tenant

5. **Scalability**
   - Independent scaling per tenant
   - Query isolation prevents noisy neighbors
   - Resource allocation per tenant tier

6. **Operational Excellence**
   - Per-tenant monitoring and alerts
   - Tenant-specific SLAs
   - Isolated troubleshooting

Implementation Pattern:
- Set X-Scope-OrgID header on all trace exports
- Configure Tempo with multitenancy_enabled: true
- Query with X-Scope-OrgID header for tenant isolation
- Use TraceQL for cross-tenant analysis (when authorized)

For production deployments:
- Use authentication/authorization for tenant access
- Implement tenant quota enforcement
- Set up tenant-specific alerting
- Configure data retention by tenant tier
    """)

    print("=" * 70)
    print("✅ Multi-tenant example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
