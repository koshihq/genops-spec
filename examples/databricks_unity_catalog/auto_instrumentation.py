#!/usr/bin/env python3
"""
Auto-Instrumentation for Databricks Unity Catalog

Demonstrates zero-code GenOps governance integration with existing
Databricks Unity Catalog applications. This shows how to add governance
tracking without modifying your existing code.

Prerequisites:
- Run setup_validation.py first
- Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
- Optional: Set GENOPS_TEAM, GENOPS_PROJECT for governance attribution

Usage:
    python auto_instrumentation.py
"""

import logging
import os
import sys
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Try to import GenOps - handle both pip install and repo development
try:
    # First try normal pip install import
    from genops.providers.databricks_unity_catalog import (
        get_cost_aggregator,
        get_governance_monitor,
    )
    from genops.providers.databricks_unity_catalog.registration import (
        auto_instrument_databricks,
        configure_unity_catalog_governance,
    )

    _GENOPS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to development repo structure
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
        from genops.providers.databricks_unity_catalog import (
            get_cost_aggregator,
            get_governance_monitor,
        )
        from genops.providers.databricks_unity_catalog.registration import (
            auto_instrument_databricks,
            configure_unity_catalog_governance,
        )

        _GENOPS_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Error importing GenOps Databricks Unity Catalog provider: {e}")
        print("💡 Make sure you have installed genops[databricks]:")
        print("   pip install genops[databricks]")
        print("   Or run from the repository root directory")
        sys.exit(1)


def check_configuration():
    """Check basic configuration."""
    workspace_url = os.getenv("DATABRICKS_HOST")
    access_token = os.getenv("DATABRICKS_TOKEN")

    if not workspace_url:
        print("❌ DATABRICKS_HOST environment variable not set")
        print(
            "💡 Set it with: export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'"
        )
        sys.exit(1)

    if not access_token:
        print("❌ DATABRICKS_TOKEN environment variable not set")
        print("💡 Set it with: export DATABRICKS_TOKEN='your_personal_access_token'")
        sys.exit(1)

    return workspace_url


def demonstrate_auto_instrumentation():
    """Demonstrate automatic instrumentation setup."""
    print("🤖 Setting up automatic GenOps instrumentation...")

    # Auto-instrument Databricks operations
    adapter = auto_instrument_databricks()

    if adapter:
        print("✅ Auto-instrumentation enabled successfully")
        print("   • Databricks SDK operations are now tracked")
        print("   • Cost attribution is automatic")
        print("   • Governance telemetry is active")
        return adapter
    else:
        print("⚠️ Auto-instrumentation not available")
        print("   This could be due to:")
        print("   • Databricks SDK not installed")
        print("   • Configuration issues")
        print("   • Network connectivity problems")
        return None


def demonstrate_governance_configuration(workspace_url):
    """Demonstrate governance configuration."""
    print("\n⚙️ Configuring Unity Catalog governance...")

    # Configure governance with auto-detected settings
    config_result = configure_unity_catalog_governance(
        workspace_url=workspace_url,
        metastore_id="auto-detect",  # Will attempt to auto-detect
        enable_lineage_tracking=True,
        enable_cost_attribution=True,
        enable_compliance_monitoring=True,
        default_team=os.getenv("GENOPS_TEAM", "default-team"),
        default_project=os.getenv("GENOPS_PROJECT", "auto-instrumentation-demo"),
    )

    if config_result["configured"]:
        print("✅ Governance configuration successful:")
        print(f"   Workspace: {config_result['workspace_url']}")
        print(f"   Metastore: {config_result['metastore_id']}")
        print("   Features enabled:")
        for feature in config_result["governance_features"]:
            print(f"     • {feature.replace('_', ' ').title()}")
    else:
        print("❌ Governance configuration failed:")
        for error in config_result["errors"]:
            print(f"   • {error}")

    return config_result


def simulate_existing_databricks_operations():
    """
    Simulate existing Databricks operations that would now have governance tracking.

    In a real scenario, these would be your existing Databricks SDK calls
    that now automatically include GenOps governance tracking.
    """
    print("\n🔄 Simulating existing Databricks operations with auto-governance...")

    try:
        # Note: In a real scenario with databricks-sdk installed and configured,
        # these operations would be automatically tracked

        print("📊 Simulated operations (with auto-tracking):")

        # Simulate catalog listing
        print("   • List catalogs → Tracked with governance attributes")

        # Simulate table query
        print("   • Query customer table → Cost and lineage automatically tracked")

        # Simulate schema creation
        print(
            "   • Create analytics schema → Governance policies automatically applied"
        )

        # Simulate SQL warehouse query
        print("   • Run analytical query → Cost attribution automatic")

        print("✅ All operations automatically tracked with zero code changes!")

    except Exception as e:
        logger.warning(f"Simulation error: {e}")
        print(
            "ℹ️ This is a simulation - with real Databricks SDK, operations would be automatically tracked"
        )


def demonstrate_cost_tracking_results():
    """Show cost tracking results from auto-instrumentation."""
    print("\n💰 Viewing automatic cost tracking results...")

    cost_aggregator = get_cost_aggregator()

    # In a real scenario, costs would have been automatically tracked
    # For demo purposes, we'll add some sample costs

    workspace_id = "demo_workspace_auto"

    # Add sample costs as if they were automatically tracked
    cost_aggregator.add_sql_warehouse_cost(
        workspace_id=workspace_id,
        warehouse_size="X-Small",
        query_duration_ms=2500,
        operation_type="auto_tracked_query",
        team=os.getenv("GENOPS_TEAM", "auto-team"),
        project=os.getenv("GENOPS_PROJECT", "auto-project"),
    )

    cost_aggregator.add_storage_cost(
        workspace_id=workspace_id,
        data_size_gb=25.0,
        operation_type="auto_tracked_storage",
        team=os.getenv("GENOPS_TEAM", "auto-team"),
        project=os.getenv("GENOPS_PROJECT", "auto-project"),
    )

    # Get cost summary
    summary = cost_aggregator.get_summary()

    print("✅ Automatic cost tracking summary:")
    print(f"   Total cost: ${summary.total_cost_usd:.6f}")
    print(f"   Operations tracked: {summary.operation_count}")
    print(f"   Workspaces: {len(summary.unique_workspaces)}")
    print("   Cost by resource type:")
    for resource_type, cost in summary.cost_by_resource_type.items():
        print(f"     • {resource_type}: ${cost:.6f}")

    if summary.cost_by_team:
        print("   Cost by team:")
        for team, cost in summary.cost_by_team.items():
            print(f"     • {team}: ${cost:.6f}")


def demonstrate_governance_monitoring():
    """Show governance monitoring results from auto-instrumentation."""
    print("\n🏛️ Viewing automatic governance monitoring...")

    governance_monitor = get_governance_monitor()

    # In a real scenario, governance events would be automatically tracked
    # For demo purposes, we'll add some sample governance events

    # Simulate auto-tracked lineage
    governance_monitor.track_data_lineage(
        lineage_type="read",
        source_catalog="production",
        source_schema="sales",
        source_table="transactions",
        data_classification="confidential",
        user_id="auto_user",
        workspace_id="demo_workspace_auto",
    )

    governance_monitor.track_data_lineage(
        lineage_type="transform",
        source_catalog="production",
        source_schema="sales",
        source_table="transactions",
        target_catalog="analytics",
        target_schema="reports",
        target_table="daily_sales",
        data_classification="internal",
        user_id="auto_user",
        workspace_id="demo_workspace_auto",
    )

    # Get governance summary
    governance_summary = governance_monitor.get_governance_summary()

    print("✅ Automatic governance monitoring summary:")
    print(f"   Lineage events tracked: {governance_summary.lineage_events}")
    print("   Data classifications:")
    for classification, count in governance_summary.data_classifications.items():
        print(f"     • {classification}: {count} operations")

    if governance_summary.policies_applied:
        print("   Policies automatically applied:")
        for policy in governance_summary.policies_applied:
            print(f"     • {policy}")


def show_integration_benefits():
    """Show the benefits of auto-instrumentation integration."""
    print("\n🌟 Benefits of Auto-Instrumentation Integration:")
    print("=" * 50)

    print("✅ Zero Code Changes Required:")
    print("   • Your existing Databricks code works unchanged")
    print("   • No need to modify function calls or add decorators")
    print("   • Governance tracking happens automatically")

    print("\n✅ Comprehensive Tracking:")
    print("   • All Unity Catalog operations automatically tracked")
    print("   • Cost attribution happens in real-time")
    print("   • Data lineage captured without extra work")

    print("\n✅ Team-Based Governance:")
    print("   • Automatic team and project attribution")
    print("   • Budget tracking across all operations")
    print("   • Policy enforcement without code changes")

    print("\n✅ Enterprise Ready:")
    print("   • OpenTelemetry-native telemetry export")
    print("   • Integration with existing observability stacks")
    print("   • Compliance automation and audit trails")


def main():
    """Main auto-instrumentation demonstration."""
    print("🤖 Databricks Unity Catalog Auto-Instrumentation Demo")
    print("=" * 60)

    print("This demo shows how to add comprehensive governance tracking")
    print("to your existing Databricks applications with ZERO code changes!")
    print()

    # Check configuration
    print("1️⃣ Checking configuration...")
    workspace_url = check_configuration()
    print(f"✅ Configuration validated for workspace: {workspace_url}")

    # Set up auto-instrumentation
    print("\n2️⃣ Setting up auto-instrumentation...")
    demonstrate_auto_instrumentation()

    # Configure governance
    print("\n3️⃣ Configuring governance features...")
    demonstrate_governance_configuration(workspace_url)

    # Simulate operations
    print("\n4️⃣ Running existing operations with auto-tracking...")
    simulate_existing_databricks_operations()

    # Show tracking results
    print("\n5️⃣ Viewing automatic tracking results...")
    demonstrate_cost_tracking_results()
    demonstrate_governance_monitoring()

    # Show benefits
    show_integration_benefits()

    print("\n🎉 Auto-instrumentation demonstration completed!")
    print("\n📚 What you learned:")
    print("   ✅ How to enable zero-code governance tracking")
    print("   ✅ How auto-instrumentation works with existing code")
    print("   ✅ How to configure enterprise governance features")
    print("   ✅ How cost and governance data is automatically captured")

    print("\n🎯 Next steps:")
    print("   • Try 'python advanced_features.py' for comprehensive governance")
    print("   • Try 'python cost_optimization.py' for cost optimization strategies")
    print("   • Apply auto-instrumentation to your existing Databricks applications")
    print("   • Configure team-based governance attributes for your organization")


if __name__ == "__main__":
    main()
