#!/usr/bin/env python3
"""
Quick Demo - Databricks Unity Catalog with GenOps (2 minutes)

This is a copy-paste ready example that demonstrates immediate value
with minimal setup. Perfect for first-time users and quick evaluation.

⚡ Zero configuration required if you have Databricks environment variables set!

Usage:
    python quick_demo.py

Expected time: Under 2 minutes to see governance results
"""

import sys
from pathlib import Path

# Add the GenOps providers path to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def main():
    """Quick demonstration with immediate value and minimal setup."""
    print("🚀 Databricks Unity Catalog + GenOps Quick Demo")
    print("=" * 50)
    print("⚡ This demo shows immediate governance value in under 2 minutes!")
    print()
    
    # Step 1: Check if basic requirements are met
    print("1️⃣ Checking requirements...")
    
    try:
        from genops.providers.databricks_unity_catalog.registration import auto_instrument_databricks
        print("   ✅ GenOps Databricks provider available")
    except ImportError:
        print("   ❌ GenOps Databricks provider not found")
        print("   💡 Install with: pip install genops[databricks]")
        return False
    
    # Step 2: Auto-configure with intelligent defaults
    print("\n2️⃣ Auto-configuring governance...")
    
    adapter = auto_instrument_databricks()
    
    if not adapter:
        print("   ⚠️ Auto-configuration not available")
        print("   💡 Set minimal environment:")
        print("      export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'")
        print("      export DATABRICKS_TOKEN='your_token'")
        print()
        print("   🔧 Using demo mode with simulated data...")
        # Fall back to demo mode
        from genops.providers.databricks_unity_catalog import instrument_databricks_unity_catalog
        adapter = instrument_databricks_unity_catalog(workspace_url="demo://localhost")
    else:
        print("   ✅ Auto-configuration successful!")
    
    # Step 3: Demonstrate immediate governance tracking
    print("\n3️⃣ Demonstrating governance tracking...")
    
    # Simulate realistic Unity Catalog operations
    operations = [
        {
            "type": "catalog",
            "operation": "read",
            "catalog": "production",
            "description": "List production catalog"
        },
        {
            "type": "table", 
            "operation": "query",
            "catalog": "production",
            "schema": "analytics",
            "table": "customer_events",
            "rows": 25000,
            "size_mb": 85,
            "description": "Query customer events table"
        },
        {
            "type": "sql_warehouse",
            "warehouse_id": "analytics-warehouse-small",
            "query_type": "transform",
            "duration_ms": 3500,
            "compute_units": 0.8,
            "description": "Run analytics transformation"
        }
    ]
    
    print("   Tracking operations:")
    for i, op in enumerate(operations, 1):
        try:
            if op["type"] == "catalog":
                result = adapter.track_catalog_operation(
                    operation=op["operation"],
                    catalog_name=op["catalog"],
                    team="demo-team",
                    project="quick-demo",
                    environment="development"
                )
                print(f"     {i}. ✅ {op['description']}")
                
            elif op["type"] == "table":
                result = adapter.track_table_operation(
                    operation=op["operation"],
                    catalog_name=op["catalog"],
                    schema_name=op["schema"],
                    table_name=op["table"],
                    row_count=op["rows"],
                    data_size_bytes=op["size_mb"] * 1024 * 1024,
                    team="demo-team",
                    project="quick-demo",
                    environment="development"
                )
                print(f"     {i}. ✅ {op['description']} ({op['rows']:,} rows, {op['size_mb']} MB)")
                
            elif op["type"] == "sql_warehouse":
                result = adapter.track_sql_warehouse_operation(
                    sql_warehouse_id=op["warehouse_id"],
                    query_type=op["query_type"],
                    query_duration_ms=op["duration_ms"],
                    compute_units=op["compute_units"],
                    team="demo-team", 
                    project="quick-demo",
                    environment="development"
                )
                print(f"     {i}. ✅ {op['description']} ({op['duration_ms']}ms, {op['compute_units']} DBU)")
                
        except Exception as e:
            print(f"     {i}. ⚠️ {op['description']} (simulated due to: {type(e).__name__})")
    
    # Step 4: Show immediate cost tracking results
    print("\n4️⃣ Viewing governance results...")
    
    try:
        from genops.providers.databricks_unity_catalog import get_cost_aggregator
        
        # Add some demo costs for illustration
        cost_aggregator = get_cost_aggregator()
        
        # Simulate costs for the operations above
        cost_aggregator.add_sql_warehouse_cost(
            workspace_id="demo-workspace",
            warehouse_size="Small",
            query_duration_ms=3500,
            operation_type="analytics_query",
            team="demo-team",
            project="quick-demo"
        )
        
        cost_aggregator.add_storage_cost(
            workspace_id="demo-workspace",
            data_size_gb=0.085,  # 85 MB
            operation_type="table_storage",
            team="demo-team",
            project="quick-demo"
        )
        
        # Get summary
        summary = cost_aggregator.get_summary()
        
        print("   💰 Cost Analysis:")
        print(f"     • Total cost: ${summary.total_cost_usd:.6f}")
        print(f"     • Operations tracked: {summary.operation_count}")
        print(f"     • Workspaces: {len(summary.unique_workspaces)}")
        
        if summary.cost_by_team:
            print("     • Cost by team:")
            for team, cost in summary.cost_by_team.items():
                print(f"       - {team}: ${cost:.6f}")
        
        if summary.cost_by_resource_type:
            print("     • Cost by resource:")
            for resource, cost in summary.cost_by_resource_type.items():
                print(f"       - {resource}: ${cost:.6f}")
                
    except Exception as e:
        print(f"   ⚠️ Cost tracking demo failed: {e}")
    
    # Step 5: Show data lineage tracking
    print("\n5️⃣ Demonstrating data lineage...")
    
    try:
        from genops.providers.databricks_unity_catalog import get_governance_monitor
        
        governance_monitor = get_governance_monitor("demo-metastore")
        
        # Track sample lineage
        lineage_examples = [
            {
                "type": "read",
                "source": ("raw_data", "events", "user_sessions"),
                "target": None,
                "classification": "internal"
            },
            {
                "type": "transform", 
                "source": ("raw_data", "events", "user_sessions"),
                "target": ("analytics", "aggregated", "daily_sessions"),
                "classification": "internal"
            }
        ]
        
        print("   📊 Data Lineage Tracking:")
        for i, lineage in enumerate(lineage_examples, 1):
            if lineage["target"]:
                source_path = f"{lineage['source'][0]}.{lineage['source'][1]}.{lineage['source'][2]}"
                target_path = f"{lineage['target'][0]}.{lineage['target'][1]}.{lineage['target'][2]}"
                
                governance_monitor.track_data_lineage(
                    lineage_type=lineage["type"],
                    source_catalog=lineage["source"][0],
                    source_schema=lineage["source"][1],
                    source_table=lineage["source"][2],
                    target_catalog=lineage["target"][0],
                    target_schema=lineage["target"][1],
                    target_table=lineage["target"][2],
                    data_classification=lineage["classification"],
                    user_id="demo-user"
                )
                print(f"     {i}. ✅ {lineage['type'].title()}: {source_path} → {target_path}")
            else:
                source_path = f"{lineage['source'][0]}.{lineage['source'][1]}.{lineage['source'][2]}"
                
                governance_monitor.track_data_lineage(
                    lineage_type=lineage["type"],
                    source_catalog=lineage["source"][0],
                    source_schema=lineage["source"][1],
                    source_table=lineage["source"][2],
                    data_classification=lineage["classification"],
                    user_id="demo-user"
                )
                print(f"     {i}. ✅ {lineage['type'].title()}: {source_path}")
        
        # Show governance summary
        governance_summary = governance_monitor.get_governance_summary()
        print(f"   🏛️ Governance Summary:")
        print(f"     • Lineage events: {governance_summary.lineage_events}")
        print(f"     • Data classifications: {dict(governance_summary.data_classifications)}")
        
    except Exception as e:
        print(f"   ⚠️ Lineage tracking demo failed: {e}")
    
    # Step 6: Show what you've achieved
    print("\n🎉 Demo Complete! What you just enabled:")
    print("=" * 50)
    print("✅ Real-time cost tracking for all Unity Catalog operations")
    print("✅ Automatic data lineage capture across catalogs and tables")
    print("✅ Team-based cost attribution and governance controls")
    print("✅ OpenTelemetry-compatible telemetry for your observability stack")
    print("✅ Zero code changes required for existing applications")
    print()
    print("📊 Telemetry Data Available In:")
    print("   • Datadog: Traces → Search 'genops.provider:databricks_unity_catalog'")
    print("   • Grafana: Explore → Traces → Filter by genops.provider")
    print("   • Honeycomb: Query genops.provider = 'databricks_unity_catalog'")
    print("   • Any OpenTelemetry-compatible platform")
    print()
    print("🎯 Next Steps:")
    print("   • Try 'python basic_tracking.py' for more detailed examples")
    print("   • Read the 5-minute quickstart guide")
    print("   • Integrate with your existing Databricks applications")
    print("   • Set up team governance attributes for your organization")
    print()
    print("⏱️ Total demo time: Under 2 minutes")
    print("🚀 Ready for production: Add one line to your existing code!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ Demo completed successfully!")
        else:
            print("\n⚠️ Demo completed with warnings - see messages above")
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        print("💡 Try running 'python setup_validation.py' for detailed diagnostics")