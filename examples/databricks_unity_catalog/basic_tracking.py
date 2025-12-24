#!/usr/bin/env python3
"""
Basic Databricks Unity Catalog Governance Tracking

Demonstrates basic GenOps governance tracking for Unity Catalog operations.
Shows cost attribution, data lineage, and team-based governance.

Prerequisites:
- Run setup_validation.py first
- Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
- Set GENOPS_TEAM and GENOPS_PROJECT for governance attribution

Usage:
    python basic_tracking.py
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Try to import GenOps - handle both pip install and repo development
try:
    # First try normal pip install import
    from genops.providers.databricks_unity_catalog import (
        instrument_databricks_unity_catalog,
        create_workspace_cost_context,
        get_governance_monitor
    )
    from genops.providers.databricks_unity_catalog_pricing import get_pricing_calculator
    _GENOPS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to development repo structure
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
        from genops.providers.databricks_unity_catalog import (
            instrument_databricks_unity_catalog,
            create_workspace_cost_context,
            get_governance_monitor
        )
        from genops.providers.databricks_unity_catalog_pricing import get_pricing_calculator
        _GENOPS_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Error importing GenOps Databricks Unity Catalog provider: {e}")
        print("💡 Make sure you have installed genops[databricks]:")
        print("   pip install genops[databricks]")
        print("   Or run from the repository root directory")
        sys.exit(1)


def check_configuration() -> Dict[str, str]:
    """Check and return configuration."""
    config = {}
    
    # Required configuration
    workspace_url = os.getenv('DATABRICKS_HOST')
    access_token = os.getenv('DATABRICKS_TOKEN')
    
    if not workspace_url:
        print("❌ DATABRICKS_HOST environment variable not set")
        print("💡 Set it with: export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'")
        sys.exit(1)
    
    if not access_token:
        print("❌ DATABRICKS_TOKEN environment variable not set")
        print("💡 Set it with: export DATABRICKS_TOKEN='your_personal_access_token'")
        sys.exit(1)
    
    config['workspace_url'] = workspace_url
    config['access_token'] = '***configured***'
    
    # Governance configuration (optional but recommended)
    config['team'] = os.getenv('GENOPS_TEAM', 'demo-team')
    config['project'] = os.getenv('GENOPS_PROJECT', 'unity-catalog-demo')
    config['environment'] = os.getenv('GENOPS_ENVIRONMENT', 'development')
    
    return config


def demonstrate_catalog_operations(adapter, governance_attrs: Dict[str, str]) -> None:
    """Demonstrate catalog-level operations with governance tracking."""
    print("\n🏛️ Demonstrating catalog operations with governance tracking...")
    
    # Track catalog creation operation
    catalog_metadata = adapter.track_catalog_operation(
        operation="read",
        catalog_name="demo_catalog",
        **governance_attrs
    )
    
    print(f"✅ Tracked catalog operation: {catalog_metadata['operation']}")
    print(f"   Catalog: {catalog_metadata['catalog_name']}")
    print(f"   Trace ID: {catalog_metadata['trace_id']}")
    print(f"   Team: {governance_attrs.get('team', 'not_set')}")
    print(f"   Project: {governance_attrs.get('project', 'not_set')}")


def demonstrate_table_operations(adapter, governance_attrs: Dict[str, str]) -> None:
    """Demonstrate table-level operations with governance tracking."""
    print("\n📊 Demonstrating table operations with governance tracking...")
    
    # Track table query operation
    table_metadata = adapter.track_table_operation(
        operation="query",
        catalog_name="demo_catalog",
        schema_name="demo_schema",
        table_name="customer_data",
        row_count=1500,
        data_size_bytes=1024 * 1024 * 50,  # 50 MB
        **governance_attrs
    )
    
    print(f"✅ Tracked table operation: {table_metadata['operation']}")
    print(f"   Table: {table_metadata['catalog_name']}.{table_metadata['schema_name']}.{table_metadata['table_name']}")
    print(f"   Rows processed: {table_metadata['row_count']}")
    print(f"   Data size: {table_metadata['data_size_bytes'] / (1024*1024):.1f} MB")
    print(f"   Trace ID: {table_metadata['trace_id']}")


def demonstrate_sql_warehouse_operations(adapter, governance_attrs: Dict[str, str]) -> None:
    """Demonstrate SQL warehouse operations with cost tracking."""
    print("\n⚡ Demonstrating SQL warehouse operations with cost tracking...")
    
    # Track SQL warehouse query
    warehouse_metadata = adapter.track_sql_warehouse_operation(
        sql_warehouse_id="demo_warehouse_123",
        query_type="select",
        query_duration_ms=5000,  # 5 seconds
        compute_units=0.25,      # 0.25 DBU
        **governance_attrs
    )
    
    print(f"✅ Tracked SQL warehouse operation: {warehouse_metadata['operation']}")
    print(f"   Warehouse ID: {warehouse_metadata['sql_warehouse_id']}")
    print(f"   Query type: {warehouse_metadata['query_type']}")
    print(f"   Duration: {warehouse_metadata['query_duration_ms']}ms")
    print(f"   Compute units: {warehouse_metadata['compute_units']} DBU")


def demonstrate_cost_calculation() -> None:
    """Demonstrate cost calculation for different operation types."""
    print("\n💰 Demonstrating cost calculation...")
    
    pricing_calc = get_pricing_calculator()
    
    # Calculate SQL warehouse cost
    warehouse_cost = pricing_calc.calculate_sql_warehouse_cost(
        warehouse_size="Small",
        duration_ms=5000,  # 5 seconds
        region="us-west-2"
    )
    
    print(f"✅ SQL warehouse cost calculation:")
    print(f"   Warehouse size: Small")
    print(f"   Duration: 5 seconds")
    print(f"   Estimated cost: ${warehouse_cost:.6f}")
    
    # Calculate compute cluster cost
    cluster_cost = pricing_calc.calculate_compute_cluster_cost(
        cluster_type="standard",
        node_count=3,
        duration_ms=300000,  # 5 minutes
        region="us-west-2"
    )
    
    print(f"✅ Compute cluster cost calculation:")
    print(f"   Cluster type: standard")
    print(f"   Nodes: 3")
    print(f"   Duration: 5 minutes")
    print(f"   Estimated cost: ${cluster_cost:.6f}")
    
    # Calculate storage cost
    storage_cost = pricing_calc.calculate_storage_cost(
        data_size_gb=100.0,
        storage_duration_days=30,
        region="us-west-2"
    )
    
    print(f"✅ Storage cost calculation:")
    print(f"   Data size: 100 GB")
    print(f"   Duration: 30 days")
    print(f"   Estimated cost: ${storage_cost:.6f}")


def demonstrate_data_lineage() -> None:
    """Demonstrate data lineage tracking."""
    print("\n🔗 Demonstrating data lineage tracking...")
    
    governance_monitor = get_governance_monitor("demo_metastore_123")
    
    # Track data transformation lineage
    lineage_metrics = governance_monitor.track_data_lineage(
        lineage_type="transform",
        source_catalog="raw_data",
        source_schema="external",
        source_table="customer_events",
        target_catalog="processed_data",
        target_schema="analytics",
        target_table="customer_metrics",
        data_owner="data_team",
        data_steward="john_doe",
        data_classification="internal",
        user_id="analyst_123"
    )
    
    print(f"✅ Tracked data lineage: {lineage_metrics.lineage_type}")
    print(f"   Source: {lineage_metrics.source_catalog}.{lineage_metrics.source_schema}.{lineage_metrics.source_table}")
    print(f"   Target: {lineage_metrics.target_catalog}.{lineage_metrics.target_schema}.{lineage_metrics.target_table}")
    print(f"   Data owner: {lineage_metrics.data_owner}")
    print(f"   Classification: {lineage_metrics.data_classification}")
    
    # Get lineage graph
    lineage_graph = governance_monitor.get_lineage_graph()
    print(f"✅ Generated lineage graph with {len(lineage_graph)} relationships")


def demonstrate_workspace_cost_context() -> None:
    """Demonstrate workspace cost context tracking."""
    print("\n📈 Demonstrating workspace cost context...")
    
    workspace_id = "demo_workspace_456"
    
    with create_workspace_cost_context(workspace_id, "demo_analysis") as cost_context:
        print(f"✅ Started cost tracking context for workspace: {workspace_id}")
        
        # Simulate some operations (would normally be real Databricks operations)
        from genops.providers.databricks_unity_catalog.cost_aggregator import get_cost_aggregator
        
        cost_aggregator = get_cost_aggregator()
        
        # Add some SQL warehouse costs
        cost_aggregator.add_sql_warehouse_cost(
            workspace_id=workspace_id,
            warehouse_size="Small",
            query_duration_ms=3000,
            operation_type="analytics_query",
            team="analytics_team",
            project="customer_insights"
        )
        
        # Add compute cluster costs
        cost_aggregator.add_compute_cluster_cost(
            workspace_id=workspace_id,
            cluster_type="standard",
            node_count=2,
            duration_ms=120000,  # 2 minutes
            operation_type="etl_job",
            team="data_engineering",
            project="pipeline_processing"
        )
        
        print("✅ Added operation costs to context")
    
    # Get final summary
    final_summary = get_cost_aggregator().get_summary()
    print(f"✅ Final cost summary:")
    print(f"   Total cost: ${final_summary.total_cost_usd:.6f}")
    print(f"   Operations: {final_summary.operation_count}")
    print(f"   Workspaces: {len(final_summary.unique_workspaces)}")
    print(f"   Cost by team: {final_summary.cost_by_team}")


def main():
    """Main demonstration function."""
    print("🚀 Databricks Unity Catalog Basic Governance Tracking Demo")
    print("="*60)
    
    # Check configuration
    print("1️⃣ Checking configuration...")
    config = check_configuration()
    print("✅ Configuration validated:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize GenOps adapter
    print("\n2️⃣ Initializing GenOps Databricks Unity Catalog adapter...")
    adapter = instrument_databricks_unity_catalog(
        workspace_url=config['workspace_url']
    )
    print("✅ GenOps adapter initialized")
    
    # Set up governance attributes
    governance_attrs = {
        'team': config['team'],
        'project': config['project'],
        'environment': config['environment'],
        'user_id': 'demo_user',
        'cost_center': 'engineering'
    }
    
    try:
        # Demonstrate different types of operations
        demonstrate_catalog_operations(adapter, governance_attrs)
        demonstrate_table_operations(adapter, governance_attrs)
        demonstrate_sql_warehouse_operations(adapter, governance_attrs)
        demonstrate_cost_calculation()
        demonstrate_data_lineage()
        demonstrate_workspace_cost_context()
        
        print("\n🎉 Basic tracking demonstration completed successfully!")
        print("\n📚 What you learned:")
        print("   ✅ How to track Unity Catalog operations with governance attributes")
        print("   ✅ How to calculate costs for different Databricks resources")
        print("   ✅ How to track data lineage across catalogs and tables")
        print("   ✅ How to use workspace cost contexts for operation grouping")
        print("   ✅ How to attribute costs to teams and projects")
        
        print("\n🎯 Next steps:")
        print("   • Try 'python auto_instrumentation.py' for zero-code setup")
        print("   • Try 'python advanced_features.py' for comprehensive governance")
        print("   • Check the README.md for more examples")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        print(f"\n❌ Error: {e}")
        print("💡 Try running 'python setup_validation.py' to check your configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()