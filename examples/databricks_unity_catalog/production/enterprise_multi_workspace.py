#!/usr/bin/env python3
"""
Enterprise Multi-Workspace Governance Example

Demonstrates enterprise-grade governance across multiple Databricks workspaces
with unified cost tracking, cross-workspace lineage, and compliance automation.

This example shows:
- Multi-workspace governance coordination
- Unified cost reporting across regions/environments
- Cross-workspace data lineage tracking
- Enterprise compliance automation
- Automated policy enforcement
- Real-time governance monitoring
"""

import os
from dataclasses import dataclass

# Import GenOps Databricks Unity Catalog components
from genops.providers.databricks_unity_catalog import (
    create_workspace_cost_context,
    get_cost_aggregator,
    get_governance_monitor,
    instrument_databricks_unity_catalog,
)
from genops.providers.databricks_unity_catalog.registration import (
    configure_unity_catalog_governance,
)


@dataclass
class WorkspaceConfig:
    """Configuration for a Databricks workspace."""

    id: str
    name: str
    url: str
    region: str
    environment: str
    metastore_id: str
    business_unit: str
    cost_center: str


class EnterpriseMultiWorkspaceGovernance:
    """Enterprise multi-workspace governance coordinator."""

    def __init__(self):
        """Initialize enterprise governance across workspaces."""
        self.workspaces = self._load_workspace_configurations()
        self.adapters = {}
        self.cost_aggregator = get_cost_aggregator()
        self.setup_enterprise_governance()

    def _load_workspace_configurations(self) -> dict[str, WorkspaceConfig]:
        """Load workspace configurations from environment or config file."""

        # Enterprise workspace configuration
        workspace_configs = {
            "prod-us-west": WorkspaceConfig(
                id="prod-us-west",
                name="Production US West",
                url=os.getenv(
                    "DATABRICKS_PROD_US_WEST_URL",
                    "https://prod-us-west.cloud.databricks.com",
                ),
                region="us-west-2",
                environment="production",
                metastore_id=os.getenv(
                    "PROD_US_WEST_METASTORE_ID", "prod-us-west-metastore"
                ),
                business_unit="data-platform",
                cost_center="engineering",
            ),
            "prod-eu-central": WorkspaceConfig(
                id="prod-eu-central",
                name="Production EU Central",
                url=os.getenv(
                    "DATABRICKS_PROD_EU_URL",
                    "https://prod-eu-central.cloud.databricks.com",
                ),
                region="eu-central-1",
                environment="production",
                metastore_id=os.getenv(
                    "PROD_EU_METASTORE_ID", "prod-eu-central-metastore"
                ),
                business_unit="data-platform",
                cost_center="engineering",
            ),
            "prod-ap-south": WorkspaceConfig(
                id="prod-ap-south",
                name="Production Asia Pacific",
                url=os.getenv(
                    "DATABRICKS_PROD_AP_URL",
                    "https://prod-ap-south.cloud.databricks.com",
                ),
                region="ap-south-1",
                environment="production",
                metastore_id=os.getenv(
                    "PROD_AP_METASTORE_ID", "prod-ap-south-metastore"
                ),
                business_unit="data-platform",
                cost_center="engineering",
            ),
            "staging-global": WorkspaceConfig(
                id="staging-global",
                name="Global Staging",
                url=os.getenv(
                    "DATABRICKS_STAGING_URL",
                    "https://staging-global.cloud.databricks.com",
                ),
                region="us-west-2",
                environment="staging",
                metastore_id=os.getenv(
                    "STAGING_METASTORE_ID", "staging-global-metastore"
                ),
                business_unit="data-platform",
                cost_center="engineering",
            ),
            "dev-shared": WorkspaceConfig(
                id="dev-shared",
                name="Development Shared",
                url=os.getenv(
                    "DATABRICKS_DEV_URL", "https://dev-shared.cloud.databricks.com"
                ),
                region="us-west-2",
                environment="development",
                metastore_id=os.getenv("DEV_METASTORE_ID", "dev-shared-metastore"),
                business_unit="data-platform",
                cost_center="engineering",
            ),
        }

        return workspace_configs

    def setup_enterprise_governance(self):
        """Set up governance for all enterprise workspaces."""
        print("🏢 Setting up enterprise multi-workspace governance...")

        # Initialize adapters for each workspace
        for workspace_id, config in self.workspaces.items():
            print(f"   📊 Configuring governance for {config.name} ({config.region})")

            try:
                # Configure governance for workspace
                governance_config = configure_unity_catalog_governance(
                    workspace_url=config.url,
                    metastore_id=config.metastore_id,
                    enable_cross_workspace_lineage=True,
                    enable_unified_cost_reporting=True,
                    compliance_level="enterprise",
                )

                if governance_config["configured"]:
                    # Create adapter with enterprise settings
                    self.adapters[workspace_id] = instrument_databricks_unity_catalog(
                        workspace_url=config.url,
                        metastore_id=config.metastore_id,
                        # Enterprise governance attributes
                        team="data-platform-engineering",
                        project="enterprise-data-governance",
                        environment=config.environment,
                        business_unit=config.business_unit,
                        cost_center=config.cost_center,
                        region=config.region,
                        # Enterprise security settings
                        enable_rbac=True,
                        audit_all_operations=True,
                        encrypt_telemetry_data=True,
                        compliance_frameworks=["SOX", "GDPR", "CCPA"],
                        # Performance settings for enterprise
                        enable_sampling=True,
                        sampling_rate=0.1,  # 10% sampling for high volume
                        async_telemetry_export=True,
                        enable_cost_optimization=True,
                    )

                    print(f"     ✅ {config.name} governance configured successfully")
                else:
                    print(f"     ❌ Failed to configure governance for {config.name}")
                    print(f"        Errors: {governance_config['errors']}")

            except Exception as e:
                print(f"     ❌ Error configuring {config.name}: {str(e)}")

        print(f"   🎯 Successfully configured {len(self.adapters)} workspaces")

    def simulate_enterprise_data_operations(self):
        """Simulate realistic enterprise data operations across workspaces."""
        print("\n📈 Simulating enterprise data operations across workspaces...")

        # Define realistic enterprise data workflows
        enterprise_workflows = [
            {
                "name": "Global Customer Analytics",
                "workspaces": ["prod-us-west", "prod-eu-central", "prod-ap-south"],
                "operations": [
                    {
                        "type": "extract",
                        "data_source": "customer_events",
                        "volume_gb": 150,
                    },
                    {
                        "type": "transform",
                        "computation": "customer_segmentation",
                        "warehouse_size": "Large",
                    },
                    {
                        "type": "load",
                        "destination": "global_customer_insights",
                        "volume_gb": 45,
                    },
                ],
            },
            {
                "name": "Financial Compliance Reporting",
                "workspaces": ["prod-us-west", "staging-global"],
                "operations": [
                    {
                        "type": "extract",
                        "data_source": "financial_transactions",
                        "volume_gb": 300,
                    },
                    {
                        "type": "validate",
                        "compliance_check": "sox_controls",
                        "warehouse_size": "Medium",
                    },
                    {
                        "type": "aggregate",
                        "computation": "quarterly_reports",
                        "warehouse_size": "Large",
                    },
                    {
                        "type": "audit",
                        "audit_type": "sox_compliance",
                        "retention_years": 7,
                    },
                ],
            },
            {
                "name": "ML Feature Engineering",
                "workspaces": ["prod-us-west", "staging-global", "dev-shared"],
                "operations": [
                    {
                        "type": "extract",
                        "data_source": "user_behavior",
                        "volume_gb": 200,
                    },
                    {
                        "type": "feature_engineering",
                        "computation": "ml_features",
                        "warehouse_size": "XLarge",
                    },
                    {
                        "type": "validation",
                        "validation_type": "data_quality",
                        "warehouse_size": "Medium",
                    },
                    {
                        "type": "load",
                        "destination": "ml_feature_store",
                        "volume_gb": 80,
                    },
                ],
            },
        ]

        total_cost = 0.0
        total_operations = 0

        # Execute workflows across workspaces
        for workflow in enterprise_workflows:
            print(f"\n   🔄 Executing workflow: {workflow['name']}")

            workflow_cost = 0.0
            workflow_operations = 0

            for workspace_id in workflow["workspaces"]:
                if workspace_id not in self.adapters:
                    continue

                workspace_config = self.workspaces[workspace_id]
                adapter = self.adapters[workspace_id]

                print(f"     📍 Operations in {workspace_config.name}")

                # Create workspace cost context for this workflow
                with create_workspace_cost_context(workspace_id, workflow["name"]):
                    for operation in workflow["operations"]:
                        try:
                            if operation["type"] in ["extract", "load"]:
                                # Table operations
                                result = adapter.track_table_operation(
                                    operation=operation["type"],
                                    catalog_name="enterprise_data",
                                    schema_name=workflow["name"]
                                    .lower()
                                    .replace(" ", "_"),
                                    table_name=operation.get(
                                        "data_source",
                                        operation.get("destination", "unknown"),
                                    ),
                                    data_size_bytes=operation.get("volume_gb", 1)
                                    * 1024**3,
                                    team="data-platform-engineering",
                                    project="enterprise-data-governance",
                                    workflow=workflow["name"],
                                    compliance_classification="confidential",
                                )

                            elif operation["type"] in [
                                "transform",
                                "aggregate",
                                "feature_engineering",
                            ]:
                                # SQL warehouse operations
                                warehouse_size = operation.get(
                                    "warehouse_size", "Medium"
                                )
                                computation_complexity = {
                                    "Small": 1000,
                                    "Medium": 3000,
                                    "Large": 8000,
                                    "XLarge": 20000,
                                }
                                duration = computation_complexity.get(
                                    warehouse_size, 3000
                                )

                                result = adapter.track_sql_warehouse_operation(
                                    sql_warehouse_id=f"{workspace_id}-{warehouse_size.lower()}",
                                    query_type=operation["type"],
                                    query_duration_ms=duration,
                                    compute_units={
                                        "Small": 1,
                                        "Medium": 4,
                                        "Large": 16,
                                        "XLarge": 64,
                                    }[warehouse_size],
                                    team="data-platform-engineering",
                                    project="enterprise-data-governance",
                                    workflow=workflow["name"],
                                    operation_category="enterprise_analytics",
                                )

                            elif operation["type"] in ["validate", "audit"]:
                                # Governance operations
                                governance_monitor = get_governance_monitor(
                                    workspace_config.metastore_id
                                )

                                if operation["type"] == "validate":
                                    governance_monitor.track_compliance_audit(
                                        audit_type=operation.get(
                                            "compliance_check", "data_validation"
                                        ),
                                        resource_path=f"enterprise_data.{workflow['name'].lower().replace(' ', '_')}",
                                        compliance_status="pass",
                                        findings=["automated_validation_passed"],
                                        auditor_id="enterprise-governance-system",
                                    )

                                elif operation["type"] == "audit":
                                    governance_monitor.track_compliance_audit(
                                        audit_type=operation.get(
                                            "audit_type", "general_audit"
                                        ),
                                        resource_path=f"enterprise_data.{workflow['name'].lower().replace(' ', '_')}",
                                        compliance_status="compliant",
                                        findings=[
                                            "sox_controls_validated",
                                            "retention_policy_applied",
                                        ],
                                        retention_years=operation.get(
                                            "retention_years", 7
                                        ),
                                    )

                                # Create mock result for consistency
                                result = {
                                    "cost_usd": 0.001,
                                    "operation": operation["type"],
                                }

                            operation_cost = result.get("cost_usd", 0)
                            workflow_cost += operation_cost
                            workflow_operations += 1

                            print(
                                f"       ✅ {operation['type']} operation: ${operation_cost:.6f}"
                            )

                        except Exception as e:
                            print(
                                f"       ❌ Failed {operation['type']} operation: {str(e)}"
                            )

            total_cost += workflow_cost
            total_operations += workflow_operations

            print(f"     💰 Workflow total cost: ${workflow_cost:.4f}")
            print(f"     📊 Operations completed: {workflow_operations}")

        print("\n   🏆 Enterprise operations summary:")
        print(f"     💰 Total cost across all workspaces: ${total_cost:.4f}")
        print(f"     📊 Total operations: {total_operations}")
        print(f"     🌍 Workspaces utilized: {len(self.adapters)}")

    def generate_enterprise_cost_report(self):
        """Generate comprehensive cost report across all workspaces."""
        print("\n💰 Generating enterprise cost report...")

        # Get unified cost summary
        cost_summary = self.cost_aggregator.get_summary()

        print("   📊 Enterprise Cost Analysis:")
        print(f"     • Total enterprise cost: ${cost_summary.total_cost_usd:.4f}")
        print(f"     • Total operations tracked: {cost_summary.operation_count}")
        print(f"     • Active workspaces: {len(cost_summary.unique_workspaces)}")

        # Cost breakdown by workspace
        print("\n   🌍 Cost by Workspace:")
        for workspace_id, cost in cost_summary.cost_by_workspace.items():
            workspace_name = (
                self.workspaces.get(workspace_id, {}).name
                if workspace_id in self.workspaces
                else workspace_id
            )
            region = (
                self.workspaces.get(workspace_id, {}).region
                if workspace_id in self.workspaces
                else "unknown"
            )
            print(f"     • {workspace_name} ({region}): ${cost:.4f}")

        # Cost breakdown by team/project
        print("\n   👥 Cost by Team:")
        for team, cost in cost_summary.cost_by_team.items():
            print(f"     • {team}: ${cost:.4f}")

        print("\n   📁 Cost by Project:")
        for project, cost in cost_summary.cost_by_project.items():
            print(f"     • {project}: ${cost:.4f}")

        # Resource utilization analysis
        print("\n   🔧 Cost by Resource Type:")
        for resource_type, cost in cost_summary.cost_by_resource_type.items():
            percentage = (
                (cost / cost_summary.total_cost_usd) * 100
                if cost_summary.total_cost_usd > 0
                else 0
            )
            print(f"     • {resource_type}: ${cost:.4f} ({percentage:.1f}%)")

    def generate_cross_workspace_lineage_report(self):
        """Generate cross-workspace data lineage report."""
        print("\n🔗 Generating cross-workspace data lineage report...")

        lineage_summary = {
            "total_lineage_events": 0,
            "cross_workspace_lineage": 0,
            "data_classifications": {},
            "compliance_events": 0,
        }

        # Aggregate lineage data from all workspaces
        for workspace_id, config in self.workspaces.items():
            if workspace_id not in self.adapters:
                continue

            try:
                governance_monitor = get_governance_monitor(config.metastore_id)
                workspace_summary = governance_monitor.get_governance_summary()

                lineage_summary["total_lineage_events"] += (
                    workspace_summary.lineage_events
                )
                lineage_summary["compliance_events"] += (
                    workspace_summary.compliance_checks
                )

                # Merge data classifications
                for (
                    classification,
                    count,
                ) in workspace_summary.data_classifications.items():
                    lineage_summary["data_classifications"][classification] = (
                        lineage_summary["data_classifications"].get(classification, 0)
                        + count
                    )

                print(f"   📍 {config.name}:")
                print(f"     • Lineage events: {workspace_summary.lineage_events}")
                print(
                    f"     • Compliance checks: {workspace_summary.compliance_checks}"
                )

            except Exception as e:
                print(f"   ❌ Error getting lineage for {config.name}: {str(e)}")

        print("\n   🌐 Cross-Workspace Lineage Summary:")
        print(f"     • Total lineage events: {lineage_summary['total_lineage_events']}")
        print(f"     • Total compliance events: {lineage_summary['compliance_events']}")

        print("\n   🔒 Data Classification Distribution:")
        total_classified = sum(lineage_summary["data_classifications"].values())
        for classification, count in lineage_summary["data_classifications"].items():
            percentage = (count / total_classified) * 100 if total_classified > 0 else 0
            print(f"     • {classification}: {count} datasets ({percentage:.1f}%)")

    def run_enterprise_governance_demo(self):
        """Run complete enterprise governance demonstration."""
        print("🏢 GenOps Enterprise Multi-Workspace Governance Demo")
        print("=" * 60)
        print("📋 This demo showcases enterprise-grade governance across")
        print("    multiple Databricks workspaces with unified reporting.")
        print()

        # Setup phase
        if not self.adapters:
            print("❌ No workspaces configured successfully")
            return False

        # Simulate enterprise operations
        self.simulate_enterprise_data_operations()

        # Generate reports
        self.generate_enterprise_cost_report()
        self.generate_cross_workspace_lineage_report()

        # Enterprise governance summary
        print("\n🎯 Enterprise Governance Summary:")
        print(f"   ✅ Multi-workspace governance: {len(self.adapters)} workspaces")
        print("   ✅ Unified cost tracking: Real-time cost attribution")
        print("   ✅ Cross-workspace lineage: Complete data lineage")
        print("   ✅ Enterprise compliance: SOX, GDPR, CCPA automation")
        print("   ✅ Real-time monitoring: Continuous governance oversight")

        print("\n💡 Enterprise Value Delivered:")
        print(f"   • Unified governance across {len(self.workspaces)} regions")
        print("   • Automated compliance monitoring and reporting")
        print("   • Real-time cost optimization recommendations")
        print("   • Complete audit trail for regulatory compliance")
        print("   • Zero-disruption integration with existing workflows")

        print("\n📊 Next Steps for Production:")
        print("   • Configure alerting for cost thresholds")
        print("   • Set up automated compliance reporting")
        print("   • Implement cross-workspace policy enforcement")
        print("   • Enable real-time governance dashboards")

        return True


def main():
    """Main execution function."""
    try:
        # Initialize enterprise governance
        enterprise_governance = EnterpriseMultiWorkspaceGovernance()

        # Run the demo
        success = enterprise_governance.run_enterprise_governance_demo()

        if success:
            print("\n✨ Enterprise governance demo completed successfully!")
            print("🚀 Your enterprise is ready for production-scale data governance!")
        else:
            print("\n⚠️ Demo completed with issues - check configuration above")

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        print("💡 Check your workspace configurations and credentials")


if __name__ == "__main__":
    main()
