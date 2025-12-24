# Databricks Unity Catalog Integration Guide

> **📍 Navigation**: [Examples](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/databricks_unity_catalog) → [Quickstart](https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/databricks-unity-catalog-quickstart.md) → **Integration Guide** → [Production Deployment](https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/production/databricks-unity-catalog-deployment.md)

Complete guide for implementing GenOps governance with Databricks Unity Catalog across enterprise data platforms.

## Overview

This integration provides comprehensive data governance, cost tracking, and compliance automation for Databricks Unity Catalog environments. It extends GenOps governance capabilities to data platforms, enabling unified telemetry across SQL warehouses, compute clusters, storage operations, and ML workloads.

### Key Capabilities

**🏛️ Data Governance Excellence**
- **Complete data lineage tracking** across catalogs, schemas, and tables
- **Automated compliance monitoring** with PII detection and retention policies
- **Policy enforcement** with real-time governance controls
- **Audit trail generation** for regulatory compliance

**💰 Enterprise Cost Intelligence**
- **Multi-workspace cost attribution** with team and project tracking
- **Real-time budget enforcement** across SQL warehouses and compute clusters
- **Resource optimization recommendations** based on usage patterns
- **Cost forecasting** and capacity planning support

**🚀 Production-Ready Integration**
- **Zero-code auto-instrumentation** for existing Databricks applications
- **OpenTelemetry-native telemetry** compatible with 15+ observability platforms
- **High-availability deployment patterns** with multi-workspace failover
- **Enterprise security and compliance** with role-based access controls

## Architecture

### Provider Structure

GenOps Databricks Unity Catalog follows the standard 4-module provider architecture:

```
src/genops/providers/databricks_unity_catalog/
├── adapter.py                 # Main GenOpsDatabricksUnityCatalogAdapter
├── cost_aggregator.py         # Multi-workspace cost tracking
├── governance_monitor.py      # Data lineage and compliance automation
└── registration.py            # Auto-instrumentation and provider registry
```

### Framework Integration

**Framework Type:** `FRAMEWORK_TYPE_DATA_PLATFORM`  
**Provider Name:** `databricks_unity_catalog`  
**Auto-Detection Modules:** `databricks`, `databricks.sdk`, `pyspark`

### Telemetry Schema

All telemetry follows OpenTelemetry standards with GenOps governance extensions:

```json
{
  "span_name": "genops.databricks.unity_catalog.table.query",
  "attributes": {
    "genops.provider": "databricks_unity_catalog",
    "genops.framework_type": "data_platform",
    "genops.operation_type": "table.query",
    "genops.catalog_name": "production",
    "genops.schema_name": "analytics",
    "genops.table_name": "customer_data",
    "genops.cost.total": 0.045,
    "genops.cost.currency": "USD",
    "genops.cost.resource_type": "sql_warehouse",
    "genops.team": "data-engineering",
    "genops.project": "customer-insights",
    "genops.data.row_count": 15000,
    "genops.data.size_bytes": 52428800,
    "genops.governance.policy_check": "passed",
    "genops.governance.data_classification": "confidential",
    "genops.governance.lineage_upstream": "raw_data.events.user_actions",
    "genops.governance.lineage_downstream": "analytics.reports.daily_metrics"
  }
}
```

## Installation & Configuration

### Dependencies

```bash
# Core installation
pip install genops[databricks]

# Optional dependencies for enhanced features
pip install databricks-sdk>=0.18.0  # Latest Databricks SDK
pip install pyspark>=3.4.0          # For Spark integration
```

### Environment Configuration

**Required Environment Variables:**

```bash
# Databricks workspace configuration
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your_personal_access_token"

# Optional: Unity Catalog metastore (auto-detected if not provided)
export DATABRICKS_METASTORE_ID="your_metastore_id"
```

**Governance Attributes (Recommended):**

```bash
# Team and project attribution
export GENOPS_TEAM="data-engineering"
export GENOPS_PROJECT="customer-analytics"
export GENOPS_ENVIRONMENT="production"
export GENOPS_COST_CENTER="engineering"

# Data governance settings
export GENOPS_DEFAULT_DATA_CLASSIFICATION="internal"
export GENOPS_ENABLE_PII_DETECTION="true"
export GENOPS_ENABLE_AUDIT_LOGGING="true"
```

### Configuration Validation

Validate your setup before using:

```python
from genops.providers.databricks_unity_catalog.validation import validate_setup, print_validation_result

result = validate_setup(check_connectivity=True, check_governance=True)
print_validation_result(result)
```

## Integration Patterns

### Pattern 1: Auto-Instrumentation (Recommended)

**Zero-code integration** for existing Databricks applications:

```python
# Add this single line to enable governance for all Databricks operations
from genops.providers.databricks_unity_catalog.registration import auto_instrument_databricks
auto_instrument_databricks()

# Your existing code now has automatic governance tracking
from databricks.sdk import WorkspaceClient

client = WorkspaceClient()
catalogs = client.catalogs.list()  # ← Automatically tracked
```

### Pattern 2: Manual Instrumentation

**Explicit governance tracking** for specific operations:

```python
from genops.providers.databricks_unity_catalog import instrument_databricks_unity_catalog

# Initialize adapter
adapter = instrument_databricks_unity_catalog()

# Track catalog operations
adapter.track_catalog_operation(
    operation="read",
    catalog_name="production",
    team="data-engineering",
    project="customer-analytics"
)

# Track table operations with detailed metrics
adapter.track_table_operation(
    operation="query",
    catalog_name="production",
    schema_name="analytics",
    table_name="customer_events",
    row_count=50000,
    data_size_bytes=1024*1024*100,  # 100 MB
    team="analytics-team",
    project="customer-insights",
    data_classification="confidential"
)
```

### Pattern 3: Context Management

**Workflow-level governance** with automatic cost aggregation:

```python
from genops.providers.databricks_unity_catalog import create_workspace_cost_context

workspace_id = "production-workspace-123"

with create_workspace_cost_context(workspace_id, "etl-pipeline") as cost_context:
    # All operations within this context are automatically tracked
    # and costs are aggregated for the entire workflow
    
    # Extract data
    adapter.track_table_operation("read", "raw", "events", "user_actions")
    
    # Transform data  
    adapter.track_sql_warehouse_operation(
        sql_warehouse_id="analytics-warehouse",
        query_type="transform",
        query_duration_ms=45000,
        compute_units=2.5
    )
    
    # Load data
    adapter.track_table_operation("write", "processed", "analytics", "user_metrics")

# Automatic cost summary available after context completion
```

### Pattern 4: Data Lineage Tracking

**Comprehensive lineage monitoring** for compliance and governance:

```python
from genops.providers.databricks_unity_catalog import get_governance_monitor

governance_monitor = get_governance_monitor("metastore-123")

# Track data transformation lineage
lineage_metrics = governance_monitor.track_data_lineage(
    lineage_type="transform",
    source_catalog="raw_data",
    source_schema="events", 
    source_table="user_sessions",
    target_catalog="analytics",
    target_schema="aggregated",
    target_table="session_metrics",
    transformation_logic="GROUP BY user_id, DATE(session_start)",
    data_owner="data-team",
    data_steward="jane.doe@company.com",
    data_classification="internal",
    compliance_tags=["gdpr", "ccpa"]
)

# Generate lineage graph for visualization
lineage_graph = governance_monitor.get_lineage_graph("analytics")
```

## Advanced Features

### Cost Optimization

**Intelligent cost analysis and recommendations:**

```python
from genops.providers.databricks_unity_catalog_pricing import get_pricing_calculator

pricing_calc = get_pricing_calculator()

# Estimate query costs before execution
estimated_cost = pricing_calc.estimate_query_cost(
    query_complexity="complex",
    data_scanned_gb=250.0,
    warehouse_size="Large",
    region="us-west-2"
)

print(f"Estimated cost: ${estimated_cost:.4f}")

# Get cost optimization recommendations
cost_aggregator = get_cost_aggregator()
summary = cost_aggregator.get_summary()

if summary.get_cost_efficiency_score() < 50:
    print("⚠️ Low efficiency detected - consider:")
    print("  • Using smaller warehouse for simple queries")
    print("  • Implementing query result caching")
    print("  • Optimizing data partitioning")
```

### Compliance Automation

**Automated governance and policy enforcement:**

```python
from genops.providers.databricks_unity_catalog import get_governance_monitor

governance_monitor = get_governance_monitor()

# Automated policy enforcement
policy_result = governance_monitor.enforce_data_classification_policy(
    catalog="customer_data",
    schema="pii", 
    table="user_profiles",
    required_classification="confidential",
    user_clearance="confidential"  # User's clearance level
)

if policy_result["access_granted"]:
    # Proceed with operation
    print("✅ Access granted - proceeding with query")
else:
    # Log violation and deny access
    print("❌ Access denied - insufficient clearance")

# Automated compliance auditing
audit_result = governance_monitor.track_compliance_audit(
    audit_type="pii_scan",
    resource_path="customer_data.profiles.users",
    compliance_status="pass",
    findings=["encrypted_email_column", "masked_phone_numbers"]
)
```

### Multi-Workspace Governance

**Enterprise-scale governance across multiple workspaces:**

```python
# Configure governance for multiple workspaces
workspaces = [
    {"id": "prod-us-west", "url": "https://prod-us-west.cloud.databricks.com"},
    {"id": "prod-eu-central", "url": "https://prod-eu-central.cloud.databricks.com"},
    {"id": "staging", "url": "https://staging.cloud.databricks.com"}
]

governance_configs = []
for workspace in workspaces:
    config = configure_unity_catalog_governance(
        workspace_url=workspace["url"],
        enable_cross_workspace_lineage=True,
        enable_unified_cost_reporting=True,
        compliance_level="enterprise"
    )
    governance_configs.append(config)

# Unified cost reporting across all workspaces
total_costs = {}
for config in governance_configs:
    if config["configured"]:
        workspace_costs = get_cost_aggregator().get_workspace_costs()
        for workspace_id, cost in workspace_costs.items():
            total_costs[workspace_id] = total_costs.get(workspace_id, 0) + cost

print(f"Total multi-workspace costs: ${sum(total_costs.values()):.2f}")
```

## Production Deployment

### High-Availability Configuration

**Multi-workspace deployment with failover:**

```python
from genops.providers.databricks_unity_catalog.registration import configure_unity_catalog_governance

# Primary workspace configuration
primary_config = configure_unity_catalog_governance(
    workspace_url="https://primary.cloud.databricks.com",
    metastore_id="primary-metastore",
    enable_high_availability=True,
    failover_workspace_url="https://secondary.cloud.databricks.com",
    sync_interval_seconds=30
)

# Health monitoring and automatic failover
if not primary_config["configured"]:
    print("⚠️ Primary workspace unavailable - failing over to secondary")
    secondary_config = configure_unity_catalog_governance(
        workspace_url="https://secondary.cloud.databricks.com",
        metastore_id="secondary-metastore"
    )
```

### Enterprise Security

**Role-based access controls and compliance:**

```python
# Configure enterprise security settings
enterprise_config = {
    "enable_rbac": True,
    "audit_all_operations": True,
    "encrypt_telemetry_data": True,
    "compliance_frameworks": ["SOX", "GDPR", "CCPA"],
    "data_retention_days": 2555,  # 7 years for financial compliance
    "enable_data_masking": True,
    "pii_detection_enabled": True
}

adapter = instrument_databricks_unity_catalog(**enterprise_config)
```

### Performance Optimization

**Configuration for high-volume workloads:**

```python
# Configure for high-performance environments
performance_config = {
    "enable_sampling": True,
    "sampling_rate": 0.1,  # Sample 10% of operations
    "async_telemetry_export": True,
    "batch_size": 1000,
    "flush_interval_seconds": 30,
    "enable_compression": True
}

adapter = instrument_databricks_unity_catalog(**performance_config)
```

## Observability Integration

### Datadog Integration

```python
from opentelemetry.exporter.datadog import DatadogExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Datadog exporter for Unity Catalog telemetry
datadog_exporter = DatadogExporter(
    agent_url="http://localhost:8126",
    service="databricks-unity-catalog"
)

span_processor = BatchSpanProcessor(datadog_exporter)
```

### Grafana Dashboard Configuration

```yaml
# Grafana dashboard configuration for Unity Catalog governance
dashboard:
  title: "Unity Catalog Governance Dashboard"
  panels:
    - title: "Cost by Team"
      type: "graph"
      query: 'sum by (genops_team) (genops_cost_total{genops_provider="databricks_unity_catalog"})'
    
    - title: "Data Lineage Operations"
      type: "table"
      query: 'genops_data_lineage{genops_provider="databricks_unity_catalog"}'
    
    - title: "Compliance Violations"
      type: "alert-list"
      query: 'genops_governance_violation{genops_provider="databricks_unity_catalog"}'
```

### Custom Metrics and Alerts

```python
# Configure custom metrics for specific business needs
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

# Custom business metrics
data_quality_score = meter.create_gauge(
    name="genops.databricks.data_quality_score",
    description="Data quality score for Unity Catalog tables"
)

governance_compliance_rate = meter.create_gauge(
    name="genops.databricks.governance_compliance_rate", 
    description="Governance compliance rate across catalogs"
)

# Update metrics based on governance monitoring
governance_summary = governance_monitor.get_governance_summary()
compliance_rate = (
    (governance_summary.schema_validation_pass) / 
    (governance_summary.schema_validation_pass + governance_summary.schema_validation_fail)
) * 100

governance_compliance_rate.set(compliance_rate, {"metastore_id": "production"})
```

## Migration Guides

### From Apache Atlas

```python
# Migration helper for Apache Atlas lineage
def migrate_atlas_lineage_to_genops(atlas_lineage_data):
    governance_monitor = get_governance_monitor()
    
    for lineage_entry in atlas_lineage_data:
        governance_monitor.track_data_lineage(
            lineage_type=lineage_entry["process_type"],
            source_catalog=lineage_entry["input_dataset"]["catalog"],
            source_schema=lineage_entry["input_dataset"]["schema"],
            source_table=lineage_entry["input_dataset"]["table"],
            target_catalog=lineage_entry["output_dataset"]["catalog"],
            target_schema=lineage_entry["output_dataset"]["schema"], 
            target_table=lineage_entry["output_dataset"]["table"],
            transformation_logic=lineage_entry.get("process_sql"),
            data_classification=lineage_entry.get("classification", "internal")
        )
```

### From AWS Glue Data Catalog

```python
# Migration helper for AWS Glue lineage
def migrate_glue_catalog_to_unity_catalog(glue_tables):
    adapter = instrument_databricks_unity_catalog()
    
    for table in glue_tables:
        # Map Glue table structure to Unity Catalog
        adapter.track_table_operation(
            operation="migrate",
            catalog_name="migrated_from_glue",
            schema_name=table["DatabaseName"],
            table_name=table["Name"],
            data_classification=table.get("Classification", "internal"),
            team="migration-team",
            project="glue-to-unity-migration"
        )
```

## Troubleshooting

### Common Issues

**Connection Problems:**
```python
# Test connectivity to Databricks workspace
from databricks.sdk import WorkspaceClient

try:
    client = WorkspaceClient()
    user = client.current_user.me()
    print(f"✅ Connected as: {user.user_name}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    # Check DATABRICKS_HOST and DATABRICKS_TOKEN
```

**Unity Catalog Access:**
```python
# Verify Unity Catalog permissions
try:
    client = WorkspaceClient()
    catalogs = list(client.catalogs.list())
    print(f"✅ Unity Catalog access: {len(catalogs)} catalogs available")
except Exception as e:
    print(f"❌ Unity Catalog access failed: {e}")
    # Ensure Unity Catalog is enabled and user has permissions
```

**Performance Issues:**
```python
# Enable debug logging for performance analysis
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor telemetry export performance
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
console_exporter = ConsoleSpanExporter()
```

### Debug Configuration

```python
# Enable comprehensive debug logging
debug_config = {
    "log_level": "DEBUG",
    "enable_trace_correlation": True,
    "export_traces_to_console": True,
    "validate_telemetry_schema": True,
    "enable_performance_metrics": True
}

adapter = instrument_databricks_unity_catalog(**debug_config)
```

## API Reference

### Core Classes

**GenOpsDatabricksUnityCatalogAdapter**
- `track_catalog_operation(operation, catalog_name, **governance_attrs)`
- `track_table_operation(operation, catalog_name, schema_name, table_name, **metrics)`
- `track_sql_warehouse_operation(warehouse_id, query_type, **performance_metrics)`

**DatabricksUnityCatalogCostAggregator**
- `add_sql_warehouse_cost(workspace_id, warehouse_size, duration_ms, **attrs)`
- `add_compute_cluster_cost(workspace_id, cluster_type, node_count, duration_ms, **attrs)`
- `get_summary() -> DatabricksCostSummary`

**DatabricksGovernanceMonitor**
- `track_data_lineage(lineage_type, source_*, target_*, **governance_attrs)`
- `enforce_data_classification_policy(catalog, schema, table, required_classification, user_clearance)`
- `track_compliance_audit(audit_type, resource_path, compliance_status, findings)`

### Utility Functions

**Registration & Configuration**
- `instrument_databricks_unity_catalog(workspace_url, **config) -> Adapter`
- `auto_instrument_databricks() -> Optional[Adapter]`
- `configure_unity_catalog_governance(**config) -> Dict[str, Any]`

**Validation & Diagnostics**
- `validate_setup(workspace_url, check_connectivity, check_governance) -> ValidationResult`
- `print_validation_result(result: ValidationResult) -> None`

## Support & Community

### Getting Help

- **📚 Quick Start**: [5-Minute Quickstart Guide](https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/databricks-unity-catalog-quickstart.md)
- **💬 Community Support**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **📖 Examples**: [Complete Example Suite](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/databricks_unity_catalog/)

### Contributing

We welcome contributions! Common contribution areas:
- Additional cost optimization algorithms
- Enhanced compliance automation features  
- New observability platform integrations
- Performance improvements for high-volume workloads

See [CONTRIBUTING.md](https://github.com/KoshiHQ/GenOps-AI/blob/main/CONTRIBUTING.md) for development setup and guidelines.

---

**Next Steps**: Ready to get started? Try the [5-Minute Quickstart Guide](https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/databricks-unity-catalog-quickstart.md) or explore the [complete example suite](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/databricks_unity_catalog/).