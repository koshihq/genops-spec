# 5-Minute Databricks Unity Catalog Quickstart

Get GenOps governance tracking working with Databricks Unity Catalog in **5 minutes or less**.

## What You'll Get

✅ **Real-time cost tracking** for SQL warehouses, compute clusters, and storage  
✅ **Data lineage governance** across catalogs, schemas, and tables  
✅ **Team-based cost attribution** with budget controls  
✅ **Zero code changes** to your existing Databricks applications  

## Prerequisites (30 seconds)

- Python 3.9+ installed
- Databricks workspace with Unity Catalog enabled
- Personal access token from your workspace

## Step 1: Install (30 seconds)

```bash
pip install genops[databricks]
```

## Step 2: Configure (60 seconds)

Set your Databricks credentials:

```bash
# Required - get these from your Databricks workspace
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your_personal_access_token"

# Optional - for team cost attribution
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name"
```

**💡 Quick credential setup:**
1. **Workspace URL**: Copy from your Databricks browser URL
2. **Access Token**: User Settings → Developer → Access Tokens → Generate New Token

## Step 3: Validate Setup (30 seconds)

```bash
# Download and run validation script
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/databricks_unity_catalog/setup_validation.py
python setup_validation.py
```

**Expected output:** `✅ Overall Status: PASSED`

## Step 4: Add Governance to Your Code (60 seconds)

### Option A: Zero-Code Auto-Instrumentation (Recommended)

Add **one line** to your existing Databricks code:

```python
# Add this single line at the top of your existing code
from genops.providers.databricks_unity_catalog.registration import auto_instrument_databricks
auto_instrument_databricks()

# Your existing Databricks code works unchanged
# All operations now have automatic governance tracking
```

### Option B: Manual Instrumentation

```python
from genops.providers.databricks_unity_catalog import instrument_databricks_unity_catalog

# Initialize GenOps governance
adapter = instrument_databricks_unity_catalog()

# Track any Unity Catalog operation
adapter.track_table_operation(
    operation="query",
    catalog_name="production",
    schema_name="analytics", 
    table_name="customer_data",
    team="data-team",
    project="customer-insights"
)
```

## Step 5: See Your Governance Data (60 seconds)

Run this to see immediate results:

```python
from genops.providers.databricks_unity_catalog import get_cost_aggregator

# Get real-time cost summary
cost_summary = get_cost_aggregator().get_summary()

print(f"💰 Total cost: ${cost_summary.total_cost_usd:.4f}")
print(f"📊 Operations: {cost_summary.operation_count}")
print(f"👥 Teams: {list(cost_summary.cost_by_team.keys())}")
```

**Expected output:**
```
💰 Total cost: $0.0045
📊 Operations: 3
👥 Teams: ['data-team', 'analytics-team']
```

## Step 6: View in Your Observability Stack (30 seconds)

GenOps exports standard OpenTelemetry data. View your governance telemetry in:

- **Datadog**: Traces → Services → `genops.databricks.unity_catalog`
- **Grafana**: Explore → Traces → Search `genops.provider:databricks_unity_catalog`
- **Honeycomb**: Query → `genops.provider = "databricks_unity_catalog"`

## ✅ Success! What You Just Enabled

🎉 **Congratulations!** Your Databricks Unity Catalog now has enterprise-grade governance:

✅ **Cost Tracking**: Every query, compute job, and storage operation tracked  
✅ **Data Lineage**: Automatic lineage capture across all catalogs  
✅ **Team Attribution**: Costs attributed to teams and projects  
✅ **Policy Enforcement**: Automated governance controls  
✅ **Real-time Monitoring**: Live telemetry in your existing dashboards  

## Next Steps (Optional)

### 🚀 Immediate Actions

```bash
# Try the complete example suite
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/databricks_unity_catalog/basic_tracking.py
python basic_tracking.py
```

### 📚 Learn More

- **[Complete Examples](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/databricks_unity_catalog)** - Advanced features and production patterns
- **[Integration Guide](integrations/databricks-unity-catalog.md)** - Comprehensive documentation
- **[Community Support](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Questions and discussions

### 🏭 Production Deployment

Ready for production? Set up enterprise governance:

```python
# Configure enterprise governance
from genops.providers.databricks_unity_catalog.registration import configure_unity_catalog_governance

configure_unity_catalog_governance(
    enable_compliance_monitoring=True,
    enable_automated_policy_enforcement=True,
    default_budget_limits={"daily": 100.0, "monthly": 2500.0}
)
```

## Troubleshooting

**❌ "DATABRICKS_HOST not set"**
```bash
# Make sure workspace URL is correct
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
```

**❌ "Authentication failed"**
```bash
# Generate new access token: User Settings → Developer → Access Tokens
export DATABRICKS_TOKEN="dapi1234567890abcdef"
```

**❌ "Unity Catalog not accessible"**  
- Ensure Unity Catalog is enabled in your workspace
- Verify your user has Unity Catalog permissions

**❌ Still having issues?**
- 📧 [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues) with error details
- 💬 [Community Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions) for questions

---

**⏱️ Total time:** Under 5 minutes  
**⚡ Zero code changes** to your existing Databricks applications  
**🎯 Immediate value:** Real-time governance for all your Unity Catalog operations