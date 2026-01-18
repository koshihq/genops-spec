# Kibana Dashboards for GenOps AI Governance

This directory contains comprehensive guides for creating Kibana dashboards for visualizing GenOps AI governance telemetry in Elasticsearch.

**Note:** Pre-built NDJSON dashboard files are planned for a future release. In the meantime, this guide provides complete instructions for manually creating dashboards tailored to your organization's needs.

## Recommended Dashboards

### 1. AI Operations Overview
**Status:** Manual creation guide (pre-built NDJSON planned for future release)

**Visualizations:**
- Request volume over time (line chart)
- Success vs error rates (pie chart)
- Latency percentiles - p50, p95, p99 (multi-line chart)
- Top operations by volume (bar chart)
- Operations by provider (pie chart)
- Operations by model (table)

**Use Cases:**
- Monitor overall AI system health
- Identify performance bottlenecks
- Track usage patterns over time
- Compare provider/model performance

### 2. Cost Attribution
**Status:** Manual creation guide (pre-built NDJSON planned for future release)

**Visualizations:**
- Total cost by team (bar chart)
- Total cost by project (bar chart)
- Cost by model and provider (heat map)
- Cost trends over time (area chart)
- Top cost drivers (table with cost, operations, avg cost)
- Cost by customer (bar chart)
- Daily/weekly/monthly cost aggregations

**Use Cases:**
- FinOps cost tracking and attribution
- Budget planning and forecasting
- Identify cost optimization opportunities
- Chargeback/showback reporting

### 3. Governance & Compliance
**Status:** Manual creation guide (pre-built NDJSON planned for future release)

**Visualizations:**
- Policy violations by type (bar chart)
- Policy violations over time (line chart)
- Budget consumption tracking (gauge/progress bars)
- Compliance status by team (heat map)
- Alert summary (table with policy, result, count)
- Policy enforcement rate (success vs blocked vs warning)
- Budget alerts (near-limit warnings)

**Use Cases:**
- Security and compliance monitoring
- Policy effectiveness analysis
- Budget enforcement tracking
- Governance audit trails

---

## Dashboard Setup Instructions

### Prerequisites

1. **Elasticsearch 8.x or 9.x** with GenOps data indexed
2. **Kibana** connected to your Elasticsearch cluster
3. **Index Pattern** created: `genops-ai-*` (with `timestamp` as time field)

### Create Index Pattern

Before importing dashboards, ensure you have the GenOps index pattern:

1. Navigate to: **Management → Stack Management → Index Patterns**
2. Click **Create index pattern**
3. Enter pattern: `genops-ai-*`
4. Click **Next step**
5. Select **timestamp** as the time field
6. Click **Create index pattern**

---

## Manual Dashboard Creation

If you prefer to create custom dashboards, follow these guidelines:

### Step 1: Navigate to Dashboard

1. Open Kibana
2. Click **Dashboard** in the left sidebar
3. Click **Create dashboard**

### Step 2: Add Visualizations

Click **Create visualization** and choose from:

#### Useful Visualization Types

**For Cost Analysis:**
- **Bar Chart**: Cost by team, project, or model
- **Area Chart**: Cost trends over time
- **Heat Map**: Cost by model and provider
- **Metric**: Total cost, average cost per operation
- **Table**: Top cost drivers with multiple metrics

**For Operations Monitoring:**
- **Line Chart**: Request volume over time
- **Pie Chart**: Success vs error rates, operations by provider
- **Bar Chart**: Top operations by volume
- **Gauge**: Error rate percentage, success rate
- **TSVB (Time Series Visual Builder)**: Advanced time series with multiple metrics

**For Governance:**
- **Bar Chart**: Policy violations by type
- **Line Chart**: Policy violations over time
- **Gauge**: Budget consumption percentage
- **Table**: Policy enforcement details
- **Markdown**: Custom text, alerts, and instructions

### Step 3: Configure Visualizations

#### Example: Cost by Team (Bar Chart)

1. Create new visualization → **Bar chart**
2. Select index pattern: `genops-ai-*`
3. Configure axes:
   - **Y-axis**: Aggregation: `Sum`, Field: `genops.cost.total`
   - **X-axis**: Aggregation: `Terms`, Field: `genops.team.keyword`, Size: 10
4. Add filters if needed: `genops.cost.total > 0`
5. Set time range: Last 7 days, 30 days, etc.
6. Click **Save**

#### Example: Operations Over Time (Line Chart)

1. Create new visualization → **Line chart**
2. Select index pattern: `genops-ai-*`
3. Configure:
   - **Y-axis**: Aggregation: `Count`
   - **X-axis**: Aggregation: `Date Histogram`, Field: `timestamp`, Interval: `Auto`
4. Add split series (optional):
   - **Split series**: Aggregation: `Terms`, Field: `genops.cost.provider.keyword`
5. Click **Save**

#### Example: Budget Consumption (Gauge)

1. Create new visualization → **Gauge**
2. Select index pattern: `genops-ai-*`
3. Configure:
   - **Metric**: Aggregation: `Max`, Field: `genops.budget.consumed`
   - **Max value**: Aggregation: `Max`, Field: `genops.budget.limit`
4. Add filter: `genops.budget.id: "team-monthly"`
5. Configure gauge ranges:
   - Green: 0-70%
   - Yellow: 70-90%
   - Red: 90-100%
6. Click **Save**

### Step 4: Save Dashboard

1. Click **Save** in the top toolbar
2. Enter dashboard name (e.g., "GenOps Cost Attribution")
3. Add description (optional)
4. Click **Save**

---

## KQL Query Examples for Dashboards

Use these KQL queries as filters or in saved searches:

### Cost Queries

```kql
# All cost data
genops.cost.total > 0

# High-cost operations (>$1)
genops.cost.total > 1.0

# Specific team
genops.team: "ml-platform" AND genops.cost.total > 0

# Specific customer
genops.customer_id: "acme-corp" AND genops.cost.total > 0

# Specific model
genops.cost.model: "gpt-4" OR genops.cost.model: "claude-3-sonnet"
```

### Policy Queries

```kql
# All policy violations
genops.policy.result: "blocked" OR genops.policy.result: "warning"

# Specific policy
genops.policy.name: "budget-constraint"

# Blocked operations
genops.policy.result: "blocked"

# Policy violations by team
genops.team: "ml-platform" AND genops.policy.result: "blocked"
```

### Performance Queries

```kql
# High-latency operations (>1s)
duration_ms > 1000

# Errors
status: "error"

# Specific provider performance
genops.cost.provider: "openai" AND duration_ms > 0
```

### Budget Queries

```kql
# Budget tracking
genops.budget.id: *

# Near-budget alerts (remaining < $100)
genops.budget.remaining > 0 AND genops.budget.remaining < 100

# Specific budget
genops.budget.id: "team-monthly"
```

---

## Dashboard Customization Tips

### Time Range Configuration

- **Relative**: Last 15 minutes, 1 hour, 24 hours, 7 days, 30 days
- **Absolute**: Specific date range for historical analysis
- **Quick select**: Presets in top-right corner of Kibana

### Refresh Interval

- **Manual**: Refresh on demand
- **Auto-refresh**: 10s, 30s, 1m, 5m for real-time monitoring
- Configure in top toolbar: 🔄 icon

### Dashboard Filters

Add global filters to entire dashboard:
1. Click **Add filter** in dashboard toolbar
2. Configure filter (field, operator, value)
3. Apply to all visualizations

Example filters:
- `genops.environment: production`
- `genops.team: ml-platform`
- `genops.cost.provider: openai`

### Drill-Downs

Enable drill-down from visualizations:
1. Edit visualization
2. Click data point
3. Configure **Action** → **Apply filter** or **Navigate to dashboard**

### Alerts (Kibana Alerting)

Create alerts for:
- High cost operations (>$10)
- Policy violations
- Budget threshold (80%, 90%, 100%)
- Error rate spikes

**Setup:**
1. Navigate to: **Stack Management → Rules**
2. Click **Create rule**
3. Configure trigger conditions (query, threshold, time window)
4. Configure actions (email, Slack, webhook)

---

## Troubleshooting

### Dashboard Shows No Data

**Solutions:**
1. Verify index pattern exists: `genops-ai-*`
2. Check time range (top-right corner) - set to "Last 7 days" or broader
3. Verify data exists: Run query in **Discover**: `genops.cost.total > 0`
4. Force refresh index: **Stack Management → Index Patterns → genops-ai-* → Refresh**

### Visualizations Not Loading

**Solutions:**
1. Check Elasticsearch cluster health: `GET /_cluster/health`
2. Verify field mappings match (e.g., `genops.cost.total` is type `float`)
3. Check for query errors in visualization editor
4. Clear Kibana cache: Browser developer tools → Clear site data

### Import Fails

**Solutions:**
1. Verify Kibana version compatibility (8.x or 9.x)
2. Check NDJSON file format (one JSON object per line)
3. Resolve index pattern conflicts before import
4. Try **Create new objects** instead of **Overwrite**

### Performance Issues

**Solutions:**
1. Reduce time range for large datasets
2. Add filters to limit query scope
3. Increase Elasticsearch cluster resources
4. Optimize index mappings (disable dynamic mapping if not needed)
5. Use index lifecycle management to archive old data

---

## Dashboard Maintenance

### Regular Updates

1. **Review and update time ranges**: Adjust based on data retention
2. **Add new visualizations**: As new metrics become relevant
3. **Archive old dashboards**: Version control for dashboard evolution
4. **Export regularly**: Backup dashboards to version control

### Version Control

Export dashboards regularly:

```bash
# Export all dashboards
curl "${KIBANA_URL}/api/saved_objects/_export" \
  -H "kbn-xsrf: true" \
  -d '{"type": "dashboard"}' \
  > genops-dashboards-backup-$(date +%Y%m%d).ndjson
```

### Team Sharing

1. **Export dashboards**: Save as NDJSON files
2. **Commit to repository**: Version control for team collaboration
3. **Document customizations**: README or inline dashboard descriptions
4. **Set permissions**: Kibana Spaces for team isolation

---

## Next Steps

1. **Create index pattern** (`genops-ai-*`) following the guide above
2. **Build custom dashboards** using the manual creation guide
3. **Start with essential visualizations** (cost by team, operations over time)
4. **Set up alerts** for critical thresholds (high costs, policy violations)
5. **Share with team** via Kibana Spaces
6. **Iterate based on feedback** (what metrics are most valuable?)
7. **Export and version control** your dashboards for team collaboration

**Future:** Pre-built NDJSON dashboard files will be available in a future release for one-click import.

---

## Resources

- **Kibana Visualizations Guide**: [elastic.co/guide/en/kibana/current/dashboard.html](https://www.elastic.co/guide/en/kibana/current/dashboard.html)
- **KQL Syntax**: [elastic.co/guide/en/kibana/current/kuery-query.html](https://www.elastic.co/guide/en/kibana/current/kuery-query.html)
- **GenOps Documentation**: [docs/integrations/elastic.md](../../../docs/integrations/elastic.md)
- **Example Integration**: [examples/observability/elastic_integration.py](../../../examples/observability/elastic_integration.py)

---

## Support

For issues or questions:
- **GitHub Issues**: [github.com/KoshiHQ/GenOps-AI/issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [github.com/KoshiHQ/GenOps-AI/discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
