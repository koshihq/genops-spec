# Splunk Integration - 5-Minute Quickstart

**🎯 Get GenOps + Splunk governance monitoring in 5 minutes**

This guide gets you from zero to routing GenOps AI governance telemetry to Splunk HEC for enterprise analytics, compliance monitoring, and cost attribution in under 5 minutes.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Splunk Enterprise v8.0+ or Splunk Cloud** installed and running
   - Get Splunk Enterprise: [https://www.splunk.com/en_us/download.html](https://www.splunk.com/en_us/download.html)
   - Or use Splunk Cloud: [https://www.splunk.com/en_us/products/splunk-cloud.html](https://www.splunk.com/en_us/products/splunk-cloud.html)

2. **HTTP Event Collector (HEC)** enabled in Splunk
   - Navigate to: Settings → Data Inputs → HTTP Event Collector

3. **GenOps AI** installed
   ```bash
   pip install genops-ai
   ```

4. **Splunk HEC** accessible at an endpoint (e.g., https://splunk.example.com:8088)

---

## ⚡ Pre-Flight Verification (30 seconds)

Before starting, verify your environment is ready:

```bash
# Check Splunk HEC is accessible
curl -k https://splunk.example.com:8088/services/collector/health
# Should return: {"text":"HEC is healthy","code":200}

# Check Splunk HEC token exists
# Navigate to: Settings → Data Inputs → HTTP Event Collector → View tokens

# Verify GenOps AI is installed
pip show genops-ai
# Should show package version info
```

**If HEC health check fails**: Ensure HTTP Event Collector is enabled in Splunk (Settings → Data Inputs → HTTP Event Collector → Global Settings → Enable)

**If GenOps is not installed**: `pip install genops-ai`

---

## 📖 Quick Glossary

New to Splunk or OpenTelemetry? Here are the key terms:

| Term | Meaning |
|------|---------|
| **HEC** | HTTP Event Collector - Splunk's HTTP endpoint for ingesting telemetry data |
| **Index** | Splunk data repository where telemetry is stored (e.g., `genops_ai`) |
| **Sourcetype** | Data classification in Splunk (e.g., `genops:telemetry`) |
| **SPL** | Search Processing Language - Splunk's query language for analyzing data |
| **OTLP** | OpenTelemetry Protocol - standard format for exporting telemetry data |

---

## ⚡ Quick Setup (2 minutes)

### Step 1: Enable Splunk HTTP Event Collector (60 seconds)

In Splunk Web UI:

1. Navigate to **Settings → Data Inputs → HTTP Event Collector**
2. Click **New Token**
3. Configure:
   - **Name**: `genops_ai_token`
   - **Source name override**: (leave empty)
   - **Description**: `GenOps AI governance telemetry`
4. Click **Next**
5. Input Settings:
   - **Source type**: Select **Structured → _json** (or create custom `genops:telemetry`)
   - **Index**: Select **main** (or create custom index `genops_ai`)
   - **Enable indexer acknowledgement**: ✓ (optional, recommended for production)
6. Click **Review → Submit**
7. **Copy the Token Value** - you'll need this in Step 2

**Verify HEC token works:**
```bash
curl -k https://splunk.example.com:8088/services/collector \
  -H "Authorization: Splunk YOUR-HEC-TOKEN" \
  -d '{"event": "test", "sourcetype": "_json"}'
# Should return: {"text":"Success","code":0}
```

### Step 2: Configure GenOps Endpoint (30 seconds)

Set environment variables to configure GenOps to send telemetry to Splunk HEC:

```bash
export SPLUNK_HEC_ENDPOINT="https://splunk.example.com:8088"
export SPLUNK_HEC_TOKEN="your-hec-token-from-step-1"
export SPLUNK_INDEX="genops_ai"  # or "main" if using default index
```

**GenOps configuration pattern:**
```python
from genops import init

# Configure GenOps to send OTLP to Splunk HEC
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={
        "Authorization": "Splunk your-hec-token",
        "X-Splunk-Request-Channel": ""  # Optional: for load balancing
    },
    default_team="ai-platform",
    default_project="genops-splunk-demo"
)
```

---

### Step 3: Send Test Telemetry (60 seconds)

Create a test file to send telemetry: `test_genops_splunk.py`

```python
from genops import init
from genops.core import track_enhanced
from genops.core.telemetry import GenOpsTelemetry

# Configure GenOps to send OTLP to Splunk HEC
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={
        "Authorization": "Splunk your-hec-token-here"
    },
    default_team="ai-platform",
    default_project="genops-splunk-demo"
)

print("✅ GenOps configured to send telemetry to Splunk\\n")

# Test telemetry export
telemetry = GenOpsTelemetry()

with track_enhanced(
    operation_name="test_operation",
    customer_id="demo-customer",
    feature="quickstart-test"
) as span:
    # Record a cost event
    telemetry.record_cost(
        span,
        provider="openai",
        model="gpt-4",
        input_tokens=100,
        output_tokens=50,
        total_cost=0.0025
    )

    print("📊 Recorded test cost event")
    print("   Provider: openai")
    print("   Model: gpt-4")
    print("   Cost: $0.0025")
    print()

print("✅ Test telemetry sent to Splunk HEC!")
print("   Check Splunk Search: index=genops_ai (or index=main)")
```

**Run it:**
```bash
python test_genops_splunk.py
```

**Expected output:**
```
✅ GenOps configured to send telemetry to Splunk

📊 Recorded test cost event
   Provider: openai
   Model: gpt-4
   Cost: $0.0025

✅ Test telemetry sent to Splunk HEC!
   Check Splunk Search: index=genops_ai (or index=main)
```

---

### Step 4: Verify in Splunk Search (30 seconds)

In Splunk Web UI:

1. Navigate to **Search & Reporting** app
2. Enter search query:
   ```spl
   index=genops_ai | head 10
   ```
   Or if using default index:
   ```spl
   index=main sourcetype=_json genops.* | head 10
   ```
3. Click **Search** (or press Enter)
4. You should see incoming events with GenOps attributes:
   - `genops.cost.total`: 0.0025
   - `genops.cost.provider`: openai
   - `genops.cost.model`: gpt-4
   - `genops.customer_id`: demo-customer
   - `genops.team`: ai-platform

**If you don't see data:**
- Check Splunk HEC token is correct (Settings → Data Inputs → HTTP Event Collector)
- Verify index exists: `| eventcount summarize=false index=* | dedup index | fields index`
- Check HEC status: Settings → Data Inputs → HTTP Event Collector → View status
- Check Splunk internal logs: `index=_internal source=*metrics.log component=Metrics group=http_event_collector_metrics`

---

## 🎯 What Just Happened?

**You successfully created a governance telemetry pipeline:**

1. ✅ **Splunk HEC** configured to receive OTLP telemetry
2. ✅ **GenOps AI** captured governance telemetry (cost, team, customer)
3. ✅ **Exported via OTLP** (OpenTelemetry Protocol) to Splunk HEC
4. ✅ **Splunk indexed** the telemetry for search and analysis
5. ✅ **Ready for analytics** with SPL queries and dashboards

**This is the foundation for:**
- Enterprise log analytics and SIEM integration
- Compliance audit trails (HIPAA, SOC 2, GDPR)
- Cost attribution across teams/customers/projects
- Policy violation monitoring and alerting
- Budget threshold enforcement

---

## 📊 See Your Data (1 minute)

### Option 1: View in Splunk Search

Basic search queries to explore your data:

```spl
# View all GenOps telemetry
index=genops_ai | table _time genops.*

# Cost by team
index=genops_ai genops.cost.total=*
| stats sum(genops.cost.total) as total_cost by genops.team

# Policy violations
index=genops_ai genops.policy.result="blocked"
| table _time genops.policy.name genops.policy.reason genops.customer_id

# Budget monitoring
index=genops_ai genops.budget.utilization=*
| where genops.budget.utilization > 80
| table genops.budget.name genops.budget.utilization genops.team
```

### Option 2: Run the Example Integration

For comprehensive examples with dashboards:

```bash
python examples/observability/splunk_integration.py
```

This will:
- Generate sample telemetry (cost, policy, budget events)
- Show example SPL queries for governance use cases
- Provide dashboard XML configurations

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps telemetry flowing into Splunk!**

### 🎯 Recommended Learning Path

For first-time users, we recommend this sequence:

**1. Start here** → **Option A: Cost Analytics** (simplest, immediate value)
- Query cost attribution by team/project/customer
- ~10 minutes to working analytics

**2. Then add** → **Option C: Dashboards** (visualization)
- Import pre-built dashboard templates
- ~15 minutes to first dashboard

**3. Next level** → **Option B: Compliance Monitoring** (governance layer)
- Set up policy violation tracking
- ~20 minutes to compliance dashboard

**4. Advanced** → **Option D: Budget Alerting** (operational requirement)
- Create real-time budget threshold alerts
- ~30 minutes to alert notifications

Choose your path below:

---

### Option A: Cost Attribution Analytics

Query and analyze AI costs with SPL:

**By Team:**
```spl
index=genops_ai genops.cost.total=*
| stats sum(genops.cost.total) as total_cost by genops.team
| sort -total_cost
| eval total_cost_formatted=printf("$%.2f", total_cost)
```

**By Model:**
```spl
index=genops_ai genops.cost.model=*
| stats sum(genops.cost.total) as total_cost by genops.cost.model, genops.cost.provider
| sort -total_cost
```

**Cost Trends:**
```spl
index=genops_ai genops.cost.total=*
| timechart span=1h sum(genops.cost.total) as total_cost by genops.project
```

### Option B: Compliance Monitoring

Track policy violations and audit trails:

**Policy Violations:**
```spl
index=genops_ai genops.policy.result="blocked"
| stats count by genops.policy.name
| sort -count
```

**Compliance Audit Trail:**
```spl
index=genops_ai (genops.policy.* OR genops.eval.*)
| table _time genops.operation.name genops.customer_id genops.team genops.policy.result genops.eval.safety
| sort -_time
```

### Option C: Import Dashboard Templates

Pre-built Splunk dashboards for GenOps governance:

**1. Cost Governance Dashboard:**
```bash
# Download dashboard XML
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/splunk_dashboards/cost_governance.xml

# Import to Splunk
splunk import dashboard cost_governance.xml
```

**Or use the Python integration to generate:**
```python
from examples.observability.splunk_integration import SplunkGenOpsIntegration

splunk = SplunkGenOpsIntegration()

# Generate cost dashboard XML
cost_dashboard_xml = splunk.create_cost_dashboard()
print(cost_dashboard_xml)

# Save to file and import to Splunk
with open("cost_dashboard.xml", "w") as f:
    f.write(cost_dashboard_xml)
```

**Available Dashboards:**
- Cost Governance (`cost_governance.xml`)
- Compliance Monitoring (`compliance_monitoring.xml`)
- Budget Alerting (`budget_alerting.xml`)

### Option D: Budget Threshold Alerting

Create real-time alerts for budget thresholds:

**1. Create Search:**
```spl
index=genops_ai genops.budget.utilization=*
| stats max(genops.budget.utilization) as max_util by genops.budget.name, genops.team
| where max_util > 80
| table genops.budget.name genops.team max_util
```

**2. Save as Alert:**
- Navigate to: Search → Save As → Alert
- **Title**: "GenOps Budget Threshold Alert"
- **Alert Type**: Real-time
- **Trigger Condition**: Custom → Number of Results is greater than 0
- **Throttle**: 5 minutes
- **Alert Action**: Send email, or trigger webhook (Slack/PagerDuty)

**3. Test Alert:**
```python
# Send test budget event over threshold
with track_enhanced(operation_name="budget_test", team="ai-platform") as span:
    telemetry.record_budget(
        span,
        budget_name="team-daily",
        budget_limit=100.0,
        budget_used=95.0,  # 95% utilized - triggers alert
        budget_remaining=5.0,
        metadata={"utilization_percent": 95.0}
    )
```

---

## 🔄 Alternative: Route via Cribl

**GenOps can also route to Splunk via Cribl Stream** for:
- Multi-destination routing (Splunk + Datadog + S3 simultaneously)
- Intelligent sampling (reduce costs by 90%+)
- Data enrichment and transformation

**Quick Cribl Setup:**
```bash
# 1. Configure GenOps → Cribl
export CRIBL_OTLP_ENDPOINT="http://cribl-stream:4318"

# 2. Add Splunk HEC destination in Cribl
# Navigate to: Data → Destinations → Splunk HEC

# 3. Create routing rule
# Filter: __inputId == 'genops_otlp_source'
# Destination: splunk_hec
```

See full Cribl integration guide: [docs/integrations/cribl.md](integrations/cribl.md)

---

## 🔍 Troubleshooting

### Issue: "HEC health check fails" or "Cannot connect to HEC"

**Fix:**
```bash
# Check Splunk HEC is enabled
# Navigate to: Settings → Data Inputs → HTTP Event Collector → Global Settings
# Verify "All Tokens" is enabled

# Check HEC port is accessible
curl -k https://splunk.example.com:8088/services/collector/health

# Verify firewall allows port 8088
netstat -an | grep 8088
```

### Issue: "Token invalid" (403 Forbidden)

**Fix:**
1. Verify token in Splunk matches GenOps configuration
2. Check token is enabled: Settings → Data Inputs → HTTP Event Collector → View tokens
3. Test token manually:
   ```bash
   curl -k https://splunk.example.com:8088/services/collector \
     -H "Authorization: Splunk YOUR-TOKEN" \
     -d '{"event": "test"}'
   ```

### Issue: "No genops.* attributes in events"

**Fix:**
1. Ensure you're using `GenOpsTelemetry().record_*()` methods
2. Verify spans are created with `track_enhanced()` context manager
3. Check OTLP exporter is configured (not console exporter):
   ```python
   init(..., exporter_type="otlp")  # Not "console"
   ```
4. Verify index and sourcetype in Splunk:
   ```spl
   | eventcount summarize=false index=* | search index=genops_ai
   ```

### Issue: "Data not showing in Splunk Search"

**Fix:**
1. Check if data is being indexed:
   ```spl
   index=_internal source=*metrics.log component=Metrics group=http_event_collector_metrics
   | stats sum(event_count) as events by series
   ```
2. Verify index exists and data is flowing:
   ```spl
   | eventcount summarize=false index=genops_ai
   ```
3. Check time range in Splunk Search (expand to "All time")
4. Verify sourcetype is correct:
   ```spl
   index=genops_ai | stats count by sourcetype
   ```

---

## ✅ Verification Checklist

Before proceeding, verify each step:

- [ ] ✅ Splunk Enterprise v8.0+ or Splunk Cloud installed and running
- [ ] ✅ HTTP Event Collector (HEC) enabled and accessible
- [ ] ✅ HEC token created and copied
- [ ] ✅ GenOps AI installed: `pip show genops-ai`
- [ ] ✅ Environment variables set: `echo $SPLUNK_HEC_ENDPOINT`
- [ ] ✅ First telemetry event sent successfully
- [ ] ✅ Event visible in Splunk Search: `index=genops_ai`
- [ ] ✅ GenOps attributes present: `genops.cost.total`, `genops.team`, etc.

**All checked?** You're ready to build dashboards and set up alerts!

---

## 📚 Learn More

- **Full Integration Guide:** [docs/integrations/splunk.md](integrations/splunk.md)
- **Example Code:** [examples/observability/splunk_integration.py](../examples/observability/splunk_integration.py)
- **Dashboard Templates:** [examples/observability/splunk_dashboards/](../examples/observability/splunk_dashboards/)
- **Splunk HEC Documentation:** [https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector](https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector)
- **SPL Reference:** [https://docs.splunk.com/Documentation/Splunk/latest/SearchReference](https://docs.splunk.com/Documentation/Splunk/latest/SearchReference)
- **GenOps Documentation:** [README.md](../README.md)
- **GitHub Repository:** [https://github.com/KoshiHQ/GenOps-AI](https://github.com/KoshiHQ/GenOps-AI)

---

## 💬 Get Help

- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Splunk Community:** [https://community.splunk.com](https://community.splunk.com)

---

## 🎉 What's Next?

**You've completed the quickstart!** Here's what you can do now:

1. **Import Dashboard Templates**: Use the 3 pre-built dashboards for cost, compliance, and budget monitoring
2. **Create SPL Queries**: Build custom queries for your governance use cases
3. **Set Up Alerts**: Configure real-time alerts for budget thresholds and policy violations
4. **Integrate with Your AI App**: Replace test code with real AI operations
5. **Scale to Production**: Enable HA, configure retention policies, optimize index sizing

**Total time: ~5 minutes** ✅

**Next level: Enterprise governance analytics in production** 🚀
