# Cribl Integration - 5-Minute Quickstart

**🎯 Get GenOps + Cribl governance pipeline in 5 minutes**

This guide gets you from zero to routing GenOps AI governance telemetry through Cribl Stream to multiple observability platforms in under 5 minutes.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Cribl Stream v4.0+** installed and running
   - Get it from: [https://cribl.io/download/](https://cribl.io/download/)
   - Or use Cribl Cloud: [https://cribl.cloud](https://cribl.cloud)

2. **GenOps AI** installed
   ```bash
   pip install genops-ai
   ```

3. **Cribl Stream accessible** at an endpoint (default: http://localhost:9000)

---

## ⚡ Pre-Flight Verification (30 seconds)

Before starting, verify your environment is ready:

```bash
# Check Cribl Stream is running
curl http://localhost:9000/api/v1/health
# Should return: {"status":"ok"}

# Check OTLP port availability (optional - Cribl will bind it in Step 1)
netstat -an | grep 4318
# Should show LISTEN state after Step 1, or no output yet

# Verify GenOps AI is installed
pip show genops-ai
# Should show package version info
```

**If Cribl health check fails**: Ensure Cribl Stream is started (`systemctl start cribl` or check Cribl Cloud status)

**If GenOps is not installed**: `pip install genops-ai`

---

## 📖 Quick Glossary

New to OpenTelemetry or Cribl? Here are the key terms:

| Term | Meaning |
|------|---------|
| **OTLP** | OpenTelemetry Protocol - standard format for exporting telemetry data |
| **Span** | A single operation or event in a trace (e.g., one AI API call) |
| **Attributes** | Key-value metadata attached to spans (e.g., `cost`, `customer_id`, `team`) |
| **Pipeline** | A Cribl workflow that filters, transforms, and routes telemetry data |
| **Sampling** | Selectively keeping a percentage of events to reduce volume and costs |

---

## ⚡ Quick Setup (2 minutes)

### Step 1: Configure Cribl HTTP Source for OTLP (60 seconds)

In Cribl Stream UI:

1. Navigate to **Data → Sources → HTTP**
2. Click **Add Source**
3. Configure:
   - **Source ID**: `genops_otlp_source`
   - **Port**: `4318`
   - **Path**: `/v1/traces`
   - **Enable TLS**: Optional (recommended for production)
4. **Authentication** (optional but recommended):
   - Click **Add Authentication**
   - Type: **Bearer Token**
   - **Token**: Generate a token:
     - **For testing**: Use `genops-cribl-test-token` (simple placeholder)
     - **For production**: Generate cryptographically secure token:
       ```bash
       openssl rand -hex 32
       ```
   - Save the token for Step 2
5. Click **Save**

**Verify source is running:**
- Status indicator should show green/active
- Endpoint URL: `http://localhost:4318/v1/traces`

### Step 2: Configure GenOps Endpoint (30 seconds)

Set environment variables and configure GenOps to send telemetry to Cribl Stream.

**Set your Cribl endpoint:**
```bash
export CRIBL_OTLP_ENDPOINT="http://localhost:4318"
export CRIBL_AUTH_TOKEN="genops-cribl-test-token"  # Use actual token from Step 1
```

**GenOps configuration pattern:**
```python
from genops import init

# Configure GenOps to send OTLP to Cribl Stream
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4318",  # Cribl OTLP HTTP receiver
    otlp_headers={
        "Authorization": "Bearer genops-cribl-test-token",  # Your token
        "X-Scope-OrgID": "my-organization"
    },
    default_team="ai-platform",
    default_project="genops-cribl-demo"
)
```

**Note**: We'll validate this configuration before sending test telemetry in the next step.

### Step 2.5: Validate Configuration (30 seconds)

Before sending test telemetry, verify your Cribl setup is correct:

```python
from genops.providers.cribl.validation import validate_setup, print_validation_result

# Check your Cribl setup
result = validate_setup()
print_validation_result(result)
```

You should see: ✅ **Status: PASSED**

**What if validation fails?**

The validation output will show specific errors and how to fix them:

```
❌ Status: FAILED
Summary: 1 errors, 0 warnings

🚨 ERRORS (must fix to proceed):

1. [Connectivity] Cannot connect to cribl-stream:4318
   Fix: Check Cribl Stream is running and port 4318 is open. Test with: telnet cribl-stream 4318
```

Follow the fix suggestions, then run validation again.

---

### Step 3: Send Test Telemetry (60 seconds)

Now that validation passed, create a test file to send telemetry: `test_genops_cribl.py`

```python
from genops import init
from genops.core import track_enhanced
from genops.core.telemetry import GenOpsTelemetry

# Configure GenOps to send OTLP to Cribl Stream
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4318",  # Cribl OTLP HTTP receiver
    otlp_headers={
        "Authorization": "Bearer genops-cribl-test-token",  # Your token
        "X-Scope-OrgID": "my-organization"
    },
    default_team="ai-platform",
    default_project="genops-cribl-demo"
)

print("✅ GenOps configured to send telemetry to Cribl\n")

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

print("✅ Test telemetry sent to Cribl!")
print("   Check Cribl UI: Data → Sources → genops_otlp_source → Live Data")
```

**Run it:**
```bash
python test_genops_cribl.py
```

**Expected output:**
```
✅ GenOps configured to send telemetry to Cribl

📊 Recorded test cost event
   Provider: openai
   Model: gpt-4
   Cost: $0.0025

✅ Test telemetry sent to Cribl!
   Check Cribl UI: Data → Sources → genops_otlp_source → Live Data
```

---

### Step 4: Verify in Cribl Stream (30 seconds)

In Cribl Stream UI:

1. Navigate to **Data → Sources → genops_otlp_source**
2. Click **Live Data** button
3. You should see incoming OTLP spans with GenOps attributes:
   - `genops.cost.total`: 0.0025
   - `genops.cost.provider`: openai
   - `genops.cost.model`: gpt-4
   - `genops.customer_id`: demo-customer
   - `genops.team`: ai-platform

**If you don't see data:**
- Check Cribl Stream logs: **Monitoring → System → Logs**
- Verify source status: **Data → Sources → genops_otlp_source**
- Check authentication token matches in both GenOps and Cribl
- Ensure Cribl is listening on port 4318

---

## 🎯 What Just Happened?

**You successfully created a governance telemetry pipeline:**

1. ✅ **GenOps AI** captured governance telemetry (cost, team, customer)
2. ✅ **Exported via OTLP** (OpenTelemetry Protocol) to Cribl Stream
3. ✅ **Cribl Stream** ingested the telemetry via HTTP source
4. ✅ **Ready for routing** to any of 100+ downstream platforms

**This is the foundation for:**
- Multi-platform distribution (Datadog, Splunk, S3, etc.)
- Intelligent sampling (reduce costs by 90%+)
- Policy-based routing (violations → SIEM, costs → dashboards)
- Compliance audit trails (regulated industries)

---

## 📊 See Your Data (1 minute)

### Option 1: View in Cribl Stream Live Data

1. **Navigate to**: Data → Sources → genops_otlp_source → Live Data
2. **Expand a span** to see all GenOps attributes
3. **Search/filter** by attributes:
   - `genops.cost.total > 0.001`
   - `genops.customer_id == "demo-customer"`
   - `genops.team == "ai-platform"`

### Option 2: Create a Simple Route to Console

Quick test route to see data flowing:

1. Navigate to **Data → Routes**
2. Click **Add Route**
3. Configure:
   - **Route ID**: `test_genops_console`
   - **Filter**: `__inputId == 'genops_otlp_source'`
   - **Output**: Select **devnull** (or create a **stdout** destination)
4. Click **Save**
5. View routed data in **Monitoring → Live Data**

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps telemetry flowing through Cribl!**

### 🎯 Recommended Learning Path

For first-time users, we recommend this sequence:

**1. Start here** → **Option A: Cost Routing** (simplest, immediate value)
- See costs flowing to Datadog/Grafana dashboards
- ~10 minutes to working dashboard

**2. Then add** → **Option C: Budget Alerting** (operational value)
- Get Slack alerts when budgets hit thresholds
- ~15 minutes to first alert

**3. Next level** → **Option B: Policy Violations** (governance layer)
- Route violations to SIEM for security review
- ~20 minutes to SIEM integration

**4. Advanced** → **Option D: Compliance Audit** (enterprise requirement)
- Preserve audit trail for regulated industries
- ~30 minutes to compliant storage

Choose your path below:

---

### Option A: Set Up Cost Routing Pipeline

Route cost telemetry to Datadog/Grafana for dashboards:

1. **Import pipeline**: Processing → Pipelines → Import
2. **Upload**: `examples/cribl/pipelines/genops_cost_governance.yml`
3. **Configure destinations**:
   - Create Datadog destination (Data → Destinations → Datadog)
   - Attach pipeline to route
4. **View costs in Datadog**: Create dashboard with `genops.cost.*` metrics

### Option B: Set Up Policy Violation Routing

Route policy violations to SIEM (Splunk/Elastic):

1. **Import pipeline**: `examples/cribl/pipelines/genops_policy_compliance.yml`
2. **Configure SIEM destination**:
   - Splunk: Data → Destinations → Splunk HEC
   - Elastic: Data → Destinations → Elasticsearch
3. **Test with policy events**:
   ```python
   telemetry.record_policy(
       span,
       policy_name="content_safety",
       policy_result="blocked",
       policy_reason="Harmful content detected"
   )
   ```

### Option C: Set Up Budget Alerting

Trigger Slack/PagerDuty alerts on budget thresholds:

1. **Import pipeline**: `examples/cribl/pipelines/genops_budget_alerting.yml`
2. **Configure webhook destinations**:
   - Slack: Data → Destinations → Webhook (Slack format)
   - PagerDuty: Data → Destinations → Webhook (PagerDuty Events API v2)
3. **Test with budget events**:
   ```python
   telemetry.record_budget(
       span,
       budget_name="team-daily",
       budget_limit=100.0,
       budget_used=95.0,
       budget_remaining=5.0,
       metadata={"utilization_percent": 95.0}
   )
   ```

### Option D: Set Up Compliance Audit Trail

Route compliance audit trail to S3/Snowflake:

1. **Import pipeline**: `examples/cribl/pipelines/genops_audit_trail.yml`
2. **Configure data lake destinations**:
   - S3: Data → Destinations → S3
   - Snowflake: Data → Destinations → Snowflake
3. **Test with compliance events**:
   ```python
   with track_enhanced(
       operation_name="phi_processing",
       customer_id="healthcare-provider-001",
       metadata={
           "compliance_framework": "HIPAA",
           "data_classification": "PHI",
           "audit_trail_required": True,
           "retention_period_years": 7
       }
   ) as span:
       # Process PHI data
       pass
   ```

---

## 🔍 Troubleshooting

### Issue: "Connection refused" or "Telemetry not appearing in Cribl"

**Fix:**
```bash
# Check Cribl Stream is running
curl http://localhost:9000/api/v1/health

# Check OTLP source is listening
netstat -an | grep 4318

# Verify source configuration in Cribl UI
# Data → Sources → genops_otlp_source → Status should be "Active"
```

### Issue: "Authentication failed" (401/403 errors)

**Fix:**
1. Verify token in Cribl source matches GenOps configuration
2. Check Cribl logs: Monitoring → System → Logs
3. Test without authentication first:
   ```python
   init(
       service_name="my-ai-service",
       exporter_type="otlp",
       otlp_endpoint="http://localhost:4318",
       otlp_headers={}  # No auth for testing
   )
   ```
4. Add authentication back once basic connection works

### Issue: "No genops.* attributes in spans"

**Fix:**
1. Ensure you're using `GenOpsTelemetry().record_*()` methods
2. Verify spans are created with `track_enhanced()` context manager
3. Check OTLP exporter is configured (not console exporter):
   ```python
   init(..., exporter_type="otlp")  # Not "console"
   ```

### Issue: "Cribl not routing telemetry to destinations"

**Fix:**
1. Check route filters match incoming data:
   - Data → Routes → Click route → View **Filter Expression**
   - Test filter: Data → Sources → Live Data → **Apply Filter**
2. Verify destinations are configured and connected:
   - Data → Destinations → Check status indicators
   - Test destination: Click destination → **Test** button
3. Check pipeline processing:
   - Processing → Pipelines → View pipeline metrics
   - Monitor dropped/failed events

---

## ✅ Verification Checklist

Before proceeding, verify each step:

- [ ] ✅ Cribl Stream v4.0+ installed and running
- [ ] ✅ OTLP HTTP source created (port 4318)
- [ ] ✅ GenOps AI installed: `pip show genops-ai`
- [ ] ✅ Environment variables set: `echo $CRIBL_OTLP_ENDPOINT`
- [ ] ✅ Validation passed: `python -c "from genops.providers.cribl.validation import validate_setup, print_validation_result; print_validation_result(validate_setup())"`
- [ ] ✅ First telemetry event sent successfully
- [ ] ✅ Event visible in Cribl UI Live Data

**All checked?** You're ready to configure pipelines!

---

## 📚 Learn More

- **Full Integration Guide:** [docs/integrations/cribl.md](integrations/cribl.md)
- **Example Code:** [examples/observability/cribl_integration.py](../examples/observability/cribl_integration.py)
- **Cribl Pipelines:** [examples/cribl/pipelines/](../examples/cribl/pipelines/)
- **Cribl Stream Docs:** [https://docs.cribl.io](https://docs.cribl.io)
- **GenOps Documentation:** [README.md](../README.md)
- **GitHub Repository:** [https://github.com/KoshiHQ/GenOps-AI](https://github.com/KoshiHQ/GenOps-AI)

---

## 💬 Get Help

- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Cribl Community:** [https://cribl.io/community](https://cribl.io/community)

---

## 🎉 What's Next?

**You've completed the quickstart!** Here's what you can do now:

1. **Import Cribl Pipelines**: Use the 4 pre-built pipelines for cost, policy, budget, and compliance routing
2. **Configure Destinations**: Set up Datadog, Splunk, S3, Slack, PagerDuty, etc.
3. **Integrate with Your AI App**: Replace test code with real AI operations
4. **Monitor and Optimize**: Use Cribl metrics to optimize sampling and routing
5. **Scale to Production**: Enable HA, add monitoring, configure retention policies

**Total time: ~5 minutes** ✅

**Next level: Multi-platform governance in production** 🚀
