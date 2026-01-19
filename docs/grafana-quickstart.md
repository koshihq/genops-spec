# Grafana Quickstart

Get GenOps AI governance telemetry flowing to Grafana in under 5 minutes.

## 🚀 Quick Setup - Choose Your Path

**Path A: Grafana Cloud** (3 minutes) - Fastest, no Docker required
**Path B: Local LGTM Stack** (10 minutes) - Complete local observability
**Path C: Existing Grafana** (7 minutes) - Connect to your current setup

---

## Path A: Grafana Cloud (3 Minutes) ⚡

Perfect for getting started quickly without local infrastructure.

### 1. Install GenOps with OpenTelemetry Support

```bash
pip install genops-ai[opentelemetry]
```

### 2. Get Your Grafana Cloud Credentials

From your Grafana Cloud account (requires Grafana Cloud v9.0+):
1. Navigate to **Connections → Add new connection → OpenTelemetry (OTLP)**
   - *If you don't see this option, verify your Grafana Cloud version is 9.0 or higher*
2. Copy your OTLP endpoint (e.g., `https://otlp-gateway-prod-us-east-0.grafana.net/otlp`)
3. Generate an **Access Policy Token** with metrics and traces write permissions

### 3. Configure OTLP Export to Grafana Cloud

Set environment variables:

```bash
export GRAFANA_CLOUD_OTLP_ENDPOINT="https://otlp-gateway-prod-<region>.grafana.net/otlp"
export GRAFANA_CLOUD_TOKEN="your_access_policy_token"
export OTEL_SERVICE_NAME="my-ai-app"
```

Configure in your Python application:

```python
from genops.exporters.otlp import configure_otlp_exporter
import os

# Configure Grafana Cloud as your OTLP endpoint
configure_otlp_exporter(
    endpoint=os.getenv("GRAFANA_CLOUD_OTLP_ENDPOINT"),
    headers={
        "Authorization": f"Bearer {os.getenv('GRAFANA_CLOUD_TOKEN')}"
    }
)
```

### 4. Enable Auto-Instrumentation (Zero Code Changes)

```python
from genops import auto_instrument

# Enable telemetry for all AI providers
auto_instrument()

# Your existing code works unchanged!
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
# ✅ Automatically exported to Grafana Cloud!
```

### 5. View Your Telemetry in Grafana Cloud

1. Navigate to **Explore** in your Grafana Cloud instance
2. Select **Tempo** as the data source
3. Search for service: `my-ai-app`
4. View traces with:
   - Cost and token metrics
   - Performance timing
   - Request/response details
   - Governance attributes

**That's it!** Your AI operations now appear in Grafana Cloud with:
- ✅ Real-time cost tracking by model and provider
- ✅ Token usage and performance metrics
- ✅ Distributed tracing across AI operations
- ✅ Full OpenTelemetry compatibility

---

## Path B: Local LGTM Stack (10 Minutes) 🐳

Complete local observability with Grafana, Tempo, Loki, and Mimir.

### 1. Install GenOps with OpenTelemetry Support

```bash
pip install genops-ai[opentelemetry]
```

### 2. Start the LGTM Observability Stack

```bash
# Clone the repository or download docker-compose.observability.yml
docker-compose -f docker-compose.observability.yml up -d

# Wait for services to start (~2-3 minutes for first-time image pulls)
docker-compose -f docker-compose.observability.yml ps
```

### 3. Validate Your Setup

```bash
python examples/observability/validate_otel_collector.py

# Expected output: All services should show ✅
# ✅ OTel Collector accessible
# ✅ Grafana accessible (http://localhost:3000)
# ✅ Tempo accessible
# ✅ Loki accessible
# ✅ Mimir accessible
```

### 4. Configure Your Application

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument

# Configure OTLP export to local OTel Collector
configure_otlp_exporter(
    endpoint="http://localhost:4318"
)

# Enable auto-instrumentation
auto_instrument()

# Your existing code works unchanged!
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
# ✅ Automatically exported to Grafana!
```

### 5. View Pre-Built Dashboards in Grafana

1. Open http://localhost:3000
2. Login with:
   - Username: `admin`
   - Password: `genops`
3. Navigate to **Dashboards → GenOps AI - Governance Overview**
4. View:
   - AI cost tracking by team/customer/model
   - Token usage distribution
   - Policy violation monitoring
   - Recent AI operations

**What You Get:**
- ✅ Complete LGTM observability stack (Grafana + Tempo + Loki + Mimir)
- ✅ Pre-built GenOps governance dashboards
- ✅ Distributed tracing with cost attribution
- ✅ Log aggregation with trace correlation
- ✅ Demo application for testing

---

## Path C: Existing Grafana (7 Minutes) 🔧

Connect GenOps to your current Grafana instance.

### Prerequisites

- Existing Grafana instance (v9.0+)
- At least one of the following data sources:
  - **Tempo** (for traces)
  - **Prometheus** or **Mimir** (for metrics)
  - **Loki** (for logs)

### 1. Install GenOps with OpenTelemetry Support

```bash
pip install genops-ai[opentelemetry]
```

### 2. Configure OTel Collector (or Direct Export)

**Option 1: Via OTel Collector (Recommended)**

Update your OTel Collector config to export to your Tempo/Prometheus/Loki backends:

```yaml
exporters:
  otlp/tempo:
    endpoint: "your-tempo-endpoint:4317"
  prometheusremotewrite:
    endpoint: "your-prometheus-endpoint/api/v1/write"
  loki:
    endpoint: "your-loki-endpoint/loki/api/v1/push"

service:
  pipelines:
    traces:
      exporters: [otlp/tempo]
    metrics:
      exporters: [prometheusremotewrite]
    logs:
      exporters: [loki]
```

**Option 2: Direct Export to Grafana Cloud**

See **Path A** above for Grafana Cloud configuration.

### 3. Configure Your Application

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument

# Point to your OTel Collector endpoint
configure_otlp_exporter(
    endpoint="http://your-otel-collector:4318"
)

# Enable auto-instrumentation
auto_instrument()

# Your existing code works unchanged!
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

### 4. Import Pre-Built Dashboards

Download and import GenOps dashboards:

**Dashboard 1: GenOps Governance Overview**
- File: `observability/grafana/dashboard-files/genops-overview.json`
- Features: Cost tracking, token usage, policy violations

**Dashboard 2: GenOps AI Governance (Prometheus)**
- File: `templates/prometheus/grafana_dashboard.json`
- Features: 14 comprehensive governance panels

**Import via Grafana UI:**
1. Navigate to **Dashboards → Import**
2. Click **Upload JSON file**
3. Select the dashboard file
4. Choose your Tempo/Prometheus data sources
5. Click **Import**

### 5. Verify Data Flow

1. Run a test AI operation
2. Open Grafana **Explore**
3. Select **Tempo** data source
4. Search for your service name
5. View traces with GenOps attributes

---

## 💰 30-Second Cost Attribution

Track costs by team, project, or customer across all paths:

```python
from genops.core.context import set_governance_context

# Set once - applies to all operations
set_governance_context({
    "team": "ai-engineering",
    "project": "customer-chatbot",
    "customer_id": "enterprise_123",
    "environment": "production"
})

# All AI operations now include attribution tags
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
```

**Query in Grafana:**

Using Tempo (TraceQL):
```traceql
{ resource.service.name="my-ai-app" && span.genops.team="ai-engineering" }
```

Using Prometheus (PromQL):
```promql
sum by (team) (genops_cost_total_usd)
```

---

## 🔍 First Queries to Try

### Cost Analysis
```promql
# Total cost by provider
sum by (provider) (genops_cost_total_usd)

# Cost per customer
sum by (customer_id) (genops_cost_total_usd)

# Hourly cost rate
rate(genops_cost_total_usd[1h])
```

### Token Usage
```promql
# Token consumption rate
rate(genops_tokens_input_total[5m]) + rate(genops_tokens_output_total[5m])

# Token efficiency by model
genops_tokens_output_total / genops_tokens_input_total
```

### Distributed Tracing
```traceql
# Find expensive operations (>$0.10)
{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 0.10 }

# Find policy violations
{ resource.service.name="my-ai-app" && span.genops.policy.status="violated" }
```

---

## ✅ Validate Your Setup

For all paths, validate your configuration:

```python
# Run the validation script
python examples/observability/validate_otel_collector.py

# Or use the validation API
from examples.observability.otel_collector_validation import validate_setup, print_validation_result

result = validate_setup(
    collector_endpoint="http://localhost:4318",  # or your endpoint
    grafana_endpoint="http://localhost:3000",     # or your Grafana URL
    check_connectivity=True,
    check_backends=True
)

print_validation_result(result)
```

**Expected Output:**
```
✅ All OpenTelemetry dependencies installed
✅ OTel Collector accessible at http://localhost:4318
✅ Grafana accessible at http://localhost:3000
✅ Tempo accessible
✅ Loki accessible
✅ Mimir accessible
```

---

## 📊 Pre-Built Dashboards

GenOps provides two production-ready Grafana dashboards:

### 1. GenOps Governance Overview
**Location:** `observability/grafana/dashboard-files/genops-overview.json`

**Panels:**
- AI Cost Overview (USD)
- Token Usage by Model (pie chart)
- Cost by Team (bar graph)
- Policy Violations (time series)
- Recent AI Operations (table with trace links)

**Data Sources:** Mimir (metrics) + Tempo (traces)

### 2. GenOps AI Governance (Prometheus Template)
**Location:** `templates/prometheus/grafana_dashboard.json`

**Panels (14 total):**
- Total Cost (Last 24h)
- Hourly Cost Rate
- Cost Over Time
- Cost by Provider/Model/Team
- Token Usage Rate & Efficiency
- Operation Latency (p50/p95/p99)
- Error Rate
- Budget Utilization (gauge with thresholds)
- Policy Violations

**Data Sources:** Prometheus/Mimir (metrics)

---

## 🐛 Troubleshooting

### No Data in Grafana

**Check 1: Verify OTel Collector is receiving data**
```bash
# Check collector metrics endpoint
curl http://localhost:8888/metrics | grep otelcol_receiver

# Check collector logs
docker logs otel-collector
```

**Check 2: Verify data sources in Grafana**
1. Navigate to **Configuration → Data Sources**
2. Test each data source (Tempo, Prometheus/Mimir, Loki)
3. All should show "Data source is working"

**Check 3: Generate test operations**
```bash
# Use the demo app
curl -X POST http://localhost:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Test", "model": "gpt-3.5-turbo"}'
```

### Connection Refused Errors

**For Grafana Cloud:**
- Verify your OTLP endpoint URL is correct
- Check your access token has proper permissions (metrics + traces write)
- Ensure no firewall blocking outbound HTTPS

**For Local Stack:**
- Verify services are running: `docker-compose ps`
- Check port conflicts: `lsof -i :3000,4318`
- Restart services: `docker-compose restart`

### Dashboard Import Failures

- Ensure you're using Grafana v9.0+
- Verify data sources are configured before importing
- Check JSON file syntax is valid
- Try importing via **Import via panel JSON** instead of file upload

---

## 📖 Next Steps

### 5-Minute Quick Wins
- ✅ Add cost attribution to your operations
- ✅ Create custom queries for your use cases
- ✅ Set up Grafana alerts for budget thresholds

### 30-Minute Deep Dive
- 📖 Read the [Comprehensive Grafana Integration Guide](integrations/grafana.md)
- 📊 Customize dashboards for your metrics
- 🔍 Explore the [Query Examples Cookbook](grafana-query-examples.md)

### 2-Hour Production Setup
- 🚀 Configure high-availability Grafana
- 🔐 Set up RBAC and team access controls
- 📈 Implement alerting and incident response workflows
- 🎯 Integrate with your existing observability stack

---

## 📚 Additional Resources

- **Full Integration Guide:** [docs/integrations/grafana.md](integrations/grafana.md)
- **Query Examples:** [docs/grafana-query-examples.md](grafana-query-examples.md)
- **LGTM Stack Details:** [observability/README.md](../observability/README.md)
- **OTel Collector Guide:** [docs/otel-collector-quickstart.md](otel-collector-quickstart.md)
- **Grafana Documentation:** <a href="https://grafana.com/docs/" target="_blank">grafana.com/docs ↗</a>
- **OpenTelemetry Docs:** <a href="https://opentelemetry.io" target="_blank">opentelemetry.io ↗</a>

---

**🎉 Congratulations!** You now have GenOps AI governance telemetry flowing to Grafana with:
- ✅ Real-time cost tracking and attribution
- ✅ Distributed tracing with governance context
- ✅ Pre-built dashboards for immediate insights
- ✅ Full OpenTelemetry compatibility

**Questions or issues?** Open an issue at <a href="https://github.com/KoshiHQ/GenOps-AI/issues" target="_blank">GitHub ↗</a>
