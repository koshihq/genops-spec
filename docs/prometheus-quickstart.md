# Prometheus Quickstart

Get GenOps AI governance metrics flowing to Prometheus in under 5 minutes.

## 🚀 Quick Setup (5 Minutes)

### 1. Install GenOps with Prometheus Support

```bash
pip install genops-ai[prometheus]
```

### 2. Start Metrics Export with Zero Code

```python
from genops.exporters.prometheus import auto_instrument

# Start metrics server at http://localhost:8000/metrics
auto_instrument()
```

### 3. Use Any AI Provider (No Code Changes)

```python
# Your existing code works unchanged!
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
# ✅ Metrics automatically exported to /metrics endpoint!
```

### 4. View Metrics in Your Browser

Open http://localhost:8000/metrics to see governance metrics:

```prometheus
# HELP genops_cost_total_usd Total cost of AI operations in USD
# TYPE genops_cost_total_usd counter
genops_cost_total_usd{provider="openai",model="gpt-3.5-turbo"} 0.000525

# HELP genops_tokens_input_total Total input tokens consumed
# TYPE genops_tokens_input_total counter
genops_tokens_input_total{provider="openai",model="gpt-3.5-turbo"} 10

# HELP genops_tokens_output_total Total output tokens generated
# TYPE genops_tokens_output_total counter
genops_tokens_output_total{provider="openai",model="gpt-3.5-turbo"} 25
```

### 5. Configure Prometheus Scraping

Create or update your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'genops-ai'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
```

Restart Prometheus and view your metrics in the Prometheus UI at http://localhost:9090.

**That's it!** Your AI operations now have:
- ✅ Real-time cost tracking by model and provider
- ✅ Token usage and efficiency metrics
- ✅ Performance latency histograms
- ✅ Policy compliance counters

---

## 💰 30-Second Cost Attribution

Track costs by team, project, or customer:

```python
from genops.core.context import set_governance_context

# Set once - applies to all operations
set_governance_context({
    "team": "ai-engineering",
    "project": "customer-chatbot",
    "customer_id": "enterprise_123",
    "environment": "production"
})

# All AI operations now include attribution labels
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
```

**Query in Prometheus:**

```promql
# Total cost by team
sum(rate(genops_cost_total_usd[1h])) by (team)

# Cost per customer
sum(genops_cost_total_usd) by (customer_id)

# Token efficiency by model
genops_tokens_total / genops_cost_total_usd
```

**View in /metrics:**

```prometheus
genops_cost_total_usd{
  provider="openai",
  model="gpt-4",
  team="ai-engineering",
  customer_id="enterprise_123",
  environment="production"
} 1.25
```

---

## 📊 Import Pre-Built Grafana Dashboard

GenOps provides a production-ready Grafana dashboard:

### Option 1: Import via Grafana UI

1. Download dashboard template:
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/templates/prometheus/grafana_dashboard.json
```

2. In Grafana:
   - Navigate to **Dashboards → Import**
   - Upload `grafana_dashboard.json`
   - Select your Prometheus data source
   - Click **Import**

### Option 2: Import via API

```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GRAFANA_API_KEY" \
  -d @templates/prometheus/grafana_dashboard.json
```

### Option 3: Docker Compose (Complete Stack)

Start Prometheus + Grafana + GenOps dashboard:

```bash
cd examples/observability/
docker-compose up -d
```

Access:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:8000/metrics

**Dashboard Features:**
- Cost breakdown by provider, model, team, and customer
- Token usage trends and efficiency
- Performance latency percentiles (p50, p95, p99)
- Budget utilization gauges
- Policy violation rates

---

## ✅ Validate Your Setup

Ensure everything is configured correctly:

```python
from genops.exporters.prometheus import validate_setup, print_validation_result

# Run comprehensive validation
result = validate_setup()
print_validation_result(result)
```

**Example output:**

```
================================================================================
GenOps Prometheus Exporter Validation
================================================================================

✅ Overall Status: PASSED
   Score: 100.0% (8/8 checks passed)

📋 System Information:
   python_version: 3.11.5
   port: 8000
   prometheus_url: http://localhost:9090
   namespace: genops

📊 Validation Results:

   DEPENDENCIES:
   ℹ️ Python Version: Python 3.11.5 detected
   ℹ️ Prometheus Client: prometheus_client 0.19.0 installed
   ℹ️ OpenTelemetry Prometheus Exporter: OpenTelemetry Prometheus exporter installed
   ℹ️ OpenTelemetry SDK: OpenTelemetry SDK packages available
   ℹ️ Requests Library: requests 2.31.0 installed

   CONFIGURATION:
   ℹ️ Port Configuration: Port 8000 is within valid range
   ℹ️ Namespace Configuration: Namespace 'genops' is valid

   CONNECTIVITY:
   ℹ️ Port Available: Port 8000 is available

💡 Recommendations:
   ✅ All checks passed! Start the exporter with: from genops.exporters.prometheus import instrument_prometheus; instrument_prometheus()

================================================================================
```

**Fix common issues:**

```bash
# Port conflict - set in your shell BEFORE running Python
export PROMETHEUS_EXPORTER_PORT=8001
python your_app.py

# Different namespace - set in your shell BEFORE running Python
export PROMETHEUS_NAMESPACE=myapp
python your_app.py
```

**Alternative**: Use a `.env` file with `python-dotenv`:

```bash
# .env file
PROMETHEUS_EXPORTER_PORT=8001
PROMETHEUS_NAMESPACE=myapp
```

```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file

from genops.exporters.prometheus import auto_instrument
auto_instrument()  # Uses values from .env
```

**Or configure programmatically**:

```python
# Disable validation
from genops.exporters.prometheus import instrument_prometheus
exporter = instrument_prometheus(port=8001, validate=False)
```

---

## 🔍 Understanding the API Functions

GenOps provides two ways to start the Prometheus exporter:

**`auto_instrument()`** - Zero-code setup (recommended for most use cases)
- Reads configuration from environment variables
- Simplest approach for getting started
- Perfect for 5-minute quickstart

**`instrument_prometheus()`** - Manual configuration
- Programmatic control over all settings
- Use when you need to set configuration in code
- Required for advanced scenarios (custom ports, sampling rates, etc.)

**Relationship**: `auto_instrument()` internally calls `instrument_prometheus()` with environment-based configuration.

**Example:**
```python
# These are equivalent:

# Option 1: Auto (reads PROMETHEUS_EXPORTER_PORT env var)
from genops.exporters.prometheus import auto_instrument
auto_instrument()

# Option 2: Manual
from genops.exporters.prometheus import instrument_prometheus
exporter = instrument_prometheus(port=8000)
```

---

## 🔍 Example PromQL Queries

Query your governance metrics in Prometheus:

### Cost Metrics

```promql
# Total cost rate per hour
sum(rate(genops_cost_total_usd[1h])) * 3600

# Cost by provider
sum(genops_cost_total_usd) by (provider)

# Cost by model
sum(genops_cost_total_usd) by (model)

# Cost by team
sum(genops_cost_total_usd) by (team)

# Daily cost burn (last 24h)
increase(genops_cost_total_usd[24h])
```

### Token Metrics

```promql
# Total tokens per hour
sum(rate(genops_tokens_total[1h])) * 3600

# Input/output token ratio
sum(rate(genops_tokens_output_total[5m])) / sum(rate(genops_tokens_input_total[5m]))

# Tokens per dollar (efficiency)
sum(rate(genops_tokens_total[5m])) / sum(rate(genops_cost_total_usd[5m]))
```

### Performance Metrics

```promql
# Median latency (p50)
histogram_quantile(0.50, rate(genops_operation_latency_seconds_bucket[5m]))

# 95th percentile latency (p95)
histogram_quantile(0.95, rate(genops_operation_latency_seconds_bucket[5m]))

# Error rate
sum(rate(genops_operation_errors_total[5m]))
```

---

## 🚨 Set Up Alerts (Optional)

Add alert rules to your Prometheus configuration:

```yaml
# prometheus_alerts.yml
groups:
  - name: genops_cost_alerts
    interval: 60s
    rules:
      # Cost spike detection
      - alert: HighCostRate
        expr: rate(genops_cost_total_usd[5m]) * 3600 > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High AI cost rate detected"
          description: "Cost rate {{ $value | humanize }}/hour exceeds threshold"

      # Budget utilization
      - alert: BudgetNearlyExceeded
        expr: genops_budget_utilization_ratio > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Budget nearly exceeded"
          description: "Budget utilization at {{ $value | humanizePercentage }}"

      # Policy violations
      - alert: PolicyViolationSpike
        expr: rate(genops_policy_violations_total[5m]) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Policy violation spike detected"
          description: "Violation rate: {{ $value | humanize }}/sec"
```

Load alert rules in `prometheus.yml`:

```yaml
rule_files:
  - "prometheus_alerts.yml"
```

See full templates:
- `templates/prometheus/alert_rules.yml` - Complete alert rule templates
- `templates/prometheus/recording_rules.yml` - Recording rules for aggregations

---

## 🌐 Multi-Provider Tracking

Track costs across multiple AI providers:

```python
from genops.exporters.prometheus import auto_instrument

# Start Prometheus exporter
auto_instrument()

# Use multiple providers
from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI()
anthropic_client = Anthropic()

# Both automatically tracked with provider labels
openai_response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

anthropic_response = anthropic_client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Metrics in Prometheus:**

```prometheus
genops_cost_total_usd{provider="openai",model="gpt-4"} 0.03
genops_cost_total_usd{provider="anthropic",model="claude-3-sonnet-20240229"} 0.015
```

**Query cross-provider costs:**

```promql
# Total cost across all providers
sum(genops_cost_total_usd)

# Cost by provider
sum(genops_cost_total_usd) by (provider)

# Most expensive models
topk(5, sum(genops_cost_total_usd) by (model))
```

---

## 📖 Next Steps

**Production Deployment:**
- [Comprehensive Integration Guide](integrations/prometheus.md) - Complete reference
- [Recording Rules Templates](../templates/prometheus/recording_rules.yml) - Pre-aggregated metrics
- [Alert Rules Templates](../templates/prometheus/alert_rules.yml) - Production alerts
- [Kubernetes Deployment](kubernetes-observability.md#prometheus-configuration) - K8s setup

**Advanced Features:**
- [High-Cardinality Management](integrations/prometheus.md#cardinality-management) - Label optimization
- [Sampling Strategies](integrations/prometheus.md#sampling-configuration) - High-volume scenarios
- [Federation Setup](integrations/prometheus.md#federation-patterns) - Multi-cluster aggregation

**Integration Guides:**
- [OpenTelemetry Collector](integrations/otel-collector.md) - Unified telemetry pipeline
- [Grafana Setup](integrations/grafana.md) - Dashboard configuration
- [Multi-Provider Tracking](guides/multi-provider-cost-tracking.md) - Unified governance

---

## 🐛 Troubleshooting

### Telemetry Not Appearing

**Problem:** Metrics endpoint returns empty or shows no GenOps metrics

**Solutions:**

1. **Verify exporter is started:**
```python
from genops.exporters.prometheus import get_exporter

exporter = get_exporter()
if exporter:
    print(f"Exporter running on port {exporter.config.port}")
else:
    print("Exporter not initialized - call auto_instrument() first")
```

2. **Check dependencies:**
```bash
pip install genops-ai[prometheus]
```

3. **Verify port is accessible:**
```bash
curl http://localhost:8000/metrics
```

### Cost Metrics Missing

**Problem:** Token metrics appear but cost metrics are zero

**Solutions:**

1. **Ensure provider instrumentation:**
```python
from genops.providers.openai import instrument_openai

# Instrument provider explicitly
client = instrument_openai()
```

2. **Check for cost calculation errors:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. **Verify model pricing:**
```python
from genops.providers.openai import OPENAI_PRICING

print(OPENAI_PRICING.get("gpt-3.5-turbo"))
```

### High Telemetry Volume

**Problem:** Metrics growing too large or Prometheus scraping slow

**Solutions:**

1. **Enable sampling:**
```python
from genops.exporters.prometheus import instrument_prometheus

exporter = instrument_prometheus(
    sampling_rate=0.1  # Sample 10% of operations
)
```

2. **Limit label cardinality:**
```python
exporter = instrument_prometheus(
    max_label_cardinality=5000,
    exclude_labels={"operation_id"}  # Exclude high-cardinality labels
)
```

3. **Use recording rules:**
```yaml
# Pre-aggregate expensive queries
- record: genops:cost:hourly_by_team
  expr: sum(rate(genops_cost_total_usd[1h])) by (team) * 3600
```

### Port Conflicts

**Problem:** `Port 8000 is already in use`

**Solution:**

```python
# Use different port
from genops.exporters.prometheus import instrument_prometheus

exporter = instrument_prometheus(port=8001)
```

Or via environment variable:

```bash
export PROMETHEUS_EXPORTER_PORT=8001
python your_app.py
```

---

## 💬 Need Help?

- **Documentation**: [Full Prometheus Integration Guide](integrations/prometheus.md)
- **Examples**: `examples/observability/prometheus_*.py`
- **GitHub Issues**: https://github.com/KoshiHQ/GenOps-AI/issues
- **Validation**: Run `validate_setup()` for diagnostics

---

**Next:** [Complete Integration Guide →](integrations/prometheus.md)
