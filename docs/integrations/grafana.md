# Grafana Integration

**Export AI governance telemetry to Grafana for comprehensive observability, cost tracking, and compliance monitoring.**

## Overview

The GenOps Grafana integration enables organizations to visualize AI governance telemetry — cost attribution, policy enforcement, budget tracking, and evaluation metrics — using Grafana's powerful visualization platform with the LGTM stack (Loki, Grafana, Tempo, Mimir) via OpenTelemetry OTLP export.

### Problems Solved

- **Cross-Stack AI Visibility:** Unified view of AI operations across OpenAI, Anthropic, Bedrock, Gemini, and 35+ providers
- **Cost Attribution:** Track and analyze AI costs by team, project, customer, and model in real-time with customizable dashboards
- **Policy Compliance:** Monitor policy enforcement and compliance violations with alerting
- **Budget Management:** Track budget consumption with proactive alerts to prevent overruns
- **Distributed Tracing:** Tempo-powered distributed tracing for AI operations with cost attribution per trace
- **Log Correlation:** Loki-based log aggregation with automatic trace correlation
- **Flexible Deployment:** Self-hosted, Grafana Cloud, or hybrid deployment options
- **Vendor Neutrality:** No lock-in — works with existing Prometheus, Tempo, Loki infrastructure

### Value Proposition

**For Platform Teams:**
- Centralized governance telemetry in your existing Grafana observability stack
- No vendor lock-in — standard OpenTelemetry/OTLP integration
- Distributed tracing shows complete AI operation flows with Tempo
- Integration with existing metrics, logs, and infrastructure monitoring
- Self-hosted option with full data control
- Open-source foundation with enterprise support available

**For FinOps Teams:**
- Per-team, per-project, per-customer cost attribution with real-time dashboards
- Budget tracking with Grafana alerting capabilities
- Cost optimization insights (model efficiency, provider comparison, token usage trends)
- Historical cost analysis with long-term Mimir/Prometheus storage
- Multi-cloud cost aggregation across AI providers

**For Compliance Teams:**
- Complete audit trail for all AI operations in Tempo traces
- Policy violation tracking with Grafana alert workflows
- Data classification and governance enforcement
- Compliance dashboard templates with pre-built panels
- Long-term retention with Loki log aggregation

---

## Installation & Setup

### Install GenOps with OpenTelemetry Support

```bash
# Install with OpenTelemetry extras
pip install genops-ai[opentelemetry]

# Or install OpenTelemetry packages directly
pip install genops-ai \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-otlp-proto-http
```

### OpenTelemetry Requirements

- **Python Version:** 3.8+ (3.10+ recommended)
- **OpenTelemetry SDK:** 1.20.0+ (latest 1.x recommended)
- **OTLP Exporter:** HTTP or gRPC (HTTP recommended)

### Grafana Stack Requirements

#### For Grafana Cloud:
- **Grafana Cloud Account:** Free or paid tier
- **Access Policy Token:** Generate from Cloud Portal → Security → Access Policies
- **Permissions:** Token needs `metrics:write`, `traces:write`, `logs:write` scopes
- **OTLP Endpoint:** Provided in Cloud Portal → Connections → OpenTelemetry

#### For Self-Hosted (Local LGTM Stack):
- **Docker:** 20.10+ with Docker Compose v2
- **Resources:** 4GB RAM minimum (8GB recommended), 10GB disk space
- **Ports:** 3000 (Grafana), 3100 (Loki), 3200 (Tempo), 4318 (OTel Collector), 9009 (Mimir)

#### For Self-Hosted (Production):
- **Kubernetes:** 1.23+ (for Helm deployments)
- **Helm:** 3.8+ (for chart installations)
- **Storage:** Persistent volumes for Tempo, Loki, Mimir
- **Resources:** Varies by scale (see Production Deployment section)

### Verify Installation

```bash
# Check GenOps installation
python -c "import genops; print(genops.__version__)"

# Check OpenTelemetry installation
python -c "import opentelemetry; print('OpenTelemetry OK')"

# For local LGTM stack, validate services
python examples/observability/validate_otel_collector.py
```

---

## Quick Start

See the **[Grafana Quickstart Guide](../grafana-quickstart.md)** for 5-minute setup.

### Path A: Grafana Cloud (Zero Infrastructure)

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument
import os

# Configure Grafana Cloud as OTLP endpoint
configure_otlp_exporter(
    endpoint=os.getenv("GRAFANA_CLOUD_OTLP_ENDPOINT"),
    headers={
        "Authorization": f"Bearer {os.getenv('GRAFANA_CLOUD_TOKEN')}"
    }
)

# Enable auto-instrumentation
auto_instrument()

# All AI operations now export to Grafana Cloud!
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Automatically tracked in Grafana Cloud
```

### Path B: Local LGTM Stack (Complete Development Environment)

**1. Start the LGTM Stack:**
```bash
# Start all services (Grafana, Tempo, Loki, Mimir, OTel Collector)
docker-compose -f docker-compose.observability.yml up -d

# Validate setup
python examples/observability/validate_otel_collector.py
```

**2. Configure Your Application:**
```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument

# Configure OTLP export to local OTel Collector
configure_otlp_exporter(
    endpoint="http://localhost:4318"
)

# Enable auto-instrumentation
auto_instrument()

# Your AI operations now flow to Grafana!
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Visible in Grafana at http://localhost:3000
```

**3. View in Grafana:**
- Open http://localhost:3000
- Login: `admin` / `genops`
- Navigate to **Dashboards → GenOps AI - Governance Overview**

### Path C: Existing Grafana (Enterprise Integration)

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument

# Point to your existing OTel Collector
configure_otlp_exporter(
    endpoint="http://your-otel-collector:4318",
    service_name="my-ai-service",
    environment="production"
)

# Enable auto-instrumentation
auto_instrument()

# Telemetry flows to your existing Tempo/Prometheus/Loki backends
```

---

## Core Concepts

### 1. LGTM Stack Architecture

GenOps integrates with Grafana's **LGTM stack** — a complete observability solution:

**LGTM Components:**
- **L**oki: Log aggregation and correlation
- **G**rafana: Visualization and dashboards
- **T**empo: Distributed tracing backend
- **M**imir: High-performance metrics storage (Prometheus-compatible)

**Architecture:**
```
GenOps AI Application
    ↓
OpenTelemetry SDK (traces, metrics, logs)
    ↓
OTLP Exporter (HTTP/gRPC)
    ↓
OpenTelemetry Collector
    ↓
    ├── Tempo (traces)
    ├── Loki (logs)
    └── Mimir/Prometheus (metrics)
         ↓
    Grafana (visualization)
```

**Benefits:**
- Standard OpenTelemetry protocol (vendor-neutral)
- Complete observability stack (traces, metrics, logs)
- Self-hosted or cloud-hosted options
- Long-term data retention with configurable policies
- Native multi-tenancy support

### 2. Deployment Options

GenOps supports three Grafana deployment patterns:

**Choose Your Deployment:**
- **< 10GB/month data?** → **Grafana Cloud Free Tier**
- **< 50GB/month + no data residency concerns?** → **Grafana Cloud Pro**
- **< 100GB/month + small team + development/testing?** → **Local LGTM Stack (Docker Compose)**
- **> 100GB/month OR regulated industry OR production workloads?** → **Self-Hosted (Kubernetes)**

---

#### Option A: Grafana Cloud (Fastest)
**Best for:** Quick start, managed service, cloud-native teams

- Fully managed Tempo, Loki, Mimir, and Grafana
- No infrastructure management required
- Auto-scaling and high availability built-in
- OTLP ingestion via Grafana Cloud Gateway
- Free tier available (14-day traces, 30-day metrics, 15-day logs)

**Setup Time:** ~3 minutes

#### Option B: Local LGTM Stack (Development)
**Best for:** Local development, testing, demos

- Complete Docker Compose stack
- Pre-configured services with auto-provisioning
- Pre-built GenOps dashboards included
- Demo application for testing
- Zero external dependencies

**Setup Time:** ~10 minutes

#### Option C: Self-Hosted Production (Enterprise)
**Best for:** Enterprise deployments, data sovereignty, full control

- Kubernetes deployment with Helm charts
- High-availability configuration
- Custom retention policies
- Integration with existing infrastructure
- Full data ownership and control

**Setup Time:** ~1-2 hours (depending on existing infrastructure)

### 3. Governance Semantic Conventions

GenOps uses standardized telemetry field names aligned with OpenTelemetry conventions:

**Core Telemetry Fields:**
- `trace_id`: Distributed trace ID (OpenTelemetry standard)
- `span_id`: Span identifier (OpenTelemetry standard)
- `service.name`: Service name (e.g., `my-ai-app`)
- `deployment.environment`: Environment (dev/staging/prod)

**Governance Attributes:**
- `genops.team`: Team attribution
- `genops.project`: Project tracking
- `genops.environment`: Environment segregation
- `genops.customer_id`: Customer attribution
- `genops.cost_center`: Financial reporting
- `genops.feature`: Feature tracking

**Cost Fields:**
- `genops.cost.total_usd`: Total cost in USD
- `genops.cost.input_usd`: Input token cost
- `genops.cost.output_usd`: Output token cost
- `genops.cost.provider`: AI provider (openai, anthropic, bedrock, gemini)
- `genops.cost.model`: Model name (gpt-4, claude-3-sonnet)
- `genops.tokens.input`: Input tokens
- `genops.tokens.output`: Output tokens
- `genops.tokens.total`: Total tokens

**Policy Fields:**
- `genops.policy.name`: Policy identifier
- `genops.policy.status`: Status (allowed, blocked, warning)
- `genops.policy.reason`: Decision reason
- `genops.policy.evaluation_time_ms`: Policy evaluation duration (ms)

**Budget Fields:**
- `genops.budget.id`: Budget identifier
- `genops.budget.limit_usd`: Budget limit (USD)
- `genops.budget.consumed_usd`: Amount consumed
- `genops.budget.remaining_usd`: Amount remaining
- `genops.budget.utilization_percent`: Utilization percentage

**Evaluation Fields:**
- `genops.eval.safety_score`: Safety score (0-1)
- `genops.eval.accuracy_score`: Accuracy score (0-1)
- `genops.eval.compliance_score`: Compliance score (0-1)
- `genops.eval.performed`: Boolean flag for evaluation

**Performance Fields:**
- `duration_ms`: Operation duration in milliseconds
- `status`: Operation status (success, error, timeout)
- `http.status_code`: HTTP response status

### 4. Data Source Configuration

Grafana requires data source configuration for each backend:

**Tempo (Traces):**
- URL: `http://tempo:3200` (Docker) or `http://localhost:3200` (local)
- Query language: TraceQL
- Use for: Distributed tracing, cost per trace, policy evaluation flows

**Prometheus/Mimir (Metrics):**
- URL: `http://mimir:9009` (Docker) or `http://localhost:9009` (local)
- Query language: PromQL
- Use for: Time-series cost metrics, token usage, performance latency

**Loki (Logs):**
- URL: `http://loki:3100` (Docker) or `http://localhost:3100` (local)
- Query language: LogQL
- Use for: Log aggregation, trace correlation, error investigation

**Grafana Cloud:**
- OTLP Endpoint: `https://otlp-gateway-prod-{region}.grafana.net/otlp`
- Authentication: Bearer token via `Authorization` header
- Auto-configured data sources (Tempo, Prometheus, Loki)

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GRAFANA_CLOUD_OTLP_ENDPOINT` | Grafana Cloud only | None | Cloud OTLP endpoint URL |
| `GRAFANA_CLOUD_TOKEN` | Grafana Cloud only | None | Access policy token |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Self-hosted | `http://localhost:4318` | OTel Collector endpoint |
| `OTEL_SERVICE_NAME` | ❌ No | `genops-ai` | Service name in traces |
| `OTEL_SERVICE_VERSION` | ❌ No | `1.0.0` | Service version |
| `OTEL_ENVIRONMENT` | ❌ No | `production` | Environment tag |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | ❌ No | `http/protobuf` | OTLP protocol |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | ❌ No | `10000` | Export timeout (ms) |

### Programmatic Configuration

```python
from genops.exporters.otlp import configure_otlp_exporter

configure_otlp_exporter(
    # Endpoint configuration
    endpoint="http://localhost:4318",  # or Grafana Cloud endpoint
    headers={},  # or {"Authorization": "Bearer <token>"} for Cloud

    # Service metadata
    service_name="my-ai-application",
    service_version="1.2.3",
    environment="production",

    # Resource attributes (optional)
    resource_attributes={
        "service.namespace": "ai-platform",
        "deployment.environment": "production",
        "cloud.provider": "aws",
        "cloud.region": "us-west-2"
    },

    # Export configuration
    timeout_ms=10000,
    compression="gzip",

    # Sampling (for high-volume applications)
    trace_sampler="always_on",  # or "trace_id_ratio" with sample_rate
    sample_rate=1.0  # 1.0 = 100% sampling
)
```

### OTel Collector Configuration

For self-hosted deployments, configure the OTel Collector to export to your Grafana backends:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

  # Add resource attributes
  resource:
    attributes:
      - key: service.namespace
        value: genops-ai
        action: upsert

  # Cost calculation processor (optional)
  transform:
    trace_statements:
      - context: span
        statements:
          # Calculate total cost from input + output costs
          - set(attributes["genops.cost.total_usd"],
              attributes["genops.cost.input_usd"] + attributes["genops.cost.output_usd"])

exporters:
  # Tempo (traces)
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

  # Mimir/Prometheus (metrics)
  prometheusremotewrite:
    endpoint: http://mimir:9009/api/v1/push
    resource_to_telemetry_conversion:
      enabled: true

  # Loki (logs)
  loki:
    endpoint: http://loki:3100/loki/api/v1/push
    labels:
      resource:
        service.name: "service_name"
        deployment.environment: "environment"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, resource, transform]
      exporters: [otlp/tempo]

    metrics:
      receivers: [otlp]
      processors: [batch, resource]
      exporters: [prometheusremotewrite]

    logs:
      receivers: [otlp]
      processors: [batch, resource]
      exporters: [loki]
```

### Governance Context (Cost Attribution)

```python
from genops.core.context import set_governance_context

# Set governance context for all subsequent operations
set_governance_context({
    "team": "ai-platform",
    "project": "customer-chatbot",
    "customer_id": "enterprise-123",
    "cost_center": "engineering",
    "environment": "production",
    "feature": "chat-completion"
})

# All AI operations now include these attributes in Grafana
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
# ✅ Visible in Grafana with full governance context
```

**Use Cases:**
- **Per-Team Cost Tracking:** Query costs grouped by `genops.team`
- **Customer Attribution:** Track AI costs per `genops.customer_id` for billing
- **Project Budgets:** Monitor spending per `genops.project`
- **Cost Center Reporting:** Financial reports grouped by `genops.cost_center`

---

## Dashboards

GenOps provides two production-ready Grafana dashboards.

### Dashboard 1: GenOps Governance Overview

**Location:** `observability/grafana/dashboard-files/genops-overview.json`

**Panels:**
1. **AI Cost Overview** (Stat panel)
   - Total AI spend in USD
   - Single value with sparkline
   - Data source: Mimir (metrics)

2. **Token Usage by Model** (Pie chart)
   - Distribution of tokens across models
   - Percentage breakdown
   - Data source: Mimir (metrics)

3. **Cost by Team** (Bar graph)
   - Team-level cost attribution
   - Sorted by total spend
   - Data source: Mimir (metrics)

4. **Policy Violations** (Time series)
   - Policy violation trends over time
   - Alerts on threshold breaches
   - Data source: Mimir (metrics)

5. **Recent AI Operations** (Table)
   - Latest AI operations with trace links
   - Columns: Timestamp, Service, Operation, Cost, Tokens, Trace ID
   - Data source: Tempo (traces)

**Import Instructions:**

**Via Grafana UI:**
1. Navigate to **Dashboards → Import**
2. Click **Upload JSON file**
3. Select `observability/grafana/dashboard-files/genops-overview.json`
4. Select data sources:
   - **Mimir/Prometheus:** Select your metrics data source
   - **Tempo:** Select your traces data source
5. Click **Import**

**Via Grafana API:**
```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @observability/grafana/dashboard-files/genops-overview.json
```

**Via Provisioning (Docker/Kubernetes):**
```yaml
# grafana/dashboards/dashboards.yml
apiVersion: 1

providers:
  - name: 'GenOps AI'
    orgId: 1
    folder: 'AI Governance'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

### Dashboard 2: GenOps AI Governance (Prometheus Template)

**Location:** `templates/prometheus/grafana_dashboard.json`

**Panels (14 comprehensive governance panels):**

1. **Total Cost (Last 24h)** - Single stat with trend
2. **Hourly Cost Rate** - Rate of spend per hour
3. **Total Tokens (24h)** - Token consumption total
4. **Operations/Second** - Request rate
5. **Cost Over Time** - Time series graph with per-provider breakdown
6. **Cost by Provider** - Pie chart (OpenAI, Anthropic, Bedrock, etc.)
7. **Cost by Model Top 10** - Bar gauge of most expensive models
8. **Cost by Team** - Pie chart of team attribution
9. **Token Usage Rate** - Time series of token consumption
10. **Token Efficiency by Model** - Output tokens per input token ratio
11. **Operation Latency** - Histogram with p50/p95/p99 percentiles
12. **Error Rate** - Failed operations percentage
13. **Budget Utilization** - Gauge with red/yellow/green thresholds
14. **Policy Violations** - Time series count of violations

**PromQL Queries Used:**

```promql
# Total cost
sum(genops_cost_total_usd)

# Cost by provider
sum by (provider) (genops_cost_total_usd)

# Hourly cost rate
rate(genops_cost_total_usd[1h]) * 3600

# Token efficiency
sum by (model) (genops_tokens_output_total) /
  sum by (model) (genops_tokens_input_total)

# Budget utilization percentage
(sum(genops_budget_consumed_usd) / sum(genops_budget_limit_usd)) * 100

# Policy violation count
sum(genops_policy_violations_total)
```

**Import:** Follow same instructions as Dashboard 1, using the Prometheus template file.

### Customizing Dashboards

**Add Custom Panels:**

1. Click **Add panel** in dashboard edit mode
2. Select visualization type (Time series, Bar chart, Table, etc.)
3. Write PromQL/TraceQL/LogQL query
4. Configure display options
5. Save panel and dashboard

**Example Custom Panel - Cost Per Customer:**

```promql
# Query
sum by (customer_id) (genops_cost_total_usd)

# Visualization: Bar chart
# Sort: Descending by value
# Legend: {{ customer_id }}
# Unit: USD ($)
```

**Example Custom Panel - Average Latency by Model:**

```promql
# Query
avg by (model) (genops_operation_duration_ms)

# Visualization: Time series
# Legend: {{ model }}
# Unit: milliseconds (ms)
```

**Dashboard Variables:**

Create dashboard variables for filtering:

```
Name: team
Type: Query
Data source: Prometheus/Mimir
Query: label_values(genops_cost_total_usd, team)
Multi-value: true
Include All option: true

Name: environment
Type: Query
Data source: Prometheus/Mimir
Query: label_values(genops_cost_total_usd, environment)
Multi-value: false
Include All option: false
```

Use in queries:
```promql
sum by (provider) (genops_cost_total_usd{team=~"$team", environment="$environment"})
```

---

## Query Patterns

### PromQL (Metrics)

**Cost Analysis:**

```promql
# Total cost across all providers
sum(genops_cost_total_usd)

# Cost by provider
sum by (provider) (genops_cost_total_usd)

# Cost by model
sum by (model) (genops_cost_total_usd)

# Cost by team
sum by (team) (genops_cost_total_usd)

# Cost per customer
sum by (customer_id) (genops_cost_total_usd)

# Hourly cost rate
rate(genops_cost_total_usd[1h]) * 3600

# Daily cost
increase(genops_cost_total_usd[1d])

# Cost trend (7-day moving average)
avg_over_time(rate(genops_cost_total_usd[1h])[7d:1h]) * 3600
```

**Token Usage:**

```promql
# Total tokens consumed
sum(genops_tokens_input_total + genops_tokens_output_total)

# Token rate (tokens per second)
rate(genops_tokens_input_total[5m]) + rate(genops_tokens_output_total[5m])

# Token efficiency (output/input ratio)
sum by (model) (genops_tokens_output_total) /
  sum by (model) (genops_tokens_input_total)

# Top 10 models by token usage
topk(10, sum by (model) (genops_tokens_total))

# Input vs output token distribution
sum(genops_tokens_input_total) / sum(genops_tokens_output_total)
```

**Performance:**

```promql
# Average operation latency
avg(genops_operation_duration_ms)

# p95 latency by model
histogram_quantile(0.95,
  sum by (model, le) (rate(genops_operation_duration_ms_bucket[5m])))

# p99 latency
histogram_quantile(0.99,
  sum by (le) (rate(genops_operation_duration_ms_bucket[5m])))

# Operations per second
rate(genops_operations_total[1m])

# Error rate percentage
(sum(rate(genops_operations_total{status="error"}[5m])) /
 sum(rate(genops_operations_total[5m]))) * 100
```

**Budget & Policy:**

```promql
# Budget utilization percentage
(sum(genops_budget_consumed_usd) / sum(genops_budget_limit_usd)) * 100

# Budget remaining
sum(genops_budget_remaining_usd)

# Policy violations per hour
rate(genops_policy_violations_total[1h]) * 3600

# Policy block rate
(sum(rate(genops_policy_violations_total{result="blocked"}[5m])) /
 sum(rate(genops_operations_total[5m]))) * 100
```

### TraceQL (Traces)

**Distributed Tracing:**

```traceql
# All AI operations for a service
{ resource.service.name="my-ai-app" }

# Operations for a specific team
{ resource.service.name="my-ai-app" && span.genops.team="ai-engineering" }

# Expensive operations (>$0.10)
{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 0.10 }

# Policy violations
{ resource.service.name="my-ai-app" && span.genops.policy.status="blocked" }

# Operations for a specific customer
{ resource.service.name="my-ai-app" && span.genops.customer_id="enterprise-123" }

# Slow operations (>5 seconds)
{ resource.service.name="my-ai-app" && duration > 5s }

# Operations using GPT-4
{ resource.service.name="my-ai-app" && span.genops.cost.model="gpt-4" }

# Failed operations
{ resource.service.name="my-ai-app" && status=error }

# Aggregate: Average cost per trace
{ resource.service.name="my-ai-app" } | avg(span.genops.cost.total_usd)

# Aggregate: Total tokens per trace
{ resource.service.name="my-ai-app" } | sum(span.genops.tokens.total)
```

### LogQL (Logs)

**Log Aggregation:**

```logql
# All logs for a service
{service_name="my-ai-app"}

# Logs with specific governance context
{service_name="my-ai-app", team="ai-engineering"}

# Error logs
{service_name="my-ai-app"} |= "error"

# Logs containing cost information
{service_name="my-ai-app"} |= "cost"

# Logs for specific trace ID (trace correlation)
{service_name="my-ai-app"} |= "trace_id=abc123def456"

# Policy violation logs
{service_name="my-ai-app"} | json | policy_result="blocked"

# Log rate (logs per second)
rate({service_name="my-ai-app"}[5m])

# Error log count
count_over_time({service_name="my-ai-app"} |= "error" [1h])
```

---

## Alerting

### Grafana Alert Rules

**Alert 1: High AI Cost Rate**

```yaml
alert: HighAICostRate
expr: rate(genops_cost_total_usd[1h]) * 3600 > 100
for: 10m
labels:
  severity: warning
  team: ai-platform
annotations:
  summary: "High AI cost rate detected"
  description: "AI costs are ${{ $value | humanize }}/hour (threshold: $100/hour)"
```

**Configure in Grafana:**
1. Navigate to **Alerting → Alert rules**
2. Click **New alert rule**
3. Set query: `rate(genops_cost_total_usd[1h]) * 3600`
4. Set condition: `IS ABOVE 100`
5. Set evaluation: **For 10 minutes**
6. Add labels and annotations
7. Configure notification channel (Slack, PagerDuty, email, etc.)

**Alert 2: Budget Utilization Threshold**

```promql
(sum(genops_budget_consumed_usd) / sum(genops_budget_limit_usd)) * 100 > 80
```

**Alert Thresholds:**
- **Warning:** >80% budget utilization
- **Critical:** >95% budget utilization

**Alert 3: Policy Violation Spike**

```promql
rate(genops_policy_violations_total[5m]) > 10
```

**Alert:** More than 10 policy violations per minute

**Alert 4: High Error Rate**

```promql
(sum(rate(genops_operations_total{status="error"}[5m])) /
 sum(rate(genops_operations_total[5m]))) * 100 > 5
```

**Alert:** Error rate exceeds 5%

### Notification Channels

**Slack Integration:**

```yaml
# grafana/provisioning/notifiers/slack.yml
notifiers:
  - name: slack-ai-alerts
    type: slack
    uid: slack-ai-alerts
    org_id: 1
    settings:
      url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
      recipient: "#ai-alerts"
      username: Grafana AI Alerts
```

**PagerDuty Integration:**

```yaml
notifiers:
  - name: pagerduty-ai-critical
    type: pagerduty
    uid: pagerduty-ai-critical
    org_id: 1
    settings:
      integrationKey: YOUR_PAGERDUTY_INTEGRATION_KEY
      severity: critical
```

**Email Integration:**

```yaml
notifiers:
  - name: email-finops-team
    type: email
    uid: email-finops-team
    org_id: 1
    settings:
      addresses: finops-team@company.com
```

---

## Production Deployment

### Docker Compose (Small Scale)

**Use Case:** Small teams, single-server deployments

**Resources:**
- 8GB RAM minimum
- 4 CPU cores
- 100GB disk space

**Configuration:**

```yaml
# docker-compose.grafana-production.yml
version: '3.8'

services:
  grafana:
    image: grafana/grafana:10.2.3
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_SERVER_ROOT_URL=${GRAFANA_ROOT_URL}
      - GF_AUTH_ANONYMOUS_ENABLED=false
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    restart: unless-stopped

  tempo:
    image: grafana/tempo:latest
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo-config.yaml:/etc/tempo.yaml
      - tempo-data:/var/tempo
    ports:
      - "3200:3200"
      - "4317:4317"
      - "4318:4318"
    restart: unless-stopped

  mimir:
    image: grafana/mimir:latest
    command: ["-config.file=/etc/mimir.yaml"]
    volumes:
      - ./mimir-config.yaml:/etc/mimir.yaml
      - mimir-data:/data
    ports:
      - "9009:9009"
    restart: unless-stopped

  loki:
    image: grafana/loki:latest
    command: ["-config.file=/etc/loki/local-config.yaml"]
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml
      - loki-data:/loki
    ports:
      - "3100:3100"
    restart: unless-stopped

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"
      - "4318:4318"
      - "8888:8888"
    restart: unless-stopped

volumes:
  grafana-data:
  tempo-data:
  mimir-data:
  loki-data:
```

**Deployment:**

```bash
# Set environment variables
export GRAFANA_ADMIN_PASSWORD="secure-password-here"
export GRAFANA_ROOT_URL="https://grafana.your-domain.com"

# Start production stack
docker-compose -f docker-compose.grafana-production.yml up -d

# Validate
curl -f http://localhost:3000/api/health
```

### Kubernetes with Helm (Production Scale)

**Use Case:** Enterprise deployments, high availability, multi-tenancy

**Prerequisites:**
- Kubernetes 1.23+
- Helm 3.8+
- Persistent volume provisioner
- Ingress controller

**Install Grafana LGTM Stack:**

```bash
# Add Grafana Helm repo
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Tempo
helm install tempo grafana/tempo \
  --namespace observability --create-namespace \
  --set persistence.enabled=true \
  --set persistence.size=100Gi

# Install Loki
helm install loki grafana/loki-stack \
  --namespace observability \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=100Gi

# Install Mimir
helm install mimir grafana/mimir-distributed \
  --namespace observability \
  --set minio.enabled=true

# Install Grafana
helm install grafana grafana/grafana \
  --namespace observability \
  --set persistence.enabled=true \
  --set adminPassword=<secure-password> \
  --set ingress.enabled=true \
  --set ingress.hosts[0]=grafana.your-domain.com

# Install OTel Collector
helm install otel-collector open-telemetry/opentelemetry-collector \
  --namespace observability \
  --values otel-collector-values.yaml
```

**OTel Collector Helm Values:**

```yaml
# otel-collector-values.yaml
mode: deployment
replicaCount: 3

resources:
  limits:
    cpu: 2
    memory: 4Gi
  requests:
    cpu: 1
    memory: 2Gi

config:
  receivers:
    otlp:
      protocols:
        http:
          endpoint: 0.0.0.0:4318
        grpc:
          endpoint: 0.0.0.0:4317

  processors:
    batch:
      timeout: 10s
    resource:
      attributes:
        - key: cluster.name
          value: production
          action: insert

  exporters:
    otlp/tempo:
      endpoint: tempo.observability.svc.cluster.local:4317
      tls:
        insecure: true

    prometheusremotewrite:
      endpoint: http://mimir.observability.svc.cluster.local:9009/api/v1/push

    loki:
      endpoint: http://loki.observability.svc.cluster.local:3100/loki/api/v1/push

  service:
    pipelines:
      traces:
        receivers: [otlp]
        processors: [batch, resource]
        exporters: [otlp/tempo]
      metrics:
        receivers: [otlp]
        processors: [batch, resource]
        exporters: [prometheusremotewrite]
      logs:
        receivers: [otlp]
        processors: [batch, resource]
        exporters: [loki]

service:
  type: LoadBalancer
  ports:
    - name: otlp-grpc
      port: 4317
      targetPort: 4317
      protocol: TCP
    - name: otlp-http
      port: 4318
      targetPort: 4318
      protocol: TCP
```

**Scaling Configuration:**

```yaml
# Horizontal Pod Autoscaler for OTel Collector
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: otel-collector-hpa
  namespace: observability
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: otel-collector
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### High Availability Configuration

**Tempo HA:**

```yaml
# tempo-ha-values.yaml
tempo:
  replicas: 3

  storage:
    trace:
      backend: s3  # or gcs, azure
      s3:
        bucket: tempo-traces
        endpoint: s3.amazonaws.com
        region: us-west-2

  memcached:
    enabled: true
    replicas: 3
```

**Mimir HA:**

```yaml
# mimir-ha-values.yaml
mimir:
  structuredConfig:
    multitenancy_enabled: true

    blocks_storage:
      backend: s3
      s3:
        bucket_name: mimir-blocks
        endpoint: s3.amazonaws.com

  ingester:
    replicas: 3
    resources:
      requests:
        cpu: 2
        memory: 8Gi

  querier:
    replicas: 3
    resources:
      requests:
        cpu: 2
        memory: 4Gi
```

### Data Retention Policies

**Tempo Retention:**

```yaml
# tempo-config.yaml
compactor:
  compaction:
    block_retention: 720h  # 30 days
```

**Mimir Retention:**

```yaml
# mimir-config.yaml
limits:
  compactor_blocks_retention_period: 90d  # 90 days
```

**Loki Retention:**

```yaml
# loki-config.yaml
table_manager:
  retention_deletes_enabled: true
  retention_period: 744h  # 31 days
```

---

## Monitoring & Troubleshooting

### Health Checks

**Check Grafana:**
```bash
curl http://localhost:3000/api/health
# Expected: {"database": "ok"}
```

**Check Tempo:**
```bash
curl http://localhost:3200/ready
# Expected: 200 OK
```

**Check Mimir:**
```bash
curl http://localhost:9009/ready
# Expected: 200 OK
```

**Check Loki:**
```bash
curl http://localhost:3100/ready
# Expected: 200 OK
```

**Check OTel Collector:**
```bash
curl http://localhost:8888/metrics | grep otelcol_receiver_accepted_spans
# Expected: Non-zero span count
```

### Common Issues

#### Issue 1: No Data in Grafana

**Symptoms:**
- Dashboards show "No data"
- Explore queries return empty results

**Diagnosis:**

```bash
# Check OTel Collector is receiving data
curl http://localhost:8888/metrics | grep otelcol_receiver

# Check Tempo has traces
curl http://localhost:3200/api/search | jq

# Check Mimir has metrics
curl http://localhost:9009/prometheus/api/v1/label/__name__/values

# Check application is exporting
python -c "from genops.exporters.otlp import validate_export; validate_export()"
```

**Solutions:**

1. Verify OTLP endpoint is correct in application config
2. Check OTel Collector logs: `docker logs otel-collector`
3. Verify network connectivity between application and collector
4. Check data sources are configured correctly in Grafana
5. Generate test data: `curl -X POST http://localhost:8000/ai/chat ...`

#### Issue 2: High Memory Usage

**Symptoms:**
- Services are OOM killed
- Slow query performance
- High pod eviction rate

**Solutions:**

1. **Increase Memory Limits:**
```yaml
resources:
  limits:
    memory: 8Gi
  requests:
    memory: 4Gi
```

2. **Configure Sampling:**
```python
configure_otlp_exporter(
    trace_sampler="trace_id_ratio",
    sample_rate=0.1  # Sample 10% of traces
)
```

3. **Reduce Retention:**
```yaml
# Shorter retention = less memory
block_retention: 168h  # 7 days instead of 30
```

4. **Enable Memcached:**
```yaml
memcached:
  enabled: true
  replicas: 3
```

#### Issue 3: Slow Query Performance

**Symptoms:**
- Grafana queries timeout
- Dashboard load times >10 seconds

**Solutions:**

1. **Add Indexes (Tempo):**
```yaml
tempo:
  search:
    enabled: true
  metrics_generator:
    enabled: true
```

2. **Use Recording Rules (Mimir):**
```yaml
# rules.yaml
groups:
  - name: genops_recordings
    interval: 1m
    rules:
      - record: genops:cost:rate1h
        expr: rate(genops_cost_total_usd[1h]) * 3600

      - record: genops:cost:by_team
        expr: sum by (team) (genops_cost_total_usd)
```

3. **Optimize Queries:**
```promql
# Bad: Unbounded time range
sum(genops_cost_total_usd)

# Good: Limited time range
sum(rate(genops_cost_total_usd[5m]))
```

4. **Scale Querier Replicas:**
```yaml
querier:
  replicas: 5  # Increase from 3
```

#### Issue 4: Data Source Connection Failures

**Symptoms:**
- "Data source is not working" in Grafana
- "Connection refused" errors

**Solutions:**

1. **Verify Service URLs:**
```bash
# From Grafana container
curl http://tempo:3200/ready
curl http://mimir:9009/ready
curl http://loki:3100/ready
```

2. **Check DNS Resolution:**
```bash
nslookup tempo.observability.svc.cluster.local
```

3. **Update Data Source URLs:**
- Grafana UI → Configuration → Data Sources
- Update URLs to correct service endpoints
- Test connection

4. **Check Firewall Rules:**
```bash
# Ensure ports are open
nc -zv tempo 3200
nc -zv mimir 9009
```

### Performance Tuning

**Batch Processing:**

```yaml
# otel-collector-config.yaml
processors:
  batch:
    timeout: 10s
    send_batch_size: 1024
    send_batch_max_size: 2048
```

**Compression:**

```python
configure_otlp_exporter(
    endpoint="http://localhost:4318",
    compression="gzip"  # Reduce network bandwidth
)
```

**Connection Pooling:**

```yaml
# otel-collector-config.yaml
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    sending_queue:
      enabled: true
      num_consumers: 10
      queue_size: 1000
```

**Query Caching:**

```yaml
# grafana.ini
[caching]
enabled = true

[caching.memory]
ttl = 24h
```

---

## Advanced Features

### Multi-Tenancy with RBAC

**Grafana RBAC:**

```yaml
# Create organizations
resource "grafana_organization" "ai_team" {
  name = "AI Engineering Team"
}

resource "grafana_organization" "finops_team" {
  name = "FinOps Team"
}

# Create folders with permissions
resource "grafana_folder" "ai_dashboards" {
  title = "AI Dashboards"
  org_id = grafana_organization.ai_team.id
}

# Assign users to roles
resource "grafana_team" "ai_engineers" {
  name = "AI Engineers"
  org_id = grafana_organization.ai_team.id
}

resource "grafana_team_member" "engineer1" {
  team_id = grafana_team.ai_engineers.id
  email = "engineer@company.com"
}

# Set folder permissions
resource "grafana_folder_permission" "ai_read" {
  folder_uid = grafana_folder.ai_dashboards.uid
  team = grafana_team.ai_engineers.id
  permission = "View"
}
```

**Tempo Multi-Tenancy:**

```yaml
# tempo-config.yaml
multitenancy_enabled: true

overrides:
  "team-ai-engineering":
    ingestion_rate_limit_bytes: 15000000
    ingestion_burst_size_bytes: 20000000

  "team-data-science":
    ingestion_rate_limit_bytes: 10000000
    ingestion_burst_size_bytes: 15000000
```

**Application Configuration:**

```python
# Set tenant ID in headers
configure_otlp_exporter(
    endpoint="http://tempo:4318",
    headers={
        "X-Scope-OrgID": "team-ai-engineering"  # Tenant identifier
    }
)
```

### SSO Integration

**SAML Configuration:**

```ini
# grafana.ini
[auth.saml]
enabled = true
certificate_path = /etc/grafana/saml-cert.crt
private_key_path = /etc/grafana/saml-key.key
idp_metadata_url = https://your-idp.com/metadata
assertion_attribute_name = displayName
assertion_attribute_login = username
assertion_attribute_email = email
assertion_attribute_groups = groups
```

**OAuth Configuration:**

```ini
# grafana.ini
[auth.generic_oauth]
enabled = true
name = OAuth
allow_sign_up = true
client_id = YOUR_CLIENT_ID
client_secret = YOUR_CLIENT_SECRET
scopes = openid profile email
auth_url = https://your-oauth.com/authorize
token_url = https://your-oauth.com/token
api_url = https://your-oauth.com/userinfo
```

### API Key Management

**Create API Key:**

```bash
# Via Grafana API
curl -X POST http://localhost:3000/api/auth/keys \
  -H "Content-Type: application/json" \
  -u admin:genops \
  -d '{
    "name": "GenOps Automation",
    "role": "Editor",
    "secondsToLive": 2592000
  }'
```

**Use API Key:**

```python
import requests

headers = {
    "Authorization": "Bearer your-api-key-here",
    "Content-Type": "application/json"
}

# Create dashboard via API
response = requests.post(
    "http://localhost:3000/api/dashboards/db",
    headers=headers,
    json=dashboard_json
)
```

### Grafana Enterprise Features

**Enterprise Data Source Permissions:**

```yaml
# Restrict data sources by team
datasource_permissions:
  - datasource: Tempo
    team: AI Engineering
    permission: Query

  - datasource: Prometheus
    team: FinOps
    permission: Query
```

**Report Scheduling:**

```yaml
# Schedule PDF reports
reports:
  - name: Daily AI Cost Report
    dashboard: GenOps AI Governance
    schedule: "0 8 * * *"  # 8 AM daily
    recipients:
      - finops-team@company.com
    format: pdf
```

**Audit Logging:**

```ini
# grafana.ini
[auditing]
enabled = true
log_dashboard_content = true
```

---

## Migration & Adoption

### Migrating from Datadog

**1. Export Datadog Dashboards:**

```bash
# Use Datadog API to export dashboard JSON
curl -X GET "https://api.datadoghq.com/api/v1/dashboard/{dashboard_id}" \
  -H "DD-API-KEY: ${DD_API_KEY}" \
  -H "DD-APPLICATION-KEY: ${DD_APP_KEY}" \
  > datadog-dashboard.json
```

**2. Convert to Grafana Format:**

Use the migration tool:

```bash
pip install datadog-to-grafana-converter
dd-to-grafana convert datadog-dashboard.json -o grafana-dashboard.json
```

**3. Update Application Configuration:**

```python
# Before (Datadog)
configure_otlp_exporter(
    endpoint="https://otlp.datadoghq.com",
    headers={"DD-API-KEY": "..."}
)

# After (Grafana Cloud)
configure_otlp_exporter(
    endpoint="https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    headers={"Authorization": "Bearer ..."}
)

# After (Self-Hosted)
configure_otlp_exporter(
    endpoint="http://otel-collector:4318"
)
```

### Grafana Cloud vs Self-Hosted Decision Matrix

| Factor | Grafana Cloud | Self-Hosted |
|--------|---------------|-------------|
| **Setup Time** | 3 minutes | 1-2 hours |
| **Infrastructure Management** | None | Full responsibility |
| **Cost** | $0-$299/month + usage | Infrastructure + labor |
| **Data Sovereignty** | Grafana Labs | Full control |
| **Scaling** | Automatic | Manual configuration |
| **High Availability** | Built-in | Must configure |
| **Upgrades** | Automatic | Manual |
| **Customization** | Limited plugins | Full customization |
| **Integration Complexity** | Low | Medium-High |
| **Best For** | Startups, cloud-first teams | Enterprises, regulated industries |

**Recommendation:**
- **< 100 developers:** Grafana Cloud (Free or Pro tier)
- **100-1000 developers:** Grafana Cloud Advanced or Self-Hosted
- **1000+ developers or regulated industry:** Self-Hosted Enterprise

### Cost Analysis

**Grafana Cloud Pricing (Estimated):**

```
Free Tier:
- 50GB metrics
- 50GB logs
- 14-day trace retention
- 3 users

Pro Tier ($299/month base + usage):
- $0.30/GB metrics
- $0.50/GB logs
- $0.50/GB traces
- Unlimited users

Example: 100GB metrics + 50GB logs + 20GB traces/month = $299 + $30 + $25 + $10 = $364/month
```

**Self-Hosted Costs (Estimated):**

```
Infrastructure:
- Kubernetes cluster: $500-2000/month (depending on scale)
- Storage (S3/equivalent): $100-500/month
- Load balancers: $50-200/month
- Data transfer: $50-300/month

Labor:
- DevOps engineer time: ~20% FTE = $3000-5000/month (amortized)

Total: ~$3700-8000/month (medium scale)

Break-even: ~1TB data/month vs Grafana Cloud
```

### Team Training Resources

**Quick Start (30 minutes):**
1. Follow [Grafana Quickstart](../grafana-quickstart.md)
2. Import pre-built dashboard
3. Run test queries
4. Set up first alert

**Deep Dive (2 hours):**
1. PromQL fundamentals
2. TraceQL query patterns
3. Dashboard customization
4. Alert rule creation
5. RBAC and permissions

**Production Readiness (1 day workshop):**
1. High-availability architecture
2. Data retention policies
3. Backup and disaster recovery
4. Performance tuning
5. Security hardening
6. Runbook creation

**External Resources:**
- Grafana Official Tutorials: <a href="https://grafana.com/tutorials" target="_blank">grafana.com/tutorials ↗</a>
- PromQL Guide: <a href="https://prometheus.io/docs/prometheus/latest/querying/basics/" target="_blank">Prometheus docs ↗</a>
- TraceQL Documentation: <a href="https://grafana.com/docs/tempo/latest/traceql/" target="_blank">Tempo TraceQL ↗</a>
- Grafana Community Forum: <a href="https://community.grafana.com" target="_blank">community.grafana.com ↗</a>

---

## Reference

### Full API Documentation

**configure_otlp_exporter():**

```python
def configure_otlp_exporter(
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    service_name: str = "genops-ai",
    service_version: str = "1.0.0",
    environment: str = "production",
    resource_attributes: Optional[Dict[str, str]] = None,
    timeout_ms: int = 10000,
    compression: str = "gzip",
    trace_sampler: str = "always_on",
    sample_rate: float = 1.0
) -> None:
    """
    Configure OpenTelemetry OTLP exporter for Grafana integration.

    Args:
        endpoint: OTLP endpoint URL (e.g., http://localhost:4318)
        headers: Optional HTTP headers (e.g., {"Authorization": "Bearer token"})
        service_name: Service name for telemetry
        service_version: Service version
        environment: Deployment environment
        resource_attributes: Additional resource attributes
        timeout_ms: Export timeout in milliseconds
        compression: Compression algorithm ("gzip" or "none")
        trace_sampler: Sampling strategy ("always_on", "always_off", "trace_id_ratio")
        sample_rate: Sampling rate (0.0-1.0) when using "trace_id_ratio"
    """
```

### Environment Variables Reference

**Core Configuration:**
```bash
OTEL_EXPORTER_OTLP_ENDPOINT         # OTLP endpoint URL
OTEL_EXPORTER_OTLP_HEADERS          # Headers (comma-separated key=value)
OTEL_SERVICE_NAME                    # Service name
OTEL_SERVICE_VERSION                 # Service version
OTEL_RESOURCE_ATTRIBUTES            # Resource attributes (key=value,key=value)
```

**Grafana Cloud:**
```bash
GRAFANA_CLOUD_OTLP_ENDPOINT         # Cloud OTLP gateway URL
GRAFANA_CLOUD_TOKEN                  # Access policy token
```

**Protocol Configuration:**
```bash
OTEL_EXPORTER_OTLP_PROTOCOL         # http/protobuf or grpc
OTEL_EXPORTER_OTLP_TIMEOUT          # Timeout in milliseconds
OTEL_EXPORTER_OTLP_COMPRESSION      # gzip or none
```

**Sampling:**
```bash
OTEL_TRACES_SAMPLER                 # always_on, always_off, trace_id_ratio
OTEL_TRACES_SAMPLER_ARG             # Sampling rate (0.0-1.0)
```

### Troubleshooting Decision Tree

```
No data in Grafana?
├─ Is OTel Collector receiving data?
│  ├─ Yes → Check Collector logs for export errors
│  └─ No → Check application OTLP configuration
│
├─ Are data sources configured?
│  ├─ Yes → Test data source connections
│  └─ No → Configure Tempo, Prometheus/Mimir, Loki
│
├─ Is data being exported from Collector?
│  ├─ Yes → Check Grafana query syntax
│  └─ No → Check Collector exporter configuration
│
└─ Are queries correct?
   ├─ PromQL errors → Check metric names and labels
   ├─ TraceQL errors → Check span attributes
   └─ LogQL errors → Check log labels
```

---

## Support & Community

### Getting Help

**Documentation:**
- GenOps Quickstart: [docs/grafana-quickstart.md](../grafana-quickstart.md)
- Query Examples: [docs/grafana-query-examples.md](../grafana-query-examples.md)
- LGTM Stack: [observability/README.md](../../observability/README.md)

**Community:**
- GitHub Issues: <a href="https://github.com/KoshiHQ/GenOps-AI/issues" target="_blank">Report bugs and request features ↗</a>
- GitHub Discussions: <a href="https://github.com/KoshiHQ/GenOps-AI/discussions" target="_blank">Ask questions and share ideas ↗</a>

**Grafana Resources:**
- Grafana Docs: <a href="https://grafana.com/docs/" target="_blank">grafana.com/docs ↗</a>
- Grafana Community: <a href="https://community.grafana.com" target="_blank">community.grafana.com ↗</a>
- Tempo Docs: <a href="https://grafana.com/docs/tempo/" target="_blank">Tempo documentation ↗</a>
- Mimir Docs: <a href="https://grafana.com/docs/mimir/" target="_blank">Mimir documentation ↗</a>
- Loki Docs: <a href="https://grafana.com/docs/loki/" target="_blank">Loki documentation ↗</a>

---

**🎉 Congratulations!** You now have a comprehensive understanding of the GenOps Grafana integration. For quick setup, see the [Grafana Quickstart Guide](../grafana-quickstart.md).

**Next Steps:**
- ✅ Complete the quickstart
- 📊 Import pre-built dashboards
- 🔍 Explore query patterns
- 📈 Set up alerting
- 🚀 Deploy to production
