# Prometheus Integration - Comprehensive Guide

Complete reference for integrating GenOps AI governance telemetry with Prometheus.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Metrics Reference](#metrics-reference)
- [Configuration Options](#configuration-options)
- [Production Patterns](#production-patterns)
- [Recording Rules](#recording-rules)
- [Alert Rules](#alert-rules)
- [Grafana Dashboards](#grafana-dashboards)
- [Performance Optimization](#performance-optimization)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

The GenOps Prometheus exporter enables governance telemetry export in Prometheus metrics format, providing:

- **Cost tracking** across all AI providers and models
- **Token usage** monitoring and efficiency metrics
- **Policy compliance** tracking and violations
- **Quality evaluation** metrics and distributions
- **Budget management** utilization and constraints
- **Performance metrics** including latency histograms

### Key Features

- ✅ **OpenTelemetry-First**: Uses OpenTelemetry metrics with Prometheus export
- ✅ **Zero-Code Setup**: Auto-instrumentation with environment configuration
- ✅ **Multi-Provider**: Unified metrics across OpenAI, Anthropic, Bedrock, Gemini, and 35+ providers
- ✅ **Production-Ready**: Sampling, cardinality controls, and performance optimization
- ✅ **Grafana Compatible**: Pre-built dashboards and query templates

---

## Architecture

### Integration Flow

```
AI Application
    ↓
GenOps Instrumentation (governance semantics)
    ↓
OpenTelemetry Metrics API
    ↓
PrometheusMetricReader (OTLP → Prometheus format)
    ↓
prometheus_client HTTP Server (/metrics endpoint)
    ↓
Prometheus Server (scraping)
    ↓
Grafana / PromQL (visualization & alerting)
```

### Design Philosophy

**Pull-Based Export**: Unlike OTLP push exporters (Datadog, Honeycomb), Prometheus **scrapes** metrics from your application's `/metrics` endpoint.

**Metrics Focus**: Prometheus specializes in time-series metrics (counters, gauges, histograms) rather than distributed traces.

**Local-First**: Common for developers to run Prometheus locally for development and testing.

---

## Installation & Setup

### Basic Installation

```bash
# Install with Prometheus support
pip install genops-ai[prometheus]

# Or install dependencies separately
pip install genops-ai prometheus-client opentelemetry-exporter-prometheus
```

### Quick Start (5 Minutes)

```python
from genops.exporters.prometheus import auto_instrument

# Start metrics server at http://localhost:8000/metrics
auto_instrument()

# Use any AI provider - metrics automatically exported
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Manual Configuration

```python
from genops.exporters.prometheus import instrument_prometheus, PrometheusConfig

# Custom configuration
config = PrometheusConfig(
    port=8001,
    namespace="myapp",
    max_label_cardinality=5000,
    sampling_rate=0.5,  # Sample 50% of operations
    exclude_labels={"operation_id"}  # Reduce cardinality
)

exporter = instrument_prometheus(
    port=config.port,
    namespace=config.namespace,
    max_label_cardinality=config.max_label_cardinality,
    sampling_rate=config.sampling_rate
)
```

### Environment Configuration

```bash
# Export configuration via environment variables
export PROMETHEUS_EXPORTER_PORT=8000
export PROMETHEUS_NAMESPACE=genops
export PROMETHEUS_URL=http://localhost:9090
export PROMETHEUS_MAX_CARDINALITY=10000
export PROMETHEUS_SAMPLING_RATE=1.0
export PROMETHEUS_INCLUDE_LABELS=team,customer_id,environment
export PROMETHEUS_EXCLUDE_LABELS=operation_id,trace_id
```

```python
from genops.exporters.prometheus import auto_instrument

# Reads environment configuration automatically
auto_instrument()
```

---

## Metrics Reference

### Cost Metrics

#### `genops_cost_total_usd`

**Type**: Counter
**Description**: Total cost of AI operations in USD
**Labels**: `provider`, `model`, `team`, `customer_id`, `environment`, `feature`

**Example**:
```prometheus
genops_cost_total_usd{provider="openai",model="gpt-4",team="ml-research"} 125.43
```

**PromQL Queries**:
```promql
# Hourly cost rate
sum(rate(genops_cost_total_usd[1h])) * 3600

# Cost by provider
sum(genops_cost_total_usd) by (provider)

# Daily cost increase
increase(genops_cost_total_usd[24h])
```

---

#### `genops_cost_by_operation_usd`

**Type**: Counter
**Description**: Cost per operation type
**Labels**: `operation_type`, `operation_id`, + standard labels

**Example**:
```prometheus
genops_cost_by_operation_usd{operation_type="completion",provider="openai"} 45.32
```

---

### Token Metrics

#### `genops_tokens_input_total`

**Type**: Counter
**Description**: Total input tokens consumed
**Labels**: Standard labels

**Example**:
```prometheus
genops_tokens_input_total{provider="openai",model="gpt-3.5-turbo"} 1250000
```

---

#### `genops_tokens_output_total`

**Type**: Counter
**Description**: Total output tokens generated
**Labels**: Standard labels

---

#### `genops_tokens_total`

**Type**: Counter
**Description**: Total tokens (input + output)
**Labels**: Standard labels

**PromQL Queries**:
```promql
# Total tokens per hour
sum(rate(genops_tokens_total[1h])) * 3600

# Input/output ratio
sum(rate(genops_tokens_output_total[5m])) / sum(rate(genops_tokens_input_total[5m]))

# Tokens per dollar (efficiency)
sum(rate(genops_tokens_total[5m])) / sum(rate(genops_cost_total_usd[5m]))
```

---

#### `genops_token_efficiency_tokens_per_usd`

**Type**: Gauge
**Description**: Tokens per dollar (cost efficiency metric)
**Labels**: Standard labels

**PromQL Queries**:
```promql
# Average efficiency by model
avg(genops_token_efficiency_tokens_per_usd) by (model)
```

---

### Policy Metrics

#### `genops_policy_violations_total`

**Type**: Counter
**Description**: Total number of policy violations
**Labels**: `policy_name`, `policy_type`, + standard labels

**Example**:
```prometheus
genops_policy_violations_total{policy_name="pii_redaction",policy_type="compliance"} 12
```

**PromQL Queries**:
```promql
# Violation rate
sum(rate(genops_policy_violations_total[5m]))

# Violations by policy
sum(genops_policy_violations_total) by (policy_name)
```

---

#### `genops_policy_evaluations_total`

**Type**: Counter
**Description**: Total number of policy evaluations
**Labels**: `policy_name`, `policy_type`, + standard labels

---

#### `genops_policy_compliance_rate_ratio`

**Type**: Gauge
**Description**: Policy compliance rate (0-1)
**Labels**: `policy_name`, `policy_type`, + standard labels

**PromQL Queries**:
```promql
# Overall compliance rate
avg(genops_policy_compliance_rate_ratio)

# Compliance by policy
genops_policy_compliance_rate_ratio{policy_name="content_filtering"}
```

---

### Evaluation Metrics

#### `genops_evaluation_score`

**Type**: Histogram
**Description**: Distribution of evaluation scores
**Labels**: `evaluation_type`, `evaluator`, + standard labels

**PromQL Queries**:
```promql
# Median evaluation score (p50)
histogram_quantile(0.5, sum(rate(genops_evaluation_score_bucket[5m])) by (le))

# 95th percentile score
histogram_quantile(0.95, sum(rate(genops_evaluation_score_bucket[5m])) by (le))

# Average score by evaluator
avg(genops_evaluation_score_sum / genops_evaluation_score_count) by (evaluator)
```

---

### Budget Metrics

#### `genops_budget_utilization_ratio`

**Type**: Gauge
**Description**: Budget utilization ratio (0-1)
**Labels**: `budget_period`, + standard labels

**Example**:
```prometheus
genops_budget_utilization_ratio{team="ml-research",budget_period="monthly"} 0.87
```

**PromQL Queries**:
```promql
# Teams near budget limit
genops_budget_utilization_ratio > 0.9

# Utilization by team
genops_budget_utilization_ratio{budget_period="monthly"}
```

---

#### `genops_budget_remaining_usd`

**Type**: Gauge
**Description**: Remaining budget in USD
**Labels**: `budget_period`, + standard labels

---

#### `genops_budget_exceeded_total`

**Type**: Counter
**Description**: Number of times budget was exceeded
**Labels**: `budget_period`, + standard labels

---

### Performance Metrics

#### `genops_operation_latency_seconds`

**Type**: Histogram
**Description**: AI operation latency distribution
**Labels**: `operation_type`, `operation_id`, + standard labels

**PromQL Queries**:
```promql
# Median latency (p50)
histogram_quantile(0.5, rate(genops_operation_latency_seconds_bucket[5m]))

# 95th percentile latency
histogram_quantile(0.95, rate(genops_operation_latency_seconds_bucket[5m]))

# 99th percentile latency
histogram_quantile(0.99, rate(genops_operation_latency_seconds_bucket[5m]))

# Latency by provider
histogram_quantile(0.95, sum(rate(genops_operation_latency_seconds_bucket[5m])) by (le, provider))
```

---

#### `genops_operation_errors_total`

**Type**: Counter
**Description**: Total number of operation errors
**Labels**: `error_type`, + operation labels

**PromQL Queries**:
```promql
# Error rate
sum(rate(genops_operation_errors_total[5m]))

# Errors by type
sum(genops_operation_errors_total) by (error_type)

# Error rate by provider
sum(rate(genops_operation_errors_total[5m])) by (provider)
```

---

#### `genops_operations_total`

**Type**: Counter
**Description**: Total number of AI operations
**Labels**: `operation_type`, `operation_id`, + standard labels

**PromQL Queries**:
```promql
# Operations per second
sum(rate(genops_operations_total[1m]))

# Operations by model
sum(rate(genops_operations_total[5m])) by (model)
```

---

## Configuration Options

### PrometheusConfig Reference

```python
from genops.exporters.prometheus import PrometheusConfig

config = PrometheusConfig(
    port=8000,                    # Metrics endpoint port
    metrics_path="/metrics",      # Metrics endpoint path
    namespace="genops",           # Metrics namespace prefix
    prometheus_url="http://localhost:9090",  # Prometheus server URL
    scrape_interval=15,           # Expected scrape interval (seconds)
    enable_recording_rules=True,  # Enable recording rules templates
    enable_alert_rules=True,      # Enable alert rules templates
    max_label_cardinality=10000,  # Max unique label combinations
    sampling_rate=1.0,            # Sampling rate (0.0-1.0)
    include_labels=set(),         # Specific labels to include
    exclude_labels=set()          # Specific labels to exclude
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROMETHEUS_EXPORTER_PORT` | Metrics endpoint port | `8000` |
| `PROMETHEUS_METRICS_PATH` | Metrics endpoint path | `/metrics` |
| `PROMETHEUS_NAMESPACE` | Metrics namespace | `genops` |
| `PROMETHEUS_URL` | Prometheus server URL | `http://localhost:9090` |
| `PROMETHEUS_SCRAPE_INTERVAL` | Scrape interval (seconds) | `15` |
| `PROMETHEUS_MAX_CARDINALITY` | Max label cardinality | `10000` |
| `PROMETHEUS_SAMPLING_RATE` | Sampling rate (0.0-1.0) | `1.0` |
| `PROMETHEUS_INCLUDE_LABELS` | Comma-separated labels to include | `` |
| `PROMETHEUS_EXCLUDE_LABELS` | Comma-separated labels to exclude | `` |

---

## Production Patterns

### High-Availability Setup

```python
from genops.exporters.prometheus import instrument_prometheus

# Instance 1
exporter_1 = instrument_prometheus(port=8000)

# Instance 2 (different port)
exporter_2 = instrument_prometheus(port=8001)
```

**Prometheus scrape config**:
```yaml
scrape_configs:
  - job_name: 'genops-ha'
    static_configs:
      - targets:
        - 'app-1:8000'
        - 'app-2:8000'
        - 'app-3:8000'
```

---

### Sampling for High-Volume Applications

```python
# Sample 10% of operations
exporter = instrument_prometheus(sampling_rate=0.1)
```

**Use cases**:
- Applications with >10k operations/minute
- Development/staging environments
- Cost optimization

---

### Label Cardinality Management

**Problem**: High-cardinality labels (e.g., `customer_id` with millions of values) can overwhelm Prometheus.

**Solutions**:

#### 1. Exclude High-Cardinality Labels

```python
exporter = instrument_prometheus(
    exclude_labels={"customer_id", "operation_id", "trace_id"}
)
```

#### 2. Include Only Essential Labels

```python
exporter = instrument_prometheus(
    include_labels={"provider", "model", "team", "environment"}
)
```

#### 3. Aggregate in Application

```python
from genops.core.context import set_governance_context

# Group customers into tiers instead of individual IDs
set_governance_context({
    "customer_tier": "enterprise",  # Instead of customer_id
    "team": "sales"
})
```

#### 4. Use Recording Rules

Pre-aggregate high-cardinality metrics (see [Recording Rules](#recording-rules) section).

---

### Multi-Instance Deployment

**Scenario**: Multiple application instances exporting metrics.

**Recommendation**: Use Prometheus service discovery or static targets.

**Kubernetes Example**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: genops-metrics
  labels:
    app: genops-ai
spec:
  ports:
  - port: 8000
    name: metrics
  selector:
    app: genops-ai
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: genops-metrics
spec:
  selector:
    matchLabels:
      app: genops-ai
  endpoints:
  - port: metrics
    interval: 15s
```

---

### Graceful Shutdown

```python
import atexit
from genops.exporters.prometheus import instrument_prometheus

exporter = instrument_prometheus()

def cleanup():
    exporter.stop()
    print("Prometheus exporter stopped")

atexit.register(cleanup)
```

---

## Recording Rules

Recording rules pre-compute frequently used queries and reduce query load.

### Cost Aggregations

```yaml
# prometheus_recording_rules.yml
groups:
  - name: genops_cost_recording
    interval: 60s
    rules:
      # Hourly cost by team
      - record: genops:cost:hourly_by_team
        expr: sum(rate(genops_cost_total_usd[1h])) by (team) * 3600

      # Hourly cost by provider
      - record: genops:cost:hourly_by_provider
        expr: sum(rate(genops_cost_total_usd[1h])) by (provider) * 3600

      # Hourly cost by customer
      - record: genops:cost:hourly_by_customer
        expr: sum(rate(genops_cost_total_usd[1h])) by (customer_id) * 3600

      # Daily cost trend
      - record: genops:cost:daily_increase
        expr: increase(genops_cost_total_usd[24h])

      # Cost per operation
      - record: genops:cost:per_operation
        expr: |
          sum(rate(genops_cost_total_usd[5m]))
          /
          sum(rate(genops_operations_total[5m]))
```

### Token Efficiency

```yaml
  - name: genops_token_efficiency
    interval: 60s
    rules:
      # Tokens per dollar by model
      - record: genops:tokens:per_dollar_by_model
        expr: |
          sum(rate(genops_tokens_total[5m])) by (model)
          /
          sum(rate(genops_cost_total_usd[5m])) by (model)

      # Input/output token ratio
      - record: genops:tokens:output_input_ratio
        expr: |
          sum(rate(genops_tokens_output_total[5m]))
          /
          sum(rate(genops_tokens_input_total[5m]))
```

### Performance Aggregations

```yaml
  - name: genops_performance
    interval: 30s
    rules:
      # p50 latency by provider
      - record: genops:latency:p50_by_provider
        expr: |
          histogram_quantile(0.50,
            sum(rate(genops_operation_latency_seconds_bucket[5m])) by (le, provider)
          )

      # p95 latency by model
      - record: genops:latency:p95_by_model
        expr: |
          histogram_quantile(0.95,
            sum(rate(genops_operation_latency_seconds_bucket[5m])) by (le, model)
          )

      # Error rate
      - record: genops:errors:rate
        expr: sum(rate(genops_operation_errors_total[5m]))
```

**Load recording rules in `prometheus.yml`**:
```yaml
rule_files:
  - "prometheus_recording_rules.yml"
```

See complete template: `templates/prometheus/recording_rules.yml`

---

## Alert Rules

Production-ready alert configurations.

### Cost Alerts

```yaml
# prometheus_alert_rules.yml
groups:
  - name: genops_cost_alerts
    rules:
      # High cost rate
      - alert: HighCostRate
        expr: rate(genops_cost_total_usd[5m]) * 3600 > 10
        for: 5m
        labels:
          severity: warning
          category: cost
        annotations:
          summary: "High AI cost rate detected"
          description: "Cost rate {{ $value | humanize }}/hour exceeds $10/hour threshold"

      # Cost spike detection (>200% of baseline)
      - alert: CostSpike
        expr: |
          rate(genops_cost_total_usd[5m])
          >
          2 * avg_over_time(rate(genops_cost_total_usd[5m])[1h:5m])
        for: 10m
        labels:
          severity: critical
          category: cost
        annotations:
          summary: "AI cost spike detected"
          description: "Cost rate is >200% of baseline"

      # Team over budget
      - alert: TeamOverBudget
        expr: genops_budget_utilization_ratio{budget_period="monthly"} > 1.0
        for: 1m
        labels:
          severity: critical
          category: budget
        annotations:
          summary: "Team {{ $labels.team }} exceeded budget"
          description: "Monthly budget exceeded: {{ $value | humanizePercentage }}"

      # Budget warning (90% utilization)
      - alert: BudgetNearlyExceeded
        expr: genops_budget_utilization_ratio{budget_period="monthly"} > 0.9
        for: 5m
        labels:
          severity: warning
          category: budget
        annotations:
          summary: "Team {{ $labels.team }} budget nearly exceeded"
          description: "Budget utilization: {{ $value | humanizePercentage }}"
```

### Policy Alerts

```yaml
  - name: genops_policy_alerts
    rules:
      # Policy violation spike
      - alert: PolicyViolationSpike
        expr: rate(genops_policy_violations_total[5m]) > 1
        for: 2m
        labels:
          severity: warning
          category: compliance
        annotations:
          summary: "Policy violation spike detected"
          description: "Violation rate: {{ $value | humanize }}/sec for policy {{ $labels.policy_name }}"

      # Low compliance rate
      - alert: LowComplianceRate
        expr: genops_policy_compliance_rate_ratio < 0.95
        for: 10m
        labels:
          severity: warning
          category: compliance
        annotations:
          summary: "Low compliance rate for policy {{ $labels.policy_name }}"
          description: "Compliance rate: {{ $value | humanizePercentage }}"
```

### Performance Alerts

```yaml
  - name: genops_performance_alerts
    rules:
      # High latency (p95 > 5s)
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(genops_operation_latency_seconds_bucket[5m])
          ) > 5
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "High AI operation latency"
          description: "p95 latency: {{ $value | humanizeDuration }}"

      # High error rate (>1% of operations)
      - alert: HighErrorRate
        expr: |
          sum(rate(genops_operation_errors_total[5m]))
          /
          sum(rate(genops_operations_total[5m]))
          > 0.01
        for: 5m
        labels:
          severity: critical
          category: reliability
        annotations:
          summary: "High AI operation error rate"
          description: "Error rate: {{ $value | humanizePercentage }}"
```

**Load alert rules in `prometheus.yml`**:
```yaml
rule_files:
  - "prometheus_alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

See complete template: `templates/prometheus/alert_rules.yml`

---

## Grafana Dashboards

GenOps provides production-ready Grafana dashboards.

### Import Dashboard

**Option 1: Via Grafana UI**

1. Download template:
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/templates/prometheus/grafana_dashboard.json
```

2. In Grafana:
   - **Dashboards → Import**
   - Upload `grafana_dashboard.json`
   - Select Prometheus data source
   - Click **Import**

**Option 2: Via API**

```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GRAFANA_API_KEY" \
  -d @templates/prometheus/grafana_dashboard.json
```

### Dashboard Panels

1. **Cost Overview** - Total cost trends over time
2. **Cost by Provider** - Bar chart of costs by AI provider
3. **Cost by Model** - Top 10 most expensive models
4. **Cost by Team** - Team-level cost breakdown
5. **Cost by Customer** - Customer attribution (if enabled)
6. **Token Usage Trends** - Input/output token rates
7. **Token Efficiency** - Tokens per dollar by model
8. **Budget Utilization** - Gauge showing budget consumption
9. **Policy Violations** - Violation rate and totals
10. **Evaluation Scores** - Distribution heatmap
11. **Operation Latency** - p50, p95, p99 percentiles
12. **Error Rate** - Operations error rate

### Example Panel Queries

**Cost Over Time**:
```promql
sum(rate(genops_cost_total_usd[5m])) * 300
```

**Top 5 Models by Cost**:
```promql
topk(5, sum(genops_cost_total_usd) by (model))
```

**Token Efficiency by Model**:
```promql
sum(rate(genops_tokens_total[5m])) by (model)
/
sum(rate(genops_cost_total_usd[5m])) by (model)
```

See complete template: `templates/prometheus/grafana_dashboard.json`

---

## Performance Optimization

### Metric Cardinality Analysis

**Check current cardinality**:
```promql
# Number of unique time series per metric
count({__name__=~"genops_.*"}) by (__name__)

# Cardinality by label
count(genops_cost_total_usd) by (customer_id)
```

### Relabeling Strategies

**Prometheus relabeling** to reduce cardinality:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'genops-ai'
    static_configs:
      - targets: ['localhost:8000']

    metric_relabel_configs:
      # Drop high-cardinality labels
      - source_labels: [__name__]
        regex: 'genops_.*'
        action: labeldrop
        regex: '(operation_id|trace_id)'

      # Aggregate customer_id to tiers
      - source_labels: [customer_id]
        regex: 'enterprise_.*'
        target_label: customer_tier
        replacement: 'enterprise'
```

### Resource Requirements

**Prometheus Server**:
- CPU: 2-4 cores
- Memory: 4-8 GB (depends on retention and cardinality)
- Disk: 20-50 GB (depends on retention period)

**Application Overhead**:
- Memory: ~10-50 MB per exporter instance
- CPU: <1% additional overhead
- Network: Minimal (Prometheus scrapes, doesn't push)

### Scaling Considerations

**Horizontal Scaling**: Run multiple application instances, Prometheus scrapes all.

**Prometheus Federation**: Aggregate metrics from multiple Prometheus instances.

```yaml
# Central Prometheus federates from edge instances
scrape_configs:
  - job_name: 'federate'
    scrape_interval: 30s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{__name__=~"genops_.*"}'
    static_configs:
      - targets:
        - 'prometheus-edge-1:9090'
        - 'prometheus-edge-2:9090'
```

---

## Kubernetes Deployment

### ServiceMonitor (Prometheus Operator)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: genops-metrics
  labels:
    app: genops-ai
    metrics: prometheus
spec:
  ports:
  - port: 8000
    name: metrics
    protocol: TCP
  selector:
    app: genops-ai
  type: ClusterIP
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: genops-ai
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: genops-ai
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

### PodMonitor (Alternative)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: genops-ai
spec:
  selector:
    matchLabels:
      app: genops-ai
  podMetricsEndpoints:
  - port: metrics
    interval: 15s
```

### Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-ai
  template:
    metadata:
      labels:
        app: genops-ai
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8000
          name: metrics
        env:
        - name: PROMETHEUS_EXPORTER_PORT
          value: "8000"
        - name: PROMETHEUS_NAMESPACE
          value: "genops"
```

---

## Troubleshooting

### Metrics Not Appearing

**Check exporter is running**:
```python
from genops.exporters.prometheus import get_exporter

exporter = get_exporter()
if exporter:
    print(f"Running on port {exporter.config.port}")
else:
    print("Not initialized")
```

**Verify endpoint**:
```bash
curl http://localhost:8000/metrics | grep genops
```

**Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Cost Metrics Zero

**Verify provider instrumentation**:
```python
from genops.providers.openai import instrument_openai
client = instrument_openai()  # Explicit instrumentation
```

**Check pricing database**:
```python
from genops.providers.openai import OPENAI_PRICING
print(OPENAI_PRICING.get("gpt-4"))
```

### High Cardinality Warnings

**Solution 1: Exclude labels**:
```python
exporter = instrument_prometheus(
    exclude_labels={"customer_id", "operation_id"}
)
```

**Solution 2: Use recording rules** to pre-aggregate.

**Solution 3: Sampling**:
```python
exporter = instrument_prometheus(sampling_rate=0.1)
```

### Port Conflicts

**Use different port**:
```bash
export PROMETHEUS_EXPORTER_PORT=8001
```

Or in code:
```python
exporter = instrument_prometheus(port=8001)
```

---

## Additional Resources

- **Quickstart Guide**: [prometheus-quickstart.md](../prometheus-quickstart.md)
- **Recording Rules Template**: [templates/prometheus/recording_rules.yml](../../templates/prometheus/recording_rules.yml)
- **Alert Rules Template**: [templates/prometheus/alert_rules.yml](../../templates/prometheus/alert_rules.yml)
- **Grafana Dashboard**: [templates/prometheus/grafana_dashboard.json](../../templates/prometheus/grafana_dashboard.json)
- **Examples**: `examples/observability/prometheus_*.py`
- **GitHub Issues**: https://github.com/KoshiHQ/GenOps-AI/issues

---

**Next Steps**: [Set up Grafana dashboards](../guides/grafana-setup.md) | [Configure alerts](../guides/prometheus-alerting.md)
