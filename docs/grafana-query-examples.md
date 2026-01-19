# Grafana Query Examples for GenOps AI

Practical query examples for analyzing AI governance telemetry in Grafana using PromQL (metrics), TraceQL (traces), and LogQL (logs).

---

## 🔍 Quick Start - Which Query Language Do I Need?

**Confused about which query language to use?**

- **Want to see numbers over time?** (costs, token counts, rates) → **PromQL** (Prometheus/Mimir)
- **Want to see individual request traces?** (complete operation flows) → **TraceQL** (Tempo)
- **Want to see text logs?** (application logs, errors) → **LogQL** (Loki)

**Where to run queries:**
1. Open Grafana (http://localhost:3000 or your Grafana Cloud URL)
2. Navigate to **Explore** tab (compass icon in left sidebar)
3. Select your data source from the dropdown:
   - Choose **Prometheus** or **Mimir** for PromQL queries
   - Choose **Tempo** for TraceQL queries
   - Choose **Loki** for LogQL queries
4. Write your query and click **Run query**

**Quick Examples:**

```promql
# PromQL - Total AI cost across all models
sum(genops_cost_total_usd)
```

```traceql
# TraceQL - Find expensive operations (>$0.10)
{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 0.10 }
```

```logql
# LogQL - Find error logs
{service_name="my-ai-app"} |= "error"
```

---

## Table of Contents

- [PromQL (Metrics)](#promql-metrics)
  - [Cost Analysis](#cost-analysis)
  - [Token Usage](#token-usage)
  - [Performance Monitoring](#performance-monitoring)
  - [Budget & Policy](#budget--policy)
  - [Provider Comparison](#provider-comparison)
- [TraceQL (Traces)](#traceql-traces)
  - [Basic Trace Queries](#basic-trace-queries)
  - [Cost Attribution](#cost-attribution)
  - [Policy Enforcement](#policy-enforcement)
  - [Performance Analysis](#performance-analysis)
- [LogQL (Logs)](#logql-logs)
  - [Log Filtering](#log-filtering)
  - [Trace Correlation](#trace-correlation)
  - [Error Analysis](#error-analysis)
- [Multi-Data Source Queries](#multi-data-source-queries)
- [Dashboard Variable Queries](#dashboard-variable-queries)

---

## PromQL (Metrics)

Use these queries with **Prometheus** or **Mimir** data sources.

### Cost Analysis

#### Total AI Cost (All Time)
```promql
sum(genops_cost_total_usd)
```
**Use case:** Display total AI spend across all providers and models.

#### Cost by Provider
```promql
sum by (provider) (genops_cost_total_usd)
```
**Use case:** Pie chart showing spend distribution (OpenAI, Anthropic, Bedrock, etc.)

**Example output:**
- `provider="openai"`: 127.45
- `provider="anthropic"`: 89.32
- `provider="bedrock"`: 43.21

#### Cost by Model
```promql
sum by (model) (genops_cost_total_usd)
```
**Use case:** Identify most expensive models (GPT-4, Claude-3-Opus, etc.)

#### Cost by Team
```promql
sum by (team) (genops_cost_total_usd)
```
**Use case:** FinOps team attribution and chargeback

#### Cost per Customer
```promql
sum by (customer_id) (genops_cost_total_usd)
```
**Use case:** Customer billing and multi-tenant cost tracking

#### Hourly Cost Rate
```promql
rate(genops_cost_total_usd[1h]) * 3600
```
**Use case:** Real-time cost burn rate in $/hour

**Panel config:**
- Visualization: Stat
- Unit: USD ($)
- Decimals: 2

#### Daily Cost Increase
```promql
increase(genops_cost_total_usd[1d])
```
**Use case:** Daily spend tracking

#### 7-Day Cost Trend (Moving Average)
```promql
avg_over_time(rate(genops_cost_total_usd[1h])[7d:1h]) * 3600
```
**Use case:** Smoothed cost trend for forecasting

#### Cost by Environment
```promql
sum by (environment) (genops_cost_total_usd)
```
**Use case:** Production vs staging vs development cost allocation

**Example output:**
- `environment="production"`: 245.67
- `environment="staging"`: 12.34
- `environment="development"`: 8.91

#### Top 10 Most Expensive Operations
```promql
topk(10, sum by (operation_name) (genops_cost_total_usd))
```
**Use case:** Identify cost optimization opportunities

#### Cost per Request (Average)
```promql
sum(genops_cost_total_usd) / sum(genops_operations_total)
```
**Use case:** Unit economics analysis

---

### Token Usage

#### Total Tokens Consumed
```promql
sum(genops_tokens_input_total + genops_tokens_output_total)
```
**Use case:** Total token consumption across all models

#### Token Consumption Rate (Tokens/Second)
```promql
rate(genops_tokens_input_total[5m]) + rate(genops_tokens_output_total[5m])
```
**Use case:** Real-time token velocity monitoring

#### Input vs Output Token Ratio
```promql
sum(genops_tokens_input_total) / sum(genops_tokens_output_total)
```
**Use case:** Identify inefficient prompts (high input, low output)

#### Token Efficiency by Model
```promql
sum by (model) (genops_tokens_output_total) / sum by (model) (genops_tokens_input_total)
```
**Use case:** Model comparison for output efficiency

**Interpretation:**
- Ratio > 1.0: Model generates more tokens than input (expansive)
- Ratio < 1.0: Model generates fewer tokens (concise)

#### Top 10 Models by Token Usage
```promql
topk(10, sum by (model) (genops_tokens_total))
```
**Use case:** Identify high-volume models

#### Tokens per Second by Provider
```promql
sum by (provider) (rate(genops_tokens_total[5m]))
```
**Use case:** Provider throughput comparison

#### Daily Token Budget Consumption
```promql
increase(genops_tokens_total{team="ai-engineering"}[1d])
```
**Use case:** Team-level token budget tracking

#### Cost per Token (Average)
```promql
sum(genops_cost_total_usd) / sum(genops_tokens_total)
```
**Use case:** Provider cost efficiency comparison

**Example interpretation:**
- $0.00003/token = GPT-3.5-turbo
- $0.00015/token = GPT-4

---

### Performance Monitoring

#### Average Operation Latency
```promql
avg(genops_operation_duration_ms)
```
**Use case:** Overall AI operation performance

#### p50 (Median) Latency
```promql
histogram_quantile(0.50, sum by (le) (rate(genops_operation_duration_ms_bucket[5m])))
```
**Use case:** Typical user experience latency

#### p95 Latency
```promql
histogram_quantile(0.95, sum by (le) (rate(genops_operation_duration_ms_bucket[5m])))
```
**Use case:** 95th percentile SLA tracking

#### p99 Latency (Tail Latency)
```promql
histogram_quantile(0.99, sum by (le) (rate(genops_operation_duration_ms_bucket[5m])))
```
**Use case:** Worst-case performance monitoring

#### p95 Latency by Model
```promql
histogram_quantile(0.95, sum by (model, le) (rate(genops_operation_duration_ms_bucket[5m])))
```
**Use case:** Identify slow models

#### Operations per Second
```promql
rate(genops_operations_total[1m])
```
**Use case:** Request rate monitoring

#### Error Rate (Percentage)
```promql
(sum(rate(genops_operations_total{status="error"}[5m])) / sum(rate(genops_operations_total[5m]))) * 100
```
**Use case:** Reliability tracking and alerting

**Alert threshold:** Error rate > 5%

#### Success Rate (Percentage)
```promql
(sum(rate(genops_operations_total{status="success"}[5m])) / sum(rate(genops_operations_total[5m]))) * 100
```
**Use case:** SLA compliance

#### Timeout Rate
```promql
(sum(rate(genops_operations_total{status="timeout"}[5m])) / sum(rate(genops_operations_total[5m]))) * 100
```
**Use case:** Identify models with frequent timeouts

#### Average Latency by Provider
```promql
avg by (provider) (genops_operation_duration_ms)
```
**Use case:** Provider performance comparison

---

### Budget & Policy

#### Budget Utilization Percentage
```promql
(sum(genops_budget_consumed_usd) / sum(genops_budget_limit_usd)) * 100
```
**Use case:** Budget exhaustion monitoring

**Panel config:**
- Visualization: Gauge
- Thresholds:
  - Green: 0-70%
  - Yellow: 70-90%
  - Red: 90-100%

#### Budget Remaining
```promql
sum(genops_budget_remaining_usd)
```
**Use case:** Available budget display

#### Budget Utilization by Team
```promql
(sum by (team) (genops_budget_consumed_usd) / sum by (team) (genops_budget_limit_usd)) * 100
```
**Use case:** Multi-team budget tracking

#### Policy Violations per Hour
```promql
rate(genops_policy_violations_total[1h]) * 3600
```
**Use case:** Compliance monitoring

#### Policy Block Rate (Percentage)
```promql
(sum(rate(genops_policy_violations_total{result="blocked"}[5m])) / sum(rate(genops_operations_total[5m]))) * 100
```
**Use case:** Policy enforcement effectiveness

#### Policy Violations by Type
```promql
sum by (policy_name) (genops_policy_violations_total)
```
**Use case:** Identify most frequently violated policies

**Example output:**
- `policy_name="pii_detection"`: 23
- `policy_name="cost_limit"`: 12
- `policy_name="content_filter"`: 8

#### Policy Evaluation Latency (Average)
```promql
avg(genops_policy_evaluation_time_ms)
```
**Use case:** Policy engine performance

#### Operations Blocked by Policy
```promql
sum(genops_operations_total{policy_result="blocked"})
```
**Use case:** Total blocked operations count

---

### Provider Comparison

#### Cost Efficiency by Provider (Cost per 1K Tokens)
```promql
(sum by (provider) (genops_cost_total_usd) / sum by (provider) (genops_tokens_total)) * 1000
```
**Use case:** Identify cheapest provider for workload

#### Latency by Provider
```promql
avg by (provider) (genops_operation_duration_ms)
```
**Use case:** Provider performance benchmarking

#### Error Rate by Provider
```promql
(sum by (provider) (rate(genops_operations_total{status="error"}[5m])) / sum by (provider) (rate(genops_operations_total[5m]))) * 100
```
**Use case:** Provider reliability comparison

#### Request Rate by Provider
```promql
sum by (provider) (rate(genops_operations_total[1m]))
```
**Use case:** Provider workload distribution

#### Cost per Request by Provider
```promql
sum by (provider) (genops_cost_total_usd) / sum by (provider) (genops_operations_total)
```
**Use case:** Provider unit economics

---

## TraceQL (Traces)

Use these queries with **Tempo** data source in Grafana **Explore**.

### Basic Trace Queries

#### All Traces for a Service
```traceql
{ resource.service.name="my-ai-app" }
```
**Use case:** View all AI operations for your service

#### Traces in Last Hour
```traceql
{ resource.service.name="my-ai-app" && duration > 0ms }
```
**Time range:** Set to "Last 1 hour" in Grafana

#### Traces for Specific Team
```traceql
{ resource.service.name="my-ai-app" && span.genops.team="ai-engineering" }
```
**Use case:** Team-level trace filtering

#### Traces for Specific Customer
```traceql
{ resource.service.name="my-ai-app" && span.genops.customer_id="enterprise-123" }
```
**Use case:** Customer support and debugging

#### Traces for Specific Environment
```traceql
{ resource.service.name="my-ai-app" && resource.deployment.environment="production" }
```
**Use case:** Production-only trace analysis

---

### Cost Attribution

#### Expensive Traces (>$0.10)
```traceql
{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 0.10 }
```
**Use case:** Identify costly operations for optimization

#### Very Expensive Traces (>$1.00)
```traceql
{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 1.00 }
```
**Use case:** Cost anomaly detection

#### Traces Using GPT-4
```traceql
{ resource.service.name="my-ai-app" && span.genops.cost.model="gpt-4" }
```
**Use case:** Track expensive model usage

#### Traces Using Claude-3-Opus
```traceql
{ resource.service.name="my-ai-app" && span.genops.cost.model="claude-3-opus-20240229" }
```
**Use case:** Provider-specific tracking

#### Average Cost per Trace
```traceql
{ resource.service.name="my-ai-app" } | avg(span.genops.cost.total_usd)
```
**Use case:** Unit economics calculation

**Example output:** `0.0234` (average $0.02 per operation)

#### Total Cost in Time Range (Aggregate)
```traceql
{ resource.service.name="my-ai-app" } | sum(span.genops.cost.total_usd)
```
**Use case:** Time-range cost analysis

#### Max Cost Trace
```traceql
{ resource.service.name="my-ai-app" } | max(span.genops.cost.total_usd)
```
**Use case:** Find single most expensive operation

#### Cost by Provider (Aggregate)
```traceql
{ resource.service.name="my-ai-app" && span.genops.cost.provider="openai" } | sum(span.genops.cost.total_usd)
```
**Use case:** Provider-specific cost aggregation

---

### Policy Enforcement

#### Policy Violations (Blocked Operations)
```traceql
{ resource.service.name="my-ai-app" && span.genops.policy.status="blocked" }
```
**Use case:** View all blocked operations

#### Policy Warnings
```traceql
{ resource.service.name="my-ai-app" && span.genops.policy.status="warning" }
```
**Use case:** Near-violation monitoring

#### Specific Policy Violations
```traceql
{ resource.service.name="my-ai-app" && span.genops.policy.name="pii_detection" && span.genops.policy.status="blocked" }
```
**Use case:** Track specific policy enforcement

#### Traces with Policy Evaluation
```traceql
{ resource.service.name="my-ai-app" && span.genops.policy.name != "" }
```
**Use case:** All operations with policy checks

#### Slow Policy Evaluations (>100ms)
```traceql
{ resource.service.name="my-ai-app" && span.genops.policy.evaluation_time_ms > 100 }
```
**Use case:** Policy engine performance issues

---

### Performance Analysis

#### Slow Traces (>5 seconds)
```traceql
{ resource.service.name="my-ai-app" && duration > 5s }
```
**Use case:** Performance bottleneck identification

#### Very Slow Traces (>30 seconds)
```traceql
{ resource.service.name="my-ai-app" && duration > 30s }
```
**Use case:** Timeout investigation

#### Fast Traces (<1 second)
```traceql
{ resource.service.name="my-ai-app" && duration < 1s }
```
**Use case:** Identify efficient operations

#### Failed Operations
```traceql
{ resource.service.name="my-ai-app" && status=error }
```
**Use case:** Error investigation and debugging

#### Successful Operations Only
```traceql
{ resource.service.name="my-ai-app" && status=ok }
```
**Use case:** Baseline performance analysis

#### Average Duration by Model
```traceql
{ resource.service.name="my-ai-app" && span.genops.cost.model != "" } | avg(duration)
```
**Use case:** Model latency comparison

#### High Token Operations (>10K tokens)
```traceql
{ resource.service.name="my-ai-app" && span.genops.tokens.total > 10000 }
```
**Use case:** Identify token-heavy operations

---

## LogQL (Logs)

Use these queries with **Loki** data source in Grafana **Explore**.

### Log Filtering

#### All Logs for Service
```logql
{service_name="my-ai-app"}
```
**Use case:** View all application logs

#### Logs for Specific Team
```logql
{service_name="my-ai-app", team="ai-engineering"}
```
**Use case:** Team-filtered log viewing

#### Logs from Production Environment
```logql
{service_name="my-ai-app", environment="production"}
```
**Use case:** Production-only log analysis

#### Logs Containing "error"
```logql
{service_name="my-ai-app"} |= "error"
```
**Use case:** Error log filtering

#### Logs Containing "cost"
```logql
{service_name="my-ai-app"} |= "cost"
```
**Use case:** Cost-related log analysis

#### Logs NOT Containing "health"
```logql
{service_name="my-ai-app"} != "health"
```
**Use case:** Filter out health check noise

#### Case-Insensitive Search
```logql
{service_name="my-ai-app"} |= `(?i)error`
```
**Use case:** Find "error", "Error", "ERROR"

---

### Trace Correlation

#### Logs for Specific Trace ID
```logql
{service_name="my-ai-app"} |= "trace_id=abc123def456"
```
**Use case:** Find all logs related to a specific trace

#### Logs with Any Trace ID
```logql
{service_name="my-ai-app"} | json | trace_id != ""
```
**Use case:** All logs with distributed tracing context

#### Logs for Expensive Operations (via trace)
```logql
{service_name="my-ai-app"} | json | cost_total_usd > 0.10
```
**Use case:** Log analysis for high-cost operations

---

### Error Analysis

#### Error Log Count (Last Hour)
```logql
count_over_time({service_name="my-ai-app"} |= "error" [1h])
```
**Use case:** Error rate monitoring

#### Error Logs by Level
```logql
{service_name="my-ai-app"} | json | level="ERROR"
```
**Use case:** Structured error log filtering

#### Rate of Error Logs
```logql
rate({service_name="my-ai-app"} |= "error" [5m])
```
**Use case:** Real-time error rate

**Panel config:**
- Visualization: Time series
- Y-axis: Logs per second

#### Top 10 Error Messages
```logql
topk(10, sum by (error_message) (count_over_time({service_name="my-ai-app"} | json [1h])))
```
**Use case:** Most common errors

#### Policy Violation Logs
```logql
{service_name="my-ai-app"} | json | policy_result="blocked"
```
**Use case:** Policy enforcement investigation

#### Policy Violation Count
```logql
count_over_time({service_name="my-ai-app"} | json | policy_result="blocked" [1h])
```
**Use case:** Policy violation tracking

---

## Multi-Data Source Queries

Combine metrics, traces, and logs for comprehensive analysis.

### Dashboard Example: Cost + Traces

**Panel 1: Total Cost (Prometheus)**
```promql
sum(genops_cost_total_usd)
```

**Panel 2: Recent Expensive Traces (Tempo)**
- Data source: Tempo
- Query: `{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 0.10 }`
- Visualization: Table
- Columns: Trace ID, Duration, Cost, Model

**Panel 3: Cost-Related Logs (Loki)**
```logql
{service_name="my-ai-app"} |= "cost"
```

### Dashboard Example: Performance + Errors

**Panel 1: p95 Latency (Prometheus)**
```promql
histogram_quantile(0.95, sum by (le) (rate(genops_operation_duration_ms_bucket[5m])))
```

**Panel 2: Slow Traces (Tempo)**
```traceql
{ resource.service.name="my-ai-app" && duration > 5s }
```

**Panel 3: Error Logs (Loki)**
```logql
{service_name="my-ai-app"} |= "error"
```

---

## Dashboard Variable Queries

Use these queries to create dynamic dashboard variables.

### Team Variable
```promql
label_values(genops_cost_total_usd, team)
```
**Use case:** Filter dashboard by team

**Usage in queries:**
```promql
sum by (provider) (genops_cost_total_usd{team=~"$team"})
```

### Environment Variable
```promql
label_values(genops_cost_total_usd, environment)
```
**Use case:** Switch between prod/staging/dev

### Provider Variable
```promql
label_values(genops_cost_total_usd, provider)
```
**Use case:** Filter by AI provider

### Model Variable
```promql
label_values(genops_cost_total_usd, model)
```
**Use case:** Compare models

### Customer Variable
```promql
label_values(genops_cost_total_usd, customer_id)
```
**Use case:** Multi-tenant dashboards

### Time Range Variable (Custom)
```
Name: time_range
Type: Interval
Values: 5m,15m,1h,6h,1d,7d
```
**Usage in queries:**
```promql
rate(genops_cost_total_usd[$time_range])
```

---

## Query Optimization Tips

### PromQL Best Practices

1. **Use rate() for counters:**
```promql
# Good
rate(genops_cost_total_usd[5m])

# Bad (counter values are cumulative)
genops_cost_total_usd
```

2. **Limit time ranges:**
```promql
# Good: 5-minute window
rate(genops_cost_total_usd[5m])

# Bad: 1-day window (slow)
rate(genops_cost_total_usd[1d])
```

3. **Use recording rules for expensive queries:**
```yaml
# Record frequently used queries
- record: genops:cost:rate1h
  expr: rate(genops_cost_total_usd[1h]) * 3600
```

4. **Filter early:**
```promql
# Good: Filter first, then aggregate
sum by (team) (genops_cost_total_usd{environment="production"})

# Bad: Aggregate all, filter later
sum by (team, environment) (genops_cost_total_usd)
```

### TraceQL Best Practices

1. **Use resource attributes for filtering:**
```traceql
# Good: Efficient resource filter
{ resource.service.name="my-ai-app" }

# Less efficient: Span attribute only
{ span.genops.team="ai-engineering" }
```

2. **Combine filters with AND:**
```traceql
# Good: Multiple filters
{ resource.service.name="my-ai-app" && span.genops.cost.total_usd > 0.10 && duration > 1s }
```

3. **Use aggregates for statistics:**
```traceql
# Average cost per trace
{ resource.service.name="my-ai-app" } | avg(span.genops.cost.total_usd)
```

### LogQL Best Practices

1. **Use label filters first:**
```logql
# Good: Label filter
{service_name="my-ai-app", environment="production"} |= "error"

# Less efficient: Line filter only
{service_name="my-ai-app"} |= "error"
```

2. **Parse JSON only when needed:**
```logql
# Good: Parse after line filter
{service_name="my-ai-app"} |= "error" | json

# Less efficient: Parse all logs
{service_name="my-ai-app"} | json | error_level="ERROR"
```

3. **Use structured logging:**
```python
# Application code: Structured logging
logger.info("AI operation completed", extra={
    "cost_usd": 0.05,
    "model": "gpt-4",
    "tokens": 1234,
    "trace_id": trace_id
})
```

Then query:
```logql
{service_name="my-ai-app"} | json | cost_usd > 0.10
```

---

## Additional Resources

- **Grafana Quickstart:** [docs/grafana-quickstart.md](grafana-quickstart.md)
- **Comprehensive Grafana Integration:** [docs/integrations/grafana.md](integrations/grafana.md)
- **PromQL Documentation:** <a href="https://prometheus.io/docs/prometheus/latest/querying/basics/" target="_blank">Prometheus docs ↗</a>
- **TraceQL Documentation:** <a href="https://grafana.com/docs/tempo/latest/traceql/" target="_blank">Tempo TraceQL ↗</a>
- **LogQL Documentation:** <a href="https://grafana.com/docs/loki/latest/logql/" target="_blank">Loki LogQL ↗</a>

---

**🎉 Happy Querying!** These examples should cover 90% of common AI governance analysis use cases. For more advanced queries, see the comprehensive Grafana integration guide.
