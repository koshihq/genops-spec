# Splunk Integration Guide

Complete guide for integrating GenOps AI governance telemetry with Splunk Enterprise/Cloud for enterprise log analytics, compliance monitoring, and cost attribution.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [HEC Setup & Configuration](#hec-setup--configuration)
6. [GenOps Configuration](#genops-configuration)
7. [GenOps Attribute Mapping](#genops-attribute-mapping)
8. [SPL Query Reference](#spl-query-reference)
9. [Dashboard Templates](#dashboard-templates)
10. [Alerting Patterns](#alerting-patterns)
11. [Cribl Routing Path](#cribl-routing-path)
12. [Production Deployment](#production-deployment)
13. [Performance Optimization](#performance-optimization)
14. [Security Best Practices](#security-best-practices)
15. [Troubleshooting](#troubleshooting)
16. [Examples](#examples)

---

## Overview

**GenOps AI + Splunk** enables enterprise-grade AI governance monitoring with:
- **Cost Attribution**: Track AI costs by team/project/customer across all providers
- **Compliance Monitoring**: Policy violations, audit trails, and regulatory compliance
- **Budget Enforcement**: Real-time budget threshold alerting and cost controls
- **SIEM Integration**: Security event correlation with AI governance data
- **Long-term Retention**: Compliance-ready archival for regulated industries

**Why Splunk?**
- Enterprise log analytics and SIEM capabilities
- Complex ad-hoc governance queries with SPL (Search Processing Language)
- Compliance audit trails (HIPAA, SOC 2, GDPR, PCI-DSS)
- Long-term retention and archival
- Strong presence in regulated industries (financial services, healthcare, government)

---

## Architecture

### Integration Paths

GenOps supports two paths for sending telemetry to Splunk:

#### Path 1: Direct OTLP → Splunk HEC

```
┌──────────────┐     OTLP/HTTP        ┌─────────────────┐
│  GenOps AI   │ ───────────────────> │  Splunk HEC     │
│  SDK         │  Port 8088           │  (HTTP Event    │
└──────────────┘                       │   Collector)    │
                                       └─────────────────┘
                                                │
                                                v
                                       ┌─────────────────┐
                                       │ Splunk Indexers │
                                       │ (index=genops_ai)│
                                       └─────────────────┘
```

**Use Case**: Simple deployments, direct integration
**Pros**: Minimal latency, simple configuration
**Cons**: Single destination, no transformation layer

#### Path 2: GenOps → Cribl → Splunk

```
┌──────────────┐     OTLP         ┌────────────┐     HEC     ┌──────────┐
│  GenOps AI   │ ───────────────> │   Cribl    │ ─────────> │ Splunk   │
│  SDK         │                   │  Stream    │            └──────────┘
└──────────────┘                   └────────────┘
                                        │
                                        ├─────────────────> Datadog
                                        ├─────────────────> S3
                                        └─────────────────> Other...
```

**Use Case**: Multi-destination routing, data transformation, sampling
**Pros**: Route to multiple platforms, intelligent sampling, enrichment
**Cons**: Additional infrastructure component

### Data Flow

1. **GenOps SDK** captures governance telemetry (cost, policy, budget)
2. **OTLP Exporter** sends data via OpenTelemetry Protocol
3. **Splunk HEC** receives and indexes telemetry
4. **SPL Queries** analyze governance metrics
5. **Dashboards** visualize cost, compliance, and budget trends
6. **Alerts** trigger on threshold violations

---

## Prerequisites

### Splunk Requirements

- **Splunk Enterprise** v8.0+ or **Splunk Cloud**
- **HTTP Event Collector (HEC)** enabled
- Index created (recommended: `genops_ai`)
- HEC token generated

### GenOps Requirements

- **GenOps AI** v1.0.0+
- **OpenTelemetry SDK**:
  ```bash
  pip install genops-ai
  pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
  ```

### Network Requirements

- Splunk HEC accessible on port 8088 (HTTPS) or 8089 (HTTP)
- Firewall rules allow OTLP traffic
- TLS certificate trusted (for HTTPS HEC)

---

## Quick Start

For 5-minute setup, see: [Splunk Quickstart Guide](../splunk-quickstart.md)

**TL;DR:**
1. Enable Splunk HEC and create token
2. Set environment variables:
   ```bash
   export SPLUNK_HEC_ENDPOINT="https://splunk.example.com:8088"
   export SPLUNK_HEC_TOKEN="your-hec-token"
   ```
3. Configure GenOps:
   ```python
   from genops import init
   init(
       service_name="my-ai-service",
       exporter_type="otlp",
       otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
       otlp_headers={"Authorization": "Splunk your-hec-token"}
   )
   ```
4. Send telemetry and query in Splunk:
   ```spl
   index=genops_ai | stats sum(genops.cost.total) by genops.team
   ```

---

## HEC Setup & Configuration

### Step 1: Enable HTTP Event Collector

1. Navigate to **Settings → Data Inputs → HTTP Event Collector**
2. Click **Global Settings**
3. **All Tokens**: Enabled ✓
4. **Enable SSL**: ✓ (recommended for production)
5. **HTTP Port Number**: 8088 (default)
6. Click **Save**

### Step 2: Create HEC Token

1. Navigate to **Settings → Data Inputs → HTTP Event Collector**
2. Click **New Token**
3. Configure:
   - **Name**: `genops_ai_token`
   - **Description**: `GenOps AI governance telemetry`
4. Click **Next**
5. Input Settings:
   - **Source type**: Select **Structured → _json** (or create custom `genops:telemetry`)
   - **App Context**: (leave default)
   - **Index**: Select **genops_ai** (or create new index)
   - **Enable indexer acknowledgement**: ✓ (optional, for reliability)
6. Click **Review → Submit**
7. **Copy Token Value** - save securely

### Step 3: Create Custom Index (Optional)

For better organization and retention management:

1. Navigate to **Settings → Indexes**
2. Click **New Index**
3. Configure:
   - **Index Name**: `genops_ai`
   - **Index Data Type**: Events
   - **Max Size of Entire Index**: 10 GB (adjust based on volume)
   - **Froze Path**: (optional, for archival)
   - **Retention**: 90 days (adjust for compliance requirements)
4. Click **Save**

### Step 4: Configure Sourcetype (Optional)

Create custom sourcetype for better parsing:

1. Navigate to **Settings → Source types**
2. Click **New Source Type**
3. Configure:
   - **Name**: `genops:telemetry`
   - **Category**: Structured
   - **Indexed Extractions**: JSON
   - **Event Breaks**: (leave default for JSON)
4. Click **Save**

### Step 5: Verify HEC Connectivity

```bash
# Test HEC health endpoint
curl -k https://splunk.example.com:8088/services/collector/health
# Expected: {"text":"HEC is healthy","code":200}

# Test HEC token
curl -k https://splunk.example.com:8088/services/collector \
  -H "Authorization: Splunk YOUR-HEC-TOKEN" \
  -d '{"event": "test", "sourcetype": "_json"}'
# Expected: {"text":"Success","code":0}
```

---

## GenOps Configuration

### Basic Configuration

```python
from genops import init

# Configure GenOps to send OTLP to Splunk HEC
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={
        "Authorization": "Splunk your-hec-token-here",
        "X-Splunk-Request-Channel": ""  # Optional: for load balancing
    },
    default_team="ai-platform",
    default_project="production"
)
```

### Environment Variables

```bash
# Set environment variables
export SPLUNK_HEC_ENDPOINT="https://splunk.example.com:8088"
export SPLUNK_HEC_TOKEN="your-hec-token"
export SPLUNK_INDEX="genops_ai"
export OTEL_SERVICE_NAME="my-ai-service"
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production"
```

```python
import os
from genops import init

# Use environment variables
init(
    service_name=os.getenv("OTEL_SERVICE_NAME", "genops-ai"),
    exporter_type="otlp",
    otlp_endpoint=f"{os.getenv('SPLUNK_HEC_ENDPOINT')}/services/collector/raw",
    otlp_headers={
        "Authorization": f"Splunk {os.getenv('SPLUNK_HEC_TOKEN')}"
    }
)
```

### Advanced Configuration

```python
from genops import init

# Advanced configuration with resource attributes
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={
        "Authorization": "Splunk your-hec-token",
        "X-Splunk-Request-Channel": "channel-1"  # Load balancing channel
    },
    resource_attributes={
        "service.name": "my-ai-service",
        "service.version": "1.0.0",
        "deployment.environment": "production",
        "splunk.index": "genops_ai",
        "splunk.sourcetype": "genops:telemetry",
        "host": "ai-app-server-01"
    },
    default_team="ai-platform",
    default_project="customer-support",
    default_cost_center="engineering"
)
```

### Programmatic Configuration with Splunk Integration Class

```python
from examples.observability.splunk_integration import SplunkGenOpsIntegration

# Use integration class for advanced features
splunk = SplunkGenOpsIntegration(
    splunk_hec_endpoint="https://splunk.example.com:8088",
    splunk_hec_token="your-hec-token",
    splunk_index="genops_ai",
    splunk_sourcetype="genops:telemetry",
    service_name="my-ai-service",
    environment="production"
)

# Integration class sets up OpenTelemetry automatically
# Now use GenOps normally - telemetry flows to Splunk
```

---

## GenOps Attribute Mapping

### Core Governance Attributes

GenOps captures governance-specific attributes that map to Splunk fields:

| GenOps Attribute | Splunk Field | Description | Example |
|------------------|--------------|-------------|---------|
| `genops.cost.total` | `genops.cost.total` | Total cost of operation | `0.0325` |
| `genops.cost.provider` | `genops.cost.provider` | AI provider | `openai`, `anthropic` |
| `genops.cost.model` | `genops.cost.model` | AI model used | `gpt-4`, `claude-3-opus` |
| `genops.tokens.input` | `genops.tokens.input` | Input token count | `1500` |
| `genops.tokens.output` | `genops.tokens.output` | Output token count | `500` |
| `genops.team` | `genops.team` | Team attribution | `ai-platform`, `product` |
| `genops.project` | `genops.project` | Project attribution | `customer-support`, `analytics` |
| `genops.customer_id` | `genops.customer_id` | Customer attribution | `enterprise-123`, `startup-456` |
| `genops.environment` | `genops.environment` | Deployment environment | `production`, `staging`, `dev` |
| `genops.cost_center` | `genops.cost_center` | Financial cost center | `engineering`, `sales` |
| `genops.feature` | `genops.feature` | Feature attribution | `chat`, `search`, `analysis` |

### Policy & Compliance Attributes

| GenOps Attribute | Splunk Field | Description | Example |
|------------------|--------------|-------------|---------|
| `genops.policy.name` | `genops.policy.name` | Policy identifier | `content_safety`, `data_privacy` |
| `genops.policy.result` | `genops.policy.result` | Policy evaluation result | `allowed`, `blocked` |
| `genops.policy.reason` | `genops.policy.reason` | Policy action reason | `Harmful content detected` |
| `genops.policy.confidence` | `genops.policy.confidence` | Confidence score | `0.95` |
| `genops.eval.quality` | `genops.eval.quality` | Quality evaluation score | `0.87` |
| `genops.eval.safety` | `genops.eval.safety` | Safety evaluation score | `0.92` |
| `genops.eval.privacy` | `genops.eval.privacy` | Privacy evaluation score | `0.88` |
| `genops.data.classification` | `genops.data.classification` | Data sensitivity | `PII`, `PHI`, `public` |

### Budget Attributes

| GenOps Attribute | Splunk Field | Description | Example |
|------------------|--------------|-------------|---------|
| `genops.budget.name` | `genops.budget.name` | Budget identifier | `team-daily`, `project-monthly` |
| `genops.budget.limit` | `genops.budget.limit` | Budget limit amount | `100.0` |
| `genops.budget.used` | `genops.budget.used` | Budget used amount | `87.50` |
| `genops.budget.remaining` | `genops.budget.remaining` | Budget remaining | `12.50` |
| `genops.budget.utilization` | `genops.budget.utilization` | Utilization percentage | `87.5` |

### Operational Attributes

| GenOps Attribute | Splunk Field | Description | Example |
|------------------|--------------|-------------|---------|
| `genops.operation.name` | `genops.operation.name` | Operation identifier | `ai.chat.completion`, `ai.embedding` |
| `genops.operation.type` | `genops.operation.type` | Operation category | `ai.inference`, `ai.training` |
| `genops.operation.duration_ms` | `genops.operation.duration_ms` | Operation duration | `1523` |
| `genops.request.id` | `genops.request.id` | Unique request ID | `req_abc123` |
| `genops.user.id` | `genops.user.id` | User identifier | `user_xyz789` |

---

## SPL Query Reference

### Cost Attribution Queries

#### Total Cost by Team

```spl
index=genops_ai genops.cost.total=*
| stats sum(genops.cost.total) as total_cost by genops.team
| sort -total_cost
| eval total_cost_formatted=printf("$%.2f", total_cost)
| rename genops.team as Team, total_cost_formatted as "Total Cost"
```

#### Cost by Model and Provider

```spl
index=genops_ai genops.cost.model=*
| stats sum(genops.cost.total) as total_cost
        count as request_count
        avg(genops.cost.total) as avg_cost
    by genops.cost.model, genops.cost.provider
| eval total_cost_fmt=printf("$%.2f", total_cost)
| eval avg_cost_fmt=printf("$%.4f", avg_cost)
| sort -total_cost
| rename genops.cost.model as Model,
         genops.cost.provider as Provider,
         total_cost_fmt as "Total Cost",
         request_count as Requests,
         avg_cost_fmt as "Avg Cost/Request"
```

#### Cost Trends Over Time

```spl
index=genops_ai genops.cost.total=*
| timechart span=1h sum(genops.cost.total) as total_cost by genops.project
| fillnull value=0
```

#### Customer Cost Attribution

```spl
index=genops_ai genops.cost.total=* genops.customer_id=*
| stats sum(genops.cost.total) as total_cost
        count as requests
        avg(genops.cost.total) as avg_cost_per_request
    by genops.customer_id
| eval total_cost_fmt=printf("$%.2f", total_cost)
| eval avg_cost_fmt=printf("$%.4f", avg_cost_per_request)
| sort -total_cost
| head 20
| rename genops.customer_id as "Customer ID",
         total_cost_fmt as "Total Cost",
         requests as Requests,
         avg_cost_fmt as "Avg Cost/Request"
```

#### Cost by Feature

```spl
index=genops_ai genops.feature=* genops.cost.total=*
| stats sum(genops.cost.total) as total_cost
        count as usage_count
    by genops.feature, genops.team
| eval cost_formatted=printf("$%.2f", total_cost)
| sort -total_cost
```

### Policy Compliance Queries

#### Recent Policy Violations

```spl
index=genops_ai genops.policy.result="blocked"
| table _time genops.policy.name genops.policy.reason genops.team genops.customer_id genops.operation.name
| sort -_time
| rename _time as Time,
         genops.policy.name as Policy,
         genops.policy.reason as Reason,
         genops.team as Team,
         genops.customer_id as Customer,
         genops.operation.name as Operation
```

#### Policy Violations by Type

```spl
index=genops_ai genops.policy.result="blocked"
| stats count as violations by genops.policy.name
| sort -violations
| rename genops.policy.name as "Policy Type", violations as "Violations"
```

#### Compliance Rate

```spl
index=genops_ai genops.policy.result=*
| stats count(eval(genops.policy.result="allowed")) as allowed
        count(eval(genops.policy.result="blocked")) as blocked
        count as total
| eval compliance_rate=round((allowed/total)*100, 2)
| eval compliance_pct=tostring(compliance_rate) + "%"
| table compliance_pct allowed blocked total
| rename compliance_pct as "Compliance Rate",
         allowed as Allowed,
         blocked as Blocked,
         total as "Total Requests"
```

#### Compliance Audit Trail

```spl
index=genops_ai (genops.policy.* OR genops.eval.*)
| table _time genops.operation.name genops.customer_id genops.team
         genops.policy.result genops.eval.safety genops.data.classification
| sort -_time
| rename _time as Timestamp,
         genops.operation.name as Operation,
         genops.customer_id as Customer,
         genops.team as Team,
         genops.policy.result as "Policy Result",
         genops.eval.safety as "Safety Score",
         genops.data.classification as "Data Classification"
```

### Budget Monitoring Queries

#### Budgets Over Threshold

```spl
index=genops_ai genops.budget.utilization=*
| stats max(genops.budget.utilization) as max_util by genops.budget.name, genops.team
| where max_util > 80
| eval utilization_pct=round(max_util, 1) + "%"
| sort -max_util
| rename genops.budget.name as Budget,
         genops.team as Team,
         utilization_pct as "Utilization %"
```

#### Budget Status Details

```spl
index=genops_ai genops.budget.*
| stats max(genops.budget.limit) as limit
        max(genops.budget.used) as used
        max(genops.budget.remaining) as remaining
        max(genops.budget.utilization) as utilization
    by genops.budget.name, genops.team
| eval limit_fmt=printf("$%.2f", limit)
| eval used_fmt=printf("$%.2f", used)
| eval remaining_fmt=printf("$%.2f", remaining)
| eval utilization_pct=round(utilization, 1) + "%"
| eval status=case(
    utilization >= 90, "CRITICAL",
    utilization >= 80, "WARNING",
    utilization >= 0, "OK"
  )
| sort -utilization
| rename genops.budget.name as Budget,
         genops.team as Team,
         limit_fmt as Limit,
         used_fmt as Used,
         remaining_fmt as Remaining,
         utilization_pct as "Utilization %",
         status as Status
```

#### Real-time Cost Monitoring

```spl
index=genops_ai genops.cost.total=*
| bin _time span=5m
| stats sum(genops.cost.total) as cost_5min by _time, genops.team
| eval cost_formatted=printf("$%.4f", cost_5min)
| timechart span=5m sum(genops.cost.total) by genops.team
```

### Performance & Evaluation Queries

#### Model Performance Metrics

```spl
index=genops_ai genops.eval.*
| stats avg(genops.eval.quality) as avg_quality
        avg(genops.eval.safety) as avg_safety
        avg(genops.eval.privacy) as avg_privacy
        count as evaluations
    by genops.cost.model
| eval avg_quality_pct=round(avg_quality*100, 1) + "%"
| eval avg_safety_pct=round(avg_safety*100, 1) + "%"
| eval avg_privacy_pct=round(avg_privacy*100, 1) + "%"
| sort -avg_quality
| rename genops.cost.model as Model,
         avg_quality_pct as "Avg Quality",
         avg_safety_pct as "Avg Safety",
         avg_privacy_pct as "Avg Privacy",
         evaluations as Evaluations
```

#### Operation Duration Analysis

```spl
index=genops_ai genops.operation.duration_ms=*
| stats avg(genops.operation.duration_ms) as avg_duration
        p50(genops.operation.duration_ms) as p50_duration
        p95(genops.operation.duration_ms) as p95_duration
        p99(genops.operation.duration_ms) as p99_duration
    by genops.operation.name
| eval avg_duration_sec=round(avg_duration/1000, 2)
| eval p50_sec=round(p50_duration/1000, 2)
| eval p95_sec=round(p95_duration/1000, 2)
| eval p99_sec=round(p99_duration/1000, 2)
| sort -avg_duration
| rename genops.operation.name as Operation,
         avg_duration_sec as "Avg (s)",
         p50_sec as "P50 (s)",
         p95_sec as "P95 (s)",
         p99_sec as "P99 (s)"
```

---

## Dashboard Templates

GenOps provides pre-built Splunk dashboard XML templates for common governance use cases.

### Cost Governance Dashboard

**Features:**
- Total cost (last 24h)
- Total requests
- Average cost per request
- Cost by team (pie chart)
- Cost by model (bar chart)
- Cost trend over time (area chart)
- Top 10 customers by cost (table)

**Import:**
```bash
# Generate from Python integration
python -c "from examples.observability.splunk_integration import SplunkGenOpsIntegration; \
           s=SplunkGenOpsIntegration(); \
           print(s.create_cost_dashboard())" > cost_dashboard.xml

# Import to Splunk
splunk import dashboard cost_dashboard.xml
```

**Or manually create in Splunk:**
1. Navigate to **Search & Reporting** → **Dashboards**
2. Click **Create New Dashboard**
3. Select **Dashboard Studio**
4. Switch to **Source** view
5. Paste XML content from integration example
6. Save dashboard

### Compliance Monitoring Dashboard

**Features:**
- Policy violations (last 24h)
- Compliance rate
- Average safety score
- Violations by policy type (bar chart)
- Violations by team (pie chart)
- Violation trend over time (line chart)
- Recent policy violations (table)
- Compliance audit trail (table)

**Import:**
```bash
# Generate from Python integration
python -c "from examples.observability.splunk_integration import SplunkGenOpsIntegration; \
           s=SplunkGenOpsIntegration(); \
           print(s.create_compliance_dashboard())" > compliance_dashboard.xml

# Import to Splunk
splunk import dashboard compliance_dashboard.xml
```

### Budget Monitoring Dashboard

**Features:**
- Budgets over 80% utilized
- Total budget allocated
- Total budget consumed
- Budget utilization by team (bar chart)
- Budget status details (table)
- Budget utilization trend (line chart)

**Import:**
```bash
# Generate from Python integration
python -c "from examples.observability.splunk_integration import SplunkGenOpsIntegration; \
           s=SplunkGenOpsIntegration(); \
           print(s.create_budget_dashboard())" > budget_dashboard.xml

# Import to Splunk
splunk import dashboard budget_dashboard.xml
```

---

## Alerting Patterns

### Budget Threshold Alert

**Trigger**: Budget utilization exceeds 80%

**Search:**
```spl
index=genops_ai genops.budget.utilization=*
| stats max(genops.budget.utilization) as max_util by genops.budget.name, genops.team
| where max_util > 80
| table genops.budget.name genops.team max_util
```

**Configuration:**
1. Navigate to **Search**
2. Run the search above
3. Click **Save As → Alert**
4. Configure:
   - **Title**: `GenOps Budget Threshold Alert (>80%)`
   - **Alert Type**: Real-time
   - **Trigger Condition**: Number of Results > 0
   - **Throttle**: 5 minutes
   - **Alert Action**: Send email, trigger webhook (Slack/PagerDuty)
5. Click **Save**

### Policy Violation Alert

**Trigger**: Any policy violation detected

**Search:**
```spl
index=genops_ai genops.policy.result="blocked"
| table _time genops.policy.name genops.policy.reason genops.team genops.customer_id
```

**Configuration:**
1. Follow same steps as budget alert
2. Configure:
   - **Title**: `GenOps Policy Violation Alert`
   - **Alert Type**: Real-time
   - **Trigger Condition**: Number of Results > 0
   - **Throttle**: 1 minute (for immediate notification)
   - **Alert Action**: Send email, log to SIEM, trigger webhook

### Cost Anomaly Alert

**Trigger**: Cost exceeds 2x historical average

**Search:**
```spl
index=genops_ai genops.cost.total=*
| bin _time span=1h
| stats sum(genops.cost.total) as hourly_cost by _time, genops.team
| eventstats avg(hourly_cost) as avg_hourly_cost by genops.team
| where hourly_cost > (avg_hourly_cost * 2)
| table _time genops.team hourly_cost avg_hourly_cost
```

**Configuration:**
1. Follow same steps as budget alert
2. Configure:
   - **Title**: `GenOps Cost Anomaly Alert (>2x avg)`
   - **Alert Type**: Scheduled (every hour)
   - **Trigger Condition**: Number of Results > 0
   - **Throttle**: 1 hour
   - **Alert Action**: Send email, create incident ticket

---

## Cribl Routing Path

GenOps can route telemetry to Splunk via **Cribl Stream** for:
- Multi-destination routing (Splunk + Datadog + S3 simultaneously)
- Intelligent sampling (reduce volume by 90%+)
- Data enrichment and transformation
- Cost optimization with conditional routing

### Architecture: GenOps → Cribl → Splunk

```
┌──────────────┐     OTLP         ┌─────────────┐     HEC      ┌──────────┐
│  GenOps AI   │ ───────────────> │   Cribl     │ ──────────> │ Splunk   │
│  SDK         │  Port 4318       │   Stream    │             └──────────┘
└──────────────┘                   └─────────────┘
                                        │
                                        ├────────────────────> Datadog
                                        ├────────────────────> S3
                                        └────────────────────> Others...
```

### Configuration Steps

#### 1. Configure GenOps → Cribl OTLP Endpoint

```bash
export CRIBL_OTLP_ENDPOINT="http://cribl-stream:4318"
export CRIBL_AUTH_TOKEN="your-cribl-token"  # Optional
```

```python
from genops import init

# Send to Cribl instead of directly to Splunk
init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="http://cribl-stream:4318",
    otlp_headers={
        "Authorization": "Bearer your-cribl-token"
    }
)
```

#### 2. Add Splunk HEC Destination in Cribl

1. Navigate to **Cribl Stream UI**
2. Go to **Data → Destinations → Splunk HEC**
3. Click **Add Destination**
4. Configure:
   - **Destination ID**: `splunk_hec_genops`
   - **HEC Endpoint**: `https://splunk.example.com:8088`
   - **Token**: (your Splunk HEC token)
   - **Default Index**: `genops_ai`
   - **Default Source Type**: `genops:telemetry`
5. Click **Save**

#### 3. Create Routing Rule in Cribl

1. Navigate to **Data → Routes**
2. Click **Add Route**
3. Configure:
   - **Route ID**: `genops_to_splunk`
   - **Filter**: `__inputId == 'genops_otlp_source'`
   - **Output**: Select `splunk_hec_genops`
   - **Pipeline**: (optional) Select processing pipeline
4. Click **Save**

#### 4. Optional: Add Sampling Pipeline

Create pipeline for intelligent sampling:

1. Navigate to **Processing → Pipelines**
2. Click **Add Pipeline**
3. Configure:
   - **Pipeline ID**: `genops_sampling`
   - Add **Sample** function:
     - **Sample Rate**: 0.1 (10% sampling)
     - **Filter**: `genops.cost.total < 0.001` (sample low-cost operations)
   - Add **Parser** function:
     - **Type**: JSON
     - **Source Field**: `_raw`
   - Add **Eval** function:
     - **Expression**: `enrichment_timestamp = Date.now()`
4. Click **Save**

#### 5. Multi-Destination Routing

Route GenOps telemetry to multiple destinations simultaneously:

1. Create multiple destinations (Splunk, Datadog, S3)
2. Update routing rule:
   - **Route ID**: `genops_multi_destination`
   - **Filter**: `__inputId == 'genops_otlp_source'`
   - **Outputs**: Select multiple destinations:
     - `splunk_hec_genops`
     - `datadog_api`
     - `s3_archive`
3. Click **Save**

### Benefits of Cribl Routing

✅ **Multi-Destination**: Route to Splunk, Datadog, S3, and 100+ destinations
✅ **Cost Optimization**: Sample low-value telemetry, keep high-value events
✅ **Data Enrichment**: Add metadata, transform attributes
✅ **Compliance**: Route sensitive data to compliant storage
✅ **Reliability**: Cribl handles retries, buffering, backpressure

---

## Production Deployment

### Index Sizing & Capacity Planning

**Estimate Daily Volume:**
```
Operations/day: 100,000
Avg event size: 2 KB
Daily volume: 100,000 × 2 KB = 200 MB/day
Monthly volume: 200 MB × 30 = 6 GB/month
```

**Index Configuration:**
- **Max Size**: 10 GB (adjust based on volume)
- **Retention**: 90 days (adjust for compliance)
- **Frozen Path**: Configure for archival beyond retention

### Performance Tuning

#### HEC Configuration

Optimize HEC for high throughput:

```
# splunk/local/inputs.conf
[http]
disabled = 0
port = 8088
enableSSL = 1
dedicatedIoThreads = 4
maxThreads = 0
maxSockets = 0
useDeploymentServer = 0

[http://genops_ai_token]
token = your-hec-token
indexes = genops_ai
sourcetype = genops:telemetry
# Enable indexer acknowledgement for reliability
useACK = 1
```

#### Indexer Configuration

```
# splunk/local/indexes.conf
[genops_ai]
homePath = $SPLUNK_DB/genops_ai/db
coldPath = $SPLUNK_DB/genops_ai/colddb
thawedPath = $SPLUNK_DB/genops_ai/thaweddb
maxTotalDataSizeMB = 10000
maxDataSize = auto
maxHotBuckets = 3
maxWarmDBCount = 300
frozenTimePeriodInSecs = 7776000  # 90 days
```

### High Availability

#### Load-Balanced HEC

Deploy multiple HEC endpoints with load balancer:

```
┌───────────┐
│  GenOps   │
└─────┬─────┘
      │
      v
┌─────────────┐
│ Load        │
│ Balancer    │
└──┬──────┬───┘
   │      │
   v      v
┌────┐  ┌────┐
│HEC1│  │HEC2│
└────┘  └────┘
```

Configure GenOps with load-balanced endpoint:
```python
init(
    otlp_endpoint="https://lb.splunk.example.com:8088/services/collector/raw",
    otlp_headers={"Authorization": "Splunk your-token"}
)
```

#### Indexer Cluster

Deploy Splunk Indexer Cluster for high availability:
- Replication factor: 3
- Search factor: 2
- Automatic failover

---

## Security Best Practices

### TLS/SSL Configuration

Always use HTTPS for HEC in production:

```python
init(
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={"Authorization": "Splunk your-hec-token"}
)
```

Verify TLS certificate:
```bash
openssl s_client -connect splunk.example.com:8088 -showcerts
```

### Token Management

**Generate Strong Tokens:**
```bash
# Use Splunk's built-in token generator
# Or generate cryptographically secure token
openssl rand -hex 32
```

**Rotate Tokens Regularly:**
- Create new token in Splunk
- Update GenOps configuration
- Disable old token after verification
- Schedule rotation every 90 days

**Store Tokens Securely:**
- Use environment variables (not hardcoded)
- Use secrets management (HashiCorp Vault, AWS Secrets Manager)
- Restrict token permissions to minimum required

### Network Security

**Firewall Rules:**
- Allow inbound 8088 (HTTPS HEC) from GenOps servers only
- Deny all other sources
- Use VPC/VNET peering for cloud deployments

**IP Allowlisting:**
Configure in Splunk HEC:
```
# splunk/local/inputs.conf
[http]
allowFrom = 10.0.1.0/24, 10.0.2.0/24
```

### Data Privacy

**PII/PHI Redaction:**
Use Cribl for redaction before Splunk indexing:
- Mask credit card numbers
- Redact SSN, email addresses
- Hash customer identifiers

**GDPR Compliance:**
- Implement data retention policies (90 days, 365 days, etc.)
- Configure frozen path for archival
- Document data processing activities

---

## Troubleshooting

### Issue: HEC Health Check Fails

**Symptoms:**
```bash
curl -k https://splunk.example.com:8088/services/collector/health
# Returns: Connection refused or timeout
```

**Solutions:**
1. Verify HEC is enabled:
   - Settings → Data Inputs → HTTP Event Collector → Global Settings
   - "All Tokens" should be **Enabled**

2. Check Splunk service is running:
   ```bash
   $SPLUNK_HOME/bin/splunk status
   # Should show: splunkd is running (PID: xxxxx)
   ```

3. Verify port 8088 is listening:
   ```bash
   netstat -an | grep 8088
   # Should show: LISTEN state
   ```

4. Check firewall rules:
   ```bash
   # Linux
   iptables -L -n | grep 8088

   # macOS
   sudo pfctl -sr | grep 8088
   ```

### Issue: Token Authentication Fails (403 Forbidden)

**Symptoms:**
```bash
curl -k https://splunk.example.com:8088/services/collector \
  -H "Authorization: Splunk YOUR-TOKEN" \
  -d '{"event": "test"}'
# Returns: {"text":"Invalid authorization","code":403}
```

**Solutions:**
1. Verify token exists and is enabled:
   - Settings → Data Inputs → HTTP Event Collector → View tokens
   - Check "Enabled" status

2. Test token manually in Splunk UI:
   - Copy token value exactly (no extra spaces)
   - Test with simple curl command

3. Check token has correct index permissions:
   - Token must have access to target index (`genops_ai`)

4. Verify Authorization header format:
   ```
   Correct:   Authorization: Splunk abc123def456
   Incorrect: Authorization: Bearer abc123def456
   ```

### Issue: No Data Appearing in Splunk

**Symptoms:**
- GenOps sends telemetry successfully
- But Splunk search returns no results: `index=genops_ai | head 10`

**Solutions:**
1. Verify index exists:
   ```spl
   | eventcount summarize=false index=* | dedup index | fields index
   ```

2. Check if data is being indexed:
   ```spl
   index=_internal source=*metrics.log component=Metrics group=http_event_collector_metrics
   | stats sum(event_count) as events by series
   ```

3. Expand time range in Splunk Search:
   - Click time range picker
   - Select "All time"
   - Re-run search

4. Verify sourcetype:
   ```spl
   index=genops_ai | stats count by sourcetype
   ```

5. Check HEC logs for errors:
   ```spl
   index=_internal sourcetype=splunkd component=HttpInputDataHandler
   | search ERROR OR WARN
   ```

### Issue: Missing GenOps Attributes

**Symptoms:**
- Data appears in Splunk
- But `genops.*` fields are missing

**Solutions:**
1. Verify OTLP exporter is configured (not console exporter):
   ```python
   init(..., exporter_type="otlp")  # Not "console"
   ```

2. Check you're using `GenOpsTelemetry().record_*()` methods:
   ```python
   from genops.core.telemetry import GenOpsTelemetry
   telemetry = GenOpsTelemetry()
   telemetry.record_cost(span, provider="openai", model="gpt-4", ...)
   ```

3. Verify spans are created with `track_enhanced()`:
   ```python
   from genops.core import track_enhanced
   with track_enhanced(operation_name="test", ...) as span:
       telemetry.record_cost(span, ...)
   ```

4. Check JSON parsing in Splunk:
   ```spl
   index=genops_ai | head 1 | spath
   ```

### Issue: High HEC Latency

**Symptoms:**
- Telemetry takes several seconds to appear in Splunk
- GenOps operations slow down

**Solutions:**
1. Enable HEC indexer acknowledgement:
   ```
   # splunk/local/inputs.conf
   [http://genops_ai_token]
   useACK = 1
   ```

2. Increase HEC threads:
   ```
   # splunk/local/inputs.conf
   [http]
   maxThreads = 0  # 0 = unlimited
   dedicatedIoThreads = 4
   ```

3. Use batch span processor (not simple):
   ```python
   # GenOps uses BatchSpanProcessor by default
   # This batches telemetry before sending to HEC
   ```

4. Consider Cribl for buffering:
   - Cribl queues telemetry during Splunk outages
   - Reduces backpressure on GenOps SDK

---

## Examples

### Example 1: Cost Tracking

```python
from genops import init
from genops.core import track_enhanced
from genops.core.telemetry import GenOpsTelemetry

# Configure Splunk integration
init(
    service_name="customer-support-ai",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={"Authorization": "Splunk your-hec-token"},
    default_team="customer-support",
    default_project="chat-assistant"
)

telemetry = GenOpsTelemetry()

# Track AI operation with cost
with track_enhanced(
    operation_name="ai.chat.completion",
    customer_id="enterprise-abc",
    feature="support-chat"
) as span:
    # Your AI operation
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Help me"}]
    )

    # Record cost
    telemetry.record_cost(
        span,
        provider="openai",
        model="gpt-4",
        input_tokens=150,
        output_tokens=300,
        total_cost=0.0195
    )

# Query in Splunk:
# index=genops_ai genops.cost.total=*
# | stats sum(genops.cost.total) by genops.customer_id
```

### Example 2: Policy Compliance

```python
from genops import init
from genops.core import track_enhanced
from genops.core.telemetry import GenOpsTelemetry

init(
    service_name="content-moderation",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={"Authorization": "Splunk your-hec-token"}
)

telemetry = GenOpsTelemetry()

with track_enhanced(
    operation_name="ai.content.moderation",
    customer_id="user-123",
    data_classification="UGC"  # User-generated content
) as span:
    # Content safety check
    safety_result = check_content_safety(user_input)

    # Record policy evaluation
    telemetry.record_policy(
        span,
        policy_name="content_safety",
        policy_result="blocked" if not safety_result.safe else "allowed",
        policy_reason=safety_result.reason,
        metadata={"confidence": safety_result.confidence}
    )

# Query in Splunk:
# index=genops_ai genops.policy.result="blocked"
# | stats count by genops.policy.name
```

### Example 3: Budget Enforcement

```python
from genops import init
from genops.core import track_enhanced
from genops.core.telemetry import GenOpsTelemetry

init(
    service_name="ai-api",
    exporter_type="otlp",
    otlp_endpoint="https://splunk.example.com:8088/services/collector/raw",
    otlp_headers={"Authorization": "Splunk your-hec-token"}
)

telemetry = GenOpsTelemetry()

# Check budget before operation
team_budget = get_team_budget("ai-research")

with track_enhanced(
    operation_name="ai.budget.check",
    team="ai-research"
) as span:
    # Record budget status
    telemetry.record_budget(
        span,
        budget_name="team-daily-budget",
        budget_limit=team_budget.limit,
        budget_used=team_budget.used,
        budget_remaining=team_budget.remaining,
        metadata={"utilization_percent": team_budget.utilization}
    )

    # Enforce budget
    if team_budget.utilization > 90:
        raise BudgetExceededError("Team budget over 90% utilized")

# Query in Splunk:
# index=genops_ai genops.budget.utilization=*
# | where genops.budget.utilization > 80
# | table genops.budget.name genops.budget.utilization genops.team
```

### Example 4: Multi-Provider Cost Analysis

```python
# Query to compare costs across multiple AI providers
spl_query = """
index=genops_ai genops.cost.total=*
| stats sum(genops.cost.total) as total_cost
        count as requests
        avg(genops.cost.total) as avg_cost_per_request
    by genops.cost.provider, genops.cost.model
| eval total_cost_fmt=printf("$%.2f", total_cost)
| eval avg_cost_fmt=printf("$%.4f", avg_cost_per_request)
| sort -total_cost
| rename genops.cost.provider as Provider,
         genops.cost.model as Model,
         total_cost_fmt as "Total Cost",
         requests as Requests,
         avg_cost_fmt as "Avg Cost/Request"
"""

# Results show cost comparison:
# Provider   Model              Total Cost  Requests  Avg Cost/Request
# openai     gpt-4              $125.50     3,200     $0.0392
# anthropic  claude-3-opus      $98.30      2,500     $0.0393
# openai     gpt-3.5-turbo      $45.20      10,000    $0.0045
```

---

## Testing & Validation

### Pre-Flight Validation

Before deploying to production, validate your Splunk HEC integration to catch configuration issues early.

#### Standalone Validation Script

Run the validation script from the command line:

```bash
cd examples/observability
python validate_splunk_setup.py
```

**With explicit credentials:**
```bash
python validate_splunk_setup.py \
  --endpoint https://splunk.example.com:8088 \
  --token YOUR_HEC_TOKEN \
  --index genops_ai
```

**Skip connectivity check** (validate config only):
```bash
python validate_splunk_setup.py --no-connectivity
```

#### Programmatic Validation

Validate within your Python code:

```python
from examples.observability.splunk_validation import validate_setup, print_validation_result

# Validate using environment variables
result = validate_setup()
print_validation_result(result)

if not result.valid:
    print("Fix errors before proceeding")
    sys.exit(1)

# Or validate with explicit credentials
result = validate_setup(
    splunk_hec_endpoint="https://splunk.example.com:8088",
    splunk_hec_token="your-hec-token",
    splunk_index="genops_ai",
    check_connectivity=True
)
```

#### Using Integration Class

```python
from examples.observability.splunk_integration import SplunkGenOpsIntegration

splunk = SplunkGenOpsIntegration()

# Quick validation with formatted output
if splunk.print_validation():
    print("Ready to send telemetry!")
else:
    print("Configuration needs fixes")

# Or get detailed validation result
result = splunk.validate_configuration()
if result.valid:
    print(f"Connected to HEC version: {result.hec_version}")
```

### Validation Checks

The validation framework performs comprehensive checks:

1. **Environment Variables**
   - `SPLUNK_HEC_ENDPOINT` is set and formatted correctly
   - `SPLUNK_HEC_TOKEN` is set and not empty
   - URL format validation (http/https, domain, port)

2. **Connectivity Tests**
   - HEC health check: `/services/collector/health`
   - Network reachability and timeout detection
   - SSL/TLS certificate validation

3. **Authentication Tests**
   - HEC token authentication with test event
   - Index write permissions verification
   - Token expiration and status checks

4. **Dependency Checks**
   - OpenTelemetry SDK installation
   - Required Python packages (requests)
   - Version compatibility

5. **Configuration Validation**
   - Index accessibility and write permissions
   - Sourcetype configuration
   - HEC global settings enabled

#### SSL Certificate Validation

**Production Recommendation:**
Always use valid SSL certificates and keep `verify_ssl=True` (default).

**Self-Signed Certificates (Development Only):**
```python
# Only for trusted development/test environments
result = validate_setup(verify_ssl=False)
```

**Security Warning:**
Disabling SSL verification (`verify_ssl=False`) makes your connection vulnerable to
man-in-the-middle attacks. Only use this in:
- Trusted internal networks
- Development/testing environments
- When using verified self-signed certificates

**Better Alternative:**
Configure your CA certificate bundle:
```bash
export REQUESTS_CA_BUNDLE=/path/to/your/ca-bundle.crt
```

**CLI Option:**
```bash
# Secure (default)
python validate_splunk_setup.py

# Self-signed certificates (development only)
python validate_splunk_setup.py --no-ssl-verify
```

### Integration Testing

#### Test Checklist

Before production deployment, verify:

- ✅ **HEC endpoint accessible** - Health check returns 200
- ✅ **HEC token authentication works** - Test event ingested successfully
- ✅ **Index write permissions verified** - Events appear in target index
- ✅ **OpenTelemetry dependencies installed** - No import errors
- ✅ **Test event successfully indexed** - Searchable in Splunk
- ✅ **SPL queries return expected results** - Cost/policy queries work
- ✅ **Dashboard XML imports correctly** - Visualizations render
- ✅ **Alerts trigger as expected** - Budget/policy alerts fire

#### Manual Integration Test

Send test telemetry and verify in Splunk:

```python
from examples.observability.splunk_integration import demonstrate_splunk_telemetry

# This will validate configuration first, then send test events
demonstrate_splunk_telemetry()
```

**Verify in Splunk Search:**
```spl
index=genops_ai earliest=-5m
| table _time genops.cost.* genops.policy.* genops.budget.*
| head 10
```

#### Automated Test Suite

If you have a test suite, add validation tests:

```python
import pytest
from examples.observability.splunk_validation import validate_setup

def test_splunk_hec_connectivity():
    """Test Splunk HEC endpoint is accessible."""
    result = validate_setup(check_connectivity=True)
    assert result.connectivity, "HEC endpoint not accessible"

def test_splunk_token_authentication():
    """Test HEC token authentication works."""
    result = validate_setup(check_connectivity=True)
    assert result.index_accessible, "HEC token authentication failed"

def test_splunk_config_validation():
    """Test environment variables are set correctly."""
    result = validate_setup(check_connectivity=False)
    assert len(result.errors) == 0, f"Config errors: {result.errors}"
```

### Common Validation Failures

#### Error: "SPLUNK_HEC_ENDPOINT not set"

**Cause:** Environment variable not configured

**Fix:**
```bash
export SPLUNK_HEC_ENDPOINT="https://splunk.example.com:8088"
```

**Verify:**
```bash
echo $SPLUNK_HEC_ENDPOINT
```

---

#### Error: "SPLUNK_HEC_TOKEN not set"

**Cause:** HEC token environment variable missing

**Fix:**
```bash
export SPLUNK_HEC_TOKEN="your-hec-token-here"
```

**Create HEC token in Splunk:**
1. Navigate to: Settings → Data Inputs → HTTP Event Collector
2. Click "New Token"
3. Configure name and settings
4. Copy token value

---

#### Error: "HEC token authentication failed (401 Unauthorized)"

**Cause:** Invalid or expired HEC token

**Fix:**
1. Verify token in Splunk UI: Settings → Data Inputs → HTTP Event Collector
2. Check token is **enabled** (not disabled)
3. Confirm token hasn't expired
4. Verify Global Settings has HEC enabled

**Test token manually:**
```bash
curl -k https://splunk.example.com:8088/services/collector \
  -H "Authorization: Splunk YOUR_TOKEN" \
  -d '{"event":"test","sourcetype":"_json"}'
```

Expected response: `{"text":"Success","code":0}`

---

#### Error: "Connection refused - HEC endpoint not accessible"

**Cause:** Network connectivity or Splunk not running

**Fix:**
1. **Check Splunk is running:**
   ```bash
   # On Splunk server
   $SPLUNK_HOME/bin/splunk status
   ```

2. **Verify port 8088 is accessible:**
   ```bash
   nc -zv splunk.example.com 8088
   # or
   telnet splunk.example.com 8088
   ```

3. **Check firewall rules:**
   - Outbound connections to port 8088 allowed
   - Splunk server firewall allows inbound on 8088

4. **Verify HEC is enabled globally:**
   - In Splunk: Settings → Data Inputs → HTTP Event Collector
   - Click "Global Settings"
   - Ensure "All Tokens" is **Enabled**

---

#### Error: "HEC token forbidden (403 Forbidden)"

**Cause:** Token lacks permissions for target index

**Fix:**
1. **Verify index exists:**
   ```spl
   | eventcount summarize=false index=*
   | dedup index
   | search index=genops_ai
   ```

2. **Check token index permissions:**
   - Settings → Data Inputs → HTTP Event Collector
   - Click on your token
   - Verify "Allowed Indexes" includes `genops_ai`

3. **Create index if missing:**
   - Settings → Indexes → New Index
   - Name: `genops_ai`
   - Configure retention and sizing

---

#### Warning: "OpenTelemetry not installed"

**Cause:** Missing Python dependencies

**Fix:**
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

**Verify installation:**
```python
import opentelemetry
print(opentelemetry.__version__)
```

---

### Validation Best Practices

1. **Run validation before every deployment**
   - Catches configuration drift
   - Verifies credentials haven't expired
   - Tests network connectivity

2. **Include validation in CI/CD pipelines**
   ```bash
   # In your CI/CD script
   python validate_splunk_setup.py || exit 1
   ```

3. **Monitor validation results**
   - Log validation failures
   - Alert on repeated failures
   - Track success rates

4. **Document environment-specific configs**
   - Development vs staging vs production endpoints
   - Different HEC tokens per environment
   - Environment-specific indexes

5. **Regular validation in production**
   - Periodic health checks (every 5 minutes)
   - Alert on validation failures
   - Automatic retry with backoff

### Troubleshooting Tips

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

result = validate_setup(check_connectivity=True)
```

**Test HEC health manually:**
```bash
curl -k https://splunk.example.com:8088/services/collector/health
# Expected: {"text":"HEC is healthy","code":200}
```

**Check Splunk internal logs:**
```spl
index=_internal source=*metrics.log component=Metrics group=http_event_collector_metrics
| stats count by name
```

**Verify index is receiving data:**
```spl
| eventcount summarize=false index=genops_ai
| eval size_mb=size_bytes/1024/1024
| table index count earliest_time latest_time size_mb
```

---

## Additional Resources

### Documentation

- **Splunk HEC Documentation**: [https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector](https://docs.splunk.com/Documentation/Splunk/latest/Data/UsetheHTTPEventCollector)
- **SPL Reference**: [https://docs.splunk.com/Documentation/Splunk/latest/SearchReference](https://docs.splunk.com/Documentation/Splunk/latest/SearchReference)
- **OpenTelemetry Specification**: [https://opentelemetry.io/docs/specs/otel/](https://opentelemetry.io/docs/specs/otel/)
- **Cribl Stream Documentation**: [https://docs.cribl.io](https://docs.cribl.io)

### GenOps Resources

- **GitHub Repository**: [https://github.com/KoshiHQ/GenOps-AI](https://github.com/KoshiHQ/GenOps-AI)
- **Quickstart Guide**: [docs/splunk-quickstart.md](../splunk-quickstart.md)
- **Example Code**: [examples/observability/splunk_integration.py](../../examples/observability/splunk_integration.py)

### Community

- **GenOps Issues**: [https://github.com/KoshiHQ/GenOps-AI/issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **GenOps Discussions**: [https://github.com/KoshiHQ/GenOps-AI/discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Splunk Community**: [https://community.splunk.com](https://community.splunk.com)

---

## Conclusion

You now have a comprehensive understanding of integrating GenOps AI governance telemetry with Splunk for:
- Enterprise-grade cost attribution and analytics
- Compliance monitoring and audit trails
- Budget enforcement and alerting
- Policy violation tracking
- Multi-provider cost analysis

For questions or support, please open an issue on GitHub or join our community discussions.

**Happy governance monitoring!** 📊🔍
