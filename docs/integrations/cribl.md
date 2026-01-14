# Cribl Integration for GenOps AI

**Complete governance-to-observability pipeline for AI systems**

## Overview

This guide demonstrates how to integrate GenOps AI with Cribl Stream to create a comprehensive telemetry governance pipeline that routes AI operations data to multiple downstream platforms.

### What is Cribl?

[Cribl](https://cribl.io) is a vendor-neutral observability pipeline platform that enables you to collect, reduce, enrich, normalize, and route telemetry data from any source to any destination at scale.

### Why GenOps + Cribl?

**GenOps** provides runtime governance for AI systems—tracking costs, enforcing policies, recording evaluations, and monitoring budgets. It emits rich OpenTelemetry spans with governance semantics.

**Cribl** provides intelligent telemetry routing—normalizing, enriching, sampling, and distributing data to 100+ observability, SIEM, and data lake platforms.

**Together**, they form a **governance-to-observability bridge**:
- GenOps = Authority & Enforcement (decides what should happen)
- Cribl = Evidence & Distribution (routes what did happen)

### Complementary Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    AI Application                             │
│                                                                │
│    ┌──────────────────────────────────────────────┐          │
│    │      GenOps AI Instrumentation                │          │
│    │  • Cost tracking    • Policy enforcement      │          │
│    │  • Token counting   • Evaluation metrics      │          │
│    │  • Budget limits    • Compliance tracking     │          │
│    └───────────────────┬──────────────────────────┘          │
└────────────────────────┼─────────────────────────────────────┘
                         │
                         │ OTLP (HTTP/gRPC)
                         │ genops.cost.*
                         │ genops.policy.*
                         │ genops.budget.*
                         │ genops.eval.*
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                    Cribl Stream                               │
│                                                                │
│    ┌──────────────────────────────────────────────┐          │
│    │  HTTP Receiver Source (OTLP)                  │          │
│    │  Endpoint: http://cribl:4318/v1/traces       │          │
│    │  Authentication: Bearer token                 │          │
│    └───────────────────┬──────────────────────────┘          │
│                        │                                       │
│    ┌──────────────────────────────────────────────┐          │
│    │            Cribl Pipelines                     │          │
│    │                                                │          │
│    │  ┌────────────────────────────────────┐      │          │
│    │  │ Pipeline 1: Cost Governance        │      │          │
│    │  │ • Parse genops.cost.* attributes   │      │          │
│    │  │ • Enrich with budget metadata      │      │          │
│    │  │ • Sample: 100% if cost > threshold │      │          │
│    │  │ • Route to Datadog/Grafana         │      │          │
│    │  └────────────────────────────────────┘      │          │
│    │                                                │          │
│    │  ┌────────────────────────────────────┐      │          │
│    │  │ Pipeline 2: Policy & Compliance    │      │          │
│    │  │ • Filter genops.policy.* events    │      │          │
│    │  │ • Route violations to SIEM         │      │          │
│    │  │ • Mask PII in evaluation metrics   │      │          │
│    │  │ • Route to Splunk/Elastic          │      │          │
│    │  └────────────────────────────────────┘      │          │
│    │                                                │          │
│    │  ┌────────────────────────────────────┐      │          │
│    │  │ Pipeline 3: Budget Alerting        │      │          │
│    │  │ • Monitor budget utilization       │      │          │
│    │  │ • Trigger alerts at 80%, 90%, 100% │      │          │
│    │  │ • Route to Slack/PagerDuty         │      │          │
│    │  └────────────────────────────────────┘      │          │
│    │                                                │          │
│    │  ┌────────────────────────────────────┐      │          │
│    │  │ Pipeline 4: Compliance Audit       │      │          │
│    │  │ • Filter regulated operations      │      │          │
│    │  │ • Preserve audit trail (7+ years)  │      │          │
│    │  │ • Route to S3/Snowflake            │      │          │
│    │  └────────────────────────────────────┘      │          │
│    └───────────────────┬──────────────────────────┘          │
└────────────────────────┼─────────────────────────────────────┘
                         │
                         │ Routed & Enriched Telemetry
                         ↓
┌──────────────────────────────────────────────────────────────┐
│           Destination Routing (Policy-Based)                  │
│                                                                │
│  ├─→ 📊 Datadog (cost dashboards & alerting)                 │
│  ├─→ 📈 Grafana/Prometheus (performance monitoring)          │
│  ├─→ 🔐 Splunk (compliance audit logs & SIEM)                │
│  ├─→ 🔍 Elastic (security analytics)                         │
│  ├─→ 💾 S3/Snowflake (long-term cost analysis & data lake)   │
│  ├─→ 🔔 Webhooks (Slack/PagerDuty for budget alerts)         │
│  └─→ 🌊 Cribl Lake (internal telemetry store)                │
└──────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before integrating GenOps with Cribl, ensure you have:

### 1. Cribl Stream Installation

- **Cribl Stream v4.0 or later** installed and running
- Access to Cribl UI for configuration
- OTLP HTTP Receiver capability enabled

**Installation Options:**
- **Cribl Cloud**: Sign up at [cribl.cloud](https://cribl.cloud)
- **Self-Hosted**: Download from [cribl.io/download](https://cribl.io/download)
- **Docker**: `docker run -p 4318:4318 cribl/cribl:latest`

### 2. GenOps AI Installation

```bash
pip install genops-ai
```

**Verify installation:**
```python
import genops
print(genops.__version__)  # Should show 0.1.0 or later
```

### 3. API Keys & Credentials

- **Cribl Authentication Token**: Generated in Cribl UI (Settings → Authentication)
- **AI Provider API Keys**: For OpenAI, Anthropic, etc. (if testing with live AI operations)
- **Downstream Platform Credentials**: For Datadog, Splunk, etc. (configured in Cribl)

### 4. Network Access

- GenOps application can reach Cribl Stream endpoint (default: `http://cribl-stream:4318`)
- Cribl Stream can reach downstream destinations (Datadog, Splunk, S3, etc.)
- Firewall rules allow OTLP HTTP/gRPC traffic on port 4318

---

## Configuration: GenOps → Cribl

### Step 1: Initialize GenOps with Cribl OTLP Endpoint

GenOps uses standard OpenTelemetry OTLP export, which Cribl ingests natively.

**Basic Configuration:**

```python
import genops

genops.init(
    service_name="my-ai-service",
    exporter_type="otlp",
    otlp_endpoint="http://cribl-stream:4318",  # Cribl OTLP HTTP receiver
    otlp_headers={
        "Authorization": "Bearer YOUR_CRIBL_AUTH_TOKEN",
        "X-Scope-OrgID": "my-organization"
    },
    default_team="ai-platform",
    default_project="genops-cribl-integration"
)
```

**Configuration Parameters:**

- **`exporter_type`**: Must be `"otlp"` for Cribl integration
- **`otlp_endpoint`**: Cribl Stream HTTP receiver endpoint (default port 4318)
- **`otlp_headers`**: Authentication and metadata headers
  - `Authorization`: Bearer token for Cribl authentication
  - `X-Scope-OrgID`: Organization identifier for multi-tenant routing
- **`default_team`**, **`default_project`**: Default governance attributes applied to all operations

### Step 2: Configure Environment-Specific Settings

**Production Configuration:**

```python
import os
import genops

# Get Cribl endpoint from environment
cribl_endpoint = os.getenv("CRIBL_OTLP_ENDPOINT", "http://cribl-stream:4318")
cribl_token = os.getenv("CRIBL_AUTH_TOKEN")

if not cribl_token:
    raise ValueError("CRIBL_AUTH_TOKEN environment variable must be set")

genops.init(
    service_name=os.getenv("SERVICE_NAME", "ai-application"),
    exporter_type="otlp",
    otlp_endpoint=cribl_endpoint,
    otlp_headers={
        "Authorization": f"Bearer {cribl_token}",
        "X-Scope-OrgID": os.getenv("ORG_ID", "default"),
        "X-GenOps-Version": genops.__version__,
        "X-Environment": os.getenv("ENVIRONMENT", "production")
    },
    default_team=os.getenv("TEAM_NAME", "default"),
    default_project=os.getenv("PROJECT_NAME", "default"),
    default_environment=os.getenv("ENVIRONMENT", "production")
)
```

**Environment Variables:**
```bash
export CRIBL_OTLP_ENDPOINT="http://cribl-stream.prod.company.com:4318"
export CRIBL_AUTH_TOKEN="your-cribl-bearer-token"
export SERVICE_NAME="customer-ai-chatbot"
export ORG_ID="acme-corp"
export TEAM_NAME="nlp-team"
export PROJECT_NAME="chatbot-v2"
export ENVIRONMENT="production"
```

### Step 3: Test Telemetry Export

**Simple Test:**

```python
from genops.providers import instrument_openai

# Instrument OpenAI client
client = instrument_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    team="test-team",
    project="cribl-integration-test"
)

# Make a test AI call
response = client.chat_completions_create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello from GenOps + Cribl!"}],
    max_tokens=50
)

print("✅ Telemetry sent to Cribl!")
print(f"Response: {response['choices'][0]['message']['content']}")
```

**Verify in Cribl UI:**
1. Navigate to **Live Data** in Cribl UI
2. Select your GenOps OTLP source
3. You should see incoming spans with `genops.*` attributes

---

## Configuration: Cribl Stream Setup

### Step 1: Create OTLP HTTP Source

In Cribl Stream UI:

1. **Navigate to Sources** → **HTTP**
2. **Click "Add Source"** or configure existing HTTP source
3. **Configure Source:**

```yaml
Source Configuration:
  Name: genops-otlp
  Input ID: genops_otlp_source
  Port: 4318
  Path: /v1/traces
  Protocol: HTTP

Authentication:
  Type: Bearer Token
  Token: <generate-secure-token>

Advanced Settings:
  Max Request Size: 10 MB
  Request Timeout: 30s
  Keep-Alive: enabled

Buffer Settings:
  Max Buffer Size: 100 MB
  Flush Interval: 10s
```

4. **Enable OTLP Parsing:**

Cribl automatically detects and parses OTLP format. Verify this is enabled:

```yaml
Data Parsing:
  Format: OTLP (auto-detected)
  Parse OTLP Attributes: Yes
  Preserve Resource Attributes: Yes
  Flatten Nested Attributes: No (preserve genops.* hierarchy)
```

5. **Test Connection:**

Use Cribl's built-in testing tool or send test data:

```bash
curl -X POST http://cribl-stream:4318/v1/traces \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "resourceSpans": [{
      "resource": {
        "attributes": [
          {"key": "service.name", "value": {"stringValue": "test-service"}}
        ]
      },
      "scopeSpans": [{
        "spans": [{
          "name": "test-span",
          "attributes": [
            {"key": "genops.cost.total", "value": {"doubleValue": 0.05}}
          ]
        }]
      }]
    }]
  }'
```

### Step 2: Configure Retry and Buffer Policies

**Retry Configuration:**

```yaml
Retry Policy:
  Max Retries: 3
  Initial Retry Delay: 1s
  Max Retry Delay: 30s
  Backoff Multiplier: 2
  Retry on Status Codes: [429, 500, 502, 503, 504]
```

**Buffer Configuration for High Availability:**

```yaml
Buffer Settings:
  Type: Persistent Queue
  Max Size: 500 MB
  Location: /opt/cribl/state/queues/genops
  Compression: gzip
  Overflow Action: Block (apply backpressure to GenOps)

Health Check:
  Interval: 30s
  Timeout: 10s
  Unhealthy Threshold: 3 consecutive failures
```

### Step 3: Set Up Health Monitoring

**Monitor Source Health:**

1. Navigate to **Monitoring** → **Sources**
2. Add alerts for GenOps source:

```yaml
Alert Rules:
  - Name: GenOps Source Down
    Condition: source_status == "down"
    Severity: Critical
    Notification: PagerDuty, Email

  - Name: GenOps High Error Rate
    Condition: error_rate > 5%
    Severity: High
    Notification: Slack

  - Name: GenOps Buffer Full
    Condition: buffer_utilization > 90%
    Severity: High
    Notification: Slack, Email
```

---

## Cribl Pipeline Definitions

Create four core pipelines to handle GenOps governance telemetry.

### Pipeline 1: GenOps Cost Governance

Routes cost telemetry to dashboards and cost monitoring platforms.

**Pipeline Configuration:**

```yaml
id: genops-cost-governance
description: Route GenOps cost telemetry to dashboards and cost platforms
enabled: true

# Filter: Only process spans with cost attributes
filters:
  - name: has_genops_cost
    description: Filter spans with genops.cost.* attributes
    expression: |
      __inputId == 'genops_otlp_source' &&
      attributes['genops.cost.total'] != null

# Processors
processors:
  # 1. Parse cost attributes into top-level fields
  - name: parse_cost_attributes
    type: eval
    description: Extract GenOps cost attributes for routing
    expressions:
      - cost_total: "parseFloat(attributes['genops.cost.total'] || '0')"
      - cost_provider: "attributes['genops.cost.provider'] || 'unknown'"
      - cost_model: "attributes['genops.cost.model'] || 'unknown'"
      - cost_currency: "attributes['genops.cost.currency'] || 'USD'"
      - tokens_input: "parseInt(attributes['genops.tokens.input'] || '0')"
      - tokens_output: "parseInt(attributes['genops.tokens.output'] || '0')"
      - tokens_total: "parseInt(attributes['genops.tokens.total'] || '0')"
      - team: "attributes['genops.team'] || 'default'"
      - project: "attributes['genops.project'] || 'default'"
      - customer_id: "attributes['genops.customer_id'] || 'none'"
      - feature: "attributes['genops.feature'] || 'none'"
      - environment: "attributes['genops.environment'] || 'unknown'"

  # 2. Enrich with budget metadata from lookup table
  - name: enrich_budget_metadata
    type: lookup
    description: Add budget information for cost analysis
    lookup_table: customer_budgets
    match_field: customer_id
    add_fields:
      - budget_limit
      - budget_tier
      - billing_account
      - cost_center
    default_values:
      budget_tier: "free"
      billing_account: "default"

  # 3. Calculate derived metrics
  - name: calculate_metrics
    type: eval
    expressions:
      - cost_per_token: "cost_total > 0 && tokens_total > 0 ? cost_total / tokens_total : 0"
      - is_high_cost: "cost_total > 1.0"  # Flag operations > $1
      - hourly_cost_estimate: "cost_total * 3600"  # Estimate cost/hour if this rate continues

  # 4. Intelligent sampling based on cost
  - name: intelligent_cost_sampling
    type: sampling
    description: Sample more aggressively for high-cost operations
    rules:
      - condition: cost_total > 10
        sample_rate: 1.0  # 100% sampling for operations > $10
        description: "High cost operations"
      - condition: cost_total > 1
        sample_rate: 0.5  # 50% sampling for operations > $1
        description: "Medium cost operations"
      - condition: customer_id != 'none' && budget_tier == 'enterprise'
        sample_rate: 1.0  # 100% sampling for enterprise customers
        description: "Enterprise customers"
      - default: 0.1  # 10% sampling for low-cost operations
        description: "Low cost operations"

  # 5. Add routing metadata
  - name: add_routing_tags
    type: eval
    expressions:
      - routing_priority: "cost_total > 5 ? 'high' : (cost_total > 1 ? 'medium' : 'low')"
      - cost_dashboard_route: "true"
      - finops_route: "cost_total > 5 ? 'true' : 'false'"

# Routes
routes:
  - name: to_datadog_cost
    description: Send cost metrics to Datadog
    destination: datadog_metrics_dest
    filter: cost_total != null
    output_format: datadog_metrics

  - name: to_grafana_prometheus
    description: Send cost metrics to Grafana/Prometheus
    destination: prometheus_remote_write
    filter: cost_total != null

  - name: to_s3_cost_analytics
    description: Store high-value operations for cost analysis
    destination: s3_cost_bucket
    filter: cost_total > 5 || is_high_cost == true
    compression: gzip

  - name: to_cribl_lake_cost
    description: Store all cost data in Cribl Lake
    destination: cribl_lake
    dataset: genops_cost
```

### Pipeline 2: GenOps Policy & Compliance

Routes policy evaluation events to SIEM and compliance platforms.

**Pipeline Configuration:**

```yaml
id: genops-policy-compliance
description: Route GenOps policy and compliance events to SIEM and audit systems
enabled: true

# Filter: Policy events
filters:
  - name: has_policy_events
    description: Filter spans with genops.policy.* attributes
    expression: |
      __inputId == 'genops_otlp_source' &&
      attributes['genops.policy.name'] != null

# Processors
processors:
  # 1. Parse policy attributes
  - name: parse_policy_attributes
    type: eval
    expressions:
      - policy_name: "attributes['genops.policy.name']"
      - policy_result: "attributes['genops.policy.result']"  # allowed, warning, blocked
      - policy_reason: "attributes['genops.policy.reason'] || 'No reason provided'"
      - policy_severity: "attributes['genops.policy.metadata.severity'] || 'medium'"
      - compliance_framework: "attributes['genops.policy.metadata.compliance_framework'] || 'none'"
      - team: "attributes['genops.team'] || 'default'"
      - customer_id: "attributes['genops.customer_id'] || 'none'"
      - environment: "attributes['genops.environment'] || 'unknown'"

  # 2. Classify policy violations
  - name: classify_violations
    type: eval
    expressions:
      - is_violation: "policy_result == 'blocked' || policy_result == 'warning'"
      - violation_severity: |
          policy_result == 'blocked' ? 'critical' :
          (policy_result == 'warning' ? 'high' : 'low')
      - requires_investigation: "policy_result == 'blocked'"
      - siem_route: "policy_result == 'blocked' || policy_result == 'warning'"

  # 3. PII masking for evaluation metrics
  - name: mask_sensitive_data
    type: mask
    description: Mask potential PII in policy metadata
    rules:
      - field: policy_reason
        type: regex
        pattern: '\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Names
        replacement: '[REDACTED_NAME]'
      - field: policy_reason
        pattern: '\b[A-Z]{2}\d{6,}\b'  # IDs
        replacement: '[REDACTED_ID]'

  # 4. Enrich with compliance context
  - name: enrich_compliance
    type: eval
    expressions:
      - audit_required: "compliance_framework != 'none' || policy_result == 'blocked'"
      - retention_years: "compliance_framework == 'HIPAA' ? 7 : (compliance_framework == 'GDPR' ? 5 : 3)"

  # 5. Sampling (100% for violations, 1% for allowed)
  - name: policy_sampling
    type: sampling
    rules:
      - condition: policy_result == 'blocked'
        sample_rate: 1.0
        description: "All blocked operations"
      - condition: policy_result == 'warning'
        sample_rate: 1.0
        description: "All warnings"
      - default: 0.01  # 1% for allowed operations
        description: "Allowed operations"

# Routes
routes:
  - name: to_splunk_siem
    description: Route violations to Splunk SIEM
    destination: splunk_hec
    filter: is_violation == true
    output_format: splunk_hec

  - name: to_elastic_security
    description: Route violations to Elastic Security
    destination: elasticsearch_security
    filter: is_violation == true

  - name: to_compliance_audit_s3
    description: Store audit trail for compliance
    destination: s3_compliance_bucket
    filter: audit_required == true
    retention: "7 years"
    encryption: AES-256

  - name: to_datadog_monitoring
    description: Send policy metrics to Datadog for monitoring
    destination: datadog_metrics_dest
    filter: policy_result != null

  - name: to_webhook_alerts
    description: Send critical violations to Slack/PagerDuty
    destination: webhook_alerts
    filter: policy_result == 'blocked' && violation_severity == 'critical'
```

### Pipeline 3: GenOps Budget Alerting

Monitors budget utilization and triggers alerts via webhooks.

**Pipeline Configuration:**

```yaml
id: genops-budget-alerting
description: Monitor GenOps budget utilization and trigger alerts
enabled: true

# Filter: Budget tracking events
filters:
  - name: has_budget_tracking
    description: Filter spans with genops.budget.* attributes
    expression: |
      __inputId == 'genops_otlp_source' &&
      attributes['genops.budget.name'] != null

# Processors
processors:
  # 1. Parse budget attributes
  - name: parse_budget_attributes
    type: eval
    expressions:
      - budget_name: "attributes['genops.budget.name']"
      - budget_limit: "parseFloat(attributes['genops.budget.allocated'] || attributes['genops.budget.limit'] || '0')"
      - budget_used: "parseFloat(attributes['genops.budget.consumed'] || attributes['genops.budget.used'] || '0')"
      - budget_remaining: "parseFloat(attributes['genops.budget.remaining'] || '0')"
      - utilization_percent: "parseFloat(attributes['genops.budget.utilization_percent'] || '0')"
      - team: "attributes['genops.team'] || 'default'"
      - project: "attributes['genops.project'] || 'default'"

  # 2. Calculate budget status
  - name: calculate_budget_status
    type: eval
    expressions:
      # Recalculate if not provided
      - actual_utilization: "budget_limit > 0 ? (budget_used / budget_limit) * 100 : 0"
      - budget_exhausted: "budget_remaining <= 0 || actual_utilization >= 100"
      - budget_critical: "actual_utilization >= 90"
      - budget_warning: "actual_utilization >= 80"
      - budget_status: |
          budget_exhausted ? 'exhausted' :
          (budget_critical ? 'critical' :
          (budget_warning ? 'warning' : 'normal'))

  # 3. Determine alert actions
  - name: alert_routing
    type: eval
    expressions:
      - should_alert: "budget_status != 'normal'"
      - alert_severity: |
          budget_exhausted ? 'critical' :
          (budget_critical ? 'high' :
          (budget_warning ? 'medium' : 'low'))
      - pagerduty_alert: "budget_exhausted || budget_critical"
      - slack_alert: "should_alert == true"

  # 4. Format alert message
  - name: format_alert_message
    type: eval
    expressions:
      - alert_title: "`Budget Alert: ${budget_name} at ${actual_utilization.toFixed(1)}%`"
      - alert_message: |
          `Budget: ${budget_name}
           Team: ${team} | Project: ${project}
           Status: ${budget_status.toUpperCase()}
           Utilization: ${actual_utilization.toFixed(1)}%
           Used: $${budget_used.toFixed(2)} / $${budget_limit.toFixed(2)}
           Remaining: $${budget_remaining.toFixed(2)}`

  # 5. Sampling (100% for alerts)
  - name: alert_sampling
    type: sampling
    rules:
      - condition: should_alert == true
        sample_rate: 1.0
        description: "All budget alerts"
      - default: 0  # Don't send normal status events
        description: "Skip normal status"

# Routes
routes:
  - name: to_pagerduty
    description: Critical budget alerts to PagerDuty
    destination: pagerduty_events
    filter: pagerduty_alert == true

  - name: to_slack
    description: Budget alerts to Slack
    destination: slack_webhook
    filter: slack_alert == true
    payload_template: |
      {
        "text": "${alert_title}",
        "attachments": [{
          "color": "${budget_status == 'exhausted' ? 'danger' : (budget_status == 'critical' ? 'warning' : '#ffcc00')}",
          "fields": [
            {"title": "Budget", "value": "${budget_name}", "short": true},
            {"title": "Team", "value": "${team}", "short": true},
            {"title": "Utilization", "value": "${actual_utilization.toFixed(1)}%", "short": true},
            {"title": "Remaining", "value": "$${budget_remaining.toFixed(2)}", "short": true},
            {"title": "Status", "value": "${budget_status.toUpperCase()}", "short": true}
          ]
        }]
      }

  - name: to_datadog_budget_metrics
    description: Send budget metrics to Datadog
    destination: datadog_metrics_dest
    filter: budget_limit > 0

  - name: to_cribl_lake_budgets
    description: Store budget tracking in Cribl Lake
    destination: cribl_lake
    dataset: genops_budgets
```

### Pipeline 4: GenOps Compliance Audit Trail

Preserves long-term audit trails for regulated operations.

**Pipeline Configuration:**

```yaml
id: genops-audit-trail
description: Preserve GenOps audit trail for compliance and regulated operations
enabled: true

# Filter: Regulated operations requiring audit trail
filters:
  - name: requires_audit_trail
    description: Operations requiring compliance audit trail
    expression: |
      __inputId == 'genops_otlp_source' &&
      (attributes['genops.compliance.audit_trail_required'] == 'true' ||
       attributes['genops.compliance.framework'] != null ||
       attributes['genops.policy.metadata.compliance_framework'] != null)

# Processors
processors:
  # 1. Parse compliance attributes
  - name: parse_compliance_attributes
    type: eval
    expressions:
      - compliance_framework: "attributes['genops.compliance.framework'] || attributes['genops.policy.metadata.compliance_framework'] || 'SOC2'"
      - data_classification: "attributes['genops.compliance.data_classification'] || 'Confidential'"
      - audit_trail_required: "true"
      - team: "attributes['genops.team'] || 'default'"
      - customer_id: "attributes['genops.customer_id'] || 'none'"
      - environment: "attributes['genops.environment'] || 'unknown'"
      - operation_name: "name"  # Span name

  # 2. Determine retention requirements
  - name: calculate_retention
    type: eval
    expressions:
      - retention_years: |
          compliance_framework == 'HIPAA' ? 7 :
          (compliance_framework == 'GDPR' ? 5 :
          (compliance_framework == 'SOC2' ? 7 :
          (compliance_framework == 'FINRA' ? 7 : 3)))
      - retention_class: |
          retention_years >= 7 ? 'long_term' :
          (retention_years >= 5 ? 'medium_term' : 'short_term')
      - storage_tier: "retention_class == 'long_term' ? 'glacier' : 'standard'"

  # 3. Add audit metadata
  - name: enrich_audit_metadata
    type: eval
    expressions:
      - audit_timestamp: "Date.now()"
      - audit_id: "`${team}-${environment}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`"
      - audit_version: "1.0"
      - immutable: "true"  # Flag for write-once storage

  # 4. Mask sensitive data (if required)
  - name: compliance_data_masking
    type: mask
    description: Mask PII/PHI according to compliance framework
    rules:
      - field: "attributes['genops.eval.metadata.*']"
        type: regex
        pattern: '\b\d{3}-\d{2}-\d{4}\b'  # SSN
        replacement: '[REDACTED_SSN]'
      - field: "attributes['genops.cost.metadata.*']"
        pattern: '\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'  # Email
        replacement: '[REDACTED_EMAIL]'

  # 5. 100% sampling for audit trail
  - name: no_sampling_audit
    type: sampling
    rules:
      - default: 1.0  # Never sample audit trail data
        description: "Preserve all audit trail events"

# Routes
routes:
  - name: to_s3_compliance_longterm
    description: Long-term storage in S3 Glacier for 7+ year retention
    destination: s3_compliance_glacier
    filter: retention_class == 'long_term'
    compression: gzip
    encryption: AES-256
    versioning: enabled
    object_lock: "COMPLIANCE mode, 7 years"

  - name: to_s3_compliance_standard
    description: Standard S3 storage for medium-term retention
    destination: s3_compliance_standard
    filter: retention_class != 'long_term'
    compression: gzip
    encryption: AES-256
    versioning: enabled

  - name: to_snowflake_audit
    description: Store audit trail in Snowflake for analytics
    destination: snowflake_audit_db
    table: genops_audit_trail
    filter: audit_trail_required == 'true'

  - name: to_cribl_lake_audit
    description: Store in Cribl Lake for searchability
    destination: cribl_lake
    dataset: genops_audit
    retention: "2555 days"  # 7 years
```

---

## Route Configuration Examples

Configure destinations for routing GenOps telemetry from Cribl pipelines.

### Destination 1: Datadog Metrics & Logs

**Datadog Configuration:**

```yaml
id: datadog_metrics_dest
type: datadog_metrics
description: Send GenOps cost and governance metrics to Datadog

# Connection
endpoint: https://api.datadoghq.com
api_key: ${DATADOG_API_KEY}  # Use environment variable or secrets manager

# Metrics Configuration
metrics:
  prefix: genops.
  tags:
    - team:${team}
    - project:${project}
    - environment:${environment}
    - source:cribl

  # Metric mappings
  mappings:
    - name: cost.total
      type: gauge
      value: ${cost_total}
      unit: USD

    - name: tokens.total
      type: count
      value: ${tokens_total}

    - name: budget.utilization
      type: gauge
      value: ${utilization_percent}
      unit: percent

    - name: policy.violations
      type: count
      value: "${is_violation ? 1 : 0}"

# Batching
batch_size: 1000
flush_interval: 10s

# Retry
max_retries: 3
retry_backoff: exponential
```

**Verify in Datadog:**
- Navigate to **Metrics Explorer**
- Search for `genops.*` metrics
- Create dashboards and monitors

### Destination 2: Splunk HTTP Event Collector (HEC)

**Splunk SIEM Configuration:**

```yaml
id: splunk_hec
type: splunk_hec
description: Send GenOps policy violations to Splunk SIEM

# Connection
endpoint: https://splunk.company.com:8088/services/collector
token: ${SPLUNK_HEC_TOKEN}
verify_tls: true

# Event Configuration
source: genops_ai
sourcetype: genops:policy:violation
index: security

# Event Format
event_template: |
  {
    "time": ${timestamp},
    "host": "${environment}",
    "source": "genops",
    "sourcetype": "genops:policy",
    "event": {
      "policy_name": "${policy_name}",
      "policy_result": "${policy_result}",
      "policy_reason": "${policy_reason}",
      "violation_severity": "${violation_severity}",
      "team": "${team}",
      "customer_id": "${customer_id}",
      "compliance_framework": "${compliance_framework}",
      "requires_investigation": ${requires_investigation}
    }
  }

# Batching
batch_size: 500
flush_interval: 5s
```

### Destination 3: AWS S3 (Compliance Storage)

**S3 Compliance Bucket Configuration:**

```yaml
id: s3_compliance_glacier
type: s3
description: Long-term compliance storage in S3 Glacier

# Connection
region: us-east-1
bucket: company-genops-compliance-audit
path: audit-trail/${_time:%Y}/${_time:%m}/${_time:%d}/
filename: genops-audit-${_time:%Y%m%d%H%M%S}-${_random:6}.json.gz

# Authentication
auth_type: iam_role  # Or access_key
role_arn: arn:aws:iam::123456789012:role/CriblGenOpsWriter

# Storage Class
storage_class: GLACIER_IR  # Glacier Instant Retrieval

# Encryption
server_side_encryption: AES256
kms_key_id: arn:aws:kms:us-east-1:123456789012:key/abc123...

# Object Lock (Compliance Mode)
object_lock: true
object_lock_mode: COMPLIANCE
object_lock_retention: 2555 days  # 7 years

# Compression
compression: gzip
compression_level: 6

# Batching (create new file every hour or 100MB)
batch_size: 100 MB
flush_interval: 3600s
```

### Destination 4: Slack Webhooks

**Slack Alerting Configuration:**

```yaml
id: slack_webhook
type: webhook
description: Send GenOps budget alerts to Slack

# Connection
url: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
method: POST
headers:
  Content-Type: application/json

# Payload Template
payload_template: |
  {
    "username": "GenOps Budget Monitor",
    "icon_emoji": ":money_with_wings:",
    "channel": "#ai-cost-alerts",
    "text": "${alert_title}",
    "attachments": [{
      "color": "${budget_status == 'exhausted' ? 'danger' : (budget_status == 'critical' ? 'warning' : '#ffcc00')}",
      "title": "Budget Alert Details",
      "fields": [
        {"title": "Budget Name", "value": "${budget_name}", "short": true},
        {"title": "Team", "value": "${team}", "short": true},
        {"title": "Utilization", "value": "${actual_utilization.toFixed(1)}%", "short": true},
        {"title": "Status", "value": "${budget_status.toUpperCase()}", "short": true},
        {"title": "Used", "value": "$${budget_used.toFixed(2)}", "short": true},
        {"title": "Remaining", "value": "$${budget_remaining.toFixed(2)}", "short": true}
      ],
      "footer": "GenOps AI + Cribl Stream",
      "ts": ${Math.floor(Date.now() / 1000)}
    }]
  }

# Retry
max_retries: 3
retry_interval: 5s
```

### Destination 5: Cribl Lake (Internal Storage)

**Cribl Lake Configuration:**

```yaml
id: cribl_lake
type: cribl_lake
description: Store GenOps telemetry in Cribl Lake for search and analytics

# Datasets
datasets:
  - name: genops_cost
    description: Cost and token usage data
    retention: 90 days
    partition_by: team, environment

  - name: genops_policy
    description: Policy violations and evaluations
    retention: 365 days
    partition_by: compliance_framework, environment

  - name: genops_budgets
    description: Budget tracking and alerts
    retention: 180 days
    partition_by: team, budget_name

  - name: genops_audit
    description: Compliance audit trail
    retention: 2555 days  # 7 years
    partition_by: compliance_framework, year

# Compression
compression: parquet
compression_codec: snappy

# Indexing
indexes:
  - customer_id
  - team
  - environment
  - compliance_framework
```

---

## Enrichment Patterns

Enrich GenOps telemetry with organizational metadata using Cribl lookup tables.

### Enrichment 1: Customer Budget Information

**Create Lookup Table:**

In Cribl UI: **Knowledge** → **Lookups** → **Add Lookup**

```csv
customer_id,budget_limit,budget_tier,billing_account,cost_center
enterprise-123,10000.00,enterprise,acct-001,engineering
enterprise-456,5000.00,enterprise,acct-002,product
startup-789,500.00,growth,acct-003,sales
free-tier-001,50.00,free,acct-004,marketing
```

**Lookup Configuration:**

```yaml
id: customer_budgets
type: csv
file: /opt/cribl/lookups/customer_budgets.csv
reload_interval: 300s  # Reload every 5 minutes
case_sensitive: false
match_type: exact

# Fields
key_field: customer_id
output_fields:
  - budget_limit
  - budget_tier
  - billing_account
  - cost_center
```

**Use in Pipeline:**

```yaml
- name: enrich_customer_budget
  type: lookup
  lookup_table: customer_budgets
  match_field: customer_id
  add_fields:
    - budget_limit
    - budget_tier
    - billing_account
    - cost_center
  default_values:
    budget_tier: "free"
    budget_limit: 50.00
```

### Enrichment 2: Team & Project Metadata

**Create Lookup Table:**

```csv
team,department,manager_email,cost_center,slack_channel
nlp-team,AI Research,research-manager@company.com,CC-001,#nlp-team
vision-team,AI Research,research-manager@company.com,CC-001,#vision-team
platform-team,Engineering,platform-lead@company.com,CC-002,#ai-platform
product-team,Product,product-vp@company.com,CC-003,#product
```

**Use in Pipeline:**

```yaml
- name: enrich_team_metadata
  type: lookup
  lookup_table: team_metadata
  match_field: team
  add_fields:
    - department
    - manager_email
    - cost_center
    - slack_channel
```

### Enrichment 3: Model Pricing Override

Allow dynamic pricing updates without code changes.

**Create Lookup Table:**

```csv
provider,model,input_cost_per_1k,output_cost_per_1k,context_window
openai,gpt-4,0.03,0.06,8192
openai,gpt-4-turbo,0.01,0.03,128000
anthropic,claude-3-opus,0.015,0.075,200000
anthropic,claude-3-sonnet,0.003,0.015,200000
```

**Use in Pipeline:**

```yaml
- name: enrich_model_pricing
  type: lookup
  lookup_table: model_pricing
  match_field: [cost_provider, cost_model]
  add_fields:
    - input_cost_per_1k
    - output_cost_per_1k
    - context_window
```

---

## Sampling Strategies

Optimize telemetry costs with intelligent sampling.

### Strategy 1: Cost-Aware Sampling

Sample more for high-cost operations, less for low-cost.

```yaml
- name: cost_aware_sampling
  type: sampling
  description: Sample based on operation cost
  rules:
    # 100% sampling for very high cost operations (>$10)
    - condition: cost_total > 10
      sample_rate: 1.0
      description: "Very high cost"

    # 50% sampling for high cost operations ($1-$10)
    - condition: cost_total > 1
      sample_rate: 0.5
      description: "High cost"

    # 10% sampling for medium cost operations ($0.10-$1)
    - condition: cost_total > 0.1
      sample_rate: 0.1
      description: "Medium cost"

    # 1% sampling for low cost operations (<$0.10)
    - default: 0.01
      description: "Low cost"
```

**Result:** Reduces telemetry volume by 90-95% while preserving high-cost operation visibility.

### Strategy 2: Policy-Based Sampling

Always preserve violations, sample normal operations.

```yaml
- name: policy_based_sampling
  type: sampling
  rules:
    # 100% for blocked operations
    - condition: policy_result == 'blocked'
      sample_rate: 1.0
      description: "Policy violations"

    # 100% for warnings
    - condition: policy_result == 'warning'
      sample_rate: 1.0
      description: "Policy warnings"

    # 1% for allowed operations
    - condition: policy_result == 'allowed'
      sample_rate: 0.01
      description: "Allowed operations"

    # 100% if compliance framework present
    - condition: compliance_framework != 'none'
      sample_rate: 1.0
      description: "Compliance-related"
```

### Strategy 3: Customer-Tier Sampling

Sample based on customer importance.

```yaml
- name: customer_tier_sampling
  type: sampling
  rules:
    # 100% for enterprise customers
    - condition: budget_tier == 'enterprise'
      sample_rate: 1.0
      description: "Enterprise customers"

    # 50% for growth tier
    - condition: budget_tier == 'growth'
      sample_rate: 0.5
      description: "Growth tier"

    # 10% for free tier
    - condition: budget_tier == 'free'
      sample_rate: 0.1
      description: "Free tier"

    # 100% for customers with violations
    - condition: customer_id != 'none' && is_violation == true
      sample_rate: 1.0
      description: "Customers with violations"
```

### Strategy 4: Environment-Based Sampling

Sample differently by environment.

```yaml
- name: environment_sampling
  type: sampling
  rules:
    # 100% for production
    - condition: environment == 'production'
      sample_rate: 1.0
      description: "Production environment"

    # 10% for staging
    - condition: environment == 'staging'
      sample_rate: 0.1
      description: "Staging environment"

    # 1% for development
    - condition: environment == 'development'
      sample_rate: 0.01
      description: "Development environment"
```

---

## Alerting & Monitoring

Configure alerting for governance events.

### Alert 1: Budget Threshold Alerts

**PagerDuty Integration:**

```yaml
- name: to_pagerduty_budget_critical
  destination: pagerduty_events
  filter: budget_status == 'exhausted' || budget_status == 'critical'

  payload_template: |
    {
      "routing_key": "${PAGERDUTY_ROUTING_KEY}",
      "event_action": "trigger",
      "dedup_key": "genops-budget-${budget_name}-${Date.now()}",
      "payload": {
        "summary": "GenOps Budget Alert: ${budget_name} at ${actual_utilization.toFixed(1)}%",
        "severity": "${budget_exhausted ? 'critical' : 'error'}",
        "source": "genops-cribl",
        "component": "budget-monitor",
        "group": "${team}",
        "class": "cost-governance",
        "custom_details": {
          "budget_name": "${budget_name}",
          "team": "${team}",
          "project": "${project}",
          "utilization": "${actual_utilization.toFixed(1)}%",
          "used": "$${budget_used.toFixed(2)}",
          "limit": "$${budget_limit.toFixed(2)}",
          "remaining": "$${budget_remaining.toFixed(2)}",
          "status": "${budget_status}"
        }
      }
    }
```

### Alert 2: Policy Violation Alerts

**Slack Integration:**

```yaml
- name: to_slack_policy_violations
  destination: slack_webhook
  filter: policy_result == 'blocked'

  payload_template: |
    {
      "channel": "#security-alerts",
      "username": "GenOps Policy Monitor",
      "icon_emoji": ":shield:",
      "attachments": [{
        "color": "danger",
        "title": ":rotating_light: Policy Violation Detected",
        "text": "*${policy_name}* was blocked",
        "fields": [
          {"title": "Policy", "value": "${policy_name}", "short": true},
          {"title": "Team", "value": "${team}", "short": true},
          {"title": "Reason", "value": "${policy_reason}", "short": false},
          {"title": "Customer", "value": "${customer_id}", "short": true},
          {"title": "Severity", "value": "${violation_severity.toUpperCase()}", "short": true},
          {"title": "Compliance", "value": "${compliance_framework || 'N/A'}", "short": true},
          {"title": "Environment", "value": "${environment}", "short": true}
        ],
        "footer": "GenOps AI + Cribl Stream",
        "ts": ${Math.floor(Date.now() / 1000)}
      }]
    }
```

### Alert 3: Cost Anomaly Detection

**Datadog Monitor:**

Create a Datadog monitor that triggers on GenOps cost anomalies.

**Monitor Configuration:**

```
Monitor Type: Anomaly Detection
Metric: sum:genops.cost.total{*} by {team,project}
Alert Threshold: 3 standard deviations above baseline
Warning Threshold: 2 standard deviations above baseline
Evaluation Window: 1 hour
Alert Message:
  GenOps Cost Anomaly Detected

  Team: {{team.name}}
  Project: {{project.name}}
  Current Cost: {{value}} USD/hour
  Expected Cost: {{threshold}} USD/hour
  Deviation: {{value - threshold}} USD/hour

  @slack-ai-cost-alerts @pagerduty-on-call
```

---

## Production Best Practices

### 1. High Availability Setup

**Cribl Stream Cluster:**

```yaml
# Worker Nodes (3+ for HA)
worker_nodes:
  - cribl-worker-01:
      role: worker
      cpu: 8 cores
      memory: 32 GB
      pipelines: [genops-cost, genops-policy]

  - cribl-worker-02:
      role: worker
      cpu: 8 cores
      memory: 32 GB
      pipelines: [genops-budget, genops-audit]

  - cribl-worker-03:
      role: worker
      cpu: 8 cores
      memory: 32 GB
      pipelines: [all]  # Backup for all pipelines

# Leader Node (1 primary + 1 standby)
leader_nodes:
  - cribl-leader-01:
      role: leader
      cpu: 4 cores
      memory: 16 GB

  - cribl-leader-02:
      role: leader-standby
      cpu: 4 cores
      memory: 16 GB

# Load Balancer
load_balancer:
  type: AWS ALB / NGINX
  health_check:
    path: /health
    interval: 30s
    timeout: 10s
    unhealthy_threshold: 3
  targets:
    - cribl-worker-01:4318
    - cribl-worker-02:4318
    - cribl-worker-03:4318
```

### 2. Disaster Recovery

**Backup Configuration:**

```yaml
backup_strategy:
  # Configuration Backup
  config_backup:
    frequency: daily
    retention: 30 days
    destination: s3://cribl-config-backup/
    includes:
      - pipelines
      - sources
      - destinations
      - lookups

  # State Backup
  state_backup:
    frequency: hourly
    retention: 7 days
    destination: s3://cribl-state-backup/
    includes:
      - persistent_queues
      - checkpoints

  # Cross-Region Replication
  replication:
    enabled: true
    regions:
      - us-west-2 (DR region)
    replication_lag: < 15 minutes
```

**Failover Procedure:**

1. **Automatic Failover** (< 5 minutes):
   - Load balancer detects worker failure
   - Routes traffic to healthy workers
   - GenOps retries failed OTLP exports

2. **Manual Failover to DR Region** (< 30 minutes):
   - Update DNS to point to DR Cribl cluster
   - Restore configuration from S3 backup
   - Verify pipelines processing correctly

### 3. Performance Tuning

**Worker Node Tuning:**

```yaml
performance_settings:
  # CPU & Memory
  worker_processes: auto  # 1 per CPU core
  max_memory_per_worker: 28 GB  # 80% of available RAM

  # Pipeline Performance
  pipeline:
    max_concurrent_events: 10000
    batch_size: 1000
    flush_interval: 10s

  # Buffer Settings
  buffer:
    type: memory + disk
    memory_size: 10 GB
    disk_size: 100 GB
    compression: true

  # Network
  network:
    max_connections: 1000
    keepalive: true
    tcp_nodelay: true
```

**Monitoring Performance:**

```yaml
monitoring:
  metrics_to_watch:
    - cpu_utilization (target: < 70%)
    - memory_utilization (target: < 80%)
    - buffer_utilization (target: < 90%)
    - pipeline_lag (target: < 10s)
    - error_rate (target: < 1%)
    - throughput (events/second)

  alerts:
    - name: High CPU Usage
      condition: cpu_utilization > 80%
      for: 10 minutes
      severity: warning

    - name: Buffer Full
      condition: buffer_utilization > 95%
      for: 5 minutes
      severity: critical
```

### 4. Security Considerations

**Authentication & Authorization:**

```yaml
security:
  # OTLP Source Authentication
  source_auth:
    type: bearer_token
    token_validation: strict
    token_rotation: 90 days

  # Destination Authentication
  destination_auth:
    secrets_manager: AWS Secrets Manager
    rotation: automatic

  # TLS/SSL
  tls:
    enabled: true
    min_version: TLS 1.2
    cipher_suites: strong_ciphers_only
    certificate: /etc/cribl/certs/server.crt
    private_key: /etc/cribl/certs/server.key

  # Network Security
  network:
    firewall_rules:
      - allow: GenOps application IPs
      - deny: all others
    rate_limiting:
      enabled: true
      max_requests: 10000/second/ip
```

**Data Protection:**

```yaml
data_protection:
  # Encryption at Rest
  encryption_at_rest:
    buffer_storage: AES-256
    persistent_queues: AES-256

  # Encryption in Transit
  encryption_in_transit:
    genops_to_cribl: TLS 1.2+
    cribl_to_destinations: TLS 1.2+

  # Data Masking
  masking:
    pii_patterns:
      - ssn: '\b\d{3}-\d{2}-\d{4}\b'
      - email: '\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
      - phone: '\b\d{3}-\d{3}-\d{4}\b'
    action: redact

  # Data Retention
  retention:
    buffer: 24 hours
    persistent_queue: 7 days
    audit_logs: 90 days
```

---

## Troubleshooting

### Issue 1: GenOps Telemetry Not Arriving in Cribl

**Symptoms:**
- No spans visible in Cribl Live Data
- GenOps logs show successful export

**Diagnosis:**

```bash
# 1. Test Cribl endpoint directly
curl -v http://cribl-stream:4318/v1/traces \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"resourceSpans": [{"scopeSpans": [{"spans": [{"name": "test"}]}]}]}'

# 2. Check GenOps OTLP exporter logs
export OTEL_LOG_LEVEL=debug
python your_app.py  # Look for OTLP export attempts

# 3. Check Cribl source status
# In Cribl UI: Sources → genops-otlp → Status
```

**Solutions:**

1. **Network connectivity**: Verify GenOps can reach Cribl endpoint
```bash
telnet cribl-stream 4318
```

2. **Authentication**: Verify bearer token is correct
```bash
# Test with curl (should return 200)
curl -H "Authorization: Bearer YOUR_TOKEN" http://cribl-stream:4318/health
```

3. **OTLP format**: Ensure GenOps is configured for OTLP export
```python
genops.init(exporter_type="otlp")  # Must be "otlp", not "console"
```

### Issue 2: High Pipeline Lag

**Symptoms:**
- Cribl UI shows pipeline lag > 60 seconds
- Buffers filling up

**Diagnosis:**

```bash
# Check pipeline metrics in Cribl UI
# Monitoring → Pipelines → genops-cost-governance

# Look for:
# - Events/second (input vs output)
# - Pipeline lag (seconds)
# - CPU usage per pipeline
```

**Solutions:**

1. **Increase worker resources**:
```yaml
# Add more CPU/memory to workers
# Or add more worker nodes
```

2. **Optimize pipeline processors**:
```yaml
# Remove expensive eval expressions
# Simplify lookup tables
# Reduce sampling rules
```

3. **Scale horizontally**:
```yaml
# Add more Cribl worker nodes
# Distribute pipelines across workers
```

### Issue 3: Destination Delivery Failures

**Symptoms:**
- Cribl shows high error rate for destination
- Telemetry not arriving in downstream platform

**Diagnosis:**

```bash
# Check destination status in Cribl UI
# Monitoring → Destinations → datadog_metrics_dest

# Look for:
# - Connection errors
# - Authentication failures
# - Rate limiting (429 errors)
```

**Solutions:**

1. **Authentication**: Verify destination credentials
```bash
# Test Datadog API key
curl -H "DD-API-KEY: YOUR_KEY" https://api.datadoghq.com/api/v1/validate

# Test Splunk HEC token
curl -H "Authorization: Splunk YOUR_TOKEN" https://splunk:8088/services/collector/health
```

2. **Rate limiting**: Increase batch size or add delays
```yaml
destination_config:
  batch_size: 500  # Increase to reduce request frequency
  flush_interval: 30s  # Increase interval
```

3. **Retry configuration**: Adjust retry policy
```yaml
retry:
  max_retries: 5
  retry_backoff: exponential
  max_retry_delay: 60s
```

---

## Next Steps

### 1. Start Small

Begin with a single pipeline (cost governance) and one destination (Datadog).

### 2. Iterate and Expand

Add more pipelines as you understand the patterns:
1. Cost governance → Works well
2. Add policy compliance routing
3. Add budget alerting
4. Add compliance audit trail

### 3. Optimize

Monitor Cribl metrics and optimize:
- Sampling rates (reduce telemetry volume)
- Pipeline processors (reduce CPU usage)
- Destination batching (reduce API calls)

### 4. Scale

As volume grows:
- Add more Cribl worker nodes
- Implement cross-region replication
- Add Cribl Lake for internal analytics

---

## Additional Resources

- **Cribl Documentation**: [docs.cribl.io](https://docs.cribl.io)
- **Cribl Community**: [community.cribl.io](https://community.cribl.io)
- **GenOps Quickstart**: [docs/cribl-quickstart.md](../cribl-quickstart.md)
- **Example Code**: [examples/observability/cribl_integration.py](../../examples/observability/cribl_integration.py)
- **Pipeline YAML Files**: [examples/cribl/pipelines/](../../examples/cribl/pipelines/)

---

## Support

**GenOps AI:**
- GitHub Issues: [github.com/KoshiHQ/GenOps-AI/issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- Discussions: [github.com/KoshiHQ/GenOps-AI/discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

**Cribl:**
- Support Portal: [cribl.io/support](https://cribl.io/support)
- Slack Community: [cribl-community.slack.com](https://cribl-community.slack.com)

---

**Congratulations!** You now have a complete governance-to-observability pipeline for AI systems using GenOps + Cribl.
