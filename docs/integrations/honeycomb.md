# Honeycomb Integration

**Export AI governance telemetry to Honeycomb for high-cardinality observability, real-time cost tracking, and interactive governance analysis.**

## Overview

The GenOps Honeycomb integration enables organizations to export AI governance telemetry — cost attribution, policy enforcement, budget tracking, and evaluation metrics — into Honeycomb's high-cardinality observability platform using OpenTelemetry OTLP export.

### Problems Solved

- **High-Cardinality AI Visibility:** Unlimited customer_id, user_id, transaction_id tracking without performance degradation
- **Real-Time Cost Attribution:** Track and analyze AI costs by any dimension with sub-second query performance
- **Interactive Root Cause Analysis:** Use BubbleUp to discover cost drivers and outliers automatically
- **Policy Compliance:** Monitor policy enforcement with fast, iterative investigation workflows
- **Budget Management:** Real-time budget tracking with instant alerting via Triggers
- **Performance Correlation:** Explore relationships between cost, latency, and usage patterns interactively

### Value Proposition

**For Platform Teams:**
- High-cardinality analysis enables per-customer, per-feature, per-transaction cost tracking
- Sub-second query performance for interactive debugging and investigation
- No cardinality limits — track as many dimensions as needed
- Native distributed tracing with OpenTelemetry compatibility
- Integration with existing Honeycomb observability workflows

**For FinOps Teams:**
- Unlimited dimension cost attribution (team, project, customer, feature, user, transaction)
- BubbleUp automatically surfaces cost anomalies and high-spend patterns
- Real-time cost visibility with no aggregation delays
- Multi-dimensional cost exploration without pre-defined dashboards
- Historical cost analysis with flexible time-based queries

**For Compliance Teams:**
- Complete audit trail for all AI operations
- Fast policy violation investigation with interactive queries
- Data classification tracking across unlimited dimensions
- Compliance dashboard templates with Honeycomb Boards

**For AI/ML Engineers:**
- Debug cost issues in real-time with fast iterative queries
- Correlate model performance with cost and usage patterns
- A/B test model changes with immediate cost feedback
- Track experiments and rollouts with deployment markers

---

## Core Concepts

### 1. OpenTelemetry OTLP Export

GenOps exports telemetry to Honeycomb using the **OpenTelemetry OTLP protocol**, ensuring vendor neutrality and interoperability.

**Architecture:**
```
GenOps AI Application
    ↓
OpenTelemetry SDK (traces, metrics, logs)
    ↓
OTLP Exporter (HTTP)
    ↓
Honeycomb OTLP Endpoint (api.honeycomb.io/v1/traces)
    ↓
Honeycomb Platform (Traces, BubbleUp, Triggers, SLOs)
```

**Benefits:**
- Standard protocol supported by 40+ observability platforms
- No Honeycomb-specific SDK required
- Easy migration between observability vendors
- Native OpenTelemetry ecosystem compatibility

### 2. High-Cardinality Analysis

**Traditional APM Challenge:**
Most observability platforms struggle with high-cardinality dimensions (customer_id, user_id, transaction_id), leading to:
- Sampling or aggregation that loses detail
- Performance degradation with many dimensions
- Pre-aggregated metrics that can't answer ad-hoc questions

**Honeycomb Solution:**
Honeycomb is architected specifically for high-cardinality analysis:
- **Unlimited dimensions:** Track customer_id, user_id, feature, transaction_id without limits
- **Fast queries:** Sub-second queries even with millions of unique values
- **Raw event retention:** No pre-aggregation — query raw events directly
- **Interactive exploration:** Answer questions you didn't anticipate

**Perfect for AI Governance:**
AI operations naturally have high cardinality:
- Per-customer cost tracking (SaaS billing)
- Per-user attribution (user-facing features)
- Per-transaction tracking (request-level costs)
- Per-feature costs (A/B tests, experiments)
- Per-model efficiency (multi-model deployments)

GenOps + Honeycomb = unlimited governance dimensions without performance trade-offs.

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
- `genops.customer_id`: Customer attribution (high-cardinality in Honeycomb!)
- `genops.user_id`: User-level tracking (perfect for Honeycomb)
- `genops.transaction_id`: Transaction tracking (unlimited cardinality)
- `genops.cost_center`: Financial reporting
- `genops.feature`: Feature tracking

**Cost Fields:**
- `genops.cost.total`: Total cost in USD
- `genops.cost.input`: Input token cost
- `genops.cost.output`: Output token cost
- `genops.cost.provider`: AI provider (openai, anthropic, bedrock, gemini)
- `genops.cost.model`: Model name (gpt-4, claude-3-sonnet)
- `genops.tokens.input`: Input tokens
- `genops.tokens.output`: Output tokens
- `genops.tokens.total`: Total tokens

**Policy Fields:**
- `genops.policy.name`: Policy identifier
- `genops.policy.result`: Result (allowed, blocked, warning)
- `genops.policy.reason`: Decision reason
- `genops.policy.response_time`: Policy evaluation duration (ms)

**Budget Fields:**
- `genops.budget.id`: Budget identifier
- `genops.budget.limit`: Budget limit (USD)
- `genops.budget.consumed`: Amount consumed
- `genops.budget.remaining`: Amount remaining
- `genops.budget.utilization_pct`: Utilization percentage (derived column candidate)

**Evaluation Fields:**
- `genops.eval.safety`: Safety score (0-1)
- `genops.eval.accuracy`: Accuracy score (0-1)
- `genops.eval.compliance`: Compliance score (0-1)
- `genops.eval.privacy`: Privacy score (0-1)
- `genops.eval.performed`: Boolean flag for evaluation

**Performance Fields:**
- `duration_ms`: Operation duration in milliseconds
- `status`: Operation status (success, error, timeout)

### 4. Authentication

Honeycomb authentication uses **API Keys** (called "Team API Keys") passed via HTTP headers:

```bash
# Required environment variable
export HONEYCOMB_API_KEY="your_honeycomb_api_key"

# Optional: Dataset name (defaults to "genops-ai")
export HONEYCOMB_DATASET="genops-ai"
```

**Generate API Key:**
1. Log in to [Honeycomb](https://ui.honeycomb.io)
2. Navigate to **Team Settings → API Keys**
3. Click **Create API Key**
4. Name: `genops-ai-production`
5. Permissions: Select **Send Events** (write access)
6. Copy the generated key
7. Set environment variable: `export HONEYCOMB_API_KEY="..."`

**Security Best Practices:**
- Store API keys in secret management systems (AWS Secrets Manager, HashiCorp Vault, etc.)
- Use different keys for production and non-production environments
- Rotate keys periodically (Honeycomb supports key rotation without downtime)
- Grant minimum required permissions (Send Events only for data export)
- Never commit keys to version control
- Use environment-specific datasets to separate dev/staging/prod data

### 5. Datasets and Environments

**Datasets** in Honeycomb are independent collections of telemetry data.

**Recommended Strategy:**
```bash
# Option 1: Single dataset with environment tags
HONEYCOMB_DATASET="genops-ai"
# Use genops.environment attribute to filter: dev, staging, production

# Option 2: Separate datasets per environment
HONEYCOMB_DATASET="genops-ai-production"  # Production
HONEYCOMB_DATASET="genops-ai-staging"     # Staging
HONEYCOMB_DATASET="genops-ai-dev"         # Development
```

**Best Practices:**
- **Small teams (<50 people):** Single dataset with environment tags (simpler)
- **Large teams (50+ people):** Separate datasets per environment (better access control)
- **Multi-tenant SaaS:** Single dataset with high-cardinality customer_id tracking
- **Compliance requirements:** Separate datasets for PII vs non-PII data

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
- **OTLP Exporter:** HTTP (required for Honeycomb)

### Honeycomb Requirements

- **Honeycomb Account:** Free or paid tier (Free tier: 20M events/month)
- **API Key:** Generate from Team Settings → API Keys
- **Permissions:** API key needs **Send Events** permission
- **Dataset:** Create dataset in Honeycomb UI (or will be auto-created on first export)

**Free Tier Limits:**
- 20 million events per month
- 60-day retention
- Unlimited columns (fields)
- Unlimited queries
- Up to 10 team members

### Verify Installation

```bash
# Check GenOps installation
python -c "import genops; print(genops.__version__)"

# Check OpenTelemetry installation
python -c "import opentelemetry; print('OpenTelemetry OK')"

# Check environment variables
echo $HONEYCOMB_API_KEY | wc -c  # Should have value
echo $HONEYCOMB_DATASET          # Should output dataset name or be empty (defaults to "genops-ai")
```

---

## Integration Patterns

### Pattern 1: Direct OTLP Export (Simplest)

**Best for:** Small applications, local development, quick prototyping

**Setup:**

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument
import os

# Configure Honeycomb OTLP endpoint
configure_otlp_exporter(
    endpoint="https://api.honeycomb.io/v1/traces",
    headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}
)

# Enable auto-instrumentation
auto_instrument()

# Your existing AI code works unchanged
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
# ✅ Telemetry automatically exported to Honeycomb
```

**Characteristics:**
- Zero-code changes to existing application logic
- Direct export from application to Honeycomb
- Minimal setup complexity
- Telemetry blocking (application waits for export to complete)

### Pattern 2: OpenTelemetry Collector (Recommended for Production)

**Best for:** Production deployments, Kubernetes, high-volume applications

**Architecture:**
```
GenOps Application → OTel Collector (sidecar/daemonset) → Honeycomb
```

**Benefits:**
- Decouples application from telemetry backend
- Enables sampling, filtering, and batching
- Supports multi-backend export (Honeycomb + Datadog simultaneously)
- Reduces application overhead (async export)
- Centralized telemetry routing and transformation

**Setup:**

**1. Application Configuration:**

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument

# Export to local OTel Collector
configure_otlp_exporter(
    endpoint="http://localhost:4318/v1/traces",  # OTel Collector HTTP endpoint
    headers={}  # No authentication needed for local collector
)

auto_instrument()
```

**2. OpenTelemetry Collector Configuration:**

Create `otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  # Batch events for efficiency
  batch:
    timeout: 10s
    send_batch_size: 100

  # Add resource attributes
  resource:
    attributes:
      - key: deployment.environment
        value: ${ENVIRONMENT}
        action: upsert

  # Sampling (optional, for high-volume)
  probabilistic_sampler:
    sampling_percentage: 100  # Set to lower value if needed (e.g., 10 for 10%)

exporters:
  otlp/honeycomb:
    endpoint: "api.honeycomb.io:443"
    headers:
      "x-honeycomb-team": "${HONEYCOMB_API_KEY}"
      "x-honeycomb-dataset": "${HONEYCOMB_DATASET}"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [otlp/honeycomb]
```

**3. Run OpenTelemetry Collector:**

```bash
# Download collector
wget https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.91.0/otelcol_0.91.0_linux_amd64.tar.gz
tar -xvf otelcol_0.91.0_linux_amd64.tar.gz

# Run with config
export HONEYCOMB_API_KEY="your_api_key"
export HONEYCOMB_DATASET="genops-ai"
export ENVIRONMENT="production"

./otelcol --config otel-collector-config.yaml
```

### Pattern 3: Kubernetes Deployment

**Best for:** Kubernetes-based applications

**Deployment Options:**
1. **DaemonSet:** One collector per node (recommended)
2. **Sidecar:** One collector per pod (simple, but higher resource usage)
3. **Deployment:** Centralized collector service (single point of failure)

**DaemonSet Example:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
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
        send_batch_size: 100

      # Add Kubernetes metadata
      k8sattributes:
        passthrough: false
        auth_type: "serviceAccount"
        extract:
          metadata:
            - k8s.namespace.name
            - k8s.deployment.name
            - k8s.pod.name
            - k8s.pod.uid
            - k8s.node.name

      # Resource detection
      resourcedetection:
        detectors: [env, system]

    exporters:
      otlp/honeycomb:
        endpoint: "api.honeycomb.io:443"
        headers:
          "x-honeycomb-team": "${HONEYCOMB_API_KEY}"
          "x-honeycomb-dataset": "genops-ai-production"

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [k8sattributes, resourcedetection, batch]
          exporters: [otlp/honeycomb]

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      serviceAccountName: otel-collector
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector:0.91.0
        command: ["--config=/conf/config.yaml"]
        env:
        - name: HONEYCOMB_API_KEY
          valueFrom:
            secretKeyRef:
              name: honeycomb-credentials
              key: api-key
        volumeMounts:
        - name: config
          mountPath: /conf
        ports:
        - containerPort: 4318  # HTTP
          name: otlp-http
        - containerPort: 4317  # gRPC
          name: otlp-grpc
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
      volumes:
      - name: config
        configMap:
          name: otel-collector-config

---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: observability
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  type: ClusterIP

---
apiVersion: v1
kind: Secret
metadata:
  name: honeycomb-credentials
  namespace: observability
type: Opaque
stringData:
  api-key: "your_honeycomb_api_key"

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: otel-collector
  namespace: observability

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: otel-collector
rules:
- apiGroups: [""]
  resources: ["pods", "namespaces", "nodes"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: otel-collector
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: otel-collector
subjects:
- kind: ServiceAccount
  name: otel-collector
  namespace: observability
```

**Application Configuration in Kubernetes:**

```python
# Use Kubernetes service DNS for collector endpoint
from genops.exporters.otlp import configure_otlp_exporter
import os

configure_otlp_exporter(
    endpoint="http://otel-collector.observability.svc.cluster.local:4318/v1/traces",
    headers={}
)
```

---

## Governance Context and Attribution

### Setting Governance Context

**Global Context (applies to all operations):**

```python
from genops.core.context import set_governance_context

set_governance_context({
    "team": "ai-engineering",
    "project": "customer-chatbot",
    "environment": "production",
    "cost_center": "engineering-ai"
})
```

**Per-Request Context (high-cardinality tracking):**

```python
from genops.core.context import set_governance_context, clear_governance_context

# For each customer request
def handle_request(request):
    # Set request-specific context
    set_governance_context({
        "customer_id": request.customer_id,
        "user_id": request.user_id,
        "transaction_id": request.transaction_id,
        "feature": request.feature_name,
        "customer_tier": request.customer.tier
    })

    # Process AI operation
    response = ai_service.generate(request.prompt)

    # Clear context after request
    clear_governance_context()

    return response
```

**Context Managers (scoped tracking):**

```python
from genops.core.context import governance_context

def process_customer_workflow(customer_id, feature):
    with governance_context(customer_id=customer_id, feature=feature):
        # All operations within this block include these attributes
        result1 = ai_service.step1()
        result2 = ai_service.step2()
        result3 = ai_service.step3()
        # Context automatically cleared on exit

    return result3
```

### High-Cardinality Attribution Examples

**SaaS Platform (per-customer billing):**

```python
set_governance_context({
    "customer_id": "acme-corp",           # Unlimited unique customers
    "customer_tier": "enterprise",        # enterprise, business, starter
    "customer_region": "us-west-2",       # Geographic tracking
    "tenant_id": "tenant-12345",          # Multi-tenant tracking
    "subscription_id": "sub-xyz-789"      # Subscription tracking
})
```

**User-Facing Feature (per-user tracking):**

```python
set_governance_context({
    "user_id": "user-98765",              # Unlimited unique users
    "user_segment": "power-user",         # User segmentation
    "feature": "document-analysis",       # Feature tracking
    "experiment_id": "exp-v2-test",       # A/B test tracking
    "variant": "treatment"                # Experiment variant
})
```

**E-Commerce (transaction-level tracking):**

```python
set_governance_context({
    "transaction_id": "order-abc-123",    # Per-transaction costs
    "cart_id": "cart-xyz-456",            # Shopping cart tracking
    "product_category": "electronics",    # Product categorization
    "recommendation_model": "v2.3",       # Model version tracking
    "session_id": "sess-789-def"          # Session tracking
})
```

**Financial Services (compliance tracking):**

```python
set_governance_context({
    "account_id": "acct-123456",          # Account-level tracking
    "transaction_type": "fraud-detection", # Use case tracking
    "data_classification": "pii",         # Data sensitivity
    "compliance_framework": "sox",        # Compliance context
    "risk_level": "high"                  # Risk classification
})
```

---

## Honeycomb Query Examples

### Cost Analysis Queries

**Total Cost by Provider:**
```
GROUP BY genops.cost.provider
| SUM(genops.cost.total)
| ORDER BY SUM DESC
```

**Cost by Customer (Top 20):**
```
GROUP BY genops.customer_id
| SUM(genops.cost.total)
| ORDER BY SUM DESC
| LIMIT 20
```

**Daily Cost Trend:**
```
GROUP BY DATE_TRUNC("day", timestamp)
| SUM(genops.cost.total)
```

**Cost by Model and Provider:**
```
GROUP BY genops.cost.provider, genops.cost.model
| SUM(genops.cost.total), COUNT
| ORDER BY SUM DESC
```

**Cost per Request (Average):**
```
AVG(genops.cost.total)
```

**Multi-Dimensional Cost Breakdown:**
```
GROUP BY genops.team, genops.project, genops.environment
| SUM(genops.cost.total)
| ORDER BY SUM DESC
```

### Performance Analysis Queries

**Latency Percentiles by Model:**
```
GROUP BY genops.cost.model
| P50(duration_ms), P95(duration_ms), P99(duration_ms)
```

**Slow Operations (>2 seconds):**
```
WHERE duration_ms > 2000
| COUNT
| GROUP BY genops.team, genops.feature
```

**Correlation: Latency vs Token Count:**
```
HEATMAP(duration_ms, genops.tokens.total)
```

**Token Throughput (tokens per second):**
```
GROUP BY genops.cost.model
| AVG(genops.tokens.total / (duration_ms / 1000))
```

### Attribution Analysis Queries

**Cost by Customer Tier:**
```
GROUP BY genops.customer_tier
| SUM(genops.cost.total), COUNT, AVG(genops.cost.total)
```

**Feature Usage and Cost:**
```
GROUP BY genops.feature
| COUNT, SUM(genops.cost.total), AVG(genops.cost.total)
| ORDER BY COUNT DESC
```

**User Segmentation:**
```
GROUP BY genops.user_segment
| SUM(genops.cost.total), COUNT
```

**Experiment Analysis (A/B Testing):**
```
WHERE genops.experiment_id = "exp-v2-test"
| GROUP BY genops.variant
| AVG(genops.cost.total), AVG(duration_ms), COUNT
```

### Policy and Compliance Queries

**Policy Violations:**
```
WHERE genops.policy.result = "blocked"
| COUNT
| GROUP BY genops.policy.name
```

**Policy Evaluation Performance:**
```
GROUP BY genops.policy.name
| AVG(genops.policy.response_time), P95(genops.policy.response_time)
```

**Data Classification Tracking:**
```
GROUP BY genops.data.classification
| COUNT, SUM(genops.cost.total)
```

**Compliance Score Distribution:**
```
WHERE genops.eval.performed = true
| HEATMAP(genops.eval.compliance, genops.eval.safety)
```

### Budget Tracking Queries

**Budget Utilization:**
```
WHERE genops.budget.id EXISTS
| AVG(genops.budget.consumed / genops.budget.limit * 100)
| GROUP BY genops.budget.id
```

**Budget Overruns:**
```
WHERE genops.budget.consumed > genops.budget.limit
| COUNT
| GROUP BY genops.team
```

**Remaining Budget:**
```
GROUP BY genops.budget.id
| MAX(genops.budget.remaining)
```

---

## BubbleUp for Root Cause Analysis

### What is BubbleUp?

**BubbleUp** is Honeycomb's signature feature for automatically discovering which attributes distinguish interesting events from normal events.

**Use Cases for AI Governance:**
- Find which customers are driving high costs
- Identify which models are underperforming
- Discover which features have the highest cost variance
- Surface policy violations by specific attributes

### Using BubbleUp for Cost Analysis

**Scenario:** Sudden cost spike detected

**Steps:**
1. Navigate to Honeycomb query interface
2. Click **BubbleUp**
3. Select metric: `SUM(genops.cost.total)`
4. Filter to time range: Last 4 hours
5. Click **Run BubbleUp**

**What BubbleUp Shows:**
- Attributes that distinguish high-cost operations from normal operations
- Automatically ranked by statistical significance

**Example Results:**
```
Top distinguishing attributes for high cost:
1. genops.customer_id = "enterprise-acme" (45% of high-cost events, 2% of normal events)
2. genops.cost.model = "gpt-4" (78% vs 12%)
3. genops.feature = "document-summarization" (67% vs 8%)
```

**Action:** Investigate why customer "enterprise-acme" is driving costs via "document-summarization" feature with GPT-4.

### BubbleUp Query Examples

**Find cost outliers:**
```
1. Create query: SUM(genops.cost.total) WHERE timestamp > ago(1h)
2. Click BubbleUp
3. Review top distinguishing attributes
```

**Find slow operations:**
```
1. Create query: P95(duration_ms) WHERE timestamp > ago(2h)
2. Click BubbleUp
3. Identify attributes correlated with slow performance
```

**Find policy violations:**
```
1. Create query: COUNT WHERE genops.policy.result = "blocked"
2. Click BubbleUp
3. Surface which teams/features have violations
```

---

## Derived Columns

### What are Derived Columns?

**Derived Columns** are computed fields created from existing telemetry fields, calculated at query time.

**Benefits:**
- Reduce cardinality by grouping values
- Create governance metrics from raw fields
- Simplify common calculations
- No application code changes needed

### Common Governance Derived Columns

**1. Cost per Token:**

```
Create Derived Column:
  Name: cost_per_token
  Type: Decimal
  Formula: genops.cost.total / genops.tokens.total
```

**Usage:**
```
GROUP BY genops.cost.model
| AVG($cost_per_token)
| ORDER BY AVG DESC
```

**2. Budget Utilization Percentage:**

```
Create Derived Column:
  Name: budget_utilization_pct
  Type: Decimal
  Formula: (genops.budget.consumed / genops.budget.limit) * 100
```

**Usage:**
```
WHERE $budget_utilization_pct > 90
| COUNT
| GROUP BY genops.team
```

**3. Token Efficiency (tokens per second):**

```
Create Derived Column:
  Name: token_throughput
  Type: Decimal
  Formula: genops.tokens.total / (duration_ms / 1000)
```

**Usage:**
```
GROUP BY genops.cost.provider
| AVG($token_throughput)
```

**4. Customer Tier Grouping:**

```
Create Derived Column:
  Name: customer_tier_group
  Type: String
  Formula:
    CASE
      WHEN genops.customer_tier IN ["enterprise", "business"] THEN "paid"
      WHEN genops.customer_tier = "free" THEN "free"
      ELSE "unknown"
    END
```

**Usage:**
```
GROUP BY $customer_tier_group
| SUM(genops.cost.total)
```

**5. Cost Bucket (categorize spend levels):**

```
Create Derived Column:
  Name: cost_bucket
  Type: String
  Formula:
    CASE
      WHEN genops.cost.total < 0.01 THEN "micro"
      WHEN genops.cost.total < 0.10 THEN "small"
      WHEN genops.cost.total < 1.00 THEN "medium"
      ELSE "large"
    END
```

**Usage:**
```
GROUP BY $cost_bucket
| COUNT
```

### Creating Derived Columns in Honeycomb

1. Navigate to **Team Settings → Derived Columns**
2. Click **Create Derived Column**
3. Enter:
   - **Name:** `cost_per_token`
   - **Type:** `Decimal`
   - **Formula:** `genops.cost.total / genops.tokens.total`
4. Click **Save**
5. Use in queries with `$cost_per_token` prefix

---

## Triggers for Budget Alerts

### What are Triggers?

**Triggers** are Honeycomb's alerting system that evaluates queries periodically and sends notifications when conditions are met.

**Use Cases for AI Governance:**
- Alert when budget thresholds are crossed
- Notify on cost spikes or anomalies
- Warn about policy violation increases
- Monitor token usage trends

### Budget Alert Examples

**Alert 1: Daily Budget Threshold (90%):**

**Query:**
```
WHERE genops.budget.id = "team-ai-engineering-daily"
| MAX(genops.budget.consumed / genops.budget.limit * 100)
```

**Trigger Configuration:**
- **Name:** "AI Engineering Daily Budget 90%"
- **Frequency:** Every 5 minutes
- **Condition:** `MAX >= 90`
- **Notification:** Slack #ai-budget-alerts

**Alert 2: Sudden Cost Spike:**

**Query:**
```
GROUP BY DATE_TRUNC("hour", timestamp)
| SUM(genops.cost.total)
```

**Trigger Configuration:**
- **Name:** "Hourly Cost Spike"
- **Frequency:** Every 10 minutes
- **Condition:** `SUM > 100` (threshold: $100/hour)
- **Notification:** PagerDuty + Slack

**Alert 3: High Policy Violation Rate:**

**Query:**
```
WHERE timestamp > ago(1h)
| COUNT WHERE genops.policy.result = "blocked"
```

**Trigger Configuration:**
- **Name:** "Policy Violations Elevated"
- **Frequency:** Every 15 minutes
- **Condition:** `COUNT > 50`
- **Notification:** Email to compliance team

**Alert 4: Customer Cost Anomaly:**

**Query:**
```
WHERE genops.customer_id = "enterprise-acme"
| SUM(genops.cost.total) WHERE timestamp > ago(1h)
```

**Trigger Configuration:**
- **Name:** "Acme Corp Hourly Cost Threshold"
- **Frequency:** Every 5 minutes
- **Condition:** `SUM > 50` ($50/hour)
- **Notification:** Account manager email

### Creating Triggers in Honeycomb

1. Create and save your query (e.g., budget utilization query)
2. Click **Create Trigger** from query interface
3. Configure:
   - **Name:** Descriptive alert name
   - **Frequency:** How often to evaluate (1min - 1day)
   - **Condition:** Threshold for alerting (>, <, >=, <=, ==)
   - **Notification Channel:** Slack, PagerDuty, Email, Webhook
4. Click **Save Trigger**
5. Test with **Send Test Alert**

---

## SLOs (Service Level Objectives)

### What are SLOs in Honeycomb?

**SLOs** track the percentage of good events (those meeting a defined quality threshold) over time.

**Use Cases for AI Governance:**
- Policy compliance rate (target: 99.9% of operations allowed)
- Budget adherence rate (target: 95% of operations within budget)
- Performance SLO (target: 95% of operations complete in <2 seconds)
- Cost efficiency SLO (target: 90% of operations under $0.10)

### Governance SLO Examples

**SLO 1: Policy Compliance Rate**

**Target:** 99.9% of operations are policy-compliant (not blocked)

**SLI Query:**
```
WHERE genops.policy.result IN ["allowed", "warning"]
```

**Total Events Query:**
```
COUNT
```

**SLO Configuration:**
- **Name:** "Policy Compliance SLO"
- **Target:** 99.9%
- **Time Window:** 30 days
- **Budget:** 0.1% error budget (allows 0.1% blocked operations)

**SLO 2: Budget Adherence Rate**

**Target:** 95% of operations stay within allocated budget

**SLI Query:**
```
WHERE genops.budget.consumed <= genops.budget.limit
```

**SLO Configuration:**
- **Name:** "Budget Adherence SLO"
- **Target:** 95%
- **Time Window:** 7 days

**SLO 3: Cost Efficiency SLO**

**Target:** 90% of operations cost less than $0.10

**SLI Query:**
```
WHERE genops.cost.total < 0.10
```

**SLO Configuration:**
- **Name:** "Cost Efficiency SLO"
- **Target:** 90%
- **Time Window:** 30 days

**SLO 4: Performance SLO**

**Target:** 95% of operations complete in <2 seconds

**SLI Query:**
```
WHERE duration_ms < 2000
```

**SLO Configuration:**
- **Name:** "AI Operation Performance SLO"
- **Target:** 95%
- **Time Window:** 7 days

### Creating SLOs in Honeycomb

1. Navigate to **SLOs → Create SLO**
2. Configure:
   - **Name:** Descriptive SLO name
   - **SLI Query:** Query defining "good" events
   - **Total Events Query:** Total events to measure against
   - **Target Percentage:** e.g., 99.9%
   - **Time Window:** 7 days, 30 days, etc.
3. Click **Create SLO**
4. View SLO dashboard for:
   - Current compliance percentage
   - Error budget remaining
   - Burn rate trends

---

## Markers for Deployments

### What are Markers?

**Markers** are annotations on your telemetry timeline that mark significant events (deployments, config changes, incidents).

**Use Cases for AI Governance:**
- Track cost changes after model deployments
- Correlate policy changes with violation rates
- Monitor performance impact of configuration updates
- Identify when budget enforcement rules were updated

### Creating Markers

**API Method (recommended for CI/CD):**

```bash
curl -X POST https://api.honeycomb.io/1/markers/${HONEYCOMB_DATASET} \
  -H "X-Honeycomb-Team: ${HONEYCOMB_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Deployed GPT-4 Turbo for premium customers",
    "type": "deploy",
    "url": "https://github.com/acme/ai-platform/releases/tag/v2.3.0"
  }'
```

**Python Example (from application):**

```python
import requests
import os

def create_deployment_marker(message, deployment_url=None):
    """Create a Honeycomb marker for deployments."""
    dataset = os.getenv("HONEYCOMB_DATASET", "genops-ai")
    api_key = os.getenv("HONEYCOMB_API_KEY")

    url = f"https://api.honeycomb.io/1/markers/{dataset}"
    headers = {
        "X-Honeycomb-Team": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "message": message,
        "type": "deploy"
    }

    if deployment_url:
        payload["url"] = deployment_url

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    print(f"✅ Marker created: {message}")

# Usage in deployment script
create_deployment_marker(
    message="Switched to Claude 3.5 Sonnet for analysis feature",
    deployment_url="https://github.com/acme/ai-platform/pull/456"
)
```

**Viewing Markers:**
- Markers appear as vertical lines on Honeycomb query timelines
- Click marker to see details (message, type, URL)
- Filter queries to time ranges around markers

---

## Production Deployment Best Practices

### Sampling Strategies

**When to Sample:**
- High-volume applications (>100K requests/day)
- Approaching Honeycomb event limits on free tier
- Cost optimization for paid tiers

**Sampling Approaches:**

**1. Head-Based Sampling (OpenTelemetry SDK):**

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.trace import TracerProvider

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)
provider = TracerProvider(sampler=sampler)
```

**2. Tail-Based Sampling (OTel Collector):**

```yaml
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 100
    policies:
      # Always sample errors
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Always sample high-cost operations
      - name: high-cost
        type: numeric_attribute
        numeric_attribute:
          key: genops.cost.total
          min_value: 1.0

      # Sample 10% of normal operations
      - name: probabilistic
        type: probabilistic
        probabilistic:
          sampling_percentage: 10
```

**3. Dynamic Sampling (per-customer):**

```python
from genops.core.sampling import set_sampling_rate

# High-value customers: 100% sampling
set_sampling_rate(customer_tier="enterprise", rate=1.0)

# Standard customers: 10% sampling
set_sampling_rate(customer_tier="business", rate=0.1)

# Free tier: 1% sampling
set_sampling_rate(customer_tier="free", rate=0.01)
```

### Multi-Environment Setup

**Strategy 1: Single Dataset with Environment Tags**

```python
# Application configuration
from genops.core.context import set_governance_context
import os

set_governance_context({
    "environment": os.getenv("ENVIRONMENT"),  # dev, staging, production
    "deployment_region": os.getenv("AWS_REGION"),
    "service_version": os.getenv("SERVICE_VERSION")
})
```

**Honeycomb Queries:**
```
# Production only
WHERE genops.environment = "production"

# Staging cost comparison
WHERE genops.environment = "staging"
| SUM(genops.cost.total)
```

**Strategy 2: Separate Datasets per Environment**

```bash
# Production
export HONEYCOMB_DATASET="genops-ai-production"

# Staging
export HONEYCOMB_DATASET="genops-ai-staging"

# Development
export HONEYCOMB_DATASET="genops-ai-dev"
```

**Access Control:**
- Create environment-specific API keys
- Grant production dataset access to ops team only
- Allow broader access to staging/dev datasets

### Cost Optimization

**1. Reduce Event Volume:**

```python
# Only track expensive operations
from genops.core.tracking import set_cost_threshold

set_cost_threshold(min_cost_usd=0.01)  # Ignore operations <$0.01
```

**2. Use Derived Columns to Reduce Cardinality:**

Create derived column `customer_tier_bucket` to group customers instead of tracking every `customer_id`:

```
Create Derived Column: customer_tier_bucket
Formula:
  CASE
    WHEN genops.customer_tier = "enterprise" THEN "paid-enterprise"
    WHEN genops.customer_tier = "business" THEN "paid-standard"
    ELSE "free"
  END
```

**3. Sampling Configuration:**

```yaml
# OTel Collector sampling for cost optimization
processors:
  probabilistic_sampler:
    sampling_percentage: 20  # 20% sampling = 80% cost reduction
```

**4. Retention Management:**

- Free tier: 60-day retention (automatic)
- Paid tier: Configure retention in Team Settings
- Archive historical data for compliance if needed

### Security and Compliance

**1. Secret Management:**

```yaml
# Kubernetes Secret
apiVersion: v1
kind: Secret
metadata:
  name: honeycomb-credentials
type: Opaque
stringData:
  api-key: "your_honeycomb_api_key"
  dataset: "genops-ai-production"
```

**2. PII Handling:**

**Option A: Separate datasets for PII vs non-PII:**
```bash
# PII-free dataset
export HONEYCOMB_DATASET="genops-ai-non-pii"

# PII dataset (restricted access)
export HONEYCOMB_DATASET="genops-ai-pii"
```

**Option B: OTel Collector filtering:**
```yaml
processors:
  attributes:
    actions:
      # Remove PII fields
      - key: genops.user.email
        action: delete
      - key: genops.user.phone
        action: delete
```

**3. Access Control:**

- Use Honeycomb Teams for role-based access control (RBAC)
- Create separate API keys per environment
- Audit API key usage periodically
- Rotate keys every 90 days

**4. Compliance:**

- **GDPR:** Use separate datasets or field filtering for EU data
- **HIPAA:** Enable Honeycomb's HIPAA-compliant plan if handling PHI
- **SOC 2:** Honeycomb is SOC 2 Type II certified
- **Audit Logs:** Export Honeycomb audit logs for compliance reporting

---

## Troubleshooting

### Issue: No Data Appearing in Honeycomb

**Symptoms:**
- Honeycomb dataset is empty
- No traces or events visible

**Diagnosis:**

1. **Check API Key:**
```bash
echo $HONEYCOMB_API_KEY
# Should output your API key (non-empty)
```

2. **Verify Dataset Exists:**
- Log in to Honeycomb UI
- Check if dataset exists in dropdown
- Create dataset if missing (will auto-create on first event)

3. **Run Validation:**
```python
from genops.exporters.validation import validate_export_setup, print_validation_result

result = validate_export_setup(provider="honeycomb")
print_validation_result(result)
```

4. **Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Re-run your application and check logs for export errors
```

**Common Causes:**
- Incorrect API key (typo, extra spaces)
- Wrong dataset name
- Network connectivity issues to `api.honeycomb.io`
- Application not instrumented correctly

**Solutions:**
- Regenerate API key and update `HONEYCOMB_API_KEY`
- Verify dataset name matches configuration
- Test network connectivity: `curl https://api.honeycomb.io`
- Re-run auto-instrumentation: `auto_instrument()`

### Issue: Authentication Failed (401 Unauthorized)

**Error Message:**
```
Failed to export to Honeycomb: 401 Unauthorized
```

**Diagnosis:**

1. **Check API Key Validity:**
- Log in to Honeycomb → Team Settings → API Keys
- Verify key exists and is active
- Regenerate key if needed

2. **Verify Header Format:**
```python
# Correct format
headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}

# Common mistake (wrong header name)
headers={"Authorization": f"Bearer {os.getenv('HONEYCOMB_API_KEY')}"}  # ❌ Wrong
```

3. **Check for Extra Characters:**
```bash
# API key should be exactly one line, no spaces
echo -n $HONEYCOMB_API_KEY | wc -c
```

**Solutions:**
- Use correct header: `X-Honeycomb-Team`
- Trim whitespace from API key
- Regenerate API key if corrupted

### Issue: High Cardinality Warning

**Honeycomb Message:**
```
Warning: High cardinality detected on field 'genops.customer_id'
```

**This is Expected and Encouraged!**

Honeycomb is designed for high-cardinality analysis. This warning is informational only.

**When to Take Action:**
- If query performance degrades (>10 seconds)
- If approaching Honeycomb event limits on free tier

**Mitigation Options:**

**Option 1: Use Derived Columns to Group**
```
# Instead of querying raw customer_id, group by tier
CREATE DERIVED COLUMN customer_tier_group AS
  CASE
    WHEN genops.customer_tier = "enterprise" THEN "paid-enterprise"
    WHEN genops.customer_tier = "business" THEN "paid-standard"
    ELSE "free"
  END
```

**Option 2: Enable Sampling**
```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

sampler = TraceIdRatioBased(0.1)  # Sample 10%
```

**Option 3: Add Time Bounds to Queries**
```
WHERE timestamp > ago(1h)  # Always filter by time
| GROUP BY genops.customer_id
```

### Issue: Slow Query Performance

**Symptoms:**
- Queries taking >10 seconds
- Timeouts on complex aggregations

**Diagnosis:**

1. **Check Query Complexity:**
- Are you grouping by >3 high-cardinality fields?
- Are you querying >7 days without time filters?
- Are you using complex CASE statements?

2. **Check Event Volume:**
- How many events are in the query time range?
- Navigate to dataset → View event rate

**Solutions:**

**1. Add Time Bounds:**
```
# Before (slow)
GROUP BY genops.customer_id | SUM(genops.cost.total)

# After (fast)
WHERE timestamp > ago(1h)
| GROUP BY genops.customer_id
| SUM(genops.cost.total)
```

**2. Reduce Grouping Dimensions:**
```
# Before (slow)
GROUP BY genops.team, genops.project, genops.customer_id, genops.feature
| SUM(genops.cost.total)

# After (fast)
GROUP BY genops.team
| SUM(genops.cost.total)
```

**3. Use Derived Columns:**
```
# Create derived column to pre-compute groupings
# Then query derived column instead of raw fields
```

**4. Use SLOs for Repeated Queries:**
- Instead of running the same query repeatedly, create an SLO
- SLOs are pre-computed and faster

### Issue: Missing Fields in Queries

**Symptoms:**
- Expected fields (e.g., `genops.customer_id`) don't appear in query builder
- Field autocomplete doesn't show governance attributes

**Diagnosis:**

1. **Check if Events Have Been Sent:**
- Fields only appear after first event with that field is received
- Send test event with all governance attributes

2. **Verify Instrumentation:**
```python
from genops import auto_instrument
from genops.core.context import set_governance_context

auto_instrument()

# Set full governance context
set_governance_context({
    "team": "test",
    "project": "test",
    "customer_id": "test-customer"
})

# Run test operation
```

3. **Check Dataset:**
- Are you viewing the correct dataset?
- Switch dataset in Honeycomb UI dropdown

**Solutions:**
- Send events with all expected governance fields
- Wait 1-2 minutes for field schema to refresh
- Refresh Honeycomb UI browser page

### Issue: OpenTelemetry Collector Configuration Errors

**Symptoms:**
- Collector fails to start
- "error parsing configuration" messages
- Telemetry not reaching Honeycomb

**Common Causes:**

1. **YAML Syntax Errors**
   ```bash
   # Check YAML validity
   ./otelcol validate --config otel-collector-config.yaml
   ```

2. **Environment Variables Not Resolved**
   ```yaml
   # Issue: ${env:HONEYCOMB_API_KEY} not expanded
   # Solution: Ensure env vars are exported before starting collector
   export HONEYCOMB_API_KEY="your_key"
   ```

3. **Kubernetes DNS Resolution**
   ```yaml
   # Issue: Can't resolve "api.honeycomb.io"
   # Solution: Use FQDN or add DNS suffix
   endpoint: "api.honeycomb.io:443"  # Correct
   ```

4. **TLS/Certificate Issues**
   ```yaml
   # Add explicit TLS config if needed
   exporters:
     otlp/honeycomb:
       endpoint: "api.honeycomb.io:443"
       tls:
         insecure: false
         insecure_skip_verify: false  # Don't skip in production
   ```

5. **Check Collector Logs**
   ```bash
   # Kubernetes
   kubectl logs -n observability -l app=otel-collector --tail=100

   # Docker
   docker logs otel-collector

   # Standalone
   ./otelcol --config config.yaml 2>&1 | tee collector.log
   ```

**Diagnostic Commands:**
```bash
# Test network connectivity to Honeycomb
curl -v https://api.honeycomb.io/1/auth \
  -H "X-Honeycomb-Team: $HONEYCOMB_API_KEY"

# Validate collector config
./otelcol validate --config otel-collector-config.yaml

# Run collector in debug mode
./otelcol --config config.yaml --set service.telemetry.logs.level=debug
```

---

## Migration from Other Platforms

### From Datadog to Honeycomb

**Why Migrate:**
- Lower cost for high-cardinality tracking
- Faster query performance
- Better support for unlimited dimensions

**Migration Steps:**

1. **Parallel Export (recommended):**
```python
# Export to both Datadog and Honeycomb simultaneously
from genops.exporters.otlp import configure_multi_backend_export

configure_multi_backend_export([
    {
        "name": "datadog",
        "endpoint": f"https://otlp.{os.getenv('DATADOG_SITE')}",
        "headers": {"DD-API-KEY": os.getenv("DATADOG_API_KEY")}
    },
    {
        "name": "honeycomb",
        "endpoint": "https://api.honeycomb.io/v1/traces",
        "headers": {"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}
    }
])
```

2. **Dashboard Migration:**
- Recreate Datadog dashboards in Honeycomb Boards
- Use equivalent queries (see query translation table below)

3. **Alert Migration:**
- Recreate Datadog monitors as Honeycomb Triggers
- Update notification channels (Slack, PagerDuty)

4. **Cutover:**
- After validation period (7-30 days), remove Datadog export
- Decommission Datadog dashboards

**Query Translation:**

| Datadog Query | Honeycomb Query |
|---------------|-----------------|
| `sum:genops.cost.total{team:ai-eng}` | `WHERE genops.team = "ai-eng" \| SUM(genops.cost.total)` |
| `avg:duration{provider:openai} by {model}` | `WHERE genops.cost.provider = "openai" \| GROUP BY genops.cost.model \| AVG(duration_ms)` |
| `count:genops.policy.result{result:blocked}` | `WHERE genops.policy.result = "blocked" \| COUNT` |

### From Prometheus to Honeycomb

**Why Migrate:**
- Distributed tracing support (Prometheus is metrics-only)
- High-cardinality support (Prometheus struggles with high cardinality)
- Interactive query interface (faster exploration)

**Key Differences:**
- **Prometheus:** Metrics-based (counters, gauges, histograms)
- **Honeycomb:** Event-based (traces with arbitrary attributes)

**Migration Steps:**

1. **Keep Prometheus for Infrastructure Metrics:**
- Honeycomb is optimized for traces, not infrastructure metrics
- Keep Prometheus for CPU, memory, disk, etc.
- Use Honeycomb for AI governance telemetry

2. **Export Traces to Honeycomb:**
```python
from genops.exporters.otlp import configure_otlp_exporter

# OTLP to Honeycomb (traces)
configure_otlp_exporter(
    endpoint="https://api.honeycomb.io/v1/traces",
    headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}
)
```

3. **Grafana Integration:**
- Keep Grafana for Prometheus dashboards
- Add Honeycomb data source to Grafana (if needed)
- Or use Honeycomb Boards for AI governance

---

## Advanced Features

### Custom Instrumentation

**Instrument Custom Operations:**

```python
from opentelemetry import trace
from genops.core.context import get_effective_attributes

tracer = trace.get_tracer(__name__)

def custom_ai_operation(prompt, model):
    with tracer.start_as_current_span("custom.ai.operation") as span:
        # Add governance attributes
        attrs = get_effective_attributes()
        for key, value in attrs.items():
            span.set_attribute(f"genops.{key}", value)

        # Add operation-specific attributes
        span.set_attribute("genops.operation.type", "custom-inference")
        span.set_attribute("genops.operation.model", model)

        # Perform operation
        result = perform_inference(prompt, model)

        # Add cost tracking
        span.set_attribute("genops.cost.total", calculate_cost(result))
        span.set_attribute("genops.tokens.total", result.token_count)

        return result
```

### Multi-Dataset Export

**Export to Multiple Datasets:**

```python
from genops.exporters.otlp import configure_multi_dataset_export

# Export PII and non-PII to separate datasets
configure_multi_dataset_export(
    api_key=os.getenv("HONEYCOMB_API_KEY"),
    datasets={
        "pii": lambda attrs: "pii" in attrs.get("genops.data.classification", ""),
        "non-pii": lambda attrs: "pii" not in attrs.get("genops.data.classification", "")
    }
)
```

### Refinery (Honeycomb Sampling Proxy)

**For Extreme Scale (millions of events/second):**

Honeycomb Refinery is a tail-based sampling proxy that:
- Samples intelligently based on event attributes
- Keeps all errors and high-value traces
- Reduces event volume while maintaining visibility

**Setup:**
1. Deploy Refinery as sidecar or standalone service
2. Configure rules for intelligent sampling
3. Point GenOps export to Refinery instead of Honeycomb directly

See [Honeycomb Refinery Documentation](https://docs.honeycomb.io/manage-data-volume/refinery/) for details.

---

## Summary

Honeycomb is uniquely suited for AI governance telemetry due to:

✅ **High-Cardinality Excellence** - Track unlimited customers, users, features, transactions
✅ **Fast Interactive Queries** - Sub-second query performance for real-time cost analysis
✅ **BubbleUp for Discovery** - Automatically surface cost drivers and anomalies
✅ **Triggers & SLOs** - Proactive alerting and governance tracking
✅ **Derived Columns** - Compute governance metrics at query time
✅ **OpenTelemetry Native** - Standard OTLP export, no vendor lock-in

**Quick Start:** [Honeycomb Quickstart Guide](../honeycomb-quickstart.md)
**GitHub:** [GenOps AI Repository](https://github.com/KoshiHQ/GenOps-AI)
**Support:** [GenOps Documentation](https://docs.genops.ai)

---

## Additional Resources

- **[Honeycomb Documentation](https://docs.honeycomb.io/)** - Official Honeycomb docs
- **[OpenTelemetry Tracing](https://opentelemetry.io/docs/concepts/signals/traces/)** - OTel tracing concepts
- **[BubbleUp Guide](https://docs.honeycomb.io/working-with-data/bubbleup/)** - Honeycomb BubbleUp documentation
- **[Triggers Guide](https://docs.honeycomb.io/working-with-data/triggers/)** - Honeycomb Triggers documentation
- **[SLOs Guide](https://docs.honeycomb.io/working-with-data/slos/)** - Honeycomb SLOs documentation
- **[GenOps GitHub](https://github.com/KoshiHQ/GenOps-AI)** - Source code and examples
- **[Honeycomb Quickstart](../honeycomb-quickstart.md)** - 5-minute quick setup guide
