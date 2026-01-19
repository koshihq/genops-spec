# Datadog Integration

**Export AI governance telemetry to Datadog for comprehensive observability, cost tracking, and compliance monitoring.**

## Overview

The GenOps Datadog integration enables organizations to export AI governance telemetry — cost attribution, policy enforcement, budget tracking, and evaluation metrics — into Datadog's observability platform using OpenTelemetry OTLP export.

### Problems Solved

- **Cross-Stack AI Visibility:** Unified view of AI operations across OpenAI, Anthropic, Bedrock, Gemini, and 35+ providers
- **Cost Attribution:** Track and analyze AI costs by team, project, customer, and model in real-time
- **Policy Compliance:** Monitor policy enforcement and compliance violations with alerting
- **Budget Management:** Track budget consumption with proactive alerts to prevent overruns
- **Performance Monitoring:** APM-style distributed tracing for AI operations
- **Custom Dashboards:** Pre-built dashboards for cost, compliance, and performance

### Value Proposition

**For Platform Teams:**
- Centralized governance telemetry in your existing Datadog observability stack
- No vendor lock-in — standard OpenTelemetry/OTLP integration
- Distributed tracing shows complete AI operation flows
- Integration with existing APM, logs, and infrastructure monitoring

**For FinOps Teams:**
- Per-team, per-project, per-customer cost attribution with real-time visibility
- Budget tracking with proactive alerting capabilities
- Cost optimization insights (model efficiency, provider comparison, token usage trends)
- Historical cost analysis and forecasting

**For Compliance Teams:**
- Complete audit trail for all AI operations
- Policy violation tracking with alert workflows
- Data classification and governance enforcement
- Compliance dashboard templates

---

## Core Concepts

### 1. OpenTelemetry OTLP Export

GenOps exports telemetry to Datadog using the **OpenTelemetry OTLP protocol**, ensuring vendor neutrality and interoperability.

**Architecture:**
```
GenOps AI Application
    ↓
OpenTelemetry SDK (traces, metrics, logs)
    ↓
OTLP Exporter (HTTP/gRPC)
    ↓
Datadog OTLP Endpoint (otlp.datadoghq.com)
    ↓
Datadog Platform (APM, Metrics, Logs)
```

**Benefits:**
- Standard protocol supported by 40+ observability platforms
- No Datadog-specific SDK required
- Easy migration between observability vendors
- Native OpenTelemetry ecosystem compatibility

### 2. Datadog Sites and Endpoints

Datadog operates in multiple geographic regions. Configure the correct site for your organization:

| Site | Region | OTLP Endpoint |
|------|--------|---------------|
| `datadoghq.com` | US1 (Virginia) | `https://otlp.datadoghq.com` |
| `us5.datadoghq.com` | US5 (Oregon) | `https://otlp.us5.datadoghq.com` |
| `datadoghq.eu` | EU (Frankfurt) | `https://otlp.datadoghq.eu` |
| `us3.datadoghq.com` | US3 (Oregon) | `https://otlp.us3.datadoghq.com` |
| `ddog-gov.com` | US1-FED (GovCloud) | `https://otlp.ddog-gov.com` |

**How to find your site:**
1. Log in to Datadog
2. Check your browser URL: `https://app.{your-site}/`
3. Use that value for `DATADOG_SITE` environment variable

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

**Evaluation Fields:**
- `genops.eval.safety`: Safety score (0-1)
- `genops.eval.accuracy`: Accuracy score (0-1)
- `genops.eval.compliance`: Compliance score (0-1)
- `genops.eval.performed`: Boolean flag for evaluation

**Performance Fields:**
- `duration_ms`: Operation duration in milliseconds
- `status`: Operation status (success, error, timeout)

### 4. Authentication

Datadog authentication uses **API Keys** passed via HTTP headers:

```bash
# Required environment variable
export DATADOG_API_KEY="your_32_char_api_key"

# Optional: Application Key for dashboard/monitor creation
export DATADOG_APP_KEY="your_40_char_app_key"
```

**Security Best Practices:**
- Store API keys in secret management systems (AWS Secrets Manager, HashiCorp Vault, etc.)
- Rotate keys every 90 days
- Use separate keys for production and non-production environments
- Grant minimum required permissions (metrics write, traces write)
- Never commit keys to version control

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
- **OTLP Exporter:** HTTP or gRPC (HTTP recommended for Datadog)

### Datadog Requirements

- **Datadog Account:** Any plan tier (Free, Pro, Enterprise)
- **API Key:** Generate from Organization Settings → API Keys
- **Permissions:** API key needs `metrics_write` and `traces_write` scopes
- **Optional App Key:** Required for programmatic dashboard/monitor creation

**Generate API Key:**
1. Navigate to **Organization Settings → API Keys**
2. Click **New Key**
3. Name: `genops-ai-production`
4. Copy the generated 32-character key
5. Set environment variable: `export DATADOG_API_KEY="..."`

### Verify Installation

```bash
# Check GenOps installation
python -c "import genops; print(genops.__version__)"

# Check OpenTelemetry installation
python -c "import opentelemetry; print('OpenTelemetry OK')"

# Check environment variables
echo $DATADOG_API_KEY | wc -c  # Should output 33 (32 chars + newline)
echo $DATADOG_SITE               # Should output your site (e.g., datadoghq.com)
```

---

## Quick Start

See the **[Datadog Quickstart Guide](../datadog-quickstart.md)** for 5-minute setup.

### Auto-Instrumentation (Zero-Code Setup)

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument
import os

# Configure Datadog as OTLP endpoint
configure_otlp_exporter(
    endpoint=f"https://otlp.{os.getenv('DATADOG_SITE', 'datadoghq.com')}",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")}
)

# Enable auto-instrumentation
auto_instrument()

# All AI operations now export to Datadog!
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Automatically tracked in Datadog
```

### Manual Instrumentation (Fine-Grained Control)

```python
from genops.providers.openai import instrument_openai
from genops.core.context import set_governance_context
from genops.exporters.otlp import configure_otlp_exporter
import os

# Configure Datadog export
configure_otlp_exporter(
    endpoint=f"https://otlp.{os.getenv('DATADOG_SITE')}",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")},
    service_name="customer-ai-service",
    environment="production"
)

# Create instrumented client
client = instrument_openai()

# Set governance context for cost attribution
set_governance_context({
    "team": "ai-platform",
    "project": "customer-chatbot",
    "customer_id": "enterprise-123",
    "cost_center": "engineering"
})

# Make AI call with automatic telemetry
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
# ✅ Exported to Datadog with full governance attributes
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATADOG_API_KEY` | ✅ Yes | None | Datadog API key (32 chars) |
| `DATADOG_SITE` | ❌ No | `datadoghq.com` | Datadog site/region |
| `DATADOG_APP_KEY` | ❌ No | None | Datadog Application Key (for dashboard/monitor creation) |
| `OTEL_SERVICE_NAME` | ❌ No | `genops-ai` | Service name in Datadog APM |
| `OTEL_ENVIRONMENT` | ❌ No | `production` | Environment tag (dev/staging/prod) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | ❌ No | `http` | OTLP protocol (http or grpc) |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | ❌ No | `10000` | Export timeout in milliseconds |

### Programmatic Configuration

```python
from genops.exporters.otlp import configure_otlp_exporter

configure_otlp_exporter(
    # Datadog endpoint
    endpoint="https://otlp.datadoghq.com",
    headers={"DD-API-KEY": "your_api_key_here"},

    # Service metadata
    service_name="my-ai-application",
    service_version="1.2.3",
    environment="production",

    # Export configuration
    protocol="http",  # or "grpc"
    timeout=10000,    # milliseconds

    # Batching and performance
    batch_size=512,
    batch_interval_ms=5000,
    max_queue_size=2048,

    # Sampling (for high-volume apps)
    sampling_rate=1.0,  # 1.0 = 100% (sample all traces)

    # Resource attributes
    resource_attributes={
        "deployment.environment": "production",
        "service.namespace": "ai-platform",
        "team": "engineering"
    }
)
```

### Sampling Configuration

For high-volume applications, configure sampling to reduce telemetry volume:

```python
from genops.exporters.otlp import configure_otlp_exporter

# Sample 10% of traces (reduce costs for high-volume apps)
configure_otlp_exporter(
    endpoint="https://otlp.datadoghq.com",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")},
    sampling_rate=0.1  # 10% sampling
)

# Or use adaptive sampling based on operation type
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

sampler = ParentBased(
    root=TraceIdRatioBased(0.1)  # 10% of root spans
)
```

---

## Integration Patterns

### Pattern 1: Context Manager for Complex Workflows

```python
from genops.core.tracker import track_ai_operation
from genops.core.context import set_governance_context

# Set governance attributes
set_governance_context({
    "team": "data-science",
    "project": "model-training",
    "customer_id": "internal",
    "environment": "production"
})

# Track complex multi-step workflow
with track_ai_operation("document-processing-pipeline") as workflow:
    # Step 1: Document extraction
    with track_ai_operation("document-extraction", parent=workflow) as step1:
        extracted_text = extract_document(file_path)
        step1.add_cost(provider="openai", model="gpt-4", cost=0.03)

    # Step 2: Entity recognition
    with track_ai_operation("entity-recognition", parent=workflow) as step2:
        entities = recognize_entities(extracted_text)
        step2.add_cost(provider="anthropic", model="claude-3-sonnet", cost=0.02)

    # Step 3: Classification
    with track_ai_operation("document-classification", parent=workflow) as step3:
        category = classify_document(entities)
        step3.add_cost(provider="openai", model="gpt-3.5-turbo", cost=0.01)

# ✅ Complete workflow appears in Datadog with nested spans and unified cost
```

### Pattern 2: Multi-Provider Cost Aggregation

```python
from genops.core.aggregation import create_cost_aggregator

# Track costs across multiple providers
with create_cost_aggregator("customer-interaction") as aggregator:
    # OpenAI call
    openai_response = openai_client.chat.completions.create(...)
    aggregator.add_provider_call("openai", "gpt-4", cost=0.05)

    # Anthropic call
    anthropic_response = anthropic_client.messages.create(...)
    aggregator.add_provider_call("anthropic", "claude-3-sonnet", cost=0.03)

    # Bedrock call
    bedrock_response = bedrock_client.invoke_model(...)
    aggregator.add_provider_call("bedrock", "titan-text", cost=0.01)

# ✅ Unified cost summary in Datadog
# Total: $0.09 across 3 providers
```

### Pattern 3: Policy-Constrained Operations

```python
from genops.policy import enforce_policy, PolicyViolation

# Define budget policy
budget_policy = {
    "name": "team-monthly-budget",
    "limit": 1000.0,
    "period": "monthly",
    "enforcement": "block"  # or "warn"
}

try:
    # Check policy before expensive operation
    with enforce_policy(budget_policy):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[...]
        )
except PolicyViolation as e:
    print(f"Policy violation: {e.policy_name} - {e.reason}")
    # ✅ Policy violation logged to Datadog with alert
```

---

## Dashboards & Visualization

### Pre-Built Dashboards

GenOps provides three ready-to-use Datadog dashboards:

#### 1. Cost Attribution Dashboard

**File:** `examples/dashboards/datadog_cost_dashboard.json`

**Features:**
- Real-time cost tracking by provider, model, team, and customer
- Token usage trends and forecasting
- Cost per operation analysis
- Provider cost comparison
- Budget consumption tracking

**Key Widgets:**
- AI Cost Overview (timeseries)
- Cost by Customer (top list)
- Token Usage by Provider (query value)
- Average Cost per Operation (query value)
- Cost per Model (distribution)

**Import:**
1. Download: `examples/dashboards/datadog_cost_dashboard.json`
2. Navigate to **Dashboards → New Dashboard → Import Dashboard JSON**
3. Paste JSON content and save

#### 2. Compliance Monitoring Dashboard

**File:** `examples/dashboards/datadog_compliance_dashboard.json`

**Features:**
- Overall compliance score
- Policy enforcement results
- Audit trail volume
- Operations by data classification
- Safety and compliance evaluation scores

**Key Widgets:**
- Overall Compliance Score (query value with threshold)
- Policy Enforcement Results (distribution by result)
- Audit Trail Volume (timeseries)
- Data Classification (sunburst chart)

#### 3. Performance Monitoring Dashboard

**File:** (Generated programmatically or custom)

**Features:**
- Operation latency (p50, p95, p99)
- Throughput (operations per minute)
- Error rates and success rates
- Provider performance comparison
- SLI tracking

**Key Queries:**
```
# P95 Latency by Provider
p95:genops.operation.duration{*} by {genops.cost.provider}

# Error Rate
sum:genops.operation.error{*} / sum:genops.operation.total{*} * 100

# Operations per Minute
rate(sum:genops.operation.total{*})
```

### Creating Custom Dashboards

```python
from genops.exporters.datadog import create_custom_dashboard

# Programmatically create custom dashboard
dashboard = create_custom_dashboard(
    title="AI Chatbot Performance",
    widgets=[
        {
            "type": "timeseries",
            "title": "Chatbot Response Time",
            "query": "avg:genops.operation.duration{feature:chatbot}",
        },
        {
            "type": "query_value",
            "title": "Daily Chatbot Cost",
            "query": "sum:genops.cost.total{feature:chatbot}",
        }
    ],
    template_variables=[
        {"name": "customer_id", "prefix": "genops.customer_id"},
        {"name": "environment", "prefix": "genops.environment"}
    ]
)

# Submit to Datadog API
submit_dashboard(dashboard)
```

---

## Alerting & Monitoring

### Recommended Alerts

#### 1. Cost Spike Alert

**Trigger:** AI costs exceed 2x normal spend in the last hour

```python
{
  "name": "AI Cost Spike Alert",
  "type": "metric alert",
  "query": "avg(last_1h):sum:genops.cost.total{*} > 2 * avg(last_4h):sum:genops.cost.total{*}",
  "message": """
AI costs are 2x higher than normal in the last hour.

**Investigation:**
- View cost by team: Filter by `genops.team`
- Check for runaway operations
- Review recent deployments

@slack-ai-governance @pagerduty-ai-ops
  """,
  "tags": ["team:ai-governance", "severity:high"]
}
```

**Create in Datadog:**
1. Navigate to **Monitors → New Monitor → Metric**
2. Query: `sum:genops.cost.total{*}`
3. Alert threshold: `> 2 * avg(last_4h)`
4. Set notification channels

#### 2. Policy Violation Alert

**Trigger:** More than 5 policy violations in 15 minutes

```python
{
  "name": "Policy Violation Rate High",
  "type": "metric alert",
  "query": "sum(last_15m):sum:genops.policy.violation{*} > 5",
  "message": """
High rate of policy violations detected.

**Common Causes:**
- Budget limit exceeded
- Compliance policy breaches
- Content safety failures

Dashboard: {{host.dashboard.link}}

@slack-compliance @pagerduty-compliance
  """,
  "tags": ["team:compliance", "severity:critical"]
}
```

#### 3. Budget Threshold Alert

**Trigger:** Team budget consumption exceeds 80%

```python
{
  "name": "Team Budget Threshold Warning",
  "type": "metric alert",
  "query": "avg(last_5m):(sum:genops.budget.consumed{*} / sum:genops.budget.limit{*}) * 100 > 80",
  "message": """
Team {{genops.team}} has consumed 80% of monthly budget.

Current consumption: {{value}}%
Remaining: {{genops.budget.remaining}}

@slack-finops
  """,
  "tags": ["team:finops", "severity:warning"]
}
```

#### 4. Safety Score Alert

**Trigger:** AI safety evaluation scores drop below 85%

```python
{
  "name": "AI Safety Score Low",
  "type": "metric alert",
  "query": "avg(last_5m):avg:genops.eval.safety{*} < 0.85",
  "message": """
AI safety scores below acceptable threshold.

Current: {{value}}
Required: 0.85

**Actions:**
1. Review recent AI operations
2. Check for new model deployments
3. Increase human review

@slack-ai-safety @pagerduty-ai-safety
  """,
  "tags": ["team:ai-safety", "severity:high"]
}
```

### Service Level Indicators (SLIs)

**AI Operation Success Rate:**
```
Query: sum:genops.operation.success{*} / sum:genops.operation.total{*} * 100
Target: ≥ 99.5%
```

**Policy Response Time:**
```
Query: avg:genops.policy.response_time{*}
Target: ≤ 100ms
```

**Compliance Evaluation Coverage:**
```
Query: sum:genops.eval.performed{*} / sum:genops.operation.total{*} * 100
Target: ≥ 95%
```

---

## Queries & Analysis

### Example Datadog Queries

#### Cost Analysis

```
# Total cost by team (last 7 days)
sum:genops.cost.total{*} by {genops.team}.rollup(sum, 86400)

# Cost per customer (top 20)
top(sum:genops.cost.total{*} by {genops.customer_id}, 20, 'mean', 'desc')

# Cost by provider and model
sum:genops.cost.total{*} by {genops.cost.provider,genops.cost.model}

# Average cost per operation
avg:genops.cost.total{*}

# Cost trend with forecast
forecast(sum:genops.cost.total{*}, 'linear', 7)
```

#### Token Usage Analysis

```
# Total tokens by provider
sum:genops.tokens.total{*} by {genops.cost.provider}

# Input vs output token ratio
sum:genops.tokens.input{*} / sum:genops.tokens.output{*}

# Token usage by feature
sum:genops.tokens.total{*} by {genops.feature}

# Anomaly detection on token usage
anomalies(avg:genops.tokens.total{*}, 'agile', 2)
```

#### Performance Analysis

```
# P95 latency by operation
p95:genops.operation.duration{*} by {operation_name}

# Error rate
sum:genops.operation.error{*} / sum:genops.operation.total{*} * 100

# Throughput (operations per minute)
per_minute(sum:genops.operation.total{*})

# Slowest operations (top 10)
top(avg:genops.operation.duration{*} by {operation_name}, 10, 'mean', 'desc')
```

#### Compliance Analysis

```
# Policy violations by type
sum:genops.policy.violation{*} by {genops.policy.name}

# Compliance score trend
avg:genops.eval.safety{*}.rollup(avg, 3600)

# Operations by data classification
sum:genops.operation.total{*} by {genops.data.classification}

# Audit coverage percentage
(sum:genops.eval.performed{*} / sum:genops.operation.total{*}) * 100
```

---

## Production Deployment

### Kubernetes with Helm Charts

**GenOps provides Helm charts for production Kubernetes deployment:**

```bash
# Add GenOps Helm repository
helm repo add genops https://helm.genops.ai
helm repo update

# Install with Datadog export
helm install genops-ai genops/genops-ai \
  --namespace genops \
  --create-namespace \
  --set export.datadog.enabled=true \
  --set export.datadog.site="datadoghq.com" \
  --set export.datadog.apiKeySecret="genops-datadog-api-key"

# Create Kubernetes secret with Datadog API key
kubectl create secret generic genops-datadog-api-key \
  --namespace genops \
  --from-literal=api-key="your_datadog_api_key"
```

**Helm Values Example:**

```yaml
# values.yaml
export:
  datadog:
    enabled: true
    site: "datadoghq.com"
    apiKeySecret: "genops-datadog-api-key"

    # OTLP configuration
    otlp:
      protocol: "http"
      timeout: 10000
      batch_size: 512
      batch_interval_ms: 5000

    # Sampling (for high volume)
    sampling:
      enabled: true
      rate: 0.1  # 10% sampling

# Service configuration
service:
  name: "genops-ai-production"
  environment: "production"
  namespace: "ai-platform"

# Resource limits
resources:
  limits:
    cpu: "1000m"
    memory: "2Gi"
  requests:
    cpu: "500m"
    memory: "1Gi"
```

### Multi-Environment Setup

**Separate configurations for dev/staging/prod:**

```python
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

DATADOG_CONFIG = {
    "development": {
        "endpoint": "https://otlp.datadoghq.com",
        "api_key": os.getenv("DATADOG_DEV_API_KEY"),
        "service_name": "genops-ai-dev",
        "sampling_rate": 1.0,  # 100% sampling in dev
    },
    "staging": {
        "endpoint": "https://otlp.datadoghq.com",
        "api_key": os.getenv("DATADOG_STAGING_API_KEY"),
        "service_name": "genops-ai-staging",
        "sampling_rate": 0.5,  # 50% sampling in staging
    },
    "production": {
        "endpoint": "https://otlp.datadoghq.com",
        "api_key": os.getenv("DATADOG_PROD_API_KEY"),
        "service_name": "genops-ai-prod",
        "sampling_rate": 0.1,  # 10% sampling in production
    }
}

from genops.exporters.otlp import configure_otlp_exporter

config = DATADOG_CONFIG[ENVIRONMENT]
configure_otlp_exporter(
    endpoint=config["endpoint"],
    headers={"DD-API-KEY": config["api_key"]},
    service_name=config["service_name"],
    environment=ENVIRONMENT,
    sampling_rate=config["sampling_rate"]
)
```

### Secret Management

**AWS Secrets Manager Integration:**

```python
import boto3
import json

def get_datadog_api_key():
    """Retrieve Datadog API key from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId='prod/genops/datadog-api-key')
    secret = json.loads(response['SecretString'])
    return secret['api_key']

# Configure with secret from AWS
from genops.exporters.otlp import configure_otlp_exporter

configure_otlp_exporter(
    endpoint="https://otlp.datadoghq.com",
    headers={"DD-API-KEY": get_datadog_api_key()},
    service_name="genops-ai-production"
)
```

### Scaling Considerations

**For high-volume applications (>10,000 operations/minute):**

1. **Enable Sampling:** Reduce telemetry volume
   ```python
   sampling_rate=0.1  # Sample 10% of traces
   ```

2. **Increase Batch Size:** Reduce network overhead
   ```python
   batch_size=2048
   batch_interval_ms=10000
   ```

3. **Use OTLP Collector:** Deploy OpenTelemetry Collector as intermediary
   ```bash
   # Deploy OTLP Collector to aggregate and batch telemetry
   helm install otel-collector open-telemetry/opentelemetry-collector \
     --set config.exporters.datadog.api.key="$DATADOG_API_KEY"
   ```

4. **Regional Endpoints:** Use closest Datadog site for lower latency

---

## Troubleshooting

### Telemetry Not Appearing in Datadog

**Check 1: Verify API Key**
```bash
# Check API key is set
echo $DATADOG_API_KEY | wc -c  # Should be 33 (32 + newline)

# Test API key with curl
curl -X POST "https://api.datadoghq.com/api/v1/validate" \
  -H "DD-API-KEY: $DATADOG_API_KEY"
# Should return: {"valid": true}
```

**Check 2: Verify OTLP Endpoint Connectivity**
```bash
# Test network connectivity
curl -v https://otlp.datadoghq.com/v1/traces \
  -H "DD-API-KEY: $DATADOG_API_KEY" \
  -H "Content-Type: application/x-protobuf" \
  --data-binary "@/dev/null"

# Should return 200 or 400 (not connection errors)
```

**Check 3: Enable Debug Logging**
```python
import logging

# Enable OpenTelemetry debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

# Run your application and check logs for export errors
```

**Check 4: Verify Service Name**
```python
# Ensure service name is set
import os
print(f"Service: {os.getenv('OTEL_SERVICE_NAME', 'default')}")

# Check in Datadog: APM → Services → Should see your service
```

### Cost Metrics Missing

**Issue:** Traces appear but no cost metrics

**Solution 1:** Ensure cost calculators installed
```bash
pip install genops-ai[openai]  # For OpenAI cost tracking
pip install genops-ai[anthropic]  # For Anthropic cost tracking
```

**Solution 2:** Verify auto-instrumentation enabled
```python
from genops import auto_instrument

# Must be called before AI operations
auto_instrument()
```

**Solution 3:** Check metric names in Datadog
```
# Search for: genops.cost.*
# If missing, cost tracking may not be enabled
```

### High Telemetry Costs

**Issue:** Datadog telemetry ingestion costs are high

**Solution 1:** Enable Sampling
```python
configure_otlp_exporter(
    endpoint="https://otlp.datadoghq.com",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")},
    sampling_rate=0.1  # Reduce to 10% sampling
)
```

**Solution 2:** Filter Low-Value Traces
```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

# Sample only important operations
sampler = ParentBased(root=TraceIdRatioBased(0.1))
```

**Solution 3:** Use Retention Filters in Datadog
- Navigate to **APM → Retention Filters**
- Create filter: Retain only high-cost operations or errors
- Reduces retained span volume

### API Key Permission Errors

**Error:** `403 Forbidden` when exporting telemetry

**Solution:** Verify API key has correct permissions

1. Navigate to **Organization Settings → API Keys**
2. Find your API key
3. Ensure it has scopes:
   - `metrics_write`
   - `traces_write`
   - `logs_write` (if logging enabled)

### Network Connectivity Issues

**Issue:** `Connection refused` or `Timeout` errors

**Solution 1:** Check corporate firewall/proxy
```bash
# Test HTTPS connectivity
curl -v https://otlp.datadoghq.com

# If using proxy, set environment variables
export HTTPS_PROXY="http://proxy.company.com:3128"
```

**Solution 2:** Verify Datadog site is correct
```bash
# Ensure DATADOG_SITE matches your account
echo $DATADOG_SITE
# Should be: datadoghq.com, datadoghq.eu, us5.datadoghq.com, etc.
```

---

## Reference

### Complete Configuration Example

```python
from genops.exporters.otlp import configure_otlp_exporter
from genops import auto_instrument
from genops.core.context import set_governance_context
import os

# Configure Datadog OTLP export
configure_otlp_exporter(
    # Datadog endpoint
    endpoint=f"https://otlp.{os.getenv('DATADOG_SITE', 'datadoghq.com')}",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")},

    # Service metadata
    service_name=os.getenv("OTEL_SERVICE_NAME", "my-ai-app"),
    service_version="1.0.0",
    environment=os.getenv("ENVIRONMENT", "production"),

    # Export configuration
    protocol="http",
    timeout=10000,

    # Performance tuning
    batch_size=512,
    batch_interval_ms=5000,
    max_queue_size=2048,

    # Sampling for high-volume apps
    sampling_rate=float(os.getenv("OTEL_SAMPLING_RATE", "1.0")),

    # Resource attributes
    resource_attributes={
        "deployment.environment": os.getenv("ENVIRONMENT"),
        "service.namespace": "ai-platform",
        "team": "engineering"
    }
)

# Enable auto-instrumentation
auto_instrument()

# Set default governance context
set_governance_context({
    "team": "ai-engineering",
    "project": "production-ai-app",
    "environment": os.getenv("ENVIRONMENT"),
    "cost_center": "engineering"
})

# Your AI application code
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, Datadog!"}]
)

# ✅ Complete telemetry exported to Datadog with governance attributes
```

### API Reference

See **[examples/observability/datadog_integration.py](../../examples/observability/datadog_integration.py)** for complete working examples.

### Related Documentation

- **[Datadog Quickstart](../datadog-quickstart.md)** - 5-minute setup guide
- **[Kubernetes Observability](../kubernetes-observability.md)** - Kubernetes deployment patterns
- **[OpenTelemetry Integration](opentelemetry.md)** - Core OpenTelemetry concepts
- **[Cost Tracking](../cost-tracking.md)** - Cost attribution and budget management

### External Resources

- **[Datadog OTLP Documentation](https://docs.datadoghq.com/opentelemetry/)** - Official Datadog OTLP guide
- **[OpenTelemetry Python SDK](https://opentelemetry-python.readthedocs.io/)** - OpenTelemetry Python documentation
- **[Datadog API Reference](https://docs.datadoghq.com/api/latest/)** - Datadog API documentation for dashboards and monitors

---

## Support

- **Documentation:** [GenOps AI Documentation](https://github.com/KoshiHQ/GenOps-AI)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Examples:** [Datadog Integration Example](../../examples/observability/datadog_integration.py)
- **Community:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Datadog Support:** [Datadog Support Portal](https://help.datadoghq.com/)
