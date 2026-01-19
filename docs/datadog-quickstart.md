# Datadog Quickstart

Get GenOps AI governance telemetry flowing to Datadog in under 5 minutes.

## 🚀 Quick Setup

### 1. Install GenOps with OpenTelemetry Support

```bash
pip install genops-ai[opentelemetry]
```

### 2. Set Environment Variables

**Important:** Set these environment variables in your terminal/shell before running the Python code in Step 3.

```bash
export DATADOG_API_KEY="your_datadog_api_key"
export DATADOG_SITE="datadoghq.com"  # or datadoghq.eu, us5.datadoghq.com, etc.
export OTEL_SERVICE_NAME="my-ai-app"
```

**Available Datadog Sites:**
- `datadoghq.com` - US1 (default)
- `us5.datadoghq.com` - US5
- `datadoghq.eu` - EU
- `us3.datadoghq.com` - US3
- `ddog-gov.com` - US1-FED

### 3. Configure Datadog OTLP Export

**Note:** This code reads the environment variables you set in Step 2.

```python
from genops.exporters.otlp import configure_otlp_exporter
import os

# Configure Datadog as your OTLP endpoint
configure_otlp_exporter(
    endpoint=f"https://otlp.{os.getenv('DATADOG_SITE', 'datadoghq.com')}",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")}
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
# ✅ Automatically exported to Datadog!
```

### 5. View Your Telemetry in Datadog

1. Navigate to **APM → Traces** in Datadog
2. Search for service: `my-ai-app`
3. View traces with:
   - Cost and token metrics
   - Performance timing
   - Request/response details
   - Governance attributes

**That's it!** Your AI operations now appear in Datadog with:
- ✅ Real-time cost tracking by model and provider
- ✅ Token usage and performance metrics
- ✅ Distributed tracing across AI operations
- ✅ Full OpenTelemetry compatibility

## 💰 Add Cost Attribution (30 seconds)

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

# All AI operations now include attribution tags in Datadog
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
```

**View in Datadog:**
- Filter traces by `team:ai-engineering`
- Group costs by `customer_id`
- Compare costs across `project` tags

## 📊 Import Pre-Built Dashboards

GenOps provides ready-to-use Datadog dashboards:

```bash
# Download dashboard templates
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI/examples/dashboards/
```

**Import to Datadog:**

1. Navigate to **Dashboards → Dashboard List** in Datadog
2. Click **New Dashboard → Import Dashboard JSON**
3. Upload one of these dashboard files:
   - `datadog_cost_dashboard.json` - Cost attribution and budget tracking
   - `datadog_compliance_dashboard.json` - Compliance monitoring and audit trails
   - `datadog_alerting_config.json` - Alerting configuration (import via Monitors API)

**Or import via API:**

```bash
curl -X POST "https://api.datadoghq.com/api/v1/dashboard" \
  -H "DD-API-KEY: $DATADOG_API_KEY" \
  -H "DD-APPLICATION-KEY: $DATADOG_APP_KEY" \
  -H "Content-Type: application/json" \
  -d @datadog_cost_dashboard.json
```

**Dashboard Features:**
- Cost breakdown by provider, model, team, and customer
- Token usage trends and forecasting
- Policy violation alerts
- Performance SLIs and latency percentiles

## ✅ Validate Your Setup

Check that telemetry is flowing correctly:

```python
from genops.exporters.validation import validate_export_setup

# Run validation
result = validate_export_setup(provider="datadog")

if result.success:
    print("✅ Datadog export configured correctly!")
else:
    print("❌ Issues detected:")
    for issue in result.issues:
        print(f"   - {issue.message}")
        print(f"     Fix: {issue.fix_suggestion}")
```

**Common issues:**
- Missing `DATADOG_API_KEY` environment variable
- Incorrect Datadog site configuration
- Network connectivity to `otlp.datadoghq.com`
- OpenTelemetry dependencies not installed

## 🔔 Set Up Alerts (Optional)

Create alerts for cost anomalies and policy violations:

```python
# Alert when cost exceeds threshold
alert_config = {
    "name": "AI Cost Spike Alert",
    "query": "sum:genops.cost.total{service:my-ai-app} > 100",
    "message": "AI costs exceeded $100 in the last hour",
    "tags": ["team:ai-engineering", "severity:high"]
}
# Import via Datadog API or UI
```

**Recommended Alerts:**
- Cost spike detection (>2x normal spend)
- Policy violation notifications
- Budget threshold warnings (80%, 90%, 100%)
- Performance degradation (p95 latency)

## 🎯 Multi-Provider Tracking

Track costs across multiple AI providers in one dashboard:

```python
from openai import OpenAI
from anthropic import Anthropic

# Both automatically tracked with governance attributes
openai_client = OpenAI()
anthropic_client = Anthropic()

# OpenAI call
response1 = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Task 1"}]
)

# Anthropic call
response2 = anthropic_client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Task 2"}]
)

# Both appear in Datadog with unified cost attribution
# Filter by: provider:openai or provider:anthropic
```

## 📈 Next Steps

- **[Comprehensive Datadog Integration Guide](integrations/datadog.md)** - Advanced features and production patterns
- **[Import Dashboards](../examples/dashboards/)** - Ready-to-use visualization templates
- **[Set Up Alerts](integrations/datadog.md#alerting-monitoring)** - Cost and compliance alerting
- **[Kubernetes Deployment](kubernetes-observability.md)** - Production deployment with Helm charts
- **[Custom Metrics](integrations/datadog.md#custom-metrics)** - Build organization-specific dashboards

## 🐛 Troubleshooting

### Telemetry not appearing in Datadog

1. **Check API key:** `echo $DATADOG_API_KEY`
2. **Verify site:** Ensure `DATADOG_SITE` matches your Datadog region
3. **Test connectivity:**
   ```bash
   curl -v https://otlp.datadoghq.com/v1/traces \
     -H "DD-API-KEY: $DATADOG_API_KEY"
   ```
4. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Cost metrics missing

- Ensure `auto_instrument()` is called before AI operations
- Verify provider-specific cost calculators are installed:
  ```bash
  pip install genops-ai[openai]  # For OpenAI cost tracking
  ```

### High telemetry volume

Configure sampling to reduce data volume:

```python
from genops.exporters.otlp import configure_otlp_exporter

configure_otlp_exporter(
    endpoint=f"https://otlp.{os.getenv('DATADOG_SITE')}",
    headers={"DD-API-KEY": os.getenv("DATADOG_API_KEY")},
    sampling_rate=0.1  # Sample 10% of traces
)
```

## 💬 Support

- **Documentation:** [Full Datadog Integration Guide](integrations/datadog.md)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Examples:** [Example Code](../examples/observability/datadog_integration.py)
- **Community:** [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
