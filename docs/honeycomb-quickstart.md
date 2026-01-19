# Honeycomb Quickstart

Get GenOps AI governance telemetry flowing to Honeycomb in under 5 minutes.

## 🚀 Quick Setup (5 Minutes)

### 1. Install GenOps with OpenTelemetry Support

```bash
pip install genops-ai[opentelemetry]
```

### 2. Set Environment Variables

**Important:** Set these environment variables in your terminal/shell before running the Python code in Step 3.

```bash
export HONEYCOMB_API_KEY="your_honeycomb_api_key"
export HONEYCOMB_DATASET="genops-ai"  # Optional: defaults to "genops-ai"
export OTEL_SERVICE_NAME="my-ai-app"
```

**Get Your API Key:**
1. Log in to [Honeycomb](https://ui.honeycomb.io)
2. Navigate to **Team Settings → API Keys**
3. Create or copy an existing API key
4. Create a dataset (or use an existing one)

### 3. Configure Honeycomb OTLP Export

**Note:** This code reads the environment variables you set in Step 2.

```python
from genops.exporters.otlp import configure_otlp_exporter
import os

# Configure Honeycomb as your OTLP endpoint
configure_otlp_exporter(
    endpoint="https://api.honeycomb.io/v1/traces",
    headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}
)
```

### 4. Enable Auto-Instrumentation (Zero Code Changes)

**Note:** This example uses OpenAI for demonstration. Before proceeding:
1. Set your OpenAI API key: `export OPENAI_API_KEY="sk-..."`
2. Or substitute with any GenOps-supported provider (Anthropic, Bedrock, Gemini, etc.)

See [provider documentation](../README.md#ai--llm-ecosystem) for other options.

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
# ✅ Automatically exported to Honeycomb!
```

### 5. View Your Telemetry in Honeycomb

1. Navigate to your **Honeycomb dataset** (e.g., `genops-ai`)
2. Click **New Query** or view recent traces
3. See high-cardinality AI governance data:
   - Cost and token metrics
   - Performance timing with distributed tracing
   - Request/response details
   - Governance attributes (team, project, customer_id)

**That's it!** Your AI operations now appear in Honeycomb with:
- ✅ Real-time cost tracking by model and provider
- ✅ High-cardinality attribution (team, customer, feature)
- ✅ Token usage and performance metrics
- ✅ Distributed tracing across AI operations
- ✅ Full OpenTelemetry compatibility

---

## 💰 Add Cost Attribution (30 Seconds)

Track costs by team, project, or customer with high-cardinality attributes:

```python
from genops.core.context import set_governance_context

# Set once - applies to all operations
set_governance_context({
    "team": "ai-engineering",
    "project": "customer-chatbot",
    "customer_id": "enterprise_123",
    "environment": "production",
    "feature": "chat"
})

# All AI operations now include attribution tags in Honeycomb
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
```

**Query in Honeycomb:**

Honeycomb's high-cardinality analysis is perfect for AI governance:

```
# Cost by customer
WHERE genops.cost.provider = "openai"
| GROUP BY genops.customer_id
| SUM(genops.cost.total)
```

```
# Token efficiency by team
GROUP BY genops.team, genops.cost.model
| AVG(genops.tokens.total / genops.cost.total)
```

```
# Performance by feature
WHERE genops.environment = "production"
| GROUP BY genops.feature
| HEATMAP(duration_ms, genops.tokens.total)
```

---

## 🔍 Essential Honeycomb Queries

### Cost Analysis

```
# Total cost by provider and model
GROUP BY genops.cost.provider, genops.cost.model
| SUM(genops.cost.total)
| ORDER BY SUM DESC
```

```
# Cost per customer (top 10)
GROUP BY genops.customer_id
| SUM(genops.cost.total)
| ORDER BY SUM DESC
| LIMIT 10
```

```
# Daily cost trend
WHERE genops.cost.total EXISTS
| GROUP BY DATE_TRUNC("day", timestamp)
| SUM(genops.cost.total)
```

### Performance Analysis

```
# P95 latency by model
GROUP BY genops.cost.model
| P95(duration_ms)
```

```
# Slow operations (>2 seconds)
WHERE duration_ms > 2000
| COUNT
| GROUP BY genops.team, genops.feature
```

```
# Token throughput correlation
HEATMAP(duration_ms, genops.tokens.total)
```

### Attribution Analysis

```
# Multi-dimensional cost breakdown
GROUP BY genops.team, genops.project, genops.environment
| SUM(genops.cost.total)
| ORDER BY SUM DESC
```

```
# Customer tier analysis
GROUP BY genops.customer_tier
| AVG(genops.cost.total), COUNT
```

### BubbleUp for Root Cause Analysis

Honeycomb's **BubbleUp** feature is perfect for finding cost outliers:

1. Click **BubbleUp** in the query interface
2. Select metric: `SUM(genops.cost.total)`
3. BubbleUp automatically finds attributes that distinguish high-cost operations
4. Examples it might surface:
   - High costs from specific `customer_id` values
   - Expensive operations in specific `genops.feature` contexts
   - Cost spikes from particular `genops.cost.model` choices

---

## 📊 Create Custom Dashboards

### Option 1: Create in Honeycomb UI

1. Navigate to **Boards → Create New Board**
2. Add queries using the examples above
3. Recommended board sections:
   - **Cost Overview**: Total spend, cost by provider, daily trends
   - **Attribution**: Cost by team, project, customer
   - **Performance**: Latency percentiles, token throughput
   - **Compliance**: Policy evaluations, data classification tracking

### Option 2: Import GenOps Template

**Coming Soon:** Pre-built Honeycomb board templates will be available at:
```bash
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI/examples/dashboards/honeycomb/
```

### Recommended Visualizations

| Metric | Visualization Type | Query |
|--------|-------------------|-------|
| **Total Cost** | Time series | `SUM(genops.cost.total)` |
| **Cost by Provider** | Bar chart | `GROUP BY genops.cost.provider \| SUM(genops.cost.total)` |
| **Latency Distribution** | Heatmap | `HEATMAP(duration_ms, genops.tokens.total)` |
| **Top Customers** | Table | `GROUP BY genops.customer_id \| SUM(genops.cost.total) \| LIMIT 10` |
| **Token Efficiency** | Line graph | `SUM(genops.tokens.total) / SUM(genops.cost.total)` |

---

## ✅ Validate Your Setup

Check that telemetry is flowing correctly:

```python
from genops.exporters.validation import validate_export_setup, print_validation_result

# Run validation
result = validate_export_setup(provider="honeycomb")

# Display results with specific fix suggestions
print_validation_result(result)
```

**Expected Output:**

```
✅ Honeycomb Setup Validation

Configuration:
  ✅ HONEYCOMB_API_KEY: Set
  ✅ HONEYCOMB_DATASET: genops-ai
  ✅ OTEL endpoint: https://api.honeycomb.io/v1/traces
  ✅ Headers: X-Honeycomb-Team configured

Connectivity:
  ✅ Honeycomb endpoint reachable
  ✅ API key valid
  ✅ Dataset accessible

✅ All checks passed! Telemetry is flowing to Honeycomb.
```

---

## ⚠️ Troubleshooting

### Issue: "No data appearing in Honeycomb"

**Check:**
1. **API Key**: Verify `HONEYCOMB_API_KEY` is set correctly
   ```bash
   echo $HONEYCOMB_API_KEY
   ```

2. **Dataset Name**: Ensure dataset exists in Honeycomb UI
   ```bash
   echo $HONEYCOMB_DATASET
   ```

3. **Run Validation**:
   ```python
   from genops.exporters.validation import validate_export_setup
   validate_export_setup(provider="honeycomb")
   ```

4. **Check Logs**: Enable debug logging
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

**Solution:**
- Create dataset in Honeycomb UI if it doesn't exist
- Verify API key has write permissions
- Check network connectivity to `api.honeycomb.io`

### Issue: "Authentication failed"

**Error:**
```
Failed to export to Honeycomb: 401 Unauthorized
```

**Solution:**
1. Verify API key is correct (check for extra spaces/newlines)
2. Ensure API key has write access to the dataset
3. Check that header is set correctly:
   ```python
   headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")}
   ```

### Issue: "High cardinality warning"

**Honeycomb Message:**
```
Warning: High cardinality detected on field 'genops.customer_id'
```

**This is expected and encouraged!** Honeycomb is designed for high-cardinality analysis.

**Best Practices:**
- Customer IDs, user IDs, transaction IDs are perfect for Honeycomb
- Use **Derived Columns** to reduce cardinality if needed
- Consider **sampling** for extremely high-volume applications

**Enable Sampling (if needed):**
```python
from genops.exporters.otlp import configure_otlp_exporter

configure_otlp_exporter(
    endpoint="https://api.honeycomb.io/v1/traces",
    headers={"X-Honeycomb-Team": os.getenv("HONEYCOMB_API_KEY")},
    sampling_rate=0.1  # Sample 10% of traces
)
```

### Issue: "Slow query performance"

**Symptoms:**
- Queries taking >5 seconds
- Timeouts on complex aggregations

**Solution:**
1. **Add time bounds**: Always filter by time range
   ```
   WHERE timestamp > ago(1h)
   ```

2. **Limit grouping dimensions**: Start with 1-2 GROUP BY fields
   ```
   GROUP BY genops.team  # Good
   GROUP BY genops.team, genops.project, genops.customer_id  # May be slow
   ```

3. **Use derived columns**: Pre-compute frequently queried aggregations
4. **Consider SLOs**: Define and track specific SLOs instead of ad-hoc queries

---

## 🚀 Next Steps

### Advanced Features

**1. Set Up Triggers for Budget Alerts**
- Alert when cost exceeds threshold
- Notify on policy violations
- Monitor token usage spikes

**2. Create SLOs for Governance**
- Policy compliance rate (target: 99.9%)
- Cost-per-request budget adherence
- Token efficiency targets

**3. Use Derived Columns**
- `cost_per_token = genops.cost.total / genops.tokens.total`
- `budget_utilization = genops.budget.consumed / genops.budget.limit * 100`

**4. Enable Markers for Deployments**
- Track cost changes after deployments
- Correlate performance with releases

### Production Deployment

For production-grade setup with Kubernetes, OTel Collector, and advanced features, see:

📘 **[Comprehensive Honeycomb Integration Guide](integrations/honeycomb.md)**

Topics covered:
- OpenTelemetry Collector configuration
- Kubernetes deployment patterns
- High-volume sampling strategies
- Multi-environment setup (dev/staging/prod)
- Derived columns for governance metrics
- Triggers and SLOs for AI operations
- Cost optimization best practices
- Enterprise SSO and RBAC integration

### Multi-Provider Tracking

Track costs across multiple AI providers simultaneously:

```python
from genops import auto_instrument

# Enable multiple providers
auto_instrument(providers=["openai", "anthropic", "bedrock"])

# All providers flow to same Honeycomb dataset
# Query with: GROUP BY genops.cost.provider
```

### Framework Integration

GenOps works with popular AI frameworks:

- **LangChain**: Automatic chain and agent tracking
- **LlamaIndex**: RAG pipeline instrumentation
- **OpenAI**: Direct API instrumentation
- **Anthropic**: Claude API monitoring
- **AWS Bedrock**: Multi-model governance

See framework-specific guides in the [documentation](../README.md).

---

## 🍯 Honeycomb-Specific Advantages

Honeycomb is uniquely suited for AI governance telemetry:

### 1. High-Cardinality Excellence
- **Traditional APM**: Struggles with high-cardinality dimensions (customer_id, transaction_id)
- **Honeycomb**: Designed for unlimited cardinality
- **GenOps + Honeycomb**: Perfect match for per-customer, per-feature cost tracking

### 2. BubbleUp for Cost Analysis
- Automatically surface attributes that correlate with high costs
- Identify which customers, features, or models drive spend
- No manual query construction needed

### 3. Fast Iterative Exploration
- Sub-second query responses even on high-volume data
- Explore cost patterns interactively
- Quickly answer "why did costs spike?" questions

### 4. Distributed Tracing Native
- See full AI operation traces across services
- Track cost attribution through complex workflows
- Correlate performance with cost

### 5. Real-Time Governance
- Query current operations in real-time
- No aggregation delays
- Immediate budget enforcement feedback

---

## 💡 Example Use Cases

### Use Case 1: Per-Customer Cost Tracking

**Scenario:** SaaS platform needs to track AI costs per customer for accurate billing.

**Setup:**
```python
from genops.core.context import set_governance_context

# For each customer request
set_governance_context({
    "customer_id": request.customer_id,
    "customer_tier": request.customer.tier,
    "feature": request.feature_name
})

# Run AI operations
response = ai_client.generate(...)
```

**Honeycomb Query:**
```
GROUP BY genops.customer_id, genops.customer_tier
| SUM(genops.cost.total)
| ORDER BY SUM DESC
```

**Result:** Real-time cost breakdown by customer for billing and budget alerts.

### Use Case 2: Model Efficiency Analysis

**Scenario:** Optimize model selection based on cost-performance trade-offs.

**Honeycomb Query:**
```
GROUP BY genops.cost.model
| AVG(duration_ms), AVG(genops.cost.total), COUNT
```

**BubbleUp:** Find which models are most cost-effective for specific use cases.

### Use Case 3: Budget Enforcement

**Scenario:** Prevent cost overruns by enforcing team budgets.

**Setup:**
```python
from genops.core.budget import set_budget_limit

set_budget_limit(team="ai-engineering", limit_usd=1000.0, period="daily")
```

**Honeycomb Trigger:**
- Alert when: `SUM(genops.cost.total WHERE genops.team = "ai-engineering") > 900`
- Action: Send Slack notification, page on-call engineer

---

## 📚 Additional Resources

- **[Honeycomb Documentation](https://docs.honeycomb.io/)** - Official Honeycomb docs
- **[OpenTelemetry Tracing](https://opentelemetry.io/docs/concepts/signals/traces/)** - OTel tracing concepts
- **[GenOps GitHub](https://github.com/KoshiHQ/GenOps-AI)** - Source code and examples
- **[Comprehensive Integration Guide](integrations/honeycomb.md)** - Advanced Honeycomb setup

---

## 💬 Get Help

- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
