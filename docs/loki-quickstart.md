# Loki Quickstart

Get GenOps AI governance logs flowing to Loki in under 5 minutes.

## 🚀 Quick Setup (5 Minutes)

### Option A: Docker Compose with Complete LGTM Stack (Recommended)

The fastest way to get started with Loki is using the complete observability stack:

```bash
# Clone the repository
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI

# Start the complete LGTM stack (Loki, Grafana, Tempo, Mimir)
docker-compose -f docker-compose.observability.yml up -d

# Verify Loki is running
curl http://localhost:3100/ready
```

**That's it!** You now have:
- ✅ **Loki** running at http://localhost:3100
- ✅ **Grafana** with data sources at http://localhost:3000 (admin/genops)
- ✅ **OTel Collector** ready to receive telemetry
- ✅ **Complete LGTM stack** for unified observability

### Option B: Standalone Loki with Docker

For a minimal Loki-only setup:

```bash
# Download Loki configuration
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/observability/loki-config.yaml

# Start Loki
docker run -d \
  --name loki \
  -p 3100:3100 \
  -v $(pwd)/loki-config.yaml:/etc/loki/local-config.yaml \
  grafana/loki:2.9.4 \
  -config.file=/etc/loki/local-config.yaml
```

---

## 📝 Configure GenOps to Export Logs

### 1. Install GenOps with OpenTelemetry Support

```bash
pip install genops-ai[opentelemetry]
```

### 2. Configure OTLP Export to Loki

```python
from genops.exporters.otlp import configure_otlp_exporter

# Configure OTLP endpoint (OTel Collector from Docker Compose)
configure_otlp_exporter(
    endpoint="http://localhost:4318",  # OTel Collector endpoint
    service_name="my-ai-app"
)
```

**Note:** The OTel Collector (included in Docker Compose) routes logs to Loki automatically.

### 3. Enable Auto-Instrumentation (Zero Code Changes)

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
# ✅ Logs automatically exported to Loki!
```

---

## 🔍 View Logs in Grafana

### 1. Access Grafana

Open http://localhost:3000 and log in with:
- **Username:** admin
- **Password:** genops

### 2. Navigate to Explore

1. Click **Explore** (compass icon) in the left sidebar
2. Select **Loki** as the data source
3. Use LogQL queries to explore your logs

### 3. Basic LogQL Queries

**All logs for your service:**
```logql
{service_name="my-ai-app"}
```

**Logs containing "cost":**
```logql
{service_name="my-ai-app"} |= "cost"
```

**Error logs only:**
```logql
{service_name="my-ai-app"} |= "error"
```

**Logs for specific team:**
```logql
{service_name="my-ai-app", team="ai-engineering"}
```

**Logs for production environment:**
```logql
{service_name="my-ai-app", environment="production"}
```

---

## 💰 Add Cost Attribution (30 Seconds)

Track logs by team, project, or customer with governance attributes:

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

# All AI operations now include attribution tags in Loki
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze customer feedback"}]
)
```

**Query logs with attribution in Grafana:**

```logql
# Logs for specific customer
{service_name="my-ai-app", customer_id="enterprise_123"}

# Logs by team
{service_name="my-ai-app", team="ai-engineering"}

# Production logs only
{service_name="my-ai-app", environment="production"}
```

---

## 📊 Essential LogQL Queries

### Log Filtering

**All logs for service:**
```logql
{service_name="my-ai-app"}
```

**Logs NOT containing "health" (filter noise):**
```logql
{service_name="my-ai-app"} != "health"
```

**Case-insensitive search:**
```logql
{service_name="my-ai-app"} |= `(?i)error`
```

### Trace Correlation

**Logs for specific trace ID:**
```logql
{service_name="my-ai-app"} |= "trace_id=abc123def456"
```

**Logs with any trace ID:**
```logql
{service_name="my-ai-app"} | json | trace_id != ""
```

**Logs for expensive operations:**
```logql
{service_name="my-ai-app"} | json | cost_total_usd > 0.10
```

### Error Analysis

**Error log count (last hour):**
```logql
count_over_time({service_name="my-ai-app"} |= "error" [1h])
```

**Error logs by level:**
```logql
{service_name="my-ai-app"} | json | level="ERROR"
```

**Rate of error logs:**
```logql
rate({service_name="my-ai-app"} |= "error" [5m])
```

**Top 10 error messages:**
```logql
topk(10, sum by (error_message) (count_over_time({service_name="my-ai-app"} | json [1h])))
```

### Policy and Compliance

**Policy violation logs:**
```logql
{service_name="my-ai-app"} | json | policy_result="blocked"
```

**Policy violation count:**
```logql
count_over_time({service_name="my-ai-app"} | json | policy_result="blocked" [1h])
```

---

## ✅ Validate Your Setup

Check that logs are flowing correctly:

```bash
# Check Loki is ready
curl http://localhost:3100/ready

# Query logs via Loki API
curl -G -s "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={service_name="my-ai-app"}' \
  --data-urlencode 'limit=10' | jq
```

**Expected response:**
```json
{
  "status": "success",
  "data": {
    "resultType": "streams",
    "result": [
      {
        "stream": {
          "service_name": "my-ai-app",
          "team": "ai-engineering"
        },
        "values": [
          ["1234567890000000000", "AI operation completed with cost: 0.0025 USD"]
        ]
      }
    ]
  }
}
```

---

## ⚠️ Troubleshooting

### Issue: "No logs appearing in Loki"

**Check:**
1. **Loki is running:**
   ```bash
   curl http://localhost:3100/ready
   # Should return: "ready"
   ```

2. **OTel Collector is running:**
   ```bash
   docker ps | grep otel-collector
   ```

3. **OTel Collector logs:**
   ```bash
   docker logs genops-otel-collector
   ```

4. **Verify configuration:**
   ```python
   from genops.exporters.otlp import get_current_configuration

   config = get_current_configuration()
   print(f"Endpoint: {config['endpoint']}")
   print(f"Service: {config['service_name']}")
   ```

**Solution:**
- Ensure Docker Compose stack is running: `docker-compose -f docker-compose.observability.yml ps`
- Check OTel Collector configuration routes logs to Loki
- Verify network connectivity between containers

### Issue: "Loki returns empty query results"

**Problem:** Queries return no results even though Loki is running

**Solutions:**

1. **Check time range in Grafana:**
   - Grafana queries default to recent time ranges
   - Expand time range to "Last 1 hour" or "Last 6 hours"

2. **Verify label names:**
   ```logql
   # List all labels
   {job="genops-ai"}
   ```

3. **Check log ingestion:**
   ```bash
   # Query Loki metrics
   curl http://localhost:3100/metrics | grep loki_ingester
   ```

### Issue: "High query latency"

**Symptoms:** Queries taking >5 seconds

**Solutions:**

1. **Add time bounds:**
   ```logql
   {service_name="my-ai-app"} [1h]  # Limit to last hour
   ```

2. **Use specific labels:**
   ```logql
   # Good - uses indexed labels
   {service_name="my-ai-app", team="ai-engineering"}

   # Slow - filters in content
   {service_name="my-ai-app"} |= "ai-engineering"
   ```

3. **Configure retention:**
   - Check `loki-config.yaml` retention settings
   - Reduce retention period if needed

### Issue: "Port 3100 already in use"

**Problem:** `Address already in use` error

**Solution:**

```bash
# Find process using port 3100
lsof -i :3100

# Stop conflicting Loki instance
docker stop loki

# Or change port in docker-compose.observability.yml
ports:
  - "3101:3100"  # Map to different local port
```

---

## 🚀 Next Steps

### Production Deployment

For production-grade setup with Kubernetes, retention policies, and high availability, see:

📘 **[Comprehensive Grafana Integration Guide](integrations/grafana.md)**

Topics covered:
- Loki retention policies and storage configuration
- Kubernetes deployment with Helm charts
- High-availability setup with distributed mode
- Log aggregation at scale
- Integration with Tempo for trace correlation
- Multi-tenant configuration
- Authentication and access control
- Performance tuning and optimization

### Advanced LogQL

**Learn advanced query patterns:**
- [LogQL Query Examples](grafana-query-examples.md#logql-logs) - Complete query reference
- [Multi-Data Source Queries](grafana-query-examples.md#multi-data-source-queries) - Correlate logs with traces and metrics

### Complete Observability Stack

**Explore the full LGTM stack:**
- **[Grafana Quickstart](grafana-quickstart.md)** - Unified observability platform
- **[Tempo Integration](integrations/grafana.md#tempo-traces)** - Distributed tracing
- **[Prometheus/Mimir Integration](prometheus-quickstart.md)** - Metrics and alerting

### Multi-Provider Tracking

Track logs across multiple AI providers:

```python
from genops import auto_instrument

# Enable multiple providers
auto_instrument(providers=["openai", "anthropic", "bedrock"])

# All providers flow to same Loki instance
# Query with: {service_name="my-ai-app"} | json | provider="openai"
```

### Framework Integration

GenOps works with popular AI frameworks:

- **LangChain**: Automatic chain and agent logging
- **LlamaIndex**: RAG pipeline log aggregation
- **OpenAI**: Direct API log capture
- **Anthropic**: Claude API logging
- **AWS Bedrock**: Multi-model log aggregation

See framework-specific guides in the [documentation](../README.md#ai--llm-ecosystem).

---

## 📚 Additional Resources

- **[Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)** - Official Loki docs
- **[LogQL Query Language](https://grafana.com/docs/loki/latest/query/)** - LogQL reference
- **[GenOps GitHub](https://github.com/KoshiHQ/GenOps-AI)** - Source code and examples
- **[Comprehensive Integration Guide](integrations/grafana.md)** - Advanced Loki setup
- **[OpenTelemetry Logging](https://opentelemetry.io/docs/concepts/signals/logs/)** - OTel logging concepts

---

## 💬 Get Help

- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
