# Grafana Tempo Quickstart (2-5 Minutes)

Get GenOps AI tracing with Grafana Tempo in 2-5 minutes. Choose your path based on your needs:

- **Path A: Standalone Tempo (2 Minutes)** ⚡ - Fastest path for seeing traces immediately
- **Path B: With OTel Collector (5 Minutes)** - Production-recommended architecture
- **Path C: Full LGTM Stack (10 Minutes)** - Complete observability with Grafana, Loki, Tempo, and Mimir

---

## Path A: Standalone Tempo (2 Minutes) ⚡

**Perfect for:** Quick testing, learning, immediate trace visualization

### 1. Start Tempo (30 seconds)

```bash
docker run -d --name tempo \
  -p 3200:3200 \
  -p 4318:4318 \
  grafana/tempo:latest
```

This starts Tempo with:
- HTTP API on port 3200 (status, search, TraceQL)
- OTLP HTTP receiver on port 4318 (trace ingestion)

### 2. Install GenOps AI (30 seconds)

```bash
pip install genops-ai
```

### 3. Configure and Validate (30 seconds)

```python
from genops.integrations.tempo import quick_start, validate_tempo_setup, print_tempo_validation

# Zero-configuration setup
quick_start()

# Verify everything works
result = validate_tempo_setup()
print_tempo_validation(result)
```

**Expected output:**
```
✅ GenOps configured for Tempo at http://localhost:3200
   View traces in Grafana at http://localhost:3000 (if using LGTM stack)

✅ Grafana Tempo Setup Validation - PASSED

Endpoint: http://localhost:3200

Status Checks:
  ✅ Tempo Accessibility: Tempo accessible at http://localhost:3200
  ✅ Tempo Version: Version 2.3.0
  ✅ TraceQL API: TraceQL queries available
  ✅ Search API: Search API available
  ✅ OTLP Write Endpoint: OTLP receiver available
  ✅ Query Capability: Can query spans via Search API

✅ All checks passed! Tempo is ready for trace ingestion.
   Tempo version: 2.3.0
   TraceQL queries: Enabled ✨
```

### 4. Send Your First Trace (30 seconds)

```python
from genops import track_usage

@track_usage(
    team="my-team",
    project="quickstart",
    customer_id="demo-customer"
)
def ask_question():
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is OpenTelemetry?"}]
    )

    return response.choices[0].message.content

# This creates a trace in Tempo!
answer = ask_question()
print(answer)
```

### 5. View Traces (30 seconds)

**Option 1: Query via TraceQL (command line)**

```bash
# Search for recent traces
curl "http://localhost:3200/api/search?q={}&limit=10"

# Search by team attribute
curl "http://localhost:3200/api/search?q={.team=\"my-team\"}&limit=10"
```

**Option 2: Use Grafana (next section)**

You now have GenOps AI traces flowing to Tempo! 🎉

---

## Path B: With OTel Collector (5 Minutes)

**Perfect for:** Production deployments, advanced processing, sampling, multi-backend export

The OpenTelemetry Collector provides:
- Centralized trace processing and routing
- Sampling configuration
- Multi-backend export (Tempo + Honeycomb + Datadog simultaneously)
- Resource attribute enrichment
- Rate limiting and batching

### 1. Create OTel Collector Configuration (1 minute)

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
  batch:
    timeout: 10s
    send_batch_size: 1024

  # Add resource attributes
  resource:
    attributes:
      - key: deployment.environment
        value: ${ENVIRONMENT}
        action: upsert

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

  # Optional: Enable debug logging
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, resource]
      exporters: [otlp/tempo, logging]
```

### 2. Start Services (1 minute)

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  tempo:
    image: grafana/tempo:latest
    ports:
      - "3200:3200"  # HTTP API
      - "4317:4317"  # OTLP gRPC
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo-config.yaml:/etc/tempo.yaml

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4318:4318"  # OTLP HTTP
      - "4317:4317"  # OTLP gRPC
    environment:
      - ENVIRONMENT=development
    depends_on:
      - tempo
```

Create minimal `tempo-config.yaml`:

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

storage:
  trace:
    backend: local
    local:
      path: /tmp/tempo/traces

compactor:
  compaction:
    block_retention: 1h
```

Start everything:

```bash
docker-compose up -d
```

### 3. Install and Configure GenOps AI (1 minute)

```bash
pip install genops-ai
```

```python
from genops.integrations.tempo import configure_tempo, validate_tempo_setup, print_tempo_validation

# Configure to use OTel Collector
configure_tempo(
    via_collector=True,
    collector_endpoint="http://localhost:4318",
    service_name="my-ai-service",
    environment="development"
)

# Validate the setup
result = validate_tempo_setup(tempo_endpoint="http://localhost:3200")
print_tempo_validation(result)
```

### 4. Send Traces (1 minute)

```python
from genops import track_usage

@track_usage(
    team="platform-team",
    project="ai-assistant",
    customer_id="enterprise-customer-123"
)
def generate_response(question):
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )

    return response.choices[0].message.content

# Traces flow: App → OTel Collector → Tempo
answer = generate_response("Explain observability in AI systems")
print(answer)
```

### 5. Query Traces (1 minute)

The OTel Collector enriches traces with additional context before sending to Tempo.

```bash
# Query by environment (added by collector)
curl "http://localhost:3200/api/search?q={.deployment.environment=\"development\"}&limit=10"

# Query by team
curl "http://localhost:3200/api/search?q={.team=\"platform-team\"}&limit=10"

# Query by customer
curl "http://localhost:3200/api/search?q={.customer_id=\"enterprise-customer-123\"}&limit=10"
```

You now have production-ready trace collection with centralized processing! 🚀

---

## Path C: Full LGTM Stack (10 Minutes)

**Perfect for:** Complete observability solution with logs, metrics, and traces in Grafana

The LGTM stack provides:
- **Loki**: Log aggregation and querying
- **Grafana**: Unified visualization dashboard
- **Tempo**: Distributed tracing (what we've been setting up)
- **Mimir**: Prometheus-compatible metrics

For the complete LGTM stack setup with Grafana dashboards, see:

📖 **[Grafana Quickstart Guide](./grafana-quickstart.md)**

The Grafana quickstart includes:
- Pre-configured Grafana dashboards for GenOps AI
- Log correlation with traces
- Cost tracking visualizations
- Multi-provider comparison charts

---

## Viewing Traces in Grafana

If you want to visualize traces from Path A or B, add Grafana:

### Quick Grafana Setup (2 minutes)

```bash
docker run -d --name grafana \
  -p 3000:3000 \
  -e "GF_AUTH_ANONYMOUS_ENABLED=true" \
  -e "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin" \
  grafana/grafana:latest
```

### Add Tempo Data Source

1. Open Grafana at http://localhost:3000
2. Go to **Configuration** → **Data Sources** → **Add data source**
3. Select **Tempo**
4. Configure:
   - **URL**: `http://host.docker.internal:3200` (Mac/Windows) or `http://172.17.0.1:3200` (Linux)
   - Click **Save & Test**

### Explore Traces

1. Go to **Explore** (compass icon in sidebar)
2. Select **Tempo** data source
3. Use **Search** tab to find traces by:
   - Service name: `genops-ai`
   - Tags: `team=my-team`, `customer_id=demo-customer`

4. Use **TraceQL** tab for advanced queries:
   ```traceql
   # Find expensive traces (>1s duration)
   {duration > 1s}

   # Find traces for specific team
   {.team = "my-team"}

   # Find traces with high token usage
   {.genops.cost.total_tokens > 1000}

   # Complex query: expensive traces for specific customer
   {duration > 500ms && .customer_id = "enterprise-customer-123"}
   ```

---

## Validation and Troubleshooting

### Run Comprehensive Validation

```python
from genops.integrations.tempo import validate_tempo_setup, print_tempo_validation

result = validate_tempo_setup(
    tempo_endpoint="http://localhost:3200",
    check_connectivity=True,
    check_write=True,
    check_read=True,
    check_traceql=True
)

print_tempo_validation(result)
```

### Common Issues

#### Issue: "Cannot connect to Tempo at http://localhost:3200"

**Fix:**
```bash
# Check if Tempo is running
docker ps | grep tempo

# If not running, start it
docker run -d --name tempo \
  -p 3200:3200 \
  -p 4318:4318 \
  grafana/tempo:latest

# Check Tempo logs
docker logs tempo
```

#### Issue: "OTLP endpoint not accessible"

**Fix:**
```bash
# Ensure port 4318 is exposed
docker run -d --name tempo \
  -p 3200:3200 \
  -p 4318:4318 \
  grafana/tempo:latest

# Test OTLP endpoint
curl http://localhost:4318/v1/traces
```

#### Issue: "TraceQL API not available (404)"

**Cause:** You're running Tempo version < 2.0

**Fix:**
```bash
# Upgrade to Tempo 2.0+
docker stop tempo && docker rm tempo
docker run -d --name tempo \
  -p 3200:3200 \
  -p 4318:4318 \
  grafana/tempo:latest
```

#### Issue: No traces appearing in Tempo

**Debug steps:**

1. **Verify GenOps configuration:**
   ```python
   from genops.integrations.tempo import validate_tempo_setup, print_tempo_validation
   result = validate_tempo_setup()
   print_tempo_validation(result)
   ```

2. **Check if traces are being exported:**
   ```python
   import os
   os.environ["OTEL_LOG_LEVEL"] = "debug"
   # Re-run your code and check logs
   ```

3. **Query Tempo directly:**
   ```bash
   # List recent traces
   curl "http://localhost:3200/api/search?q={}&limit=10" | jq
   ```

4. **Check OTel Collector logs** (if using Path B):
   ```bash
   docker logs otel-collector
   ```

---

## Configuration Reference

### Environment Variables

GenOps AI respects standard OpenTelemetry environment variables:

```bash
# Service identification
export OTEL_SERVICE_NAME="my-ai-service"
export ENVIRONMENT="production"

# OTLP endpoint (overrides defaults)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://tempo:4318"

# Tempo-specific
export TEMPO_ENDPOINT="http://localhost:3200"
export TEMPO_AUTH_HEADER="Bearer your-token"  # For authenticated deployments
```

### Programmatic Configuration

```python
from genops.integrations.tempo import configure_tempo

# Development: Direct to Tempo
configure_tempo(
    endpoint="http://localhost:3200",
    service_name="ai-dev-service",
    environment="development"
)

# Production: Via OTel Collector
configure_tempo(
    via_collector=True,
    collector_endpoint="http://otel-collector:4318",
    service_name="ai-prod-service",
    environment="production",
    sampling_rate=0.1  # 10% sampling for high-volume production
)

# Multi-tenant: Grafana Cloud Tempo
configure_tempo(
    endpoint="https://tempo-us-central1.grafana.net",
    tenant_id="123456",  # Your Grafana Cloud instance ID
    service_name="ai-service",
    environment="production"
)
```

---

## Next Steps

### 1. Add Governance Attributes

Enrich your traces with GenOps governance attributes:

```python
from genops import track_usage

@track_usage(
    team="ml-platform",           # Cost attribution
    project="customer-support",   # Project tracking
    customer_id="acme-corp",      # Customer attribution
    environment="production",     # Environment segregation
    cost_center="engineering",    # Financial reporting
    feature="ai-assistant"        # Feature-level tracking
)
def handle_customer_query(query):
    # Your AI logic here
    pass
```

### 2. Query Traces by Governance Attributes

Use TraceQL to slice traces by governance dimensions:

```traceql
# Cost by team
{.team = "ml-platform"} | rate() by (.team)

# Customer usage patterns
{.customer_id = "acme-corp"}

# Feature cost analysis
{.feature = "ai-assistant"} | sum(.genops.cost.total_cost) by (.feature)

# Environment-specific traces
{.environment = "production" && duration > 1s}
```

### 3. Set Up Grafana Dashboards

See the **[Grafana Integration Guide](./integrations/grafana.md)** for:
- Pre-built GenOps AI dashboards
- Cost tracking visualizations
- Multi-provider comparison charts
- Trace-to-log correlation

### 4. Explore Advanced Features

- **[Tempo Integration Guide](./integrations/tempo.md)** - Deep dive into Tempo features
- **[Multi-Provider Cost Tracking](./guides/multi-provider-cost-tracking.md)** - Track costs across OpenAI, Anthropic, Bedrock, etc.
- **[Cost Optimization with Tempo](./integrations/tempo.md#cost-attribution)** - Analyze and reduce AI costs using TraceQL
- **[Production Deployment Patterns](./integrations/tempo.md#production-deployment)** - Scale Tempo for production workloads

---

## Performance Considerations

### Sampling Strategies

For high-volume production workloads, use sampling to reduce overhead:

```python
from genops.integrations.tempo import configure_tempo

# Sample 10% of traces
configure_tempo(
    via_collector=True,
    sampling_rate=0.1
)

# Or use head-based sampling in OTel Collector
# See: otel-collector-config.yaml processors section
```

### Resource Requirements

**Tempo (standalone):**
- CPU: 0.5-1 core
- Memory: 512MB-1GB
- Storage: 1GB per 1M spans (with 1h retention)

**With OTel Collector:**
- Add: 0.5 core CPU, 512MB memory
- Benefits: Better batching, sampling, multi-backend export

**Full LGTM Stack:**
- See [Grafana Quickstart](./grafana-quickstart.md) for complete resource requirements

---

## Additional Resources

- **[Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)** - Official Tempo docs
- **[TraceQL Query Language](https://grafana.com/docs/tempo/latest/traceql/)** - Query language reference
- **[OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)** - OTLP protocol specs
- **[GenOps AI Examples](../examples/tempo/)** - Code examples for Tempo integration

---

## Support

- **GitHub Issues**: [GenOps AI Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Documentation**: [Full Integration Guide](./integrations/tempo.md)
- **Community**: [GenOps AI Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**You're now ready to track AI governance telemetry with Grafana Tempo!** 🎉

Continue with the [Grafana Integration Guide](./integrations/grafana.md) to add visualization and dashboards.
