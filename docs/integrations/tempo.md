## Grafana Tempo Integration Guide

Comprehensive guide for integrating GenOps AI with Grafana Tempo for distributed tracing and governance telemetry.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture Patterns](#architecture-patterns)
4. [Configuration](#configuration)
5. [Validation & Troubleshooting](#validation--troubleshooting)
6. [TraceQL Queries](#traceql-queries)
7. [Cost Attribution](#cost-attribution)
8. [Multi-Tenancy](#multi-tenancy)
9. [Production Deployment](#production-deployment)
10. [Performance Optimization](#performance-optimization)
11. [Security](#security)
12. [Examples](#examples)

---

## Overview

### What is Grafana Tempo?

Grafana Tempo is an open-source, high-scale distributed tracing backend designed for:
- **Cost-effective trace storage** using object storage (S3, GCS, Azure)
- **TraceQL** - powerful query language for trace analysis
- **OpenTelemetry native** - full OTLP support
- **Grafana integration** - seamless visualization

### Why Tempo for GenOps AI?

1. **Cost-Effective Governance Tracking**
   - Store millions of traces with governance attributes
   - Low-cost object storage backend
   - Efficient compression and compaction

2. **Powerful TraceQL Queries**
   - Query by team, customer, project
   - Cost aggregation and analysis
   - Performance investigation

3. **Production-Ready**
   - High availability and scalability
   - Multi-tenancy support
   - Cloud-native architecture

4. **Open Source & Vendor Neutral**
   - No vendor lock-in
   - Standard OpenTelemetry protocol
   - Self-hosted or cloud options

---

## Quick Start

For fastest path to working traces, see **[Tempo Quickstart](../tempo-quickstart.md)**.

**2-minute quick start:**

```bash
# Start Tempo
docker run -d -p 3200:3200 -p 4318:4318 grafana/tempo:latest

# Install and configure GenOps
pip install genops-ai

python -c "
from genops.integrations.tempo import quick_start
quick_start()
"
```

---

## Architecture Patterns

### Pattern 1: Direct Export (Development)

```
┌─────────────┐
│  GenOps AI  │──OTLP/HTTP──┐
└─────────────┘              │
                             ↓
                      ┌──────────────┐
                      │    Tempo     │
                      └──────────────┘
```

**Use when:**
- Local development
- Simple deployments
- Single service

**Configuration:**
```python
from genops.integrations.tempo import configure_tempo

configure_tempo(endpoint="http://localhost:3200")
```

---

### Pattern 2: Via OTel Collector (Recommended)

```
┌─────────────┐           ┌──────────────────┐
│  GenOps AI  │──OTLP────▶│ OTel Collector   │
└─────────────┘           │  - Batching      │
                          │  - Sampling      │──▶ Tempo
                          │  - Processing    │
                          └──────────────────┘
```

**Use when:**
- Production deployments
- Multiple services
- Need sampling/processing
- Multi-backend export

**Configuration:**
```python
from genops.integrations.tempo import configure_tempo

configure_tempo(
    via_collector=True,
    collector_endpoint="http://otel-collector:4318"
)
```

**OTel Collector Config:**
```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

  # Add resource attributes
  resource:
    attributes:
      - key: deployment.environment
        value: production
        action: upsert

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, resource]
      exporters: [otlp/tempo]
```

---

### Pattern 3: LGTM Stack (Complete Observability)

```
┌─────────────┐           ┌──────────────────┐
│  GenOps AI  │──OTLP────▶│ OTel Collector   │
└─────────────┘           └─────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              ┌─────────┐     ┌─────────┐    ┌────────┐
              │  Loki   │     │  Tempo  │    │ Mimir  │
              │ (Logs)  │     │(Traces) │    │(Metrics)│
              └─────────┘     └─────────┘    └────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ↓
                              ┌──────────┐
                              │  Grafana │
                              └──────────┘
```

**Use when:**
- Need logs, metrics, and traces
- Unified observability platform
- Correlation across signals

**See:** [Grafana Quickstart](../grafana-quickstart.md) for full LGTM stack setup.

---

## Configuration

### Python API

#### Basic Configuration

```python
from genops.integrations.tempo import configure_tempo

# Default localhost
configure_tempo()

# Custom endpoint
configure_tempo(endpoint="http://tempo.example.com:3200")

# Via OTel Collector
configure_tempo(
    via_collector=True,
    collector_endpoint="http://otel-collector:4318"
)
```

#### Multi-Tenancy

```python
configure_tempo(
    endpoint="https://tempo.grafana.net",
    tenant_id="team-platform",
    service_name="ai-service",
    environment="production"
)
```

#### Sampling

```python
# Sample 10% of traces
configure_tempo(
    endpoint="http://localhost:3200",
    sampling_rate=0.1
)
```

#### Complete Options

```python
configure_tempo(
    endpoint="http://tempo:3200",        # Tempo endpoint
    via_collector=False,                  # Route via collector
    collector_endpoint=None,              # Collector endpoint
    tenant_id=None,                       # X-Scope-OrgID header
    service_name="genops-ai",            # Service name
    environment=None,                     # Environment
    sampling_rate=1.0,                    # Sampling (0.0-1.0)
    headers={}                            # Custom headers
)
```

### Environment Variables

GenOps respects standard OpenTelemetry variables:

```bash
# Service identification
export OTEL_SERVICE_NAME="my-ai-service"
export ENVIRONMENT="production"

# OTLP endpoint (overrides configure_tempo)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://tempo:4318"

# Tempo-specific
export TEMPO_ENDPOINT="http://localhost:3200"
export TEMPO_AUTH_HEADER="Bearer token"
```

### Tempo Configuration

**Minimal `tempo.yaml`:**

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        http:
          endpoint: 0.0.0.0:4318
        grpc:
          endpoint: 0.0.0.0:4317

storage:
  trace:
    backend: local
    local:
      path: /tmp/tempo/traces

compactor:
  compaction:
    block_retention: 24h
```

**Production `tempo.yaml`:**

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        http:
          endpoint: 0.0.0.0:4318
        grpc:
          endpoint: 0.0.0.0:4317

  # Rate limiting
  rate_limit:
    traces_per_second: 1000

storage:
  trace:
    backend: s3
    s3:
      bucket: tempo-traces
      endpoint: s3.amazonaws.com
      region: us-east-1

    # Write-ahead log
    wal:
      path: /var/tempo/wal

    # Search & TraceQL
    cache: memcached
    memcached:
      consistent_hash: true
      host: memcached:11211

compactor:
  compaction:
    block_retention: 168h  # 7 days
    compacted_block_retention: 24h

querier:
  max_concurrent_queries: 20
  search:
    max_duration: 24h

# Multi-tenancy
multitenancy_enabled: true
multitenancy_tenant_header: X-Scope-OrgID
```

---

## Validation & Troubleshooting

### Comprehensive Validation

```python
from genops.integrations.tempo import validate_tempo_setup, print_tempo_validation

result = validate_tempo_setup(
    tempo_endpoint="http://localhost:3200",
    check_connectivity=True,
    check_write=True,
    check_read=True,
    check_traceql=True,
    timeout=5
)

print_tempo_validation(result)
```

**Output:**
```
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

### Common Issues & Fixes

#### Issue: Connection Refused

```
❌ Cannot connect to Tempo at http://localhost:3200
   Fix: Start Tempo:
     Docker: docker run -d -p 3200:3200 -p 4318:4318 grafana/tempo:latest
     Or check if Tempo is running: curl http://localhost:3200/status/buildinfo
```

**Solution:**
```bash
# Check if Tempo is running
docker ps | grep tempo

# Start Tempo if not running
docker run -d --name tempo \
  -p 3200:3200 -p 4318:4318 \
  grafana/tempo:latest

# Verify
curl http://localhost:3200/status/buildinfo
```

#### Issue: OTLP Endpoint Not Accessible

```
❌ OTLP endpoint not accessible
   Fix: Enable OTLP receiver in Tempo config (default port 4318)
```

**Solution:**
Check `tempo.yaml`:
```yaml
distributor:
  receivers:
    otlp:
      protocols:
        http:
          endpoint: 0.0.0.0:4318  # Ensure this is present
```

#### Issue: TraceQL Not Available

```
❌ TraceQL API not available (404)
   Fix: TraceQL requires Tempo 2.0+. Upgrade Tempo version.
```

**Solution:**
```bash
# Check Tempo version
curl http://localhost:3200/status/buildinfo | jq '.version'

# Upgrade to 2.0+
docker pull grafana/tempo:latest
docker stop tempo && docker rm tempo
docker run -d --name tempo \
  -p 3200:3200 -p 4318:4318 \
  grafana/tempo:latest
```

#### Issue: No Traces Appearing

**Debug steps:**

1. **Verify configuration:**
   ```python
   from genops.integrations.tempo import validate_tempo_setup, print_tempo_validation
   result = validate_tempo_setup()
   print_tempo_validation(result)
   ```

2. **Enable debug logging:**
   ```python
   import os
   os.environ["OTEL_LOG_LEVEL"] = "debug"
   # Re-run your code
   ```

3. **Query Tempo directly:**
   ```bash
   curl "http://localhost:3200/api/search?q={}&limit=10" | jq
   ```

4. **Check OTel Collector logs** (if using collector):
   ```bash
   docker logs otel-collector
   ```

---

## TraceQL Queries

TraceQL is Tempo's powerful query language for trace analysis.

### Basic Syntax

```traceql
# All traces
{}

# Filter by attribute
{.team = "platform"}

# Filter by duration
{duration > 1s}

# Complex conditions
{duration > 500ms && .cost > 0.05}
```

### GenOps Governance Queries

#### Team Attribution

```traceql
# Traces for specific team
{.team = "ml-platform"}

# Multiple teams
{.team = "platform" || .team = "research"}

# Team cost aggregation (via curl)
curl 'http://localhost:3200/api/search?q={.team="platform"}&limit=100' | \
  jq '[.traces[].spans[].attributes[] | select(.key=="genops.cost.total_cost")] | add'
```

#### Customer Tracking

```traceql
# Specific customer
{.customer_id = "acme-corp"}

# High-value customers (many traces)
{.customer_id != ""} | rate() by (.customer_id)

# Customer cost
{.customer_id = "acme-corp"} | sum(.genops.cost.total_cost)
```

#### Cost Analysis

```traceql
# High cost operations
{.genops.cost.total_cost > 0.10}

# Cost by provider
{.genops.provider = "openai"}
{.genops.provider = "anthropic"}

# Expensive slow operations
{duration > 1s && .genops.cost.total_cost > 0.05}

# Token usage
{.genops.cost.total_tokens > 2000}

# Cost aggregation by team
{} | sum(.genops.cost.total_cost) by (.team)
```

#### Performance Analysis

```traceql
# Slow traces
{duration > 1s}

# P95 latency (approximation)
{} | quantile(.duration, 0.95)

# Error traces
{status = error}

# Traces with exceptions
{.exception.message != ""}
```

### Advanced Queries

#### Cost per Customer

```bash
# Get all customer traces and sum costs
curl 'http://localhost:3200/api/search?q={.customer_id!=""}&limit=1000' | \
  jq 'group_by(.traces[].spans[].attributes[] | select(.key=="customer_id").value) |
      map({customer: .[0], total_cost: [.[].spans[].attributes[] |
      select(.key=="genops.cost.total_cost").value] | add})'
```

#### Budget Utilization

```traceql
# Operations near budget limit
{.genops.budget.utilization_pct > 80}

# Budget exceeded
{.genops.budget.remaining < 0}
```

#### Multi-Dimensional Analysis

```traceql
# Production errors for specific team
{.deployment.environment = "production" && status = error && .team = "platform"}

# Expensive operations in specific region
{.genops.cost.total_cost > 0.05 && .cloud.region = "us-east-1"}
```

### Query Performance Tips

1. **Use attribute indexes** - Tempo indexes span attributes
2. **Limit time range** - Narrower time windows = faster queries
3. **Use specific filters** - More specific queries are faster
4. **Leverage caching** - Repeated queries benefit from query frontend cache

---

## Cost Attribution

GenOps provides comprehensive cost tracking via trace attributes.

### Cost Attributes

```python
from genops import track_usage

@track_usage(
    team="platform",
    project="ai-assistant",
    customer_id="acme-corp",
    cost_center="engineering"
)
def ai_operation():
    # GenOps automatically adds:
    # - genops.cost.total_cost
    # - genops.cost.total_tokens
    # - genops.cost.prompt_tokens
    # - genops.cost.completion_tokens
    # - genops.provider
    # - genops.model
    pass
```

### Cost Queries

See [examples/tempo/cost_attribution.py](../../examples/tempo/cost_attribution.py) for comprehensive cost tracking examples.

**Key patterns:**
- Team cost attribution
- Customer billing
- Multi-provider comparison
- Budget tracking
- Cost center allocation

---

## Multi-Tenancy

Tempo supports multi-tenancy using the `X-Scope-OrgID` header.

### Configuration

**Tempo:**
```yaml
multitenancy_enabled: true
multitenancy_tenant_header: X-Scope-OrgID
```

**GenOps:**
```python
configure_tempo(
    endpoint="https://tempo.example.com",
    tenant_id="team-platform"
)
```

### Tenant Isolation

```bash
# Query specific tenant
curl -H "X-Scope-OrgID: team-platform" \
  "http://tempo:3200/api/search?q={}&limit=10"

# Different tenant (isolated)
curl -H "X-Scope-OrgID: team-research" \
  "http://tempo:3200/api/search?q={}&limit=10"
```

### Use Cases

1. **Team Isolation** - Each team's traces are separate
2. **Customer Data** - Per-customer trace storage
3. **Environment Separation** - Dev/staging/prod isolation
4. **Cost Allocation** - Per-tenant billing

See [examples/tempo/multi_tenant.py](../../examples/tempo/multi_tenant.py) for complete multi-tenancy example.

---

## Production Deployment

### High Availability

**Multi-replica deployment:**

```yaml
# Kubernetes example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tempo-distributor
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: tempo
        image: grafana/tempo:latest
        args:
        - -target=distributor
        - -config.file=/etc/tempo.yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tempo-querier
spec:
  replicas: 3
  # ...
```

### Storage Backend

**S3 (Recommended):**
```yaml
storage:
  trace:
    backend: s3
    s3:
      bucket: tempo-traces
      endpoint: s3.amazonaws.com
      region: us-east-1
      access_key: ${S3_ACCESS_KEY}
      secret_key: ${S3_SECRET_KEY}
```

**GCS:**
```yaml
storage:
  trace:
    backend: gcs
    gcs:
      bucket_name: tempo-traces
      chunk_buffer_size: 10485760
```

**Azure:**
```yaml
storage:
  trace:
    backend: azure
    azure:
      container_name: tempo-traces
      storage_account_name: ${AZURE_STORAGE_ACCOUNT}
      storage_account_key: ${AZURE_STORAGE_KEY}
```

### Retention & Compaction

```yaml
compactor:
  compaction:
    # How long to keep traces
    block_retention: 168h  # 7 days

    # How long to keep compacted blocks
    compacted_block_retention: 24h

    # Compaction workers
    compaction_workers: 10

    # Flush size
    flush_size_bytes: 5242880  # 5MB
```

### Resource Requirements

**Small Deployment (< 1000 spans/sec):**
- Distributor: 2 cores, 4GB RAM
- Ingester: 4 cores, 8GB RAM
- Querier: 2 cores, 4GB RAM
- Compactor: 2 cores, 4GB RAM
- Storage: 50GB/day (1M spans/day)

**Medium Deployment (< 10K spans/sec):**
- Distributor: 4 cores, 8GB RAM × 3 replicas
- Ingester: 8 cores, 16GB RAM × 3 replicas
- Querier: 4 cores, 8GB RAM × 3 replicas
- Compactor: 4 cores, 8GB RAM
- Storage: 500GB/day (10M spans/day)

**Large Deployment (> 10K spans/sec):**
- Consult [Tempo Scaling Documentation](https://grafana.com/docs/tempo/latest/operations/scaling/)

---

## Performance Optimization

### 1. Sampling

```python
# Sample 10% of traces
configure_tempo(sampling_rate=0.1)
```

**Or in OTel Collector:**
```yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 10

service:
  pipelines:
    traces:
      processors: [probabilistic_sampler]
```

### 2. Batching

```python
# Larger batches = fewer exports
from opentelemetry.sdk.trace.export import BatchSpanProcessor

processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    schedule_delay_millis=5000,
    max_export_batch_size=512
)
```

### 3. Async Export

GenOps uses async batch export by default - no blocking on span creation.

### 4. Attribute Reduction

```python
# Only include essential attributes
span.set_attribute("team", team)  # ✅ Essential
span.set_attribute("verbose_debug_info", "...")  # ❌ Avoid
```

### 5. Caching (Tempo)

```yaml
storage:
  trace:
    cache: memcached
    memcached:
      consistent_hash: true
      host: memcached:11211
      service: memcached
      timeout: 500ms
```

---

## Security

### Authentication

**Basic Auth:**
```python
configure_tempo(
    endpoint="https://tempo.example.com",
    headers={"Authorization": "Basic " + base64_encoded_credentials}
)
```

**Bearer Token:**
```python
configure_tempo(
    endpoint="https://tempo.example.com",
    headers={"Authorization": "Bearer your-token"}
)
```

**API Key:**
```python
configure_tempo(
    endpoint="https://tempo.grafana.net",
    headers={"X-API-Key": "your-api-key"}
)
```

### TLS/SSL

**Grafana Cloud Tempo (TLS enabled by default):**
```python
configure_tempo(endpoint="https://tempo-us-central1.grafana.net")
```

**Self-signed certificates:**
```python
import os
os.environ["OTEL_EXPORTER_OTLP_CERTIFICATE"] = "/path/to/cert.pem"
```

### Data Privacy

**Sensitive attribute filtering** (in OTel Collector):
```yaml
processors:
  attributes:
    actions:
      - key: customer_email
        action: delete
      - key: api_key
        action: delete

service:
  pipelines:
    traces:
      processors: [attributes]
```

---

## Examples

### Complete Examples

1. **[Direct Export](../../examples/tempo/direct_export.py)**
   - Basic Tempo setup
   - Validation
   - Simple trace creation

2. **[TraceQL Queries](../../examples/tempo/traceql_queries.py)**
   - Comprehensive query examples
   - Governance attribute queries
   - Cost analysis patterns

3. **[Cost Attribution](../../examples/tempo/cost_attribution.py)**
   - Multi-provider cost tracking
   - Customer billing
   - Budget management

4. **[Multi-Tenant](../../examples/tempo/multi_tenant.py)**
   - Tenant isolation
   - Per-tenant policies
   - Cross-tenant analysis

---

## Additional Resources

- **[Tempo Quickstart](../tempo-quickstart.md)** - 2-5 minute setup guide
- **[Grafana Quickstart](../grafana-quickstart.md)** - Full LGTM stack
- **[Official Tempo Docs](https://grafana.com/docs/tempo/latest/)** - Grafana documentation
- **[TraceQL Guide](https://grafana.com/docs/tempo/latest/traceql/)** - Query language reference
- **[OpenTelemetry Spec](https://opentelemetry.io/docs/specs/otel/)** - OTLP protocol

---

## Support

- **GitHub Issues**: [GenOps AI Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [GenOps AI Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Grafana Community**: [Grafana Community Forums](https://community.grafana.com/)

---

**You're now ready to leverage Grafana Tempo for comprehensive AI governance telemetry!** 🎉
