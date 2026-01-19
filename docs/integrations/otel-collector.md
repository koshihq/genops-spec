# OpenTelemetry Collector Integration - Comprehensive Guide

**Complete reference for integrating GenOps AI with OpenTelemetry Collector**

This guide covers everything from local development to production deployment, including architecture, configuration, validation, and troubleshooting.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Installation Patterns](#installation-patterns)
5. [Configuration Deep-Dive](#configuration-deep-dive)
6. [Integration Patterns](#integration-patterns)
7. [Validation and Testing](#validation-and-testing)
8. [Production Deployment](#production-deployment)
9. [Multi-Platform Export](#multi-platform-export)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)
12. [Examples and Templates](#examples-and-templates)

---

## Overview

### What is OpenTelemetry Collector?

The **OpenTelemetry Collector** is a vendor-agnostic implementation to receive, process, and export telemetry data. It removes the need to run, operate, and maintain multiple agents/collectors to support open-source observability data formats.

**Key capabilities:**
- **Receives** telemetry from multiple sources (OTLP, Jaeger, Prometheus, etc.)
- **Processes** data through pipelines (batching, filtering, enrichment)
- **Exports** to multiple backends (Tempo, Jaeger, Datadog, Splunk, etc.)
- **Standard Protocol**: OTLP (OpenTelemetry Protocol) for interoperability

### Why Use OTel Collector with GenOps AI?

**Vendor Neutrality**: Export to any observability backend without code changes
**Centralized Processing**: Single pipeline for all governance telemetry
**Performance**: Batching, sampling, and efficient resource usage
**Flexibility**: Route data to multiple destinations simultaneously
**Standards-Based**: OpenTelemetry is a CNCF graduated project

### GenOps + OpenTelemetry Architecture

GenOps AI extends OpenTelemetry with governance semantics:

```
┌───────────────────────────────────────────────────────────────┐
│                       Your AI Application                     │
│                  (Instrumented with GenOps SDK)               │
└───────────────────────────┬───────────────────────────────────┘
                            │ OTLP Export
                            │ (gRPC 4317 or HTTP 4318)
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  OpenTelemetry Collector                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Receivers   │  │  Processors  │  │    Exporters      │  │
│  │              │  │              │  │                   │  │
│  │  • OTLP      │→ │  • Batch     │→ │  • Tempo         │  │
│  │  • Jaeger    │  │  • Transform │  │  • Loki          │  │
│  │  • Prometheus│  │  • Resource  │  │  • Mimir         │  │
│  │              │  │  • Memory    │  │  • Prometheus    │  │
│  └──────────────┘  │  • Governance│  │  • Datadog       │  │
│                    │    Semantics │  │  • Splunk        │  │
│                    └──────────────┘  │  • Custom        │  │
│                                      └───────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Observability Backends     │
              │  (Grafana, Tempo, Datadog,  │
              │   Splunk, Honeycomb, etc.)  │
              └─────────────────────────────┘
```

**GenOps Governance Semantic Conventions** extend standard OTel attributes:

```
Standard OTel Attributes:
- service.name
- service.version
- deployment.environment

GenOps Governance Extensions:
- genops.cost.total           # Total cost of operation
- genops.cost.provider        # AI provider (openai, anthropic)
- genops.cost.model           # Model used (gpt-4, claude-3-sonnet)
- genops.policy.name          # Policy evaluated
- genops.policy.result        # Policy result (passed, blocked, warning)
- genops.budget.name          # Budget constraint
- genops.budget.utilization   # Budget utilization percentage
- genops.eval.metric_name     # Evaluation metric
- genops.eval.score           # Evaluation score
- genops.team                 # Team attribution
- genops.customer_id          # Customer attribution
- genops.project              # Project attribution
```

---

## Quick Start

### 5-Minute Quickstart

**New to OTel Collector?** Follow our [5-Minute Quickstart Guide](../otel-collector-quickstart.md) to get from zero to live governance dashboards.

The quickstart covers:
- Starting the LGTM stack with Docker Compose
- Validating your setup
- Running your first instrumented application
- Viewing data in Grafana dashboards

**After completing the quickstart**, return to this guide for production deployment and advanced configurations.

---

## Installation Patterns

### Local Development (Docker Compose)

**Use Case**: Development, testing, learning

**Architecture**: Complete LGTM stack running locally

**Setup**:

```bash
# Clone repository
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI

# Start observability stack
docker-compose -f docker-compose.observability.yml up -d

# Verify services
docker-compose -f docker-compose.observability.yml ps
```

**Services Included**:
- OTel Collector (ports 4317/4318 for OTLP)
- Grafana (port 3000)
- Tempo (port 3200) - Distributed tracing
- Loki (port 3100) - Log aggregation
- Mimir (port 9009) - Metrics storage
- Prometheus (port 9090) - Metrics scraping
- Redis (port 6379) - Demo app cache
- Demo API (port 8000) - Sample application

**Configuration Files**:
- `observability/otel-collector-config.yaml` - Collector configuration
- `observability/tempo-config.yaml` - Tempo backend
- `observability/loki-config.yaml` - Loki backend
- `observability/mimir-config.yaml` - Mimir backend
- `observability/grafana/` - Grafana datasources and dashboards

**Default Endpoints**:
- OTLP HTTP: `http://localhost:4318`
- OTLP gRPC: `http://localhost:4317`
- Grafana: `http://localhost:3000` (admin/genops)

---

### Kubernetes with Helm

**Use Case**: Production deployment, scalable infrastructure

**Prerequisites**:
- Kubernetes cluster (v1.19+)
- Helm 3.x installed
- kubectl configured

**Quick Deploy**:

```bash
# Add GenOps Helm repository
helm repo add genops-ai https://genops-ai.github.io/helm-charts
helm repo update

# Install with OTel Collector enabled
helm install genops-ai genops-ai/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set otelCollector.enabled=true \
  --set otelCollector.endpoint="http://otel-collector:4318"

# Verify deployment
kubectl get pods -n genops-system
kubectl logs -n genops-system deployment/genops-ai
```

**Helm Values Configuration**:

```yaml
# values.yaml
otelCollector:
  enabled: true
  endpoint: "http://otel-collector:4318"
  protocol: "http"  # or "grpc"

  # Optional: Custom headers for authentication
  headers:
    Authorization: "Bearer token-here"

  # Sampling configuration
  sampling:
    enabled: true
    rate: 0.1  # 10% sampling for high-volume

governance:
  # Default governance attributes
  team: "platform-team"
  project: "ai-platform"
  environment: "production"

  # Cost limits
  costLimits:
    daily: 1000.00
    monthly: 25000.00

  # Rate limiting
  rateLimiting:
    enabled: true
    requestsPerMinute: 100

# Observability integration
observability:
  grafana:
    enabled: true
    dashboardsConfigMap: "genops-dashboards"

  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: "30s"
```

**Production Configuration**:

```bash
# Deploy with production values
helm install genops-ai genops-ai/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --values production-values.yaml

# Upgrade existing deployment
helm upgrade genops-ai genops-ai/genops-ai \
  --namespace genops-system \
  --values production-values.yaml

# Rollback if needed
helm rollback genops-ai --namespace genops-system
```

**Service Mesh Integration** (Istio):

```yaml
# Enable service mesh sidecar injection
apiVersion: v1
kind: Namespace
metadata:
  name: genops-system
  labels:
    istio-injection: enabled

---
# VirtualService for external access
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai
  namespace: genops-system
spec:
  hosts:
  - genops-ai.example.com
  gateways:
  - genops-gateway
  http:
  - route:
    - destination:
        host: genops-ai
        port:
          number: 8080
```

---

### Cloud Platform Deployments

#### AWS Deployment

**Prerequisites**:
- AWS account with EKS cluster
- AWS Load Balancer Controller installed
- IRSA (IAM Roles for Service Accounts) configured

**Architecture Options**:

**Option 1: EKS with AWS X-Ray**
```yaml
# Export to AWS X-Ray for distributed tracing
exporters:
  awsxray:
    region: us-west-2
    no_verify_ssl: false
```

**Option 2: EKS with AWS CloudWatch**
```yaml
# Export to CloudWatch for logs and metrics
exporters:
  awscloudwatch:
    region: us-west-2
    log_group_name: /aws/eks/genops-ai
    log_stream_name: governance-telemetry
```

**Full AWS Deployment Guide**: [kubernetes-aws-deployment.md](../kubernetes-aws-deployment.md)

---

#### GCP Deployment

**Prerequisites**:
- GCP project with GKE cluster
- Cloud Trace API enabled
- Workload Identity configured

**Architecture Options**:

**Option 1: GKE with Cloud Trace**
```yaml
# Export to Google Cloud Trace
exporters:
  googlecloud:
    project: "your-gcp-project"
    use_insecure: false
```

**Option 2: GKE with Cloud Logging**
```yaml
# Export to Cloud Logging
exporters:
  googlecloudlogging:
    project_id: "your-gcp-project"
    log_name: "genops-ai-governance"
```

**Full GCP Deployment Guide**: [kubernetes-gcp-deployment.md](../kubernetes-gcp-deployment.md)

---

#### Azure Deployment

**Prerequisites**:
- Azure subscription with AKS cluster
- Azure Monitor enabled
- Managed Identity configured

**Architecture Options**:

**Option 1: AKS with Azure Monitor**
```yaml
# Export to Azure Monitor Application Insights
exporters:
  azuremonitor:
    instrumentation_key: "your-instrumentation-key"
    endpoint: "https://dc.services.visualstudio.com/v2/track"
```

**Full Azure Deployment Guide**: [kubernetes-azure-deployment.md](../kubernetes-azure-deployment.md)

---

## Configuration Deep-Dive

### OTel Collector Configuration Structure

The collector uses a pipeline-based configuration:

```yaml
# otel-collector-config.yaml structure
receivers:   # How telemetry enters the collector
processors:  # How telemetry is processed
exporters:   # Where telemetry is sent
extensions:  # Additional capabilities (health checks, pprof)
service:     # Pipeline definitions connecting receivers → processors → exporters
```

### Receivers Configuration

**OTLP Receiver** (Primary for GenOps):

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        max_recv_msg_size_mib: 16  # Max message size

      http:
        endpoint: 0.0.0.0:4318
        cors:
          allowed_origins:
            - "http://*"
            - "https://*"

        # Optional: Authentication
        auth:
          authenticator: oauth2client
```

**Additional Receivers** (Optional):

```yaml
receivers:
  # Prometheus metrics receiver
  prometheus:
    config:
      scrape_configs:
        - job_name: 'genops-ai'
          scrape_interval: 30s
          static_configs:
            - targets: ['genops-ai:8001']

  # Jaeger receiver (for existing Jaeger instrumentation)
  jaeger:
    protocols:
      grpc:
        endpoint: 0.0.0.0:14250
      thrift_http:
        endpoint: 0.0.0.0:14268
```

### Processors Configuration

**Batch Processor** (Recommended for performance):

```yaml
processors:
  batch:
    # Wait time before sending batch
    timeout: 1s

    # Send batch when this many spans accumulated
    send_batch_size: 1024

    # Maximum batch size (spans)
    send_batch_max_size: 2048
```

**Memory Limiter** (Prevent OOM):

```yaml
processors:
  memory_limiter:
    # Check memory usage every 1 second
    check_interval: 1s

    # Soft limit - start dropping data
    limit_mib: 512

    # Hard limit - force GC
    spike_limit_mib: 128
```

**Transform Processor** (GenOps Governance Semantics):

```yaml
processors:
  transform:
    trace_statements:
      # Extract cost information to root span attribute
      - context: span
        statements:
          - set(attributes["genops.cost.total"], attributes["genops.cost.amount"]) where attributes["genops.cost.amount"] != nil

          # Mark as governance-processed
          - set(attributes["genops.governance.processed"], true)

          # Normalize team names to lowercase
          - set(attributes["genops.team"], LowerCase(attributes["genops.team"])) where attributes["genops.team"] != nil

    metric_statements:
      # Transform GenOps cost metrics
      - context: metric
        statements:
          - set(name, "genops_ai_cost_total") where name == "genops.cost"
          - set(name, "genops_ai_tokens_total") where name == "genops.tokens"
          - set(unit, "USD") where name == "genops_ai_cost_total"
```

**Resource Detection** (Auto-detect environment):

```yaml
processors:
  resourcedetection:
    detectors:
      - env       # Environment variables
      - system    # System metadata (hostname, OS)
      - docker    # Docker container metadata
      - eks       # AWS EKS metadata (if on EKS)
      - gcp       # GCP metadata (if on GCP)
      - azure     # Azure metadata (if on Azure)

    timeout: 5s
    override: false  # Don't override existing resource attributes
```

**Resource Processor** (Add custom attributes):

```yaml
processors:
  resource:
    attributes:
      - key: service.namespace
        value: "genops-ai"
        action: upsert

      - key: deployment.environment
        from_attribute: ENVIRONMENT
        action: insert

      - key: cloud.region
        value: "us-west-2"
        action: insert
```

### Exporters Configuration

**Tempo Exporter** (Distributed Tracing):

```yaml
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true  # For local dev; use proper TLS in production

    # Optional: Compression
    compression: gzip

    # Optional: Retry configuration
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s
```

**Loki Exporter** (Log Aggregation):

```yaml
exporters:
  loki:
    endpoint: http://loki:3100/loki/api/v1/push

    # Labels for log streams
    labels:
      resource:
        service.name: "service_name"
        deployment.environment: "env"
      attributes:
        genops.team: "team"
        genops.customer_id: "customer"
```

**Mimir/Prometheus Exporter** (Metrics):

```yaml
exporters:
  prometheusremotewrite:
    endpoint: http://mimir:9009/api/v1/push

    # Add external labels
    external_labels:
      cluster: "production"
      region: "us-west-2"

    # Resource to metric labels
    resource_to_telemetry_conversion:
      enabled: true
```

**Datadog Exporter**:

```yaml
exporters:
  datadog:
    api:
      key: "${env:DD_API_KEY}"
      site: "datadoghq.com"

    # Host metadata
    host_metadata:
      enabled: true
      hostname_source: "config_or_system"
```

**Splunk HEC Exporter**:

```yaml
exporters:
  splunk_hec:
    endpoint: "https://splunk.example.com:8088/services/collector"
    token: "${env:SPLUNK_HEC_TOKEN}"
    index: "genops_ai"
    source: "genops:telemetry"
    sourcetype: "_json"
```

### Service Pipeline Configuration

**Traces Pipeline** (Most important for GenOps):

```yaml
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [
        memory_limiter,
        resourcedetection,
        resource,
        transform,
        batch
      ]
      exporters: [otlp/tempo, datadog, logging]
```

**Metrics Pipeline**:

```yaml
service:
  pipelines:
    metrics:
      receivers: [otlp, prometheus]
      processors: [
        memory_limiter,
        resourcedetection,
        resource,
        transform,
        batch
      ]
      exporters: [prometheusremotewrite, datadog]
```

**Logs Pipeline**:

```yaml
service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [
        memory_limiter,
        resourcedetection,
        resource,
        batch
      ]
      exporters: [loki, splunk_hec]
```

---

## Integration Patterns

### Auto-Instrumentation (Zero-Code)

**Simplest approach** - GenOps auto-detects OTel Collector:

```python
from genops import auto_instrument
from genops.providers.openai import instrument_openai

# Auto-instruments and configures OTLP export
# Looks for OTEL_EXPORTER_OTLP_ENDPOINT env var
# Falls back to http://localhost:4318
auto_instrument()

# Instrument your AI provider
openai_client = instrument_openai()

# All operations now automatically export to OTel Collector
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Environment Variables**:

```bash
# OTLP endpoint (HTTP or gRPC)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Service identification
export OTEL_SERVICE_NAME="my-ai-app"
export OTEL_SERVICE_VERSION="1.0.0"

# Optional: Custom headers (for authentication)
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer token123"

# Optional: Protocol selection
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"  # or "grpc"
```

### Manual Instrumentation

**Full control over telemetry**:

```python
from genops.core.telemetry import GenOpsTelemetry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry manually
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()

# OTLP exporter to collector
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
    headers={"Authorization": "Bearer your-token"}
)

tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Use GenOps telemetry
telemetry = GenOpsTelemetry()

with telemetry.trace_operation(
    operation_name="complex_ai_workflow",
    team="engineering",
    customer_id="customer-123",
    project="ai-assistant"
) as span:
    # Your AI operations
    result = perform_ai_operation()

    # Record governance telemetry
    telemetry.record_cost(span, cost=0.05, provider="openai", model="gpt-4")
    telemetry.record_policy(span, "cost_limit", "passed", "Within budget")
    telemetry.record_evaluation(span, "quality", score=0.95, threshold=0.8)
```

### Framework-Specific Integration

**LangChain Auto-Instrumentation**:

```python
from genops.providers.langchain import instrument_langchain
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Auto-instrument LangChain
instrument_langchain()

# All chains automatically tracked
chain = LLMChain(llm=OpenAI(model="gpt-4"), prompt=prompt_template)
result = chain.run(input="Analyze this data")

# Governance telemetry automatically exported to OTel Collector
```

**LlamaIndex Auto-Instrumentation**:

```python
from genops.providers.llamaindex import instrument_llamaindex
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Auto-instrument LlamaIndex
instrument_llamaindex()

# All queries automatically tracked
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("What is the total cost?")

# Governance telemetry automatically exported
```

---

## Validation and Testing

### Automated Validation

**Use the validation script**:

```bash
# Run full validation
python examples/observability/validate_otel_collector.py

# Expected output:
# ✅ [SUCCESS] Collector Status: Healthy
# ✅ [SUCCESS] OTLP HTTP Endpoint: Accessible (port 4318)
# ✅ [SUCCESS] OTLP gRPC Endpoint: Accessible (port 4317)
# ✅ [SUCCESS] Grafana: Accessible
# ✅ [SUCCESS] Tempo: Accessible
```

**Validation Checks**:
1. OTel Collector health endpoint (port 13133)
2. OTLP HTTP endpoint accessibility (port 4318)
3. OTLP gRPC endpoint accessibility (port 4317)
4. Backend services (Grafana, Tempo, Loki, Mimir)
5. OpenTelemetry dependencies installed

**Programmatic Validation**:

```python
from examples.observability.otel_collector_validation import (
    validate_setup,
    print_validation_result
)

# Run validation
result = validate_setup(
    collector_endpoint="http://localhost:4318",
    check_connectivity=True,
    check_backends=True
)

# Display results
print_validation_result(result)

# Check validation status
if result.valid:
    print("Setup validated successfully!")
    # Proceed with telemetry export
else:
    print(f"Validation failed with {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - {error}")
```

### Manual Testing

**Test OTel Collector Health**:

```bash
# Health check endpoint
curl http://localhost:13133/
# Expected: {"status":"Server available","upSince":"..."}

# Check OTLP HTTP endpoint
curl -v http://localhost:4318/v1/traces
# Expected: Connection successful (405 Method Not Allowed is OK)

# Check OTLP gRPC endpoint
nc -zv localhost 4317
# Expected: Connection to localhost port 4317 [tcp/*] succeeded!
```

**Send Test Telemetry**:

```bash
# Run the quickstart example
python examples/quickstarts/otel_collector_quickstart.py

# Or use the demo API
curl -X POST http://localhost:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Test message",
    "model": "gpt-4",
    "team": "test-team",
    "customer_id": "test-customer"
  }'
```

**Verify in Grafana**:

1. Open http://localhost:3000 (admin/genops)
2. Navigate to **Dashboards → GenOps AI - Governance Overview**
3. Verify data appears in panels
4. Click **Explore → Tempo** to search traces
5. Query: `{.genops.team="test-team"}`

### Integration Testing

**Unit Test Example** (Mock Collector):

```python
import pytest
from unittest.mock import Mock, patch
from opentelemetry.sdk.trace import TracerProvider
from genops.core.telemetry import GenOpsTelemetry

@pytest.fixture
def mock_collector():
    """Mock OTel Collector for testing"""
    with patch('opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter') as mock:
        yield mock

def test_telemetry_export(mock_collector):
    """Test telemetry is exported to collector"""
    telemetry = GenOpsTelemetry()

    with telemetry.trace_operation(
        operation_name="test_op",
        team="test-team"
    ) as span:
        telemetry.record_cost(span, cost=0.01, provider="openai", model="gpt-4")

    # Verify exporter was called
    assert mock_collector.called
```

**End-to-End Test Example**:

```python
import requests
import time
from examples.observability.otel_collector_validation import validate_setup

def test_end_to_end_data_flow():
    """Test complete data flow from GenOps to Grafana"""

    # 1. Validate setup
    result = validate_setup()
    assert result.valid, f"Setup validation failed: {result.errors}"

    # 2. Send test telemetry
    from genops.core.telemetry import GenOpsTelemetry
    telemetry = GenOpsTelemetry()

    test_customer = "e2e-test-customer"

    with telemetry.trace_operation(
        operation_name="e2e_test",
        team="test-team",
        customer_id=test_customer
    ) as span:
        telemetry.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

    # 3. Wait for telemetry to be processed
    time.sleep(10)

    # 4. Query Tempo for the trace
    tempo_url = "http://localhost:3200/api/search"
    params = {
        "tags": f"genops.customer_id={test_customer}",
        "limit": 10
    }

    response = requests.get(tempo_url, params=params)
    assert response.status_code == 200

    traces = response.json().get("traces", [])
    assert len(traces) > 0, "No traces found in Tempo"

    # 5. Verify trace has governance attributes
    trace = traces[0]
    assert "genops.cost.total" in str(trace)
    assert "genops.customer_id" in str(trace)
```

---

## Production Deployment

### Scaling Considerations

**Horizontal Scaling** (Multiple Collector Replicas):

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
spec:
  replicas: 3  # Scale based on load
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-contrib:0.90.1
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

**Load Balancing** (Service):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
spec:
  type: LoadBalancer  # Or ClusterIP for internal
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: health
    port: 13133
    targetPort: 13133
  selector:
    app: otel-collector
```

**Sampling for High Volume**:

```yaml
processors:
  probabilistic_sampler:
    # Sample 10% of traces
    sampling_percentage: 10

    # Hash seed for consistent sampling
    hash_seed: 22
```

**Tail-Based Sampling** (Keep important traces):

```yaml
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 100
    expected_new_traces_per_sec: 10
    policies:
      # Always sample errors
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Always sample high-cost operations
      - name: high_cost
        type: numeric_attribute
        numeric_attribute:
          key: genops.cost.total
          min_value: 1.0  # Sample if cost > $1.00

      # Always sample policy violations
      - name: policy_violations
        type: string_attribute
        string_attribute:
          key: genops.policy.result
          values: [blocked, warning]

      # Probabilistic sampling for everything else
      - name: probabilistic
        type: probabilistic
        probabilistic:
          sampling_percentage: 10
```

### Performance Tuning

**Batch Processing**:

```yaml
processors:
  batch:
    # Increase batch size for higher throughput
    timeout: 5s
    send_batch_size: 8192
    send_batch_max_size: 16384
```

**Concurrent Exports**:

```yaml
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    sending_queue:
      enabled: true
      num_consumers: 10  # Concurrent export workers
      queue_size: 5000
```

**Resource Limits**:

```yaml
extensions:
  memory_ballast:
    # Reserve memory to reduce GC pressure
    size_mib: 256

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 2048  # Increase for high-volume
    spike_limit_mib: 512
```

### Security Best Practices

**TLS/mTLS for OTLP**:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        tls:
          cert_file: /certs/server.crt
          key_file: /certs/server.key
          client_ca_file: /certs/ca.crt  # For mTLS
          client_auth_type: RequireAndVerifyClientCert
```

**Authentication**:

```yaml
extensions:
  oauth2client:
    client_id: "genops-ai"
    client_secret: "${env:OAUTH_CLIENT_SECRET}"
    token_url: "https://auth.example.com/token"
    scopes: ["telemetry.write"]

receivers:
  otlp:
    protocols:
      http:
        auth:
          authenticator: oauth2client
```

**Network Policies** (Kubernetes):

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: otel-collector-policy
spec:
  podSelector:
    matchLabels:
      app: otel-collector
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Allow OTLP from application pods
  - from:
    - namespaceSelector:
        matchLabels:
          name: genops-apps
    ports:
    - protocol: TCP
      port: 4318
    - protocol: TCP
      port: 4317
  egress:
  # Allow export to backends
  - to:
    - namespaceSelector:
        matchLabels:
          name: observability
    ports:
    - protocol: TCP
      port: 4317  # Tempo
    - protocol: TCP
      port: 3100  # Loki
```

### High Availability

**Multi-Zone Deployment**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      affinity:
        # Spread across availability zones
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - otel-collector
              topologyKey: topology.kubernetes.io/zone
```

**Health Checks**:

```yaml
spec:
  containers:
  - name: otel-collector
    livenessProbe:
      httpGet:
        path: /
        port: 13133
      initialDelaySeconds: 10
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /
        port: 13133
      initialDelaySeconds: 5
      periodSeconds: 5
```

---

## Multi-Platform Export

### Simultaneous Multi-Destination Export

**Export to Multiple Backends**:

```yaml
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, memory_limiter]
      exporters: [
        otlp/tempo,      # Local Tempo
        datadog,          # Datadog APM
        splunk_hec,       # Splunk Enterprise
        otlp/honeycomb    # Honeycomb
      ]
```

**Per-Backend Sampling** (Cost Optimization):

```yaml
exporters:
  # Full fidelity to local Tempo
  otlp/tempo:
    endpoint: tempo:4317

  # Sampled export to Datadog (cost optimization)
  datadog:
    api:
      key: "${env:DD_API_KEY}"
    # Sampling handled by probabilistic_sampler processor

  # High-value events to Splunk (compliance)
  splunk_hec:
    endpoint: "https://splunk.example.com:8088/services/collector"
    token: "${env:SPLUNK_HEC_TOKEN}"
    # Only send policy violations and high-cost operations
```

**Routing Processor** (Route by attributes):

```yaml
processors:
  routing:
    from_attribute: "genops.environment"
    table:
      - value: "production"
        exporters: [otlp/tempo, datadog, splunk_hec]
      - value: "staging"
        exporters: [otlp/tempo]
      - value: "development"
        exporters: [logging]
    default_exporters: [otlp/tempo]
```

### Platform-Specific Configurations

**Datadog**:

```yaml
exporters:
  datadog:
    api:
      key: "${env:DD_API_KEY}"
      site: "datadoghq.com"  # or datadoghq.eu

    # Map GenOps attributes to Datadog tags
    host_metadata:
      enabled: true
      tags:
        - "env:${env:ENVIRONMENT}"
        - "service:genops-ai"

    # Resource to metric labels
    resource_to_telemetry_conversion:
      enabled: true
```

**Honeycomb**:

```yaml
exporters:
  otlp/honeycomb:
    endpoint: "api.honeycomb.io:443"
    headers:
      "x-honeycomb-team": "${env:HONEYCOMB_API_KEY}"
      "x-honeycomb-dataset": "genops-ai"
```

**Grafana Cloud**:

```yaml
exporters:
  otlphttp/grafanacloud:
    endpoint: "https://otlp-gateway-prod-us-central-0.grafana.net/otlp"
    headers:
      authorization: "Basic ${env:GRAFANA_CLOUD_API_TOKEN}"
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Connection refused" to OTel Collector

**Symptoms:**
- Application logs show: `Failed to export traces: connection refused`
- Telemetry not appearing in backends

**Diagnosis:**
```bash
# Check if collector is running
docker ps | grep otel-collector
# Or for Kubernetes:
kubectl get pods -l app=otel-collector

# Check collector logs
docker logs otel-collector
# Or for Kubernetes:
kubectl logs deployment/otel-collector

# Test OTLP endpoint
curl -v http://localhost:4318/v1/traces
```

**Solutions:**

1. **Collector not running**:
   ```bash
   # Start Docker Compose stack
   docker-compose -f docker-compose.observability.yml up -d

   # Or restart Kubernetes deployment
   kubectl rollout restart deployment/otel-collector
   ```

2. **Wrong endpoint URL**:
   ```bash
   # Check environment variable
   echo $OTEL_EXPORTER_OTLP_ENDPOINT

   # Should be: http://localhost:4318 (for local)
   # Or: http://otel-collector:4318 (for Kubernetes)
   ```

3. **Port conflict**:
   ```bash
   # Check what's using the port
   lsof -i :4318

   # Stop conflicting process or change port
   ```

---

#### Issue: "No data in Grafana dashboards"

**Symptoms:**
- Grafana shows empty dashboards
- Tempo queries return no results
- Collector is running and receiving data

**Diagnosis:**
```bash
# Check collector is exporting
docker logs otel-collector | grep "Exporting"

# Check Tempo has data
curl http://localhost:3200/api/search | jq

# Check Grafana data sources
curl http://admin:genops@localhost:3000/api/datasources
```

**Solutions:**

1. **Telemetry not reaching collector**:
   ```python
   # Verify OTLP exporter is configured
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

   # Ensure exporter is added
   tracer_provider = TracerProvider()
   tracer_provider.add_span_processor(
       BatchSpanProcessor(
           OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
       )
   )
   ```

2. **Data not being exported from collector**:
   ```yaml
   # Check service pipelines in otel-collector-config.yaml
   service:
     pipelines:
       traces:
         receivers: [otlp]
         processors: [batch]
         exporters: [otlp/tempo]  # Verify exporter is listed
   ```

3. **Time range issue in Grafana**:
   - Change time range to "Last 15 minutes" or "Last 5 minutes"
   - Click refresh button in top-right

4. **Grafana data source not configured**:
   ```bash
   # Verify Tempo data source
   curl http://admin:genops@localhost:3000/api/datasources | jq '.[] | select(.type=="tempo")'
   ```

---

#### Issue: "High memory usage in collector"

**Symptoms:**
- Collector container OOMKilled
- Slow telemetry processing
- Collector crashes under load

**Diagnosis:**
```bash
# Check collector memory usage
docker stats otel-collector

# Check collector logs for memory warnings
docker logs otel-collector | grep -i "memory"

# Check processor configuration
grep -A 10 "memory_limiter" observability/otel-collector-config.yaml
```

**Solutions:**

1. **Increase memory limit**:
   ```yaml
   # docker-compose.observability.yml
   services:
     otel-collector:
       deploy:
         resources:
           limits:
             memory: 1Gi  # Increase from 512Mi
   ```

2. **Configure memory_limiter processor**:
   ```yaml
   processors:
     memory_limiter:
       check_interval: 1s
       limit_mib: 800  # 80% of container limit
       spike_limit_mib: 200
   ```

3. **Increase batch processing**:
   ```yaml
   processors:
     batch:
       timeout: 5s  # Increase from 1s
       send_batch_size: 2048  # Increase batch size
   ```

4. **Enable sampling**:
   ```yaml
   processors:
     probabilistic_sampler:
       sampling_percentage: 10  # Sample 10% for high volume
   ```

---

#### Issue: "Slow trace queries in Tempo"

**Symptoms:**
- Grafana Explore → Tempo queries timeout
- Trace search takes >30 seconds
- "Query timeout" errors in Grafana

**Diagnosis:**
```bash
# Check Tempo ingestion rate
curl http://localhost:3200/metrics | grep tempo_ingester

# Check Tempo query frontend logs
docker logs tempo | grep "query"

# Check trace count
curl http://localhost:3200/api/search?limit=1 | jq '.traces | length'
```

**Solutions:**

1. **Increase Tempo query timeout**:
   ```yaml
   # tempo-config.yaml
   query_frontend:
     search:
       max_duration: 0  # No duration limit
       query_timeout: 2m  # Increase timeout
   ```

2. **Add indexes for common queries**:
   ```yaml
   # tempo-config.yaml
   overrides:
     defaults:
       index:
         trace_id_column: "trace_id"
         span_id_column: "span_id"
   ```

3. **Use more specific queries**:
   ```
   # Instead of broad search:
   {}

   # Use specific attributes:
   {.genops.team="engineering" && .genops.customer_id="customer-123"}
   ```

---

#### Issue: "Policy violations not appearing"

**Symptoms:**
- Policy evaluations run but not visible in dashboards
- `genops.policy.*` attributes missing from traces

**Diagnosis:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check if policies are registered
from genops.core.policy import _policy_engine
print(_policy_engine.list_policies())

# Check spans have policy attributes
with telemetry.trace_operation(...) as span:
    # After recording policy
    print(span.attributes)
```

**Solutions:**

1. **Ensure policy is registered before evaluation**:
   ```python
   from genops.core.policy import register_policy, PolicyResult

   # Register policy first
   register_policy(
       name="cost_limit",
       enforcement_level=PolicyResult.WARNING,
       conditions={"max_cost": 1.0}
   )

   # Then evaluate
   result = _policy_engine.evaluate_policy("cost_limit", {"cost": 0.05})
   ```

2. **Record policy result in span**:
   ```python
   telemetry.record_policy(
       span,
       policy_name="cost_limit",
       result=result.result.value,  # "passed", "warning", "blocked"
       reason=result.reason
   )
   ```

3. **Check transform processor doesn't remove attributes**:
   ```yaml
   # Verify no attribute deletion in transform processor
   processors:
     transform:
       trace_statements:
         - context: span
           statements:
             # Don't delete genops.policy.* attributes
   ```

---

### Debug Mode

**Enable verbose logging**:

```yaml
# otel-collector-config.yaml
service:
  telemetry:
    logs:
      level: debug  # Change from "info"

  # Enable internal telemetry
  telemetry:
    metrics:
      address: ":8888"  # Expose collector metrics

extensions:
  zpages:
    endpoint: "0.0.0.0:55679"  # Debug pages
  pprof:
    endpoint: "0.0.0.0:1777"  # Profiling
```

**Access debug information**:

```bash
# Collector metrics
curl http://localhost:8888/metrics

# ZPages (service status)
open http://localhost:55679/debug/servicez

# Trace debug page
open http://localhost:55679/debug/tracez

# CPU profiling
curl http://localhost:1777/debug/pprof/profile > cpu.prof
go tool pprof cpu.prof
```

---

## Advanced Topics

### Custom Processors

**Create custom processor in Go**:

```go
package customprocessor

import (
    "context"
    "go.opentelemetry.io/collector/pdata/ptrace"
    "go.opentelemetry.io/collector/processor"
)

type genopsProcessor struct {
    config *Config
}

func (p *genopsProcessor) processTraces(ctx context.Context, td ptrace.Traces) (ptrace.Traces, error) {
    // Iterate through spans
    rss := td.ResourceSpans()
    for i := 0; i < rss.Len(); i++ {
        rs := rss.At(i)
        ilss := rs.ScopeSpans()

        for j := 0; j < ilss.Len(); j++ {
            ils := ilss.At(j)
            spans := ils.Spans()

            for k := 0; k < spans.Len(); k++ {
                span := spans.At(k)
                attrs := span.Attributes()

                // Custom logic: Calculate cost tier
                if cost, ok := attrs.Get("genops.cost.total"); ok {
                    costValue := cost.AsDouble()
                    var tier string

                    if costValue < 0.01 {
                        tier = "low"
                    } else if costValue < 0.10 {
                        tier = "medium"
                    } else {
                        tier = "high"
                    }

                    attrs.PutStr("genops.cost.tier", tier)
                }
            }
        }
    }

    return td, nil
}
```

### Governance-Specific Metrics

**Custom metric generation from traces**:

```yaml
# Use spanmetrics connector (experimental)
connectors:
  spanmetrics:
    # Generate metrics from span attributes
    dimensions:
      - name: genops.team
      - name: genops.customer_id
      - name: genops.cost.provider
      - name: genops.cost.model

    # Custom metrics
    metrics:
      - name: genops_cost_total
        description: "Total AI operation cost"
        unit: "USD"
        sum:
          value_attribute: genops.cost.total

      - name: genops_tokens_total
        description: "Total tokens used"
        sum:
          value_attribute: genops.tokens.total

      - name: genops_operations_total
        description: "Total AI operations"
        count: {}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [spanmetrics, otlp/tempo]

    # Metrics pipeline consumes from spanmetrics connector
    metrics:
      receivers: [spanmetrics]
      exporters: [prometheusremotewrite]
```

### Multi-Tenancy

**Tenant isolation and routing**:

```yaml
processors:
  # Route by customer_id
  routing/customer:
    from_attribute: "genops.customer_id"
    table:
      # Enterprise customers → dedicated backend
      - value: "enterprise-*"
        exporters: [otlp/tempo-dedicated]

      # Standard customers → shared backend
      - value: "*"
        exporters: [otlp/tempo-shared]

  # Add tenant-specific resource attributes
  resource/tenant:
    attributes:
      - key: tenant.tier
        from_attribute: genops.customer_id
        action: insert

      - key: tenant.region
        value: "us-west-2"
        action: insert

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [routing/customer, resource/tenant, batch]
      # Exporters determined by routing processor
```

### Cost Attribution and Chargebacks

**Example Grafana dashboard query for cost attribution**:

```promql
# Total cost by team (last 24h)
sum by (genops_team) (
  rate(genops_cost_total[24h])
)

# Cost by customer (monthly)
sum by (genops_customer_id) (
  increase(genops_cost_total[30d])
)

# Cost by model and provider
sum by (genops_cost_model, genops_cost_provider) (
  rate(genops_cost_total[1h])
)

# Budget utilization
(
  sum(genops_budget_used) / sum(genops_budget_limit)
) * 100
```

---

## Examples and Templates

### Production-Ready Configuration Template

**Complete `otel-collector-config-production.yaml`**:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        max_recv_msg_size_mib: 32
        tls:
          cert_file: /certs/server.crt
          key_file: /certs/server.key
          client_ca_file: /certs/ca.crt
          client_auth_type: RequireAndVerifyClientCert

      http:
        endpoint: 0.0.0.0:4318
        cors:
          allowed_origins:
            - "https://*.example.com"

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 2048
    spike_limit_mib: 512

  resourcedetection:
    detectors: [env, system, docker, eks]
    timeout: 5s

  resource:
    attributes:
      - key: service.namespace
        value: "genops-ai"
        action: upsert
      - key: deployment.environment
        from_attribute: ENVIRONMENT
        action: insert

  transform:
    trace_statements:
      - context: span
        statements:
          - set(attributes["genops.cost.total"], attributes["genops.cost.amount"]) where attributes["genops.cost.amount"] != nil
          - set(attributes["genops.governance.processed"], true)
          - set(attributes["genops.team"], LowerCase(attributes["genops.team"])) where attributes["genops.team"] != nil

  # Tail-based sampling for cost optimization
  tail_sampling:
    decision_wait: 10s
    num_traces: 1000
    expected_new_traces_per_sec: 100
    policies:
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: high_cost
        type: numeric_attribute
        numeric_attribute:
          key: genops.cost.total
          min_value: 1.0
      - name: policy_violations
        type: string_attribute
        string_attribute:
          key: genops.policy.result
          values: [blocked, warning]
      - name: probabilistic
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

  batch:
    timeout: 5s
    send_batch_size: 8192
    send_batch_max_size: 16384

exporters:
  # Primary: Tempo for distributed tracing
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: false
      cert_file: /certs/client.crt
      key_file: /certs/client.key
    compression: gzip
    retry_on_failure:
      enabled: true
      max_elapsed_time: 300s
    sending_queue:
      enabled: true
      num_consumers: 10
      queue_size: 5000

  # Secondary: Datadog for additional observability
  datadog:
    api:
      key: "${env:DD_API_KEY}"
      site: "datadoghq.com"
    host_metadata:
      enabled: true
      tags:
        - "env:production"
        - "service:genops-ai"

  # Compliance: Splunk for audit trails
  splunk_hec:
    endpoint: "https://splunk.example.com:8088/services/collector"
    token: "${env:SPLUNK_HEC_TOKEN}"
    index: "genops_ai_prod"
    source: "genops:telemetry"
    sourcetype: "_json"
    tls:
      insecure_skip_verify: false

extensions:
  health_check:
    endpoint: "0.0.0.0:13133"

  pprof:
    endpoint: "0.0.0.0:1777"

  zpages:
    endpoint: "0.0.0.0:55679"

service:
  extensions: [health_check, pprof, zpages]

  telemetry:
    logs:
      level: info
    metrics:
      address: "0.0.0.0:8888"

  pipelines:
    traces:
      receivers: [otlp]
      processors: [
        memory_limiter,
        resourcedetection,
        resource,
        transform,
        tail_sampling,
        batch
      ]
      exporters: [otlp/tempo, datadog, splunk_hec]
```

### Kubernetes Production Deployment

**Complete Kubernetes manifests** (`k8s/otel-collector.yaml`):

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: genops-system
data:
  collector-config.yaml: |
    # Full production config from above
    # (contents of otel-collector-config-production.yaml)

---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: genops-system
spec:
  type: LoadBalancer
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: metrics
    port: 8888
    targetPort: 8888
  - name: health
    port: 13133
    targetPort: 13133
  selector:
    app: otel-collector

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: genops-system
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      serviceAccountName: otel-collector
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001

      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-contrib:0.90.1
        command: ["/otelcol-contrib"]
        args:
          - "--config=/conf/collector-config.yaml"

        ports:
        - containerPort: 4317
          name: otlp-grpc
        - containerPort: 4318
          name: otlp-http
        - containerPort: 8888
          name: metrics
        - containerPort: 13133
          name: health

        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

        livenessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 10
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 5
          periodSeconds: 5

        volumeMounts:
        - name: config
          mountPath: /conf
        - name: certs
          mountPath: /certs
          readOnly: true

        env:
        - name: DD_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: datadog-api-key
        - name: SPLUNK_HEC_TOKEN
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: splunk-hec-token

      volumes:
      - name: config
        configMap:
          name: otel-collector-config
      - name: certs
        secret:
          secretName: otel-collector-tls

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: otel-collector
  namespace: genops-system

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: otel-collector-pdb
  namespace: genops-system
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: otel-collector

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: otel-collector-hpa
  namespace: genops-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: otel-collector
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Additional Resources

### Documentation

- **Quickstart Guide**: [docs/otel-collector-quickstart.md](../otel-collector-quickstart.md)
- **Kubernetes Getting Started**: [docs/kubernetes-getting-started.md](../kubernetes-getting-started.md)
- **Splunk Integration**: [docs/splunk-quickstart.md](../splunk-quickstart.md)
- **GenOps Main Docs**: [README.md](../../README.md)

### External Resources

- **OpenTelemetry Collector**: [https://opentelemetry.io/docs/collector/](https://opentelemetry.io/docs/collector/)
- **OTLP Specification**: [https://github.com/open-telemetry/opentelemetry-proto](https://github.com/open-telemetry/opentelemetry-proto)
- **Grafana Tempo**: [https://grafana.com/docs/tempo/](https://grafana.com/docs/tempo/)
- **Grafana Loki**: [https://grafana.com/docs/loki/](https://grafana.com/docs/loki/)

### Community and Support

- **GitHub Issues**: [https://github.com/KoshiHQ/GenOps-AI/issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **GitHub Discussions**: [https://github.com/KoshiHQ/GenOps-AI/discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **OpenTelemetry Slack**: [https://cloud-native.slack.com](https://cloud-native.slack.com) (#otel-collector)

---

**This comprehensive guide covers the complete GenOps AI + OpenTelemetry Collector integration from development to production. For quick setup, start with the [5-Minute Quickstart](../otel-collector-quickstart.md).**
