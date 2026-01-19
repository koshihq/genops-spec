# Advanced Kubernetes Observability for GenOps AI

Complete guide for implementing comprehensive monitoring, tracing, and alerting for AI workloads in Kubernetes with GenOps governance integration.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Metrics Collection](#metrics-collection)
5. [Distributed Tracing](#distributed-tracing)
6. [Visualization & Dashboards](#visualization-dashboards)
7. [Alerting & Incident Response](#alerting-incident-response)
8. [Log Aggregation](#log-aggregation)
9. [Platform Integration](#platform-integration)
10. [Troubleshooting](#troubleshooting)

## Quick Start

Deploy complete observability stack in 5 minutes:

```bash
# 1. Install Prometheus + Grafana + Jaeger
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set grafana.adminPassword=admin

# 2. Install Jaeger for distributed tracing
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm install jaeger jaegertracing/jaeger \
  --namespace monitoring \
  --set provisionDataStore.cassandra=false \
  --set storage.type=memory

# 3. Configure GenOps to export telemetry
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-observability
  namespace: genops-system
data:
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.monitoring:4318"
  OTEL_METRICS_EXPORTER: "otlp"
  OTEL_TRACES_EXPORTER: "otlp"
  OTEL_LOGS_EXPORTER: "otlp"
EOF
```

✅ **Result:** Full observability stack with Prometheus, Grafana, and Jaeger monitoring GenOps AI.

## Architecture Overview

### Observability Stack Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   GenOps AI Workloads                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │ Inference  │  │ Evaluation │  │ Governance │               │
│  │  Service   │  │  Service   │  │   Engine   │               │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘               │
│        │                │                │                       │
│        │    OpenTelemetry Instrumentation│                       │
│        └────────────────┼────────────────┘                       │
└─────────────────────────┼──────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │    OpenTelemetry Collector          │
        │  - Receive: OTLP (gRPC/HTTP)        │
        │  - Process: Batch, Filter, Enrich   │
        │  - Export: Multiple backends         │
        └─────────────────┬───────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Prometheus  │  │   Jaeger     │  │   Grafana    │
│             │  │              │  │   Loki       │
│ - Metrics   │  │ - Traces     │  │ - Logs       │
│ - Alerts    │  │ - Sampling   │  │ - Dashboards │
│ - Recording │  │ - Storage    │  │ - Queries    │
└─────────────┘  └──────────────┘  └──────────────┘
      │                  │                  │
      └──────────────────┼──────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │   Unified Observability View    │
        │                                 │
        │  - Cost Attribution             │
        │  - Policy Compliance            │
        │  - Performance Metrics          │
        │  - Distributed Traces           │
        │  - Audit Logs                   │
        └─────────────────────────────────┘
```

### Key Components

- **OpenTelemetry Collector**: Centralized telemetry pipeline
- **Prometheus**: Time-series metrics database with alerting
- **Grafana**: Unified visualization and dashboards
- **Jaeger/Tempo**: Distributed tracing system
- **Loki**: Log aggregation and querying
- **AlertManager**: Alert routing and notification

## Prerequisites

### Required Tools

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify Helm installation
helm version

# Add required Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update
```

### Kubernetes Cluster Requirements

- **Minimum Kubernetes version**: 1.21+
- **Recommended resources**:
  - 4 vCPUs
  - 8 GB RAM
  - 100 GB storage for metrics retention
- **Persistent volume support** for metrics storage

### Network Requirements

**Required ports:**
```yaml
# Prometheus
- 9090: Prometheus UI and API
- 9093: AlertManager

# Grafana
- 3000: Grafana UI

# Jaeger
- 16686: Jaeger UI
- 14250: Jaeger gRPC collector
- 14268: Jaeger HTTP collector

# OpenTelemetry Collector
- 4317: OTLP gRPC
- 4318: OTLP HTTP
- 8888: Metrics endpoint
```

## Metrics Collection

### OpenTelemetry Collector Deployment

Deploy the OpenTelemetry Collector to receive and process telemetry:

```yaml
# otel-collector-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: monitoring
data:
  otel-collector-config.yaml: |
    receivers:
      # OTLP receiver for GenOps AI telemetry
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

      # Prometheus receiver for Kubernetes metrics
      prometheus:
        config:
          scrape_configs:
          - job_name: 'genops-pods'
            kubernetes_sd_configs:
            - role: pod
              namespaces:
                names:
                - genops-system
            relabel_configs:
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
              action: keep
              regex: true
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
              action: replace
              target_label: __metrics_path__
              regex: (.+)
            - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
              action: replace
              regex: ([^:]+)(?::\d+)?;(\d+)
              replacement: $1:$2
              target_label: __address__

      # Kubernetes cluster metrics
      k8s_cluster:
        collection_interval: 30s
        node_conditions_to_report: [Ready, MemoryPressure, DiskPressure]
        allocatable_types_to_report: [cpu, memory, storage, ephemeral-storage]

    processors:
      # Batch processor for efficiency
      batch:
        timeout: 10s
        send_batch_size: 1024

      # Memory limiter to prevent OOM
      memory_limiter:
        check_interval: 1s
        limit_mib: 512
        spike_limit_mib: 128

      # Resource processor to add governance attributes
      resource:
        attributes:
        - key: cluster.name
          value: ${CLUSTER_NAME}
          action: upsert
        - key: environment
          value: ${ENVIRONMENT}
          action: upsert

      # Attributes processor for enrichment
      attributes:
        actions:
        - key: genops.cost_center
          from_attribute: team
          action: insert
        - key: genops.governance_enabled
          value: true
          action: insert

    exporters:
      # Prometheus exporter
      prometheus:
        endpoint: 0.0.0.0:8889
        namespace: genops
        const_labels:
          cluster: ${CLUSTER_NAME}

      # OTLP exporter to Jaeger
      otlp/jaeger:
        endpoint: jaeger-collector.monitoring:4317
        tls:
          insecure: true

      # Logging exporter for debugging
      logging:
        loglevel: info

      # Prometheus remote write
      prometheusremotewrite:
        endpoint: http://prometheus-kube-prometheus-prometheus.monitoring:9090/api/v1/write
        tls:
          insecure: true

    service:
      pipelines:
        # Metrics pipeline
        metrics:
          receivers: [otlp, prometheus, k8s_cluster]
          processors: [memory_limiter, batch, resource, attributes]
          exporters: [prometheus, prometheusremotewrite, logging]

        # Traces pipeline
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch, resource]
          exporters: [otlp/jaeger, logging]

        # Logs pipeline
        logs:
          receivers: [otlp]
          processors: [memory_limiter, batch, resource]
          exporters: [logging]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: monitoring
spec:
  replicas: 2
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
        image: otel/opentelemetry-collector-contrib:0.91.0
        args:
        - --config=/conf/otel-collector-config.yaml
        env:
        - name: CLUSTER_NAME
          value: "genops-production"
        - name: ENVIRONMENT
          value: "production"
        ports:
        - containerPort: 4317
          name: otlp-grpc
          protocol: TCP
        - containerPort: 4318
          name: otlp-http
          protocol: TCP
        - containerPort: 8888
          name: metrics
          protocol: TCP
        - containerPort: 8889
          name: prometheus
          protocol: TCP
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
        volumeMounts:
        - name: config
          mountPath: /conf
        livenessProbe:
          httpGet:
            path: /
            port: 13133
        readinessProbe:
          httpGet:
            path: /
            port: 13133
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: monitoring
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
    protocol: TCP
  - name: otlp-http
    port: 4318
    targetPort: 4318
    protocol: TCP
  - name: metrics
    port: 8888
    targetPort: 8888
    protocol: TCP
  - name: prometheus
    port: 8889
    targetPort: 8889
    protocol: TCP
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: otel-collector
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: otel-collector
rules:
- apiGroups: [""]
  resources: ["nodes", "nodes/stats", "pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "statefulsets", "daemonsets"]
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
  namespace: monitoring
```

Apply the collector configuration:

```bash
kubectl apply -f otel-collector-deployment.yaml

# Verify deployment
kubectl get pods -n monitoring -l app=otel-collector
kubectl logs -n monitoring -l app=otel-collector --tail=50
```

### Prometheus Configuration

Configure Prometheus to scrape GenOps metrics:

```yaml
# prometheus-genops-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-genops-scrape
  namespace: monitoring
data:
  genops-scrape.yaml: |
    - job_name: 'genops-ai'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - genops-system
      relabel_configs:
      # Only scrape pods with prometheus.io/scrape annotation
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

      # Use custom metrics path if specified
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

      # Use custom port if specified
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

      # Add governance labels
      - source_labels: [__meta_kubernetes_pod_label_genops_ai_team]
        target_label: team
      - source_labels: [__meta_kubernetes_pod_label_genops_ai_project]
        target_label: project
      - source_labels: [__meta_kubernetes_pod_label_genops_ai_environment]
        target_label: environment
      - source_labels: [__meta_kubernetes_pod_label_genops_ai_cost_center]
        target_label: cost_center

      # Preserve pod information
      - source_labels: [__meta_kubernetes_namespace]
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: kubernetes_pod_name

      metric_relabel_configs:
      # Drop high-cardinality metrics
      - source_labels: [__name__]
        regex: 'go_.*|process_.*'
        action: drop

      # Add genops prefix to all metrics
      - source_labels: [__name__]
        regex: '(.+)'
        replacement: 'genops_${1}'
        target_label: __name__

    # Scrape OpenTelemetry Collector metrics
    - job_name: 'otel-collector'
      static_configs:
      - targets: ['otel-collector.monitoring:8888']
        labels:
          component: otel-collector

    # Scrape OTel Collector Prometheus endpoint
    - job_name: 'genops-otel-metrics'
      static_configs:
      - targets: ['otel-collector.monitoring:8889']
        labels:
          source: otel-collector
```

Apply the scrape configuration:

```bash
# Add scrape config to Prometheus
kubectl apply -f prometheus-genops-config.yaml

# Reload Prometheus configuration
kubectl exec -n monitoring prometheus-monitoring-kube-prometheus-prometheus-0 -- \
  curl -X POST http://localhost:9090/-/reload
```

### Custom Metrics Exporter

Create a custom metrics exporter for GenOps-specific telemetry:

```python
# genops-metrics-exporter.py
#!/usr/bin/env python3
"""
Custom Prometheus metrics exporter for GenOps AI governance telemetry.
"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
import time
import os

# Configuration
METRICS_PORT = int(os.getenv('METRICS_PORT', '8080'))
OTEL_ENDPOINT = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://otel-collector:4317')

# Prometheus metrics
cost_total = Gauge('genops_cost_total_usd', 'Total cost in USD', ['team', 'project', 'provider'])
cost_by_model = Gauge('genops_cost_by_model_usd', 'Cost by model', ['model', 'provider'])
budget_remaining = Gauge('genops_budget_remaining_usd', 'Remaining budget', ['team', 'project'])
budget_utilization = Gauge('genops_budget_utilization_percent', 'Budget utilization %', ['team'])

policy_violations = Counter('genops_policy_violations_total', 'Policy violations', ['policy_type', 'severity'])
policy_enforcements = Counter('genops_policy_enforcements_total', 'Policy enforcements', ['action'])

request_count = Counter('genops_requests_total', 'Total requests', ['provider', 'model', 'status'])
request_duration = Histogram(
    'genops_request_duration_seconds',
    'Request duration',
    ['provider', 'model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

token_usage = Counter('genops_tokens_total', 'Token usage', ['provider', 'model', 'type'])
token_cost = Gauge('genops_token_cost_per_1k', 'Cost per 1K tokens', ['provider', 'model', 'type'])

eval_quality_score = Gauge('genops_eval_quality_score', 'Evaluation quality score', ['model', 'metric'])
eval_latency = Summary('genops_eval_latency_seconds', 'Evaluation latency', ['model'])

def update_metrics():
    """Update metrics with live data from GenOps."""
    # This would integrate with GenOps data sources
    # For demonstration, showing the metric structure

    # Cost metrics
    cost_total.labels(team='ml-platform', project='inference', provider='openai').set(125.50)
    cost_total.labels(team='research', project='experiments', provider='anthropic').set(87.30)
    cost_by_model.labels(model='gpt-4', provider='openai').set(95.20)
    cost_by_model.labels(model='claude-3-opus', provider='anthropic').set(65.10)

    # Budget metrics
    budget_remaining.labels(team='ml-platform', project='inference').set(874.50)
    budget_utilization.labels(team='ml-platform').set(12.55)

    # Policy metrics
    policy_violations.labels(policy_type='budget_exceeded', severity='warning').inc(0)
    policy_enforcements.labels(action='throttle').inc(0)

    # Request metrics
    request_count.labels(provider='openai', model='gpt-4', status='success').inc(1250)
    request_count.labels(provider='anthropic', model='claude-3', status='success').inc(890)

    # Token metrics
    token_usage.labels(provider='openai', model='gpt-4', type='input').inc(125000)
    token_usage.labels(provider='openai', model='gpt-4', type='output').inc(45000)

    # Evaluation metrics
    eval_quality_score.labels(model='gpt-4', metric='relevance').set(0.89)
    eval_quality_score.labels(model='gpt-4', metric='coherence').set(0.92)

def main():
    """Start metrics server and update loop."""
    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    print(f"Metrics server started on port {METRICS_PORT}")

    # Configure OpenTelemetry
    exporter = OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=30000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    # Update metrics periodically
    while True:
        try:
            update_metrics()
            time.sleep(30)
        except Exception as e:
            print(f"Error updating metrics: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
```

Deploy the metrics exporter:

```yaml
# genops-metrics-exporter.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-metrics-exporter
  namespace: genops-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: genops-metrics-exporter
  template:
    metadata:
      labels:
        app: genops-metrics-exporter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: exporter
        image: genopsai/metrics-exporter:latest
        ports:
        - containerPort: 8080
          name: metrics
        env:
        - name: METRICS_PORT
          value: "8080"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://otel-collector.monitoring:4317"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: genops-metrics-exporter
  namespace: genops-system
spec:
  selector:
    app: genops-metrics-exporter
  ports:
  - port: 8080
    targetPort: 8080
    name: metrics
```

## Distributed Tracing

### Jaeger Deployment

Deploy Jaeger for distributed tracing:

```bash
# Deploy Jaeger with production storage
helm install jaeger jaegertracing/jaeger \
  --namespace monitoring \
  --set provisionDataStore.cassandra=false \
  --set storage.type=elasticsearch \
  --set storage.elasticsearch.host=elasticsearch.monitoring \
  --set storage.elasticsearch.port=9200 \
  --set agent.enabled=false \
  --set collector.enabled=true \
  --set query.enabled=true \
  --set ingester.enabled=false

# Verify Jaeger deployment
kubectl get pods -n monitoring -l app.kubernetes.io/instance=jaeger
```

Alternative: Deploy Grafana Tempo for simpler setup:

```yaml
# tempo-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tempo-config
  namespace: monitoring
data:
  tempo.yaml: |
    server:
      http_listen_port: 3200

    distributor:
      receivers:
        otlp:
          protocols:
            grpc:
              endpoint: 0.0.0.0:4317
            http:
              endpoint: 0.0.0.0:4318

    ingester:
      trace_idle_period: 30s
      max_block_duration: 5m

    compactor:
      compaction:
        block_retention: 48h

    storage:
      trace:
        backend: local
        local:
          path: /var/tempo/traces
        wal:
          path: /var/tempo/wal

    overrides:
      per_tenant_override_config: /conf/overrides.yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tempo
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tempo
  template:
    metadata:
      labels:
        app: tempo
    spec:
      containers:
      - name: tempo
        image: grafana/tempo:latest
        args:
        - -config.file=/conf/tempo.yaml
        ports:
        - containerPort: 3200
          name: http
        - containerPort: 4317
          name: otlp-grpc
        - containerPort: 4318
          name: otlp-http
        volumeMounts:
        - name: config
          mountPath: /conf
        - name: storage
          mountPath: /var/tempo
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
      volumes:
      - name: config
        configMap:
          name: tempo-config
      - name: storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: tempo
  namespace: monitoring
spec:
  selector:
    app: tempo
  ports:
  - port: 3200
    name: http
  - port: 4317
    name: otlp-grpc
  - port: 4318
    name: otlp-http
```

### Trace Instrumentation

Configure GenOps AI for distributed tracing:

```python
# genops-tracing-config.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from genops import track_usage, create_governance_context

# Configure resource with governance attributes
resource = Resource.create({
    "service.name": "genops-ai-inference",
    "service.namespace": "genops-system",
    "service.version": "1.0.0",
    "deployment.environment": "production",
    # Governance attributes
    "genops.team": "ml-platform",
    "genops.project": "inference-service",
    "genops.cost_center": "engineering",
})

# Set up tracer provider
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint="http://otel-collector.monitoring:4317",
        insecure=True
    )
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Auto-instrument HTTP libraries
RequestsInstrumentor().instrument()
FlaskInstrumentor().instrument_app(app)

# Get tracer
tracer = trace.get_tracer(__name__)

# Example: Trace AI inference with governance context
@track_usage(team="ml-platform", project="inference-service")
def inference_request(prompt: str, model: str = "gpt-4"):
    """Handle AI inference request with tracing."""

    with tracer.start_as_current_span("inference_request") as span:
        # Add governance context to span
        span.set_attribute("genops.customer_id", "customer-123")
        span.set_attribute("genops.model", model)
        span.set_attribute("genops.prompt_length", len(prompt))

        # Create governance context
        with create_governance_context(
            customer_id="customer-123",
            feature="chat-completion"
        ) as ctx:
            # Make API call (automatically traced)
            response = call_llm_api(prompt, model)

            # Add cost and token metrics to span
            span.set_attribute("genops.cost_usd", ctx.get_cost())
            span.set_attribute("genops.tokens_input", ctx.get_tokens_input())
            span.set_attribute("genops.tokens_output", ctx.get_tokens_output())

            return response

def call_llm_api(prompt: str, model: str):
    """Make LLM API call (automatically traced by instrumentation)."""
    with tracer.start_as_current_span("llm_api_call") as span:
        span.set_attribute("ai.model", model)
        span.set_attribute("ai.prompt.length", len(prompt))

        # Simulate API call
        import requests
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        )

        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("ai.response.length", len(response.text))

        return response.json()
```

### Trace Sampling Strategy

Configure intelligent sampling for high-volume workloads:

```yaml
# trace-sampling-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-sampling-config
  namespace: monitoring
data:
  sampling.yaml: |
    # Sampling configuration for OpenTelemetry Collector
    processors:
      # Probabilistic sampler: sample X% of traces
      probabilistic_sampler:
        sampling_percentage: 10  # 10% of all traces

      # Tail-based sampler: intelligent sampling based on trace characteristics
      tail_sampling:
        decision_wait: 10s  # Wait for complete trace
        num_traces: 10000
        expected_new_traces_per_sec: 100

        policies:
          # Always sample errors
          - name: error-policy
            type: status_code
            status_code:
              status_codes: [ERROR]

          # Always sample slow requests
          - name: latency-policy
            type: latency
            latency:
              threshold_ms: 5000  # > 5 seconds

          # Always sample policy violations
          - name: governance-policy
            type: string_attribute
            string_attribute:
              key: genops.policy_violation
              values: ["true"]
              enabled_regex_matching: false

          # Sample expensive requests
          - name: high-cost-policy
            type: numeric_attribute
            numeric_attribute:
              key: genops.cost_usd
              min_value: 1.00  # Cost > $1.00

          # Sample specific customers at higher rate
          - name: premium-customer-policy
            type: string_attribute
            string_attribute:
              key: genops.customer_tier
              values: ["premium", "enterprise"]

          # Default sampling for everything else
          - name: probabilistic-policy
            type: probabilistic
            probabilistic:
              sampling_percentage: 1  # 1% of normal traffic
```

Apply sampling configuration:

```bash
# Update OpenTelemetry Collector with sampling config
kubectl patch configmap otel-collector-config -n monitoring --patch-file trace-sampling-config.yaml

# Restart collector to apply changes
kubectl rollout restart deployment/otel-collector -n monitoring
```

## Visualization & Dashboards

### Grafana Dashboard Templates

Deploy production-ready Grafana dashboards for GenOps AI:

#### Dashboard 1: Cost Tracking Dashboard

```yaml
# grafana-cost-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-cost-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  genops-cost-tracking.json: |
    {
      "dashboard": {
        "title": "GenOps AI - Cost Tracking",
        "tags": ["genops", "cost", "ai"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Total Cost (Last 24h)",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
            "targets": [{
              "expr": "sum(increase(genops_cost_total_usd[24h]))",
              "legendFormat": "Total Cost"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "currencyUSD",
                "thresholds": {
                  "steps": [
                    {"value": 0, "color": "green"},
                    {"value": 100, "color": "yellow"},
                    {"value": 500, "color": "red"}
                  ]
                }
              }
            }
          },
          {
            "id": 2,
            "title": "Cost by Team",
            "type": "piechart",
            "gridPos": {"x": 6, "y": 0, "w": 6, "h": 8},
            "targets": [{
              "expr": "sum by (team) (genops_cost_total_usd)",
              "legendFormat": "{{team}}"
            }]
          },
          {
            "id": 3,
            "title": "Cost by Provider",
            "type": "bargauge",
            "gridPos": {"x": 12, "y": 0, "w": 6, "h": 8},
            "targets": [{
              "expr": "sum by (provider) (genops_cost_total_usd)",
              "legendFormat": "{{provider}}"
            }],
            "options": {
              "orientation": "horizontal",
              "displayMode": "gradient"
            }
          },
          {
            "id": 4,
            "title": "Cost Over Time",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
            "targets": [{
              "expr": "sum by (team) (rate(genops_cost_total_usd[5m]))",
              "legendFormat": "{{team}}"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "currencyUSD/s"
              }
            }
          },
          {
            "id": 5,
            "title": "Cost by Model",
            "type": "table",
            "gridPos": {"x": 12, "y": 8, "w": 6, "h": 8},
            "targets": [{
              "expr": "sum by (model, provider) (genops_cost_by_model_usd)",
              "format": "table",
              "instant": true
            }],
            "transformations": [{
              "id": "organize",
              "options": {
                "excludeByName": {"Time": true},
                "indexByName": {},
                "renameByName": {
                  "model": "Model",
                  "provider": "Provider",
                  "Value": "Cost (USD)"
                }
              }
            }]
          },
          {
            "id": 6,
            "title": "Budget Utilization",
            "type": "gauge",
            "gridPos": {"x": 18, "y": 0, "w": 6, "h": 8},
            "targets": [{
              "expr": "sum by (team) (genops_budget_utilization_percent)",
              "legendFormat": "{{team}}"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "thresholds": {
                  "steps": [
                    {"value": 0, "color": "green"},
                    {"value": 80, "color": "yellow"},
                    {"value": 95, "color": "red"}
                  ]
                }
              }
            }
          }
        ],
        "refresh": "30s",
        "time": {
          "from": "now-24h",
          "to": "now"
        }
      }
    }
```

#### Dashboard 2: Policy Compliance Dashboard

```yaml
# grafana-policy-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-policy-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  genops-policy-compliance.json: |
    {
      "dashboard": {
        "title": "GenOps AI - Policy Compliance",
        "tags": ["genops", "policy", "compliance"],
        "panels": [
          {
            "id": 1,
            "title": "Policy Violations (Last Hour)",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
            "targets": [{
              "expr": "sum(increase(genops_policy_violations_total[1h]))",
              "legendFormat": "Violations"
            }],
            "fieldConfig": {
              "defaults": {
                "thresholds": {
                  "steps": [
                    {"value": 0, "color": "green"},
                    {"value": 1, "color": "yellow"},
                    {"value": 10, "color": "red"}
                  ]
                }
              }
            }
          },
          {
            "id": 2,
            "title": "Violations by Policy Type",
            "type": "timeseries",
            "gridPos": {"x": 6, "y": 0, "w": 12, "h": 8},
            "targets": [{
              "expr": "sum by (policy_type) (rate(genops_policy_violations_total[5m]))",
              "legendFormat": "{{policy_type}}"
            }]
          },
          {
            "id": 3,
            "title": "Policy Enforcement Actions",
            "type": "piechart",
            "gridPos": {"x": 18, "y": 0, "w": 6, "h": 8},
            "targets": [{
              "expr": "sum by (action) (genops_policy_enforcements_total)",
              "legendFormat": "{{action}}"
            }]
          },
          {
            "id": 4,
            "title": "Compliance Score by Team",
            "type": "bargauge",
            "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
            "targets": [{
              "expr": "100 - (sum by (team) (rate(genops_policy_violations_total[1h])) * 100)",
              "legendFormat": "{{team}}"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 100,
                "thresholds": {
                  "steps": [
                    {"value": 0, "color": "red"},
                    {"value": 90, "color": "yellow"},
                    {"value": 95, "color": "green"}
                  ]
                }
              }
            }
          }
        ]
      }
    }
```

#### Dashboard 3: AI Performance Metrics

```yaml
# grafana-performance-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-performance-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  genops-ai-performance.json: |
    {
      "dashboard": {
        "title": "GenOps AI - Performance Metrics",
        "tags": ["genops", "performance", "ai"],
        "panels": [
          {
            "id": 1,
            "title": "Request Rate",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
            "targets": [{
              "expr": "sum by (provider, model) (rate(genops_requests_total[5m]))",
              "legendFormat": "{{provider}}/{{model}}"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "reqps"
              }
            }
          },
          {
            "id": 2,
            "title": "Average Latency (p50, p95, p99)",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
            "targets": [
              {
                "expr": "histogram_quantile(0.50, sum by (provider, le) (rate(genops_request_duration_seconds_bucket[5m])))",
                "legendFormat": "p50"
              },
              {
                "expr": "histogram_quantile(0.95, sum by (provider, le) (rate(genops_request_duration_seconds_bucket[5m])))",
                "legendFormat": "p95"
              },
              {
                "expr": "histogram_quantile(0.99, sum by (provider, le) (rate(genops_request_duration_seconds_bucket[5m])))",
                "legendFormat": "p99"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "s"
              }
            }
          },
          {
            "id": 3,
            "title": "Token Usage Rate",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
            "targets": [{
              "expr": "sum by (type) (rate(genops_tokens_total[5m]))",
              "legendFormat": "{{type}} tokens"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "tokens/s"
              }
            }
          },
          {
            "id": 4,
            "title": "Quality Scores",
            "type": "gauge",
            "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
            "targets": [{
              "expr": "genops_eval_quality_score",
              "legendFormat": "{{metric}}"
            }],
            "fieldConfig": {
              "defaults": {
                "unit": "percentunit",
                "min": 0,
                "max": 1,
                "thresholds": {
                  "steps": [
                    {"value": 0, "color": "red"},
                    {"value": 0.7, "color": "yellow"},
                    {"value": 0.85, "color": "green"}
                  ]
                }
              }
            }
          }
        ]
      }
    }
```

Apply dashboard configurations:

```bash
# Apply all dashboards
kubectl apply -f grafana-cost-dashboard.yaml
kubectl apply -f grafana-policy-dashboard.yaml
kubectl apply -f grafana-performance-dashboard.yaml

# Restart Grafana to load dashboards
kubectl rollout restart deployment/monitoring-grafana -n monitoring
```

### Dashboard as Code

Automate dashboard provisioning with Terraform:

```hcl
# terraform-grafana-dashboards.tf
provider "grafana" {
  url  = "http://grafana.monitoring:3000"
  auth = var.grafana_api_key
}

resource "grafana_dashboard" "genops_cost" {
  config_json = file("${path.module}/dashboards/genops-cost-tracking.json")
  folder      = grafana_folder.genops.id
}

resource "grafana_dashboard" "genops_policy" {
  config_json = file("${path.module}/dashboards/genops-policy-compliance.json")
  folder      = grafana_folder.genops.id
}

resource "grafana_dashboard" "genops_performance" {
  config_json = file("${path.module}/dashboards/genops-ai-performance.json")
  folder      = grafana_folder.genops.id
}

resource "grafana_folder" "genops" {
  title = "GenOps AI"
}

resource "grafana_data_source" "prometheus" {
  type = "prometheus"
  name = "Prometheus"
  url  = "http://prometheus-kube-prometheus-prometheus.monitoring:9090"

  json_data {
    http_method     = "POST"
    timeout_seconds = 60
  }
}

resource "grafana_data_source" "tempo" {
  type = "tempo"
  name = "Tempo"
  url  = "http://tempo.monitoring:3200"

  json_data {
    http_method = "GET"
    trace_id_tag = "trace_id"
  }
}
```

## Alerting & Incident Response

### Prometheus Alert Rules

Configure comprehensive alerting rules:

```yaml
# prometheus-alert-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-alert-rules
  namespace: monitoring
data:
  genops-alerts.yaml: |
    groups:
    - name: genops-cost-alerts
      interval: 30s
      rules:
      # Alert when 80% of budget consumed
      - alert: BudgetThreshold80Percent
        expr: genops_budget_utilization_percent > 80
        for: 5m
        labels:
          severity: warning
          category: cost
        annotations:
          summary: "Budget threshold exceeded ({{ $labels.team }})"
          description: "Team {{ $labels.team }} has consumed {{ $value }}% of monthly budget"

      # Alert when 95% of budget consumed
      - alert: BudgetThreshold95Percent
        expr: genops_budget_utilization_percent > 95
        for: 5m
        labels:
          severity: critical
          category: cost
        annotations:
          summary: "Budget critically low ({{ $labels.team }})"
          description: "Team {{ $labels.team }} has consumed {{ $value }}% of monthly budget - URGENT"

      # Alert when budget fully consumed
      - alert: BudgetExhausted
        expr: genops_budget_remaining_usd <= 0
        for: 1m
        labels:
          severity: critical
          category: cost
        annotations:
          summary: "Budget exhausted ({{ $labels.team }}/{{ $labels.project }})"
          description: "Budget for {{ $labels.team }}/{{ $labels.project }} has been fully consumed"

      # Alert on cost anomalies (50% increase)
      - alert: CostAnomaly
        expr: |
          (
            sum by (team) (rate(genops_cost_total_usd[1h]))
            >
            sum by (team) (rate(genops_cost_total_usd[1h] offset 24h)) * 1.5
          )
        for: 30m
        labels:
          severity: warning
          category: cost
        annotations:
          summary: "Unusual cost increase detected ({{ $labels.team }})"
          description: "Cost for {{ $labels.team }} has increased by 50% compared to yesterday"

    - name: genops-policy-alerts
      interval: 30s
      rules:
      # Alert on policy violations
      - alert: PolicyViolationsDetected
        expr: rate(genops_policy_violations_total[5m]) > 0
        for: 5m
        labels:
          severity: warning
          category: policy
        annotations:
          summary: "Policy violations detected ({{ $labels.policy_type }})"
          description: "{{ $value }} violations/sec of policy type {{ $labels.policy_type }}"

      # Alert on high violation rate
      - alert: HighPolicyViolationRate
        expr: rate(genops_policy_violations_total[5m]) > 0.1
        for: 10m
        labels:
          severity: critical
          category: policy
        annotations:
          summary: "High rate of policy violations"
          description: "Policy violation rate is {{ $value }} violations/sec - investigate immediately"

    - name: genops-performance-alerts
      interval: 30s
      rules:
      # Alert on high latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, sum by (provider, le) (rate(genops_request_duration_seconds_bucket[5m]))) > 10
        for: 10m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "High latency detected ({{ $labels.provider }})"
          description: "P95 latency for {{ $labels.provider }} is {{ $value }}s"

      # Alert on high error rate
      - alert: HighErrorRate
        expr: |
          (
            sum by (provider) (rate(genops_requests_total{status="error"}[5m]))
            /
            sum by (provider) (rate(genops_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          category: performance
        annotations:
          summary: "High error rate ({{ $labels.provider }})"
          description: "Error rate for {{ $labels.provider }} is {{ $value | humanizePercentage }}"

      # Alert on low quality scores
      - alert: LowQualityScore
        expr: genops_eval_quality_score < 0.7
        for: 15m
        labels:
          severity: warning
          category: quality
        annotations:
          summary: "Low quality score ({{ $labels.model }}/{{ $labels.metric }})"
          description: "Quality score for {{ $labels.model }} {{ $labels.metric }} is {{ $value }}"

    - name: genops-infrastructure-alerts
      interval: 30s
      rules:
      # Alert when GenOps pods are not running
      - alert: GenOpsPodDown
        expr: kube_pod_status_phase{namespace="genops-system", phase!="Running"} > 0
        for: 5m
        labels:
          severity: critical
          category: infrastructure
        annotations:
          summary: "GenOps pod not running ({{ $labels.pod }})"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is not running"

      # Alert on high memory usage
      - alert: HighMemoryUsage
        expr: |
          (
            container_memory_working_set_bytes{namespace="genops-system"}
            /
            container_spec_memory_limit_bytes{namespace="genops-system"}
          ) > 0.9
        for: 5m
        labels:
          severity: warning
          category: infrastructure
        annotations:
          summary: "High memory usage ({{ $labels.pod }})"
          description: "Pod {{ $labels.pod }} is using {{ $value | humanizePercentage }} of memory limit"

      # Alert on OTel Collector issues
      - alert: OTelCollectorDown
        expr: up{job="otel-collector"} == 0
        for: 5m
        labels:
          severity: critical
          category: observability
        annotations:
          summary: "OpenTelemetry Collector is down"
          description: "The OTel Collector has been down for more than 5 minutes - telemetry collection is impaired"
```

Apply alert rules:

```bash
# Apply alert rules to Prometheus
kubectl apply -f prometheus-alert-rules.yaml

# Reload Prometheus configuration
kubectl exec -n monitoring prometheus-monitoring-kube-prometheus-prometheus-0 -- \
  curl -X POST http://localhost:9090/-/reload
```

### AlertManager Configuration

Configure alert routing and notifications:

```yaml
# alertmanager-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-genops-config
  namespace: monitoring
stringData:
  alertmanager.yaml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

    route:
      group_by: ['alertname', 'cluster', 'team']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'default'
      routes:
      # Critical cost alerts
      - match:
          severity: critical
          category: cost
        receiver: 'cost-critical'
        group_wait: 1m
        continue: false

      # Warning cost alerts
      - match:
          severity: warning
          category: cost
        receiver: 'cost-warning'
        continue: true

      # Policy violations
      - match:
          category: policy
        receiver: 'policy-violations'
        continue: true

      # Performance issues
      - match:
          category: performance
        receiver: 'performance-alerts'
        continue: true

      # Infrastructure issues
      - match:
          category: infrastructure
        receiver: 'infrastructure-alerts'
        continue: false

    receivers:
    - name: 'default'
      slack_configs:
      - channel: '#genops-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    - name: 'cost-critical'
      slack_configs:
      - channel: '#genops-cost-critical'
        title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        actions:
        - type: button
          text: 'View Dashboard'
          url: 'http://grafana.monitoring/d/genops-cost'
        - type: button
          text: 'Runbook'
          url: 'https://wiki.example.com/runbooks/budget-exhausted'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        severity: 'critical'
      email_configs:
      - to: 'platform-team@example.com'
        subject: 'CRITICAL: GenOps Budget Alert'

    - name: 'cost-warning'
      slack_configs:
      - channel: '#genops-cost'
        title: '⚠️  WARNING: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    - name: 'policy-violations'
      slack_configs:
      - channel: '#genops-policy'
        title: '📋 Policy: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
      webhook_configs:
      - url: 'http://genops-ai.genops-system/api/policy/alert'
        send_resolved: true

    - name: 'performance-alerts'
      slack_configs:
      - channel: '#genops-performance'
        title: '⚡ Performance: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    - name: 'infrastructure-alerts'
      slack_configs:
      - channel: '#genops-infrastructure'
        title: '🔧 Infrastructure: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        severity: 'warning'

    inhibit_rules:
    # Inhibit warning if critical is firing
    - source_match:
        severity: 'critical'
      target_match:
        severity: 'warning'
      equal: ['alertname', 'team', 'project']

    # Inhibit budget threshold alerts if budget is exhausted
    - source_match:
        alertname: 'BudgetExhausted'
      target_match_re:
        alertname: 'BudgetThreshold.*'
      equal: ['team', 'project']
```

Apply AlertManager configuration:

```bash
kubectl apply -f alertmanager-config.yaml

# Restart AlertManager
kubectl rollout restart statefulset/alertmanager-monitoring-kube-prometheus-alertmanager -n monitoring
```

### Incident Response Runbooks

Create automated runbooks for common alerts:

```yaml
# incident-response-runbook.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-runbooks
  namespace: monitoring
data:
  budget-exhausted.md: |
    # Runbook: Budget Exhausted

    ## Alert Description
    A team or project has exhausted their monthly AI budget.

    ## Impact
    - AI operations for the affected team/project will be throttled or blocked
    - Users may experience service degradation

    ## Investigation Steps

    1. Check current cost breakdown:
       ```bash
       kubectl exec -n genops-system deployment/genops-ai -- \
         genops-cli cost-report --team <TEAM> --period today
       ```

    2. Identify cost drivers:
       ```bash
       # View top models by cost
       kubectl exec -n genops-system deployment/genops-ai -- \
         genops-cli cost-breakdown --by model --team <TEAM>

       # View top customers by cost
       kubectl exec -n genops-system deployment/genops-ai -- \
         genops-cli cost-breakdown --by customer --team <TEAM>
       ```

    3. Check for anomalies:
       - Unexpected spike in API calls?
       - New expensive model deployed?
       - Cost optimization disabled?

    ## Remediation Steps

    ### Immediate Actions
    1. Increase budget temporarily (if approved):
       ```bash
       kubectl exec -n genops-system deployment/genops-ai -- \
         genops-cli budget-update --team <TEAM> --limit <NEW_LIMIT>
       ```

    2. Enable cost optimization:
       ```bash
       kubectl patch deployment genops-ai -n genops-system --patch '{
         "spec": {
           "template": {
             "spec": {
               "containers": [{
                 "name": "genops-ai",
                 "env": [{"name": "COST_OPTIMIZATION_ENABLED", "value": "true"}]
               }]
             }
           }
         }
       }'
       ```

    ### Long-term Actions
    1. Review and optimize expensive operations
    2. Implement model selection based on task complexity
    3. Set up graduated budget alerts (50%, 75%, 90%)
    4. Review pricing with AI providers

    ## Escalation
    - Notify: FinOps team, Engineering Manager
    - PagerDuty: Yes (if critical service)
    - Slack: #genops-cost-critical

  high-latency.md: |
    # Runbook: High Latency

    ## Alert Description
    P95 latency for AI requests has exceeded threshold (>10 seconds).

    ## Investigation Steps

    1. Check current latency:
       ```bash
       # Get latency metrics
       kubectl exec -n monitoring prometheus-monitoring-kube-prometheus-prometheus-0 -- \
         promtool query instant \
         'histogram_quantile(0.95, sum by (provider, le) (rate(genops_request_duration_seconds_bucket[5m])))'
       ```

    2. Identify slow provider:
       - Check dashboard: http://grafana.monitoring/d/genops-performance
       - Review traces in Jaeger: http://jaeger.monitoring:16686

    3. Check infrastructure health:
       ```bash
       kubectl top pods -n genops-system
       kubectl get pods -n genops-system
       ```

    ## Remediation Steps

    1. Scale up if resource constrained:
       ```bash
       kubectl scale deployment genops-ai --replicas=5 -n genops-system
       ```

    2. Enable request queueing:
       ```bash
       kubectl patch configmap genops-config -n genops-system --patch '{
         "data": {"ENABLE_REQUEST_QUEUE": "true", "MAX_QUEUE_SIZE": "100"}
       }'
       ```

    3. Route traffic to faster provider:
       ```bash
       kubectl patch configmap genops-routing -n genops-system --patch '{
         "data": {"FALLBACK_PROVIDER": "anthropic"}
       }'
       ```

    ## Escalation
    - Notify: Platform team, On-call engineer
    - PagerDuty: If latency >30s
    - Slack: #genops-performance
```

## Log Aggregation

### Loki Deployment

Deploy Grafana Loki for log aggregation:

```bash
# Install Loki with Helm
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=50Gi \
  --set promtail.enabled=true \
  --set grafana.enabled=false

# Verify Loki deployment
kubectl get pods -n monitoring -l app=loki
```

### Promtail Configuration

Configure Promtail to collect GenOps logs:

```yaml
# promtail-genops-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-genops-config
  namespace: monitoring
data:
  promtail.yaml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    positions:
      filename: /tmp/positions.yaml

    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
      # Scrape GenOps system logs
      - job_name: genops-system
        kubernetes_sd_configs:
        - role: pod
          namespaces:
            names:
            - genops-system
        pipeline_stages:
        # Parse JSON logs
        - json:
            expressions:
              level: level
              timestamp: timestamp
              message: message
              team: governance.team
              project: governance.project
              cost: governance.cost_usd

        # Extract governance context
        - labels:
            team:
            project:
            level:

        # Parse cost from logs
        - regex:
            expression: '.*cost_usd=(?P<cost>[0-9.]+)'
            source: message

        # Add metrics
        - metrics:
            cost_total:
              type: Counter
              description: "Total cost from logs"
              source: cost
              config:
                action: add

        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          target_label: app
        - source_labels: [__meta_kubernetes_namespace]
          target_label: namespace
        - source_labels: [__meta_kubernetes_pod_name]
          target_label: pod

      # Scrape audit logs
      - job_name: genops-audit
        kubernetes_sd_configs:
        - role: pod
          namespaces:
            names:
            - genops-system
        pipeline_stages:
        - json:
            expressions:
              level: level
              action: audit.action
              resource: audit.resource
              user: audit.user
              result: audit.result

        - labels:
            action:
            user:
            result:

        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_audit_enabled]
          action: keep
          regex: true
```

### Structured Logging Best Practices

Implement structured logging in GenOps applications:

```python
# structured-logging-example.py
import logging
import json
from pythonjsonlogger import jsonlogger
from opentelemetry import trace

class GenOpsJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with governance context."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record['timestamp'] = record.created

        # Add trace context
        span = trace.get_current_span()
        if span:
            span_context = span.get_span_context()
            log_record['trace_id'] = format(span_context.trace_id, '032x')
            log_record['span_id'] = format(span_context.span_id, '016x')

        # Add governance context
        from genops import get_current_governance_context
        ctx = get_current_governance_context()
        if ctx:
            log_record['governance'] = {
                'team': ctx.team,
                'project': ctx.project,
                'customer_id': ctx.customer_id,
                'cost_usd': ctx.get_cost()
            }

# Configure logger
logger = logging.getLogger('genops')
handler = logging.StreamHandler()
handler.setFormatter(GenOpsJSONFormatter('%(timestamp)s %(level)s %(name)s %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
@track_usage(team="ml-platform", project="inference")
def process_request(prompt: str):
    """Process AI request with structured logging."""

    logger.info("Processing request", extra={
        "prompt_length": len(prompt),
        "model": "gpt-4"
    })

    try:
        result = call_llm_api(prompt)
        logger.info("Request completed successfully", extra={
            "tokens_used": result.tokens,
            "cost_usd": result.cost
        })
        return result
    except Exception as e:
        logger.error("Request failed", extra={
            "error": str(e),
            "error_type": type(e).__name__
        }, exc_info=True)
        raise
```

## Platform Integration

### Datadog Integration

Integrate GenOps telemetry with Datadog:

```yaml
# datadog-integration.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: datadog-genops-config
  namespace: monitoring
data:
  genops-integration.yaml: |
    init_config:

    instances:
      # Prometheus metrics integration
      - prometheus_url: http://genops-metrics-exporter.genops-system:8080/metrics
        namespace: "genops"
        metrics:
          - genops_cost_*
          - genops_budget_*
          - genops_policy_*
          - genops_requests_*
          - genops_tokens_*
          - genops_eval_*
        type_overrides:
          genops_cost_total_usd: gauge
          genops_budget_remaining_usd: gauge
          genops_policy_violations_total: count
        tags:
          - team:$team
          - project:$project
          - environment:$environment

      # OpenTelemetry traces
      - otlp_config:
          receiver:
            protocols:
              grpc:
                endpoint: 0.0.0.0:4317
          exporter:
            datadog:
              api:
                key: ${DD_API_KEY}
                site: datadoghq.com
```

### Honeycomb Integration

Send structured events to Honeycomb:

```python
# honeycomb-integration.py
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure Honeycomb OTLP export
headers = {
    "x-honeycomb-team": os.getenv("HONEYCOMB_API_KEY"),
    "x-honeycomb-dataset": "genops-ai"
}

exporter = OTLPSpanExporter(
    endpoint="https://api.honeycomb.io:443",
    headers=headers
)

provider = TracerProvider()
processor = BatchSpanProcessor(exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
```

## Troubleshooting

### Common Observability Issues

#### Issue: Metrics Not Appearing in Prometheus

**Diagnosis:**
```bash
# Check if OTel Collector is running
kubectl get pods -n monitoring -l app=otel-collector

# Check collector logs
kubectl logs -n monitoring -l app=otel-collector --tail=100

# Verify metrics endpoint is accessible
kubectl exec -n monitoring deployment/prometheus-monitoring-kube-prometheus-prometheus -- \
  curl -s http://otel-collector.monitoring:8889/metrics | head -20

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-monitoring-kube-prometheus-prometheus 9090:9090 &
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="genops-ai")'
```

**Solutions:**

1. **Verify OTLP endpoint configuration:**
   ```bash
   kubectl get pods -n genops-system -o yaml | grep OTEL_EXPORTER
   ```

2. **Check ServiceMonitor:**
   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: genops-metrics
     namespace: genops-system
   spec:
     selector:
       matchLabels:
         app: genops-ai
     endpoints:
     - port: metrics
       interval: 30s
   ```

3. **Restart collector:**
   ```bash
   kubectl rollout restart deployment/otel-collector -n monitoring
   ```

#### Issue: Traces Not Appearing in Jaeger

**Diagnosis:**
```bash
# Check Jaeger collector status
kubectl get pods -n monitoring -l app.kubernetes.io/component=collector

# Check collector logs
kubectl logs -n monitoring -l app.kubernetes.io/component=collector

# Verify OTLP endpoint
kubectl port-forward -n monitoring svc/jaeger-collector 14268:14268 &
curl http://localhost:14268/api/traces

# Check trace export from application
kubectl logs -n genops-system -l app=genops-ai | grep -i trace
```

**Solutions:**

1. **Verify trace sampling is not too restrictive:**
   ```bash
   kubectl get configmap otel-collector-config -n monitoring -o yaml | grep -A 20 "tail_sampling"
   ```

2. **Increase sampling rate temporarily:**
   ```bash
   kubectl patch configmap otel-collector-config -n monitoring --patch '{
     "data": {
       "sampling_percentage": "100"
     }
   }'
   ```

3. **Check application instrumentation:**
   ```python
   # Verify tracer is configured
   from opentelemetry import trace
   tracer = trace.get_tracer(__name__)
   print(f"Tracer configured: {tracer is not None}")
   ```

#### Issue: High Cardinality Metrics

**Diagnosis:**
```bash
# Check metric cardinality
kubectl exec -n monitoring prometheus-monitoring-kube-prometheus-prometheus-0 -- \
  promtool tsdb analyze /prometheus

# Find high-cardinality metrics
kubectl port-forward -n monitoring svc/prometheus-monitoring-kube-prometheus-prometheus 9090:9090 &
curl 'http://localhost:9090/api/v1/label/__name__/values' | jq -r '.data[]' | while read metric; do
  echo "$metric: $(curl -s "http://localhost:9090/api/v1/query?query=count($metric)" | jq '.data.result[0].value[1]')"
done | sort -t: -k2 -n | tail -20
```

**Solutions:**

1. **Drop high-cardinality labels:**
   ```yaml
   # In prometheus-genops-config.yaml
   metric_relabel_configs:
   - source_labels: [customer_id]  # Example: drop customer_id label
     action: labeldrop
   ```

2. **Aggregate metrics:**
   ```yaml
   # Use recording rules to pre-aggregate
   - record: genops:cost:team_total
     expr: sum by (team) (genops_cost_total_usd)
   ```

3. **Implement sampling:**
   ```bash
   # Reduce scrape frequency for high-volume metrics
   kubectl patch servicemonitor genops-metrics -n genops-system --patch '{
     "spec": {
       "endpoints": [{
         "interval": "60s"
       }]
     }
   }'
   ```

---

## Next Steps

1. **Deploy full observability stack** - Prometheus, Grafana, Jaeger, Loki
2. **Import GenOps dashboards** - Cost tracking, policy compliance, performance
3. **Configure alerts** - Budget thresholds, policy violations, performance degradation
4. **Set up log aggregation** - Centralized logging with structured logs
5. **Integrate with platforms** - Datadog, Honeycomb, or Splunk
6. **Create runbooks** - Incident response procedures for common alerts

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboard Gallery](https://grafana.com/grafana/dashboards/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Docs](https://opentelemetry.io/docs/)
- [Loki Documentation](https://grafana.com/docs/loki/)
- [GenOps AI Documentation](https://github.com/KoshiHQ/GenOps-AI)

---

This guide provides a comprehensive foundation for implementing production-grade observability for GenOps AI on Kubernetes with metrics, tracing, logging, and alerting.
