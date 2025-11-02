# Production Monitoring Guide for GenOps OpenRouter

This guide provides comprehensive monitoring, alerting, and observability configurations for production GenOps OpenRouter deployments.

## Table of Contents

- [Overview](#overview)
- [OpenTelemetry Configuration](#opentelemetry-configuration)
- [Platform Integrations](#platform-integrations)
- [Key Metrics and Alerts](#key-metrics-and-alerts)
- [Dashboard Configuration](#dashboard-configuration)
- [Log Management](#log-management)
- [Performance Monitoring](#performance-monitoring)
- [Cost Monitoring](#cost-monitoring)
- [Troubleshooting Runbooks](#troubleshooting-runbooks)

## Overview

### Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  OpenTelemetry  │───▶│  Observability  │
│   (OpenRouter)  │    │    Collector    │    │   Platform      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                      │
         ▼                        ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Metrics     │    │      Traces     │    │      Logs       │
│   (Prometheus)  │    │    (Jaeger)     │    │  (Elasticsearch)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Monitoring Objectives

- **Service Health**: Uptime, response times, error rates
- **Cost Attribution**: Per-team, per-project, per-customer costs
- **Provider Performance**: OpenRouter provider selection and performance
- **Capacity Planning**: Resource utilization and scaling metrics
- **Security Monitoring**: Authentication failures, rate limiting
- **Business Intelligence**: Usage patterns and optimization opportunities

## OpenTelemetry Configuration

### Core Configuration

Create `otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
  
  prometheus:
    config:
      scrape_configs:
        - job_name: 'openrouter-service'
          static_configs:
            - targets: ['openrouter-service:8000']
          scrape_interval: 30s
          metrics_path: /metrics

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  resource:
    attributes:
      - key: service.name
        value: genops-openrouter
        action: upsert
      - key: service.version
        from_attribute: service_version
        action: upsert
      - key: environment
        from_attribute: deployment_environment
        action: upsert
  
  # GenOps-specific processors
  attributes/genops:
    actions:
      - key: genops.cost.currency
        value: "USD"
        action: upsert
      - key: genops.provider
        value: "openrouter"
        action: upsert
  
  # Cost attribution processor
  transform:
    metric_statements:
      - context: metric
        statements:
          - set(description, "OpenRouter request cost in USD") where name == "genops.cost.total"
          - set(unit, "USD") where name == "genops.cost.total"

exporters:
  # Honeycomb
  otlp/honeycomb:
    endpoint: https://api.honeycomb.io:443
    headers:
      x-honeycomb-team: "${HONEYCOMB_API_KEY}"
  
  # Datadog
  datadog:
    api:
      key: "${DATADOG_API_KEY}"
      site: datadoghq.com
    hostname: "openrouter-collector"
  
  # Prometheus
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: genops_openrouter
    const_labels:
      service: "openrouter"
  
  # Jaeger
  jaeger:
    endpoint: jaeger-collector:14250
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, resource, attributes/genops]
      exporters: [otlp/honeycomb, jaeger]
    
    metrics:
      receivers: [otlp, prometheus]
      processors: [batch, resource, attributes/genops, transform]
      exporters: [otlp/honeycomb, datadog, prometheus]
    
    logs:
      receivers: [otlp]
      processors: [batch, resource]
      exporters: [otlp/honeycomb]

extensions:
  health_check:
    endpoint: 0.0.0.0:13133
  
  pprof:
    endpoint: 0.0.0.0:1777
  
  zpages:
    endpoint: 0.0.0.0:55679
```

### Kubernetes Deployment

Create `monitoring/otel-collector-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: genops-monitoring
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
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-contrib:0.89.0
        command:
          - "/otelcol-contrib"
          - "--config=/conf/otel-collector-config.yaml"
        env:
        - name: HONEYCOMB_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: honeycomb-api-key
        - name: DATADOG_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: datadog-api-key
        ports:
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 4318  # OTLP HTTP
        - containerPort: 8889  # Prometheus metrics
        - containerPort: 13133 # Health check
        volumeMounts:
        - name: config
          mountPath: /conf
        resources:
          limits:
            memory: 512Mi
            cpu: 500m
          requests:
            memory: 256Mi
            cpu: 250m
      volumes:
      - name: config
        configMap:
          name: otel-collector-config

---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: genops-monitoring
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: prometheus
    port: 8889
    targetPort: 8889
```

## Platform Integrations

### Honeycomb Integration

#### Custom Queries for OpenRouter

```sql
-- Request rate by team
COUNT 
| WHERE genops.provider = "openrouter" 
| GROUP BY genops.team 
| ORDER BY COUNT DESC

-- High-cost requests analysis
AVG(genops.cost.total), MAX(genops.cost.total), COUNT 
| WHERE genops.cost.total > 0.01 
| GROUP BY genops.model, genops.openrouter.actual_provider 
| ORDER BY AVG(genops.cost.total) DESC

-- Provider routing effectiveness
COUNT 
| WHERE genops.openrouter.predicted_provider != genops.openrouter.actual_provider 
| GROUP BY genops.openrouter.predicted_provider, genops.openrouter.actual_provider

-- Error rate by endpoint and model
COUNT 
| WHERE http.status_code >= 400 
| GROUP BY http.route, genops.model 
| ORDER BY COUNT DESC

-- Performance by provider
P95(duration_ms), P50(duration_ms), COUNT 
| WHERE genops.provider = "openrouter" 
| GROUP BY genops.openrouter.actual_provider 
| ORDER BY P95(duration_ms) DESC

-- Cost optimization opportunities
SUM(genops.cost.total), COUNT 
| WHERE genops.provider = "openrouter" 
| GROUP BY genops.model 
| HAVING SUM(genops.cost.total) > 1.0 
| ORDER BY SUM(genops.cost.total) DESC
```

#### Honeycomb Triggers

```json
{
  "triggers": [
    {
      "name": "OpenRouter High Error Rate",
      "query": "COUNT | WHERE genops.provider = 'openrouter' AND http.status_code >= 400 | GROUP BY time_bucket(60)",
      "threshold": {
        "op": ">",
        "value": 10
      },
      "frequency": "60s",
      "alert_type": "on_change"
    },
    {
      "name": "OpenRouter High Cost Alert",
      "query": "SUM(genops.cost.total) | WHERE genops.provider = 'openrouter' | GROUP BY time_bucket(300), genops.team",
      "threshold": {
        "op": ">",
        "value": 10.0
      },
      "frequency": "300s",
      "alert_type": "on_true"
    },
    {
      "name": "OpenRouter Provider Failover",
      "query": "COUNT | WHERE genops.openrouter.fallback_used = true | GROUP BY time_bucket(60)",
      "threshold": {
        "op": ">",
        "value": 5
      },
      "frequency": "60s",
      "alert_type": "on_true"
    }
  ]
}
```

### Datadog Integration

#### Custom Metrics Dashboard

```json
{
  "title": "GenOps OpenRouter Production Dashboard",
  "widgets": [
    {
      "definition": {
        "type": "timeseries",
        "title": "Request Rate by Team",
        "requests": [
          {
            "q": "sum:genops_openrouter.requests.total{service:openrouter} by {genops_team}.as_rate()",
            "display_type": "line"
          }
        ]
      }
    },
    {
      "definition": {
        "type": "timeseries",
        "title": "Cost per Request",
        "requests": [
          {
            "q": "avg:genops_openrouter.cost.total{service:openrouter} by {genops_model}",
            "display_type": "line"
          }
        ]
      }
    },
    {
      "definition": {
        "type": "toplist",
        "title": "Top Models by Usage",
        "requests": [
          {
            "q": "top(sum:genops_openrouter.requests.total{service:openrouter} by {genops_model}.as_count(), 10, 'sum', 'desc')"
          }
        ]
      }
    },
    {
      "definition": {
        "type": "heatmap",
        "title": "Response Time Distribution",
        "requests": [
          {
            "q": "avg:genops_openrouter.request.duration{service:openrouter} by {genops_openrouter_actual_provider}"
          }
        ]
      }
    }
  ]
}
```

#### Datadog Monitors

```json
{
  "monitors": [
    {
      "name": "OpenRouter High Error Rate",
      "type": "metric alert",
      "query": "avg(last_5m):sum:genops_openrouter.requests.total{service:openrouter,http_status_code:5xx}.as_rate() > 0.05",
      "message": "@slack-#alerts OpenRouter error rate is above 5%",
      "options": {
        "thresholds": {
          "critical": 0.05,
          "warning": 0.02
        }
      }
    },
    {
      "name": "OpenRouter High Latency",
      "type": "metric alert",
      "query": "avg(last_10m):avg:genops_openrouter.request.duration{service:openrouter} > 5000",
      "message": "@pagerduty-openrouter OpenRouter response time is above 5 seconds",
      "options": {
        "thresholds": {
          "critical": 5000,
          "warning": 3000
        }
      }
    },
    {
      "name": "OpenRouter Daily Cost Threshold",
      "type": "metric alert",
      "query": "sum(last_1d):sum:genops_openrouter.cost.total{service:openrouter} by {genops_team} > 100",
      "message": "@email-finance Team {{genops_team.name}} OpenRouter daily cost exceeded $100",
      "options": {
        "thresholds": {
          "critical": 100,
          "warning": 75
        }
      }
    }
  ]
}
```

### Grafana Configuration

#### OpenRouter Dashboard JSON

Create `monitoring/grafana/openrouter-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "GenOps OpenRouter Production Dashboard",
    "tags": ["genops", "openrouter", "ai"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(genops_openrouter_requests_total[5m])",
            "legendFormat": "{{genops_team}} - {{genops_model}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 10},
                {"color": "red", "value": 50}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Cost Tracking",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(genops_openrouter_cost_total) by (genops_team)",
            "legendFormat": "{{genops_team}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "linear",
              "pointSize": 5
            },
            "color": {
              "mode": "palette-classic"
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Provider Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum(genops_openrouter_requests_total) by (genops_openrouter_actual_provider)",
            "legendFormat": "{{genops_openrouter_actual_provider}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(genops_openrouter_requests_total{http_status_code=~\"5..\"}[5m]) / rate(genops_openrouter_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {
                "params": [0.05],
                "type": "gt"
              },
              "operator": {
                "type": "and"
              },
              "query": {
                "params": ["A", "5m", "now"]
              },
              "reducer": {
                "params": [],
                "type": "avg"
              },
              "type": "query"
            }
          ]
        }
      },
      {
        "id": 5,
        "title": "Response Time Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "sum(rate(genops_openrouter_request_duration_seconds_bucket[5m])) by (le)",
            "format": "heatmap",
            "legendFormat": "{{le}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

#### Prometheus Rules

Create `monitoring/prometheus/openrouter-rules.yaml`:

```yaml
groups:
  - name: genops_openrouter
    rules:
      # Request rate
      - record: genops_openrouter:request_rate
        expr: rate(genops_openrouter_requests_total[5m])
      
      # Error rate
      - record: genops_openrouter:error_rate
        expr: rate(genops_openrouter_requests_total{http_status_code=~"5.."}[5m]) / rate(genops_openrouter_requests_total[5m])
      
      # Cost per request
      - record: genops_openrouter:cost_per_request
        expr: rate(genops_openrouter_cost_total[5m]) / rate(genops_openrouter_requests_total[5m])
      
      # Provider success rate
      - record: genops_openrouter:provider_success_rate
        expr: 1 - (rate(genops_openrouter_requests_total{http_status_code=~"5.."}[5m]) / rate(genops_openrouter_requests_total[5m]))
      
      # Daily cost by team
      - record: genops_openrouter:daily_cost_by_team
        expr: increase(genops_openrouter_cost_total[1d])

  - name: genops_openrouter_alerts
    rules:
      # High error rate alert
      - alert: OpenRouterHighErrorRate
        expr: genops_openrouter:error_rate > 0.05
        for: 2m
        labels:
          severity: warning
          service: openrouter
        annotations:
          summary: "OpenRouter error rate is high"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 2 minutes"
      
      # High latency alert
      - alert: OpenRouterHighLatency
        expr: histogram_quantile(0.95, rate(genops_openrouter_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
          service: openrouter
        annotations:
          summary: "OpenRouter latency is high"
          description: "95th percentile latency is {{ $value }}s"
      
      # Cost threshold alert
      - alert: OpenRouterHighDailyCost
        expr: genops_openrouter:daily_cost_by_team > 100
        for: 0m
        labels:
          severity: info
          service: openrouter
        annotations:
          summary: "OpenRouter daily cost threshold exceeded"
          description: "Team {{ $labels.genops_team }} daily cost is ${{ $value }}"
      
      # Service down alert
      - alert: OpenRouterServiceDown
        expr: up{job="openrouter-service"} == 0
        for: 1m
        labels:
          severity: critical
          service: openrouter
        annotations:
          summary: "OpenRouter service is down"
          description: "OpenRouter service has been down for more than 1 minute"
```

## Key Metrics and Alerts

### Core Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `genops_openrouter_requests_total` | Total requests by team/model | > 1000/min |
| `genops_openrouter_request_duration_seconds` | Request latency | p95 > 5s |
| `genops_openrouter_cost_total` | Cost in USD | > $100/day |
| `genops_openrouter_errors_total` | Error count by type | > 5% error rate |
| `genops_openrouter_provider_switches_total` | Provider failovers | > 10/hour |

### Business Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| Cost per team | Daily/monthly spend by team | Budget tracking |
| Model utilization | Usage distribution across models | Optimization |
| Provider performance | Latency/cost by provider | Routing decisions |
| Error attribution | Errors by team/project | SLA tracking |
| Token efficiency | Cost per token by model | Cost optimization |

### Alert Configuration

#### Slack Integration

Create `monitoring/alerts/slack-webhook.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-slack
  namespace: monitoring
type: Opaque
stringData:
  webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      slack_api_url: '/secrets/webhook_url'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'default'
      routes:
      - match:
          severity: critical
        receiver: 'critical'
      - match:
          service: openrouter
        receiver: 'openrouter-alerts'
    
    receivers:
    - name: 'default'
      slack_configs:
      - channel: '#alerts'
        title: 'GenOps Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    
    - name: 'critical'
      slack_configs:
      - channel: '#critical-alerts'
        title: 'CRITICAL: GenOps Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
        send_resolved: true
    
    - name: 'openrouter-alerts'
      slack_configs:
      - channel: '#openrouter-alerts'
        title: 'OpenRouter: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Service:* {{ .Labels.service }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          {{ end }}
```

## Log Management

### Structured Logging Configuration

#### Application Logging

```python
# examples/openrouter/monitoring/structured_logging.py
"""
Enhanced structured logging for OpenRouter service monitoring.
"""

import structlog
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class OpenRouterLogger:
    """Structured logger for OpenRouter service with monitoring context."""
    
    def __init__(self, service_name: str = "openrouter-service"):
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                self._add_timestamp,
                self._add_service_context,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.service_name = service_name
        self.logger = structlog.get_logger()
    
    def _add_timestamp(self, logger, method_name, event_dict):
        """Add ISO timestamp to log entries."""
        event_dict["timestamp"] = datetime.utcnow().isoformat()
        return event_dict
    
    def _add_service_context(self, logger, method_name, event_dict):
        """Add service context to all log entries."""
        event_dict["service"] = self.service_name
        event_dict["provider"] = "openrouter"
        return event_dict
    
    def log_request(self, 
                   model: str, 
                   team: str, 
                   customer_id: Optional[str] = None,
                   request_id: Optional[str] = None,
                   **kwargs):
        """Log incoming request with governance context."""
        self.logger.info(
            "openrouter_request_started",
            model=model,
            team=team,
            customer_id=customer_id,
            request_id=request_id,
            **kwargs
        )
    
    def log_response(self, 
                    model: str,
                    status_code: int,
                    duration_ms: float,
                    tokens_used: int,
                    cost_usd: float,
                    provider: str,
                    team: str,
                    **kwargs):
        """Log response with performance and cost metrics."""
        self.logger.info(
            "openrouter_request_completed",
            model=model,
            status_code=status_code,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            provider=provider,
            team=team,
            **kwargs
        )
    
    def log_error(self, 
                  error: Exception,
                  model: str,
                  team: str,
                  **kwargs):
        """Log error with full context."""
        self.logger.error(
            "openrouter_request_error",
            error=str(error),
            error_type=type(error).__name__,
            model=model,
            team=team,
            **kwargs,
            exc_info=True
        )
    
    def log_cost_alert(self,
                      team: str,
                      daily_cost: float,
                      threshold: float,
                      **kwargs):
        """Log cost threshold alerts."""
        self.logger.warning(
            "openrouter_cost_threshold_exceeded",
            team=team,
            daily_cost_usd=daily_cost,
            threshold_usd=threshold,
            **kwargs
        )
    
    def log_provider_failover(self,
                             original_provider: str,
                             fallback_provider: str,
                             model: str,
                             reason: str,
                             **kwargs):
        """Log provider failover events."""
        self.logger.warning(
            "openrouter_provider_failover",
            original_provider=original_provider,
            fallback_provider=fallback_provider,
            model=model,
            reason=reason,
            **kwargs
        )
```

#### Log Aggregation with Fluentd

Create `monitoring/logging/fluentd-config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: genops-monitoring
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*genops-openrouter*.log
      pos_file /var/log/fluentd-openrouter.log.pos
      tag kubernetes.openrouter.*
      format json
      time_key timestamp
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <filter kubernetes.openrouter.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
      kubernetes_url "#{ENV['FLUENT_FILTER_KUBERNETES_URL'] || 'https://' + ENV['KUBERNETES_SERVICE_HOST'] + ':' + ENV['KUBERNETES_SERVICE_PORT'] + '/api'}"
      verify_ssl "#{ENV['KUBERNETES_VERIFY_SSL'] || true}"
      preserve_json_log true
      merge_json_log true
      flatten_hashes true
      flatten_hashes_separator _
    </filter>
    
    <filter kubernetes.openrouter.**>
      @type record_transformer
      <record>
        service genops-openrouter
        environment "#{ENV['ENVIRONMENT'] || 'production'}"
        cluster "#{ENV['CLUSTER_NAME'] || 'default'}"
      </record>
    </filter>
    
    # Route to different outputs based on log level
    <match kubernetes.openrouter.**>
      @type rewrite_tag_filter
      <rule>
        key level
        pattern /^(ERROR|CRITICAL)$/
        tag alerts.openrouter.${tag_suffix[2]}
      </rule>
      <rule>
        key level
        pattern /^(WARN|WARNING)$/
        tag warnings.openrouter.${tag_suffix[2]}
      </rule>
      <rule>
        key $.level
        pattern /.*/
        tag logs.openrouter.${tag_suffix[2]}
      </rule>
    </match>
    
    # Send errors to alerting system
    <match alerts.openrouter.**>
      @type elasticsearch
      host elasticsearch-master
      port 9200
      index_name genops-openrouter-alerts
      type_name _doc
      include_tag_key true
      tag_key @log_name
      flush_interval 1s
    </match>
    
    # Send all logs to main index
    <match logs.openrouter.**>
      @type elasticsearch
      host elasticsearch-master
      port 9200
      index_name genops-openrouter-logs
      type_name _doc
      include_tag_key true
      tag_key @log_name
      flush_interval 10s
    </match>
```

## Performance Monitoring

### Application Performance Monitoring (APM)

#### Custom Performance Metrics

```python
# monitoring/performance/openrouter_apm.py
"""
Application Performance Monitoring for OpenRouter service.
"""

from typing import Dict, List, Optional
import time
import psutil
import threading
from dataclasses import dataclass
from collections import defaultdict, deque
from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    request_count: int = 0
    total_duration: float = 0.0
    error_count: int = 0
    active_requests: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

class OpenRouterAPM:
    """Application Performance Monitoring for OpenRouter."""
    
    def __init__(self, service_name: str = "openrouter-service"):
        self.service_name = service_name
        self.metrics_by_endpoint = defaultdict(PerformanceMetrics)
        self.recent_response_times = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        # Initialize OpenTelemetry metrics
        self.meter = metrics.get_meter("genops.openrouter.apm")
        
        # Create custom metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up custom performance metrics."""
        
        # Request duration histogram
        self.request_duration = self.meter.create_histogram(
            name="genops_openrouter_request_duration_seconds",
            description="Request duration in seconds",
            unit="s"
        )
        
        # Active requests gauge
        self.active_requests_gauge = self.meter.create_up_down_counter(
            name="genops_openrouter_active_requests",
            description="Number of active requests"
        )
        
        # Throughput counter
        self.throughput_counter = self.meter.create_counter(
            name="genops_openrouter_throughput_total",
            description="Total throughput counter"
        )
        
        # Resource usage gauges
        self.meter.create_observable_gauge(
            name="genops_openrouter_cpu_usage_percent",
            description="CPU usage percentage",
            callbacks=[self._get_cpu_usage]
        )
        
        self.meter.create_observable_gauge(
            name="genops_openrouter_memory_usage_bytes",
            description="Memory usage in bytes",
            callbacks=[self._get_memory_usage]
        )
        
        # Custom business metrics
        self.meter.create_observable_gauge(
            name="genops_openrouter_avg_response_time",
            description="Average response time over last 100 requests",
            callbacks=[self._get_avg_response_time]
        )
    
    def _get_cpu_usage(self, options: CallbackOptions):
        """Callback for CPU usage metric."""
        cpu_percent = psutil.cpu_percent(interval=None)
        yield Observation(cpu_percent, {"service": self.service_name})
    
    def _get_memory_usage(self, options: CallbackOptions):
        """Callback for memory usage metric."""
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        yield Observation(memory_bytes, {"service": self.service_name})
    
    def _get_avg_response_time(self, options: CallbackOptions):
        """Callback for average response time metric."""
        with self._lock:
            if self.recent_response_times:
                avg_time = sum(self.recent_response_times) / len(self.recent_response_times)
                yield Observation(avg_time, {"service": self.service_name})
    
    def record_request_start(self, endpoint: str, model: str, team: str):
        """Record the start of a request."""
        attributes = {
            "endpoint": endpoint,
            "model": model,
            "team": team,
            "service": self.service_name
        }
        
        with self._lock:
            self.metrics_by_endpoint[endpoint].active_requests += 1
        
        self.active_requests_gauge.add(1, attributes)
        return time.time()  # Return start time
    
    def record_request_end(self, 
                          endpoint: str, 
                          model: str, 
                          team: str,
                          start_time: float, 
                          status_code: int,
                          tokens_used: int = 0,
                          cost: float = 0.0):
        """Record the completion of a request."""
        duration = time.time() - start_time
        
        attributes = {
            "endpoint": endpoint,
            "model": model,
            "team": team,
            "status_code": str(status_code),
            "service": self.service_name
        }
        
        with self._lock:
            metrics = self.metrics_by_endpoint[endpoint]
            metrics.active_requests = max(0, metrics.active_requests - 1)
            metrics.request_count += 1
            metrics.total_duration += duration
            
            if status_code >= 400:
                metrics.error_count += 1
            
            self.recent_response_times.append(duration)
        
        # Record metrics
        self.request_duration.record(duration, attributes)
        self.active_requests_gauge.add(-1, attributes)
        self.throughput_counter.add(1, attributes)
        
        # Record business metrics if available
        if tokens_used > 0:
            self.meter.create_histogram(
                name="genops_openrouter_tokens_per_request",
                description="Tokens used per request"
            ).record(tokens_used, attributes)
        
        if cost > 0:
            self.meter.create_histogram(
                name="genops_openrouter_cost_per_request",
                description="Cost per request in USD"
            ).record(cost, attributes)
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        with self._lock:
            total_requests = sum(m.request_count for m in self.metrics_by_endpoint.values())
            total_errors = sum(m.error_count for m in self.metrics_by_endpoint.values())
            total_active = sum(m.active_requests for m in self.metrics_by_endpoint.values())
            
            avg_response_time = 0.0
            if self.recent_response_times:
                avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times)
            
            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": total_errors / total_requests if total_requests > 0 else 0,
                "active_requests": total_active,
                "avg_response_time_ms": avg_response_time * 1000,
                "cpu_usage_percent": psutil.cpu_percent(interval=None),
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "endpoints": {
                    endpoint: {
                        "request_count": metrics.request_count,
                        "error_count": metrics.error_count,
                        "error_rate": metrics.error_count / metrics.request_count if metrics.request_count > 0 else 0,
                        "avg_duration_ms": (metrics.total_duration / metrics.request_count * 1000) if metrics.request_count > 0 else 0,
                        "active_requests": metrics.active_requests
                    }
                    for endpoint, metrics in self.metrics_by_endpoint.items()
                }
            }

# Global APM instance
apm = OpenRouterAPM()

# Context manager for request tracking
class RequestTracker:
    def __init__(self, endpoint: str, model: str, team: str):
        self.endpoint = endpoint
        self.model = model
        self.team = team
        self.start_time = None
    
    def __enter__(self):
        self.start_time = apm.record_request_start(
            self.endpoint, self.model, self.team
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status_code = 500 if exc_type else 200
        apm.record_request_end(
            self.endpoint, self.model, self.team,
            self.start_time, status_code
        )
```

## Cost Monitoring

### Cost Analytics Dashboard

Create `monitoring/cost/cost-analytics.py`:

```python
"""
Cost analytics and monitoring for OpenRouter service.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class CostMetric:
    """Cost metric container."""
    team: str
    project: str
    customer_id: Optional[str]
    model: str
    provider: str
    cost_usd: float
    tokens_used: int
    timestamp: datetime

class CostAnalytics:
    """Cost analytics and monitoring system."""
    
    def __init__(self):
        self.cost_data = []
        self.daily_budgets = {}
        self.monthly_budgets = {}
        self.alert_thresholds = {}
    
    def record_cost(self, 
                   team: str, 
                   project: str,
                   model: str,
                   provider: str,
                   cost_usd: float,
                   tokens_used: int,
                   customer_id: Optional[str] = None):
        """Record a cost metric."""
        metric = CostMetric(
            team=team,
            project=project,
            customer_id=customer_id,
            model=model,
            provider=provider,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            timestamp=datetime.utcnow()
        )
        self.cost_data.append(metric)
        
        # Check for budget alerts
        self._check_budget_alerts(metric)
    
    def get_daily_cost_by_team(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """Get daily costs grouped by team."""
        if date is None:
            date = datetime.utcnow()
        
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        costs_by_team = defaultdict(float)
        
        for metric in self.cost_data:
            if start_of_day <= metric.timestamp < end_of_day:
                costs_by_team[metric.team] += metric.cost_usd
        
        return dict(costs_by_team)
    
    def get_cost_by_model(self, 
                         team: Optional[str] = None,
                         days: int = 7) -> Dict[str, float]:
        """Get cost breakdown by model."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        costs_by_model = defaultdict(float)
        
        for metric in self.cost_data:
            if metric.timestamp >= cutoff_date:
                if team is None or metric.team == team:
                    costs_by_model[metric.model] += metric.cost_usd
        
        return dict(costs_by_model)
    
    def get_cost_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate cost efficiency metrics."""
        if not self.cost_data:
            return {}
        
        # Cost per token by model
        cost_per_token = defaultdict(lambda: {'total_cost': 0.0, 'total_tokens': 0})
        
        for metric in self.cost_data:
            cost_per_token[metric.model]['total_cost'] += metric.cost_usd
            cost_per_token[metric.model]['total_tokens'] += metric.tokens_used
        
        efficiency = {}
        for model, data in cost_per_token.items():
            if data['total_tokens'] > 0:
                efficiency[model] = data['total_cost'] / data['total_tokens']
        
        return efficiency
    
    def get_provider_cost_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare costs across providers."""
        provider_costs = defaultdict(lambda: defaultdict(float))
        provider_usage = defaultdict(lambda: defaultdict(int))
        
        for metric in self.cost_data:
            provider_costs[metric.provider][metric.model] += metric.cost_usd
            provider_usage[metric.provider][metric.model] += metric.tokens_used
        
        comparison = {}
        for provider in provider_costs:
            comparison[provider] = {}
            for model in provider_costs[provider]:
                total_cost = provider_costs[provider][model]
                total_tokens = provider_usage[provider][model]
                comparison[provider][model] = {
                    'total_cost': total_cost,
                    'cost_per_token': total_cost / total_tokens if total_tokens > 0 else 0
                }
        
        return comparison
    
    def set_budget_alert(self, 
                        team: str, 
                        daily_budget: Optional[float] = None,
                        monthly_budget: Optional[float] = None,
                        alert_threshold: float = 0.8):
        """Set budget alerts for a team."""
        if daily_budget:
            self.daily_budgets[team] = daily_budget
        if monthly_budget:
            self.monthly_budgets[team] = monthly_budget
        
        self.alert_thresholds[team] = alert_threshold
    
    def _check_budget_alerts(self, metric: CostMetric):
        """Check if budget thresholds are exceeded."""
        team = metric.team
        
        # Daily budget check
        if team in self.daily_budgets:
            daily_cost = self.get_daily_cost_by_team()[team]
            daily_budget = self.daily_budgets[team]
            threshold = self.alert_thresholds.get(team, 0.8)
            
            if daily_cost >= daily_budget * threshold:
                self._send_budget_alert(
                    team, "daily", daily_cost, daily_budget, threshold
                )
    
    def _send_budget_alert(self, 
                          team: str, 
                          period: str, 
                          current_cost: float, 
                          budget: float, 
                          threshold: float):
        """Send budget alert (integrate with your alerting system)."""
        alert_data = {
            "team": team,
            "period": period,
            "current_cost": current_cost,
            "budget": budget,
            "percentage_used": current_cost / budget,
            "threshold": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Here you would integrate with your alerting system
        print(f"BUDGET ALERT: {json.dumps(alert_data, indent=2)}")
    
    def generate_cost_report(self, 
                            start_date: datetime, 
                            end_date: datetime) -> Dict:
        """Generate comprehensive cost report."""
        filtered_data = [
            metric for metric in self.cost_data
            if start_date <= metric.timestamp <= end_date
        ]
        
        if not filtered_data:
            return {"error": "No data found for the specified period"}
        
        total_cost = sum(metric.cost_usd for metric in filtered_data)
        total_tokens = sum(metric.tokens_used for metric in filtered_data)
        
        # Group by various dimensions
        cost_by_team = defaultdict(float)
        cost_by_project = defaultdict(float)
        cost_by_model = defaultdict(float)
        cost_by_provider = defaultdict(float)
        
        for metric in filtered_data:
            cost_by_team[metric.team] += metric.cost_usd
            cost_by_project[metric.project] += metric.cost_usd
            cost_by_model[metric.model] += metric.cost_usd
            cost_by_provider[metric.provider] += metric.cost_usd
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "average_cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
                "total_requests": len(filtered_data)
            },
            "breakdown": {
                "by_team": dict(cost_by_team),
                "by_project": dict(cost_by_project),
                "by_model": dict(cost_by_model),
                "by_provider": dict(cost_by_provider)
            },
            "top_consumers": {
                "teams": sorted(cost_by_team.items(), key=lambda x: x[1], reverse=True)[:10],
                "projects": sorted(cost_by_project.items(), key=lambda x: x[1], reverse=True)[:10],
                "models": sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        }

# Global cost analytics instance
cost_analytics = CostAnalytics()
```

## Troubleshooting Runbooks

### Common Issues and Solutions

#### 1. High Error Rate

**Symptoms:**
- Error rate > 5%
- Multiple 5xx responses
- Alert: "OpenRouterHighErrorRate"

**Investigation Steps:**
```bash
# Check error distribution
kubectl logs -n genops-openrouter -l app.kubernetes.io/name=openrouter-service --tail=100 | grep ERROR

# Check provider status
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models

# Check resource usage
kubectl top pods -n genops-openrouter

# Check OpenTelemetry traces
# Query in Honeycomb: WHERE genops.provider = "openrouter" AND http.status_code >= 400
```

**Resolution:**
1. **API Key Issues**: Rotate API key if authentication errors
2. **Rate Limiting**: Implement exponential backoff
3. **Resource Constraints**: Scale up deployment
4. **Provider Outage**: Enable fallback providers

#### 2. High Latency

**Symptoms:**
- P95 latency > 5 seconds
- Timeout errors
- Alert: "OpenRouterHighLatency"

**Investigation:**
```bash
# Check provider response times
# Honeycomb query: P95(duration_ms) WHERE genops.provider = "openrouter" GROUP BY genops.openrouter.actual_provider

# Check network connectivity
kubectl exec -n genops-openrouter deployment/openrouter-service -- curl -w "%{time_total}" -o /dev/null -s https://openrouter.ai/api/v1/models

# Check resource usage
kubectl describe pods -n genops-openrouter
```

**Resolution:**
1. **Provider Selection**: Route to faster providers
2. **Request Optimization**: Reduce token limits
3. **Scaling**: Increase replica count
4. **Circuit Breaker**: Implement request timeouts

#### 3. High Costs

**Symptoms:**
- Daily costs exceed budget
- Alert: "OpenRouterHighDailyCost"
- Unexpected cost spikes

**Investigation:**
```bash
# Cost analysis query (Honeycomb)
# SUM(genops.cost.total) WHERE genops.provider = "openrouter" GROUP BY genops.team, genops.model

# Check high-cost models
python -c "
from genops.providers.openrouter_pricing import get_cost_breakdown
models = ['anthropic/claude-3-opus', 'openai/gpt-4o', 'meta-llama/llama-3.1-405b-instruct']
for model in models:
    cost = get_cost_breakdown(model, 1000, 500)
    print(f'{model}: \${cost[\"total_cost\"]:.4f}')
"
```

**Resolution:**
1. **Model Optimization**: Switch to cost-effective models
2. **Budget Limits**: Implement per-team quotas
3. **Request Filtering**: Add content length limits
4. **Usage Analytics**: Identify optimization opportunities

---

This production monitoring guide provides comprehensive observability, alerting, and troubleshooting capabilities for GenOps OpenRouter deployments, ensuring reliability, performance, and cost optimization in production environments.