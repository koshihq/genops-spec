# Databricks Unity Catalog Production Deployment Guide

Complete guide for deploying GenOps governance with Databricks Unity Catalog in production environments.

## Overview

This guide covers enterprise-grade deployment patterns, high-availability configurations, security best practices, and performance optimization for production Databricks Unity Catalog governance.

## Production Architecture Patterns

### 1. High-Availability Multi-Workspace Deployment

**Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Primary       │    │   Secondary     │    │   DR Site       │
│   US-West-2     │◄──►│   US-East-1     │◄──►│   EU-Central    │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   GenOps    │ │    │ │   GenOps    │ │    │ │   GenOps    │ │
│ │ Governance  │ │    │ │ Governance  │ │    │ │ Governance  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ Unity Catalog   │    │ Unity Catalog   │    │ Unity Catalog   │
│ Primary         │    │ Secondary       │    │ DR Backup       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Implementation:**
```python
from genops.providers.databricks_unity_catalog.registration import configure_unity_catalog_governance

# Primary workspace configuration
primary_config = configure_unity_catalog_governance(
    workspace_url="https://primary-prod-us-west-2.cloud.databricks.com",
    metastore_id="primary-metastore",
    enable_high_availability=True,
    failover_workspace_url="https://secondary-prod-us-east-1.cloud.databricks.com",
    sync_interval_seconds=30,
    health_check_interval_seconds=60,
    governance_config={
        "enable_cross_workspace_lineage": True,
        "enable_unified_cost_reporting": True,
        "compliance_level": "enterprise",
        "audit_retention_days": 2555,  # 7 years
        "enable_real_time_alerts": True
    }
)

# Secondary workspace configuration
secondary_config = configure_unity_catalog_governance(
    workspace_url="https://secondary-prod-us-east-1.cloud.databricks.com",
    metastore_id="secondary-metastore",
    is_failover_target=True,
    primary_workspace_url="https://primary-prod-us-west-2.cloud.databricks.com",
    enable_automatic_failover=True,
    governance_config={
        "enable_cross_workspace_lineage": True,
        "enable_unified_cost_reporting": True,
        "compliance_level": "enterprise"
    }
)

# Health monitoring and automatic failover
import time
while True:
    if not primary_config.is_healthy():
        print("⚠️ Primary workspace unhealthy - initiating failover")
        secondary_config.promote_to_primary()
        break
    time.sleep(60)
```

### 2. Enterprise Security Configuration

**Role-Based Access Controls:**
```python
# Enterprise RBAC configuration
enterprise_security = {
    "rbac_enabled": True,
    "authentication": {
        "method": "azure_ad",  # or "aws_iam", "google_identity"
        "mfa_required": True,
        "session_timeout_minutes": 480  # 8 hours
    },
    "authorization": {
        "data_classification_enforcement": True,
        "row_level_security": True,
        "column_masking": True,
        "minimum_clearance_levels": {
            "restricted": ["data_steward", "compliance_officer"],
            "confidential": ["data_analyst", "data_engineer"],
            "internal": ["all_authenticated_users"]
        }
    },
    "encryption": {
        "data_at_rest": "customer_managed_keys",
        "data_in_transit": "tls_1_3",
        "telemetry_encryption": True,
        "key_rotation_days": 90
    },
    "audit_logging": {
        "enabled": True,
        "log_level": "detailed",
        "destinations": ["splunk", "datadog", "s3_bucket"],
        "real_time_alerting": True
    }
}

# Apply enterprise security configuration
adapter = instrument_databricks_unity_catalog(
    workspace_url="https://enterprise-prod.cloud.databricks.com",
    **enterprise_security
)
```

**Network Security:**
```yaml
# VPC/Network configuration (infrastructure as code)
network_security:
  vpc_endpoints:
    - databricks_workspace
    - databricks_backend
    - s3_root_bucket
  
  security_groups:
    databricks_workspace:
      ingress:
        - port: 443
          protocol: tcp
          source: corporate_network_cidrs
        - port: 2443
          protocol: tcp  
          source: databricks_control_plane
      egress:
        - port: 443
          protocol: tcp
          destination: databricks_control_plane
        - port: 443
          protocol: tcp
          destination: genops_telemetry_endpoints
  
  private_subnet_configuration:
    workspace_subnets:
      - subnet-prod-databricks-private-1a
      - subnet-prod-databricks-private-1b
    storage_endpoint: vpce-databricks-s3-prod
```

### 3. Performance Optimization for High-Volume Workloads

**High-Performance Configuration:**
```python
# Performance optimization for large-scale deployments
performance_config = {
    "telemetry_optimization": {
        "enable_sampling": True,
        "sampling_strategy": "adaptive",  # Adjust based on volume
        "sampling_rates": {
            "table_operations": 0.1,      # 10% for high-volume table ops
            "sql_warehouse": 1.0,          # 100% for expensive warehouse ops
            "governance_events": 1.0       # 100% for compliance
        },
        "batch_processing": {
            "enabled": True,
            "batch_size": 1000,
            "flush_interval_seconds": 30,
            "max_memory_mb": 512
        }
    },
    
    "cost_calculation": {
        "enable_caching": True,
        "cache_ttl_seconds": 300,  # 5 minutes
        "enable_async_processing": True,
        "cost_aggregation_interval": 60  # 1 minute
    },
    
    "lineage_tracking": {
        "enable_compression": True,
        "async_lineage_processing": True,
        "lineage_graph_cache_ttl": 3600,  # 1 hour
        "max_lineage_depth": 10
    },
    
    "resource_management": {
        "max_concurrent_operations": 50,
        "connection_pool_size": 20,
        "request_timeout_seconds": 30,
        "retry_policy": {
            "max_retries": 3,
            "backoff_multiplier": 2.0,
            "max_backoff_seconds": 60
        }
    }
}

adapter = instrument_databricks_unity_catalog(
    workspace_url="https://high-volume-prod.cloud.databricks.com",
    **performance_config
)
```

**Scaling Recommendations:**
```python
# Auto-scaling configuration based on workload
def configure_adaptive_scaling():
    from genops.providers.databricks_unity_catalog import get_cost_aggregator
    
    cost_aggregator = get_cost_aggregator()
    
    # Monitor operation volume and adjust sampling
    while True:
        summary = cost_aggregator.get_summary()
        operations_per_minute = summary.operation_count / 
            ((time.time() - summary.start_time) / 60)
        
        if operations_per_minute > 1000:
            # High volume - increase sampling efficiency
            adapter.update_sampling_rate("table_operations", 0.05)  # 5%
            adapter.enable_aggressive_caching()
        elif operations_per_minute < 100:
            # Low volume - increase sampling for accuracy
            adapter.update_sampling_rate("table_operations", 0.5)   # 50%
            
        time.sleep(60)  # Check every minute
```

## Deployment Templates

### 1. Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install GenOps with Databricks support
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 genops
RUN chown -R genops:genops /app
USER genops

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "from genops.providers.databricks_unity_catalog.validation import validate_setup; \
                 result = validate_setup(); \
                 exit(0 if result.is_valid else 1)"

ENTRYPOINT ["python", "src/main.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  genops-databricks:
    build: .
    environment:
      - DATABRICKS_HOST=${DATABRICKS_HOST}
      - DATABRICKS_TOKEN=${DATABRICKS_TOKEN}
      - GENOPS_TEAM=${GENOPS_TEAM}
      - GENOPS_PROJECT=${GENOPS_PROJECT}
      - GENOPS_ENVIRONMENT=production
      - OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_ENDPOINT}
      - OTEL_EXPORTER_OTLP_HEADERS=${OTEL_HEADERS}
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from genops.providers.databricks_unity_catalog.validation import validate_setup; exit(0 if validate_setup().is_valid else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  # Observability stack
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
```

### 2. Kubernetes Deployment

**kubernetes/namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: genops-databricks
  labels:
    name: genops-databricks
    environment: production
```

**kubernetes/configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-databricks-config
  namespace: genops-databricks
data:
  config.yaml: |
    genops:
      providers:
        databricks_unity_catalog:
          enable_high_availability: true
          performance_mode: "production"
          telemetry:
            sampling_rate: 0.1
            batch_size: 1000
            flush_interval: 30
          governance:
            compliance_level: "enterprise"
            audit_retention_days: 2555
            enable_real_time_alerts: true
```

**kubernetes/secret.yaml:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: genops-databricks-secrets
  namespace: genops-databricks
type: Opaque
stringData:
  databricks-host: "https://prod.cloud.databricks.com"
  databricks-token: "your-production-token-here"
  otel-endpoint: "https://api.honeycomb.io"
  otel-headers: "x-honeycomb-team=your-team-key"
```

**kubernetes/deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-databricks
  namespace: genops-databricks
  labels:
    app: genops-databricks
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: genops-databricks
  template:
    metadata:
      labels:
        app: genops-databricks
    spec:
      serviceAccountName: genops-databricks
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: genops-databricks
        image: genops/databricks-unity-catalog:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: health
        env:
        - name: DATABRICKS_HOST
          valueFrom:
            secretKeyRef:
              name: genops-databricks-secrets
              key: databricks-host
        - name: DATABRICKS_TOKEN
          valueFrom:
            secretKeyRef:
              name: genops-databricks-secrets
              key: databricks-token
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: genops-databricks-secrets
              key: otel-endpoint
        - name: OTEL_EXPORTER_OTLP_HEADERS
          valueFrom:
            secretKeyRef:
              name: genops-databricks-secrets
              key: otel-headers
        - name: GENOPS_ENVIRONMENT
          value: "production"
        - name: GENOPS_TEAM
          value: "data-platform"
        - name: GENOPS_PROJECT
          value: "unity-catalog-governance"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: genops-databricks-config
      - name: logs-volume
        emptyDir: {}
```

**kubernetes/service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: genops-databricks
  namespace: genops-databricks
  labels:
    app: genops-databricks
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 8081
    targetPort: 8081
    protocol: TCP
    name: health
  selector:
    app: genops-databricks
```

**kubernetes/hpa.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genops-databricks-hpa
  namespace: genops-databricks
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-databricks
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Monitoring and Alerting

### Production Monitoring Setup

**Prometheus Configuration:**
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "genops_databricks_rules.yml"

scrape_configs:
  - job_name: 'genops-databricks'
    kubernetes_sd_configs:
    - role: endpoints
      namespaces:
        names:
        - genops-databricks
    relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: genops-databricks

alerting:
  alertmanagers:
  - kubernetes_sd_configs:
    - role: endpoints
      namespaces:
        names:
        - monitoring
```

**Alerting Rules:**
```yaml
# monitoring/genops_databricks_rules.yml
groups:
- name: genops.databricks.unity_catalog
  rules:
  # High error rate
  - alert: GenOpsDatabricksHighErrorRate
    expr: rate(genops_databricks_errors_total[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GenOps Databricks error rate is high"
      description: "Error rate is {{ $value }} errors/sec for the last 5 minutes"

  # Cost anomaly detection
  - alert: GenOpsDatabricksUnexpectedCosts
    expr: increase(genops_databricks_cost_usd_total[1h]) > 100
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Unexpected high costs detected"
      description: "Hourly cost increase of ${{ $value }} detected"

  # Compliance violations
  - alert: GenOpsDatabricksComplianceViolation
    expr: increase(genops_databricks_compliance_violations_total[10m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Data governance compliance violation"
      description: "{{ $value }} compliance violations in the last 10 minutes"

  # Workspace connectivity issues
  - alert: GenOpsDatabricksWorkspaceDown
    expr: genops_databricks_workspace_healthy == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Databricks workspace connectivity lost"
      description: "Cannot connect to workspace {{ $labels.workspace_id }}"
```

### Grafana Dashboards

**Cost Monitoring Dashboard:**
```json
{
  "dashboard": {
    "title": "GenOps Databricks Unity Catalog - Cost Monitoring",
    "panels": [
      {
        "title": "Total Cost Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(genops_databricks_cost_usd_total[5m])) * 3600",
            "legendFormat": "Hourly Cost Rate"
          }
        ]
      },
      {
        "title": "Cost by Team",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (team) (genops_databricks_cost_usd_total)",
            "legendFormat": "{{ team }}"
          }
        ]
      },
      {
        "title": "Cost by Resource Type",
        "type": "bargraph",
        "targets": [
          {
            "expr": "sum by (resource_type) (genops_databricks_cost_usd_total)",
            "legendFormat": "{{ resource_type }}"
          }
        ]
      }
    ]
  }
}
```

## Disaster Recovery

### Backup and Recovery Procedures

**Automated Backup Script:**
```python
#!/usr/bin/env python3
"""
Databricks Unity Catalog Governance Backup Script
"""

import json
import boto3
from datetime import datetime
from genops.providers.databricks_unity_catalog import get_governance_monitor, get_cost_aggregator

def backup_governance_data():
    """Backup governance data to S3."""
    
    # Initialize components
    governance_monitor = get_governance_monitor()
    cost_aggregator = get_cost_aggregator()
    
    # Create backup data structure
    backup_data = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        "governance_summary": governance_monitor.get_governance_summary().to_dict(),
        "cost_summary": cost_aggregator.get_summary().to_dict(),
        "lineage_graph": governance_monitor.get_lineage_graph(),
        "compliance_reports": governance_monitor.get_compliance_reports(days=30),
        "policy_configurations": governance_monitor.export_policies()
    }
    
    # Upload to S3
    s3 = boto3.client('s3')
    backup_key = f"genops-databricks-backup/{datetime.now().strftime('%Y/%m/%d')}/governance-backup.json"
    
    s3.put_object(
        Bucket='genops-backup-bucket',
        Key=backup_key,
        Body=json.dumps(backup_data, indent=2),
        ServerSideEncryption='aws:kms',
        SSEKMSKeyId='arn:aws:kms:us-west-2:123456789012:key/backup-key-id'
    )
    
    print(f"✅ Backup completed: s3://genops-backup-bucket/{backup_key}")

def restore_governance_data(backup_date: str):
    """Restore governance data from backup."""
    
    s3 = boto3.client('s3')
    backup_key = f"genops-databricks-backup/{backup_date}/governance-backup.json"
    
    try:
        response = s3.get_object(Bucket='genops-backup-bucket', Key=backup_key)
        backup_data = json.loads(response['Body'].read())
        
        # Restore governance configuration
        governance_monitor = get_governance_monitor()
        governance_monitor.import_policies(backup_data["policy_configurations"])
        
        print(f"✅ Restore completed from: {backup_date}")
        return True
        
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        return False

if __name__ == "__main__":
    backup_governance_data()
```

## Performance Benchmarking

### Production Performance Targets

```python
# Performance benchmark tests
performance_targets = {
    "operation_latency": {
        "table_operation_tracking": "< 50ms",
        "sql_warehouse_tracking": "< 100ms", 
        "governance_policy_check": "< 25ms",
        "cost_calculation": "< 10ms"
    },
    "throughput": {
        "operations_per_second": "> 1000 ops/sec",
        "concurrent_workspaces": "> 10 workspaces",
        "cost_aggregation_rate": "> 5000 cost_entries/min"
    },
    "resource_usage": {
        "memory_usage": "< 1GB per 100k operations",
        "cpu_usage": "< 70% sustained",
        "network_bandwidth": "< 10MB/s"
    },
    "reliability": {
        "uptime": "99.9%",
        "error_rate": "< 0.1%",
        "recovery_time": "< 5 minutes"
    }
}

def benchmark_production_performance():
    """Benchmark production performance against targets."""
    import time
    import psutil
    
    # Test operation latency
    start_time = time.time()
    for i in range(1000):
        adapter.track_table_operation(
            operation="benchmark_test",
            catalog_name="benchmark",
            schema_name="performance",
            table_name=f"test_{i}",
            team="benchmark-team",
            project="performance-test"
        )
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 1000
    print(f"Average operation latency: {avg_latency*1000:.2f}ms")
    
    # Memory usage check
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage: {memory_usage:.2f}MB")
    
    return {
        "avg_latency_ms": avg_latency * 1000,
        "memory_usage_mb": memory_usage,
        "meets_targets": avg_latency < 0.05 and memory_usage < 1024
    }
```

This production deployment guide provides comprehensive coverage of enterprise deployment patterns, security configurations, monitoring setups, and disaster recovery procedures for Databricks Unity Catalog governance in production environments.