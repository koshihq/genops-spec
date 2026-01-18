# GenOps AI Helm Chart

A comprehensive Helm chart for deploying GenOps AI governance framework on Kubernetes with enterprise-grade security, observability, and multi-provider AI support.

## Features

✅ **Multi-Provider AI Support**: OpenAI, Anthropic, OpenRouter, and more  
✅ **Enterprise Security**: RBAC, Network Policies, Pod Security Standards  
✅ **Auto-scaling**: HPA with custom metrics and VPA support  
✅ **Observability**: Prometheus metrics, OpenTelemetry integration  
✅ **Environment-Specific Configurations**: Dev, staging, production profiles  
✅ **Cloud-Native**: Service mesh ready, admission controller support  

## Quick Start

### Prerequisites

- Kubernetes 1.20+ 
- Helm 3.0+
- AI provider API keys
- Optional: Prometheus Operator for monitoring

### Installation

1. **Add the repository**:
```bash
helm repo add genops https://charts.genops.ai
helm repo update
```

2. **Create namespace**:
```bash
kubectl create namespace genops-ai
```

3. **Configure values**:
```bash
# Create values file with your configuration
cat > values-production.yaml <<EOF
global:
  environment: production
  team: ai-platform
  project: genops-deployment

secrets:
  apiKeys:
    openai: "sk-..."
    anthropic: "sk-ant-..."
    honeycomb: "hcaik_..."

opentelemetry:
  endpoint: "http://otel-collector:4318"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
EOF
```

4. **Install the chart**:
```bash
helm install genops-ai genops/genops-ai \
  --namespace genops-ai \
  --values values-production.yaml
```

### Verify Installation

```bash
# Check deployment status
kubectl get pods -n genops-ai

# Test health endpoint
kubectl port-forward -n genops-ai service/genops-ai 8000:8000
curl http://localhost:8000/health

# Check logs
kubectl logs -n genops-ai -l app.kubernetes.io/name=genops-ai -f
```

### Validate OpenTelemetry Collector Setup

**Validate that your OTel Collector is properly configured and accessible:**

```bash
# Run automated validation from within a pod
kubectl run genops-validator --rm -it --restart=Never \
  --image=python:3.11 \
  --namespace genops-ai \
  -- bash -c "
    pip install requests && \
    curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/otel_collector_validation.py && \
    curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/validate_otel_collector.py && \
    python validate_otel_collector.py --endpoint http://otel-collector:4318
  "
```

**Manual validation:**

```bash
# Check OTel Collector health
kubectl run curl-test --rm -it --restart=Never \
  --image=curlimages/curl \
  --namespace genops-ai \
  -- http://otel-collector:13133/

# Test OTLP HTTP endpoint
kubectl run curl-test --rm -it --restart=Never \
  --image=curlimages/curl \
  --namespace genops-ai \
  -- -v http://otel-collector:4318/v1/traces

# Check if OTel Collector pods are running
kubectl get pods -n genops-ai -l app=otel-collector
```

**Expected validation results:**
- ✅ Collector Status: Healthy
- ✅ OTLP HTTP Endpoint: Accessible (port 4318)
- ✅ OTLP gRPC Endpoint: Accessible (port 4317)

**If validation fails**, see the [Troubleshooting](#troubleshooting) section below.

## Configuration

### Environment Profiles

The chart includes pre-configured environments:

```yaml
# Development
helm install genops-ai genops/genops-ai \
  --set global.environment=development \
  --set deployment.replicaCount=1

# Staging  
helm install genops-ai genops/genops-ai \
  --set global.environment=staging \
  --set autoscaling.minReplicas=2

# Production
helm install genops-ai genops/genops-ai \
  --set global.environment=production \
  --set autoscaling.minReplicas=3
```

### AI Provider Configuration

```yaml
providers:
  openai:
    enabled: true
    apiKeySecret:
      name: genops-ai-secrets
      key: openai-api-key
  
  anthropic:
    enabled: true
    apiKeySecret:
      name: genops-ai-secrets
      key: anthropic-api-key
```

### Governance Policies

```yaml
governance:
  policies:
    enabled: true
    costLimits:
      daily: 100.00
      monthly: 2500.00
    rateLimits:
      requestsPerMinute: 60
    contentSafety:
      enabled: true
      minimumScore: 0.85
```

### Observability Integration

```yaml
opentelemetry:
  enabled: true
  endpoint: "http://otel-collector:4318"
  headers:
    x-honeycomb-team: "your-api-key"

monitoring:
  serviceMonitor:
    enabled: true  # Requires Prometheus Operator
    interval: 30s
```

## Advanced Configuration

### Custom Resource Limits

```yaml
deployment:
  container:
    resources:
      limits:
        cpu: 1000m
        memory: 2Gi
        nvidia.com/gpu: 1  # GPU support
      requests:
        cpu: 500m
        memory: 1Gi
```

### Multi-Zone High Availability

```yaml
deployment:
  replicaCount: 6
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - genops-ai
        topologyKey: topology.kubernetes.io/zone
```

### Network Security

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: api-gateway
      ports:
      - protocol: TCP
        port: 8000
  
  egress:
    - to: []  # AI provider APIs
      ports:
      - protocol: TCP
        port: 443
```

### Service Mesh Integration

```yaml
# Istio VirtualService
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: genops-ai
spec:
  hosts:
  - genops-ai
  http:
  - route:
    - destination:
        host: genops-ai
        port:
          number: 8000
    timeout: 30s
    retries:
      attempts: 3
```

## Monitoring & Observability

### Built-in Metrics

The chart exposes comprehensive metrics:

- **AI Operations**: Request counts, latencies, costs
- **Resource Usage**: CPU, memory, GPU utilization  
- **Governance**: Policy violations, compliance scores
- **Business**: Cost attribution by team/customer

### Grafana Dashboards

Import pre-built dashboards:

```bash
# Add Grafana dashboard ConfigMap
kubectl apply -f charts/genops-ai/dashboards/
```

### Alerting Rules

```yaml
# PrometheusRule for alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: genops-ai-alerts
spec:
  groups:
  - name: genops-ai.rules
    rules:
    - alert: GenOpsHighCost
      expr: sum(rate(genops_cost_total[1h])) > 10
      for: 5m
      annotations:
        summary: "GenOps AI costs are high"
```

## Security

### RBAC Configuration

The chart creates minimal RBAC permissions:

```yaml
rbac:
  rules:
  - apiGroups: [""]
    resources: ["pods", "nodes"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["resourcequotas"] 
    verbs: ["get", "list"]
```

### Pod Security Standards

```yaml
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

containerSecurityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop: ["ALL"]
```

### Secrets Management

```yaml
# External secrets integration
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: genops-ai-secrets
spec:
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: genops-ai-secrets
  data:
  - secretKey: openai-api-key
    remoteRef:
      key: genops/openai
      property: api-key
```

## Production Deployment

### Multi-Region Setup

```bash
# Deploy to multiple regions
for region in us-east-1 us-west-2 eu-west-1; do
  helm install genops-ai-$region genops/genops-ai \
    --namespace genops-ai-$region \
    --set global.region=$region \
    --values values-$region.yaml
done
```

### Blue-Green Deployment

```yaml
# Blue deployment
helm install genops-ai-blue genops/genops-ai \
  --namespace genops-ai \
  --set deployment.podLabels.version=blue

# Green deployment  
helm install genops-ai-green genops/genops-ai \
  --namespace genops-ai \
  --set deployment.podLabels.version=green

# Switch traffic via ingress
```

### Disaster Recovery

```yaml
# Cross-region backup
velero backup create genops-ai-backup \
  --include-namespaces genops-ai \
  --storage-location aws-us-west-2
```

## Troubleshooting

### Common Issues

#### Issue: Pods not starting

**Diagnosis:**
```bash
kubectl describe pods -n genops-ai
kubectl logs -n genops-ai -l app.kubernetes.io/name=genops-ai
```

**Common causes:**
- Insufficient resources (check resource quotas)
- Image pull errors (verify image registry access)
- ConfigMap/Secret missing (verify secrets are created)

---

#### Issue: API key issues

**Diagnosis:**
```bash
kubectl get secrets -n genops-ai
kubectl exec -n genops-ai deployment/genops-ai -- \
  python -c "from genops.providers.openai import validate_setup; print(validate_setup())"
```

**Solutions:**
1. Verify secrets exist and have correct keys:
   ```bash
   kubectl get secret genops-ai-secrets -n genops-ai -o yaml
   ```

2. Re-create secrets with correct values:
   ```bash
   kubectl delete secret genops-ai-secrets -n genops-ai
   kubectl create secret generic genops-ai-secrets \
     --from-literal=openai-api-key="sk-..." \
     --from-literal=anthropic-api-key="sk-ant-..." \
     --namespace genops-ai
   ```

3. Restart pods to pick up new secrets:
   ```bash
   kubectl rollout restart deployment/genops-ai -n genops-ai
   ```

---

#### Issue: OpenTelemetry Collector connection failures

**Symptoms:**
- Application logs show: "Failed to export traces"
- No telemetry data appearing in observability backend
- Timeout errors connecting to collector

**Diagnosis:**
```bash
# Check if OTel Collector is deployed
kubectl get pods -n genops-ai -l app=otel-collector

# Check collector logs
kubectl logs -n genops-ai -l app=otel-collector

# Verify service exists
kubectl get svc -n genops-ai otel-collector

# Test connectivity from GenOps pod
kubectl exec -n genops-ai deployment/genops-ai -- \
  curl -v http://otel-collector:4318/v1/traces
```

**Solutions:**

1. **Collector not deployed:**
   ```yaml
   # In values.yaml, ensure OTel Collector is enabled
   opentelemetry:
     enabled: true
     endpoint: "http://otel-collector:4318"
   ```

2. **Wrong endpoint URL:**
   ```bash
   # Check ConfigMap for OTEL_EXPORTER_OTLP_ENDPOINT
   kubectl get configmap genops-ai-config -n genops-ai -o yaml | grep OTEL

   # Should be: http://otel-collector:4318 (or 4317 for gRPC)
   ```

3. **Network policy blocking traffic:**
   ```bash
   # Check if network policies exist
   kubectl get networkpolicies -n genops-ai

   # Update network policy to allow egress to collector
   ```

4. **Collector service not ready:**
   ```bash
   # Check collector readiness
   kubectl get pods -n genops-ai -l app=otel-collector

   # Wait for collector to be ready
   kubectl wait --for=condition=ready pod \
     -l app=otel-collector \
     -n genops-ai \
     --timeout=120s
   ```

5. **Use validation utilities:**
   ```bash
   # Run automated validation
   kubectl run genops-validator --rm -it --restart=Never \
     --image=python:3.11 \
     --namespace genops-ai \
     -- bash -c "
       pip install requests && \
       curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/otel_collector_validation.py && \
       curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/validate_otel_collector.py && \
       python validate_otel_collector.py --endpoint http://otel-collector:4318
     "
   ```

---

#### Issue: No telemetry data in observability backend

**Symptoms:**
- OTel Collector is running but no data appears in Grafana/Datadog/Splunk
- Collector receiving data but not exporting

**Diagnosis:**
```bash
# Check collector metrics to see if data is being received
kubectl port-forward -n genops-ai service/otel-collector 8888:8888
curl http://localhost:8888/metrics | grep otelcol_receiver

# Check collector logs for export errors
kubectl logs -n genops-ai -l app=otel-collector | grep -i export

# Verify exporter configuration in collector config
kubectl get configmap otel-collector-config -n genops-ai -o yaml
```

**Solutions:**

1. **Verify exporters are configured:**
   ```yaml
   # In collector ConfigMap, check service.pipelines
   service:
     pipelines:
       traces:
         receivers: [otlp]
         processors: [batch]
         exporters: [otlp/tempo, datadog]  # Ensure exporters are listed
   ```

2. **Check backend connectivity:**
   ```bash
   # Test connection to Tempo
   kubectl exec -n genops-ai -l app=otel-collector -- \
     curl -v http://tempo:4317

   # Test connection to Datadog (if using)
   kubectl exec -n genops-ai -l app=otel-collector -- \
     curl -v https://api.datadoghq.com
   ```

3. **Verify authentication tokens:**
   ```bash
   # Check if secrets for backends exist
   kubectl get secrets -n genops-ai | grep observability

   # Verify environment variables in collector deployment
   kubectl describe deployment otel-collector -n genops-ai | grep -A 10 "Environment:"
   ```

---

#### Issue: Resource constraints

**Diagnosis:**
```bash
kubectl top pods -n genops-ai
kubectl get hpa -n genops-ai
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**Solutions:**
1. Increase resource limits in values.yaml
2. Enable autoscaling with appropriate thresholds
3. Add more nodes to the cluster

---

#### Issue: High latency or slow responses

**Diagnosis:**
```bash
# Check pod resource usage
kubectl top pods -n genops-ai

# Check API latency metrics
kubectl port-forward -n genops-ai service/genops-ai 8000:8000
curl http://localhost:8000/metrics | grep http_request_duration

# Check if HPA is scaling
kubectl get hpa -n genops-ai -w
```

**Solutions:**
1. Increase replica count or HPA targets
2. Optimize AI provider configurations
3. Enable caching for frequently-used operations
4. Review and optimize governance policies

---

### Validation Utilities

**For comprehensive validation, use the built-in validation utilities:**

```bash
# Local validation (if you have Python installed)
python examples/observability/validate_otel_collector.py \
  --endpoint http://otel-collector.genops-ai.svc.cluster.local:4318

# From within cluster
kubectl run genops-validator --rm -it --restart=Never \
  --image=python:3.11 \
  --namespace genops-ai \
  -- bash -c "
    pip install requests && \
    curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/otel_collector_validation.py && \
    curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/observability/validate_otel_collector.py && \
    python validate_otel_collector.py --endpoint http://otel-collector:4318 --verbose
  "
```

**Validation checks:**
- ✅ OTel Collector health endpoint (port 13133)
- ✅ OTLP HTTP endpoint accessibility (port 4318)
- ✅ OTLP gRPC endpoint accessibility (port 4317)
- ✅ Backend services connectivity
- ✅ OpenTelemetry dependencies installed

---

### Health Checks

The chart provides comprehensive health endpoints:

- `/health` - Basic liveness check
- `/ready` - Readiness with dependency validation
- `/metrics` - Prometheus metrics endpoint

**Test health endpoints:**
```bash
# Port forward to service
kubectl port-forward -n genops-ai service/genops-ai 8000:8000

# Check liveness
curl http://localhost:8000/health

# Check readiness (includes dependency checks)
curl http://localhost:8000/ready

# View metrics
curl http://localhost:8000/metrics
```

---

### Debug Mode

**Enable debug logging for troubleshooting:**

```yaml
# In values.yaml
deployment:
  env:
    - name: LOG_LEVEL
      value: "DEBUG"
    - name: OTEL_LOG_LEVEL
      value: "debug"
```

**Apply changes:**
```bash
helm upgrade genops-ai genops/genops-ai \
  --namespace genops-ai \
  --values values-debug.yaml \
  --wait
```

**View debug logs:**
```bash
kubectl logs -n genops-ai -l app.kubernetes.io/name=genops-ai -f --tail=100
```

---

### Additional Resources

- **5-Minute Quickstart**: [docs/otel-collector-quickstart.md](../../docs/otel-collector-quickstart.md)
- **Comprehensive Integration Guide**: [docs/integrations/otel-collector.md](../../docs/integrations/otel-collector.md)
- **Kubernetes Troubleshooting**: [docs/kubernetes-troubleshooting.md](../../docs/kubernetes-troubleshooting.md)
- **GitHub Issues**: [https://github.com/KoshiHQ/GenOps-AI/issues](https://github.com/KoshiHQ/GenOps-AI/issues)

## Upgrading

```bash
# Update repository
helm repo update

# Check what will change
helm diff upgrade genops-ai genops/genops-ai \
  --namespace genops-ai \
  --values values-production.yaml

# Upgrade with zero downtime
helm upgrade genops-ai genops/genops-ai \
  --namespace genops-ai \
  --values values-production.yaml \
  --wait
```

## Values Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.environment` | Environment name | `production` |
| `deployment.replicaCount` | Number of replicas | `3` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `providers.openai.enabled` | Enable OpenAI provider | `true` |
| `opentelemetry.enabled` | Enable telemetry | `true` |
| `governance.policies.enabled` | Enable policy enforcement | `true` |
| `monitoring.serviceMonitor.enabled` | Create ServiceMonitor | `false` |

See [values.yaml](values.yaml) for complete configuration options.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Test your changes: `helm lint charts/genops-ai`
4. Submit a pull request

## Support

- **Documentation**: [GenOps AI Docs](https://docs.genops.ai)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)  
- **Community**: [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

## License

Licensed under the Apache License, Version 2.0.