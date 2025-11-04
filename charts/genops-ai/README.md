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

**Pods not starting:**
```bash
kubectl describe pods -n genops-ai
kubectl logs -n genops-ai -l app.kubernetes.io/name=genops-ai
```

**API key issues:**
```bash
kubectl get secrets -n genops-ai
kubectl exec -n genops-ai deployment/genops-ai -- \
  python -c "from genops.providers.openai import validate_setup; print(validate_setup())"
```

**Resource constraints:**
```bash
kubectl top pods -n genops-ai
kubectl get hpa -n genops-ai
```

### Health Checks

The chart provides comprehensive health endpoints:

- `/health` - Basic liveness check
- `/ready` - Readiness with dependency validation  
- `/metrics` - Prometheus metrics endpoint

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