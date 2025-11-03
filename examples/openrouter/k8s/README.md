# Kubernetes Deployment Guide for GenOps OpenRouter Service

This directory contains production-ready Kubernetes manifests for deploying the GenOps OpenRouter service with comprehensive AI governance capabilities.

## Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured for your cluster
- OpenRouter API key from [openrouter.ai/keys](https://openrouter.ai/keys)
- Observability platform (Honeycomb, Datadog, etc.)

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/your-org/genops-ai.git
cd genops-ai/examples/openrouter/k8s

# Update secrets with your API keys
# Edit secrets.yaml and replace base64 encoded values:
echo -n "your-openrouter-api-key" | base64
echo -n "your-honeycomb-api-key" | base64
```

### 2. Deploy to Kubernetes

```bash
# Create namespace and deploy all resources
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml
kubectl apply -f serviceaccount.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
kubectl apply -f networkpolicy.yaml

# Or deploy everything at once
kubectl apply -f .
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n genops-openrouter

# Check service health
kubectl port-forward -n genops-openrouter service/openrouter-service-internal 8080:8000
curl http://localhost:8080/health

# Check logs
kubectl logs -n genops-openrouter -l app.kubernetes.io/name=openrouter-service -f
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│   Ingress       │───▶│   Pods (3-20)   │
│   (AWS NLB)     │    │   (NGINX)       │    │   Auto-scaling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TLS/SSL       │    │   Rate Limiting │    │   OpenRouter    │
│   (Let's Encrypt)│    │   CORS Headers  │    │   400+ Models   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────┐
                                          │  OpenTelemetry  │
                                          │  (Honeycomb)    │
                                          └─────────────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes | - |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Observability endpoint | Yes | - |
| `OTEL_EXPORTER_OTLP_HEADERS` | OTLP headers | Yes | - |
| `ENVIRONMENT` | Environment name | No | `production` |
| `DEFAULT_TEAM` | Default team for attribution | No | `platform` |
| `DEFAULT_PROJECT` | Default project name | No | `k8s-openrouter-service` |

### Resource Requirements

**Per Pod:**
- **CPU**: 250m requests, 500m limits
- **Memory**: 512Mi requests, 1Gi limits
- **Storage**: 500Mi ephemeral storage

**Cluster Requirements:**
- **Minimum**: 3 worker nodes (for pod anti-affinity)
- **Recommended**: 5+ worker nodes for high availability
- **Auto-scaling**: 3-20 pods based on CPU/memory usage

## Security Features

### Pod Security

- **Non-root user**: Runs as UID 1000
- **Read-only filesystem**: Prevents runtime modification
- **No privilege escalation**: Enhanced container security
- **Dropped capabilities**: All Linux capabilities removed

### Network Security

- **Network policies**: Restrict ingress/egress traffic
- **TLS encryption**: End-to-end HTTPS with Let's Encrypt
- **Rate limiting**: 100 requests/minute per IP
- **CORS protection**: Configured for web API access

### Secrets Management

- **Kubernetes secrets**: API keys stored securely
- **Base64 encoding**: Standard Kubernetes secret format
- **RBAC**: Minimal permissions for service account
- **Auto-mounting**: Service account tokens when needed

## Scaling and Performance

### Horizontal Pod Autoscaling (HPA)

```yaml
# Automatic scaling based on:
CPU Utilization: 70% target
Memory Utilization: 80% target
Min Replicas: 3
Max Replicas: 20

# Scale-up: Aggressive (100% increase every 30s)
# Scale-down: Conservative (50% decrease every 60s)
```

### Load Balancing

- **Service type**: LoadBalancer (AWS NLB)
- **Pod anti-affinity**: Distributes pods across nodes  
- **Rolling updates**: Zero-downtime deployments
- **Health checks**: Liveness, readiness, and startup probes

## Observability

### Health Endpoints

- **`/health`**: Liveness probe - basic service health
- **`/ready`**: Readiness probe - dependency validation
- **`/metrics`**: Prometheus metrics (if enabled)

### OpenTelemetry Integration

All requests automatically generate rich telemetry:
- **Traces**: Request flow and timing
- **Metrics**: Request counts, latencies, errors
- **Logs**: Structured JSON logging
- **Attributes**: Team, project, customer attribution

### Supported Platforms

- **Honeycomb**: `x-honeycomb-team=API_KEY`
- **Datadog**: `dd-api-key=API_KEY`
- **New Relic**: `api-key=API_KEY`
- **Jaeger**: Native OTLP support
- **Grafana Tempo**: Native OTLP support

## API Usage

### Chat Completions

```bash
curl -X POST https://openrouter-api.your-domain.com/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "team": "production",
    "customer_id": "customer-123"
  }'
```

### Cost Estimation

```bash
curl -X POST https://openrouter-api.your-domain.com/cost/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "input_tokens": 100,
    "output_tokens": 50
  }'
```

### Health Check

```bash
curl https://openrouter-api.your-domain.com/health
```

## Deployment Strategies

### Blue-Green Deployment

```bash
# 1. Deploy new version to staging namespace
kubectl apply -f . -n genops-openrouter-staging

# 2. Validate staging deployment
kubectl port-forward -n genops-openrouter-staging service/openrouter-service-internal 8080:8000
curl http://localhost:8080/health

# 3. Switch traffic (update ingress)
kubectl patch ingress openrouter-service-ingress -n genops-openrouter \
  -p '{"spec":{"rules":[{"host":"openrouter-api.your-domain.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"openrouter-service-internal","port":{"number":8000}}}}]}}]}}'

# 4. Monitor and rollback if needed
kubectl rollout undo deployment/openrouter-service -n genops-openrouter
```

### Canary Deployment

```bash
# 1. Deploy canary version
kubectl patch deployment openrouter-service -n genops-openrouter \
  -p '{"spec":{"template":{"metadata":{"labels":{"version":"v2"}}}}}'

# 2. Configure traffic split (using Istio/Linkerd)
# Route 10% to v2, 90% to v1

# 3. Monitor metrics and gradually increase traffic
# 4. Complete rollout when confident
```

## Troubleshooting

### Common Issues

**Pods not starting:**
```bash
# Check pod events
kubectl describe pods -n genops-openrouter

# Check logs
kubectl logs -n genops-openrouter -l app.kubernetes.io/name=openrouter-service --tail=50
```

**Health check failures:**
```bash
# Test health endpoint directly
kubectl exec -n genops-openrouter deployment/openrouter-service -- \
  curl -f http://localhost:8000/health

# Check OpenRouter API connectivity
kubectl exec -n genops-openrouter deployment/openrouter-service -- \
  python -c "from genops.providers.openrouter import validate_setup; print(validate_setup().is_valid)"
```

**High error rates:**
```bash
# Check resource usage
kubectl top pods -n genops-openrouter

# Check HPA status
kubectl get hpa -n genops-openrouter

# Scale manually if needed
kubectl scale deployment openrouter-service -n genops-openrouter --replicas=10
```

### Monitoring Queries

**Honeycomb Queries:**
```sql
-- Request rate by team
COUNT | WHERE genops.provider = "openrouter" | GROUP BY genops.team

-- High-cost requests  
AVG(genops.cost.total) | WHERE genops.cost.total > 0.01 | GROUP BY genops.model

-- Error rate by endpoint
COUNT | WHERE http.status_code >= 400 | GROUP BY http.route
```

**Prometheus Queries:**
```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

## Maintenance

### Update Deployment

```bash
# Update image version
kubectl set image deployment/openrouter-service -n genops-openrouter \
  openrouter-service=your-registry/openrouter-service:v1.1.0

# Check rollout status
kubectl rollout status deployment/openrouter-service -n genops-openrouter
```

### Backup Configuration

```bash
# Export all configurations
kubectl get all,secrets,configmaps,ingress,networkpolicies -n genops-openrouter -o yaml > backup.yaml

# Restore from backup
kubectl apply -f backup.yaml
```

### Resource Cleanup

```bash
# Delete all resources
kubectl delete namespace genops-openrouter

# Delete specific resources
kubectl delete -f .
```

## Production Checklist

### Pre-deployment

- [ ] OpenRouter API key configured
- [ ] Observability platform configured  
- [ ] TLS certificates configured
- [ ] Resource limits appropriate
- [ ] Network policies reviewed
- [ ] RBAC permissions minimal

### Post-deployment

- [ ] Health checks passing
- [ ] Telemetry flowing to observability platform
- [ ] Auto-scaling working correctly
- [ ] Load balancer routing traffic
- [ ] API endpoints responding
- [ ] Error monitoring configured
- [ ] Backup procedures tested

## Support

- **Documentation**: [Full Integration Guide](../../docs/integrations/openrouter.md)
- **Examples**: [OpenRouter Examples](../)
- **Issues**: GitHub Issues
- **Community**: Discussions

---

**Production Ready**: This deployment has been tested with production workloads and includes enterprise-grade security, monitoring, and scalability features.