# OpenAI Kubernetes Deployment

Production-ready Kubernetes deployment for GenOps AI with OpenAI integration, featuring comprehensive governance, monitoring, and security.

## Features

✅ **OpenAI GPT Integration**: GPT-4, GPT-3.5 Turbo support  
✅ **Cost Attribution**: Team and customer-level cost tracking  
✅ **Policy Enforcement**: Rate limiting, content safety, budget controls  
✅ **High Availability**: Multi-replica deployment with auto-scaling  
✅ **Enterprise Security**: RBAC, NetworkPolicies, Pod Security Standards  
✅ **Observability**: OpenTelemetry integration with Prometheus metrics  

## Quick Start

### Prerequisites

- Kubernetes 1.20+
- OpenAI API key
- GenOps Helm chart or manual deployment

### Using Helm Chart

```bash
# Install with OpenAI configuration
helm install openai-service genops/genops-ai \
  --namespace genops-openai \
  --create-namespace \
  --values values-openai.yaml
```

### Manual Deployment

```bash
# Create namespace
kubectl create namespace genops-openai

# Apply all resources
kubectl apply -f .
```

## Configuration

### OpenAI API Key

Create secret with your OpenAI API key:

```bash
# Create secret
kubectl create secret generic openai-secrets \
  --namespace genops-openai \
  --from-literal=api-key="sk-..."

# Or using YAML
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: openai-secrets
  namespace: genops-openai
type: Opaque
data:
  api-key: $(echo -n "sk-..." | base64)
EOF
```

### Environment Variables

Key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_ORG_ID` | OpenAI organization ID | Optional |
| `DEFAULT_MODEL` | Default GPT model | `gpt-3.5-turbo` |
| `MAX_TOKENS` | Default max tokens | `4000` |
| `TEMPERATURE` | Default temperature | `0.7` |
| `DEFAULT_TEAM` | Default team attribution | `openai-team` |

## Governance Policies

### Cost Management

```yaml
# Apply cost limits
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: openai-cost-policy
  namespace: genops-openai
spec:
  costLimits:
    enabled: true
    daily: 100.00
    monthly: 2500.00
    enforcement: throttle
  selector:
    matchLabels:
      ai-provider: openai
```

### Rate Limiting

```yaml
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: openai-rate-policy
  namespace: genops-openai
spec:
  rateLimits:
    enabled: true
    requestsPerMinute: 60    # OpenAI tier-1 limit
    requestsPerHour: 3000
    enforcement: throttle
  selector:
    matchLabels:
      ai-provider: openai
```

### Model Governance

```yaml
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: openai-model-policy
  namespace: genops-openai
spec:
  modelGovernance:
    enabled: true
    allowedProviders: ["openai"]
    allowedModels:
    - "gpt-4*"
    - "gpt-3.5-turbo*"
    - "text-embedding-*"
    costPerToken:
      input: 0.00001   # GPT-4 Turbo pricing
      output: 0.00003
  selector:
    matchLabels:
      ai-provider: openai
```

## Budget Allocation

```yaml
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: openai-team-budget
  namespace: genops-openai
spec:
  allocation:
    amount: 5000.00
    currency: USD
    period: monthly
  attribution:
    team: openai-team
    project: chat-assistant
    costCenter: engineering
  alerts:
    enabled: true
    thresholds:
    - percentage: 80
      severity: warning
    - percentage: 95
      severity: critical
  providerBudgets:
  - provider: openai
    allocation: 5000.00
    models: ["gpt-4*", "gpt-3.5-turbo*"]
  selector:
    matchLabels:
      ai-provider: openai
```

## API Usage Examples

### Chat Completions

```bash
# Basic chat completion
curl -X POST http://openai-service:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "team": "engineering",
    "customer_id": "customer-123"
  }'

# GPT-4 with governance attributes
curl -X POST http://openai-service:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Analyze this data"}],
    "max_tokens": 500,
    "temperature": 0.3,
    "team": "data-science",
    "project": "analytics",
    "customer_id": "enterprise-456",
    "data_classification": "confidential"
  }'
```

### Embeddings

```bash
# Text embeddings
curl -X POST http://openai-service:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-3-small",
    "input": "Sample text for embedding",
    "team": "search-team",
    "customer_id": "customer-789"
  }'
```

### Cost Estimation

```bash
# Estimate costs before making requests
curl -X POST http://openai-service:8000/cost/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "input_tokens": 100,
    "output_tokens": 200
  }'
```

## Monitoring

### Health Checks

```bash
# Service health
curl http://openai-service:8000/health

# Detailed status with OpenAI connectivity
curl http://openai-service:8000/ready

# Metrics for Prometheus
curl http://openai-service:8000/metrics
```

### Key Metrics

Monitor these OpenAI-specific metrics:

```promql
# Request rate
rate(genops_openai_requests_total[5m])

# Cost tracking
genops_openai_cost_total_usd

# Token usage
rate(genops_openai_tokens_total[5m])

# Error rate
rate(genops_openai_errors_total[5m]) / rate(genops_openai_requests_total[5m])

# Model usage distribution
genops_openai_requests_total by (model)

# Response latency
histogram_quantile(0.95, genops_openai_duration_seconds_bucket)
```

### Grafana Queries

```promql
# Top models by cost
topk(5, sum by (model) (genops_openai_cost_total_usd))

# Daily cost trend
increase(genops_openai_cost_total_usd[1d])

# Cost by team
sum by (team) (genops_openai_cost_total_usd)

# Token efficiency (cost per token)
genops_openai_cost_total_usd / genops_openai_tokens_total
```

## Security

### OpenAI API Key Management

- Store API key in Kubernetes Secret
- Use service account for RBAC
- Rotate keys regularly via external secret management
- Monitor for unusual usage patterns

### Network Security

```yaml
# Network policy for OpenAI service
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: openai-service-policy
spec:
  podSelector:
    matchLabels:
      app: openai-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
  egress:
  - to: []  # OpenAI APIs
    ports:
    - protocol: TCP
      port: 443
```

### Content Safety

```yaml
# Content safety policy for OpenAI
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: openai-safety-policy
spec:
  contentSafety:
    enabled: true
    minimumSafetyScore: 0.85
    categories: ["hate", "harassment", "self-harm", "sexual", "violence"]
    enforcement: block
  selector:
    matchLabels:
      ai-provider: openai
```

## Scaling

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: openai-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openai-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: genops_openai_requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: openai-service-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openai-service
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: openai-service
      maxAllowed:
        cpu: 1000m
        memory: 2Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

## Production Checklist

### Pre-deployment

- [ ] OpenAI API key configured and tested
- [ ] Resource limits and requests set appropriately
- [ ] Network policies configured for security
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery plan
- [ ] Load testing completed

### Post-deployment

- [ ] Health checks passing
- [ ] Metrics flowing to monitoring system
- [ ] Auto-scaling working correctly
- [ ] Policy enforcement active
- [ ] Cost attribution functioning
- [ ] Security scans completed

## Troubleshooting

### Common Issues

**OpenAI API authentication failures:**
```bash
# Check secret
kubectl get secret openai-secrets -n genops-openai -o yaml

# Test API key manually
kubectl exec -n genops-openai deployment/openai-service -- \
  python -c "
import openai
openai.api_key = '$OPENAI_API_KEY'
print(openai.models.list())
"
```

**Rate limit exceeded:**
```bash
# Check current rate limits
kubectl logs -n genops-openai -l app=openai-service | grep "rate_limit"

# Review policy configuration
kubectl get aipolicy openai-rate-policy -o yaml
```

**High costs:**
```bash
# Check cost breakdown
curl http://openai-service:8000/metrics | grep cost

# Review budget status
kubectl get aibudget openai-team-budget -o yaml
```

### Debug Mode

Enable debug logging:

```bash
kubectl patch deployment openai-service -n genops-openai -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "openai-service",
            "env": [
              {
                "name": "LOG_LEVEL",
                "value": "debug"
              }
            ]
          }
        ]
      }
    }
  }
}'
```

## Support

- **OpenAI Documentation**: https://platform.openai.com/docs
- **GenOps Integration**: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/integrations/openai
- **Community**: https://github.com/KoshiHQ/GenOps-AI/discussions