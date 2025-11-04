# Multi-Provider AI + Kubernetes - 5 Minute Quickstart

Deploy a unified AI gateway supporting OpenAI, Anthropic, and OpenRouter in Kubernetes with intelligent routing, cost optimization, and unified governance - all in under 5 minutes.

## ⚡ Quick Setup (2 minutes)

### 1. Deploy Multi-Provider AI Gateway

```bash
# Deploy with multiple AI providers configured
helm repo add genops https://charts.genops.ai && helm repo update
helm install ai-gateway genops/genops-ai \
  --namespace genops-multi \
  --create-namespace \
  --set global.environment=production \
  --set providers.openai.enabled=true \
  --set providers.anthropic.enabled=true \
  --set providers.openrouter.enabled=true \
  --set secrets.apiKeys.openai="YOUR_OPENAI_API_KEY" \
  --set secrets.apiKeys.anthropic="YOUR_ANTHROPIC_API_KEY" \
  --set secrets.apiKeys.openrouter="YOUR_OPENROUTER_API_KEY" \
  --set governance.policies.costLimits.daily=200.00
```

**Done!** You now have an intelligent AI gateway that routes between providers for optimal cost and performance.

## ✅ Test Multi-Provider Intelligence (1 minute)

### Make Smart-Routed Requests

```bash
# Port-forward to access the gateway
kubectl port-forward -n genops-multi service/genops-ai 8080:8000 &

# Let GenOps choose the best provider for you
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello! Route me optimally."}],
    "routing_preference": "cost-optimized",
    "team": "engineering",
    "customer_id": "demo-customer"
  }'

# Force a specific provider if needed
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Use Claude for this request"}],
    "provider": "anthropic",
    "model": "claude-3-sonnet",
    "team": "engineering"
  }'
```

**The gateway automatically:**
- ✅ **Selects optimal provider** based on cost and performance
- ✅ **Tracks costs** across all providers with unified attribution  
- ✅ **Handles failover** if a provider is unavailable
- ✅ **Enforces policies** consistently across all providers

## 🎯 Immediate Multi-Provider Value

Your AI gateway provides:

| Feature | What You Get | Example |
|---------|--------------|---------|
| **Intelligent Routing** | Best provider per request | `OpenRouter for simple tasks, Claude for analysis` |
| **Unified Cost Tracking** | Single view across all providers | `$0.0012 OpenAI + $0.0018 Anthropic = $0.003 total` |
| **Automatic Failover** | No downtime when providers fail | `OpenAI down → Auto-route to Anthropic` |
| **Cost Optimization** | Always use cheapest suitable provider | `30% cost savings through smart routing` |
| **Performance Balancing** | Route based on response times | `High priority → fastest provider` |

## 🔍 See Cross-Provider Analytics (1 minute)

### Compare Provider Performance

```bash
# Get cross-provider metrics
curl http://localhost:8080/metrics | grep -E "(openai|anthropic|openrouter)"

# Key metrics to watch:
# genops_openai_cost_total_usd
# genops_anthropic_cost_total_usd  
# genops_openrouter_cost_total_usd
# genops_routing_decisions_total
```

### View Cost Optimization

```bash
# Get routing analytics
curl http://localhost:8080/analytics/routing | jq '.'

# Example response:
# {
#   "cost_savings": 0.0456,
#   "routing_decisions": {
#     "openrouter": 45,
#     "openai": 32, 
#     "anthropic": 23
#   },
#   "average_cost_per_request": 0.0089
# }
```

## 🚀 Advanced Multi-Provider Features (2 minutes)

### Set Up Provider-Specific Budgets

```bash
# Create cross-provider budget allocation
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: multi-provider-budget
  namespace: genops-multi
spec:
  allocation:
    amount: 5000.00
    currency: USD
    period: monthly
  attribution:
    team: engineering
    project: multi-provider-ai
  # Allocate budget across providers
  providerBudgets:
  - provider: openai
    allocation: 2000.00  # 40% to OpenAI
  - provider: anthropic
    allocation: 2000.00  # 40% to Anthropic
  - provider: openrouter
    allocation: 1000.00  # 20% to OpenRouter
  alerts:
    thresholds: [60, 80, 95]
  selector:
    matchLabels:
      deployment: multi-provider-gateway
EOF
```

### Configure Intelligent Routing Rules

```bash
# Set up routing optimization
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: routing-config
  namespace: genops-multi
data:
  routing.yaml: |
    strategies:
      cost-optimized:
        enabled: true
        providers:
          - name: openrouter
            priority: 1
            cost_weight: 0.8
          - name: openai
            priority: 2
            cost_weight: 1.0
          - name: anthropic
            priority: 3
            cost_weight: 1.2
        rules:
          - condition: "tokens < 500"
            provider: "openrouter"
          - condition: "safety_critical == true"
            provider: "anthropic"
          - condition: "task_type == 'embedding'"
            provider: "openai"
EOF
```

### Set Up Cross-Provider Policies

```bash
# Unified governance across all providers
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: unified-provider-policy
  namespace: genops-multi
spec:
  costLimits:
    enabled: true
    daily: 200.00
    enforcement: throttle
  rateLimits:
    enabled: true
    requestsPerMinute: 120  # Combined across providers
    enforcement: throttle
  modelGovernance:
    enabled: true
    routing:
      gpt-4: 
        provider: openai
        fallback: openrouter
      claude-3:
        provider: anthropic
        fallback: openrouter
  contentSafety:
    enabled: true
    minimumSafetyScore: 0.85
    crossProviderValidation: true
  selector:
    matchLabels:
      component: multi-provider-gateway
EOF
```

## 📊 Monitor Multi-Provider Performance

### Real-Time Provider Health

```bash
# Check provider health and routing decisions
curl http://localhost:8080/health/providers | jq '.'

# Example response:
# {
#   "openai": {"status": "healthy", "latency_ms": 245, "error_rate": 0.02},
#   "anthropic": {"status": "healthy", "latency_ms": 189, "error_rate": 0.01}, 
#   "openrouter": {"status": "healthy", "latency_ms": 312, "error_rate": 0.03}
# }
```

### Cost Optimization Analytics

```bash
# View cost savings from intelligent routing
kubectl get aibudget multi-provider-budget -o yaml | grep -A 20 breakdown

# Check routing effectiveness
kubectl logs -n genops-multi -l app=genops-ai | grep -i "routing_decision" | tail -10
```

## 🎨 Integration Patterns

### Auto-Failover Integration

```python
# Your applications automatically get failover
import requests

# The gateway handles provider selection and failover
response = requests.post("http://ai-gateway.genops-multi:8000/chat/completions", 
    json={
        "messages": [{"role": "user", "content": "Hello!"}],
        "routing_preference": "performance-optimized",
        "team": "engineering",
        "fallback_enabled": True  # Automatic provider failover
    }
)
```

### Cost-Aware Applications

```python
# Get cost estimates before making requests
cost_estimate = requests.post("http://ai-gateway.genops-multi:8000/cost/estimate",
    json={
        "messages": [{"role": "user", "content": "Complex analysis task"}],
        "providers": ["openai", "anthropic", "openrouter"],
        "return_cheapest": True
    }
)

print(f"Cheapest option: {cost_estimate.json()['recommended_provider']}")
print(f"Estimated cost: ${cost_estimate.json()['estimated_cost']}")
```

## 🔧 Troubleshooting Multi-Provider Setup

### Issue: Provider routing not working

**Quick Fix:**
```bash
# Check routing configuration
kubectl get configmap routing-config -n genops-multi -o yaml

# Verify provider health
kubectl exec -n genops-multi deployment/genops-ai -- \
  python -c "
from genops.providers import validate_multi_provider_setup
print(validate_multi_provider_setup())
"
```

### Issue: Cost tracking inconsistent across providers

**Quick Fix:**
```bash
# Verify all provider API keys
kubectl get secret genops-secrets -n genops-multi -o yaml

# Check provider-specific logs
kubectl logs -n genops-multi -l app=genops-ai | grep -E "(openai|anthropic|openrouter)"
```

### Issue: High failover rate

**Quick Fix:**
```bash
# Check provider health and error rates
curl http://localhost:8080/health/providers
kubectl get events -n genops-multi --field-selector reason=ProviderFailover
```

## ✨ What You've Achieved

In 5 minutes you now have:

✅ **Unified AI Gateway** supporting OpenAI, Anthropic, and OpenRouter  
✅ **Intelligent Routing** optimizing for cost and performance automatically  
✅ **Cross-Provider Budgets** with unified cost tracking and attribution  
✅ **Automatic Failover** ensuring high availability across providers  
✅ **Cost Optimization** saving 20-40% through smart provider selection  
✅ **Unified Governance** with consistent policies across all providers  

### Ready for Enterprise Scale?

This quickstart handles production workloads. For advanced features:

- 📖 **Complete Multi-Provider Guide**: [Advanced Routing Strategies](examples/kubernetes/multi-provider/)
- 🔍 **Cost Analysis**: [Cross-Provider Analytics](examples/kubernetes/multi-provider/#cost-optimization-analysis)
- 🛡️ **Security**: [Multi-Provider Security](examples/kubernetes/multi-provider/#security--compliance)
- 🏢 **Enterprise**: [Migration Strategies](examples/kubernetes/multi-provider/#migration-strategies)

---

**⏱️ Setup time: 5 minutes**  
**🎯 Result: Intelligent multi-provider AI gateway**  
**💰 Automatic cost optimization across providers**  
**🚀 High availability with automatic failover**  
**📊 Unified governance and monitoring**