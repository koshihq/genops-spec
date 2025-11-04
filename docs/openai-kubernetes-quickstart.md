# OpenAI + Kubernetes - 5 Minute Quickstart

Deploy OpenAI with GenOps AI governance in Kubernetes in under 5 minutes. Get immediate cost tracking, policy enforcement, and performance monitoring for your GPT applications.

## ⚡ Quick Setup (2 minutes)

### 1. Deploy OpenAI Service with GenOps

```bash
# Single command deployment with OpenAI configuration
helm repo add genops https://charts.genops.ai && helm repo update
helm install openai-service genops/genops-ai \
  --namespace genops-openai \
  --create-namespace \
  --set global.environment=production \
  --set providers.openai.enabled=true \
  --set secrets.apiKeys.openai="YOUR_OPENAI_API_KEY" \
  --set governance.policies.costLimits.daily=100.00
```

**Done!** You now have a production-ready OpenAI service with governance.

## ✅ Test Your OpenAI Service (1 minute)

### Make Your First Governed AI Request

```bash
# Port-forward to access the service
kubectl port-forward -n genops-openai service/genops-ai 8080:8000 &

# Make an OpenAI request with automatic governance
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello from Kubernetes!"}],
    "max_tokens": 50,
    "team": "engineering",
    "customer_id": "demo-customer"
  }'
```

**Response includes:**
- ✅ **OpenAI completion** (your AI response)
- ✅ **Cost tracking** (automatically calculated)
- ✅ **Performance metrics** (response time, tokens used)
- ✅ **Governance data** (team attribution, policy compliance)

## 🎯 Immediate Value

Your OpenAI service now provides:

| Feature | What You Get | Example |
|---------|--------------|---------|
| **Cost Tracking** | Real-time spend by team/customer | `$0.0023 for demo-customer` |
| **Performance Monitoring** | Response times and throughput | `245ms avg response time` |
| **Token Analytics** | Usage patterns and optimization | `1,250 tokens/hour peak usage` |
| **Policy Enforcement** | Rate limits and budget controls | `60 req/min limit enforced` |
| **Security Auditing** | Complete request audit trail | `All requests logged with context` |

## 🔍 See Your Governance Data (1 minute)

### View Real-Time Metrics

```bash
# Check OpenAI-specific metrics
curl http://localhost:8080/metrics | grep openai

# Expected metrics:
# genops_openai_requests_total
# genops_openai_cost_total_usd
# genops_openai_tokens_total
# genops_openai_duration_seconds
```

### Check Cost and Usage

```bash
# Get cost summary
curl http://localhost:8080/cost/summary | jq '.'

# Example response:
# {
#   "total_cost": 0.0156,
#   "requests_today": 12,
#   "tokens_used": 1240,
#   "average_cost_per_request": 0.0013
# }
```

## 🚀 Advanced Features (2 minutes)

### Set Up Team Budgets

```bash
# Create monthly budget for engineering team
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: engineering-openai-budget
  namespace: genops-openai
spec:
  allocation:
    amount: 2000.00
    currency: USD
    period: monthly
  attribution:
    team: engineering
    project: openai-service
  providerBudgets:
  - provider: openai
    allocation: 2000.00
    models: ["gpt-4*", "gpt-3.5-turbo*"]
  alerts:
    thresholds: [50, 80, 95]
  selector:
    matchLabels:
      team: engineering
EOF
```

### Configure Model Policies

```bash
# Set up OpenAI-specific policies
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: openai-production-policy
  namespace: genops-openai
spec:
  modelGovernance:
    enabled: true
    allowedProviders: ["openai"]
    allowedModels: ["gpt-4*", "gpt-3.5-turbo*"]
  rateLimits:
    enabled: true
    requestsPerMinute: 60
    enforcement: throttle
  contentSafety:
    enabled: true
    minimumSafetyScore: 0.85
  selector:
    matchLabels:
      ai-provider: openai
EOF
```

## 📊 Monitor Your OpenAI Usage

### Set Up Alerts

```bash
# Create budget alerts (automatically configured with the budget above)
# You'll get notifications at 50%, 80%, and 95% of budget usage

# View current budget status
kubectl get aibudget engineering-openai-budget -o yaml | grep -A 10 status
```

### Scale Based on Usage

```bash
# Auto-scaling is automatically configured based on:
# - CPU usage (70% target)
# - Request rate (10 req/sec per pod)
# - Queue length (5 requests max per pod)

# Check current scaling status
kubectl get hpa -n genops-openai
```

## 🎨 Integration Examples

### Add to Existing Applications

```yaml
# Update your existing deployment to use GenOps OpenAI service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-chat-app
spec:
  template:
    spec:
      containers:
      - name: chat-app
        env:
        - name: OPENAI_API_BASE
          value: "http://genops-ai.genops-openai:8000"
        - name: OPENAI_API_KEY
          value: "not-needed"  # GenOps handles authentication
        # Add governance labels
        - name: GENOPS_TEAM
          value: "engineering"
        - name: GENOPS_PROJECT
          value: "chat-app"
```

### Python SDK Integration

```python
# Your existing code works unchanged
import openai

# Point to GenOps service instead of OpenAI directly
openai.api_base = "http://genops-ai.genops-openai:8000"
openai.api_key = "not-needed"  # GenOps handles this

# Make requests normally - governance is automatic
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    # Optional: Add governance context
    extra_headers={
        "X-GenOps-Team": "engineering",
        "X-GenOps-Customer": "demo-customer"
    }
)
```

## 🔧 Common Issues & Quick Fixes

### Issue: OpenAI API key not working

**Quick Fix:**
```bash
# Verify secret is created correctly
kubectl get secret genops-secrets -n genops-openai -o yaml
echo "YOUR_KEY" | base64  # Compare with stored value
```

### Issue: No cost data appearing

**Quick Fix:**
```bash
# Check OpenAI connectivity
kubectl exec -n genops-openai deployment/genops-ai -- \
  python -c "
from genops.providers.openai import validate_setup
print(validate_setup())
"
```

### Issue: Requests being rejected

**Quick Fix:**
```bash
# Check policy violations
kubectl get events -n genops-openai --field-selector reason=PolicyViolation
kubectl logs -n genops-openai -l app=genops-ai | grep -i policy
```

## ✨ What You've Achieved

In 5 minutes you now have:

✅ **Production OpenAI Service** with high availability and auto-scaling  
✅ **Complete Cost Visibility** with real-time tracking and attribution  
✅ **Policy Enforcement** with budget limits and rate limiting  
✅ **Performance Monitoring** with detailed metrics and alerting  
✅ **Security Compliance** with audit trails and content safety  
✅ **Zero Code Changes** to existing applications  

### Ready for Scale?

This quickstart works for production workloads. For enterprise features:

- 📖 **Full OpenAI Guide**: [Complete OpenAI Integration](examples/kubernetes/openai/)
- 🔒 **Security**: [OpenAI Security Patterns](examples/kubernetes/openai/#security)
- 📊 **Advanced Monitoring**: [OpenAI Dashboards](examples/kubernetes/monitoring/)
- 🏢 **Multi-Tenant**: [Team Isolation](examples/kubernetes/production-patterns/multi-tenant/)

---

**⏱️ Setup time: 5 minutes**  
**🎯 Result: Production-ready OpenAI with governance**  
**💰 Immediate cost visibility and control**  
**🚀 Scales automatically with your usage**