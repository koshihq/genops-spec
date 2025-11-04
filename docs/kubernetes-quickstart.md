# GenOps AI Kubernetes - 5 Minute Quickstart

Get GenOps AI governance running in your Kubernetes cluster in under 5 minutes with zero configuration changes to your existing AI applications.

## ⚡ Quick Setup (2 minutes)

### 1. Install GenOps with Helm

```bash
# Add repository and install in one command
helm repo add genops https://charts.genops.ai && helm repo update
helm install genops genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set secrets.apiKeys.openai="YOUR_OPENAI_API_KEY"
```

### 2. Enable AI Governance on Your Namespace

```bash
# Enable governance on your existing AI workloads
kubectl label namespace YOUR_NAMESPACE genops.ai/injection=enabled
```

**That's it!** GenOps is now tracking all AI requests in your labeled namespaces.

## ✅ Verify It's Working (1 minute)

### Check GenOps Status

```bash
# Verify GenOps is running
kubectl get pods -n genops-system

# Should show:
# NAME                              READY   STATUS    RESTARTS
# genops-ai-xxx                    1/1     Running   0
```

### View Your AI Governance Dashboard

```bash
# Port-forward to access metrics
kubectl port-forward -n genops-system service/genops-ai 8080:8000

# Open in browser: http://localhost:8080/health
# Should return: {"status": "healthy", "kubernetes": true}
```

### Test with a Sample AI Request (if you have existing AI workloads)

Your existing AI applications now automatically include:
- ✅ **Cost tracking** by team/customer/project
- ✅ **Performance monitoring** with traces
- ✅ **Policy compliance** checking
- ✅ **Budget enforcement** (optional)

## 🎯 Immediate Value

You now have:

| Feature | What You Get |
|---------|--------------|
| **Cost Visibility** | See AI spending by team, project, and customer |
| **Performance Tracking** | Monitor response times and error rates |
| **Usage Analytics** | Track token consumption and model usage |
| **Compliance Monitoring** | Automatic policy compliance checking |
| **Security Auditing** | Complete audit trail of AI operations |

## 🔍 See Your Data

### View Metrics

```bash
# Get real-time metrics
kubectl port-forward -n genops-system service/genops-ai 8080:8000
curl http://localhost:8080/metrics | grep genops
```

### Check Governance Status

```bash
# View any policy violations or budget alerts
kubectl get events -n genops-system --field-selector reason=PolicyViolation
kubectl get events -n genops-system --field-selector reason=BudgetAlert
```

## 🚀 Next Steps (Optional - takes 5 more minutes)

### Add More Detailed Governance

```bash
# Create a budget for your team
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: my-team-budget
  namespace: genops-system
spec:
  allocation:
    amount: 1000.00
    currency: USD
    period: monthly
  attribution:
    team: my-team
  selector:
    matchLabels:
      team: my-team
EOF
```

### Set Up Cost Limits

```bash
# Create cost policies
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: my-team-policy
  namespace: genops-system
spec:
  costLimits:
    daily: 50.00
    monthly: 1000.00
    enforcement: warn
  rateLimits:
    requestsPerMinute: 60
  selector:
    matchLabels:
      team: my-team
EOF
```

## 🔧 Common Issues & Fixes

### Issue: Pods not showing governance data

**Quick Fix:**
```bash
# Ensure your pods have the right labels
kubectl label pods -l app=my-ai-app team=my-team
kubectl label pods -l app=my-ai-app genops.ai/enable=true
```

### Issue: No cost data appearing

**Quick Fix:**
```bash
# Verify your API key is set correctly
kubectl get secret genops-secrets -n genops-system -o yaml
```

### Issue: GenOps not detecting Kubernetes environment

**Quick Fix:**
```bash
# Run validation check
kubectl exec -n genops-system deployment/genops-ai -- \
  python -c "
from genops.providers.kubernetes import validate_kubernetes_setup, print_kubernetes_validation_result
result = validate_kubernetes_setup()
print_kubernetes_validation_result(result)
"
```

## 🎉 Success! You now have AI governance running in Kubernetes

Your AI workloads are now automatically tracked for:
- **Cost attribution** across teams and projects
- **Performance monitoring** with detailed metrics  
- **Policy compliance** with automatic enforcement
- **Budget management** with real-time tracking
- **Security auditing** with comprehensive logs

### What Just Happened?

1. **GenOps was deployed** to your cluster with governance capabilities
2. **Your AI workloads were labeled** for automatic tracking
3. **Telemetry collection started** capturing cost, performance, and compliance data
4. **Policies were activated** to ensure responsible AI usage

### Ready for Production?

This quickstart gives you immediate value. For production deployments:

- 📖 **Full Guide**: [Complete Kubernetes Integration](kubernetes/)
- 🔒 **Security**: [Enterprise Security Setup](kubernetes/security/)
- 📊 **Monitoring**: [Observability Stack](kubernetes/monitoring/)
- 🏢 **Multi-Tenant**: [Enterprise Patterns](kubernetes/production-patterns/)

---

**⏱️ Total time: 5 minutes or less**  
**🎯 Immediate value: Complete AI governance visibility**  
**🚀 Zero application changes required**