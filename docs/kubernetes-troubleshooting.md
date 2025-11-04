# GenOps AI Kubernetes Troubleshooting Runbook

A comprehensive troubleshooting guide for GenOps AI deployments in Kubernetes. This runbook provides step-by-step solutions for common issues, diagnostic commands, and preventive measures.

## 🎯 Quick Diagnosis

**Start Here:** Run these commands to get an overview of your GenOps deployment health:

```bash
# Check GenOps pods status
kubectl get pods -n genops-system -l app.kubernetes.io/name=genops-ai

# Check recent events
kubectl get events -n genops-system --sort-by=.metadata.creationTimestamp | tail -10

# Quick health check
kubectl exec -n genops-system deployment/genops-ai -- curl -s http://localhost:8000/health 2>/dev/null || echo "Health endpoint unreachable"

# Check governance resources
kubectl get aibudgets,aipolicies -A --no-headers | wc -l
```

**Health Status Indicators:**
- ✅ **Healthy**: All pods Running, health endpoint returns `{"status": "healthy"}`
- ⚠️ **Degraded**: Pods running but health check issues or warnings in logs
- ❌ **Unhealthy**: Pods not running, CrashLoopBackOff, or health endpoint unreachable

---

## 🚨 Emergency Response

### GenOps Pods Not Starting

**Symptom:** Pods stuck in `Pending`, `CrashLoopBackOff`, or `ImagePullBackOff`

**Quick Fix:**
```bash
# Check pod status and events
kubectl describe pods -n genops-system -l app.kubernetes.io/name=genops-ai

# Check logs from failed pods
kubectl logs -n genops-system -l app.kubernetes.io/name=genops-ai --previous

# Restart deployment
kubectl rollout restart deployment/genops-ai -n genops-system
```

**Common Causes & Solutions:**

| Error | Cause | Solution |
|-------|-------|----------|
| `ImagePullBackOff` | Image repository access issue | Check image name and registry credentials |
| `CrashLoopBackOff` | Application startup failure | Check logs for configuration errors |
| `Pending` | Resource constraints | Check node resources and resource requests |
| `FailedScheduling` | Node selection issues | Check node selectors and taints/tolerations |

**Detailed Diagnostics:**
```bash
# Check node resources
kubectl top nodes

# Check resource quotas
kubectl describe resourcequotas -n genops-system

# Check persistent volume claims
kubectl get pvc -n genops-system

# Check service account permissions
kubectl auth can-i get pods --as=system:serviceaccount:genops-system:genops-ai
```

### Complete Service Outage

**Immediate Response:**
```bash
# Scale up replicas for faster recovery
kubectl scale deployment/genops-ai --replicas=5 -n genops-system

# Check if LoadBalancer/Ingress is working
kubectl get services,ingress -n genops-system

# Bypass GenOps temporarily (if needed)
# Direct calls to AI providers while investigating
```

**Recovery Steps:**
```bash
# 1. Restore from backup configuration
kubectl apply -f backup/genops-config-backup.yaml

# 2. Force recreate deployment
kubectl delete deployment genops-ai -n genops-system
kubectl apply -f original-deployment.yaml

# 3. Check external dependencies
kubectl get secrets -n genops-system
kubectl describe configmaps -n genops-system
```

---

## 🔧 Installation & Configuration Issues

### Helm Installation Failures

**Symptom:** Helm install/upgrade commands fail

**Diagnosis:**
```bash
# Check Helm release status
helm status genops -n genops-system

# List all releases
helm list -A

# Check Helm values
helm get values genops -n genops-system
```

**Common Solutions:**

**1. CRD Installation Issues:**
```bash
# Manual CRD installation
kubectl apply -f https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/operators/genops-controller/config/crd/bases/genops.ai_aipolicies.yaml
kubectl apply -f https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/operators/genops-controller/config/crd/bases/genops.ai_aibudgets.yaml

# Then retry Helm installation
helm install genops genops/genops-ai --namespace genops-system
```

**2. Namespace Issues:**
```bash
# Create namespace if missing
kubectl create namespace genops-system

# Check namespace labels
kubectl describe namespace genops-system
```

**3. RBAC Permission Issues:**
```bash
# Check service account
kubectl get serviceaccount -n genops-system

# Verify RBAC permissions
kubectl auth can-i create aipolicies --as=system:serviceaccount:genops-system:genops-ai
kubectl auth can-i get secrets --as=system:serviceaccount:genops-system:genops-ai
```

### Secret and Configuration Problems

**Symptom:** API keys not working or configuration not loading

**Diagnosis:**
```bash
# Check secrets exist and are properly formatted
kubectl get secrets -n genops-system
kubectl get secret genops-secrets -n genops-system -o yaml | grep -E "(openai|anthropic|azure)"

# Verify ConfigMap content
kubectl get configmaps -n genops-system
kubectl describe configmap genops-config -n genops-system
```

**Solutions:**

**1. Fix Secret Encoding:**
```bash
# Secrets must be base64 encoded
echo -n "sk-your-actual-key" | base64

# Update secret
kubectl create secret generic genops-secrets \
  --namespace genops-system \
  --from-literal=openai-api-key="sk-your-key" \
  --from-literal=anthropic-api-key="sk-ant-your-key" \
  --dry-run=client -o yaml | kubectl apply -f -
```

**2. ConfigMap Updates:**
```bash
# Update ConfigMap and restart pods
kubectl create configmap genops-config \
  --namespace genops-system \
  --from-file=config.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/genops-ai -n genops-system
```

---

## 🚫 Policy and Budget Issues

### Policies Not Enforcing

**Symptom:** AI requests not being blocked/throttled despite policy violations

**Diagnosis:**
```bash
# Check policy status
kubectl get aipolicies -A -o wide

# Check policy selectors
kubectl describe aipolicy your-policy-name -n genops-system

# Check GenOps controller logs
kubectl logs -n genops-system -l control-plane=genops-controller --since=1h
```

**Solutions:**

**1. Fix Policy Selectors:**
```bash
# Ensure labels match
kubectl get pods --show-labels | grep genops

# Update policy selector
kubectl patch aipolicy your-policy \
  --type='merge' \
  --patch='{"spec":{"selector":{"matchLabels":{"app.kubernetes.io/name":"genops-ai"}}}}'
```

**2. Verify Controller is Running:**
```bash
# Check GenOps controller
kubectl get deployment -n genops-system genops-controller

# Check webhook configuration
kubectl get validatingwebhookconfiguration genops-validating-webhook-configuration
kubectl get mutatingwebhookconfiguration genops-mutating-webhook-configuration
```

### Budget Tracking Not Working

**Symptom:** Costs not being tracked or budget limits not enforced

**Diagnosis:**
```bash
# Check budget resources
kubectl get aibudgets -A -o yaml

# Check budget status
kubectl describe aibudget your-budget-name -n genops-system

# Look for budget-related events
kubectl get events --field-selector reason=BudgetExceeded,reason=BudgetAlert -A
```

**Solutions:**

**1. Verify Budget Configuration:**
```bash
# Check budget selector matches your workloads
kubectl get pods --show-labels -l team=engineering

# Update budget selector if needed
kubectl patch aibudget team-engineering-budget \
  --type='merge' \
  --patch='{"spec":{"selector":{"matchLabels":{"team":"engineering"}}}}'
```

**2. Check Cost Attribution:**
```bash
# Verify telemetry attributes are being set
kubectl logs -n genops-system deployment/genops-ai | grep -i "cost\|attribution\|budget"

# Test a request with explicit attributes
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-GenOps-Team: engineering" \
  -H "X-GenOps-Customer: test-customer" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}'
```

---

## 💰 Cost Tracking Issues

### No Cost Data Appearing

**Symptom:** Metrics show zero costs or no cost metrics at all

**Step-by-Step Fix:**

**1. Verify API Keys:**
```bash
# Check if secrets exist and are valid
kubectl get secret genops-secrets -n genops-system -o jsonpath='{.data}' | jq -r 'to_entries[] | "\(.key): \(.value | @base64d | length) characters"'

# Test API key manually
kubectl exec -n genops-system deployment/genops-ai -- python -c "
import openai
import os
openai.api_key = os.environ.get('OPENAI_API_KEY')
try:
    models = openai.Model.list()
    print('✅ OpenAI API key valid')
except Exception as e:
    print(f'❌ OpenAI API key invalid: {e}')
"
```

**2. Check Provider Configuration:**
```bash
# Verify providers are enabled in Helm values
helm get values genops -n genops-system | grep -A 10 providers

# Check provider status in logs
kubectl logs -n genops-system deployment/genops-ai | grep -i "provider.*enabled\|provider.*disabled"
```

**3. Verify Request Flow:**
```bash
# Make a test request and watch logs
kubectl logs -n genops-system deployment/genops-ai --follow &
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Calculate cost for this request"}],
    "max_tokens": 10
  }'
```

### Incorrect Cost Calculations

**Symptom:** Costs seem too high, too low, or inconsistent

**Diagnosis:**
```bash
# Check cost calculation logs
kubectl logs -n genops-system deployment/genops-ai | grep -i "cost.*calculation\|token.*count\|pricing"

# Verify model pricing configuration
kubectl get configmap genops-config -n genops-system -o yaml | grep -A 20 pricing
```

**Solutions:**

**1. Update Pricing Data:**
```bash
# Force update pricing configuration
kubectl create configmap genops-pricing \
  --namespace genops-system \
  --from-literal=openai-pricing='{"gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}}' \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/genops-ai -n genops-system
```

**2. Debug Token Counting:**
```bash
# Enable debug logging for cost calculations
kubectl set env deployment/genops-ai -n genops-system LOG_LEVEL=DEBUG
kubectl logs -n genops-system deployment/genops-ai | grep -i "token.*count"
```

---

## 🌐 Network and Connectivity Issues

### Cannot Reach AI Providers

**Symptom:** Timeouts or connection errors to OpenAI, Anthropic, etc.

**Diagnosis:**
```bash
# Test external connectivity from pods
kubectl exec -n genops-system deployment/genops-ai -- nslookup api.openai.com
kubectl exec -n genops-system deployment/genops-ai -- wget -qO- --timeout=10 https://api.openai.com/v1/models || echo "Connection failed"

# Check network policies
kubectl get networkpolicies -n genops-system
```

**Solutions:**

**1. Fix DNS Resolution:**
```bash
# Check DNS configuration
kubectl exec -n genops-system deployment/genops-ai -- cat /etc/resolv.conf

# Test with different DNS servers
kubectl exec -n genops-system deployment/genops-ai -- nslookup api.openai.com 8.8.8.8
```

**2. Update Network Policies:**
```bash
# Allow HTTPS egress to AI providers
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genops-ai-egress
  namespace: genops-system
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: genops-ai
  policyTypes:
  - Egress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
EOF
```

**3. Corporate Firewall Issues:**
```bash
# Test through proxy if required
kubectl set env deployment/genops-ai -n genops-system \
  HTTPS_PROXY=http://your-proxy:8080 \
  HTTP_PROXY=http://your-proxy:8080
```

### Service Discovery Problems

**Symptom:** Internal services cannot reach GenOps API

**Quick Fix:**
```bash
# Check service configuration
kubectl get services -n genops-system
kubectl describe service genops-ai -n genops-system

# Test service resolution
kubectl exec -n default deployment/test-app -- nslookup genops-ai.genops-system.svc.cluster.local

# Port-forward for testing
kubectl port-forward -n genops-system service/genops-ai 8080:8000
```

---

## 📊 Monitoring and Observability Issues

### Missing Metrics

**Symptom:** No GenOps metrics in Prometheus/Grafana

**Diagnosis:**
```bash
# Check metrics endpoint
kubectl exec -n genops-system deployment/genops-ai -- curl http://localhost:8000/metrics

# Verify ServiceMonitor (if using Prometheus Operator)
kubectl get servicemonitor -n genops-system

# Check Prometheus targets
kubectl port-forward -n monitoring service/prometheus 9090:9090
# Navigate to http://localhost:9090/targets
```

**Solutions:**

**1. Fix ServiceMonitor:**
```bash
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: genops-ai
  namespace: genops-system
  labels:
    app.kubernetes.io/name: genops-ai
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: genops-ai
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF
```

**2. Verify Prometheus Configuration:**
```bash
# Check if Prometheus is scraping the namespace
kubectl get prometheus -n monitoring -o yaml | grep -A 5 serviceMonitorNamespaceSelector
```

### Log Aggregation Issues

**Symptom:** GenOps logs not appearing in logging system

**Solutions:**

**1. Check Log Format:**
```bash
# Verify logs are in expected format
kubectl logs -n genops-system deployment/genops-ai | head -5

# Enable structured logging
kubectl set env deployment/genops-ai -n genops-system LOG_FORMAT=json
```

**2. Fix Log Collection:**
```bash
# Check if log collection pods are running
kubectl get pods -n kube-system | grep -E "(fluentd|filebeat|fluent-bit)"

# Verify log collection configuration
kubectl get configmap -n kube-system fluentd-config -o yaml | grep -A 10 genops
```

---

## 🔄 Performance and Scaling Issues

### High Latency

**Symptom:** Slow response times from GenOps API

**Diagnosis:**
```bash
# Check pod resource usage
kubectl top pods -n genops-system

# Check HPA status
kubectl get hpa -n genops-system

# Monitor request latency
kubectl logs -n genops-system deployment/genops-ai | grep -E "duration|latency|slow"
```

**Solutions:**

**1. Scale Horizontally:**
```bash
# Increase replicas
kubectl scale deployment/genops-ai --replicas=5 -n genops-system

# Configure HPA
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genops-ai-hpa
  namespace: genops-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-ai
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
EOF
```

**2. Optimize Resources:**
```bash
# Increase CPU/memory limits
kubectl patch deployment genops-ai -n genops-system -p '{"spec":{"template":{"spec":{"containers":[{"name":"genops-ai","resources":{"requests":{"cpu":"1000m","memory":"2Gi"},"limits":{"cpu":"2000m","memory":"4Gi"}}}]}}}}'
```

### Memory Leaks

**Symptom:** Pods consuming increasing amounts of memory over time

**Diagnosis:**
```bash
# Monitor memory usage over time
kubectl top pods -n genops-system --containers=true

# Check for memory-related events
kubectl get events -n genops-system --field-selector reason=Killing,reason=FailedScheduling
```

**Solutions:**

**1. Restart Deployment Regularly:**
```bash
# Create CronJob to restart deployment periodically
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: genops-restart
  namespace: genops-system
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: genops-restart
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - kubectl
            - rollout
            - restart
            - deployment/genops-ai
            - -n
            - genops-system
          restartPolicy: OnFailure
EOF
```

---

## 🧪 Testing and Validation

### Automated Health Checks

**Create Health Check Script:**
```bash
cat > genops-health-check.sh <<'EOF'
#!/bin/bash
set -e

echo "🏥 GenOps Health Check Starting..."

# Check pods
echo "1. Checking pod status..."
kubectl get pods -n genops-system -l app.kubernetes.io/name=genops-ai --no-headers | while read pod_info; do
    pod_name=$(echo $pod_info | awk '{print $1}')
    status=$(echo $pod_info | awk '{print $3}')
    if [ "$status" != "Running" ]; then
        echo "❌ Pod $pod_name is not running: $status"
        exit 1
    fi
done
echo "✅ All pods running"

# Check health endpoint
echo "2. Checking health endpoint..."
kubectl exec -n genops-system deployment/genops-ai -- curl -sf http://localhost:8000/health > /dev/null
echo "✅ Health endpoint responding"

# Check governance resources
echo "3. Checking governance resources..."
budget_count=$(kubectl get aibudgets -A --no-headers | wc -l)
policy_count=$(kubectl get aipolicies -A --no-headers | wc -l)
echo "✅ Found $budget_count budgets and $policy_count policies"

# Test AI request (if API key available)
echo "4. Testing AI request..."
if kubectl get secret genops-secrets -n genops-system -o jsonpath='{.data.openai-api-key}' >/dev/null 2>&1; then
    kubectl exec -n genops-system deployment/genops-ai -- curl -sf \
        -X POST http://localhost:8000/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":1}' \
        > /dev/null
    echo "✅ AI request successful"
else
    echo "⚠️  No API key configured - skipping AI request test"
fi

echo "🎉 GenOps health check completed successfully!"
EOF

chmod +x genops-health-check.sh
./genops-health-check.sh
```

### Performance Testing

```bash
# Load test script
cat > genops-load-test.sh <<'EOF'
#!/bin/bash

echo "🚀 GenOps Load Test Starting..."

# Port forward
kubectl port-forward -n genops-system service/genops-ai 8080:8000 &
PF_PID=$!

sleep 5

# Run concurrent requests
for i in {1..10}; do
    {
        for j in {1..10}; do
            curl -s -X POST http://localhost:8080/chat/completions \
                -H "Content-Type: application/json" \
                -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"load test '$i'-'$j'"}],"max_tokens":1}'
        done
    } &
done

wait

# Cleanup
kill $PF_PID
echo "🏁 Load test completed"
EOF

chmod +x genops-load-test.sh
```

---

## 📞 Getting Additional Help

### Collect Diagnostic Information

**Run this script to collect comprehensive diagnostic data:**

```bash
cat > collect-genops-diagnostics.sh <<'EOF'
#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DIAG_DIR="genops-diagnostics-$TIMESTAMP"
mkdir -p $DIAG_DIR

echo "🔍 Collecting GenOps diagnostic information..."

# Basic cluster info
kubectl cluster-info > $DIAG_DIR/cluster-info.txt
kubectl version > $DIAG_DIR/kubectl-version.txt
kubectl get nodes -o wide > $DIAG_DIR/nodes.txt

# GenOps resources
kubectl get all -n genops-system > $DIAG_DIR/genops-resources.txt
kubectl get aibudgets,aipolicies -A -o yaml > $DIAG_DIR/governance-resources.yaml
kubectl get secrets -n genops-system > $DIAG_DIR/secrets.txt
kubectl get configmaps -n genops-system > $DIAG_DIR/configmaps.txt

# Logs
kubectl logs -n genops-system -l app.kubernetes.io/name=genops-ai --tail=500 > $DIAG_DIR/genops-logs.txt
kubectl logs -n genops-system -l control-plane=genops-controller --tail=500 > $DIAG_DIR/controller-logs.txt

# Events
kubectl get events -n genops-system --sort-by=.metadata.creationTimestamp > $DIAG_DIR/events.txt

# Describe resources
kubectl describe deployment genops-ai -n genops-system > $DIAG_DIR/deployment-describe.txt
kubectl describe pods -n genops-system -l app.kubernetes.io/name=genops-ai > $DIAG_DIR/pods-describe.txt

# Network
kubectl get services,endpoints -n genops-system > $DIAG_DIR/network.txt
kubectl get networkpolicies -n genops-system > $DIAG_DIR/network-policies.txt

# RBAC
kubectl get serviceaccounts,roles,rolebindings -n genops-system > $DIAG_DIR/rbac.txt

# Helm
helm list -n genops-system > $DIAG_DIR/helm-releases.txt
helm get values genops -n genops-system > $DIAG_DIR/helm-values.txt 2>/dev/null || true

tar -czf genops-diagnostics-$TIMESTAMP.tar.gz $DIAG_DIR/
echo "📦 Diagnostic data collected: genops-diagnostics-$TIMESTAMP.tar.gz"
echo "Please attach this file when requesting support."
EOF

chmod +x collect-genops-diagnostics.sh
./collect-genops-diagnostics.sh
```

### Support Channels

- **GitHub Issues**: [Report bugs with diagnostic data](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community Discord**: [Get help from the community](https://discord.gg/genops-ai)  
- **Documentation**: [Search the knowledge base](https://docs.genops.ai)
- **Enterprise Support**: [Priority support for enterprise customers](mailto:support@genops.ai)

### Before Contacting Support

Please include:
1. **Diagnostic data** (from the collection script above)
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Environment details** (cluster type, version, etc.)
5. **Recent changes** to configuration or deployment

---

## 📚 Related Documentation

- **[Getting Started Guide](kubernetes-getting-started.md)**: Complete setup instructions
- **[Configuration Reference](kubernetes-configuration.md)**: All configuration options
- **[Best Practices](kubernetes-best-practices.md)**: Production deployment patterns
- **[Security Guide](kubernetes-security.md)**: Security hardening instructions
- **[Monitoring Guide](kubernetes-monitoring.md)**: Observability setup

---

**💡 Pro Tip:** Most issues can be resolved by checking logs, verifying configuration, and ensuring proper RBAC permissions. The diagnostic scripts in this guide will help you quickly identify and fix common problems.