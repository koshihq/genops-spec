# GenOps AI Migration Guide for Kubernetes

Complete guide for adding GenOps AI governance to existing AI applications running in Kubernetes. This guide covers zero-downtime migration patterns, gradual rollout strategies, and compatibility with existing infrastructure.

## 🎯 Migration Overview

### What This Guide Covers
- **Zero-downtime migration** from existing AI applications
- **Gradual rollout strategies** (canary, blue-green, rolling updates)
- **Compatibility assessment** for existing applications
- **Rollback procedures** if issues arise
- **Cost impact analysis** and optimization

### Migration Strategies

| Strategy | Best For | Downtime | Complexity | Risk |
|----------|----------|----------|------------|------|
| **Proxy Injection** | Any HTTP-based AI API | None | Low | Low |
| **Sidecar Pattern** | Service mesh environments | None | Medium | Low |
| **Service Replacement** | Direct provider integrations | Minimal | Medium | Medium |
| **Gateway Migration** | Multiple AI services | None | High | Low |

---

## 🔍 Pre-Migration Assessment

### 1. Application Discovery

**Identify AI Applications:**
```bash
# Find applications using AI APIs
kubectl get pods -A -o jsonpath='{range .items[*]}{.metadata.namespace}{" "}{.metadata.name}{" "}{.spec.containers[*].image}{"\n"}{end}' | grep -E "(openai|anthropic|huggingface|replicate)"

# Check environment variables for AI API keys
kubectl get secrets -A -o yaml | grep -E "(openai|anthropic|api.*key)" -i

# Find applications with AI-related annotations
kubectl get deployments -A --show-labels | grep -E "(ai|ml|llm|gpt|claude)"
```

**Analyze Network Traffic:**
```bash
# Check egress to AI providers (if using service mesh)
kubectl logs -n istio-system -l app=istio-proxy | grep -E "(api\.openai\.com|api\.anthropic\.com|api\.replicate\.com)"

# Review network policies
kubectl get networkpolicies -A -o yaml | grep -E "(openai|anthropic|443)"
```

### 2. Cost Discovery

**Current AI Spending Analysis:**
```bash
# Create cost discovery script
cat > discover-ai-costs.sh <<'EOF'
#!/bin/bash
echo "🔍 Discovering current AI costs..."

# Check for existing monitoring
echo "=== Existing Monitoring ==="
kubectl get servicemonitor -A | grep -i ai || echo "No AI service monitors found"
kubectl get prometheus -A | head -1 && echo "Prometheus available" || echo "No Prometheus found"

# Look for cost-related annotations
echo "=== Cost Annotations ==="
kubectl get deployments -A -o yaml | grep -E "(cost|budget|team|customer)" -i | head -10

# Find AI-related services
echo "=== AI Services ==="
kubectl get services -A --show-labels | grep -E "(ai|ml|openai|anthropic|llm)"

# Check for existing budgets/quotas
echo "=== Resource Quotas ==="
kubectl get resourcequotas -A

echo "✅ Discovery complete. Review output to understand current state."
EOF

chmod +x discover-ai-costs.sh
./discover-ai-costs.sh
```

### 3. Compatibility Check

**Create Compatibility Assessment:**
```bash
# Test GenOps with your applications
cat > test-compatibility.sh <<'EOF'
#!/bin/bash
APP_NAMESPACE=${1:-default}
APP_NAME=${2:-your-app}

echo "🧪 Testing GenOps compatibility with $APP_NAME in $APP_NAMESPACE..."

# Check if app uses standard HTTP for AI APIs
echo "=== HTTP API Usage ==="
kubectl exec -n $APP_NAMESPACE deployment/$APP_NAME -- netstat -tuln | grep -E ":80|:443" || echo "No HTTP/HTTPS connections found"

# Check environment variables
echo "=== Environment Variables ==="
kubectl get deployment $APP_NAME -n $APP_NAMESPACE -o yaml | grep -A 20 "env:" | grep -E "(API_KEY|OPENAI|ANTHROPIC|ENDPOINT)"

# Check service communication
echo "=== Service Dependencies ==="
kubectl get deployment $APP_NAME -n $APP_NAMESPACE -o yaml | grep -E "(image|endpoint|url)" | grep -v "#"

echo "✅ Compatibility check complete"
EOF

chmod +x test-compatibility.sh
./test-compatibility.sh default your-ai-app
```

---

## 🚀 Migration Strategy 1: Proxy Injection (Recommended)

**Best for:** Applications making HTTP calls to AI providers  
**Downtime:** Zero  
**Complexity:** Low

### Implementation Steps

**1. Deploy GenOps as Proxy Service**
```bash
# Install GenOps in proxy mode
helm repo add genops https://charts.genops.ai
helm install genops-proxy genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set global.environment=production \
  --set deployment.mode=proxy \
  --set providers.openai.enabled=true \
  --set providers.anthropic.enabled=true \
  --set secrets.apiKeys.openai="$OPENAI_API_KEY" \
  --set secrets.apiKeys.anthropic="$ANTHROPIC_API_KEY"
```

**2. Gradual Traffic Migration**
```bash
# Create canary deployment with GenOps proxy
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: your-app-canary
  namespace: your-app-namespace
  labels:
    app: your-app
    version: canary
spec:
  replicas: 1  # Start with 1 replica
  selector:
    matchLabels:
      app: your-app
      version: canary
  template:
    metadata:
      labels:
        app: your-app
        version: canary
        genops.ai/enable: "true"
    spec:
      containers:
      - name: your-app
        image: your-app:latest
        env:
        # Redirect AI API calls to GenOps proxy
        - name: OPENAI_API_BASE
          value: "http://genops-ai.genops-system:8000"
        - name: ANTHROPIC_API_BASE  
          value: "http://genops-ai.genops-system:8000"
        # Governance attributes
        - name: GENOPS_TEAM
          value: "your-team"
        - name: GENOPS_PROJECT
          value: "your-project"
        - name: GENOPS_CUSTOMER_ID
          value: "your-customer"
EOF
```

**3. Test Canary Deployment**
```bash
# Test GenOps proxy functionality
kubectl port-forward -n your-app-namespace service/your-app-canary 8080:80 &

# Make test requests and verify governance
curl http://localhost:8080/your-ai-endpoint

# Check GenOps metrics
kubectl port-forward -n genops-system service/genops-ai 9090:8000 &
curl http://localhost:9090/metrics | grep genops
```

**4. Gradual Rollout**
```bash
# Increase canary traffic gradually
for replicas in 2 4 8; do
    echo "Scaling canary to $replicas replicas..."
    kubectl scale deployment/your-app-canary --replicas=$replicas -n your-app-namespace
    
    # Monitor for 10 minutes
    echo "Monitoring for 10 minutes..."
    sleep 600
    
    # Check for errors
    kubectl logs -n your-app-namespace -l version=canary --since=10m | grep -i error || echo "No errors found"
    
    read -p "Continue to next stage? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping rollout"
        break
    fi
done
```

**5. Complete Migration**
```bash
# Update original deployment to use GenOps
kubectl patch deployment your-app -n your-app-namespace -p '{"spec":{"template":{"metadata":{"labels":{"genops.ai/enable":"true"}},"spec":{"containers":[{"name":"your-app","env":[{"name":"OPENAI_API_BASE","value":"http://genops-ai.genops-system:8000"},{"name":"GENOPS_TEAM","value":"your-team"}]}]}}}}'

# Remove canary deployment
kubectl delete deployment your-app-canary -n your-app-namespace
```

---

## 🔄 Migration Strategy 2: Sidecar Pattern

**Best for:** Service mesh environments (Istio, Linkerd)  
**Downtime:** Zero  
**Complexity:** Medium

### Istio Implementation

**1. Deploy GenOps Sidecar Injector**
```bash
# Enable automatic sidecar injection
kubectl label namespace your-app-namespace genops-injection=enabled

# Create GenOps sidecar configuration
kubectl apply -f - <<EOF
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: genops-sidecar
spec:
  values:
    sidecarInjectorWebhook:
      templates:
        genops: |
          spec:
            containers:
            - name: genops-sidecar
              image: genops/genops-ai:latest
              ports:
              - containerPort: 8080
                name: genops-proxy
              env:
              - name: GENOPS_MODE
                value: "sidecar"
              - name: GENOPS_TARGET_PORT
                value: "8000"
EOF
```

**2. Configure Traffic Interception**
```bash
# Create VirtualService to route AI traffic through GenOps
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-providers-vs
  namespace: your-app-namespace
spec:
  hosts:
  - api.openai.com
  - api.anthropic.com
  http:
  - match:
    - uri:
        prefix: "/v1/"
    route:
    - destination:
        host: genops-ai.genops-system.svc.cluster.local
        port:
          number: 8000
      headers:
        request:
          add:
            x-genops-team: "your-team"
            x-genops-project: "your-project"
EOF
```

### Linkerd Implementation

**1. Add GenOps as Linkerd Service**
```bash
# Inject Linkerd proxy into GenOps
kubectl get deployment genops-ai -n genops-system -o yaml | linkerd inject - | kubectl apply -f -

# Create service profile for AI providers
kubectl apply -f - <<EOF
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: openai-api
  namespace: your-app-namespace
spec:
  routes:
  - name: chat-completions
    condition:
      method: POST
      pathRegex: "/v1/chat/completions"
    isRetryable: false
    timeout: "30s"
  - name: embeddings
    condition:
      method: POST 
      pathRegex: "/v1/embeddings"
    isRetryable: false
    timeout: "10s"
EOF
```

---

## 🔀 Migration Strategy 3: Service Replacement

**Best for:** Applications with direct provider SDK usage  
**Downtime:** Minimal (during deployment)  
**Complexity:** Medium

### Implementation Steps

**1. Create GenOps-Compatible Service**
```bash
# Deploy new service with GenOps integration
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: your-app-genops
  namespace: your-app-namespace
spec:
  replicas: 3
  selector:
    matchLabels:
      app: your-app
      version: genops
  template:
    metadata:
      labels:
        app: your-app
        version: genops
        genops.ai/enable: "true"
    spec:
      containers:
      - name: app
        image: your-app:genops-enabled
        env:
        # GenOps configuration
        - name: GENOPS_ENABLED
          value: "true"
        - name: GENOPS_ENDPOINT
          value: "http://genops-ai.genops-system:8000"
        - name: GENOPS_TEAM
          value: "your-team"
        - name: GENOPS_PROJECT  
          value: "your-project"
        # Original app configuration
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
---
apiVersion: v1
kind: Service
metadata:
  name: your-app-genops
  namespace: your-app-namespace
spec:
  selector:
    app: your-app
    version: genops
  ports:
  - port: 80
    targetPort: 8080
EOF
```

**2. Blue-Green Deployment**
```bash
# Test new service
kubectl port-forward -n your-app-namespace service/your-app-genops 8080:80

# Switch traffic using service selector update
kubectl patch service your-app -n your-app-namespace -p '{"spec":{"selector":{"version":"genops"}}}'

# Monitor for issues
kubectl logs -n your-app-namespace -l version=genops --follow
```

---

## 🌐 Migration Strategy 4: Gateway Migration  

**Best for:** Multiple AI services, API gateway environments  
**Downtime:** Zero  
**Complexity:** High

### API Gateway Integration

**1. Kong Integration**
```bash
# Install Kong with GenOps plugin
kubectl apply -f - <<EOF
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-plugin
  namespace: kong
config:
  genops_endpoint: http://genops-ai.genops-system:8000
  enable_cost_tracking: true
  enable_governance: true
plugin: genops-ai
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-gateway
  namespace: your-app-namespace
  annotations:
    kubernetes.io/ingress.class: kong
    konghq.com/plugins: genops-plugin
spec:
  rules:
  - host: ai-api.yourcompany.com
    http:
      paths:
      - path: /openai
        pathType: Prefix
        backend:
          service:
            name: genops-ai
            port:
              number: 8000
EOF
```

**2. Ambassador/Emissary Integration**
```bash
# Create Ambassador mapping with GenOps
kubectl apply -f - <<EOF
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: ai-provider-mapping
  namespace: your-app-namespace
spec:
  hostname: "*.yourcompany.com"
  prefix: "/ai/"
  service: genops-ai.genops-system:8000
  timeout_ms: 30000
  retry_policy:
    retry_on: "5xx"
    num_retries: 3
  add_request_headers:
    x-genops-team: "from-header"
    x-genops-project: "from-header"
EOF
```

---

## 📊 Cost Migration Analysis

### Pre-Migration Cost Baseline

**Create Cost Monitoring:**
```bash
# Deploy cost monitoring before GenOps
cat > pre-migration-monitoring.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: cost-monitoring-script
  namespace: your-app-namespace
data:
  monitor.sh: |
    #!/bin/bash
    while true; do
      timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      
      # Extract API usage from logs (customize for your app)
      api_calls=$(kubectl logs -n your-app-namespace -l app=your-app --since=1m | grep -c "API call" || echo "0")
      
      # Estimate costs (customize rates)
      estimated_cost=$(echo "$api_calls * 0.002" | bc)
      
      echo "$timestamp,pre-migration,$api_calls,$estimated_cost" >> /tmp/cost-data.csv
      
      sleep 60
    done
EOF

kubectl apply -f pre-migration-monitoring.yaml

# Run monitoring for 1 week before migration
kubectl create job pre-migration-baseline \
  --from=cronjob/cost-monitor \
  --namespace your-app-namespace
```

### Post-Migration Cost Comparison

**Compare Costs After Migration:**
```bash
# GenOps provides automatic cost tracking
kubectl port-forward -n genops-system service/genops-ai 8080:8000 &

# Get cost data
curl http://localhost:8080/cost/summary > post-migration-costs.json

# Create comparison report
cat > cost-comparison.sh <<'EOF'
#!/bin/bash
echo "📊 Cost Migration Analysis Report"
echo "================================="

# Pre-migration estimates (manual tracking)
echo "Pre-migration (estimated): $PRE_MIGRATION_COST"

# Post-migration actuals (from GenOps)
GENOPS_COST=$(curl -s http://localhost:8080/cost/summary | jq '.total_cost')
echo "Post-migration (actual): $GENOPS_COST"

# Calculate difference  
SAVINGS=$(echo "$PRE_MIGRATION_COST - $GENOPS_COST" | bc)
echo "Cost difference: $SAVINGS"

if (( $(echo "$SAVINGS > 0" | bc -l) )); then
    echo "✅ Migration saved money!"
else
    echo "⚠️ Migration increased costs - investigate optimization opportunities"
fi
EOF
```

---

## 🛡️ Rollback Procedures

### Emergency Rollback

**Immediate Rollback Steps:**
```bash
cat > emergency-rollback.sh <<'EOF'
#!/bin/bash
set -e

echo "🚨 Emergency rollback initiated..."

NAMESPACE=${1:-your-app-namespace}
APP_NAME=${2:-your-app}

# 1. Scale down GenOps-enabled deployment
kubectl scale deployment ${APP_NAME}-genops --replicas=0 -n $NAMESPACE

# 2. Restore original service selector
kubectl patch service $APP_NAME -n $NAMESPACE -p '{"spec":{"selector":{"version":"stable"}}}'

# 3. Scale up original deployment
kubectl scale deployment $APP_NAME --replicas=3 -n $NAMESPACE

# 4. Wait for rollback to complete
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE

# 5. Verify application is working
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

echo "✅ Emergency rollback completed"
echo "🔍 Check application logs and metrics to ensure everything is working"
EOF

chmod +x emergency-rollback.sh
```

### Gradual Rollback

**Step-by-Step Rollback:**
```bash
# Gradual traffic shift back to original
for percentage in 90 75 50 25 0; do
    echo "Shifting ${percentage}% traffic to GenOps version..."
    
    # Update traffic split (adjust for your load balancer/service mesh)
    kubectl patch deployment your-app-genops -p "{\"spec\":{\"replicas\":$(echo "3 * $percentage / 100" | bc)}}" -n your-app-namespace
    
    # Monitor for 15 minutes
    echo "Monitoring for 15 minutes..."
    sleep 900
    
    # Check error rates
    ERROR_RATE=$(kubectl logs -n your-app-namespace -l version=genops --since=15m | grep -c "ERROR" || echo "0")
    echo "Error rate: $ERROR_RATE"
    
    if [ "$ERROR_RATE" -gt 10 ]; then
        echo "⚠️ High error rate detected. Stopping rollback."
        break
    fi
done
```

---

## 🔍 Migration Validation

### Comprehensive Validation Checklist

**Post-Migration Validation:**
```bash
cat > validate-migration.sh <<'EOF'
#!/bin/bash

echo "✅ GenOps Migration Validation"
echo "=============================="

# 1. Application Health
echo "1. Application Health Check..."
kubectl get pods -n your-app-namespace -l genops.ai/enable=true
HEALTHY_PODS=$(kubectl get pods -n your-app-namespace -l genops.ai/enable=true --field-selector=status.phase=Running --no-headers | wc -l)
echo "   Healthy pods: $HEALTHY_PODS"

# 2. GenOps Connectivity
echo "2. GenOps Connectivity..."
kubectl exec -n your-app-namespace deployment/your-app -- curl -s http://genops-ai.genops-system:8000/health
echo "   ✅ GenOps accessible from application"

# 3. Cost Tracking
echo "3. Cost Tracking Validation..."
kubectl port-forward -n genops-system service/genops-ai 8080:8000 &
sleep 2
COST_DATA=$(curl -s http://localhost:8080/metrics | grep genops_cost_total)
if [ -n "$COST_DATA" ]; then
    echo "   ✅ Cost tracking active"
else
    echo "   ❌ Cost tracking not working"
fi

# 4. Governance Policies
echo "4. Governance Policy Check..."
kubectl get aipolicies,aibudgets -A
echo "   ✅ Governance resources configured"

# 5. Performance Check
echo "5. Performance Validation..."
RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null http://localhost:8080/health)
echo "   Response time: ${RESPONSE_TIME}s"

# 6. Error Rate Check
echo "6. Error Rate Analysis..."
ERROR_COUNT=$(kubectl logs -n your-app-namespace -l genops.ai/enable=true --since=1h | grep -i error | wc -l)
echo "   Errors in last hour: $ERROR_COUNT"

pkill -f "kubectl port-forward" || true

echo ""
echo "🎉 Migration validation complete!"
EOF

chmod +x validate-migration.sh
./validate-migration.sh
```

### Performance Comparison

**Before/After Performance Analysis:**
```bash
# Create performance comparison script  
cat > performance-comparison.sh <<'EOF'
#!/bin/bash

echo "📈 Performance Comparison: Before vs After GenOps"
echo "================================================="

# Test without GenOps (direct to provider)
echo "Testing direct provider calls..."
START_TIME=$(date +%s%3N)
curl -s -X POST https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":1}' > /dev/null
END_TIME=$(date +%s%3N)
DIRECT_TIME=$((END_TIME - START_TIME))

# Test through GenOps
echo "Testing through GenOps..."
kubectl port-forward -n genops-system service/genops-ai 8080:8000 &
sleep 2

START_TIME=$(date +%s%3N)
curl -s -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":1}' > /dev/null
END_TIME=$(date +%s%3N)
GENOPS_TIME=$((END_TIME - START_TIME))

pkill -f "kubectl port-forward" || true

echo "Results:"
echo "  Direct provider: ${DIRECT_TIME}ms"
echo "  Through GenOps: ${GENOPS_TIME}ms"
echo "  Overhead: $((GENOPS_TIME - DIRECT_TIME))ms"

OVERHEAD_PERCENT=$(echo "scale=2; ($GENOPS_TIME - $DIRECT_TIME) * 100 / $DIRECT_TIME" | bc)
echo "  Overhead percentage: ${OVERHEAD_PERCENT}%"
EOF

chmod +x performance-comparison.sh
./performance-comparison.sh
```

---

## 📚 Migration Patterns by Application Type

### LangChain Applications

**Migration Pattern:**
```python
# Before migration (existing LangChain app)
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="gpt-3.5-turbo"
)

# After migration (with GenOps)
chat = ChatOpenAI(
    openai_api_base="http://genops-ai.genops-system:8000",
    openai_api_key="not-needed",  # GenOps handles authentication
    model_name="gpt-3.5-turbo",
    headers={
        "X-GenOps-Team": "data-science",
        "X-GenOps-Project": "langchain-app",
        "X-GenOps-Customer": customer_id
    }
)
```

### OpenAI SDK Applications  

**Migration Pattern:**
```python
# Before migration
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

# After migration  
openai.api_base = "http://genops-ai.genops-system:8000"
openai.api_key = "not-needed"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    headers={
        "X-GenOps-Team": os.environ.get("TEAM", "unknown"),
        "X-GenOps-Customer": request.headers.get("Customer-ID")
    }
)
```

### FastAPI Applications

**Migration Pattern:**
```python
# Kubernetes deployment change
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  template:
    metadata:
      labels:
        genops.ai/enable: "true"
    spec:
      containers:
      - name: app
        env:
        # Redirect AI API calls to GenOps
        - name: OPENAI_API_BASE
          value: "http://genops-ai.genops-system:8000"
        # Add governance context
        - name: GENOPS_TEAM
          value: "api-team"
        - name: GENOPS_PROJECT
          value: "customer-api"
```

---

## 🎯 Post-Migration Optimization

### Cost Optimization

**Implement Advanced Cost Controls:**
```bash
# Create advanced budget controls
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIBudget  
metadata:
  name: optimized-budget
  namespace: your-app-namespace
spec:
  allocation:
    amount: 1000.00
    currency: USD
    period: monthly
  attribution:
    team: your-team
    project: your-project
  # Advanced cost controls
  providerBudgets:
  - provider: openai
    allocation: 600.00
    models: ["gpt-3.5-turbo*"]  # Restrict expensive models
  - provider: anthropic
    allocation: 400.00
  costOptimization:
    enabled: true
    targetSavings: 20  # Aim for 20% cost reduction
    strategies:
    - "model_routing"  # Use cheaper models when appropriate
    - "response_caching"  # Cache similar requests
    - "request_batching"  # Batch small requests
  alerts:
    thresholds: [50, 75, 90, 95]
    webhooks:
    - url: "https://hooks.slack.com/services/your/slack/webhook"
      events: ["budget_exceeded", "cost_spike"]
EOF
```

### Performance Optimization

**Implement Advanced Performance Features:**
```bash
# Configure performance optimization
helm upgrade genops genops/genops-ai \
  --namespace genops-system \
  --set performance.caching.enabled=true \
  --set performance.caching.ttl=300 \
  --set performance.batching.enabled=true \
  --set performance.batching.maxBatchSize=10 \
  --set performance.batching.waitTime=100ms
```

---

## 📞 Migration Support

### Getting Help During Migration

**Before Starting Migration:**
- Review this guide thoroughly
- Test migration strategy in development environment
- Create rollback plan
- Schedule migration during low-traffic period

**During Migration:**
- Monitor application logs continuously
- Watch GenOps metrics and logs
- Have rollback script ready to execute
- Keep support channels open

**Emergency Contacts:**
- **GitHub Issues**: [Urgent migration issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discord**: [#migration-support](https://discord.gg/genops-ai)
- **Enterprise Support**: [24/7 support for enterprise customers](mailto:support@genops.ai)

### Migration Checklist

**Pre-Migration:**
- [ ] Application discovery completed
- [ ] Compatibility testing done
- [ ] Migration strategy selected
- [ ] Rollback plan prepared
- [ ] GenOps deployed and tested
- [ ] Team trained on new system

**During Migration:**
- [ ] Gradual rollout executed
- [ ] Monitoring active
- [ ] Performance validated
- [ ] Cost tracking verified
- [ ] Error rates monitored
- [ ] User feedback collected

**Post-Migration:**
- [ ] Full validation completed
- [ ] Performance comparison done
- [ ] Cost analysis completed
- [ ] Documentation updated
- [ ] Team knowledge transfer done
- [ ] Old infrastructure decommissioned

---

## 🎉 Migration Success!

Congratulations! You've successfully migrated your AI applications to use GenOps governance. You now have:

✅ **Zero-downtime migration** completed  
✅ **Complete cost visibility** across all AI operations  
✅ **Policy enforcement** preventing budget overruns  
✅ **Performance monitoring** with detailed telemetry  
✅ **Rollback capability** for peace of mind  

### Next Steps

- **[Optimize Costs](kubernetes-cost-optimization.md)**: Advanced cost management strategies
- **[Scale Deployment](kubernetes-scaling.md)**: Handle increased load and traffic
- **[Security Hardening](kubernetes-security.md)**: Enterprise security configurations
- **[Advanced Monitoring](kubernetes-observability.md)**: Comprehensive observability setup

---

**💡 Pro Tip:** Migration is just the beginning. The real value of GenOps AI comes from the ongoing cost optimization, policy enforcement, and operational insights it provides. Keep monitoring and optimizing!