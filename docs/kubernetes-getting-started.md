# Getting Started with GenOps AI on Kubernetes

The complete guide to deploying AI governance in your Kubernetes clusters. This guide takes you from zero to production-ready GenOps AI deployment with comprehensive cost tracking, policy enforcement, and observability.

## 🎯 What You'll Achieve

By the end of this guide, you'll have:

- ✅ **Zero-code AI governance** for existing applications
- ✅ **Real-time cost tracking** with team/customer attribution  
- ✅ **Multi-provider support** (OpenAI, Anthropic, OpenRouter, etc.)
- ✅ **Production-ready deployment** with security and observability
- ✅ **Policy enforcement** with budgets and rate limiting
- ✅ **Complete monitoring** integrated with your existing observability stack

## 🗺️ Learning Path Overview

This guide follows our proven learning progression:

| Phase | Time | Focus | Outcome |
|-------|------|-------|---------|
| **Phase 1** | 5 minutes | Quick wins | Zero-code instrumentation working |
| **Phase 2** | 30 minutes | Hands-on control | Cost tracking and governance |
| **Phase 3** | 2 hours | Production mastery | Enterprise-ready deployment |

Choose your path based on your immediate needs and available time.

---

## 🚀 Phase 1: Quick Wins (5 minutes)

### Prerequisites Check

**Required:**
- Kubernetes cluster (any version 1.20+)
- `kubectl` configured and working
- Python 3.8+ (for validation)

**Validation:**
```bash
# Quick cluster check
kubectl cluster-info
kubectl get nodes

# If you don't have a cluster, jump to "Local Development Setup" below
```

### Option A: I Have a Kubernetes Cluster

**1. Install GenOps AI**
```bash
# Add Helm repository
helm repo add genops https://charts.genops.ai && helm repo update

# Install with minimal configuration
helm install genops genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set global.environment=quickstart
```

**2. Verify Installation**
```bash
# Check GenOps is running
kubectl get pods -n genops-system

# Should show genops-ai pod in Running state
```

**3. Test Auto-Instrumentation**
```bash
# Port-forward to access GenOps
kubectl port-forward -n genops-system service/genops-ai 8080:8000 &

# Test health endpoint
curl http://localhost:8080/health
# Should return: {"status": "healthy", "kubernetes": true}
```

### Option B: I Need a Local Cluster

**Quick Local Setup with kind:**
```bash
# Install kind (if not already installed)
# macOS
brew install kind
# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 && chmod +x ./kind && sudo mv ./kind /usr/local/bin/

# Create cluster
kind create cluster --name genops-demo

# Install GenOps AI
helm repo add genops https://charts.genops.ai && helm repo update
helm install genops genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set global.environment=development
```

### ✅ Phase 1 Success Check

You should now have:
- GenOps AI running in your cluster
- Health endpoint responding
- Zero-code instrumentation enabled

**Next:** Jump to Phase 2 for cost tracking, or continue reading for more details.

---

## ⚙️ Phase 2: Hands-On Control (30 minutes)

Now we'll add cost tracking, governance policies, and see real AI operations with attribution.

### Add AI Provider Integration

**Option A: OpenAI (Recommended for testing)**
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Update GenOps with OpenAI integration
helm upgrade genops genops/genops-ai \
  --namespace genops-system \
  --set providers.openai.enabled=true \
  --set secrets.apiKeys.openai="$OPENAI_API_KEY"
```

**Option B: Multi-Provider Setup**
```bash
# Set multiple API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Enable multiple providers
helm upgrade genops genops/genops-ai \
  --namespace genops-system \
  --set providers.openai.enabled=true \
  --set providers.anthropic.enabled=true \
  --set secrets.apiKeys.openai="$OPENAI_API_KEY" \
  --set secrets.apiKeys.anthropic="$ANTHROPIC_API_KEY"
```

### Set Up Cost Tracking and Governance

**1. Create a Budget**
```bash
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: team-engineering-daily
  namespace: genops-system
spec:
  allocation:
    amount: 50.00
    currency: USD
    period: daily
  attribution:
    team: engineering
    project: ai-experimentation
  alerts:
    thresholds: [50, 75, 90]
  selector:
    matchLabels:
      team: engineering
EOF
```

**2. Create Governance Policies**
```bash
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: engineering-policy
  namespace: genops-system
spec:
  costLimits:
    enabled: true
    daily: 50.00
    enforcement: warn
  rateLimits:
    enabled: true
    requestsPerMinute: 100
    enforcement: throttle
  contentSafety:
    enabled: true
    minimumSafetyScore: 0.8
  selector:
    matchLabels:
      team: engineering
EOF
```

### Test Real AI Operations

**1. Deploy a Test Application**
```bash
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-test-app
  namespace: default
  labels:
    app: ai-test-app
    team: engineering
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-test-app
  template:
    metadata:
      labels:
        app: ai-test-app
        team: engineering
        genops.ai/enable: "true"
    spec:
      containers:
      - name: ai-test
        image: python:3.9-slim
        command: ["sleep", "3600"]
        env:
        - name: GENOPS_TEAM
          value: "engineering"
        - name: GENOPS_PROJECT
          value: "ai-experimentation"
EOF
```

**2. Make Tracked AI Requests**
```bash
# Port-forward GenOps service
kubectl port-forward -n genops-system service/genops-ai 8080:8000 &

# Make an AI request through GenOps (automatically tracked)
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello! This is a test of GenOps governance."}],
    "max_tokens": 50,
    "team": "engineering",
    "project": "ai-experimentation",
    "customer_id": "demo-customer"
  }'
```

**3. View Cost and Usage Data**
```bash
# Check metrics endpoint
curl http://localhost:8080/metrics | grep genops

# View budget status
kubectl get aibudget team-engineering-daily -o yaml

# Check for any policy violations
kubectl get events --field-selector reason=PolicyViolation
```

### ✅ Phase 2 Success Check

You should now have:
- AI requests being tracked with cost attribution
- Budget and policy enforcement active
- Real-time metrics being collected
- Team and project attribution working

---

## 🏢 Phase 3: Production Mastery (2 hours)

Time to build a production-ready deployment with enterprise features.

### Production Architecture Setup

**1. Multi-Namespace Isolation**
```bash
# Create production namespaces
kubectl create namespace genops-production
kubectl create namespace genops-staging
kubectl create namespace genops-monitoring

# Label namespaces for governance
kubectl label namespace genops-production genops.ai/environment=production
kubectl label namespace genops-staging genops.ai/environment=staging
```

**2. Deploy Production Configuration**
```bash
# Production Helm values
cat > values-production.yaml <<EOF
global:
  environment: production

deployment:
  replicaCount: 3
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

security:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop: ["ALL"]

providers:
  openai:
    enabled: true
  anthropic:
    enabled: true
  openrouter:
    enabled: true

governance:
  policies:
    costLimits:
      daily: 1000.00
      enforcement: throttle
    rateLimits:
      requestsPerMinute: 1000
      enforcement: throttle
    contentSafety:
      minimumSafetyScore: 0.85

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  grafana:
    dashboards:
      enabled: true

networkPolicy:
  enabled: true

podDisruptionBudget:
  enabled: true
  minAvailable: 2
EOF

# Deploy production instance
helm install genops-prod genops/genops-ai \
  --namespace genops-production \
  --values values-production.yaml
```

### Enterprise Security Configuration

**1. RBAC Setup**
```bash
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: genops-production
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["genops.ai"]
  resources: ["aipolicies", "aibudgets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: genops-production
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: genops-production
subjects:
- kind: ServiceAccount
  name: genops-ai
  namespace: genops-production
EOF
```

**2. Network Security**
```bash
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genops-network-policy
  namespace: genops-production
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: genops-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS to AI providers
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53  # DNS
    - protocol: UDP
      port: 53  # DNS
EOF
```

### Comprehensive Monitoring Setup

**1. Deploy Monitoring Stack**
```bash
# Add monitoring repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Deploy Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace genops-monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword=admin123
```

**2. Import GenOps Dashboards**
```bash
# Create Grafana dashboard ConfigMap
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-dashboards
  namespace: genops-monitoring
  labels:
    grafana_dashboard: "1"
data:
  genops-overview.json: |
    {
      "dashboard": {
        "id": null,
        "title": "GenOps AI Overview",
        "tags": ["genops", "ai", "kubernetes"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "AI Requests per Second",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(genops_ai_requests_total[5m])",
                "legendFormat": "{{provider}}"
              }
            ]
          },
          {
            "id": 2,
            "title": "Cost by Provider",
            "type": "piechart", 
            "targets": [
              {
                "expr": "sum by (provider) (genops_ai_cost_total_usd)",
                "legendFormat": "{{provider}}"
              }
            ]
          },
          {
            "id": 3,
            "title": "Budget Utilization",
            "type": "gauge",
            "targets": [
              {
                "expr": "genops_budget_utilization_percent",
                "legendFormat": "{{team}}"
              }
            ]
          }
        ]
      }
    }
EOF
```

### Multi-Tenant Production Setup

**1. Team Isolation**
```bash
# Create team-specific budgets
for team in engineering marketing sales; do
  kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: ${team}-monthly-budget
  namespace: genops-production
spec:
  allocation:
    amount: 5000.00
    currency: USD
    period: monthly
  attribution:
    team: ${team}
  providerBudgets:
  - provider: openai
    allocation: 2000.00
  - provider: anthropic
    allocation: 2000.00
  - provider: openrouter
    allocation: 1000.00
  alerts:
    thresholds: [60, 80, 95]
    webhooks:
    - url: "https://hooks.slack.com/services/your/slack/webhook"
  selector:
    matchLabels:
      team: ${team}
EOF
done
```

**2. Environment-Specific Policies**
```bash
# Production policy (strict)
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: production-policy
  namespace: genops-production
spec:
  costLimits:
    enabled: true
    daily: 500.00
    enforcement: throttle
  rateLimits:
    enabled: true
    requestsPerMinute: 500
    enforcement: throttle
  modelGovernance:
    enabled: true
    allowedProviders: ["openai", "anthropic"]
    allowedModels: ["gpt-4*", "gpt-3.5-turbo*", "claude-3*"]
  contentSafety:
    enabled: true
    minimumSafetyScore: 0.90
    enforcement: block
  auditPolicy:
    logLevel: info
    retentionDays: 90
  selector:
    matchLabels:
      environment: production
EOF

# Staging policy (permissive)
kubectl apply -f - <<EOF
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: staging-policy
  namespace: genops-staging
spec:
  costLimits:
    enabled: true
    daily: 100.00
    enforcement: warn
  rateLimits:
    enabled: true
    requestsPerMinute: 1000
    enforcement: warn
  contentSafety:
    enabled: true
    minimumSafetyScore: 0.70
    enforcement: warn
  selector:
    matchLabels:
      environment: staging
EOF
```

### Disaster Recovery Setup

**1. Backup Configuration**
```bash
# Create backup CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: genops-backup
  namespace: genops-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: bitnami/kubectl:latest
            command:
            - /bin/bash
            - -c
            - |
              # Backup GenOps configurations
              kubectl get aibudgets -o yaml > /backup/aibudgets-$(date +%Y%m%d).yaml
              kubectl get aipolicies -o yaml > /backup/aipolicies-$(date +%Y%m%d).yaml
              kubectl get configmaps -l app.kubernetes.io/name=genops-ai -o yaml > /backup/configmaps-$(date +%Y%m%d).yaml
              kubectl get secrets -l app.kubernetes.io/name=genops-ai -o yaml > /backup/secrets-$(date +%Y%m%d).yaml
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          restartPolicy: OnFailure
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: genops-backup-pvc
EOF
```

### ✅ Phase 3 Success Check

You should now have:
- Production-ready GenOps deployment with HA
- Multi-tenant isolation with team-specific budgets
- Comprehensive monitoring and alerting
- Enterprise security with RBAC and network policies
- Disaster recovery and backup procedures
- Complete cost governance across all teams and environments

---

## 🔧 Advanced Configuration

### Custom Provider Integration

```bash
# Example: Adding Azure OpenAI
helm upgrade genops-prod genops/genops-ai \
  --namespace genops-production \
  --reuse-values \
  --set providers.azure.enabled=true \
  --set providers.azure.endpoint="https://your-resource.openai.azure.com/" \
  --set secrets.apiKeys.azure="your-azure-key"
```

### GitOps Integration

```bash
# ArgoCD Application
kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: genops-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://charts.genops.ai
    targetRevision: "1.0.0"
    chart: genops-ai
    helm:
      valueFiles:
      - values-production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: genops-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
EOF
```

### Service Mesh Integration

```bash
# Istio VirtualService
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai
  namespace: genops-production
spec:
  http:
  - match:
    - uri:
        prefix: "/chat/completions"
    route:
    - destination:
        host: genops-ai
        port:
          number: 8000
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 10s
EOF
```

---

## 🚨 Troubleshooting

### Quick Health Checks

```bash
# Check GenOps health
kubectl get pods -n genops-production
kubectl logs -n genops-production -l app.kubernetes.io/name=genops-ai --tail=50

# Check governance resources
kubectl get aibudgets,aipolicies -A

# Verify network connectivity
kubectl exec -n genops-production deployment/genops-ai -- wget -qO- http://localhost:8000/health
```

### Common Issues

**GenOps pods not starting:**
```bash
# Check events
kubectl describe pods -n genops-production -l app.kubernetes.io/name=genops-ai

# Check RBAC permissions
kubectl auth can-i get pods --as=system:serviceaccount:genops-production:genops-ai
```

**No cost data appearing:**
```bash
# Verify API keys
kubectl get secrets -n genops-production genops-secrets -o yaml

# Check provider connectivity
kubectl logs -n genops-production -l app.kubernetes.io/name=genops-ai | grep -i "provider\|api\|error"
```

**Policy violations not working:**
```bash
# Check GenOps controller logs
kubectl logs -n genops-system -l control-plane=genops-controller

# Verify policy selectors
kubectl get aipolicies -o yaml | grep -A 10 selector
```

---

## 🎯 Next Steps

### Expand Your Deployment
- **[Multi-Cloud Setup](kubernetes-multi-cloud.md)**: Deploy across AWS, GCP, Azure
- **[Advanced Monitoring](kubernetes-observability.md)**: Detailed dashboards and alerting
- **[Cost Optimization](kubernetes-cost-optimization.md)**: Advanced cost management strategies

### Integrate with Your Stack
- **[CI/CD Integration](kubernetes-cicd.md)**: GitOps and deployment pipelines
- **[Service Mesh](kubernetes-service-mesh.md)**: Istio, Linkerd integration patterns
- **[API Gateway](kubernetes-api-gateway.md)**: Kong, Ambassador, Ingress patterns

### Enterprise Features
- **[Multi-Tenant Architecture](kubernetes-multi-tenant.md)**: Complete tenant isolation
- **[Security Hardening](kubernetes-security.md)**: Advanced security configurations
- **[Disaster Recovery](kubernetes-dr.md)**: Multi-region and backup strategies

---

## 📚 Additional Resources

- **[Examples](../examples/kubernetes/)**: Hands-on code examples and patterns
- **[Helm Chart Documentation](../charts/genops-ai/)**: Complete configuration reference
- **[Operator Guide](../operators/genops-controller/)**: Custom resource management
- **[API Reference](api-reference.md)**: Complete API documentation
- **[Best Practices](kubernetes-best-practices.md)**: Production deployment patterns

---

## 🆘 Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community Discord**: [Join our community](https://discord.gg/genops-ai)
- **Documentation**: [Complete documentation site](https://docs.genops.ai)
- **Enterprise Support**: [Contact us](mailto:support@genops.ai)

---

**🎉 Congratulations!** You now have production-ready GenOps AI governance running in Kubernetes with comprehensive cost tracking, policy enforcement, and enterprise security features.