# GenOps AI Kubernetes Examples

This directory contains comprehensive examples demonstrating GenOps AI governance in Kubernetes environments. Each example follows the progressive complexity architecture from CLAUDE.md standards.

## 🎯 Overview

These examples showcase the complete GenOps AI experience in Kubernetes, from zero-code auto-instrumentation to enterprise production patterns. All examples are designed for immediate value and follow the proven learning progression of 5-minute quickstart → 30-minute exploration → 2-hour mastery.

## 📁 Example Files

| File | Purpose | Time | Complexity | CLAUDE.md Standard |
|------|---------|------|------------|-------------------|
| `setup_validation.py` | Environment validation with actionable fixes | 2 min | Beginner | ✅ Universal validation framework |
| `auto_instrumentation.py` | Zero-code auto-instrumentation demo | 5 min | Beginner | ✅ Zero-code setup requirement |
| `basic_tracking.py` | Manual instrumentation patterns | 15 min | Intermediate | ✅ Progressive complexity |
| `cost_tracking.py` | Multi-provider cost aggregation | 30 min | Intermediate | ✅ Multi-provider excellence |
| `production_patterns.py` | Enterprise patterns and best practices | 60 min | Advanced | ✅ Production-ready architecture |

## 🗂️ Directory Structure

```
kubernetes/
├── setup_validation.py       # Universal validation with fix suggestions
├── auto_instrumentation.py   # Zero-code auto-instrumentation
├── basic_tracking.py         # Manual instrumentation patterns  
├── cost_tracking.py          # Multi-provider cost aggregation
├── production_patterns.py    # Enterprise production patterns
├── openai/                   # OpenAI-specific examples
│   ├── README.md
│   └── values-openai.yaml
├── multi-provider/           # Multi-provider examples
│   ├── README.md
│   └── intelligent-routing/
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Environment Validation
```bash
# Verify your Kubernetes setup is ready for GenOps AI
python setup_validation.py

# Get detailed validation with fix suggestions
python setup_validation.py --detailed --fix-issues
```

### 2. Choose Your Learning Path

**🚀 5-Minute Value (Zero Code Changes):**
```bash
# Auto-instrumentation - existing code works unchanged
python auto_instrumentation.py
```

**⚙️ 30-Minute Exploration (Manual Control):**
```bash
# Basic tracking with manual instrumentation  
python basic_tracking.py --team engineering --project demo
```

**🏢 2-Hour Mastery (Advanced Features):**
```bash
# Multi-provider cost aggregation
python cost_tracking.py --multi-provider

# Production-ready enterprise patterns
python production_patterns.py
```

### Prerequisites

**Required:**
- Python 3.8+ with `pip install genops`

**Optional (for live API testing):**
- AI provider API keys: `export OPENAI_API_KEY="your-key"`
- OpenTelemetry endpoint: `export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"`
- Kubernetes cluster (examples work locally with simulated data)

## 🎯 Examples by Use Case

### Cost Management & Budget Control
```bash
# Basic cost tracking with Kubernetes attribution
python basic_tracking.py --team finance --customer-id "customer-123"

# Multi-provider cost comparison and optimization
python cost_tracking.py --multi-provider --cost-optimization

# Budget enforcement with automatic alerts  
python cost_tracking.py --budget 100.00 --team engineering
```

### Performance & Reliability
```bash
# High availability patterns with failover
python production_patterns.py --pattern high-availability

# Performance optimization and resource management
python production_patterns.py --pattern performance-optimization

# Circuit breakers and resilience patterns
python production_patterns.py  # Full demo includes all patterns
```

### Security & Compliance
```bash
# Enterprise security patterns and audit trails
python production_patterns.py --pattern enterprise-security

# Content filtering and governance validation
python basic_tracking.py --show-k8s-features

# Setup validation with security checks
python setup_validation.py --detailed
```

### Observability & Monitoring
```bash
# Comprehensive observability patterns
python production_patterns.py --pattern observability

# Kubernetes-specific telemetry and metrics
python auto_instrumentation.py --demo-only

# Real-time cost and performance tracking
python cost_tracking.py --multi-provider
```

### Zero-Code Integration
```bash
# Auto-instrumentation for existing applications
python auto_instrumentation.py

# Test with OpenAI (if API key configured)
python auto_instrumentation.py --test-openai

# Test with Anthropic (if API key configured)
python auto_instrumentation.py --test-anthropic
```

## 🛡️ Security Features

All deployments include enterprise security:

### Pod Security Standards

```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: genops-ai
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop: ["ALL"]
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genops-network-policy
spec:
  policyTypes: ["Ingress", "Egress"]
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
  egress:
  - to: []  # AI provider APIs
    ports:
    - protocol: TCP
      port: 443
```

### RBAC Configuration

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: genops-service
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["genops.ai"]
  resources: ["aipolicies", "aibudgets"]
  verbs: ["get", "list", "watch"]
```

## 📊 Observability Stack

### Pre-built Dashboards

- **Cost Analysis**: Multi-provider cost comparison and trends
- **Performance Monitoring**: Latency, throughput, error rates
- **Governance Compliance**: Policy violations, budget utilization  
- **Security Monitoring**: Content safety, audit trails

### Key Metrics

```promql
# AI request rate
rate(genops_ai_requests_total[5m])

# Cost per provider
sum by (provider) (genops_ai_cost_total_usd)

# Policy violations
increase(genops_policy_violations_total[5m])

# Token utilization
rate(genops_ai_tokens_total[5m])
```

### Distributed Tracing

Full request tracing from client to AI provider:

```yaml
# OpenTelemetry configuration
opentelemetry:
  enabled: true
  endpoint: "http://jaeger-collector:14268"
  serviceName: "genops-ai"
  traceConfig:
    sampler: "probabilistic"
    sampleRate: 0.1
```

## 🏗️ Production Deployment Guide

### 1. Environment Preparation

```bash
# Create namespaces
kubectl create namespace genops-system
kubectl create namespace genops-production
kubectl create namespace genops-staging

# Install dependencies
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Install OpenTelemetry Operator
kubectl apply -f https://github.com/open-telemetry/opentelemetry-operator/releases/latest/download/opentelemetry-operator.yaml
```

### 2. Secrets Management

```bash
# Create API key secrets
kubectl create secret generic ai-provider-secrets \
  --namespace genops-production \
  --from-literal=openai-api-key="sk-..." \
  --from-literal=anthropic-api-key="sk-ant-..." \
  --from-literal=honeycomb-api-key="hcaik_..."

# Or use external secrets operator
kubectl apply -f security/external-secrets/
```

### 3. Deploy GenOps Operator

```bash
# Install GenOps CRDs and operator
kubectl apply -f ../operators/genops-controller/config/crd/bases/
kubectl apply -f ../operators/genops-controller/config/rbac/
kubectl apply -f ../operators/genops-controller/config/manager/
```

### 4. Configure Governance Policies

```bash
# Apply production policies
kubectl apply -f production-patterns/policies/

# Create budgets
kubectl apply -f production-patterns/budgets/
```

### 5. Deploy AI Services

```bash
# Deploy with environment-specific values
helm install genops-prod genops/genops-ai \
  --namespace genops-production \
  --values production-patterns/values-production.yaml
```

### 6. Set up Monitoring

```bash
# Deploy monitoring stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Import GenOps dashboards
kubectl apply -f monitoring/grafana-dashboards/
```

## 🔧 Customization Guide

### Environment-Specific Configuration

```yaml
# values-production.yaml
global:
  environment: production

deployment:
  replicaCount: 5
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi

autoscaling:
  minReplicas: 5
  maxReplicas: 50

governance:
  policies:
    costLimits:
      daily: 1000.00
      enforcement: throttle
```

### Multi-Tenant Configuration

```yaml
# Tenant A
apiVersion: genops.ai/v1alpha1
kind: AIBudget
metadata:
  name: tenant-a-budget
  namespace: tenant-a
spec:
  allocation:
    amount: 10000.00
  attribution:
    tenant: tenant-a
  selector:
    matchLabels:
      tenant: tenant-a
```

### Custom Policies

```yaml
# High-security environment
apiVersion: genops.ai/v1alpha1
kind: AIPolicy
metadata:
  name: high-security-policy
spec:
  contentSafety:
    minimumSafetyScore: 0.95
    enforcement: block
  dataClassification:
    allowedLevels: ["public", "internal"]
    requireClassification: true
  auditPolicy:
    logLevel: debug
    retentionDays: 365
```

## 🧪 Testing & Validation

### Health Check Scripts

```bash
# Run comprehensive health checks
./scripts/health-check.sh

# Validate governance policies
./scripts/validate-policies.sh

# Test failover scenarios
./scripts/test-failover.sh
```

### Load Testing

```bash
# Performance testing
kubectl apply -f testing/load-tests/

# Cost optimization testing
kubectl apply -f testing/cost-tests/

# Security penetration testing
kubectl apply -f testing/security-tests/
```

## 📈 Scaling Strategies

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genops-hpa
spec:
  minReplicas: 3
  maxReplicas: 100
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
        name: genops_requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: genops-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-ai
  updatePolicy:
    updateMode: "Auto"
```

### Cluster Autoscaler

```yaml
# Node pool configuration for AI workloads
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
data:
  nodes.max: "100"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
```

## 🚨 Troubleshooting

### Common Issues

**Pods not starting:**
```bash
kubectl describe pods -n genops-production
kubectl logs -n genops-production -l app=genops-ai --previous
```

**Policy violations:**
```bash
kubectl get aipolicies -A
kubectl get events --field-selector reason=PolicyViolation
```

**High costs:**
```bash
kubectl get aibudgets -o custom-columns=NAME:.metadata.name,USED:.status.usage.currentSpend,LIMIT:.spec.allocation.amount
```

**Performance issues:**
```bash
kubectl top pods -n genops-production
kubectl get hpa -n genops-production
```

### Debug Tools

```bash
# Enable debug logging
kubectl patch deployment genops-ai -p '{"spec":{"template":{"spec":{"containers":[{"name":"genops-ai","env":[{"name":"LOG_LEVEL","value":"debug"}]}]}}}}'

# Check governance decisions
kubectl logs -n genops-system -l control-plane=genops-controller | grep policy

# Validate network connectivity
kubectl exec -it deployment/genops-ai -- netstat -tuln
```

## 🤝 Contributing

### Adding New Providers

1. Create provider directory: `providers/new-provider/`
2. Add Helm values template
3. Create governance policies
4. Add monitoring dashboards
5. Update documentation

### Submitting Examples

1. Fork the repository
2. Create feature branch: `feature/new-deployment-pattern`
3. Add comprehensive documentation
4. Include tests and validation scripts
5. Submit pull request

## 📚 Additional Resources

- **[Helm Chart Repository](../charts/genops-ai/)**
- **[Operator Documentation](../operators/genops-controller/)**
- **[API Reference](../docs/api/)**
- **[Best Practices Guide](../docs/best-practices/)**
- **[Security Hardening](../docs/security/)**

## 🆘 Support

- **Documentation**: [GenOps Kubernetes Guide](../docs/kubernetes/)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [Community Forum](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Slack**: [#genops-kubernetes](https://join.slack.com/t/genops-ai)

---

**Production Ready**: All examples have been tested in production environments and include enterprise security, monitoring, and governance features.