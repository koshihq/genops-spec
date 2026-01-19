# Multi-Tenant Kubernetes Architecture for GenOps AI

> **Status:** 📋 Documentation in progress
> **Last Updated:** 2026-01-18

Build secure, isolated multi-tenant AI platforms with comprehensive governance tracking per tenant.

---

## Overview

Multi-tenant architectures enable platform teams to serve multiple customers or internal teams from a shared infrastructure while maintaining:
- **Strong Isolation** between tenants at compute, network, and data layers
- **Per-Tenant Cost Attribution** with accurate usage tracking and billing
- **Governance Boundaries** with tenant-specific policies and budgets
- **Resource Quotas** to prevent noisy neighbor problems
- **Security Segmentation** with RBAC and network policies

GenOps AI provides built-in multi-tenant governance tracking, making it ideal for SaaS platforms and internal AI service platforms.

---

## Quick Reference

### Tenant Isolation Levels

**Namespace Isolation (Soft Multi-Tenancy):**
- Each tenant gets dedicated Kubernetes namespace
- Shared cluster control plane and worker nodes
- Network policies for traffic isolation
- Resource quotas per namespace
- **Best for:** Internal teams, trusted tenants

**Node Pool Isolation (Medium Isolation):**
- Dedicated node pools per tenant or tenant group
- Node taints and tolerations for scheduling
- Separate compute resources
- **Best for:** Different SLAs, compliance requirements

**Cluster Isolation (Hard Multi-Tenancy):**
- Dedicated Kubernetes cluster per tenant
- Complete infrastructure separation
- Maximum security and performance isolation
- **Best for:** Enterprise customers, strict compliance

### GenOps Multi-Tenant Configuration

```python
from genops import track_usage

@track_usage(
    team="platform-team",
    project="saas-inference",
    customer_id="tenant-abc-123",  # Unique tenant identifier
    environment="production",
    budget_limit=500.0,  # Per-tenant budget
    budget_period="monthly"
)
def serve_tenant_request(tenant_id, request):
    # Automatically tracked and attributed to tenant
    response = ai_model.generate(request)
    return response
```

---

## Table of Contents

### Planned Documentation Sections

1. **Multi-Tenant Architecture Patterns**
   - Namespace-based soft multi-tenancy
   - Node pool isolation strategies
   - Cluster-per-tenant architectures
   - Hybrid approaches for different tenant tiers

2. **Resource Isolation**
   - Kubernetes namespaces and RBAC
   - ResourceQuotas and LimitRanges
   - Node affinity and anti-affinity
   - PodDisruptionBudgets for availability

3. **Network Segmentation**
   - NetworkPolicies for tenant isolation
   - Service mesh authorization policies
   - Ingress and egress controls
   - DNS isolation strategies

4. **Cost Attribution and Billing**
   - Per-tenant cost tracking with GenOps
   - Usage-based billing integration
   - Cost allocation for shared resources
   - Chargeback and showback reporting

5. **Security and Compliance**
   - RBAC for tenant administrators
   - Pod Security Standards per tenant
   - Secret management and isolation
   - Audit logging per tenant
   - Compliance frameworks (SOC2, HIPAA, GDPR)

6. **Tenant Onboarding and Management**
   - Automated tenant provisioning
   - Self-service tenant portals
   - Tenant lifecycle management
   - Monitoring and alerting per tenant

7. **Performance and Scalability**
   - Preventing noisy neighbor issues
   - QoS classes and priority
   - Autoscaling strategies per tenant
   - Capacity planning and forecasting

---

## Related Documentation

**Kubernetes Guides:**
- [Kubernetes Getting Started](kubernetes-getting-started.md)
- [Security Hardening](kubernetes-security.md)
- [Cost Optimization](kubernetes-cost-optimization.md)

**Integration Guides:**
- [API Gateway Integration](kubernetes-api-gateway.md)
- [Advanced Observability](kubernetes-observability.md)

---

## Quick Examples

### Example 1: Namespace-Based Tenant Isolation

```yaml
# Tenant namespace with resource quotas
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-abc-123
  labels:
    genops.ai/tenant-id: "abc-123"
    genops.ai/tenant-tier: "premium"

---
# Resource quota for tenant
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-quota
  namespace: tenant-abc-123
spec:
  hard:
    requests.cpu: "10"
    requests.memory: "20Gi"
    requests.nvidia.com/gpu: "2"
    limits.cpu: "20"
    limits.memory: "40Gi"
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"

---
# Limit range for pod defaults
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-limits
  namespace: tenant-abc-123
spec:
  limits:
  - max:
      cpu: "4"
      memory: "8Gi"
    min:
      cpu: "100m"
      memory: "128Mi"
    default:
      cpu: "500m"
      memory: "1Gi"
    defaultRequest:
      cpu: "250m"
      memory: "512Mi"
    type: Container
```

### Example 2: Network Policies for Tenant Isolation

```yaml
# Default deny all ingress traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: tenant-abc-123
spec:
  podSelector: {}
  policyTypes:
  - Ingress

---
# Allow ingress from API gateway only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-gateway
  namespace: tenant-abc-123
spec:
  podSelector:
    matchLabels:
      app: tenant-inference
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    ports:
    - protocol: TCP
      port: 8080

---
# Allow egress to AI providers only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-egress-ai-providers
  namespace: tenant-abc-123
spec:
  podSelector:
    matchLabels:
      app: tenant-inference
  policyTypes:
  - Egress
  egress:
  # DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  # AI provider APIs
  - to:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

### Example 3: Node Pool Isolation with Taints and Tolerations

```yaml
# Node pool for premium tenants
# Apply taint to nodes:
# kubectl taint nodes node-premium-1 tenant-tier=premium:NoSchedule

# Deployment with toleration for premium node pool
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tenant-inference
  namespace: tenant-abc-123
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tenant-inference
      tenant-id: abc-123
  template:
    metadata:
      labels:
        app: tenant-inference
        tenant-id: abc-123
      annotations:
        genops.ai/team: "platform-team"
        genops.ai/customer-id: "abc-123"
    spec:
      # Tolerate premium node pool taint
      tolerations:
      - key: "tenant-tier"
        operator: "Equal"
        value: "premium"
        effect: "NoSchedule"

      # Schedule only on premium nodes
      nodeSelector:
        tenant-tier: "premium"

      # Anti-affinity to spread across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - tenant-inference
              topologyKey: kubernetes.io/hostname

      containers:
      - name: inference
        image: genops-ai-inference:latest
        env:
        - name: GENOPS_CUSTOMER_ID
          value: "abc-123"
        - name: GENOPS_TEAM
          value: "platform-team"
        - name: GENOPS_BUDGET_LIMIT
          value: "500.0"
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
```

### Example 4: RBAC for Tenant Administrators

```yaml
# Role for tenant admin (namespace-scoped)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: tenant-admin
  namespace: tenant-abc-123
rules:
# Manage pods and deployments
- apiGroups: ["", "apps"]
  resources: ["pods", "deployments", "replicasets", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# View logs
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get", "list"]
# Cannot modify resource quotas or network policies
- apiGroups: [""]
  resources: ["resourcequotas", "limitranges"]
  verbs: ["get", "list"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies"]
  verbs: ["get", "list"]

---
# RoleBinding for tenant user
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: tenant-admin-binding
  namespace: tenant-abc-123
subjects:
- kind: User
  name: "admin@tenant-abc.com"
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: tenant-admin
  apiGroup: rbac.authorization.k8s.io
```

### Example 5: Tenant Provisioning Automation

```python
# Automated tenant onboarding script
from kubernetes import client, config

def provision_tenant(tenant_id: str, tier: str, quota_config: dict):
    """Provision a new tenant namespace with governance configuration."""

    config.load_kube_config()
    v1 = client.CoreV1Api()
    rbac_v1 = client.RbacAuthorizationV1Api()

    # Create namespace
    namespace = client.V1Namespace(
        metadata=client.V1ObjectMeta(
            name=f"tenant-{tenant_id}",
            labels={
                "genops.ai/tenant-id": tenant_id,
                "genops.ai/tenant-tier": tier,
                "genops.ai/managed-by": "platform-team"
            }
        )
    )
    v1.create_namespace(namespace)

    # Create resource quota
    quota = client.V1ResourceQuota(
        metadata=client.V1ObjectMeta(name="tenant-quota"),
        spec=client.V1ResourceQuotaSpec(
            hard=quota_config
        )
    )
    v1.create_namespaced_resource_quota(
        namespace=f"tenant-{tenant_id}",
        body=quota
    )

    # Deploy GenOps AI with tenant configuration
    deployment = create_genops_deployment(
        tenant_id=tenant_id,
        tier=tier
    )
    apps_v1 = client.AppsV1Api()
    apps_v1.create_namespaced_deployment(
        namespace=f"tenant-{tenant_id}",
        body=deployment
    )

    print(f"Tenant {tenant_id} provisioned successfully!")
    return f"tenant-{tenant_id}"

# Example usage
provision_tenant(
    tenant_id="abc-123",
    tier="premium",
    quota_config={
        "requests.cpu": "10",
        "requests.memory": "20Gi",
        "requests.nvidia.com/gpu": "2"
    }
)
```

---

## Multi-Tenant Architecture Patterns (Detailed)

### Pattern 1: Namespace-Based Isolation (Soft Multi-Tenancy)

**Architecture:**
- Single Kubernetes cluster shared across tenants
- Each tenant gets one or more dedicated namespaces
- Resource quotas prevent resource monopolization
- Network policies isolate tenant traffic
- RBAC restricts cross-tenant access

**Best For:**
- Internal teams within same organization
- Trusted tenants with similar security requirements
- Cost-sensitive deployments
- Development and staging environments

**Pros:**
- **Cost-Efficient**: Maximum resource utilization through sharing
- **Operational Simplicity**: Single cluster to manage
- **Fast Provisioning**: New tenants onboarded in seconds
- **Resource Sharing**: Efficient use of node capacity

**Cons:**
- **Limited Isolation**: Noisy neighbor problems possible
- **Shared Control Plane**: Control plane issues affect all tenants
- **Security Risk**: Kernel-level vulnerabilities affect all tenants
- **Performance**: Resource contention during high load

**Implementation Example:**
```yaml
# Comprehensive namespace-based tenant setup
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-acme-corp
  labels:
    genops.ai/tenant-id: "acme-corp"
    genops.ai/tenant-tier: "enterprise"
    genops.ai/billing-code: "BC-12345"
    genops.ai/cost-center: "engineering"

---
# Hierarchical resource quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-compute-quota
  namespace: tenant-acme-corp
spec:
  hard:
    # Compute resources
    requests.cpu: "20"
    requests.memory: "40Gi"
    requests.nvidia.com/gpu: "4"
    limits.cpu: "40"
    limits.memory: "80Gi"

    # Storage
    requests.storage: "500Gi"
    persistentvolumeclaims: "20"

    # Network
    services.loadbalancers: "3"
    services.nodeports: "0"  # Disallow NodePort

    # Objects
    pods: "100"
    configmaps: "50"
    secrets: "50"

---
# LimitRange for pod defaults
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-limits
  namespace: tenant-acme-corp
spec:
  limits:
  - max:
      cpu: "8"
      memory: "16Gi"
    min:
      cpu: "100m"
      memory: "128Mi"
    default:
      cpu: "1"
      memory: "2Gi"
    defaultRequest:
      cpu: "500m"
      memory: "1Gi"
    type: Container

  - max:
      storage: "100Gi"
    min:
      storage: "1Gi"
    type: PersistentVolumeClaim
```

### Pattern 2: Node Pool Isolation (Medium Isolation)

**Architecture:**
- Dedicated node pools per tenant or tenant tier
- Taints and tolerations enforce scheduling boundaries
- Separate autoscaling configurations per pool
- Shared control plane, isolated compute

**Best For:**
- Multi-tier SaaS platforms (free/standard/premium)
- Compliance requirements (PCI-DSS, HIPAA)
- Performance-sensitive workloads
- Mixed workload types (CPU vs GPU)

**Implementation:**
```yaml
# Node pool for premium tenants
# Apply taint: kubectl taint nodes node-pool-premium tenant-tier=premium:NoSchedule

# Deployment with node affinity and tolerations
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-inference
  namespace: tenant-acme-corp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-ai-inference
      tenant-id: acme-corp
  template:
    metadata:
      labels:
        app: genops-ai-inference
        tenant-id: acme-corp
      annotations:
        genops.ai/team: "ml-platform"
        genops.ai/customer-id: "acme-corp"
        genops.ai/budget-limit: "5000"
    spec:
      # Tolerate premium node taint
      tolerations:
      - key: "tenant-tier"
        operator: "Equal"
        value: "premium"
        effect: "NoSchedule"

      # Prefer premium nodes
      nodeSelector:
        tenant-tier: "premium"
        node.kubernetes.io/instance-type: "c5.2xlarge"

      # Anti-affinity to spread across nodes
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - genops-ai-inference
            topologyKey: kubernetes.io/hostname

        # Prefer different availability zones
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - genops-ai-inference
              topologyKey: topology.kubernetes.io/zone

      containers:
      - name: inference
        image: genops-ai-inference:latest
        env:
        - name: GENOPS_CUSTOMER_ID
          value: "acme-corp"
        - name: GENOPS_TEAM
          value: "ml-platform"
        - name: GENOPS_BUDGET_LIMIT
          value: "5000.0"
        resources:
          requests:
            cpu: "2000m"
            memory: "4Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"

      # Priority class for tenant tier
      priorityClassName: premium-tenant-priority
```

### Pattern 3: Cluster-Per-Tenant (Hard Multi-Tenancy)

**Architecture:**
- Completely isolated Kubernetes clusters
- Dedicated control plane per tenant
- No resource sharing between tenants
- Maximum isolation and security

**Best For:**
- Enterprise customers with strict compliance
- Regulated industries (finance, healthcare)
- Customers requiring dedicated infrastructure
- High-security government contracts

**Management:**
```python
# Automated cluster provisioning script
import boto3
from kubernetes import client, config

def provision_tenant_cluster(tenant_id: str, tier: str, region: str):
    """Provision dedicated EKS cluster for enterprise tenant."""
    eks = boto3.client('eks', region_name=region)

    cluster_config = {
        'name': f'genops-{tenant_id}',
        'version': '1.28',
        'roleArn': 'arn:aws:iam::ACCOUNT:role/EKSClusterRole',
        'resourcesVpcConfig': {
            'subnetIds': get_tenant_subnets(tenant_id),
            'securityGroupIds': [get_tenant_security_group(tenant_id)],
            'endpointPrivateAccess': True,
            'endpointPublicAccess': False
        },
        'logging': {
            'clusterLogging': [{
                'types': ['api', 'audit', 'authenticator'],
                'enabled': True
            }]
        },
        'tags': {
            'genops.ai/tenant-id': tenant_id,
            'genops.ai/tenant-tier': tier,
            'genops.ai/managed-by': 'genops-platform'
        }
    }

    # Create cluster
    response = eks.create_cluster(**cluster_config)

    # Wait for cluster to be active
    waiter = eks.get_waiter('cluster_active')
    waiter.wait(name=f'genops-{tenant_id}')

    # Create node group
    node_group_config = {
        'clusterName': f'genops-{tenant_id}',
        'nodegroupName': f'{tier}-nodes',
        'scalingConfig': {
            'minSize': 3,
            'maxSize': 20,
            'desiredSize': 3
        },
        'subnets': get_tenant_subnets(tenant_id),
        'instanceTypes': get_instance_types_for_tier(tier),
        'amiType': 'AL2_x86_64',
        'nodeRole': 'arn:aws:iam::ACCOUNT:role/EKSNodeRole',
        'labels': {
            'genops.ai/tenant-id': tenant_id,
            'genops.ai/tier': tier
        },
        'tags': {
            'genops.ai/tenant-id': tenant_id
        }
    }

    eks.create_nodegroup(**node_group_config)

    print(f"✅ Provisioned cluster for tenant: {tenant_id}")
    return f'genops-{tenant_id}'
```

---

## Resource Isolation (Advanced Techniques)

### Hierarchical Resource Quotas

**Parent-Child Resource Allocation:**
```yaml
# Organization-level quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: org-acme-quota
  namespace: acme-org
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"

---
# Team-level quota (subset of org quota)
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-ml-platform-quota
  namespace: acme-ml-platform
spec:
  hard:
    requests.cpu: "40"
    requests.memory: "80Gi"

---
# Project-level quota (subset of team quota)
apiVersion: v1
kind: ResourceQuota
metadata:
  name: project-inference-quota
  namespace: acme-inference-prod
spec:
  hard:
    requests.cpu: "20"
    requests.memory: "40Gi"
```

### Quality of Service (QoS) Classes

**Priority-Based Scheduling:**
```yaml
# Critical production workloads
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: tenant-critical
value: 1000000
globalDefault: false
description: "Critical production workloads for paying customers"

---
# Standard production workloads
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: tenant-high
value: 100000
description: "Standard production workloads"

---
# Development/test workloads
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: tenant-low
value: 1000
description: "Development and testing workloads"

---
# Use priority in deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-production
spec:
  template:
    spec:
      priorityClassName: tenant-critical
      containers:
      - name: app
        image: genops-ai:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "2"
            memory: "4Gi"  # Guaranteed QoS
```

### PodDisruptionBudgets for Availability

**Ensure Minimum Availability During Disruptions:**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: genops-ai-pdb
  namespace: tenant-acme-corp
spec:
  minAvailable: 2  # Always maintain 2 pods
  selector:
    matchLabels:
      app: genops-ai-inference
      tenant-id: acme-corp

---
# Alternative: percentage-based
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: genops-ai-pdb-percent
  namespace: tenant-acme-corp
spec:
  minAvailable: 60%  # Always maintain 60% of pods
  selector:
    matchLabels:
      app: genops-ai-inference
```

---

## Network Segmentation (Complete Isolation)

### Zero-Trust Network Policies

**Default Deny All + Selective Allow:**
```yaml
# Step 1: Deny all ingress and egress by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: tenant-acme-corp
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Step 2: Allow ingress from API gateway only
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-api-gateway
  namespace: tenant-acme-corp
spec:
  podSelector:
    matchLabels:
      app: genops-ai-inference
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    ports:
    - protocol: TCP
      port: 8080

---
# Step 3: Allow egress to AI providers and observability
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-egress-controlled
  namespace: tenant-acme-corp
spec:
  podSelector:
    matchLabels:
      app: genops-ai-inference
  policyTypes:
  - Egress
  egress:
  # DNS resolution
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53

  # OTLP telemetry export
  - to:
    - namespaceSelector:
        matchLabels:
          name: observability
      podSelector:
        matchLabels:
          app: otel-collector
    ports:
    - protocol: TCP
      port: 4318

  # HTTPS to external AI APIs
  - to:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 443

  # Explicitly block cross-tenant communication
  - to:
    - namespaceSelector:
        matchExpressions:
        - key: genops.ai/tenant-id
          operator: In
          values: ["acme-corp"]  # Only same tenant
```

### Service Mesh Authorization for Tenants

**Istio AuthorizationPolicy for Tenant Isolation:**
```yaml
# Deny cross-tenant service calls
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: tenant-isolation
  namespace: tenant-acme-corp
spec:
  selector:
    matchLabels:
      app: genops-ai-inference

  action: ALLOW

  rules:
  # Allow only requests from same tenant
  - from:
    - source:
        principals:
        - cluster.local/ns/tenant-acme-corp/sa/*
        namespaces:
        - tenant-acme-corp
    when:
    - key: source.labels[genops.ai/tenant-id]
      values: ["acme-corp"]

  # Allow from API gateway with tenant validation
  - from:
    - source:
        namespaces:
        - api-gateway
    when:
    - key: request.headers[x-genops-customer-id]
      values: ["acme-corp"]
```

---

## Cost Attribution and Billing

### Real-Time Cost Tracking

**GenOps Cost Attribution Integration:**
```python
from genops import track_usage, get_tenant_cost_summary
from datetime import datetime, timedelta

@track_usage(
    team="platform-team",
    project="saas-inference",
    customer_id="acme-corp",  # Tenant ID
    budget_limit=5000.0,
    budget_period="monthly"
)
def serve_tenant_request(tenant_id: str, request_data: dict):
    """
    Serve AI inference request with automatic cost tracking.

    All costs automatically attributed to tenant for billing.
    """
    response = ai_model.generate(request_data)
    return response

def generate_tenant_invoice(tenant_id: str, month: str):
    """Generate monthly invoice for tenant."""
    # Get detailed cost breakdown
    summary = get_tenant_cost_summary(
        customer_id=tenant_id,
        start_date=datetime.fromisoformat(f"{month}-01"),
        end_date=datetime.fromisoformat(f"{month}-01") + timedelta(days=30)
    )

    invoice = {
        "tenant_id": tenant_id,
        "billing_period": month,
        "total_cost": summary.total_cost,
        "breakdown": {
            "compute": summary.compute_cost,
            "storage": summary.storage_cost,
            "network": summary.network_cost,
            "ai_api_calls": summary.ai_api_cost
        },
        "usage_metrics": {
            "api_calls": summary.total_requests,
            "tokens_processed": summary.total_tokens,
            "storage_gb_hours": summary.storage_usage
        },
        "cost_by_project": summary.cost_by_project,
        "cost_by_team": summary.cost_by_team
    }

    return invoice
```

### Kubecost Integration for Multi-Tenant Cost Allocation

**Install and Configure Kubecost:**
```bash
# Install Kubecost
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm install kubecost kubecost/cost-analyzer \
  --namespace kubecost \
  --create-namespace \
  --set kubecostToken="<TOKEN>"

# Configure tenant label allocation
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubecost-cost-model
  namespace: kubecost
data:
  allocation.yaml: |
    tenant_label: genops.ai/tenant-id
    team_label: genops.ai/team
    project_label: genops.ai/project

    # Shared cost allocation
    shared_namespace_cost:
      - namespace: kube-system
        allocation_method: even  # Split evenly across tenants
      - namespace: istio-system
        allocation_method: proportional  # Proportional to usage
EOF
```

---

## Security and Compliance

### Pod Security Standards Enforcement

**Apply Pod Security Standards Per Tenant:**
```yaml
# Restricted standard for production tenants
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-acme-corp
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
    genops.ai/tenant-id: "acme-corp"

---
# Baseline for development tenants
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-dev-sandbox
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: baseline
    pod-security.kubernetes.io/warn: baseline
    genops.ai/tenant-id: "dev-sandbox"
```

### OPA Gatekeeper Policies for Tenants

**Enforce Governance Labels:**
```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          type: object
          properties:
            labels:
              type: array
              items:
                type: string
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8srequiredlabels

      violation[{"msg": msg, "details": {"missing_labels": missing}}] {
        provided := {label | input.review.object.metadata.labels[label]}
        required := {label | label := input.parameters.labels[_]}
        missing := required - provided
        count(missing) > 0
        msg := sprintf("Missing required labels: %v", [missing])
      }

---
# Require GenOps governance labels on all pods
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: require-genops-labels
spec:
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
    namespaceSelector:
      matchExpressions:
      - key: genops.ai/tenant-id
        operator: Exists
  parameters:
    labels:
    - genops.ai/team
    - genops.ai/customer-id
    - genops.ai/project
```

### Audit Logging Per Tenant

**Configure Audit Policy:**
```yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Log all tenant operations
- level: RequestResponse
  namespaces: ["tenant-*"]
  verbs: ["create", "update", "patch", "delete"]
  resources:
  - group: ""
    resources: ["pods", "services", "secrets", "configmaps"]

# Log tenant RBAC changes
- level: RequestResponse
  verbs: ["create", "update", "patch", "delete"]
  resources:
  - group: "rbac.authorization.k8s.io"

# Forward tenant logs to GenOps
omitStages:
- RequestReceived
```

---

## Tenant Onboarding and Lifecycle Management

### Automated Tenant Provisioning

**Complete Tenant Onboarding Script:**
```python
#!/usr/bin/env python3
"""
Automated tenant provisioning for GenOps multi-tenant platform.
"""
from kubernetes import client, config
from typing import Dict, List

def provision_tenant(
    tenant_id: str,
    tier: str,
    quota_config: Dict[str, str],
    team: str = "platform-team"
) -> Dict[str, str]:
    """
    Provision complete tenant environment with all resources.

    Args:
        tenant_id: Unique tenant identifier
        tier: Tenant tier (free/standard/premium/enterprise)
        quota_config: Resource quota configuration
        team: Managing team name

    Returns:
        Dict with provisioned resource names and endpoints
    """
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    rbac_v1 = client.RbacAuthorizationV1Api()
    networking_v1 = client.NetworkingV1Api()

    namespace_name = f"tenant-{tenant_id}"

    # 1. Create namespace with labels
    namespace = client.V1Namespace(
        metadata=client.V1ObjectMeta(
            name=namespace_name,
            labels={
                "genops.ai/tenant-id": tenant_id,
                "genops.ai/tenant-tier": tier,
                "genops.ai/team": team,
                "pod-security.kubernetes.io/enforce": "restricted"
            },
            annotations={
                "genops.ai/created-at": datetime.utcnow().isoformat(),
                "genops.ai/managed-by": "genops-platform"
            }
        )
    )
    v1.create_namespace(namespace)

    # 2. Create resource quota
    quota = client.V1ResourceQuota(
        metadata=client.V1ObjectMeta(name="tenant-quota"),
        spec=client.V1ResourceQuotaSpec(hard=quota_config)
    )
    v1.create_namespaced_resource_quota(namespace_name, quota)

    # 3. Create limit range
    limit_range = client.V1LimitRange(
        metadata=client.V1ObjectMeta(name="tenant-limits"),
        spec=client.V1LimitRangeSpec(
            limits=[
                client.V1LimitRangeItem(
                    type="Container",
                    default={"cpu": "1", "memory": "2Gi"},
                    default_request={"cpu": "500m", "memory": "1Gi"},
                    max={"cpu": "4", "memory": "8Gi"},
                    min={"cpu": "100m", "memory": "128Mi"}
                )
            ]
        )
    )
    v1.create_namespaced_limit_range(namespace_name, limit_range)

    # 4. Create network policies
    default_deny = client.V1NetworkPolicy(
        metadata=client.V1ObjectMeta(name="default-deny-all"),
        spec=client.V1NetworkPolicySpec(
            pod_selector=client.V1LabelSelector(),
            policy_types=["Ingress", "Egress"]
        )
    )
    networking_v1.create_namespaced_network_policy(namespace_name, default_deny)

    # 5. Create RBAC for tenant admin
    role = client.V1Role(
        metadata=client.V1ObjectMeta(name="tenant-admin"),
        rules=[
            client.V1PolicyRule(
                api_groups=["", "apps"],
                resources=["pods", "deployments", "services"],
                verbs=["get", "list", "watch", "create", "update", "patch", "delete"]
            )
        ]
    )
    rbac_v1.create_namespaced_role(namespace_name, role)

    # 6. Deploy GenOps-enabled application
    deployment = create_genops_deployment(tenant_id, tier)
    apps_v1.create_namespaced_deployment(namespace_name, deployment)

    # 7. Create service
    service = create_tenant_service(tenant_id)
    v1.create_namespaced_service(namespace_name, service)

    print(f"✅ Tenant {tenant_id} provisioned successfully!")
    return {
        "namespace": namespace_name,
        "tier": tier,
        "status": "active"
    }


def deprovision_tenant(tenant_id: str):
    """Safely deprovision tenant and clean up all resources."""
    config.load_kube_config()
    v1 = client.CoreV1Api()

    namespace_name = f"tenant-{tenant_id}"

    # Delete namespace (cascades to all resources)
    v1.delete_namespace(namespace_name)

    print(f"✅ Tenant {tenant_id} deprovisioned successfully!")


# Example usage
if __name__ == "__main__":
    provision_tenant(
        tenant_id="acme-corp",
        tier="enterprise",
        quota_config={
            "requests.cpu": "20",
            "requests.memory": "40Gi",
            "requests.nvidia.com/gpu": "4"
        }
    )
```

---

## Performance and Scalability

### Preventing Noisy Neighbor Problems

**CPU Throttling and Priority:**
```yaml
# High-priority tenant workload
apiVersion: apps/v1
kind: Deployment
metadata:
  name: premium-tenant-inference
spec:
  template:
    spec:
      priorityClassName: tenant-critical
      containers:
      - name: inference
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
          limits:
            cpu: "4"     # No throttling - guaranteed
            memory: "8Gi"

---
# Lower-priority background jobs
apiVersion: batch/v1
kind: Job
metadata:
  name: free-tier-batch-job
spec:
  template:
    spec:
      priorityClassName: tenant-low
      containers:
      - name: batch
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"       # Can be throttled
            memory: "2Gi"
```

### Tenant-Specific Autoscaling

**HPA Per Tenant:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tenant-acme-hpa
  namespace: tenant-acme-corp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-ai-inference
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # Custom metric: requests per second per tenant
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
        selector:
          matchLabels:
            tenant_id: "acme-corp"
      target:
        type: AverageValue
        averageValue: "100"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

---

## Multi-Tenant Best Practices

✅ **Isolation:**
- [ ] Use dedicated namespaces per tenant
- [ ] Implement network policies for traffic isolation
- [ ] Apply resource quotas to prevent resource exhaustion
- [ ] Use node pools for different tenant tiers

✅ **Security:**
- [ ] Enable RBAC with least-privilege principles
- [ ] Implement Pod Security Standards
- [ ] Isolate secrets per tenant namespace
- [ ] Enable audit logging for compliance
- [ ] Regular security scanning and updates

✅ **Cost Management:**
- [ ] Track costs per tenant with GenOps governance attributes
- [ ] Set budget limits per tenant
- [ ] Implement automated cost alerts
- [ ] Generate per-tenant billing reports

✅ **Performance:**
- [ ] Monitor resource usage per tenant
- [ ] Implement autoscaling policies
- [ ] Use PodDisruptionBudgets for availability
- [ ] Load test with realistic multi-tenant scenarios

✅ **Operations:**
- [ ] Automate tenant provisioning and deprovisioning
- [ ] Implement self-service tenant portals
- [ ] Monitor tenant health and SLAs
- [ ] Plan for tenant migrations and upgrades

---

## Tenant Isolation Comparison

| Aspect | Namespace Isolation | Node Pool Isolation | Cluster Isolation |
|--------|-------------------|-------------------|------------------|
| **Security** | Medium | High | Very High |
| **Performance Isolation** | Low | Medium | High |
| **Cost Efficiency** | High | Medium | Low |
| **Operational Complexity** | Low | Medium | High |
| **Best For** | Internal teams | Mixed workloads | Enterprise customers |

---

## Next Steps

Ready to build a multi-tenant AI platform? Start with:

1. **Define Tenant Isolation Requirements** - Choose the right isolation level
2. **Design Resource Allocation** - Plan quotas and node pools
3. **Implement Network Policies** - Secure tenant communication
4. **Configure GenOps Governance** - Set up per-tenant cost tracking
5. **Automate Tenant Lifecycle** - Build provisioning and management tools
6. **Monitor and Optimize** - Track tenant metrics and costs

Return to [Kubernetes Getting Started](kubernetes-getting-started.md) for the complete deployment overview.

---

## Support

- **Documentation:** [GenOps AI Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community:** [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
