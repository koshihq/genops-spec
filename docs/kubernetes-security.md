# Security Hardening for GenOps AI on Kubernetes

> **Status:** ✅ Production Ready
> **Last Updated:** 2026-01-18

Secure your GenOps AI deployment with comprehensive Kubernetes security best practices and governance-aware policies.

---

## Overview

Security in Kubernetes requires a defense-in-depth approach across multiple layers:
- **Access Control** with RBAC and service accounts for least-privilege access
- **Network Security** with policies, encryption, and segmentation
- **Container Security** with image scanning, Pod Security Standards, and runtime protection
- **Data Protection** with secret management, encryption at rest, and audit logging
- **Compliance** with automated policy enforcement and governance tracking

GenOps AI integrates security governance into your AI workloads, providing visibility and control over sensitive operations.

---

## Quick Start (5 Minutes)

**Deploy GenOps AI with security hardening:**

```bash
# 1. Create namespace with Pod Security Standards
kubectl create namespace genops
kubectl label namespace genops \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted

# 2. Apply default-deny NetworkPolicy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: genops
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

# 3. Deploy GenOps AI with security context
kubectl apply -f https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/kubernetes/secure-deployment.yaml

# 4. Verify security posture
kubectl get psp,networkpolicy,serviceaccount -n genops
```

---

## Table of Contents

1. [Authentication and Authorization](#authentication-and-authorization)
2. [Network Security](#network-security)
3. [Container and Pod Security](#container-and-pod-security)
4. [Secret Management](#secret-management)
5. [Data Protection](#data-protection)
6. [Audit and Compliance](#audit-and-compliance)
7. [Supply Chain Security](#supply-chain-security)
8. [Runtime Security Monitoring](#runtime-security-monitoring)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Authentication and Authorization

### RBAC Configuration

**Least-Privilege Service Account:**

```yaml
# Service account with minimal permissions
apiVersion: v1
kind: ServiceAccount
metadata:
  name: genops-ai
  namespace: genops
  labels:
    app: genops-ai
    genops.ai/team: platform
automountServiceAccountToken: false  # Only mount when needed

---
# Role with minimal required permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: genops-ai-role
  namespace: genops
rules:
# Read access to ConfigMaps (for configuration)
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch"]
  resourceNames: ["genops-config"]  # Specific ConfigMap only

# Read access to Secrets (only specific secrets)
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["genops-api-keys", "genops-db-credentials"]
  verbs: ["get"]

# Pod logs access for debugging
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
# Bind role to service account
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: genops-ai-rolebinding
  namespace: genops
subjects:
- kind: ServiceAccount
  name: genops-ai
  namespace: genops
roleRef:
  kind: Role
  name: genops-ai-role
  apiGroup: rbac.authorization.k8s.io
```

### Advanced RBAC Patterns

**Multi-Team Access Control:**

```yaml
# Team-specific namespace access
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: genops-team-developer
  namespace: genops
rules:
# Full access to team's resources
- apiGroups: ["apps", ""]
  resources: ["deployments", "pods", "services", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]

# Read-only access to secrets
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]

# No delete permissions (require admin approval)
# No access to cluster-wide resources

---
# ClusterRole for platform administrators
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: genops-platform-admin
rules:
# Full control of GenOps resources across all namespaces
- apiGroups: ["apps", "", "networking.k8s.io", "rbac.authorization.k8s.io"]
  resources: ["*"]
  verbs: ["*"]

# Audit logging access
- apiGroups: ["audit.k8s.io"]
  resources: ["events"]
  verbs: ["get", "list", "watch"]

---
# ClusterRole for read-only auditors
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: genops-auditor
rules:
# Read-only access to all resources
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["get", "list", "watch"]

# Access to audit logs
- nonResourceURLs: ["/logs", "/logs/*"]
  verbs: ["get"]
```

### OIDC Integration

**Corporate Identity Provider Integration:**

```yaml
# API Server configuration (add to kube-apiserver flags)
# --oidc-issuer-url=https://accounts.example.com
# --oidc-client-id=kubernetes
# --oidc-username-claim=email
# --oidc-groups-claim=groups
# --oidc-ca-file=/etc/kubernetes/pki/oidc-ca.crt

---
# ClusterRoleBinding using OIDC groups
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: genops-developers-oidc
subjects:
- kind: Group
  name: genops-developers@example.com  # OIDC group claim
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: genops-team-developer
  apiGroup: rbac.authorization.k8s.io

---
# RoleBinding for specific team namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: team-alpha-binding
  namespace: genops-team-alpha
subjects:
- kind: Group
  name: team-alpha@example.com
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: genops-team-developer
  apiGroup: rbac.authorization.k8s.io
```

### Admission Controllers

**ValidatingWebhook for Governance Enforcement:**

```yaml
# ValidatingWebhookConfiguration
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: genops-governance-validator
webhooks:
- name: validate.genops.ai
  clientConfig:
    service:
      name: genops-webhook
      namespace: genops
      path: "/validate"
    caBundle: LS0tLS1CRUdJTi...  # Base64 CA cert
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: ["apps", "batch"]
    apiVersions: ["v1"]
    resources: ["deployments", "jobs", "cronjobs"]
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Fail  # Reject on webhook failure

---
# Webhook service implementation
apiVersion: v1
kind: Service
metadata:
  name: genops-webhook
  namespace: genops
spec:
  selector:
    app: genops-webhook
  ports:
  - port: 443
    targetPort: 8443
    protocol: TCP

---
# Webhook server deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-webhook
  namespace: genops
spec:
  replicas: 2
  selector:
    matchLabels:
      app: genops-webhook
  template:
    metadata:
      labels:
        app: genops-webhook
    spec:
      serviceAccountName: genops-webhook
      containers:
      - name: webhook
        image: genops-ai/webhook:latest
        ports:
        - containerPort: 8443
        env:
        - name: GENOPS_POLICY_ENDPOINT
          value: "http://opa.genops:8181/v1/data/genops/allow"
        volumeMounts:
        - name: webhook-certs
          mountPath: /etc/webhook/certs
          readOnly: true
      volumes:
      - name: webhook-certs
        secret:
          secretName: genops-webhook-certs
```

**Webhook Server Implementation (Python):**

```python
#!/usr/bin/env python3
"""GenOps Governance Validation Webhook"""

from flask import Flask, request, jsonify
import base64
import json

app = Flask(__name__)

REQUIRED_LABELS = [
    "genops.ai/team",
    "genops.ai/project",
    "genops.ai/cost-center"
]

BUDGET_LIMITS = {
    "team-alpha": 1000.00,
    "team-beta": 500.00,
    "team-gamma": 2000.00
}

@app.route('/validate', methods=['POST'])
def validate():
    """Validate admission request against governance policies"""
    admission_review = request.get_json()

    # Extract admission request
    admission_request = admission_review.get('request', {})
    uid = admission_request.get('uid')
    kind = admission_request.get('kind', {}).get('kind')
    namespace = admission_request.get('namespace')
    operation = admission_request.get('operation')

    # Decode object
    obj = admission_request.get('object', {})
    metadata = obj.get('metadata', {})
    labels = metadata.get('labels', {})

    # Validation checks
    allowed = True
    message = "Request allowed"

    # Check 1: Required governance labels
    missing_labels = [label for label in REQUIRED_LABELS if label not in labels]
    if missing_labels:
        allowed = False
        message = f"Missing required governance labels: {', '.join(missing_labels)}"

    # Check 2: Budget validation
    if allowed:
        team = labels.get('genops.ai/team')
        if team in BUDGET_LIMITS:
            # Query current spend from metrics
            current_spend = query_team_spend(team, namespace)
            budget_limit = BUDGET_LIMITS[team]

            if current_spend >= budget_limit * 0.95:
                allowed = False
                message = f"Team {team} at 95% budget utilization ({current_spend}/{budget_limit})"

    # Check 3: Security context validation
    if allowed and kind in ["Deployment", "StatefulSet"]:
        spec = obj.get('spec', {}).get('template', {}).get('spec', {})
        security_context = spec.get('securityContext', {})

        if not security_context.get('runAsNonRoot'):
            allowed = False
            message = "Deployment must set runAsNonRoot=true"

        if not security_context.get('seccompProfile'):
            allowed = False
            message = "Deployment must set seccompProfile"

    # Build admission response
    admission_response = {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": allowed,
            "status": {
                "message": message
            }
        }
    }

    return jsonify(admission_response)

def query_team_spend(team: str, namespace: str) -> float:
    """Query current team spend from Prometheus"""
    # Implementation would query Prometheus for:
    # sum(genops_cost_total{team="$team", namespace="$namespace"})
    return 0.0  # Placeholder

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443, ssl_context=('/etc/webhook/certs/tls.crt', '/etc/webhook/certs/tls.key'))
```

---

## Network Security

### NetworkPolicy Patterns

**Default Deny All Traffic:**

```yaml
# Deny all ingress and egress by default
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: genops
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

**Allow Ingress from Ingress Controller:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-controller
  namespace: genops
spec:
  podSelector:
    matchLabels:
      app: genops-ai
  policyTypes:
  - Ingress
  ingress:
  - from:
    # Allow from ingress-nginx namespace
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
      podSelector:
        matchLabels:
          app.kubernetes.io/name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
```

**Allow Egress to Specific Services:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-egress-selective
  namespace: genops
spec:
  podSelector:
    matchLabels:
      app: genops-ai
  policyTypes:
  - Egress
  egress:
  # DNS resolution (CoreDNS)
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    - podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53

  # PostgreSQL database
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432

  # Redis cache
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379

  # OTLP exporter (observability namespace)
  - to:
    - namespaceSelector:
        matchLabels:
          name: observability
      podSelector:
        matchLabels:
          app: otel-collector
    ports:
    - protocol: TCP
      port: 4317
    - protocol: TCP
      port: 4318

  # External HTTPS (AI provider APIs)
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443

  # Allow intra-namespace communication
  - to:
    - podSelector: {}
```

**Micro-Segmentation for Multi-Tenant:**

```yaml
# Isolate customer workloads from each other
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: isolate-customer-workloads
  namespace: genops
spec:
  podSelector:
    matchLabels:
      genops.ai/workload-type: customer
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Only allow from API gateway
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080

  egress:
  # DNS
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

  # Shared services only (no cross-customer traffic)
  - to:
    - podSelector:
        matchLabels:
          genops.ai/workload-type: shared-service
    ports:
    - protocol: TCP
      port: 8080

  # External AI APIs
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

### Service Mesh Security (Istio)

**Mutual TLS Enforcement:**

```yaml
# Enforce mTLS for all services in namespace
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default-mtls
  namespace: genops
spec:
  mtls:
    mode: STRICT  # Require mTLS for all traffic

---
# Destination rule for mTLS
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: default-mtls
  namespace: genops
spec:
  host: "*.genops.svc.cluster.local"
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
```

**Authorization Policies:**

```yaml
# Allow only API gateway to access backend services
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: api-gateway-to-backend
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-backend
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/genops/sa/api-gateway"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/*"]

---
# Deny all by default, require JWT
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: require-jwt
  namespace: genops
spec:
  action: DENY
  rules:
  - from:
    - source:
        notRequestPrincipals: ["*"]  # No JWT present

---
# Request authentication (JWT validation)
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai
  jwtRules:
  - issuer: "https://auth.example.com"
    jwksUri: "https://auth.example.com/.well-known/jwks.json"
    audiences:
    - "genops-ai-api"
    forwardOriginalToken: true
```

---

## Container and Pod Security

### Pod Security Standards

**Restricted Profile Enforcement:**

```yaml
# Namespace with restricted Pod Security Standard
apiVersion: v1
kind: Namespace
metadata:
  name: genops
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest

---
# Deployment meeting restricted requirements
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai
  namespace: genops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-ai
  template:
    metadata:
      labels:
        app: genops-ai
        genops.ai/team: platform
        genops.ai/project: core
    spec:
      # Use dedicated service account
      serviceAccountName: genops-ai
      automountServiceAccountToken: false

      # Pod-level security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 3000
        fsGroup: 2000
        fsGroupChangePolicy: "OnRootMismatch"
        seccompProfile:
          type: RuntimeDefault
        supplementalGroups: [4000]

      containers:
      - name: genops-ai
        image: genops-ai:latest

        # Container-level security context
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true

        # Resource limits
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"

        # Health probes
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

        # Volume mounts (read-only where possible)
        volumeMounts:
        - name: config
          mountPath: /etc/genops
          readOnly: true
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /var/cache/genops

      volumes:
      - name: config
        configMap:
          name: genops-config
          defaultMode: 0440
      - name: tmp
        emptyDir:
          sizeLimit: 1Gi
      - name: cache
        emptyDir:
          sizeLimit: 5Gi
```

### Runtime Security with Falco

**Falco Installation:**

```bash
# Install Falco with Helm
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm repo update

helm install falco falcosecurity/falco \
  --namespace falco-system \
  --create-namespace \
  --set falcosidekick.enabled=true \
  --set falcosidekick.webui.enabled=true
```

**Custom Falco Rules for GenOps:**

```yaml
# ConfigMap with custom Falco rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: falco-custom-rules
  namespace: falco-system
data:
  custom-rules.yaml: |
    # Detect unauthorized API key access
    - rule: Unauthorized Secret Access
      desc: Detect access to secrets outside allowed paths
      condition: >
        open_read and
        container.image.repository contains "genops-ai" and
        fd.name glob "/var/run/secrets/kubernetes.io/*" and
        not fd.name glob "/var/run/secrets/kubernetes.io/serviceaccount/token"
      output: >
        Unauthorized secret access
        (user=%user.name command=%proc.cmdline file=%fd.name container=%container.name)
      priority: WARNING
      tags: [filesystem, secrets]

    # Detect privilege escalation attempts
    - rule: Privilege Escalation Attempt
      desc: Detect attempts to escalate privileges
      condition: >
        spawned_process and
        container and
        proc.name in (sudo, su, setuid) and
        container.image.repository contains "genops-ai"
      output: >
        Privilege escalation attempt detected
        (user=%user.name command=%proc.cmdline container=%container.name)
      priority: CRITICAL
      tags: [process, privilege_escalation]

    # Detect unexpected network connections
    - rule: Unexpected Outbound Connection
      desc: Detect connections to unexpected external IPs
      condition: >
        outbound and
        container.image.repository contains "genops-ai" and
        not fd.sip.name in (allowed_domains)
      output: >
        Unexpected outbound connection
        (connection=%fd.name container=%container.name dest=%fd.rip:%fd.rport)
      priority: WARNING
      tags: [network]

    # Detect file modification in read-only paths
    - rule: Write to Read-Only Path
      desc: Detect writes to paths that should be read-only
      condition: >
        open_write and
        container.image.repository contains "genops-ai" and
        fd.name glob "/etc/*"
      output: >
        Write to read-only path detected
        (user=%user.name file=%fd.name container=%container.name)
      priority: ERROR
      tags: [filesystem]

    # Detect sensitive data exfiltration
    - rule: Sensitive Data Exfiltration
      desc: Detect potential exfiltration of sensitive data
      condition: >
        outbound and
        evt.buffer contains "api_key" or
        evt.buffer contains "password" or
        evt.buffer contains "secret"
      output: >
        Potential sensitive data exfiltration
        (connection=%fd.name container=%container.name)
      priority: CRITICAL
      tags: [network, data_loss]

    # List of allowed domains for outbound connections
    - list: allowed_domains
      items:
        - api.openai.com
        - api.anthropic.com
        - bedrock-runtime.us-east-1.amazonaws.com
        - generativelanguage.googleapis.com
```

**Falco Alert Integration:**

```yaml
# FalcoSidekick configuration for alert routing
apiVersion: v1
kind: ConfigMap
metadata:
  name: falcosidekick-config
  namespace: falco-system
data:
  config.yaml: |
    slack:
      webhookurl: "https://hooks.slack.com/services/XXX/YYY/ZZZ"
      minimumpriority: "warning"
      messageformat: "Alert: *{{.Rule}}* - {{.Output}}"

    webhook:
      address: "http://genops-alerting.genops:8080/falco"
      minimumpriority: "error"

    pagerduty:
      routingkey: "YOUR_PAGERDUTY_KEY"
      minimumpriority: "critical"

    prometheus:
      address: "http://prometheus.observability:9090"
```

### AppArmor Profiles

**AppArmor Profile for GenOps AI:**

```yaml
# AppArmor profile ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: apparmor-profiles
  namespace: genops
data:
  genops-ai-profile: |
    #include <tunables/global>

    profile genops-ai flags=(attach_disconnected,mediate_deleted) {
      #include <abstractions/base>

      # Allow read access to application files
      /app/** r,
      /etc/genops/** r,

      # Allow write to temporary directories
      /tmp/** rw,
      /var/cache/genops/** rw,

      # Allow network access
      network inet stream,
      network inet6 stream,

      # Allow specific system calls
      capability setuid,
      capability setgid,
      capability net_bind_service,

      # Deny everything else
      deny /proc/sys/** w,
      deny /sys/** w,
      deny @{HOME}/.ssh/** rw,
      deny /etc/shadow r,
      deny /etc/passwd w,
    }

---
# Pod with AppArmor annotation
apiVersion: v1
kind: Pod
metadata:
  name: genops-ai-apparmor
  namespace: genops
  annotations:
    container.apparmor.security.beta.kubernetes.io/genops-ai: localhost/genops-ai-profile
spec:
  containers:
  - name: genops-ai
    image: genops-ai:latest
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
```

### Seccomp Profiles

**Custom Seccomp Profile:**

```yaml
# Seccomp profile ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: seccomp-profiles
  namespace: genops
data:
  genops-ai-seccomp.json: |
    {
      "defaultAction": "SCMP_ACT_ERRNO",
      "architectures": [
        "SCMP_ARCH_X86_64",
        "SCMP_ARCH_X86",
        "SCMP_ARCH_AARCH64"
      ],
      "syscalls": [
        {
          "names": [
            "accept",
            "accept4",
            "access",
            "arch_prctl",
            "bind",
            "brk",
            "clone",
            "close",
            "connect",
            "dup",
            "dup2",
            "epoll_create",
            "epoll_ctl",
            "epoll_wait",
            "exit",
            "exit_group",
            "fcntl",
            "fstat",
            "futex",
            "getcwd",
            "getdents",
            "getpeername",
            "getpid",
            "getsockname",
            "getsockopt",
            "listen",
            "mmap",
            "mprotect",
            "munmap",
            "open",
            "openat",
            "poll",
            "read",
            "recvfrom",
            "recvmsg",
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "sendmsg",
            "sendto",
            "setsockopt",
            "socket",
            "stat",
            "write"
          ],
          "action": "SCMP_ACT_ALLOW"
        }
      ]
    }

---
# Pod using custom seccomp profile
apiVersion: v1
kind: Pod
metadata:
  name: genops-ai-seccomp
  namespace: genops
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: genops-ai-seccomp.json
  containers:
  - name: genops-ai
    image: genops-ai:latest
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
```

---

## Secret Management

### External Secrets Operator

**Installation:**

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets-system \
  --create-namespace
```

**AWS Secrets Manager Integration:**

```yaml
# SecretStore for AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: genops
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: genops-ai

---
# ServiceAccount with IRSA (IAM Roles for Service Accounts)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: genops-ai
  namespace: genops
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/genops-secrets-reader

---
# ExternalSecret syncing from AWS
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: genops-api-keys
  namespace: genops
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: genops-api-keys
    creationPolicy: Owner
    template:
      engineVersion: v2
      data:
        OPENAI_API_KEY: "{{ .openai_key }}"
        ANTHROPIC_API_KEY: "{{ .anthropic_key }}"
        DB_PASSWORD: "{{ .db_password }}"
  data:
  - secretKey: openai_key
    remoteRef:
      key: prod/genops/openai-api-key
  - secretKey: anthropic_key
    remoteRef:
      key: prod/genops/anthropic-api-key
  - secretKey: db_password
    remoteRef:
      key: prod/genops/database-credentials
      property: password
```

**Azure Key Vault Integration:**

```yaml
# SecretStore for Azure Key Vault
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: azure-keyvault
  namespace: genops
spec:
  provider:
    azurekv:
      vaultUrl: "https://genops-vault.vault.azure.net"
      authType: WorkloadIdentity
      serviceAccountRef:
        name: genops-ai

---
# ExternalSecret syncing from Azure
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: genops-azure-secrets
  namespace: genops
spec:
  refreshInterval: 30m
  secretStoreRef:
    name: azure-keyvault
    kind: SecretStore
  target:
    name: genops-azure-secrets
  data:
  - secretKey: api-key
    remoteRef:
      key: openai-api-key
  - secretKey: connection-string
    remoteRef:
      key: cosmos-connection-string
```

**HashiCorp Vault Integration:**

```yaml
# SecretStore for Vault
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: genops
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "genops-ai"
          serviceAccountRef:
            name: genops-ai

---
# ExternalSecret syncing from Vault
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: genops-vault-secrets
  namespace: genops
spec:
  refreshInterval: 15m
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: genops-vault-secrets
  data:
  - secretKey: database-url
    remoteRef:
      key: database/config
      property: url
  - secretKey: api-token
    remoteRef:
      key: api/tokens
      property: genops-prod-token
```

### Secret Rotation

**Automatic Secret Rotation with Reloader:**

```bash
# Install Reloader
kubectl apply -f https://raw.githubusercontent.com/stakater/Reloader/master/deployments/kubernetes/reloader.yaml
```

```yaml
# Deployment with automatic reload on secret change
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai
  namespace: genops
  annotations:
    reloader.stakater.com/auto: "true"  # Auto-reload on ConfigMap/Secret change
    # OR specific secrets:
    # reloader.stakater.com/search: "true"
    # secret.reloader.stakater.com/reload: "genops-api-keys,genops-db-creds"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-ai
  template:
    metadata:
      labels:
        app: genops-ai
    spec:
      containers:
      - name: genops-ai
        image: genops-ai:latest
        envFrom:
        - secretRef:
            name: genops-api-keys
```

### Certificate Management

**cert-manager Installation:**

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

**Certificate Issuance:**

```yaml
# ClusterIssuer for Let's Encrypt
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx

---
# Certificate for GenOps AI
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: genops-ai-tls
  namespace: genops
spec:
  secretName: genops-ai-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - genops.example.com
  - api.genops.example.com

---
# Ingress using the certificate
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genops-ai
  namespace: genops
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - genops.example.com
    secretName: genops-ai-tls
  rules:
  - host: genops.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: genops-ai
            port:
              number: 80
```

---

## Data Protection

### Encryption in Transit

**Service-to-Service mTLS (Linkerd):**

```bash
# Install Linkerd
linkerd install --crds | kubectl apply -f -
linkerd install | kubectl apply -f -
linkerd check

# Inject Linkerd into namespace
kubectl annotate namespace genops linkerd.io/inject=enabled
```

**TLS for External Traffic:**

```yaml
# Ingress with TLS termination
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genops-ai-tls
  namespace: genops
  annotations:
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
    nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - genops.example.com
    secretName: genops-ai-tls
  rules:
  - host: genops.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: genops-ai
            port:
              number: 443
```

### Encryption at Rest

**etcd Encryption Configuration:**

```yaml
# EncryptionConfiguration for etcd
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
    - secrets
    - configmaps
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: BASE64_ENCODED_32_BYTE_KEY
    - identity: {}  # Fallback to unencrypted

# Apply to API server:
# --encryption-provider-config=/etc/kubernetes/enc/encryption-config.yaml
```

**Persistent Volume Encryption:**

```yaml
# StorageClass with encryption (AWS EBS)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: encrypted-gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
  kmsKeyId: arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true

---
# PVC using encrypted storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: genops-data-encrypted
  namespace: genops
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: encrypted-gp3
  resources:
    requests:
      storage: 100Gi
```

---

## Audit and Compliance

### Kubernetes Audit Logging

**Audit Policy:**

```yaml
# Audit policy ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: audit-policy
  namespace: kube-system
data:
  audit-policy.yaml: |
    apiVersion: audit.k8s.io/v1
    kind: Policy
    rules:
    # Log all requests at RequestResponse level
    - level: RequestResponse
      resources:
      - group: ""
        resources: ["secrets", "configmaps"]

    # Log metadata for resource modifications
    - level: Metadata
      verbs: ["create", "update", "patch", "delete"]

    # Log metadata for authentication/authorization
    - level: Metadata
      nonResourceURLs:
      - "/api*"
      - "/apis*"
      - "/version"

    # Don't log read-only requests
    - level: None
      verbs: ["get", "list", "watch"]
      resources:
      - group: ""
        resources: ["pods/log", "pods/status"]

    # Don't log health checks
    - level: None
      users: ["system:kube-proxy"]
      verbs: ["watch"]
      resources:
      - group: ""
        resources: ["services", "endpoints"]

    # Log everything else at Metadata level
    - level: Metadata
      omitStages:
      - "RequestReceived"

# Apply to API server:
# --audit-policy-file=/etc/kubernetes/audit/audit-policy.yaml
# --audit-log-path=/var/log/kubernetes/audit.log
# --audit-log-maxage=30
# --audit-log-maxbackup=10
# --audit-log-maxsize=100
```

### OPA Gatekeeper Policies

**Installation:**

```bash
# Install Gatekeeper
kubectl apply -f https://raw.githubusercontent.com/open-policy-agent/gatekeeper/master/deploy/gatekeeper.yaml
```

**Constraint Templates:**

```yaml
# Template: Required governance labels
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequiredgovernancelabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredGovernanceLabels
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
        package k8srequiredgovernancelabels

        violation[{"msg": msg, "details": {"missing_labels": missing}}] {
          provided := {label | input.review.object.metadata.labels[label]}
          required := {label | label := input.parameters.labels[_]}
          missing := required - provided
          count(missing) > 0
          msg := sprintf("Required governance labels missing: %v", [missing])
        }

---
# Constraint: Enforce governance labels
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredGovernanceLabels
metadata:
  name: require-genops-labels
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment", "StatefulSet"]
    namespaces:
      - genops
  parameters:
    labels:
      - "genops.ai/team"
      - "genops.ai/project"
      - "genops.ai/cost-center"
      - "genops.ai/environment"

---
# Template: Budget limit enforcement
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sbudgetlimit
spec:
  crd:
    spec:
      names:
        kind: K8sBudgetLimit
      validation:
        openAPIV3Schema:
          type: object
          properties:
            teamBudgets:
              type: object
              additionalProperties:
                type: number
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8sbudgetlimit

        violation[{"msg": msg}] {
          team := input.review.object.metadata.labels["genops.ai/team"]
          budget_limit := input.parameters.teamBudgets[team]

          # Query current spend from external data source
          current_spend := data.genops.team_spend[team]

          # Check if adding this resource would exceed budget
          projected_cost := estimate_cost(input.review.object)
          total := current_spend + projected_cost

          total > budget_limit
          msg := sprintf("Team %v would exceed budget: %v + %v > %v",
                        [team, current_spend, projected_cost, budget_limit])
        }

        estimate_cost(obj) = cost {
          # Simple cost estimation based on resource requests
          cpu := obj.spec.template.spec.containers[_].resources.requests.cpu
          memory := obj.spec.template.spec.containers[_].resources.requests.memory
          replicas := obj.spec.replicas

          # Example: $0.05/vCPU-hour + $0.01/GB-hour
          cost := (cpu * 0.05 + memory * 0.01) * replicas * 24 * 30
        }

---
# Constraint: Enforce team budgets
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sBudgetLimit
metadata:
  name: enforce-team-budgets
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment"]
    namespaces:
      - genops
  parameters:
    teamBudgets:
      team-alpha: 1000.00
      team-beta: 500.00
      team-gamma: 2000.00

---
# Template: Prevent privileged containers
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8spspprivilegedcontainer
spec:
  crd:
    spec:
      names:
        kind: K8sPSPPrivilegedContainer
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8spspprivilegedcontainer

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          container.securityContext.privileged
          msg := sprintf("Privileged container not allowed: %v", [container.name])
        }

---
# Constraint: Block privileged containers
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPPrivilegedContainer
metadata:
  name: block-privileged-containers
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    excludedNamespaces:
      - kube-system
      - kube-public
```

---

## Supply Chain Security

### Image Signing with Cosign

**Sign Container Images:**

```bash
# Generate key pair
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key genops-ai:latest

# Verify signature
cosign verify --key cosign.pub genops-ai:latest
```

**Policy Enforcement:**

```yaml
# ClusterImagePolicy for Cosign
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata:
  name: genops-image-policy
spec:
  images:
  - glob: "genops-ai/*:*"
  authorities:
  - key:
      data: |
        -----BEGIN PUBLIC KEY-----
        MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE...
        -----END PUBLIC KEY-----
```

### SBOM Generation

**Generate SBOM with Syft:**

```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM
syft genops-ai:latest -o cyclonedx-json > sbom.json

# Attach SBOM to image
cosign attach sbom --sbom sbom.json genops-ai:latest
```

**GitHub Actions Workflow:**

```yaml
name: Build, Sign, and Generate SBOM

on:
  push:
    branches: [main]

jobs:
  secure-build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write  # For keyless signing

    steps:
      - uses: actions/checkout@v3

      - name: Build image
        run: docker build -t genops-ai:${{ github.sha }} .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: genops-ai:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Fail on critical vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: genops-ai:${{ github.sha }}
          exit-code: '1'
          severity: 'CRITICAL'

      - name: Install Cosign
        uses: sigstore/cosign-installer@v3

      - name: Sign image (keyless)
        run: cosign sign genops-ai:${{ github.sha }}
        env:
          COSIGN_EXPERIMENTAL: 1

      - name: Install Syft
        uses: anchore/sbom-action/download-syft@v0

      - name: Generate SBOM
        run: syft genops-ai:${{ github.sha }} -o cyclonedx-json > sbom.json

      - name: Attach SBOM to image
        run: cosign attach sbom --sbom sbom.json genops-ai:${{ github.sha }}

      - name: Push image
        run: docker push genops-ai:${{ github.sha }}
```

---

## Runtime Security Monitoring

### Security Metrics Dashboard

**Prometheus Metrics:**

```yaml
# ServiceMonitor for Falco
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: falco
  namespace: falco-system
spec:
  selector:
    matchLabels:
      app: falco
  endpoints:
  - port: metrics
    interval: 30s

---
# Grafana Dashboard ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-dashboard
  namespace: observability
data:
  security-dashboard.json: |
    {
      "dashboard": {
        "title": "GenOps Security Dashboard",
        "panels": [
          {
            "title": "Security Alerts by Severity",
            "targets": [
              {
                "expr": "sum by (priority) (rate(falco_events_total[5m]))"
              }
            ]
          },
          {
            "title": "Unauthorized Secret Access Attempts",
            "targets": [
              {
                "expr": "sum(rate(falco_events_total{rule=\"Unauthorized Secret Access\"}[5m]))"
              }
            ]
          },
          {
            "title": "Privilege Escalation Attempts",
            "targets": [
              {
                "expr": "sum(rate(falco_events_total{rule=\"Privilege Escalation Attempt\"}[5m]))"
              }
            ]
          },
          {
            "title": "Network Policy Violations",
            "targets": [
              {
                "expr": "sum(rate(networkpolicy_drop_total[5m]))"
              }
            ]
          },
          {
            "title": "Failed Authentication Attempts",
            "targets": [
              {
                "expr": "sum(rate(apiserver_audit_event_total{verb=\"create\",objectRef_resource=\"tokenreviews\",responseStatus_code!=\"201\"}[5m]))"
              }
            ]
          }
        ]
      }
    }
```

### Security Incident Response

**Automated Response Workflow:**

```python
#!/usr/bin/env python3
"""Security Incident Response Automation"""

import os
import requests
from kubernetes import client, config
from datetime import datetime

# Load Kubernetes config
config.load_incluster_config()
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
PAGERDUTY_KEY = os.getenv("PAGERDUTY_INTEGRATION_KEY")

def handle_security_alert(alert: dict):
    """Handle security alert with automated response"""

    severity = alert.get("priority", "WARNING")
    rule = alert.get("rule")
    container = alert.get("output_fields", {}).get("container.name")
    namespace = alert.get("output_fields", {}).get("k8s.ns.name")
    pod = alert.get("output_fields", {}).get("k8s.pod.name")

    print(f"Security alert: {rule} - {severity}")

    # Send alert to Slack
    send_slack_alert(rule, severity, container, namespace, pod)

    # Critical alerts: immediate response
    if severity == "CRITICAL":
        page_oncall(rule, container, namespace, pod)

        # Quarantine pod
        if pod and namespace:
            quarantine_pod(namespace, pod)

    # High severity: scale down deployment
    elif severity == "ERROR" and container:
        scale_down_deployment(namespace, container)

    # All alerts: log to audit system
    log_security_event(alert)

def send_slack_alert(rule: str, severity: str, container: str, namespace: str, pod: str):
    """Send alert to Slack"""
    color = {
        "CRITICAL": "danger",
        "ERROR": "warning",
        "WARNING": "warning",
        "INFO": "good"
    }.get(severity, "warning")

    message = {
        "attachments": [{
            "color": color,
            "title": f"🚨 Security Alert: {rule}",
            "fields": [
                {"title": "Severity", "value": severity, "short": True},
                {"title": "Container", "value": container or "N/A", "short": True},
                {"title": "Namespace", "value": namespace or "N/A", "short": True},
                {"title": "Pod", "value": pod or "N/A", "short": True},
                {"title": "Time", "value": datetime.utcnow().isoformat(), "short": True}
            ]
        }]
    }

    requests.post(SLACK_WEBHOOK, json=message)

def page_oncall(rule: str, container: str, namespace: str, pod: str):
    """Page on-call engineer via PagerDuty"""
    event = {
        "routing_key": PAGERDUTY_KEY,
        "event_action": "trigger",
        "payload": {
            "summary": f"Critical Security Alert: {rule}",
            "severity": "critical",
            "source": f"{namespace}/{pod}",
            "custom_details": {
                "rule": rule,
                "container": container,
                "namespace": namespace,
                "pod": pod
            }
        }
    }

    requests.post("https://events.pagerduty.com/v2/enqueue", json=event)

def quarantine_pod(namespace: str, pod_name: str):
    """Quarantine pod by applying restrictive NetworkPolicy"""
    print(f"Quarantining pod {namespace}/{pod_name}")

    # Label pod for quarantine
    v1.patch_namespaced_pod(
        name=pod_name,
        namespace=namespace,
        body={"metadata": {"labels": {"security.genops.ai/quarantined": "true"}}}
    )

    # Apply quarantine NetworkPolicy
    quarantine_policy = client.V1NetworkPolicy(
        metadata=client.V1ObjectMeta(
            name=f"quarantine-{pod_name}",
            namespace=namespace
        ),
        spec=client.V1NetworkPolicySpec(
            pod_selector=client.V1LabelSelector(
                match_labels={"security.genops.ai/quarantined": "true"}
            ),
            policy_types=["Ingress", "Egress"],
            ingress=[],  # Deny all ingress
            egress=[]    # Deny all egress
        )
    )

    networking_v1 = client.NetworkingV1Api()
    networking_v1.create_namespaced_network_policy(
        namespace=namespace,
        body=quarantine_policy
    )

def scale_down_deployment(namespace: str, deployment_name: str):
    """Scale down deployment to 0 replicas"""
    print(f"Scaling down deployment {namespace}/{deployment_name}")

    apps_v1.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=namespace,
        body={"spec": {"replicas": 0}}
    )

def log_security_event(alert: dict):
    """Log security event to audit system"""
    # Implementation would send to central audit logging system
    print(f"Logging security event: {alert}")

if __name__ == "__main__":
    # Example: Listen for Falco alerts via webhook
    from flask import Flask, request

    app = Flask(__name__)

    @app.route('/falco', methods=['POST'])
    def falco_webhook():
        alert = request.get_json()
        handle_security_alert(alert)
        return {"status": "processed"}, 200

    app.run(host='0.0.0.0', port=8080)
```

---

## Security Best Practices

### Production Security Checklist

**✅ Authentication & Authorization:**
- [x] Enable RBAC and remove default cluster-admin bindings
- [x] Use dedicated service accounts for each application
- [x] Implement least-privilege access policies
- [x] Integrate with corporate identity provider (OIDC/SAML)
- [x] Regular access review and audit (quarterly minimum)
- [x] Implement admission controllers for policy enforcement

**✅ Network Security:**
- [x] Implement default-deny NetworkPolicies in all namespaces
- [x] Use service mesh for mTLS between services
- [x] Restrict egress to only required external services
- [x] Deploy API gateway with rate limiting
- [x] Enable DDoS protection on ingress
- [x] Implement network segmentation by workload sensitivity

**✅ Container Security:**
- [x] Enforce Pod Security Standards (Restricted profile)
- [x] Scan all images for vulnerabilities before deployment
- [x] Use minimal base images (distroless, alpine, scratch)
- [x] Run containers as non-root with read-only root filesystem
- [x] Implement runtime security monitoring (Falco)
- [x] Deploy AppArmor/Seccomp profiles for critical workloads

**✅ Secret Management:**
- [x] Never commit secrets to Git (use git-secrets/gitleaks)
- [x] Use external secret management (Vault, AWS Secrets Manager, Azure Key Vault)
- [x] Enable encryption at rest for etcd
- [x] Rotate secrets regularly (90 days maximum)
- [x] Audit secret access and implement alerts
- [x] Use short-lived credentials where possible

**✅ Data Protection:**
- [x] Enable TLS for all service-to-service communication
- [x] Encrypt persistent volumes at rest
- [x] Implement PII data handling policies
- [x] Regular backup and disaster recovery testing
- [x] Data retention and deletion policies
- [x] Compliance with data residency requirements

**✅ Audit & Compliance:**
- [x] Enable Kubernetes audit logging
- [x] Implement policy enforcement with OPA/Gatekeeper
- [x] Regular compliance scanning (CIS benchmarks)
- [x] Security event monitoring and alerting
- [x] Incident response plan and runbooks
- [x] Regular security drills and tabletop exercises

**✅ Supply Chain Security:**
- [x] Sign and verify container images (Cosign)
- [x] Generate and track SBOMs for all images
- [x] Secure build pipeline (signed commits, protected branches)
- [x] Use private artifact repositories
- [x] Dependency scanning and updates (Dependabot)
- [x] Base image vulnerability management

---

## Troubleshooting

### Issue 1: Pod Failing Security Context Constraints

**Symptoms:**
```
Error creating: pods "genops-ai-" is forbidden: violates PodSecurity "restricted:latest"
```

**Diagnosis:**
```bash
# Check namespace Pod Security Standard
kubectl get namespace genops -o jsonpath='{.metadata.labels}'

# Check pod security context
kubectl get pod POD_NAME -n genops -o jsonpath='{.spec.securityContext}'
```

**Solutions:**

**Option 1: Fix SecurityContext:**
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000
  seccompProfile:
    type: RuntimeDefault
```

**Option 2: Relax Namespace Policy (not recommended):**
```bash
kubectl label namespace genops pod-security.kubernetes.io/enforce=baseline --overwrite
```

---

### Issue 2: NetworkPolicy Blocking Legitimate Traffic

**Symptoms:**
- Application cannot reach dependencies
- `Connection refused` or `Timeout` errors
- Works without NetworkPolicy

**Diagnosis:**
```bash
# Check applied NetworkPolicies
kubectl get networkpolicy -n genops

# Describe specific policy
kubectl describe networkpolicy POLICY_NAME -n genops

# Test connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot -n genops -- bash
# Inside pod:
curl -v http://service-name:port
```

**Solution:**
```yaml
# Add egress rule for missing service
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-egress-to-service
  namespace: genops
spec:
  podSelector:
    matchLabels:
      app: genops-ai
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: target-service
    ports:
    - protocol: TCP
      port: 8080
```

---

### Issue 3: ExternalSecret Not Syncing

**Symptoms:**
- Secret not created in namespace
- `SecretSync failed` events

**Diagnosis:**
```bash
# Check ExternalSecret status
kubectl get externalsecret -n genops
kubectl describe externalsecret SECRET_NAME -n genops

# Check SecretStore status
kubectl get secretstore -n genops
kubectl describe secretstore STORE_NAME -n genops

# Check operator logs
kubectl logs -n external-secrets-system deployment/external-secrets
```

**Common Issues:**

**Issue 3a: IAM Permissions:**
```bash
# Verify ServiceAccount has correct IAM role
kubectl get serviceaccount genops-ai -n genops -o yaml | grep eks.amazonaws.com/role-arn

# Test IAM permissions
aws sts assume-role-with-web-identity \
  --role-arn ROLE_ARN \
  --role-session-name test \
  --web-identity-token $(kubectl create token genops-ai -n genops)
```

**Issue 3b: Secret Path Incorrect:**
```yaml
# Verify secret path in AWS Secrets Manager
data:
- secretKey: api_key
  remoteRef:
    key: prod/genops/openai-api-key  # Must match exact path
```

---

### Issue 4: Admission Webhook Failing

**Symptoms:**
- Deployments rejected with webhook timeout
- `Internal error occurred: failed calling webhook`

**Diagnosis:**
```bash
# Check webhook configuration
kubectl get validatingwebhookconfiguration genops-governance-validator

# Check webhook service
kubectl get svc -n genops genops-webhook
kubectl get endpoints -n genops genops-webhook

# Check webhook pod logs
kubectl logs -n genops -l app=genops-webhook
```

**Solutions:**

**Option 1: Fix Webhook Certificate:**
```bash
# Regenerate webhook certificates
./scripts/generate-webhook-certs.sh

# Update secret
kubectl create secret tls genops-webhook-certs \
  --cert=webhook.crt \
  --key=webhook.key \
  -n genops \
  --dry-run=client -o yaml | kubectl apply -f -
```

**Option 2: Temporarily Disable Webhook:**
```bash
# Change failurePolicy to Ignore
kubectl patch validatingwebhookconfiguration genops-governance-validator \
  --type='json' -p='[{"op": "replace", "path": "/webhooks/0/failurePolicy", "value":"Ignore"}]'
```

---

### Issue 5: Falco High CPU Usage

**Symptoms:**
- Falco DaemonSet consuming excessive CPU
- Node performance degradation

**Diagnosis:**
```bash
# Check Falco resource usage
kubectl top pod -n falco-system

# Check Falco event rate
kubectl logs -n falco-system -l app=falco | grep -c "rule="
```

**Solutions:**

**Option 1: Tune Falco Rules:**
```yaml
# Reduce rule scope
- rule: Sensitive File Access
  condition: >
    open_read and
    container.image.repository contains "genops-ai" and
    fd.name in (sensitive_files) and
    not proc.name in (allowed_processes)  # Add exceptions
```

**Option 2: Increase Resource Limits:**
```yaml
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi
```

**Option 3: Reduce Event Buffer Size:**
```yaml
# Falco config
syscall_event_drops:
  threshold: 0.1  # Increase threshold
  actions:
    - log
```

---

### Issue 6: OPA Gatekeeper Performance Issues

**Symptoms:**
- Slow deployment creation
- Admission webhook timeout errors

**Diagnosis:**
```bash
# Check Gatekeeper audit logs
kubectl logs -n gatekeeper-system -l control-plane=audit-controller

# Check constraint status
kubectl get constraints

# Check webhook latency
kubectl get validatingwebhookconfiguration gatekeeper-validating-webhook-configuration -o yaml
```

**Solutions:**

**Option 1: Optimize Rego Policies:**
```rego
# Use efficient data structures
package k8srequiredlabels

# Cache expensive operations
cached_labels[label] {
  label := input.parameters.labels[_]
}

violation[{"msg": msg}] {
  provided := {l | input.review.object.metadata.labels[l]}
  missing := cached_labels - provided  # Use cached result
  count(missing) > 0
  msg := sprintf("Missing: %v", [missing])
}
```

**Option 2: Increase Replica Count:**
```bash
kubectl scale deployment -n gatekeeper-system gatekeeper-controller-manager --replicas=3
```

---

### Issue 7: Certificate Renewal Failures

**Symptoms:**
- cert-manager failing to renew certificates
- TLS errors on ingress

**Diagnosis:**
```bash
# Check certificate status
kubectl get certificate -n genops
kubectl describe certificate genops-ai-tls -n genops

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Check certificate expiry
kubectl get secret genops-ai-tls -n genops -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -enddate
```

**Solutions:**

**Option 1: Manually Trigger Renewal:**
```bash
# Delete secret to trigger renewal
kubectl delete secret genops-ai-tls -n genops

# cert-manager will recreate it
```

**Option 2: Fix ACME Challenge:**
```bash
# Check challenge status
kubectl get challenge -n genops

# Check ingress for .well-known/acme-challenge
kubectl get ingress -n genops
```

---

## Next Steps

Ready to secure your GenOps AI deployment? Start with:

1. **Security Audit** - Run CIS Kubernetes Benchmark scan with kube-bench
2. **Apply Pod Security Standards** - Enforce restricted profile on all namespaces
3. **Implement RBAC** - Configure least-privilege service accounts
4. **Deploy NetworkPolicies** - Default deny with selective allow rules
5. **Set Up Secret Management** - Integrate External Secrets Operator
6. **Enable Audit Logging** - Configure comprehensive audit policies
7. **Deploy Runtime Security** - Install Falco with custom rules
8. **Continuous Monitoring** - Implement security dashboards and alerts

**Related Documentation:**
- [Kubernetes Best Practices](kubernetes-best-practices.md) - Security best practices
- [Multi-Tenant Architecture](kubernetes-multi-tenant.md) - Tenant isolation patterns
- [Observability](kubernetes-observability.md) - Security monitoring integration

Return to [Kubernetes Getting Started](kubernetes-getting-started.md) for the complete deployment overview.

---

## Security Resources

**Kubernetes Official:**
- [Security Best Practices](https://kubernetes.io/docs/concepts/security/security-best-practices/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [RBAC Documentation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

**Industry Standards:**
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Kubernetes Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Kubernetes_Security_Cheat_Sheet.html)

**Security Tools:**
- [Falco](https://falco.org/) - Runtime security monitoring
- [OPA Gatekeeper](https://open-policy-agent.github.io/gatekeeper/) - Policy enforcement
- [Trivy](https://trivy.dev/) - Vulnerability scanning
- [Cosign](https://github.com/sigstore/cosign) - Container signing
- [cert-manager](https://cert-manager.io/) - Certificate management
- [External Secrets Operator](https://external-secrets.io/) - Secret management

---

## Support

- **Documentation:** [GenOps AI Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community:** [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Security Advisories:** [Security Policy](https://github.com/KoshiHQ/GenOps-AI/security/policy)
