# Service Mesh Integration for GenOps AI

> **Status:** 📋 Documentation in progress
> **Last Updated:** 2026-01-18

Enhance GenOps AI with service mesh capabilities for advanced traffic management, security, and observability.

---

## Overview

Service meshes provide infrastructure-level capabilities that complement GenOps AI governance:
- **Distributed tracing** with automatic span injection across service boundaries
- **mTLS encryption** for zero-trust security between AI services
- **Traffic management** including retry logic, timeouts, and circuit breakers
- **Policy enforcement** at the network layer with governance context propagation

GenOps AI integrates with popular service meshes to provide unified governance tracking across your entire AI workload mesh.

---

## Quick Reference

### Supported Service Meshes

**Istio:**
- Most feature-rich service mesh
- Native OpenTelemetry integration
- Advanced traffic routing and policy enforcement

**Linkerd:**
- Lightweight and performant
- Automatic mTLS with minimal overhead
- Simple setup and operation

**Consul Connect:**
- Multi-cloud service mesh
- Native Consul integration for service discovery
- Cross-datacenter support

### Key Benefits for AI Workloads

**Automatic Observability:**
```yaml
# Service mesh automatically adds tracing headers
# GenOps AI propagates governance context through these headers
apiVersion: v1
kind: Service
metadata:
  name: ai-inference
  annotations:
    # Istio automatically instruments this service
    sidecar.istio.io/inject: "true"
```

**Traffic Control:**
```yaml
# Canary deployment with traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-inference
spec:
  hosts:
  - ai-inference
  http:
  - match:
    - headers:
        customer-tier:
          exact: "premium"
    route:
    - destination:
        host: ai-inference
        subset: v2  # New model version
      weight: 100
  - route:
    - destination:
        host: ai-inference
        subset: v1
      weight: 90
    - destination:
        host: ai-inference
        subset: v2
      weight: 10
```

---

## Table of Contents

### Planned Documentation Sections

1. **Service Mesh Fundamentals**
   - Architecture overview and components
   - Control plane vs data plane
   - Sidecar proxy pattern
   - When to use a service mesh vs direct Kubernetes services

2. **Istio Integration**
   - Installation and configuration
   - VirtualService and DestinationRule patterns
   - Telemetry configuration for GenOps AI
   - Authorization policies with governance attributes

3. **Linkerd Integration**
   - Lightweight installation for AI workloads
   - Automatic mTLS configuration
   - Traffic split for A/B testing
   - Observability integration

4. **Traffic Management Patterns**
   - Circuit breakers for AI service resilience
   - Retry policies with exponential backoff
   - Timeout configuration for LLM API calls
   - Rate limiting per customer/team

5. **Security Enhancements**
   - mTLS for service-to-service encryption
   - Authorization policies based on governance attributes
   - JWT authentication and propagation
   - Network segmentation and policy enforcement

6. **Observability Deep-Dive**
   - Distributed tracing across service mesh
   - Governance context propagation through mesh headers
   - Service-level metrics and SLIs
   - Integration with Prometheus and Grafana

7. **Advanced Patterns**
   - Multi-cluster service mesh for high availability
   - Egress gateway for controlled external API access
   - Fault injection for chaos engineering
   - Progressive delivery with Flagger

---

## Related Documentation

**Kubernetes Guides:**
- [Kubernetes Getting Started](kubernetes-getting-started.md)
- [Advanced Observability](kubernetes-observability.md)
- [Security Hardening](kubernetes-security.md)

**Integration Guides:**
- [API Gateway Integration](kubernetes-api-gateway.md)
- [Multi-Cloud Deployment](kubernetes-multi-cloud.md)

---

## Quick Examples

### Example 1: Istio Installation with GenOps AI

```bash
# Install Istio with OpenTelemetry integration
istioctl install --set profile=demo \
  --set meshConfig.defaultConfig.tracing.zipkin.address=jaeger-collector.observability:9411 \
  --set meshConfig.enableTracing=true

# Label namespace for automatic sidecar injection
kubectl label namespace genops istio-injection=enabled

# Deploy GenOps AI (Istio automatically injects sidecar)
helm upgrade --install genops-ai ./helm-charts/genops-ai \
  --namespace genops \
  --values helm-values/istio-enabled.yaml
```

### Example 2: Traffic Management with Governance Context

```yaml
# Route traffic based on governance attributes
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-inference-routing
  namespace: genops
spec:
  hosts:
  - ai-inference.genops.svc.cluster.local
  http:
  # Route high-value customers to premium model
  - match:
    - headers:
        x-genops-customer-tier:
          exact: "premium"
    route:
    - destination:
        host: ai-inference
        subset: claude-opus
  # Route standard customers to efficient model
  - match:
    - headers:
        x-genops-customer-tier:
          exact: "standard"
    route:
    - destination:
        host: ai-inference
        subset: claude-haiku
  # Default route
  - route:
    - destination:
        host: ai-inference
        subset: claude-sonnet
```

### Example 3: Circuit Breaker for LLM API Calls

```yaml
# Prevent cascading failures when LLM provider is slow
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ai-inference-circuit-breaker
  namespace: genops
spec:
  host: ai-inference.genops.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 25
```

### Example 4: mTLS and Authorization Policies

```yaml
# Enforce mTLS for all services
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: genops
spec:
  mtls:
    mode: STRICT

---
# Authorization policy based on governance attributes
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: ai-inference-authz
  namespace: genops
spec:
  selector:
    matchLabels:
      app: ai-inference
  action: ALLOW
  rules:
  # Allow requests with valid team and project headers
  - from:
    - source:
        principals: ["cluster.local/ns/genops/sa/genops-client"]
    when:
    - key: request.headers[x-genops-team]
      values: ["ml-platform", "product-team"]
    - key: request.headers[x-genops-project]
      notValues: [""] # Must have a project
```

---

## Service Mesh Fundamentals

### Architecture Overview

Service meshes provide infrastructure-level capabilities for microservices communication through a dedicated control and data plane architecture:

**Control Plane:**
- **Configuration Management**: Central policy distribution
- **Service Discovery**: Dynamic service registry
- **Certificate Authority**: Automated certificate issuance and rotation
- **Telemetry Collection**: Metrics, logs, and traces aggregation
- **Policy Enforcement**: Authorization and rate limiting rules

**Data Plane:**
- **Sidecar Proxies**: Envoy proxies injected alongside each pod
- **Traffic Interception**: Automatic request/response capture
- **Policy Execution**: Runtime enforcement of control plane policies
- **Telemetry Export**: Metrics and traces sent to control plane

**Sidecar Proxy Pattern:**
```
┌─────────────────────────────────────┐
│           Application Pod           │
│  ┌──────────────┐  ┌─────────────┐ │
│  │              │  │   Envoy     │ │
│  │  App Container│◄─┤   Sidecar   │ │
│  │              │  │   Proxy     │ │
│  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────┘
         ▲                    │
         │                    ▼
    Inbound Traffic     Outbound Traffic
```

### When to Use Service Mesh vs Alternatives

**Use Service Mesh When:**
- **Many Microservices**: 10+ services with complex communication patterns
- **Multi-Team Ownership**: Different teams managing different services
- **Security Requirements**: mTLS encryption and zero-trust networking
- **Advanced Traffic Management**: A/B testing, canary releases, circuit breakers
- **Detailed Observability**: Per-request tracing across all services
- **Polyglot Environment**: Services written in different languages

**Avoid Service Mesh When:**
- **Monolithic Application**: Single application with limited service-to-service calls
- **Small Scale**: < 5 microservices
- **Performance Critical**: Latency overhead (5-10ms) is unacceptable
- **Simple Requirements**: Basic load balancing is sufficient
- **Limited Resources**: No team capacity for mesh operation

**Alternatives to Service Mesh:**

| Requirement | Alternative Solution |
|-------------|---------------------|
| Load balancing | Kubernetes Service + Ingress |
| mTLS | cert-manager + application-level TLS |
| Observability | OpenTelemetry SDK instrumentation |
| Rate limiting | API Gateway (Kong, NGINX) |
| Circuit breaking | Application libraries (Hystrix, resilience4j) |

### Performance Overhead Analysis

**Latency Impact:**
```
Without Service Mesh:
  Service A → Service B: ~2ms

With Service Mesh (Istio):
  Service A → Envoy Sidecar → Envoy Sidecar → Service B: ~7-12ms
  Additional latency: 5-10ms per hop
```

**Resource Overhead (per pod):**

| Service Mesh | CPU (sidecar) | Memory (sidecar) | Control Plane CPU | Control Plane Memory |
|--------------|---------------|------------------|-------------------|---------------------|
| **Istio** | 100-500m | 50-100Mi | 500m-2 | 1-4Gi |
| **Linkerd** | 20-100m | 20-50Mi | 100-500m | 500Mi-2Gi |
| **Consul Connect** | 50-200m | 30-80Mi | 500m-1 | 512Mi-2Gi |

**Throughput Impact:**
- **Without Mesh**: 50,000 RPS baseline
- **With Istio**: 40,000 RPS (-20% throughput)
- **With Linkerd**: 45,000 RPS (-10% throughput)

**When Performance Overhead is Acceptable:**
- Request latency > 100ms (5-10ms overhead is < 10%)
- Network I/O bound workloads
- Security/observability benefits outweigh performance cost
- Can scale horizontally to compensate

### Service Mesh Comparison Matrix

| Feature | Istio | Linkerd | Consul Connect |
|---------|-------|---------|----------------|
| **Complexity** | High | Low | Medium |
| **Performance** | Good | Excellent | Good |
| **Features** | Most comprehensive | Focused/minimal | Service discovery + mesh |
| **Multi-Cluster** | Yes (advanced) | Yes | Yes (native) |
| **Protocol Support** | HTTP/gRPC/TCP | HTTP/gRPC/TCP | HTTP/gRPC/TCP |
| **Observability** | Excellent (built-in) | Good (Prometheus) | Good (built-in) |
| **mTLS** | Automatic | Automatic (faster) | Automatic |
| **Traffic Management** | Advanced (VirtualService) | Basic (TrafficSplit) | Medium |
| **Community** | Large (CNCF) | Medium (CNCF) | Large (HashiCorp) |
| **Best For** | Large enterprises | Simplicity seekers | Consul users |

---

## Istio Integration Deep-Dive

### Istio Installation and Configuration

**Install Istio with istioctl:**
```bash
# Download Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.20.0
export PATH=$PWD/bin:$PATH

# Install Istio with OpenTelemetry integration
istioctl install --set profile=demo \
  --set meshConfig.enableTracing=true \
  --set meshConfig.defaultConfig.tracing.zipkin.address=jaeger-collector.observability:9411 \
  --set meshConfig.accessLogFile=/dev/stdout \
  --set meshConfig.defaultConfig.holdApplicationUntilProxyStarts=true

# Enable sidecar injection for namespace
kubectl label namespace genops istio-injection=enabled

# Verify installation
kubectl get pods -n istio-system
istioctl verify-install
```

**Configure Istio for GenOps Telemetry:**
```yaml
# IstioOperator for custom telemetry
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: genops-istio-config
  namespace: istio-system
spec:
  meshConfig:
    # Enable distributed tracing
    enableTracing: true
    defaultConfig:
      tracing:
        sampling: 100.0  # 100% sampling for testing, reduce in production
        zipkin:
          address: jaeger-collector.observability:9411

    # Enable access logs with governance context
    accessLogFile: /dev/stdout
    accessLogFormat: |
      [%START_TIME%] "%REQ(:METHOD)% %REQ(X-ENVOY-ORIGINAL-PATH?:PATH)% %PROTOCOL%"
      %RESPONSE_CODE% %RESPONSE_FLAGS% %BYTES_RECEIVED% %BYTES_SENT% %DURATION%
      "%REQ(X-FORWARDED-FOR)%" "%REQ(USER-AGENT)%" "%REQ(X-REQUEST-ID)%"
      "%REQ(X-GENOPS-TEAM)%" "%REQ(X-GENOPS-CUSTOMER-ID)%" "%REQ(X-GENOPS-PROJECT)%"

    # Custom headers to propagate
    defaultConfig:
      proxyHeaders:
        requestId:
          name: X-Request-ID
        attemptCount:
          name: X-Envoy-Attempt-Count
        forwardedClientCert:
          name: X-Forwarded-Client-Cert
```

### VirtualService Traffic Routing

**Basic VirtualService for GenOps AI:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-vs
  namespace: genops
spec:
  hosts:
  - genops-ai-inference.genops.svc.cluster.local
  http:
  - match:
    - headers:
        x-genops-customer-tier:
          exact: "premium"
    route:
    - destination:
        host: genops-ai-inference
        subset: high-performance
        port:
          number: 8080
      weight: 100

  - match:
    - headers:
        x-genops-customer-tier:
          exact: "standard"
    route:
    - destination:
        host: genops-ai-inference
        subset: standard
        port:
          number: 8080
      weight: 100

  - route:
    - destination:
        host: genops-ai-inference
        subset: basic
        port:
          number: 8080
      weight: 100
```

**Canary Deployment with Traffic Splitting:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-canary
  namespace: genops
spec:
  hosts:
  - genops-ai-inference
  http:
  # Route 10% of traffic to canary version
  - match:
    - headers:
        x-canary-user:
          exact: "true"
    route:
    - destination:
        host: genops-ai-inference
        subset: v2-canary
      weight: 100

  # Regular traffic split: 90% v1, 10% v2
  - route:
    - destination:
        host: genops-ai-inference
        subset: v1-stable
      weight: 90
    - destination:
        host: genops-ai-inference
        subset: v2-canary
      weight: 10

    # Add canary header to track in telemetry
    headers:
      response:
        set:
          x-canary-version: v2
```

**A/B Testing Based on Governance Context:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-ab-test
  namespace: genops
spec:
  hosts:
  - genops-ai-inference
  http:
  # Model A: GPT-4 for specific teams
  - match:
    - headers:
        x-genops-team:
          regex: "(ml-research|data-science)"
    route:
    - destination:
        host: genops-ai-inference
        subset: model-gpt4

  # Model B: Claude for everyone else
  - route:
    - destination:
        host: genops-ai-inference
        subset: model-claude
```

### DestinationRule for Load Balancing

**Connection Pool and Load Balancer Settings:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: genops-ai-dr
  namespace: genops
spec:
  host: genops-ai-inference.genops.svc.cluster.local

  # Define subsets for different versions
  subsets:
  - name: v1-stable
    labels:
      version: v1
    trafficPolicy:
      loadBalancer:
        consistentHash:
          httpHeaderName: x-genops-customer-id  # Session affinity per customer

  - name: v2-canary
    labels:
      version: v2
    trafficPolicy:
      loadBalancer:
        simple: LEAST_REQUEST  # Load balance by least requests

  - name: high-performance
    labels:
      tier: premium
    trafficPolicy:
      connectionPool:
        http:
          http1MaxPendingRequests: 100
          http2MaxRequests: 1000
          maxRequestsPerConnection: 10

  # Default traffic policy for all subsets
  trafficPolicy:
    # Connection pooling
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100

    # Outlier detection (circuit breaker)
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 25
```

### Authorization Policies with Governance Attributes

**Service-Level Authorization:**
```yaml
# Deny all by default
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: genops
spec:
  {}  # Empty spec denies all requests

---
# Allow specific services with governance validation
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: genops-ai-authz
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference

  action: ALLOW

  rules:
  # Allow requests from API gateway with valid governance context
  - from:
    - source:
        namespaces: ["api-gateway"]
    to:
    - operation:
        methods: ["POST", "GET"]
        paths: ["/ai/v1/*"]
    when:
    # Require governance headers
    - key: request.headers[x-genops-team]
      notValues: [""]
    - key: request.headers[x-genops-customer-id]
      notValues: [""]

  # Allow internal service-to-service calls
  - from:
    - source:
        namespaces: ["genops"]
        principals: ["cluster.local/ns/genops/sa/*"]
```

**Team-Based Access Control:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: team-isolation
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference

  action: ALLOW

  rules:
  # ml-platform team can access all models
  - when:
    - key: request.headers[x-genops-team]
      values: ["ml-platform"]

  # product-team can only access standard models
  - when:
    - key: request.headers[x-genops-team]
      values: ["product-team"]
    - key: request.headers[x-model-type]
      values: ["standard", "basic"]
      notValues: ["premium"]
```

### Telemetry Configuration for GenOps

**Custom Metrics for Governance:**
```yaml
# Istio Telemetry v2 configuration
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: genops-telemetry
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference

  # Custom metrics
  metrics:
  - providers:
    - name: prometheus
    dimensions:
      genops_team: request.headers["x-genops-team"]
      genops_customer_id: request.headers["x-genops-customer-id"]
      genops_project: request.headers["x-genops-project"]
      genops_tier: request.headers["x-genops-customer-tier"]
    overrides:
    - match:
        metric: REQUEST_COUNT
      tagOverrides:
        genops_team:
          value: request.headers["x-genops-team"]
        genops_customer_id:
          value: request.headers["x-genops-customer-id"]

  # Distributed tracing
  tracing:
  - providers:
    - name: zipkin
    randomSamplingPercentage: 1.0  # 1% sampling in production
    customTags:
      genops.team:
        header:
          name: x-genops-team
      genops.customer_id:
        header:
          name: x-genops-customer-id
      genops.project:
        header:
          name: x-genops-project

  # Access logging
  accessLogging:
  - providers:
    - name: envoy
```

---

## Linkerd Integration

### Linkerd Installation (Lightweight)

**Install Linkerd CLI and Control Plane:**
```bash
# Install Linkerd CLI
curl -fsL https://run.linkerd.io/install | sh
export PATH=$PATH:$HOME/.linkerd2/bin

# Validate cluster
linkerd check --pre

# Install Linkerd CRDs
linkerd install --crds | kubectl apply -f -

# Install Linkerd control plane
linkerd install \
  --set proxyInit.runAsRoot=false \
  --set controllerLogLevel=info \
  | kubectl apply -f -

# Verify installation
linkerd check

# Install viz extension for observability
linkerd viz install | kubectl apply -f -
```

**Inject Linkerd Proxy into GenOps Namespace:**
```bash
# Annotate namespace for automatic injection
kubectl annotate namespace genops linkerd.io/inject=enabled

# Or inject manually into deployment
kubectl get deployment genops-ai-inference -n genops -o yaml \
  | linkerd inject - \
  | kubectl apply -f -

# Verify injection
linkerd -n genops check --proxy
```

### Automatic mTLS Configuration

**Linkerd automatically enables mTLS for all meshed services. No additional configuration required!**

**Verify mTLS Status:**
```bash
# Check mTLS for services in namespace
linkerd -n genops edges deployment

# View traffic split and mTLS status
linkerd -n genops stat deployments

# Detailed traffic metrics with mTLS
linkerd -n genops tap deployment/genops-ai-inference
```

**mTLS Policy (if you need to customize):**
```yaml
# MeshTLSAuthentication for stricter validation
apiVersion: policy.linkerd.io/v1alpha1
kind: MeshTLSAuthentication
metadata:
  name: genops-mtls
  namespace: genops
spec:
  identities:
  - "genops-ai-inference.genops.serviceaccount.identity.linkerd.cluster.local"
  - "genops-api-gateway.api-gateway.serviceaccount.identity.linkerd.cluster.local"
```

### Traffic Split for A/B Testing

**Use SMI TrafficSplit for Canary:**
```yaml
# Deploy stable and canary versions
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-v1
  namespace: genops
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-ai
      version: v1
  template:
    metadata:
      labels:
        app: genops-ai
        version: v1
    spec:
      containers:
      - name: inference
        image: genops-ai:v1
        ports:
        - containerPort: 8080

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-v2
  namespace: genops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: genops-ai
      version: v2
  template:
    metadata:
      labels:
        app: genops-ai
        version: v2
    spec:
      containers:
      - name: inference
        image: genops-ai:v2
        ports:
        - containerPort: 8080

---
# Services for each version
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-v1
  namespace: genops
spec:
  selector:
    app: genops-ai
    version: v1
  ports:
  - port: 8080

---
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-v2
  namespace: genops
spec:
  selector:
    app: genops-ai
    version: v2
  ports:
  - port: 8080

---
# Root service
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-inference
  namespace: genops
spec:
  selector:
    app: genops-ai
  ports:
  - port: 8080

---
# TrafficSplit: 90% v1, 10% v2
apiVersion: split.smi-spec.io/v1alpha2
kind: TrafficSplit
metadata:
  name: genops-ai-split
  namespace: genops
spec:
  service: genops-ai-inference
  backends:
  - service: genops-ai-v1
    weight: 900  # 90%
  - service: genops-ai-v2
    weight: 100  # 10%
```

### ServiceProfile for Per-Route Metrics

**Define ServiceProfile for Detailed Metrics:**
```yaml
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: genops-ai-inference.genops.svc.cluster.local
  namespace: genops
spec:
  routes:
  # Define routes for granular metrics
  - name: POST /ai/v1/inference
    condition:
      method: POST
      pathRegex: /ai/v1/inference
    timeout: 30s
    retryBudget:
      retryRatio: 0.2
      minRetriesPerSecond: 10
      ttl: 10s

  - name: POST /ai/v1/embeddings
    condition:
      method: POST
      pathRegex: /ai/v1/embeddings
    timeout: 10s
    isRetryable: true

  - name: GET /health
    condition:
      method: GET
      pathRegex: /health
    timeout: 5s
    isRetryable: true

  # Default route for unmatched requests
  - name: default
    timeout: 30s
```

**View Per-Route Metrics:**
```bash
# Real-time per-route metrics
linkerd -n genops routes deployment/genops-ai-inference

# Success rate by route
linkerd -n genops routes deployment/genops-ai-inference --to service/genops-ai-inference

# Aggregate stats
linkerd -n genops stat --from deployment/api-gateway deployment/genops-ai-inference
```

### Integration with Prometheus/Grafana

**Linkerd Metrics in Prometheus:**
```yaml
# ServiceMonitor for Linkerd metrics
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: linkerd-proxy
  namespace: monitoring
spec:
  selector:
    matchLabels:
      linkerd.io/control-plane-component: proxy
  endpoints:
  - port: linkerd-admin
    interval: 30s
    path: /metrics
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_label_genops_team]
      targetLabel: genops_team
    - sourceLabels: [__meta_kubernetes_pod_label_genops_customer_id]
      targetLabel: genops_customer_id
```

**Import Linkerd Grafana Dashboards:**
```bash
# Linkerd provides pre-built dashboards
linkerd viz dashboard &

# Export Linkerd dashboards for Grafana
kubectl -n linkerd-viz get configmap linkerd-grafana-config -o jsonpath='{.data}' > linkerd-dashboards.json
```

---

## Traffic Management Patterns

### Circuit Breakers for LLM API Resilience

**Istio Circuit Breaker Configuration:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: genops-ai-circuit-breaker
  namespace: genops
spec:
  host: genops-ai-inference.genops.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2

    outlierDetection:
      # Eject pod after 5 consecutive errors
      consecutiveErrors: 5

      # Check every 30 seconds
      interval: 30s

      # Pod ejected for 30 seconds minimum
      baseEjectionTime: 30s

      # Maximum 50% of pods can be ejected
      maxEjectionPercent: 50

      # At least 25% must remain healthy
      minHealthPercent: 25

      # Split external/local origin errors
      splitExternalLocalOriginErrors: true
      consecutiveLocalOriginFailures: 5
```

**Test Circuit Breaker with Fault Injection:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-fault-test
  namespace: genops
spec:
  hosts:
  - genops-ai-inference
  http:
  - fault:
      abort:
        percentage:
          value: 50  # 50% of requests fail
        httpStatus: 500
    route:
    - destination:
        host: genops-ai-inference
```

### Retry Policies with Exponential Backoff

**Istio Retry Configuration:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-retry
  namespace: genops
spec:
  hosts:
  - genops-ai-inference
  http:
  - route:
    - destination:
        host: genops-ai-inference
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream

      # Exponential backoff
      retryRemoteLocalities: true
```

**Linkerd Automatic Retry:**
```yaml
# ServiceProfile with retry budget
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: genops-ai-inference.genops.svc.cluster.local
  namespace: genops
spec:
  routes:
  - name: POST /ai/v1/inference
    condition:
      method: POST
      pathRegex: /ai/v1/inference
    isRetryable: true
    timeout: 30s
    retryBudget:
      # Allow 20% retry ratio
      retryRatio: 0.2
      # Minimum 10 retries per second
      minRetriesPerSecond: 10
      # Retry budget TTL
      ttl: 10s
```

### Timeout Configuration for Long-Running Requests

**Per-Route Timeouts:**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-timeouts
  namespace: genops
spec:
  hosts:
  - genops-ai-inference
  http:
  # Long timeout for inference requests
  - match:
    - uri:
        prefix: /ai/v1/inference
    route:
    - destination:
        host: genops-ai-inference
    timeout: 60s  # 60 seconds for inference

  # Short timeout for health checks
  - match:
    - uri:
        exact: /health
    route:
    - destination:
        host: genops-ai-inference
    timeout: 5s

  # Default timeout
  - route:
    - destination:
        host: genops-ai-inference
    timeout: 30s
```

### Rate Limiting at Service Mesh Level

**Envoy Rate Limiting (Istio):**
```yaml
# EnvoyFilter for local rate limiting
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: genops-ratelimit
  namespace: genops
spec:
  workloadSelector:
    labels:
      app: genops-ai-inference
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
      listener:
        filterChain:
          filter:
            name: envoy.filters.network.http_connection_manager
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.local_ratelimit
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
          stat_prefix: http_local_rate_limiter
          token_bucket:
            max_tokens: 100
            tokens_per_fill: 100
            fill_interval: 60s  # 100 requests per minute
          filter_enabled:
            runtime_key: local_rate_limit_enabled
            default_value:
              numerator: 100
              denominator: HUNDRED
          filter_enforced:
            runtime_key: local_rate_limit_enforced
            default_value:
              numerator: 100
              denominator: HUNDRED
          response_headers_to_add:
          - append: false
            header:
              key: x-local-rate-limit
              value: 'true'
```

### Fault Injection for Chaos Testing

**Istio Fault Injection:**
```yaml
# Inject delays and errors for testing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-ai-chaos
  namespace: genops
spec:
  hosts:
  - genops-ai-inference
  http:
  # Inject 500ms delay for 10% of requests
  - match:
    - headers:
        x-chaos-test:
          exact: "delay"
    fault:
      delay:
        percentage:
          value: 10
        fixedDelay: 500ms
    route:
    - destination:
        host: genops-ai-inference

  # Inject 503 errors for 5% of requests
  - match:
    - headers:
        x-chaos-test:
          exact: "error"
    fault:
      abort:
        percentage:
          value: 5
        httpStatus: 503
    route:
    - destination:
        host: genops-ai-inference

  # Normal traffic
  - route:
    - destination:
        host: genops-ai-inference
```

---

## Security Enhancements

### mTLS for Service-to-Service Encryption

**Istio PeerAuthentication (Strict mTLS):**
```yaml
# Enforce strict mTLS for entire namespace
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: genops
spec:
  mtls:
    mode: STRICT

---
# Allow specific port in PERMISSIVE mode (for health checks)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: genops-ai-mtls
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference
  mtls:
    mode: STRICT
  portLevelMtls:
    8080:
      mode: STRICT
    8081:
      mode: PERMISSIVE  # Health check port
```

**Linkerd mTLS (Automatic):**
```bash
# Linkerd automatically enables mTLS for all meshed services
# Verify mTLS status
linkerd -n genops edges deployment

# Check certificate expiration
linkerd identity -n genops
```

### JWT Authentication Propagation

**Istio RequestAuthentication:**
```yaml
# Validate JWT tokens
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: genops-jwt-auth
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference
  jwtRules:
  - issuer: "https://auth.example.com"
    jwksUri: "https://auth.example.com/.well-known/jwks.json"
    audiences:
    - "genops-ai-api"
    forwardOriginalToken: true
    outputPayloadToHeader: "x-jwt-payload"

---
# Extract JWT claims to headers
apiVersion: networking.istio.io/v1beta1
kind: EnvoyFilter
metadata:
  name: jwt-claims-extraction
  namespace: genops
spec:
  workloadSelector:
    labels:
      app: genops-ai-inference
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
      listener:
        filterChain:
          filter:
            name: envoy.filters.network.http_connection_manager
    patch:
      operation: INSERT_AFTER
      value:
        name: envoy.filters.http.jwt_authn
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.jwt_authn.v3.JwtAuthentication
          providers:
            genops_provider:
              issuer: https://auth.example.com
              forward_payload_header: x-jwt-payload
              payload_in_metadata: jwt_payload
```

### Authorization Policies Based on Governance Labels

**Label-Based Authorization:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: genops-label-authz
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference

  action: ALLOW

  rules:
  # Allow only pods with governance labels
  - from:
    - source:
        namespaces: ["genops", "api-gateway"]
        principals: ["cluster.local/ns/*/sa/*"]
    to:
    - operation:
        methods: ["POST", "GET"]
    when:
    # Validate source has required labels
    - key: source.labels[genops.ai/team]
      notValues: [""]
    - key: source.labels[genops.ai/project]
      notValues: [""]
```

### Network Segmentation with Mesh Policies

**Istio Sidecar for Egress Control:**
```yaml
# Restrict outbound traffic to specific services
apiVersion: networking.istio.io/v1beta1
kind: Sidecar
metadata:
  name: genops-ai-sidecar
  namespace: genops
spec:
  workloadSelector:
    labels:
      app: genops-ai-inference

  # Only allow outbound to these services
  egress:
  - hosts:
    - "./*"  # Same namespace
    - "observability/*"  # Observability namespace
    - "istio-system/*"  # Istio system

  # Inbound configuration
  ingress:
  - port:
      number: 8080
      protocol: HTTP
      name: http
    defaultEndpoint: 127.0.0.1:8080
```

### Zero-Trust Networking Patterns

**Default Deny All + Selective Allow:**
```yaml
# Step 1: Deny all traffic by default
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: deny-all
  namespace: genops
spec:
  {}

---
# Step 2: Allow specific paths with strong authentication
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-authenticated
  namespace: genops
spec:
  selector:
    matchLabels:
      app: genops-ai-inference

  action: ALLOW

  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/api-gateway/sa/gateway"]
        namespaces: ["api-gateway"]
    to:
    - operation:
        methods: ["POST"]
        paths: ["/ai/v1/*"]
    when:
    # Require JWT authentication
    - key: request.auth.claims[iss]
      values: ["https://auth.example.com"]
    # Require governance context
    - key: request.headers[x-genops-customer-id]
      notValues: [""]
    - key: request.headers[x-genops-team]
      notValues: [""]
```

---

## Observability Deep-Dive

### Distributed Tracing Integration

**Istio + Jaeger Configuration:**
```yaml
# Deploy Jaeger
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:latest
        ports:
        - containerPort: 14268  # Jaeger collector
        - containerPort: 16686  # Jaeger UI
        env:
        - name: COLLECTOR_ZIPKIN_HTTP_PORT
          value: "9411"

---
# Update Istio telemetry
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: tracing-config
  namespace: istio-system
spec:
  tracing:
  - providers:
    - name: jaeger
    randomSamplingPercentage: 1.0  # 1% sampling
    customTags:
      # Add governance context to traces
      genops.team:
        header:
          name: x-genops-team
      genops.customer_id:
        header:
          name: x-genops-customer-id
      genops.project:
        header:
          name: x-genops-project
      genops.cost_center:
        header:
          name: x-genops-cost-center
```

### Governance Context in Mesh Headers

**Header Propagation Configuration:**
```yaml
# EnvoyFilter to propagate governance headers
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: governance-header-propagation
  namespace: genops
spec:
  workloadSelector:
    labels:
      app: genops-ai-inference
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_OUTBOUND
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.header_to_metadata
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.header_to_metadata.v3.Config
          request_rules:
          - header: x-genops-team
            on_header_present:
              metadata_namespace: governance
              key: team
              type: STRING
          - header: x-genops-customer-id
            on_header_present:
              metadata_namespace: governance
              key: customer_id
              type: STRING
          - header: x-genops-project
            on_header_present:
              metadata_namespace: governance
              key: project
              type: STRING
```

### Service-Level Metrics and SLIs

**Prometheus Queries for SLIs:**
```yaml
# Availability SLI (success rate)
sum(rate(istio_requests_total{
  destination_service="genops-ai-inference.genops.svc.cluster.local",
  response_code!~"5.."
}[5m]))
/
sum(rate(istio_requests_total{
  destination_service="genops-ai-inference.genops.svc.cluster.local"
}[5m]))

# Latency SLI (P95 < 1000ms)
histogram_quantile(0.95,
  sum(rate(istio_request_duration_milliseconds_bucket{
    destination_service="genops-ai-inference.genops.svc.cluster.local"
  }[5m])) by (le)
)

# Throughput by customer tier
sum(rate(istio_requests_total{
  destination_service="genops-ai-inference.genops.svc.cluster.local"
}[5m])) by (genops_tier)
```

### Grafana Dashboards for Service Mesh

**Import Istio Dashboards:**
```bash
# Istio provides pre-built Grafana dashboards
kubectl -n istio-system port-forward svc/grafana 3000:3000

# Import GenOps-specific dashboard
cat > genops-service-mesh-dashboard.json <<'EOF'
{
  "dashboard": {
    "title": "GenOps AI Service Mesh",
    "panels": [
      {
        "title": "Requests by Customer Tier",
        "targets": [{
          "expr": "sum(rate(istio_requests_total{destination_service=~\"genops.*\"}[5m])) by (genops_tier)"
        }]
      },
      {
        "title": "Cost by Team",
        "targets": [{
          "expr": "sum(genops_cost_total) by (genops_team)"
        }]
      }
    ]
  }
}
EOF

# Apply dashboard
kubectl create configmap genops-grafana-dashboard \
  --from-file=genops-service-mesh-dashboard.json \
  -n observability
```

### Kiali for Mesh Visualization

**Install and Configure Kiali:**
```bash
# Install Kiali operator
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml

# Access Kiali dashboard
kubectl -n istio-system port-forward svc/kiali 20001:20001

# Open browser to http://localhost:20001
```

**Kiali Custom Labels for GenOps:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kiali
  namespace: istio-system
data:
  config.yaml: |
    server:
      port: 20001

    # Custom labels for GenOps visualization
    external_services:
      custom_dashboards:
        genops_governance:
          - name: "Cost by Team"
            query: "sum(genops_cost_total) by (genops_team)"
          - name: "Requests by Customer"
            query: "sum(rate(istio_requests_total[5m])) by (genops_customer_id)"

    kubernetes_config:
      cluster_name: "production"

    # Health configuration
    health_config:
      rate:
        - namespace: "genops"
          kind: "service"
          name: "genops-ai-inference"
          tolerance:
            - code: "5xx"
              failure: 10
              protocol: "http"
```

---

## Service Mesh Selection Guide

### Choose Istio if:
✅ You need advanced traffic management features
✅ Multi-cluster deployments are required
✅ Complex authorization policies are needed
✅ Team has experience with service mesh operations

### Choose Linkerd if:
✅ Simplicity and ease of operation are priorities
✅ Performance overhead must be minimal
✅ Automatic mTLS is the primary use case
✅ You want a lightweight, opinionated solution

### Choose Consul Connect if:
✅ Already using Consul for service discovery
✅ Multi-cloud deployments across different Kubernetes clusters
✅ Integration with non-Kubernetes workloads (VMs, etc.)
✅ Hybrid cloud architecture

---

## Next Steps

Ready to enhance GenOps AI with a service mesh? Start with:

1. **Choose Your Service Mesh** - Evaluate Istio, Linkerd, or Consul Connect
2. **Install Control Plane** - Set up the service mesh in a test environment
3. **Enable Sidecar Injection** - Add service mesh proxies to GenOps AI pods
4. **Configure Observability** - Integrate with your existing monitoring stack
5. **Implement Traffic Policies** - Add circuit breakers, retries, and timeouts

Return to [Kubernetes Getting Started](kubernetes-getting-started.md) for the complete deployment overview.

---

## Support

- **Documentation:** [GenOps AI Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community:** [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
