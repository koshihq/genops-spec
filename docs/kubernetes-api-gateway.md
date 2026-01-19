# API Gateway Integration for GenOps AI

> **Status:** 📋 Documentation in progress
> **Last Updated:** 2026-01-18

Expose GenOps AI services through API gateways with governance-aware routing, rate limiting, and authentication.

---

## Overview

API gateways provide essential capabilities for exposing AI services to external consumers:
- **Authentication and Authorization** with JWT validation and API key management
- **Rate Limiting** with governance-aware quotas per team, project, or customer
- **Cost Attribution** by automatically injecting governance context into requests
- **Request/Response Transformation** for standardized API contracts
- **Analytics and Monitoring** with detailed request telemetry

GenOps AI integrates seamlessly with popular API gateways to provide unified governance tracking for all inbound requests.

---

## Quick Reference

### Supported API Gateways

**Kong:**
- Enterprise-grade API gateway
- Rich plugin ecosystem
- Native Kubernetes Ingress support
- Advanced rate limiting and authentication

**Ambassador (Emissary-ingress):**
- Kubernetes-native API gateway
- Built on Envoy proxy
- GitOps-friendly configuration
- Integrated with service mesh patterns

**NGINX Ingress Controller:**
- Lightweight and performant
- Wide community adoption
- Simple configuration
- Extensive customization options

**Traefik:**
- Modern, dynamic configuration
- Automatic service discovery
- Let's Encrypt integration
- Real-time metrics and tracing

### Key Benefits for AI Workloads

**Governance Context Injection:**
```yaml
# API Gateway automatically adds governance headers
apiVersion: gateway.networking.k8s.io/v1beta1
kind: HTTPRoute
metadata:
  name: ai-inference-route
spec:
  parentRefs:
  - name: genops-gateway
  hostnames:
  - "api.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /ai/inference
    filters:
    - type: RequestHeaderModifier
      requestHeaderModifier:
        add:
        - name: x-genops-team
          value: "extracted-from-jwt"
        - name: x-genops-customer-id
          value: "extracted-from-api-key"
    backendRefs:
    - name: ai-inference
      port: 8080
```

**Cost-Aware Rate Limiting:**
```yaml
# Kong plugin for governance-aware rate limiting
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-rate-limit
plugin: rate-limiting
config:
  minute: 100
  policy: redis
  redis_host: redis.default.svc.cluster.local
  # Rate limits vary by customer tier
  limit_by: header
  header_name: x-genops-customer-tier
```

---

## Table of Contents

### Planned Documentation Sections

1. **API Gateway Fundamentals**
   - Architecture patterns for AI service exposure
   - Ingress vs API Gateway vs Service Mesh
   - When to use which pattern
   - Performance considerations

2. **Kong Integration**
   - Installation and configuration
   - Custom plugins for governance tracking
   - Rate limiting strategies per team/customer
   - Authentication with JWT and API keys
   - Cost attribution at the gateway level

3. **Ambassador/Emissary Integration**
   - Kubernetes-native gateway setup
   - Mapping resources for AI services
   - Authentication with AuthService
   - Rate limiting and circuit breakers

4. **NGINX Ingress Integration**
   - Ingress resource configuration
   - Custom annotations for governance
   - ModSecurity WAF integration
   - SSL/TLS termination

5. **Traefik Integration**
   - Dynamic configuration with IngressRoute
   - Middleware for request transformation
   - Let's Encrypt automatic SSL
   - Distributed tracing integration

6. **Authentication and Authorization**
   - JWT validation and claim extraction
   - API key management and rotation
   - OAuth2/OIDC integration
   - mTLS for service-to-service communication

7. **Rate Limiting and Quotas**
   - Per-customer rate limiting strategies
   - Budget-aware throttling
   - Burst handling and queue management
   - Distributed rate limiting with Redis

8. **Cost Attribution and Billing**
   - Request-level cost tracking
   - Customer billing integration
   - Usage analytics and reporting
   - Chargeback mechanisms

---

## Related Documentation

**Kubernetes Guides:**
- [Kubernetes Getting Started](kubernetes-getting-started.md)
- [Service Mesh Integration](kubernetes-service-mesh.md)
- [Security Hardening](kubernetes-security.md)

**Multi-Tenant Patterns:**
- [Multi-Tenant Architecture](kubernetes-multi-tenant.md)
- [Cost Optimization](kubernetes-cost-optimization.md)

---

## Quick Examples

### Example 1: Kong Gateway with GenOps AI

```bash
# Install Kong Ingress Controller
helm repo add kong https://charts.konghq.com
helm install kong kong/kong \
  --namespace kong \
  --create-namespace \
  --set ingressController.installCRDs=false \
  --set proxy.type=LoadBalancer

# Deploy GenOps AI with Kong annotations
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ai-inference
  namespace: genops
  annotations:
    konghq.com/plugins: genops-rate-limit,genops-auth
spec:
  selector:
    app: ai-inference
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-inference-ingress
  namespace: genops
  annotations:
    konghq.com/strip-path: "true"
spec:
  ingressClassName: kong
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /ai/inference
        pathType: Prefix
        backend:
          service:
            name: ai-inference
            port:
              number: 8080
EOF
```

### Example 2: Governance-Aware Rate Limiting with Kong

```yaml
# Custom Kong plugin for per-customer rate limiting
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-customer-rate-limit
  namespace: genops
plugin: rate-limiting-advanced
config:
  limit:
    - 1000  # Premium customers
    - 100   # Standard customers
    - 10    # Free tier customers
  window_size:
    - 3600
  identifier: header
  header_name: x-genops-customer-id
  sync_rate: 10
  namespace: genops-rate-limits
  strategy: redis
  redis:
    host: redis.default.svc.cluster.local
    port: 6379
    database: 0

---
# Apply to service
apiVersion: v1
kind: Service
metadata:
  name: ai-inference
  namespace: genops
  annotations:
    konghq.com/plugins: genops-customer-rate-limit
spec:
  selector:
    app: ai-inference
  ports:
  - port: 8080
```

### Example 3: JWT Authentication with Claim Extraction

```yaml
# Kong JWT plugin with governance claim extraction
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-jwt-auth
  namespace: genops
plugin: jwt
config:
  key_claim_name: kid
  claims_to_verify:
    - exp
  run_on_preflight: false

---
# Header transformer to add governance context
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-header-transformer
  namespace: genops
plugin: request-transformer
config:
  add:
    headers:
      - x-genops-team:$(claims.team)
      - x-genops-project:$(claims.project)
      - x-genops-customer-id:$(claims.sub)
      - x-genops-customer-tier:$(claims.tier)

---
# Apply both plugins to service
apiVersion: v1
kind: Service
metadata:
  name: ai-inference
  namespace: genops
  annotations:
    konghq.com/plugins: genops-jwt-auth,genops-header-transformer
spec:
  selector:
    app: ai-inference
  ports:
  - port: 8080
```

### Example 4: Ambassador Mapping with Governance

```yaml
# Ambassador Mapping for GenOps AI service
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: ai-inference-mapping
  namespace: genops
spec:
  hostname: api.example.com
  prefix: /ai/inference/
  service: ai-inference.genops:8080
  timeout_ms: 30000

  # Add governance headers from JWT claims
  add_request_headers:
    x-genops-team:
      value: "%REQ(x-jwt-claim-team)%"
    x-genops-customer-id:
      value: "%REQ(x-jwt-claim-sub)%"
    x-genops-project:
      value: "%REQ(x-jwt-claim-project)%"

  # Enable rate limiting
  labels:
    ambassador:
      - request_label:
        - genops_customer_id:
            header: "x-genops-customer-id"

---
# Rate limit configuration
apiVersion: getambassador.io/v3alpha1
kind: RateLimitService
metadata:
  name: genops-ratelimit
  namespace: genops
spec:
  service: ratelimit.genops:8081
  protocol_version: v3
```

### Example 5: Kubernetes Gateway API with GenOps

```yaml
# Modern Kubernetes Gateway API configuration
apiVersion: gateway.networking.k8s.io/v1beta1
kind: Gateway
metadata:
  name: genops-gateway
  namespace: genops
spec:
  gatewayClassName: kong
  listeners:
  - name: https
    protocol: HTTPS
    port: 443
    hostname: api.example.com
    tls:
      certificateRefs:
      - name: api-tls-cert

---
# HTTPRoute with governance filters
apiVersion: gateway.networking.k8s.io/v1beta1
kind: HTTPRoute
metadata:
  name: ai-inference-route
  namespace: genops
spec:
  parentRefs:
  - name: genops-gateway
  hostnames:
  - api.example.com
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /ai/inference
    filters:
    # Extract governance context from JWT
    - type: ExtensionRef
      extensionRef:
        group: configuration.konghq.com
        kind: KongPlugin
        name: jwt-to-headers
    # Apply rate limiting
    - type: ExtensionRef
      extensionRef:
        group: configuration.konghq.com
        kind: KongPlugin
        name: customer-rate-limit
    backendRefs:
    - name: ai-inference
      port: 8080
```

---

## API Gateway Fundamentals

### Architecture Patterns for AI Service Exposure

API gateways serve as the single entry point for external clients to access your AI services, providing essential capabilities:

**Core Responsibilities:**
- **Routing and Load Balancing**: Distribute requests across service instances
- **Authentication and Authorization**: Validate identities and enforce access policies
- **Rate Limiting and Throttling**: Protect services from overload
- **Request/Response Transformation**: Standardize API contracts
- **Observability**: Collect metrics, logs, and traces
- **Governance Context Injection**: Add team, project, and customer attribution headers

**Ingress vs API Gateway vs Service Mesh:**

| Feature | Ingress Controller | API Gateway | Service Mesh |
|---------|-------------------|-------------|--------------|
| **Primary Function** | HTTP(S) routing | API management | Service-to-service communication |
| **Scope** | North-south traffic | North-south traffic | East-west traffic |
| **Protocol Support** | HTTP/HTTPS | HTTP/HTTPS/gRPC/WebSocket | All TCP/UDP |
| **Authentication** | Basic | Advanced (JWT, OAuth, API keys) | mTLS |
| **Rate Limiting** | Basic | Advanced (per-key, per-user) | Service-level |
| **Complexity** | Low | Medium | High |
| **Best For** | Simple routing | External APIs | Microservices mesh |

**When to Use Each Pattern:**

Use **Ingress Controller** when:
- Simple HTTP/HTTPS routing is sufficient
- Minimal operational overhead is desired
- Built-in cloud load balancer integration is needed
- Certificate management with cert-manager

Use **API Gateway** when:
- External APIs require advanced authentication
- Per-customer rate limiting and quotas are needed
- Request transformation and validation required
- Monetization and billing integration
- Analytics and detailed usage tracking

Use **Service Mesh** when:
- Many microservices communicate internally
- Automatic mTLS encryption is required
- Advanced traffic management (circuit breakers, retries)
- Detailed observability for all service-to-service calls

### Performance Considerations

**Latency Impact:**
```
Direct service call:         ~1-2ms
+ Ingress Controller:        +1-3ms
+ API Gateway:               +5-15ms
+ API Gateway + Service Mesh: +10-30ms
```

**Throughput Benchmarks (requests/second):**
- **NGINX Ingress**: 50,000+ RPS
- **Kong (open-source)**: 20,000-30,000 RPS
- **Ambassador**: 15,000-25,000 RPS
- **Traefik**: 25,000-40,000 RPS

**Resource Requirements (typical):**

| Gateway | CPU (per instance) | Memory (per instance) | Minimum Replicas |
|---------|-------------------|---------------------|------------------|
| NGINX | 100-500m | 256-512Mi | 2 |
| Kong | 500m-1 | 512Mi-1Gi | 2 |
| Ambassador | 500m-1 | 512Mi-1Gi | 2 |
| Traefik | 200-500m | 256-512Mi | 2 |

### Cost Comparison

**Open-Source Options (Infrastructure Only):**
- NGINX Ingress Controller: Free (infrastructure costs only)
- Kong Open-Source: Free (infrastructure costs only)
- Traefik: Free (infrastructure costs only)
- Ambassador (Emissary-ingress): Free (infrastructure costs only)

**Commercial/Enterprise Editions:**
- **NGINX Plus**: $2,500+/instance/year
- **Kong Enterprise**: $50,000+/year (varies by features)
- **Ambassador Cloud**: Usage-based pricing
- **Traefik Enterprise**: Contact for pricing

**Estimated Monthly Infrastructure Cost (AWS):**
- 2x t3.medium instances: ~$60/month
- 2x t3.large instances: ~$120/month
- Load Balancer: ~$20-30/month
- **Total**: $80-150/month for basic production setup

---

## Kong Integration Deep-Dive

### Kong Ingress Controller Installation

**Install Kong with Helm:**
```bash
# Add Kong Helm repository
helm repo add kong https://charts.konghq.com
helm repo update

# Create namespace
kubectl create namespace kong

# Install Kong Ingress Controller
helm install kong kong/kong \
  --namespace kong \
  --set ingressController.installCRDs=false \
  --set proxy.type=LoadBalancer \
  --set proxy.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb" \
  --set admin.enabled=true \
  --set admin.type=ClusterIP \
  --set postgresql.enabled=true \
  --set env.database=postgres \
  --set env.log_level=info

# Verify installation
kubectl get pods -n kong
kubectl get svc -n kong
```

**Enable Prometheus Metrics:**
```yaml
# Kong configuration for metrics
apiVersion: v1
kind: ConfigMap
metadata:
  name: kong-prometheus
  namespace: kong
data:
  servers.conf: |
    server {
      listen 8100;
      location /metrics {
        content_by_lua_block {
          require("kong.plugins.prometheus.exporter").collect()
        }
      }
    }
---
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: kong-metrics
  namespace: kong
spec:
  selector:
    matchLabels:
      app: kong
  endpoints:
  - port: metrics
    interval: 30s
```

### Custom Kong Plugins for GenOps Governance

**Governance Context Extractor Plugin:**
```lua
-- kong/plugins/genops-governance/handler.lua
local BasePlugin = require "kong.plugins.base_plugin"

local GenOpsGovernanceHandler = BasePlugin:extend()

GenOpsGovernanceHandler.PRIORITY = 1000
GenOpsGovernanceHandler.VERSION = "1.0.0"

function GenOpsGovernanceHandler:access(conf)
  GenOpsGovernanceHandler.super.access(self)

  local jwt_claims = kong.request.get_header("X-JWT-Claim-*")

  -- Extract governance attributes from JWT
  kong.service.request.set_header("X-GenOps-Team",
    kong.request.get_header("X-JWT-Claim-Team") or "default")
  kong.service.request.set_header("X-GenOps-Project",
    kong.request.get_header("X-JWT-Claim-Project") or "unknown")
  kong.service.request.set_header("X-GenOps-Customer-ID",
    kong.request.get_header("X-JWT-Claim-Sub") or "")
  kong.service.request.set_header("X-GenOps-Customer-Tier",
    kong.request.get_header("X-JWT-Claim-Tier") or "free")

  -- Add request ID for tracing
  local request_id = kong.request.get_header("X-Request-ID") or kong.tools.uuid.uuid()
  kong.service.request.set_header("X-Request-ID", request_id)

  kong.log.info("GenOps governance context injected for customer: ",
    kong.request.get_header("X-GenOps-Customer-ID"))
end

return GenOpsGovernanceHandler
```

**Plugin Configuration:**
```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-governance
  namespace: genops
plugin: genops-governance
config:
  extract_from_jwt: true
  default_team: "platform"
  require_customer_id: true
```

### Multi-Tier Rate Limiting

**Redis-Backed Rate Limiting:**
```yaml
# Deploy Redis for distributed rate limiting
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-ratelimit
  namespace: kong
spec:
  replicas: 2
  selector:
    matchLabels:
      app: redis-ratelimit
  template:
    metadata:
      labels:
        app: redis-ratelimit
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-ratelimit
  namespace: kong
spec:
  selector:
    app: redis-ratelimit
  ports:
  - port: 6379
    targetPort: 6379
```

**Tier-Based Rate Limiting Plugin:**
```yaml
# Rate limiting based on customer tier
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-tiered-ratelimit
  namespace: genops
plugin: rate-limiting-advanced
config:
  # Define limits per tier
  limit:
    - name: free
      limit: 100
      window_size: 3600  # 100 requests per hour
    - name: standard
      limit: 1000
      window_size: 3600  # 1,000 requests per hour
    - name: premium
      limit: 10000
      window_size: 3600  # 10,000 requests per hour

  # Identify tier from header
  identifier: header
  header_name: x-genops-customer-tier

  # Use Redis for distributed tracking
  strategy: redis
  redis:
    host: redis-ratelimit.kong.svc.cluster.local
    port: 6379
    database: 0
    timeout: 2000

  # Sync rate limit counters
  sync_rate: 10
  namespace: genops-rate-limits

  # Hide client details in response
  hide_client_headers: false

  # Error handling
  error_code: 429
  error_message: "API rate limit exceeded for your tier"
```

### Cost-Per-Request Tracking

**Custom Plugin for Cost Attribution:**
```yaml
# Cost tracking plugin configuration
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-cost-tracker
  namespace: genops
plugin: request-termination  # Base plugin, extended for cost tracking
config:
  # Integrate with GenOps telemetry
  telemetry_endpoint: "http://otel-collector.observability:4318/v1/traces"

  # Cost attribution rules
  cost_per_request:
    default: 0.001  # $0.001 per request
    premium: 0.0005  # Discounted for premium tier

  # Track request metadata
  track_request_body_size: true
  track_response_body_size: true
  track_latency: true

  # Export to GenOps
  export_format: "otlp"
  batch_size: 100
  batch_timeout: 5000  # ms
```

**Apply Plugins to Service:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-inference
  namespace: genops
  annotations:
    # Chain multiple plugins
    konghq.com/plugins: >
      genops-jwt-auth,
      genops-governance,
      genops-tiered-ratelimit,
      genops-cost-tracker
spec:
  selector:
    app: genops-ai-inference
  ports:
  - port: 8080
    targetPort: 8080
```

---

## Ambassador (Emissary-Ingress) Integration

### Installation and Configuration

**Install Ambassador Edge Stack:**
```bash
# Add Ambassador Helm repository
helm repo add datawire https://app.getambassador.io
helm repo update

# Install Ambassador Edge Stack
kubectl create namespace ambassador
kubectl apply -f https://app.getambassador.io/yaml/edge-stack/3.9.0/aes-crds.yaml

helm install ambassador datawire/edge-stack \
  --namespace ambassador \
  --set service.type=LoadBalancer \
  --set enableAES=false  # Use open-source version

# Verify installation
kubectl get pods -n ambassador
kubectl get svc -n ambassador
```

### Mapping Resources for AI Services

**Basic Mapping with Governance:**
```yaml
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: genops-ai-mapping
  namespace: genops
spec:
  hostname: api.example.com
  prefix: /ai/v1/
  service: genops-ai-inference.genops:8080
  timeout_ms: 30000

  # Extract JWT claims to headers
  add_request_headers:
    x-genops-team:
      value: "%REQ(x-jwt-claim-team)%"
    x-genops-customer-id:
      value: "%REQ(x-jwt-claim-sub)%"
    x-genops-project:
      value: "%REQ(x-jwt-claim-project)%"
    x-genops-customer-tier:
      value: "%REQ(x-jwt-claim-tier)%"

  # Add request tracking
  labels:
    ambassador:
      - request_label:
        - customer_id:
            header: "x-genops-customer-id"
        - team:
            header: "x-genops-team"
```

**Advanced Mapping with Retry and Circuit Breaking:**
```yaml
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: genops-ai-resilient
  namespace: genops
spec:
  hostname: api.example.com
  prefix: /ai/v1/
  service: genops-ai-inference.genops:8080

  # Retry configuration
  retry_policy:
    retry_on: "5xx"
    num_retries: 3
    per_try_timeout: "10s"

  # Circuit breaker
  circuit_breakers:
    max_connections: 1000
    max_pending_requests: 1000
    max_requests: 1000
    max_retries: 3

  # Load balancing
  load_balancer:
    policy: least_request
```

### AuthService for JWT Validation

**Deploy External Auth Service:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-auth-service
  namespace: genops
spec:
  replicas: 2
  selector:
    matchLabels:
      app: genops-auth-service
  template:
    metadata:
      labels:
        app: genops-auth-service
    spec:
      containers:
      - name: auth
        image: genops/auth-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: JWT_PUBLIC_KEY
          valueFrom:
            secretKeyRef:
              name: jwt-keys
              key: public-key
        - name: ALLOWED_ISSUERS
          value: "https://auth.example.com"
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: genops-auth-service
  namespace: genops
spec:
  selector:
    app: genops-auth-service
  ports:
  - port: 8080
    targetPort: 8080
```

**Configure Ambassador AuthService:**
```yaml
apiVersion: getambassador.io/v3alpha1
kind: AuthService
metadata:
  name: genops-authentication
  namespace: genops
spec:
  auth_service: genops-auth-service.genops:8080
  proto: http
  timeout_ms: 5000

  # Pass JWT claims as headers
  allowed_request_headers:
    - "authorization"
  allowed_authorization_headers:
    - "x-jwt-claim-*"

  # Include request body for validation
  include_body:
    max_bytes: 4096
    allow_partial: true
```

### Distributed Rate Limiting with Redis

**RateLimitService Configuration:**
```yaml
apiVersion: getambassador.io/v3alpha1
kind: RateLimitService
metadata:
  name: genops-ratelimit
  namespace: genops
spec:
  service: ratelimit.genops:8081
  protocol_version: v3
  timeout_ms: 500

---
# Deploy Envoy Rate Limit Service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ratelimit
  namespace: genops
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ratelimit
  template:
    metadata:
      labels:
        app: ratelimit
    spec:
      containers:
      - name: ratelimit
        image: envoyproxy/ratelimit:latest
        ports:
        - containerPort: 8081
        env:
        - name: REDIS_SOCKET_TYPE
          value: tcp
        - name: REDIS_URL
          value: redis-ratelimit.kong:6379
        - name: USE_STATSD
          value: "false"
        - name: LOG_LEVEL
          value: info
        - name: RUNTIME_ROOT
          value: /data
        - name: RUNTIME_SUBDIRECTORY
          value: ratelimit
        volumeMounts:
        - name: config
          mountPath: /data/ratelimit/config
      volumes:
      - name: config
        configMap:
          name: ratelimit-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ratelimit-config
  namespace: genops
data:
  config.yaml: |
    domain: genops-api
    descriptors:
      # Free tier: 100 requests per hour
      - key: customer_tier
        value: free
        rate_limit:
          unit: hour
          requests_per_unit: 100

      # Standard tier: 1,000 requests per hour
      - key: customer_tier
        value: standard
        rate_limit:
          unit: hour
          requests_per_unit: 1000

      # Premium tier: 10,000 requests per hour
      - key: customer_tier
        value: premium
        rate_limit:
          unit: hour
          requests_per_unit: 10000
```

### Traffic Shadowing for Testing

**Shadow Traffic to Canary Version:**
```yaml
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: genops-ai-production
  namespace: genops
spec:
  hostname: api.example.com
  prefix: /ai/v1/
  service: genops-ai-inference-v1.genops:8080

  # Shadow 10% of traffic to v2
  shadow: true
  weight: 100

---
apiVersion: getambassador.io/v3alpha1
kind: Mapping
metadata:
  name: genops-ai-canary
  namespace: genops
spec:
  hostname: api.example.com
  prefix: /ai/v1/
  service: genops-ai-inference-v2.genops:8080
  weight: 10  # 10% of traffic goes here

  # Label shadowed requests
  add_request_headers:
    x-shadow-traffic:
      value: "true"
```

---

## NGINX Ingress and Traefik Integration

### NGINX Ingress Controller Setup

**Install NGINX Ingress Controller:**
```bash
# Install with Helm
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

kubectl create namespace ingress-nginx

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.replicaCount=2 \
  --set controller.nodeSelector."kubernetes\.io/os"=linux \
  --set controller.service.type=LoadBalancer \
  --set controller.metrics.enabled=true \
  --set controller.metrics.serviceMonitor.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"="true" \
  --set controller.podAnnotations."prometheus\.io/port"="10254"

# Verify installation
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx
```

### Custom Annotations for Governance

**Ingress with GenOps Governance Annotations:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genops-ai-ingress
  namespace: genops
  annotations:
    # NGINX Ingress Controller class
    kubernetes.io/ingress.class: nginx

    # SSL/TLS
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "10"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"

    # Request/response size limits
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"

    # Timeouts
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"

    # Custom headers for governance
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-GenOps-Gateway: nginx";
      more_set_headers "X-Request-ID: $request_id";

    # Auth annotations
    nginx.ingress.kubernetes.io/auth-url: "http://genops-auth-service.genops.svc.cluster.local:8080/verify"
    nginx.ingress.kubernetes.io/auth-response-headers: "X-JWT-Claim-Team,X-JWT-Claim-Sub,X-JWT-Claim-Project,X-JWT-Claim-Tier"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-example-com-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /ai/v1
        pathType: Prefix
        backend:
          service:
            name: genops-ai-inference
            port:
              number: 8080
```

### ModSecurity WAF Integration

**Enable ModSecurity Web Application Firewall:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: modsecurity-config
  namespace: ingress-nginx
data:
  modsecurity.conf: |
    SecRuleEngine On
    SecRequestBodyAccess On
    SecResponseBodyAccess On
    SecAuditEngine RelevantOnly
    SecAuditLog /dev/stdout

    # OWASP Core Rule Set
    Include /etc/nginx/modsecurity/crs-setup.conf
    Include /etc/nginx/modsecurity/rules/*.conf
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genops-ai-waf
  namespace: genops
  annotations:
    nginx.ingress.kubernetes.io/enable-modsecurity: "true"
    nginx.ingress.kubernetes.io/enable-owasp-core-rules: "true"
    nginx.ingress.kubernetes.io/modsecurity-snippet: |
      SecRuleRemoveById 920350  # Adjust rules as needed
spec:
  ingressClassName: nginx
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /ai/v1
        pathType: Prefix
        backend:
          service:
            name: genops-ai-inference
            port:
              number: 8080
```

### SSL/TLS Termination with cert-manager

**Install cert-manager:**
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
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
EOF
```

### Traefik IngressRoute Configuration

**Install Traefik:**
```bash
helm repo add traefik https://traefik.github.io/charts
helm repo update

kubectl create namespace traefik

helm install traefik traefik/traefik \
  --namespace traefik \
  --set service.type=LoadBalancer \
  --set ports.websecure.tls.enabled=true \
  --set experimental.plugins.enabled=true
```

**IngressRoute with Middleware:**
```yaml
# Middleware for governance headers
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: genops-headers
  namespace: genops
spec:
  headers:
    customRequestHeaders:
      X-GenOps-Gateway: "traefik"
    customResponseHeaders:
      X-Request-ID: ""

---
# Middleware for rate limiting
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: genops-ratelimit
  namespace: genops
spec:
  rateLimit:
    average: 100
    burst: 50
    period: 1m

---
# IngressRoute
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: genops-ai-route
  namespace: genops
spec:
  entryPoints:
    - websecure
  routes:
  - match: Host(`api.example.com`) && PathPrefix(`/ai/v1`)
    kind: Rule
    middlewares:
    - name: genops-headers
    - name: genops-ratelimit
    services:
    - name: genops-ai-inference
      port: 8080
  tls:
    secretName: api-example-com-tls
```

---

## Authentication and Authorization

### JWT Validation and Claim Extraction

**JWT Validation Flow:**
```
1. Client sends request with JWT in Authorization header
2. API Gateway validates JWT signature and expiration
3. Gateway extracts claims (team, project, customer_id, tier)
4. Claims injected as X-GenOps-* headers to backend service
5. GenOps SDK reads headers for governance tracking
```

**Example JWT Structure:**
```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key-2024-01"
  },
  "payload": {
    "iss": "https://auth.example.com",
    "sub": "customer-abc-123",
    "aud": "genops-ai-api",
    "exp": 1706486400,
    "iat": 1706482800,
    "team": "ml-platform",
    "project": "production-inference",
    "tier": "premium",
    "cost_center": "engineering"
  }
}
```

### API Key Management and Rotation

**API Key Storage in Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: customer-api-keys
  namespace: genops
type: Opaque
data:
  # Base64-encoded API keys with metadata
  customer-abc-123: |
    a2V5OnByZW1pdW0tdGllcg==
stringData:
  key-metadata.json: |
    {
      "customer-abc-123": {
        "tier": "premium",
        "team": "ml-platform",
        "created": "2024-01-01T00:00:00Z",
        "expires": "2025-01-01T00:00:00Z"
      }
    }
```

**API Key Rotation Script:**
```python
#!/usr/bin/env python3
"""Automated API key rotation for GenOps customers."""
import base64
import secrets
from datetime import datetime, timedelta
from kubernetes import client, config

def rotate_api_key(customer_id: str, tier: str):
    """Generate new API key and update Kubernetes secret."""
    # Generate cryptographically secure key
    new_key = secrets.token_urlsafe(32)

    # Load Kubernetes config
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # Update secret
    secret = v1.read_namespaced_secret("customer-api-keys", "genops")
    secret.data[customer_id] = base64.b64encode(
        f"key:{tier}".encode()
    ).decode()

    v1.replace_namespaced_secret("customer-api-keys", "genops", secret)

    print(f"✅ Rotated API key for customer: {customer_id}")
    return new_key

if __name__ == "__main__":
    # Rotate keys expiring in next 30 days
    rotate_api_key("customer-abc-123", "premium")
```

### OAuth2/OIDC Integration with Dex

**Deploy Dex OIDC Provider:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dex
  namespace: genops
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dex
  template:
    metadata:
      labels:
        app: dex
    spec:
      containers:
      - name: dex
        image: ghcr.io/dexidp/dex:v2.37.0
        ports:
        - containerPort: 5556
        volumeMounts:
        - name: config
          mountPath: /etc/dex
      volumes:
      - name: config
        configMap:
          name: dex-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: dex-config
  namespace: genops
data:
  config.yaml: |
    issuer: https://dex.example.com

    storage:
      type: kubernetes
      config:
        inCluster: true

    web:
      http: 0.0.0.0:5556

    connectors:
    - type: oidc
      id: google
      name: Google
      config:
        issuer: https://accounts.google.com
        clientID: $GOOGLE_CLIENT_ID
        clientSecret: $GOOGLE_CLIENT_SECRET
        redirectURI: https://dex.example.com/callback

    oauth2:
      skipApprovalScreen: true

    staticClients:
    - id: genops-api-gateway
      redirectURIs:
      - 'https://api.example.com/callback'
      name: 'GenOps API Gateway'
      secret: $DEX_CLIENT_SECRET
```

### mTLS for Service-to-Service Authentication

**cert-manager Certificate for mTLS:**
```yaml
# CA for internal services
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: internal-ca
spec:
  ca:
    secretName: internal-ca-key-pair

---
# Certificate for API Gateway
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: api-gateway-mtls
  namespace: genops
spec:
  secretName: api-gateway-mtls-cert
  issuerRef:
    name: internal-ca
    kind: ClusterIssuer
  commonName: api-gateway.genops.svc.cluster.local
  dnsNames:
  - api-gateway.genops.svc.cluster.local
  - api-gateway.genops
  usages:
  - digital signature
  - key encipherment
  - client auth
  - server auth
```

### Multi-Tenant Authentication Patterns

**Tenant Isolation with JWT:**
```yaml
# Kong plugin for tenant isolation
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-tenant-isolation
  namespace: genops
plugin: request-validator
config:
  # Require tenant ID in JWT
  allowed_content_types:
    - application/json
  body_schema: |
    {
      "type": "object",
      "required": ["tenant_id"],
      "properties": {
        "tenant_id": {
          "type": "string",
          "pattern": "^[a-z0-9-]+$"
        }
      }
    }
```

---

## Rate Limiting and Quotas

### Per-Customer Rate Limiting Strategies

**Sliding Window Rate Limiting:**
```yaml
# Kong sliding window algorithm
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-sliding-window-ratelimit
  namespace: genops
plugin: rate-limiting-advanced
config:
  # Sliding window configuration
  window_size:
    - 3600  # 1 hour window
  limit:
    - 1000  # 1,000 requests per hour

  # Sliding window type
  window_type: sliding

  # Identify requests by customer ID
  identifier: header
  header_name: x-genops-customer-id

  # Redis for distributed tracking
  strategy: redis
  redis:
    host: redis-ratelimit.kong
    port: 6379
    database: 1

  # Synchronization
  sync_rate: 10

  # Response headers
  hide_client_headers: false
```

### Budget-Aware Throttling

**Budget Threshold Enforcement:**
```python
# Custom middleware for budget checking
from fastapi import Request, HTTPException
from genops import get_budget_status

async def budget_enforcement_middleware(request: Request, call_next):
    """Check customer budget before processing request."""
    customer_id = request.headers.get("x-genops-customer-id")

    if customer_id:
        budget_status = get_budget_status(customer_id=customer_id)

        # Block if budget exceeded
        if budget_status.consumed_percent >= 100:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "budget_exceeded",
                    "message": "Your monthly budget has been exceeded",
                    "consumed": budget_status.consumed,
                    "limit": budget_status.limit,
                    "reset_date": budget_status.reset_date
                }
            )

        # Warn if approaching limit
        if budget_status.consumed_percent >= 90:
            response = await call_next(request)
            response.headers["X-Budget-Warning"] = (
                f"Budget {budget_status.consumed_percent:.1f}% consumed"
            )
            return response

    return await call_next(request)
```

### Burst Handling and Queue Management

**Token Bucket Algorithm:**
```yaml
# Traefik rate limit with burst
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: genops-burst-ratelimit
  namespace: genops
spec:
  rateLimit:
    average: 100      # Average 100 requests per minute
    burst: 200        # Allow bursts up to 200 requests
    period: 1m
    sourceCriterion:
      requestHeaderName: X-GenOps-Customer-ID
```

### Cost-Per-Request Quotas

**Monthly Request Quota Enforcement:**
```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: genops-monthly-quota
  namespace: genops
plugin: rate-limiting-advanced
config:
  limit:
    - 100000  # 100,000 requests per month
  window_size:
    - 2592000  # 30 days in seconds

  identifier: header
  header_name: x-genops-customer-id

  strategy: redis
  redis:
    host: redis-ratelimit.kong
    port: 6379
    database: 2

  # Reset on first of month
  namespace: monthly-quotas
  sync_rate: 60

  # Error response
  error_code: 429
  error_message: |
    {
      "error": "quota_exceeded",
      "message": "Monthly request quota exceeded",
      "quota_limit": 100000,
      "quota_remaining": 0,
      "quota_reset": "<first_of_next_month>"
    }
```

---

## Cost Attribution and Billing

### Request-Level Cost Tracking

**OpenTelemetry Integration for Cost Tracking:**
```python
from opentelemetry import trace
from genops import get_current_span

def track_request_cost(customer_id: str, endpoint: str, response_time: float):
    """Track cost metrics for billing."""
    span = get_current_span()

    if span:
        # Calculate cost based on response time and endpoint
        cost = calculate_request_cost(endpoint, response_time)

        # Add cost attributes to span
        span.set_attribute("genops.cost.request", cost)
        span.set_attribute("genops.cost.currency", "USD")
        span.set_attribute("genops.customer_id", customer_id)
        span.set_attribute("genops.billing_endpoint", endpoint)
        span.set_attribute("genops.response_time_ms", response_time)

    return cost

def calculate_request_cost(endpoint: str, response_time_ms: float) -> float:
    """Calculate cost based on endpoint and response time."""
    base_costs = {
        "/ai/v1/inference": 0.001,      # $0.001 per request
        "/ai/v1/embeddings": 0.0005,    # $0.0005 per request
        "/ai/v1/completions": 0.002,    # $0.002 per request
    }

    base_cost = base_costs.get(endpoint, 0.001)

    # Add latency premium for slow requests
    if response_time_ms > 1000:
        latency_premium = (response_time_ms - 1000) / 1000 * 0.0001
        return base_cost + latency_premium

    return base_cost
```

### Usage Analytics and Reporting

**Daily Usage Aggregation:**
```python
from datetime import datetime, timedelta
from genops.analytics import query_usage_metrics

def generate_daily_usage_report(customer_id: str, date: datetime):
    """Generate daily usage and cost report for customer."""
    metrics = query_usage_metrics(
        customer_id=customer_id,
        start_time=date,
        end_time=date + timedelta(days=1)
    )

    report = {
        "customer_id": customer_id,
        "date": date.isoformat(),
        "total_requests": metrics.request_count,
        "total_cost": metrics.total_cost,
        "cost_by_endpoint": metrics.cost_by_endpoint,
        "avg_response_time_ms": metrics.avg_response_time,
        "error_rate": metrics.error_rate,
        "top_projects": metrics.top_projects_by_cost
    }

    return report
```

### Customer Billing Integration

**Stripe Integration Example:**
```python
import stripe
from genops.billing import get_monthly_usage

def create_monthly_invoice(customer_id: str, month: str):
    """Create Stripe invoice for customer's monthly usage."""
    stripe.api_key = os.getenv("STRIPE_API_KEY")

    # Get usage data from GenOps
    usage = get_monthly_usage(customer_id=customer_id, month=month)

    # Create Stripe invoice
    invoice = stripe.Invoice.create(
        customer=customer_id,
        auto_advance=True,
        collection_method="charge_automatically"
    )

    # Add line items for each service
    for service, cost in usage.cost_by_service.items():
        stripe.InvoiceItem.create(
            customer=customer_id,
            invoice=invoice.id,
            amount=int(cost * 100),  # Convert to cents
            currency="usd",
            description=f"GenOps AI - {service}",
            metadata={
                "customer_id": customer_id,
                "service": service,
                "request_count": usage.request_count_by_service[service],
                "month": month
            }
        )

    return invoice.finalize_invoice(invoice.id)
```

### Chargeback Mechanisms

**Internal Chargeback Report:**
```python
def generate_chargeback_report(organization_id: str, month: str):
    """Generate chargeback report for internal cost allocation."""
    from genops.reporting import get_org_usage

    usage = get_org_usage(org_id=organization_id, month=month)

    chargeback_report = {
        "organization": organization_id,
        "month": month,
        "total_cost": usage.total_cost,
        "teams": []
    }

    # Break down by team
    for team, team_usage in usage.by_team.items():
        team_report = {
            "team": team,
            "total_cost": team_usage.total_cost,
            "request_count": team_usage.request_count,
            "projects": []
        }

        # Break down by project within team
        for project, project_usage in team_usage.by_project.items():
            team_report["projects"].append({
                "project": project,
                "cost": project_usage.total_cost,
                "requests": project_usage.request_count,
                "cost_by_provider": project_usage.cost_by_provider
            })

        chargeback_report["teams"].append(team_report)

    return chargeback_report
```

---

## API Gateway Selection Guide

### Choose Kong if:
✅ Enterprise features are required (rate limiting, auth, plugins)
✅ Need extensive plugin ecosystem
✅ Hybrid deployment (Kubernetes + VMs)
✅ Advanced traffic control and transformations

### Choose Ambassador if:
✅ Kubernetes-native is a priority
✅ GitOps workflow for configuration
✅ Integration with Envoy-based service mesh
✅ Developer self-service model

### Choose NGINX Ingress if:
✅ Simplicity and performance are key
✅ Wide community support needed
✅ Standard Ingress resources are sufficient
✅ Minimal operational overhead

### Choose Traefik if:
✅ Automatic service discovery is important
✅ Dynamic configuration without restarts
✅ Modern, cloud-native architecture
✅ Integrated Let's Encrypt support

---

## Next Steps

Ready to expose GenOps AI through an API gateway? Start with:

1. **Choose Your Gateway** - Evaluate Kong, Ambassador, NGINX, or Traefik
2. **Deploy Gateway Controller** - Install in your Kubernetes cluster
3. **Configure Authentication** - Set up JWT or API key validation
4. **Add Governance Context** - Extract and inject team/customer headers
5. **Implement Rate Limiting** - Configure per-customer quotas
6. **Monitor and Optimize** - Track request metrics and costs

Return to [Kubernetes Getting Started](kubernetes-getting-started.md) for the complete deployment overview.

---

## Support

- **Documentation:** [GenOps AI Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community:** [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
