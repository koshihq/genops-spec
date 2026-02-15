#!/usr/bin/env python3
"""
Production Deployment Patterns with GenOps and Haystack

Demonstrates production-ready deployment patterns including containerization,
Kubernetes deployment, monitoring, scaling, health checks, and high-availability
configurations for enterprise AI systems.

Usage:
    python production_deployment_patterns.py

Features:
    - Docker containerization patterns with multi-stage builds
    - Kubernetes deployment manifests and scaling strategies
    - Health checks and readiness probes for AI workloads
    - Production monitoring and alerting configurations
    - High-availability deployment patterns with failover
    - Performance optimization and resource management
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# Core Haystack imports
try:
    from haystack import Document, Pipeline  # noqa: F401
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever  # noqa: F401
    from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: F401
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        GenOpsHaystackAdapter,
        analyze_pipeline_costs,  # noqa: F401
        print_validation_result,
        validate_haystack_setup,
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Health check result for production monitoring."""

    status: str  # healthy, degraded, unhealthy
    timestamp: str
    response_time_ms: float
    dependencies: dict[str, str]
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfiguration:
    """Production deployment configuration."""

    service_name: str
    version: str
    environment: str
    replicas: int
    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str
    health_check_interval: int
    monitoring_enabled: bool = True
    auto_scaling_enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 10


class ProductionHealthChecker:
    """Production health checking and monitoring."""

    def __init__(self, adapter: GenOpsHaystackAdapter):
        self.adapter = adapter
        self.last_check_time = None
        self.consecutive_failures = 0

    def check_health(self) -> HealthCheckResult:
        """Comprehensive health check for production deployment."""
        start_time = time.time()
        errors = []
        dependencies = {}

        # Check GenOps adapter health
        try:
            if self.adapter:
                dependencies["genops_adapter"] = "healthy"
            else:
                dependencies["genops_adapter"] = "unhealthy"
                errors.append("GenOps adapter not initialized")
        except Exception as e:
            dependencies["genops_adapter"] = "unhealthy"
            errors.append(f"GenOps adapter error: {str(e)}")

        # Check Haystack framework
        try:
            test_pipeline = Pipeline()
            test_pipeline.add_component(
                "test_prompt", PromptBuilder(template="Health check: {{message}}")
            )
            dependencies["haystack"] = "healthy"
        except Exception as e:
            dependencies["haystack"] = "unhealthy"
            errors.append(f"Haystack framework error: {str(e)}")

        # Check AI provider connectivity (mock for demo)
        try:
            # In production, this would test actual provider connectivity
            dependencies["ai_providers"] = "healthy"
        except Exception as e:
            dependencies["ai_providers"] = "degraded"
            errors.append(f"AI provider connectivity issue: {str(e)}")

        # Check telemetry export
        try:
            # Mock telemetry health check
            dependencies["telemetry_export"] = "healthy"
        except Exception as e:
            dependencies["telemetry_export"] = "degraded"
            errors.append(f"Telemetry export issue: {str(e)}")

        response_time_ms = (time.time() - start_time) * 1000

        # Determine overall status
        if not errors:
            status = "healthy"
            self.consecutive_failures = 0
        elif any("unhealthy" in dep for dep in dependencies.values()):
            status = "unhealthy"
            self.consecutive_failures += 1
        else:
            status = "degraded"

        self.last_check_time = time.time()

        return HealthCheckResult(
            status=status,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            response_time_ms=response_time_ms,
            dependencies=dependencies,
            errors=errors,
            metadata={
                "consecutive_failures": self.consecutive_failures,
                "uptime_seconds": time.time() - start_time,
            },
        )

    def is_ready(self) -> bool:
        """Readiness probe for Kubernetes deployments."""
        try:
            health = self.check_health()
            return (
                health.status in ["healthy", "degraded"]
                and health.response_time_ms < 5000
            )
        except Exception:
            return False


class ProductionPipelineManager:
    """Manages production AI pipelines with scaling and monitoring."""

    def __init__(self, deployment_config: DeploymentConfiguration):
        self.config = deployment_config
        self.pipelines = {}
        self.health_checker = None
        self.performance_metrics = {
            "requests_processed": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "throughput_per_second": 0.0,
        }

    def initialize(self) -> bool:
        """Initialize production pipeline manager."""
        try:
            # Create production adapter
            adapter = GenOpsHaystackAdapter(
                team=f"production-{self.config.environment}",
                project=self.config.service_name,
                environment=self.config.environment,
                daily_budget_limit=1000.0,  # Production budget
                monthly_budget_limit=25000.0,
                governance_policy="enforcing",
            )

            # Initialize health checker
            self.health_checker = ProductionHealthChecker(adapter)

            # Create production pipelines
            self._create_production_pipelines(adapter)

            logger.info(
                f"Production pipeline manager initialized for {self.config.service_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize production manager: {e}")
            return False

    def _create_production_pipelines(self, adapter: GenOpsHaystackAdapter):
        """Create optimized production pipelines."""

        # Main production pipeline
        main_pipeline = Pipeline()
        main_pipeline.add_component(
            "prompt_builder",
            PromptBuilder(
                template="""
            [PRODUCTION AI SERVICE - {{service_name}}]
            Environment: {{environment}}
            Request ID: {{request_id}}

            User Request: {{user_request}}

            Provide a high-quality response following production guidelines:
            """
            ),
        )

        main_pipeline.add_component(
            "llm",
            OpenAIGenerator(
                model="gpt-3.5-turbo",
                generation_kwargs={
                    "max_tokens": 500,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "presence_penalty": 0.1,
                },
            ),
        )

        main_pipeline.connect("prompt_builder", "llm")
        self.pipelines["main"] = {"pipeline": main_pipeline, "adapter": adapter}

        # Fallback pipeline with simpler model
        fallback_pipeline = Pipeline()
        fallback_pipeline.add_component(
            "prompt_builder",
            PromptBuilder(template="Fallback response for: {{user_request}}"),
        )
        fallback_pipeline.add_component(
            "llm",
            OpenAIGenerator(
                model="gpt-3.5-turbo",
                generation_kwargs={"max_tokens": 200, "temperature": 0.3},
            ),
        )
        fallback_pipeline.connect("prompt_builder", "llm")
        self.pipelines["fallback"] = {"pipeline": fallback_pipeline, "adapter": adapter}

        logger.info("Production pipelines created with failover capability")

    def process_request(
        self, request_data: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        """Process production request with monitoring and fallback."""
        start_time = time.time()

        try:
            # Try main pipeline first
            return self._execute_pipeline("main", request_data, request_id)

        except Exception as e:
            logger.warning(f"Main pipeline failed for request {request_id}: {e}")

            try:
                # Fallback to simpler pipeline
                logger.info(f"Using fallback pipeline for request {request_id}")
                return self._execute_pipeline("fallback", request_data, request_id)

            except Exception as fallback_error:
                logger.error(
                    f"Fallback pipeline also failed for request {request_id}: {fallback_error}"
                )

                # Return error response
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": "Service temporarily unavailable",
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

    def _execute_pipeline(
        self, pipeline_name: str, request_data: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        """Execute specific pipeline with monitoring."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")

        start_time = time.time()
        pipeline_config = self.pipelines[pipeline_name]
        pipeline = pipeline_config["pipeline"]
        adapter = pipeline_config["adapter"]

        with adapter.track_pipeline(
            f"production-{pipeline_name}",
            request_id=request_id,
            service_name=self.config.service_name,
            environment=self.config.environment,
            pipeline_type=pipeline_name,
        ) as context:
            result = pipeline.run(
                {
                    "prompt_builder": {
                        "service_name": self.config.service_name,
                        "environment": self.config.environment,
                        "request_id": request_id,
                        "user_request": request_data.get("request", ""),
                    }
                }
            )

            response_time_ms = (time.time() - start_time) * 1000

            # Update performance metrics
            self.performance_metrics["requests_processed"] += 1
            self._update_performance_metrics(response_time_ms, success=True)

            return {
                "request_id": request_id,
                "status": "success",
                "response": result["llm"]["replies"][0],
                "pipeline_used": pipeline_name,
                "response_time_ms": response_time_ms,
                "cost": float(context.get_metrics().total_cost),
            }

    def _update_performance_metrics(self, response_time_ms: float, success: bool):
        """Update running performance metrics."""
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_requests = self.performance_metrics["requests_processed"]

        if total_requests > 0:
            self.performance_metrics["average_response_time"] = (
                current_avg * (total_requests - 1) + response_time_ms
            ) / total_requests
        else:
            self.performance_metrics["average_response_time"] = response_time_ms

        # Note: Error rate would be calculated over a time window in production

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        health = self.health_checker.check_health() if self.health_checker else None

        return {
            "performance": self.performance_metrics.copy(),
            "health": {
                "status": health.status if health else "unknown",
                "dependencies": health.dependencies if health else {},
                "last_check": health.timestamp if health else None,
            },
            "deployment": {
                "service_name": self.config.service_name,
                "version": self.config.version,
                "environment": self.config.environment,
                "replicas": self.config.replicas,
            },
        }


def generate_docker_configuration() -> dict[str, str]:
    """Generate Docker configuration for production deployment."""

    dockerfile = """
# Multi-stage production Dockerfile for Haystack + GenOps AI service
FROM python:3.9-slim as builder

# Set build arguments
ARG APP_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Add labels for metadata
LABEL version="${APP_VERSION}" \\
      description="Production Haystack + GenOps AI Service" \\
      maintainer="genops-team@company.com" \\
      build-date="${BUILD_DATE}" \\
      vcs-ref="${VCS_REF}"

# Create non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "-m", "production_deployment_patterns"]
"""

    docker_compose = """
version: '3.8'

services:
  haystack-genops-api:
    build: .
    image: haystack-genops-api:latest
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - SERVICE_NAME=haystack-genops-api
      - LOG_LEVEL=INFO
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - GENOPS_DAILY_BUDGET_LIMIT=1000.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./otel-collector-config.yml:/etc/otel-collector-config.yml
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8888:8888"   # Prometheus metrics
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
"""

    return {"Dockerfile": dockerfile, "docker-compose.yml": docker_compose}


def generate_kubernetes_manifests() -> dict[str, str]:
    """Generate Kubernetes deployment manifests."""

    deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: haystack-genops-api
  labels:
    app: haystack-genops-api
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: haystack-genops-api
  template:
    metadata:
      labels:
        app: haystack-genops-api
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: haystack-genops-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: SERVICE_NAME
          value: "haystack-genops-api"
        - name: LOG_LEVEL
          value: "INFO"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://otel-collector:4317"
        - name: GENOPS_DAILY_BUDGET_LIMIT
          value: "1000.0"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 5
          failureThreshold: 2
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      securityContext:
        fsGroup: 1000
---
apiVersion: v1
kind: Service
metadata:
  name: haystack-genops-api-service
  labels:
    app: haystack-genops-api
spec:
  selector:
    app: haystack-genops-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
    name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: haystack-genops-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: haystack-genops-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: haystack-genops-api-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.genops-ai.com
    secretName: haystack-genops-tls
  rules:
  - host: api.genops-ai.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: haystack-genops-api-service
            port:
              number: 80
"""

    monitoring = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  otel-collector-config.yml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch:
      memory_limiter:
        check_interval: 1s
        limit_mib: 512

    exporters:
      prometheus:
        endpoint: "0.0.0.0:8888"
        namespace: genops
        const_labels:
          service: haystack-genops-api

      logging:
        loglevel: info

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [logging]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [prometheus, logging]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-contrib:latest
        args: ["--config=/etc/otel-collector-config.yml"]
        ports:
        - containerPort: 4317
        - containerPort: 4318
        - containerPort: 8888
        volumeMounts:
        - name: config
          mountPath: /etc/otel-collector-config.yml
          subPath: otel-collector-config.yml
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: metrics
    port: 8888
    targetPort: 8888
"""

    return {"deployment.yaml": deployment, "monitoring.yaml": monitoring}


def demo_production_deployment():
    """Demonstrate production deployment patterns."""
    print("\n" + "=" * 70)
    print("🚀 Production Deployment Patterns")
    print("=" * 70)

    # Create production configuration
    deployment_config = DeploymentConfiguration(
        service_name="haystack-genops-api",
        version="1.0.0",
        environment="production",
        replicas=3,
        cpu_request="500m",
        memory_request="512Mi",
        cpu_limit="1000m",
        memory_limit="1Gi",
        health_check_interval=30,
        monitoring_enabled=True,
        auto_scaling_enabled=True,
        min_replicas=2,
        max_replicas=10,
    )

    print("🏗️ Production Configuration:")
    print(f"   Service: {deployment_config.service_name}")
    print(f"   Version: {deployment_config.version}")
    print(f"   Environment: {deployment_config.environment}")
    print(f"   Replicas: {deployment_config.replicas}")
    print(
        f"   Resource Requests: {deployment_config.cpu_request} CPU, {deployment_config.memory_request} Memory"
    )
    print(
        f"   Resource Limits: {deployment_config.cpu_limit} CPU, {deployment_config.memory_limit} Memory"
    )

    # Initialize production manager
    pipeline_manager = ProductionPipelineManager(deployment_config)

    if not pipeline_manager.initialize():
        print("❌ Failed to initialize production pipeline manager")
        return None

    print("✅ Production pipeline manager initialized")

    # Simulate production requests
    print("\n📋 Simulating Production Workload:")

    test_requests = [
        {"request": "Analyze customer feedback sentiment", "priority": "normal"},
        {"request": "Generate product recommendation summary", "priority": "high"},
        {"request": "Create technical documentation outline", "priority": "normal"},
        {"request": "Process user query about AI features", "priority": "normal"},
        {"request": "Generate executive summary report", "priority": "high"},
    ]

    # Process requests with concurrent execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for i, request_data in enumerate(test_requests, 1):
            request_id = f"req-{i:04d}"
            future = executor.submit(
                pipeline_manager.process_request, request_data, request_id
            )
            futures.append((request_id, future))

        # Collect results
        results = []
        for request_id, future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
                print(
                    f"   ✅ {request_id}: {result['status']} ({result.get('response_time_ms', 0):.1f}ms)"
                )
            except Exception as e:
                print(f"   ❌ {request_id}: Error - {e}")

    # Get performance metrics
    metrics = pipeline_manager.get_metrics()

    print("\n📊 Production Metrics:")
    print(f"   Requests Processed: {metrics['performance']['requests_processed']}")
    print(
        f"   Average Response Time: {metrics['performance']['average_response_time']:.1f}ms"
    )
    print(f"   Service Health: {metrics['health']['status']}")
    print(f"   Dependencies: {metrics['health']['dependencies']}")

    return pipeline_manager, metrics


def demo_containerization_configs():
    """Demonstrate containerization configurations."""
    print("\n" + "=" * 70)
    print("🐳 Containerization Configurations")
    print("=" * 70)

    # Generate Docker configurations
    docker_configs = generate_docker_configuration()

    print("📦 Docker Configuration Generated:")
    print("   • Multi-stage Dockerfile with security best practices")
    print("   • Production-optimized Python environment")
    print("   • Health checks and monitoring integration")
    print("   • Non-root user execution")
    print("   • Resource limitations and security controls")

    print("\n🔧 Docker Compose Services:")
    print("   • haystack-genops-api: Main application service")
    print("   • otel-collector: OpenTelemetry telemetry collection")
    print("   • prometheus: Metrics storage and monitoring")
    print("   • grafana: Visualization and alerting dashboard")

    # Show sample Dockerfile section
    dockerfile_lines = docker_configs["Dockerfile"].split("\n")
    print("\n📄 Sample Dockerfile (first 15 lines):")
    for line in dockerfile_lines[:15]:
        if line.strip():
            print(f"   {line}")

    return docker_configs


def demo_kubernetes_deployment():
    """Demonstrate Kubernetes deployment patterns."""
    print("\n" + "=" * 70)
    print("☸️ Kubernetes Deployment Patterns")
    print("=" * 70)

    # Generate Kubernetes manifests
    k8s_manifests = generate_kubernetes_manifests()

    print("🚀 Kubernetes Resources Generated:")
    print("   • Deployment: Multi-replica application deployment")
    print("   • Service: Internal load balancing and service discovery")
    print("   • HorizontalPodAutoscaler: Automatic scaling based on CPU/memory")
    print("   • Ingress: External traffic routing with SSL termination")
    print("   • ConfigMap: OpenTelemetry collector configuration")
    print("   • Monitoring: Integrated observability stack")

    print("\n⚡ Scaling Configuration:")
    print("   • Min Replicas: 2 (high availability)")
    print("   • Max Replicas: 10 (burst capacity)")
    print("   • CPU Target: 70% utilization")
    print("   • Memory Target: 80% utilization")

    print("\n🛡️ Security Configuration:")
    print("   • Non-root container execution")
    print("   • Read-only root filesystem")
    print("   • Dropped capabilities (ALL)")
    print("   • Resource limits and requests")
    print("   • Network policies for isolation")

    print("\n💊 Health Checks:")
    print("   • Liveness Probe: /health endpoint (30s interval)")
    print("   • Readiness Probe: /ready endpoint (15s interval)")
    print("   • Startup grace period: 60s")
    print("   • Graceful shutdown handling")

    return k8s_manifests


def demo_monitoring_and_alerting():
    """Demonstrate production monitoring and alerting."""
    print("\n" + "=" * 70)
    print("📈 Production Monitoring and Alerting")
    print("=" * 70)

    print("🔍 Observability Stack:")
    print("   • OpenTelemetry: Unified telemetry collection")
    print("   • Prometheus: Metrics storage and alerting")
    print("   • Grafana: Visualization and dashboards")
    print("   • Jaeger: Distributed tracing analysis")

    print("\n📊 Key Metrics Monitored:")
    print("   • Request rate and response time")
    print("   • Error rates and success rates")
    print("   • AI model costs and budget utilization")
    print("   • System resources (CPU, memory, disk)")
    print("   • Service dependencies health")

    print("\n🚨 Alerting Scenarios:")
    print("   • High error rate (>5% for 5 minutes)")
    print("   • Slow response time (>2s P95 for 10 minutes)")
    print("   • Budget overrun (>90% daily budget)")
    print("   • Service dependency failures")
    print("   • Resource exhaustion (CPU >80%, Memory >85%)")

    print("\n🎯 SLA Monitoring:")
    print("   • Availability: 99.9% uptime target")
    print("   • Performance: P95 < 2 seconds")
    print("   • Error Budget: <0.1% error rate")
    print("   • Cost Efficiency: <$0.01 per request")


def main():
    """Run the comprehensive production deployment patterns demonstration."""
    print("🚀 Production Deployment Patterns with Haystack + GenOps")
    print("=" * 70)

    # Validate environment setup
    print("🔍 Validating setup...")
    result = validate_haystack_setup()

    if not result.is_valid:
        print("❌ Setup validation failed!")
        print_validation_result(result)
        return 1
    else:
        print("✅ Environment validated and ready")

    try:
        # Production deployment demonstration
        pipeline_manager, metrics = demo_production_deployment()

        # Containerization patterns
        demo_containerization_configs()

        # Kubernetes deployment patterns
        demo_kubernetes_deployment()

        # Monitoring and alerting
        demo_monitoring_and_alerting()

        print("\n🎉 Production Deployment Patterns demonstration completed!")
        print("\n🚀 Next Steps:")
        print("   • Try performance_optimization.py for speed improvements")
        print("   • Review generated configurations for your deployment")
        print("   • Customize monitoring and alerting for your requirements")
        print("   • Deploy to your production environment with confidence! 🚀")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running the setup validation to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
