#!/usr/bin/env python3
"""
PromptLayer Production Deployment Patterns with GenOps

This example demonstrates enterprise-ready production deployment patterns for PromptLayer
with GenOps governance, including Docker containerization, Kubernetes deployment,
monitoring integration, and scaling strategies.

This is the Level 3 (2-hour) example - Production deployment and enterprise patterns.

Usage:
    python production_patterns.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    export OPENAI_API_KEY="your-openai-key"  # For actual LLM calls

    # Production environment variables
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    export GENOPS_ENVIRONMENT="production"
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:14268/api/traces"

    # Optional: Advanced monitoring
    export PROMETHEUS_GATEWAY_URL="http://prometheus-gateway:9091"
    export DATADOG_API_KEY="your-datadog-key"
"""

import asyncio
import logging
import os
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionMetrics:
    """Production-ready metrics with full observability context."""

    service_name: str
    service_version: str
    deployment_id: str

    # Request metrics
    request_count: int = 0
    error_count: int = 0
    success_count: int = 0

    # Performance metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Cost metrics
    total_cost_usd: float = 0.0
    avg_cost_per_request: float = 0.0
    cost_by_team: dict[str, float] = field(default_factory=dict)

    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    active_connections: int = 0

    # Business metrics
    customer_count: int = 0
    revenue_attributed: float = 0.0

    # Health metrics
    health_status: str = "healthy"
    last_health_check: Optional[datetime] = None


class ProductionPromptLayerService:
    """Production-ready PromptLayer service with enterprise patterns."""

    def __init__(
        self,
        service_name: str = "promptlayer-service",
        service_version: str = "1.0.0",
        environment: str = "production",
        enable_circuit_breaker: bool = True,
        enable_rate_limiting: bool = True,
        enable_caching: bool = True,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.deployment_id = f"{service_name}-{int(time.time())}"

        # Feature flags
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_caching = enable_caching

        # Production state
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.metrics = ProductionMetrics(
            service_name=service_name,
            service_version=service_version,
            deployment_id=self.deployment_id,
        )

        # Thread pool for concurrent request handling
        self.executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="promptlayer-worker"
        )

        # Request queue for load balancing
        self.request_queue = queue.Queue(maxsize=1000)

        # Initialize adapters
        self.adapter = None
        self._initialize_adapter()

        # Monitoring setup
        self.metrics_export_interval = 60  # seconds
        self.health_check_interval = 30  # seconds

        logger.info(f"Production PromptLayer service initialized: {self.deployment_id}")

    def _initialize_adapter(self):
        """Initialize PromptLayer adapter with production configuration."""
        try:
            from genops.providers.promptlayer import (
                GovernancePolicy,
                instrument_promptlayer,
            )

            self.adapter = instrument_promptlayer(
                promptlayer_api_key=os.getenv("PROMPTLAYER_API_KEY"),
                team=os.getenv("GENOPS_TEAM", "production-team"),
                project=os.getenv("GENOPS_PROJECT", "promptlayer-service"),
                environment=self.environment,
                customer_id=None,  # Will be set per request
                enable_governance=True,
                daily_budget_limit=1000.0,  # $1000 daily limit for production
                max_operation_cost=5.0,  # $5 max per operation
                governance_policy=GovernancePolicy.ENFORCED,  # Strict enforcement in production
                enable_cost_alerts=True,
            )

            logger.info("PromptLayer adapter initialized for production")

        except ImportError as e:
            logger.error(f"Failed to initialize PromptLayer adapter: {e}")
            raise
        except Exception as e:
            logger.error(f"Adapter initialization error: {e}")
            raise

    async def start_service(self):
        """Start the production service with all monitoring components."""
        logger.info(f"Starting production PromptLayer service: {self.deployment_id}")

        self.is_running = True

        # Start background tasks
        tasks = [
            asyncio.create_task(self._metrics_export_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._request_processor_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]

        try:
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Wait for shutdown signal
            while self.is_running and not self.shutdown_event.is_set():
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Service error: {e}")
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Shutdown executor
            self.executor.shutdown(wait=True)

            logger.info("Production service shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
        self.shutdown_event.set()

    async def _metrics_export_loop(self):
        """Continuously export metrics to observability platforms."""
        while self.is_running:
            try:
                await self._export_metrics()
                await asyncio.sleep(self.metrics_export_interval)
            except Exception as e:
                logger.error(f"Metrics export error: {e}")
                await asyncio.sleep(10)  # Brief retry delay

    async def _health_check_loop(self):
        """Continuously monitor service health."""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief retry delay

    async def _request_processor_loop(self):
        """Process queued requests with load balancing."""
        while self.is_running:
            try:
                # Process requests from queue
                while not self.request_queue.empty() and self.is_running:
                    request_data = self.request_queue.get_nowait()

                    # Submit to thread pool for processing
                    self.executor.submit(self._process_request_sync, request_data)

                    # Don't wait for completion to maintain throughput

                await asyncio.sleep(0.1)  # Brief pause to prevent busy waiting

            except Exception as e:
                logger.error(f"Request processor error: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self):
        """Periodic cleanup and maintenance tasks."""
        while self.is_running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)

    def _process_request_sync(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Synchronously process a single request."""
        start_time = time.time()
        request_id = request_data.get("request_id", f"req_{int(time.time() * 1000)}")

        try:
            # Circuit breaker check
            if self.enable_circuit_breaker and not self._circuit_breaker_check():
                raise Exception("Circuit breaker open")

            # Rate limiting check
            if self.enable_rate_limiting and not self._rate_limit_check():
                raise Exception("Rate limit exceeded")

            # Process with governance
            with self.adapter.track_prompt_operation(
                prompt_name=request_data.get("prompt_name", "production_prompt"),
                operation_type="production_request",
                operation_name=f"process_{request_id}",
                customer_id=request_data.get("customer_id"),
                tags=request_data.get("tags", []),
            ) as span:
                result = self.adapter.run_prompt_with_governance(
                    prompt_name=request_data["prompt_name"],
                    input_variables=request_data["input_variables"],
                    tags=request_data.get("tags", []),
                )

                # Update metrics
                duration = (time.time() - start_time) * 1000
                self.metrics.request_count += 1
                self.metrics.success_count += 1
                self.metrics.total_latency_ms += duration
                self.metrics.avg_latency_ms = (
                    self.metrics.total_latency_ms / self.metrics.request_count
                )

                # Cost tracking
                cost = span.estimated_cost if hasattr(span, "estimated_cost") else 0.0
                self.metrics.total_cost_usd += cost
                self.metrics.avg_cost_per_request = (
                    self.metrics.total_cost_usd / self.metrics.request_count
                )

                # Team attribution
                team = request_data.get("team", "unknown")
                if team not in self.metrics.cost_by_team:
                    self.metrics.cost_by_team[team] = 0.0
                self.metrics.cost_by_team[team] += cost

                logger.info(
                    f"Request {request_id} processed successfully (Duration: {duration:.2f}ms, Cost: ${cost:.6f})"
                )

                return {
                    "request_id": request_id,
                    "status": "success",
                    "result": result,
                    "duration_ms": duration,
                    "cost_usd": cost,
                }

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.metrics.request_count += 1
            self.metrics.error_count += 1

            logger.error(
                f"Request {request_id} failed: {e} (Duration: {duration:.2f}ms)"
            )

            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "duration_ms": duration,
            }

    def _circuit_breaker_check(self) -> bool:
        """Check circuit breaker status."""
        if self.metrics.request_count == 0:
            return True

        error_rate = self.metrics.error_count / self.metrics.request_count
        return error_rate < 0.5  # Open circuit if >50% errors

    def _rate_limit_check(self) -> bool:
        """Check rate limiting constraints."""
        # Simple rate limiting - could be enhanced with sliding window
        return self.request_queue.qsize() < 900  # Leave buffer in queue

    async def _export_metrics(self):
        """Export metrics to observability platforms."""
        metrics_payload = {
            "timestamp": datetime.now().isoformat(),
            "service": {
                "name": self.service_name,
                "version": self.service_version,
                "deployment_id": self.deployment_id,
                "environment": self.environment,
            },
            "metrics": {
                "requests": {
                    "total": self.metrics.request_count,
                    "success": self.metrics.success_count,
                    "error": self.metrics.error_count,
                    "error_rate": self.metrics.error_count
                    / max(1, self.metrics.request_count),
                },
                "latency": {
                    "avg_ms": self.metrics.avg_latency_ms,
                    "total_ms": self.metrics.total_latency_ms,
                },
                "cost": {
                    "total_usd": self.metrics.total_cost_usd,
                    "avg_per_request": self.metrics.avg_cost_per_request,
                    "by_team": self.metrics.cost_by_team,
                },
                "health": {
                    "status": self.metrics.health_status,
                    "active_connections": self.metrics.active_connections,
                },
            },
        }

        # Export to multiple platforms
        await self._export_to_prometheus(metrics_payload)
        await self._export_to_datadog(metrics_payload)
        await self._export_to_otel(metrics_payload)

        logger.debug(
            f"Metrics exported: {self.metrics.request_count} requests, ${self.metrics.total_cost_usd:.6f} total cost"
        )

    async def _export_to_prometheus(self, metrics: dict[str, Any]):
        """Export metrics to Prometheus via pushgateway."""
        try:
            # Simulate Prometheus export
            prometheus_url = os.getenv("PROMETHEUS_GATEWAY_URL")
            if prometheus_url:
                logger.debug("Exporting to Prometheus (simulated)")
                # In production: Use prometheus_client to push metrics
        except Exception as e:
            logger.warning(f"Prometheus export failed: {e}")

    async def _export_to_datadog(self, metrics: dict[str, Any]):
        """Export metrics to Datadog."""
        try:
            datadog_key = os.getenv("DATADOG_API_KEY")
            if datadog_key:
                logger.debug("Exporting to Datadog (simulated)")
                # In production: Use datadog library to send metrics
        except Exception as e:
            logger.warning(f"Datadog export failed: {e}")

    async def _export_to_otel(self, metrics: dict[str, Any]):
        """Export metrics via OpenTelemetry."""
        try:
            otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otel_endpoint:
                logger.debug("Exporting via OpenTelemetry (simulated)")
                # In production: Use OpenTelemetry SDK to export
        except Exception as e:
            logger.warning(f"OpenTelemetry export failed: {e}")

    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            # Check adapter health
            adapter_healthy = self.adapter is not None

            # Check circuit breaker
            circuit_healthy = self._circuit_breaker_check()

            # Check queue capacity
            queue_healthy = self.request_queue.qsize() < 950

            # Determine overall health
            if adapter_healthy and circuit_healthy and queue_healthy:
                self.metrics.health_status = "healthy"
            elif adapter_healthy and queue_healthy:
                self.metrics.health_status = "degraded"
            else:
                self.metrics.health_status = "unhealthy"

            self.metrics.last_health_check = datetime.now()

            logger.debug(f"Health check: {self.metrics.health_status}")

        except Exception as e:
            self.metrics.health_status = "unhealthy"
            logger.error(f"Health check failed: {e}")

    async def _perform_cleanup(self):
        """Perform periodic cleanup tasks."""
        try:
            # Reset metrics if they get too large
            if self.metrics.request_count > 1000000:  # 1M requests
                logger.info("Resetting metrics counters")
                self.metrics = ProductionMetrics(
                    service_name=self.service_name,
                    service_version=self.service_version,
                    deployment_id=self.deployment_id,
                )

            logger.debug("Cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def submit_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Submit a request for processing."""
        try:
            request_id = f"req_{int(time.time() * 1000)}"
            request_data["request_id"] = request_id

            # Add to queue
            self.request_queue.put(request_data, timeout=1.0)

            return {
                "request_id": request_id,
                "status": "queued",
                "queue_position": self.request_queue.qsize(),
            }

        except queue.Full:
            logger.warning("Request queue full")
            return {"status": "rejected", "reason": "queue_full"}


def demonstrate_production_deployment():
    """
    Demonstrates production deployment patterns.

    Shows enterprise-ready deployment with monitoring, scaling,
    and governance integration for PromptLayer operations.
    """
    print("🏭 Production PromptLayer Deployment Patterns")
    print("=" * 50)

    try:
        # Initialize production service
        service = ProductionPromptLayerService(
            service_name="promptlayer-prod-service",
            service_version="1.2.3",
            environment="production",
            enable_circuit_breaker=True,
            enable_rate_limiting=True,
            enable_caching=True,
        )
        print("✅ Production service initialized")

    except ImportError as e:
        print(f"❌ Failed to initialize production service: {e}")
        print("💡 Fix: Run 'pip install genops[promptlayer]'")
        return False
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        return False

    print("\n🚀 Running Production Deployment Scenarios...")
    print("-" * 50)

    # Simulate production request patterns
    print("\n1️⃣ Production Request Processing")
    try:
        # Sample production requests
        production_requests = [
            {
                "prompt_name": "customer_support_v3",
                "input_variables": {
                    "query": "Billing issue with enterprise account",
                    "priority": "high",
                },
                "customer_id": "enterprise_customer_001",
                "team": "customer-success",
                "tags": ["billing", "enterprise", "high-priority"],
            },
            {
                "prompt_name": "product_recommendation_v2",
                "input_variables": {
                    "user_profile": "premium_user",
                    "category": "productivity",
                },
                "customer_id": "premium_user_456",
                "team": "product-team",
                "tags": ["recommendation", "personalization"],
            },
            {
                "prompt_name": "content_moderation_v4",
                "input_variables": {
                    "content": "User-generated content review",
                    "severity": "standard",
                },
                "customer_id": "platform_moderation",
                "team": "trust-safety",
                "tags": ["moderation", "safety"],
            },
        ]

        # Process requests synchronously for demo
        results = []
        for i, request in enumerate(production_requests):
            print(f"   Processing request {i + 1}: {request['prompt_name']}")
            result = service._process_request_sync(request)
            results.append(result)

            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"   {status_icon} {result['request_id']}: {result['status']}")
            if result["status"] == "success":
                print(
                    f"      Duration: {result['duration_ms']:.2f}ms, Cost: ${result['cost_usd']:.6f}"
                )

        print("\n   📊 Request Processing Summary:")
        successful = sum(1 for r in results if r["status"] == "success")
        print(
            f"      Success Rate: {successful}/{len(results)} ({successful / len(results) * 100:.1f}%)"
        )
        print(f"      Total Service Requests: {service.metrics.request_count}")
        print(f"      Total Cost: ${service.metrics.total_cost_usd:.6f}")

    except Exception as e:
        print(f"❌ Production request processing failed: {e}")
        return False

    # Demonstrate monitoring integration
    print("\n2️⃣ Production Monitoring Integration")
    try:
        # Simulate metrics export
        print("   📊 Exporting metrics to observability platforms:")

        # Get current metrics
        {
            "timestamp": datetime.now().isoformat(),
            "service_metrics": {
                "requests_total": service.metrics.request_count,
                "requests_success": service.metrics.success_count,
                "requests_error": service.metrics.error_count,
                "latency_avg_ms": service.metrics.avg_latency_ms,
                "cost_total_usd": service.metrics.total_cost_usd,
                "health_status": service.metrics.health_status,
            },
            "governance_metrics": {
                "cost_by_team": service.metrics.cost_by_team,
                "deployment_id": service.deployment_id,
                "environment": service.environment,
            },
        }

        print("      • Prometheus: Service metrics and SLI/SLO tracking")
        print("      • Datadog: APM traces and custom business metrics")
        print("      • OpenTelemetry: Distributed tracing and span data")
        print("      • Internal Dashboard: Real-time cost and usage analytics")

        print("\n   📈 Current Production Metrics:")
        print(f"      Service Health: {service.metrics.health_status}")
        print(f"      Request Count: {service.metrics.request_count}")
        print(f"      Average Latency: {service.metrics.avg_latency_ms:.2f}ms")
        print(f"      Total Cost: ${service.metrics.total_cost_usd:.6f}")
        print(
            f"      Error Rate: {service.metrics.error_count / max(1, service.metrics.request_count) * 100:.1f}%"
        )

        # Team cost attribution
        if service.metrics.cost_by_team:
            print("      Cost by Team:")
            for team, cost in service.metrics.cost_by_team.items():
                print(f"        • {team}: ${cost:.6f}")

    except Exception as e:
        print(f"❌ Monitoring integration failed: {e}")
        return False

    # Demonstrate scaling patterns
    print("\n3️⃣ Production Scaling Patterns")
    try:
        print("   🔧 Enterprise Scaling Configuration:")

        scaling_config = {
            "horizontal_scaling": {
                "min_replicas": 3,
                "max_replicas": 20,
                "target_cpu_utilization": "70%",
                "target_memory_utilization": "80%",
                "scale_up_threshold": "avg_latency_ms > 2000",
                "scale_down_threshold": "avg_latency_ms < 500",
            },
            "vertical_scaling": {
                "min_resources": {"cpu": "500m", "memory": "1Gi"},
                "max_resources": {"cpu": "4", "memory": "8Gi"},
                "resource_requests": {"cpu": "1", "memory": "2Gi"},
            },
            "circuit_breaker": {
                "enabled": service.enable_circuit_breaker,
                "failure_threshold": "50%",
                "recovery_timeout": "60s",
                "half_open_max_calls": 10,
            },
            "rate_limiting": {
                "enabled": service.enable_rate_limiting,
                "requests_per_second": 100,
                "burst_capacity": 200,
                "queue_size": 1000,
            },
            "governance_limits": {
                "daily_budget": "$1000",
                "max_operation_cost": "$5",
                "policy_enforcement": "enforced",
                "cost_alerts": True,
            },
        }

        for category, config in scaling_config.items():
            print(f"      • {category.replace('_', ' ').title()}:")
            for key, value in config.items():
                print(f"        - {key}: {value}")

        print("\n   🎯 Scaling Decision Logic:")
        current_latency = service.metrics.avg_latency_ms
        if current_latency > 2000:
            print(
                f"      ⬆️ SCALE UP: Current latency {current_latency:.0f}ms exceeds 2000ms threshold"
            )
        elif current_latency < 500:
            print(
                f"      ⬇️ SCALE DOWN: Current latency {current_latency:.0f}ms below 500ms threshold"
            )
        else:
            print(
                f"      ➡️ MAINTAIN: Current latency {current_latency:.0f}ms within optimal range"
            )

        # Cost-based scaling
        avg_cost = service.metrics.avg_cost_per_request
        if avg_cost > 0.10:
            print(
                f"      💰 COST OPTIMIZATION: Average cost ${avg_cost:.6f} suggests model optimization"
            )
        else:
            print(
                f"      💰 COST OPTIMAL: Average cost ${avg_cost:.6f} within target range"
            )

    except Exception as e:
        print(f"❌ Scaling patterns demo failed: {e}")
        return False

    return True


def show_docker_kubernetes_configs():
    """Show Docker and Kubernetes configuration examples."""
    print("\n🐳 Docker & Kubernetes Configuration")
    print("-" * 40)

    print("📦 Production Dockerfile:")
    print("""
    # Multi-stage production Dockerfile
    FROM python:3.11-slim AS builder
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    FROM python:3.11-slim AS runtime
    WORKDIR /app
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    COPY . .

    # Production environment
    ENV GENOPS_ENVIRONMENT=production
    ENV PYTHONUNBUFFERED=1
    ENV OTEL_RESOURCE_ATTRIBUTES="service.name=promptlayer-service,service.version=1.0.0"

    EXPOSE 8080
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
        CMD python -c "import requests; requests.get('http://localhost:8080/health')"

    CMD ["python", "production_patterns.py"]
    """)

    print("\n☸️ Kubernetes Deployment:")
    print("""
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: promptlayer-service
      labels:
        app: promptlayer-service
        version: v1.0.0
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: promptlayer-service
      template:
        metadata:
          labels:
            app: promptlayer-service
            version: v1.0.0
        spec:
          containers:
          - name: promptlayer-service
            image: your-registry/promptlayer-service:v1.0.0
            ports:
            - containerPort: 8080
            env:
            - name: GENOPS_ENVIRONMENT
              value: "production"
            - name: PROMPTLAYER_API_KEY
              valueFrom:
                secretKeyRef:
                  name: promptlayer-secret
                  key: api-key
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://jaeger:14268/api/traces"
            resources:
              requests:
                memory: "2Gi"
                cpu: "1"
              limits:
                memory: "8Gi"
                cpu: "4"
            livenessProbe:
              httpGet:
                path: /health
                port: 8080
              initialDelaySeconds: 30
              periodSeconds: 10
            readinessProbe:
              httpGet:
                path: /ready
                port: 8080
              initialDelaySeconds: 5
              periodSeconds: 5
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: promptlayer-service
    spec:
      selector:
        app: promptlayer-service
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8080
    ---
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: promptlayer-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: promptlayer-service
      minReplicas: 3
      maxReplicas: 20
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
    """)


def show_monitoring_dashboards():
    """Show monitoring dashboard configurations."""
    print("\n📊 Production Monitoring Dashboards")
    print("-" * 40)

    print("🎯 Grafana Dashboard Configuration:")
    print("""
    {
      "dashboard": {
        "title": "PromptLayer Production Metrics",
        "panels": [
          {
            "title": "Request Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(promptlayer_requests_total[5m])",
                "legend": "Requests/sec"
              }
            ]
          },
          {
            "title": "Success Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(promptlayer_requests_success[5m]) / rate(promptlayer_requests_total[5m]) * 100",
                "legend": "Success %"
              }
            ]
          },
          {
            "title": "Cost Attribution by Team",
            "type": "piechart",
            "targets": [
              {
                "expr": "promptlayer_cost_by_team",
                "legend": "{{team}}"
              }
            ]
          },
          {
            "title": "Latency Percentiles",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.50, promptlayer_latency_histogram)",
                "legend": "P50"
              },
              {
                "expr": "histogram_quantile(0.95, promptlayer_latency_histogram)",
                "legend": "P95"
              },
              {
                "expr": "histogram_quantile(0.99, promptlayer_latency_histogram)",
                "legend": "P99"
              }
            ]
          }
        ]
      }
    }
    """)

    print("\n📈 Datadog Custom Metrics:")
    print("""
    # Custom metrics for Datadog
    {
      "promptlayer.requests.rate": {
        "type": "rate",
        "tags": ["service:promptlayer", "env:production"]
      },
      "promptlayer.cost.total": {
        "type": "gauge",
        "tags": ["service:promptlayer", "team:*"]
      },
      "promptlayer.governance.violations": {
        "type": "count",
        "tags": ["service:promptlayer", "policy:*"]
      },
      "promptlayer.quality.score": {
        "type": "gauge",
        "tags": ["service:promptlayer", "prompt:*"]
      }
    }
    """)


async def main():
    """Main execution function."""
    print("🚀 Starting PromptLayer Production Patterns Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check prerequisites
    if not os.getenv("PROMPTLAYER_API_KEY"):
        print("❌ PROMPTLAYER_API_KEY not found")
        print(
            "💡 Set your PromptLayer API key: export PROMPTLAYER_API_KEY='pl-your-key'"
        )
        print("📖 Get your API key from: https://promptlayer.com/")
        return False

    # Run demonstrations
    success = True

    # Production deployment patterns
    if not demonstrate_production_deployment():
        success = False

    # Configuration examples
    if success:
        show_docker_kubernetes_configs()
        show_monitoring_dashboards()

    if success:
        print("\n" + "🌟" * 65)
        print("🎉 PromptLayer Production Patterns Demo Complete!")
        print("\n📊 What You've Mastered:")
        print("   ✅ Enterprise-ready production service architecture")
        print("   ✅ Comprehensive monitoring and observability integration")
        print("   ✅ Auto-scaling and load balancing with governance")
        print("   ✅ Docker containerization and Kubernetes deployment")

        print("\n🔍 Your Production-Ready Stack:")
        print("   • PromptLayer: Prompt management and execution platform")
        print("   • GenOps: Production governance and cost intelligence")
        print("   • OpenTelemetry: Enterprise observability and tracing")
        print("   • Kubernetes: Container orchestration and auto-scaling")
        print("   • Multi-Platform: Prometheus, Datadog, Grafana integration")

        print("\n📚 Next Steps:")
        print("   • Deploy to your Kubernetes cluster using provided configs")
        print("   • Integrate with your existing observability stack")
        print("   • Set up alerting based on SLI/SLO thresholds")
        print("   • Run complete test suite: pytest tests/promptlayer/")

        print("\n💡 Production Deployment Checklist:")
        print("   ✅ Container image built and pushed to registry")
        print("   ✅ Kubernetes manifests applied to cluster")
        print("   ✅ Secrets and ConfigMaps configured")
        print("   ✅ Monitoring dashboards and alerts set up")
        print("   ✅ Load testing and performance validation completed")
        print("   ✅ Disaster recovery and backup procedures documented")

        print("\n🏗️ Architecture Pattern:")
        print("   ```yaml")
        print("   # Production deployment with governance")
        print("   apiVersion: v1")
        print("   kind: Service")
        print("   metadata:")
        print("     annotations:")
        print("       genops.ai/governance: 'enforced'")
        print("       genops.ai/cost-center: 'ai-platform'")
        print("       genops.ai/team: 'ai-engineering'")
        print("   ```")

        print("🌟" * 65)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
