#!/usr/bin/env python3
"""
Example: Production Monitoring and Alerting

Complexity: ⭐⭐⭐ Advanced

This example demonstrates comprehensive production monitoring for Flowise
deployments including health checks, metrics collection, alerting, and
dashboard setup for observability platforms.

Prerequisites:
- Flowise instance running
- GenOps package installed
- Flask and prometheus_client for monitoring endpoints

Usage:
    python 07_production_monitoring.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key
    MONITORING_PORT: Port for monitoring endpoints (default: 8080)
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

# Production monitoring dependencies
try:
    from flask import Flask, Response, jsonify
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAS_MONITORING_DEPS = True
except ImportError:
    print("⚠️  Install monitoring dependencies: pip install flask prometheus_client")
    HAS_MONITORING_DEPS = False

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Health check configuration and results."""

    name: str
    check_function: callable
    interval_seconds: int = 60
    timeout_seconds: int = 10
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    condition: str  # e.g., "error_rate > 0.1"
    severity: str  # "info", "warning", "critical"
    threshold: float
    duration_minutes: int = 5  # How long condition must persist
    enabled: bool = True


class ProductionMonitor:
    """Production monitoring system for Flowise deployments."""

    def __init__(self, flowise_base_url: str, api_key: Optional[str] = None):
        self.flowise_base_url = flowise_base_url
        self.api_key = api_key

        # Initialize Prometheus metrics
        if HAS_MONITORING_DEPS:
            self._init_prometheus_metrics()

        # Health checks
        self.health_checks: list[HealthCheck] = []
        self._setup_health_checks()

        # Alert rules
        self.alert_rules: list[AlertRule] = []
        self._setup_alert_rules()

        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_cost": Decimal("0.0"),
            "average_response_time": 0.0,
            "uptime_start": datetime.now(),
        }

        logger.info("Production monitor initialized")

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for monitoring."""

        # Request metrics
        self.request_counter = Counter(
            "flowise_requests_total",
            "Total number of Flowise requests",
            ["chatflow_id", "team", "project", "status"],
        )

        self.request_duration = Histogram(
            "flowise_request_duration_seconds",
            "Time spent on Flowise requests",
            ["chatflow_id", "team", "project"],
        )

        self.request_cost = Counter(
            "flowise_request_cost_total",
            "Total cost of Flowise requests in USD",
            ["chatflow_id", "team", "project"],
        )

        # System metrics
        self.active_sessions = Gauge(
            "flowise_active_sessions", "Number of active Flowise sessions"
        )

        self.health_status = Gauge(
            "flowise_health_status",
            "Health status of Flowise components",
            ["component"],
        )

        self.error_rate = Gauge(
            "flowise_error_rate", "Error rate percentage for Flowise requests"
        )

    def _setup_health_checks(self):
        """Setup health check configurations."""

        # Flowise API health
        self.health_checks.append(
            HealthCheck(
                name="flowise_api",
                check_function=self._check_flowise_health,
                interval_seconds=30,
            )
        )

        # Flowise chatflows availability
        self.health_checks.append(
            HealthCheck(
                name="chatflows_available",
                check_function=self._check_chatflows_health,
                interval_seconds=60,
            )
        )

        # System resource health
        self.health_checks.append(
            HealthCheck(
                name="system_resources",
                check_function=self._check_system_resources,
                interval_seconds=60,
            )
        )

    def _setup_alert_rules(self):
        """Setup alerting rules."""

        self.alert_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 10",
                severity="warning",
                threshold=10.0,
                duration_minutes=5,
            ),
            AlertRule(
                name="very_high_error_rate",
                condition="error_rate > 25",
                severity="critical",
                threshold=25.0,
                duration_minutes=2,
            ),
            AlertRule(
                name="flowise_down",
                condition="flowise_api_health == false",
                severity="critical",
                threshold=0.0,
                duration_minutes=1,
            ),
            AlertRule(
                name="high_response_time",
                condition="avg_response_time > 10",
                severity="warning",
                threshold=10.0,
                duration_minutes=10,
            ),
        ]

    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Production monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Production monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                self._run_health_checks()

                # Update metrics
                self._update_metrics()

                # Check alert rules
                self._check_alerts()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on errors

    def _run_health_checks(self):
        """Run all configured health checks."""
        for check in self.health_checks:
            # Skip if not time for next check
            if check.last_check and datetime.now() - check.last_check < timedelta(
                seconds=check.interval_seconds
            ):
                continue

            try:
                # Run health check
                result = check.check_function()
                check.last_check = datetime.now()
                check.last_result = result
                check.last_error = None

                if result:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1

                # Update Prometheus metric
                if HAS_MONITORING_DEPS:
                    self.health_status.labels(component=check.name).set(
                        1 if result else 0
                    )

            except Exception as e:
                check.last_check = datetime.now()
                check.last_result = False
                check.last_error = str(e)
                check.consecutive_failures += 1

                if HAS_MONITORING_DEPS:
                    self.health_status.labels(component=check.name).set(0)

                logger.error(f"Health check {check.name} failed: {e}")

    def _check_flowise_health(self) -> bool:
        """Check if Flowise API is healthy."""
        try:
            result = validate_flowise_setup(
                self.flowise_base_url, self.api_key, timeout=5
            )
            return result.is_valid
        except Exception:
            return False

    def _check_chatflows_health(self) -> bool:
        """Check if chatflows are available and accessible."""
        try:
            flowise = instrument_flowise(
                base_url=self.flowise_base_url,
                api_key=self.api_key,
                team="health-check",
                project="monitoring",
            )
            chatflows = flowise.get_chatflows()
            return len(chatflows) > 0
        except Exception:
            return False

    def _check_system_resources(self) -> bool:
        """Check system resource utilization."""
        try:
            # Simplified resource check
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            # Consider healthy if CPU < 90% and Memory < 85%
            return cpu_percent < 90 and memory_percent < 85
        except ImportError:
            # psutil not available, assume healthy
            return True
        except Exception:
            return False

    def _update_metrics(self):
        """Update monitoring metrics."""
        if not HAS_MONITORING_DEPS:
            return

        # Update error rate
        if self.stats["total_requests"] > 0:
            error_rate = (
                self.stats["failed_requests"] / self.stats["total_requests"]
            ) * 100
            self.error_rate.set(error_rate)

    def _check_alerts(self):
        """Check alert rules and trigger alerts."""
        current_metrics = {
            "error_rate": (
                self.stats["failed_requests"] / max(self.stats["total_requests"], 1)
            )
            * 100,
            "avg_response_time": self.stats["average_response_time"],
            "flowise_api_health": any(
                check.name == "flowise_api" and check.last_result
                for check in self.health_checks
            ),
        }

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Simple rule evaluation (in production, use more sophisticated logic)
            should_alert = False

            if "error_rate >" in rule.condition:
                should_alert = current_metrics["error_rate"] > rule.threshold
            elif "avg_response_time >" in rule.condition:
                should_alert = current_metrics["avg_response_time"] > rule.threshold
            elif "flowise_api_health == false" in rule.condition:
                should_alert = not current_metrics["flowise_api_health"]

            if should_alert:
                self._trigger_alert(rule, current_metrics)

    def _trigger_alert(self, rule: AlertRule, current_metrics: dict):
        """Trigger an alert based on rule violation."""
        logger.warning(f"ALERT [{rule.severity}]: {rule.name} - {rule.condition}")

        # In production, this would send to alerting systems
        # (PagerDuty, Slack, email, etc.)

    def record_request(
        self,
        chatflow_id: str,
        team: str,
        project: str,
        success: bool,
        duration_seconds: float,
        cost: Decimal = None,
    ):
        """Record metrics for a Flowise request."""

        # Update statistics
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1

        if cost:
            self.stats["total_cost"] += cost

        # Update average response time (simple moving average)
        self.stats["average_response_time"] = (
            self.stats["average_response_time"] * (self.stats["total_requests"] - 1)
            + duration_seconds
        ) / self.stats["total_requests"]

        # Update Prometheus metrics
        if HAS_MONITORING_DEPS:
            status = "success" if success else "error"
            self.request_counter.labels(
                chatflow_id=chatflow_id, team=team, project=project, status=status
            ).inc()

            if success:
                self.request_duration.labels(
                    chatflow_id=chatflow_id, team=team, project=project
                ).observe(duration_seconds)

            if cost:
                self.request_cost.labels(
                    chatflow_id=chatflow_id, team=team, project=project
                ).inc(float(cost))

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary."""
        overall_healthy = all(
            check.last_result is True
            for check in self.health_checks
            if check.last_result is not None
        )

        health_details = {}
        for check in self.health_checks:
            health_details[check.name] = {
                "healthy": check.last_result,
                "last_check": check.last_check.isoformat()
                if check.last_check
                else None,
                "consecutive_failures": check.consecutive_failures,
                "error": check.last_error,
            }

        uptime_seconds = (datetime.now() - self.stats["uptime_start"]).total_seconds()

        return {
            "overall_healthy": overall_healthy,
            "uptime_seconds": uptime_seconds,
            "health_checks": health_details,
            "statistics": {
                "total_requests": self.stats["total_requests"],
                "success_rate": (
                    (
                        self.stats["successful_requests"]
                        / max(self.stats["total_requests"], 1)
                    )
                    * 100
                ),
                "error_rate": (
                    (
                        self.stats["failed_requests"]
                        / max(self.stats["total_requests"], 1)
                    )
                    * 100
                ),
                "average_response_time": self.stats["average_response_time"],
                "total_cost": float(self.stats["total_cost"]),
            },
        }


def create_monitoring_server(monitor: ProductionMonitor) -> Flask:
    """Create Flask server for monitoring endpoints."""

    app = Flask(__name__)

    @app.route("/health")
    def health_check():
        """Health check endpoint for load balancers."""
        health = monitor.get_health_summary()

        if health["overall_healthy"]:
            return jsonify(
                {
                    "status": "healthy",
                    "uptime_seconds": health["uptime_seconds"],
                    "statistics": health["statistics"],
                }
            ), 200
        else:
            return jsonify(
                {
                    "status": "unhealthy",
                    "health_checks": health["health_checks"],
                    "statistics": health["statistics"],
                }
            ), 503

    @app.route("/health/detailed")
    def detailed_health():
        """Detailed health information."""
        return jsonify(monitor.get_health_summary())

    @app.route("/metrics")
    def prometheus_metrics():
        """Prometheus metrics endpoint."""
        if not HAS_MONITORING_DEPS:
            return "Prometheus client not available", 503

        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    @app.route("/stats")
    def statistics():
        """Statistics endpoint."""
        return jsonify(monitor.stats)

    return app


def demonstrate_production_monitoring():
    """Demonstrate production monitoring setup."""

    print("📊 Production Monitoring and Alerting")
    print("=" * 50)

    if not HAS_MONITORING_DEPS:
        print("❌ Missing monitoring dependencies. Install with:")
        print("   pip install flask prometheus_client psutil")
        return False

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")
    monitoring_port = int(os.getenv("MONITORING_PORT", "8080"))

    # Step 1: Initialize monitoring
    print("📋 Step 1: Initializing production monitoring...")

    try:
        monitor = ProductionMonitor(base_url, api_key)
        monitor.start_monitoring()

        print("✅ Production monitoring initialized")
        print(f"   Health checks: {len(monitor.health_checks)}")
        print(f"   Alert rules: {len(monitor.alert_rules)}")

    except Exception as e:
        logger.error(f"Monitoring initialization failed: {e}")
        return False

    # Step 2: Setup monitoring server
    print(f"\n🌐 Step 2: Starting monitoring server on port {monitoring_port}...")

    try:
        app = create_monitoring_server(monitor)

        # Start server in separate thread
        server_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=monitoring_port, debug=False),
            daemon=True,
        )
        server_thread.start()

        # Give server time to start
        time.sleep(2)

        print("✅ Monitoring server started")
        print(f"   Health endpoint: http://localhost:{monitoring_port}/health")
        print(f"   Metrics endpoint: http://localhost:{monitoring_port}/metrics")
        print(f"   Detailed health: http://localhost:{monitoring_port}/health/detailed")

    except Exception as e:
        logger.error(f"Monitoring server failed: {e}")
        monitor.stop_monitoring()
        return False

    # Step 3: Simulate requests with monitoring
    print("\n🔄 Step 3: Simulating monitored requests...")

    try:
        flowise = instrument_flowise(
            base_url=base_url,
            api_key=api_key,
            team="monitoring-demo",
            project="production-test",
            environment="production",
        )

        chatflows = flowise.get_chatflows()
        if not chatflows:
            print("❌ No chatflows available for testing")
            monitor.stop_monitoring()
            return False

        chatflow_id = chatflows[0].get("id")
        chatflows[0].get("name", "Unnamed")

        # Simulate various request scenarios
        scenarios = [
            ("Successful request", "What are the benefits of AI?", True),
            ("Another successful request", "Explain machine learning.", True),
            (
                "Complex request",
                "Generate a comprehensive analysis of market trends with detailed competitive intelligence across all sectors",
                True,
            ),
            (
                "Failed request simulation",
                "",
                False,
            ),  # Empty request to simulate failure
        ]

        for scenario_name, question, should_succeed in scenarios:
            print(f"   📡 {scenario_name}...")

            start_time = time.time()

            try:
                if should_succeed and question:
                    flowise.predict_flow(chatflow_id, question)
                    success = True
                else:
                    # Simulate failure
                    raise Exception("Simulated failure")

            except Exception as e:
                success = False
                logger.debug(f"Request failed (simulated): {e}")

            duration = time.time() - start_time
            cost = Decimal("0.002") if success else Decimal("0.0")

            # Record metrics
            monitor.record_request(
                chatflow_id=chatflow_id,
                team="monitoring-demo",
                project="production-test",
                success=success,
                duration_seconds=duration,
                cost=cost,
            )

            print(
                f"      {'✅' if success else '❌'} Duration: {duration:.3f}s, Cost: ${cost:.4f}"
            )

        # Wait for health checks to run
        print("\n⏰ Waiting for health checks to complete...")
        time.sleep(35)  # Wait for at least one health check cycle

    except Exception as e:
        logger.error(f"Request simulation failed: {e}")
        monitor.stop_monitoring()
        return False

    # Step 4: Show monitoring results
    print("\n📊 Step 4: Monitoring Results")
    print("=" * 30)

    health_summary = monitor.get_health_summary()

    print(
        f"Overall Health: {'✅ Healthy' if health_summary['overall_healthy'] else '❌ Unhealthy'}"
    )
    print(f"Uptime: {health_summary['uptime_seconds']:.0f} seconds")

    stats = health_summary["statistics"]
    print("\nRequest Statistics:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Error Rate: {stats['error_rate']:.1f}%")
    print(f"   Average Response Time: {stats['average_response_time']:.3f}s")
    print(f"   Total Cost: ${stats['total_cost']:.4f}")

    print("\nHealth Check Details:")
    for check_name, check_data in health_summary["health_checks"].items():
        status = "✅ Healthy" if check_data["healthy"] else "❌ Unhealthy"
        print(f"   {check_name}: {status}")
        if check_data["error"]:
            print(f"      Error: {check_data['error']}")
        if check_data["consecutive_failures"] > 0:
            print(f"      Consecutive Failures: {check_data['consecutive_failures']}")

    # Step 5: Show monitoring integration examples
    print("\n🔗 Step 5: Integration Examples")
    print("=" * 30)

    print("Prometheus Metrics Collection:")
    print("   scrape_configs:")
    print("     - job_name: 'flowise-genops'")
    print("       static_configs:")
    print(f"         - targets: ['localhost:{monitoring_port}']")
    print("       metrics_path: '/metrics'")
    print("       scrape_interval: 30s")

    print("\nGrafana Dashboard Queries:")
    print("   Request Rate: rate(flowise_requests_total[5m])")
    print(
        '   Error Rate: rate(flowise_requests_total{status="error"}[5m]) / rate(flowise_requests_total[5m])'
    )
    print(
        "   Response Time: histogram_quantile(0.95, rate(flowise_request_duration_seconds_bucket[5m]))"
    )
    print("   Cost Rate: rate(flowise_request_cost_total[1h])")

    print("\nAlertmanager Rules:")
    print("   - alert: FlowiseHighErrorRate")
    print("     expr: flowise_error_rate > 10")
    print("     for: 5m")
    print("     labels:")
    print("       severity: warning")

    # Cleanup
    monitor.stop_monitoring()

    return True


def demonstrate_observability_integrations():
    """Show observability platform integrations."""

    print("\n🔍 Observability Platform Integrations")
    print("=" * 50)

    integrations = [
        {
            "platform": "Datadog",
            "setup": [
                'Export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.datadoghq.com"',
                'Export OTEL_EXPORTER_OTLP_HEADERS="dd-api-key=your-key"',
                "Enable Datadog APM and custom metrics",
            ],
            "benefits": [
                "Native OpenTelemetry support",
                "Pre-built AI/ML dashboards",
                "Intelligent alerting and anomaly detection",
            ],
        },
        {
            "platform": "Grafana + Prometheus",
            "setup": [
                "Deploy Prometheus with Flowise scraping config",
                "Import GenOps Grafana dashboard",
                "Configure Alertmanager for notifications",
            ],
            "benefits": [
                "Open source and self-hosted",
                "Highly customizable dashboards",
                "Flexible alerting rules",
            ],
        },
        {
            "platform": "Honeycomb",
            "setup": [
                'Export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"',
                'Export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-key"',
                "Enable structured logging and tracing",
            ],
            "benefits": [
                "Excellent for debugging complex workflows",
                "Advanced query and exploration capabilities",
                "Built-in SLI/SLO tracking",
            ],
        },
    ]

    for integration in integrations:
        print(f"\n📊 {integration['platform']}:")
        print("   Setup:")
        for step in integration["setup"]:
            print(f"     • {step}")
        print("   Benefits:")
        for benefit in integration["benefits"]:
            print(f"     • {benefit}")


def main():
    """Main example function."""

    try:
        print("🚀 Production Monitoring and Alerting Example")
        print("=" * 60)

        # Run main demonstration
        success = demonstrate_production_monitoring()

        if success:
            # Show observability integrations
            demonstrate_observability_integrations()

            print("\n🎉 Production Monitoring Example Complete!")
            print("=" * 50)
            print("✅ You've learned how to:")
            print("   • Set up comprehensive production monitoring")
            print("   • Create health checks and alerting rules")
            print("   • Export Prometheus metrics for observability")
            print("   • Build monitoring dashboards and endpoints")
            print("   • Integrate with major observability platforms")

            print("\n📊 Production Monitoring Features:")
            print("   • Real-time health checks and status monitoring")
            print("   • Prometheus metrics for request, error, and cost tracking")
            print("   • Configurable alerting rules with severity levels")
            print("   • RESTful health and metrics endpoints")
            print("   • Integration with Grafana, Datadog, and other platforms")

            print("\n📚 Next Steps:")
            print("   • Deploy monitoring in your production environment")
            print("   • Set up Grafana dashboards for visualization")
            print("   • Configure alerting to your incident management system")
            print(
                "   • Try async high-performance patterns (08_async_high_performance.py)"
            )

        return success

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
