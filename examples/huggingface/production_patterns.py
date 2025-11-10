#!/usr/bin/env python3
"""
Hugging Face Production Patterns Example

This example demonstrates enterprise-ready deployment patterns for GenOps
with Hugging Face in production environments.

Example usage:
    python production_patterns.py

Features demonstrated:
- High-volume instrumentation strategies
- Async telemetry export patterns
- Error handling and circuit breakers
- Performance optimization techniques
- Monitoring and alerting integration
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for production monitoring."""
    operation_count: int = 0
    total_duration: float = 0.0
    error_count: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.operation_count if self.operation_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.operation_count if self.operation_count > 0 else 0.0

    @property
    def throughput(self) -> float:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.operation_count / elapsed if elapsed > 0 else 0.0


class ProductionHuggingFaceAdapter:
    """Production-ready Hugging Face adapter with enhanced monitoring."""

    def __init__(self,
                 max_retries: int = 3,
                 timeout: float = 30.0,
                 circuit_breaker_threshold: int = 5,
                 enable_metrics: bool = True):
        self.max_retries = max_retries
        self.timeout = timeout
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.enable_metrics = enable_metrics

        self.metrics = PerformanceMetrics()
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False
        self._lock = threading.Lock()

        try:
            from genops.providers.huggingface import GenOpsHuggingFaceAdapter
            self.adapter = GenOpsHuggingFaceAdapter()
        except ImportError:
            self.adapter = None
            logger.error("GenOps Hugging Face adapter not available")

    @contextmanager
    def _performance_tracking(self, operation_name: str):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        try:
            yield
            if self.enable_metrics:
                with self._lock:
                    self.metrics.operation_count += 1
                    self.metrics.total_duration += time.time() - start_time

        except Exception as e:
            if self.enable_metrics:
                with self._lock:
                    self.metrics.operation_count += 1
                    self.metrics.error_count += 1
                    self.failure_count += 1
                    self.last_failure_time = datetime.now()

                    # Circuit breaker logic
                    if self.failure_count >= self.circuit_breaker_threshold:
                        self.circuit_open = True
                        logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            logger.error(f"Operation {operation_name} failed: {e}")
            raise

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should allow operation."""
        if not self.circuit_open:
            return True

        # Auto-reset circuit breaker after 60 seconds
        if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() > 60:
            with self._lock:
                self.circuit_open = False
                self.failure_count = 0
            logger.info("Circuit breaker reset")
            return True

        return False

    def generate_text_with_retry(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text with retry logic and circuit breaker."""
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - too many recent failures")

        if not self.adapter:
            raise Exception("GenOps adapter not available")

        for attempt in range(self.max_retries):
            try:
                with self._performance_tracking(f"text_generation_attempt_{attempt + 1}"):
                    response = self.adapter.text_generation(
                        prompt=prompt,
                        **kwargs
                    )

                    # Reset failure count on success
                    with self._lock:
                        self.failure_count = 0
                        if self.circuit_open:
                            self.circuit_open = False
                            logger.info("Circuit breaker reset after successful operation")

                    return response

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        return None

    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary."""
        return {
            'operation_count': self.metrics.operation_count,
            'avg_duration': self.metrics.avg_duration,
            'error_rate': self.metrics.error_rate,
            'throughput': self.metrics.throughput,
            'total_cost': self.metrics.total_cost,
            'circuit_breaker_open': self.circuit_open,
            'failure_count': self.failure_count
        }


def demonstrate_high_volume_processing():
    """Demonstrate high-volume request processing with monitoring."""

    print("📈 High-Volume Processing Demo")
    print("="*40)
    print("Simulating production-scale request processing:")
    print()

    try:
        # Create production adapter
        prod_adapter = ProductionHuggingFaceAdapter(
            max_retries=2,
            timeout=15.0,
            circuit_breaker_threshold=3
        )

        # Simulate high-volume requests
        requests = [
            {
                "prompt": f"Summarize the key points from customer feedback #{i}",
                "governance": {
                    "team": "support-team",
                    "project": "feedback-analysis",
                    "customer_id": f"batch-{i // 10}",
                    "operation_id": f"op-{i:04d}"
                }
            }
            for i in range(1, 26)  # 25 requests for demo
        ]

        print(f"📊 Processing {len(requests)} requests with production patterns...")
        print()

        # Process requests with concurrent execution
        successful_operations = 0
        failed_operations = 0

        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit concurrency
            # Submit all tasks
            future_to_request = {
                executor.submit(
                    prod_adapter.generate_text_with_retry,
                    req['prompt'],
                    model='microsoft/DialoGPT-medium',
                    max_new_tokens=50,
                    **req['governance']
                ): req for req in requests[:10]  # Process first 10 for demo
            }

            # Collect results
            for i, future in enumerate(as_completed(future_to_request), 1):
                request = future_to_request[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        successful_operations += 1
                        if i <= 3:  # Show first few results
                            print(f"   ✅ Operation {request['governance']['operation_id']}: Success")
                        elif i == 4:
                            print("   ... processing remaining operations ...")
                    else:
                        failed_operations += 1
                        print(f"   ❌ Operation {request['governance']['operation_id']}: Failed")

                except Exception as e:
                    failed_operations += 1
                    print(f"   ❌ Operation {request['governance']['operation_id']}: Error - {str(e)[:50]}...")

        print()

        # Display performance metrics
        metrics = prod_adapter.get_metrics_summary()
        print("📊 Performance Metrics:")
        print(f"   Operations Completed: {metrics['operation_count']}")
        print(f"   Success Rate: {((successful_operations / (successful_operations + failed_operations)) * 100):.1f}%")
        print(f"   Average Duration: {metrics['avg_duration']:.3f}s")
        print(f"   Throughput: {metrics['throughput']:.1f} ops/sec")
        print(f"   Error Rate: {metrics['error_rate']:.1%}")
        print(f"   Circuit Breaker Status: {'🔴 OPEN' if metrics['circuit_breaker_open'] else '🟢 CLOSED'}")
        print()

        return True

    except ImportError as e:
        print(f"❌ High-volume processing unavailable: {e}")
        return False


def demonstrate_async_telemetry_export():
    """Demonstrate asynchronous telemetry export patterns."""

    print("🚀 Async Telemetry Export Demo")
    print("="*40)
    print("Demonstrating non-blocking telemetry export for production:")
    print()

    class AsyncTelemetryExporter:
        """Example async telemetry exporter."""

        def __init__(self, batch_size: int = 100, flush_interval: float = 5.0):
            self.batch_size = batch_size
            self.flush_interval = flush_interval
            self.telemetry_queue = []
            self.last_flush = time.time()
            self._lock = threading.Lock()
            self._export_thread = None
            self._stop_event = threading.Event()

        def start(self):
            """Start the background export thread."""
            if not self._export_thread or not self._export_thread.is_alive():
                self._stop_event.clear()
                self._export_thread = threading.Thread(target=self._export_worker)
                self._export_thread.daemon = True
                self._export_thread.start()
                logger.info("Async telemetry exporter started")

        def stop(self):
            """Stop the background export thread."""
            if self._export_thread:
                self._stop_event.set()
                self._export_thread.join(timeout=10)
                # Flush any remaining data
                self._flush_telemetry()
                logger.info("Async telemetry exporter stopped")

        def add_telemetry(self, operation_data: Dict):
            """Add telemetry data to export queue."""
            with self._lock:
                self.telemetry_queue.append({
                    'timestamp': datetime.now().isoformat(),
                    'data': operation_data
                })

                # Check if we need to flush
                if (len(self.telemetry_queue) >= self.batch_size or
                    time.time() - self.last_flush > self.flush_interval):
                    self._flush_telemetry()

        def _export_worker(self):
            """Background worker for exporting telemetry."""
            while not self._stop_event.wait(1.0):  # Check every second
                with self._lock:
                    if (time.time() - self.last_flush > self.flush_interval and
                        len(self.telemetry_queue) > 0):
                        self._flush_telemetry()

        def _flush_telemetry(self):
            """Flush telemetry data to export destination."""
            if not self.telemetry_queue:
                return

            batch = self.telemetry_queue.copy()
            self.telemetry_queue.clear()
            self.last_flush = time.time()

            # Simulate async export (in production, send to OTLP endpoint)
            logger.info(f"📤 Exporting batch of {len(batch)} telemetry records")

            # Simulate export processing
            try:
                # In production: send to OpenTelemetry collector
                # otel_exporter.export(batch)
                time.sleep(0.1)  # Simulate network delay
                logger.debug(f"✅ Successfully exported {len(batch)} records")

            except Exception as e:
                logger.error(f"❌ Telemetry export failed: {e}")
                # In production: implement retry logic or dead letter queue

    # Demonstrate async export
    print("   🔄 Setting up async telemetry export...")
    exporter = AsyncTelemetryExporter(batch_size=5, flush_interval=2.0)
    exporter.start()

    print("   📡 Simulating AI operations with telemetry...")

    # Simulate operations generating telemetry
    for i in range(12):
        telemetry_data = {
            'operation_id': f'op-{i:03d}',
            'operation_type': 'text-generation',
            'provider': 'huggingface',
            'model': 'microsoft/DialoGPT-medium',
            'cost': 0.001 * (i + 1),
            'tokens_input': 100 + i * 10,
            'tokens_output': 50 + i * 5,
            'team': f'team-{i % 3}',
            'duration': 0.5 + i * 0.1
        }

        exporter.add_telemetry(telemetry_data)

        if i < 5:
            print(f"      📊 Operation {i+1}: Telemetry queued")
        elif i == 5:
            print("      ... continuing operations ...")

        time.sleep(0.2)  # Simulate operation interval

    print()
    print("   ⏱️ Waiting for final batch export...")
    time.sleep(3)  # Allow final flush

    exporter.stop()

    print("   ✅ Async telemetry export completed")
    print()

    print("💡 Production Telemetry Best Practices:")
    print("   • Use batched export to reduce network overhead")
    print("   • Implement async export to avoid blocking AI operations")
    print("   • Add retry logic with exponential backoff for failed exports")
    print("   • Monitor telemetry export health and set up alerts")
    print("   • Use compression for large telemetry payloads")
    print("   • Implement sampling for extremely high-volume scenarios")
    print()

    return True


def demonstrate_error_resilience():
    """Demonstrate comprehensive error handling and resilience patterns."""

    print("🛡️ Error Resilience Patterns Demo")
    print("="*45)
    print("Demonstrating production error handling and recovery:")
    print()

    class ResilientAIService:
        """Example resilient AI service with comprehensive error handling."""

        def __init__(self):
            self.health_status = "healthy"
            self.error_counts = {
                'rate_limit': 0,
                'timeout': 0,
                'model_error': 0,
                'network': 0,
                'auth': 0
            }
            self.fallback_models = [
                'microsoft/DialoGPT-medium',
                'gpt-3.5-turbo',
                'claude-3-haiku'
            ]

        def classify_error(self, error: Exception) -> str:
            """Classify error type for appropriate handling."""
            error_msg = str(error).lower()

            if 'rate limit' in error_msg or '429' in error_msg:
                return 'rate_limit'
            elif 'timeout' in error_msg:
                return 'timeout'
            elif 'model' in error_msg or '404' in error_msg:
                return 'model_error'
            elif 'network' in error_msg or 'connection' in error_msg:
                return 'network'
            elif 'auth' in error_msg or '401' in error_msg or '403' in error_msg:
                return 'auth'
            else:
                return 'unknown'

        def handle_error_with_fallback(self,
                                     prompt: str,
                                     primary_model: str,
                                     **kwargs) -> Dict:
            """Handle errors with intelligent fallback strategies."""

            models_to_try = [primary_model] + [m for m in self.fallback_models if m != primary_model]

            for model_index, model in enumerate(models_to_try):
                try:
                    print(f"      🎯 Attempting with model: {model}")

                    # Simulate API call with various potential errors
                    import random
                    if random.random() < 0.3:  # 30% chance of simulated error
                        error_types = ['rate_limit', 'timeout', 'model_error', 'network']
                        simulated_error = random.choice(error_types)
                        raise Exception(f"Simulated {simulated_error} error for demo")

                    # Success case
                    result = {
                        'model_used': model,
                        'response': f"Response from {model} for: {prompt[:30]}...",
                        'attempt_number': model_index + 1,
                        'fallback_used': model_index > 0,
                        'cost': 0.001 * (model_index + 1)  # Simulate cost variation
                    }

                    print(f"      ✅ Success with {model}")
                    return result

                except Exception as e:
                    error_type = self.classify_error(e)
                    self.error_counts[error_type] += 1

                    print(f"      ❌ {model} failed: {error_type}")

                    # Error-specific handling
                    if error_type == 'rate_limit':
                        print("         ⏱️ Rate limit detected - waiting before retry...")
                        time.sleep(1)  # In production: exponential backoff
                    elif error_type == 'auth':
                        print("         🔑 Auth error - this model may require different credentials")
                        continue  # Skip to next model immediately
                    elif error_type == 'model_error':
                        print("         🔄 Model unavailable - trying alternative...")
                        continue

                    # If this is the last model, re-raise the error
                    if model_index == len(models_to_try) - 1:
                        print("         💥 All fallback options exhausted")
                        raise Exception(f"All models failed. Last error: {e}")

            return None

        def get_health_status(self) -> Dict:
            """Get service health status and metrics."""
            total_errors = sum(self.error_counts.values())

            if total_errors > 10:
                self.health_status = "degraded"
            elif total_errors > 20:
                self.health_status = "unhealthy"
            else:
                self.health_status = "healthy"

            return {
                'status': self.health_status,
                'error_counts': self.error_counts.copy(),
                'total_errors': total_errors,
                'uptime': '99.5%',  # Simulated
                'last_check': datetime.now().isoformat()
            }

    # Demonstrate resilient service
    service = ResilientAIService()

    test_scenarios = [
        {
            "name": "Normal Operation",
            "prompt": "Generate a welcome message for new users",
            "model": "microsoft/DialoGPT-medium"
        },
        {
            "name": "Primary Model Failure",
            "prompt": "Create a summary of quarterly results",
            "model": "unavailable-model-123"
        },
        {
            "name": "High-Load Scenario",
            "prompt": "Process customer feedback for sentiment analysis",
            "model": "gpt-4"
        },
        {
            "name": "Network Issues Recovery",
            "prompt": "Generate product description for new feature",
            "model": "claude-3-opus"
        }
    ]

    print("🧪 Testing Error Resilience Scenarios:")
    print()

    successful_operations = 0

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"   {i}. {scenario['name']}:")
        print(f"      Prompt: {scenario['prompt'][:50]}...")

        try:
            result = service.handle_error_with_fallback(
                prompt=scenario['prompt'],
                primary_model=scenario['model'],
                team='resilience-test',
                project='error-handling-demo'
            )

            if result:
                successful_operations += 1
                print("      ✅ Operation completed successfully")
                print(f"      📊 Model used: {result['model_used']}")
                print(f"      🔄 Fallback used: {'Yes' if result['fallback_used'] else 'No'}")
                print(f"      💰 Cost: ${result['cost']:.6f}")

        except Exception as e:
            print(f"      ❌ Operation failed completely: {str(e)[:60]}...")

        print()

    # Health status report
    health = service.get_health_status()
    print("📊 Service Health Report:")
    print(f"   Overall Status: {health['status'].upper()}")
    print(f"   Total Errors: {health['total_errors']}")
    print("   Error Breakdown:")
    for error_type, count in health['error_counts'].items():
        if count > 0:
            print(f"      • {error_type}: {count}")
    print(f"   Success Rate: {(successful_operations / len(test_scenarios) * 100):.1f}%")
    print()

    print("🛡️ Resilience Best Practices Demonstrated:")
    print("   ✅ Intelligent error classification and handling")
    print("   ✅ Model fallback chains with cost awareness")
    print("   ✅ Rate limit detection and backoff strategies")
    print("   ✅ Health monitoring and status reporting")
    print("   ✅ Graceful degradation under load")
    print()

    return True


def demonstrate_monitoring_integration():
    """Demonstrate integration with monitoring and alerting systems."""

    print("📊 Production Monitoring Integration")
    print("="*45)
    print("Demonstrating monitoring, alerting, and observability patterns:")
    print()

    class ProductionMonitor:
        """Production monitoring and alerting system."""

        def __init__(self):
            self.metrics = {
                'request_count': 0,
                'success_count': 0,
                'error_count': 0,
                'total_cost': 0.0,
                'avg_latency': 0.0,
                'p95_latency': 0.0
            }
            self.alerts = []
            self.thresholds = {
                'error_rate': 0.05,  # 5%
                'avg_latency': 2.0,  # 2 seconds
                'hourly_cost': 10.0,  # $10/hour
                'success_rate': 0.95  # 95%
            }

        def record_operation(self, success: bool, latency: float, cost: float):
            """Record operation metrics."""
            self.metrics['request_count'] += 1

            if success:
                self.metrics['success_count'] += 1
            else:
                self.metrics['error_count'] += 1

            self.metrics['total_cost'] += cost

            # Update latency (simplified moving average)
            self.metrics['avg_latency'] = (
                (self.metrics['avg_latency'] * (self.metrics['request_count'] - 1) + latency)
                / self.metrics['request_count']
            )

            # Check for alerts
            self._check_alerts()

        def _check_alerts(self):
            """Check metrics against thresholds and generate alerts."""
            current_error_rate = (
                self.metrics['error_count'] / self.metrics['request_count']
                if self.metrics['request_count'] > 0 else 0
            )

            # Error rate alert
            if current_error_rate > self.thresholds['error_rate']:
                self.alerts.append({
                    'type': 'HIGH_ERROR_RATE',
                    'message': f"Error rate {current_error_rate:.1%} exceeds threshold {self.thresholds['error_rate']:.1%}",
                    'severity': 'CRITICAL',
                    'timestamp': datetime.now()
                })

            # Latency alert
            if self.metrics['avg_latency'] > self.thresholds['avg_latency']:
                self.alerts.append({
                    'type': 'HIGH_LATENCY',
                    'message': f"Average latency {self.metrics['avg_latency']:.2f}s exceeds threshold {self.thresholds['avg_latency']}s",
                    'severity': 'WARNING',
                    'timestamp': datetime.now()
                })

            # Cost alert (simplified hourly projection)
            projected_hourly_cost = self.metrics['total_cost'] * (3600 / max(1, self.metrics['request_count']))
            if projected_hourly_cost > self.thresholds['hourly_cost']:
                self.alerts.append({
                    'type': 'HIGH_COST',
                    'message': f"Projected hourly cost ${projected_hourly_cost:.2f} exceeds threshold ${self.thresholds['hourly_cost']}",
                    'severity': 'WARNING',
                    'timestamp': datetime.now()
                })

        def get_dashboard_data(self) -> Dict:
            """Get data for monitoring dashboard."""
            success_rate = (
                self.metrics['success_count'] / self.metrics['request_count']
                if self.metrics['request_count'] > 0 else 0
            )

            return {
                'metrics': self.metrics.copy(),
                'derived_metrics': {
                    'success_rate': success_rate,
                    'error_rate': 1 - success_rate,
                    'requests_per_minute': self.metrics['request_count'] / max(1,
                        (datetime.now() - datetime.now().replace(minute=0, second=0)).seconds / 60)
                },
                'alerts': self.alerts.copy(),
                'thresholds': self.thresholds.copy()
            }

    # Demonstrate monitoring
    monitor = ProductionMonitor()

    print("📈 Simulating production traffic with monitoring...")

    # Simulate various operation scenarios
    scenarios = [
        # Normal operations
        *[{'success': True, 'latency': 0.5 + i * 0.1, 'cost': 0.002} for i in range(10)],
        # Some slower operations
        *[{'success': True, 'latency': 1.5 + i * 0.2, 'cost': 0.005} for i in range(5)],
        # A few failures
        *[{'success': False, 'latency': 3.0, 'cost': 0.001} for i in range(3)],
        # High-cost operations
        *[{'success': True, 'latency': 0.8, 'cost': 0.02} for i in range(3)],
    ]

    for i, scenario in enumerate(scenarios):
        monitor.record_operation(
            success=scenario['success'],
            latency=scenario['latency'],
            cost=scenario['cost']
        )

        if i % 5 == 0:  # Show progress every 5 operations
            print(f"   📊 Processed {i + 1}/{len(scenarios)} operations...")

    print()

    # Display dashboard
    dashboard = monitor.get_dashboard_data()

    print("📊 Production Dashboard:")
    print("   🎯 Request Metrics:")
    print(f"      Total Requests: {dashboard['metrics']['request_count']:,}")
    print(f"      Success Rate: {dashboard['derived_metrics']['success_rate']:.1%}")
    print(f"      Error Rate: {dashboard['derived_metrics']['error_rate']:.1%}")
    print()

    print("   ⚡ Performance Metrics:")
    print(f"      Average Latency: {dashboard['metrics']['avg_latency']:.2f}s")
    print(f"      Requests/Minute: {dashboard['derived_metrics']['requests_per_minute']:.1f}")
    print()

    print("   💰 Cost Metrics:")
    print(f"      Total Cost: ${dashboard['metrics']['total_cost']:.4f}")
    print(f"      Average Cost/Request: ${dashboard['metrics']['total_cost'] / dashboard['metrics']['request_count']:.6f}")
    print()

    # Display alerts
    if dashboard['alerts']:
        print(f"🚨 Active Alerts ({len(dashboard['alerts'])}):")
        for alert in dashboard['alerts'][-3:]:  # Show last 3 alerts
            severity_icon = "🔴" if alert['severity'] == 'CRITICAL' else "🟡"
            print(f"      {severity_icon} {alert['type']}: {alert['message']}")
        print()
    else:
        print("✅ No active alerts")
        print()

    print("📊 Monitoring Integration Examples:")
    print("""
    # Datadog Integration
    from datadog import initialize, statsd
    
    def send_metrics_to_datadog(metrics):
        statsd.increment('ai.requests.total', metrics['request_count'])
        statsd.gauge('ai.latency.avg', metrics['avg_latency'])
        statsd.gauge('ai.cost.total', metrics['total_cost'])
    
    # Prometheus Integration
    from prometheus_client import Counter, Histogram, Gauge
    
    REQUEST_COUNT = Counter('ai_requests_total', 'Total AI requests')
    REQUEST_DURATION = Histogram('ai_request_duration_seconds', 'Request duration')
    COST_TOTAL = Gauge('ai_cost_total_dollars', 'Total AI cost')
    
    # OpenTelemetry Metrics
    from opentelemetry import metrics
    
    meter = metrics.get_meter(__name__)
    request_counter = meter.create_counter('ai_requests_total')
    latency_histogram = meter.create_histogram('ai_request_duration')
    """)

    return True


def main():
    """Main demonstration function."""

    print("Welcome to the Hugging Face Production Patterns Demo!")
    print()
    print("This example demonstrates enterprise-ready deployment patterns")
    print("for GenOps with Hugging Face in production environments.")
    print()

    success_count = 0
    total_demos = 4

    # Run all production pattern demonstrations
    demos = [
        ("High-Volume Processing", demonstrate_high_volume_processing),
        ("Async Telemetry Export", demonstrate_async_telemetry_export),
        ("Error Resilience", demonstrate_error_resilience),
        ("Monitoring Integration", demonstrate_monitoring_integration)
    ]

    for demo_name, demo_func in demos:
        print(f"🚀 Running {demo_name} Demo...")
        try:
            success = demo_func()
            if success:
                success_count += 1
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"⚠️ {demo_name} demo encountered issues")
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")

        print("-" * 60)
        print()

    # Summary
    if success_count >= 3:
        print("🎉 Production Patterns Demo Completed Successfully!")
        print()
        print("🏭 Enterprise Patterns Demonstrated:")
        print("   ✅ High-volume request processing with monitoring")
        print("   ✅ Asynchronous telemetry export for performance")
        print("   ✅ Comprehensive error handling and resilience")
        print("   ✅ Production monitoring and alerting integration")
        print()
        print("🛡️ Production-Ready Features:")
        print("   ✅ Circuit breaker patterns for fault tolerance")
        print("   ✅ Model fallback chains for reliability")
        print("   ✅ Performance metrics and health monitoring")
        print("   ✅ Cost tracking and budget alerting")
        print("   ✅ Observability platform integration")
        print()
        print("🚀 Production Deployment Checklist:")
        print("   1. Configure OpenTelemetry export to your observability platform")
        print("   2. Set up monitoring dashboards for key metrics")
        print("   3. Implement alerting rules for error rates and costs")
        print("   4. Configure circuit breakers and fallback models")
        print("   5. Set up automated scaling based on request volume")
        print("   6. Implement comprehensive error logging and debugging")
        print("   7. Create runbooks for common operational scenarios")
        print()
        print("📖 Advanced Topics:")
        print("   → Set up multi-region deployment for global availability")
        print("   → Implement A/B testing for model performance optimization")
        print("   → Configure auto-scaling based on cost and performance metrics")
        print("   → Set up compliance and audit logging for regulated industries")

    else:
        print(f"⚠️ {success_count}/{total_demos} production demos completed successfully")
        print()
        print("🔧 Production Deployment Considerations:")
        print("   • Ensure all dependencies are properly installed")
        print("   • Configure observability and monitoring systems")
        print("   • Set up proper error handling and alerting")
        print("   • Test failure scenarios and recovery procedures")
        print("   • Plan for scaling and capacity management")

    return 0 if success_count >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
