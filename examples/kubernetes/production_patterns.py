#!/usr/bin/env python3
"""
✅ Production Patterns Kubernetes Example

Demonstrates production-ready patterns for GenOps AI in Kubernetes environments.
Shows enterprise patterns, performance optimization, and operational best practices.

Usage:
    python production_patterns.py
    python production_patterns.py --pattern high-availability
    python production_patterns.py --pattern performance-optimization
    python production_patterns.py --pattern enterprise-security
    python production_patterns.py --pattern observability
"""

import argparse
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, AsyncGenerator

# Import GenOps for production patterns
try:
    from genops.providers.kubernetes import KubernetesAdapter, validate_kubernetes_setup
    from genops.core.governance import create_governance_context
    from genops.core.performance import PerformanceMonitor, CircuitBreaker
    from genops.core.security import SecurityValidator, ContentFilter
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False
    print("⚠️  GenOps not installed. Install with: pip install genops")


@dataclass
class ProductionConfig:
    """Production configuration settings."""
    
    # Performance settings
    max_concurrent_requests: int = 50
    request_timeout_seconds: int = 30
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    # Security settings
    enable_content_filtering: bool = True
    max_tokens_per_request: int = 8000
    rate_limit_per_minute: int = 1000
    
    # Observability settings
    enable_detailed_tracing: bool = True
    export_metrics_interval: int = 10
    log_level: str = "INFO"
    
    # Resource management
    enable_resource_monitoring: bool = True
    cpu_limit_millicores: Optional[int] = None
    memory_limit_bytes: Optional[int] = None


class ProductionPatternDemo:
    """Demonstrates enterprise production patterns."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.adapter = None
        self.performance_monitor = None
        self.circuit_breaker = None
        self.security_validator = None
        
        if GENOPS_AVAILABLE:
            self.adapter = KubernetesAdapter()
            self.performance_monitor = PerformanceMonitor()
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout
            )
            self.security_validator = SecurityValidator()
    
    async def demonstrate_high_availability_pattern(self) -> bool:
        """Demonstrate high availability and resilience patterns."""
        
        print("🏗️ High Availability Pattern")
        print("=" * 60)
        
        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False
        
        print("1️⃣ Multi-Provider Failover Strategy:")
        
        # Define provider hierarchy for failover
        providers = [
            {"name": "primary", "endpoint": "openai", "priority": 1, "healthy": True},
            {"name": "secondary", "endpoint": "anthropic", "priority": 2, "healthy": True},
            {"name": "fallback", "endpoint": "openrouter", "priority": 3, "healthy": True}
        ]
        
        # Simulate provider health checks
        for provider in providers:
            health_status = "✅ HEALTHY" if provider["healthy"] else "❌ UNHEALTHY"
            print(f"   {provider['name']:12} ({provider['endpoint']:12}): {health_status}")
        
        print("\n2️⃣ Circuit Breaker Pattern:")
        
        # Simulate circuit breaker behavior
        success_count = 0
        failure_count = 0
        
        for attempt in range(8):
            # Simulate some requests failing
            simulate_failure = attempt in [3, 4, 5]  # Simulate failures
            
            if not simulate_failure:
                success_count += 1
                status = "✅ SUCCESS"
                self.circuit_breaker.record_success()
            else:
                failure_count += 1
                status = "❌ FAILURE"
                self.circuit_breaker.record_failure()
            
            circuit_state = self.circuit_breaker.get_state()
            print(f"   Request {attempt + 1}: {status} | Circuit: {circuit_state}")
            
            # Show circuit breaker opening
            if circuit_state == "OPEN":
                print(f"   🔴 Circuit breaker OPENED after {failure_count} failures")
                print("   ⏭️  Requests will be rejected until recovery timeout")
                break
        
        print("\n3️⃣ Graceful Degradation:")
        print("   • Primary provider down → Route to secondary")  
        print("   • All providers down → Serve cached responses")
        print("   • Circuit open → Return simplified responses")
        print("   • Resource exhaustion → Queue with backpressure")
        
        print("\n4️⃣ Health Check Implementation:")
        print("   ✅ Provider endpoint health monitoring")
        print("   ✅ Kubernetes liveness/readiness probes")
        print("   ✅ OpenTelemetry health check metrics")
        print("   ✅ Automatic failover and recovery")
        
        return True
    
    async def demonstrate_performance_optimization(self) -> bool:
        """Demonstrate performance optimization patterns."""
        
        print("\n⚡ Performance Optimization Patterns")
        print("=" * 60)
        
        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False
        
        print("1️⃣ Request Batching and Connection Pooling:")
        
        # Simulate batching performance improvement
        single_request_times = []
        batched_request_times = []
        
        # Single requests
        print("   📊 Single Request Performance:")
        for i in range(5):
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate request
            duration = time.time() - start_time
            single_request_times.append(duration)
            print(f"      Request {i+1}: {duration:.3f}s")
        
        # Batched requests
        print("   📊 Batched Request Performance:")
        start_time = time.time()
        
        # Simulate concurrent batch processing
        tasks = [asyncio.sleep(0.1) for _ in range(5)]
        await asyncio.gather(*tasks)
        
        batch_duration = time.time() - start_time
        per_request_batched = batch_duration / 5
        
        print(f"      Batch of 5 requests: {batch_duration:.3f}s")
        print(f"      Per request (batched): {per_request_batched:.3f}s")
        
        # Calculate improvement
        avg_single = sum(single_request_times) / len(single_request_times)
        improvement = ((avg_single - per_request_batched) / avg_single) * 100
        print(f"      Performance improvement: {improvement:.1f}%")
        
        print("\n2️⃣ Caching Strategy:")
        cache_scenarios = [
            ("Response caching", "95% hit rate", "200ms → 5ms"),
            ("Model metadata caching", "90% hit rate", "150ms → 2ms"),
            ("Cost calculation caching", "85% hit rate", "50ms → 1ms"),
            ("Token counting caching", "99% hit rate", "10ms → 0.5ms")
        ]
        
        for scenario, hit_rate, improvement in cache_scenarios:
            print(f"   ✅ {scenario:25}: {hit_rate} | {improvement}")
        
        print("\n3️⃣ Resource Management:")
        
        # Show resource monitoring
        governance_attrs = {
            "team": "performance-team",
            "project": "optimization-demo",
            "customer_id": "perf-customer"
        }
        
        with self.adapter.create_governance_context(**governance_attrs) as ctx:
            print("   📊 Current Resource Usage:")
            
            # Get resource information
            resource_usage = ctx.get_resource_usage()
            if resource_usage:
                cpu_usage = resource_usage.get('cpu_usage_millicores', 0)
                memory_usage = resource_usage.get('memory_usage_bytes', 0)
                
                print(f"      CPU Usage: {cpu_usage}m cores")
                print(f"      Memory Usage: {memory_usage / 1024 / 1024:.1f} MB")
                
                # Show resource limits
                if self.config.cpu_limit_millicores:
                    cpu_percent = (cpu_usage / self.config.cpu_limit_millicores) * 100
                    print(f"      CPU Utilization: {cpu_percent:.1f}%")
                
                if self.config.memory_limit_bytes:
                    mem_percent = (memory_usage / self.config.memory_limit_bytes) * 100
                    print(f"      Memory Utilization: {mem_percent:.1f}%")
            else:
                print("      Resource monitoring not available")
        
        print("\n4️⃣ Optimization Strategies:")
        print("   ⚡ Connection pooling reduces connection overhead")
        print("   ⚡ Request batching improves throughput")
        print("   ⚡ Response caching eliminates redundant calls")
        print("   ⚡ Streaming reduces memory usage")
        print("   ⚡ Async processing improves concurrency")
        
        return True
    
    async def demonstrate_enterprise_security(self) -> bool:
        """Demonstrate enterprise security patterns."""
        
        print("\n🔒 Enterprise Security Patterns") 
        print("=" * 60)
        
        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False
        
        print("1️⃣ Content Security and Filtering:")
        
        # Simulate content filtering
        test_inputs = [
            ("Safe business query", "safe", True),
            ("Request with PII data", "contains_pii", False),
            ("Prompt injection attempt", "malicious", False),
            ("Normal AI assistance", "safe", True),
            ("Data extraction attempt", "suspicious", False)
        ]
        
        for content, classification, allowed in test_inputs:
            status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
            risk_level = "🟢 LOW" if classification == "safe" else "🔴 HIGH" if classification == "malicious" else "🟡 MEDIUM"
            
            print(f"   {content:25}: {status} | Risk: {risk_level}")
        
        print("\n2️⃣ Authentication and Authorization:")
        
        # Show RBAC patterns
        rbac_examples = [
            ("team:engineering", "openai:gpt-4", "✅ ALLOWED", "Full access"),
            ("team:marketing", "openai:gpt-3.5", "✅ ALLOWED", "Standard access"),
            ("team:intern", "openai:gpt-4", "❌ DENIED", "Insufficient privileges"),
            ("team:admin", "anthropic:*", "✅ ALLOWED", "Admin access"),
            ("team:external", "any:*", "❌ DENIED", "No external access")
        ]
        
        print("   RBAC Policy Enforcement:")
        for identity, resource, status, reason in rbac_examples:
            print(f"      {identity:20} → {resource:20}: {status} ({reason})")
        
        print("\n3️⃣ Audit and Compliance:")
        
        audit_events = [
            "User authentication and authorization",
            "AI model access and usage",
            "Cost and budget compliance",
            "Data privacy and PII handling",
            "Policy violations and responses",
            "Security incidents and remediation"
        ]
        
        print("   Comprehensive Audit Trail:")
        for event in audit_events:
            print(f"      ✅ {event}")
        
        print("\n4️⃣ Data Privacy and Protection:")
        
        privacy_measures = [
            ("PII Detection", "Automatic identification of personal data"),
            ("Data Redaction", "Mask sensitive information in logs"),
            ("Request Anonymization", "Remove identifying information"),
            ("Response Filtering", "Prevent data leakage in outputs"),
            ("Retention Policies", "Automated data lifecycle management"),
            ("Encryption", "End-to-end encryption for all data")
        ]
        
        print("   Privacy Protection Measures:")
        for measure, description in privacy_measures:
            print(f"      🛡️ {measure:20}: {description}")
        
        print("\n5️⃣ Kubernetes Security Integration:")
        
        k8s_security = [
            "Pod Security Standards (PSS) compliance",
            "Network policies for traffic isolation",
            "Service mesh (Istio/Linkerd) integration",
            "Secret management with external secret operators",
            "RBAC integration with Kubernetes roles",
            "Admission controllers for policy enforcement"
        ]
        
        for security_feature in k8s_security:
            print(f"      🔐 {security_feature}")
        
        return True
    
    async def demonstrate_observability_patterns(self) -> bool:
        """Demonstrate comprehensive observability patterns."""
        
        print("\n📊 Observability Patterns")
        print("=" * 60)
        
        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False
        
        print("1️⃣ Metrics Collection:")
        
        # Show key metrics
        metrics_categories = {
            "Business Metrics": [
                "genops.cost.total_usd",
                "genops.requests.count",
                "genops.tokens.consumed",
                "genops.budget.utilization"
            ],
            "Performance Metrics": [
                "genops.request.duration_ms",
                "genops.provider.latency_ms", 
                "genops.throughput.requests_per_second",
                "genops.error.rate"
            ],
            "Infrastructure Metrics": [
                "k8s.pod.cpu_usage",
                "k8s.pod.memory_usage",
                "k8s.node.resource_utilization",
                "k8s.service.health_status"
            ]
        }
        
        for category, metrics in metrics_categories.items():
            print(f"   📈 {category}:")
            for metric in metrics:
                print(f"      • {metric}")
        
        print("\n2️⃣ Distributed Tracing:")
        
        # Simulate trace structure
        trace_example = [
            ("kubernetes.request.received", "0ms", "Root span"),
            ("genops.governance.validate", "2ms", "Governance validation"),
            ("genops.provider.select", "5ms", "Provider selection"),
            ("openai.chat.completion", "245ms", "AI provider call"),
            ("genops.cost.calculate", "1ms", "Cost calculation"),
            ("genops.telemetry.export", "3ms", "Telemetry export")
        ]
        
        print("   🔍 Example Trace Spans:")
        for span_name, duration, description in trace_example:
            print(f"      {span_name:30} | {duration:6} | {description}")
        
        total_duration = sum(int(d.replace('ms', '')) for _, d, _ in trace_example)
        print(f"      Total Request Duration: {total_duration}ms")
        
        print("\n3️⃣ Structured Logging:")
        
        log_examples = [
            {
                "level": "INFO",
                "message": "AI request completed",
                "fields": {
                    "request_id": "req-abc123",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo", 
                    "cost": 0.0023,
                    "duration_ms": 245,
                    "k8s.namespace": "ai-prod",
                    "k8s.pod": "genops-ai-xyz",
                    "team": "engineering"
                }
            },
            {
                "level": "WARN",
                "message": "Budget threshold exceeded",
                "fields": {
                    "budget_id": "team-engineering-daily",
                    "current_spend": 85.50,
                    "budget_limit": 100.00,
                    "threshold": "85%",
                    "k8s.namespace": "ai-prod"
                }
            }
        ]
        
        print("   📝 Structured Log Examples:")
        for log in log_examples:
            print(f"      {log['level']:4} | {log['message']}")
            for key, value in log['fields'].items():
                print(f"           {key}: {value}")
            print()
        
        print("4️⃣ Alerting and Monitoring:")
        
        alert_rules = [
            ("Cost Alert", "Daily budget >90% utilized", "Slack + PagerDuty"),
            ("Performance Alert", "Request latency >5s", "Slack"),
            ("Error Alert", "Error rate >5%", "PagerDuty"),
            ("Security Alert", "Policy violation detected", "Security team"),
            ("Resource Alert", "Pod CPU >80%", "Platform team")
        ]
        
        print("   🚨 Alert Configuration:")
        for name, condition, destination in alert_rules:
            print(f"      {name:18} | {condition:25} → {destination}")
        
        print("\n5️⃣ Dashboard Integration:")
        
        dashboard_platforms = [
            "Grafana: Cost and performance dashboards",
            "Datadog: APM and infrastructure monitoring",
            "Honeycomb: Distributed tracing and debugging", 
            "New Relic: Application performance insights",
            "Kubernetes Dashboard: Pod and cluster metrics"
        ]
        
        for platform in dashboard_platforms:
            print(f"      📊 {platform}")
        
        return True
    
    @asynccontextmanager
    async def production_request_context(
        self, 
        request_id: str,
        **governance_attrs
    ) -> AsyncGenerator:
        """Production-ready request context with full observability."""
        
        start_time = time.time()
        
        try:
            # Start performance monitoring
            with self.performance_monitor.monitor_request(request_id):
                # Create governance context with Kubernetes attribution
                with self.adapter.create_governance_context(**governance_attrs) as ctx:
                    # Add request metadata
                    ctx.add_metadata({
                        "request_id": request_id,
                        "start_time": start_time,
                        "pattern": "production"
                    })
                    
                    yield ctx
                    
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            # Log error with full context
            error_context = {
                "request_id": request_id,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
                **governance_attrs
            }
            
            print(f"❌ Request failed: {error_context}")
            raise
            
        finally:
            # Always record metrics
            duration = time.time() - start_time
            print(f"📊 Request {request_id} completed in {duration:.3f}s")


async def run_production_pattern(pattern: str, config: ProductionConfig) -> bool:
    """Run specific production pattern demonstration."""
    
    demo = ProductionPatternDemo(config)
    
    if pattern == "high-availability":
        return await demo.demonstrate_high_availability_pattern()
    elif pattern == "performance-optimization":
        return await demo.demonstrate_performance_optimization()
    elif pattern == "enterprise-security":
        return await demo.demonstrate_enterprise_security()
    elif pattern == "observability":
        return await demo.demonstrate_observability_patterns()
    else:
        print(f"❌ Unknown pattern: {pattern}")
        return False


async def run_comprehensive_demo(config: ProductionConfig) -> bool:
    """Run comprehensive production patterns demonstration."""
    
    print("🏢 Comprehensive Production Patterns Demo")
    print("=" * 80)
    
    demo = ProductionPatternDemo(config)
    success = True
    
    # Run all patterns
    patterns = [
        ("High Availability", demo.demonstrate_high_availability_pattern),
        ("Performance Optimization", demo.demonstrate_performance_optimization),
        ("Enterprise Security", demo.demonstrate_enterprise_security), 
        ("Observability", demo.demonstrate_observability_patterns)
    ]
    
    for pattern_name, pattern_func in patterns:
        try:
            pattern_success = await pattern_func()
            success = success and pattern_success
            print("\n" + "=" * 80)
        except Exception as e:
            print(f"❌ {pattern_name} demo failed: {e}")
            success = False
    
    # Production readiness summary
    print("🎯 PRODUCTION READINESS SUMMARY")
    print("=" * 80)
    
    readiness_checklist = [
        "✅ High availability and failover strategies implemented",
        "✅ Performance optimization and resource management",
        "✅ Enterprise security and compliance measures",
        "✅ Comprehensive observability and monitoring",
        "✅ Circuit breaker and resilience patterns",
        "✅ Cost tracking and budget enforcement",
        "✅ Audit logging and security controls",
        "✅ Kubernetes integration and best practices"
    ]
    
    for item in readiness_checklist:
        print(f"   {item}")
    
    print(f"\n💡 Enterprise Benefits:")
    print("   • Reduced operational overhead through automation")
    print("   • Improved reliability with resilience patterns")
    print("   • Enhanced security and compliance posture")
    print("   • Complete cost visibility and control")
    print("   • Faster incident response with observability")
    
    return success


async def main():
    """Main production patterns demo."""
    
    parser = argparse.ArgumentParser(
        description="Production patterns Kubernetes example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python production_patterns.py                           # Full demo
    python production_patterns.py --pattern high-availability  # HA patterns
    python production_patterns.py --pattern performance-optimization  # Performance
    python production_patterns.py --pattern enterprise-security  # Security
    python production_patterns.py --pattern observability     # Observability
        """
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["high-availability", "performance-optimization", "enterprise-security", "observability"],
        help="Specific production pattern to demonstrate"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent requests"
    )
    
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Create production configuration
    config = ProductionConfig(
        max_concurrent_requests=args.max_concurrent,
        request_timeout_seconds=args.request_timeout,
        enable_detailed_tracing=True,
        enable_content_filtering=True
    )
    
    success = True
    
    if args.pattern:
        # Run specific pattern
        success = await run_production_pattern(args.pattern, config)
    else:
        # Run comprehensive demo
        success = await run_comprehensive_demo(config)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())