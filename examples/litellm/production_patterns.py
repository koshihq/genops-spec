#!/usr/bin/env python3
"""
LiteLLM Production Deployment Patterns with GenOps

Demonstrates enterprise deployment patterns and scaling strategies for
LiteLLM + GenOps integration in production environments. This showcases
patterns for high-availability, performance optimization, and enterprise
governance requirements.

Usage:
    export OPENAI_API_KEY="your_key_here"
    python production_patterns.py

Features:
    - High-availability deployment patterns
    - Performance optimization for scale
    - Enterprise governance configurations
    - Monitoring and alerting integration
    - Circuit breaker and fallback strategies
    - Multi-tenant isolation patterns
"""

import os
import sys
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for production examples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration for LiteLLM + GenOps deployment."""
    
    # High Availability
    primary_providers: List[str] = field(default_factory=lambda: ["openai", "anthropic"])
    fallback_providers: List[str] = field(default_factory=lambda: ["google", "cohere"])
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Performance
    max_concurrent_requests: int = 100
    request_rate_limit: float = 10.0  # requests per second
    enable_request_batching: bool = True
    batch_size: int = 10
    
    # Governance
    daily_budget_limit: float = 1000.0
    governance_policy: str = "enforced"  # advisory, enforced, or strict
    enable_cost_tracking: bool = True
    enable_compliance_logging: bool = True
    
    # Monitoring
    enable_health_checks: bool = True
    health_check_interval: int = 60  # seconds
    alert_cost_threshold: float = 800.0  # 80% of budget
    
    # Multi-tenant
    tenant_isolation: bool = True
    per_tenant_budgets: bool = True


class ProductionLiteLLMManager:
    """Production-ready LiteLLM manager with GenOps governance."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.is_initialized = False
        self.health_status = {}
        self.current_costs = {}
        self.request_counts = {}
        self._lock = threading.RLock()
        
    def initialize(self) -> bool:
        """Initialize the production LiteLLM manager."""
        try:
            import litellm
            from genops.providers.litellm import auto_instrument, get_usage_stats
            
            # Configure LiteLLM for production
            litellm.set_verbose = False  # Reduce logging noise
            litellm.drop_params = True   # Drop unsupported params
            
            # Enable GenOps governance
            success = auto_instrument(
                team="production-ai",
                project="enterprise-service", 
                environment="production",
                daily_budget_limit=self.config.daily_budget_limit,
                governance_policy=self.config.governance_policy,
                enable_cost_tracking=self.config.enable_cost_tracking
            )
            
            if success:
                self.is_initialized = True
                logger.info("Production LiteLLM manager initialized successfully")
                
                # Start health monitoring
                if self.config.enable_health_checks:
                    self._start_health_monitoring()
                
                return True
            else:
                logger.error("Failed to initialize GenOps auto-instrumentation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize production manager: {e}")
            return False
    
    def _start_health_monitoring(self):
        """Start background health monitoring."""
        def health_monitor():
            while self.is_initialized:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
        
        monitoring_thread = threading.Thread(target=health_monitor, daemon=True)
        monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def _perform_health_checks(self):
        """Perform health checks on configured providers."""
        for provider in self.config.primary_providers + self.config.fallback_providers:
            try:
                # Simple health check - test provider availability
                health_result = self._check_provider_health(provider)
                with self._lock:
                    self.health_status[provider] = {
                        'healthy': health_result,
                        'last_check': time.time(),
                        'check_count': self.health_status.get(provider, {}).get('check_count', 0) + 1
                    }
            except Exception as e:
                logger.warning(f"Health check failed for {provider}: {e}")
                with self._lock:
                    self.health_status[provider] = {
                        'healthy': False,
                        'last_check': time.time(),
                        'error': '[Error details redacted for security]'
                    }
    
    def _check_provider_health(self, provider: str) -> bool:
        """Check if a provider is healthy and accessible."""
        # In production, this would be more sophisticated
        # For demo, we'll simulate health checks
        api_key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'cohere': 'COHERE_API_KEY'
        }
        
        return bool(os.getenv(api_key_mapping.get(provider)))
    
    def get_available_providers(self) -> List[str]:
        """Get list of currently healthy providers."""
        available = []
        with self._lock:
            for provider, status in self.health_status.items():
                if status.get('healthy', False):
                    available.append(provider)
        return available
    
    def select_optimal_provider(
        self, 
        use_case: str = "general",
        cost_preference: float = 0.5  # 0.0 = cheapest, 1.0 = highest quality
    ) -> Optional[str]:
        """Select optimal provider based on availability and preferences."""
        available_providers = self.get_available_providers()
        
        if not available_providers:
            logger.warning("No healthy providers available")
            return None
        
        # Simple provider selection logic (in production, this would be more sophisticated)
        primary_available = [p for p in self.config.primary_providers if p in available_providers]
        
        if primary_available:
            return primary_available[0]
        
        fallback_available = [p for p in self.config.fallback_providers if p in available_providers]
        return fallback_available[0] if fallback_available else None


def check_production_setup():
    """Check production environment setup."""
    print("🔍 Checking production environment setup...")
    
    # Check imports
    try:
        import litellm
        from genops.providers.litellm import auto_instrument, get_usage_stats, get_cost_summary
        print("✅ LiteLLM and GenOps available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install litellm genops[litellm]")
        return False
    
    # Check for production-grade API keys
    production_providers = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
    configured_providers = [key for key in production_providers if os.getenv(key)]
    
    if len(configured_providers) < 2:
        print(f"⚠️  Only {len(configured_providers)} provider(s) configured")
        print("💡 Production deployments should configure multiple providers for redundancy")
        print("   Configure at least: OpenAI and Anthropic for high availability")
    else:
        print(f"✅ {len(configured_providers)} providers configured for high availability")
    
    return True


def demo_high_availability_patterns():
    """Demonstrate high availability deployment patterns."""
    print("\n" + "="*60)
    print("🏗️ Demo: High Availability Patterns")
    print("="*60)
    
    print("Enterprise HA patterns for production LiteLLM + GenOps:")
    print("• Multi-provider redundancy with automatic failover")
    print("• Health monitoring and circuit breaker patterns")
    print("• Graceful degradation under load or failures")
    
    # Initialize production manager
    config = ProductionConfig(
        primary_providers=["openai", "anthropic"],
        fallback_providers=["google", "cohere"],
        max_retries=3,
        timeout_seconds=10
    )
    
    manager = ProductionLiteLLMManager(config)
    
    print(f"\n📋 Initializing production manager...")
    if not manager.initialize():
        print("❌ Failed to initialize production manager")
        return
    
    print("✅ Production manager initialized with HA configuration")
    
    # Simulate health monitoring
    print(f"\n🏥 Health monitoring active:")
    print(f"   • Primary providers: {config.primary_providers}")
    print(f"   • Fallback providers: {config.fallback_providers}")
    print(f"   • Health check interval: {config.health_check_interval}s")
    
    # Wait for initial health checks
    time.sleep(2)
    
    available_providers = manager.get_available_providers()
    print(f"   • Currently healthy: {available_providers}")
    
    # Demonstrate provider selection
    print(f"\n🎯 Optimal provider selection:")
    for use_case, cost_pref in [("general", 0.3), ("premium", 0.8), ("bulk", 0.1)]:
        provider = manager.select_optimal_provider(use_case, cost_pref)
        print(f"   • {use_case} use case (cost pref {cost_pref}): {provider}")


def demo_performance_optimization():
    """Demonstrate performance optimization patterns."""
    print("\n" + "="*60)
    print("⚡ Demo: Performance Optimization")
    print("="*60)
    
    print("Production performance patterns:")
    print("• Concurrent request handling with rate limiting")
    print("• Request batching for efficiency")
    print("• Asynchronous processing with governance")
    
    import litellm
    from genops.providers.litellm import track_completion
    
    # Simulate concurrent request processing
    print(f"\n📋 Concurrent Request Processing Demo:")
    
    def process_request(request_id: int, customer_id: str) -> Dict[str, Any]:
        """Process a single request with governance tracking."""
        try:
            with track_completion(
                model="gpt-3.5-turbo",
                team="production-service",
                project="customer-api", 
                customer_id=customer_id,
                custom_tags={
                    "request_id": f"req-{request_id}",
                    "processing_mode": "concurrent"
                }
            ) as context:
                
                # Simulate API request processing time
                processing_time = 0.1 + (request_id % 3) * 0.1  # Variable processing time
                time.sleep(processing_time)
                
                return {
                    "request_id": request_id,
                    "customer_id": customer_id,
                    "status": "completed",
                    "processing_time": processing_time,
                    "cost": context.cost if hasattr(context, 'cost') else 0.001
                }
                
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "failed",
                "error": "[Error details redacted for security]"
            }
    
    # Process requests concurrently
    requests = [
        (i, f"customer-{i % 5}") for i in range(20)  # 20 requests across 5 customers
    ]
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all requests
        future_to_request = {
            executor.submit(process_request, req_id, customer_id): (req_id, customer_id)
            for req_id, customer_id in requests
        }
        
        # Collect results
        completed = 0
        failed = 0
        
        for future in as_completed(future_to_request):
            result = future.result()
            if result["status"] == "completed":
                completed += 1
            else:
                failed += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"   📊 Performance Results:")
    print(f"      • Total requests: {len(requests)}")
    print(f"      • Completed: {completed}")
    print(f"      • Failed: {failed}")
    print(f"      • Total time: {total_time:.2f}s")
    print(f"      • Requests/second: {len(requests)/total_time:.1f}")
    print(f"      • Concurrent governance tracking: ✅")


def demo_enterprise_governance():
    """Demonstrate enterprise governance patterns."""
    print("\n" + "="*60)
    print("🏢 Demo: Enterprise Governance")
    print("="*60)
    
    print("Enterprise governance patterns:")
    print("• Multi-tenant isolation and budget allocation")
    print("• Compliance logging and audit trails")
    print("• Cost center attribution and reporting")
    
    from genops.providers.litellm import auto_instrument, get_cost_summary
    
    # Enterprise governance configuration
    enterprise_teams = [
        {
            "team": "customer-support",
            "cost_center": "operations", 
            "budget_limit": 200.0,
            "compliance_level": "standard"
        },
        {
            "team": "product-ai",
            "cost_center": "engineering",
            "budget_limit": 500.0, 
            "compliance_level": "strict"
        },
        {
            "team": "sales-ai",
            "cost_center": "revenue",
            "budget_limit": 300.0,
            "compliance_level": "standard"
        }
    ]
    
    print(f"\n📋 Enterprise Team Configuration:")
    
    for team_config in enterprise_teams:
        print(f"\n   🏢 Team: {team_config['team']}")
        print(f"      • Cost center: {team_config['cost_center']}")
        print(f"      • Budget: ${team_config['budget_limit']}")
        print(f"      • Compliance: {team_config['compliance_level']}")
        
        # Configure team-specific governance
        governance_policy = "strict" if team_config['compliance_level'] == "strict" else "enforced"
        
        success = auto_instrument(
            team=team_config['team'],
            project="enterprise-ai",
            environment="production",
            daily_budget_limit=team_config['budget_limit'],
            governance_policy=governance_policy,
            enable_cost_tracking=True,
            
            # Enterprise attributes
            cost_center=team_config['cost_center'],
            compliance_level=team_config['compliance_level']
        )
        
        if success:
            print(f"      ✅ Governance configured")
        else:
            print(f"      ⚠️ Governance configuration failed")
    
    # Demonstrate cost reporting
    print(f"\n📊 Enterprise Cost Reporting:")
    
    cost_summary = get_cost_summary(group_by="team")
    
    if cost_summary.get('cost_by_team'):
        total_cost = cost_summary['total_cost']
        print(f"   💰 Total enterprise cost: ${total_cost:.6f}")
        
        for team, cost in cost_summary['cost_by_team'].items():
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
            print(f"   • {team}: ${cost:.6f} ({percentage:.1f}%)")
    else:
        print(f"   📈 No cost data yet - configure with live API calls for reporting")


def demo_monitoring_integration():
    """Demonstrate monitoring and alerting integration."""
    print("\n" + "="*60)
    print("📊 Demo: Monitoring & Alerting Integration") 
    print("="*60)
    
    print("Production monitoring patterns:")
    print("• Real-time cost tracking with budget alerts")
    print("• Performance metrics and SLA monitoring")
    print("• Provider health and failover monitoring")
    
    from genops.providers.litellm import get_usage_stats, get_cost_summary
    
    # Production monitoring configuration
    monitoring_config = {
        "cost_alert_threshold": 0.8,  # 80% of budget
        "latency_sla_ms": 2000,       # 2 second SLA
        "error_rate_threshold": 0.05,  # 5% error rate
        "health_check_interval": 60    # 1 minute
    }
    
    print(f"\n📋 Monitoring Configuration:")
    for metric, threshold in monitoring_config.items():
        print(f"   • {metric}: {threshold}")
    
    # Simulate monitoring dashboard
    print(f"\n📊 Production Monitoring Dashboard:")
    
    # Get current usage statistics
    stats = get_usage_stats()
    cost_summary = get_cost_summary(group_by="provider")
    
    print(f"   🎯 Current Session Metrics:")
    print(f"   • Total requests: {stats.get('total_requests', 0)}")
    print(f"   • Total cost: ${stats.get('total_cost', 0):.6f}")
    print(f"   • Average latency: {stats.get('avg_latency_ms', 0):.0f}ms")
    print(f"   • Error rate: {stats.get('error_rate', 0):.2%}")
    
    if cost_summary.get('cost_by_provider'):
        print(f"   📈 Provider Cost Breakdown:")
        for provider, cost in cost_summary['cost_by_provider'].items():
            print(f"      • {provider}: ${cost:.6f}")
    
    # Simulate alerting logic
    budget_limit = 1000.0
    current_cost = stats.get('total_cost', 0)
    cost_percentage = (current_cost / budget_limit) * 100
    
    print(f"\n🚨 Alert Status:")
    if cost_percentage > 80:
        print(f"   ⚠️ BUDGET ALERT: {cost_percentage:.1f}% of budget used")
        print(f"   💡 Action: Review spending and consider cost optimization")
    else:
        print(f"   ✅ Budget healthy: {cost_percentage:.1f}% of budget used")
    
    # Health monitoring summary
    print(f"\n🏥 Provider Health Summary:")
    providers = ["openai", "anthropic", "google", "cohere"]
    for provider in providers:
        has_key = bool(os.getenv(f"{provider.upper()}_API_KEY"))
        status = "🟢 Healthy" if has_key else "🔴 Unavailable"
        print(f"   • {provider}: {status}")


def demo_circuit_breaker_patterns():
    """Demonstrate circuit breaker and resilience patterns."""
    print("\n" + "="*60)
    print("🔄 Demo: Circuit Breaker & Resilience Patterns")
    print("="*60)
    
    print("Resilience patterns for production stability:")
    print("• Circuit breaker for failing providers")
    print("• Automatic retry with exponential backoff")
    print("• Graceful degradation and fallback strategies")
    
    class SimpleCircuitBreaker:
        """Simple circuit breaker for demonstration."""
        
        def __init__(self, failure_threshold: int = 3, timeout: int = 60):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "closed"  # closed, open, half-open
        
        def call(self, func, *args, **kwargs):
            """Execute function with circuit breaker protection."""
            if self.state == "open":
                if time.time() - self.last_failure_time < self.timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = "half-open"
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    print(f"   🔴 Circuit breaker OPENED for {func.__name__}")
                
                raise e
    
    # Simulate circuit breaker usage
    def simulate_api_call(provider: str, fail_rate: float = 0.3) -> str:
        """Simulate API call with configurable failure rate."""
        import random
        if random.random() < fail_rate:
            raise Exception(f"Simulated {provider} API failure")
        return f"Success from {provider}"
    
    print(f"\n📋 Circuit Breaker Demo:")
    
    providers = ["openai", "anthropic", "google"]
    circuit_breakers = {provider: SimpleCircuitBreaker() for provider in providers}
    
    # Simulate requests with failures
    for round_num in range(3):
        print(f"\n   🔄 Request Round {round_num + 1}:")
        
        for provider in providers:
            cb = circuit_breakers[provider]
            try:
                # Simulate higher failure rate for round 2 to trigger circuit breaker
                fail_rate = 0.8 if round_num == 1 else 0.2
                result = cb.call(simulate_api_call, provider, fail_rate)
                print(f"      ✅ {provider}: {result} (state: {cb.state})")
            except Exception as e:
                print(f"      ❌ {provider}: [Error details redacted for security] (state: {cb.state})")
    
    print(f"\n   📊 Circuit Breaker States:")
    for provider, cb in circuit_breakers.items():
        print(f"      • {provider}: {cb.state.upper()} (failures: {cb.failure_count})")


def main():
    """Run the complete production patterns demonstration."""
    
    print("🏗️ LiteLLM + GenOps: Production Deployment Patterns")
    print("=" * 70)
    print("Enterprise-grade deployment strategies for scaled AI governance")
    print("High availability, performance optimization, and enterprise governance")
    
    # Check setup
    if not check_production_setup():
        print("\n❌ Production setup incomplete. Please resolve issues above.")
        return 1
    
    try:
        # Run demonstrations
        demo_high_availability_patterns()
        demo_performance_optimization()
        demo_enterprise_governance()
        demo_monitoring_integration()
        demo_circuit_breaker_patterns()
        
        print("\n" + "="*60)
        print("🎉 Production Deployment Patterns Complete!")
        
        print("\n🏗️ Production Patterns Demonstrated:")
        print("   ✅ High availability with multi-provider redundancy")
        print("   ✅ Performance optimization for concurrent workloads")
        print("   ✅ Enterprise governance with multi-tenant isolation")
        print("   ✅ Monitoring and alerting integration")
        print("   ✅ Circuit breaker and resilience patterns")
        
        print("\n🚀 Production Deployment Checklist:")
        print("   • Configure multiple providers for high availability")
        print("   • Set appropriate budget limits and governance policies") 
        print("   • Implement health monitoring and alerting")
        print("   • Configure circuit breakers for resilience")
        print("   • Set up multi-tenant isolation for enterprise use")
        print("   • Monitor cost trends and optimize regularly")
        
        print("\n📖 Next Steps:")
        print("   • Deploy with your observability stack (Datadog, Grafana, etc.)")
        print("   • Configure alerts for budget and performance thresholds")
        print("   • Implement provider rotation strategies")
        print("   • Set up compliance monitoring for audit requirements")
        print("   • Scale with container orchestration (Kubernetes, Docker)")
        
        print("\n🏢 Enterprise Integration:")
        print("   • Single instrumentation → ecosystem-wide governance")
        print("   • OpenTelemetry export → existing observability tools")
        print("   • Multi-tenant → customer attribution and billing")
        print("   • Production-ready → high availability and compliance")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        print("💡 For debugging, check your API key configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)