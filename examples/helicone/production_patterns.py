#!/usr/bin/env python3
"""
Helicone Production Deployment Patterns

This example demonstrates enterprise-ready production deployment patterns
for GenOps + Helicone integration, including high availability, monitoring,
error handling, and scalability considerations.

Usage:
    python production_patterns.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    export OPENAI_API_KEY="your_openai_api_key"
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager


class ProductionHeliconeManager:
    """Production-ready Helicone manager with enterprise features."""
    
    def __init__(self):
        self.adapter = None
        self.logger = self._setup_logging()
        self._initialize_adapter()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up production logging."""
        logger = logging.getLogger('helicone_production')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_adapter(self):
        """Initialize Helicone adapter with production settings."""
        try:
            from genops.providers.helicone import instrument_helicone
            
            self.adapter = instrument_helicone(
                # Production environment settings
                helicone_api_key=os.getenv('HELICONE_API_KEY'),
                provider_keys={
                    'openai': os.getenv('OPENAI_API_KEY'),
                    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
                    'groq': os.getenv('GROQ_API_KEY')
                },
                
                # Production governance
                team=os.getenv('TEAM_NAME', 'production'),
                project=os.getenv('PROJECT_NAME', 'main-application'),
                environment=os.getenv('ENVIRONMENT', 'production'),
                auto_instrument_providers=True
            )
            
            self.logger.info("✅ Production Helicone adapter initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize adapter: {e}")
            raise
    
    @contextmanager
    def production_request_context(self, operation_name: str, **kwargs):
        """Production request context with comprehensive monitoring."""
        start_time = time.time()
        operation_id = f"{operation_name}-{int(start_time)}"
        
        self.logger.info(f"🚀 Starting operation: {operation_id}")
        
        try:
            yield operation_id
            
        except Exception as e:
            self.logger.error(f"❌ Operation {operation_id} failed: {e}")
            # In production, you might send to alerting system
            self._send_alert(f"Operation failed: {operation_id}", str(e))
            raise
            
        finally:
            duration = time.time() - start_time
            self.logger.info(f"✅ Operation {operation_id} completed in {duration:.2f}s")
    
    def _send_alert(self, title: str, message: str):
        """Send alert to monitoring system (stubbed for demo)."""
        self.logger.warning(f"🚨 ALERT: {title} - {message}")
        # In production: send to PagerDuty, Slack, etc.
    
    def make_resilient_request(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make a resilient request with comprehensive error handling."""
        
        with self.production_request_context("ai_request") as operation_id:
            
            # Production request with fallbacks
            providers = ['openai', 'anthropic', 'groq']
            
            for attempt, provider in enumerate(providers, 1):
                try:
                    self.logger.info(f"🎯 Attempt {attempt}: Using {provider}")
                    
                    response = self.adapter.chat(
                        message=query,
                        provider=provider,
                        model=self._get_optimal_model(provider),
                        
                        # Production metadata
                        customer_id=kwargs.get('customer_id', 'default'),
                        operation_id=operation_id,
                        **kwargs
                    )
                    
                    # Extract production metrics
                    result = {
                        'content': response.content if hasattr(response, 'content') else str(response),
                        'provider': provider,
                        'model': getattr(response, 'model', 'unknown'),
                        'cost': getattr(response.usage, 'total_cost', 0.0) if hasattr(response, 'usage') else 0.0,
                        'tokens': getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                        'operation_id': operation_id
                    }
                    
                    self.logger.info(f"✅ Success with {provider}: ${result['cost']:.6f}")
                    return result
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  {provider} failed (attempt {attempt}): {e}")
                    
                    if attempt == len(providers):
                        # All providers failed
                        self.logger.error("❌ All providers failed")
                        self._send_alert("All AI providers failed", str(e))
                        raise
                    
                    continue
        
        return None
    
    def _get_optimal_model(self, provider: str) -> str:
        """Get optimal model for each provider."""
        model_map = {
            'openai': 'gpt-3.5-turbo',
            'anthropic': 'claude-3-haiku-20240307',
            'groq': 'mixtral-8x7b-32768'
        }
        return model_map.get(provider, 'default')


def demonstrate_production_deployment():
    """Show production deployment patterns."""
    
    print("🏭 Production Deployment Patterns")
    print("=" * 35)
    
    try:
        manager = ProductionHeliconeManager()
        
        # Production test scenarios
        scenarios = [
            {
                'name': 'Customer Support Query',
                'query': 'How can I reset my password?',
                'customer_id': 'customer-12345',
                'priority': 'high'
            },
            {
                'name': 'Product Recommendation',
                'query': 'Suggest products similar to wireless headphones',
                'customer_id': 'customer-67890', 
                'priority': 'medium'
            },
            {
                'name': 'Technical Documentation',
                'query': 'Explain how to integrate our API',
                'customer_id': 'developer-54321',
                'priority': 'low'
            }
        ]
        
        print("🎯 Testing production scenarios...")
        
        results = []
        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            
            try:
                result = manager.make_resilient_request(
                    query=scenario['query'],
                    customer_id=scenario['customer_id'],
                    priority=scenario['priority']
                )
                
                if result:
                    results.append(result)
                    print(f"   ✅ Success: {result['provider']} (${result['cost']:.6f})")
                else:
                    print("   ❌ Failed: No response")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Production metrics summary
        if results:
            total_cost = sum(r['cost'] for r in results)
            avg_cost = total_cost / len(results)
            
            print(f"\n📊 Production Metrics:")
            print(f"   • Successful requests: {len(results)}/{len(scenarios)}")
            print(f"   • Total cost: ${total_cost:.6f}")
            print(f"   • Average cost: ${avg_cost:.6f}")
            print(f"   • Success rate: {len(results)/len(scenarios)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Production deployment demo failed: {e}")
        return False

    return True


def demonstrate_monitoring_and_alerting():
    """Show production monitoring and alerting."""
    
    print("\n📊 Production Monitoring & Alerting")
    print("=" * 38)
    
    monitoring_components = [
        {
            'component': 'Request Monitoring',
            'metrics': ['Request rate', 'Success rate', 'Latency percentiles', 'Error rates'],
            'alerts': ['High error rate', 'Latency degradation', 'Provider failures']
        },
        {
            'component': 'Cost Monitoring',
            'metrics': ['Cost per request', 'Daily spend', 'Budget utilization'],
            'alerts': ['Budget exceeded', 'Cost anomaly', 'Unexpected spikes']
        },
        {
            'component': 'Performance Monitoring', 
            'metrics': ['Response time', 'Token throughput', 'Queue depth'],
            'alerts': ['Performance degradation', 'Queue overflow', 'Timeout increases']
        },
        {
            'component': 'Business Monitoring',
            'metrics': ['Customer satisfaction', 'Feature usage', 'Revenue impact'],
            'alerts': ['Customer complaints', 'Feature failures', 'Revenue loss']
        }
    ]
    
    for comp in monitoring_components:
        print(f"\n🔍 {comp['component']}:")
        print(f"   📈 Metrics: {', '.join(comp['metrics'])}")
        print(f"   🚨 Alerts: {', '.join(comp['alerts'])}")
    
    # Example monitoring configuration
    print(f"\n⚙️  Example Monitoring Stack:")
    print("""
    # OpenTelemetry export to multiple backends
    OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
    OTEL_EXPORTER_JAEGER_ENDPOINT=https://jaeger.company.com
    
    # Custom metrics export
    GENOPS_CUSTOM_METRICS=business_value,customer_satisfaction
    GENOPS_ALERT_WEBHOOKS=https://alerts.company.com/webhook
    
    # Cost monitoring
    GENOPS_BUDGET_DAILY=100.00
    GENOPS_BUDGET_MONTHLY=2500.00
    GENOPS_COST_ALERT_THRESHOLD=0.80
    """)

    return True


def demonstrate_scalability_patterns():
    """Show scalability patterns for high-volume deployments."""
    
    print("\n🚀 Scalability Patterns")
    print("=" * 23)
    
    scalability_patterns = [
        {
            'pattern': 'Horizontal Scaling',
            'description': 'Multiple adapter instances with load balancing',
            'use_case': 'High request volume (>1000 req/min)'
        },
        {
            'pattern': 'Connection Pooling',
            'description': 'Reuse HTTP connections across requests',
            'use_case': 'Reduce connection overhead'
        },
        {
            'pattern': 'Request Batching',
            'description': 'Batch multiple requests into single calls',
            'use_case': 'Improve throughput efficiency'
        },
        {
            'pattern': 'Async Processing',
            'description': 'Non-blocking request processing',
            'use_case': 'Handle concurrent requests'
        },
        {
            'pattern': 'Caching Layer',
            'description': 'Cache frequent queries and responses',
            'use_case': 'Reduce API calls and costs'
        },
        {
            'pattern': 'Circuit Breakers',
            'description': 'Fail fast when services are unavailable',
            'use_case': 'Maintain system stability'
        }
    ]
    
    for pattern in scalability_patterns:
        print(f"\n🔧 {pattern['pattern']}:")
        print(f"   📝 {pattern['description']}")
        print(f"   🎯 Use case: {pattern['use_case']}")
    
    # Example high-scale configuration
    print(f"\n⚡ High-Scale Configuration Example:")
    print("""
    # Multiple worker processes
    gunicorn app:app --workers 8 --worker-class gevent

    # Connection pooling
    adapter = instrument_helicone(
        max_connections=100,
        connection_pool_size=20,
        keep_alive_timeout=30
    )
    
    # Async processing
    import asyncio
    async def process_requests(requests):
        tasks = [adapter.chat_async(req) for req in requests]
        return await asyncio.gather(*tasks)
    """)

    return True


def main():
    """Main function to run production patterns demonstration."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    required_vars = ['HELICONE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Run demonstrations
    success = True
    success &= demonstrate_production_deployment()
    success &= demonstrate_monitoring_and_alerting()
    success &= demonstrate_scalability_patterns()
    
    if success:
        print("\n🎉 SUCCESS! Production patterns demonstration completed.")
        print("\n🏭 Production-Ready Features Demonstrated:")
        print("   • Comprehensive error handling and resilience")
        print("   • Production logging and monitoring")
        print("   • Multi-provider failover strategies") 
        print("   • Scalability patterns for high volume")
        print("   • Enterprise monitoring and alerting")
        
        print("\n🚀 Ready for Production:")
        print("   • Implement these patterns in your deployment")
        print("   • Set up monitoring and alerting")
        print("   • Configure scalability features")
        print("   • Test failover scenarios")
        
        print("\n📚 Additional Resources:")
        print("   • See docs/integrations/helicone.md for detailed configuration")
        print("   • Review monitoring setup in your observability platform")
        print("   • Consider enterprise features for large deployments")
    else:
        print("\n❌ Production patterns demo encountered issues.")
    
    return success


if __name__ == "__main__":
    """Entry point for the production patterns example."""
    success = main()
    
    if success:
        print("\n" + "🏭" * 20)
        print("Production-ready AI gateway: Enterprise-grade reliability!")
        print("🏭" * 20)
    
    sys.exit(0 if success else 1)