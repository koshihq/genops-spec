#!/usr/bin/env python3
"""
Helicone Advanced Features Example

This example demonstrates advanced GenOps + Helicone features including
streaming responses, custom routing logic, performance optimization,
and enterprise-grade functionality.

Usage:
    python advanced_features.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    export OPENAI_API_KEY="your_openai_api_key"
    export ANTHROPIC_API_KEY="your_anthropic_api_key"
"""

import asyncio
import os
import sys
from datetime import datetime


def demonstrate_custom_routing():
    """Show custom routing logic implementation."""

    print("🎯 Custom Intelligent Routing Logic")
    print("=" * 38)

    try:
        from genops.providers.helicone import instrument_helicone

        adapter = instrument_helicone(
            team="advanced-features-team", project="custom-routing-demo"
        )

        # Custom routing function
        def smart_routing_strategy(query, providers, context):
            """Custom routing based on query characteristics."""
            import re

            # Simple query detection
            simple_patterns = [
                r"\b\d+\s*[+\-*/]\s*\d+\b",
                r"^(what|who|when|where) is",
                r"^hello\b",
            ]
            if any(re.search(pattern, query.lower()) for pattern in simple_patterns):
                return "groq"  # Fast and cheap for simple queries

            # Code-related queries
            code_patterns = [
                r"\bcode\b",
                r"\bfunction\b",
                r"\bpython\b",
                r"\bjavascript\b",
            ]
            if any(re.search(pattern, query.lower()) for pattern in code_patterns):
                return "openai"  # Good for code generation

            # Complex reasoning
            complex_patterns = [r"\banalyz", r"\bcompare\b", r"\bexplain.*why\b"]
            if any(re.search(pattern, query.lower()) for pattern in complex_patterns):
                return "anthropic"  # Best for reasoning

            # Default fallback
            return "openai"

        # Register custom strategy
        adapter.register_routing_strategy("smart_custom", smart_routing_strategy)

        # Test custom routing
        test_queries = [
            "What is 15 * 23?",
            "Write a Python function to sort a list",
            "Analyze the pros and cons of renewable energy",
            "Hello, how are you today?",
        ]

        print("🧪 Testing custom routing logic...")

        for query in test_queries:
            try:
                response = adapter.multi_provider_chat(
                    message=query,
                    providers=["openai", "anthropic", "groq"],
                    routing_strategy="smart_custom",
                )

                provider_used = getattr(response, "provider_used", "unknown")
                cost = (
                    getattr(response.usage, "total_cost", 0.0)
                    if hasattr(response, "usage")
                    else 0.0
                )

                print(f"   Query: {query[:50]}...")
                print(f"   Routed to: {provider_used} (${cost:.6f})")

            except Exception as e:
                print(f"   ❌ Failed: {query[:30]}... - {e}")

    except Exception as e:
        print(f"❌ Custom routing demo failed: {e}")
        return False

    return True


async def demonstrate_streaming_responses():
    """Show streaming response handling with telemetry."""

    print("\n🌊 Streaming Responses with Real-time Telemetry")
    print("=" * 48)

    try:
        from genops.providers.helicone import instrument_helicone

        adapter = instrument_helicone(
            team="streaming-demo-team", project="streaming-responses"
        )

        print("🚀 Starting streaming demonstration...")

        # Simulate streaming (actual implementation would use real streaming)
        query = (
            "Explain the benefits of streaming AI responses in production applications."
        )

        print(f"📝 Query: {query}")
        print("🌊 Streaming response:")

        try:
            # In a real implementation, this would be actual streaming
            response = adapter.chat(
                message=query,
                provider="openai",
                model="gpt-3.5-turbo",
                stream=True,  # This would enable streaming
                customer_id="streaming-demo",
            )

            # Simulate streaming chunks
            content = (
                response.content
                if hasattr(response, "content")
                else "Streaming response content..."
            )
            words = content.split()

            print("   ", end="")
            for _i, word in enumerate(words[:20]):  # Show first 20 words
                print(word, end=" ", flush=True)
                await asyncio.sleep(0.1)  # Simulate streaming delay
            print("...")

            # Final telemetry
            if hasattr(response, "usage"):
                cost = getattr(response.usage, "total_cost", 0.0)
                tokens = getattr(response.usage, "output_tokens", 0)
                print(f"✅ Streaming complete: ${cost:.6f}, {tokens} tokens")

        except Exception as e:
            print(f"❌ Streaming failed: {e}")

    except Exception as e:
        print(f"❌ Streaming demo setup failed: {e}")
        return False

    return True


def demonstrate_performance_optimization():
    """Show performance optimization techniques."""

    print("\n⚡ Performance Optimization Techniques")
    print("=" * 39)

    optimization_techniques = [
        {
            "name": "Request Batching",
            "description": "Batch multiple requests for efficiency",
            "benefit": "Reduced latency overhead",
        },
        {
            "name": "Connection Pooling",
            "description": "Reuse HTTP connections across requests",
            "benefit": "Lower connection establishment cost",
        },
        {
            "name": "Response Caching",
            "description": "Cache frequent queries to avoid API calls",
            "benefit": "Dramatic cost and latency reduction",
        },
        {
            "name": "Provider Load Balancing",
            "description": "Distribute load across providers",
            "benefit": "Better throughput and reliability",
        },
        {
            "name": "Circuit Breakers",
            "description": "Fail fast when providers are down",
            "benefit": "Improved reliability and user experience",
        },
    ]

    print("🔧 Available Optimization Techniques:")
    for tech in optimization_techniques:
        print(f"   • {tech['name']:>20}: {tech['description']}")
        print(f"     {'':>20}  Benefit: {tech['benefit']}")

    # Example configuration
    print("\n⚙️  Example Performance Configuration:")
    print("""
    adapter = instrument_helicone(
        # Connection optimization
        max_connections=50,
        connection_timeout=10,

        # Caching configuration
        enable_caching=True,
        cache_ttl=3600,  # 1 hour

        # Load balancing
        load_balance_strategy='round_robin',
        health_check_interval=30,

        # Circuit breaker
        failure_threshold=5,
        recovery_timeout=60
    )
    """)

    return True


def demonstrate_enterprise_features():
    """Show enterprise-grade features."""

    print("\n🏢 Enterprise-Grade Features")
    print("=" * 31)

    enterprise_features = [
        "🔐 Advanced Authentication (OAuth2, SAML, Custom)",
        "🛡️  Role-based Access Control (RBAC)",
        "📊 Advanced Analytics and Reporting",
        "🔍 Audit Logging and Compliance",
        "🌐 Self-hosted Gateway Deployment",
        "💾 Data Residency and Privacy Controls",
        "📈 Custom Metrics and Dashboards",
        "🚨 Advanced Alerting and Monitoring",
        "⚖️  SLA Management and Guarantees",
        "🔄 Disaster Recovery and Backup",
        "🎛️  Fine-grained Policy Controls",
        "📡 Custom Telemetry Export",
    ]

    for feature in enterprise_features:
        print(f"   {feature}")

    # Example enterprise configuration
    print("\n🏭 Example Enterprise Configuration:")
    print("""
    adapter = instrument_helicone(
        # Authentication
        auth_mode='oauth2',
        oauth_config={
            'provider': 'okta',
            'client_id': 'your_client_id'
        },

        # Compliance
        audit_logging=True,
        data_residency='us-east-1',
        compliance_mode='soc2',

        # Self-hosted gateway
        gateway_url='https://gateway.yourcompany.com',

        # Advanced monitoring
        custom_metrics=['business_value', 'quality_score'],
        alert_webhooks=['https://alerts.yourcompany.com']
    )
    """)

    return True


def main():
    """Main function to run advanced features demonstration."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv("HELICONE_API_KEY"):
        print("❌ Missing HELICONE_API_KEY")
        return False

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY (required for advanced features)")
        return False

    # Run demonstrations
    success = True
    success &= demonstrate_custom_routing()

    # Run async streaming demo
    try:
        loop = asyncio.get_event_loop()
        success &= loop.run_until_complete(demonstrate_streaming_responses())
    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")
        success = False

    success &= demonstrate_performance_optimization()
    success &= demonstrate_enterprise_features()

    if success:
        print("\n🎉 SUCCESS! Advanced features demonstration completed.")
        print("\n🚀 Advanced Capabilities Demonstrated:")
        print("   • Custom intelligent routing logic")
        print("   • Streaming responses with real-time telemetry")
        print("   • Performance optimization techniques")
        print("   • Enterprise-grade features and configurations")

        print("\n🎯 Production Implementation:")
        print("   • Implement custom routing for your use cases")
        print("   • Enable streaming for better user experience")
        print("   • Configure performance optimizations")
        print("   • Consider enterprise features for production deployments")

        print("\n📚 Next Steps:")
        print("   • Try 'python production_patterns.py' for deployment patterns")
        print("   • Implement these patterns in your applications")
        print("   • Monitor performance and optimize further")
    else:
        print("\n❌ Advanced features demo encountered issues.")

    return success


if __name__ == "__main__":
    """Entry point for the advanced features example."""
    success = main()

    if success:
        print("\n" + "🚀" * 20)
        print("Advanced AI gateway features: Production-ready intelligence!")
        print("🚀" * 20)

    sys.exit(0 if success else 1)
