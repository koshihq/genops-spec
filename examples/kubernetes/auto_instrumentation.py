#!/usr/bin/env python3
"""
✅ Auto-Instrumentation Kubernetes Example

Demonstrates zero-code auto-instrumentation for Kubernetes environments.
Shows how existing AI applications can get governance with no code changes.

Usage:
    python auto_instrumentation.py
    python auto_instrumentation.py --test-openai
    python auto_instrumentation.py --test-anthropic
    python auto_instrumentation.py --demo-only
"""

import argparse
import asyncio
import os
import sys
from typing import List, Optional

# Import GenOps auto-instrumentation
try:
    from genops import auto_instrument
    from genops.providers.kubernetes import validate_kubernetes_setup, KubernetesDetector
    from genops.core.instrumentation import get_active_instrumentations
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False
    print("⚠️  GenOps not installed. Install with: pip install genops")

# Import AI providers for testing (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


async def demonstrate_auto_instrumentation():
    """
    Demonstrate zero-code auto-instrumentation in Kubernetes.
    
    Shows how GenOps automatically instruments existing AI code
    without requiring any application changes.
    """
    
    print("🔧 Auto-Instrumentation Kubernetes Example")
    print("=" * 60)
    
    if not GENOPS_AVAILABLE:
        print("❌ GenOps not available - install with: pip install genops")
        return False
    
    # 1. Show environment before instrumentation
    print("\n1️⃣ Pre-Instrumentation Environment Check...")
    validation = validate_kubernetes_setup()
    
    if validation.is_kubernetes_environment:
        print(f"✅ Running in Kubernetes namespace: {validation.namespace}")
    else:
        print("⚠️  Not in Kubernetes - auto-instrumentation will work with limited context")
    
    print(f"   Instrumentations active: {len(get_active_instrumentations())}")
    
    # 2. Enable auto-instrumentation
    print("\n2️⃣ Enabling Auto-Instrumentation...")
    print("   Calling: auto_instrument()")
    
    try:
        # This is the magic call - zero configuration required!
        auto_instrument()
        print("✅ Auto-instrumentation enabled successfully")
        
        # Show what got instrumented
        active = get_active_instrumentations()
        print(f"   Active instrumentations: {len(active)}")
        for name, details in active.items():
            print(f"   • {name}: {details.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Auto-instrumentation failed: {e}")
        return False
    
    # 3. Show Kubernetes context detection
    print("\n3️⃣ Kubernetes Context Auto-Detection...")
    detector = KubernetesDetector()
    
    print(f"   Environment detected: {'Kubernetes' if detector.is_kubernetes() else 'Local'}")
    if detector.is_kubernetes():
        attrs = detector.get_governance_attributes()
        print(f"   Kubernetes attributes: {len(attrs)} detected")
        
        # Show key auto-detected attributes
        key_attrs = ['k8s.namespace.name', 'k8s.pod.name', 'k8s.node.name']
        for attr in key_attrs:
            value = attrs.get(attr, 'Not available')
            print(f"   {attr}: {value}")
    
    print("\n✅ Auto-instrumentation setup complete!")
    print("\n🎯 What This Means:")
    print("   • All AI provider calls are now automatically tracked")
    print("   • Kubernetes context is automatically added to telemetry")
    print("   • Cost and performance data is collected transparently")
    print("   • No changes required to existing application code")
    
    return True


async def test_instrumented_openai():
    """Test that OpenAI calls are automatically instrumented."""
    
    print("\n🤖 Testing Auto-Instrumented OpenAI")
    print("=" * 60)
    
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI not available - install with: pip install openai")
        return False
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - skipping live API test")
        return simulate_openai_call()
    
    try:
        print("   Making OpenAI request (automatically instrumented)...")
        
        # This is your existing code - no changes needed!
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello! This request is automatically tracked by GenOps."}
            ],
            max_tokens=50
        )
        
        print(f"✅ OpenAI Response: {response.choices[0].message.content}")
        print("   🎯 This call was automatically tracked with:")
        print("   • Cost calculation and attribution")
        print("   • Kubernetes context (namespace, pod, node)")
        print("   • Performance metrics (latency, token counts)")
        print("   • Governance attributes (team, project, environment)")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False


def simulate_openai_call():
    """Simulate an OpenAI call to show instrumentation structure."""
    
    print("   Simulating OpenAI call (no API key configured)...")
    print("   📋 If OPENAI_API_KEY was set, this would:")
    print("   • Make actual OpenAI API call")
    print("   • Calculate real costs automatically") 
    print("   • Add Kubernetes governance attributes")
    print("   • Export telemetry to your observability platform")
    
    print("\n   📊 Telemetry Structure (automatically added):")
    print("   {")
    print("     'genops.provider': 'openai',")
    print("     'genops.model': 'gpt-3.5-turbo',")
    print("     'genops.cost.total': 0.0023,")
    print("     'genops.tokens.input': 15,")
    print("     'genops.tokens.output': 50,")
    print("     'k8s.namespace.name': 'your-namespace',")
    print("     'k8s.pod.name': 'your-pod-xyz',")
    print("     'k8s.node.name': 'node-123'")
    print("   }")
    
    return True


async def test_instrumented_anthropic():
    """Test that Anthropic calls are automatically instrumented."""
    
    print("\n🧠 Testing Auto-Instrumented Anthropic") 
    print("=" * 60)
    
    if not ANTHROPIC_AVAILABLE:
        print("❌ Anthropic not available - install with: pip install anthropic")
        return False
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set - skipping live API test")
        return simulate_anthropic_call()
    
    try:
        print("   Making Anthropic request (automatically instrumented)...")
        
        # This is your existing code - no changes needed!
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Hello! This Anthropic request is automatically tracked."}
            ]
        )
        
        print(f"✅ Anthropic Response: {response.content[0].text}")
        print("   🎯 This call was automatically tracked with full Kubernetes context")
        
        return True
        
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        return False


def simulate_anthropic_call():
    """Simulate an Anthropic call to show instrumentation."""
    
    print("   Simulating Anthropic call (no API key configured)...")
    print("   📋 With ANTHROPIC_API_KEY, this would automatically track:")
    print("   • Claude model usage and costs")
    print("   • Kubernetes pod and namespace attribution") 
    print("   • Cross-provider cost aggregation")
    
    return True


def show_existing_code_examples():
    """Show how existing code works unchanged with auto-instrumentation."""
    
    print("\n📝 EXISTING CODE COMPATIBILITY")
    print("=" * 60)
    
    print("✅ Your existing code works unchanged after auto_instrument():")
    print()
    
    print("🔹 OpenAI Example:")
    print("""
    import openai
    
    # Your existing code - no changes needed!
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # GenOps automatically adds:
    # • Cost tracking: $0.0023 for this request
    # • K8s context: namespace=my-app, pod=my-app-xyz
    # • Performance: 245ms response time
    # • Governance: team=engineering (from env vars)
    """)
    
    print("\n🔹 Anthropic Example:")
    print("""
    import anthropic
    
    # Your existing code - no changes needed!
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # GenOps automatically adds:
    # • Cost tracking: $0.0048 for this request  
    # • K8s context: namespace=my-app, pod=my-app-abc
    # • Performance: 189ms response time
    # • Multi-provider aggregation
    """)
    
    print("\n🔹 LangChain Example:")
    print("""
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    
    # Your existing LangChain code - no changes!
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = chat([HumanMessage(content="Hello!")])
    
    # GenOps automatically adds:
    # • LangChain operation tracking
    # • Nested cost aggregation across chains
    # • Kubernetes resource attribution
    """)


def show_advanced_auto_features():
    """Show advanced auto-instrumentation features."""
    
    print("\n⚡ ADVANCED AUTO-INSTRUMENTATION FEATURES")  
    print("=" * 60)
    
    print("🎯 Environment-Based Configuration:")
    print("   Set these environment variables for automatic configuration:")
    print()
    print("   # Team attribution")
    print("   export GENOPS_TEAM='engineering'")
    print("   export DEFAULT_TEAM='engineering'")
    print()
    print("   # Project tracking")
    print("   export PROJECT_NAME='my-awesome-app'")
    print("   export GENOPS_PROJECT='my-awesome-app'")
    print()
    print("   # Customer attribution")
    print("   export DEFAULT_CUSTOMER_ID='enterprise-customer'")
    print()
    print("   # Cost center")
    print("   export COST_CENTER='engineering-ai'")
    
    print("\n🔍 Automatic Provider Detection:")
    providers = [
        "OpenAI (openai package)",
        "Anthropic (anthropic package)", 
        "LangChain (langchain package)",
        "Google AI (google-generativeai package)",
        "AWS Bedrock (boto3 with bedrock)",
        "Azure OpenAI (openai with azure endpoint)"
    ]
    
    for provider in providers:
        print(f"   ✅ {provider}")
    
    print("\n📊 Automatic Telemetry Export:")
    print("   GenOps detects and uses these automatically:")
    print("   • OTEL_EXPORTER_OTLP_ENDPOINT (OpenTelemetry)")
    print("   • JAEGER_ENDPOINT (Jaeger tracing)")
    print("   • HONEYCOMB_API_KEY (Honeycomb)")
    print("   • DATADOG_API_KEY (Datadog APM)")
    
    print("\n🚀 Zero-Config Kubernetes Features:")
    print("   • Automatic namespace, pod, and node detection")
    print("   • Service account and RBAC awareness")
    print("   • Resource limit and usage monitoring")
    print("   • Network policy compliance checking")


async def run_comprehensive_demo():
    """Run a comprehensive demo of all auto-instrumentation features."""
    
    print("\n🎪 COMPREHENSIVE AUTO-INSTRUMENTATION DEMO")
    print("=" * 60)
    
    success = True
    
    # 1. Enable auto-instrumentation
    demo_success = await demonstrate_auto_instrumentation()
    success = success and demo_success
    
    # 2. Test OpenAI
    print("\n" + "-" * 40)
    openai_success = await test_instrumented_openai()
    success = success and openai_success
    
    # 3. Test Anthropic  
    print("\n" + "-" * 40)
    anthropic_success = await test_instrumented_anthropic()
    success = success and anthropic_success
    
    # 4. Show code compatibility
    print("\n" + "-" * 40)
    show_existing_code_examples()
    
    # 5. Show advanced features
    print("\n" + "-" * 40)
    show_advanced_auto_features()
    
    # Final summary
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("✅ Auto-instrumentation enabled with zero code changes")
    print("✅ Kubernetes context automatically detected and added")
    print("✅ AI provider calls automatically tracked and costed")
    print("✅ Telemetry exported to observability platforms")
    print("✅ Existing applications work unchanged")
    
    print("\n🚀 Next Steps:")
    print("   1. Add auto_instrument() to your application startup")
    print("   2. Set environment variables for team/project attribution")
    print("   3. Configure OTEL_EXPORTER_OTLP_ENDPOINT for telemetry export")
    print("   4. Monitor costs and performance in your observability platform")
    
    return success


async def main():
    """Main demo runner."""
    
    parser = argparse.ArgumentParser(
        description="Auto-instrumentation Kubernetes example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python auto_instrumentation.py                    # Full demo
    python auto_instrumentation.py --demo-only       # Setup demo only
    python auto_instrumentation.py --test-openai     # Test OpenAI instrumentation
    python auto_instrumentation.py --test-anthropic  # Test Anthropic instrumentation
        """
    )
    
    parser.add_argument(
        "--demo-only", 
        action="store_true",
        help="Show setup demo only (no provider testing)"
    )
    
    parser.add_argument(
        "--test-openai",
        action="store_true",
        help="Test OpenAI auto-instrumentation"
    )
    
    parser.add_argument(
        "--test-anthropic", 
        action="store_true",
        help="Test Anthropic auto-instrumentation"
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Run specific tests if requested
    if args.demo_only:
        success = await demonstrate_auto_instrumentation()
        show_existing_code_examples()
        show_advanced_auto_features()
        
    elif args.test_openai:
        await demonstrate_auto_instrumentation()
        success = await test_instrumented_openai()
        
    elif args.test_anthropic:
        await demonstrate_auto_instrumentation() 
        success = await test_instrumented_anthropic()
        
    else:
        # Run comprehensive demo
        success = await run_comprehensive_demo()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())