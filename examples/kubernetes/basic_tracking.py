#!/usr/bin/env python3
"""
✅ Basic Kubernetes Tracking Example

Demonstrates fundamental GenOps AI tracking in Kubernetes environments.
Shows how to add governance to existing AI applications with minimal code changes.

Usage:
    python basic_tracking.py
    python basic_tracking.py --team engineering --project demo-app
    python basic_tracking.py --customer-id "customer-123"
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

# Import GenOps Kubernetes provider
try:
    from genops.providers.kubernetes import KubernetesAdapter, validate_kubernetes_setup
    from genops.core.governance import create_governance_context
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False
    print("⚠️  GenOps not installed. Install with: pip install genops")

# Import OpenAI for demonstration (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


async def basic_tracking_example(
    team: Optional[str] = None,
    project: Optional[str] = None,
    customer_id: Optional[str] = None
):
    """
    Demonstrate basic tracking in Kubernetes environment.
    
    Shows how to add governance tracking to existing AI operations
    with minimal code changes.
    """
    
    print("🚢 Basic Kubernetes Tracking Example")
    print("=" * 60)
    
    if not GENOPS_AVAILABLE:
        print("❌ GenOps not available - install with: pip install genops")
        return False
    
    # 1. Validate Kubernetes environment
    print("\n1️⃣ Validating Kubernetes Environment...")
    validation = validate_kubernetes_setup(enable_resource_monitoring=True)
    
    if not validation.is_kubernetes_environment:
        print("⚠️  Not running in Kubernetes - governance will work but with limited context")
    else:
        print(f"✅ Running in Kubernetes namespace: {validation.namespace}")
        if validation.pod_name:
            print(f"   Pod: {validation.pod_name}")
        if validation.node_name:
            print(f"   Node: {validation.node_name}")
    
    # 2. Initialize Kubernetes adapter
    print("\n2️⃣ Initializing Kubernetes Adapter...")
    try:
        adapter = KubernetesAdapter()
        print(f"✅ Kubernetes adapter initialized: {adapter.get_framework_name()}")
        print(f"   Environment available: {adapter.is_available()}")
    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return False
    
    # 3. Create governance context with Kubernetes attributes
    print("\n3️⃣ Creating Governance Context...")
    governance_attrs = {
        "team": team or os.getenv("DEFAULT_TEAM", "unknown"),
        "project": project or os.getenv("PROJECT_NAME", "basic-tracking-demo"),
        "customer_id": customer_id or "demo-customer",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "feature": "basic-tracking-example"
    }
    
    # Get Kubernetes-specific attributes from adapter
    k8s_attrs = adapter.get_telemetry_attributes(**governance_attrs)
    print(f"✅ Governance context created with {len(k8s_attrs)} attributes")
    
    # Show key governance attributes
    print("\n📊 Governance Attributes:")
    key_attrs = [
        "team", "project", "customer_id", "environment",
        "k8s.namespace.name", "k8s.pod.name", "k8s.node.name"
    ]
    for attr in key_attrs:
        value = k8s_attrs.get(attr, "Not available")
        print(f"   {attr}: {value}")
    
    # 4. Demonstrate tracked AI operation
    print("\n4️⃣ Running Tracked AI Operation...")
    
    # Use Kubernetes adapter context manager for automatic governance
    with adapter.create_governance_context(**governance_attrs) as governance_context:
        print(f"✅ Governance context active: {governance_context.context_id}")
        
        # Simulate AI operation (replace with actual AI calls)
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                # Example: OpenAI request with automatic Kubernetes governance
                print("   Making OpenAI request with Kubernetes governance...")
                
                client = openai.AsyncOpenAI()
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Hello from Kubernetes with GenOps governance!"}
                    ],
                    max_tokens=50
                )
                
                print(f"   ✅ OpenAI response: {response.choices[0].message.content[:100]}...")
                
                # Cost information is automatically tracked via governance context
                cost_info = governance_context.get_cost_summary()
                if cost_info:
                    print(f"   💰 Estimated cost: ${cost_info.get('total_cost', 0):.4f}")
                
            except Exception as e:
                print(f"   ⚠️ OpenAI request failed: {e}")
                print("   (This is expected if OPENAI_API_KEY is not set)")
        else:
            # Simulate operation without external API
            print("   Simulating AI operation (no external APIs configured)...")
            await asyncio.sleep(0.5)  # Simulate operation time
            
            # Manually add cost tracking for demonstration
            governance_context.add_cost_data(
                provider="simulated",
                model="demo-model",
                cost=0.0023,
                tokens_in=15,
                tokens_out=50,
                operation="chat_completion"
            )
            
            print("   ✅ Simulated operation completed")
            print("   💰 Simulated cost: $0.0023")
        
        # Show final governance summary
        print("\n📋 Operation Summary:")
        print(f"   Context ID: {governance_context.context_id}")
        print(f"   Duration: {governance_context.get_duration():.3f}s")
        
        telemetry = governance_context.get_telemetry_data()
        print(f"   Telemetry attributes: {len(telemetry)} captured")
        
        # Show resource usage if available
        if validation.has_resource_monitoring:
            resource_usage = governance_context.get_resource_usage()
            if resource_usage:
                print(f"   CPU usage: {resource_usage.get('cpu_usage_millicores', 'N/A')}m")
                print(f"   Memory usage: {resource_usage.get('memory_usage_bytes', 'N/A')} bytes")
    
    # 5. Show telemetry export
    print("\n5️⃣ Telemetry Export...")
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otel_endpoint:
        print(f"✅ Telemetry exported to: {otel_endpoint}")
        print("   Check your observability platform for governance metrics:")
        print("   - genops.kubernetes.cost_total")
        print("   - genops.kubernetes.request_duration")
        print("   - genops.kubernetes.resource_usage")
    else:
        print("⚠️  OTEL_EXPORTER_OTLP_ENDPOINT not set")
        print("   Telemetry captured but not exported to external systems")
    
    print("\n🎉 Basic tracking example completed!")
    print("\nKey Benefits Demonstrated:")
    print("✅ Automatic Kubernetes context detection and attribution")
    print("✅ Minimal code changes to existing AI applications") 
    print("✅ Real-time cost and performance tracking")
    print("✅ Governance attributes propagated to telemetry")
    print("✅ Resource usage monitoring (when available)")
    
    return True


def demonstrate_tracking_patterns():
    """Show different tracking patterns available."""
    
    print("\n🔍 AVAILABLE TRACKING PATTERNS")
    print("=" * 60)
    
    if not GENOPS_AVAILABLE:
        print("❌ GenOps not available for demonstration")
        return
    
    print("1️⃣ Context Manager Pattern (Recommended):")
    print("""
    from genops.providers.kubernetes import KubernetesAdapter
    
    adapter = KubernetesAdapter()
    with adapter.create_governance_context(team="engineering") as ctx:
        # Your AI operations here
        result = ai_operation()
        # Cost and performance automatically tracked
    """)
    
    print("\n2️⃣ Manual Tracking Pattern:")
    print("""
    adapter = KubernetesAdapter()
    telemetry = adapter.get_telemetry_attributes(
        team="engineering",
        project="my-app",
        customer_id="customer-123"
    )
    
    # Use telemetry attributes in your AI calls
    result = ai_operation_with_attributes(telemetry)
    """)
    
    print("\n3️⃣ Auto-Instrumentation Pattern:")
    print("""
    from genops import auto_instrument
    auto_instrument()  # Automatic governance for supported frameworks
    
    # Existing code works unchanged
    result = openai.ChatCompletion.create(...)
    """)


def show_kubernetes_specific_features():
    """Demonstrate Kubernetes-specific governance features."""
    
    print("\n⚙️ KUBERNETES-SPECIFIC FEATURES")
    print("=" * 60)
    
    if not GENOPS_AVAILABLE:
        print("❌ GenOps not available")
        return
    
    try:
        from genops.providers.kubernetes import KubernetesDetector, KubernetesResourceMonitor
        
        # Show detection capabilities
        detector = KubernetesDetector()
        print("🔍 Environment Detection:")
        print(f"   Running in Kubernetes: {detector.is_kubernetes()}")
        print(f"   Namespace: {detector.get_namespace() or 'Unknown'}")
        print(f"   Pod Name: {detector.get_pod_name() or 'Unknown'}")
        print(f"   Node Name: {detector.get_node_name() or 'Unknown'}")
        
        # Show governance attributes
        print("\n📊 Kubernetes Governance Attributes:")
        attrs = detector.get_governance_attributes()
        for key, value in sorted(attrs.items()):
            if key.startswith('k8s.'):
                print(f"   {key}: {value}")
        
        # Show resource monitoring
        print("\n💾 Resource Monitoring:")
        try:
            monitor = KubernetesResourceMonitor()
            usage = monitor.get_current_usage()
            if usage.cpu_usage_millicores is not None:
                print(f"   Current CPU: {usage.cpu_usage_millicores}m")
            if usage.memory_usage_bytes is not None:
                print(f"   Current Memory: {usage.memory_usage_bytes / 1024 / 1024:.1f} MB")
            
            resources = monitor.get_current_resources()
            if resources.get('cpu_limit'):
                print(f"   CPU Limit: {resources['cpu_limit']}")
            if resources.get('memory_limit'):
                print(f"   Memory Limit: {resources['memory_limit']}")
                
        except Exception as e:
            print(f"   Resource monitoring unavailable: {e}")
        
    except ImportError:
        print("❌ Kubernetes provider not available")


async def main():
    """Main example runner."""
    
    parser = argparse.ArgumentParser(
        description="Basic Kubernetes tracking example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python basic_tracking.py                                    # Basic demo
    python basic_tracking.py --team engineering                # With team attribution
    python basic_tracking.py --team engineering --project app  # With project attribution
    python basic_tracking.py --customer-id "customer-123"     # With customer attribution
    python basic_tracking.py --show-patterns                  # Show tracking patterns
    python basic_tracking.py --show-k8s-features             # Show K8s-specific features
        """
    )
    
    parser.add_argument(
        "--team",
        type=str,
        help="Team name for cost attribution"
    )
    
    parser.add_argument(
        "--project", 
        type=str,
        help="Project name for tracking"
    )
    
    parser.add_argument(
        "--customer-id",
        type=str,
        help="Customer ID for billing attribution"
    )
    
    parser.add_argument(
        "--show-patterns",
        action="store_true",
        help="Show available tracking patterns"
    )
    
    parser.add_argument(
        "--show-k8s-features",
        action="store_true", 
        help="Show Kubernetes-specific features"
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Run basic example by default
    if not args.show_patterns and not args.show_k8s_features:
        success = await basic_tracking_example(
            team=args.team,
            project=args.project, 
            customer_id=args.customer_id
        )
    
    # Show patterns if requested
    if args.show_patterns:
        demonstrate_tracking_patterns()
    
    # Show Kubernetes features if requested
    if args.show_k8s_features:
        show_kubernetes_specific_features()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())