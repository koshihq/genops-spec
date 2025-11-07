#!/usr/bin/env python3
"""
PromptLayer Auto-Instrumentation with GenOps

This example demonstrates zero-code auto-instrumentation for PromptLayer operations,
automatically adding GenOps governance, cost attribution, and policy enforcement
to existing PromptLayer code without any code changes required.

This is the Level 1 (5-minute) example - immediately usable auto-instrumentation.

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    
    # Optional but recommended for full governance
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any

def demonstrate_auto_instrumentation():
    """
    Demonstrates auto-instrumentation that requires ZERO code changes.
    
    This example shows how GenOps can be enabled for existing PromptLayer code
    with a single auto_instrument() call, automatically adding governance
    intelligence to all PromptLayer operations.
    """
    print("🎯 PromptLayer Auto-Instrumentation Demo")
    print("=" * 45)
    print("📌 Zero-code governance for existing PromptLayer applications")
    print()
    
    try:
        # Import and enable GenOps auto-instrumentation
        from genops.providers.promptlayer import auto_instrument
        print("✅ GenOps PromptLayer auto-instrumentation available")
        
        # Enable auto-instrumentation with a single call
        # This automatically adds governance to ALL PromptLayer operations
        auto_instrument(
            promptlayer_api_key=os.getenv('PROMPTLAYER_API_KEY'),
            team=os.getenv('GENOPS_TEAM', 'auto-instrumentation-demo'),
            project=os.getenv('GENOPS_PROJECT', 'zero-code-governance'),
            environment="development",
            customer_id="demo-customer",
            cost_center="rd-department",
            enable_cost_alerts=True,
            daily_budget_limit=5.0  # $5 daily budget for demo
        )
        print("🚀 Auto-instrumentation enabled! All PromptLayer operations now include:")
        print("   • Automatic cost tracking and attribution")
        print("   • Team and project governance")
        print("   • Budget enforcement and alerts")
        print("   • Policy compliance monitoring")
        print()
        
    except ImportError as e:
        print(f"❌ Failed to import GenOps auto-instrumentation: {e}")
        print("💡 Fix: Run 'pip install genops[promptlayer]'")
        return False
    
    # Simulate existing PromptLayer code (unchanged!)
    print("📝 Running your EXISTING PromptLayer code (zero changes required):")
    print("-" * 50)
    
    try:
        # Import PromptLayer as you normally would
        import promptlayer
        
        # Your existing PromptLayer code works exactly the same!
        # Auto-instrumentation automatically adds governance behind the scenes
        
        # Example 1: Standard PromptLayer prompt execution
        print("\n1️⃣ Standard PromptLayer Operations (your existing code)")
        
        # Mock PromptLayer operations since auto-instrumentation enhances them
        demo_operations = [
            {
                "name": "customer_support_v1",
                "input": {"customer_query": "How do I reset my password?"},
                "description": "Customer support prompt execution"
            },
            {
                "name": "product_description_v2",
                "input": {"product": "Smart Home Assistant", "features": ["Voice control", "Smart scheduling"]},
                "description": "Product description generation"
            },
            {
                "name": "email_writer_v3",
                "input": {"recipient": "valued customer", "subject": "Product update"},
                "description": "Email writing assistant"
            }
        ]
        
        for i, operation in enumerate(demo_operations):
            print(f"   Running: {operation['description']}")
            
            # In a real scenario, this would be your actual PromptLayer call:
            # response = promptlayer_client.run(
            #     prompt_name=operation['name'], 
            #     input_variables=operation['input']
            # )
            
            # For demo, we'll simulate the enhanced operation
            print(f"   ✅ {operation['name']} executed with automatic governance")
            print(f"      💰 Auto-tracked cost: ${0.002 + (i * 0.001):.6f}")
            print(f"      🏷️ Team attribution: auto-instrumentation-demo")
            print(f"      📊 Customer attribution: demo-customer")
            print()
        
        print("🌟 Key Benefits of Auto-Instrumentation:")
        print("   • ZERO code changes to your existing PromptLayer code")
        print("   • Automatic cost tracking for all prompt executions")
        print("   • Team and project attribution without manual tagging")
        print("   • Built-in budget enforcement and cost alerts")
        print("   • Policy compliance monitoring out of the box")
        print("   • Works with all PromptLayer features (versioning, A/B testing, etc.)")
        print()
        
    except ImportError:
        print("⚠️ PromptLayer SDK not found - simulating operations")
        print("💡 Install with: pip install promptlayer")
        print()
        
        print("✅ Auto-instrumentation simulation complete")
        print("   Your existing PromptLayer code would work exactly as shown")
        print("   with automatic governance intelligence added")
    
    return True

def show_before_after_comparison():
    """Show the before/after comparison of code with auto-instrumentation."""
    print("\n🔄 Before vs After Auto-Instrumentation")
    print("-" * 40)
    
    print("📝 BEFORE (Your existing code):")
    print("""
    import promptlayer
    
    client = promptlayer.PromptLayer(api_key="pl-your-key")
    response = client.run(
        prompt_name="customer_support",
        input_variables={"query": "Help request"}
    )
    # No governance, cost tracking, or attribution
    """)
    
    print("📝 AFTER (With GenOps auto-instrumentation):")
    print("""
    # Add just ONE line at the top of your application:
    from genops.providers.promptlayer import auto_instrument
    auto_instrument(team="support-team", project="customer-service")
    
    # Your existing code works exactly the same:
    import promptlayer
    
    client = promptlayer.PromptLayer(api_key="pl-your-key")
    response = client.run(
        prompt_name="customer_support",
        input_variables={"query": "Help request"}
    )
    # NOW automatically includes:
    # ✅ Cost tracking and attribution
    # ✅ Team and project governance
    # ✅ Budget enforcement
    # ✅ Policy compliance
    # ✅ OpenTelemetry export to your observability stack
    """)
    
    print("💡 Migration Strategy:")
    print("   1. Add auto_instrument() to your application startup")
    print("   2. Set team/project environment variables")
    print("   3. Your existing PromptLayer code gains governance automatically")
    print("   4. No changes to business logic or prompt execution")

def demonstrate_configuration_options():
    """Show various configuration options for auto-instrumentation."""
    print("\n⚙️ Auto-Instrumentation Configuration Options")
    print("-" * 40)
    
    try:
        from genops.providers.promptlayer import auto_instrument
        
        print("🎛️ Basic Configuration (minimal setup):")
        print("""
        auto_instrument(
            team="engineering",
            project="ai-features"
        )
        """)
        
        print("🎛️ Advanced Configuration (full governance):")
        print("""
        auto_instrument(
            team="engineering",
            project="ai-features",
            environment="production",
            customer_id="enterprise-123",
            cost_center="rd-department",
            daily_budget_limit=50.0,      # $50/day budget
            max_operation_cost=2.0,       # $2 max per operation
            enable_cost_alerts=True,
            governance_policy="advisory"   # or "enforced"
        )
        """)
        
        print("🌍 Environment Variable Configuration:")
        print("   # Set once, works everywhere")
        print("   export PROMPTLAYER_API_KEY='pl-your-key'")
        print("   export GENOPS_TEAM='engineering'")
        print("   export GENOPS_PROJECT='ai-features'")
        print("   export GENOPS_ENVIRONMENT='production'")
        print()
        print("   # Then just call:")
        print("   auto_instrument()  # Uses environment variables")
        
        print("\n✅ Auto-instrumentation adapts to your workflow:")
        print("   • Development: Minimal setup for quick testing")
        print("   • Staging: Full governance with warnings")
        print("   • Production: Strict policies with enforcement")
        
    except ImportError:
        print("❌ Auto-instrumentation not available")
        print("💡 Fix: pip install genops[promptlayer]")

def show_enterprise_patterns():
    """Demonstrate enterprise-ready auto-instrumentation patterns."""
    print("\n🏢 Enterprise Auto-Instrumentation Patterns")
    print("-" * 40)
    
    print("🎯 Multi-Team Application Pattern:")
    print("""
    # Different teams can have different governance policies
    if team == "customer-support":
        auto_instrument(
            team=team,
            project="support-automation",
            daily_budget_limit=10.0,
            governance_policy="advisory"
        )
    elif team == "sales":
        auto_instrument(
            team=team, 
            project="sales-enablement",
            daily_budget_limit=25.0,
            governance_policy="enforced"
        )
    """)
    
    print("🎯 Multi-Environment Pattern:")
    print("""
    import os
    
    environment = os.getenv('ENVIRONMENT', 'development')
    
    if environment == "production":
        auto_instrument(
            governance_policy="enforced",
            daily_budget_limit=100.0,
            enable_cost_alerts=True
        )
    else:
        auto_instrument(
            governance_policy="advisory",
            daily_budget_limit=10.0
        )
    """)
    
    print("🎯 Customer-Aware SaaS Pattern:")
    print("""
    def setup_customer_governance(customer_id: str, tier: str):
        budget_limits = {
            "free": 1.0,      # $1/day for free tier
            "premium": 10.0,  # $10/day for premium
            "enterprise": 100.0  # $100/day for enterprise
        }
        
        auto_instrument(
            customer_id=customer_id,
            daily_budget_limit=budget_limits.get(tier, 1.0),
            cost_center=f"customer-{tier}",
            governance_policy="enforced"
        )
    """)

async def main():
    """Main execution function."""
    print("🚀 Starting PromptLayer Auto-Instrumentation Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not os.getenv('PROMPTLAYER_API_KEY'):
        print("⚠️ PROMPTLAYER_API_KEY not found")
        print("💡 Set your PromptLayer API key: export PROMPTLAYER_API_KEY='pl-your-key'")
        print("📖 Get your API key from: https://promptlayer.com/")
        print()
        print("🎯 Demo will continue with simulation...")
        print()
    
    # Run demonstrations
    success = True
    
    # Main auto-instrumentation demo
    if not demonstrate_auto_instrumentation():
        success = False
    
    # Show before/after comparison
    show_before_after_comparison()
    
    # Configuration options
    demonstrate_configuration_options()
    
    # Enterprise patterns
    show_enterprise_patterns()
    
    if success:
        print("\n" + "🌟" * 50)
        print("🎉 PromptLayer Auto-Instrumentation Demo Complete!")
        print("\n📊 What You've Learned:")
        print("   ✅ Zero-code governance for existing PromptLayer applications")
        print("   ✅ Automatic cost tracking and team attribution")
        print("   ✅ Budget enforcement without changing business logic")
        print("   ✅ Enterprise-ready configuration patterns")
        
        print("\n🔍 Your Auto-Instrumented Stack:")
        print("   • PromptLayer: Existing prompt management workflows")
        print("   • GenOps: Automatic governance and cost intelligence")
        print("   • OpenTelemetry: Standard observability export")
        print("   • Zero Changes: Your existing code works exactly the same")
        
        print("\n📚 Next Steps:")
        print("   • Try manual instrumentation: python prompt_management.py")
        print("   • Explore A/B testing: python evaluation_integration.py")
        print("   • Advanced patterns: python production_patterns.py")
        
        print("\n💡 Integration Checklist:")
        print("   ✅ Add auto_instrument() to your application startup")
        print("   ✅ Set team/project environment variables")
        print("   ✅ Configure budget limits for your use case")
        print("   ✅ Your existing PromptLayer code now has governance!")
        
        print("🌟" * 50)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)