#!/usr/bin/env python3
"""
Together AI Auto-Instrumentation with GenOps

Demonstrates zero-code instrumentation for Together AI operations.
Shows how to add governance to existing Together AI code with minimal changes.

Usage:
    python auto_instrumentation.py

Features:
    - Zero-code governance for existing Together AI applications
    - Automatic cost tracking and attribution
    - Drop-in replacement for existing Together code
    - Seamless integration with OpenTelemetry observability
"""

import asyncio
import os
import sys

try:
    # Standard Together AI import (what users already have)
    from together import Together

    from genops.providers.together import TogetherModel, auto_instrument
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[together] together")
    print("Then run: python setup_validation.py")
    sys.exit(1)


def demonstrate_manual_approach():
    """Show traditional approach without auto-instrumentation."""
    print("📝 Traditional Approach (without GenOps)")
    print("-" * 40)

    try:
        # Traditional Together AI usage (what users already do)
        client = Together()

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "What are the benefits of auto-instrumentation?"}
            ],
            max_tokens=100
        )

        print("✅ Traditional approach works")
        print(f"   Response: {response.choices[0].message.content[:100]}...")
        print("   ❌ No cost tracking")
        print("   ❌ No governance attributes")
        print("   ❌ No budget controls")
        print("   ❌ No observability telemetry")

    except Exception as e:
        print(f"❌ Traditional approach failed: {e}")


def demonstrate_auto_instrumentation():
    """Show how auto-instrumentation adds governance with zero code changes."""
    print("\n🔧 Auto-Instrumentation Approach")
    print("-" * 40)

    # STEP 1: Enable auto-instrumentation with ONE line
    print("Step 1: Enable auto-instrumentation")
    adapter = auto_instrument(
        team=os.getenv('GENOPS_TEAM', 'auto-instrumented'),
        project=os.getenv('GENOPS_PROJECT', 'zero-code-demo'),
        environment=os.getenv('GENOPS_ENVIRONMENT', 'development'),
        daily_budget_limit=25.0,
        governance_policy='advisory'
    )
    print("✅ Auto-instrumentation enabled with one line!")

    # STEP 2: Use existing Together AI code unchanged
    print("\nStep 2: Use existing Together AI code (unchanged)")
    try:
        # Same exact code as before - but now with governance!
        client = Together()

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": "What are the benefits of auto-instrumentation?"}
            ],
            max_tokens=100
        )

        print("✅ Same code now has governance!")
        print(f"   Response: {response.choices[0].message.content[:100]}...")
        print("   ✅ Automatic cost tracking")
        print("   ✅ Governance attributes applied")
        print("   ✅ Budget monitoring active")
        print("   ✅ OpenTelemetry traces generated")

        # Show cost summary
        cost_summary = adapter.get_cost_summary()
        print("\n💰 Automatic Cost Tracking:")
        print(f"   Daily costs: ${cost_summary['daily_costs']:.6f}")
        print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")

    except Exception as e:
        print(f"❌ Auto-instrumented approach failed: {e}")
        return


def demonstrate_mixed_approaches():
    """Show how manual and auto-instrumented approaches can coexist."""
    print("\n🔀 Mixed Approaches")
    print("-" * 40)

    # Get the current auto-instrumented adapter
    from src.genops.providers.together import get_current_adapter
    adapter = get_current_adapter()

    if not adapter:
        print("❌ No auto-instrumentation active")
        return

    print("Combining auto-instrumentation with manual governance:")

    try:
        # Use the adapter directly for fine-grained control
        result = adapter.chat_with_governance(
            messages=[
                {"role": "user", "content": "Compare auto vs manual instrumentation approaches"}
            ],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=150,
            temperature=0.7,
            # Fine-grained governance attributes
            feature="instrumentation-comparison",
            approach="mixed",
            demo_type="governance-showcase"
        )

        print("✅ Manual governance with fine-grained control:")
        print(f"   Response: {result.response[:120]}...")
        print(f"   Model: {result.model_used}")
        print(f"   Cost: ${result.cost:.6f}")
        print("   Custom attributes: feature, approach, demo_type")

        # Also show that regular Together calls still work
        client = Together()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "This is automatically tracked too!"}],
            max_tokens=50
        )

        print("\n✅ Regular Together calls automatically tracked:")
        print(f"   Response: {response.choices[0].message.content[:80]}...")
        print("   (Cost and governance automatically applied)")

    except Exception as e:
        print(f"❌ Mixed approach failed: {e}")


def demonstrate_async_auto_instrumentation():
    """Show auto-instrumentation with async operations."""
    print("\n⚡ Async Auto-Instrumentation")
    print("-" * 40)

    async def async_operations():
        """Demonstrate async operations with auto-instrumentation."""
        from together import AsyncTogether

        try:
            client = AsyncTogether()

            # Multiple concurrent operations
            tasks = [
                client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    messages=[{"role": "user", "content": f"Task {i}: What is AI?"}],
                    max_tokens=50
                )
                for i in range(3)
            ]

            print("🚀 Running 3 concurrent operations...")
            responses = await asyncio.gather(*tasks)

            print("✅ All async operations completed with governance:")
            for i, response in enumerate(responses, 1):
                print(f"   Task {i}: {response.choices[0].message.content[:60]}...")

            # Show updated cost tracking
            from src.genops.providers.together import get_current_adapter
            adapter = get_current_adapter()
            if adapter:
                cost_summary = adapter.get_cost_summary()
                print("\n💰 Updated costs after async operations:")
                print(f"   Total daily costs: ${cost_summary['daily_costs']:.6f}")

        except Exception as e:
            print(f"❌ Async operations failed: {e}")

    # Run async demo
    try:
        asyncio.run(async_operations())
    except Exception as e:
        print(f"❌ Async demo failed: {e}")


def main():
    """Demonstrate auto-instrumentation capabilities."""
    print("🔧 Together AI Auto-Instrumentation Demo")
    print("=" * 50)

    # Show the difference
    demonstrate_manual_approach()
    demonstrate_auto_instrumentation()
    demonstrate_mixed_approaches()
    demonstrate_async_auto_instrumentation()

    # Final summary
    print("\n" + "=" * 50)
    print("📊 Auto-Instrumentation Benefits")
    print("=" * 50)

    from src.genops.providers.together import get_current_adapter
    adapter = get_current_adapter()

    if adapter:
        cost_summary = adapter.get_cost_summary()
        print("✅ Zero-code governance achieved:")
        print(f"   • Cost tracking: ${cost_summary['daily_costs']:.6f} spent today")
        print(f"   • Budget monitoring: {cost_summary['daily_budget_utilization']:.1f}% utilized")
        print(f"   • Team attribution: {cost_summary['team']}")
        print(f"   • Project tracking: {cost_summary['project']}")
        print(f"   • Governance policy: {cost_summary['governance_policy']}")
        print(f"   • Active sessions: {cost_summary['active_sessions']}")

        print("\n🎯 Key Advantages:")
        print("   ✅ Drop-in replacement for existing code")
        print("   ✅ No refactoring required")
        print("   ✅ Automatic cost and performance tracking")
        print("   ✅ Governance attributes applied globally")
        print("   ✅ OpenTelemetry integration")
        print("   ✅ Can mix with manual governance for fine control")

    print("\n🚀 Next Steps:")
    print("   • Add auto_instrument() to your existing Together AI code")
    print("   • Configure team, project, and budget limits")
    print("   • Monitor costs and performance automatically")
    print("   • Use manual governance for fine-grained control when needed")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure to run setup_validation.py first")
        sys.exit(1)
