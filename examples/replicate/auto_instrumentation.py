#!/usr/bin/env python3
"""
🔧 GenOps Replicate Auto-Instrumentation - Phase 2 (15 minutes)

Zero-code instrumentation for existing Replicate applications.
Perfect for adding GenOps tracking to apps you already have running.

This example shows how to add comprehensive cost tracking and governance
to existing Replicate code without changing any of your application logic.

Requirements:
- REPLICATE_API_TOKEN environment variable
- pip install replicate genops-ai

Key Benefits:
- Works with existing Replicate code unchanged
- Automatic cost tracking across all model types  
- Team/project attribution for governance
- Real-time budget monitoring and alerts
"""

import os
import time


def demonstrate_auto_instrumentation():
    """Show how auto-instrumentation works with existing Replicate code."""

    print("🔧 GenOps Auto-Instrumentation Demo")
    print("=" * 50)

    # Step 1: Enable auto-instrumentation (ONE LINE!)
    print("Step 1: Enabling GenOps auto-instrumentation...")
    from genops.providers.replicate import auto_instrument
    auto_instrument()
    print("✅ Auto-instrumentation enabled - all replicate.run() calls now tracked!")
    print()

    # Step 2: Your existing Replicate code works unchanged
    print("Step 2: Running existing Replicate code (unchanged)...")
    import replicate

    # This is how your existing code probably looks - NO CHANGES NEEDED!
    try:
        # Text generation (existing code pattern)
        print("🔤 Text generation with Llama-2...")
        start_time = time.time()

        text_output = replicate.run(
            "meta/llama-2-7b-chat",
            input={
                "prompt": "Write a haiku about AI and cost tracking",
                "max_length": 100
            }
        )

        print(f"   Output: {text_output[:100]}...")
        print(f"   ⏱️  Time: {(time.time() - start_time)*1000:.0f}ms")
        print("   💰 Cost: Automatically tracked by GenOps!")
        print()

        # Image generation (existing code pattern)
        print("🎨 Image generation with FLUX...")
        start_time = time.time()

        image_output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": "A robot accountant calculating AI costs, digital art",
                "num_outputs": 1
            }
        )

        print(f"   Output: {len(image_output) if isinstance(image_output, list) else 1} image(s)")
        print(f"   ⏱️  Time: {(time.time() - start_time)*1000:.0f}ms")
        print("   💰 Cost: Automatically tracked by GenOps!")
        print()

        print("✅ SUCCESS! Both operations automatically tracked with GenOps")
        print("📊 All costs, latency, and governance data captured automatically")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Check your REPLICATE_API_TOKEN and network connection")
        return False

    return True

def demonstrate_governance_attributes():
    """Show how to add governance attributes for team tracking."""

    print("\n🏛️ Adding Governance Attributes")
    print("=" * 50)

    print("Step 3: Adding team/project attribution to existing calls...")

    # Import the adapter for manual control with governance
    from genops.providers.replicate import GenOpsReplicateAdapter

    adapter = GenOpsReplicateAdapter()

    try:
        # Your existing replicate.run() calls can be enhanced with governance
        response = adapter.run_model(
            model="meta/llama-2-7b-chat",
            input={
                "prompt": "Explain the benefits of AI cost tracking in one sentence",
                "max_length": 50
            },
            # Add governance attributes (no change to core logic!)
            team="engineering-team",
            project="cost-optimization",
            customer_id="internal-demo",
            environment="development"
        )

        print("✅ Enhanced governance tracking enabled!")
        print(f"   💬 Response: {response.content[:100]}...")
        print(f"   💰 Cost: ${response.cost_usd:.6f}")
        print("   📊 Team: engineering-team")
        print("   🏷️  Project: cost-optimization")
        print(f"   ⏱️  Latency: {response.latency_ms:.0f}ms")

    except Exception as e:
        print(f"❌ Error in governance demo: {e}")
        return False

    return True

def demonstrate_multi_modal_tracking():
    """Show cost tracking across different model types."""

    print("\n🎭 Multi-Modal Cost Tracking")
    print("=" * 50)

    # Use cost aggregator for workflow-level tracking
    from genops.providers.replicate_cost_aggregator import create_replicate_cost_context

    try:
        with create_replicate_cost_context("multi_modal_demo", budget_limit=1.0) as context:
            print("Step 4: Multi-modal workflow with unified cost tracking...")

            # Different model types in same workflow
            models_to_test = [
                ("meta/llama-2-7b-chat", "text", {"prompt": "Hello AI world!", "max_length": 20}),
                ("black-forest-labs/flux-schnell", "image", {"prompt": "Simple AI icon", "num_outputs": 1})
            ]

            for model, category, input_params in models_to_test:
                try:
                    print(f"   🤖 Testing {category} model: {model}")

                    import replicate
                    output = replicate.run(model, input=input_params)

                    # The auto-instrumentation automatically feeds the cost aggregator
                    print(f"      ✅ {category.title()} generation completed")

                except Exception as model_error:
                    print(f"      ⚠️ Skipped {model}: {model_error}")
                    continue

            # Get comprehensive summary
            summary = context.get_current_summary()

            print("\n📊 WORKFLOW SUMMARY:")
            print(f"   💰 Total Cost: ${summary.total_cost:.6f}")
            print(f"   🔄 Operations: {summary.operation_count}")
            print(f"   🎯 Models Used: {len(summary.unique_models)}")
            print(f"   📋 Categories: {', '.join(summary.unique_categories)}")

            if summary.optimization_recommendations:
                print("   💡 Optimization Tips:")
                for tip in summary.optimization_recommendations[:2]:
                    print(f"      • {tip}")

    except Exception as e:
        print(f"❌ Error in multi-modal demo: {e}")
        return False

    return True

def main():
    """Main demonstration of Replicate auto-instrumentation."""

    print("🚀 GenOps Replicate Auto-Instrumentation Demo")
    print("This shows how to add GenOps tracking to existing Replicate apps")
    print()

    # Check prerequisites
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not set")
        print("🔧 Get token: https://replicate.com/account/api-tokens")
        print("   export REPLICATE_API_TOKEN='r8_your_token_here'")
        return False

    success = True

    # Run demonstrations
    success &= demonstrate_auto_instrumentation()
    success &= demonstrate_governance_attributes()
    success &= demonstrate_multi_modal_tracking()

    if success:
        print("\n🎉 AUTO-INSTRUMENTATION DEMO COMPLETE!")
        print("=" * 50)
        print("✅ Your existing Replicate code now has:")
        print("   • Automatic cost tracking across all models")
        print("   • Team/project attribution for governance")
        print("   • Multi-modal workflow cost aggregation")
        print("   • Real-time optimization recommendations")
        print()
        print("🎯 PHASE 2 COMPLETE - Ready for production deployment!")
        print()
        print("🚀 NEXT STEPS:")
        print("   → python basic_tracking.py       # Learn manual adapter patterns")
        print("   → python cost_optimization.py   # Advanced cost intelligence")
        print("   → examples/replicate/README.md  # Complete documentation")

    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
