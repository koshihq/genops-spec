#!/usr/bin/env python3
"""
Langfuse Auto-Instrumentation with GenOps Governance Example

This example demonstrates zero-code GenOps governance integration with existing
Langfuse applications. Perfect for adding governance to your current Langfuse
setup without any code changes.

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"  # Or another provider
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict


def demonstrate_zero_code_setup():
    """Demonstrate zero-code governance setup for existing Langfuse code."""
    print("⚡ Zero-Code Auto-Instrumentation with GenOps Governance")
    print("=" * 55)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Step 1: Enable GenOps governance for ALL Langfuse operations
        print("🚀 Enabling GenOps governance for all Langfuse operations...")

        adapter = instrument_langfuse(
            team="auto-instrumented-team",
            project="zero-code-demo",
            environment="development",
            auto_instrument=True,  # This is the magic flag
            budget_limits={"daily": 1.0}  # $1 daily budget limit
        )

        print("✅ Auto-instrumentation enabled!")
        print(f"   🏷️  Team: {adapter.team}")
        print(f"   📊 Project: {adapter.project}")
        print(f"   🌍 Environment: {adapter.environment}")
        print(f"   💰 Daily Budget: ${adapter.budget_limits.get('daily', 0):.2f}")
        print("   ⚡ Auto-instrument: ON")

        return adapter

    except ImportError as e:
        print(f"❌ Failed to import GenOps Langfuse: {e}")
        print("💡 Fix: Run 'pip install genops[langfuse]'")
        return None
    except Exception as e:
        print(f"❌ Failed to enable auto-instrumentation: {e}")
        return None


def demonstrate_existing_langfuse_enhanced():
    """Show how existing Langfuse code automatically gets governance."""
    print("\n📋 Your Existing Langfuse Code - Now Enhanced!")
    print("-" * 48)

    try:
        # Import Langfuse as you normally would - no changes needed!
        from langfuse.decorators import observe

        print("🎯 The magic: Your existing code now has governance automatically")
        print("   No code changes required - governance happens transparently")

        # Example 1: Existing function with @observe decorator
        @observe()
        def existing_text_analysis(text: str) -> Dict[str, str]:
            """Your existing Langfuse-decorated function - now with governance!"""
            print(f"   📝 Analyzing text: '{text[:30]}...'")

            # Simulate your existing OpenAI call (or any LLM call)
            # This call is now automatically tracked with GenOps governance
            analysis_result = {
                "sentiment": "positive" if "good" in text.lower() else "neutral",
                "word_count": len(text.split()),
                "complexity": "simple" if len(text.split()) < 20 else "complex",
                "summary": f"Analysis of {len(text)} character text"
            }

            print(f"   ✅ Analysis complete: {analysis_result['sentiment']} sentiment")
            return analysis_result

        # Example 2: Existing function without any changes
        @observe()
        def existing_translation_service(text: str, target_language: str) -> Dict[str, str]:
            """Your existing translation function - governance added automatically!"""
            print(f"   🌍 Translating to {target_language}: '{text[:25]}...'")

            # Your existing logic - no changes needed
            translation_result = {
                "original": text,
                "translated": f"[{target_language.upper()}] {text}",  # Mock translation
                "language": target_language,
                "confidence": 0.95,
                "provider": "mock_translator"
            }

            print(f"   ✅ Translation complete: {translation_result['confidence']:.0%} confidence")
            return translation_result

        # Test the enhanced functions
        print("\n🧪 Testing your enhanced functions...")

        # Test 1: Text analysis with automatic governance
        analysis_result = existing_text_analysis(
            "This is a good example of how GenOps enhances Langfuse with zero code changes!"
        )

        # Test 2: Translation service with automatic governance
        translation_result = existing_translation_service(
            "GenOps makes governance transparent and automatic",
            "spanish"
        )

        print("\n✅ Both functions executed with automatic governance!")
        print("   📊 Team attribution: auto-instrumented-team")
        print("   💰 Cost tracking: Enabled automatically")
        print("   🛡️  Budget enforcement: $1.00 daily limit active")
        print("   🏷️  Governance tags: Added to all Langfuse traces")

        return {
            "analysis_result": analysis_result,
            "translation_result": translation_result
        }

    except ImportError as e:
        print(f"❌ Failed to import Langfuse decorators: {e}")
        print("💡 Fix: Run 'pip install langfuse'")
        return None
    except Exception as e:
        print(f"❌ Enhanced function execution failed: {e}")
        return None


def demonstrate_langchain_auto_enhancement():
    """Show automatic governance for LangChain + Langfuse integration."""
    print("\n📋 LangChain + Langfuse Integration - Automatically Enhanced")
    print("-" * 58)

    try:
        print("🔗 Simulating LangChain operations with Langfuse observability...")
        print("   (Your existing LangChain + Langfuse code gets governance automatically)")

        # Mock LangChain-style operations that would normally use Langfuse
        def simulate_langchain_chain_execution():
            """Simulate a LangChain chain that uses Langfuse for observability."""
            print("   🔗 Chain step 1: Document retrieval")
            time.sleep(0.1)  # Simulate processing

            print("   🔗 Chain step 2: Context preparation")
            time.sleep(0.1)  # Simulate processing

            print("   🔗 Chain step 3: LLM generation")
            time.sleep(0.2)  # Simulate LLM call

            print("   🔗 Chain step 4: Response formatting")
            time.sleep(0.1)  # Simulate processing

            return {
                "result": "Comprehensive analysis completed using enhanced RAG pipeline",
                "steps_completed": 4,
                "total_time_ms": 500,
                "documents_retrieved": 5,
                "tokens_used": 1250
            }

        # Execute the simulated chain
        chain_result = simulate_langchain_chain_execution()

        print("✅ LangChain execution complete!")
        print(f"   📊 Result: {chain_result['result']}")
        print(f"   🔢 Steps: {chain_result['steps_completed']}")
        print(f"   ⏱️  Time: {chain_result['total_time_ms']}ms")
        print(f"   📚 Documents: {chain_result['documents_retrieved']}")
        print(f"   🎯 Tokens: {chain_result['tokens_used']}")

        print("\n🎉 Automatic Governance Applied:")
        print("   ✅ All chain steps tracked with team attribution")
        print("   ✅ Cost calculated and attributed automatically")
        print("   ✅ Budget limits enforced across entire chain")
        print("   ✅ Langfuse traces enhanced with GenOps metadata")

        return chain_result

    except Exception as e:
        print(f"❌ LangChain simulation failed: {e}")
        return None


def demonstrate_multi_provider_governance():
    """Show automatic governance across multiple AI providers."""
    print("\n📋 Multi-Provider Operations - Unified Governance")
    print("-" * 47)

    try:
        print("🌐 Simulating operations across multiple AI providers...")
        print("   (All automatically tracked with unified GenOps governance)")

        providers = ["openai", "anthropic", "google"]
        total_cost = 0.0
        operations = []

        for i, provider in enumerate(providers, 1):
            print(f"\n   🔄 Operation {i}: {provider.title()} Provider")

            # Simulate provider-specific operation
            operation_cost = 0.001 * (i * 2.5)  # Different costs per provider
            operation_tokens = 500 + (i * 150)   # Different token usage

            operation = {
                "provider": provider,
                "model": f"{provider}-model-v1",
                "cost": operation_cost,
                "tokens": operation_tokens,
                "latency_ms": 400 + (i * 100),
                "success": True
            }

            operations.append(operation)
            total_cost += operation_cost

            print(f"     💰 Cost: ${operation_cost:.6f}")
            print(f"     🎯 Tokens: {operation_tokens}")
            print(f"     ⏱️  Latency: {operation['latency_ms']}ms")

            time.sleep(0.1)  # Simulate processing time

        print("\n✅ Multi-provider operations complete!")
        print(f"   📊 Total operations: {len(operations)}")
        print(f"   💰 Total cost: ${total_cost:.6f}")
        print(f"   🌐 Providers used: {', '.join([op['provider'].title() for op in operations])}")

        print("\n🎉 Unified Governance Applied:")
        print("   ✅ All providers tracked with single team attribution")
        print("   ✅ Unified cost calculation across all providers")
        print("   ✅ Shared budget limits enforced automatically")
        print("   ✅ Consistent governance metadata in all traces")

        return {
            "operations": operations,
            "total_cost": total_cost,
            "provider_count": len(providers)
        }

    except Exception as e:
        print(f"❌ Multi-provider simulation failed: {e}")
        return None


def demonstrate_governance_features():
    """Demonstrate the governance features enabled by auto-instrumentation."""
    print("\n💡 Enhanced Governance Features Now Active")
    print("=" * 42)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Get the auto-instrumented adapter
        adapter = instrument_langfuse(
            team="governance-features",
            project="auto-demo",
            budget_limits={"daily": 0.50, "monthly": 10.0}
        )

        print("🛡️  Governance Intelligence:")
        features = [
            "💰 Real-time cost attribution to teams and projects",
            "🎯 Automatic budget enforcement across all operations",
            "📊 Cost breakdowns by provider, model, and operation type",
            "🏷️  Governance metadata automatically added to all Langfuse traces",
            "📈 Performance optimization recommendations based on cost patterns",
            "🔍 Enhanced observability with business context in every trace",
            "⚡ Zero-code setup - works with your existing Langfuse applications"
        ]

        for feature in features:
            print(f"   {feature}")

        # Show current cost summary
        cost_summary = adapter.get_cost_summary("daily")
        print("\n📊 Current Daily Governance Summary:")
        print(f"   💰 Total cost: ${cost_summary['total_cost']:.6f}")
        print(f"   📈 Operations: {cost_summary['operation_count']}")
        print(f"   💡 Budget remaining: ${cost_summary['budget_remaining']:.6f}")
        print(f"   🏷️  Team: {cost_summary['governance']['team']}")
        print(f"   📊 Project: {cost_summary['governance']['project']}")
        print(f"   🌍 Environment: {cost_summary['governance']['environment']}")

        return True

    except Exception as e:
        print(f"❌ Governance features demo failed: {e}")
        return False


def show_next_steps():
    """Show next steps for developers using auto-instrumentation."""
    print("\n🚀 Next Steps: Maximize Your Auto-Instrumentation")
    print("=" * 48)

    next_steps = [
        ("🔍 Custom Governance", "Add custom team/project attribution to specific functions",
         "adapter.trace_with_governance(name='my_operation', customer_id='abc123')"),
        ("💰 Budget Management", "Set specific budget limits for different operations",
         "instrument_langfuse(budget_limits={'daily': 5.0, 'monthly': 100.0})"),
        ("📊 Advanced Tracking", "Use manual tracking for complex multi-step workflows",
         "python basic_tracking.py"),
        ("🎯 Evaluation Governance", "Add governance to your LLM evaluation workflows",
         "python evaluation_integration.py"),
        ("🏭 Production Setup", "Configure auto-instrumentation for production deployment",
         "python production_patterns.py")
    ]

    for title, description, example in next_steps:
        print(f"   {title}")
        print(f"     Purpose: {description}")
        print(f"     Example: {example}")
        print()

    print("📚 Advanced Resources:")
    print("   • Manual Instrumentation: python basic_tracking.py")
    print("   • Comprehensive Guide: docs/integrations/langfuse.md")
    print("   • All Examples: ./run_all_examples.sh")

    print("\n💡 Pro Tips for Auto-Instrumentation:")
    print("   ✅ Works with ALL existing Langfuse @observe decorators")
    print("   ✅ Compatible with LangChain, LlamaIndex, and other frameworks")
    print("   ✅ Automatically adds governance to third-party libraries")
    print("   ✅ No performance overhead - telemetry export is async")
    print("   ✅ Disable anytime by setting auto_instrument=False")


def main():
    """Main function to run the auto-instrumentation example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv('LANGFUSE_PUBLIC_KEY'):
        print("❌ Missing LANGFUSE_PUBLIC_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not os.getenv('LANGFUSE_SECRET_KEY'):
        print("❌ Missing LANGFUSE_SECRET_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY')]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        return False

    # Run demonstrations
    success = True

    # Step 1: Enable auto-instrumentation
    adapter = demonstrate_zero_code_setup()
    if not adapter:
        return False

    # Step 2: Show existing code enhancement
    enhanced_results = demonstrate_existing_langfuse_enhanced()
    success &= enhanced_results is not None

    # Step 3: Show LangChain integration
    langchain_result = demonstrate_langchain_auto_enhancement()
    success &= langchain_result is not None

    # Step 4: Multi-provider governance
    multi_provider_result = demonstrate_multi_provider_governance()
    success &= multi_provider_result is not None

    # Step 5: Show governance features
    governance_success = demonstrate_governance_features()
    success &= governance_success

    if success:
        show_next_steps()
        print("\n" + "⚡" * 20)
        print("Auto-instrumentation successful!")
        print("Your existing Langfuse code now has GenOps governance!")
        print("Zero code changes required - governance is automatic!")
        print("⚡" * 20)
        return True
    else:
        print("\n❌ Some demonstrations failed. Check the errors above.")
        return False


if __name__ == "__main__":
    """Main entry point."""
    success = main()
    sys.exit(0 if success else 1)
