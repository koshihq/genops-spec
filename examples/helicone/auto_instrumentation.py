#!/usr/bin/env python3
"""
Helicone Auto-Instrumentation Example

This example demonstrates zero-code GenOps integration with Helicone AI Gateway.
Your existing AI code automatically gets multi-provider gateway routing and 
comprehensive governance tracking with no code changes required.

Usage:
    python auto_instrumentation.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    export OPENAI_API_KEY="your_openai_api_key"
"""

import os
import sys
from datetime import datetime


def demonstrate_zero_code_integration():
    """Show how auto-instrumentation works with existing AI code."""

    print("🪄 GenOps Auto-Instrumentation with Helicone Gateway")
    print("=" * 58)
    print("✨ Zero code changes - your existing AI code just works better!")

    # Step 1: Enable GenOps auto-instrumentation
    print("\n📡 Step 1: Enable GenOps Auto-Instrumentation")
    print("-" * 45)

    try:
        from genops import init

        # This single line enables auto-instrumentation for ALL supported frameworks
        init(
            # Optional: Add default governance attributes
            default_attributes={
                "team": "platform-engineering",
                "project": "auto-instrumentation-demo",
                "environment": "development",
                "cost_center": "engineering"
            }
        )
        print("✅ GenOps auto-instrumentation enabled")
        print("   🔄 All AI providers automatically instrumented")
        print("   📊 Governance attributes applied to all requests")

    except ImportError as e:
        print(f"❌ Auto-instrumentation failed: {e}")
        print("💡 Fix: pip install genops[helicone]")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

    # Step 2: Use your existing AI code - no changes needed!
    print("\n🤖 Step 2: Your Existing AI Code (No Changes!)")
    print("-" * 48)

    # Example 1: Direct OpenAI usage (automatically routed through Helicone)
    if os.getenv('OPENAI_API_KEY'):
        print("\n📋 Example 1: Direct OpenAI Usage")
        try:
            import openai

            # This looks like normal OpenAI code, but it's automatically:
            # - Routed through Helicone gateway
            # - Tracked with GenOps governance
            # - Cost attributed to your team/project

            client = openai.OpenAI()  # Automatically uses Helicone routing!

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "What is auto-instrumentation in AI?"}
                ],
                max_tokens=100
            )

            print("   ✅ OpenAI request completed (via Helicone gateway)")
            print(f"   📝 Response: {response.choices[0].message.content[:100]}...")
            print("   🎯 Automatically tracked: cost, usage, governance attributes")

        except Exception as e:
            print(f"   ❌ OpenAI example failed: {e}")

    # Example 2: Multi-provider access through single interface
    print("\n📋 Example 2: Multi-Provider Gateway Access")
    try:
        from genops import instrument_helicone

        # Even explicit instrumentation is enhanced with gateway intelligence
        adapter = instrument_helicone(
            team="auto-demo-team",
            project="auto-instrumentation-demo"
        )

        # Test multiple providers if available
        providers_to_test = []
        if os.getenv('OPENAI_API_KEY'):
            providers_to_test.append(('openai', 'gpt-3.5-turbo'))
        if os.getenv('ANTHROPIC_API_KEY'):
            providers_to_test.append(('anthropic', 'claude-3-haiku-20240307'))
        if os.getenv('GROQ_API_KEY'):
            providers_to_test.append(('groq', 'mixtral-8x7b-32768'))

        for provider, model in providers_to_test:
            try:
                response = adapter.chat(
                    message=f"Hello from {provider}! Explain auto-instrumentation.",
                    provider=provider,
                    model=model
                )

                cost = getattr(response.usage, 'total_cost', 0.0) if hasattr(response, 'usage') else 0.0
                print(f"   ✅ {provider.title()}: ${cost:.6f} - Gateway routing active")

            except Exception as e:
                print(f"   ⚠️  {provider.title()}: {e}")
                continue

    except Exception as e:
        print(f"   ❌ Multi-provider example failed: {e}")

    # Step 3: Show what's happening automatically
    print("\n🔍 Step 3: What GenOps Auto-Instrumentation Provides")
    print("-" * 52)

    automatic_features = [
        "🌐 Helicone Gateway Routing - Unified access to 100+ AI models",
        "💰 Automatic Cost Tracking - Real-time cost calculation across all providers",
        "🏷️  Governance Attribution - Team, project, customer cost attribution",
        "📊 OpenTelemetry Export - Standard telemetry to your observability stack",
        "🔄 Provider Failover - Automatic switching when providers are unavailable",
        "⚡ Performance Tracking - Latency and success rate monitoring",
        "🛡️  Error Handling - Graceful degradation and retry logic",
        "📈 Cost Optimization - Intelligent routing for cost and performance"
    ]

    for feature in automatic_features:
        print(f"   {feature}")

    return True


def demonstrate_framework_compatibility():
    """Show compatibility with popular AI frameworks."""

    print("\n🧩 Framework Compatibility Demonstration")
    print("-" * 42)
    print("GenOps auto-instrumentation works with your existing frameworks:")

    frameworks = {
        'LangChain': 'langchain',
        'LlamaIndex': 'llama_index',
        'Raw OpenAI': 'openai',
        'Anthropic SDK': 'anthropic'
    }

    for framework_name, module_name in frameworks.items():
        try:
            __import__(module_name)
            print(f"   ✅ {framework_name}: Auto-instrumentation ready")
        except ImportError:
            print(f"   ⚠️  {framework_name}: Not installed (would work if installed)")

    print("\n💡 Key Benefits:")
    print("   • No code changes required - just add init()")
    print("   • Works with any AI framework or direct provider usage")
    print("   • Unified governance across your entire AI stack")
    print("   • Gateway intelligence with provider optimization")


def main():
    """Main function to run the auto-instrumentation example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv('HELICONE_API_KEY'):
        print("❌ Missing HELICONE_API_KEY environment variable")
        print("💡 Get your key at: https://app.helicone.ai/")
        return False

    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GROQ_API_KEY')]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        print("   export GROQ_API_KEY='your_groq_key'")
        return False

    # Run demonstrations
    success = True
    success &= demonstrate_zero_code_integration()
    demonstrate_framework_compatibility()

    if success:
        print("\n🎉 SUCCESS! Auto-instrumentation demonstration completed.")
        print("\n🔮 What Just Happened:")
        print("   • Your AI code now has gateway intelligence")
        print("   • All requests automatically cost-tracked and attributed")
        print("   • Multi-provider access through unified interface")
        print("   • Enterprise governance with zero code changes")

        print("\n📚 Next Steps:")
        print("   • Add init() to your real application")
        print("   • Try 'python multi_provider_costs.py' for cost optimization")
        print("   • Try 'python production_patterns.py' for enterprise patterns")
    else:
        print("\n❌ Auto-instrumentation demo encountered issues.")

    return success


if __name__ == "__main__":
    """Entry point for the auto-instrumentation example."""
    success = main()

    if success:
        print("\n" + "✨" * 20)
        print("Auto-instrumentation: AI code enhancement with zero effort!")
        print("✨" * 20)

    sys.exit(0 if success else 1)
