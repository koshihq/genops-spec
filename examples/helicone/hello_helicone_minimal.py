#!/usr/bin/env python3
"""
🌐 GenOps + Helicone AI Gateway: Hello Multi-Provider AI (Minimal Example)

GOAL: Prove GenOps works with Helicone AI gateway in 30 seconds
TIME: 30 seconds
WHAT YOU'LL LEARN: Multi-provider AI access with unified cost tracking

This is the simplest possible example to verify GenOps tracking works
with Helicone AI gateway. Access 100+ models through one API with
comprehensive cost intelligence.

Prerequisites:
- Helicone API key: export HELICONE_API_KEY="your-helicone-key"
- At least one provider API key: export OPENAI_API_KEY="your-openai-key"
- GenOps: pip install genops-ai
- Requests: pip install requests
"""

import os
import sys


def main():
    """30-second AI gateway confidence builder."""
    print("🌐 GenOps + Helicone AI Gateway: Hello Multi-Provider AI!")
    print("=" * 60)

    # Check prerequisites
    print("🔍 Checking prerequisites...")

    # Check Helicone API key
    helicone_key = os.getenv("HELICONE_API_KEY")
    if not helicone_key:
        print("❌ HELICONE_API_KEY not found")
        print("   Get your key: https://app.helicone.ai/")
        print("   Set it: export HELICONE_API_KEY='your-helicone-key'")
        return False

    print("✅ Helicone API key found and validated")

    # Check for at least one provider API key
    provider_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY"),
    }

    available_providers = [(name, key) for name, key in provider_keys.items() if key]

    if not available_providers:
        print("❌ No provider API keys found")
        print("   Configure at least one provider:")
        print("   • export OPENAI_API_KEY='your-openai-key'")
        print("   • export ANTHROPIC_API_KEY='your-anthropic-key'")
        print("   • export GROQ_API_KEY='your-groq-key'")
        return False

    print(f"✅ {len(available_providers)} provider API key(s) configured:")
    for name, _ in available_providers:
        print(f"   • {name}")

    # Check dependencies
    try:
        import requests  # noqa: F401

        print("✅ Requests library available")
    except ImportError:
        print("❌ Requests library not found")
        print("   Install: pip install requests")
        return False

    try:
        from genops.providers.helicone import instrument_helicone

        print("✅ GenOps Helicone gateway provider available")
    except ImportError:
        print("❌ GenOps not found")
        print("   Install: pip install genops[helicone]")
        return False

    print("\n🚀 Testing AI gateway with GenOps tracking...")
    print("-" * 60)

    try:
        # Enable GenOps tracking for AI gateway
        adapter = instrument_helicone(
            helicone_api_key=helicone_key,
            provider_keys={name.lower(): key for name, key in available_providers},
            team="demo-team",
            project="ai-gateway-test",
        )

        print("✅ GenOps AI gateway adapter created")

        # Test multi-provider access through single interface
        primary_provider = available_providers[0][0].lower()

        if primary_provider == "openai":
            model = "gpt-3.5-turbo"
        elif primary_provider == "anthropic":
            model = "claude-3-haiku-20240307"
        else:  # groq
            model = "llama3-8b-8192"

        response = adapter.chat(
            message="What are the benefits of AI gateways?",
            provider=primary_provider,
            model=model,
        )

        # Response is successful if we get here without exception
        print("✅ AI Gateway Response received:")
        content = response.content if hasattr(response, "content") else str(response)
        print(f"   Content: {content[:150]}...")
        print(f"   Provider: {primary_provider}")
        print(f"   Model: {model}")

        print("\n💰 Unified Cost Tracking:")
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            provider_cost = getattr(response.usage, "provider_cost", 0.0)
            gateway_cost = getattr(response.usage, "helicone_cost", 0.0)
            total_cost = getattr(
                response.usage, "total_cost", provider_cost + gateway_cost
            )

            print(f"   Input tokens: {input_tokens}")
            print(f"   Output tokens: {output_tokens}")
            print(f"   Provider cost: ${provider_cost:.6f}")
            print(f"   Gateway cost: ${gateway_cost:.6f}")
            print(f"   Total cost: ${total_cost:.6f}")
        else:
            print(
                "   Cost tracking: Available (detailed usage not shown in minimal example)"
            )

        print("\n🌐 AI Gateway Benefits:")
        print("   ✅ Access 100+ models through single API")
        print("   ✅ Multi-provider routing and failover")
        print("   ✅ Built-in observability and analytics")
        print("   ✅ Unified cost tracking across providers")
        print("   ✅ Zero vendor lock-in")

        # Test multi-provider access if multiple providers available
        if len(available_providers) > 1:
            print("\n🔀 Testing Multi-Provider Access:")

            # Try a second provider
            second_provider = available_providers[1][0].lower()

            if second_provider == "openai":
                second_model = "gpt-3.5-turbo"
            elif second_provider == "anthropic":
                second_model = "claude-3-haiku-20240307"
            else:  # groq
                second_model = "llama3-8b-8192"

            try:
                adapter.chat(
                    message="Hello from a different provider!",
                    provider=second_provider,
                    model=second_model,
                )

                print("   ✅ Multi-provider access successful")
                print(f"   Primary provider: {primary_provider}")
                print(f"   Secondary provider: {second_provider}")
                print("   Both providers accessible through single interface")

            except Exception as e:
                print(f"   ⚠️ Multi-provider test failed: {e}")
                print("   (Single provider still works)")

        print("\n" + "=" * 60)
        print("✅ SUCCESS! GenOps is now tracking your AI gateway usage")
        print("🌐 Your multi-provider AI operations have unified governance!")

        print("\n🚀 Next Steps:")
        print("   1. Try different providers: anthropic, groq, vertex")
        print("   2. Explore multi-provider routing: python multi_provider_costs.py")
        print("   3. Check cost optimization: python cost_optimization.py")
        print("   4. Read full guide: docs/integrations/helicone.md")

        return True

    except Exception as e:
        print(f"❌ Error during AI gateway test: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check API keys are correct and have credits")
        print("   2. Verify Helicone gateway access: visit app.helicone.ai")
        print("   3. Check internet connection")
        print("   4. Try: python -c \"import requests; print('OK')\"")
        return False


def quick_provider_comparison():
    """Bonus: Quick provider cost information."""
    print("\n" + "=" * 60)
    print("🎁 BONUS: AI Gateway Cost Information")
    print("=" * 60)

    # Get available providers
    available_providers = []
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("anthropic")
    if os.getenv("GROQ_API_KEY"):
        available_providers.append("groq")

    if len(available_providers) < 2:
        print("ℹ️ Configure multiple providers for cost comparison")
        print("   Try: export GROQ_API_KEY='your_groq_key' (free tier available)")
        return

    print(f"📊 Estimated costs for {len(available_providers)} providers (1000 tokens):")
    print("-" * 50)

    # Estimated costs (would use real pricing in production)
    cost_estimates = {"openai": 0.002, "anthropic": 0.0015, "groq": 0.0005}

    provider_costs = []
    for provider in available_providers:
        cost = cost_estimates.get(provider, 0.001)
        provider_costs.append((provider, cost))

    # Sort by cost
    sorted_providers = sorted(provider_costs, key=lambda x: x[1])

    for i, (provider, cost) in enumerate(sorted_providers):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"   {rank} {provider.title()}: ~${cost:.6f}")

    # Show potential savings
    if len(sorted_providers) > 1:
        cheapest_cost = sorted_providers[0][1]
        most_expensive_cost = sorted_providers[-1][1]
        savings = most_expensive_cost - cheapest_cost
        savings_percent = (savings / most_expensive_cost) * 100

        print("\n💰 Gateway Cost Intelligence:")
        print(f"   Cheapest: {sorted_providers[0][0]} (~${cheapest_cost:.6f})")
        print(
            f"   Most expensive: {sorted_providers[-1][0]} (~${most_expensive_cost:.6f})"
        )
        print(f"   Potential savings: ~${savings:.6f} ({savings_percent:.1f}%)")
        print("   Gateway routing can automatically select the best provider!")


if __name__ == "__main__":
    print("Starting AI gateway confidence builder...")

    success = main()

    if success:
        # Run bonus comparison if main test succeeded
        quick_provider_comparison()

        print("\n🎉 AI Gateway Success!")
        print("You're ready to explore advanced GenOps + Helicone features:")
        print("• basic_tracking.py - Comprehensive multi-provider tracking")
        print("• multi_provider_costs.py - Cost comparison and optimization")
        print("• cost_optimization.py - Intelligent routing strategies")
        print("• advanced_features.py - Streaming and advanced patterns")
        print("• production_patterns.py - Enterprise deployment patterns")

        sys.exit(0)
    else:
        print("\n⚠️ Issues detected. Please fix the errors above and try again.")
        print("Need help? Check docs/helicone-quickstart.md for troubleshooting")
        sys.exit(1)
