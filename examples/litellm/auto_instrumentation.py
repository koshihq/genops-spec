#!/usr/bin/env python3
"""
LiteLLM Zero-Code Auto-Instrumentation with GenOps

Demonstrates the highest-leverage GenOps integration: single instrumentation
layer providing governance telemetry across 100+ LLM providers through
LiteLLM's unified interface.

Usage:
    export OPENAI_API_KEY="your_key_here"
    python auto_instrumentation.py

Features:
    - Zero-code instrumentation for existing LiteLLM applications
    - Automatic cost tracking across all 100+ supported providers
    - Unified governance telemetry with team/project attribution
    - Budget controls and compliance monitoring
    - Provider-agnostic usage analytics
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_requirements():
    """Check if required packages and configuration are available."""
    print("🔍 Checking requirements...")

    # Check LiteLLM
    try:
        import litellm  # noqa: F401

        print("✅ LiteLLM available")
    except ImportError:
        print("❌ LiteLLM not found")
        print("💡 Install: pip install litellm")
        return False

    # Check GenOps
    try:
        from genops.providers.litellm import auto_instrument  # noqa: F401

        print("✅ GenOps LiteLLM provider available")
    except ImportError:
        print("❌ GenOps LiteLLM provider not found")
        print("💡 Install: pip install genops[litellm]")
        return False

    # Check API keys (at least one required)
    api_keys_found = []
    api_key_checks = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "Azure": "AZURE_API_KEY",
        "Cohere": "COHERE_API_KEY",
    }

    for provider, env_var in api_key_checks.items():
        if os.getenv(env_var):
            api_keys_found.append(provider)
            print("✅ API key configured")

    if not api_keys_found:
        print("⚠️  No API keys configured")
        print("💡 Set at least one: export OPENAI_API_KEY=your_key")
        print("   Supported: OpenAI, Anthropic, Google, Azure, Cohere, and 95+ more")
        return False

    print(f"🎯 Ready with {len(api_keys_found)} provider(s) configured")
    return True


def demo_zero_code_instrumentation():
    """Demonstrate zero-code auto-instrumentation."""
    print("\n" + "=" * 60)
    print("🚀 Demo: Zero-Code Auto-Instrumentation")
    print("=" * 60)

    # Import LiteLLM and GenOps
    import litellm

    from genops.providers.litellm import auto_instrument, get_usage_stats

    print("📋 Step 1: Enable GenOps auto-instrumentation")
    print("   This adds governance to ALL LiteLLM requests across 100+ providers")

    # Enable auto-instrumentation with governance settings
    success = auto_instrument(
        team="demo-team",
        project="litellm-demo",
        environment="development",
        daily_budget_limit=10.0,  # $10 daily limit for demo
        governance_policy="advisory",  # Warnings only, don't block
    )

    if not success:
        print("❌ Failed to enable auto-instrumentation")
        return False

    print("✅ Auto-instrumentation enabled!")
    print("   • All LiteLLM requests now include GenOps governance")
    print("   • Cost tracking active across all 100+ providers")
    print("   • Team attribution: demo-team / litellm-demo")

    print("\n📋 Step 2: Use LiteLLM normally - governance added automatically")

    # Test with different providers (use whatever API keys are available)
    test_models = []

    # Add models based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        test_models.append(("gpt-3.5-turbo", "OpenAI"))
    if os.getenv("ANTHROPIC_API_KEY"):
        test_models.append(("claude-3-haiku", "Anthropic"))
    if os.getenv("GOOGLE_API_KEY"):
        test_models.append(("gemini-pro", "Google"))
    if os.getenv("COHERE_API_KEY"):
        test_models.append(("command-light", "Cohere"))

    if not test_models:
        # Fallback - try OpenAI with demo key (will fail but show instrumentation)
        test_models = [("gpt-3.5-turbo", "OpenAI")]
        print("⚠️  Using demo mode (API calls will fail but instrumentation will work)")

    for model, provider in test_models:
        print(f"\n🔄 Testing {provider} via LiteLLM ({model})...")

        try:
            start_time = time.time()

            # This is normal LiteLLM usage - GenOps instrumentation is automatic!
            response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the capital of France? (one word answer)",
                    }
                ],
                max_tokens=10,
                timeout=10,
            )

            end_time = time.time()

            # Extract response text
            if hasattr(response, "choices") and response.choices:
                result_text = response.choices[0].message.content.strip()
                print(f"✅ {provider} response: {result_text}")
                print(f"   Latency: {(end_time - start_time) * 1000:.0f}ms")

                # Show usage info if available
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    total_tokens = getattr(usage, "total_tokens", "unknown")
                    print(f"   Tokens: {total_tokens}")
            else:
                print(f"✅ {provider} request completed")

        except Exception:
            print(
                f"⚠️  {provider} request failed: [Error details redacted for security]"
            )
            print("   (This is normal if API key not configured)")

    print("\n📋 Step 3: View GenOps governance data")

    # Get usage statistics
    stats = get_usage_stats()

    print("\n📊 Usage Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total cost: ${stats['total_cost']:.6f}")

    if stats["provider_usage"]:
        print("   Provider breakdown:")
        for provider, data in stats["provider_usage"].items():
            print(
                f"     • {provider}: {data['requests']} requests, ${data['cost']:.6f}"
            )

    if stats["instrumentation_active"]:
        print(
            f"   ✅ Instrumentation active for: {stats['instrumentation_config']['team']}"
        )

    return True


def demo_multi_provider_usage():
    """Demonstrate multi-provider usage with unified governance."""
    print("\n" + "=" * 60)
    print("🌐 Demo: Multi-Provider Unified Governance")
    print("=" * 60)

    import litellm

    from genops.providers.litellm import get_cost_summary

    print("This demonstrates the key value of LiteLLM + GenOps:")
    print("• Single instrumentation layer")
    print("• Unified governance across ALL providers")
    print("• Provider-agnostic cost optimization")

    # Demonstrate model equivalents across providers
    model_equivalents = [
        ("gpt-3.5-turbo", "OpenAI - Fast, cost-effective"),
        ("claude-3-haiku", "Anthropic - Fast, thoughtful"),
        ("gemini-pro", "Google - Multimodal capable"),
        ("command-light", "Cohere - Enterprise focused"),
    ]

    print("\n🎯 Testing equivalent models across providers:")

    successful_requests = 0

    for model, description in model_equivalents:
        try:
            print(f"\n   • {model} ({description})")

            # Same request across different providers
            litellm.completion(
                model=model,
                messages=[
                    {"role": "user", "content": "Hello! Respond with just 'Hi there!'"}
                ],
                max_tokens=5,
                timeout=5,
            )

            successful_requests += 1
            print("     ✅ Success")

        except Exception:
            print("     ⚠️  Skipped (likely missing API key)")

    print("\n📊 Multi-Provider Summary:")
    cost_summary = get_cost_summary(group_by="provider")

    print(f"   Total cost: ${cost_summary['total_cost']:.6f}")

    if cost_summary.get("cost_by_provider"):
        print("   Cost by provider:")
        for provider, cost in cost_summary["cost_by_provider"].items():
            print(f"     • {provider}: ${cost:.6f}")

    print(
        f"\n🎉 Result: {successful_requests} providers tested through single GenOps integration!"
    )

    return True


def main():
    """Run the complete LiteLLM auto-instrumentation demonstration."""

    print("🌟 LiteLLM + GenOps: Highest-Leverage AI Governance Integration")
    print("=" * 70)
    print("Single instrumentation layer → Governance across 100+ LLM providers")
    print("Provider-agnostic cost tracking → Unified AI operations intelligence")

    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please resolve the issues above.")
        return 1

    try:
        # Run demonstrations
        print("\n🚀 Starting demonstrations...")

        success = demo_zero_code_instrumentation()
        if not success:
            print("❌ Auto-instrumentation demo failed")
            return 1

        demo_multi_provider_usage()

        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\n🚀 Key Takeaways:")
        print("   ✅ Single GenOps integration covers 100+ providers")
        print("   ✅ Zero-code instrumentation for existing apps")
        print("   ✅ Unified cost tracking and governance")
        print("   ✅ Provider-agnostic optimization opportunities")

        print("\n📖 Next Steps:")
        print("   • Explore multi_provider_costs.py for cost optimization")
        print("   • Try production_patterns.py for scaling strategies")
        print("   • Integrate into your existing LiteLLM applications!")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1

    except Exception:
        print("\n❌ Demo failed: [Error details redacted for security]")
        print("💡 For debugging, check your API key configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
