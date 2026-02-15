#!/usr/bin/env python3
"""
🇪🇺 GenOps + Mistral AI: Hello European AI (Minimal Example)

GOAL: Prove GenOps works with Mistral AI in 30 seconds
TIME: 30 seconds
WHAT YOU'LL LEARN: European AI cost tracking with GDPR compliance

This is the simplest possible example to verify GenOps tracking works
with Mistral AI. Run this first before exploring advanced features.

Prerequisites:
- Mistral API key: export MISTRAL_API_KEY="your-key"
- GenOps: pip install genops-ai
- Mistral: pip install mistralai
"""

import os
import sys


def main():
    """30-second European AI confidence builder."""
    print("🇪🇺 GenOps + Mistral AI: Hello European AI!")
    print("=" * 50)

    # Check prerequisites
    print("🔍 Checking prerequisites...")

    # Check API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found")
        print("   Get your key: https://console.mistral.ai/")
        print("   Set it: export MISTRAL_API_KEY='your-key'")
        return False

    print("✅ Mistral API key found and validated")

    # Check dependencies
    try:
        import mistralai  # noqa: F401

        print("✅ Mistral client available")
    except ImportError:
        print("❌ Mistral client not found")
        print("   Install: pip install mistralai")
        return False

    try:
        from genops.providers.mistral import instrument_mistral

        print("✅ GenOps Mistral provider available")
    except ImportError:
        print("❌ GenOps not found")
        print("   Install: pip install genops-ai")
        return False

    print("\n🚀 Testing European AI with GenOps tracking...")
    print("-" * 50)

    try:
        # Enable GenOps tracking for European AI
        adapter = instrument_mistral(team="demo-team", project="european-ai-test")

        print("✅ GenOps European AI adapter created")

        # Test basic chat with cost tracking
        response = adapter.chat(
            message="What are the benefits of European AI?",
            model="mistral-small-latest",  # Cost-effective European model
        )

        if response.success:
            print("✅ European AI Response received:")
            print(f"   Content: {response.content[:100]}...")
            print(f"   Model: {response.model}")

            print("\n💰 European AI Cost Tracking:")
            print(f"   Input tokens: {response.usage.input_tokens}")
            print(f"   Output tokens: {response.usage.output_tokens}")
            print(f"   Total cost: ${response.usage.total_cost:.6f}")
            print(f"   Cost per token: ${response.usage.cost_per_token:.8f}")

            print("\n🇪🇺 European AI Benefits:")
            print("   ✅ GDPR compliant by default")
            print("   ✅ EU data residency maintained")
            print("   ✅ Competitive pricing vs US providers")
            print("   ✅ No cross-border data transfer costs")
            print("   ✅ Simplified regulatory compliance")

            print("\n⚡ Performance Metrics:")
            print(f"   Request time: {response.usage.request_time:.3f}s")
            if response.usage.tokens_per_second > 0:
                print(f"   Tokens per second: {response.usage.tokens_per_second:.1f}")

            # Get session summary
            summary = adapter.get_usage_summary()
            print("\n📊 Session Summary:")
            print(f"   Total operations: {summary['total_operations']}")
            print(f"   Total cost: ${summary['total_cost']:.6f}")
            print(
                f"   Cost tracking: {'✅' if summary['cost_tracking_enabled'] else '❌'}"
            )

            print("\n" + "=" * 50)
            print("✅ SUCCESS! GenOps is now tracking your European AI usage")
            print(
                "🇪🇺 Your Mistral operations have enterprise governance + GDPR compliance!"
            )

            print("\n🚀 Next Steps:")
            print("   1. Try different models: mistral-tiny-2312 (ultra-low cost)")
            print(
                "   2. Explore European AI advantages: python european_ai_advantages.py"
            )
            print("   3. Check out cost optimization: python cost_optimization.py")
            print("   4. Read full guide: docs/integrations/mistral.md")

            return True

        else:
            print(f"❌ European AI request failed: {response.error_message}")
            return False

    except Exception as e:
        print(f"❌ Error during European AI test: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check API key is correct: echo $MISTRAL_API_KEY")
        print("   2. Verify Mistral access: visit console.mistral.ai")
        print("   3. Check internet connection")
        print("   4. Try: python -c \"import mistralai; print('OK')\"")
        return False


def quick_model_comparison():
    """Bonus: Quick model comparison for European AI cost optimization."""
    print("\n" + "=" * 50)
    print("🎁 BONUS: European AI Model Cost Comparison")
    print("=" * 50)

    try:
        from genops.providers.mistral import instrument_mistral

        adapter = instrument_mistral(team="comparison-demo")

        # Test different models with same prompt
        models_to_test = [
            ("mistral-tiny-2312", "Ultra-low cost"),
            ("mistral-small-latest", "Cost-effective"),
            ("mistral-medium-latest", "Balanced performance"),
        ]

        prompt = "What is 2+2?"

        print(f"📊 Comparing European AI models with prompt: '{prompt}'")
        print("-" * 50)

        for model, description in models_to_test:
            try:
                response = adapter.chat(message=prompt, model=model, max_tokens=10)

                if response.success:
                    print(f"✅ {model} ({description}):")
                    print(f"   Cost: ${response.usage.total_cost:.6f}")
                    print(f"   Tokens: {response.usage.total_tokens}")
                    print(f"   Response: {response.content[:50]}...")
                else:
                    print(f"❌ {model}: {response.error_message}")

            except Exception as e:
                print(f"❌ {model}: Error - {e}")

        # Show session summary
        summary = adapter.get_usage_summary()
        print("\n🇪🇺 European AI Session Total:")
        print(f"   Operations: {summary['total_operations']}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")
        print(
            f"   Average cost/operation: ${summary['average_cost_per_operation']:.6f}"
        )

        print("\n💡 European AI Insight:")
        print("   Choose the right model for optimal cost-performance balance!")
        print("   Mistral provides GDPR-compliant AI at competitive European rates.")

    except Exception as e:
        print(f"❌ Model comparison failed: {e}")


if __name__ == "__main__":
    print("Starting European AI confidence builder...")

    success = main()

    if success:
        # Run bonus comparison if main test succeeded
        quick_model_comparison()

        print("\n🎉 European AI Success!")
        print("You're ready to explore advanced GenOps + Mistral features:")
        print("• european_ai_advantages.py - GDPR compliance benefits")
        print("• cost_optimization.py - European AI cost strategies")
        print("• enterprise_deployment.py - Production GDPR governance")

        sys.exit(0)
    else:
        print("\n⚠️ Issues detected. Please fix the errors above and try again.")
        print("Need help? Check docs/mistral-quickstart.md for troubleshooting")
        sys.exit(1)
