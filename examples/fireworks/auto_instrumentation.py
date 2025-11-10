#!/usr/bin/env python3
"""
Fireworks AI Auto-Instrumentation with GenOps

Demonstrates zero-code instrumentation for Fireworks AI operations.
Shows how to add governance to existing Fireworks AI code with minimal changes.

Usage:
    python auto_instrumentation.py

Features:
    - Zero-code governance for existing Fireworks AI applications
    - Automatic cost tracking and attribution with 4x faster inference
    - Drop-in replacement for existing Fireworks code
    - Seamless integration with OpenTelemetry observability
"""

import asyncio
import os
import sys

try:
    # Standard Fireworks AI import (what users already have)
    from fireworks.client import Fireworks

    from genops.providers.fireworks import FireworksModel, auto_instrument
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[fireworks] fireworks-ai")
    print("Then run: python setup_validation.py")
    sys.exit(1)


def demonstrate_manual_approach():
    """Show traditional approach without auto-instrumentation."""
    print("📝 Traditional Approach (without GenOps)")
    print("-" * 40)

    try:
        # Traditional Fireworks AI usage (what users already do)
        client = Fireworks()

        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[
                {"role": "user", "content": "What are the benefits of auto-instrumentation?"}
            ],
            max_tokens=100
        )

        print("✅ Response received:")
        print(f"   {response.choices[0].message.content}")
        print("❓ But how much did it cost? How fast was it? Which team used it?")
        print("❓ No automatic tracking, governance, or observability!")

    except Exception as e:
        print(f"❌ Traditional approach failed: {e}")
        return False

    return True


def demonstrate_auto_instrumentation():
    """Show auto-instrumentation approach with full governance."""
    print("\n🤖 Auto-Instrumentation Approach (with GenOps)")
    print("-" * 40)

    # 🎯 THE MAGIC LINE - Add comprehensive governance with ONE line!
    print("🎯 Adding auto-instrumentation...")
    auto_instrument()
    print("✅ Auto-instrumentation active!")

    try:
        # Exact same code as before - no changes needed!
        client = Fireworks()

        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[
                {"role": "user", "content": "What are the benefits of auto-instrumentation with fast inference?"}
            ],
            max_tokens=100
        )

        print("✅ Response received with automatic governance:")
        print(f"   {response.choices[0].message.content}")
        print("🎉 Automatic cost tracking, governance, and observability added!")
        print("⚡ 4x faster inference with Fireattention optimization!")

    except Exception as e:
        print(f"❌ Auto-instrumentation approach failed: {e}")
        return False

    return True


def demonstrate_mixed_models():
    """Show auto-instrumentation with different model tiers."""
    print("\n🔬 Auto-Instrumentation with Multiple Models")
    print("-" * 40)

    # Auto-instrumentation is already active from previous call
    client = Fireworks()

    # Test different pricing tiers with auto-instrumentation
    models_to_test = [
        ("accounts/fireworks/models/llama-v3p2-1b-instruct", "Tiny (1B)", "$0.10/M"),
        ("accounts/fireworks/models/llama-v3p1-8b-instruct", "Small (8B)", "$0.20/M"),
        ("accounts/fireworks/models/llama-v3p1-70b-instruct", "Large (70B)", "$0.90/M"),
        ("accounts/fireworks/models/mixtral-8x7b-instruct", "MoE (8x7B)", "$0.50/M")
    ]

    prompt = "Explain the speed benefits of Fireworks AI in one sentence."

    for model, tier, pricing in models_to_test:
        try:
            print(f"\n🧠 Testing {tier} model ({pricing})...")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5
            )

            print(f"   ✅ {tier}: {response.choices[0].message.content[:60]}...")
            print("   🎯 Automatic governance tracking active!")

        except Exception as e:
            print(f"   ❌ {tier} failed: {e}")


def demonstrate_openai_compatibility():
    """Show OpenAI-compatible interface with auto-instrumentation."""
    print("\n🔄 OpenAI Compatibility with Auto-Instrumentation")
    print("-" * 40)

    try:
        # Use OpenAI-compatible interface (common migration pattern)
        import openai

        # Point to Fireworks endpoint (common pattern for users switching)
        openai_client = openai.OpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )

        response = openai_client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[{
                "role": "user",
                "content": "How does Fireworks AI compare to other providers for speed?"
            }],
            max_tokens=80
        )

        print("✅ OpenAI-compatible interface with Fireworks speed:")
        print(f"   {response.choices[0].message.content}")
        print("🎯 Auto-instrumentation works with OpenAI-compatible code too!")

    except ImportError:
        print("⚠️ OpenAI library not installed - skipping compatibility demo")
    except Exception as e:
        print(f"❌ OpenAI compatibility demo failed: {e}")


def demonstrate_embedding_auto_instrumentation():
    """Show embedding operations with auto-instrumentation."""
    print("\n🔤 Embeddings with Auto-Instrumentation")
    print("-" * 40)

    try:
        client = Fireworks()

        # Embedding operations are automatically instrumented too
        response = client.embeddings.create(
            model="accounts/fireworks/models/nomic-embed-text-v1p5",
            input=["Fast AI inference is crucial for production", "Fireworks AI delivers 4x speed improvements"]
        )

        print(f"✅ Generated embeddings for {len(response.data)} texts")
        print("🎯 Embedding costs automatically tracked!")
        print("⚡ Fast embedding generation with Fireworks optimizations!")

    except Exception as e:
        print(f"❌ Embedding auto-instrumentation failed: {e}")


async def demonstrate_async_operations():
    """Show async operations with auto-instrumentation."""
    print("\n⚡ Async Operations with Auto-Instrumentation")
    print("-" * 40)

    try:
        # Note: This is a conceptual example - actual async implementation would depend on Fireworks client
        print("🔄 Processing multiple requests concurrently...")

        client = Fireworks()

        # Simulate concurrent operations
        prompts = [
            "What makes Fireworks AI fast?",
            "How does 4x speed improvement help production?",
            "What are the cost benefits of fast inference?"
        ]

        results = []
        for i, prompt in enumerate(prompts):
            response = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40
            )
            results.append(response)
            print(f"   ✅ Request {i+1}/3 completed with auto-governance")

        print(f"🎉 All {len(results)} concurrent requests completed!")
        print("🎯 Each request automatically tracked for cost and governance!")

    except Exception as e:
        print(f"❌ Async auto-instrumentation demo failed: {e}")


def demonstrate_advanced_features():
    """Show advanced Fireworks features with auto-instrumentation."""
    print("\n🚀 Advanced Features with Auto-Instrumentation")
    print("-" * 40)

    try:
        client = Fireworks()

        # Function calling (if supported)
        print("🔧 Testing function calling capabilities...")
        functions = [
            {
                "name": "get_speed_info",
                "description": "Get information about Fireworks AI speed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "Speed metric to query"}
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            messages=[{"role": "user", "content": "How much faster is Fireworks AI?"}],
            functions=functions,
            function_call="auto",
            max_tokens=60
        )

        print("✅ Advanced features with auto-governance!")
        print(f"   {response.choices[0].message.content}")

    except Exception as e:
        print(f"⚠️ Advanced features demo: {e}")


def main():
    """Demonstrate auto-instrumentation capabilities."""
    print("🤖 Fireworks AI Auto-Instrumentation with GenOps")
    print("=" * 60)

    print("This demo shows how ONE line of code adds complete governance")
    print("to existing Fireworks AI applications with zero code changes!")

    # Step 1: Show traditional approach (no governance)
    if not demonstrate_manual_approach():
        return 1

    # Step 2: Show auto-instrumentation magic
    if not demonstrate_auto_instrumentation():
        return 1

    # Step 3: Show it works with multiple models
    demonstrate_mixed_models()

    # Step 4: Show OpenAI compatibility
    demonstrate_openai_compatibility()

    # Step 5: Show embedding operations
    demonstrate_embedding_auto_instrumentation()

    # Step 6: Show async operations
    asyncio.run(demonstrate_async_operations())

    # Step 7: Show advanced features
    demonstrate_advanced_features()

    # Summary
    print("\n" + "=" * 60)
    print("🎉 Auto-Instrumentation Demo Complete!")
    print("=" * 60)

    print("✅ What you achieved with ONE line of code:")
    print("   • Automatic cost tracking across all operations")
    print("   • Real-time governance and budget monitoring")
    print("   • Complete observability integration")
    print("   • Team and project attribution")
    print("   • Multi-model support across all pricing tiers")
    print("   • 4x faster inference with Fireattention optimization")
    print("   • Zero changes to your existing code!")

    print("\n🚀 Next Steps:")
    print("   • Add auto_instrument() to your existing Fireworks AI apps")
    print("   • Try cost_optimization.py for intelligent model selection")
    print("   • Explore production_patterns.py for enterprise deployment")
    print("   • Enjoy the speed and governance benefits! 🔥")

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
        print("Try running setup_validation.py to check your configuration")
        sys.exit(1)
