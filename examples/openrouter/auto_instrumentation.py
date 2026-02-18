#!/usr/bin/env python3
"""
OpenRouter Auto-Instrumentation Example

Demonstrates zero-code auto-instrumentation with OpenRouter.
Shows how existing OpenRouter code gets automatic governance telemetry.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python auto_instrumentation.py

Key features demonstrated:
- Zero-code auto-instrumentation setup
- Existing OpenRouter code works unchanged
- Automatic governance telemetry capture
- Global default governance attributes
"""

import os


def demonstrate_auto_instrumentation():
    """Show how auto-instrumentation works with existing OpenRouter code."""

    print("🎯 OpenRouter Auto-Instrumentation Demo")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set OPENROUTER_API_KEY environment variable.")
        return

    try:
        print("🔧 Step 1: Initialize GenOps auto-instrumentation")
        print("   Code: genops.init()")

        # Initialize GenOps auto-instrumentation - this is the ONLY change needed
        import genops

        genops.init(
            service_name="openrouter-demo",
            default_team="ai-platform-team",
            default_project="multi-provider-experiment",
            default_environment="development",
        )
        print("   ✅ Auto-instrumentation enabled!")

        print("\n📱 Step 2: Use existing OpenRouter code (unchanged!)")
        print("   Code: Standard OpenAI SDK with OpenRouter base URL")

        # This is standard OpenRouter code - no changes needed!
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            # Optional: Add OpenRouter-specific headers
            default_headers={
                "HTTP-Referer": "https://genops-demo.com",
                "X-Title": "GenOps Auto-Instrumentation Demo",
            },
        )

        print("   ✅ OpenRouter client created (standard code)")

        print("\n🚀 Step 3: Make requests - telemetry is automatic!")

        test_requests = [
            {
                "name": "Fast & Cheap: Llama 3.2 3B",
                "model": "meta-llama/llama-3.2-3b-instruct",
                "prompt": "What is the capital of France?",
            },
            {
                "name": "Balanced: GPT-4o",
                "model": "openai/gpt-4o",
                "prompt": "Explain quantum computing in simple terms.",
            },
            {
                "name": "Reasoning: Claude 3.5 Sonnet",
                "model": "anthropic/claude-3-5-sonnet",
                "prompt": "What are the ethical considerations of AI in healthcare?",
            },
        ]

        total_tokens = 0
        successful_requests = 0

        for i, request in enumerate(test_requests, 1):
            print(f"\n   {i}. {request['name']}")
            print(f"      Model: {request['model']}")
            print(f"      Prompt: {request['prompt']}")

            try:
                # Standard OpenAI SDK call - GenOps automatically captures telemetry
                response = client.chat.completions.create(
                    model=request["model"],
                    messages=[{"role": "user", "content": request["prompt"]}],
                    max_tokens=80,
                )

                # Extract response
                content = response.choices[0].message.content
                usage = response.usage

                print(
                    f"      ✅ Success! Tokens: {usage.total_tokens}, Cost tracked automatically"
                )
                print(
                    f"      Response: {content[:60]}{'...' if len(content) > 60 else ''}"
                )

                total_tokens += usage.total_tokens
                successful_requests += 1

            except Exception as e:
                print(f"      ❌ Error: {str(e)}")

        print("\n" + "=" * 50)
        print("📊 Auto-Instrumentation Results")
        print("=" * 50)
        print(f"✅ Successful Requests: {successful_requests}/{len(test_requests)}")
        print(f"📊 Total Tokens Used: {total_tokens}")
        print("🎯 Zero Code Changes Required!")

        print("\n🔍 What GenOps Captured Automatically:")
        print("   • Request/response for each model")
        print("   • Token usage and cost calculations")
        print("   • Provider routing decisions (OpenAI vs Anthropic vs Meta)")
        print("   • Governance attributes (team, project, environment)")
        print("   • OpenTelemetry traces for observability integration")
        print("   • Multi-provider cost attribution")

        print("\n📈 Telemetry Attributes Added:")
        print("   • genops.service.name: openrouter-demo")
        print("   • genops.team: ai-platform-team")
        print("   • genops.project: multi-provider-experiment")
        print("   • genops.environment: development")
        print("   • genops.provider: openrouter")
        print("   • genops.openrouter.actual_provider: [varies by model]")
        print("   • genops.cost.total: [calculated per request]")

        print("\n🔄 How It Works:")
        print("   1. genops.init() patches the OpenAI client globally")
        print("   2. When base_url contains 'openrouter.ai', GenOps intercepts")
        print("   3. Requests flow through GenOps telemetry layer")
        print("   4. Original response returned unchanged")
        print("   5. Telemetry exported to configured observability backend")

        print("\n✨ Benefits:")
        print("   • No code changes to existing OpenRouter applications")
        print("   • Automatic cost tracking across 400+ models")
        print("   • Unified governance across all AI providers")
        print("   • Drop-in observability for existing systems")
        print("   • Multi-provider cost attribution and budgeting")

        print("\n🚀 Next Steps:")
        print(
            "   • Add per-request governance: client.chat.completions.create(..., team='new-team')"
        )
        print("   • Set up budget alerts in your observability dashboard")
        print("   • Try production_patterns.py for deployment best practices")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install required packages: pip install genops-ai openai")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your API key and network connection")


def show_comparison():
    """Show before/after code comparison."""
    print("\n📋 Code Comparison: Before vs After GenOps")
    print("=" * 50)

    print("❌ BEFORE (No governance):")
    print("""
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello"}]
)
# No cost tracking, no governance, no observability
""")

    print("✅ AFTER (With GenOps):")
    print("""
import genops
genops.init()  # <-- Only addition needed!

from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello"}]
)
# Automatic cost tracking, governance attributes, full observability!
""")

    print("🎯 Result: 1 line addition = Complete AI governance")


if __name__ == "__main__":
    demonstrate_auto_instrumentation()
    show_comparison()
