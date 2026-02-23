#!/usr/bin/env python3
"""
Basic OpenRouter Cost Tracking Example

Demonstrates simple cost and usage tracking with OpenRouter using GenOps.
Shows how to track costs across multiple models and providers with governance attributes.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python basic_tracking.py

Key features demonstrated:
- Basic cost tracking across multiple OpenRouter models
- Governance attributes for team/project attribution
- Multi-provider cost visibility
- Usage metrics and token counting
"""

import os


def basic_tracking_example():
    """Demonstrate basic OpenRouter cost tracking with GenOps."""

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set OPENROUTER_API_KEY environment variable.")
        print("Get your key from: https://openrouter.ai/keys")
        return

    print("🚀 Basic OpenRouter + GenOps Cost Tracking")
    print("=" * 50)

    try:
        # Import GenOps OpenRouter integration
        from genops.providers.openrouter import instrument_openrouter

        # Create instrumented OpenRouter client
        print("📡 Creating instrumented OpenRouter client...")
        client = instrument_openrouter(openrouter_api_key=api_key)
        print("   ✅ Client created successfully")

        # Test different models with governance attributes
        test_scenarios = [
            {
                "name": "💬 Anthropic Claude 3.5 Sonnet (High-end reasoning)",
                "model": "anthropic/claude-3-5-sonnet",
                "message": "Explain the benefits of renewable energy in 2 sentences.",
                "governance": {
                    "team": "sustainability-team",
                    "project": "green-energy-chatbot",
                    "customer_id": "demo-customer-001",
                },
            },
            {
                "name": "⚡ Meta Llama 3.2 3B (Fast, cost-effective)",
                "model": "meta-llama/llama-3.2-3b-instruct",
                "message": "What is machine learning?",
                "governance": {
                    "team": "ml-team",
                    "project": "educational-content",
                    "environment": "development",
                },
            },
            {
                "name": "🧠 OpenAI GPT-4o (Balanced performance)",
                "model": "openai/gpt-4o",
                "message": "Summarize the key principles of software architecture.",
                "governance": {
                    "team": "engineering-team",
                    "project": "code-assistant",
                    "cost_center": "R&D",
                },
            },
        ]

        total_cost = 0.0
        results = []

        print("\n🔄 Running test scenarios...")

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Model: {scenario['model']}")
            print(f"   Query: {scenario['message']}")

            try:
                # Make request with governance attributes
                response = client.chat_completions_create(
                    model=scenario["model"],
                    messages=[{"role": "user", "content": scenario["message"]}],
                    max_tokens=100,  # Keep costs low for demo
                    **scenario["governance"],  # Add governance attributes
                )

                # Extract response details
                content = response.choices[0].message.content
                usage = response.usage if hasattr(response, "usage") else None

                # Calculate cost (GenOps automatically tracks this)
                if usage:
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # Get cost estimate from GenOps pricing engine
                    from genops.providers.openrouter_pricing import (
                        calculate_openrouter_cost,
                    )

                    estimated_cost = calculate_openrouter_cost(
                        scenario["model"],
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                    print("   ✅ Success!")
                    print(
                        f"      Tokens: {input_tokens} in, {output_tokens} out ({total_tokens} total)"
                    )
                    print(f"      Est. Cost: ${estimated_cost:.6f}")
                    print(
                        f"      Response: {content[:100]}{'...' if len(content) > 100 else ''}"
                    )

                    total_cost += estimated_cost
                    results.append(
                        {
                            "model": scenario["model"],
                            "cost": estimated_cost,
                            "tokens": total_tokens,
                            "governance": scenario["governance"],
                        }
                    )
                else:
                    print("   ⚠️  No usage data available")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                continue

        # Display summary
        print("\n" + "=" * 50)
        print("📊 Cost Tracking Summary")
        print("=" * 50)

        if results:
            print(f"💰 Total Estimated Cost: ${total_cost:.6f}")
            print(f"📈 Models Tested: {len(results)}")

            print("\n📋 Breakdown by Model:")
            for result in results:
                print(
                    f"   • {result['model']}: ${result['cost']:.6f} ({result['tokens']} tokens)"
                )

            print("\n🏷️  Governance Attribution:")
            teams = {r["governance"].get("team", "unknown") for r in results}
            projects = {r["governance"].get("project", "unknown") for r in results}
            print(f"   • Teams: {', '.join(teams)}")
            print(f"   • Projects: {', '.join(projects)}")
        else:
            print("❌ No successful requests completed")

        print("\n🔍 Telemetry Notes:")
        print("   • All requests automatically tracked in OpenTelemetry traces")
        print("   • Governance attributes propagated to observability backend")
        print("   • Cost data available for dashboards and alerting")
        print("   • Multi-provider routing decisions captured")

        print("\n✨ Next Steps:")
        print("   • Check your observability dashboard for detailed traces")
        print("   • Set up budget alerts based on team/project attribution")
        print("   • Try advanced_features.py for routing control")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install required packages: pip install genops-ai openai")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your API key and network connection")


if __name__ == "__main__":
    basic_tracking_example()
