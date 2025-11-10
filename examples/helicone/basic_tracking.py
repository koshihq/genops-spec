#!/usr/bin/env python3
"""
Helicone Basic Multi-Provider Tracking Example

This example demonstrates basic GenOps tracking with Helicone AI Gateway
across multiple providers. Perfect for understanding the fundamentals of
multi-provider AI operations with unified cost tracking.

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    export OPENAI_API_KEY="your_openai_api_key"
    export ANTHROPIC_API_KEY="your_anthropic_api_key"  # Optional
"""

import os
import sys
from datetime import datetime


def basic_multi_provider_example():
    """Demonstrate basic multi-provider tracking through Helicone gateway."""

    print("🚀 GenOps + Helicone: Basic Multi-Provider Tracking")
    print("=" * 55)

    # Step 1: Import and initialize GenOps Helicone adapter
    try:
        from genops.providers.helicone import instrument_helicone
        print("✅ GenOps Helicone provider imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Fix: pip install genops[helicone]")
        return False

    # Step 2: Set up the adapter with your API keys
    try:
        adapter = instrument_helicone(
            helicone_api_key=os.getenv('HELICONE_API_KEY'),
            provider_keys={
                'openai': os.getenv('OPENAI_API_KEY'),
                'anthropic': os.getenv('ANTHROPIC_API_KEY'),
                'groq': os.getenv('GROQ_API_KEY')
            },
            # Governance attributes for cost attribution
            team="engineering-team",
            project="helicone-basic-example",
            environment="development"
        )
        print("✅ Helicone gateway adapter initialized")
        print(f"   📊 Providers configured: {len(adapter.provider_keys)}")
    except Exception as e:
        print(f"❌ Adapter initialization failed: {e}")
        print("💡 Check your API keys and try again")
        return False

    # Step 3: Make requests to different providers through unified interface
    examples = [
        {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'message': 'Explain artificial intelligence in one sentence.',
            'description': 'OpenAI GPT-3.5 Turbo (Fast, cost-effective)'
        }
    ]

    # Add Anthropic if available
    if os.getenv('ANTHROPIC_API_KEY'):
        examples.append({
            'provider': 'anthropic',
            'model': 'claude-3-haiku-20240307',
            'message': 'What are the benefits of AI gateways?',
            'description': 'Anthropic Claude 3 Haiku (Fast, reasoning-focused)'
        })

    # Add Groq if available
    if os.getenv('GROQ_API_KEY'):
        examples.append({
            'provider': 'groq',
            'model': 'mixtral-8x7b-32768',
            'message': 'How do AI gateways help with cost optimization?',
            'description': 'Groq Mixtral (Ultra-fast, cost-efficient)'
        })

    print(f"\n🎯 Running {len(examples)} multi-provider examples...")
    print("-" * 55)

    total_cost = 0.0
    results = []

    for i, example in enumerate(examples, 1):
        print(f"\n📋 Example {i}: {example['description']}")
        print(f"   Provider: {example['provider']}")
        print(f"   Model: {example['model']}")
        print(f"   Query: {example['message']}")

        try:
            # Make the request through Helicone gateway
            response = adapter.chat(
                message=example['message'],
                provider=example['provider'],
                model=example['model'],

                # Additional governance attributes
                customer_id="demo-customer",
                cost_center="engineering",
                feature="basic-tracking-demo"
            )

            # Extract and display results
            content = response.content if hasattr(response, 'content') else str(response)
            cost = response.usage.total_cost if hasattr(response, 'usage') else 0.0
            provider_cost = response.usage.provider_cost if hasattr(response, 'usage') else 0.0
            gateway_cost = response.usage.helicone_cost if hasattr(response, 'usage') else 0.0

            total_cost += cost
            results.append({
                'provider': example['provider'],
                'model': example['model'],
                'cost': cost,
                'provider_cost': provider_cost,
                'gateway_cost': gateway_cost,
                'content': content[:150] + '...' if len(content) > 150 else content
            })

            print(f"   ✅ Response: {content[:100]}...")
            print(f"   💰 Provider cost: ${provider_cost:.6f}")
            print(f"   🌐 Gateway cost: ${gateway_cost:.6f}")
            print(f"   📊 Total cost: ${cost:.6f}")

        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            continue

    # Step 4: Display comprehensive results
    print("\n" + "=" * 55)
    print("📊 MULTI-PROVIDER SESSION SUMMARY")
    print("=" * 55)

    if results:
        print(f"✅ Successful requests: {len(results)}")
        print(f"💰 Total session cost: ${total_cost:.6f}")
        print()

        # Cost breakdown by provider
        print("💸 Cost Breakdown by Provider:")
        for result in results:
            print(f"   • {result['provider'].title():>10}: ${result['cost']:.6f} "
                  f"(Provider: ${result['provider_cost']:.6f}, "
                  f"Gateway: ${result['gateway_cost']:.6f})")

        # Provider comparison
        if len(results) > 1:
            print("\n📈 Provider Comparison:")
            cheapest = min(results, key=lambda x: x['cost'])
            most_expensive = max(results, key=lambda x: x['cost'])

            print(f"   🥇 Most cost-effective: {cheapest['provider']} (${cheapest['cost']:.6f})")
            print(f"   💎 Most expensive: {most_expensive['provider']} (${most_expensive['cost']:.6f})")

            if most_expensive['cost'] > cheapest['cost']:
                savings = most_expensive['cost'] - cheapest['cost']
                print(f"   💡 Potential savings: ${savings:.6f} per request")

    else:
        print("❌ No successful requests completed")
        return False

    # Step 5: Show what GenOps tracked automatically
    print("\n🔍 AUTOMATIC GENOPS TRACKING")
    print("-" * 30)
    print("✅ Multi-provider cost attribution")
    print("✅ Gateway fee analysis")
    print("✅ Team and project cost tracking")
    print("✅ Customer billing attribution")
    print("✅ Environment segregation")
    print("✅ OpenTelemetry trace export")
    print("✅ Real-time cost aggregation")

    print("\n💡 WHAT YOU'VE LEARNED")
    print("-" * 25)
    print("• How to access multiple AI providers through single interface")
    print("• Unified cost tracking across all providers and gateway fees")
    print("• Governance attribute propagation for cost attribution")
    print("• Provider cost comparison for optimization insights")
    print("• Zero-code integration with existing AI workflows")

    return True


def main():
    """Main function to run the basic tracking example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    required_env_vars = ['HELICONE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n💡 Set them with:")
        print("   export HELICONE_API_KEY='your_helicone_key'")
        print("   export OPENAI_API_KEY='your_openai_key'")
        return False

    # Run the example
    success = basic_multi_provider_example()

    if success:
        print("\n🎉 SUCCESS! Basic multi-provider tracking completed.")
        print("\n📚 Next Steps:")
        print("   • Try 'python multi_provider_costs.py' for cost comparison")
        print("   • Try 'python cost_optimization.py' for intelligent routing")
        print("   • Try 'python advanced_features.py' for streaming & advanced features")
    else:
        print("\n❌ Example failed. Check the errors above.")

    return success


if __name__ == "__main__":
    """Entry point for the script."""
    success = main()
    sys.exit(0 if success else 1)
