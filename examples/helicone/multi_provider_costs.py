#!/usr/bin/env python3
"""
Helicone Multi-Provider Cost Comparison Example

This example demonstrates comprehensive cost comparison and analysis across
multiple AI providers using Helicone AI Gateway. Perfect for understanding
cost optimization opportunities and making data-driven provider decisions.

Usage:
    python multi_provider_costs.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    # At least 2 provider keys for meaningful comparison:
    export OPENAI_API_KEY="your_openai_api_key"
    export ANTHROPIC_API_KEY="your_anthropic_api_key"
    export GROQ_API_KEY="your_groq_api_key"  # Optional (free tier)
"""

import os
import sys
import time
from datetime import datetime


def run_cost_comparison_analysis():
    """Run comprehensive cost comparison across multiple providers."""

    print("💰 GenOps + Helicone: Multi-Provider Cost Analysis")
    print("=" * 58)

    # Initialize the adapter
    try:
        from genops.providers.helicone import instrument_helicone
        from genops.providers.helicone_cost_aggregator import create_cost_aggregator

        adapter = instrument_helicone(
            helicone_api_key=os.getenv("HELICONE_API_KEY"),
            provider_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                "groq": os.getenv("GROQ_API_KEY"),
            },
            team="cost-analysis-team",
            project="provider-comparison",
            environment="analysis",
        )
        print("✅ Multi-provider adapter initialized")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Fix: pip install genops[helicone]")
        return False
    except Exception as e:
        print(f"❌ Adapter initialization failed: {e}")
        return False

    # Define test scenarios for comparison
    test_scenarios = [
        {
            "name": "Simple Q&A",
            "complexity": "Low",
            "message": "What is machine learning?",
            "expected_tokens": 50,
        },
        {
            "name": "Technical Explanation",
            "complexity": "Medium",
            "message": "Explain the transformer architecture in neural networks.",
            "expected_tokens": 150,
        },
        {
            "name": "Complex Analysis",
            "complexity": "High",
            "message": "Compare and contrast different approaches to AI safety, including alignment research, interpretability, and robustness testing.",
            "expected_tokens": 300,
        },
    ]

    # Provider configurations for testing
    provider_configs = []
    if os.getenv("OPENAI_API_KEY"):
        provider_configs.extend(
            [
                {"provider": "openai", "model": "gpt-3.5-turbo", "tier": "Standard"},
                {"provider": "openai", "model": "gpt-4", "tier": "Premium"},
            ]
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        provider_configs.extend(
            [
                {
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "tier": "Fast",
                },
                {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet-20240229",
                    "tier": "Balanced",
                },
            ]
        )

    if os.getenv("GROQ_API_KEY"):
        provider_configs.append(
            {"provider": "groq", "model": "mixtral-8x7b-32768", "tier": "Ultra-Fast"}
        )

    if len(provider_configs) < 2:
        print("❌ Need at least 2 providers for meaningful comparison")
        print("💡 Set additional API keys: ANTHROPIC_API_KEY, GROQ_API_KEY")
        return False

    print(
        f"🎯 Testing {len(test_scenarios)} scenarios across {len(provider_configs)} provider/model combinations"
    )

    # Run cost analysis with tracking
    all_results = []

    # Create cost aggregator for session tracking
    aggregator = create_cost_aggregator("cost-comparison-session")

    try:
        print("\n📊 COST COMPARISON ANALYSIS")
        print("=" * 40)

        for scenario_idx, scenario in enumerate(test_scenarios, 1):
            print(
                f"\n🔍 Scenario {scenario_idx}: {scenario['name']} ({scenario['complexity']} Complexity)"
            )
            print("-" * 60)

            scenario_results = []

            for config in provider_configs:
                provider = config["provider"]
                model = config["model"]
                tier = config["tier"]

                print(f"   Testing: {provider.title()} {model} ({tier})")

                try:
                    start_time = time.time()

                    response = adapter.chat(
                        message=scenario["message"],
                        provider=provider,
                        model=model,
                        customer_id=f"scenario-{scenario_idx}",
                        feature=f"cost-analysis-{scenario['complexity'].lower()}",
                    )

                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds

                    # Extract cost information
                    usage = response.usage if hasattr(response, "usage") else None
                    if usage:
                        provider_cost = getattr(usage, "provider_cost", 0.0)
                        gateway_cost = getattr(usage, "helicone_cost", 0.0)
                        total_cost = getattr(
                            usage, "total_cost", provider_cost + gateway_cost
                        )
                        input_tokens = getattr(usage, "input_tokens", 0)
                        output_tokens = getattr(usage, "output_tokens", 0)
                    else:
                        provider_cost = gateway_cost = total_cost = 0.0
                        input_tokens = output_tokens = 0

                    result = {
                        "scenario": scenario["name"],
                        "complexity": scenario["complexity"],
                        "provider": provider,
                        "model": model,
                        "tier": tier,
                        "provider_cost": provider_cost,
                        "gateway_cost": gateway_cost,
                        "total_cost": total_cost,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": latency,
                        "cost_per_token": total_cost
                        / max(input_tokens + output_tokens, 1),
                        "response_length": len(response.content)
                        if hasattr(response, "content")
                        else 0,
                    }

                    scenario_results.append(result)
                    all_results.append(result)

                    print(
                        f"      💰 Cost: ${total_cost:.6f} (Provider: ${provider_cost:.6f}, Gateway: ${gateway_cost:.6f})"
                    )
                    print(f"      ⚡ Latency: {latency:.0f}ms")
                    print(f"      📊 Tokens: {input_tokens} in, {output_tokens} out")

                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    continue

            # Scenario summary
            if scenario_results:
                cheapest = min(scenario_results, key=lambda x: x["total_cost"])
                fastest = min(scenario_results, key=lambda x: x["latency_ms"])

                print(
                    f"\n   🥇 Cheapest: {cheapest['provider']} {cheapest['model']} (${cheapest['total_cost']:.6f})"
                )
                print(
                    f"   ⚡ Fastest: {fastest['provider']} {fastest['model']} ({fastest['latency_ms']:.0f}ms)"
                )

    finally:
        # Finalize aggregator
        aggregator.finalize()

    # Comprehensive analysis
    if all_results:
        print("\n" + "=" * 60)
        print("📈 COMPREHENSIVE COST ANALYSIS")
        print("=" * 60)

        # Overall statistics
        total_requests = len(all_results)
        total_cost = sum(r["total_cost"] for r in all_results)
        avg_cost = total_cost / total_requests

        print("📊 Overall Statistics:")
        print(f"   • Total requests: {total_requests}")
        print(f"   • Total cost: ${total_cost:.6f}")
        print(f"   • Average cost per request: ${avg_cost:.6f}")

        # Provider comparison
        print("\n💸 Cost by Provider:")
        provider_costs = {}
        for result in all_results:
            provider = result["provider"]
            if provider not in provider_costs:
                provider_costs[provider] = {"total": 0.0, "count": 0, "models": set()}
            provider_costs[provider]["total"] += result["total_cost"]
            provider_costs[provider]["count"] += 1
            provider_costs[provider]["models"].add(result["model"])

        for provider, data in provider_costs.items():
            avg_cost = data["total"] / data["count"]
            print(
                f"   • {provider.title():>10}: ${data['total']:.6f} total, ${avg_cost:.6f} avg ({data['count']} requests)"
            )

        # Complexity analysis
        print("\n🎯 Cost by Complexity:")
        complexity_costs = {}
        for result in all_results:
            complexity = result["complexity"]
            if complexity not in complexity_costs:
                complexity_costs[complexity] = {"total": 0.0, "count": 0}
            complexity_costs[complexity]["total"] += result["total_cost"]
            complexity_costs[complexity]["count"] += 1

        for complexity, data in complexity_costs.items():
            avg_cost = data["total"] / data["count"]
            print(f"   • {complexity:>6} complexity: ${avg_cost:.6f} average cost")

        # Best value analysis
        print("\n🏆 Best Value Analysis:")

        # Best overall value (lowest cost)
        cheapest_overall = min(all_results, key=lambda x: x["total_cost"])
        print(
            f"   🥇 Most cost-effective: {cheapest_overall['provider']} {cheapest_overall['model']} (${cheapest_overall['total_cost']:.6f})"
        )

        # Best performance value (cost per ms)
        performance_value = min(
            all_results, key=lambda x: x["total_cost"] / x["latency_ms"]
        )
        cost_per_ms = performance_value["total_cost"] / performance_value["latency_ms"]
        print(
            f"   ⚡ Best performance value: {performance_value['provider']} {performance_value['model']} (${cost_per_ms:.8f}/ms)"
        )

        # Best token efficiency
        token_efficient = min(all_results, key=lambda x: x["cost_per_token"])
        print(
            f"   📊 Most token-efficient: {token_efficient['provider']} {token_efficient['model']} (${token_efficient['cost_per_token']:.8f}/token)"
        )

        # Cost savings opportunities
        print("\n💡 Cost Optimization Recommendations:")
        most_expensive = max(all_results, key=lambda x: x["total_cost"])
        potential_savings = (
            most_expensive["total_cost"] - cheapest_overall["total_cost"]
        )

        if potential_savings > 0:
            savings_percent = (potential_savings / most_expensive["total_cost"]) * 100
            print(
                f"   • Switch from most expensive to cheapest: Save ${potential_savings:.6f} ({savings_percent:.1f}%) per request"
            )
            print(
                f"   • At 1000 requests/month: Save ${potential_savings * 1000:.2f}/month"
            )
            print(
                f"   • At 10000 requests/month: Save ${potential_savings * 10000:.2f}/month"
            )

    return True


def demonstrate_cost_tracking_features():
    """Demonstrate advanced cost tracking features."""

    print("\n🔧 ADVANCED COST TRACKING FEATURES")
    print("=" * 42)

    features = [
        "✅ Real-time cost calculation across all providers",
        "✅ Gateway fee tracking and analysis",
        "✅ Token-level cost attribution",
        "✅ Provider cost comparison and optimization",
        "✅ Session-based cost aggregation",
        "✅ Historical cost trend analysis",
        "✅ Budget monitoring and alerts",
        "✅ Customer-specific cost attribution",
        "✅ Team and project cost segregation",
        "✅ Multi-currency cost reporting",
    ]

    for feature in features:
        print(f"   {feature}")


def main():
    """Main function to run the multi-provider cost comparison."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv("HELICONE_API_KEY"):
        print("❌ Missing HELICONE_API_KEY")
        return False

    provider_count = sum(
        [
            bool(os.getenv("OPENAI_API_KEY")),
            bool(os.getenv("ANTHROPIC_API_KEY")),
            bool(os.getenv("GROQ_API_KEY")),
        ]
    )

    if provider_count < 2:
        print("❌ Need at least 2 AI provider API keys for meaningful comparison")
        print("💡 Available providers:")
        print("   • OpenAI: export OPENAI_API_KEY='your_key'")
        print("   • Anthropic: export ANTHROPIC_API_KEY='your_key'")
        print("   • Groq: export GROQ_API_KEY='your_key' (free tier available)")
        return False

    # Run the analysis
    success = run_cost_comparison_analysis()
    demonstrate_cost_tracking_features()

    if success:
        print("\n🎉 SUCCESS! Multi-provider cost analysis completed.")
        print("\n📊 Key Insights:")
        print("   • Identified most cost-effective provider/model combinations")
        print("   • Discovered performance vs cost trade-offs")
        print("   • Quantified potential cost savings opportunities")
        print("   • Established baseline for ongoing cost optimization")

        print("\n📚 Next Steps:")
        print("   • Try 'python cost_optimization.py' for intelligent routing")
        print(
            "   • Try 'python advanced_features.py' for streaming & advanced patterns"
        )
        print("   • Implement findings in your production applications")
    else:
        print("\n❌ Cost analysis encountered issues. Check errors above.")

    return success


if __name__ == "__main__":
    """Entry point for the multi-provider cost comparison."""
    success = main()

    if success:
        print("\n" + "💰" * 20)
        print("Multi-provider cost intelligence: Make informed AI decisions!")
        print("💰" * 20)

    sys.exit(0 if success else 1)
