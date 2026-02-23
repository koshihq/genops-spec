#!/usr/bin/env python3
"""
Helicone Cost Optimization Example

This example demonstrates intelligent cost optimization strategies using
Helicone AI Gateway with GenOps tracking. Learn how to minimize AI costs
while maintaining quality through smart routing and budget management.

Usage:
    python cost_optimization.py

Prerequisites:
    pip install genops[helicone]
    export HELICONE_API_KEY="your_helicone_api_key"
    export OPENAI_API_KEY="your_openai_api_key"
    export GROQ_API_KEY="your_groq_api_key"  # Recommended for cost optimization
"""

import os
import sys
from datetime import datetime


def demonstrate_intelligent_routing():
    """Show intelligent routing strategies for cost optimization."""

    print("🧠 Intelligent Cost-Optimized Routing Strategies")
    print("=" * 52)

    try:
        from genops.providers.helicone import instrument_helicone

        adapter = instrument_helicone(
            team="cost-optimization-team",
            project="smart-routing-demo",
            environment="production",
        )

        print("✅ Cost-optimization adapter initialized")

    except Exception as e:
        print(f"❌ Adapter setup failed: {e}")
        return False

    # Test different routing strategies
    test_queries = [
        {
            "query": "What is 2+2?",
            "complexity": "simple",
            "strategy": "cost_optimized",
            "description": "Simple math - route to cheapest provider",
        },
        {
            "query": "Explain quantum computing and its applications.",
            "complexity": "complex",
            "strategy": "quality_optimized",
            "description": "Complex topic - prioritize quality over cost",
        },
        {
            "query": "Write a professional email.",
            "complexity": "medium",
            "strategy": "balanced",
            "description": "Medium task - balance cost and quality",
        },
    ]

    print(f"\n🎯 Testing {len(test_queries)} routing strategies...")

    for i, test in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {test['description']}")
        print(f"   Query: {test['query']}")
        print(f"   Strategy: {test['strategy']}")

        try:
            response = adapter.multi_provider_chat(
                message=test["query"],
                providers=["openai", "groq", "anthropic"],  # Ordered preference
                routing_strategy=test["strategy"],
                max_cost=0.01,  # Budget constraint
                customer_id=f"cost-opt-test-{i}",
            )

            if hasattr(response, "usage") and response.usage:
                cost = getattr(response.usage, "total_cost", 0.0)
                provider_used = getattr(response, "provider_used", "unknown")
                print(f"   ✅ Routed to: {provider_used}")
                print(f"   💰 Cost: ${cost:.6f}")

                # Show cost optimization logic
                if test["strategy"] == "cost_optimized" and cost < 0.005:
                    print("   🎯 Optimization success: Used cheapest provider")
                elif test["strategy"] == "quality_optimized":
                    print("   🎯 Quality priority: Selected best reasoning model")
                elif test["strategy"] == "balanced":
                    print("   🎯 Balanced approach: Cost-quality trade-off optimized")

            else:
                print("   ⚠️  Response received but cost data unavailable")

        except Exception as e:
            print(f"   ❌ Routing failed: {e}")
            continue

    return True


def demonstrate_budget_management():
    """Show budget management and cost controls."""

    print("\n💸 Budget Management & Cost Controls")
    print("=" * 40)

    try:
        from genops.providers.helicone import instrument_helicone

        # Initialize for budget management demonstration
        adapter = instrument_helicone(
            team="budget-demo-team",
            project="budget-management",
            environment="production",
        )

        # Simulate budget scenarios
        budget_tests = [
            {
                "name": "Under Budget Request",
                "query": "Hello, how are you?",
                "max_cost": 0.01,
                "should_succeed": True,
            },
            {
                "name": "Budget-Constrained Request",
                "query": "Write a detailed analysis of machine learning trends.",
                "max_cost": 0.001,  # Very tight budget
                "should_succeed": False,
            },
        ]

        print("🎯 Testing budget enforcement...")

        for test in budget_tests:
            print(f"\n📋 {test['name']}")
            print(f"   Query: {test['query']}")
            print(f"   Max budget: ${test['max_cost']:.6f}")

            try:
                response = adapter.chat(
                    message=test["query"],
                    provider="groq",  # Usually cheapest
                    model="mixtral-8x7b-32768",
                    max_cost=test["max_cost"],
                    customer_id="budget-test",
                )

                cost = (
                    getattr(response.usage, "total_cost", 0.0)
                    if hasattr(response, "usage")
                    else 0.0
                )

                if cost <= test["max_cost"]:
                    print(f"   ✅ Success: Cost ${cost:.6f} within budget")
                else:
                    print(f"   ⚠️  Warning: Cost ${cost:.6f} exceeded budget")

            except Exception as e:
                if "budget" in str(e).lower() or "cost" in str(e).lower():
                    print(f"   ✅ Budget enforced: {e}")
                else:
                    print(f"   ❌ Unexpected error: {e}")

        # Budget monitoring features
        print("\n💰 Budget Management Features:")
        features = [
            "✅ Per-request cost limits with automatic enforcement",
            "✅ Team and project budget allocation",
            "✅ Real-time budget tracking and alerts",
            "✅ Cost forecasting based on usage patterns",
            "✅ Automatic provider switching for budget compliance",
            "✅ Monthly and daily budget caps",
            "✅ Customer-specific budget controls",
            "✅ Cost anomaly detection and alerts",
        ]

        for feature in features:
            print(f"   {feature}")

    except Exception as e:
        print(f"❌ Budget management demo failed: {e}")
        return False

    return True


def demonstrate_cost_analytics():
    """Show cost analytics and optimization insights."""

    print("\n📊 Cost Analytics & Optimization Insights")
    print("=" * 43)

    # Simulated cost analytics (would use real data in production)
    analytics = {
        "monthly_spend": 127.45,
        "requests_this_month": 15420,
        "avg_cost_per_request": 0.00827,
        "top_cost_teams": [
            {"team": "ml-research", "cost": 45.20, "requests": 4200},
            {"team": "product", "cost": 38.15, "requests": 5800},
            {"team": "engineering", "cost": 28.90, "requests": 3920},
        ],
        "provider_breakdown": [
            {"provider": "openai", "cost": 78.30, "percentage": 61.4},
            {"provider": "anthropic", "cost": 35.20, "percentage": 27.6},
            {"provider": "groq", "cost": 13.95, "percentage": 10.9},
        ],
        "optimization_opportunities": [
            {
                "description": "Switch simple queries to Groq",
                "monthly_savings": 24.60,
                "impact": "High",
            },
            {
                "description": "Use Claude Haiku for medium complexity",
                "monthly_savings": 15.30,
                "impact": "Medium",
            },
        ],
    }

    print("📈 Monthly Cost Analytics:")
    print(f"   💰 Total spend: ${analytics['monthly_spend']:.2f}")
    print(f"   📊 Total requests: {analytics['requests_this_month']:,}")
    print(f"   📉 Average cost/request: ${analytics['avg_cost_per_request']:.5f}")

    print("\n👥 Top Spending Teams:")
    for team in analytics["top_cost_teams"]:
        avg_cost = team["cost"] / team["requests"]
        print(
            f"   • {team['team']:>12}: ${team['cost']:>6.2f} ({team['requests']:,} requests, ${avg_cost:.5f} avg)"
        )

    print("\n🔄 Provider Cost Breakdown:")
    for provider in analytics["provider_breakdown"]:
        print(
            f"   • {provider['provider'].title():>10}: ${provider['cost']:>6.2f} ({provider['percentage']:>4.1f}%)"
        )

    print("\n💡 Optimization Opportunities:")
    total_potential_savings = sum(
        opp["monthly_savings"] for opp in analytics["optimization_opportunities"]
    )
    for opp in analytics["optimization_opportunities"]:
        print(
            f"   • {opp['description']}: ${opp['monthly_savings']:>5.2f}/month ({opp['impact']} impact)"
        )

    print(
        f"\n🎯 Total Potential Monthly Savings: ${total_potential_savings:.2f} ({total_potential_savings / analytics['monthly_spend'] * 100:.1f}%)"
    )

    return True


def main():
    """Main function to run cost optimization demonstrations."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv("HELICONE_API_KEY"):
        print("❌ Missing HELICONE_API_KEY")
        return False

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY (required for cost comparison)")
        return False

    # Run demonstrations
    success = True
    success &= demonstrate_intelligent_routing()
    success &= demonstrate_budget_management()
    success &= demonstrate_cost_analytics()

    if success:
        print("\n🎉 SUCCESS! Cost optimization demonstration completed.")
        print("\n💰 Key Cost Optimization Strategies:")
        print("   • Use intelligent routing based on query complexity")
        print("   • Set budget limits to prevent cost overruns")
        print("   • Monitor usage patterns for optimization opportunities")
        print("   • Choose the right provider/model for each use case")

        print("\n📚 Next Steps:")
        print("   • Implement routing strategies in your applications")
        print("   • Set up budget monitoring and alerts")
        print("   • Try 'python advanced_features.py' for more advanced patterns")
    else:
        print("\n❌ Cost optimization demo encountered issues.")

    return success


if __name__ == "__main__":
    """Entry point for the cost optimization example."""
    success = main()

    if success:
        print("\n" + "💡" * 20)
        print("Smart cost optimization: Maximum AI value, minimum spend!")
        print("💡" * 20)

    sys.exit(0 if success else 1)
