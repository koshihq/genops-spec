#!/usr/bin/env python3
"""
SkyRouter Route Optimization and Cost Intelligence Example

This example demonstrates advanced route optimization strategies with SkyRouter
and GenOps, showing how to optimize costs across 150+ models through intelligent
routing, volume discounts, and performance-aware model selection.

Features demonstrated:
- Advanced routing strategy comparison and optimization
- Volume discount optimization across model ecosystem
- Cost efficiency analysis and recommendations
- Multi-model performance vs cost analysis
- Route intelligence with automated suggestions
- Budget optimization with smart alerting

Usage:
    export SKYROUTER_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python route_optimization.py

Author: GenOps AI Contributors
"""

import os
import sys
from pathlib import Path
from typing import Any

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def demonstrate_routing_strategy_optimization():
    """Demonstrate comprehensive routing strategy optimization."""

    print("🚀 SkyRouter Route Optimization & Cost Intelligence")
    print("=" * 55)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        from genops.providers.skyrouter_pricing import (
            RaindropPricingConfig,  # noqa: F401
        )
    except ImportError as e:
        print(f"❌ Error importing GenOps SkyRouter: {e}")
        print(
            "💡 Make sure you're in the project root directory and GenOps is properly installed"
        )
        return False

    # Configuration
    api_key = os.getenv("SKYROUTER_API_KEY")
    team = os.getenv("GENOPS_TEAM", "route-optimization-team")
    project = os.getenv("GENOPS_PROJECT", "cost-intelligence-demo")

    print("🔧 Configuration:")
    print(f"  👥 Team: {team}")
    print(f"  📊 Project: {project}")
    print("  🎯 Focus: Route optimization across 150+ models")
    print()

    # Initialize adapter with optimization focus
    adapter = GenOpsSkyRouterAdapter(
        skyrouter_api_key=api_key,
        team=team,
        project=project,
        environment="production",
        daily_budget_limit=100.0,
        enable_cost_alerts=True,
        governance_policy="enforced",
    )

    print("✅ Optimization-focused adapter initialized")
    print()

    # Test scenarios for optimization
    test_scenarios = [
        {
            "name": "Content Generation",
            "description": "Blog post creation with creative requirements",
            "complexity": "complex",
            "models": ["gpt-4", "claude-3-opus", "claude-3-sonnet", "gemini-pro"],
            "input_size": "large",
            "quality_requirements": "high",
        },
        {
            "name": "Code Review",
            "description": "Automated code analysis and suggestions",
            "complexity": "enterprise",
            "models": ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"],
            "input_size": "medium",
            "quality_requirements": "critical",
        },
        {
            "name": "Data Analysis",
            "description": "Large dataset summarization and insights",
            "complexity": "moderate",
            "models": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-pro", "llama-2"],
            "input_size": "large",
            "quality_requirements": "standard",
        },
        {
            "name": "Customer Support",
            "description": "Real-time customer query processing",
            "complexity": "simple",
            "models": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-pro"],
            "input_size": "small",
            "quality_requirements": "good",
        },
    ]

    routing_strategies = [
        "cost_optimized",
        "balanced",
        "latency_optimized",
        "reliability_first",
    ]
    optimization_results = {}

    print("📊 Running Route Optimization Analysis")
    print("-" * 42)

    for scenario in test_scenarios:
        print(f"\n🧪 Scenario: {scenario['name']}")
        print(f"   📝 {scenario['description']}")
        print(f"   🎛️  Complexity: {scenario['complexity']}")
        print(f"   📏 Input size: {scenario['input_size']}")
        print(f"   ⭐ Quality req: {scenario['quality_requirements']}")

        scenario_results = {}

        for strategy in routing_strategies:
            print(f"     🔄 Testing {strategy}...")

            with adapter.track_routing_session(
                f"optimization-{scenario['name'].lower().replace(' ', '_')}-{strategy}"
            ) as session:
                result = session.track_multi_model_routing(
                    models=scenario["models"],
                    input_data={
                        "scenario": scenario["name"],
                        "complexity": scenario["complexity"],
                        "input_size": scenario["input_size"],
                        "quality_requirements": scenario["quality_requirements"],
                    },
                    routing_strategy=strategy,
                    complexity=scenario["complexity"],
                )

                scenario_results[strategy] = {
                    "cost": float(result.total_cost),
                    "model": result.model,
                    "efficiency_score": result.route_efficiency_score,
                    "savings": float(result.optimization_savings),
                    "route": result.route,
                }

                print(
                    f"       💰 ${result.total_cost:.4f} | 🤖 {result.model} | ⚡ {result.route_efficiency_score:.2f}"
                )

        optimization_results[scenario["name"]] = scenario_results

        # Find optimal strategy for this scenario
        best_cost = min(scenario_results.items(), key=lambda x: x[1]["cost"])
        best_efficiency = max(
            scenario_results.items(), key=lambda x: x[1]["efficiency_score"]
        )

        print(f"     🏆 Best cost: {best_cost[0]} (${best_cost[1]['cost']:.4f})")
        print(
            f"     🏆 Best efficiency: {best_efficiency[0]} (score: {best_efficiency[1]['efficiency_score']:.2f})"
        )

    return optimization_results


def analyze_optimization_results(optimization_results: dict[str, dict[str, Any]]):
    """Analyze optimization results and provide recommendations."""

    print("\n📈 Route Optimization Analysis & Recommendations")
    print("=" * 52)

    # Overall cost analysis
    total_costs = {}
    total_efficiency = {}
    strategy_wins = {}

    for strategy in [
        "cost_optimized",
        "balanced",
        "latency_optimized",
        "reliability_first",
    ]:
        total_costs[strategy] = 0
        total_efficiency[strategy] = 0
        strategy_wins[strategy] = {"cost": 0, "efficiency": 0}

    for _scenario_name, scenario_results in optimization_results.items():
        # Find best performers
        best_cost = min(scenario_results.items(), key=lambda x: x[1]["cost"])
        best_efficiency = max(
            scenario_results.items(), key=lambda x: x[1]["efficiency_score"]
        )

        strategy_wins[best_cost[0]]["cost"] += 1
        strategy_wins[best_efficiency[0]]["efficiency"] += 1

        # Accumulate totals
        for strategy, result in scenario_results.items():
            total_costs[strategy] += result["cost"]
            total_efficiency[strategy] += result["efficiency_score"]

    print("🏆 Strategy Performance Summary:")
    print()

    num_scenarios = len(optimization_results)
    for strategy in total_costs.keys():
        avg_cost = total_costs[strategy] / num_scenarios
        avg_efficiency = total_efficiency[strategy] / num_scenarios
        cost_wins = strategy_wins[strategy]["cost"]
        efficiency_wins = strategy_wins[strategy]["efficiency"]

        print(f"📊 **{strategy}**:")
        print(f"   💰 Avg cost: ${avg_cost:.4f}")
        print(f"   ⚡ Avg efficiency: {avg_efficiency:.2f}")
        print(f"   🏅 Cost wins: {cost_wins}/{num_scenarios}")
        print(f"   🏅 Efficiency wins: {efficiency_wins}/{num_scenarios}")
        print()

    # Generate specific recommendations
    print("💡 Optimization Recommendations:")
    print()

    # Find overall best strategy
    best_overall_cost = min(total_costs.items(), key=lambda x: x[1])
    best_overall_efficiency = max(total_efficiency.items(), key=lambda x: x[1])

    recommendations = []

    if best_overall_cost[0] == best_overall_efficiency[0]:
        recommendations.append(
            f"🎯 **Use '{best_overall_cost[0]}' as your default strategy** - it provides the best balance of cost and efficiency."
        )
    else:
        recommendations.append(
            f"💰 **For cost optimization**: Use '{best_overall_cost[0]}' (saves ${(max(total_costs.values()) - best_overall_cost[1]):.4f} vs worst)"
        )
        recommendations.append(
            f"⚡ **For efficiency optimization**: Use '{best_overall_efficiency[0]}' (efficiency score: {best_overall_efficiency[1] / num_scenarios:.2f})"
        )

    # Scenario-specific recommendations
    scenario_recommendations = {}
    for scenario_name, scenario_results in optimization_results.items():
        best_cost = min(scenario_results.items(), key=lambda x: x[1]["cost"])
        scenario_recommendations[scenario_name] = best_cost[0]

    if len(set(scenario_recommendations.values())) > 1:
        recommendations.append(
            "🎛️ **Use scenario-specific strategies for optimal results:**"
        )
        for scenario, strategy in scenario_recommendations.items():
            cost = optimization_results[scenario][strategy]["cost"]
            recommendations.append(f"   • {scenario}: {strategy} (${cost:.4f})")

    # Volume considerations
    high_volume_scenarios = ["Customer Support", "Data Analysis"]
    cost_savings = sum(
        max(optimization_results[scenario].values(), key=lambda x: x["cost"])["cost"]
        - min(optimization_results[scenario].values(), key=lambda x: x["cost"])["cost"]
        for scenario in high_volume_scenarios
        if scenario in optimization_results
    )

    if cost_savings > 0:
        monthly_savings = cost_savings * 1000  # Assuming 1000 operations per month
        recommendations.append(
            f"📊 **Volume optimization**: Optimizing high-volume scenarios could save ~${monthly_savings:.2f}/month"
        )

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print()


def demonstrate_volume_discount_optimization():
    """Demonstrate volume discount optimization across model tiers."""

    print("💎 Volume Discount Optimization")
    print("-" * 35)

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        from genops.providers.skyrouter_pricing import SkyRouterPricingConfig

        # Configure enterprise volume pricing
        enterprise_pricing = SkyRouterPricingConfig()
        enterprise_pricing.volume_tiers = {
            1000: 0.05,  # 5% discount for 1K+ tokens
            10000: 0.12,  # 12% discount for 10K+ tokens
            50000: 0.20,  # 20% discount for 50K+ tokens
            200000: 0.30,  # 30% discount for 200K+ tokens
            1000000: 0.40,  # 40% discount for enterprise volume
        }

        adapter = GenOpsSkyRouterAdapter(
            team="volume-optimization",
            project="enterprise-pricing",
            daily_budget_limit=500.0,
        )

        # Update pricing configuration
        adapter.pricing_calculator.config = enterprise_pricing

        # Simulate different volume scenarios
        volume_scenarios = [
            {"name": "Small Team", "monthly_volume": 5000, "operations": 200},
            {"name": "Medium Team", "monthly_volume": 25000, "operations": 1000},
            {"name": "Large Team", "monthly_volume": 100000, "operations": 4000},
            {"name": "Enterprise", "monthly_volume": 500000, "operations": 20000},
        ]

        print("📊 Volume Discount Analysis:")
        print()

        for scenario in volume_scenarios:
            adapter.pricing_calculator.update_monthly_volume(scenario["monthly_volume"])
            volume_info = adapter.pricing_calculator.get_volume_discount_info()

            # Estimate monthly costs
            cost_estimate = adapter.pricing_calculator.estimate_monthly_cost(
                daily_operations=scenario["operations"] // 30,  # Daily operations
                avg_tokens_per_operation=500,
                model_distribution={
                    "gpt-4": 0.3,
                    "claude-3-sonnet": 0.3,
                    "gpt-3.5-turbo": 0.3,
                    "gemini-pro": 0.1,
                },
                optimization_strategy="balanced",
            )

            print(
                f"🏢 **{scenario['name']}** ({scenario['monthly_volume']:,} tokens/month)"
            )
            print(
                f"   📈 Volume discount: {volume_info['current_discount_percentage']:.1f}%"
            )
            print(f"   💰 Monthly cost: ${cost_estimate['final_monthly_cost']:.2f}")
            print(
                f"   💾 Discount savings: ${cost_estimate['volume_discount_amount']:.2f}"
            )
            print(
                f"   📊 Cost per operation: ${cost_estimate['cost_per_operation']:.4f}"
            )

            if volume_info["next_threshold"]:
                tokens_to_next = volume_info["tokens_to_next_tier"]
                next_discount = volume_info["next_discount_percentage"]
                print(
                    f"   🎯 Next tier: {tokens_to_next:,} more tokens for {next_discount:.1f}% discount"
                )

            print()

        return True

    except Exception as e:
        print(f"❌ Volume discount demo failed: {e}")
        return False


def demonstrate_intelligent_route_selection():
    """Demonstrate intelligent route selection based on context."""

    print("🧠 Intelligent Route Selection")
    print("-" * 33)

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        adapter = GenOpsSkyRouterAdapter(
            team="intelligent-routing",
            project="context-aware-optimization",
            daily_budget_limit=150.0,
        )

        # Define context-aware routing scenarios
        intelligent_scenarios = [
            {
                "context": "Real-time customer chat",
                "requirements": {
                    "max_latency": "500ms",
                    "quality": "good",
                    "cost_sensitivity": "medium",
                },
                "recommended_strategy": "latency_optimized",
                "models": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-pro"],
            },
            {
                "context": "Financial analysis report",
                "requirements": {
                    "accuracy": "critical",
                    "compliance": "required",
                    "cost_sensitivity": "low",
                },
                "recommended_strategy": "reliability_first",
                "models": ["gpt-4", "claude-3-opus"],
            },
            {
                "context": "Batch content moderation",
                "requirements": {
                    "volume": "high",
                    "speed": "important",
                    "cost_sensitivity": "high",
                },
                "recommended_strategy": "cost_optimized",
                "models": ["gpt-3.5-turbo", "claude-3-haiku", "llama-2", "gemini-pro"],
            },
            {
                "context": "Creative writing assistance",
                "requirements": {
                    "creativity": "high",
                    "quality": "excellent",
                    "user_experience": "premium",
                },
                "recommended_strategy": "balanced",
                "models": ["gpt-4", "claude-3-opus", "claude-3-sonnet", "gemini-pro"],
            },
        ]

        print("🎯 Context-Aware Route Selection Results:")
        print()

        route_intelligence = {}

        for scenario in intelligent_scenarios:
            context = scenario["context"]
            requirements = scenario["requirements"]
            recommended = scenario["recommended_strategy"]

            print(f"📋 **{context}**")
            print(
                f"   📝 Requirements: {', '.join(f'{k}={v}' for k, v in requirements.items())}"
            )
            print(f"   💡 Recommended: {recommended}")

            # Test the recommended strategy
            with adapter.track_routing_session(
                f"intelligent-{context.lower().replace(' ', '_')}"
            ) as session:
                result = session.track_multi_model_routing(
                    models=scenario["models"],
                    input_data={"context": context, "requirements": requirements},
                    routing_strategy=recommended,
                    complexity="moderate",
                )

                route_intelligence[context] = {
                    "strategy": recommended,
                    "selected_model": result.model,
                    "cost": float(result.total_cost),
                    "efficiency": result.route_efficiency_score,
                    "requirements_met": True,  # Assume requirements are met
                }

                print(f"   ✅ Selected: {result.model}")
                print(f"   💰 Cost: ${result.total_cost:.4f}")
                print(f"   ⚡ Efficiency: {result.route_efficiency_score:.2f}")
                print()

        # Analyze route intelligence patterns
        print("🧠 Route Intelligence Analysis:")
        print()

        strategy_usage = {}
        for _context, result in route_intelligence.items():
            strategy = result["strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1

        total_contexts = len(route_intelligence)
        for strategy, count in strategy_usage.items():
            percentage = (count / total_contexts) * 100
            print(
                f"   📊 {strategy}: {count}/{total_contexts} scenarios ({percentage:.1f}%)"
            )

        # Cost efficiency by context
        print()
        print("💡 Context-Specific Insights:")

        avg_cost_by_context = {
            context: result["cost"] for context, result in route_intelligence.items()
        }

        sorted_contexts = sorted(avg_cost_by_context.items(), key=lambda x: x[1])

        print(
            f"   💰 Most cost-effective: {sorted_contexts[0][0]} (${sorted_contexts[0][1]:.4f})"
        )
        print(
            f"   💎 Most premium: {sorted_contexts[-1][0]} (${sorted_contexts[-1][1]:.4f})"
        )
        print()

        return route_intelligence

    except Exception as e:
        print(f"❌ Intelligent route selection demo failed: {e}")
        return {}


def demonstrate_cost_optimization_recommendations():
    """Demonstrate automated cost optimization recommendations."""

    print("📈 Automated Cost Optimization Recommendations")
    print("-" * 48)

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        adapter = GenOpsSkyRouterAdapter(
            team="cost-optimization",
            project="automated-recommendations",
            daily_budget_limit=200.0,
        )

        # Simulate some usage to generate recommendations
        usage_patterns = [
            {"type": "high_volume_simple", "operations": 50, "strategy": "balanced"},
            {
                "type": "premium_complex",
                "operations": 5,
                "strategy": "reliability_first",
            },
            {
                "type": "mixed_workload",
                "operations": 20,
                "strategy": "latency_optimized",
            },
        ]

        print("🔄 Simulating usage patterns for recommendation engine...")

        for pattern in usage_patterns:
            for i in range(pattern["operations"]):
                with adapter.track_routing_session(f"{pattern['type']}-{i}") as session:
                    session.track_multi_model_routing(
                        models=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
                        input_data={"pattern_type": pattern["type"], "operation_id": i},
                        routing_strategy=pattern["strategy"],
                    )

        print("✅ Usage simulation completed")
        print()

        # Get optimization recommendations
        recommendations = (
            adapter.cost_aggregator.get_cost_optimization_recommendations()
        )

        if recommendations:
            print("🚀 Personalized Optimization Recommendations:")
            print()

            for i, rec in enumerate(recommendations, 1):
                priority_icon = (
                    "🔥"
                    if rec["priority_score"] >= 80
                    else "⭐"
                    if rec["priority_score"] >= 60
                    else "💡"
                )
                effort_icon = (
                    "🟢"
                    if rec["effort_level"] == "low"
                    else "🟡"
                    if rec["effort_level"] == "medium"
                    else "🔴"
                )

                print(f"{priority_icon} **Recommendation {i}: {rec['title']}**")
                print(
                    f"   📝 {rec.get('description', 'Optimize your routing for better cost efficiency')}"
                )
                print(f"   💰 Potential savings: ${rec['potential_savings']:.2f}/month")
                print(f"   {effort_icon} Effort level: {rec['effort_level']}")
                print(f"   🎯 Priority: {rec['priority_score']:.0f}/100")
                print(f"   🏷️  Type: {rec['optimization_type']}")

                # Add implementation guidance
                if rec["optimization_type"] == "model_optimization":
                    print(
                        "   🔧 Implementation: Review model selection for high-cost operations"
                    )
                elif rec["optimization_type"] == "route_optimization":
                    print(
                        "   🔧 Implementation: Switch to cost_optimized routing strategy"
                    )
                elif rec["optimization_type"] == "volume_optimization":
                    print(
                        "   🔧 Implementation: Consolidate operations to unlock volume discounts"
                    )

                print()
        else:
            print("🎉 Great! Your current routing patterns are well-optimized.")
            print("No specific recommendations at this time.")
            print()

        # Show current cost summary
        summary = adapter.cost_aggregator.get_summary()

        print("📊 Current Usage Summary:")
        print(f"   💰 Total cost: ${summary.total_cost:.4f}")
        print(f"   📈 Operations: {summary.total_operations}")
        print(f"   📉 Avg cost/op: ${summary.average_cost_per_operation:.4f}")
        print(f"   💾 Total savings: ${summary.optimization_savings:.4f}")
        print()

        # Budget utilization
        budget_status = adapter.cost_aggregator.check_budget_status()
        current_cost = budget_status["current_daily_cost"]
        budget_limit = budget_status["daily_budget_limit"]

        if budget_limit:
            utilization = (current_cost / budget_limit) * 100
            status_icon = (
                "🟢" if utilization < 50 else "🟡" if utilization < 80 else "🔴"
            )
            print(
                f"📊 Budget Utilization: {status_icon} {utilization:.1f}% (${current_cost:.4f}/${budget_limit:.2f})"
            )

        return True

    except Exception as e:
        print(f"❌ Cost optimization recommendations failed: {e}")
        return False


def main():
    """Main execution function."""

    print("🚀 SkyRouter Route Optimization & Cost Intelligence Demo")
    print("=" * 60)
    print()

    print("This example demonstrates advanced route optimization strategies")
    print("for cost-effective multi-model routing across 150+ models.")
    print()

    # Check prerequisites
    api_key = os.getenv("SKYROUTER_API_KEY")
    if not api_key:
        print("❌ Missing required environment variables:")
        print("   SKYROUTER_API_KEY - Your SkyRouter API key")
        print()
        print("💡 Set up your environment:")
        print("   export SKYROUTER_API_KEY='your-api-key'")
        print("   export GENOPS_TEAM='optimization-team'")
        print("   export GENOPS_PROJECT='route-optimization'")
        return

    try:
        success = True

        # Run routing strategy optimization
        if success:
            optimization_results = demonstrate_routing_strategy_optimization()
            if optimization_results:
                analyze_optimization_results(optimization_results)
            else:
                success = False

        # Volume discount optimization
        if success:
            print("\n" + "=" * 60 + "\n")
            success = demonstrate_volume_discount_optimization()

        # Intelligent route selection
        if success:
            print("\n" + "=" * 60 + "\n")
            route_intelligence = demonstrate_intelligent_route_selection()
            success = bool(route_intelligence)

        # Cost optimization recommendations
        if success:
            print("\n" + "=" * 60 + "\n")
            success = demonstrate_cost_optimization_recommendations()

        if success:
            print("🎉 Route Optimization demonstration completed successfully!")
            print()
            print("🔑 **Key Takeaways:**")
            print("• Different routing strategies excel in different scenarios")
            print("• Volume discounts can significantly reduce costs at scale")
            print("• Context-aware routing improves both cost and performance")
            print(
                "• Automated recommendations help identify optimization opportunities"
            )
            print("• Regular analysis ensures continued cost efficiency")
            print()
            print("🚀 **Next Steps:**")
            print("1. Implement scenario-specific routing strategies in production")
            print("2. Set up volume discount monitoring for your team")
            print("3. Try agent_workflows.py for multi-agent optimization patterns")
            print("4. Explore enterprise_patterns.py for production deployment")
            print()
            print("📖 **Advanced Topics:**")
            print("• Custom pricing tiers for enterprise volume")
            print("• Real-time route optimization based on performance metrics")
            print("• Multi-region routing with cost-aware failover")
            print("• Compliance-aware routing for regulated industries")

    except KeyboardInterrupt:
        print()
        print("👋 Demo cancelled.")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("1. Verify your SKYROUTER_API_KEY is correct and has sufficient credits")
        print("2. Check your internet connection")
        print("3. Ensure GenOps is properly installed: pip install genops[skyrouter]")


if __name__ == "__main__":
    main()
