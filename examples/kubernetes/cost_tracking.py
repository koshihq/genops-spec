#!/usr/bin/env python3
"""
✅ Cost Tracking Kubernetes Example

Demonstrates comprehensive cost tracking and budget management in Kubernetes.
Shows multi-provider cost aggregation, budget enforcement, and cost optimization.

Usage:
    python cost_tracking.py
    python cost_tracking.py --budget 50.00
    python cost_tracking.py --team engineering --project demo-app
    python cost_tracking.py --multi-provider
    python cost_tracking.py --cost-optimization
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

# Import GenOps for cost tracking
try:
    from genops.core.cost import BudgetManager, CostSummary, CostTracker
    from genops.core.governance import create_governance_context

    from genops.providers.kubernetes import KubernetesAdapter, validate_kubernetes_setup
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False
    print("⚠️  GenOps not installed. Install with: pip install genops")

# Import AI providers for multi-provider cost tracking
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class KubernetesCostDemo:
    """Demonstrates cost tracking features in Kubernetes environments."""

    def __init__(self):
        self.adapter = None
        self.cost_tracker = None
        self.budget_manager = None

        if GENOPS_AVAILABLE:
            self.adapter = KubernetesAdapter()
            self.cost_tracker = CostTracker()
            self.budget_manager = BudgetManager()

    async def demonstrate_basic_cost_tracking(
        self,
        team: Optional[str] = None,
        project: Optional[str] = None
    ) -> bool:
        """Demonstrate basic cost tracking in Kubernetes."""

        print("💰 Basic Cost Tracking in Kubernetes")
        print("=" * 60)

        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False

        # 1. Validate setup
        validation = validate_kubernetes_setup()
        if not validation.is_kubernetes_environment:
            print("⚠️  Not in Kubernetes - cost tracking will work with limited context")

        # 2. Set up governance context
        governance_attrs = {
            "team": team or os.getenv("DEFAULT_TEAM", "demo-team"),
            "project": project or os.getenv("PROJECT_NAME", "cost-demo"),
            "customer_id": "cost-demo-customer",
            "environment": os.getenv("ENVIRONMENT", "development")
        }

        print("\n1️⃣ Setting up cost tracking for:")
        print(f"   Team: {governance_attrs['team']}")
        print(f"   Project: {governance_attrs['project']}")
        print(f"   Customer: {governance_attrs['customer_id']}")

        # 3. Track costs with Kubernetes context
        with self.adapter.create_governance_context(**governance_attrs) as ctx:
            print(f"\n2️⃣ Tracking costs in context: {ctx.context_id}")

            # Simulate multiple AI operations with different costs
            operations = [
                ("openai", "gpt-3.5-turbo", 0.0023, 15, 50, "chat_completion"),
                ("anthropic", "claude-3-haiku", 0.0018, 12, 45, "text_generation"),
                ("openai", "gpt-4", 0.0156, 20, 60, "chat_completion"),
                ("anthropic", "claude-3-sonnet", 0.0089, 25, 75, "analysis")
            ]

            total_simulated_cost = 0

            for provider, model, cost, tokens_in, tokens_out, operation in operations:
                print(f"   💸 {provider} {model}: ${cost:.4f} ({tokens_in} → {tokens_out} tokens)")

                # Add cost data to context
                ctx.add_cost_data(
                    provider=provider,
                    model=model,
                    cost=cost,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    operation=operation
                )

                total_simulated_cost += cost

                # Simulate operation time
                await asyncio.sleep(0.1)

            # 4. Get cost summary
            print("\n3️⃣ Cost Summary:")
            cost_summary = ctx.get_cost_summary()

            print(f"   Total Cost: ${cost_summary.get('total_cost', total_simulated_cost):.4f}")
            print(f"   Operations: {len(operations)}")
            print(f"   Duration: {ctx.get_duration():.2f}s")

            # Show cost breakdown by provider
            cost_by_provider = {}
            for provider, _, cost, _, _, _ in operations:
                cost_by_provider[provider] = cost_by_provider.get(provider, 0) + cost

            print("\n4️⃣ Cost Breakdown by Provider:")
            for provider, provider_cost in cost_by_provider.items():
                percentage = (provider_cost / total_simulated_cost) * 100
                print(f"   {provider}: ${provider_cost:.4f} ({percentage:.1f}%)")

            # Show Kubernetes attribution
            k8s_attrs = ctx.get_telemetry_data()
            print("\n5️⃣ Kubernetes Attribution:")
            for key in ['k8s.namespace.name', 'k8s.pod.name', 'k8s.node.name']:
                value = k8s_attrs.get(key, 'Not available')
                print(f"   {key}: {value}")

        print("✅ Basic cost tracking completed!")
        return True

    async def demonstrate_budget_management(self, budget_limit: float = 100.0) -> bool:
        """Demonstrate budget management and enforcement."""

        print(f"\n💳 Budget Management (${budget_limit:.2f} limit)")
        print("=" * 60)

        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False

        # 1. Create budget
        budget_config = {
            "daily_limit": budget_limit,
            "monthly_limit": budget_limit * 30,
            "team": "demo-team",
            "project": "budget-demo",
            "alert_thresholds": [50, 75, 90]  # Percentage thresholds
        }

        print(f"1️⃣ Creating budget with ${budget_limit:.2f} daily limit")
        print(f"   Alert thresholds: {budget_config['alert_thresholds']}%")

        # 2. Simulate spending approaching budget
        current_spend = 0.0
        operations = [
            ("Initial batch", budget_limit * 0.3),   # 30% of budget
            ("Mid-day usage", budget_limit * 0.25),  # 25% more = 55% total
            ("Afternoon spike", budget_limit * 0.20), # 20% more = 75% total
            ("Evening batch", budget_limit * 0.18),  # 18% more = 93% total
            ("Late request", budget_limit * 0.10)    # Would exceed budget
        ]

        print("\n2️⃣ Simulating spending against budget:")

        for i, (description, cost) in enumerate(operations, 1):
            potential_total = current_spend + cost
            percentage = (potential_total / budget_limit) * 100

            print(f"\n   Operation {i}: {description}")
            print(f"   Cost: ${cost:.2f}")
            print(f"   Would bring total to: ${potential_total:.2f} ({percentage:.1f}%)")

            # Check budget enforcement
            if potential_total <= budget_limit:
                current_spend = potential_total
                status = "✅ APPROVED"

                # Check alert thresholds
                for threshold in budget_config['alert_thresholds']:
                    if percentage >= threshold and (current_spend - cost) / budget_limit * 100 < threshold:
                        print(f"   🚨 ALERT: {threshold}% budget threshold exceeded!")

            else:
                status = "❌ REJECTED - Budget exceeded"
                print(f"   🛑 This operation would exceed the daily budget of ${budget_limit:.2f}")
                break

            print(f"   Status: {status}")
            print(f"   Remaining budget: ${budget_limit - current_spend:.2f}")

        print("\n3️⃣ Final Budget Status:")
        print(f"   Used: ${current_spend:.2f} / ${budget_limit:.2f}")
        print(f"   Utilization: {(current_spend / budget_limit) * 100:.1f}%")
        print(f"   Remaining: ${budget_limit - current_spend:.2f}")

        return True

    async def demonstrate_multi_provider_cost_aggregation(self) -> bool:
        """Demonstrate cost aggregation across multiple AI providers."""

        print("\n🔄 Multi-Provider Cost Aggregation")
        print("=" * 60)

        if not GENOPS_AVAILABLE:
            print("❌ GenOps not available")
            return False

        # Define multi-provider scenario
        providers = {
            "openai": {
                "models": ["gpt-3.5-turbo", "gpt-4", "text-embedding-ada-002"],
                "base_costs": [0.002, 0.03, 0.0001]
            },
            "anthropic": {
                "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
                "base_costs": [0.0015, 0.015, 0.075]
            },
            "openrouter": {
                "models": ["meta-llama/llama-2-70b", "mistralai/mixtral-8x7b"],
                "base_costs": [0.0008, 0.0005]
            }
        }

        print("1️⃣ Multi-Provider Cost Simulation:")

        total_cost = 0
        cost_by_provider = {}
        cost_by_model = {}
        operations_count = 0

        # Create aggregation context
        governance_attrs = {
            "team": "multi-provider-team",
            "project": "cost-aggregation-demo",
            "customer_id": "enterprise-customer"
        }

        with self.adapter.create_governance_context(**governance_attrs) as ctx:

            # Simulate operations across all providers
            for provider, config in providers.items():
                print(f"\n   🔹 {provider.upper()} Operations:")
                provider_cost = 0

                for model, base_cost in zip(config["models"], config["base_costs"]):
                    # Simulate variable usage
                    operations = 3 + (hash(model) % 5)  # 3-7 operations per model

                    for op in range(operations):
                        # Variable cost based on usage
                        cost_multiplier = 1 + (op * 0.2)  # Increasing cost per operation
                        operation_cost = base_cost * cost_multiplier

                        ctx.add_cost_data(
                            provider=provider,
                            model=model,
                            cost=operation_cost,
                            tokens_in=15 + (op * 5),
                            tokens_out=50 + (op * 10),
                            operation=f"operation_{op+1}"
                        )

                        provider_cost += operation_cost
                        total_cost += operation_cost
                        operations_count += 1

                        # Track by model
                        cost_by_model[f"{provider}/{model}"] = \
                            cost_by_model.get(f"{provider}/{model}", 0) + operation_cost

                    print(f"     {model}: {operations} ops, ${provider_cost:.4f}")

                cost_by_provider[provider] = provider_cost

            print("\n2️⃣ Aggregated Cost Summary:")
            print(f"   Total Operations: {operations_count}")
            print(f"   Total Cost: ${total_cost:.4f}")
            print(f"   Average Cost/Operation: ${total_cost/operations_count:.4f}")

            print("\n3️⃣ Cost by Provider:")
            for provider, cost in sorted(cost_by_provider.items(), key=lambda x: x[1], reverse=True):
                percentage = (cost / total_cost) * 100
                print(f"   {provider:12}: ${cost:8.4f} ({percentage:5.1f}%)")

            print("\n4️⃣ Top Cost Models:")
            top_models = sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True)[:5]
            for model, cost in top_models:
                percentage = (cost / total_cost) * 100
                print(f"   {model:30}: ${cost:8.4f} ({percentage:5.1f}%)")

            # Cost optimization suggestions
            print("\n5️⃣ Cost Optimization Suggestions:")

            # Find most expensive provider
            most_expensive = max(cost_by_provider.items(), key=lambda x: x[1])
            cheapest = min(cost_by_provider.items(), key=lambda x: x[1])

            potential_savings = most_expensive[1] - cheapest[1]
            print(f"   • Consider migrating from {most_expensive[0]} to {cheapest[0]}")
            print(f"     Potential savings: ${potential_savings:.4f} ({potential_savings/total_cost*100:.1f}%)")

            # Model-level optimization
            most_expensive_model = max(cost_by_model.items(), key=lambda x: x[1])
            print(f"   • Review usage of {most_expensive_model[0]}")
            print(f"     This model accounts for ${most_expensive_model[1]:.4f} ({most_expensive_model[1]/total_cost*100:.1f}%)")

        return True

    async def demonstrate_cost_optimization_strategies(self) -> bool:
        """Demonstrate intelligent cost optimization strategies."""

        print("\n🎯 Cost Optimization Strategies")
        print("=" * 60)

        # Define task complexity and model capabilities
        optimization_scenarios = [
            {
                "task": "Simple chat completion",
                "complexity": "low",
                "recommended_models": [
                    ("openrouter/meta-llama-2-7b", 0.0002),
                    ("openai/gpt-3.5-turbo", 0.002),
                    ("anthropic/claude-3-haiku", 0.0015)
                ]
            },
            {
                "task": "Complex analysis and reasoning",
                "complexity": "high",
                "recommended_models": [
                    ("anthropic/claude-3-sonnet", 0.015),
                    ("openai/gpt-4", 0.03),
                    ("anthropic/claude-3-opus", 0.075)
                ]
            },
            {
                "task": "Code generation",
                "complexity": "medium",
                "recommended_models": [
                    ("openai/gpt-3.5-turbo", 0.002),
                    ("anthropic/claude-3-sonnet", 0.015),
                    ("openrouter/codellama-34b", 0.001)
                ]
            }
        ]

        print("1️⃣ Intelligent Model Selection:")

        total_savings = 0

        for scenario in optimization_scenarios:
            print(f"\n   📋 Task: {scenario['task']} ({scenario['complexity']} complexity)")

            models = scenario['recommended_models']
            cheapest_cost = min(model[1] for model in models)
            most_expensive_cost = max(model[1] for model in models)

            print("   Model Options (cost per 1K tokens):")
            for model, cost in models:
                savings_vs_expensive = most_expensive_cost - cost
                if cost == cheapest_cost:
                    marker = "🟢 RECOMMENDED"
                elif cost == most_expensive_cost:
                    marker = "🔴 EXPENSIVE"
                else:
                    marker = "🟡 MODERATE"

                print(f"     {marker} {model}: ${cost:.4f}")
                if savings_vs_expensive > 0:
                    print(f"       Savings vs most expensive: ${savings_vs_expensive:.4f}")

            scenario_savings = most_expensive_cost - cheapest_cost
            total_savings += scenario_savings

        print("\n2️⃣ Cost Optimization Impact:")
        print(f"   Total potential savings per 1K tokens: ${total_savings:.4f}")

        # Scale to realistic usage
        monthly_tokens = 1_000_000  # 1M tokens per month
        monthly_savings = (total_savings * monthly_tokens) / 1000

        print(f"   For {monthly_tokens:,} tokens/month:")
        print(f"   Potential monthly savings: ${monthly_savings:.2f}")
        print(f"   Annual savings: ${monthly_savings * 12:.2f}")

        print("\n3️⃣ Smart Routing Strategies:")
        print("   ✅ Route simple tasks to cost-effective models")
        print("   ✅ Use premium models only for complex reasoning")
        print("   ✅ Implement fallback chains for availability")
        print("   ✅ Monitor performance vs cost trade-offs")
        print("   ✅ Auto-scale model selection based on budget")

        print("\n4️⃣ Budget-Aware Operations:")
        print("   • Set model selection based on remaining budget")
        print("   • Implement cost caps with graceful degradation")
        print("   • Use cached responses to reduce redundant calls")
        print("   • Batch operations for volume discounts")

        return True


async def main():
    """Main cost tracking demo."""

    parser = argparse.ArgumentParser(
        description="Kubernetes cost tracking example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cost_tracking.py                          # Full cost tracking demo
    python cost_tracking.py --budget 50.00          # Demo with $50 budget
    python cost_tracking.py --multi-provider        # Multi-provider aggregation
    python cost_tracking.py --cost-optimization     # Cost optimization strategies
    python cost_tracking.py --team eng --project app # With attribution
        """
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Budget limit for budget management demo"
    )

    parser.add_argument(
        "--team",
        type=str,
        help="Team name for cost attribution"
    )

    parser.add_argument(
        "--project",
        type=str,
        help="Project name for cost tracking"
    )

    parser.add_argument(
        "--multi-provider",
        action="store_true",
        help="Run multi-provider cost aggregation demo"
    )

    parser.add_argument(
        "--cost-optimization",
        action="store_true",
        help="Run cost optimization strategies demo"
    )

    args = parser.parse_args()

    demo = KubernetesCostDemo()
    success = True

    # Run specific demos if requested
    if args.multi_provider:
        success = await demo.demonstrate_multi_provider_cost_aggregation()
    elif args.cost_optimization:
        success = await demo.demonstrate_cost_optimization_strategies()
    else:
        # Run comprehensive cost tracking demo
        print("🚀 Comprehensive Kubernetes Cost Tracking Demo")
        print("=" * 80)

        # 1. Basic cost tracking
        basic_success = await demo.demonstrate_basic_cost_tracking(
            team=args.team,
            project=args.project
        )
        success = success and basic_success

        # 2. Budget management
        budget_success = await demo.demonstrate_budget_management(args.budget)
        success = success and budget_success

        # 3. Multi-provider aggregation
        multi_success = await demo.demonstrate_multi_provider_cost_aggregation()
        success = success and multi_success

        # 4. Cost optimization
        opt_success = await demo.demonstrate_cost_optimization_strategies()
        success = success and opt_success

        # Final summary
        print("\n🎉 COST TRACKING DEMO COMPLETE!")
        print("=" * 80)
        print("✅ Basic cost tracking with Kubernetes attribution")
        print("✅ Budget management and enforcement")
        print("✅ Multi-provider cost aggregation")
        print("✅ Intelligent cost optimization strategies")

        print("\n💡 Key Takeaways:")
        print("   • Real-time cost tracking across all AI providers")
        print("   • Kubernetes context automatically added to cost data")
        print("   • Budget enforcement prevents cost overruns")
        print("   • Multi-provider aggregation enables cost comparison")
        print("   • Smart routing optimizes cost vs performance")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
