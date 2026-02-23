#!/usr/bin/env python3
"""
Arize AI + GenOps Cost Optimization Example

This example demonstrates comprehensive cost intelligence and optimization
features for Arize AI model monitoring operations.

Features demonstrated:
- Multi-model cost aggregation and analysis
- Cost optimization recommendations with actionable insights
- Volume discount optimization and pricing tier analysis
- Budget forecasting and cost trend analysis
- Dynamic cost-aware monitoring strategies
- Enterprise cost management patterns

Run this example:
    python cost_optimization.py

Prerequisites:
    pip install genops[arize]
    export ARIZE_API_KEY="your-arize-api-key"
    export ARIZE_SPACE_KEY="your-arize-space-key"
"""

import random
import sys


def print_header():
    """Print example header."""
    print("=" * 70)
    print("💰 Arize AI + GenOps Cost Optimization Example")
    print("=" * 70)
    print()


def demonstrate_cost_aggregation():
    """Demonstrate comprehensive cost aggregation across multiple models."""
    print("📊 Multi-Model Cost Aggregation:")
    print()

    try:
        from genops.providers.arize_cost_aggregator import ArizeCostAggregator

        # Initialize cost aggregator for multiple models
        cost_aggregator = ArizeCostAggregator(
            team="ml-platform",
            project="multi-model-monitoring",
            budget_limit=500.0,
            retention_days=90,
        )

        print("  ✅ Cost aggregator initialized:")
        print(f"     • Team: {cost_aggregator.team}")
        print(f"     • Project: {cost_aggregator.project}")
        print(f"     • Budget Limit: ${cost_aggregator.budget_limit:.2f}")
        print()

        # Simulate monitoring costs for multiple models
        models_config = [
            ("fraud-detection-v3", "3.1", "production", 150000, 75, 8),
            ("credit-scoring-v2", "2.3", "production", 200000, 100, 12),
            ("risk-assessment-v1", "1.5", "staging", 50000, 25, 4),
            ("churn-prediction-v4", "4.0", "production", 300000, 150, 15),
            ("recommendation-engine", "2.1", "production", 500000, 200, 20),
        ]

        print("  🔄 Calculating costs for multiple models...")

        total_cost = 0.0
        cost_by_model = {}
        cost_by_environment = {}

        for (
            model_id,
            version,
            environment,
            predictions,
            quality_checks,
            alerts,
        ) in models_config:
            session_cost = cost_aggregator.calculate_monitoring_session_cost(
                model_id=model_id,
                model_version=version,
                environment=environment,
                prediction_count=predictions,
                data_quality_checks=quality_checks,
                active_alerts=alerts,
                session_duration_hours=720,  # Monthly (30 days * 24 hours)
                dashboard_views=100,
                storage_mb=predictions * 0.001,  # Estimate storage
            )

            model_key = f"{model_id}-{version}"
            cost_by_model[model_key] = session_cost.total_cost
            total_cost += session_cost.total_cost

            # Aggregate by environment
            if environment not in cost_by_environment:
                cost_by_environment[environment] = 0.0
            cost_by_environment[environment] += session_cost.total_cost

            print(f"     • {model_key}: ${session_cost.total_cost:.2f}")
            print(
                f"       - Predictions: {predictions:,} (${session_cost.prediction_logging_cost:.2f})"
            )
            print(
                f"       - Data Quality: {quality_checks} checks (${session_cost.data_quality_cost:.2f})"
            )
            print(
                f"       - Alerts: {alerts} active (${session_cost.alert_management_cost:.2f})"
            )
            print(f"       - Efficiency: {session_cost.efficiency_score:.1f} pred/hour")

        print("\n  💰 Cost Summary:")
        print(f"     • Total Monthly Cost: ${total_cost:.2f}")
        print(
            f"     • Budget Utilization: {(total_cost / cost_aggregator.budget_limit) * 100:.1f}%"
        )
        print(f"     • Average Cost per Model: ${total_cost / len(models_config):.2f}")

        print("\n  🏗️  Cost by Environment:")
        for env, cost in cost_by_environment.items():
            print(
                f"     • {env.capitalize()}: ${cost:.2f} ({(cost / total_cost) * 100:.1f}%)"
            )

        print("\n  🏆 Top 3 Cost Drivers:")
        sorted_models = sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True)
        for i, (model, cost) in enumerate(sorted_models[:3], 1):
            print(f"     {i}. {model}: ${cost:.2f} ({(cost / total_cost) * 100:.1f}%)")

        return cost_aggregator, total_cost, cost_by_model

    except ImportError as e:
        print(f"❌ Required package not available: {e}")
        return None, 0.0, {}
    except Exception as e:
        print(f"❌ Cost aggregation failed: {e}")
        return None, 0.0, {}


def demonstrate_cost_optimization_recommendations(cost_aggregator):
    """Demonstrate cost optimization recommendations."""
    print("\n🔧 Cost Optimization Recommendations:")
    print()

    if not cost_aggregator:
        print("❌ Cost aggregator not available")
        return

    try:
        # Get optimization recommendations
        recommendations = cost_aggregator.get_cost_optimization_recommendations()

        if not recommendations:
            print("  ℹ️  No optimization recommendations available (insufficient data)")
            return

        print(f"  📋 Found {len(recommendations)} optimization opportunities:")
        print()

        total_potential_savings = 0.0

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. 🎯 {rec.title}")
            print(f"     💡 {rec.description}")
            print(f"     💰 Potential Savings: ${rec.potential_savings:.2f}")
            print(f"     ⚡ Effort Level: {rec.effort_level}")
            print(f"     ⚠️  Risk Level: {rec.risk_level}")
            print(f"     📊 Priority Score: {rec.priority_score:.1f}/100")

            if rec.implementation_steps:
                print("     🔧 Implementation Steps:")
                for step_num, step in enumerate(rec.implementation_steps, 1):
                    print(f"        {step_num}. {step}")

            if rec.affected_models:
                print(f"     🎯 Affected Models: {', '.join(rec.affected_models)}")

            total_potential_savings += rec.potential_savings
            print()

        print(f"  💰 Total Potential Monthly Savings: ${total_potential_savings:.2f}")

        # Demonstrate monthly summary
        monthly_summary = cost_aggregator.get_monthly_cost_summary()

        print("\n  📈 Monthly Summary:")
        print(f"     • Current Total: ${monthly_summary.total_cost:.2f}")
        print(
            f"     • Optimized Estimate: ${monthly_summary.total_cost - total_potential_savings:.2f}"
        )
        print(
            f"     • Savings Percentage: {(total_potential_savings / monthly_summary.total_cost) * 100:.1f}%"
        )
        print(f"     • Model Count: {monthly_summary.model_count}")
        print(f"     • Prediction Volume: {monthly_summary.prediction_volume:,}")

    except Exception as e:
        print(f"❌ Optimization recommendations failed: {e}")


def demonstrate_pricing_optimization():
    """Demonstrate pricing tier optimization and volume discounts."""
    print("\n💎 Pricing Tier Optimization:")
    print()

    try:
        from genops.providers.arize_pricing import (
            ArizePricingCalculator,
            PricingTier,
            quick_monthly_estimate,  # noqa: F401
        )

        # Test different pricing tiers for cost optimization
        usage_scenario = {
            "models": 8,
            "predictions_per_model": 125000,  # 1M total predictions
            "quality_checks_per_model": 50,
            "alerts_per_model": 6,
            "dashboards": 12,
        }

        print("  📊 Usage Scenario:")
        print(f"     • Models: {usage_scenario['models']}")
        print(
            f"     • Predictions per Model: {usage_scenario['predictions_per_model']:,}"
        )
        print(
            f"     • Total Monthly Predictions: {usage_scenario['models'] * usage_scenario['predictions_per_model']:,}"
        )
        print(
            f"     • Quality Checks per Model: {usage_scenario['quality_checks_per_model']}"
        )
        print(f"     • Alerts per Model: {usage_scenario['alerts_per_model']}")
        print(f"     • Dashboards: {usage_scenario['dashboards']}")
        print()

        # Compare pricing tiers
        print("  💰 Pricing Tier Comparison:")

        tiers_to_compare = [
            PricingTier.STARTER,
            PricingTier.PROFESSIONAL,
            PricingTier.ENTERPRISE,
        ]
        tier_costs = {}

        for tier in tiers_to_compare:
            calculator = ArizePricingCalculator(
                tier=tier, region="us-east-1", currency="USD"
            )

            estimate = calculator.estimate_monthly_cost(
                models=usage_scenario["models"],
                predictions_per_model=usage_scenario["predictions_per_model"],
                quality_checks_per_model=usage_scenario["quality_checks_per_model"],
                alerts_per_model=usage_scenario["alerts_per_model"],
                dashboards=usage_scenario["dashboards"],
                optimize_for_cost=True,
            )

            tier_costs[tier] = estimate.total_estimated_cost

            print(
                f"     • {tier.value.capitalize()}: ${estimate.total_estimated_cost:.2f}"
            )
            print(
                f"       - Recommended: {'✅' if estimate.recommended_tier == tier else '❌'}"
            )
            print(f"       - Potential Savings: ${estimate.potential_savings:.2f}")

        # Find optimal tier
        optimal_tier = min(tier_costs.items(), key=lambda x: x[1])
        print(
            f"\n  🏆 Optimal Tier: {optimal_tier[0].value.capitalize()} (${optimal_tier[1]:.2f})"
        )

        # Calculate savings from tier optimization
        starter_cost = tier_costs[PricingTier.STARTER]
        optimal_cost = optimal_tier[1]
        savings = starter_cost - optimal_cost

        if savings > 0:
            print(
                f"     • Savings vs Starter: ${savings:.2f} ({(savings / starter_cost) * 100:.1f}%)"
            )

        # Demonstrate volume discount analysis
        print("\n  📈 Volume Discount Analysis:")

        enterprise_calculator = ArizePricingCalculator(tier=PricingTier.ENTERPRISE)

        volume_levels = [10000, 50000, 100000, 500000, 1000000, 2000000]

        print("     Prediction Volume → Cost per Prediction → Discount Tier")
        for volume in volume_levels:
            pricing_breakdown = enterprise_calculator.calculate_prediction_logging_cost(
                prediction_count=volume, time_period_days=30
            )

            discount_tier = enterprise_calculator.get_volume_discount_tier(volume)

            print(
                f"     {volume:,} → ${pricing_breakdown.effective_rate:.6f} → {discount_tier.tier_name} ({discount_tier.discount_percentage:.0f}%)"
            )

        return optimal_tier[0], optimal_cost

    except ImportError as e:
        print(f"❌ Pricing optimization requires additional packages: {e}")
        return None, 0.0
    except Exception as e:
        print(f"❌ Pricing optimization failed: {e}")
        return None, 0.0


def demonstrate_cost_forecasting(cost_aggregator):
    """Demonstrate cost forecasting and budget planning."""
    print("\n🔮 Cost Forecasting & Budget Planning:")
    print()

    if not cost_aggregator:
        print("❌ Cost aggregator not available")
        return

    try:
        # Generate 3-month forecast
        forecast = cost_aggregator.generate_cost_forecast(forecast_months=3)

        print("  📈 3-Month Cost Forecast:")
        print(f"     • Period: {forecast.forecast_period}")
        print(f"     • Forecasted Cost: ${forecast.forecasted_cost:.2f}")
        print(
            f"     • Confidence Range: ${forecast.confidence_interval[0]:.2f} - ${forecast.confidence_interval[1]:.2f}"
        )
        print(f"     • Budget Recommendation: ${forecast.budget_recommendation:.2f}")

        print("\n  📋 Key Assumptions:")
        for assumption in forecast.key_assumptions:
            print(f"     • {assumption}")

        print("\n  ⚠️  Risk Factors:")
        for risk in forecast.risk_factors:
            print(f"     • {risk}")

        # Demonstrate monthly cost trending
        print("\n  📊 Cost Trend Analysis:")

        # Simulate historical data for demonstration
        historical_months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
        base_cost = 150.0

        print("     Month      Cost      Change    Trend")
        print("     -----      ----      ------    -----")

        for i, month in enumerate(historical_months):
            # Simulate cost growth with some variation
            month_cost = base_cost * (1 + i * 0.08 + random.uniform(-0.02, 0.04))

            if i == 0:
                change = 0.0
                trend = "→"
            else:
                prev_cost = base_cost * (
                    1 + (i - 1) * 0.08 + 0.02
                )  # Approximate previous
                change = ((month_cost - prev_cost) / prev_cost) * 100
                trend = "↗" if change > 2 else "↘" if change < -2 else "→"

            print(f"     {month}    ${month_cost:.2f}    {change:+.1f}%     {trend}")

        # Budget planning recommendations
        print("\n  💰 Budget Planning Recommendations:")
        current_monthly = base_cost * 1.3  # Current estimated monthly

        print(f"     • Current Monthly Average: ${current_monthly:.2f}")
        print(
            f"     • Recommended Q4 Budget: ${current_monthly * 3 * 1.15:.2f} (+15% buffer)"
        )
        print(
            f"     • Annual Budget Estimate: ${current_monthly * 12 * 1.20:.2f} (+20% growth)"
        )

        # Cost optimization timeline
        print("\n  🗓️  Optimization Timeline:")
        print("     • Week 1-2: Implement high-priority, low-effort optimizations")
        print("     • Week 3-4: Configure sampling and alert consolidation")
        print("     • Month 2: Review tier optimization and volume discounts")
        print("     • Month 3: Evaluate environment consolidation opportunities")
        print("     • Quarterly: Review and adjust budget allocations")

    except Exception as e:
        print(f"❌ Cost forecasting failed: {e}")


def demonstrate_dynamic_cost_monitoring():
    """Demonstrate dynamic cost-aware monitoring strategies."""
    print("\n⚡ Dynamic Cost-Aware Monitoring:")
    print()

    try:
        from genops.providers.arize import GenOpsArizeAdapter

        # Create adapter with cost monitoring
        adapter = GenOpsArizeAdapter(
            team="ml-platform",
            project="dynamic-monitoring",
            daily_budget_limit=100.0,
            enable_cost_alerts=True,
        )

        print("  🎛️  Dynamic Monitoring Strategy:")
        print("     Automatically adjust monitoring behavior based on cost usage")
        print()

        # Simulate different cost scenarios
        scenarios = [
            ("Morning Peak", 15.0, "Normal sampling (10%)"),
            ("Mid-day High", 45.0, "Reduced sampling (5%)"),
            ("Afternoon Critical", 78.0, "Minimal sampling (1%)"),
            ("Evening Recovery", 55.0, "Moderate sampling (3%)"),
        ]

        print("  📊 Adaptive Sampling Based on Budget Usage:")
        print("     Time Period       Usage    Budget %    Sampling Rate")
        print("     -----------       -----    --------    -------------")

        for period, usage, _strategy in scenarios:
            budget_pct = (usage / adapter.daily_budget_limit) * 100

            # Determine sampling rate based on budget usage
            if budget_pct < 30:
                sampling_rate = "10%"
                sampling_color = "🟢"
            elif budget_pct < 60:
                sampling_rate = "5%"
                sampling_color = "🟡"
            elif budget_pct < 80:
                sampling_rate = "2%"
                sampling_color = "🟠"
            else:
                sampling_rate = "1%"
                sampling_color = "🔴"

            print(
                f"     {period:<15}   ${usage:>5.1f}    {budget_pct:>6.1f}%    {sampling_color} {sampling_rate}"
            )

        # Demonstrate cost-aware decision making
        print("\n  🤖 Automated Cost Controls:")

        current_usage = 75.0
        remaining_budget = adapter.daily_budget_limit - current_usage

        print(f"     • Current Usage: ${current_usage:.1f}")
        print(f"     • Remaining Budget: ${remaining_budget:.1f}")

        # Simulate decision logic
        if remaining_budget > 20:
            decision = "Continue normal monitoring"
            color = "🟢"
        elif remaining_budget > 10:
            decision = "Reduce monitoring frequency by 50%"
            color = "🟡"
        elif remaining_budget > 5:
            decision = "Switch to critical alerts only"
            color = "🟠"
        else:
            decision = "Suspend non-critical monitoring"
            color = "🔴"

        print(f"     • Decision: {color} {decision}")

        # Cost optimization automation
        print("\n  ⚙️  Automated Optimizations:")
        print("     • Prediction sampling: Automatically adjust based on budget")
        print("     • Alert consolidation: Merge similar alerts during high usage")
        print("     • Dashboard caching: Cache expensive analytics queries")
        print("     • Quality check scheduling: Defer non-critical checks")
        print("     • Environment throttling: Reduce dev/staging monitoring")

        return True

    except Exception as e:
        print(f"❌ Dynamic monitoring demo failed: {e}")
        return False


def demonstrate_enterprise_cost_patterns():
    """Demonstrate enterprise cost management patterns."""
    print("\n🏢 Enterprise Cost Management Patterns:")
    print()

    # Multi-team cost allocation
    print("  👥 Multi-Team Cost Allocation:")

    teams_config = [
        ("fraud-team", 3, 200000, 85.50),
        ("credit-team", 2, 150000, 65.25),
        ("risk-team", 4, 300000, 125.75),
        ("ops-team", 1, 50000, 22.15),
    ]

    total_cost = sum(cost for _, _, _, cost in teams_config)

    print("     Team          Models    Predictions    Monthly Cost    % of Total")
    print("     ----          ------    -----------    ------------    ----------")

    for team, models, predictions, cost in teams_config:
        pct_total = (cost / total_cost) * 100
        print(
            f"     {team:<12}  {models:>6}    {predictions:>11,}    ${cost:>10.2f}    {pct_total:>8.1f}%"
        )

    print(
        f"     {'TOTAL':<12}  {sum(m for _, m, _, _ in teams_config):>6}    {sum(p for _, _, p, _ in teams_config):>11,}    ${total_cost:>10.2f}    {'100.0':>8}%"
    )

    # Cost center attribution
    print("\n  🏗️  Cost Center Attribution:")

    cost_centers = [
        ("ML Infrastructure", 45.0),
        ("Data Platform", 35.0),
        ("Product Engineering", 15.0),
        ("Research & Development", 5.0),
    ]

    for center, pct in cost_centers:
        allocated_cost = (pct / 100) * total_cost
        print(f"     • {center}: ${allocated_cost:.2f} ({pct:.0f}%)")

    # Budget governance
    print("\n  📊 Budget Governance:")
    print(f"     • Monthly Budget: ${total_cost * 1.1:.2f} (10% buffer)")
    print("     • Quarterly Review: Adjust allocations based on usage trends")
    print("     • Annual Planning: Forecast growth and optimization opportunities")
    print("     • Cost Controls: Automated alerts at 80% budget utilization")
    print("     • Chargeback Model: Teams charged based on actual usage")

    # Compliance and audit
    print("\n  🔍 Compliance & Audit:")
    print("     • Cost Attribution: All operations tagged with team/project/customer")
    print("     • Audit Trail: Complete history of monitoring costs and decisions")
    print("     • Policy Compliance: Automated enforcement of budget policies")
    print("     • Reporting: Monthly cost reports with optimization recommendations")


def print_cost_optimization_summary():
    """Print comprehensive cost optimization summary."""
    print("\n📋 Cost Optimization Summary:")
    print()

    print("  🎯 Key Strategies Demonstrated:")
    print("     1. Multi-model cost aggregation for comprehensive visibility")
    print("     2. Automated optimization recommendations with priority scoring")
    print("     3. Pricing tier optimization for maximum cost efficiency")
    print("     4. Volume discount analysis for large-scale operations")
    print("     5. Cost forecasting and budget planning for financial control")
    print("     6. Dynamic cost-aware monitoring with automatic adjustments")
    print("     7. Enterprise-grade cost allocation and governance patterns")
    print()

    print("  💰 Potential Cost Savings:")
    print("     • Tier Optimization: 15-30% reduction in monthly costs")
    print("     • Volume Discounts: 10-40% reduction at scale")
    print("     • Dynamic Sampling: 20-50% reduction during peak usage")
    print("     • Alert Consolidation: 5-15% reduction in management overhead")
    print("     • Environment Optimization: 10-25% reduction in non-prod costs")
    print()

    print("  🚀 Implementation Roadmap:")
    print("     Week 1: Implement basic cost tracking and monitoring")
    print("     Week 2: Configure budget limits and cost alerts")
    print("     Week 3: Enable dynamic sampling and optimization")
    print("     Month 2: Analyze usage patterns and optimize pricing tier")
    print("     Month 3: Implement enterprise governance and allocation")
    print("     Ongoing: Monitor trends and adjust optimization strategies")


def main():
    """Main cost optimization demonstration."""
    print_header()

    # Step 1: Multi-model cost aggregation
    cost_aggregator, total_cost, cost_by_model = demonstrate_cost_aggregation()

    if not cost_aggregator:
        print("❌ Cost aggregation failed. Cannot continue with optimization demos.")
        return 1

    # Step 2: Cost optimization recommendations
    demonstrate_cost_optimization_recommendations(cost_aggregator)

    # Step 3: Pricing optimization
    optimal_tier, optimal_cost = demonstrate_pricing_optimization()

    # Step 4: Cost forecasting
    demonstrate_cost_forecasting(cost_aggregator)

    # Step 5: Dynamic cost monitoring
    demonstrate_dynamic_cost_monitoring()

    # Step 6: Enterprise patterns
    demonstrate_enterprise_cost_patterns()

    # Summary
    print_cost_optimization_summary()

    print("\n" + "=" * 70)
    print("💰 Cost Optimization Example Completed Successfully!")
    print("=" * 70)
    print("🎉 You now have comprehensive cost intelligence for Arize AI monitoring!")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
