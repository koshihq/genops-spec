#!/usr/bin/env python3
"""
PostHog Cost Optimization Example

This example demonstrates comprehensive cost optimization strategies for PostHog
with GenOps governance, including volume discounts, usage pattern analysis,
budget forecasting, and intelligent cost reduction recommendations.

Usage:
    python cost_optimization.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your-project-api-key"
"""

import os
import random
from decimal import Decimal


def main():
    """Demonstrate PostHog cost optimization with GenOps intelligence."""
    print("💡 PostHog + GenOps Cost Optimization Example")
    print("=" * 55)

    # Initialize adapter
    try:
        from genops.providers.posthog import GenOpsPostHogAdapter

        adapter = GenOpsPostHogAdapter(
            posthog_api_key=os.getenv("POSTHOG_API_KEY"),
            team=os.getenv("GENOPS_TEAM", "cost-optimization-team"),
            project=os.getenv("GENOPS_PROJECT", "analytics-optimization"),
            environment="production",
            daily_budget_limit=150.0,
            enable_governance=True,
            governance_policy="advisory",
        )

        print("✅ Cost optimization adapter initialized")

    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return

    # Demo 1: Current Usage Analysis
    print("\n📊 Analyzing current PostHog usage costs...")
    analyze_current_usage(adapter)

    # Demo 2: Volume Discount Analysis
    print("\n📈 Volume Discount Optimization Analysis")
    print("-" * 45)
    analyze_volume_discounts(adapter)

    # Demo 3: Usage Pattern Optimization
    print("\n⚡ Usage Pattern Optimization Strategies")
    print("-" * 45)
    demonstrate_usage_optimization(adapter)

    # Demo 4: Budget Forecasting
    print("\n📅 Budget Forecasting & Planning")
    print("-" * 35)
    demonstrate_budget_forecasting(adapter)

    # Demo 5: Cost-Aware Analytics
    print("\n💰 Cost-Aware Analytics Implementation")
    print("-" * 42)
    demonstrate_cost_aware_analytics(adapter)

    # Demo 6: Multi-Tier Optimization
    print("\n🎯 Multi-Tier Cost Optimization")
    print("-" * 35)
    demonstrate_multi_tier_optimization(adapter)

    print("\n✅ Cost optimization analysis completed!")


def analyze_current_usage(adapter):
    """Analyze current PostHog usage patterns and costs."""

    # Simulate current usage patterns
    usage_scenarios = [
        {
            "name": "web_analytics",
            "monthly_events": 850000,
            "identified_ratio": 0.3,
            "feature_flag_requests": 120000,
            "session_recordings": 8000,
        },
        {
            "name": "mobile_analytics",
            "monthly_events": 650000,
            "identified_ratio": 0.6,
            "feature_flag_requests": 95000,
            "session_recordings": 5500,
        },
        {
            "name": "api_analytics",
            "monthly_events": 420000,
            "identified_ratio": 0.9,
            "feature_flag_requests": 200000,
            "session_recordings": 1200,
        },
    ]

    calculator = PostHogCostCalculator()  # noqa: F821
    total_monthly_cost = Decimal("0")

    print("📋 Current Usage Breakdown:")

    for scenario in usage_scenarios:
        # Calculate costs for this usage pattern
        events = scenario["monthly_events"]
        identified = int(events * scenario["identified_ratio"])

        cost_result = calculator.calculate_session_cost(
            event_count=events,
            identified_events=identified,
            feature_flag_requests=scenario["feature_flag_requests"],
            session_recordings=scenario["session_recordings"],
        )

        total_monthly_cost += cost_result.total_cost

        print(f"\n  📊 {scenario['name'].replace('_', ' ').title()}:")
        print(f"     Monthly events: {events:,}")
        print(
            f"     Identified events: {identified:,} ({scenario['identified_ratio'] * 100:.1f}%)"
        )
        print(f"     Feature flag requests: {scenario['feature_flag_requests']:,}")
        print(f"     Session recordings: {scenario['session_recordings']:,}")
        print(f"     Monthly cost: ${cost_result.total_cost:.2f}")

        # Cost breakdown
        print("     Cost breakdown:")
        for component, cost in cost_result.cost_breakdown.items():
            if cost > 0:
                percentage = (cost / cost_result.total_cost) * 100
                print(
                    f"       {component.replace('_', ' ').title()}: ${cost:.2f} ({percentage:.1f}%)"
                )

    print("\n💰 Monthly Cost Summary:")
    print(f"  Total Cost: ${total_monthly_cost:.2f}")
    print(
        f"  Average Cost per Scenario: ${total_monthly_cost / len(usage_scenarios):.2f}"
    )

    # Cost efficiency metrics
    total_events = sum(s["monthly_events"] for s in usage_scenarios)
    cost_per_event = total_monthly_cost / total_events if total_events > 0 else 0
    events_per_dollar = 1 / cost_per_event if cost_per_event > 0 else 0

    print(f"  Cost per Event: ${cost_per_event:.6f}")
    print(f"  Events per Dollar: {events_per_dollar:,.0f}")

    return total_monthly_cost, usage_scenarios


def analyze_volume_discounts(adapter):
    """Analyze volume discount opportunities."""

    calculator = PostHogCostCalculator()  # noqa: F821

    # Test different monthly volumes
    volume_scenarios = [
        500000,  # Current usage level
        1000000,  # 2x growth
        2500000,  # 5x growth
        5000000,  # 10x growth
        10000000,  # 20x growth
        25000000,  # Significant scale
    ]

    print("📈 Volume Discount Analysis:")

    volume_results = []
    for volume in volume_scenarios:
        cost = calculator.calculate_event_cost(volume)
        cost_per_event = cost / volume if volume > 0 else 0

        volume_results.append(
            {"volume": volume, "cost": cost, "cost_per_event": cost_per_event}
        )

        # Format volume for display
        if volume >= 1000000:
            volume_display = f"{volume / 1000000:.1f}M"
        else:
            volume_display = f"{volume / 1000:.0f}K"

        print(
            f"  {volume_display:>8} events -> ${cost:>8.2f} (${cost_per_event:.6f}/event)"
        )

    # Calculate potential savings
    print("\n💰 Volume Discount Opportunities:")
    current_volume = 500000  # Assumed current usage
    current_cost_per_event = volume_results[0]["cost_per_event"]

    for result in volume_results[1:]:
        volume = result["volume"]
        cost_per_event = result["cost_per_event"]

        savings_per_event = current_cost_per_event - cost_per_event
        total_savings = savings_per_event * current_volume

        if savings_per_event > 0:
            volume_display = (
                f"{volume / 1000000:.1f}M"
                if volume >= 1000000
                else f"{volume / 1000:.0f}K"
            )
            savings_percent = (savings_per_event / current_cost_per_event) * 100

            print(
                f"  At {volume_display:>8} volume: {savings_percent:>5.1f}% cheaper per event"
            )
            print(
                f"            Monthly savings on current usage: ${total_savings:>6.2f}"
            )

    # Get recommendations from the calculator
    recommendations = calculator.get_volume_discount_recommendations(current_volume)

    if recommendations:
        print("\n🎯 Volume Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['optimization_type']}")
            print(f"     Current tier: {rec['current_tier']}")
            print(f"     Next tier: {rec['next_tier']}")
            print(f"     Events needed: {rec['events_needed']:,} more")
            print(
                f"     Potential savings: ${rec['potential_savings_per_month']:.2f}/month"
            )
            print(f"     Priority score: {rec['priority_score']:.1f}/100")
            print()


def demonstrate_usage_optimization(adapter):
    """Demonstrate usage pattern optimization strategies."""

    print("⚡ Usage Pattern Optimization Strategies:")

    # Strategy 1: Event Sampling
    print("\n  1. Intelligent Event Sampling")
    print("     ─" * 30)

    sampling_strategies = [
        {
            "name": "High-frequency events",
            "current_rate": 100,
            "optimal_rate": 10,
            "impact": "minimal",
        },
        {
            "name": "Debug/dev events",
            "current_rate": 100,
            "optimal_rate": 5,
            "impact": "none",
        },
        {
            "name": "Page view events",
            "current_rate": 100,
            "optimal_rate": 50,
            "impact": "low",
        },
        {
            "name": "User interaction events",
            "current_rate": 100,
            "optimal_rate": 90,
            "impact": "none",
        },
    ]

    total_savings_sampling = Decimal("0")

    for strategy in sampling_strategies:
        current_events = 10000  # Monthly events for this category
        optimized_events = int(current_events * strategy["optimal_rate"] / 100)
        event_savings = current_events - optimized_events

        # Estimate cost savings (using average cost per event)
        avg_cost_per_event = Decimal("0.00005")  # PostHog average
        cost_savings = event_savings * avg_cost_per_event
        total_savings_sampling += cost_savings

        print(
            f"     {strategy['name']:25} -> {strategy['optimal_rate']:3d}% sampling, "
            f"${cost_savings:6.2f} savings ({strategy['impact']} impact)"
        )

    print(f"     {'Total Sampling Savings':25} -> ${total_savings_sampling:8.2f}/month")

    # Strategy 2: Feature Flag Optimization
    print("\n  2. Feature Flag Request Optimization")
    print("     ─" * 35)

    flag_optimizations = [
        {"strategy": "Local evaluation caching", "savings_pct": 60, "effort": "Medium"},
        {"strategy": "Batch flag evaluations", "savings_pct": 25, "effort": "Low"},
        {"strategy": "Smart flag refresh logic", "savings_pct": 15, "effort": "High"},
        {"strategy": "Remove unused flags", "savings_pct": 10, "effort": "Low"},
    ]

    base_flag_cost = Decimal("15.00")  # Monthly feature flag cost
    total_savings_flags = Decimal("0")

    for opt in flag_optimizations:
        savings = base_flag_cost * Decimal(str(opt["savings_pct"] / 100))
        total_savings_flags += savings

        print(
            f"     {opt['strategy']:25} -> {opt['savings_pct']:3d}% reduction, "
            f"${savings:6.2f} savings ({opt['effort']} effort)"
        )

    print(f"     {'Total Flag Savings':25} -> ${total_savings_flags:8.2f}/month")

    # Strategy 3: Session Recording Optimization
    print("\n  3. Session Recording Optimization")
    print("     ─" * 30)

    recording_strategies = [
        {
            "strategy": "Record high-value sessions only",
            "savings_pct": 40,
            "quality_impact": "Low",
        },
        {
            "strategy": "Reduce recording quality",
            "savings_pct": 20,
            "quality_impact": "Medium",
        },
        {
            "strategy": "Intelligent session sampling",
            "savings_pct": 30,
            "quality_impact": "Low",
        },
        {
            "strategy": "Shorter retention periods",
            "savings_pct": 15,
            "quality_impact": "None",
        },
    ]

    base_recording_cost = Decimal("25.00")  # Monthly recording cost
    total_savings_recordings = Decimal("0")

    for strategy in recording_strategies:
        savings = base_recording_cost * Decimal(str(strategy["savings_pct"] / 100))
        total_savings_recordings += savings

        print(
            f"     {strategy['strategy']:25} -> {strategy['savings_pct']:3d}% reduction, "
            f"${savings:6.2f} savings ({strategy['quality_impact']} impact)"
        )

    print(
        f"     {'Total Recording Savings':25} -> ${total_savings_recordings:8.2f}/month"
    )

    # Total optimization potential
    total_optimization_savings = (
        total_savings_sampling + total_savings_flags + total_savings_recordings
    )

    print(f"\n💡 Total Optimization Potential: ${total_optimization_savings:.2f}/month")

    # Implementation priority
    print("\n🎯 Implementation Priority:")
    priorities = [
        ("Remove unused feature flags", "Low effort, immediate savings"),
        ("Implement event sampling", "Medium effort, high impact"),
        ("Optimize session recording", "Medium effort, good ROI"),
        ("Add local flag evaluation", "High effort, long-term benefit"),
    ]

    for i, (action, description) in enumerate(priorities, 1):
        print(f"  {i}. {action}")
        print(f"     → {description}")


def demonstrate_budget_forecasting(adapter):
    """Demonstrate budget forecasting and planning."""

    print("📅 Budget Forecasting & Planning:")

    # Current usage baseline
    current_monthly_cost = Decimal("125.50")

    # Growth scenarios
    growth_scenarios = [
        {"name": "Conservative Growth", "monthly_growth": 0.05, "period_months": 12},
        {"name": "Moderate Growth", "monthly_growth": 0.10, "period_months": 12},
        {"name": "Aggressive Growth", "monthly_growth": 0.20, "period_months": 12},
        {"name": "Startup Scale", "monthly_growth": 0.35, "period_months": 12},
    ]

    print("\n📊 Growth Scenario Analysis:")
    print(f"   Current monthly cost: ${current_monthly_cost}")
    print()

    for scenario in growth_scenarios:
        print(
            f"  📈 {scenario['name']} ({scenario['monthly_growth'] * 100:.0f}% monthly growth):"
        )

        cost = current_monthly_cost
        total_cost_12_months = Decimal("0")

        # Calculate monthly costs
        monthly_costs = []
        for month in range(scenario["period_months"]):
            if month > 0:
                cost *= 1 + Decimal(str(scenario["monthly_growth"]))
            monthly_costs.append(cost)
            total_cost_12_months += cost

        # Show key milestones
        cost_3_months = monthly_costs[2] if len(monthly_costs) > 2 else cost
        cost_6_months = monthly_costs[5] if len(monthly_costs) > 5 else cost
        cost_12_months = monthly_costs[11] if len(monthly_costs) > 11 else cost

        print(f"     3 months:  ${cost_3_months:8.2f}")
        print(f"     6 months:  ${cost_6_months:8.2f}")
        print(f"     12 months: ${cost_12_months:8.2f}")
        print(f"     Total year: ${total_cost_12_months:8.2f}")

        # Budget recommendations
        recommended_annual_budget = total_cost_12_months * Decimal("1.2")  # 20% buffer
        print(f"     Recommended annual budget: ${recommended_annual_budget:8.2f}")
        print()

    # Seasonal variations
    print("  📅 Seasonal Variation Considerations:")
    seasonal_factors = [
        {"period": "Q1 (Jan-Mar)", "factor": 0.9, "reason": "Post-holiday dip"},
        {"period": "Q2 (Apr-Jun)", "factor": 1.1, "reason": "Spring growth"},
        {"period": "Q3 (Jul-Sep)", "factor": 0.95, "reason": "Summer slowdown"},
        {"period": "Q4 (Oct-Dec)", "factor": 1.25, "reason": "Holiday peak"},
    ]

    for factor_info in seasonal_factors:
        seasonal_cost = current_monthly_cost * Decimal(str(factor_info["factor"]))
        variation = (factor_info["factor"] - 1) * 100
        print(
            f"     {factor_info['period']:15} -> ${seasonal_cost:7.2f} ({variation:+.0f}%) - {factor_info['reason']}"
        )

    # Budget alerting thresholds
    print("\n🚨 Recommended Budget Alert Thresholds:")
    alert_thresholds = [
        (
            "Daily Warning",
            current_monthly_cost / 30 * Decimal("1.5"),
            "Monitor usage spike",
        ),
        (
            "Weekly Caution",
            current_monthly_cost / 4 * Decimal("1.3"),
            "Review usage patterns",
        ),
        (
            "Monthly Alert",
            current_monthly_cost * Decimal("1.2"),
            "Budget variance check",
        ),
        (
            "Emergency Stop",
            current_monthly_cost * Decimal("2.0"),
            "Immediate investigation",
        ),
    ]

    for alert_name, threshold, action in alert_thresholds:
        print(f"  {alert_name:15} -> ${threshold:7.2f} - {action}")


def demonstrate_cost_aware_analytics(adapter):
    """Demonstrate cost-aware analytics implementation."""

    print("💰 Cost-Aware Analytics Implementation:")

    # Simulate cost-aware decision making
    print("\n  📊 Dynamic Cost-Based Analytics Strategies:")

    # Strategy 1: Tiered event tracking
    print("\n     1. Tiered Event Tracking by Importance")
    event_tiers = [
        {
            "tier": "Critical",
            "sample_rate": 100,
            "events": ["conversion", "signup", "payment"],
        },
        {
            "tier": "Important",
            "sample_rate": 80,
            "events": ["feature_use", "page_view", "click"],
        },
        {
            "tier": "Nice-to-have",
            "sample_rate": 20,
            "events": ["hover", "scroll", "focus"],
        },
        {"tier": "Debug", "sample_rate": 5, "events": ["debug", "trace", "verbose"]},
    ]

    total_cost_savings = Decimal("0")
    base_events_per_tier = 25000  # Monthly events per tier
    base_cost_per_event = Decimal("0.00005")

    for tier_info in event_tiers:
        full_cost = base_events_per_tier * base_cost_per_event
        sampled_events = base_events_per_tier * tier_info["sample_rate"] / 100
        actual_cost = sampled_events * base_cost_per_event
        cost_savings = full_cost - actual_cost
        total_cost_savings += cost_savings

        print(
            f"        {tier_info['tier']:12} -> {tier_info['sample_rate']:3d}% sampling, "
            f"${actual_cost:6.2f} cost, ${cost_savings:6.2f} saved"
        )

    print(f"        {'Total Savings':12} -> ${total_cost_savings:28.2f}")

    # Strategy 2: Budget-constrained analytics
    print("\n     2. Budget-Constrained Analytics Sessions")

    with adapter.track_analytics_session(
        session_name="budget_aware_analytics",
        budget_limit=10.0,
        cost_optimization_enabled=True,
    ) as session:
        # Simulate intelligent event prioritization
        high_priority_events = [
            ("user_conversion", {"value": 299.0, "source": "organic"}),
            ("feature_adoption", {"feature": "premium", "success": True}),
            ("error_critical", {"error": "payment_failed", "severity": "high"}),
        ]

        medium_priority_events = [
            ("page_interaction", {"element": "cta_button", "location": "header"}),
            ("content_engagement", {"duration": 45, "scroll_depth": 0.8}),
            ("navigation_flow", {"from": "/pricing", "to": "/signup"}),
        ]

        low_priority_events = [
            ("ui_interaction", {"element": "tooltip", "action": "hover"}),
            ("performance_metric", {"load_time": 1.2, "ttfb": 200}),
            ("debug_trace", {"component": "analytics", "level": "info"}),
        ]

        # Process events with cost awareness
        session_cost = Decimal("0")
        events_processed = 0

        # Always process high priority
        for event_name, properties in high_priority_events:
            result = adapter.capture_event_with_governance(
                event_name=event_name,
                properties=properties,
                distinct_id=f"cost_aware_user_{events_processed}",
                is_identified=True,
                session_id=session.session_id,
            )
            session_cost += Decimal(str(result["cost"]))
            events_processed += 1
            print(f"        High priority '{event_name}': ${result['cost']:.6f}")

        # Process medium priority if budget allows
        budget_remaining = Decimal("10.0") - session_cost
        for event_name, properties in medium_priority_events:
            estimated_cost = Decimal("0.000198")  # Identified event cost
            if budget_remaining >= estimated_cost:
                result = adapter.capture_event_with_governance(
                    event_name=event_name,
                    properties=properties,
                    distinct_id=f"cost_aware_user_{events_processed}",
                    is_identified=True,
                    session_id=session.session_id,
                )
                session_cost += Decimal(str(result["cost"]))
                budget_remaining -= Decimal(str(result["cost"]))
                events_processed += 1
                print(f"        Medium priority '{event_name}': ${result['cost']:.6f}")
            else:
                print(f"        Medium priority '{event_name}': Skipped (budget)")

        # Process low priority with sampling if budget allows
        for event_name, properties in low_priority_events:
            if (
                budget_remaining >= Decimal("0.00005") and random.random() < 0.3
            ):  # 30% sampling
                result = adapter.capture_event_with_governance(
                    event_name=event_name,
                    properties=properties,
                    distinct_id=f"cost_aware_user_{events_processed}",
                    is_identified=False,  # Anonymous to save cost
                    session_id=session.session_id,
                )
                session_cost += Decimal(str(result["cost"]))
                budget_remaining -= Decimal(str(result["cost"]))
                events_processed += 1
                print(
                    f"        Low priority '{event_name}': ${result['cost']:.6f} (sampled)"
                )

        print(
            f"        Session summary: {events_processed} events, ${session_cost:.4f} total"
        )
        print(
            f"        Budget utilized: {((10.0 - float(budget_remaining)) / 10.0) * 100:.1f}%"
        )


def demonstrate_multi_tier_optimization(adapter):
    """Demonstrate multi-tier cost optimization strategies."""

    print("🎯 Multi-Tier Cost Optimization:")

    # Define customer tiers with different optimization strategies
    customer_tiers = [
        {
            "tier": "Free",
            "monthly_budget": 0,  # Free tier usage limits
            "optimization": "maximum",
            "sample_rates": {"events": 10, "flags": 50, "recordings": 0},
        },
        {
            "tier": "Starter",
            "monthly_budget": 25,
            "optimization": "aggressive",
            "sample_rates": {"events": 50, "flags": 80, "recordings": 20},
        },
        {
            "tier": "Professional",
            "monthly_budget": 100,
            "optimization": "balanced",
            "sample_rates": {"events": 85, "flags": 95, "recordings": 70},
        },
        {
            "tier": "Enterprise",
            "monthly_budget": 500,
            "optimization": "minimal",
            "sample_rates": {"events": 100, "flags": 100, "recordings": 100},
        },
    ]

    print("\n  📊 Tier-Based Optimization Strategies:")

    base_usage = {"events": 50000, "flags": 10000, "recordings": 1000}

    base_costs = {
        "events": Decimal("2.50"),
        "flags": Decimal("0.50"),
        "recordings": Decimal("7.10"),
    }

    for tier_info in customer_tiers:
        tier_name = tier_info["tier"]
        budget = tier_info["monthly_budget"]
        sample_rates = tier_info["sample_rates"]

        print(f"\n     🏷️  {tier_name} Tier (${budget}/month budget):")

        total_cost = Decimal("0")
        total_savings = Decimal("0")

        for usage_type, base_cost in base_costs.items():
            sample_rate = sample_rates[usage_type]
            optimized_cost = base_cost * Decimal(str(sample_rate / 100))
            savings = base_cost - optimized_cost

            total_cost += optimized_cost
            total_savings += savings

            usage_count = int(base_usage[usage_type] * sample_rate / 100)

            print(
                f"        {usage_type.capitalize():12} -> {sample_rate:3d}% sampling, "
                f"{usage_count:6,} items, ${optimized_cost:6.2f} cost"
            )

        budget_utilization = (float(total_cost) / budget * 100) if budget > 0 else 0

        print(f"        {'Total Cost':12} -> ${total_cost:18.2f}")
        print(f"        {'Savings':12} -> ${total_savings:18.2f}")

        if budget > 0:
            print(f"        {'Budget Usage':12} -> {budget_utilization:17.1f}%")
        else:
            print(f"        {'Budget Usage':12} -> {'Free tier limits':>17}")

    # ROI Analysis
    print("\n  💡 Optimization ROI Analysis:")

    roi_scenarios = [
        {
            "optimization": "Event Sampling",
            "implementation_hours": 8,
            "monthly_savings": 15.75,
            "maintenance_hours_monthly": 1,
        },
        {
            "optimization": "Smart Feature Flags",
            "implementation_hours": 16,
            "monthly_savings": 8.50,
            "maintenance_hours_monthly": 2,
        },
        {
            "optimization": "Recording Optimization",
            "implementation_hours": 12,
            "monthly_savings": 22.30,
            "maintenance_hours_monthly": 0.5,
        },
        {
            "optimization": "Tier-Based Analytics",
            "implementation_hours": 24,
            "monthly_savings": 45.80,
            "maintenance_hours_monthly": 3,
        },
    ]

    developer_hourly_rate = 75  # USD per hour

    for scenario in roi_scenarios:
        impl_cost = scenario["implementation_hours"] * developer_hourly_rate
        monthly_maintenance_cost = (
            scenario["maintenance_hours_monthly"] * developer_hourly_rate
        )
        net_monthly_savings = scenario["monthly_savings"] - monthly_maintenance_cost
        payback_months = (
            impl_cost / net_monthly_savings if net_monthly_savings > 0 else float("inf")
        )
        annual_roi = (
            (net_monthly_savings * 12 - impl_cost) / impl_cost * 100
            if impl_cost > 0
            else 0
        )

        print(f"\n     {scenario['optimization']}:")
        print(f"       Implementation cost: ${impl_cost:,.0f}")
        print(f"       Monthly savings: ${scenario['monthly_savings']:.2f}")
        print(f"       Monthly maintenance: ${monthly_maintenance_cost:.2f}")
        print(f"       Net monthly benefit: ${net_monthly_savings:.2f}")
        print(f"       Payback period: {payback_months:.1f} months")
        print(f"       Annual ROI: {annual_roi:.0f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cost optimization example interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(
            "🐛 Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues"
        )
