#!/usr/bin/env python3
"""
PostHog Advanced Features Demo with GenOps Governance

This example demonstrates advanced PostHog features including feature flags,
session recordings, LLM analytics integration, A/B testing, and comprehensive
governance with cost intelligence and multi-tenant attribution.

Usage:
    python advanced_features.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your-project-api-key"
    export GENOPS_TEAM="your-team-name"
"""

import os
import random
import time
from datetime import datetime
from decimal import Decimal


def main():
    """Demonstrate advanced PostHog features with GenOps governance."""
    print("🚀 PostHog + GenOps Advanced Features Demo")
    print("=" * 55)

    # Initialize adapter
    try:
        from genops.providers.posthog import GenOpsPostHogAdapter

        adapter = GenOpsPostHogAdapter(
            posthog_api_key=os.getenv("POSTHOG_API_KEY"),
            team=os.getenv("GENOPS_TEAM", "advanced-features-team"),
            project=os.getenv("GENOPS_PROJECT", "advanced-analytics-demo"),
            environment="production",
            daily_budget_limit=200.0,
            enable_governance=True,
            governance_policy="enforced",
        )

        print("✅ Advanced PostHog adapter initialized")

    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return

    # Demo 1: Multi-tenant Feature Flag Management
    print("\n🚩 Multi-Tenant Feature Flag Management Demo")
    print("-" * 50)

    demonstrate_feature_flag_management(adapter)

    # Demo 2: LLM Analytics Integration
    print("\n🤖 LLM Analytics Integration Demo")
    print("-" * 40)

    demonstrate_llm_analytics(adapter)

    # Demo 3: Session Recording with Governance
    print("\n🎬 Session Recording & Analytics Demo")
    print("-" * 45)

    demonstrate_session_analytics(adapter)

    # Demo 4: A/B Testing with Cost Intelligence
    print("\n🧪 A/B Testing with Cost Intelligence Demo")
    print("-" * 48)

    demonstrate_ab_testing(adapter)

    # Demo 5: Multi-Customer Analytics
    print("\n🏢 Multi-Customer Analytics Governance Demo")
    print("-" * 50)

    demonstrate_multi_customer_analytics(adapter)

    # Demo 6: Real-time Dashboard Analytics
    print("\n📊 Real-time Dashboard Analytics Demo")
    print("-" * 42)

    demonstrate_dashboard_analytics(adapter)

    # Final summary
    print("\n💰 Comprehensive Cost & Governance Summary")
    print("=" * 50)

    display_final_summary(adapter)

    print("\n✅ Advanced features demo completed successfully!")


def demonstrate_feature_flag_management(adapter):
    """Demonstrate advanced feature flag management with governance."""

    # Multi-environment feature flag scenarios
    user_segments = ["free_tier", "premium", "enterprise", "beta_tester"]

    feature_flags = [
        {
            "flag": "new_dashboard_v3",
            "description": "Next generation dashboard interface",
            "rollout_percentage": 25,
            "target_segments": ["premium", "enterprise"],
        },
        {
            "flag": "ai_powered_insights",
            "description": "AI-generated analytics insights",
            "rollout_percentage": 10,
            "target_segments": ["enterprise", "beta_tester"],
        },
        {
            "flag": "advanced_filtering",
            "description": "Enhanced data filtering capabilities",
            "rollout_percentage": 50,
            "target_segments": ["premium", "enterprise"],
        },
        {
            "flag": "mobile_app_redesign",
            "description": "Redesigned mobile application UI",
            "rollout_percentage": 75,
            "target_segments": ["free_tier", "premium", "enterprise"],
        },
    ]

    print("🎯 Evaluating feature flags across user segments...")

    total_evaluations = 0
    total_cost = Decimal("0")

    for flag_config in feature_flags:
        flag_key = flag_config["flag"]

        print(f"\n  🚩 Feature: {flag_config['description']}")
        print(f"     Flag: {flag_key}")
        print(f"     Rollout: {flag_config['rollout_percentage']}%")

        # Evaluate flag for different user segments
        segment_results = {}
        for segment in user_segments:
            user_id = f"user_{segment}_{random.randint(1000, 9999)}"

            flag_value, metadata = adapter.evaluate_feature_flag_with_governance(
                flag_key=flag_key,
                distinct_id=user_id,
                properties={
                    "user_segment": segment,
                    "signup_date": "2024-01-15",
                    "plan_type": segment,
                    "region": random.choice(["us_east", "us_west", "eu", "asia"]),
                },
            )

            segment_results[segment] = {"enabled": flag_value, "cost": metadata["cost"]}
            total_evaluations += 1
            total_cost += Decimal(str(metadata["cost"]))

        # Display results
        for segment, result in segment_results.items():
            status = "✅ Enabled" if result["enabled"] else "❌ Disabled"
            print(f"     {segment:15} -> {status} (${result['cost']:.6f})")

    print("\n📊 Feature Flag Summary:")
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average cost per evaluation: ${total_cost / total_evaluations:.6f}")


def demonstrate_llm_analytics(adapter):
    """Demonstrate LLM analytics integration with PostHog."""

    print("🤖 Simulating LLM-powered product features...")

    # Simulate AI features in a product analytics context
    llm_features = [
        {
            "feature": "smart_insights",
            "prompt": "Generate insights from user behavior data",
            "model": "gpt-4-turbo",
            "context": "product_analytics",
        },
        {
            "feature": "automated_reports",
            "prompt": "Create weekly analytics report summary",
            "model": "gpt-3.5-turbo",
            "context": "business_intelligence",
        },
        {
            "feature": "anomaly_detection",
            "prompt": "Identify unusual patterns in user activity",
            "model": "claude-3-sonnet",
            "context": "data_science",
        },
        {
            "feature": "personalized_recommendations",
            "prompt": "Generate product recommendations for user",
            "model": "gpt-4-turbo",
            "context": "user_experience",
        },
    ]

    llm_total_cost = Decimal("0")

    for feature_config in llm_features:
        print(f"\n  🧠 LLM Feature: {feature_config['feature']}")

        # Simulate LLM analytics events
        with adapter.track_analytics_session(
            session_name=f"llm_{feature_config['feature']}",
            feature=feature_config["feature"],
            ai_model=feature_config["model"],
            context=feature_config["context"],
        ) as session:
            # LLM request event
            llm_request_result = adapter.capture_event_with_governance(
                event_name="llm_request_started",
                properties={
                    "feature": feature_config["feature"],
                    "model": feature_config["model"],
                    "prompt_length": len(feature_config["prompt"]),
                    "context": feature_config["context"],
                    "request_id": f"llm_req_{int(time.time())}",
                },
                distinct_id=f"system_llm_{int(time.time())}",
                session_id=session.session_id,
            )

            # Simulate processing time
            processing_time = random.uniform(1.5, 4.2)
            time.sleep(0.1)  # Demo timing

            # LLM response event
            llm_response_result = adapter.capture_event_with_governance(
                event_name="llm_response_completed",
                properties={
                    "feature": feature_config["feature"],
                    "model": feature_config["model"],
                    "processing_time_seconds": processing_time,
                    "response_length": random.randint(150, 800),
                    "success": True,
                    "cost_estimated": random.uniform(0.001, 0.01),
                },
                distinct_id=f"system_llm_{int(time.time())}",
                session_id=session.session_id,
            )

            session_cost = llm_request_result["cost"] + llm_response_result["cost"]
            llm_total_cost += Decimal(str(session_cost))

            print(f"     Model: {feature_config['model']}")
            print(f"     Processing: {processing_time:.1f}s")
            print(f"     Analytics cost: ${session_cost:.6f}")

    print("\n🤖 LLM Analytics Summary:")
    print(f"  Features analyzed: {len(llm_features)}")
    print(f"  Total analytics cost: ${llm_total_cost:.4f}")
    print(f"  Average cost per feature: ${llm_total_cost / len(llm_features):.6f}")


def demonstrate_session_analytics(adapter):
    """Demonstrate session recording and analytics with governance."""

    print("🎬 Simulating user session recordings with governance...")

    # Simulate different types of user sessions
    session_types = [
        {
            "type": "onboarding",
            "duration": random.randint(120, 300),
            "actions": [
                "signup",
                "tutorial_start",
                "tutorial_complete",
                "first_action",
            ],
            "user_segment": "new_user",
        },
        {
            "type": "power_user_session",
            "duration": random.randint(600, 1200),
            "actions": [
                "login",
                "dashboard_view",
                "report_generate",
                "data_export",
                "logout",
            ],
            "user_segment": "enterprise",
        },
        {
            "type": "troubleshooting",
            "duration": random.randint(180, 450),
            "actions": [
                "error_encountered",
                "help_search",
                "support_contact",
                "issue_resolved",
            ],
            "user_segment": "premium",
        },
    ]

    session_costs = []

    for session_config in session_types:
        user_id = f"user_{session_config['type']}_{random.randint(1000, 9999)}"

        print(f"\n  📹 Recording session: {session_config['type']}")

        with adapter.track_analytics_session(
            session_name=f"session_recording_{session_config['type']}",
            customer_id=f"customer_{random.randint(100, 999)}",
            user_segment=session_config["user_segment"],
            session_type=session_config["type"],
        ) as session:
            # Session start
            session_start_result = adapter.capture_event_with_governance(
                event_name="session_recording_started",
                properties={
                    "session_type": session_config["type"],
                    "user_segment": session_config["user_segment"],
                    "estimated_duration": session_config["duration"],
                    "recording_quality": "high",
                },
                distinct_id=user_id,
                session_id=session.session_id,
            )

            # Session actions
            action_costs = []
            for action in session_config["actions"]:
                action_result = adapter.capture_event_with_governance(
                    event_name=f"session_action_{action}",
                    properties={
                        "action_type": action,
                        "session_type": session_config["type"],
                        "timestamp": datetime.now().isoformat(),
                    },
                    distinct_id=user_id,
                    is_identified=True,
                    session_id=session.session_id,
                )
                action_costs.append(action_result["cost"])
                time.sleep(0.05)  # Demo timing

            # Session end
            session_end_result = adapter.capture_event_with_governance(
                event_name="session_recording_completed",
                properties={
                    "session_type": session_config["type"],
                    "actual_duration": session_config["duration"],
                    "actions_count": len(session_config["actions"]),
                    "recording_size_mb": random.uniform(5.2, 15.8),
                },
                distinct_id=user_id,
                session_id=session.session_id,
            )

            total_session_cost = (
                session_start_result["cost"]
                + sum(action_costs)
                + session_end_result["cost"]
            )

            session_costs.append(total_session_cost)

            print(f"     Duration: {session_config['duration']}s")
            print(f"     Actions: {len(session_config['actions'])}")
            print(f"     Cost: ${total_session_cost:.6f}")

    print("\n📹 Session Analytics Summary:")
    print(f"  Sessions recorded: {len(session_types)}")
    print(f"  Total recording cost: ${sum(session_costs):.4f}")
    print(f"  Average cost per session: ${sum(session_costs) / len(session_costs):.6f}")


def demonstrate_ab_testing(adapter):
    """Demonstrate A/B testing with cost intelligence."""

    print("🧪 Running A/B tests with cost tracking...")

    # Define A/B tests
    ab_tests = [
        {
            "test_name": "checkout_flow_optimization",
            "variants": ["control", "variant_a", "variant_b"],
            "traffic_split": [0.33, 0.33, 0.34],
            "success_metric": "conversion_rate",
        },
        {
            "test_name": "pricing_page_layout",
            "variants": ["current", "simplified", "detailed"],
            "traffic_split": [0.4, 0.3, 0.3],
            "success_metric": "engagement_time",
        },
        {
            "test_name": "onboarding_tutorial",
            "variants": ["interactive", "video", "text_only"],
            "traffic_split": [0.4, 0.3, 0.3],
            "success_metric": "completion_rate",
        },
    ]

    test_results = {}

    for test_config in ab_tests:
        test_name = test_config["test_name"]
        print(f"\n  🧪 A/B Test: {test_name}")

        variant_results = {}

        # Simulate users for each variant
        for variant, traffic_pct in zip(
            test_config["variants"], test_config["traffic_split"]
        ):
            variant_users = int(100 * traffic_pct)  # Simulate 100 total users
            variant_cost = Decimal("0")

            for user_num in range(variant_users):
                user_id = f"test_user_{test_name}_{variant}_{user_num}"

                # Test assignment event
                assignment_result = adapter.capture_event_with_governance(
                    event_name="ab_test_assignment",
                    properties={
                        "test_name": test_name,
                        "variant": variant,
                        "assignment_timestamp": datetime.now().isoformat(),
                        "user_segment": random.choice(
                            ["free", "premium", "enterprise"]
                        ),
                    },
                    distinct_id=user_id,
                    is_identified=True,
                )

                # Success metric event (simulate some succeeding)
                success_probability = random.uniform(0.1, 0.8)  # Varying success rates
                if random.random() < success_probability:
                    success_result = adapter.capture_event_with_governance(
                        event_name=f"ab_test_success_{test_config['success_metric']}",
                        properties={
                            "test_name": test_name,
                            "variant": variant,
                            "success_metric": test_config["success_metric"],
                            "success_value": random.uniform(0.5, 1.0),
                        },
                        distinct_id=user_id,
                        is_identified=True,
                    )
                    variant_cost += Decimal(str(success_result["cost"]))

                variant_cost += Decimal(str(assignment_result["cost"]))

            variant_results[variant] = {
                "users": variant_users,
                "cost": float(variant_cost),
                "cost_per_user": float(variant_cost / variant_users)
                if variant_users > 0
                else 0,
            }

            print(f"     {variant:12} -> {variant_users:3} users, ${variant_cost:.4f}")

        test_results[test_name] = variant_results

    print("\n🧪 A/B Testing Summary:")
    sum(
        sum(
            variant["cost"]
            for variant in test["variants"].values()
            if "variants" in test
        )
        for test in test_results.values()
    )

    # Calculate test cost (need to fix the summary calculation)
    actual_total_cost = Decimal("0")
    total_users = 0
    for _test_name, variants in test_results.items():
        for _variant_name, variant_data in variants.items():
            actual_total_cost += Decimal(str(variant_data["cost"]))
            total_users += variant_data["users"]

    print(f"  Tests conducted: {len(ab_tests)}")
    print(f"  Total test users: {total_users}")
    print(f"  Total testing cost: ${actual_total_cost:.4f}")
    print(f"  Average cost per user: ${actual_total_cost / total_users:.6f}")


def demonstrate_multi_customer_analytics(adapter):
    """Demonstrate multi-customer analytics governance."""

    print("🏢 Processing multi-customer analytics with governance...")

    customers = [
        {"id": "enterprise_corp", "tier": "enterprise", "events_per_day": 10000},
        {"id": "startup_inc", "tier": "premium", "events_per_day": 2500},
        {"id": "freelancer_llc", "tier": "free", "events_per_day": 500},
        {"id": "agency_partners", "tier": "premium", "events_per_day": 5000},
    ]

    customer_costs = {}

    for customer in customers:
        customer_id = customer["id"]
        daily_events = customer["events_per_day"]

        print(f"\n  🏢 Customer: {customer_id}")
        print(f"     Tier: {customer['tier']}")
        print(f"     Daily events: {daily_events:,}")

        with adapter.track_analytics_session(
            session_name=f"daily_analytics_{customer_id}",
            customer_id=customer_id,
            cost_center=f"customer_{customer['tier']}",
            tier=customer["tier"],
            daily_event_volume=daily_events,
        ) as session:
            # Simulate a sample of the daily events
            sample_events = min(50, daily_events // 100)  # Sample for demo
            customer_cost = Decimal("0")

            for event_num in range(sample_events):
                event_types = ["page_view", "button_click", "conversion", "feature_use"]
                event_name = random.choice(event_types)

                result = adapter.capture_event_with_governance(
                    event_name=event_name,
                    properties={
                        "customer_tier": customer["tier"],
                        "event_sequence": event_num,
                        "daily_volume_estimate": daily_events,
                    },
                    distinct_id=f"user_{customer_id}_{event_num}",
                    is_identified=customer["tier"] != "free",
                    session_id=session.session_id,
                )

                customer_cost += Decimal(str(result["cost"]))

            # Extrapolate to full daily cost
            full_daily_cost = customer_cost * (daily_events / sample_events)
            customer_costs[customer_id] = {
                "daily_cost": float(full_daily_cost),
                "events": daily_events,
                "tier": customer["tier"],
                "cost_per_event": float(full_daily_cost / daily_events)
                if daily_events > 0
                else 0,
            }

            print(f"     Sample events processed: {sample_events}")
            print(f"     Estimated daily cost: ${full_daily_cost:.2f}")
            print(f"     Cost per event: ${full_daily_cost / daily_events:.6f}")

    print("\n🏢 Multi-Customer Summary:")
    total_daily_cost = sum(
        customer["daily_cost"] for customer in customer_costs.values()
    )
    total_daily_events = sum(customer["events"] for customer in customer_costs.values())

    print(f"  Customers managed: {len(customers)}")
    print(f"  Total daily events: {total_daily_events:,}")
    print(f"  Total daily cost: ${total_daily_cost:.2f}")
    print(f"  Average cost per event: ${total_daily_cost / total_daily_events:.6f}")

    # Customer tier breakdown
    tier_summary = {}
    for _customer_id, data in customer_costs.items():
        tier = data["tier"]
        if tier not in tier_summary:
            tier_summary[tier] = {"customers": 0, "cost": 0, "events": 0}
        tier_summary[tier]["customers"] += 1
        tier_summary[tier]["cost"] += data["daily_cost"]
        tier_summary[tier]["events"] += data["events"]

    print("\n  By customer tier:")
    for tier, summary in tier_summary.items():
        print(
            f"    {tier:10} -> {summary['customers']} customers, ${summary['cost']:.2f}/day"
        )


def demonstrate_dashboard_analytics(adapter):
    """Demonstrate real-time dashboard analytics."""

    print("📊 Generating real-time dashboard analytics...")

    # Simulate dashboard usage patterns
    dashboard_sessions = [
        {"name": "executive_summary", "complexity": "low", "update_freq": "hourly"},
        {
            "name": "user_behavior_deep_dive",
            "complexity": "high",
            "update_freq": "real-time",
        },
        {"name": "conversion_funnel", "complexity": "medium", "update_freq": "daily"},
        {"name": "revenue_analytics", "complexity": "high", "update_freq": "hourly"},
    ]

    dashboard_costs = []

    for dashboard in dashboard_sessions:
        dashboard_name = dashboard["name"]

        print(f"\n  📊 Dashboard: {dashboard_name}")

        with adapter.track_analytics_session(
            session_name=f"dashboard_{dashboard_name}",
            dashboard_type=dashboard_name,
            complexity=dashboard["complexity"],
            update_frequency=dashboard["update_freq"],
        ) as session:
            # Dashboard load event
            load_result = adapter.capture_event_with_governance(
                event_name="dashboard_loaded",
                properties={
                    "dashboard_name": dashboard_name,
                    "complexity": dashboard["complexity"],
                    "load_time_ms": random.randint(200, 2000),
                    "data_points": random.randint(50, 500),
                },
                distinct_id=f"dashboard_user_{int(time.time())}",
                session_id=session.session_id,
            )

            # Data refresh events
            refresh_count = {"low": 2, "medium": 4, "high": 8}[dashboard["complexity"]]
            refresh_costs = []

            for refresh_num in range(refresh_count):
                refresh_result = adapter.capture_event_with_governance(
                    event_name="dashboard_data_refresh",
                    properties={
                        "dashboard_name": dashboard_name,
                        "refresh_sequence": refresh_num,
                        "data_freshness_seconds": random.randint(30, 300),
                        "query_complexity": dashboard["complexity"],
                    },
                    distinct_id=f"dashboard_user_{int(time.time())}",
                    session_id=session.session_id,
                )
                refresh_costs.append(refresh_result["cost"])

            total_dashboard_cost = load_result["cost"] + sum(refresh_costs)
            dashboard_costs.append(total_dashboard_cost)

            print(f"     Complexity: {dashboard['complexity']}")
            print(f"     Refreshes: {refresh_count}")
            print(f"     Total cost: ${total_dashboard_cost:.6f}")

    print("\n📊 Dashboard Analytics Summary:")
    print(f"  Dashboards active: {len(dashboard_sessions)}")
    print(f"  Total dashboard cost: ${sum(dashboard_costs):.4f}")
    print(
        f"  Average cost per dashboard: ${sum(dashboard_costs) / len(dashboard_costs):.6f}"
    )


def display_final_summary(adapter):
    """Display comprehensive summary of all advanced features."""

    cost_summary = adapter.get_cost_summary()

    print("💰 Overall Cost & Governance Summary:")
    print(f"  Total daily cost: ${cost_summary['daily_costs']:.4f}")
    print(f"  Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(
        f"  Remaining budget: ${cost_summary['daily_budget_limit'] - cost_summary['daily_costs']:.4f}"
    )

    print("\n🏛️ Governance Configuration:")
    print(f"  Team: {cost_summary['team']}")
    print(f"  Project: {cost_summary['project']}")
    print(f"  Environment: {cost_summary['environment']}")
    print(f"  Policy: {cost_summary['governance_policy']}")
    print("  Cost tracking: Enabled")
    print(
        f"  Alerts: {'Enabled' if cost_summary['cost_alerts_enabled'] else 'Disabled'}"
    )

    print("\n📊 Advanced Features Demonstrated:")
    features = [
        "✅ Multi-tenant feature flag management with governance",
        "✅ LLM analytics integration with cost tracking",
        "✅ Session recording analytics with attribution",
        "✅ A/B testing with cost intelligence",
        "✅ Multi-customer analytics governance",
        "✅ Real-time dashboard analytics",
    ]

    for feature in features:
        print(f"  {feature}")

    # Cost optimization recommendations
    volume_analysis = adapter.get_volume_discount_analysis(
        projected_monthly_events=100000
    )

    print("\n💡 Advanced Optimization Insights:")
    print(
        f"  Monthly cost projection: ${volume_analysis['projected_monthly_cost']:.2f}"
    )
    print(f"  Cost per event: ${volume_analysis['cost_per_event']:.6f}")

    if volume_analysis["optimization_recommendations"]:
        print(
            f"  Available optimizations: {len(volume_analysis['optimization_recommendations'])}"
        )
        for i, rec in enumerate(volume_analysis["optimization_recommendations"][:2], 1):
            print(
                f"    {i}. {rec['optimization_type']}: ${rec['potential_savings_per_month']:.2f}/month"
            )

    print("\n🚀 Next Steps for Advanced Usage:")
    print("  1. Integrate with your observability platform")
    print("  2. Set up automated cost alerts and budgets")
    print("  3. Deploy to production with governance policies")
    print("  4. Explore production patterns: python production_patterns.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Advanced features demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(
            "🐛 Please report this issue: https://github.com/KoshiHQ/GenOps-AI/issues"
        )
